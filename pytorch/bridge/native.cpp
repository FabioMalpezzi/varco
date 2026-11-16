// The container end, in C++: an MPS backend for PyTorch on Linux whose
// kernels send every operation to the Mac.
//
// A tensor "on mps" here has a shape, strides, a type, the MPS device and
// dispatch key, and no data: the data is on the Mac, under a number. Being a
// real MPS tensor to PyTorch, every operation on it is treated exactly as on
// a Mac (`linear` and attention stay whole) and ends at this backend's one
// kernel, `fallback`. That kernel recognises an operation it has seen before
// with the same shapes and constants, makes the result tensors, and writes a
// few integers to the socket (wire.py's O frame): no Python, no waiting.
// Anything else goes to Python (client.py), which works out the result's
// shape, tells the Mac, and has the operation remembered for next time.
//
// Built into the image (Dockerfile) as the module `bridge_native`. It must be
// imported before torch: see MPSHooks.
#include <torch/extension.h>

#include <pybind11/stl.h>

#include <ATen/EmptyTensor.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/detail/MPSHooksInterface.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/GeneratorImpl.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/Storage.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/tensor_dtypes.h>
#include <torch/version.h>

#include <sys/socket.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

// --- telling PyTorch there is an MPS device -------------------------------------

// Linux builds know the name "mps" but refuse to move a tensor to it ("not
// linked with support for mps devices"): no device guard is registered for
// it. This one has one device, one stream, and events that are always done:
// enough for the autograd engine, which handles an accelerator's gradients
// by stream and event. Waiting for the GPU is the bridge's to do, not this.
namespace {
struct Guard final : c10::impl::DeviceGuardImplInterface {
  using Device = c10::Device;
  using Stream = c10::Stream;
  static Device device() { return Device(c10::DeviceType::MPS, 0); }
  static Stream stream() { return Stream(Stream::DEFAULT, device()); }
  c10::DeviceType type() const override { return c10::DeviceType::MPS; }
  Device exchangeDevice(Device) const override { return device(); }
  Device getDevice() const override { return device(); }
  void setDevice(Device) const override {}
  void uncheckedSetDevice(Device) const noexcept override {}
  Stream getStream(Device) const override { return stream(); }
  Stream getDefaultStream(Device) const override { return stream(); }
  Stream getNewStream(Device, int = 0) const override { return stream(); }
  Stream exchangeStream(Stream) const override { return stream(); }
  c10::DeviceIndex deviceCount() const noexcept override { return 1; }
  void record(void**, const Stream&, const c10::DeviceIndex, const c10::EventFlag) const override {}
  void block(void*, const Stream&) const override {}
  bool queryEvent(void*) const override { return true; }
  void destroyEvent(void*, const c10::DeviceIndex) const noexcept override {}
  bool queryStream(const Stream&) const override { return true; }
  void synchronizeStream(const Stream&) const override {}
  void synchronizeEvent(void*) const override {}
  void synchronizeDevice(const c10::DeviceIndex) const override {}
  void recordDataPtrOnStream(const c10::DataPtr&, const Stream&) const override {}
};
}  // namespace
C10_REGISTER_GUARD_IMPL(MPS, Guard);

// What PyTorch asks the MPS backend itself, not an operator: is there a GPU, how much memory is in
// use, wait for it, a random generator of its own. The answers are the Mac's: client.py's
// Bridge.call, by name. Defined further down, with the rest of the state.
namespace {
py::object call_mac(const char* name, const py::tuple& arguments);  // with Python's lock held
bool reachable();

// PyTorch calls the hooks with or without Python's lock (torch.mps.synchronize() lets go of it
// first), and nothing of Python's may be made, read or dropped without it: everything from the
// arguments to the answer's conversion happens here, inside the lock.
template <typename Result, typename... Arguments>
Result mac(const char* name, Arguments&&... arguments) {
  py::gil_scoped_acquire lock;
  py::object value = call_mac(name, py::make_tuple(std::forward<Arguments>(arguments)...));
  if constexpr (!std::is_void_v<Result>) return value.template cast<Result>();
}

// A random generator "on mps" is on the Mac, under a number: 0 is the Mac's default one. Random
// operators get it by that number (key_of), so a seeded program draws what it would on a Mac.
std::atomic<uint32_t> next_generator{1};
struct RemoteGenerator final : c10::GeneratorImpl {
  uint32_t id;
  explicit RemoteGenerator(uint32_t id)
      : c10::GeneratorImpl(c10::Device(c10::DeviceType::MPS, 0), c10::DispatchKeySet(c10::DispatchKey::MPS)), id(id) {}
  void set_current_seed(uint64_t seed) override { mac<void>("generator.manual_seed", id, seed); }
  uint64_t current_seed() const override { return mac<uint64_t>("generator.initial_seed", id); }
  uint64_t seed() override {
    uint64_t random = c10::detail::getNonDeterministicRandom(true);
    set_current_seed(random);
    return random;
  }
  void set_offset(uint64_t) override { TORCH_CHECK_NOT_IMPLEMENTED(false, "an MPS generator has no offset"); }
  uint64_t get_offset() const override { TORCH_CHECK_NOT_IMPLEMENTED(false, "an MPS generator has no offset"); }
  void set_state(const c10::TensorImpl& state) override {
    at::Tensor tensor(c10::intrusive_ptr<c10::TensorImpl>::unsafe_reclaim_from_nonowning(const_cast<c10::TensorImpl*>(&state)));
    mac<void>("generator.set_state", id, tensor);
  }
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override {
    return mac<at::Tensor>("generator.get_state", id).getIntrusivePtr();
  }
  static c10::DeviceType device_type() { return c10::DeviceType::MPS; }
  RemoteGenerator* clone_impl() const override {
    auto* twin = new RemoteGenerator(next_generator++);
    mac<void>("generator.clone", id, twin->id);
    return twin;
  }
};
}  // namespace

// PyTorch asks these hooks whether there is an MPS device at all. Saying yes
// is what lets a backward pass run: the autograd engine asks for "the
// accelerator" and, on Linux, would find none. The name is the one PyTorch
// looks up, so this module has to be loaded before PyTorch first asks:
// import bridge_native before torch. The rest is what torch.mps and
// torch.backends.mps call: each is done on the Mac.
namespace at {
struct MPSHooks : MPSHooksInterface {
  template <typename... Arguments>
  explicit MPSHooks(Arguments&&...) {}  // PyTorch's registry gives it an argument in some versions, none in others
  void init() const override {}
  bool isBuilt() const override { return true; }
  bool isAvailable() const override { return reachable(); }  // torch.backends.mps.is_available(): the Mac answers
  bool hasMPS() const override { return true; }
  bool hasPrimaryContext(c10::DeviceIndex) const override { return true; }
  c10::DeviceIndex deviceCount() const override { return 1; }
  bool isOnMacOSorNewer(unsigned major, unsigned minor) const override {
    return mac<bool>("is_macos_or_newer", major, minor);
  }
  const Generator& getDefaultGenerator(c10::DeviceIndex = -1) const override {
    static Generator generator = make_generator<RemoteGenerator>(0u);
    return generator;
  }
  Generator getNewGenerator(c10::DeviceIndex = -1) const override {
    uint32_t id = next_generator++;
    mac<void>("generator.new", id);
    return make_generator<RemoteGenerator>(id);
  }
  void deviceSynchronize() const override { mac<void>("synchronize"); }
  void commitStream() const override { mac<void>("synchronize"); }
  void emptyCache() const override { mac<void>("empty_cache"); }
  size_t getCurrentAllocatedMemory() const override { return mac<size_t>("current_allocated_memory"); }
  size_t getDriverAllocatedMemory() const override { return mac<size_t>("driver_allocated_memory"); }
  size_t getRecommendedMaxMemory() const override { return mac<size_t>("recommended_max_memory"); }
  void setMemoryFraction(double ratio) const override { mac<void>("set_per_process_memory_fraction", ratio); }
  void profilerStartTrace(const std::string& mode, bool wait) const override { mac<void>("profiler_start", mode, wait); }
  void profilerStopTrace() const override { mac<void>("profiler_stop"); }
  uint32_t acquireEvent(bool timing) const override { return mac<uint32_t>("event.acquire", timing); }
  void releaseEvent(uint32_t event) const override { mac<void>("event.release", event); }
  void recordEvent(uint32_t event) const override { mac<void>("event.record", event); }
  void waitForEvent(uint32_t event) const override { mac<void>("event.wait", event); }
  void synchronizeEvent(uint32_t event) const override { mac<void>("event.synchronize", event); }
  bool queryEvent(uint32_t event) const override { return mac<bool>("event.query", event); }
  double elapsedTimeOfEvents(uint32_t start, uint32_t end) const override { return mac<double>("event.elapsed", start, end); }
  // Pinned memory is for copying to a GPU with memory of its own. There is nothing to pin here:
  // tensor.pin_memory() gives an ordinary copy, as harmless as it is useless.
  bool isPinnedPtr(const void*) const override { return false; }
  Allocator* getPinnedMemoryAllocator() const override { return c10::GetCPUAllocator(); }
};
REGISTER_MPS_HOOKS(MPSHooks);
}  // namespace at

namespace {

// --- the tensors ------------------------------------------------------------------

// What a tensor here stands for. It hangs off the tensor's storage, as the
// storage's context, so that finding it is following two pointers, and so
// that it goes when the storage does: PyTorch re-wraps tensors (a module's
// parameters, on .to()), and the storage is what survives that.
struct Info {
  uint32_t handle;      // the Mac's number for the tensor
  uint32_t tag;         // stands for its shape, strides and type: an operation's key is made of these
  uint32_t generation;  // which connection to the Mac the number belongs to
  int64_t offset;       // where the tensor starts in its storage, as a view does on a Mac: see make
  bool view;            // made by an operator that returns a view of its argument: its data is another tensor's
  c10::Storage root;    // for a view: the storage of the tensor whose data it is, see root_of
  uint32_t via = 0;     // for a storage that is no longer a tensor's own (handle 0, see rebind): a tensor that still reads it
};

// The connection a tensor's number means something on. A process made by fork() inherits its
// parent's tensors and socket, and must use neither (forked, below): it gets a connection of its
// own when it first needs one, and that is a new generation.
std::atomic<uint32_t> generation{1};
// The first generation of this process itself: those before it were its parent's (see forked).
std::atomic<uint32_t> own_generation{1};

std::mutex* freed_lock = new std::mutex;  // storages can go on any thread
std::vector<uint32_t> freed;              // the Mac's numbers of the tensors that went, not yet told to the Mac

void gone(void* context) {
  auto* info = static_cast<Info*>(context);
  if (info->handle && info->generation == generation.load(std::memory_order_relaxed)) {
    std::lock_guard<std::mutex> lock(*freed_lock);
    freed.push_back(info->handle);
  }
  delete info;
}

// A remembered result: a new tensor of this shape, or one of the arguments
// handed back (reshaped, if the operation reshapes it in place), or none.
struct Out {
  enum Kind { Fresh, Argument, None } kind;
  uint32_t argument = 0;
  bool reshaped = false;
  std::vector<int64_t> sizes, strides;
  int64_t offset = 0;
  caffe2::TypeMeta type;
  uint32_t tag = 0;
};

struct Remembered {
  uint32_t signature;
  bool list;  // the operator returns one list of tensors, not one value per result
  std::vector<Out> outs;
};

struct Hash {
  size_t operator()(const std::vector<uint64_t>& words) const {
    uint64_t h = 0x9e3779b97f4a7c15ull;
    for (uint64_t w : words) {
      h ^= w + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
      h *= 0xff51afd7ed558ccdull;
    }
    return h;
  }
};

// Everything below is touched only with Python's lock held: it is this module's lock too.
struct State {
  int fd = -1;
  size_t flush_at = 1024;  // small: the Mac should start while the program is still issuing
  std::vector<char> out;   // bytes for the Mac, not yet sent
  uint32_t next_handle = 1;
  uint32_t next_signature = 1;
  std::unordered_map<std::vector<uint64_t>, uint32_t, Hash> tags;
  std::vector<std::vector<uint64_t>> shapes;  // each tag's type, sizes and strides: tags, the other way round
  std::unordered_map<std::vector<uint64_t>, Remembered, Hash> remembered;
  bool learn = true;   // false: nothing is remembered, every operation goes through Python (BRIDGE_SYNC)
  py::object slow;     // Python's side: client.py, Bridge._slow
  py::object call;     // Bridge.call: a question for the Mac that is not an operator
  py::object connect;  // Bridge.connect: opens the connection, when something first needs the Mac
  uint64_t sent = 0;
  uint64_t missed = 0;
  // How far the program may run ahead of the Mac: after `window` operations sent without waiting
  // for anything, it waits for the Mac to have taken them all (pace, below). 0: no limit.
  uint32_t ahead = 0, window = 1024, widest = 4096;
};

// Keys of the operations Python is working on, innermost last. Per thread: Python can hand the
// lock to another thread in the middle of one, and that thread's operation is not this one's.
thread_local std::vector<std::vector<uint64_t>> pending;

State& state() {
  static State* s = new State();  // never destroyed: tensors can outlive the interpreter's teardown
  return *s;
}

uint32_t tag_of(c10::IntArrayRef sizes, c10::IntArrayRef strides, caffe2::TypeMeta type) {
  std::vector<uint64_t> shape;
  shape.reserve(2 + 2 * sizes.size());
  shape.push_back(static_cast<uint64_t>(c10::typeMetaToScalarType(type)));
  shape.push_back(sizes.size());
  for (auto v : sizes) shape.push_back(static_cast<uint64_t>(v));
  for (auto v : strides) shape.push_back(static_cast<uint64_t>(v));
  auto& tags = state().tags;
  auto found = tags.find(shape);
  if (found != tags.end()) return found->second;
  uint32_t tag = static_cast<uint32_t>(tags.size()) + 1;
  state().shapes.push_back(shape);
  tags.emplace(std::move(shape), tag);
  return tag;
}

// `view_of`: the tensor this one is a view of, for an operator that returns a view (select, detach,
// transpose). A view is the kind of tensor its base is, whatever the mode it is made in: a view
// of an ordinary tensor made inside torch.inference_mode() is an ordinary tensor, and shares its
// base's version counter, which an inference tensor cannot. PyTorch's own backends get this by
// building a view with its base's keys; so does this.
//
// `offset`: where the tensor starts in its storage. Here every tensor has a storage of its own,
// views too (the Mac has the real ones, which do share), but a view still says where in its
// base it starts, as on a Mac: autograd reads a view's place in its base from its sizes,
// strides and offset, to send the gradient of an operation in place on a view (y[1:3] += x) to
// the right elements. The storage is as long as that takes, and saving lays the data out so.
Info* info_of(const at::Tensor& tensor);

// The storage a view's data is really in: its base's, or if its base is a view too, that one's.
// On a Mac a tensor and its views have one storage between them, and saving them writes it once:
// loaded again, they still share it, and a change made through one shows in the others. Here
// every tensor has a storage of its own (see make), so each view remembers whose data it is, and
// saving asks (install.py, _saving). The storage is kept for as long as a view of it lives, as
// on a Mac; the Mac keeps the tensor it stands for, which costs it nothing more than the view does.
c10::Storage root_of(const at::Tensor& base) {
  Info* info = info_of(base);
  if (info && info->view && info->root) return info->root;
  return base.storage();
}

at::Tensor make(c10::IntArrayRef sizes, std::optional<c10::IntArrayRef> strides, caffe2::TypeMeta type,
                uint32_t handle, uint32_t tag, const at::Tensor* view_of = nullptr, int64_t offset = 0) {
  // As many bytes as the real tensor has, so that sizes read true; none allocated.
  size_t bytes = strides ? at::detail::computeStorageNbytes(sizes, *strides, type.itemsize(), offset)
                         : at::detail::computeStorageNbytesContiguous(sizes, type.itemsize(), offset);
  auto* info = new Info{handle, tag, generation.load(std::memory_order_relaxed), offset, view_of != nullptr,
                        view_of ? root_of(*view_of) : c10::Storage()};
  // The storage's pointer is not to the data, which is not here, but it is not null either: PyTorch
  // takes a storage with a null pointer for a broken one, and tells storages apart by their pointers
  // (saving does both). The note above serves: an address no other storage has.
  c10::Storage storage(c10::Storage::use_byte_size_t(), bytes,
                       at::DataPtr(info, info, &gone, c10::Device(c10::DeviceType::MPS, 0)), nullptr, false);
  auto tensor = view_of
      ? at::detail::make_tensor<c10::TensorImpl>(c10::TensorImpl::VIEW, std::move(storage), view_of->key_set(), type)
      : at::detail::make_tensor<c10::TensorImpl>(std::move(storage), c10::DispatchKeySet(c10::DispatchKey::MPS), type);
  if (strides) {
    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(sizes, *strides);
  } else {
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(sizes);
  }
  if (offset) tensor.unsafeGetTensorImpl()->set_storage_offset(offset);
  if (!tag) info->tag = tag_of(tensor.sizes(), tensor.strides(), type);
  return tensor;
}

Info* info_of(const c10::Storage& storage) {
  const auto& data = storage.data_ptr();
  if (data.get_deleter() != &gone) return nullptr;
  auto* info = static_cast<Info*>(data.get_context());
  if (info->generation != generation.load(std::memory_order_relaxed)) {
    if (info->generation >= own_generation.load(std::memory_order_relaxed)) {
      throw std::runtime_error("this MPS tensor was on a connection to the Mac that has been closed (the bridge lost the "
                               "Mac, or its place among the Mac's answers): what the program had on the GPU went with it. "
                               "Tensors made from now on are on a new connection.");
    }
    throw std::runtime_error("this MPS tensor was made before os.fork(), in the parent process: the GPU's tensors do "
                             "not cross a fork (on a Mac neither). Make them in the process that uses them, or move "
                             "them to the CPU before forking.");
  }
  return info;
}

Info* info_of(const at::Tensor& tensor) {
  if (!tensor.defined() || !tensor.is_mps() || !tensor.has_storage()) return nullptr;
  return info_of(tensor.storage());
}

// An operation that reshapes its argument in place: the tensor here follows, and its storage
// grows with it, as it would have (its size is what saving the tensor writes).
void reshape(const at::Tensor& tensor, c10::IntArrayRef sizes, c10::IntArrayRef strides, uint32_t tag, int64_t offset) {
  auto* impl = tensor.unsafeGetTensorImpl();
  impl->set_sizes_and_strides(sizes, strides);
  impl->set_storage_offset(offset);
  size_t bytes = at::detail::computeStorageNbytes(sizes, strides, impl->dtype().itemsize(), offset);
  if (bytes > impl->storage().nbytes()) impl->storage().unsafeGetStorageImpl()->set_nbytes(bytes);
  if (Info* info = info_of(tensor)) {
    info->tag = tag ? tag : tag_of(tensor.sizes(), tensor.strides(), tensor.dtype());
    info->offset = offset;
  }
}

// x.set_(y): the tensor is given another tensor's memory (or, with no y, none). On a Mac its
// storage is replaced, and so it is here: the tensor gets a new storage, which stands for it
// from now on (the Mac's number moves to it) and says whose memory it now is, so that saving
// writes that memory once for the two. The storage it had stays what the views made of it
// before still share, as on a Mac, where they go on reading the first memory: it is no tensor's
// own any more (number 0), and what is in it is fetched through one of those views (via).
void rebind(const at::Tensor& tensor, c10::Storage shared, c10::IntArrayRef sizes, c10::IntArrayRef strides,
            int64_t offset) {
  Info* was = info_of(tensor);
  if (!was) throw std::runtime_error("set_: not a tensor of the bridge's");
  auto* impl = tensor.unsafeGetTensorImpl();
  const auto type = impl->dtype();
  size_t bytes = at::detail::computeStorageNbytes(sizes, strides, type.itemsize(), offset);
  const bool view = static_cast<bool>(shared);
  auto* info = new Info{was->handle, 0, was->generation, offset, view, std::move(shared)};
  was->handle = 0;
  c10::Storage storage(c10::Storage::use_byte_size_t(), bytes,
                       at::DataPtr(info, info, &gone, c10::Device(c10::DeviceType::MPS, 0)), nullptr, false);
  impl->set_storage_keep_dtype(std::move(storage));  // (the old one goes here, if no view holds it: `was` with it)
  impl->set_sizes_and_strides(sizes, strides);
  impl->set_storage_offset(offset);
  info->tag = tag_of(tensor.sizes(), tensor.strides(), type);
}

// --- the wire -----------------------------------------------------------------------

void put(std::vector<char>& out, char kind, uint32_t n) {
  size_t at = out.size();
  out.resize(at + 5);
  out[at] = kind;
  std::memcpy(out.data() + at + 1, &n, 4);  // little-endian, as the Macs and these containers are
}

void put(std::vector<char>& out, uint32_t n) {
  size_t at = out.size();
  out.resize(at + 4);
  std::memcpy(out.data() + at, &n, 4);
}

void flush() {
  auto& s = state();
  size_t at = 0;
  while (at < s.out.size()) {
    ssize_t n = ::send(s.fd, s.out.data() + at, s.out.size() - at, MSG_NOSIGNAL);
    if (n < 0) {
      if (errno == EINTR) continue;
      s.out.clear();
      throw std::runtime_error(std::string("the bridge lost the Mac: ") + std::strerror(errno));
    }
    at += static_cast<size_t>(n);
  }
  s.out.clear();
}

// Tensors that went are told to the Mac before the next operation, which is
// when a program on the Mac would have freed them.
void tell_freed() {
  std::vector<uint32_t> went;
  {
    std::lock_guard<std::mutex> lock(*freed_lock);
    if (freed.empty()) return;
    went.swap(freed);
  }
  auto& out = state().out;
  put(out, 'F', static_cast<uint32_t>(went.size()));
  for (uint32_t handle : went) put(out, handle);
}

// --- every operation on an MPS tensor comes through here ----------------------

enum : uint64_t { TENSOR = 1ull << 56, NONE = 2ull << 56, BOOL = 3ull << 56, INT = 4ull << 56, DOUBLE = 5ull << 56,
                  STRING = 6ull << 56, DEVICE = 7ull << 56, INTS = 8ull << 56, DOUBLES = 9ull << 56,
                  BOOLS = 10ull << 56, TENSORS = 11ull << 56, SCALAR = 12ull << 56, GENERATOR = 13ull << 56 };

uint64_t bits(double value) {
  uint64_t out;
  std::memcpy(&out, &value, 8);
  return out;
}

// Whether an operator's real numbers may change from one call to the next without its being
// another operation: an optimizer's step size, a learning rate on a schedule, a loss's weight
// are new numbers at every step, and a training of a hundred thousand steps would otherwise be
// a hundred thousand operations to remember, here and on the Mac. A real number never decides a
// result's shape or type when the operator has a tensor to take them from; the few that make a
// tensor out of numbers (arange and its like) are left out, as are lists of them (the scale
// factors of an upsampling). Integers always stay in the key: they are sizes and dimensions.
bool varies(const c10::FunctionSchema& schema) {
  static auto* known = new std::unordered_map<const c10::FunctionSchema*, bool>();
  auto found = known->find(&schema);
  if (found != known->end()) return found->second;
  bool yes = false;
  const std::string& name = schema.name();
  static const char* made_of_numbers[] = {"aten::arange", "aten::range", "aten::linspace", "aten::logspace"};
  bool numbers = false;
  for (const char* prefix : made_of_numbers) numbers = numbers || name.rfind(prefix, 0) == 0;
  if (!numbers) {
    for (const auto& argument : schema.arguments()) {
      if (argument.is_out()) continue;
      auto kind = argument.type()->kind();
      if (kind == c10::TypeKind::TensorType) yes = true;
      if (kind == c10::TypeKind::ListType &&
          argument.type()->expectRef<c10::ListType>().getElementType()->kind() == c10::TypeKind::TensorType) yes = true;
    }
  }
  known->emplace(&schema, yes);
  return yes;
}

// Whether an operator returns a view of its argument (select, transpose, detach): `Tensor(a)` in
// its schema, not written to.
bool views(const c10::FunctionSchema& schema) {
  static auto* known = new std::unordered_map<const c10::FunctionSchema*, bool>();
  auto found = known->find(&schema);
  if (found != known->end()) return found->second;
  bool yes = false;
  for (const auto& result : schema.returns()) yes = yes || (result.alias_info() && !result.alias_info()->isWrite());
  known->emplace(&schema, yes);
  return yes;
}

// The operation as a key: operator, the tensors' tags, the constants. False
// if something in it is not routine (data still here, a move to the CPU, a
// list with gaps): Python deals with it, and it is not remembered. Real
// numbers that may vary (see varies) are not in the key: they go in `reals`,
// with their arguments' positions in `where`, and travel with each call.
bool key_of(const c10::OperatorHandle& op, c10::ArrayRef<c10::IValue> arguments, std::vector<uint64_t>& key,
            std::vector<uint32_t>& handles, std::vector<double>& reals, std::vector<uint32_t>& where) {
  key.push_back(reinterpret_cast<uint64_t>(&op.schema()));
  const bool vary = varies(op.schema());
  const bool view = views(op.schema());  // a view's place in its storage follows from its base's: part of the key
  for (size_t i = 0; i < arguments.size(); i++) {
    const auto& v = arguments[i];
    if (v.isTensor()) {
      const auto& tensor = v.toTensor();
      if (!tensor.defined()) {
        key.push_back(NONE);
      } else if (Info* info = info_of(tensor)) {
        key.push_back(TENSOR | info->tag);
        if (view) key.push_back(static_cast<uint64_t>(info->offset));
        handles.push_back(info->handle);
      } else if (tensor.dim() == 0 && tensor.device().is_cpu()) {  // a Python number PyTorch wrapped in a tensor
        auto scalar = tensor.item();
        // A Python number counts for less than a tensor when the result's type is decided: part of the key.
        key.push_back(SCALAR | static_cast<uint64_t>(tensor.scalar_type()) |
                      (tensor.unsafeGetTensorImpl()->is_wrapped_number() ? 1ull << 16 : 0));
        if (vary && scalar.isFloatingPoint()) {
          reals.push_back(scalar.toDouble());
          where.push_back(static_cast<uint32_t>(i));
        } else {
          key.push_back(scalar.isFloatingPoint() ? bits(scalar.toDouble())
                        : scalar.isBoolean() ? uint64_t(scalar.toBool()) : static_cast<uint64_t>(scalar.toLong()));
        }
      } else {
        return false;
      }
    } else if (v.isNone()) {
      key.push_back(NONE);
    } else if (v.isBool()) {
      key.push_back(BOOL | uint64_t(v.toBool()));
    } else if (v.isInt()) {
      key.push_back(INT);
      key.push_back(static_cast<uint64_t>(v.toInt()));
    } else if (v.isDouble()) {
      key.push_back(DOUBLE);
      if (vary) {
        reals.push_back(v.toDouble());
        where.push_back(static_cast<uint32_t>(i));
      } else {
        key.push_back(bits(v.toDouble()));
      }
    } else if (v.isString()) {
      key.push_back(STRING | (std::hash<std::string>()(v.toStringRef()) >> 8));
    } else if (v.isDevice()) {
      if (v.toDevice().is_cpu()) return false;
      key.push_back(DEVICE);
    } else if (v.isIntList()) {
      auto list = v.toIntList();
      key.push_back(INTS | list.size());
      for (int64_t e : list) key.push_back(static_cast<uint64_t>(e));
    } else if (v.isDoubleList()) {
      auto list = v.toDoubleList();
      key.push_back(DOUBLES | list.size());
      for (double e : list) key.push_back(bits(e));
    } else if (v.isBoolList()) {
      auto list = v.toBoolList();
      key.push_back(BOOLS | list.size());
      for (bool e : list) key.push_back(uint64_t(e));
    } else if (v.isGenerator()) {  // one of the Mac's, or a CPU one: an error there as here, raised by Python
      auto* remote = dynamic_cast<RemoteGenerator*>(v.toGenerator().unsafeGetGeneratorImpl());
      if (!remote) return false;
      key.push_back(GENERATOR | remote->id);
    } else if (v.isTensorList()) {
      auto list = v.toTensorList();
      key.push_back(TENSORS | list.size());
      for (const at::Tensor& tensor : list) {
        Info* info = info_of(tensor);
        if (!info) return false;
        key.push_back(TENSOR | info->tag);
        handles.push_back(info->handle);
      }
    } else {
      return false;
    }
  }
  return true;
}

// The operators a Mac has no way to run on its GPU: no MPS kernel, and no kernel that works on
// every backend by being made of other operators. A Mac raises NotImplementedError for these, or
// runs them on its CPU if PYTORCH_ENABLE_MPS_FALLBACK=1 was set when PyTorch loaded; so does this.
std::unordered_set<const c10::FunctionSchema*>& not_on_mps() {
  static auto* set = new std::unordered_set<const c10::FunctionSchema*>();
  return *set;
}

bool cpu_fallback() {
  static const bool enabled = [] {
    const char* value = std::getenv("PYTORCH_ENABLE_MPS_FALLBACK");
    return value && std::string(value) == "1";
  }();
  return enabled;
}

// A program that never asks for a value never waits, and could send operations much faster than
// the Mac runs them: thousands of heavy ones queued, which the Mac would still be working through
// long after the program was interrupted, or killed. On a Mac a program cannot get ahead of its own
// operations. So every so many operations sent without a wait, the program waits for the Mac to
// have taken them all. How many adapts: fewer when the wait was long (the Mac is the one behind),
// more when it was nothing (the Mac keeps up, and the wait is pure cost: one round trip in thousands
// of operations). A failure on the Mac is also heard of here, within that many operations.
void pace() {
  auto& s = state();
  s.ahead = 0;
  auto start = std::chrono::steady_clock::now();
  mac<void>("ping");
  double waited = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  if (waited > 0.02) {
    s.window = std::max<uint32_t>(16, s.window / 2);
  } else if (waited < 0.002) {
    s.window = std::min<uint32_t>(s.widest, s.window * 2);
  }
}

// The tensor a result is a view of, if the operator says its result is one (`Tensor(a)` in its
// schema, not written to): for every such operator that is its first argument.
const at::Tensor* base_of(const c10::FunctionSchema& schema, size_t result, c10::ArrayRef<c10::IValue> arguments) {
  const auto& returns = schema.returns();
  const auto* alias = returns[result < returns.size() ? result : 0].alias_info();
  if (!alias || alias->isWrite() || arguments.empty() || !arguments[0].isTensor()) return nullptr;
  const at::Tensor& base = arguments[0].toTensor();
  return base.defined() && base.is_mps() ? &base : nullptr;
}

void fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  py::gil_scoped_acquire lock;
  auto& s = state();
  if (s.fd < 0) {  // the first operation on the GPU: time to reach the Mac (or to say that it cannot be)
    if (!s.connect || s.connect.is_none()) {
      throw std::runtime_error(op.schema().name() + ": this runs on the Mac, through the bridge (import bridge)");
    }
    s.connect();
  }
  if (s.widest && s.ahead >= s.window) pace();
  const auto& schema = op.schema();
  const size_t count = schema.arguments().size();
  auto arguments = torch::jit::last(*stack, count);
  std::vector<uint64_t> key;
  std::vector<uint32_t> handles;
  key.reserve(16);
  handles.reserve(8);
  std::vector<double> reals;
  std::vector<uint32_t> where;
  const bool routine = key_of(op, arguments, key, handles, reals, where) && s.learn;

  if (routine) {
    auto found = s.remembered.find(key);
    if (found != s.remembered.end()) {  // seen before: a few integers to the Mac, and the results made here
      const Remembered& how = found->second;
      std::vector<c10::IValue> results;
      results.reserve(how.outs.size());
      for (const Out& out : how.outs) {
        if (out.kind == Out::Fresh) {
          uint32_t handle = s.next_handle++;
          results.emplace_back(make(out.sizes, c10::IntArrayRef(out.strides), out.type, handle, out.tag,
                                    base_of(schema, results.size(), arguments), out.offset));
          handles.push_back(handle);
        } else if (out.kind == Out::Argument) {
          const auto& tensor = arguments[out.argument].toTensor();
          Info* info = info_of(tensor);
          if (out.reshaped) reshape(tensor, out.sizes, out.strides, out.tag, out.offset);
          results.push_back(arguments[out.argument]);
          handles.push_back(info->handle);
        } else {
          results.emplace_back(at::Tensor());  // a result the operator does not produce here
          handles.push_back(0);
        }
      }
      tell_freed();
      put(s.out, 'O', how.signature);
      for (uint32_t handle : handles) put(s.out, handle);
      if (!reals.empty()) {
        size_t at = s.out.size();
        s.out.resize(at + 8 * reals.size());
        std::memcpy(s.out.data() + at, reals.data(), 8 * reals.size());
      }
      if (s.out.size() >= s.flush_at) flush();
      s.sent++;
      s.ahead++;
      torch::jit::drop(*stack, count);
      if (how.list) {
        c10::List<at::Tensor> list;
        for (auto& result : results) list.push_back(result.toTensor());
        stack->emplace_back(std::move(list));
      } else {
        for (auto& result : results) stack->push_back(std::move(result));
      }
      return;
    }
  }

  // Not seen before, or not routine: Python's turn (client.py, Bridge._slow).
  s.missed++;
  if (not_on_mps().count(&schema) && !cpu_fallback()) {
    // What a Mac says, where a Mac says it: at the call, as a NotImplementedError a program can catch.
    TORCH_CHECK_NOT_IMPLEMENTED(false, "The operator '", schema.operator_name(), "' is not currently implemented for the MPS ",
                                "device. If you want this op to be considered for addition please comment on ",
                                "https://github.com/pytorch/pytorch/issues/141287 and mention use-case, that resulted in ",
                                "missing op as well as commit hash ", TORCH_VERSION, ". As a temporary fix, you can set the ",
                                "environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for ",
                                "this op. WARNING: this will be slower than running natively on MPS.");
  }
  py::list positional;
  py::dict keywords;
  for (size_t i = 0; i < count; i++) {
    const auto& parameter = schema.arguments()[i];
    // A tensor goes as a tensor, and a Python number that PyTorch wrapped in a tensor (x + 1.5) goes
    // as the number it was: when a result's type is decided a number counts for less than a
    // tensor, and Python's side and the Mac's must decide as PyTorch does.
    py::object value;
    if (!arguments[i].isTensor()) {
      value = torch::jit::toPyObject(arguments[i]);
    } else if (const auto& tensor = arguments[i].toTensor(); !tensor.defined()) {
      value = py::none();
    } else if (tensor.unsafeGetTensorImpl()->is_wrapped_number() && tensor.device().is_cpu() && tensor.dim() == 0) {
      auto scalar = tensor.item();
      if (scalar.isBoolean()) value = py::bool_(scalar.toBool());
      else if (scalar.isIntegral(false)) value = py::int_(scalar.toLong());
      else if (scalar.isFloatingPoint()) value = py::float_(scalar.toDouble());
      else value = py::cast(tensor);
    } else {
      value = py::cast(tensor);
    }
    if (!parameter.kwarg_only()) {
      positional.append(std::move(value));
    } else if (!value.is_none()) {
      keywords[py::str(parameter.name())] = std::move(value);
    }
  }
  pending.push_back(routine ? std::move(key) : std::vector<uint64_t>());
  py::object result;
  try {
    result = s.slow(schema.name(), schema.overload_name(), positional, keywords, routine, where);
  } catch (...) {
    pending.pop_back();
    throw;
  }
  pending.pop_back();
  // A tensor Python made comes back with a Python object attached, and PyTorch may be about to
  // wrap it in a class of its own (a module's Parameter): hand over a twin with none. A result
  // that is one of the arguments (an operation in place) stays the tensor it is.
  size_t taken = 0;  // which of the operator's results is being taken
  auto bare = [&](const at::Tensor& tensor) {
    if (!tensor.defined()) return tensor;
    for (const auto& argument : arguments) {
      if (argument.isTensor() && argument.toTensor().unsafeGetTensorImpl() == tensor.unsafeGetTensorImpl()) return tensor;
    }
    if (const at::Tensor* base = base_of(schema, taken, arguments); base && info_of(tensor)) {  // a view: see make
      auto twin = at::detail::make_tensor<c10::TensorImpl>(c10::TensorImpl::VIEW, c10::Storage(tensor.storage()),
                                                           base->key_set(), tensor.dtype());
      twin.unsafeGetTensorImpl()->set_sizes_and_strides(tensor.sizes(), tensor.strides());
      twin.unsafeGetTensorImpl()->set_storage_offset(tensor.storage_offset());
      Info* made = info_of(tensor);
      if (!made->view) {
        made->view = true;
        made->root = root_of(*base);
      }
      return twin;
    }
    // (A tensor made in inference mode counts no versions, and its twin must not either.)
    auto* impl = tensor.unsafeGetTensorImpl();
    auto version = impl->is_inference() ? c10::VariableVersion(c10::VariableVersion::DISABLED) : c10::VariableVersion(0);
    return at::Tensor(impl->shallow_copy_and_detach(std::move(version), true));
  };
  std::vector<c10::IValue> results;
  auto take = [&](const py::object& value, const c10::Argument& slot) {
    if (value.is_none() && slot.type()->kind() == c10::TypeKind::TensorType) {
      results.emplace_back(at::Tensor());  // a result the operator does not produce here
      return;
    }
    auto converted = torch::jit::toIValue(value, slot.type());
    if (converted.isTensor()) {
      results.emplace_back(bare(converted.toTensor()));
    } else if (converted.isTensorList()) {
      c10::List<at::Tensor> list;
      for (const at::Tensor& tensor : converted.toTensorList()) list.push_back(bare(tensor));
      results.emplace_back(std::move(list));
    } else {
      results.push_back(std::move(converted));
    }
  };
  const auto& returns = schema.returns();
  if (returns.empty()) {
    // An operator that returns nothing (the fused optimizers, the gradient scaler's check): its effect is on its arguments.
  } else if (returns.size() == 1) {
    take(result, returns[0]);
  } else {
    auto values = result.cast<py::sequence>();
    for (; taken < returns.size(); taken++) take(values[taken], returns[taken]);
  }
  torch::jit::drop(*stack, count);
  for (auto& value : results) stack->push_back(std::move(value));
}

TORCH_LIBRARY_IMPL(_, MPS, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&fallback>());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm(const at::Tensor& input, at::TensorList hx, at::TensorList params,
                                                    bool has_biases, int64_t num_layers, double dropout, bool train,
                                                    bool bidirectional, bool batch_first);

// Which operators come here whole. PyTorch breaks an operator into smaller
// ones in two places: above a backend, the same way on every machine, and
// inside a backend's own code, where a Mac takes branches that a Linux build
// does not have (convolution, the recurrent layers, copying). So: an operator
// comes here whole unless PyTorch breaks it up above the backend, and those
// too when a Mac has an MPS kernel for them (`linear`). The Mac then does
// with each what it would have done with it anyway.
void take_operators() {
  static torch::Library library(torch::Library::IMPL, "aten", c10::DispatchKey::MPS, __FILE__, __LINE__);
  // Every operator PyTorch for macOS has an MPS kernel for.
  static const std::unordered_set<std::string> on_a_mac = {
#include "mps_ops.inc"  // this PyTorch version's list: bridge/ops/, made by gen_ops.py
  };
  auto& dispatcher = c10::Dispatcher::singleton();
  for (const auto& name : dispatcher.getAllOpNames()) {
    if (name.name.rfind("aten::", 0) != 0) continue;
    auto op = dispatcher.findOp(name);
    if (!op || !op->hasSchema()) continue;
    std::string full = name.name.substr(6) + (name.overload_name.empty() ? "" : "." + name.overload_name);
    const bool on_gpu = on_a_mac.count(full);
    if (op->hasKernelForDispatchKey(c10::DispatchKey::CompositeImplicitAutograd) && !on_gpu) continue;
    if (!on_gpu && !op->hasKernelForDispatchKey(c10::DispatchKey::CompositeImplicitAutograd) &&
        !op->hasKernelForDispatchKey(c10::DispatchKey::CompositeExplicitAutograd) &&
        !op->hasKernelForDispatchKey(c10::DispatchKey::CompositeExplicitAutogradNonFunctional)) {
      not_on_mps().insert(&op->schema());
    }
    library.impl(full.c_str(), torch::CppFunction::makeFromBoxedFunction<&fallback>());
  }
  // The two operators this file takes above the backend (attention, lstm, further down) come to the
  // backend itself inside torch.inference_mode(), which skips the autograd key they are taken at:
  // attention then goes to the Mac whole, as it does whenever no gradient is wanted.
  library.impl("scaled_dot_product_attention", torch::CppFunction::makeFromBoxedFunction<&fallback>());
  library.impl("lstm.input", TORCH_FN(lstm));
}

// Attention takes its MPS path inside the operator, after asking a table of backends that, on
// Linux, stops at the name "mps": left to itself the operator would be broken up here, into
// the generic matmul and softmax, where a Mac runs its GPU's own attention. Which of the two a
// Mac takes, and for which arguments (dropout, grouped queries, a mask's type), is the Mac's
// PyTorch's business and changes with the version. So:
// - when no gradient is wanted, the operator goes to the Mac whole, and the Mac decides;
// - when one is, it is broken up here as on a Mac, where a backward pass needs the pieces
//   (aten/src/ATen/native/transformers/attention.cpp).
at::Tensor attention(const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
                     const std::optional<at::Tensor>& mask, double dropout, bool causal,
                     std::optional<double> scale, bool gqa) {
  const bool grad = at::GradMode::is_enabled() && (query.requires_grad() || key.requires_grad() || value.requires_grad());
  if (!grad) {
    static auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::scaled_dot_product_attention", "");
    torch::jit::Stack stack{query, key, value, mask ? c10::IValue(*mask) : c10::IValue(), dropout, causal,
                            scale ? c10::IValue(*scale) : c10::IValue(), gqa};
    fallback(op, &stack);
    return stack.at(0).toTensor();
  }
  if (query.dim() == 3) {  // unbatched: give it a batch dimension, as the original does
    auto batched = mask && mask->dim() == 3 ? std::optional<at::Tensor>(mask->unsqueeze(0)) : mask;
    return attention(query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0), batched, dropout, causal, scale, gqa)
        .squeeze(0);
  }
  std::optional<at::Tensor> additive = mask;
  if (mask && mask->dtype() == at::kBool) {  // a boolean mask says what to keep
    constexpr double neg_inf = -std::numeric_limits<double>::infinity();
    additive = at::where(*mask, 0.0,
                         at::scalar_tensor(neg_inf, at::TensorOptions().dtype(query.dtype()).device(mask->device())));
  }
  return std::get<0>(at::_scaled_dot_product_attention_math(query, key, value, additive, dropout, causal,
                                                            std::nullopt, scale, gqa));
}

TORCH_LIBRARY_IMPL(aten, AutogradMPS, m) {
  m.impl("scaled_dot_product_attention", TORCH_FN(attention));
}

// An LSTM over a whole sequence is one operator of the GPU's on a Mac: PyTorch's lstm hands it
// to _lstm_mps, in a branch that is compiled for macOS only (`#ifdef USE_MPS`,
// aten/src/ATen/native/RNN.cpp). Left to itself the operator would be broken up here, cell by
// cell, into linear, sigmoid and tanh: the same mathematics, and other bits (seen: an LSTM
// differed from the Mac's in 557 elements of 672, by up to 1.2e-7). So it goes to the Mac as a
// Mac sends it to its GPU. _lstm_mps has a gradient of its own (lstm_mps_backward), which the
// Mac runs too. The packed form and the other recurrent layers have no such branch on a Mac, and
// are broken up there as here.
std::tuple<at::Tensor, at::Tensor, at::Tensor> lstm(const at::Tensor& input, at::TensorList hx, at::TensorList params,
                                                    bool has_biases, int64_t num_layers, double dropout, bool train,
                                                    bool bidirectional, bool batch_first) {
  if (!input.is_mps()) {  // (only a hidden state or a weight is on the GPU: PyTorch's own code, which fails as on a Mac)
    return at::native::lstm(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
  }
  TORCH_CHECK(hx.size() == 2, "lstm expects two hidden states");
  auto output = at::_lstm_mps(input, hx, params, has_biases, num_layers, dropout, train, bidirectional, batch_first);
  return std::make_tuple(std::move(std::get<0>(output)), std::move(std::get<1>(output)), std::move(std::get<2>(output)));
}

// For the autograd key and for the backend's: with torch.inference_mode() an operation skips the
// first, and without a kernel on the second PyTorch would take its own code for every backend.
TORCH_LIBRARY_IMPL(aten, AutogradMPS, m) {
  m.impl("lstm.input", TORCH_FN(lstm));
}

py::object call_mac(const char* name, const py::tuple& arguments) {
  auto& s = state();
  if (!s.call || s.call.is_none()) throw std::runtime_error(std::string(name) + ": this is the Mac's to answer, through the bridge (import bridge)");
  return s.call(name, *arguments);
}

bool reachable() {
  py::gil_scoped_acquire lock;
  auto& s = state();
  if (s.fd >= 0) return true;
  if (!s.connect || s.connect.is_none()) return false;
  try {
    s.connect();
  } catch (py::error_already_set&) {
    return false;
  }
  return s.fd >= 0;
}

caffe2::TypeMeta type_of(const py::object& dtype) {
  if (!THPDtype_Check(dtype.ptr())) throw py::type_error("expected a torch.dtype");
  return c10::scalarTypeToTypeMeta(reinterpret_cast<THPDtype*>(dtype.ptr())->scalar_type);
}

}  // namespace

// --- for client.py ---------------------------------------------------------------------

PYBIND11_MODULE(bridge_native, m) {
  take_operators();
  // Python's side: what is not routine (slow), questions that are not operators (call), and how
  // to open the connection (connect), which happens when something first needs the Mac.
  m.def("start", [](py::object slow, py::object call, py::object connect, bool learn) {
    auto& s = state();
    s.slow = std::move(slow);
    s.call = std::move(call);
    s.connect = std::move(connect);
    s.learn = learn;
  });
  // The connection is open: the socket to the Mac.
  m.def("attach", [](int fd, size_t flush_at, uint32_t ahead) {
    auto& s = state();
    s.fd = fd;
    s.flush_at = flush_at;
    s.widest = ahead;
    s.window = std::min<uint32_t>(1024, ahead);
    s.ahead = 0;
  });
  // The program has just waited for the Mac: it is not ahead of it.
  m.def("caught_up", []() { state().ahead = 0; });
  // The connection is closed, whether the program ended or the bridge gave the connection up. The Mac's
  // process for it ends, and with it every tensor and every signature it knew: the tensors made on
  // it are of a generation that is over (using one says so), and what was learnt about the Mac's side
  // is forgotten, since the next connection is to a process that knows nothing of it.
  m.def("stop", []() {
    auto& s = state();
    if (s.fd >= 0) {
      try {
        flush();
      } catch (const std::exception&) {  // the Mac has gone already: nothing left to tell it
      }
    }
    s.fd = -1;
    generation++;
    {
      std::lock_guard<std::mutex> lock(*freed_lock);
      freed.clear();
    }
    s.out.clear();
    s.remembered.clear();
    s.next_signature = 1;
  });
  // In a process made by fork(): the socket, the bytes not yet sent and the tensors are the
  // parent's. Nothing of them is touched; what was learnt about the Mac's side is forgotten,
  // because the connection this process will open starts from nothing.
  m.def("forked", []() {
    auto& s = state();
    generation++;
    own_generation = generation.load();
    freed_lock = new std::mutex;  // the old one may have been held by a thread that does not exist here
    freed.clear();
    s.fd = -1;
    s.out.clear();
    s.remembered.clear();
    s.next_signature = 1;
    pending.clear();
  });
  m.def("connected", []() { return state().fd >= 0; });
  // A tensor of this shape and type (strides None for the usual, in-order layout), under a new number or the one given.
  m.def("fake", [](const std::vector<int64_t>& sizes, const std::optional<std::vector<int64_t>>& strides,
                   const py::object& dtype, uint32_t handle, int64_t offset) {
    auto& s = state();
    return make(sizes, strides ? std::optional<c10::IntArrayRef>(*strides) : std::nullopt, type_of(dtype),
                handle ? handle : s.next_handle++, 0, nullptr, offset);
  }, py::arg("sizes"), py::arg("strides"), py::arg("dtype"), py::arg("handle") = 0, py::arg("offset") = 0);
  // The Mac's number for a tensor; 0 if it is not one of these.
  m.def("handle", [](const at::Tensor& tensor) {
    Info* info = info_of(tensor);
    return info ? info->handle : 0u;
  });
  // For the operations that reshape their argument in place.
  m.def("set_geometry", [](const at::Tensor& tensor, const std::vector<int64_t>& sizes,
                           const std::vector<int64_t>& strides, int64_t offset) { reshape(tensor, sizes, strides, 0, offset); });
  // What a storage of ours stands for: the Mac's number of a tensor that is in that memory (the tensor
  // the storage is of, or for a storage that is no tensor's own any more a view that still reads it;
  // 0 if there is none). None for any other storage. Saving a tensor starts from its storage, and
  // fetches all of the memory that tensor is in on the Mac: not the tensor's values, which are as
  // much of it as its shape of the moment covers.
  m.def("storage_handle", [](py::handle storage) -> py::object {
    if (!THPStorage_Check(storage.ptr())) return py::none();
    Info* info = info_of(THPStorage_Unpack(storage.ptr()));
    if (!info) return py::none();
    return py::int_(info->handle ? info->handle : info->via);
  });
  // A view's place in the storage it shares: the storage of the tensor whose data it is (root_of),
  // or None for a tensor that is nobody's view. Saving writes that storage, once.
  m.def("root", [](const at::Tensor& tensor) -> py::object {
    if (!tensor.defined() || !tensor.is_mps() || !tensor.has_storage()) return py::none();
    const auto& data = tensor.storage().data_ptr();
    if (data.get_deleter() != &gone) return py::none();
    auto* info = static_cast<Info*>(data.get_context());
    if (!info->view || !info->root) return py::none();
    // (The tensor that memory was first of may have been given other memory since: see rebind.)
    if (Info* shared = info_of(info->root); shared && !shared->handle) shared->via = info->handle;
    return py::reinterpret_steal<py::object>(THPStorage_Wrap(info->root));
  });
  // x.set_(source): the tensor is given the memory of that tensor, or of the tensor that storage is of, at
  // this place in it; with None, memory of its own and nothing in it.
  m.def("rebind", [](const at::Tensor& tensor, py::handle source, const std::vector<int64_t>& sizes,
                     const std::vector<int64_t>& strides, int64_t offset) {
    c10::Storage shared;
    if (THPStorage_Check(source.ptr())) {
      c10::Storage given = THPStorage_Unpack(source.ptr());
      Info* info = info_of(given);
      if (!info) throw std::runtime_error("set_: not a storage of the bridge's");
      shared = info->view && info->root ? info->root : given;
    } else if (!source.is_none()) {
      auto from = source.cast<at::Tensor>();
      if (info_of(from)) shared = root_of(from);
    }
    rebind(tensor, std::move(shared), sizes, strides, offset);
  });
  // A tensor Python made by asking the Mac for a view (loading: install.py, _loading) is a view of `base`.
  m.def("view_of", [](const at::Tensor& tensor, const at::Tensor& base) {
    Info* info = info_of(tensor);
    if (!info || info->view || !info_of(base)) return;
    info->view = true;
    info->root = root_of(base);
  });
  // The Mac's number for a generator "on mps"; -1 for a generator of the CPU's.
  m.def("generator", [](const at::Generator& generator) -> int64_t {
    auto* remote = dynamic_cast<RemoteGenerator*>(generator.unsafeGetGeneratorImpl());
    return remote ? int64_t(remote->id) : -1;
  });
  // Numbers set aside for the results of an operation the Mac runs before saying what came of it.
  m.def("handles", [](uint32_t count) {
    auto& s = state();
    uint32_t first = s.next_handle;
    s.next_handle += count;
    return first;
  });
  m.def("signature", []() { return state().next_signature++; });
  // Whether this is the tensor made last and its data is its own (not a view of another's): then
  // nothing else can be looking at its data yet, and new data for all of it can simply take its place.
  m.def("fresh", [](const at::Tensor& tensor) {
    Info* info = info_of(tensor);
    return info && !info->view && info->handle == state().next_handle - 1;
  });
  // Bytes for the Mac, in order with the operations. Several buffers go as one piece: a frame
  // written in parts must not have another thread's operation land in the middle of it.
  m.def("write", [](const py::args& buffers) {
    auto& s = state();
    if (s.fd < 0) throw std::runtime_error("the bridge is not connected to the Mac");
    tell_freed();
    for (auto item : buffers) {
      py::buffer_info info = py::reinterpret_borrow<py::buffer>(item).request();
      const char* data = static_cast<const char*>(info.ptr);
      s.out.insert(s.out.end(), data, data + info.size * info.itemsize);
    }
    if (s.out.size() >= s.flush_at) flush();
  });
  m.def("flush", []() {
    if (state().fd >= 0) flush();
  });
  // The operation Python has just worked out is routine from now on: its signature
  // on the Mac, and for each result None, the argument handed back, or a tensor to copy the shape of.
  m.def("remember", [](uint32_t signature, const py::list& outs, bool list) {
    auto& s = state();
    if (pending.empty() || pending.back().empty()) return false;
    Remembered how{signature, list, {}};
    for (auto item : outs) {
      Out out{Out::None};
      if (py::isinstance<py::int_>(item)) {
        out.kind = Out::Argument;
        out.argument = item.cast<uint32_t>();
      } else if (!item.is_none()) {
        at::Tensor model;
        if (py::isinstance<py::tuple>(item)) {  // (argument, the tensor it has become)
          auto pair = item.cast<py::tuple>();
          out.kind = Out::Argument;
          out.argument = pair[0].cast<uint32_t>();
          out.reshaped = true;
          model = pair[1].cast<at::Tensor>();
        } else {
          out.kind = Out::Fresh;
          model = item.cast<at::Tensor>();
        }
        out.sizes = model.sizes().vec();
        out.strides = model.strides().vec();
        out.offset = model.storage_offset();
        out.type = model.dtype();
        out.tag = tag_of(model.sizes(), model.strides(), model.dtype());
      }
      how.outs.push_back(std::move(out));
    }
    s.remembered.insert_or_assign(pending.back(), std::move(how));
    return true;
  });
  m.def("counts", []() {
    auto& s = state();
    return std::make_tuple(s.sent, s.missed, s.remembered.size());
  });
}
