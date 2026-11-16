// The Mac end's hot path, in C++: runs the operations in the bytes the
// container sent, straight through PyTorch's dispatcher, with no Python per
// operation. server.py keeps everything rare (data, questions, definitions)
// and hands this the received bytes; build-mac.sh builds it.
//
// Frames are wire.py's: O (an operation: signature, then its tensors' and
// results' numbers) and F (numbers to forget). It stops at the first frame
// that is not whole, or is of another kind, and says where.
//
// The bytes are a container's, and a container is not trusted. policy.py has
// checked a signature before it is defined here; what this file reads memory
// by is checked here as well, where it is read: a signature's places are
// inside the frame it declares (define), and no tensor reaches outside its
// own memory (usable), whichever operator made it.
#include <torch/extension.h>

#include <pybind11/stl.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/mps/MPSStream.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/jit/python/pybind_utils.h>

#include <array>
#include <cctype>
#include <chrono>
#include <cstring>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

namespace {

uint32_t u32(const char* p) {
  uint32_t v;
  std::memcpy(&v, p, 4);  // the wire is little-endian, and so are the Macs this runs on
  return v;
}

// The most tensors, results and real numbers an operation may have: policy.py's MOST.
constexpr uint32_t kMost = 4096;

// Whether a tensor is plain memory of the GPU's or the CPU's that it does not reach outside of:
// policy.py's usable, here for the operations Python does not see. PyTorch has operators that
// make a view without checking it against the memory (_reshape_alias), and one that fails half
// way leaves its argument with the new shape and the old memory (resize_): such a tensor is
// never given to an operator, nor back to the container.
bool usable(const at::Tensor& tensor) {
  if (!tensor.defined()) return false;
  auto* impl = tensor.unsafeGetTensorImpl();
  if (impl->layout() != c10::kStrided || impl->is_quantized() || impl->is_nested() || !impl->has_storage()) return false;
  auto device = impl->device();
  if (!(device.is_mps() || device.is_cpu()) || (device.has_index() && device.index() != 0)) return false;
  if (impl->numel() == 0) return true;
  int64_t reach = impl->storage_offset();
  if (reach < 0) return false;
  auto sizes = impl->sizes();
  auto strides = impl->strides();
  for (size_t i = 0; i < sizes.size(); i++) {
    int64_t span;
    if (strides[i] < 0 || __builtin_mul_overflow(sizes[i] - 1, strides[i], &span) ||
        __builtin_add_overflow(reach, span, &reach)) {
      return false;
    }
  }
  int64_t bytes;
  if (__builtin_add_overflow(reach, int64_t(1), &reach) ||
      __builtin_mul_overflow(reach, static_cast<int64_t>(impl->itemsize()), &bytes)) {
    return false;
  }
  return static_cast<uint64_t>(bytes) <= impl->storage().nbytes();
}

// An operation with its constants in place: only the tensors change from one call to the next.
struct Signature {
  c10::OperatorHandle op;
  std::string name;
  std::vector<c10::IValue> arguments;                 // every argument of the schema; tensors' places empty
  std::vector<std::pair<uint32_t, uint32_t>> tensors;  // (argument, which of the frame's numbers)
  std::vector<std::array<uint32_t, 3>> lists;          // lists of tensors: (argument, first number, how many)
  uint32_t inputs;
  uint32_t outputs;
  uint32_t numbers;  // how many real numbers a frame carries, whichever of them the arguments use
  // Real numbers that come with each call (a step size, a learning rate), after the frame's
  // numbers: as the number itself, or where the operator takes a tensor as the tensor PyTorch
  // makes of a Python number (marked as one), or as a 0-dim tensor of a given type.
  struct Real {
    uint32_t argument, which;
    enum Kind { Number, Wrapped, Tensor } kind;
    c10::ScalarType type;
  };
  std::vector<Real> reals;
  bool gpu = false;    // one of its constants, or of its real numbers, is a tensor on the GPU
  std::string broken;  // why it could not be defined, if it could not: its operations fail with this
};

struct Executor {
  std::unordered_map<uint32_t, at::Tensor> tensors;
  std::unordered_map<uint32_t, Signature> signatures;
  std::unordered_set<std::string> allowed;  // the operators that may be defined, "aten::name.overload": policy.py's list
  std::string failed;  // the first operation that failed while nobody was waiting
  uint64_t count = 0;
  bool paused = false;  // run stopped for its time, with whole frames still to run

  // `arguments` are the positional ones, None where a tensor goes; `which`
  // says, for each, which of the frame's numbers it is, or -1 for a constant.
  void define(uint32_t id, const std::string& name, const std::string& overload, const py::list& arguments,
              const std::vector<int>& which, const std::vector<std::array<uint32_t, 3>>& lists,
              const py::dict& keywords, const std::map<std::string, uint32_t>& named,
              const std::vector<std::tuple<std::string, uint32_t, std::string>>& reals, uint32_t inputs,
              uint32_t outputs, uint32_t carried) {
    if (!allowed.count(overload.empty() ? name : name + "." + overload)) {
      throw std::runtime_error(name + " is not in the list of operators the Mac end runs for a container");
    }
    if (inputs > kMost || outputs > kMost || carried > kMost) throw std::runtime_error(name + ": too many tensors or numbers");
    auto op = c10::Dispatcher::singleton().findSchemaOrThrow(name.c_str(), overload.c_str());
    const auto& parameters = op.schema().arguments();
    // Every place the signature names is one of the frame's: its operations read no other.
    for (int number : which) {
      if (number >= static_cast<int>(inputs)) throw std::runtime_error(name + ": a tensor outside the frame");
    }
    for (const auto& list : lists) {
      if (list[0] >= parameters.size() || list[1] > inputs || list[2] > inputs - list[1]) {
        throw std::runtime_error(name + ": a list of tensors outside the frame");
      }
    }
    for (const auto& [keyword, number] : named) {
      if (number >= inputs) throw std::runtime_error(name + ": a tensor outside the frame");
    }
    for (const auto& real : reals) {
      if (std::get<1>(real) >= carried) throw std::runtime_error(name + ": a real number outside the frame");
    }
    // As many results as the operator gives: one list of tensors, of any length, or one value each.
    const auto& returns = op.schema().returns();
    const bool whole = returns.size() == 1 && returns[0].type()->kind() == c10::TypeKind::ListType;
    if (!whole && outputs != returns.size()) throw std::runtime_error(name + ": not the operator's number of results");
    Signature s{op, name, std::vector<c10::IValue>(parameters.size()), {}, lists, inputs, outputs, carried};
    // A plain number where the operator takes a tensor (x + 1.5) becomes what PyTorch makes of it in
    // a program: a tensor marked as a number, which counts for less when the result's type is decided.
    torch::jit::ToIValueAllowNumbersAsTensors numbers(true);
    std::vector<bool> is_list(parameters.size(), false);
    for (const auto& list : lists) is_list.at(list[0]) = true;
    std::vector<bool> given(parameters.size(), false);
    if (arguments.size() > parameters.size() || which.size() != arguments.size()) {
      throw std::runtime_error(name + ": more arguments than the operator takes");
    }
    for (const auto& [place, which_real, type] : reals) {  // by position ("3") or by name
      size_t i = 0;
      if (!place.empty() && std::isdigit(static_cast<unsigned char>(place[0]))) {
        i = std::stoul(place);
      } else {
        while (i < parameters.size() && parameters[i].name() != place) i++;
      }
      if (i >= parameters.size()) throw std::runtime_error(name + ": no argument " + place);
      Signature::Real real{static_cast<uint32_t>(i), which_real, Signature::Real::Number, c10::ScalarType::Float};
      if (parameters[i].type()->kind() == c10::TypeKind::TensorType) {
        real.kind = type.empty() ? Signature::Real::Wrapped : Signature::Real::Tensor;
        if (!type.empty()) {
          auto dtype = py::module_::import("torch").attr(type.c_str());
          if (!THPDtype_Check(dtype.ptr())) throw std::runtime_error(name + ": not a type: " + type);
          real.type = reinterpret_cast<THPDtype*>(dtype.ptr())->scalar_type;
          if (real.type == c10::ScalarType::Double) real.type = c10::ScalarType::Float;  // MPS has no float64
        }
      }
      s.gpu = s.gpu || real.kind == Signature::Real::Tensor;
      s.reals.push_back(real);
      given[i] = true;
    }
    for (size_t i = 0; i < arguments.size(); i++) {
      if (given[i] && which[i] >= 0) throw std::runtime_error(name + ": " + parameters[i].name() + " is given twice");
      if (which[i] >= 0) {
        s.tensors.emplace_back(i, which[i]);
      } else if (is_list[i] || given[i]) {
        // filled in at each call: a list of tensors, or a real number
      } else {
        s.arguments[i] = torch::jit::toIValue(arguments[i], parameters[i].type());
      }
      given[i] = true;
    }
    for (auto item : keywords) {
      auto keyword = item.first.cast<std::string>();
      size_t i = 0;
      while (i < parameters.size() && parameters[i].name() != keyword) i++;
      if (i == parameters.size()) throw std::runtime_error(name + ": no argument named " + keyword);
      if (given[i]) throw std::runtime_error(name + ": " + keyword + " is given twice");
      s.arguments[i] = torch::jit::toIValue(item.second, parameters[i].type());
      given[i] = true;
    }
    for (const auto& [keyword, which_number] : named) {  // tensors given by name: out=
      size_t i = 0;
      while (i < parameters.size() && parameters[i].name() != keyword) i++;
      if (i == parameters.size()) throw std::runtime_error(name + ": no argument named " + keyword);
      if (given[i]) throw std::runtime_error(name + ": " + keyword + " is given twice");
      s.tensors.emplace_back(i, which_number);
      given[i] = true;
    }
    for (size_t i = 0; i < parameters.size(); i++) {
      if (given[i]) continue;
      if (!parameters[i].default_value()) throw std::runtime_error(name + ": missing " + parameters[i].name());
      s.arguments[i] = *parameters[i].default_value();
    }
    for (const auto& constant : s.arguments) {  // a number the container had as a 0-dim tensor is one of the GPU's here
      s.gpu = s.gpu || (constant.isTensor() && constant.toTensor().defined() && constant.toTensor().is_mps());
    }
    signatures.insert_or_assign(id, std::move(s));
  }

  // A signature that could not be defined (an operator this PyTorch has not, an argument it does
  // not take): its operations are still frames of a known length, and each fails with the reason.
  void define_broken(uint32_t id, const std::string& name, const std::string& why, uint32_t inputs, uint32_t outputs,
                     uint32_t carried) {
    if (inputs > kMost || outputs > kMost || carried > kMost) throw std::runtime_error(name + ": too many tensors or numbers");
    auto op = c10::Dispatcher::singleton().findSchemaOrThrow("aten::alias", "");
    Signature s{op, name, {}, {}, {}, inputs, outputs, carried};
    s.broken = why.empty() ? "the signature was refused" : why;
    signatures.insert_or_assign(id, std::move(s));
  }

  const at::Tensor& fetch(uint32_t handle) {
    auto found = tensors.find(handle);
    if (found == tensors.end()) throw std::runtime_error("tensor " + std::to_string(handle) + " is not here");
    if (!usable(found->second)) {
      tensors.erase(found);
      throw std::runtime_error("tensor " + std::to_string(handle) + " is not memory the Mac end lets a container use (it "
                               "reaches outside what it was given): an operation on it failed half way, or made it so");
    }
    return found->second;
  }

  void store(uint32_t handle, at::Tensor tensor) {
    if (!tensor.defined()) {  // a result the operator did not produce this time (a gradient nobody asked for)
      tensors.erase(handle);
      return;
    }
    if (!usable(tensor)) {
      throw std::runtime_error("a result is not memory the Mac end lets a container use (it reaches outside what it was "
                               "given, or is not a plain tensor of the GPU or the CPU)");
    }
    tensors.insert_or_assign(handle, std::move(tensor));
  }

  void one(const Signature& s, const char* numbers) {
    torch::jit::Stack stack = s.arguments;
    // The tensors a container names must not all be on the CPU: PyTorch would then run the CPU's
    // kernel on them, and the Mac end computes with the GPU's (policy.py, placed).
    bool named = false, gpu = s.gpu;
    for (const auto& [argument, which] : s.tensors) {
      const at::Tensor& tensor = fetch(u32(numbers + 4 * which));
      named = true;
      gpu = gpu || tensor.is_mps();
      stack[argument] = tensor;
    }
    for (const auto& list : s.lists) {
      c10::List<at::Tensor> members;
      for (uint32_t i = 0; i < list[2]; i++) {
        const at::Tensor& tensor = fetch(u32(numbers + 4 * (list[1] + i)));
        named = true;
        gpu = gpu || tensor.is_mps();
        members.push_back(tensor);
      }
      stack[list[0]] = c10::IValue(std::move(members));
    }
    if (named && !gpu) {
      throw std::runtime_error("Refused: an operation on tensors of the CPU's alone: the Mac end computes on the GPU, and "
                               "takes a tensor of the CPU's only next to one of the GPU's");
    }
    const char* reals = numbers + 4 * (size_t(s.inputs) + s.outputs);
    for (const auto& real : s.reals) {
      double value;
      std::memcpy(&value, reals + 8 * real.which, 8);
      if (real.kind == Signature::Real::Number) {
        stack[real.argument] = value;
      } else if (real.kind == Signature::Real::Wrapped) {
        at::Tensor number = at::scalar_to_tensor(value);
        number.unsafeGetTensorImpl()->set_wrapped_number(true);
        stack[real.argument] = std::move(number);
      } else {
        stack[real.argument] = at::scalar_tensor(value, at::TensorOptions().dtype(real.type).device(at::kMPS));
      }
    }
    s.op.callBoxed(stack);
    const char* results = numbers + 4 * s.inputs;
    if (stack.size() == 1 && stack[0].isTensorList()) {  // one list of them, even of one
      auto list = stack[0].toTensorVector();
      // As many as the container made tensors for when it first ran the operation: with fewer, or
      // more, the tensors it goes on to use would not be the ones the operator gave.
      if (list.size() != s.outputs) {
        throw std::runtime_error("the operator gave " + std::to_string(list.size()) + " tensors, and the container made " +
                                 std::to_string(s.outputs) + " for them");
      }
      for (uint32_t i = 0; i < s.outputs; i++) {
        store(u32(results + 4 * i), list[i]);
      }
    } else if (stack.size() == s.outputs) {  // one value per result
      for (uint32_t i = 0; i < s.outputs; i++) {
        if (!stack[i].isTensor()) continue;  // a result the operator did not produce this time
        store(u32(results + 4 * i), std::move(stack[i]).toTensor());
      }
    } else {
      throw std::runtime_error("unexpected results");
    }
  }

  // Runs every whole O and F frame from `at` on; returns where it stopped.
  //
  // With `commit`, what was just encoded is handed to the GPU before
  // returning: the caller is about to wait for more bytes, and the GPU can
  // work meanwhile. Left to itself, MPS keeps it until something asks for a value.
  //
  // It runs for `budget` seconds at most, and says so (`paused`): a program that was killed can
  // leave a long queue behind it, and the caller looks, between one stretch and the next, whether
  // the program is still there to want it.
  size_t run(const py::buffer& bytes, size_t at, bool commit, double budget) {
    uint64_t before = count;
    paused = false;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>(budget);
    py::buffer_info info = bytes.request();
    const char* data = static_cast<const char*>(info.ptr);
    size_t end = static_cast<size_t>(info.size * info.itemsize);
    while (end - at >= 5) {
      char kind = data[at];
      uint32_t n = u32(data + at + 1);
      if (kind == 'O') {
        auto found = signatures.find(n);
        if (found == signatures.end()) throw std::runtime_error("an operation with a signature never defined");
        const Signature& s = found->second;
        size_t after = at + 5 + 4 * (size_t(s.inputs) + s.outputs) + 8 * size_t(s.numbers);
        if (after > end) break;
        try {
          if (!s.broken.empty()) throw std::runtime_error(s.broken);
          one(s, data + at + 5);
        } catch (const c10::Error& e) {  // "operator: Kind: message", as server.py reports Python's
          const char* kind = dynamic_cast<const c10::NotImplementedError*>(&e) ? "NotImplementedError"
                             : dynamic_cast<const c10::TypeError*>(&e) ? "TypeError"
                             : dynamic_cast<const c10::ValueError*>(&e) ? "ValueError"
                             : dynamic_cast<const c10::IndexError*>(&e) ? "IndexError"
                             : dynamic_cast<const c10::OutOfMemoryError*>(&e) ? "OutOfMemoryError" : "RuntimeError";
          if (failed.empty()) failed = s.name + ": " + kind + ": " + e.what_without_backtrace();
        } catch (const std::exception& e) {
          if (failed.empty()) failed = s.name + ": RuntimeError: " + e.what();
        }
        count++;
        at = after;
        if ((count & 15) == 0 && std::chrono::steady_clock::now() > deadline) {
          paused = true;
          break;
        }
      } else if (kind == 'F') {
        size_t after = at + 5 + 4 * size_t(n);
        if (after > end) break;
        for (uint32_t i = 0; i < n; i++) tensors.erase(u32(data + at + 5 + 4 * i));
        at = after;
      } else {
        break;
      }
    }
    if (commit && count != before) {
      at::mps::getDefaultMPSStream()->synchronize(at::mps::SyncType::COMMIT);
    }
    return at;
  }

  std::string take_failed() {
    std::string was;
    was.swap(failed);
    return was;
  }
};

}  // namespace

PYBIND11_MODULE(_executor, m) {
  m.def("usable", &usable);
  py::class_<Executor>(m, "Executor")
      .def(py::init([](std::unordered_set<std::string> allowed) {
        auto executor = std::make_unique<Executor>();
        executor->allowed = std::move(allowed);
        return executor;
      }))
      .def("define", &Executor::define)
      .def("define_broken", &Executor::define_broken)
      .def("run", &Executor::run)
      .def("take_failed", &Executor::take_failed)
      .def("get", [](Executor& e, uint32_t handle) { return e.fetch(handle); })
      .def("put", [](Executor& e, uint32_t handle, const at::Tensor& t) { e.store(handle, t); })
      .def("forget", [](Executor& e, uint32_t handle) { e.tensors.erase(handle); })
      .def_readonly("paused", &Executor::paused)
      .def_readonly("count", &Executor::count);
}
