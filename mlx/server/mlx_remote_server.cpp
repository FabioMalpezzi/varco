// mlx-remote-server: the Mac end of the remote gpu backend of MLX.
//
// A build of MLX in a Linux container (backend/remote) sends its graphs here;
// this process rebuilds them with mlx's own import_function and runs them on
// the Mac's GPU with the real MLX it is linked to. It keeps handle -> array
// per connection and does nothing but: store uploaded data, import and
// evaluate MLX graphs, send array data back, free handles, and read the
// arrays of the files of weights under the directories it is started with.
// With --custom-kernels it also compiles and runs the Metal kernels the
// containers make with mx.fast.metal_kernel.
//
// Usage: mlx-remote-server [--port N | --unix PATH] [-v] [--timing]
//        [--compile-graphs] [--slow-build F] [--custom-kernels]
//        [--models DIR]... [--map CONTAINER_DIR=DIR]...
// Listens on 127.0.0.1 only (default port 47800) or on a Unix socket.
// What a container marks urgent, a small graph whose result its program is
// about to read, is run on the cpu and answered ahead of the work queued
// before it when that gives the same bytes (see "Urgent graphs" below).
// --models DIR: the files under DIR may be read here for a container that
// sees them under the same path; --map: for one that sees DIR as
// CONTAINER_DIR. Without either no file is ever read here.
// --custom-kernels: off unless given, since the source of a kernel is code
// that a container sends and this Mac's GPU runs (see "Custom kernels"
// below for what that code can and cannot do).

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/param.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mlx/mlx.h"
#include "mlx/backend/metal/device.h"
#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

#include "../upstream/mlx/backend/remote/protocol.h"

namespace mx = mlx::core;
using namespace mlx_remote;

namespace {

bool verbose = false;
// With --timing: where the time of the Mac side goes, printed at each SYNC
// and READ (microseconds since the previous line).
bool timing = false;
using Clock = std::chrono::steady_clock;
struct Timers {
  uint64_t receive{0}, queued{0}, key{0}, inputs{0}, parse{0}, build{0},
      eval{0}, store{0}, other{0};
  // Of parse: making the primitives that were not known yet, and how many.
  uint64_t import{0}, imports{0};
  uint64_t messages{0}, graphs{0}, nodes{0}, bytes{0};
} timers;
uint64_t ns_since(Clock::time_point t) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t)
      .count();
}
struct Lap {
  uint64_t& total;
  Clock::time_point start;
  explicit Lap(uint64_t& total) : total(total) {
    if (timing) {
      start = Clock::now();
    }
  }
  ~Lap() {
    if (timing) {
      total += ns_since(start);
    }
  }
};
std::string unix_path;

///////////////////////////////////////////////////////////////////////////////
// Jobs: connection threads read whole messages, the main thread runs them.
// All MLX work happens on the main thread, since streams belong to a thread.
///////////////////////////////////////////////////////////////////////////////

struct Job {
  int conn{0};
  int fd{-1};
  uint8_t type{0}; // 0: connection closed
  std::unique_ptr<char[]> payload;
  size_t size{0};
  // For --timing: when its header arrived and when it was queued.
  Clock::time_point arrived, queued;
  // Its number among the GRAPH, OP and LOAD messages of its connection, as
  // they arrived (the n of the DONE frames and of SYNC).
  uint64_t seq{0};
  // A GRAPH that says it is urgent (see "Urgent graphs").
  bool urgent{false};
};

std::mutex queue_mtx;
std::condition_variable queue_cv;
std::deque<Job> queue;
// The urgent jobs in the queue.
std::atomic<int> urgent_queued{0};

void push(Job job) {
  {
    std::lock_guard lk(queue_mtx);
    queue.push_back(std::move(job));
  }
  queue_cv.notify_one();
}

bool read_all(int fd, void* p, size_t n) {
  auto* c = static_cast<char*>(p);
  while (n > 0) {
    auto m = ::recv(fd, c, n, 0);
    if (m < 0 && errno == EINTR) {
      continue;
    }
    if (m <= 0) {
      return false;
    }
    c += m;
    n -= m;
  }
  return true;
}

bool write_all(int fd, const void* p, size_t n) {
  auto* c = static_cast<const char*>(p);
  while (n > 0) {
    auto m = ::send(fd, c, n, 0);
    if (m < 0 && errno == EINTR) {
      continue;
    }
    if (m <= 0) {
      return false;
    }
    c += m;
    n -= m;
  }
  return true;
}

bool write_iov(int fd, std::vector<iovec>& iov) {
  size_t first = 0;
  while (first < iov.size()) {
    auto m = ::writev(
        fd, iov.data() + first, static_cast<int>(std::min<size_t>(iov.size() - first, 64)));
    if (m < 0 && errno == EINTR) {
      continue;
    }
    if (m <= 0) {
      return false;
    }
    auto left = static_cast<size_t>(m);
    while (first < iov.size() && left >= iov[first].iov_len) {
      left -= iov[first].iov_len;
      first++;
    }
    if (left > 0) {
      iov[first].iov_base = static_cast<char*>(iov[first].iov_base) + left;
      iov[first].iov_len -= left;
    }
  }
  return true;
}

void reader_thread(int conn, int fd) {
  uint64_t seq = 0;
  while (true) {
    char head[kFrameHeader];
    if (!read_all(fd, head, sizeof(head))) {
      break;
    }
    Job job;
    job.conn = conn;
    job.fd = fd;
    job.type = static_cast<uint8_t>(head[0]);
    uint64_t len;
    std::memcpy(&len, head + 1, sizeof(len));
    if (job.type == 0 || job.type > kLastMsg || len > (1ull << 37) ||
        (job.type == kKernel && len > kMaxKernelSource + (64 << 10))) {
      std::cerr << "conn " << conn << ": malformed frame, closing" << std::endl;
      break;
    }
    job.size = len;
    if (timing) {
      job.arrived = Clock::now();
    }
    job.payload.reset(new char[len > 0 ? len : 1]);
    if (!read_all(fd, job.payload.get(), len)) {
      break;
    }
    if (timing || verbose) {
      job.queued = Clock::now();
    }
    if (job.type == kGraph || job.type == kOp || job.type == kLoad) {
      job.seq = ++seq;
    }
    if (job.type == kGraph && len > 0 && (job.payload[0] & kGraphUrgent)) {
      job.urgent = true;
      urgent_queued++;
    }
    push(std::move(job));
  }
  Job closed;
  closed.conn = conn;
  closed.fd = fd;
  push(std::move(closed));
}

///////////////////////////////////////////////////////////////////////////////
// Telling the container that a graph is done (the DONE frames of protocol.h)
//
// The main thread cannot do it: when the GPU finishes a graph it is usually
// busy handing the next one to the GPU, and the container would wait for a
// result that is ready. So each connection has a thread that waits for the
// GPU on copies of the events of the arrays (never on the arrays themselves,
// which only the main thread touches) and then writes the frame.
//
// The prebuilt library does not export Event::wait, only array::wait: so the
// thread waits on arrays of its own, made for the purpose, that hold nothing
// but a copy of an event.
///////////////////////////////////////////////////////////////////////////////

mx::array waiter(const mx::Event& event) {
  mx::array a(mx::Shape{}, mx::bool_, nullptr, std::vector<mx::array>{});
  a.set_status(mx::array::Status::evaluated);
  a.attach_event(event);
  return a;
}

struct Told {
  uint64_t seq{0};
  std::string error;
  std::vector<mx::array> waiters;
  struct Bytes {
    uint64_t handle;
    // Keeps the buffer alive until the bytes are written.
    mx::array array;
    const char* data;
    size_t nbytes;
  };
  std::vector<Bytes> arrays;
  // The large ones, for the MORE frame.
  std::vector<Bytes> later;
  // For --timing: when the main thread was done with the graph.
  Clock::time_point handed;
};

struct Channel {
  int fd{-1};
  // Replies (main thread) and DONE frames (this thread) share the socket.
  std::mutex write_mtx;
  std::mutex mtx;
  std::condition_variable cv;
  std::deque<Told> queue;
  bool closed{false};
  std::thread thread;

  explicit Channel(int fd) : fd(fd) {
    thread = std::thread([this]() { run(); });
  }

  void tell(Told told) {
    if (timing) {
      told.handed = Clock::now();
    }
    {
      std::lock_guard lk(mtx);
      queue.push_back(std::move(told));
    }
    cv.notify_one();
  }

  void run() {
    while (true) {
      Told told;
      {
        std::unique_lock lk(mtx);
        cv.wait(lk, [this] { return closed || !queue.empty(); });
        if (closed) {
          return;
        }
        told = std::move(queue.front());
        queue.pop_front();
      }
      for (auto& w : told.waiters) {
        try {
          w.wait();
        } catch (const std::exception& ex) {
          if (told.error.empty()) {
            told.error = encode_error(ex);
          }
        }
      }
      ByteWriter w;
      w.put<uint8_t>(kReplyDone);
      w.put<uint64_t>(0);
      w.put<uint64_t>(told.seq);
      w.put<uint32_t>(static_cast<uint32_t>(told.error.size()));
      w.put_bytes(told.error.data(), told.error.size());
      if (!told.error.empty()) {
        told.later.clear();
      }
      w.put<uint32_t>(
          static_cast<uint32_t>(told.arrays.size()) |
          (told.later.empty() ? 0 : kDoneHasMore));
      for (auto& a : told.arrays) {
        w.put<uint64_t>(a.handle);
        w.put<uint64_t>(a.nbytes);
        w.put_bytes(a.data, a.nbytes);
      }
      uint64_t len = w.str().size() - kFrameHeader;
      std::memcpy(w.str().data() + 1, &len, sizeof(len));
      auto waited = timing ? ns_since(told.handed) : 0;
      {
        std::lock_guard lk(write_mtx);
        write_all(fd, w.str().data(), w.str().size());
      }
      if (!told.later.empty()) {
        // The large ones, straight from the arrays.
        ByteWriter head;
        head.put<uint8_t>(kReplyMore);
        head.put<uint64_t>(0);
        head.put<uint64_t>(told.seq);
        head.put<uint32_t>(0);
        head.put<uint32_t>(static_cast<uint32_t>(told.later.size()));
        std::vector<uint64_t> entries;
        entries.reserve(2 * told.later.size());
        uint64_t more_len = head.str().size() - kFrameHeader;
        for (auto& a : told.later) {
          entries.push_back(a.handle);
          entries.push_back(a.nbytes);
          more_len += 2 * sizeof(uint64_t) + a.nbytes;
        }
        std::memcpy(head.str().data() + 1, &more_len, sizeof(more_len));
        std::vector<iovec> iov;
        iov.push_back({head.str().data(), head.str().size()});
        for (size_t i = 0; i < told.later.size(); ++i) {
          iov.push_back({&entries[2 * i], 2 * sizeof(uint64_t)});
          if (told.later[i].nbytes > 0) {
            iov.push_back(
                {const_cast<char*>(told.later[i].data), told.later[i].nbytes});
          }
        }
        std::lock_guard lk(write_mtx);
        write_iov(fd, iov);
      }
      if (timing) {
        std::cerr << "  timing done " << told.seq << ": the GPU finished "
                  << (waited + 500) / 1000 << " us after the main thread, "
                  << "written after " << (ns_since(told.handed) + 500) / 1000
                  << " us" << std::endl;
      }
    }
  }

  // Called once the container's end is closed: nothing more can be written.
  void close() {
    {
      std::lock_guard lk(mtx);
      closed = true;
    }
    cv.notify_one();
    ::shutdown(fd, SHUT_RDWR);
    thread.join();
    queue.clear();
  }
};

///////////////////////////////////////////////////////////////////////////////
// Files of weights that are read here (OPEN and LOAD in protocol.h)
//
// A container only names a file. What is read is decided here: a regular
// file under one of the directories given at start, found by resolving the
// name on this machine (symlinks and ".." included), opened read only, and
// checked again once open, so that no name a container sends, and nothing it
// does to the names meanwhile, reaches a file outside those directories.
///////////////////////////////////////////////////////////////////////////////

struct Root {
  // The directory as containers name it, and what it is here (resolved).
  std::string container;
  std::string mac;
};
std::vector<Root> roots;

// True if path is below dir (both absolute, dir without a trailing slash).
bool below(const std::string& path, const std::string& dir) {
  if (dir == "/") {
    return path.size() > 1 && path[0] == '/';
  }
  return path.size() > dir.size() + 1 &&
      path.compare(0, dir.size(), dir) == 0 && path[dir.size()] == '/';
}

bool allowed(const std::string& resolved) {
  for (auto& r : roots) {
    if (below(resolved, r.mac)) {
      return true;
    }
  }
  return false;
}

bool add_root(std::string container, const std::string& mac) {
  while (container.size() > 1 && container.back() == '/') {
    container.pop_back();
  }
  char resolved[PATH_MAX];
  struct stat st;
  if (container.empty() || container[0] != '/') {
    std::cerr << "--models and --map take absolute paths: " << container
              << std::endl;
    return false;
  }
  if (::realpath(mac.c_str(), resolved) == nullptr ||
      ::stat(resolved, &st) != 0 || !S_ISDIR(st.st_mode)) {
    std::cerr << mac << " is not a directory" << std::endl;
    return false;
  }
  roots.push_back({container, resolved});
  return true;
}

struct File {
  int fd{-1};
  uint64_t size{0};
  // When it was last modified, as it was when it was opened: a file that
  // changes after that is not the one the container read the layout of.
  int64_t mtime_ns{0};
  std::string path;
  bool changed() const {
    struct stat st;
    return ::fstat(fd, &st) != 0 || static_cast<uint64_t>(st.st_size) != size ||
        st.st_mtimespec.tv_sec * 1000000000ll + st.st_mtimespec.tv_nsec != mtime_ns;
  }
  ~File() {
    if (fd >= 0) {
      ::close(fd);
    }
  }
};

// For the log: a name sent by a container, without control characters.
std::string printable(std::string name) {
  for (auto& ch : name) {
    if (static_cast<unsigned char>(ch) < 0x20 || ch == 0x7f) {
      ch = '?';
    }
  }
  return name.size() > 512 ? name.substr(0, 512) + "..." : name;
}

// Opens the file a container named, or says why not: `why` goes to the
// container, `log` stays here (a container must not learn from the answers
// what exists outside the allowed directories).
std::shared_ptr<File> open_file(
    const std::string& name,
    uint64_t size,
    int64_t mtime_ns,
    uint64_t digest,
    std::string& why,
    std::string& log) {
  static const std::string refused =
      "it is not a regular file that the server may read under its --models "
      "and --map directories (the server's log says more)";
  if (roots.empty()) {
    why = "the server was started without --models or --map";
    return nullptr;
  }
  if (name.empty() || name[0] != '/' || name.size() >= PATH_MAX ||
      name.find('\0') != std::string::npos) {
    why = refused;
    log = "not an absolute path";
    return nullptr;
  }
  const Root* root = nullptr;
  for (auto& r : roots) {
    if (below(name, r.container) &&
        (root == nullptr || r.container.size() > root->container.size())) {
      root = &r;
    }
  }
  if (root == nullptr) {
    why = "no --models or --map of the server covers that path";
    return nullptr;
  }
  auto mapped = root->mac +
      name.substr(root->container == "/" ? 0 : root->container.size());
  char resolved[PATH_MAX];
  if (::realpath(mapped.c_str(), resolved) == nullptr) {
    why = refused;
    log = mapped + ": " + std::strerror(errno);
    return nullptr;
  }
  if (!allowed(resolved)) {
    why = refused;
    log = mapped + " is " + resolved + ", outside the allowed directories";
    return nullptr;
  }
  auto file = std::make_shared<File>();
  file->fd = ::open(resolved, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
  if (file->fd < 0) {
    why = refused;
    log = std::string(resolved) + ": " + std::strerror(errno);
    return nullptr;
  }
  struct stat st;
  if (::fstat(file->fd, &st) != 0 || !S_ISREG(st.st_mode)) {
    why = refused;
    log = std::string(resolved) + " is not a regular file";
    return nullptr;
  }
  // What was really opened, whatever happened to the names in between.
  char opened[PATH_MAX];
  if (::fcntl(file->fd, F_GETPATH, opened) != 0 || !allowed(opened)) {
    why = refused;
    log = std::string(resolved) + " was replaced while it was being opened";
    return nullptr;
  }
  file->size = st.st_size;
  file->mtime_ns =
      st.st_mtimespec.tv_sec * 1000000000ll + st.st_mtimespec.tv_nsec;
  file->path = opened;
  // The container read the names and offsets of the arrays in its file: the
  // one found here must be the same. Its size, the time it was last modified
  // and a digest of parts of it say so, as far as that can be said without
  // reading it whole (see protocol.h).
  if (file->size != size) {
    why = "the file the Mac finds for it, " + file->path +
        ", has another size";
    return nullptr;
  }
  if (!same_file_time(file->mtime_ns, mtime_ns)) {
    why = "the file the Mac finds for it, " + file->path +
        ", was last modified at another time: it is not the one the "
        "container has";
    return nullptr;
  }
  uint64_t found = 0;
  int fd = file->fd;
  if (!file_digest(
          size,
          [fd](uint64_t offset, char* p, size_t n) {
            while (n > 0) {
              auto m = ::pread(fd, p, n, offset);
              if (m < 0 && errno == EINTR) {
                continue;
              }
              if (m <= 0) {
                return false;
              }
              p += m;
              n -= m;
              offset += m;
            }
            return true;
          },
          found)) {
    why = refused;
    log = file->path + " cannot be read";
    return nullptr;
  }
  if (found != digest) {
    why = "the file the Mac finds for it, " + file->path +
        ", has other contents";
    return nullptr;
  }
  return file;
}

// The bytes are read by a few threads, as mlx reads a file of weights
// natively (in pieces of 32 MB, half the cores at most), into arrays that
// the main thread has made. The main thread waits for all of them before it
// runs any message other than the next LOAD (wait_loads), so that nothing
// ever sees an array that is still being read.
struct ReadJob {
  std::shared_ptr<File> file;
  char* dst{nullptr};
  size_t n{0};
  uint64_t offset{0};
  // The size of the elements whose bytes are to be swapped, or 0.
  uint8_t swap{0};
  int conn{0};
  uint64_t handle{0};
};
constexpr size_t kReadPiece = 32 << 20;
std::mutex load_mtx;
std::condition_variable load_cv;
std::condition_variable load_done_cv;
std::deque<ReadJob> load_queue;
std::vector<std::thread> load_threads;
std::atomic<size_t> loads_pending{0};
// The arrays whose bytes could not be read, by connection.
struct LoadError {
  std::string text;
  std::vector<uint64_t> handles;
};
std::unordered_map<int, LoadError> load_errors;
std::atomic<bool> loads_failed{false};
// When the reads now pending began, and how long those that ended since the
// last wait_loads took, from the first LOAD to the last byte.
Clock::time_point loads_began;
std::atomic<uint64_t> loads_took{0};

void load_worker() {
  while (true) {
    ReadJob job;
    {
      std::unique_lock lk(load_mtx);
      load_cv.wait(lk, [] { return !load_queue.empty(); });
      job = std::move(load_queue.front());
      load_queue.pop_front();
    }
    std::string error;
    auto* p = job.dst;
    auto n = job.n;
    auto offset = job.offset;
    while (n > 0) {
      auto m = ::pread(job.file->fd, p, n, offset);
      if (m < 0 && errno == EINTR) {
        continue;
      }
      if (m <= 0) {
        error = m == 0 ? "the file ends before the array (it was changed "
                         "after the container opened it)"
                       : std::strerror(errno);
        break;
      }
      p += m;
      n -= m;
      offset += m;
    }
    if (error.empty() && job.file->changed()) {
      error = "the file was changed while it was being read: what was read "
              "of it is not the container's file";
    }
    if (error.empty() && job.swap > 1) {
      // As mlx's Load does: the bytes of each element are reversed.
      for (size_t i = 0; i + job.swap <= job.n; i += job.swap) {
        std::reverse(job.dst + i, job.dst + i + job.swap);
      }
    }
    std::lock_guard lk(load_mtx);
    if (!error.empty()) {
      auto& e = load_errors[job.conn];
      if (e.text.empty()) {
        e.text = "reading " + job.file->path + ": " + error;
      }
      e.handles.push_back(job.handle);
      loads_failed = true;
    }
    if (--loads_pending == 0) {
      loads_took += ns_since(loads_began);
      load_done_cv.notify_all();
    }
  }
}

void queue_read(ReadJob job) {
  std::lock_guard lk(load_mtx);
  if (load_threads.empty()) {
    auto n = std::clamp<size_t>(std::thread::hardware_concurrency() / 2, 4, 16);
    for (size_t i = 0; i < n; ++i) {
      load_threads.emplace_back(load_worker);
      load_threads.back().detach();
    }
  }
  if (loads_pending == 0) {
    loads_began = Clock::now();
  }
  auto total = job.n;
  for (size_t done = 0; done < total; done += kReadPiece) {
    auto piece = job;
    piece.dst = job.dst + done;
    piece.offset = job.offset + done;
    piece.n = std::min(kReadPiece, total - done);
    load_queue.push_back(std::move(piece));
    loads_pending++;
  }
  load_cv.notify_all();
}

///////////////////////////////////////////////////////////////////////////////
// Connection state (main thread only)
///////////////////////////////////////////////////////////////////////////////

struct Conn {
  std::unique_ptr<Channel> channel;
  // Events of the graphs sent without "tell" since the last one with it: the
  // pieces of an evaluation, done when its last piece is.
  std::vector<mx::Event> piece_events;
  std::unordered_map<uint64_t, mx::array> arrays;
  // The first error of a message that has no reply; given to the next one
  // that has, or with the next DONE frame. As it travels: its kind, then its
  // text (encode_error in protocol.h).
  std::string error;
  // It has said which protocol and which MLX it speaks, and they are this
  // server's: nothing else is taken from it before.
  bool greeted{false};
  uint64_t graphs{0}, nodes{0}, imports{0}, compiles{0}, ops{0};
  // Urgent graphs: those it marked, those answered early, the number of the
  // last of these, and after how many nodes a long program of this
  // connection pauses for them (see do_graph).
  uint64_t urgent{0}, early{0}, last_early{0};
  size_t nodes_per_stop{16};
  int fast_stops{0};
  uint64_t uploads{0}, upload_bytes{0};
  uint64_t reads{0}, read_bytes{0};
  uint64_t syncs{0}, frees{0};
  // The files it had opened here, and what was read from them: arrays,
  // bytes, and the time the reads took from the first LOAD of a run of them
  // to the last byte.
  std::unordered_map<uint32_t, std::shared_ptr<File>> files;
  uint64_t opens{0}, refused{0}, loads{0}, load_bytes{0}, load_ns{0};
  // The custom kernels it sent, by the number it gave them, and what stands
  // for them in the key of a cached program (see do_graph).
  std::unordered_map<uint32_t, std::shared_ptr<struct Kernel>> kernels;
  std::string kernel_key;
  // Those that were refused, and why: said again to a node that runs one.
  std::unordered_map<uint32_t, std::string> refused_kernels;
  // The number of the GRAPH, OP or LOAD message being run, and for those
  // not waited for yet the arrays they made.
  uint64_t seq{0};
  std::deque<std::pair<uint64_t, std::vector<uint64_t>>> unsynced;
};

std::unordered_map<int, Conn> conns;
constexpr size_t kMaxOpenFiles = 256;

// Wait for the reads in flight. Those that failed for this connection leave
// no array behind, and their error is thrown.
void wait_loads(int conn, Conn& c, bool can_throw = true) {
  if (loads_pending == 0 && !loads_failed && loads_took == 0) {
    return;
  }
  LoadError failed;
  {
    std::unique_lock lk(load_mtx);
    load_done_cv.wait(lk, [] { return loads_pending == 0; });
    auto ns = loads_took.exchange(0);
    c.load_ns += ns;
    if (verbose && ns > 0) {
      std::cerr << "  loads: the files were read in "
                << (ns + 500000) / 1000000 << " ms" << std::endl;
    }
    if (auto it = load_errors.find(conn); it != load_errors.end()) {
      failed = std::move(it->second);
      load_errors.erase(it);
    }
    loads_failed = !load_errors.empty();
  }
  for (auto handle : failed.handles) {
    c.arrays.erase(handle);
  }
  if (can_throw && !failed.text.empty()) {
    throw std::runtime_error(failed.text);
  }
}

// Arrays the container freed while the GPU may still be writing them.
//
// MLX keeps the inputs of a kernel alive until it has run, but not its
// output: if that is dropped at once, its buffer goes back to MLX's cache and
// the next allocation can get it. Natively that is harmless, the next user is
// normally another kernel, queued behind the first. Here the next user can be
// an upload, written by the cpu right away and then overwritten by the kernel
// still in flight (seen as rare wrong results in mlx's own test suite). So a
// freed array is let go only once it is evaluated.
std::vector<mx::array> in_flight;

void sweep_in_flight() {
  if (in_flight.empty()) {
    return;
  }
  if (in_flight.size() > 16384) {
    for (auto& s : mx::get_streams()) {
      mx::synchronize(s);
    }
    in_flight.clear();
    return;
  }
  size_t kept = 0;
  for (size_t i = 0; i < in_flight.size(); ++i) {
    if (!in_flight[i].is_available()) {
      if (kept != i) {
        in_flight[kept] = std::move(in_flight[i]);
      }
      kept++;
    }
  }
  in_flight.erase(in_flight.begin() + kept, in_flight.end());
}

void release_array(Conn& c, uint64_t handle) {
  auto it = c.arrays.find(handle);
  if (it == c.arrays.end()) {
    return;
  }
  if (!it->second.is_available()) {
    in_flight.push_back(std::move(it->second));
  }
  c.arrays.erase(it);
}

// A type as the container names it. Its size travels with it, and must be
// the type's own: an array is as many bytes as its type's size says, and the
// kernels read as many as the type itself has.
mx::Dtype to_dtype(const ArraySpec& spec) {
  static const mx::Dtype known[] = {
      mx::bool_,   mx::uint8,   mx::uint16,   mx::uint32,  mx::uint64,
      mx::int8,    mx::int16,   mx::int32,    mx::int64,   mx::float16,
      mx::float32, mx::float64, mx::bfloat16, mx::complex64};
  for (auto dtype : known) {
    if (spec.dtype_val == static_cast<uint32_t>(dtype.val()) &&
        spec.dtype_size == dtype.size()) {
      return dtype;
    }
  }
  throw std::runtime_error("malformed message: unknown dtype");
}

// The bytes of an array of that shape and type: more than a size can count
// is refused, here and not where a product has already wrapped around.
size_t bytes_of(const mx::Shape& shape, mx::Dtype dtype) {
  size_t nbytes = dtype.size();
  for (auto d : shape) {
    if (d < 0 || __builtin_mul_overflow(nbytes, static_cast<size_t>(d), &nbytes)) {
      throw std::runtime_error("malformed message: an array too large to count");
    }
  }
  if (nbytes > (size_t(1) << 46)) {
    throw std::runtime_error("malformed message: an array too large to count");
  }
  return nbytes;
}

mx::Shape to_shape(const ArraySpec& spec) {
  for (auto d : spec.shape) {
    if (d < 0) {
      throw std::runtime_error("malformed message: negative dimension");
    }
  }
  mx::Shape shape(spec.shape.begin(), spec.shape.end());
  // (As bytes of the largest type: whatever type it is seen with, it counts.)
  bytes_of(shape, mx::complex64);
  return shape;
}

// The bytes of an array that may be sent to a container: those of an array
// that is computed, laid out in order, and wholly inside the memory it has.
// An array says how large it is by its shape, and a primitive that shares its
// input's memory with its output (a copy, a reshape) leaves the output the
// shape the container declared for it: without this, the bytes sent could be
// more than the memory holds, and what lies after it in this process.
// (inside: for any layout, what this process's own code may read of an array
// with the cpu; sendable: laid out in order as well, as the bytes travel.)
bool inside(const mx::array& a) {
  auto& data = a.data_shared_ptr();
  if (data == nullptr || a.strides().size() != a.shape().size()) {
    return false;
  }
  if (a.size() == 0) {
    return true;
  }
  auto* start = static_cast<const char*>(data->buffer().raw_ptr());
  auto size = static_cast<int64_t>(mx::allocator::allocator().size(data->buffer()));
  if (start == nullptr) {
    return false;
  }
  // The first and the last element it reaches, counted from its own first.
  int64_t low = 0, high = 0;
  auto& strides = a.strides();
  for (size_t d = 0; d < a.shape().size(); ++d) {
    int64_t span;
    if (__builtin_mul_overflow(
            static_cast<int64_t>(a.shape(d) - 1), static_cast<int64_t>(strides[d]), &span) ||
        __builtin_add_overflow(span < 0 ? low : high, span, span < 0 ? &low : &high)) {
      return false;
    }
  }
  int64_t item = a.itemsize();
  int64_t first = a.data<char>() - start;
  int64_t before, after;
  if (__builtin_mul_overflow(low, item, &before) ||
      __builtin_mul_overflow(high + 1, item, &after)) {
    return false;
  }
  return first + before >= 0 && first <= size && after <= size - first;
}

// The memory an array has is what mlx's allocator gave for it, which can be
// more than was asked: a buffer that was used before, with what it held. So
// the bytes sent are also no more than the array was given to count as its
// own when its data was made (data_size: the elements written by the kernel
// that computed it, or shared by the view that it is), whatever its shape
// says. The nodes are checked so that the two agree (check_shapes): this is
// for the node that slipped through.
bool sendable(const mx::array& a) {
  return a.flags().row_contiguous && inside(a) && a.size() <= a.data_size();
}

// Memory of mlx's allocator for so many bytes of a container's data (an
// upload, an array read from a file, a constant): never from mlx's heap of
// small buffers (see do_upload), and with nothing in it beyond those bytes.
// mlx gives a buffer it has used before as it was left, and one larger than
// asked: what it held is zeroed.
auto data_buffer(size_t nbytes) {
  auto data = mx::allocator::malloc(std::max<size_t>(nbytes, 256));
  auto capacity = mx::allocator::allocator().size(data.buffer());
  if (capacity > nbytes) {
    std::memset(static_cast<char*>(data.buffer().raw_ptr()) + nbytes, 0, capacity - nbytes);
  }
  return data;
}

// The array of a handle, seen with the shape and type the container uses for
// it (they differ only when the container made a view of the same bytes).
mx::array input(Conn& c, const ArraySpec& spec) {
  auto it = c.arrays.find(spec.handle);
  if (it == c.arrays.end()) {
    throw std::runtime_error(
        "array #" + std::to_string(spec.handle) +
        " does not exist (an earlier graph failed, or it was freed)");
  }
  auto a = it->second;
  auto dtype = to_dtype(spec);
  auto shape = to_shape(spec);
  if (a.dtype() != dtype) {
    a = mx::view(a, dtype);
  }
  if (a.shape() != shape) {
    a = mx::reshape(a, shape);
  }
  return a;
}

///////////////////////////////////////////////////////////////////////////////
// Rebuilding nodes
//
// Only mlx's import_function can make a primitive from its saved bytes, and
// it only reads files. So each distinct primitive (stream, name, state) is
// imported once, as a function of that single node written to a temporary
// file, and the primitive object is kept: nodes are then made directly with
// it, as mlx itself does when it calls an imported or compiled function.
// (Importing whole graphs instead costs about 10 us per node, since the
// reader makes one system call per field: measured with
// probe/probe_interpose.cpp.)
///////////////////////////////////////////////////////////////////////////////

std::unordered_map<std::string, std::shared_ptr<mx::Primitive>> primitives;
// What a compiled primitive was made for (make_compiled): the types of its
// inputs, which of them are constants, the types of its outputs.
struct CompiledInfo {
  std::vector<mx::Dtype> inputs;
  std::vector<bool> constant;
  std::vector<mx::Dtype> outputs;
};
std::unordered_map<const mx::Primitive*, CompiledInfo> compiled_infos;

constexpr size_t kMaxPrimitives = 1 << 17;

// The temporary file the primitives are imported from, made at the first
// import and removed when the server ends.
int function_fd = -1;
std::string function_path;

const std::string& function_file() {
  if (function_fd < 0) {
    const char* tmpdir = std::getenv("TMPDIR");
    std::string path = std::string(tmpdir && *tmpdir ? tmpdir : "/tmp");
    if (path.back() != '/') {
      path.push_back('/');
    }
    path += "mlx-remote-" + std::to_string(getpid()) + ".mlxfn";
    ::unlink(path.c_str());
    function_fd =
        ::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (function_fd < 0) {
      throw std::runtime_error("cannot write " + path);
    }
    function_path = path;
  }
  return function_path;
}

// Writes the file format of mlx's export_function (mlx/export.cpp, 0.32.x).
struct FunctionFile {
  std::string bytes;
  template <typename T>
  void put(T v) {
    bytes.append(reinterpret_cast<const char*>(&v), sizeof(T));
  }
  void put_string(const std::string& v) {
    put<uint64_t>(v.size());
    bytes.append(v);
  }
  void put_shape(const mx::Shape& shape) {
    put<uint64_t>(shape.size());
    for (auto d : shape) {
      put<int32_t>(d);
    }
  }
  void put_dtype(mx::Dtype dtype) {
    put<int32_t>(static_cast<int32_t>(dtype.val()));
    put<uint8_t>(dtype.size());
  }
  void put_ids(uint64_t first, uint64_t n) {
    put<uint64_t>(n);
    for (uint64_t i = 0; i < n; ++i) {
      put<uint64_t>(first + i);
    }
  }
};

std::shared_ptr<mx::Primitive> import_primitive(
    Conn& c,
    const std::string& saved,
    const std::vector<mx::Shape>& in_shapes,
    const std::vector<mx::Dtype>& in_dtypes,
    const std::vector<mx::Shape>& out_shapes,
    const std::vector<mx::Dtype>& out_dtypes) {
  Lap lap(timers.import);
  timers.imports++;
  // A custom kernel never comes as a saved primitive, which would hold shader
  // source ready to compile: only through KERNEL, if the server takes them.
  if (saved.find("CustomKernel") != std::string::npos) {
    throw std::runtime_error(
        "graphs with a saved CustomKernel are refused (a kernel is sent with "
        "KERNEL, to a server started with --custom-kernels)");
  }
  auto n_in = in_shapes.size();
  auto n_out = out_shapes.size();
  FunctionFile f;
  f.put_string(mx::version());
  f.put<int32_t>(1); // functions in the file
  f.put<uint8_t>(0); // not shapeless
  f.put_string(""); // metadata
  f.put<uint64_t>(0); // keyword names
  f.put_ids(1, n_in);
  f.put<uint64_t>(n_in);
  for (size_t i = 0; i < n_in; ++i) {
    f.put_shape(in_shapes[i]);
    f.put_dtype(in_dtypes[i]);
  }
  f.put_ids(n_in + 1, n_out);
  f.put<uint64_t>(1); // nodes
  f.put<uint64_t>(n_in + 1);
  f.put<uint8_t>(1); // has a primitive
  f.put_ids(1, n_in);
  f.bytes.append(saved);
  f.put<uint64_t>(n_out - 1); // siblings
  if (n_out == 1) {
    f.put_shape(out_shapes[0]);
    f.put_dtype(out_dtypes[0]);
  } else {
    f.put_ids(n_in + 1, n_out);
    f.put<uint64_t>(n_out);
    for (auto& shape : out_shapes) {
      f.put_shape(shape);
    }
    f.put<uint64_t>(n_out);
    for (auto dtype : out_dtypes) {
      f.put_dtype(dtype);
    }
  }

  // One file for as long as the server runs, written over each time (making
  // a file and removing it again took twice as long as mlx takes to read it:
  // 86 and about 40 against 55 us, and a model's step has primitives that
  // are new at every token, those that hold a position in its cache).
  auto& path = function_file();
  bool ok = ::ftruncate(function_fd, 0) == 0;
  for (size_t done = 0; ok && done < f.bytes.size();) {
    auto m = ::pwrite(
        function_fd, f.bytes.data() + done, f.bytes.size() - done, done);
    ok = m > 0 || (m < 0 && errno == EINTR);
    done += m > 0 ? m : 0;
  }
  if (!ok) {
    throw std::runtime_error("cannot write " + path);
  }
  auto fn = mx::import_function(path);
  c.imports++;
  // Call it on placeholders, never evaluated, to get at the primitive.
  std::vector<mx::array> placeholders;
  for (size_t i = 0; i < n_in; ++i) {
    placeholders.emplace_back(
        in_shapes[i], in_dtypes[i], nullptr, std::vector<mx::array>{});
  }
  auto outs = fn(placeholders);
  if (outs.size() != n_out || !outs[0].has_primitive()) {
    throw std::runtime_error("import_function did not rebuild the node");
  }
  return outs[0].primitive_ptr();
}

///////////////////////////////////////////////////////////////////////////////
// Custom kernels (KERNEL in protocol.h), only with --custom-kernels
//
// The source of a kernel is code of a container that this Mac compiles and
// runs on its GPU. Whatever is checked here, that code reads and writes the
// GPU's memory through pointers that nothing bounds: it can read or change
// arrays of this process that it was not given, those of other connections
// included, make the process fail, and keep the GPU or the compiler busy for
// as long as it likes. What it must not do is reach the files of this Mac:
// the compiler reads any file for an #include, and quotes its lines in its
// errors, so a source that could make it is refused (reads_files).
//
// A kernel is made by mlx's own metal_kernel from the arguments the program
// in the container gave to its metal_kernel. The signature of the function,
// and so the buffers it is given, is generated here from the arrays of each
// node. The function metal_kernel returns is kept, for every connection
// that sends the same kernel, and a node that runs it is one more operation
// that mlx builds here (lower_call): like any other node it is then part of
// a cached program, and costs a model's step nothing more.
///////////////////////////////////////////////////////////////////////////////

bool custom_kernels = false;

struct Kernel {
  std::string name;
  size_t n_in{0}, n_out{0};
  mx::fast::CustomKernelFunction fn;
  // Which one it is among all the kernels this server was ever sent: never
  // given to another, since the programs kept for the graphs that come again
  // are kept under it (see read_head).
  uint64_t uid{0};
  // The bytes it was sent as, and what mlx compiled of it, by the names mlx
  // keeps its libraries under.
  size_t bytes{0};
  std::vector<std::string> libraries;
};

// The kernels of the containers connected now, by what was sent of them (the
// KERNEL message after the number). A kernel goes when the last connection
// that sent it does (release_kernels), with what mlx compiled of it: what
// this server keeps of kernels, and so what it can refuse for having too
// many, is what its containers use now, not what the containers before them
// left.
std::unordered_map<std::string, std::shared_ptr<Kernel>> kernels;
uint64_t next_kernel_uid = 1;
// What mlx compiles for them: a library for each kernel, values of its
// template arguments and types of its arrays, kept by mlx under these names
// until it is told to let go of them.
std::unordered_set<std::string> kernel_libraries;
constexpr size_t kMaxKernelsTotal = 4096;
constexpr size_t kMaxKernelsBytes = 64 << 20;
size_t kernels_bytes = 0;
constexpr size_t kMaxKernelLibraries = 16384;

// Lets go of the kernels that no connection has any more: their source, the
// function mlx made of it, and the libraries mlx compiled and keeps (which is
// most of what a kernel costs this process). Called when a connection has
// closed. What the GPU is still running of them holds its own pipeline.
void release_kernels() {
  for (auto it = kernels.begin(); it != kernels.end();) {
    if (it->second.use_count() > 1) {
      ++it;
      continue;
    }
    for (auto& library : it->second->libraries) {
      mx::metal::device(mx::Device::gpu).clear_library(library);
      kernel_libraries.erase(library);
    }
    kernels_bytes -= it->second->bytes;
    it = kernels.erase(it);
  }
}
// mlx multiplies the three sizes of a threadgroup as ints before it compares
// them with what the GPU allows.
constexpr int64_t kMaxThreadgroup = 1 << 20;

bool is_identifier(const std::string& s) {
  if (s.empty() || s.size() > kMaxKernelName || (s[0] >= '0' && s[0] <= '9')) {
    return false;
  }
  for (auto ch : s) {
    if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
          (ch >= '0' && ch <= '9') || ch == '_')) {
      return false;
    }
  }
  return true;
}

// True if the source could make the compiler read a file. That takes a
// directive (#include, #import, #if __has_include) or a pragma (GCC
// dependency), which _Pragma gives without a '#'. They are looked for as
// text, since what the compiler accepts between the parts of a directive
// (line splices, comments over several lines, "%:" for '#') is more than is
// worth parsing: once blanks and backslashes are taken out, none of the
// words may come anywhere after the first '#'. A kernel may use #define and
// #if; one that also says "include" further down, in a comment even, is
// refused too.
bool reads_files(const std::string& text) {
  std::string squeezed;
  squeezed.reserve(text.size());
  for (size_t i = 0; i < text.size(); ++i) {
    auto ch = text[i];
    if (text.compare(i, 3, "?\?/") == 0) {
      // The trigraph of a backslash.
      i += 2;
    } else if (ch != '\\' && ch != ' ' && ch != '\t' && ch != '\n' &&
               ch != '\r' && ch != '\f' && ch != '\v' && ch != '\0') {
      squeezed.push_back(ch);
    }
  }
  if (squeezed.find("_Pragma") != std::string::npos) {
    return true;
  }
  auto hash = std::min(
      {squeezed.find('#'), squeezed.find("%:"), squeezed.find("?\?=")});
  if (hash == std::string::npos) {
    return false;
  }
  for (auto* word : {"include", "import", "pragma"}) {
    if (squeezed.find(word, hash) != std::string::npos) {
      return true;
    }
  }
  return false;
}

void add_kernel(int conn, Conn& c, uint32_t id, ByteReader& r);

void do_kernel(int conn, Conn& c, ByteReader& r) {
  if (!custom_kernels) {
    throw std::runtime_error(
        "custom kernels are refused: this server was started without "
        "--custom-kernels");
  }
  auto id = r.get<uint32_t>();
  try {
    add_kernel(conn, c, id, r);
  } catch (const std::exception& e) {
    if (c.kernels.count(id) == 0 && c.refused_kernels.size() < kMaxKernels) {
      c.refused_kernels.emplace(id, e.what());
    }
    throw;
  }
}

void add_kernel(int conn, Conn& c, uint32_t id, ByteReader& r) {
  std::string sent(r.position(), r.remaining());
  auto name = r.get_text();
  if (!is_identifier(name)) {
    throw std::runtime_error(
        "kernel refused: its name must be an identifier of at most " +
        std::to_string(kMaxKernelName) + " characters");
  }
  auto refused = [&name](const std::string& why) {
    return std::runtime_error("kernel " + name + " refused: " + why);
  };
  if (id == 0 || c.kernels.count(id) != 0) {
    throw refused("malformed message (its number)");
  }
  if (c.kernels.size() >= kMaxKernels) {
    throw refused(
        "this connection has sent " + std::to_string(kMaxKernels) +
        " kernels already");
  }
  std::vector<std::string> names[2];
  for (auto& list : names) {
    auto n = r.get<uint32_t>();
    if (n > kMaxKernelArrays) {
      throw refused(
          "more than " + std::to_string(kMaxKernelArrays) +
          " inputs or outputs");
    }
    for (uint32_t i = 0; i < n; ++i) {
      list.push_back(r.get_text());
      if (!is_identifier(list.back())) {
        throw refused(
            "the names of its arrays must be identifiers of at most " +
            std::to_string(kMaxKernelName) + " characters");
      }
    }
  }
  if (names[1].empty()) {
    throw refused("it has no output");
  }
  auto source = r.get_text();
  auto header = r.get_text();
  if (source.size() + header.size() > kMaxKernelSource) {
    throw refused(
        "its source and header are more than " +
        std::to_string(kMaxKernelSource) + " bytes");
  }
  bool ensure_row_contiguous = r.get<uint8_t>();
  bool atomic_outputs = r.get<uint8_t>();
  auto options = r.get<int32_t>();
  if (r.remaining() != 0) {
    throw refused("malformed message");
  }
  if (options < static_cast<int32_t>(mx::MathMode::Safe) ||
      options > static_cast<int32_t>(mx::MathMode::Fast)) {
    throw refused("unknown compile options");
  }
  if (reads_files(header + "\n" + source)) {
    throw refused(
        "its source has an #include, #import or #pragma (or one of those "
        "words somewhere after a '#'), or a _Pragma: a kernel must not make "
        "the compiler read files of the Mac");
  }
  auto it = kernels.find(sent);
  bool is_new = it == kernels.end();
  if (is_new) {
    if (kernels.size() >= kMaxKernelsTotal ||
        kernels_bytes + sent.size() > kMaxKernelsBytes) {
      throw refused(
          "the containers connected to this server have sent it " +
          std::to_string(kernels.size()) +
          " different kernels, as many as it keeps at a time");
    }
    kernels_bytes += sent.size();
    auto kernel = std::make_shared<Kernel>();
    kernel->name = name;
    kernel->n_in = names[0].size();
    kernel->n_out = names[1].size();
    kernel->uid = next_kernel_uid++;
    kernel->bytes = sent.size();
    kernel->fn = mx::fast::metal_kernel(
        name,
        names[0],
        names[1],
        source,
        header,
        ensure_row_contiguous,
        atomic_outputs,
        mx::CompileOptions(options));
    it = kernels.emplace(std::move(sent), std::move(kernel)).first;
  }
  auto& kernel = it->second;
  c.kernels.emplace(id, kernel);
  c.kernel_key.append(reinterpret_cast<const char*>(&id), sizeof(id));
  c.kernel_key.append(
      reinterpret_cast<const char*>(&kernel->uid), sizeof(kernel->uid));
  // Always in the log: whose source this Mac is about to compile.
  std::cerr << "conn " << conn << ": kernel #" << id << " " << name
            << (is_new ? ", new to this server: " : ", sent before: ")
            << kernel->n_in << " inputs, " << kernel->n_out << " outputs, "
            << source.size() + header.size() << " bytes of source"
            << (ensure_row_contiguous ? "" : ", inputs as they lie in memory")
            << (atomic_outputs ? ", atomic outputs" : "")
            << (options == 0 ? "" : options == 1 ? ", relaxed math" : ", fast math")
            << std::endl;
}

// A parsed GRAPH message: what to build from the inputs.
// The inputs of a node are inputs of the message (bit 31), constants of the
// program (bit 30) or outputs of its nodes, numbered in order.
constexpr uint32_t kValueIsConstant = 0x40000000u;

struct Node {
  std::shared_ptr<mx::Primitive> primitive;
  std::vector<uint32_t> inputs;
  std::vector<mx::Shape> shapes;
  std::vector<mx::Dtype> dtypes;
};

struct Program {
  std::vector<Node> nodes;
  std::vector<uint32_t> keep;
  // Built for the cpu, to answer an urgent graph (see "Urgent graphs"), and
  // the inputs it uses as indices: each has to be a place in an axis of that
  // size where a slice of that size starts.
  bool cpu{false};
  struct Index {
    uint32_t input;
    int64_t size;
    int64_t slice;
  };
  std::vector<Index> indices;
  // Arrays that mlx made while building an operation here (see lower).
  std::vector<mx::array> constants;
  size_t n_values{0};
  // The size of the largest array it computes.
  size_t max_bytes{0};
};

mx::array placeholder(const mx::Shape& shape, mx::Dtype dtype) {
  return mx::array(shape, dtype, nullptr, std::vector<mx::array>{});
}

std::shared_ptr<mx::Primitive> find_primitive(
    Conn& c,
    std::string saved,
    const std::vector<mx::Shape>& in_shapes,
    const std::vector<mx::Dtype>& in_dtypes,
    const std::vector<mx::Shape>& out_shapes,
    const std::vector<mx::Dtype>& out_dtypes) {
  auto it = primitives.find(saved);
  if (it == primitives.end()) {
    auto primitive =
        import_primitive(c, saved, in_shapes, in_dtypes, out_shapes, out_dtypes);
    // Some are new at every step of a model (those that hold a position in
    // its cache): not kept for ever. The programs keep theirs.
    if (primitives.size() >= kMaxPrimitives) {
      primitives.clear();
      compiled_infos.clear();
    }
    it = primitives.emplace(std::move(saved), std::move(primitive)).first;
  }
  return it->second;
}

// What a node of a container's graph must be before mlx is given it. mlx's
// primitives take their arrays, and their own state, on trust: natively the
// operations that make them (mlx/ops.cpp) have checked. Here a container
// makes them, and what the code that runs a primitive relies on is checked
// again, primitive by primitive, before an array is made of the node:
// - the node runs on the GPU (one sent on the cpu's stream would have this
//   process's own code read wherever the shapes say);
// - it has the arrays its primitive reads: with fewer, mlx reads past its
//   list of them;
// - its state goes with its arrays: an axis is one of the array's, a list of
//   axes, sizes or strides is as long as the array has dimensions, a slice or
//   a view by strides lies inside its input. Much of what runs a primitive is
//   this process's own code, before any kernel of the GPU: it fills in the
//   strides of a transposition axis by axis, walks a slice's starts, takes an
//   axis out of a shape, and with an axis that is not there it writes
//   outside its own vectors;
// - its outputs have the shape and the type the primitive makes of those
//   inputs, and its inputs the types its kernel reads them as. A primitive
//   that shares its input's memory with its output (a copy, a reshape) leaves
//   the output the shape and type the container declared for it: declared
//   larger, it would be an array of more bytes than were ever written, and
//   what was sent back of it would be what the memory held before. A kernel
//   given a shape or a type that does not go with its inputs reads or writes
//   the GPU's memory outside its arrays.
// What is left unchecked is what is in the values: an index out of range, on
// the GPU, reads or writes what the GPU has there (see graph_refusals.py).
// The rules are those of the operations that make each primitive in mlx
// 0.32.4, read there. A primitive that is not here is refused, and so is a
// node that mlx's own operations would not have made (the few they make in
// vmap with arrays that are not broadcast, a pad with a value that is not a
// scalar): what is refused fails in the container with this text.
using Shapes = std::vector<mx::Shape>;
using Dtypes = std::vector<mx::Dtype>;

enum class Rule {
  unary, real_part, as_type, same, reshape, squeeze, expand_dims, transpose,
  as_strided, slice, split, view, number_of_elements, arange, random_bits,
  binary, to_bool, divmod, select, broadcast, broadcast_axes, concatenate, pad,
  dynamic_slice, dynamic_slice_update, slice_update, search_sorted, depends,
  cpu_only, gather, scatter, gather_axis, scatter_axis, masked_scatter, sort,
  arg_sort, partition, arg_partition, scan, reduce, arg_reduce, softmax,
  log_sum_exp, matmul, addmm, block_masked_mm, gather_mm, quantized_matmul,
  gather_qmm, quantize, convolution, rms_norm, layer_norm, rms_norm_vjp,
  layer_norm_vjp, rope, hadamard, fft
};

struct Known {
  Rule rule;
  size_t least, most;
  // One of those mlx's compile fuses into a kernel: the only ones a compiled
  // primitive may be made of.
  bool fusable{false};
};
constexpr size_t kAny = 1 << 16;

// The name a primitive is saved under: after its stream (three integers).
std::string saved_name(const std::string& saved) {
  uint64_t length = 0;
  if (saved.size() < 12 + sizeof(length)) {
    throw std::runtime_error("malformed graph: a primitive too short to have a name");
  }
  std::memcpy(&length, saved.data() + 12, sizeof(length));
  if (length > 64 || saved.size() - 12 - sizeof(length) < length) {
    throw std::runtime_error("malformed graph: a primitive without a name");
  }
  return saved.substr(12 + sizeof(length), length);
}

// (Dimensions are not negative and their product can be counted: to_shape.)
int64_t elements(const mx::Shape& shape) {
  int64_t n = 1;
  for (auto d : shape) {
    n *= d;
  }
  return n;
}

bool is_axis(int64_t axis, size_t ndim) {
  return axis >= 0 && static_cast<uint64_t>(axis) < ndim;
}

// A list of axes of an array of that rank, each named once.
template <typename Axes>
bool are_axes(const Axes& axes, size_t ndim) {
  uint64_t seen = 0; // (an array has 64 dimensions at most: get_type)
  for (auto axis : axes) {
    if (!is_axis(static_cast<int64_t>(axis), ndim) || (seen >> axis & 1)) {
      return false;
    }
    seen |= uint64_t(1) << axis;
  }
  return true;
}

bool is_index(mx::Dtype dtype) {
  return mx::issubdtype(dtype, mx::integer) && dtype != mx::bool_;
}

// Whether an array of shape `from` broadcasts to `to`, as mlx's broadcast_to.
bool broadcasts_to(const mx::Shape& from, const mx::Shape& to) {
  if (from.size() > to.size()) {
    return false;
  }
  auto skip = to.size() - from.size();
  for (size_t i = 0; i < from.size(); ++i) {
    if (from[i] != to[i + skip] && from[i] != 1) {
      return false;
    }
  }
  return true;
}

// The shape those shapes broadcast to, as mlx's broadcast_shapes; false if
// they do not go together.
bool broadcast_of(const Shapes& shapes, mx::Shape& out) {
  size_t ndim = 0;
  for (auto& shape : shapes) {
    ndim = std::max(ndim, shape.size());
  }
  out = mx::Shape(ndim, 1);
  for (auto& shape : shapes) {
    auto skip = ndim - shape.size();
    for (size_t i = 0; i < shape.size(); ++i) {
      auto& d = out[i + skip];
      if (shape[i] != 1 && d != 1 && shape[i] != d) {
        return false;
      }
      if (shape[i] != 1) {
        d = shape[i];
      }
    }
  }
  return true;
}

const std::unordered_map<std::string, Known>& known_primitives() {
  static const std::unordered_map<std::string, Known> known = [] {
    std::unordered_map<std::string, Known> k;
    for (auto* name :
         {"Abs", "ArcCos", "ArcSin", "ArcTan", "Cos", "Cosh", "Exp", "Log1p",
          "Sin", "Sinh", "Tan", "Tanh", "ArcCosh", "ArcSinh", "ArcTanh",
          "Expm1", "Sigmoid", "Erf", "ErfInv", "Log", "Sqrt", "Ceil", "Floor",
          "Negative", "Sign", "Square", "Conjugate", "LogicalNot", "Round"}) {
      k[name] = {Rule::unary, 1, 1, true};
    }
    for (auto* name : {"Real", "Imag"}) {
      k[name] = {Rule::real_part, 1, 1, true};
    }
    k["AsType"] = {Rule::as_type, 1, 1, true};
    for (auto* name : {"Full", "Copy", "StopGradient"}) {
      k[name] = {Rule::same, 1, 1};
    }
    for (auto* name : {"Reshape", "Flatten", "Unflatten"}) {
      k[name] = {Rule::reshape, 1, 1};
    }
    for (auto* name :
         {"Add", "Subtract", "Multiply", "Maximum", "Minimum", "Divide",
          "FloorDivide", "Remainder", "Power", "ArcTan2", "LogAddExp",
          "BitwiseBinary"}) {
      k[name] = {Rule::binary, 2, 2, true};
    }
    for (auto* name :
         {"NotEqual", "Greater", "GreaterEqual", "Less", "LessEqual", "Equal",
          "LogicalAnd", "LogicalOr"}) {
      k[name] = {Rule::to_bool, 2, 2, true};
    }
    for (auto* name : {"QRF", "SVD", "Inverse", "Cholesky", "Eig", "Eigh"}) {
      // (mlx has none of these on the GPU: each fails when it is run.)
      k[name] = {Rule::cpu_only, 1, 1};
    }
    k["Squeeze"] = {Rule::squeeze, 1, 1};
    k["ExpandDims"] = {Rule::expand_dims, 1, 1};
    k["Transpose"] = {Rule::transpose, 1, 1};
    k["AsStrided"] = {Rule::as_strided, 1, 1};
    k["Slice"] = {Rule::slice, 1, 1};
    k["Split"] = {Rule::split, 1, 1};
    k["View"] = {Rule::view, 1, 1};
    k["NumberOfElements"] = {Rule::number_of_elements, 1, 1};
    k["Arange"] = {Rule::arange, 0, 0};
    k["RandomBits"] = {Rule::random_bits, 1, 1};
    k["DivMod"] = {Rule::divmod, 2, 2};
    k["Select"] = {Rule::select, 3, 3, true};
    k["Broadcast"] = {Rule::broadcast, 1, kAny, true};
    k["BroadcastAxes"] = {Rule::broadcast_axes, 2, kAny};
    k["Concatenate"] = {Rule::concatenate, 1, kAny};
    k["Pad"] = {Rule::pad, 2, 2};
    k["DynamicSlice"] = {Rule::dynamic_slice, 2, 2};
    k["DynamicSliceUpdate"] = {Rule::dynamic_slice_update, 3, 3};
    k["SliceUpdate"] = {Rule::slice_update, 2, 2};
    k["SearchSorted"] = {Rule::search_sorted, 2, 2};
    k["Depends"] = {Rule::depends, 1, kAny};
    k["Gather"] = {Rule::gather, 1, kAny};
    k["Scatter"] = {Rule::scatter, 2, kAny};
    k["GatherAxis"] = {Rule::gather_axis, 2, 2};
    k["ScatterAxis"] = {Rule::scatter_axis, 3, 3};
    k["MaskedScatter"] = {Rule::masked_scatter, 3, 3};
    k["Sort"] = {Rule::sort, 1, 1};
    k["ArgSort"] = {Rule::arg_sort, 1, 1};
    k["Partition"] = {Rule::partition, 1, 1};
    k["ArgPartition"] = {Rule::arg_partition, 1, 1};
    k["Scan"] = {Rule::scan, 1, 1};
    k["Reduce"] = {Rule::reduce, 1, 1};
    k["ArgReduce"] = {Rule::arg_reduce, 1, 1};
    k["Softmax"] = {Rule::softmax, 1, 1};
    k["LogSumExp"] = {Rule::log_sum_exp, 1, 1};
    k["Matmul"] = {Rule::matmul, 2, 2};
    k["AddMM"] = {Rule::addmm, 3, 3};
    k["BlockMaskedMM"] = {Rule::block_masked_mm, 3, 5};
    k["GatherMM"] = {Rule::gather_mm, 4, 4};
    k["QuantizedMatmul"] = {Rule::quantized_matmul, 3, 4};
    k["GatherQMM"] = {Rule::gather_qmm, 5, 6};
    k["Quantize"] = {Rule::quantize, 1, 3};
    k["Convolution"] = {Rule::convolution, 2, 2};
    k["RMSNorm"] = {Rule::rms_norm, 2, 2};
    k["LayerNorm"] = {Rule::layer_norm, 3, 3};
    k["RMSNormVJP"] = {Rule::rms_norm_vjp, 3, 3};
    k["LayerNormVJP"] = {Rule::layer_norm_vjp, 4, 4};
    k["RoPE"] = {Rule::rope, 2, 3};
    k["Hadamard"] = {Rule::hadamard, 1, 1};
    k["FFT"] = {Rule::fft, 1, 1};
    return k;
  }();
  return known;
}

// Throws unless the node is one mlx's own operations make. Whatever device
// its primitive is on: check_node adds that it is the GPU, check_for_cpu
// what the cpu may compute in its place.
void check_shapes(
    const std::string& name,
    mx::Primitive& primitive,
    const Shapes& in,
    const Dtypes& it,
    const Shapes& out,
    const Dtypes& ot,
    bool in_tape = false) {
  auto& known = known_primitives();
  auto found = known.find(name);
  if (found == known.end()) {
    throw std::runtime_error(
        "malformed graph: " + name + " is not a primitive this server knows the arrays of");
  }
  auto rule = found->second;
  auto n_in = in.size();
  auto n_out = out.size();
  if (n_in < rule.least || n_in > rule.most) {
    throw std::runtime_error(
        "malformed graph: " + name + " with " + std::to_string(n_in) + " arrays");
  }
  auto need = [&name](bool ok, const char* what) {
    if (!ok) {
      throw std::runtime_error(
          "malformed graph: " + name + " whose " + what +
          " not what mlx makes of its inputs");
    }
  };
  static const char* outputs = "number of outputs is";
  static const char* shape = "output's shape is";
  static const char* type = "types are";
  static const char* state = "axes, sizes or strides are";
  static const char* inputs = "inputs' shapes are";
  if (in_tape) {
    // A compiled function is made of these and no others, one output each.
    // What runs it reads the shapes of the node it is, not those of what it
    // is made of (see check_compiled).
    need(rule.fusable && n_out == 1, "place in a compiled function is");
    return;
  }
  // One output, of the shape and the type of the first input.
  auto like_first = [&] {
    need(n_out == 1, outputs);
    need(out[0] == in[0], shape);
    need(ot[0] == it[0], type);
  };
  switch (rule.rule) {
    case Rule::unary:
    case Rule::same:
      like_first();
      break;
    case Rule::real_part:
      need(n_out == 1, outputs);
      need(out[0] == in[0], shape);
      need(it[0] == mx::complex64 && ot[0] == mx::float32, type);
      break;
    case Rule::as_type:
      need(n_out == 1, outputs);
      need(out[0] == in[0], shape);
      need(ot[0] == static_cast<const mx::AsType&>(primitive).state(), type);
      break;
    case Rule::reshape:
      // (What runs it reads the two shapes and nothing of its state.)
      need(n_out == 1, outputs);
      need(elements(out[0]) == elements(in[0]), shape);
      need(ot[0] == it[0], type);
      break;
    case Rule::squeeze: {
      auto axes = static_cast<const mx::Squeeze&>(primitive).state();
      need(n_out == 1, outputs);
      need(are_axes(axes, in[0].size()) && std::is_sorted(axes.begin(), axes.end()), state);
      mx::Shape expected;
      size_t next = 0;
      for (size_t d = 0; d < in[0].size(); ++d) {
        if (next < axes.size() && static_cast<size_t>(axes[next]) == d) {
          need(in[0][d] == 1, state);
          next++;
        } else {
          expected.push_back(in[0][d]);
        }
      }
      need(out[0] == expected, shape);
      need(ot[0] == it[0], type);
      break;
    }
    case Rule::expand_dims: {
      auto axes = static_cast<const mx::ExpandDims&>(primitive).state();
      need(n_out == 1, outputs);
      need(axes.size() <= 64 && std::is_sorted(axes.begin(), axes.end()), state);
      mx::Shape expected = in[0];
      for (auto axis : axes) {
        need(axis >= 0 && static_cast<size_t>(axis) <= expected.size(), state);
        expected.insert(expected.begin() + axis, 1);
      }
      need(out[0] == expected, shape);
      need(ot[0] == it[0], type);
      break;
    }
    case Rule::transpose: {
      auto axes = static_cast<const mx::Transpose&>(primitive).state();
      need(n_out == 1, outputs);
      need(axes.size() == in[0].size() && are_axes(axes, in[0].size()), state);
      need(out[0].size() == in[0].size(), shape);
      for (size_t d = 0; d < axes.size(); ++d) {
        need(out[0][d] == in[0][axes[d]], shape);
      }
      need(ot[0] == it[0], type);
      break;
    }
    case Rule::as_strided: {
      // Every element it reaches is one of its input's.
      auto [view, strides, offset] = static_cast<const mx::AsStrided&>(primitive).state();
      need(n_out == 1, outputs);
      auto total = elements(in[0]);
      int64_t low = 0, high = 0;
      bool ok = view.size() == strides.size() && offset <= static_cast<size_t>(total);
      bool empty = false;
      for (size_t d = 0; ok && d < view.size(); ++d) {
        empty = empty || view[d] == 0;
        int64_t span;
        ok = view[d] >= 0 &&
            !__builtin_mul_overflow(
                static_cast<int64_t>(std::max(view[d], 1) - 1),
                static_cast<int64_t>(strides[d]),
                &span) &&
            !__builtin_add_overflow(span < 0 ? low : high, span, span < 0 ? &low : &high);
      }
      if (!ok || (!empty && (static_cast<int64_t>(offset) + low < 0 ||
                             static_cast<int64_t>(offset) + high >= total))) {
        throw std::runtime_error(
            "malformed graph: a view by strides that is not inside its array");
      }
      need(out[0] == view, shape);
      need(ot[0] == it[0], type);
      break;
    }
    case Rule::slice: {
      // As many elements along each axis as the output has, each of them in
      // the input. (What runs it reads the starts, the strides and the
      // output's shape, and nothing of the ends.)
      auto [start, end, strides] = static_cast<const mx::Slice&>(primitive).state();
      auto ndim = in[0].size();
      need(n_out == 1, outputs);
      need(start.size() == ndim && strides.size() == ndim && end.size() == ndim, state);
      need(out[0].size() == ndim, shape);
      for (size_t d = 0; d < ndim; ++d) {
        need(strides[d] != 0, state);
        int64_t count = out[0][d];
        int64_t last = start[d] + (count - 1) * int64_t(strides[d]);
        if (count > 0 && !(start[d] >= 0 && start[d] < in[0][d] && last >= 0 && last < in[0][d])) {
          throw std::runtime_error("malformed graph: a slice that is not inside its input");
        }
      }
      need(ot[0] == it[0], type);
      break;
    }
    case Rule::split: {
      auto [indices, axis] = static_cast<const mx::Split&>(primitive).state();
      auto ndim = in[0].size();
      need(is_axis(axis, ndim), state);
      need(n_out == indices.size() + 1 && n_out >= 2, outputs);
      int64_t before = 0;
      for (size_t i = 0; i < n_out; ++i) {
        int64_t upto = i < indices.size() ? indices[i] : in[0][axis];
        need(upto >= before && upto <= in[0][axis], state);
        auto expected = in[0];
        expected[axis] = static_cast<int32_t>(upto - before);
        need(out[i] == expected, shape);
        need(ot[i] == it[0], type);
        before = upto;
      }
      break;
    }
    case Rule::view: {
      auto dtype = static_cast<const mx::View&>(primitive).state();
      need(n_out == 1, outputs);
      need(ot[0] == dtype, type);
      auto expected = in[0];
      size_t from = it[0].size(), to = dtype.size();
      if (from != to) {
        need(!expected.empty(), inputs);
        int64_t last = expected.back();
        if (from > to) {
          last *= from / to;
        } else {
          need(last % (to / from) == 0, inputs);
          last /= to / from;
        }
        need(last <= std::numeric_limits<int32_t>::max(), shape);
        expected.back() = static_cast<int32_t>(last);
      }
      need(out[0] == expected, shape);
      break;
    }
    case Rule::number_of_elements: {
      auto [axes, inverted, dtype] = static_cast<const mx::NumberOfElements&>(primitive).state();
      need(n_out == 1, outputs);
      need(are_axes(axes, in[0].size()), state);
      need(out[0].empty(), shape);
      need(ot[0] == dtype, type);
      break;
    }
    case Rule::arange: {
      auto [start, stop, step] = static_cast<const mx::Arange&>(primitive).state();
      need(n_out == 1, outputs);
      // (A step can be infinite: mlx makes the empty range of it.)
      need(std::isfinite(start) && std::isfinite(stop) && !std::isnan(step) && step != 0, state);
      double count = std::ceil((stop - start) / step);
      need(count <= std::numeric_limits<int32_t>::max(), state);
      need(out[0] == mx::Shape{count > 0 ? static_cast<int32_t>(count) : 0}, shape);
      need(ot[0] != mx::bool_, type);
      break;
    }
    case Rule::random_bits: {
      auto [made, width] = static_cast<const mx::RandomBits&>(primitive).state();
      need(n_out == 1, outputs);
      need(!in[0].empty() && in[0].back() == 2 && it[0] == mx::uint32, inputs);
      // (One key, or with vmap one for each of the output's first dimensions.)
      need(out[0] == made && out[0].size() + 1 >= in[0].size() &&
               std::equal(in[0].begin(), in[0].end() - 1, out[0].begin()),
           shape);
      need(ot[0] == (width == 1 ? mx::uint8 : width == 2 ? mx::uint16 : mx::uint32) &&
               (width == 1 || width == 2 || width == 4),
           type);
      break;
    }
    case Rule::binary:
      need(in[0] == in[1], inputs);
      need(it[0] == it[1], type);
      like_first();
      break;
    case Rule::to_bool:
      need(n_out == 1, outputs);
      need(in[0] == in[1], inputs);
      need(out[0] == in[0], shape);
      need(it[0] == it[1] && ot[0] == mx::bool_, type);
      break;
    case Rule::divmod:
      need(n_out == 2, outputs);
      need(in[0] == in[1], inputs);
      need(out[0] == in[0] && out[1] == in[0], shape);
      need(it[0] == it[1] && ot[0] == it[0] && ot[1] == it[0], type);
      break;
    case Rule::select:
      need(n_out == 1, outputs);
      need(in[0] == in[1] && in[0] == in[2], inputs);
      need(out[0] == in[0], shape);
      need(it[0] == mx::bool_ && it[1] == ot[0] && it[2] == ot[0], type);
      break;
    case Rule::broadcast: {
      need(n_out == 1, outputs);
      if (n_in == 1) {
        need(out[0] == static_cast<const mx::Broadcast&>(primitive).state(), shape);
        need(broadcasts_to(in[0], out[0]), inputs);
      } else {
        // (As mlx traces a function of shapes not yet known: the shape all
        // its inputs broadcast to.)
        mx::Shape all;
        need(broadcast_of(in, all), inputs);
        need(out[0] == all, shape);
      }
      need(ot[0] == it[0], type);
      break;
    }
    case Rule::broadcast_axes: {
      // The inputs broadcast to each other but for the axes named, counted
      // from the last, which the output takes from the first input.
      auto ignore = static_cast<const mx::BroadcastAxes&>(primitive).state();
      need(n_out == 1, outputs);
      size_t least = 64;
      for (auto& s : in) {
        least = std::min(least, s.size());
      }
      need(ignore.size() <= least, state);
      for (size_t i = 0; i < ignore.size(); ++i) {
        need(ignore[i] < 0 && ignore[i] >= -static_cast<int64_t>(least) &&
                 (i == 0 || ignore[i] > ignore[i - 1]),
             state);
      }
      Shapes rest;
      for (auto& s : in) {
        auto without = s;
        for (auto axis = ignore.rbegin(); axis != ignore.rend(); ++axis) {
          without.erase(without.begin() + s.size() + *axis);
        }
        rest.push_back(std::move(without));
      }
      mx::Shape expected;
      need(broadcast_of(rest, expected), inputs);
      int64_t dims = ignore.size() + expected.size();
      for (auto axis : ignore) {
        expected.insert(expected.begin() + dims + axis, in[0][in[0].size() + axis]);
      }
      need(out[0] == expected, shape);
      need(ot[0] == it[0], type);
      break;
    }
    case Rule::concatenate: {
      auto axis = static_cast<const mx::Concatenate&>(primitive).state();
      need(n_out == 1, outputs);
      need(is_axis(axis, in[0].size()), state);
      auto expected = in[0];
      int64_t along = 0;
      for (size_t i = 0; i < n_in; ++i) {
        need(in[i].size() == in[0].size() && it[i] == it[0], inputs);
        for (size_t d = 0; d < in[0].size(); ++d) {
          need(d == static_cast<size_t>(axis) || in[i][d] == in[0][d], inputs);
        }
        along += in[i][axis];
      }
      need(along <= std::numeric_limits<int32_t>::max(), shape);
      expected[axis] = static_cast<int32_t>(along);
      need(out[0] == expected, shape);
      need(ot[0] == it[0], type);
      break;
    }
    case Rule::pad: {
      auto [axes, low, high] = static_cast<const mx::Pad&>(primitive).state();
      int64_t ndim = in[0].size();
      need(n_out == 1, outputs);
      need(axes.size() == low.size() && axes.size() == high.size(), state);
      // (The value to pad with: one element. mlx also takes an array that
      // broadcasts to the output, which its code for the GPU does not check.)
      need(elements(in[1]) == 1 && it[1] == it[0], inputs);
      std::vector<int64_t> expected(in[0].begin(), in[0].end());
      for (size_t i = 0; i < axes.size(); ++i) {
        int64_t axis = axes[i] < 0 ? axes[i] + ndim : axes[i];
        need(is_axis(axis, ndim) && low[i] >= 0 && high[i] >= 0, state);
        expected[axis] += int64_t(low[i]) + high[i];
      }
      need(out[0].size() == in[0].size(), shape);
      for (int64_t d = 0; d < ndim; ++d) {
        need(out[0][d] == expected[d], shape);
      }
      need(ot[0] == it[0], type);
      break;
    }
    case Rule::dynamic_slice:
    case Rule::dynamic_slice_update: {
      // Where the slice starts is in an array, and so is not checked here:
      // see the note on indices above.
      bool update = rule.rule == Rule::dynamic_slice_update;
      std::vector<int> axes;
      mx::Shape sizes;
      if (update) {
        axes = static_cast<const mx::DynamicSliceUpdate&>(primitive).state();
        sizes = in[1];
        need(it[1] == it[0], type);
      } else {
        std::tie(axes, sizes) = static_cast<const mx::DynamicSlice&>(primitive).state();
      }
      auto& start = in[update ? 2 : 1];
      need(n_out == 1, outputs);
      need(are_axes(axes, in[0].size()) && sizes.size() == in[0].size(), state);
      for (size_t d = 0; d < sizes.size(); ++d) {
        need(sizes[d] >= 0 && sizes[d] <= in[0][d], state);
      }
      need(start.size() <= 1 && elements(start) == static_cast<int64_t>(axes.size()) &&
               is_index(it[update ? 2 : 1]),
           inputs);
      need(out[0] == (update ? in[0] : sizes), shape);
      need(ot[0] == it[0], type);
      break;
    }
    case Rule::slice_update: {
      // The update is written along each axis from the start, by the strides:
      // each of its elements goes to one of the source's.
      auto [reduce, start, end, strides] = static_cast<const mx::SliceUpdate&>(primitive).state();
      auto ndim = in[0].size();
      need(start.size() == ndim && strides.size() == ndim && end.size() == ndim, state);
      need(in[1].size() == ndim && ndim >= 1, inputs);
      for (size_t d = 0; d < ndim; ++d) {
        need(strides[d] != 0, state);
        int64_t count = in[1][d];
        int64_t last = start[d] + (count - 1) * int64_t(strides[d]);
        if (count > 0 && !(start[d] >= 0 && start[d] < in[0][d] && last >= 0 && last < in[0][d])) {
          throw std::runtime_error("malformed graph: an update of a slice that is not inside its array");
        }
      }
      need(it[1] == it[0], type);
      like_first();
      break;
    }
    case Rule::search_sorted:
      need(n_out == 1, outputs);
      need(in[0].size() == 1, inputs);
      need(out[0] == in[1], shape);
      need(it[0] == it[1] && ot[0] == mx::uint32, type);
      break;
    case Rule::depends:
      need(n_out >= 1 && n_out <= n_in, outputs);
      for (size_t i = 0; i < n_out; ++i) {
        need(out[i] == in[i], shape);
        need(ot[i] == it[i], type);
      }
      break;
    case Rule::cpu_only:
      break;
    case Rule::gather: {
      auto [axes, sizes] = static_cast<const mx::Gather&>(primitive).state();
      need(n_out == 1, outputs);
      need(n_in == axes.size() + 1 && are_axes(axes, in[0].size()) && sizes.size() == in[0].size(), state);
      mx::Shape expected;
      for (size_t i = 1; i < n_in; ++i) {
        need(in[i] == in[1] && is_index(it[i]) && it[i] == it[1], inputs);
      }
      // Whole slices of the input; of an input with nothing in it, none (no
      // index, or slices of no element).
      bool nothing = n_in > 1 && elements(in[1]) == 0;
      for (size_t d = 0; d < sizes.size(); ++d) {
        need(sizes[d] >= 0, state);
        nothing = nothing || sizes[d] == 0;
      }
      for (size_t d = 0; d < sizes.size(); ++d) {
        need(elements(in[0]) == 0 ? nothing : sizes[d] <= in[0][d], state);
      }
      if (n_in > 1) {
        expected = in[1];
      }
      expected.insert(expected.end(), sizes.begin(), sizes.end());
      need(out[0] == expected, shape);
      need(ot[0] == it[0], type);
      break;
    }
    case Rule::scatter: {
      auto [reduce, axes] = static_cast<const mx::Scatter&>(primitive).state();
      need(n_in == axes.size() + 2 && are_axes(axes, in[0].size()), state);
      auto& updates = in[n_in - 1];
      size_t index_dims = n_in > 2 ? in[1].size() : 0;
      for (size_t i = 1; i + 1 < n_in; ++i) {
        need(in[i] == in[1] && is_index(it[i]) && it[i] == it[1], inputs);
      }
      need(updates.size() == index_dims + in[0].size(), inputs);
      for (size_t d = 0; d < index_dims; ++d) {
        need(updates[d] == in[1][d], inputs);
      }
      for (size_t d = 0; d < in[0].size(); ++d) {
        need(updates[index_dims + d] <= in[0][d], inputs);
      }
      need(it[n_in - 1] == it[0], type);
      like_first();
      break;
    }
    case Rule::gather_axis:
    case Rule::scatter_axis: {
      bool scatter = rule.rule == Rule::scatter_axis;
      int axis = scatter ? static_cast<const mx::ScatterAxis&>(primitive).state().second
                         : static_cast<const mx::GatherAxis&>(primitive).state();
      need(n_out == 1, outputs);
      need(is_axis(axis, in[0].size()), state);
      need(in[1].size() == in[0].size() && is_index(it[1]), inputs);
      for (size_t d = 0; d < in[0].size(); ++d) {
        need(d == static_cast<size_t>(axis) || in[1][d] == in[0][d], inputs);
      }
      if (scatter) {
        need(in[2] == in[1], inputs);
        need(it[2] == it[0], type);
      }
      need(out[0] == (scatter ? in[0] : in[1]), shape);
      need(ot[0] == it[0], type);
      break;
    }
    case Rule::masked_scatter:
      need(in[1] == in[0] && !in[2].empty() && in[2][0] > 0, inputs);
      need(it[1] == mx::bool_ && it[2] == it[0], type);
      like_first();
      break;
    case Rule::sort:
    case Rule::arg_sort:
    case Rule::partition:
    case Rule::arg_partition: {
      int axis = rule.rule == Rule::sort ? static_cast<const mx::Sort&>(primitive).state()
          : rule.rule == Rule::arg_sort ? static_cast<const mx::ArgSort&>(primitive).state()
          : rule.rule == Rule::partition ? static_cast<const mx::Partition&>(primitive).state().second
          : static_cast<const mx::ArgPartition&>(primitive).state().second;
      bool arg = rule.rule == Rule::arg_sort || rule.rule == Rule::arg_partition;
      need(n_out == 1, outputs);
      need(is_axis(axis, in[0].size()), state);
      need(out[0] == in[0], shape);
      need(ot[0] == (arg ? mx::uint32 : it[0]), type);
      break;
    }
    case Rule::scan: {
      auto [reduce, axis, reverse, inclusive] = static_cast<const mx::Scan&>(primitive).state();
      need(n_out == 1, outputs);
      need(is_axis(axis, in[0].size()), state);
      need(out[0] == in[0], shape);
      // (A sum of booleans counts them.)
      need(ot[0] == it[0] || (it[0] == mx::bool_ && ot[0] == mx::int32 && reduce == mx::Scan::Sum), type);
      break;
    }
    case Rule::reduce: {
      auto [reduce, axes] = static_cast<const mx::Reduce&>(primitive).state();
      need(n_out == 1, outputs);
      need(!axes.empty() && are_axes(axes, in[0].size()) && std::is_sorted(axes.begin(), axes.end()), state);
      auto expected = in[0];
      for (auto axis : axes) {
        expected[axis] = 1;
      }
      need(out[0] == expected, shape);
      // The kernel writes what the operation makes of the input's type,
      // whatever the output says it is.
      auto made = it[0];
      if (reduce == mx::Reduce::And || reduce == mx::Reduce::Or) {
        made = mx::bool_;
      } else if (reduce == mx::Reduce::Sum || reduce == mx::Reduce::Prod) {
        if (mx::issubdtype(it[0], mx::signedinteger)) {
          made = it[0].size() <= 4 ? mx::int32 : mx::int64;
        } else if (mx::issubdtype(it[0], mx::unsignedinteger)) {
          made = it[0].size() <= 4 ? mx::uint32 : mx::uint64;
        } else if (it[0] == mx::bool_) {
          made = mx::int32;
        }
      }
      need(ot[0] == made, type);
      break;
    }
    case Rule::arg_reduce: {
      auto [reduce, axis] = static_cast<const mx::ArgReduce&>(primitive).state();
      need(n_out == 1, outputs);
      need(is_axis(axis, in[0].size()), state);
      auto expected = in[0];
      expected[axis] = 1;
      need(elements(in[0]) > 0, inputs);
      need(out[0] == expected, shape);
      need(ot[0] == mx::uint32, type);
      break;
    }
    case Rule::softmax:
      need(!in[0].empty(), inputs);
      like_first();
      break;
    case Rule::log_sum_exp: {
      need(n_out == 1, outputs);
      need(!in[0].empty(), inputs);
      auto expected = in[0];
      expected.back() = 1;
      need(out[0] == expected, shape);
      need(ot[0] == it[0], type);
      break;
    }
    case Rule::matmul:
    case Rule::addmm:
    case Rule::block_masked_mm: {
      // Two stacks of matrices of the same batch: (.., M, K) by (.., K, N).
      auto ndim = in[0].size();
      need(n_out == 1, outputs);
      need(ndim >= 2 && in[1].size() == ndim && in[0][ndim - 1] == in[1][ndim - 2] &&
               std::equal(in[0].begin(), in[0].end() - 2, in[1].begin()),
           inputs);
      auto expected = in[0];
      expected.back() = in[1].back();
      need(out[0] == expected, shape);
      need(it[0] == it[1] && ot[0] == it[0], type);
      if (rule.rule == Rule::addmm) {
        need(in[2] == expected, inputs);
        need(it[2] == it[0], type);
      } else if (rule.rule == Rule::block_masked_mm) {
        // Masks by blocks: of the output (3 or 5 inputs), then of the two
        // operands (4 or 5).
        int64_t block = static_cast<const mx::BlockMaskedMM&>(primitive).state();
        need(block == 32 || block == 64, state);
        auto blocks = [block](int64_t n) { return static_cast<int32_t>((n + block - 1) / block); };
        int32_t m = blocks(in[0][ndim - 2]), k = blocks(in[0][ndim - 1]), n = blocks(in[1][ndim - 1]);
        auto mask = [&](size_t i, int32_t rows, int32_t columns) {
          auto wanted = in[0];
          wanted[ndim - 2] = rows;
          wanted[ndim - 1] = columns;
          need(in[i] == wanted, inputs);
          need(it[i] == mx::bool_ || it[i] == it[0], type);
        };
        if (n_in != 4) {
          mask(2, m, n);
        }
        if (n_in >= 4) {
          mask(n_in - 2, m, k);
          mask(n_in - 1, k, n);
        }
      }
      break;
    }
    case Rule::gather_mm: {
      need(n_out == 1, outputs);
      need(in[0].size() >= 2 && in[1].size() >= 2 && in[0].back() == in[1][in[1].size() - 2] && in[2] == in[3], inputs);
      auto expected = in[2];
      expected.push_back(in[0][in[0].size() - 2]);
      expected.push_back(in[1].back());
      need(out[0] == expected, shape);
      need(it[0] == it[1] && it[2] == mx::uint32 && it[3] == mx::uint32 && ot[0] == it[0], type);
      break;
    }
    case Rule::quantized_matmul:
    case Rule::gather_qmm: {
      // As mlx's quantized_matmul and gather_qmm check them: x, the packed
      // matrix w, its scales, its biases (affine) or one scale for each
      // matrix (nvfp4), and for gather_qmm which x and which w.
      bool gathers = rule.rule == Rule::gather_qmm;
      int group_size, bits;
      mx::QuantizationMode mode;
      bool transpose;
      if (gathers) {
        auto s = static_cast<const mx::GatherQMM&>(primitive).state();
        group_size = std::get<0>(s), bits = std::get<1>(s), mode = std::get<2>(s), transpose = std::get<3>(s);
      } else {
        std::tie(group_size, bits, mode, transpose) = static_cast<const mx::QuantizedMatmul&>(primitive).state();
      }
      bool affine = mode == mx::QuantizationMode::Affine;
      size_t n_matrix = gathers ? n_in - 2 : n_in;
      need(n_out == 1, outputs);
      need(group_size >= 1 && bits >= 1 && bits <= 32, state);
      need(n_matrix == (affine ? 4u : 3u) || (gathers && !affine && n_matrix == 4), inputs);
      auto& x = in[0];
      auto& w = in[1];
      auto& scales = in[2];
      need(w.size() >= 2 && scales.size() == w.size() && !x.empty() &&
               std::equal(w.begin(), w.end() - 2, scales.begin()) &&
               int64_t(w.back()) * 32 / bits == int64_t(scales.back()) * group_size,
           inputs);
      int64_t unpacked = int64_t(w.back()) * 32 / bits;
      int64_t inner = transpose ? unpacked : w[w.size() - 2];
      int64_t outer = transpose ? w[w.size() - 2] : unpacked;
      need(x.back() == inner && outer <= std::numeric_limits<int32_t>::max(), inputs);
      need(it[1] == mx::uint32 && ot[0] == it[0] && mx::issubdtype(ot[0], mx::floating), type);
      if (affine) {
        need(in[3] == scales, inputs);
        need(it[2] == it[0] && it[3] == it[0], type);
      } else if (n_matrix == 4) {
        need(in[3] == mx::Shape(w.begin(), w.end() - 2), inputs);
        need(it[3] == mx::float32 && mode == mx::QuantizationMode::Nvfp4, type);
      }
      mx::Shape expected;
      if (gathers) {
        need(x.size() >= 2 && in[n_in - 2] == in[n_in - 1], inputs);
        need(it[n_in - 2] == mx::uint32 && it[n_in - 1] == mx::uint32, type);
        expected = in[n_in - 2];
        expected.push_back(x[x.size() - 2]);
        expected.push_back(static_cast<int32_t>(outer));
      } else {
        // (Stacks of matrices on both sides are of one batch.)
        need(w.size() == 2 || x.size() <= 2 ||
                 (x.size() == w.size() && std::equal(x.begin(), x.end() - 2, w.begin())),
             inputs);
        expected = x;
        expected.back() = static_cast<int32_t>(outer);
      }
      need(out[0] == expected, shape);
      break;
    }
    case Rule::quantize: {
      // Packing a matrix into so many bits an element with a scale (and a
      // bias, affine) for each group of them, and the way back.
      auto s = static_cast<const mx::fast::Quantize&>(primitive).state();
      int64_t group_size = std::get<1>(s), bits = std::get<2>(s);
      bool affine = std::get<3>(s) == mx::QuantizationMode::Affine;
      bool dequantize = std::get<4>(s);
      need(group_size >= 1 && bits >= 1 && bits <= 32, state);
      auto& w = in[0];
      need(w.size() >= 2, inputs);
      if (dequantize) {
        need(n_out == 1, outputs);
        need(n_in >= 2 && (!affine || n_in == 3), inputs);
        auto& scales = in[1];
        need(scales.size() == w.size() && std::equal(w.begin(), w.end() - 2, scales.begin()) &&
                 w[w.size() - 2] == scales[scales.size() - 2] &&
                 int64_t(w.back()) * 32 / bits == int64_t(scales.back()) * group_size,
             inputs);
        need(!affine || in[2] == scales, inputs);
        need(it[0] == mx::uint32 && (!affine || (it[2] == it[1] && ot[0] == it[1])), type);
        auto expected = w;
        int64_t unpacked = int64_t(w.back()) * 32 / bits;
        need(unpacked <= std::numeric_limits<int32_t>::max(), shape);
        expected.back() = static_cast<int32_t>(unpacked);
        need(out[0] == expected, shape);
      } else {
        need(n_out == (affine ? 3u : 2u), outputs);
        need(n_in <= (affine ? 1u : 2u), inputs);
        need(w.back() % group_size == 0 && int64_t(w.back()) * bits % 32 == 0, inputs);
        auto packed = w, groups = w;
        packed.back() = static_cast<int32_t>(int64_t(w.back()) * bits / 32);
        groups.back() = static_cast<int32_t>(w.back() / group_size);
        need(out[0] == packed && out[1] == groups && (!affine || out[2] == groups), shape);
        need(ot[0] == mx::uint32 && (affine ? ot[1] == it[0] && ot[2] == it[0] : ot[1] == mx::uint8), type);
      }
      break;
    }
    case Rule::convolution: {
      auto [strides, pad_lo, pad_hi, dilation, in_dilation, groups, flip] =
          static_cast<const mx::Convolution&>(primitive).state();
      auto ndim = in[0].size();
      need(n_out == 1, outputs);
      need(ndim >= 3 && ndim <= 5 && in[1].size() == ndim, inputs);
      auto spatial = ndim - 2;
      need(strides.size() == spatial && pad_lo.size() == spatial && pad_hi.size() == spatial &&
               dilation.size() == spatial && in_dilation.size() == spatial && groups >= 1,
           state);
      need(int64_t(in[1].back()) * groups == in[0].back() && in[1][0] % groups == 0, inputs);
      auto expected = in[0];
      expected.back() = in[1][0];
      for (size_t d = 0; d < spatial; ++d) {
        need(strides[d] > 0 && dilation[d] > 0 && in_dilation[d] > 0 && pad_lo[d] >= 0 && pad_hi[d] >= 0, state);
        int64_t image = int64_t(in_dilation[d]) * (in[0][d + 1] - 1) + 1 + pad_lo[d] + pad_hi[d];
        int64_t kernel = int64_t(dilation[d]) * (in[1][d + 1] - 1) + 1;
        need(in[0][d + 1] >= 1 && in[1][d + 1] >= 1 && image >= kernel &&
                 image <= std::numeric_limits<int32_t>::max(),
             inputs);
        expected[d + 1] = static_cast<int32_t>(1 + (image - kernel) / strides[d]);
      }
      need(out[0] == expected, shape);
      need(it[0] == it[1] && ot[0] == it[0], type);
      break;
    }
    case Rule::rms_norm:
    case Rule::layer_norm:
    case Rule::rms_norm_vjp:
    case Rule::layer_norm_vjp: {
      // x, its weight (and bias): one for each element of the last axis, or
      // the one number mlx puts in their place when there is none.
      bool layer = rule.rule == Rule::layer_norm || rule.rule == Rule::layer_norm_vjp;
      bool vjp = rule.rule == Rule::rms_norm_vjp || rule.rule == Rule::layer_norm_vjp;
      size_t n_weights = layer ? 2 : 1;
      need(!in[0].empty(), inputs);
      for (size_t i = 1; i <= n_weights; ++i) {
        need(in[i].empty() || in[i] == mx::Shape{in[0].back()}, inputs);
        need(it[i] == it[0], type);
      }
      need(mx::issubdtype(it[0], mx::floating), type);
      if (vjp) {
        need(in[n_weights + 1] == in[0], inputs);
        need(it[n_weights + 1] == it[0], type);
        need(n_out == n_weights + 1, outputs);
        for (size_t i = 0; i < n_out; ++i) {
          need(out[i] == in[i], shape);
          need(ot[i] == it[i], type);
        }
      } else {
        like_first();
      }
      break;
    }
    case Rule::rope: {
      auto s = static_cast<const mx::fast::RoPE&>(primitive).state();
      int64_t dims = std::get<1>(s);
      need(in[0].size() >= 3 && dims > 0 && dims % 2 == 0 && dims <= in[0].back(), state);
      need(in[1].size() <= 1 && (elements(in[1]) == 1 || elements(in[1]) == in[0][0]), inputs);
      need(is_index(it[1]) && it[1].size() == 4, type);
      if (n_in == 3) {
        need(in[2] == mx::Shape{static_cast<int32_t>(dims / 2)}, inputs);
      }
      like_first();
      break;
    }
    case Rule::hadamard:
      need(!in[0].empty() && in[0].back() > 1, inputs);
      like_first();
      break;
    case Rule::fft: {
      auto [axes, inverse, real, odd] = static_cast<const mx::FFT&>(primitive).state();
      need(n_out == 1, outputs);
      need(!axes.empty() && are_axes(axes, in[0].size()), state);
      auto expected = in[0];
      for (auto axis : axes) {
        need(in[0][axis] >= 1, inputs);
      }
      auto last = axes.back();
      if (real && inverse) {
        expected[last] = 2 * (in[0][last] - 1) + (odd ? 1 : 0);
        need(expected[last] >= 1, shape);
        need(it[0] == mx::complex64 && ot[0] == mx::float32, type);
      } else if (real) {
        expected[last] = in[0][last] / 2 + 1;
        need(it[0] == mx::float32 && ot[0] == mx::complex64, type);
      } else {
        need(it[0] == mx::complex64 && ot[0] == mx::complex64, type);
      }
      need(out[0] == expected, shape);
      break;
    }
  }
}

// A node of a container's graph for the GPU: one mlx's operations make
// (check_shapes), on the GPU's stream.
void check_node(
    const std::string& name,
    mx::Primitive& primitive,
    const Shapes& in_shapes,
    const Dtypes& in_dtypes,
    const Shapes& out_shapes,
    const Dtypes& out_dtypes,
    bool in_tape) {
  check_shapes(name, primitive, in_shapes, in_dtypes, out_shapes, out_dtypes, in_tape);
  if (primitive.stream().device != mx::Device::gpu) {
    throw std::runtime_error("malformed graph: " + name + " is not on the gpu");
  }
}

// A node that is a compiled function: the arrays it is given are those it
// was compiled for. What runs it lines its inputs up with its output
// dimension by dimension, and takes every output for one of the first's
// shape.
void check_compiled(
    const mx::Primitive& primitive,
    const Shapes& in,
    const Dtypes& it,
    const Shapes& out,
    const Dtypes& ot) {
  auto found = compiled_infos.find(&primitive);
  auto need = [](bool ok, const char* what) {
    if (!ok) {
      throw std::runtime_error(
          std::string("malformed graph: a compiled function whose ") + what +
          " not those it was compiled for");
    }
  };
  need(found != compiled_infos.end(), "arrays are");
  auto& info = found->second;
  need(!in.empty() && in.size() == info.inputs.size() && out.size() == info.outputs.size(), "arrays are");
  // Every output has one shape, which every input broadcasts to (it can be
  // larger than all of them: the function can broadcast inside).
  for (size_t i = 0; i < in.size(); ++i) {
    need(it[i] == info.inputs[i], "inputs' types are");
    need(!info.constant[i] || elements(in[i]) == 1, "constants are");
    need(broadcasts_to(in[i], out[0]), "inputs' shapes are");
  }
  for (size_t i = 0; i < out.size(); ++i) {
    need(out[i] == out[0], "outputs' shapes are");
    need(ot[i] == info.outputs[i], "outputs' types are");
  }
}

// A Compiled primitive from what the container sent of its own (see
// protocol.h): the same object mlx's compile makes natively, so the same
// fused kernel.
std::shared_ptr<mx::Primitive> make_compiled(Conn& c, ByteReader r) {
  std::vector<mx::array> inputs;
  std::unordered_set<uintptr_t> constant_ids;
  CompiledInfo info;
  auto n_in = r.get<uint32_t>();
  for (uint32_t i = 0; i < n_in; ++i) {
    auto spec = r.get_type();
    auto dtype = to_dtype(spec);
    auto shape = to_shape(spec);
    bool constant = r.get<uint8_t>();
    info.inputs.push_back(dtype);
    info.constant.push_back(constant);
    if (constant) {
      if (bytes_of(shape, dtype) != dtype.size()) {
        throw std::runtime_error("malformed graph: constant with many elements");
      }
      auto data = data_buffer(dtype.size());
      std::memcpy(data.buffer().raw_ptr(), r.get_bytes(dtype.size()), dtype.size());
      inputs.emplace_back(std::move(data), shape, dtype);
      constant_ids.insert(inputs.back().id());
    } else {
      inputs.push_back(placeholder(shape, dtype));
    }
  }
  std::vector<mx::array> tape;
  auto value = [&](uint32_t v) -> const mx::array& {
    bool is_input = v & kValueIsInput;
    auto index = v & ~kValueIsInput;
    if (index >= (is_input ? inputs.size() : tape.size())) {
      throw std::runtime_error("malformed graph: value out of range");
    }
    return is_input ? inputs[index] : tape[index];
  };
  auto n_tape = r.get<uint32_t>();
  for (uint32_t i = 0; i < n_tape; ++i) {
    auto len = r.get<uint32_t>();
    bool is_op = len & kNodeCall;
    len &= ~kNodeFlags;
    std::string saved(r.get_bytes(len), len);
    std::vector<mx::array> in;
    std::vector<mx::Shape> in_shapes;
    std::vector<mx::Dtype> in_dtypes;
    auto n = r.get<uint16_t>();
    for (uint16_t j = 0; j < n; ++j) {
      in.push_back(value(r.get<uint32_t>()));
      in_shapes.push_back(in.back().shape());
      in_dtypes.push_back(in.back().dtype());
    }
    auto spec = r.get_type();
    auto dtype = to_dtype(spec);
    auto shape = to_shape(spec);
    std::shared_ptr<mx::Primitive> primitive;
    if (is_op) {
      // A primitive that cannot be saved: taken from the operation itself.
      uint32_t op = 0;
      if (saved.size() == sizeof(op)) {
        std::memcpy(&op, saved.data(), sizeof(op));
      }
      if (op != kOpBitwiseInvert || in.size() != 1) {
        throw std::runtime_error("unknown operation in a compiled primitive");
      }
      primitive = mx::bitwise_invert(in[0]).primitive_ptr();
    } else {
      primitive = find_primitive(c, saved, in_shapes, in_dtypes, {shape}, {dtype});
      check_node(saved_name(saved), *primitive, in_shapes, in_dtypes, {shape}, {dtype}, /* in_tape */ true);
    }
    tape.emplace_back(shape, dtype, std::move(primitive), std::move(in));
  }
  std::vector<mx::array> outputs;
  auto n_out = r.get<uint32_t>();
  for (uint32_t i = 0; i < n_out; ++i) {
    auto v = r.get<uint32_t>();
    if (v & kValueIsInput) {
      throw std::runtime_error("malformed graph: output that is an input");
    }
    outputs.push_back(value(v));
    info.outputs.push_back(outputs.back().dtype());
  }
  if (outputs.empty()) {
    throw std::runtime_error("malformed graph: compiled primitive without outputs");
  }
  auto stream = outputs.back().primitive().stream();
  auto compiled = std::make_shared<mx::Compiled>(
      stream,
      std::move(inputs),
      std::move(outputs),
      std::move(tape),
      std::move(constant_ids));
  compiled_infos[compiled.get()] = std::move(info);
  return compiled;
}

// Add to a program the nodes of an array that mlx built here on
// placeholders, whose values in the program are given; returns the value of
// the array.
uint32_t lower(
    Program& p,
    const mx::array& a,
    std::unordered_map<uintptr_t, uint32_t>& known,
    int depth = 0) {
  if (auto it = known.find(a.id()); it != known.end()) {
    return it->second;
  }
  if (depth > 256) {
    throw std::runtime_error("operation too deep to rebuild");
  }
  if (!a.has_primitive()) {
    // A constant: kept with the program, out of MLX's heap of small buffers
    // as the uploads are (see do_upload).
    auto copy = a;
    copy.eval();
    auto data = data_buffer(copy.nbytes());
    std::memcpy(data.buffer().raw_ptr(), copy.data<char>(), copy.nbytes());
    auto v = kValueIsConstant | static_cast<uint32_t>(p.constants.size());
    p.constants.emplace_back(std::move(data), copy.shape(), copy.dtype());
    known.emplace(a.id(), v);
    return v;
  }
  Node node;
  node.primitive = a.primitive_ptr();
  for (auto& in : a.inputs()) {
    node.inputs.push_back(lower(p, in, known, depth + 1));
  }
  for (auto& o : a.outputs()) {
    node.shapes.push_back(o.shape());
    node.dtypes.push_back(o.dtype());
    known.emplace(o.id(), static_cast<uint32_t>(p.n_values++));
  }
  p.nodes.push_back(std::move(node));
  return known.at(a.id());
}

// A node that runs a custom kernel: made by the function mlx's metal_kernel
// returned for it, on placeholders.
std::vector<mx::array> call_kernel(
    int conn,
    Conn& c,
    ByteReader& r,
    const std::vector<mx::array>& args,
    const Node& expected) {
  if (!custom_kernels) {
    throw std::runtime_error(
        "custom kernels are refused: this server was started without "
        "--custom-kernels");
  }
  auto id = r.get<uint32_t>();
  auto it = c.kernels.find(id);
  if (it == c.kernels.end()) {
    auto why = c.refused_kernels.find(id);
    throw std::runtime_error(
        why != c.refused_kernels.end()
            ? why->second
            : "kernel #" + std::to_string(id) +
                  " was not sent on this connection");
  }
  auto& kernel = *it->second;
  auto refused = [&kernel](const std::string& why) {
    return std::runtime_error("kernel " + kernel.name + ": " + why);
  };
  int sizes[6];
  for (auto& size : sizes) {
    size = r.get<int32_t>();
  }
  auto threads = static_cast<int64_t>(sizes[3]) * sizes[4] * sizes[5];
  if (*std::min_element(sizes, sizes + 6) < 1 || threads > kMaxThreadgroup) {
    throw refused(
        "the sizes of the grid and of the threadgroup must be positive, and "
        "a threadgroup at most " + std::to_string(kMaxThreadgroup) +
        " threads");
  }
  bool has_init = r.get<uint8_t>();
  auto init = r.get<float>();
  std::vector<std::pair<std::string, mx::fast::TemplateArg>> template_args;
  auto n_template = r.get<uint16_t>();
  if (n_template > kMaxKernelTemplate) {
    throw refused("too many template arguments");
  }
  for (uint16_t i = 0; i < n_template; ++i) {
    auto kind = r.get<uint8_t>();
    auto name = r.get_text<uint16_t>();
    if (!is_identifier(name)) {
      throw refused("the names of template arguments must be identifiers");
    }
    if (kind == kTemplateInt) {
      template_args.emplace_back(std::move(name), static_cast<int>(r.get<int32_t>()));
    } else if (kind == kTemplateBool) {
      template_args.emplace_back(std::move(name), r.get<uint8_t>() != 0);
    } else if (kind == kTemplateDtype) {
      ArraySpec spec;
      spec.dtype_val = r.get<uint32_t>();
      spec.dtype_size = r.get<uint8_t>();
      template_args.emplace_back(std::move(name), to_dtype(spec));
    } else {
      throw refused("malformed template argument");
    }
  }
  if (r.remaining() != 0) {
    throw refused("malformed node");
  }
  if (args.size() != kernel.n_in || expected.shapes.size() != kernel.n_out) {
    throw refused(
        "it takes " + std::to_string(kernel.n_in) + " arrays and makes " +
        std::to_string(kernel.n_out) + ", the node has " +
        std::to_string(args.size()) + " and " +
        std::to_string(expected.shapes.size()));
  }
  auto outs = kernel.fn(
      args,
      expected.shapes,
      expected.dtypes,
      {sizes[0], sizes[1], sizes[2]},
      {sizes[3], sizes[4], sizes[5]},
      template_args,
      has_init ? std::optional<float>(init) : std::nullopt,
      false,
      {});
  if (outs.size() != kernel.n_out || !outs[0].has_primitive() ||
      std::string(outs[0].primitive().name()) != "CustomKernel") {
    throw refused("metal_kernel did not make the node");
  }
  // What mlx will compile when the node first runs: its name says the kernel,
  // the template arguments and the types of the arrays.
  // (The library's name is the one mlx gives it, in
  // backend/metal/custom_kernel.cpp: it is what mlx is later told to let go.)
  auto state =
      static_cast<const mx::fast::CustomKernel&>(outs[0].primitive()).state();
  auto& full_name = std::get<0>(state);
  std::ostringstream hash;
  hash << std::hex << std::hash<std::string>{}(std::get<1>(state));
  auto library = full_name + "_" + hash.str() + "_" +
      std::to_string(std::get<10>(state));
  if (kernel_libraries.count(library) == 0) {
    if (kernel_libraries.size() >= kMaxKernelLibraries) {
      throw refused(
          "the containers connected to this server have had it compile " +
          std::to_string(kMaxKernelLibraries) +
          " kernels, as many as it keeps at a time");
    }
    kernel.libraries.push_back(library);
    kernel_libraries.insert(std::move(library));
    std::cerr << "conn " << conn << ": compiles kernel " << full_name
              << " (#" << id << ", when it first runs)" << std::endl;
  }
  return outs;
}

// An operation that mlx builds here as it would natively (see protocol.h);
// returns the values of its outputs.
std::vector<uint32_t> lower_call(
    int conn,
    Conn& c,
    Program& p,
    ByteReader r,
    const std::vector<uint32_t>& inputs,
    const std::vector<mx::Shape>& in_shapes,
    const std::vector<mx::Dtype>& in_dtypes,
    const Node& expected,
    bool has_kernels) {
  auto call = r.get<uint32_t>();
  if (call == kCallContiguous || call == kCallKernel) {
    std::vector<mx::array> args;
    std::unordered_map<uintptr_t, uint32_t> known;
    for (size_t i = 0; i < inputs.size(); ++i) {
      args.push_back(placeholder(in_shapes[i], in_dtypes[i]));
      known.emplace(args.back().id(), inputs[i]);
    }
    std::vector<mx::array> outs;
    if (call == kCallKernel) {
      if (!has_kernels) {
        throw std::runtime_error(
            "malformed graph: a kernel in a graph that does not say it has "
            "one");
      }
      outs = call_kernel(conn, c, r, args, expected);
    } else {
      bool allow_col_major = r.get<uint8_t>();
      if (args.size() != 1 || expected.shapes.size() != 1) {
        throw std::runtime_error("malformed graph: inputs of a contiguous");
      }
      outs.push_back(mx::contiguous(args[0], allow_col_major));
    }
    for (size_t i = 0; i < outs.size(); ++i) {
      if (outs.size() != expected.shapes.size() ||
          outs[i].shape() != expected.shapes[i] ||
          outs[i].dtype() != expected.dtypes[i]) {
        throw std::runtime_error("operation of an unexpected shape or type");
      }
    }
    std::vector<uint32_t> values;
    lower(p, outs[0], known);
    for (auto& out : outs) {
      values.push_back(known.at(out.id()));
    }
    return values;
  }
  if (call == kCallCrossEntropy || call == kCallCrossEntropyVJP) {
    if (r.remaining() != 0) {
      throw std::runtime_error("malformed graph: a cross entropy");
    }
    std::vector<mx::array> args;
    std::unordered_map<uintptr_t, uint32_t> known;
    for (size_t i = 0; i < inputs.size(); ++i) {
      args.push_back(placeholder(in_shapes[i], in_dtypes[i]));
      known.emplace(args.back().id(), inputs[i]);
    }
    mx::array out(0.0f);
    if (call == kCallCrossEntropy) {
      if (args.size() != 2) {
        throw std::runtime_error("malformed graph: inputs of a cross entropy");
      }
      // mlx's own operation, which checks its arrays and makes the node of
      // the fused kernel: what a program on the Mac gets.
      out = mx::fast::cross_entropy(args[0], args[1]);
    } else {
      // Its gradient. mlx makes this node in one place only, the gradient of
      // the node above (CrossEntropy::vjp in mlx/fast.cpp), of the logits,
      // the targets, the loss and the loss's cotangent: made here as there,
      // of arrays that are what those are.
      auto leading = [&](size_t i) {
        return in_shapes[i].size() + 1 == in_shapes[0].size() &&
            std::equal(in_shapes[i].begin(), in_shapes[i].end(), in_shapes[0].begin());
      };
      if (args.size() != 4 || in_shapes[0].empty() || in_shapes[0].back() < 1 ||
          !mx::issubdtype(in_dtypes[0], mx::floating) || in_dtypes[1] != mx::int32 ||
          in_dtypes[2] != mx::float32 || in_dtypes[3] != mx::float32 ||
          !leading(1) || !leading(2) || !leading(3)) {
        throw std::runtime_error(
            "malformed graph: inputs of the gradient of a cross entropy");
      }
      // (mlx's library does not let a program make the node itself: it is
      // taken from mlx's own gradient of the operation, traced on these
      // arrays, and given the four of the message in the place of its own:
      // the copy mlx traces the logits through, and the loss it makes again.)
      auto traced = mx::vjp(
          [&](const std::vector<mx::array>& logits) {
            return std::vector<mx::array>{mx::fast::cross_entropy(logits[0], args[1])};
          },
          {args[0]},
          {args[3]});
      if (traced.second.size() != 1 || !traced.second[0].has_primitive() ||
          std::string(traced.second[0].primitive().name()) != "CrossEntropyVJP" ||
          traced.second[0].inputs().size() != 4 ||
          traced.second[0].inputs()[1].id() != args[1].id() ||
          traced.second[0].inputs()[3].id() != args[3].id()) {
        throw std::runtime_error("mlx did not make the gradient of a cross entropy");
      }
      out = traced.second[0];
      for (size_t i = 0; i < 4; ++i) {
        known.emplace(out.inputs()[i].id(), inputs[i]);
      }
    }
    if (expected.shapes.size() != 1 || out.shape() != expected.shapes[0] ||
        out.dtype() != expected.dtypes[0]) {
      throw std::runtime_error("cross entropy of an unexpected shape or type");
    }
    return {lower(p, out, known)};
  }
  if (call != kCallAttention) {
    throw std::runtime_error("unknown operation " + std::to_string(call));
  }
  auto scale = r.get<float>();
  bool causal = r.get<uint8_t>();
  bool has_mask = r.get<uint8_t>();
  bool has_sinks = r.get<uint8_t>();
  bool force_fused = r.get<uint8_t>();
  if (inputs.size() != 3u + has_mask + has_sinks || (causal && has_mask)) {
    throw std::runtime_error("malformed graph: inputs of an attention");
  }
  std::vector<mx::array> args;
  std::unordered_map<uintptr_t, uint32_t> known;
  for (size_t i = 0; i < inputs.size(); ++i) {
    args.push_back(placeholder(in_shapes[i], in_dtypes[i]));
    known.emplace(args.back().id(), inputs[i]);
  }
  auto out = mx::fast::scaled_dot_product_attention(
      args[0],
      args[1],
      args[2],
      scale,
      causal ? "causal" : "",
      has_mask ? std::optional<mx::array>(args[3]) : std::nullopt,
      has_sinks ? std::optional<mx::array>(args.back()) : std::nullopt,
      force_fused);
  if (expected.shapes.size() != 1 || out.shape() != expected.shapes[0] ||
      out.dtype() != expected.dtypes[0]) {
    throw std::runtime_error("attention of an unexpected shape or type");
  }
  return {lower(p, out, known)};
}

// Urgent graphs (see protocol.h): what the cpu may compute in the place of
// the GPU.
//
// Only what gives the same bytes on both: nodes that select or move elements
// (an index, a slice, a reshape, a concatenation), and comparisons and logic
// on integers and booleans. Nothing that computes with floating point, where
// the two devices may round differently, and no comparison of floating point
// either (the GPU's kernels need not treat a NaN as the cpu does). Such a
// program is built with the primitives of the message moved to the cpu
// stream: in what mlx saves of a primitive the stream comes first, as three
// integers (its index, the kind of device, the index of the device).
//
// The cpu reads memory where the message says, and a wrong read ends this
// process, for every container. So the nodes are checked as they are not for
// the GPU: the shapes are those mlx itself derives from the inputs, the types
// those the cpu's code expects, a slice lies inside its input, and the
// indices of a gather, which are data, are looked at each time before the
// program is run.
struct NotForCpu : std::runtime_error {
  using std::runtime_error::runtime_error;
};
constexpr size_t kMaxEarlyNodes = 64;
constexpr size_t kMaxEarlyBytes = 4 << 20;
// How many GRAPH, OP and LOAD messages after an early answer a connection
// counts as one whose program reads values as it goes.
constexpr uint64_t kEarlyRecent = 256;
// A long program of such a connection pauses for urgent graphs after this
// many nodes at least, and at most.
constexpr size_t kMinNodesPerStop = 16;
constexpr size_t kMaxNodesPerStop = 256;
// How long handing over the nodes between two pauses should take.
constexpr uint64_t kStopNs = 500000;
// With more command buffers than 10 in flight mlx holds the thread that
// evaluates (MAX_ACTIVE_TASKS in its transforms.cpp): the hand-over waits
// two before, the room for what it hands over between two pauses.
constexpr int kMaxActiveTasks = 8;

std::string on_cpu(std::string saved) {
  int32_t stream[3];
  if (saved.size() < sizeof(stream)) {
    throw NotForCpu("malformed primitive");
  }
  std::memcpy(stream, saved.data(), sizeof(stream));
  if (stream[1] != static_cast<int32_t>(mx::Device::gpu)) {
    throw NotForCpu("not a primitive of the gpu");
  }
  stream[1] = static_cast<int32_t>(mx::Device::cpu);
  stream[2] = 0;
  std::memcpy(saved.data(), stream, sizeof(stream));
  return saved;
}

bool is_integer(mx::Dtype dtype) {
  return !mx::issubdtype(dtype, mx::inexact);
}

// The elements of an evaluated, row contiguous array of integers.
template <typename F>
bool each_integer(const mx::array& a, F&& f) {
  auto walk = [&](auto* data) {
    for (size_t i = 0; i < a.size(); ++i) {
      if (!f(static_cast<int64_t>(data[i]))) {
        return false;
      }
    }
    return true;
  };
  switch (a.dtype().val()) {
    case mx::Dtype::Val::uint8:
      return walk(a.data<uint8_t>());
    case mx::Dtype::Val::uint16:
      return walk(a.data<uint16_t>());
    case mx::Dtype::Val::uint32:
      return walk(a.data<uint32_t>());
    case mx::Dtype::Val::int8:
      return walk(a.data<int8_t>());
    case mx::Dtype::Val::int16:
      return walk(a.data<int16_t>());
    case mx::Dtype::Val::int32:
      return walk(a.data<int32_t>());
    case mx::Dtype::Val::int64:
      return walk(a.data<int64_t>());
    default:
      return false;
  }
}

// A node of a program for the cpu: throws unless it is one the cpu computes
// as the GPU would, with the shapes and types of the message.
void check_for_cpu(
    Program& p,
    Node& node,
    const std::string& saved_as,
    const std::vector<mx::Shape>& in_shapes,
    const std::vector<mx::Dtype>& in_dtypes) {
  // How many inputs, and which types: 'm' moves elements (the output has the
  // type of the inputs), 'g' gathers them, 's' selects between two arrays,
  // 'c' compares or combines two arrays of integers or booleans, 'n' negates
  // one, 'r' reduces booleans with and or or, 't' changes the type of
  // integers.
  struct Rule {
    int inputs;
    char kind;
  };
  static const std::unordered_map<std::string, Rule> rules = {
      {"Slice", {1, 'm'}},        {"Squeeze", {1, 'm'}},
      {"ExpandDims", {1, 'm'}},   {"Reshape", {1, 'm'}},
      {"Flatten", {1, 'm'}},      {"Unflatten", {1, 'm'}},
      {"Transpose", {1, 'm'}},    {"Broadcast", {1, 'm'}},
      {"BroadcastAxes", {1, 'm'}}, {"Copy", {1, 'm'}},
      {"StopGradient", {1, 'm'}}, {"Concatenate", {0, 'm'}},
      {"Gather", {0, 'g'}},       {"Select", {3, 's'}},
      {"Equal", {2, 'c'}},        {"NotEqual", {2, 'c'}},
      {"Greater", {2, 'c'}},      {"GreaterEqual", {2, 'c'}},
      {"Less", {2, 'c'}},         {"LessEqual", {2, 'c'}},
      {"LogicalAnd", {2, 'c'}},   {"LogicalOr", {2, 'c'}},
      {"LogicalNot", {1, 'n'}},   {"And", {1, 'r'}},
      {"Or", {1, 'r'}},           {"AsType", {1, 't'}}};
  auto& primitive = *node.primitive;
  if (primitive.stream().device != mx::Device::cpu) {
    throw NotForCpu("the primitive is not on the cpu");
  }
  std::string name = primitive.name();
  auto it = rules.find(name);
  if (it == rules.end()) {
    throw NotForCpu(name + " is computed by the GPU");
  }
  // What mlx's own code relies on to derive a shape from the node's state is
  // looked at before it is asked to (it throws otherwise: the graph takes
  // its turn, where it is refused). By the name the primitive is saved
  // under: the one it gives itself can be that of what it does (a Reduce
  // calls itself And).
  check_shapes(saved_as, primitive, in_shapes, in_dtypes, node.shapes, node.dtypes);
  auto rule = it->second;
  auto n_in = in_shapes.size();
  if (n_in == 0 || (rule.inputs != 0 && n_in != size_t(rule.inputs)) ||
      node.shapes.size() != 1) {
    throw NotForCpu("malformed " + name);
  }
  auto out = node.dtypes[0];
  bool types = true;
  switch (rule.kind) {
    case 'm':
      for (auto& dtype : in_dtypes) {
        types = types && dtype == out;
      }
      break;
    case 'g':
      types = in_dtypes[0] == out;
      for (size_t i = 1; i < n_in; ++i) {
        types = types && is_integer(in_dtypes[i]) && in_dtypes[i] != mx::bool_;
      }
      break;
    case 's':
      types = in_dtypes[0] == mx::bool_ && in_dtypes[1] == out &&
          in_dtypes[2] == out;
      break;
    case 'c':
      types = in_dtypes[0] == in_dtypes[1] && is_integer(in_dtypes[0]) &&
          out == mx::bool_;
      break;
    case 'n':
    case 'r':
      types = in_dtypes[0] == mx::bool_ && out == mx::bool_;
      break;
    case 't':
      types = is_integer(in_dtypes[0]) && is_integer(out);
      break;
  }
  if (!types) {
    throw NotForCpu(name + " of these types is computed by the GPU");
  }
  auto& shape = node.shapes[0];
  if (name == "Slice") {
    // mlx derives no shape for it: every element it takes is in the input.
    auto [start, end, strides] = static_cast<mx::Slice&>(primitive).state();
    auto& in = in_shapes[0];
    bool inside = start.size() == in.size() && strides.size() == in.size() &&
        shape.size() == in.size();
    inside = inside && end.size() == in.size();
    for (size_t d = 0; inside && d < in.size(); ++d) {
      // As many elements as from start to end, each of them in the input.
      int64_t step = strides[d];
      int64_t span = step > 0 ? int64_t(end[d]) - start[d]
                              : int64_t(start[d]) - end[d];
      int64_t count = step == 0 || span <= 0
          ? 0
          : (span + std::abs(step) - 1) / std::abs(step);
      int64_t last = start[d] + (count - 1) * step;
      inside = step != 0 && shape[d] == count &&
          (count == 0 ||
           (start[d] >= 0 && start[d] < in[d] && last >= 0 && last < in[d]));
    }
    if (!inside) {
      throw NotForCpu("a slice that is not inside its input");
    }
    return;
  }
  std::vector<mx::array> inputs;
  for (size_t i = 0; i < n_in; ++i) {
    inputs.push_back(placeholder(in_shapes[i], in_dtypes[i]));
  }
  if (primitive.output_shapes(inputs) != node.shapes) {
    throw NotForCpu(name + " of a shape that mlx does not derive");
  }
  if (rule.kind == 'g') {
    // Whole slices of the input, which start at the indices: where they may
    // start is checked when the program is run, on indices that are inputs
    // of the message.
    auto [axes, slice_sizes] = static_cast<mx::Gather&>(primitive).state();
    auto& in = in_shapes[0];
    bool inside = axes.size() == n_in - 1 && slice_sizes.size() == in.size();
    for (size_t d = 0; inside && d < in.size(); ++d) {
      inside = slice_sizes[d] >= 0 && slice_sizes[d] <= in[d];
    }
    for (size_t i = 0; inside && i < axes.size(); ++i) {
      inside = axes[i] >= 0 && size_t(axes[i]) < in.size() &&
          (node.inputs[i + 1] & kValueIsInput);
      if (inside) {
        p.indices.push_back(
            {node.inputs[i + 1] & ~kValueIsInput,
             in[axes[i]],
             slice_sizes[axes[i]]});
      }
    }
    if (!inside) {
      throw NotForCpu("a gather whose indices cannot be checked");
    }
  }
}

// With cpu the program is one that answers an urgent graph, and anything
// else is refused.
Program parse_program(
    int conn,
    Conn& c,
    ByteReader& r,
    const std::vector<ArraySpec>& in_specs,
    std::vector<uint32_t> keep,
    bool has_kernels,
    bool cpu = false) {
  Program p;
  p.cpu = cpu;
  std::vector<mx::Shape> in_shapes;
  std::vector<mx::Dtype> in_dtypes;
  for (auto& spec : in_specs) {
    in_shapes.push_back(to_shape(spec));
    in_dtypes.push_back(to_dtype(spec));
  }
  // For each value of the message: its shape and type, and what it is in the
  // program (which has more values, when an operation is built here).
  std::vector<mx::Shape> value_shapes;
  std::vector<mx::Dtype> value_dtypes;
  std::vector<uint32_t> value_of;
  auto n_nodes = r.get<uint32_t>();
  if (cpu && n_nodes > kMaxEarlyNodes) {
    throw NotForCpu("too many nodes");
  }
  p.nodes.reserve(n_nodes);
  for (uint32_t i = 0; i < n_nodes; ++i) {
    Node node;
    auto len = r.get<uint32_t>();
    auto flags = len & kNodeFlags;
    len &= ~kNodeFlags;
    if (cpu && flags != 0) {
      throw NotForCpu("an operation that the GPU computes");
    }
    auto* bytes = r.get_bytes(len);
    auto n_in = r.get<uint16_t>();
    std::vector<mx::Shape> node_in_shapes;
    std::vector<mx::Dtype> node_in_dtypes;
    for (uint16_t j = 0; j < n_in; ++j) {
      auto v = r.get<uint32_t>();
      bool is_input = v & kValueIsInput;
      auto index = v & ~kValueIsInput;
      if (index >= (is_input ? in_shapes.size() : value_shapes.size())) {
        throw std::runtime_error("malformed graph: value out of range");
      }
      node.inputs.push_back(is_input ? v : value_of[index]);
      node_in_shapes.push_back(is_input ? in_shapes[index] : value_shapes[index]);
      node_in_dtypes.push_back(is_input ? in_dtypes[index] : value_dtypes[index]);
    }
    auto n_out = r.get<uint16_t>();
    if (n_out == 0) {
      throw std::runtime_error("malformed graph: node without outputs");
    }
    for (uint16_t j = 0; j < n_out; ++j) {
      auto spec = r.get_type();
      node.shapes.push_back(to_shape(spec));
      node.dtypes.push_back(to_dtype(spec));
    }
    value_shapes.insert(value_shapes.end(), node.shapes.begin(), node.shapes.end());
    value_dtypes.insert(value_dtypes.end(), node.dtypes.begin(), node.dtypes.end());
    if (flags == kNodeCall) {
      auto values = lower_call(
          conn,
          c,
          p,
          ByteReader(bytes, len),
          node.inputs,
          node_in_shapes,
          node_in_dtypes,
          node,
          has_kernels);
      value_of.insert(value_of.end(), values.begin(), values.end());
      continue;
    }
    if (flags == kNodeCompiled) {
      // Kept with the other primitives, under a key no saved one can have.
      std::string key("\001compiled");
      key.append(bytes, len);
      auto it = primitives.find(key);
      if (it == primitives.end()) {
        it = primitives
                 .emplace(std::move(key), make_compiled(c, ByteReader(bytes, len)))
                 .first;
      }
      node.primitive = it->second;
      check_compiled(*node.primitive, node_in_shapes, node_in_dtypes, node.shapes, node.dtypes);
    } else if (flags == 0) {
      std::string saved(bytes, len);
      node.primitive = find_primitive(
          c,
          cpu ? on_cpu(saved) : saved,
          node_in_shapes,
          node_in_dtypes,
          node.shapes,
          node.dtypes);
      if (cpu) {
        check_for_cpu(p, node, saved_name(saved), node_in_shapes, node_in_dtypes);
      } else {
        check_node(
            saved_name(saved),
            *node.primitive,
            node_in_shapes,
            node_in_dtypes,
            node.shapes,
            node.dtypes,
            /* in_tape */ false);
      }
    } else {
      throw std::runtime_error("malformed graph: unknown kind of node");
    }
    // Operations that only make a view of their input do not count.
    static const std::unordered_set<std::string> views = {
        "Transpose", "Reshape", "Broadcast", "BroadcastAxes", "Slice",
        "ExpandDims", "Squeeze", "Flatten", "Unflatten", "AsStrided", "Copy",
        "StopGradient", "Depends"};
    bool is_view = views.count(node.primitive->name()) > 0;
    for (size_t j = 0; j < node.shapes.size() && !is_view; ++j) {
      p.max_bytes = std::max(p.max_bytes, bytes_of(node.shapes[j], node.dtypes[j]));
    }
    for (size_t j = 0; j < node.shapes.size(); ++j) {
      value_of.push_back(static_cast<uint32_t>(p.n_values++));
    }
    p.nodes.push_back(std::move(node));
  }
  for (auto k : keep) {
    if (k >= value_of.size()) {
      throw std::runtime_error("malformed graph: output out of range");
    }
    p.keep.push_back(value_of[k]);
  }
  if (cpu && p.max_bytes > kMaxEarlyBytes) {
    throw NotForCpu("arrays too large");
  }
  return p;
}

// Build the lazy arrays of a program on the given inputs; returns the kept
// values.
// With stops: also an array of each node, at which the evaluation of a long
// program can pause (see do_graph).
std::vector<mx::array> run_program(
    const Program& p,
    const std::vector<mx::array>& inputs,
    std::vector<mx::array>* stops = nullptr) {
  std::vector<mx::array> values;
  values.reserve(p.n_values);
  if (stops != nullptr) {
    stops->reserve(p.nodes.size());
  }
  for (auto& node : p.nodes) {
    std::vector<mx::array> in;
    in.reserve(node.inputs.size());
    for (auto v : node.inputs) {
      in.push_back(
          (v & kValueIsInput) ? inputs[v & ~kValueIsInput]
          : (v & kValueIsConstant) ? p.constants[v & ~kValueIsConstant]
                                   : values[v]);
    }
    if (node.shapes.size() == 1) {
      values.emplace_back(
          node.shapes[0], node.dtypes[0], node.primitive, std::move(in));
    } else {
      auto outs = mx::array::make_arrays(
          node.shapes, node.dtypes, node.primitive, in);
      for (auto& o : outs) {
        values.push_back(std::move(o));
      }
    }
    if (stops != nullptr) {
      stops->push_back(values.back());
    }
  }
  std::vector<mx::array> kept;
  kept.reserve(p.keep.size());
  for (auto k : p.keep) {
    kept.push_back(
        (k & kValueIsInput) ? inputs[k & ~kValueIsInput]
        : (k & kValueIsConstant) ? p.constants[k & ~kValueIsConstant]
                                 : values[k]);
  }
  return kept;
}

// Programs by the bytes of their message (without the handles): the same
// graph, a model's step, is sent again and again.
//
// With --compile-graphs a program that comes back is also passed through
// mlx's compile, which fuses chains of element-wise operations into single
// kernels. It was how the functions that mlx.nn compiles (gelu, silu, ...)
// got their fused kernel when mx.compile did nothing in the container; now
// those arrive as Compiled primitives, the graph is the one native MLX would
// run, and compiling it again makes no measurable difference (BGE-M3, 8 and
// 32 sentences). When on, it leaves out graphs seen only a couple of times
// and those that work on small arrays only (a step of token generation):
// measured with tests/genloop.py, fused kernels make those slower.
using ArrayFn =
    std::function<std::vector<mx::array>(const std::vector<mx::array>&)>;
struct Cached {
  std::shared_ptr<Program> program;
  ArrayFn compiled;
  uint64_t uses{0};
  bool compile_failed{false};
};
bool use_compile = false;
// With --slow-build F: building the nodes of a graph takes 1 + F times as
// long (see do_graph). With F = 1, 0.25 ms more per BGE-M3 forward on the
// main thread, the container's times do not change: the GPU is behind the
// main thread by milliseconds, so the time building takes is not waited for.
double slow_build = 0;
constexpr uint64_t kCompileAtUse = 3;
constexpr size_t kCompileMinBytes = 1 << 20;
std::unordered_map<std::string, Cached> programs;
size_t programs_bytes = 0;
constexpr size_t kMaxProgramsBytes = 256 << 20;

std::vector<mx::array> call(
    Conn& c,
    Cached& entry,
    const std::vector<mx::array>& args,
    std::vector<mx::array>* stops = nullptr) {
  entry.uses++;
  if (use_compile && !entry.compile_failed && entry.uses >= kCompileAtUse &&
      entry.program->max_bytes >= kCompileMinBytes) {
    try {
      if (!entry.compiled) {
        auto program = entry.program;
        entry.compiled = mx::compile(
            ArrayFn([program](const std::vector<mx::array>& in) {
              return run_program(*program, in);
            }),
            /* shapeless */ false);
        c.compiles++;
      }
      return entry.compiled(args);
    } catch (const std::exception& e) {
      // Run it as it was sent.
      std::cerr << "compile failed, not used for this graph: " << e.what()
                << std::endl;
      entry.compile_failed = true;
      entry.compiled = nullptr;
    }
  }
  return run_program(*entry.program, args, stops);
}

void reply(Channel& ch, uint8_t status, const void* p, size_t n) {
  char head[kFrameHeader];
  head[0] = static_cast<char>(status);
  uint64_t len = n;
  std::memcpy(head + 1, &len, sizeof(len));
  std::lock_guard lk(ch.write_mtx);
  if (n <= 4096) {
    // One write: a reply in two pieces can be delivered in two.
    char small[kFrameHeader + 4096];
    std::memcpy(small, head, sizeof(head));
    std::memcpy(small + sizeof(head), p, n);
    write_all(ch.fd, small, sizeof(head) + n);
    return;
  }
  write_all(ch.fd, head, sizeof(head));
  write_all(ch.fd, p, n);
}

void reply_error(Channel& ch, const std::string& msg) {
  reply(ch, kReplyError, msg.data(), msg.size());
}

// The same event is attached to all the arrays an evaluation computes in a
// stream.
void add_event(std::vector<mx::Event>& events, const mx::array& a) {
  auto& e = a.event();
  if (!e.valid()) {
    return;
  }
  for (auto& have : events) {
    if (&have.cast<char>() == &e.cast<char>() && have.value() == e.value()) {
      return;
    }
  }
  events.push_back(e);
}

std::string stats_text(const Conn& c) {
  std::ostringstream os;
  os << "graphs=" << c.graphs << " (urgent " << c.urgent << ", early "
     << c.early << ") nodes=" << c.nodes
     << " imports=" << c.imports << " compiles=" << c.compiles
     << " ops=" << c.ops << " kernels=" << c.kernels.size()
     << " uploads=" << c.uploads << " (" << c.upload_bytes << " B)"
     << " reads=" << c.reads << " (" << c.read_bytes << " B)"
     << " files=" << c.opens << " (refused " << c.refused << ")"
     << " loads=" << c.loads << " (" << c.load_bytes << " B, "
     << (c.load_ns + 500000) / 1000000 << " ms)"
     << " syncs=" << c.syncs << " frees=" << c.frees
     << " live_arrays=" << c.arrays.size()
     << " mac_active_memory=" << mx::get_active_memory()
     << " server_kernels=" << kernels.size()
     << " kernel_libraries=" << kernel_libraries.size();
  return os.str();
}

void do_upload(Conn& c, ByteReader& r) {
  auto spec = r.get_spec();
  auto dtype = to_dtype(spec);
  auto shape = to_shape(spec);
  size_t nbytes = bytes_of(shape, dtype);
  if (r.remaining() != nbytes) {
    throw std::runtime_error("malformed upload: size does not match shape");
  }
  auto* bytes = r.get_bytes(nbytes);
  sweep_in_flight();
  // Copied into memory of MLX's own allocator. (The array constructor that
  // wraps a pointer in place gives the GPU wrong data when the pointer is not
  // page aligned, as measured with probe/probe_upload.cpp.)
  // Never from MLX's heap of small buffers (those under 256 bytes): the
  // container keeps small constants here for as long as it runs, and a kernel
  // of mlx 0.32.4 that writes past its own small buffer (the scan along an
  // axis of size one of a strided view does, see
  // probe/native_scan_overflow.py) would silently change them for good.
  auto data = data_buffer(nbytes);
  std::memcpy(data.buffer().raw_ptr(), bytes, nbytes);
  mx::array a(std::move(data), shape, dtype);
  c.arrays.insert_or_assign(spec.handle, std::move(a));
  c.uploads++;
  c.upload_bytes += nbytes;
  if (verbose) {
    std::cerr << "  upload #" << spec.handle << " " << shape << " " << dtype
              << std::endl;
  }
}

void do_open(int conn, Conn& c, ByteReader& r) {
  auto id = r.get<uint32_t>();
  auto size = r.get<uint64_t>();
  auto mtime_ns = r.get<int64_t>();
  auto digest = r.get<uint64_t>();
  std::string name(r.position(), r.remaining());
  std::string why, log;
  std::shared_ptr<File> file;
  if (id == 0) {
    why = "malformed message";
  } else if (c.files.size() >= kMaxOpenFiles && c.files.count(id) == 0) {
    why = "too many files are open on this connection";
  } else {
    file = open_file(name, size, mtime_ns, digest, why, log);
  }
  std::string answer(1, file ? '\1' : '\0');
  if (file) {
    if (verbose) {
      std::cerr << "  open " << printable(name) << ": " << file->path
                << std::endl;
    }
    c.files.insert_or_assign(id, std::move(file));
    c.opens++;
  } else {
    // Always in the log: a container asked for a file it does not get.
    std::cerr << "conn " << conn << ": not reading " << printable(name)
              << ": " << (log.empty() ? why : log) << std::endl;
    answer += why;
    c.refused++;
  }
  reply(*c.channel, kReplyOk, answer.data(), answer.size());
}

void do_load(int conn, Conn& c, ByteReader& r) {
  auto spec = r.get_spec();
  auto dtype = to_dtype(spec);
  auto shape = to_shape(spec);
  auto id = r.get<uint32_t>();
  auto offset = r.get<uint64_t>();
  bool swap = r.get<uint8_t>();
  auto it = c.files.find(id);
  if (it == c.files.end()) {
    throw std::runtime_error("malformed load: the file is not open");
  }
  auto& file = it->second;
  // Inside the file as it was when it was opened, whatever the shape says.
  size_t nbytes = bytes_of(shape, dtype);
  if (offset > file->size || nbytes > file->size - offset) {
    throw std::runtime_error(
        "load refused: the array is not inside " + file->path);
  }
  if (file->changed()) {
    throw std::runtime_error(
        "load refused: " + file->path + " was changed after the container "
        "opened it");
  }
  if (c.arrays.count(spec.handle) != 0) {
    wait_loads(conn, c);
  }
  sweep_in_flight();
  // In memory of MLX's own allocator, never from its heap of small buffers:
  // as for an upload (see do_upload).
  auto data = data_buffer(nbytes);
  auto* dst = static_cast<char*>(data.buffer().raw_ptr());
  mx::array a(std::move(data), shape, dtype);
  c.arrays.insert_or_assign(spec.handle, std::move(a));
  c.unsynced.back().second.push_back(spec.handle);
  c.loads++;
  c.load_bytes += nbytes;
  if (verbose) {
    std::cerr << "  load #" << spec.handle << " " << shape << " " << dtype
              << " at " << offset << std::endl;
  }
  if (nbytes > 0) {
    ReadJob job;
    job.file = file;
    job.dst = dst;
    job.n = nbytes;
    job.offset = offset;
    job.swap = swap ? dtype.size() : 0;
    job.conn = conn;
    job.handle = spec.handle;
    queue_read(std::move(job));
  }
}

bool serve_urgent();

// The start of a GRAPH message, up to its nodes.
struct GraphHead {
  uint8_t flags{0};
  std::vector<ArraySpec> in_specs;
  std::vector<uint32_t> keep;
  std::vector<uint64_t> out_handles;
  std::vector<uint32_t> with_bytes;
  // The key of the program: everything but the handles. The programs are
  // shared by all connections, and a kernel is a number that each connection
  // chose: a graph that says it has kernels is cached under what those
  // numbers stand for on its connection as well (one that does not say so
  // may not have any, see lower_call). The program built for the cpu from
  // the same message has a key of its own.
  std::string key;
};

GraphHead read_head(Conn& c, ByteReader& r, bool cpu = false) {
  GraphHead h;
  h.flags = r.get<uint8_t>();
  auto n_in = r.get<uint32_t>();
  h.in_specs.reserve(n_in);
  bool has_kernels = h.flags & kGraphKernels;
  auto& key = h.key;
  key.assign(1, cpu ? '\2' : has_kernels ? '\1' : '\0');
  if (has_kernels) {
    auto size = static_cast<uint32_t>(c.kernel_key.size());
    key.append(reinterpret_cast<const char*>(&size), sizeof(size));
    key.append(c.kernel_key);
  }
  key.append(reinterpret_cast<const char*>(&n_in), sizeof(n_in));
  for (uint32_t i = 0; i < n_in; ++i) {
    h.in_specs.push_back(r.get_spec());
    auto& spec = h.in_specs.back();
    key.push_back(static_cast<char>(spec.dtype_val));
    key.push_back(static_cast<char>(spec.dtype_size));
    key.push_back(static_cast<char>(spec.shape.size()));
    key.append(
        reinterpret_cast<const char*>(spec.shape.data()),
        spec.shape.size() * sizeof(int32_t));
  }
  auto n_keep = r.get<uint32_t>();
  h.keep.reserve(n_keep);
  for (uint32_t i = 0; i < n_keep; ++i) {
    auto value = r.get<uint32_t>();
    h.keep.push_back(value & ~kKeepFlags);
    h.with_bytes.push_back(value & kKeepFlags);
    h.out_handles.push_back(r.get<uint64_t>());
  }
  key.append(reinterpret_cast<const char*>(&n_keep), sizeof(n_keep));
  key.append(
      reinterpret_cast<const char*>(h.keep.data()),
      h.keep.size() * sizeof(uint32_t));
  key.push_back('|');
  key.append(r.position(), r.remaining());
  return h;
}

void do_graph(int conn, Conn& c, ByteReader& r) {
  auto lap = std::make_unique<Lap>(timers.key);
  auto head = read_head(c, r);
  auto flags = head.flags;
  bool has_kernels = flags & kGraphKernels;
  auto& in_specs = head.in_specs;
  auto& keep = head.keep;
  auto& out_handles = head.out_handles;
  auto& with_bytes = head.with_bytes;
  auto& key = head.key;
  auto n_in = in_specs.size();
  auto n_keep = keep.size();
  c.graphs++;
  if (flags & kGraphUrgent) {
    c.urgent++;
  }

  lap = std::make_unique<Lap>(timers.inputs);
  std::vector<mx::array> args;
  args.reserve(n_in);
  for (auto& spec : in_specs) {
    args.push_back(input(c, spec));
  }

  lap = std::make_unique<Lap>(timers.key);
  auto it = programs.find(key);
  if (it == programs.end()) {
    lap = std::make_unique<Lap>(timers.parse);
    auto program = std::make_shared<Program>(
        parse_program(conn, c, r, in_specs, std::move(keep), has_kernels));
    if (programs_bytes + key.size() > kMaxProgramsBytes) {
      programs.clear();
      programs_bytes = 0;
    }
    programs_bytes += key.size();
    it = programs.emplace(std::move(key), Cached{std::move(program)}).first;
  }
  auto n_nodes = it->second.program->nodes.size();
  c.nodes += n_nodes;
  timers.graphs++;
  timers.nodes += n_nodes;
  lap = std::make_unique<Lap>(timers.build);
  auto build_start = Clock::now();
  // A program that reads values while its next step is already here (it had
  // an urgent graph answered early not long ago) must not wait for that
  // whole step to be handed to the GPU, which takes this thread milliseconds,
  // or longer when mlx holds it back until the GPU has caught up: the step is
  // handed over some nodes at a time, and the urgent graphs that arrived
  // meanwhile are answered in between.
  std::vector<mx::array> stops;
  bool pauses = (flags & kGraphEval) && c.last_early != 0 &&
      c.last_early + kEarlyRecent > c.seq && n_nodes >= 2 * kMinNodesPerStop;
  auto outs = call(c, it->second, args, pauses ? &stops : nullptr);
  // (The programs may change while urgent graphs are answered.)
  it = programs.end();
  if (slow_build > 0) {
    // A measurement: is the time building takes time the container waits?
    auto until = Clock::now() +
        std::chrono::nanoseconds(
            static_cast<int64_t>(slow_build * ns_since(build_start)));
    while (Clock::now() < until) {
    }
  }
  lap = std::make_unique<Lap>(timers.store);
  if (verbose) {
    std::cerr << "  graph: " << n_nodes << " nodes, " << n_in << " inputs, "
              << n_keep << " outputs:";
    for (size_t i = 0; i < outs.size() && i < 8; ++i) {
      std::cerr << " #" << out_handles[i] << "="
                << (outs[i].has_primitive() ? outs[i].primitive().name()
                                            : std::string("input"))
                << outs[i].shape();
    }
    std::cerr << std::endl;
  }
  for (size_t i = 0; i < outs.size(); ++i) {
    c.arrays.insert_or_assign(out_handles[i], outs[i]);
  }
  lap = std::make_unique<Lap>(timers.eval);
  if (flags & kGraphEval) {
    // Start the GPU without waiting for it.
    // With pauses: a few nodes at a time, then more while handing them over
    // takes this thread less than half a millisecond, up to a few hundred,
    // and fewer as soon as it takes longer. When the GPU has nearly as many
    // command buffers still to run as mlx lets it have, mlx would hold this
    // thread inside async_eval until one is done: the wait is here instead,
    // and ends when an urgent graph arrives.
    size_t at = c.nodes_per_stop;
    size_t let_go = 0;
    while (at + c.nodes_per_stop / 2 < stops.size()) {
      bool held = false;
      while (mx::scheduler::n_active_tasks() > kMaxActiveTasks) {
        held = true;
        {
          std::unique_lock lk(queue_mtx);
          queue_cv.wait_for(lk, std::chrono::microseconds(200), [] {
            return urgent_queued > 0;
          });
        }
        while (urgent_queued > 0 && serve_urgent()) {
        }
      }
      auto start = Clock::now();
      if (stops[at - 1].status() == mx::array::Status::unscheduled) {
        mx::async_eval({stops[at - 1]});
      }
      auto took = ns_since(start);
      // What was handed over is kept by what still uses it, as natively.
      while (let_go < at) {
        auto drop = std::move(stops[let_go++]);
      }
      while (urgent_queued > 0 && serve_urgent()) {
      }
      if (took > 2 * kStopNs) {
        c.nodes_per_stop = std::max<size_t>(
            kMinNodesPerStop, c.nodes_per_stop * kStopNs / took);
        c.fast_stops = 0;
      } else if (held || took > kStopNs) {
        c.fast_stops = 0;
      } else if (++c.fast_stops >= 16 && c.nodes_per_stop < kMaxNodesPerStop) {
        c.nodes_per_stop *= 2;
        c.fast_stops = 0;
      }
      at += c.nodes_per_stop;
    }
    stops.clear();
    mx::async_eval(outs);
    for (auto& out : outs) {
      add_event(c.piece_events, out);
    }
  }
  if (flags & kGraphTell) {
    Told told;
    told.seq = c.seq;
    // A message before this one that failed and had no reply to say so: said
    // here. This frame tells the container that everything up to this graph
    // is done, and it would take that for "and went well".
    told.error = std::move(c.error);
    c.error.clear();
    for (auto& e : c.piece_events) {
      told.waiters.push_back(waiter(e));
    }
    c.piece_events.clear();
    for (size_t i = 0; i < outs.size() && (flags & kGraphEval); ++i) {
      auto& a = outs[i];
      // What is not sent here the container asks for with READ.
      if (with_bytes[i] && a.status() != mx::array::Status::unscheduled &&
          sendable(a)) {
        auto& list =
            (with_bytes[i] & kKeepWithBytes) ? told.arrays : told.later;
        list.push_back({out_handles[i], a, a.data<char>(), a.nbytes()});
      }
    }
    c.channel->tell(std::move(told));
  } else if (c.piece_events.size() > 256) {
    // A client that never asks to be told.
    c.piece_events.erase(c.piece_events.begin(), c.piece_events.begin() + 128);
  }
  c.unsynced.back().second = std::move(out_handles);
}

// An urgent graph (see protocol.h), taken out of the queue ahead of the
// messages before it: answer it now if the cpu can, with an EARLY frame.
// False if it cannot, and then nothing was done: the graph takes its turn.
// It can when every input is an array that exists and is computed (so no
// message still queued makes or changes it: a container never uses a handle
// twice), seen with its own shape and type, and when the nodes are of those
// the cpu computes as the GPU would (see check_for_cpu).
// And only when nothing is owed to the container. An EARLY frame carries no
// error, and a container whose graphs were all answered early takes
// everything it sent for done: so no graph is answered early while a failure
// of an earlier message waits to be told (in its turn the graph gets a DONE
// frame, which carries it), nor ahead of a message that could still fail
// without a reply (serve_urgent), nor for a connection that has not greeted
// (in its turn the graph is refused, like any other message).
bool run_urgent(Job& job) {
  auto found = conns.find(job.conn);
  if (found == conns.end() || !found->second.channel) {
    return false;
  }
  auto& c = found->second;
  if (!c.greeted || !c.error.empty()) {
    if (verbose) {
      std::cerr << "  urgent graph " << job.seq << " takes its turn: "
                << (c.greeted ? "a failure is still to be told"
                              : "the container has not greeted")
                << std::endl;
    }
    return false;
  }
  // With -v: why it is left to its turn.
  auto no = [&job](const char* why) {
    if (verbose) {
      std::cerr << "  urgent graph " << job.seq << " takes its turn: " << why
                << std::endl;
    }
    return false;
  };
  // No array may be one that a file is still being read into.
  if (loads_pending != 0 || loads_failed || loads_took != 0) {
    return no("files are being read");
  }
  auto start = Clock::now();
  std::vector<mx::array> outs;
  std::vector<mx::array> bytes;
  GraphHead head;
  size_t n_nodes = 0;
  try {
    ByteReader r(job.payload.get(), job.size);
    head = read_head(c, r, /* cpu */ true);
    if ((head.flags & (kGraphEval | kGraphTell | kGraphKernels)) !=
        (kGraphEval | kGraphTell)) {
      return no("not a graph to evaluate and tell about");
    }
    std::vector<mx::array> args;
    args.reserve(head.in_specs.size());
    for (auto& spec : head.in_specs) {
      auto it = c.arrays.find(spec.handle);
      if (it == c.arrays.end()) {
        return no("an input is still to be made");
      }
      if (!it->second.is_available()) {
        return no("an input is still being computed");
      }
      if (it->second.dtype() != to_dtype(spec) ||
          it->second.shape() != to_shape(spec)) {
        return no("an input is seen with another shape or type");
      }
      // The cpu reads it with this process's own code.
      if (!inside(it->second)) {
        return no("an input is larger than the memory it has");
      }
      args.push_back(it->second);
    }
    for (auto with : head.with_bytes) {
      if (with & kKeepWithBytesLater) {
        return no("a large result");
      }
    }
    auto it = programs.find(head.key);
    if (it == programs.end()) {
      // What the cpu cannot answer is remembered as such.
      std::shared_ptr<Program> program;
      try {
        program = std::make_shared<Program>(parse_program(
            job.conn, c, r, head.in_specs, head.keep, false, /* cpu */ true));
      } catch (const std::exception& e) {
        no(e.what());
      }
      if (programs_bytes + head.key.size() > kMaxProgramsBytes) {
        programs.clear();
        programs_bytes = 0;
      }
      programs_bytes += head.key.size();
      it = programs.emplace(head.key, Cached{std::move(program)}).first;
    }
    if (!it->second.program) {
      return no("not for the cpu");
    }
    n_nodes = it->second.program->nodes.size();
    for (auto& index : it->second.program->indices) {
      auto& a = args[index.input];
      bool is_signed = mx::issubdtype(a.dtype(), mx::signedinteger);
      if (!a.flags().row_contiguous ||
          !each_integer(a, [&](int64_t i) {
            i += (is_signed && i < 0) ? index.size : 0;
            return i >= 0 && i + index.slice <= index.size;
          })) {
        return no("an index that the cpu cannot check, or out of range");
      }
    }
    outs = run_program(*it->second.program, args);
    // The bytes that go back, row major. (Only what is still to be computed
    // is evaluated: with nothing to compute mlx would wait on the GPU.)
    std::vector<mx::array> todo;
    for (size_t i = 0; i < outs.size(); ++i) {
      if (outs[i].status() == mx::array::Status::unscheduled) {
        todo.push_back(outs[i]);
      }
    }
    if (!todo.empty()) {
      mx::eval(todo);
    }
    todo.clear();
    for (size_t i = 0; i < outs.size(); ++i) {
      bytes.push_back(outs[i]);
      if (head.with_bytes[i] && !outs[i].flags().row_contiguous) {
        bytes.back() = mx::contiguous(outs[i], false, mx::Device::cpu);
        todo.push_back(bytes.back());
      }
    }
    if (!todo.empty()) {
      mx::eval(todo);
    }
    for (size_t i = 0; i < outs.size(); ++i) {
      if (head.with_bytes[i] && !sendable(bytes[i])) {
        return no("a result is larger than the memory it has");
      }
    }
  } catch (const std::exception& e) {
    // Whatever is wrong with it is said when its turn comes.
    return no(e.what());
  }
  for (size_t i = 0; i < outs.size(); ++i) {
    c.arrays.insert_or_assign(head.out_handles[i], outs[i]);
  }
  c.graphs++;
  c.urgent++;
  c.early++;
  c.last_early = job.seq;
  c.nodes += n_nodes;
  ByteWriter w;
  w.put<uint8_t>(kReplyEarly);
  w.put<uint64_t>(0);
  w.put<uint64_t>(job.seq);
  w.put<uint32_t>(0);
  uint32_t count = 0;
  for (size_t i = 0; i < outs.size(); ++i) {
    count += head.with_bytes[i] != 0;
  }
  w.put<uint32_t>(count);
  for (size_t i = 0; i < outs.size(); ++i) {
    if (head.with_bytes[i]) {
      w.put<uint64_t>(head.out_handles[i]);
      w.put<uint64_t>(bytes[i].nbytes());
      w.put_bytes(bytes[i].data<char>(), bytes[i].nbytes());
    }
  }
  uint64_t len = w.str().size() - kFrameHeader;
  std::memcpy(w.str().data() + 1, &len, sizeof(len));
  {
    std::lock_guard lk(c.channel->write_mtx);
    write_all(c.channel->fd, w.str().data(), w.str().size());
  }
  if (verbose || timing) {
    std::cerr << "  " << (timing ? "timing " : "") << "early " << job.seq
              << ": " << n_nodes << " nodes on the cpu, answered "
              << (ns_since(job.queued) + 500) / 1000
              << " us after it arrived, of which here "
              << (ns_since(start) + 500) / 1000 << " us" << std::endl;
  }
  return true;
}

void do_op(Conn& c, ByteReader& r) {
  auto op = r.get<uint32_t>();
  auto n_in = r.get<uint32_t>();
  std::vector<mx::array> args;
  for (uint32_t i = 0; i < n_in; ++i) {
    auto spec = r.get_spec();
    args.push_back(input(c, spec));
  }
  auto n_out = r.get<uint32_t>();
  std::vector<uint64_t> out_handles;
  for (uint32_t i = 0; i < n_out; ++i) {
    out_handles.push_back(r.get<uint64_t>());
  }
  c.ops++;
  std::vector<mx::array> outs;
  switch (op) {
    case kOpBitwiseInvert:
      if (args.size() != 1) {
        throw std::runtime_error("bitwise_invert takes one array");
      }
      outs.push_back(mx::bitwise_invert(args[0]));
      break;
    default:
      throw std::runtime_error("unknown operation " + std::to_string(op));
  }
  if (outs.size() != out_handles.size()) {
    throw std::runtime_error("operation returned an unexpected number of arrays");
  }
  for (size_t i = 0; i < outs.size(); ++i) {
    c.arrays.insert_or_assign(out_handles[i], outs[i]);
  }
  c.unsynced.back().second = std::move(out_handles);
  mx::async_eval(outs);
}

void do_read(Conn& c, ByteReader& r) {
  auto handle = r.get<uint64_t>();
  auto it = c.arrays.find(handle);
  if (it == c.arrays.end()) {
    throw std::runtime_error(
        "array #" + std::to_string(handle) +
        " does not exist (an earlier graph failed, or it was freed)");
  }
  auto a = it->second;
  a.eval();
  if (!a.flags().row_contiguous) {
    a = mx::contiguous(a);
    a.eval();
  }
  if (!sendable(a)) {
    throw std::runtime_error(
        "array #" + std::to_string(handle) + " is larger than the memory it "
        "has: the shape or type its node was given is not what the node makes");
  }
  c.reads++;
  c.read_bytes += a.nbytes();
  reply(*c.channel, kReplyOk, a.data<char>(), a.nbytes());
}

void run(Job& job) {
  auto& c = conns[job.conn];
  if (job.type == 0) {
    wait_loads(job.conn, c, false);
    std::cerr << "conn " << job.conn << " closed: " << stats_text(c)
              << std::endl;
    for (auto& [handle, a] : c.arrays) {
      if (!a.is_available()) {
        in_flight.push_back(std::move(a));
      }
    }
    if (c.channel) {
      c.channel->close();
    }
    conns.erase(job.conn);
    sweep_in_flight();
    ::close(job.fd);
    // The kernels that only this connection had sent go with it.
    release_kernels();
    if (conns.empty()) {
      // Nobody is left to use the buffers MLX keeps for reuse: a model that
      // was loaded here gives its memory back to the Mac. So do the programs
      // and primitives kept for the graphs that come again: they hold the
      // bytes of the messages they were built from, and arrays of their own
      // (the constants of the operations built here, the scalars of compiled
      // functions), which would otherwise stay until the server ends. The
      // next container builds its own.
      for (auto& s : mx::get_streams()) {
        mx::synchronize(s);
      }
      in_flight.clear();
      programs.clear();
      programs_bytes = 0;
      primitives.clear();
      compiled_infos.clear();
      mx::clear_cache();
    }
    return;
  }
  if (!c.channel) {
    c.channel = std::make_unique<Channel>(job.fd);
  }
  bool has_reply = job.type == kHello || job.type == kRead ||
      job.type == kSync || job.type == kStats || job.type == kOpen;
  // With -v: when each message started (since the previous one ended) and
  // how long it took.
  static auto last_end = std::chrono::steady_clock::now();
  auto start = std::chrono::steady_clock::now();
  if (timing) {
    timers.messages++;
    timers.bytes += job.size;
    timers.receive += (job.queued - job.arrived).count();
    timers.queued += (start - job.queued).count();
  }
  struct Timing {
    std::chrono::steady_clock::time_point start;
    uint8_t type;
    bool told;
    ~Timing() {
      auto end = std::chrono::steady_clock::now();
      if (timing && (type == kSync || type == kRead || told)) {
        auto us = [](uint64_t ns) { return (ns + 500) / 1000; };
        auto& t = timers;
        auto took = static_cast<uint64_t>((end - start).count());
        if (type == kGraph) {
          t.other += took;
        }
        auto other = t.other - t.key - t.inputs - t.parse - t.build - t.store -
            t.eval;
        std::cerr << "  timing "
                  << (type == kSync ? "sync" : type == kRead ? "read" : "told")
                  << ": "
                  << t.messages << " messages, " << t.bytes << " B, "
                  << t.graphs << " graphs, " << t.nodes << " nodes; receive "
                  << us(t.receive) << ", queued " << us(t.queued) << ", key "
                  << us(t.key) << ", inputs " << us(t.inputs) << ", parse "
                  << us(t.parse) << " (" << t.imports << " new primitives "
                  << us(t.import) << "), build " << us(t.build) << ", store "
                  << us(t.store) << ", start gpu " << us(t.eval)
                  << ", other messages " << us(other) << ", this "
                  << us(took) << ", idle before " << us((start - last_end).count())
                  << " us" << std::endl;
        timers = Timers{};
      } else if (timing) {
        timers.other += (end - start).count();
      }
      if (verbose && (type == kGraph || type == kSync || type == kRead)) {
        auto us = [](auto d) {
          return std::chrono::duration_cast<std::chrono::microseconds>(d)
              .count();
        };
        std::cerr << "  " << (type == kGraph ? "graph" : type == kSync ? "sync" : "read")
                  << " took " << us(end - start) << " us (idle before: "
                  << us(start - last_end) << " us)" << std::endl;
      }
      last_end = end;
    }
  } timing{
      start,
      job.type,
      job.type == kGraph && job.size > 0 && (job.payload[0] & kGraphTell)};
  try {
    ByteReader r(job.payload.get(), job.size);
    // The arrays being read from files are complete before anything but the
    // next of them goes on.
    if (job.type != kLoad && job.type != kOpen && job.type != kHello) {
      wait_loads(job.conn, c);
    }
    if (job.type != kHello && !c.greeted) {
      throw std::runtime_error(
          "a message before the greeting: the container must first say which "
          "protocol and which MLX it speaks");
    }
    // An error of an earlier one-way message comes first.
    if (has_reply && job.type != kHello && !c.error.empty()) {
      auto msg = std::move(c.error);
      c.error.clear();
      if (verbose) {
        std::cerr << "conn " << job.conn << ": error: " << msg.substr(1) << std::endl;
      }
      reply_error(*c.channel, msg);
      return;
    }
    switch (job.type) {
      case kHello: {
        auto version = r.get<uint32_t>();
        if (version != kProtocolVersion) {
          throw std::runtime_error(
              "protocol version " + std::to_string(version) +
              " not supported, this server speaks " +
              std::to_string(kProtocolVersion));
        }
        // The same MLX on both ends: a node travels as the bytes MLX saves a
        // primitive in, and another MLX saves, and means, another thing.
        auto theirs = r.get_text();
        if (theirs.size() > 64) {
            throw std::runtime_error("malformed greeting");
        }
        if (theirs != mx::version()) {
          throw std::runtime_error(
              "the container runs MLX " + printable(theirs) + ", this server " +
              mx::version() + ": the two must be the same version");
        }
        c.greeted = true;
        ByteWriter w;
        w.put<uint32_t>(custom_kernels ? kFeatureKernels : 0);
        w.put_text(std::string(mx::version()));
        std::string text = std::string("mlx ") + mx::version() + " on " +
            (mx::gpu::is_available() ? "the Mac's GPU" : "cpu only");
        w.put_bytes(text.data(), text.size());
        reply(*c.channel, kReplyOk, w.str().data(), w.str().size());
        break;
      }
      case kUpload:
        do_upload(c, r);
        break;
      case kGraph:
        if (c.unsynced.size() > 4096) {
          // A client that never waits: only the recent ones matter.
          c.unsynced.pop_front();
        }
        c.seq = job.seq;
        c.unsynced.emplace_back(c.seq, std::vector<uint64_t>{});
        do_graph(job.conn, c, r);
        break;
      case kOp:
        c.seq = job.seq;
        c.unsynced.emplace_back(c.seq, std::vector<uint64_t>{});
        do_op(c, r);
        break;
      case kFree: {
        auto n = r.get<uint32_t>();
        sweep_in_flight();
        for (uint32_t i = 0; i < n; ++i) {
          release_array(c, r.get<uint64_t>());
        }
        c.frees += n;
        break;
      }
      case kRead:
        do_read(c, r);
        break;
      case kSync: {
        auto upto = r.get<uint64_t>();
        if (upto == 0) {
          c.unsynced.clear();
          for (auto& s : mx::get_streams()) {
            mx::synchronize(s);
          }
        }
        // The arrays a message made are evaluated once those it kept are.
        while (!c.unsynced.empty() && c.unsynced.front().first <= upto) {
          for (auto handle : c.unsynced.front().second) {
            if (auto it = c.arrays.find(handle); it != c.arrays.end()) {
              it->second.eval();
            }
          }
          c.unsynced.pop_front();
        }
        c.syncs++;
        reply(*c.channel, kReplyOk, "", 0);
        break;
      }
      case kStats: {
        auto text = stats_text(c);
        reply(*c.channel, kReplyOk, text.data(), text.size());
        break;
      }
      case kOpen:
        do_open(job.conn, c, r);
        break;
      case kLoad:
        if (c.unsynced.size() > 4096) {
          c.unsynced.pop_front();
        }
        c.seq = job.seq;
        c.unsynced.emplace_back(c.seq, std::vector<uint64_t>{});
        do_load(job.conn, c, r);
        break;
      case kClose:
        c.files.erase(r.get<uint32_t>());
        break;
      case kKernel:
        do_kernel(job.conn, c, r);
        break;
    }
  } catch (const std::exception& e) {
    if (verbose || !has_reply) {
      std::cerr << "conn " << job.conn << ": error: " << e.what() << std::endl;
    }
    if (has_reply) {
      // (The answer to a greeting is text alone: an end of another version
      // must be able to read why it is not greeted.)
      reply_error(*c.channel, job.type == kHello ? std::string(e.what()) : encode_error(e));
    } else if (job.type == kGraph && job.size > 0 &&
               (job.payload[0] & kGraphTell)) {
      // The container is told about this graph in any case, and waits for it.
      Told told;
      told.seq = job.seq;
      told.error = c.error.empty() ? encode_error(e) : c.error;
      c.error.clear();
      c.piece_events.clear();
      c.channel->tell(std::move(told));
    } else if (c.error.empty()) {
      c.error = encode_error(e);
    }
  }
}

// Take the first urgent graph out of the queue, ahead of the messages
// before it and with the uploads right before it (its own operands: an
// upload makes a new array, whenever it is run), and answer it; if it cannot
// be answered now it goes back to its place. False if there was none.
// (Only this thread takes jobs out: a place in the queue stays as far from
// the front as it was.)
bool serve_urgent() {
  Job job;
  std::vector<Job> uploads;
  size_t place = 0;
  {
    std::lock_guard lk(queue_mtx);
    size_t i = 0;
    while (i < queue.size() && !queue[i].urgent) {
      i++;
    }
    if (i == queue.size()) {
      return false;
    }
    place = i;
    while (place > 0 && queue[place - 1].conn == queue[i].conn &&
           queue[place - 1].type == kUpload) {
      place--;
    }
    // What it would be answered ahead of, of its own connection, must be
    // messages whose failure the container hears in order whatever happens:
    // those it waits for (their reply says so), the numbered ones (a
    // container counts nothing done past one of them before its DONE frame
    // or a SYNC, which carry the failure), and those that cannot fail (a
    // well-formed FREE or CLOSE). Behind any other (a kernel that may be
    // refused, an upload that is not its own operand) it takes its turn.
    for (size_t k = 0; k < place; ++k) {
      auto& before = queue[k];
      if (before.conn != queue[i].conn) {
        continue;
      }
      uint32_t n = 0;
      if (before.type == kFree && before.size >= sizeof(n)) {
        std::memcpy(&n, before.payload.get(), sizeof(n));
      }
      bool in_order = before.type == kGraph || before.type == kOp ||
          before.type == kLoad || before.type == kHello ||
          before.type == kRead || before.type == kSync ||
          before.type == kStats || before.type == kOpen ||
          (before.type == kFree && before.size == sizeof(n) + 8 * size_t(n)) ||
          (before.type == kClose && before.size >= sizeof(uint32_t));
      if (!in_order) {
        queue[i].urgent = false;
        urgent_queued--;
        return true;
      }
    }
    for (size_t k = place; k < i; ++k) {
      uploads.push_back(std::move(queue[k]));
    }
    job = std::move(queue[i]);
    queue.erase(queue.begin() + place, queue.begin() + i + 1);
    urgent_queued--;
  }
  for (auto& upload : uploads) {
    run(upload);
  }
  job.urgent = false;
  if (!run_urgent(job)) {
    std::lock_guard lk(queue_mtx);
    queue.insert(queue.begin() + place, std::move(job));
  }
  return true;
}

void on_signal(int) {
  if (!unix_path.empty()) {
    ::unlink(unix_path.c_str());
  }
  if (function_fd >= 0) {
    ::unlink(function_path.c_str());
  }
  _exit(0);
}

} // namespace

int main(int argc, char** argv) {
  int port = kDefaultPort;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-v") {
      verbose = true;
    } else if (arg == "--timing") {
      timing = true;
    } else if (arg == "--compile-graphs") {
      use_compile = true;
    } else if (arg == "--no-compile") {
      // The default now; accepted for the scripts that pass it.
      use_compile = false;
    } else if (arg == "--custom-kernels") {
      custom_kernels = true;
    } else if (arg == "--slow-build" && i + 1 < argc) {
      slow_build = std::atof(argv[++i]);
    } else if (arg == "--port" && i + 1 < argc) {
      port = std::atoi(argv[++i]);
    } else if (arg == "--unix" && i + 1 < argc) {
      unix_path = argv[++i];
    } else if (arg == "--models" && i + 1 < argc) {
      std::string dir = argv[++i];
      if (!add_root(dir, dir)) {
        return 2;
      }
    } else if (arg == "--map" && i + 1 < argc) {
      std::string map = argv[++i];
      auto eq = map.find('=');
      if (eq == std::string::npos) {
        std::cerr << "--map takes CONTAINER_DIR=DIR" << std::endl;
        return 2;
      }
      if (!add_root(map.substr(0, eq), map.substr(eq + 1))) {
        return 2;
      }
    } else {
      std::cerr << "usage: mlx-remote-server [--port N | --unix PATH] [-v] "
                   "[--timing] [--compile-graphs] [--slow-build F] "
                   "[--custom-kernels] "
                   "[--models DIR]... [--map CONTAINER_DIR=DIR]..."
                << std::endl;
      return 2;
    }
  }
  std::signal(SIGPIPE, SIG_IGN);
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  int lfd;
  if (!unix_path.empty()) {
    lfd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (unix_path.size() >= sizeof(addr.sun_path)) {
      std::cerr << "socket path too long" << std::endl;
      return 1;
    }
    std::strcpy(addr.sun_path, unix_path.c_str());
    ::unlink(unix_path.c_str());
    if (::bind(lfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      std::perror("bind");
      return 1;
    }
    ::chmod(unix_path.c_str(), 0600);
  } else {
    lfd = ::socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    ::setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    // Loopback only: never reachable from other machines.
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::bind(lfd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
      std::perror("bind");
      return 1;
    }
  }
  if (::listen(lfd, 16) != 0) {
    std::perror("listen");
    return 1;
  }

  // The streams the imported graphs resolve to belong to this thread.
  if (!mx::gpu::is_available()) {
    std::cerr << "no GPU available to MLX on this machine" << std::endl;
    return 1;
  }
  mx::default_stream(mx::Device::gpu);
  std::cerr << "mlx-remote-server: mlx " << mx::version() << ", listening on "
            << (unix_path.empty() ? "127.0.0.1:" + std::to_string(port)
                                  : unix_path)
            << std::endl;
  if (custom_kernels) {
    std::cerr << "compiles and runs on this Mac's GPU the Metal kernels the "
              << "containers send (--custom-kernels)" << std::endl;
  }
  for (auto& r : roots) {
    std::cerr << "reads the files of weights under " << r.mac
              << (r.container == r.mac
                      ? std::string()
                      : " (" + r.container + " in a container)")
              << std::endl;
  }

  std::thread([lfd]() {
    int next_conn = 1;
    while (true) {
      int fd = ::accept(lfd, nullptr, nullptr);
      if (fd < 0) {
        if (errno == EINTR) {
          continue;
        }
        std::perror("accept");
        _exit(1);
      }
      int one = 1;
      ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
      ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one));
      int conn = next_conn++;
      std::cerr << "conn " << conn << " opened" << std::endl;
      std::thread(reader_thread, conn, fd).detach();
    }
  }).detach();

  while (true) {
    Job job;
    {
      std::unique_lock lk(queue_mtx);
      queue_cv.wait(lk, [] { return !queue.empty(); });
      if (urgent_queued == 0) {
        job = std::move(queue.front());
        queue.pop_front();
      }
    }
    if (job.fd < 0) {
      serve_urgent();
      continue;
    }
    run(job);
  }
}
