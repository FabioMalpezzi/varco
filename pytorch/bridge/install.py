"""Makes the PyTorch of this process one with an "mps" device: the Mac's GPU, through the bridge.

install() runs once, right after `import torch` (auto.py arranges that for every Python process
in the container, with no change to the program). Most of what a program asks about MPS is
answered by PyTorch's own code, through native.cpp's hooks; what is here is the rest:

- what PyTorch decided when it was built for Linux, and the libraries that look at the machine
  instead of asking PyTorch;
- saving and loading, which work on a tensor's storage, and here an MPS tensor has none;
- torch.Event, for the copies a program asks for without waiting.
"""

import functools
import platform

import bridge_native
import torch

from .client import Bridge

bridge = None


def install():
    global bridge
    if bridge is not None:
        return bridge
    bridge = Bridge()

    # "Was this PyTorch built with MPS?" is a constant of the build, read by torch.backends.mps.is_built(),
    # torch.mps.manual_seed() and torch.mps.device_count(). It was not; it has MPS all the same.
    torch._C._has_mps = True
    if hasattr(torch.backends.mps.is_built, "cache_clear"):
        torch.backends.mps.is_built.cache_clear()

    # "Is MPS available?" is what a program asks before choosing its device: here the answer is whether
    # the Mac's end can be reached, found out by reaching it. No is not remembered: the server may be
    # started later. (PyTorch's own function is cached, and other code reads its __wrapped__.)
    def is_available():
        try:
            bridge.connect()
        except RuntimeError:
            return False
        return True

    is_available.__wrapped__ = is_available
    is_available.cache_clear = lambda: None
    is_available.__doc__ = torch.backends.mps.is_available.__doc__
    torch.backends.mps.is_available = is_available
    count_here = torch.mps.device_count

    def device_count():  # torch.mps.is_available() is this, more than zero
        return int(is_available())

    device_count.__doc__ = count_here.__doc__
    device_count.cache_clear = lambda: None
    torch.mps.device_count = device_count

    # Lightning asks the machine, not PyTorch, whether it is an Apple one: MPSAccelerator.is_available()
    # wants platform.processor() to say "arm". In a Linux container on Apple silicon it says nothing
    # ("" or "aarch64"). The GPU this process computes on is an Apple one.
    platform.processor = lambda: "arm"

    # torch.manual_seed on a Mac seeds the CPU's generator and the GPU's. The GPU's is reached through
    # the hooks (torch.mps.manual_seed is PyTorch's own); the Mac's CPU generator, which the few MPS
    # operators that fall back to the CPU draw from, is seeded here.
    seed_here = torch.manual_seed

    def manual_seed(seed):
        bridge.call("manual_seed", int(seed))
        return seed_here(seed)

    manual_seed.__doc__ = seed_here.__doc__
    torch.manual_seed = torch.random.manual_seed = manual_seed

    # torch.use_deterministic_algorithms (and set_deterministic_debug_mode, which comes to the same function) is
    # a setting of the process that computes. What runs on the GPU runs in the Mac's process for this program:
    # it is told too, so that an operator PyTorch refuses under that setting on a Mac is refused here.
    deterministic_here = torch._C._set_deterministic_algorithms

    def set_deterministic_algorithms(mode, *, warn_only=False):
        deterministic_here(mode, warn_only=warn_only)
        bridge.call("use_deterministic_algorithms", bool(mode), bool(warn_only))

    torch._C._set_deterministic_algorithms = set_deterministic_algorithms

    # Backward passes for a GPU run on a thread of their own: keep them on the
    # program's thread, so that operations reach the Mac in the program's order.
    torch.autograd.set_multithreading_enabled(False)

    _saving()
    _loading()
    _events()
    _newer()
    return bridge


def _newer():
    """What torch.mps has gained since, which PyTorch for Linux was built without (torch._C lacks the
    functions torch.mps calls). What has a meaning through the bridge is given one; the rest fails
    by saying so, not with "module torch._C has no attribute"."""
    core = torch._C
    for name, call in (("_mps_get_name", "get_name"), ("_mps_get_core_count", "get_core_count")):  # which GPU it is: the Mac's
        if not hasattr(core, name):
            setattr(core, name, functools.partial(bridge.call, call))

    def unsupported(name):
        def call(*arguments, **options):
            raise NotImplementedError(f"torch.mps: {name[5:]} is not available through the bridge (it would have to run on the Mac)")

        return call

    for name in ("_mps_compileShader", "_mps_loadMetallibFromPath", "_mps_loadMetalllib", "_mps_startCapture", "_mps_stopCapture",
                 "_mps_isCaptureEnabled", "_mps_isCapturing"):  # Metal shaders of the program's own, GPU captures
        if not hasattr(core, name):
            setattr(core, name, unsupported(name))

    streams = getattr(torch.mps, "streams", None)
    if streams is None or hasattr(core, "_mps_getCurrentStream"):
        return

    class Stream:
        """torch.mps.Stream. A program's operations run on the Mac in one sequence, the default stream: that
        one exists, and can be asked for and waited on; making another is refused."""

        def __init__(self, stream_id=None, **options):
            if stream_id != 0:
                raise NotImplementedError("the bridge runs a program's operations on one MPS stream, the default one: another cannot be made")
            self.stream_id = 0

        def synchronize(self):
            torch.mps.synchronize()

        def __eq__(self, other):
            return isinstance(other, Stream) and other.stream_id == self.stream_id

        def __hash__(self):
            return hash(self.stream_id)

        def __repr__(self):
            return f"<torch.mps.Stream stream_id={self.stream_id}>"

    core._mps_getCurrentStream = lambda: Stream(0)
    core._mps_setStream = lambda stream: None
    streams.Stream = Stream
    if hasattr(torch.mps, "Stream"):
        torch.mps.Stream = Stream


def _saving():
    """torch.save, pickle and tensor.storage() read a tensor's storage. Ours stands for the memory a tensor is
    in on the Mac: its bytes, all of them, are fetched when something asks for the storage on the CPU, which
    is what saving does. The file is the one a Mac writes: storages marked "mps", loadable anywhere with
    map_location."""
    storage = torch.UntypedStorage
    cpu_here, write_here = storage.cpu, storage._write_file
    typed_here = torch.Tensor._typed_storage

    def _typed_storage(self):
        """The storage a tensor is saved with. A view's is its base's, as on a Mac, where they are one storage:
        the file then holds that storage once, and the tensors loaded from it share it as those saved did."""
        if self.is_mps:
            root = bridge_native.root(self)
            if root is not None:
                return torch.TypedStorage(wrap_storage=root, dtype=self.dtype, _internal=True)
        return typed_here(self)

    torch.Tensor._typed_storage = _typed_storage

    def cpu(self):
        if self.device.type == "mps":
            fetched = bridge.storage_bytes(self)
            if fetched is not None:
                return fetched
        return cpu_here(self)

    def _write_file(self, *arguments):  # the older file format, which pickle.dumps(tensor) still uses
        return write_here(cpu(self), *arguments)

    storage.cpu, storage._write_file = cpu, _write_file


def _loading():
    """torch.load onto "mps" (map_location, or a file saved on a Mac) builds MPS storages, which Linux
    cannot. The storage stays on the CPU, marked; each tensor made of a marked storage is made on the
    Mac instead: the storage goes there once, whole, and the tensors are views of it, as on a Mac."""
    storage = torch.UntypedStorage
    mps_here, to_here, rebuild_here = storage.mps, storage.to, torch._utils._rebuild_tensor

    def mps(self):
        if self.device.type == "cpu":
            self._bridge_mps = {}  # the storage on the Mac, by the type its bytes are read as
            return self
        return mps_here(self)

    def to(self, *, device, non_blocking=False):
        if torch.device(device).type == "mps" and self.device.type == "cpu":
            return mps(self)
        return to_here(self, device=device, non_blocking=non_blocking)

    def _rebuild_tensor(typed, offset, size, stride):
        untyped = typed._untyped_storage
        whole = getattr(untyped, "_bridge_mps", None)
        if whole is None:
            return rebuild_here(typed, offset, size, stride)
        base = whole.get(typed.dtype)
        if base is None:
            base = whole[typed.dtype] = torch.empty((0,), dtype=typed.dtype).set_(untyped).to("mps")
        # Not through autograd: a tensor read from a file is nobody's view. Its data is the storage's all the
        # same, with the other tensors of that storage: saved again, they are saved as one.
        tensor = bridge._general("aten::as_strided", "", [base, list(size), list(stride), offset], {}, False)
        bridge_native.view_of(tensor, base)
        return tensor

    storage.mps, storage.to, torch._utils._rebuild_tensor = mps, to, _rebuild_tensor


def _events():
    """An event on the GPU marks a point in its work that a program can wait for. Here the only work a
    program can outrun is a copy it asked for without waiting (client.py, _later)."""
    event_here = torch.Event

    class Event:
        def __new__(cls, device=None, **options):
            if device is None or torch.device(device).type != "mps":
                return event_here(device, **options)
            return super().__new__(cls)

        def __init__(self, device=None, **options):
            self.mark = self.epoch = 0

        def record(self, stream=None):
            self.mark, self.epoch = bridge.asked, bridge.epoch

        def landed(self):
            """Waits for the copies asked for up to the mark. They were asked of one connection: if that one has been
            closed since, they never arrive, and the tensors they were to fill are not what the program computed."""
            if self.epoch != bridge.epoch:
                if self.mark:
                    raise RuntimeError("the bridge lost the Mac before a copy asked of it arrived: what the program had on "
                                       "the GPU is gone")
                return
            bridge._land(self.mark)

        def synchronize(self):
            self.landed()

        def wait(self, stream=None):
            self.landed()

        block = wait

        def query(self):
            return self.epoch == bridge.epoch and bridge.landed >= self.mark

    torch.Event = Event
