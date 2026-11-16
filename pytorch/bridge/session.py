"""Mac end of the bridge, one program's worth: runs the operations a container's process sends on the GPU (MPS).

server.py starts one of these for every connection, and hands it the connection. So a program
in a container has, on the Mac, what a program run on the Mac has: a process of its own, with
its own GPU memory (all of it given back when the process ends, however the program ended), its
own random generators, its own MPS settings.

It keeps the tensors, by number, and answers only when asked to: most
operations get no reply, so the container does not wait for them.

A container is not trusted. Only the operators of a list are run, on arguments that are plain
data of the kind each takes, and no tensor it can name reaches outside its own memory:
policy.py, the one check that an operation asked for whole, a signature run by Python and a
signature run by executor.cpp all go through.
"""

import collections
import importlib.machinery
import importlib.util
import json
import os
import select
import socket
import struct
import sys
import sysconfig

import torch

from .policy import DEVICES, DTYPES, INT64, MOST, Policy, Refused
from .wire import HEAD, PROTOCOL, raw, send


def load_executor():
    """The hot path in C++ (executor.cpp), as build-mac.sh built it for this PyTorch; without it, Python does the same."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "_build", torch.__version__.split("+")[0], "_executor" + sysconfig.get_config_var("EXT_SUFFIX"))
    if not os.path.exists(path) or os.environ.get("BRIDGE_EXECUTOR") == "0":  # 0: Python's path, for the tests of it
        return None
    spec = importlib.util.spec_from_loader("_executor", importlib.machinery.ExtensionFileLoader("_executor", path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_executor = load_executor()

POLICY = Policy()
# The PyTorch this repository builds both ends for and tests them with (versions.sh, bridge/ops/):
# what a container with another version is told.
BUILT_FOR = "2.15.0"
DEVICE = torch.device("mps")
# Hand the GPU what has been encoded whenever the container has sent nothing more (executor.cpp).
COMMIT = os.environ.get("BRIDGE_COMMIT", "1") != "0"


def seed(value):
    torch.manual_seed(value)


# The container's random generators "on mps", by its numbers for them. 0 is this process's default one.
generators = {}


def generator(number):
    if number == 0:
        return torch.mps._get_default_mps_generator()
    return generators[number]


def new_generator(number):
    generators[number] = torch.Generator(device="mps")


def clone_generator(number, twin):
    generators[twin] = generator(number).clone_state()


def nothing(function):
    def call(*arguments):
        function(*arguments)

    return call


# The kinds of argument a named call takes: each says whether a value a container sent is one.
def whole(value):  # seeds go up to 64 bits without a sign
    return type(value) is int and -INT64 <= value < 2 * INT64


def real(value):
    return type(value) in (int, float)


def flag(value):
    return type(value) is bool


def word(value):  # the profiler's mode: "interval", "event,signpost"
    return isinstance(value, str) and len(value) <= 64 and all(c.isalpha() or c in ",_" for c in value)


def state(value):  # a generator's state, as bytes
    return isinstance(value, list) and len(value) <= MOST and all(type(byte) is int and 0 <= byte < 256 for byte in value)


# What a container may ask for by name: what acts on the GPU itself and is not an operator, each
# with the arguments it takes. native.cpp's MPS hooks come here, so these are what torch.mps.*
# does in the container.
CALLS = {
    "manual_seed": (seed, whole),
    "generator.new": (new_generator, whole),
    "generator.clone": (clone_generator, whole, whole),
    "generator.manual_seed": (lambda number, value: nothing(generator(number).manual_seed)(value), whole, whole),
    "generator.initial_seed": (lambda number: generator(number).initial_seed(), whole),
    "generator.get_state": (lambda number: generator(number).get_state().tolist(), whole),
    "generator.set_state": (lambda number, state: nothing(generator(number).set_state)(torch.tensor(state, dtype=torch.uint8)),
                            whole, state),
    "get_name": (getattr(torch._C, "_mps_get_name", lambda: "Apple GPU"),),
    "get_core_count": (getattr(torch._C, "_mps_get_core_count", lambda: 0),),
    # A setting of the process that computes: with it PyTorch refuses the operators that do not give the same
    # bits twice (the gradient of an embedding, on the GPU), where they run, which is here.
    "use_deterministic_algorithms": (lambda mode, warn_only: torch.use_deterministic_algorithms(mode, warn_only=warn_only), flag, flag),
    "ping": (lambda: None,),  # answered when everything sent before it has been taken: the container's pace (native.cpp)
    "synchronize": (torch.mps.synchronize,),
    "empty_cache": (torch.mps.empty_cache,),
    "current_allocated_memory": (torch.mps.current_allocated_memory,),
    "driver_allocated_memory": (torch.mps.driver_allocated_memory,),
    "recommended_max_memory": (torch.mps.recommended_max_memory,),
    "set_per_process_memory_fraction": (torch.mps.set_per_process_memory_fraction, real),
    "is_macos_or_newer": (torch.backends.mps.is_macos_or_newer, whole, whole),
    "profiler_start": (torch._C._mps_profilerStartTrace, word, flag),
    "profiler_stop": (torch._C._mps_profilerStopTrace,),
    "event.acquire": (torch._C._mps_acquireEvent, flag),
    "event.release": (torch._C._mps_releaseEvent, whole),
    "event.record": (torch._C._mps_recordEvent, whole),
    "event.wait": (torch._C._mps_waitForEvent, whole),
    "event.synchronize": (torch._C._mps_synchronizeEvent, whole),
    "event.query": (torch._C._mps_queryEvent, whole),
    "event.elapsed": (torch._C._mps_elapsedTimeOfEvents, whole, whole),
}


def call(name, arguments):
    """One of CALLS, if it is one and these are the arguments it takes."""
    if not isinstance(name, str) or name not in CALLS:
        raise Refused(f"not something the Mac end does by name: {name!r:.60}")
    function, *kinds = CALLS[name]
    if not isinstance(arguments, list) or len(arguments) != len(kinds) or not all(k(a) for k, a in zip(kinds, arguments)):
        raise Refused(f"{name} takes {len(kinds)} argument{'s' if len(kinds) != 1 else ''}"
                      + (": " + ", ".join(k.__name__ for k in kinds) if kinds else ""))
    return function(*arguments)


def named(kind, name):
    """A type, layout or memory format by its name in torch: of those policy.py lets through, and no other attribute."""
    if not isinstance(name, str) or not name.isidentifier() or (kind is torch.dtype and name not in DTYPES):
        raise Refused(f"not a {kind.__name__}: {name!r:.60}")
    value = getattr(torch, name, None)
    if not isinstance(value, kind):
        raise Refused(f"not a {kind.__name__}: {name}")
    return value


def decode(x, tensors):
    if isinstance(x, list):
        return [decode(v, tensors) for v in x]
    if not isinstance(x, dict):
        return x
    if "t" in x:
        return tensors[x["t"]]
    if "s" in x:  # a number the container had as a 0-dim tensor
        dtype = named(torch.dtype, x["dt"])
        if dtype == torch.float64:  # MPS has no float64
            dtype = torch.float32
        return torch.tensor(x["s"], dtype=dtype, device=DEVICE)
    if "dtype" in x:
        return named(torch.dtype, x["dtype"])
    if "device" in x:
        if x["device"] not in DEVICES:
            raise Refused(f"not a device of the Mac end: {x['device']!r:.40}")
        return torch.device(x["device"])
    if "layout" in x:
        return named(torch.layout, x["layout"])
    if "mf" in x:
        return named(torch.memory_format, x["mf"])
    if "g" in x:  # 0, the default generator, is what an operator uses when given none
        return generators[x["g"]] if x["g"] else None
    raise ValueError(f"unknown argument: {x}")


def unusable(what):
    return RuntimeError(f"{what} is not memory the Mac end lets a container use (it reaches outside what it was given, or is "
                        "not a plain tensor of the GPU or the CPU): an operation on it failed half way, or made it so")


def describe(result, base, room, tensors):
    """What a waiting container needs to know of a result, storing its tensors from `base` on: (the answer, its blobs).

    A result that is a tensor on the CPU here (a packed sequence's batch sizes) is one there
    too: it goes back whole. One that is a list of tensors (the gradients of an LSTM's weights)
    goes back as a list. The container set `room` numbers aside for the tensors, from `base`:
    no more are taken, or a tensor would be stored under a number that is another tensor's.
    """
    if type(base) is not int or type(room) is not int or not 0 <= room <= MOST:
        raise Refused("an operation that is waited for says where its results go, and how many there is room for")
    blobs, used = [], 0

    def one(item, nested):
        nonlocal used
        if isinstance(item, torch.Tensor):
            if not POLICY.usable(item):
                raise unusable("a result")
            if item.device.type == "cpu":
                blobs.append(raw(item))
                return {"c": [list(item.shape), str(item.dtype)[6:], len(blobs) - 1]}
            if used >= room:
                raise RuntimeError(f"the operation gave more than the {room} tensors the container set numbers aside for")
            tensors[base + used] = item
            used += 1
            return {"t": [list(item.shape), str(item.dtype)[6:], list(item.stride()), item.storage_offset()], "n": used - 1}
        if isinstance(item, (tuple, list)) and not nested:
            return {"l": [one(element, True) for element in item]}
        if item is None or type(item) in (bool, int, float):
            return {"v": item}
        raise RuntimeError(f"a result that is a {type(item).__name__}, which the bridge does not carry")

    items = list(result) if isinstance(result, (tuple, list)) else [result]
    return {"res": [one(item, False) for item in items], "seq": isinstance(result, (tuple, list))}, blobs


class Signature:
    """An operation with its constants in place: only the tensors change from one call to the next."""

    def __init__(self, spec):
        self.name = spec["op"]
        checked = POLICY.check(spec, counts=True)
        op = checked._op  # the built-in under the Python wrapper
        returns = checked._schema.returns
        as_list = len(returns) == 1 and returns[0].type.kind() == "ListType"  # one list of tensors, of any length
        inputs, outputs, reals = spec["nin"], spec["nout"], spec.get("nf", 0)
        self.count = inputs + outputs
        self.size = 4 * self.count + 8 * reals
        self.unpack_from = struct.Struct(f"<{self.count}I{reals}d").unpack_from  # the reals follow the numbers in `h`

        def real(a):
            if a.get("dt"):  # a 0-dim tensor the container had on its CPU
                dtype = named(torch.dtype, a["dt"])
                constants["dt%d" % a["f"]] = torch.float32 if dtype == torch.float64 else dtype
                return f"tensor(h[{self.count + int(a['f'])}], dtype=dt{int(a['f'])}, device=DEVICE)"
            return f"h[{self.count + int(a['f'])}]"  # a plain number: Python's call wraps it where a tensor is taken

        def on_gpu(*arguments, **named_arguments):  # as executor.cpp does: never the CPU's kernel on a container's tensors
            POLICY.placed(arguments, named_arguments)
            return op(*arguments, **named_arguments)

        # The call is written out as source, once: `T[h[2]] = op(T[h[0]], T[h[1]], c2)`.
        # Only positions and names made here go into it, never text from the container.
        constants = {"op": on_gpu, "tensor": torch.tensor, "DEVICE": DEVICE}
        call = []
        for i, a in enumerate(spec["a"]):
            if isinstance(a, dict) and "t" in a:
                call.append(f"T[h[{int(a['t'])}]]")
            elif isinstance(a, dict) and "f" in a:
                call.append(real(a))
            else:
                constants[f"c{i}"] = decode(a, None)
                call.append(f"c{i}")
        for i, (name, a) in enumerate(spec["kw"].items()):
            if not name.isidentifier():
                raise ValueError(f"not an argument name: {name}")
            if isinstance(a, dict) and "f" in a:
                call.append(f"{name}={real(a)}")
                continue
            constants[f"k{i}"] = decode(a, None)
            call.append(f"{name}=k{i}")
        call = f"op({', '.join(call)})"
        if outputs == 1 and not as_list:
            body_ = f"T[h[{inputs}]] = {call}"
        else:  # as many results as the container made tensors for, as executor.cpp asks
            body_ = (f"r = {call}\n    if len(r) != {outputs}: raise RuntimeError('the operator gave %d results, and the "
                     f"container made tensors for {outputs}' % len(r))\n"
                     + "\n".join(f"    T[h[{inputs + i}]] = r[{i}]" for i in range(outputs)))
        exec(f"def run(h, T):\n    {body_}", constants)  # noqa: S102 - see above
        self.run = constants["run"]


class Broken:
    """A signature that was refused: its operations are still frames of a known length, and each fails with the reason."""

    def __init__(self, name, why, inputs, outputs, reals):
        self.name = name
        self.size = 4 * (inputs + outputs) + 8 * reals
        self.unpack_from = lambda buffer, at: None

        def run(numbers, tensors):
            raise RuntimeError(why)

        self.run = run


class Table:
    """The executor's tensors, for the Python that handles the rare frames. The executor checks each (usable)."""

    def __init__(self, executor):
        self.executor = executor

    def __getitem__(self, handle):
        return self.executor.get(handle)

    def __setitem__(self, handle, tensor):
        self.executor.put(handle, tensor)


class Tensors(dict):
    """The tensors when there is no executor: none that reaches outside its memory goes in, or comes out."""

    def __getitem__(self, handle):
        tensor = dict.__getitem__(self, handle)
        if not POLICY.usable(tensor):
            del self[handle]
            raise unusable(f"tensor {handle}")
        return tensor

    def __setitem__(self, handle, tensor):
        if type(handle) is not int or not 0 <= handle < 1 << 32:
            raise Refused(f"not a tensor's number: {handle!r:.40}")
        if not isinstance(tensor, torch.Tensor) or not POLICY.usable(tensor):
            raise unusable("a result")
        dict.__setitem__(self, handle, tensor)


def define(executor, signature, spec):
    POLICY.check(spec, counts=True)
    arguments, which, lists = [], [], []
    reals = []  # real numbers sent with each call: (argument's position or name, which of the call's, type if a tensor's)
    for i, a in enumerate(spec["a"]):
        if isinstance(a, dict) and "f" in a:
            reals.append((str(i), int(a["f"]), a.get("dt") or ""))
            arguments.append(None)
            which.append(-1)
            continue
        if isinstance(a, dict) and "tl" in a:  # a list of tensors: (argument, its first number, how many)
            lists.append((i, int(a["tl"][0]), int(a["tl"][1])))
            arguments.append(None)
            which.append(-1)
            continue
        tensor = isinstance(a, dict) and "t" in a
        # (A plain number where the operator takes a tensor stays a number: PyTorch wraps it in a
        # tensor marked as a number, as it does in a program on the Mac, and the result's type is
        # decided accordingly.)
        arguments.append(None if tensor else decode(a, None))
        which.append(int(a["t"]) if tensor else -1)
    named = {k: int(v["t"]) for k, v in spec["kw"].items() if isinstance(v, dict) and "t" in v}  # tensors given by name (out=)
    reals += [(k, int(v["f"]), v.get("dt") or "") for k, v in spec["kw"].items() if isinstance(v, dict) and "f" in v]
    keywords = {k: decode(v, None) for k, v in spec["kw"].items() if k not in named and not (isinstance(v, dict) and "f" in v)}
    overload = "" if spec["ov"] == "default" else spec["ov"]  # Python's name for the unnamed one
    executor.define(signature, spec["op"], overload, arguments, which, lists, keywords, named, reals, spec["nin"], spec["nout"],
                    spec.get("nf", 0))


def u32(buffer, at):
    return int.from_bytes(buffer[at:at + 4], "little")


class Reader:
    """The connection as frames. Operations come many to a packet: they are
    taken straight from the bytes received, without a read call each."""

    def __init__(self, conn):
        self.conn = conn
        self.buffer = b""
        self.at = 0

    def ready(self):
        """Whether more bytes have already arrived."""
        return bool(select.select([self.conn], [], [], 0)[0])

    def gone(self):
        """Whether the program has closed its end (it ended, or was killed), even with bytes of its still unread."""
        watch = select.poll()
        watch.register(self.conn, select.POLLHUP)
        return any(event & (select.POLLHUP | select.POLLERR | select.POLLNVAL) for _, event in watch.poll(0))

    def more(self):
        chunk = self.conn.recv(1 << 20)
        if not chunk:
            return False
        self.buffer = self.buffer[self.at:] + chunk if self.at < len(self.buffer) else chunk
        self.at = 0
        return True

    def exactly(self, length):
        """`length` bytes from here on: for the rare frames, and for tensors' data."""
        have = len(self.buffer) - self.at
        if have >= length:
            data = self.buffer[self.at:self.at + length]
            self.at += length
            return data
        data = bytearray(length)
        data[:have] = self.buffer[self.at:]
        self.buffer, self.at = b"", 0
        view = memoryview(data)
        while have < length:
            got = self.conn.recv_into(view[have:])
            if not got:
                raise OSError("the connection closed inside a frame")
            have += got
        return data


class Writer:
    """wire.py writes a frame as several pieces at once."""

    def __init__(self, conn):
        self.file = conn.makefile("wb", buffering=1 << 20)
        self.flush = self.file.flush

    def write(self, *pieces):
        for piece in pieces:
            self.file.write(piece)


def serve(conn):
    reader = Reader(conn)
    w = Writer(conn)
    executor = _executor.Executor(POLICY.names()) if _executor else None
    tensors = Table(executor) if executor else Tensors()
    signatures = {}
    failed = None  # the first operation that failed while nobody was waiting
    count = 0
    # Values the container asked for without waiting ("get" with "later"): they are copied off the
    # GPU in the background, each with an event that tells when its copy is done, and answered in
    # order. Meanwhile the operations that follow are encoded: the GPU and this loop work at once.
    later = collections.deque()

    def answer(wait):
        """Sends the answers that are ready; all of them, waiting for the GPU, if `wait`.

        Like every answer, each carries the first failure since the one before it: a program that
        waits for a copy it asked for must hear that an operation before it failed, since the
        values it gets are then not those it computed.
        """
        nonlocal failed
        sent = False
        while later and (wait or later[0][0].query()):
            event, into, number = later.popleft()
            event.synchronize()
            send(w, {"shape": list(into.shape), "dtype": str(into.dtype)[6:], "err": failed, "i": number}, [raw(into)])
            failed = None
            sent = True
        if sent:
            w.flush()

    while True:
        buffer, at = reader.buffer, reader.at
        end = len(buffer)
        # The hot path: every whole operation or free already received.
        if executor:
            at = executor.run(buffer, at, COMMIT, 0.25)
            # Taken every time, not only when there is no earlier failure to report: what failed after the
            # first failure failed because of it, and left there it would be reported later as a failure of its own.
            then = executor.take_failed()
            failed = failed or then or None
        while not executor and end - at >= 5:
            kind = buffer[at]
            if kind == 79:  # O
                signature = signatures[u32(buffer, at + 1)]
                after = at + 5 + signature.size
                if after > end:
                    break
                try:
                    signature.run(signature.unpack_from(buffer, at + 5), tensors)
                except Exception as e:  # noqa: BLE001 - reported at the next question
                    if failed is None:
                        failed = f"{signature.name}: {type(e).__name__}: {e}"
                count += 1
                at = after
            elif kind == 70:  # F
                n = u32(buffer, at + 1)
                after = at + 5 + 4 * n
                if after > end:
                    break
                for handle in struct.unpack_from(f"<{n}I", buffer, at + 5):
                    tensors.pop(handle, None)
                at = after
            else:
                break
        reader.at = at
        if executor and executor.paused:  # more to run: but not for a program that is no longer there
            if reader.gone():
                return executor.count
            if later:
                answer(False)
            continue
        if later:
            answer(False)
        if end - at < 5 or buffer[at] in (79, 70):  # nothing whole left: wait for more
            if later and not reader.ready():
                answer(True)  # nothing to encode meanwhile, and the container may be waiting for these
            if not reader.more():
                return executor.count if executor else count
            continue

        kind, n = HEAD.unpack(reader.exactly(5))
        if kind == b"D":
            spec = json.loads(reader.exactly(u32(reader.exactly(4), 0)))
            try:
                if executor:
                    define(executor, n, spec)
                else:
                    signatures[n] = Signature(spec)
            except Exception as e:  # noqa: BLE001 - reported at the next question, like an operation that failed
                name = str(spec.get("op") if isinstance(spec, dict) else None)[:80]
                why = f"{type(e).__name__}: {e}"
                failed = failed or f"{name}: {why}"
                # Its operations will come all the same, and their length is what the signature said: they
                # are skipped, and fail with this. Without a length that can be believed, what follows
                # cannot be told apart from them: the connection ends.
                lengths = [spec.get(key, 0) for key in ("nin", "nout", "nf")] if isinstance(spec, dict) else [None]
                if not all(type(length) is int and 0 <= length <= MOST for length in lengths):
                    raise Refused(f"a signature of no known length ({why})") from None
                if executor:
                    executor.define_broken(n, name, why, *lengths)
                else:
                    signatures[n] = Broken(name, why, *lengths)
            continue
        if kind != b"J":
            raise Refused(f"not a frame of the bridge: {kind!r}")
        msg = json.loads(reader.exactly(n))
        if not isinstance(msg, dict):
            raise Refused("a message that is not a dictionary")
        sizes = msg.pop("b", [])
        if not isinstance(sizes, list) or len(sizes) > MOST or not all(type(k) is int and 0 <= k < 1 << 40 for k in sizes):
            raise Refused("a message whose data has no known length")
        blobs = [reader.exactly(k) for k in sizes]
        kind = msg.get("k")
        reply, results = None, ()
        try:
            if kind == "op":
                count += 1
                fn = POLICY.check(msg)
                args = decode(msg["a"], tensors)
                kwargs = {k: decode(v, tensors) for k, v in msg["kw"].items()}
                POLICY.placed(args, kwargs)
                result = fn(*args, **kwargs)
                if msg.get("sync"):
                    reply, results = describe(result, msg.get("base"), msg.get("room"), tensors)
                else:
                    outs = list(result) if isinstance(result, (tuple, list)) else [result]
                    # As many results as the container made tensors for: with fewer, or more, the tensors it
                    # goes on to use would not be the ones the operator gave.
                    # (None of them, for an operator that gives nothing back: its effect is on its arguments.)
                    expected = msg.get("out")
                    if not isinstance(expected, list) or (expected and len(expected) != len(outs)):
                        raise RuntimeError(f"the operator gave {len(outs)} results, and the container made tensors for "
                                           f"{len(expected) if isinstance(expected, list) else 'none'}")
                    for handle, out in zip(expected, outs):
                        if out is not None:  # a result the operator did not produce this time
                            tensors[handle] = out
            elif kind == "put":
                dtype = named(torch.dtype, msg.get("dtype"))
                shape = msg.get("shape")
                elements = 1
                for size in shape if isinstance(shape, list) and len(shape) <= 64 else [None]:
                    if type(size) is not int or not 0 <= size < INT64:
                        raise Refused("data of no known shape")
                    elements *= size  # as long a number as it takes: no product here overflows
                # The bytes must be the shape's: a tensor is never made of memory nobody sent.
                if len(blobs) != 1 or elements * dtype.itemsize != len(blobs[0]):
                    raise Refused(f"data of {elements} elements of {dtype.itemsize} bytes, sent as "
                                  f"{len(blobs[0]) if len(blobs) == 1 else 'no'} bytes")
                if blobs[0]:
                    t = torch.frombuffer(bytearray(blobs[0]) if isinstance(blobs[0], bytes) else blobs[0], dtype=dtype).reshape(shape)
                else:
                    t = torch.empty(shape, dtype=dtype)
                tensors[msg["t"]] = t if msg.get("cpu") else t.to(DEVICE)  # "cpu": lent for one operation, as the CPU tensor it is
            elif kind == "get" and msg.get("later"):
                source = tensors[msg["t"]].detach()
                into = torch.empty(source.shape, dtype=source.dtype)
                into.copy_(source, non_blocking=True)
                event = torch.Event(device=DEVICE, blocking=True)
                event.record()
                later.append((event, into, msg.get("i")))
                continue
            elif kind == "get":
                t = tensors[msg["t"]].detach()
                if msg.get("storage"):
                    # All of the memory the tensor is in, as bytes: what a Mac saves of it, and of the tensors that
                    # share that memory. (Not the tensor's values: a tensor made smaller covers less of its memory
                    # than the views made of it before, which still read the rest.)
                    whole = t.untyped_storage()
                    t = torch.empty(0, dtype=torch.uint8, device=t.device).set_(whole, 0, (whole.nbytes(),), (1,))
                t = t.cpu()
                reply, results = {"shape": list(t.shape), "dtype": str(t.dtype)[6:]}, [raw(t)]
            elif kind == "sync":
                torch.mps.synchronize()
                reply = {}
            elif kind == "call":
                value = call(msg.get("fn"), msg.get("a"))
                reply = None if msg.get("quiet") else {"v": value}  # quiet: told, not asked (a seed)
            else:
                raise Refused(f"not a message of the bridge: {kind!r:.40}")
        except Exception as e:  # noqa: BLE001 - reported to the container, whatever it is
            error = f"{str(msg.get('op', kind))[:80]}: {type(e).__name__}: {e}"
            if (kind in ("get", "sync", "call") or msg.get("sync")) and not msg.get("quiet"):
                reply, results = {"err": failed or error}, ()
                failed = None
            elif failed is None:
                failed = error
        finally:
            # Nothing of a message outlives it. Left alone, these names would hold on to its tensors
            # (an operation's arguments and results, the data just sent or fetched) until the next
            # message of the kind replaced them: the container would have let go of a tensor, told
            # this process to forget it, and the GPU would have it still.
            args = kwargs = result = outs = out = t = source = into = blobs = None
        if reply is not None:
            answer(True)  # answers go out in the order of the questions
            reply.setdefault("err", failed)
            failed = None
            reply["i"] = msg.get("i")  # which question this answers: the container checks that it is the one it waits for
            send(w, reply, results)
            w.flush()
            reply = results = None


def main():
    """python -m bridge.session: serves the one connection server.py hands over, then ends.

    The connection comes over this process's standard input, a socket to server.py, together
    with the container's greeting. A process is started before there is a program for it, so
    that PyTorch is loaded by the time one connects; if none ever does, the socket closes and
    this process ends.
    """
    control = socket.socket(fileno=0)
    greeting, descriptors, _, _ = socket.recv_fds(control, 1 << 16, 1)
    if not descriptors:
        return
    control.send(b"k")  # server.py keeps its copy of the connection open until it hears this
    control.close()
    hello = json.loads(greeting)  # a dictionary: server.py took nothing else for a greeting
    conn = socket.socket(fileno=descriptors[0])
    if conn.family != socket.AF_UNIX:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    # (For the log: what a container says of itself, cut short and without control characters.)
    program = "".join(c if c.isprintable() else "?" for c in str(hello.get("program") or "a program"))[:120]
    who = f"{program} (pid {str(hello.get('pid'))[:20]} there, {os.getpid()} here)"
    ours = torch.__version__.split("+")[0]
    w = Writer(conn)
    theirs = str(hello.get("torch"))[:40]
    if theirs != ours:
        send(w, {"protocol": PROTOCOL, "err": f"the container has PyTorch {theirs}, the Mac's bridge server {ours}: they must be "
                                              "the same. The image and the Mac end have to be built for the same version, and "
                                              f"PyTorch {BUILT_FOR} is the one this repository is built and tested for"})
        w.flush()
        print(f"bridge server: refused {who}: PyTorch {theirs} there, {ours} here", flush=True)
        return
    if hello.get("protocol") != PROTOCOL:
        send(w, {"protocol": PROTOCOL, "err": f"the container's bridge speaks version {hello.get('protocol') or 1} of its protocol, "
                                              f"the Mac's bridge server version {PROTOCOL}: build the image and start the server "
                                              "from the same state of the repository"})
        w.flush()
        print(f"bridge server: refused {who}: protocol {hello.get('protocol') or 1} there, {PROTOCOL} here", flush=True)
        return
    send(w, {"torch": ours, "protocol": PROTOCOL, "pid": os.getpid(), "err": None})
    w.flush()
    print(f"bridge server: {who} connected", flush=True)
    try:
        count = serve(conn)
        print(f"bridge server: {who} left after {count} operations", flush=True)
    except Exception as e:  # noqa: BLE001 - a program that died mid-message, or sent what is not the bridge's: this process is its own
        print(f"bridge server: {who} dropped: {type(e).__name__}: {str(e)[:300]}", flush=True)


if __name__ == "__main__":
    main()
