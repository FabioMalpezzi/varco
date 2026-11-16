"""Container end of the bridge, Python's part: what is not routine.

native.cpp is an MPS backend for PyTorch on Linux: every operation on an
"mps" tensor ends in its one kernel, which sends an operation it has seen
before to the Mac as a few integers, without Python. This is where the
others come (Bridge._slow):

- an operation seen for the first time with these shapes and constants: its
  result's shape is worked out here, by running it on empty (meta) tensors,
  or by asking the Mac when PyTorch cannot; then it is remembered;
- data to send to the Mac or to bring back (.to("mps"), .cpu());
- operations whose answer is in the values (.item(), a comparison): the
  program waits for those.

A tensor "on mps" has a shape and a type and no data: the Mac keeps the real
one under a number, which native.cpp keeps with the tensor's storage.

The connection to the Mac is opened when something first needs the GPU, not
before: a process that imports PyTorch and never touches "mps" (a data
loader's worker, a script that only reads a file) costs the Mac nothing. On
the Mac every connection is a process of its own (server.py), so a program
here has the GPU memory, the random generator and the lifetime that a
program has on a Mac. A process made by fork() gets a connection of its own
when it first needs one, and leaves its parent's alone.
"""

import bridge_native  # before torch: it tells PyTorch there is an MPS device (native.cpp)  # isort: skip

import atexit
import collections
import json
import os
import signal
import socket
import struct
import sys
import threading
import time

import torch
from torch import Tensor

from .wire import ENVIRONMENT, PROTOCOL, define, raw, recv, send

META = torch.device("meta")
_empty_strided = torch.empty_strided
_fake = bridge_native.fake
_handle = bridge_native.handle

# Operations whose answer is in the values, not the shapes: only the Mac can
# give it, every time, so the program waits for these.
NEEDS_VALUES = {
    "aten::_local_scalar_dense", "aten::item", "aten::is_nonzero", "aten::equal",
    "aten::nonzero", "aten::masked_select", "aten::_unique2", "aten::unique_dim",
}

# Operations whose results' shapes PyTorch cannot work out here without data (they have no kernel
# for empty, "meta", tensors) and which are, all the same, decided by their arguments' shapes and
# constants alone: the Mac is asked the first time, and the answer is remembered. Each is here
# because its result's shape was read off its definition; an operation that is not here and
# cannot be worked out is asked of the Mac every time, which is slower and never wrong. One whose
# result's shape is in the values (unique_consecutive: as long as there are distinct values) must
# never be here: remembered, the second call would be given the first call's shape.
SHAPE_FROM_SHAPES = {
    "aten::max_pool2d_backward",  # the gradient with respect to the input: the input's shape
    "aten::_ctc_loss_backward",  # the gradient with respect to the log-probabilities: their shape
    "aten::_logcumsumexp",  # its argument's shape
    "aten::binomial", "aten::_sample_dirichlet", "aten::_standard_gamma_grad", "aten::_dirichlet_grad",  # their arguments' shape
    # A Mac's own convolutions and recurrent layers: sizes, strides, paddings and layers are constants.
    "aten::_mps_convolution", "aten::_mps_convolution_transpose", "aten::mps_convolution_backward",
    "aten::mps_convolution_transpose_backward", "aten::_lstm_mps", "aten::lstm_mps_backward",
}

# What a program sets before it has touched the GPU, and the Mac must know when it does: the last
# value of each is kept and sent when the connection opens. A seed set by a program that never
# uses the GPU then costs nothing, and fails nothing where there is no Mac to reach.
BEFORE_CONNECTING = ("manual_seed", "generator.manual_seed", "use_deterministic_algorithms")

# What is told to the Mac without waiting for an answer, there being none to give. Not only to
# save the wait: PyTorch seeds a generator holding the generator's own lock and Python's, and a
# wait would let go of Python's; another thread seeding the same generator (the default one, in
# every thread that calls torch.manual_seed) would then take Python's lock and wait for the
# generator's, for ever.
ONE_WAY = {"manual_seed", "generator.manual_seed", "generator.new", "generator.clone", "generator.set_state",
           "use_deterministic_algorithms"}

# MPS has no 64-bit floats. A Mac says so where the tensor is made, with these words.
NO_DOUBLE = "Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead."
DOUBLES = (torch.float64, torch.complex128)


# An operation that failed on the Mac fails here with the kind of error it was there, when it is one of these.
FAILURES = {kind.__name__: kind for kind in (NotImplementedError, TypeError, ValueError, IndexError, OverflowError, torch.OutOfMemoryError)}


def failure(text):
    """The error to raise for one the Mac reported as "operator: Kind: message"."""
    kind = FAILURES.get(text.split(": ")[1] if text.count(": ") >= 2 else "", RuntimeError)
    return kind(f"on the Mac: {text}")


# The operators that take tensors on the CPU among their indices when the tensor indexed is on the
# GPU (x[rows] with `rows` on the CPU): PyTorch moves such indices to the GPU itself. Every other
# tensor that is on the CPU goes to the Mac as a CPU tensor, see Bridge._mixed.
INDEXING = {"aten::index", "aten::index_put", "aten::index_put_", "aten::_index_put_impl_", "aten::_unsafe_index", "aten::_unsafe_index_put"}


class Uninterrupted:
    """While a question to the Mac is out, signals wait.

    Ctrl-C raises KeyboardInterrupt wherever the program happens to be, and a program can catch
    it and go on (Lightning does, to save and close). If that were in the middle of a question,
    the answer would still arrive, and be taken for the answer to the next one. So the handlers
    Python would run (in the main thread, the only one that runs them) are held until the answer
    is in, and run then: the interrupt arrives, a moment later, with the connection in order.
    """

    SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)

    def __enter__(self):
        self.previous = {}
        if threading.current_thread() is threading.main_thread():
            self.held = []
            for number in self.SIGNALS:
                handler = signal.getsignal(number)
                if callable(handler):
                    self.previous[number] = handler
                    signal.signal(number, self.hold)

    def hold(self, number, frame):
        self.held.append((number, frame))

    def __exit__(self, *failure):
        for number, handler in self.previous.items():
            signal.signal(number, handler)
        if self.previous:
            for number, frame in self.held:
                self.previous[number](number, frame)


def is_mps(x):
    return isinstance(x, torch.device) and x.type == "mps"


# The readers of connections that were the parent's, in a process made by fork(): see Bridge._forked.
ORPHANS = []


class Writer:
    """wire.py writes frames to a file: this one is native.cpp's buffer, so that they keep their place among the operations."""

    write = staticmethod(bridge_native.write)
    flush = staticmethod(bridge_native.flush)


class Bridge:
    def __init__(self):
        self.sock = self.r = None
        self.w = Writer()
        # One question to the Mac at a time, with its answer: threads take turns. Operations are
        # written with Python's lock held throughout (native.cpp), and need no other.
        self.lock = threading.RLock()
        self.operators = {}
        self.positions = {}
        self.in_values = {}  # an operator -> whether its results' shapes are in its arguments' values (_in_values)
        self.every_time = collections.Counter()  # operators asked of the Mac at every call, for want of a shape worked out here
        self.signatures = {}  # an operation's signature, as the Mac was told it -> its number there
        self.before = {}  # BEFORE_CONNECTING: what to tell the Mac when the connection opens
        self.waits = 0  # round trips the program waited for
        self.waited = 0.0  # seconds spent waiting for the Mac's answers
        self.uploaded = 0  # bytes of tensors sent to the Mac
        self.pending = collections.deque()  # (tensor here, its question's number) for values asked for without waiting, oldest first
        self.asked = self.landed = 0  # how many such were asked for, how many have arrived
        self.questions = 0  # questions sent on this connection: each carries its number, and its answer carries it back
        self.epoch = 0  # which connection this is, counted from the first: a copy asked for on one never arrives on the next
        self.slow = collections.Counter() if os.environ.get("BRIDGE_DEBUG") else None  # what leaves the fast path
        self.slowly = 0.0  # seconds spent in Python for it
        # BRIDGE_SYNC=1: every operation waits for the Mac. Slow, and for one thing only: an
        # operation that fails on the Mac is otherwise reported at the next question the program
        # asks, which is later than where it happened. With this it fails where it is.
        self.every = os.environ.get("BRIDGE_SYNC", "0") not in ("", "0")
        bridge_native.start(self._slow, self.call, self.connect, not self.every)
        os.register_at_fork(after_in_child=self._forked)
        atexit.register(self.close)
        if self.slow is not None or os.environ.get("BRIDGE_REPORT"):  # what the bridge did, when the program ends
            atexit.register(lambda: self.sock is not None and self.report())

    # --- the connection -----------------------------------------------------

    def where(self):
        # Kiri sets KIRI_GPU_BRIDGE in a container started with --gpus, to an address that exists
        # only in that container. Without it, the Mac's loopback through the engine.
        given = os.environ.get("KIRI_GPU_BRIDGE", "")
        host = os.environ.get("BRIDGE_HOST") or given.rpartition(":")[0] or "host.docker.internal"
        return host, int(os.environ.get("BRIDGE_PORT") or given.rpartition(":")[2] or "47700")

    def nobody(self, host, port):
        """What a program is told when the connection opened and was closed before the server's greeting.

        That is what Kiri does for a container it gave no bridge: it accepts a connection to any port of
        host.docker.internal and closes it at once when nothing listens there on the Mac. A container it
        started with --gpus has KIRI_GPU_BRIDGE, and the same happens to it when no server listens on
        the socket Kiri's channel leads to.
        """
        said = f"no bridge server answered at {host}:{port}: the connection was closed before the server's greeting arrived. "
        if os.environ.get("KIRI_GPU_BRIDGE"):
            return said + ("This container was started with --gpus in Kiri (KIRI_GPU_BRIDGE is set), and Kiri's channel leads to a "
                           "socket on the Mac: is the bridge's server listening there (ENGINE=kiri pytorch/server.sh start)?")
        return said + ("With Kiri, the container must be started with --gpus all (a GPU request in Compose): without it the "
                       "container has no bridge, and Kiri accepts the connection and closes it at once. With another engine, "
                       "the server's log on the Mac says why it closed (pytorch/runs/server.log)")

    def connect(self):
        """Opens the connection, if it is not open. Raises if the Mac cannot be reached, or runs another PyTorch."""
        with self.lock:
            if self.sock is not None:
                return
            host, port = self.where()
            try:
                sock = socket.create_connection((host, port), timeout=float(os.environ.get("BRIDGE_TIMEOUT", "10")))
            except OSError as e:
                raise RuntimeError(f"the bridge cannot reach the Mac's GPU at {host}:{port} ({e}): "
                                   "is `python -m bridge.server` running on the Mac?") from None
            sock.settimeout(None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock, self.r = sock, sock.makefile("rb", buffering=1 << 20)
            # Operations are sent a few at a time (this many bytes), so that the Mac starts on them
            # while the program is still issuing the next ones. Measured: larger and the Mac starts
            # late, smaller and the sends themselves cost more than they save.
            # BRIDGE_AHEAD: how many operations the program may send without waiting before it waits for
            # the Mac to have taken them (native.cpp, pace); 0 for no limit.
            bridge_native.attach(sock.fileno(), int(os.environ.get("BRIDGE_BUFFER", "1024")), int(os.environ.get("BRIDGE_AHEAD", "4096")))
            self.questions = 0
            try:
                hello = {"k": "hello", "protocol": PROTOCOL, "torch": torch.__version__.split("+")[0], "pid": os.getpid(),
                         "program": sys.argv[0], "env": {name: os.environ[name] for name in ENVIRONMENT if name in os.environ}}
                try:
                    answer, _ = self._ask(hello)
                except (OSError, RuntimeError) as e:
                    # Closed before the server's greeting arrived: nothing there was a bridge server to lose.
                    if isinstance(e, OSError) or str(e).startswith("the bridge lost the Mac"):
                        raise RuntimeError(self.nobody(host, port)) from None
                    raise
                if answer.get("protocol") != PROTOCOL:  # a server from before the protocol had versions says none
                    raise RuntimeError(f"the Mac's bridge server speaks version {answer.get('protocol') or 1} of the bridge's "
                                       f"protocol, this image version {PROTOCOL}: build the image and start the server from "
                                       "the same state of the repository")
                for name, arguments in list(self.before.items()):
                    self._send({"k": "call", "fn": name, "a": arguments, "quiet": True})
            except BaseException:
                self._drop()
                raise

    def _drop(self):
        """Ends the connection, and with it what was known of the Mac's side.

        The Mac's process for this program ends with the connection: the tensors it kept are gone
        (native.cpp says so of each when it is next used), and the next connection is to a new
        process, which has never been told a signature and has no copy to send.
        """
        bridge_native.stop()
        if self.sock is not None:
            self.r.close()
            self.sock.close()
        self.sock = self.r = None
        self.signatures = {}
        self.pending = collections.deque()
        self.asked = self.landed = 0
        self.questions = 0
        self.epoch += 1

    def close(self):
        with self.lock:
            self._drop()

    def _forked(self):
        """In a process made by fork(): the socket and the tensors are the parent's. This process starts with none."""
        bridge_native.forked()
        self.lock = threading.RLock()
        if self.sock is not None:
            # This process's copy of the socket is closed; the parent's connection stays as it is. Not
            # through the reader: a thread of the parent's may have been reading an answer when this
            # process was made, and the reader's own lock is then held here by a thread that does not
            # exist, for ever. Closing it, or letting go of it (which closes it), would wait for that
            # lock: it is kept, untouched, and the descriptor is closed under it.
            ORPHANS.append(self.r)
            try:
                os.close(self.sock.detach())
            except OSError:
                pass
        self.sock = self.r = None
        self.signatures = {}
        self.pending = collections.deque()
        self.asked = self.landed = 0
        self.questions = 0
        self.epoch += 1

    # --- the tensors --------------------------------------------------------

    def handle_of(self, x):
        return (_handle(x) or None) if isinstance(x, Tensor) else None

    def any_remote(self, x):
        if isinstance(x, (list, tuple)):
            return any(self.any_remote(v) for v in x)
        return is_mps(x) or self.handle_of(x) is not None

    # --- the wire -----------------------------------------------------------

    def _send(self, header, blobs=()):
        if self.sock is None:
            self.connect()
        send(self.w, header, blobs)

    def _number(self):
        """The number of the next question. Its answer must carry it: see _mine."""
        self.questions += 1
        return self.questions

    def _mine(self, reply, number):
        """Checks that an answer is the one to the question `number`.

        Answers come in the order of the questions, and nothing else says which is which. Should the
        two ever get out of step (an answer left unread, by a fault here or on the Mac), every value
        read from then on would be the answer to another question: so the connection ends, loudly.
        """
        if reply is not None and reply.get("i") != number:
            self._drop()
            raise RuntimeError(f"the bridge lost its place among the Mac's answers (it waited for the answer to its question "
                               f"{number} and was given the one to {reply.get('i')}): the connection to the Mac was closed, "
                               "and what the program had on the GPU is gone")

    def _ask(self, header, blobs=()):
        with self.lock, Uninterrupted():
            number = self._number()
            self._send({**header, "i": number}, blobs)
            self.w.flush()
            self.waits += 1
            # Answers come in the order of the questions: the deferred ones first. A failure one of them
            # reports is raised below, once this question's own answer has been read too: raised before,
            # that answer would be left behind, and taken for the answer to the next question.
            earlier = self._land(self.asked, raising=False)
            started = time.perf_counter()
            try:
                reply, got = recv(self.r)
            except BaseException:  # the answer was not read to its end: nothing after it can be trusted
                self._drop()
                raise
            if header.get("k") != "hello":  # (the greeting's answer is the first frame, whoever sends it)
                self._mine(reply, number)
            bridge_native.caught_up()
            self.waited += time.perf_counter() - started
        if reply is None:
            raise RuntimeError("the bridge lost the Mac: its process for this program ended (it crashed, as the program itself "
                               "would have on a Mac, or was killed for the memory it took, or the server was stopped)")
        if earlier or reply.get("err"):
            raise failure(earlier or reply["err"])
        return reply, got

    def synchronize(self):
        self._ask({"k": "sync"})

    def call(self, name, *arguments):
        """Something that acts on the GPU itself and is not an operator (a seed, the memory in use): done on the Mac.

        This is what native.cpp's MPS hooks come to, so torch.mps.* and torch.backends.mps.* are PyTorch's own.
        """
        if name in BEFORE_CONNECTING and (name != "generator.manual_seed" or arguments[0] == 0):
            self.before[name] = list(arguments)
            if self.sock is None:
                return None
        if name == "generator.set_state":
            arguments = (arguments[0], arguments[1].tolist())
        if name in ONE_WAY:
            self._send({"k": "call", "fn": name, "a": list(arguments), "quiet": True})
            return None
        value = self._ask({"k": "call", "fn": name, "a": list(arguments)})[0].get("v")
        if name == "generator.get_state":
            value = torch.tensor(value, dtype=torch.uint8)
        return value

    # --- values asked for without waiting (copy_ with non_blocking=True) ----
    #
    # A program that copies a GPU tensor to the CPU "without blocking" goes on
    # issuing work and looks at the copy later, after waiting on an event it
    # recorded (torch.Event; install.py gives it one that counts these
    # copies). Text generation does this once per token, so that the GPU works
    # on a token while the next is being issued.

    def _later(self, into, source):
        with self.lock:
            number = self._number()
            self._send({"k": "get", "t": self.handle_of(source), "later": True, "i": number})
            self.w.flush()
            self.pending.append((into, number))
            self.asked += 1

    def _land(self, upto, raising=True):
        """Waits until the first `upto` of those copies have arrived.

        An answer can say that an operation failed on the Mac (the first since the last answer: the
        values copied are then not those the program computed, or there are none). Every answer
        read is one copy done with, failure or not, so that the next answer is the next copy's; the
        first failure is raised once they are all in, or returned if not `raising`.
        """
        failed = None
        with self.lock, Uninterrupted():
            started = time.perf_counter()
            while self.landed < upto:
                if self.r is None:
                    raise RuntimeError("the bridge lost the Mac before a copy asked of it arrived: what the program had on "
                                       "the GPU is gone")
                try:
                    reply, blobs = recv(self.r)
                except BaseException:
                    self._drop()
                    raise
                if reply is None:
                    raise RuntimeError("the bridge lost the Mac: its end closed the connection")
                into, number = self.pending.popleft()
                self.landed += 1
                self._mine(reply, number)
                failed = failed or reply.get("err")
                if blobs and blobs[0]:
                    data = torch.frombuffer(bytearray(blobs[0]), dtype=getattr(torch, reply["dtype"]))
                    into.copy_(data.reshape(reply["shape"]))
            self.waited += time.perf_counter() - started
        if failed and raising:
            raise failure(failed)
        return failed

    # --- moving data --------------------------------------------------------

    def upload(self, t):
        if t.dtype in DOUBLES:
            raise TypeError(NO_DOUBLE)
        t = t.detach().contiguous()
        if t.is_conj() or t.is_neg():
            t = t.resolve_conj().resolve_neg()
        tensor = _fake(t.shape, None, t.dtype)
        data = raw(t)
        self.uploaded += len(data)
        self._send({"k": "put", "t": _handle(tensor), "dtype": str(t.dtype)[6:], "shape": list(t.shape)}, [data])
        return tensor

    def download(self, tensor):
        reply, blobs = self._ask({"k": "get", "t": self.handle_of(tensor)})
        dtype = getattr(torch, reply["dtype"])
        if not blobs[0]:
            return torch.empty(reply["shape"], dtype=dtype)
        return torch.frombuffer(bytearray(blobs[0]), dtype=dtype).reshape(reply["shape"])

    def storage_bytes(self, storage):
        """A storage of ours as the CPU storage a Mac would save: all of the memory its tensor is in, byte for byte.

        The memory, not the tensor's values: a tensor that was made smaller covers less of its
        memory than the views made of it before, and those still read the rest. What cannot be
        fetched whole is not saved: an error, never a file with something else in its place.
        """
        handle = bridge_native.storage_handle(storage)
        if handle is None:
            return None
        if not handle:
            raise RuntimeError("the bridge cannot save this storage: the tensor it was the memory of has been given other "
                               "memory (set_), and no view of the first is left to fetch it through")
        _, blobs = self._ask({"k": "get", "t": handle, "storage": True})
        wanted = storage.nbytes()
        if len(blobs[0]) < wanted:
            raise RuntimeError(f"the bridge cannot save this storage: it stands for {wanted} bytes, and the memory its "
                               f"tensor is in on the Mac has {len(blobs[0])}")
        if not wanted:
            return torch.empty(0, dtype=torch.uint8).untyped_storage()
        return torch.frombuffer(bytearray(blobs[0][:wanted]), dtype=torch.uint8).untyped_storage()

    # --- arguments ----------------------------------------------------------

    def _prepare(self, x, keep=None):
        """One argument as (its stand-in for working out shapes, what the Mac gets).

        `keep`: where to hold the tensors sent to the Mac for this operation alone (indices that were
        on the CPU), until the operation itself is sent: let go before, the Mac would be told to
        forget them first.
        """
        if isinstance(x, (list, tuple)):
            pairs = [self._prepare(v, keep) for v in x]
            return type(x)(p[0] for p in pairs), [p[1] for p in pairs]
        if isinstance(x, Tensor):
            handle = self.handle_of(x)
            if handle is None:
                if x.device.type != "cpu":
                    raise NotImplementedError(f"the bridge does not know this {x.device} tensor: it was not made by an operation on the GPU")
                if x.dim() == 0:  # a Python number PyTorch wrapped in a tensor
                    return x.to(META), {"s": x.item(), "dt": str(x.dtype)[6:]}
                x = self.upload(x)
                handle = _handle(x)
                if keep is not None:
                    keep.append(x)
            offset = x.storage_offset()
            if offset:  # a view, which starts further on in its storage: what is made of it starts from there
                extent = offset + 1 + sum((size - 1) * stride for size, stride in zip(x.shape, x.stride())) if x.numel() else offset
                return torch.empty(extent, dtype=x.dtype, device=META).as_strided(x.shape, x.stride(), offset), {"t": handle}
            return _empty_strided(x.shape, x.stride(), dtype=x.dtype, device=META), {"t": handle}
        if isinstance(x, torch.dtype):
            return x, {"dtype": str(x)[6:]}
        if isinstance(x, torch.device):
            return (META, {"device": "mps"}) if is_mps(x) else (x, {"device": str(x)})
        if isinstance(x, torch.layout):
            return x, {"layout": str(x)[6:]}
        if isinstance(x, torch.memory_format):
            return x, {"mf": str(x)[6:]}
        if isinstance(x, torch.Generator):  # one of the Mac's, by its number there; the shapes need none
            number = bridge_native.generator(x)
            if number < 0:
                raise RuntimeError(f"Expected a 'mps:0' device type for generator but found '{x.device}'")
            return None, {"g": number}
        if x is None or isinstance(x, (bool, int, float, str)):
            return x, x
        raise NotImplementedError(f"the bridge cannot send a {type(x).__name__} to the Mac")

    # --- what native.cpp does not know how to do ----------------------------

    def _slow(self, name, overload, args, kwargs, routine, varying=()):
        """An operation seen for the first time (`routine`), or one that never will be routine.

        `varying`: which of its arguments are real numbers that may change from one call to the
        next without its being another operation (native.cpp, varies).
        """
        if self.slow is None:
            result = self._general(name, overload, args, kwargs, routine, varying)
        else:
            self.slow[(name, "first time" if routine else "unusual arguments")] += 1
            started, waited = time.perf_counter(), self.waited
            try:
                result = self._general(name, overload, args, kwargs, routine, varying)
            finally:
                self.slowly += time.perf_counter() - started - (self.waited - waited)
        if self.every:
            self.synchronize()
        return result

    def _operator(self, name, overload):
        func = self.operators.get((name, overload))
        if func is None:
            func = self.operators[name, overload] = getattr(getattr(torch.ops.aten, name[6:]), overload or "default")
        return func

    def _general(self, name, overload, args, kwargs, routine, varying=()):
        if name == "aten::set_" and overload.startswith("source_Storage"):
            return self._rebound_to_storage(overload, args, kwargs)
        func = self._operator(name, overload)
        if name == "aten::_to_copy":
            src, device = args[0], kwargs.get("device")
            remote = self.handle_of(src) is not None
            if remote and device is not None and device.type == "cpu":
                return func(self.download(src), **kwargs)
            if not remote and is_mps(device):
                return self.upload(func(src, **{**kwargs, "device": None}))
        if name == "aten::copy_" and not args[0].is_mps:  # the Mac's values, copied into a tensor that lives here
            if len(args) > 2 and args[2] and args[0].shape == args[1].shape:
                self._later(args[0], args[1])
                return args[0]
            return args[0].copy_(self.download(args[1]))
        if name == "aten::copy_" and not args[1].is_mps and args[1].shape == args[0].shape and bridge_native.fresh(args[0]):
            # .to("mps"): PyTorch has just made the tensor that is to receive this data. It goes straight into it.
            data = raw(args[1].to(args[0].dtype).resolve_conj().resolve_neg())
            self.uploaded += len(data)
            self._send({"k": "put", "t": _handle(args[0]), "dtype": str(args[0].dtype)[6:], "shape": list(args[0].shape)}, [data])
            return args[0]

        if name in INDEXING:  # the indices may be on the CPU; the tensor indexed and the values may not
            lent = self._on_cpu(args[0], args[2:], kwargs)
        else:
            lent = name != "aten::copy_" and self._on_cpu(args, kwargs)
        if lent:
            return self._mixed(func, name, overload, args, kwargs)
        # Indices that are still here, and what is being copied to the GPU, go to the Mac's GPU first:
        # PyTorch itself moves them there.
        args = [self.upload(a) if isinstance(a, Tensor) and not a.is_mps and a.dim() else a for a in args]
        keep = []
        stand_ins, wire_args = self._prepare(args, keep)
        prepared = {k: self._prepare(v, keep) for k, v in kwargs.items()}
        wire_kwargs = {k: v[1] for k, v in prepared.items()}
        header = {"k": "op", "op": name, "ov": overload or "default", "a": wire_args, "kw": wire_kwargs}

        if name == "aten::set_":
            return self._rebound(overload, args, kwargs, header)
        if name in NEEDS_VALUES:
            return self._wait_for(header, False)
        header["varying"] = varying
        if not func._schema.returns:  # nothing comes back (a fused optimizer's step): its effect is on its arguments, on the Mac
            header["out"] = []
            if routine:
                self._remember(func, header, [], False, run=True)
            else:
                self._send(header)
            return None
        if self._in_values(func):
            return self._wait_for(header, False)
        try:
            result = func(*stand_ins, **{k: v[0] for k, v in prepared.items()})
        except NotImplementedError:
            # PyTorch cannot work this one out without data: the Mac can. What it answers is remembered
            # only for an operation known to give the same shapes for the same shapes (SHAPE_FROM_SHAPES):
            # for any other the Mac is asked every time.
            if routine and name not in SHAPE_FROM_SHAPES:
                self.every_time[name] += 1
            return self._wait_for(header, routine and name in SHAPE_FROM_SHAPES)

        outs = list(result) if isinstance(result, (tuple, list)) else [result]
        if not all(o is None or isinstance(o, Tensor) for o in outs) or all(o is None for o in outs):
            return result  # about shapes only: worked out here, nothing to run
        if any(o is not None and o.dtype in DOUBLES for o in outs):
            raise TypeError(NO_DOUBLE)
        tensors, spec = [], []
        for o in outs:
            if o is None:  # a result the operator does not produce here (a gradient nobody asked for)
                tensors.append(None)
                spec.append(None)
                continue
            same = next((i for i, a in enumerate(stand_ins) if a is o), None)
            named = next((k for k, v in prepared.items() if v[0] is o), None)
            if named is not None:  # the result goes into a tensor that was given for it (out=)
                tensor = kwargs[named]
                position = self._positions(func)[named]
                if (o.shape, o.stride()) != (tensor.shape, tensor.stride()):  # ...and takes the result's shape
                    bridge_native.set_geometry(tensor, list(o.shape), list(o.stride()), o.storage_offset())
                    position = (position, tensor)
                spec.append(position)
            elif same is None:
                tensor = _fake(o.shape, None if o.is_contiguous() else o.stride(), o.dtype, 0, o.storage_offset())
                spec.append(tensor)
            else:  # the operation handed back its argument
                tensor = args[same]
                if (o.shape, o.stride()) != (tensor.shape, tensor.stride()):  # ...reshaped in place: the tensor here follows
                    bridge_native.set_geometry(tensor, list(o.shape), list(o.stride()), o.storage_offset())
                    same = (same, tensor)
                spec.append(same)
            tensors.append(tensor)
        header["out"] = [0 if t is None else _handle(t) for t in tensors]
        if routine:
            self._remember(func, header, spec, isinstance(result, list), run=True)
        else:
            self._send(self._plain(header))
        return type(result)(tensors) if isinstance(result, (tuple, list)) else tensors[0]

    def _rebound(self, overload, args, kwargs, header):
        """x.set_(source): the tensor is given another tensor's memory, or with no source none at all.

        Not an operation like the others: what changes is which memory the tensor is in, and the
        tensor here must follow (native.cpp, rebind), or saving it would write the memory it had.
        The Mac is waited for first: refused there (another type, a place outside the source's
        memory), nothing changes here.
        """
        target = args[0]
        if overload == "source_Tensor":
            source = args[1]
            sizes, strides, offset = list(source.shape), list(source.stride()), source.storage_offset()
        elif overload == "source_Tensor_storage_offset":
            source = args[1]
            given = dict(zip(("storage_offset", "size", "stride"), args[2:]), **kwargs)
            sizes, strides, offset = list(given["size"]), list(given["stride"]), int(given["storage_offset"])
        elif not overload:
            source, sizes, strides, offset = None, [0], [1], 0
        else:
            raise NotImplementedError(f"the bridge does not carry set_ of a storage ({overload}): give it a tensor")
        header["out"] = [_handle(target)]
        self._send(self._plain(header))
        self._ask({"k": "call", "fn": "ping", "a": []})
        bridge_native.rebind(target, source if self.handle_of(source) else None, sizes, strides, offset)
        return target

    def _rebound_to_storage(self, overload, args, kwargs):
        """x.set_(storage, offset, size, stride), which is also what PyTorch makes of x.set_(tensor, offset, size, stride).

        A storage does not travel. One of ours stands for the memory a tensor is in on the Mac: the
        Mac is told to give x that tensor's memory, at that place in it, which is the same thing there.
        """
        target, storage = args[0], args[1]
        handle = bridge_native.storage_handle(storage)
        if not handle:
            raise NotImplementedError("the bridge cannot give a tensor on the GPU the memory of a storage that is not on the GPU: "
                                      "move the data there first (.to('mps'))")
        if overload == "source_Storage":
            sizes, strides, offset = [storage.nbytes() // target.element_size()], [1], 0
        else:
            given = dict(zip(("storage_offset", "size", "stride"), args[2:]), **kwargs)
            sizes, strides, offset = list(given["size"]), list(given["stride"]), int(given["storage_offset"])
        self._send({"k": "op", "op": "aten::set_", "ov": "source_Tensor_storage_offset", "kw": {}, "out": [_handle(target)],
                    "a": [{"t": _handle(target)}, {"t": handle}, offset, sizes, strides]})
        self._ask({"k": "call", "fn": "ping", "a": []})
        bridge_native.rebind(target, storage, sizes, strides, offset)
        return target

    def _in_values(self, func):
        """Whether an operator's results have shapes that depend on its arguments' values, by PyTorch's own account
        of it (the tag `dynamic_output_shape`: unique_consecutive, repeat_interleave, bincount), when no argument
        lets PyTorch work the shape out here. Such a result is never remembered: the program waits, every time.

        (An index by tensors has the tag for the case of a mask. With indices that are numbers PyTorch
        works its shape out here; with a mask it cannot, and says so: that is caught where it is tried.)
        """
        known = self.in_values.get(func)
        if known is None:
            dynamic = getattr(torch.Tag, "dynamic_output_shape", None)
            known = self.in_values[func] = dynamic is not None and dynamic in func.tags and func._schema.name != "aten::index"
        return known

    @staticmethod
    def _on_cpu(*values):
        """Whether there is a tensor with data here, on the CPU, among an operation's arguments (a 0-dim one is a number)."""
        for value in values:
            if isinstance(value, Tensor):
                if value.dim() and not value.is_mps:
                    return True
            elif isinstance(value, (list, tuple)):
                if Bridge._on_cpu(*value):
                    return True
            elif isinstance(value, dict):
                if Bridge._on_cpu(*value.values()):
                    return True
        return False

    def _mixed(self, func, name, overload, args, kwargs):
        """An operation on the GPU with a tensor of the CPU's among its arguments (x_gpu + y_cpu, a
        packed sequence's lengths). The tensor goes to the Mac as what it is, a CPU tensor, and the Mac
        does with the operation what it does in a program of its own: most refuse it ("Expected all
        tensors to be on the same device"), a few are made for it. The program waits, so that a
        refusal is raised here, at the call. A result that is a CPU tensor on the Mac is one here."""
        lent = []

        def wire(value):
            if isinstance(value, (list, tuple)):
                return [wire(v) for v in value]
            if isinstance(value, Tensor) and value.dim() and not value.is_mps:
                if value.device.type != "cpu":
                    raise NotImplementedError(f"the bridge does not know this {value.device} tensor")
                handle = bridge_native.handles(1)
                value = value.detach().resolve_conj().resolve_neg()
                self._send({"k": "put", "t": handle, "cpu": True, "dtype": str(value.dtype)[6:], "shape": list(value.shape)}, [raw(value)])
                lent.append(handle)
                return {"t": handle}
            return self._prepare(value)[1]

        header = {"k": "op", "op": name, "ov": overload or "default", "a": wire(args), "kw": {k: wire(v) for k, v in kwargs.items()},
                  "sync": True, **self._room(args, kwargs)}
        try:
            reply, blobs = self._ask(header)
        finally:
            if lent and self.sock is not None:
                self.w.write(struct.pack(f"<cI{len(lent)}I", b"F", len(lent), *lent))
        values, _ = self._results(reply, blobs, header["base"])
        return tuple(values) if reply["seq"] else values[0]

    def _room(self, *arguments):
        """Numbers set aside for the tensors an operation that is waited for will give: where they start, and how many.

        Eight, and for an operator that gives lists of tensors (the gradients of an LSTM's weights,
        one for each weight it was given) twice as many again as it takes. The Mac stores no more
        than that: an operator that gave more would otherwise have its results stored under
        numbers that are other tensors'.
        """
        def count(value):
            if isinstance(value, (list, tuple)):
                return sum(count(v) for v in value)
            if isinstance(value, dict):  # as it goes to the Mac, a tensor is {"t": its number}
                return 1 if set(value) == {"t"} else sum(count(v) for v in value.values())
            return int(isinstance(value, Tensor))

        room = 8 + 2 * count(arguments)
        return {"base": bridge_native.handles(room), "room": room}

    def _results(self, reply, blobs, base):
        """What an operation the Mac ran and was waited for gave: (its results, whether they are tensors of the GPU's and no other thing)."""
        plain = True

        def one(item):
            nonlocal plain
            if "t" in item:
                shape, dtype, stride, offset = item["t"]
                return _fake(shape, stride, getattr(torch, dtype), base + item["n"], offset)
            plain = False
            if "l" in item:
                return [one(element) for element in item["l"]]
            if "c" in item:
                shape, dtype, which = item["c"]
                data = blobs[which]
                return (torch.frombuffer(bytearray(data), dtype=getattr(torch, dtype)).reshape(shape) if data
                        else torch.empty(shape, dtype=getattr(torch, dtype)))
            return item["v"]

        return [one(item) for item in reply["res"]], plain

    @staticmethod
    def _plain(header):
        """An operation's header as it goes on the wire when it is sent whole, not as a remembered one."""
        return {k: v for k, v in header.items() if k != "varying"}

    def _positions(self, func):
        """Where each of an operator's arguments sits among them all: native.cpp counts them that way."""
        positions = self.positions.get(func)
        if positions is None:
            positions = self.positions[func] = {a.name: i for i, a in enumerate(func._schema.arguments)}
        return positions

    def _remember(self, func, header, spec, as_list, run=False):
        """Makes the operation routine: next time, with these shapes, it is a few integers and no Python.

        The Mac knows an operation by its signature: operator, constants, how many tensors. Shapes
        are not in it, so an operation met again with new shapes (text generation: every token is
        one longer) has its signature already, and `run` sends it as the few integers it will be.
        """
        template, keywords, inputs, reals = [], {}, [], []
        varying = set(header.get("varying", ()))

        def real(a, position):
            """A real number that may change from call to call: not in the signature, sent with each call."""
            if position not in varying:
                return a
            if isinstance(a, float):  # a plain number, also where the operator takes a tensor (x * 0.5)
                reals.append(a)
                return {"f": len(reals) - 1}
            if isinstance(a, dict) and "s" in a:  # a 0-dim CPU tensor
                reals.append(float(a["s"]))
                return {"f": len(reals) - 1, "dt": a["dt"]}
            return a

        for position, a in enumerate(header["a"]):
            if isinstance(a, dict) and "t" in a:
                inputs.append(a["t"])
                a = {"t": len(inputs) - 1}
            elif isinstance(a, list) and a and all(isinstance(e, dict) and "t" in e for e in a):
                first = len(inputs)
                inputs.extend(e["t"] for e in a)
                a = {"tl": [first, len(a)]}  # a list of tensors: its first number, how many
            template.append(real(a, position))
        positions = self._positions(func)
        for name, a in sorted(header["kw"].items(), key=lambda item: positions[item[0]]):  # the order native.cpp meets them in
            if isinstance(a, dict) and "t" in a:
                inputs.append(a["t"])
                a = {"t": len(inputs) - 1}
            keywords[name] = real(a, positions[name])
        spec_ = {"op": header["op"], "ov": header["ov"], "a": template, "kw": keywords, "nin": len(inputs), "nout": len(spec),
                 "nf": len(reals)}
        known = json.dumps(spec_, sort_keys=True)
        signature = self.signatures.get(known)
        if signature is None:
            signature = self.signatures[known] = bridge_native.signature()
            define(self.w, signature, spec_)
        if run:
            self.w.write(struct.pack(f"<cI{len(inputs) + len(spec)}I{len(reals)}d", b"O", signature, *inputs, *header["out"], *reals))
        # With a real number left out of what the operation is remembered by, its results must be the
        # same kind of thing whatever the number. A new tensor is; the argument handed back is only
        # for an operator that says so (x.add_(y, alpha=...)). One that hands its argument back for
        # one value and makes a tensor for another (a dropout with p=0) is not remembered.
        returns = func._schema.returns
        for i, out in enumerate(spec):
            if reals and isinstance(out, (int, tuple)) and not (i < len(returns) and returns[i].alias_info and returns[i].alias_info.is_write):
                return
        bridge_native.remember(signature, spec, as_list)

    def _wait_for(self, header, routine):
        """Runs an operation on the Mac and waits for what came of it: values, or the results' shapes."""
        room = self._room(header["a"], header["kw"])
        reply, blobs = self._ask({**self._plain(header), "sync": True, **room})
        values, plain = self._results(reply, blobs, room["base"])
        if routine and plain and values:
            self._remember(self._operator(header["op"], "" if header["ov"] == "default" else header["ov"]), header, values, False)
        return tuple(values) if reply["seq"] else values[0]

    def report(self, file=sys.stderr):
        sent, missed, remembered = bridge_native.counts()
        for (name, why), count in sorted((self.slow or {}).items(), key=lambda item: -item[1])[:25]:
            print(f"[bridge] slow path {count:6d}  {name}  ({why})", file=file)
        for name, count in self.every_time.most_common(25):
            print(f"[bridge] asked of the Mac at every call {count:6d}  {name}  (PyTorch cannot work out its result's shape here, "
                  "and it is not in client.py's SHAPE_FROM_SHAPES)", file=file)
        print(
            f"[bridge] {sent} operations sent without Python, {missed} through it ({remembered} remembered), {self.waits} round trips waited for, "
            f"{self.uploaded / 1e6:.0f} MB of tensors sent to the Mac, {self.waited:.2f} s waiting for the Mac"
            + (f", {self.slowly:.2f} s off the fast path" if self.slow is not None else ""),
            file=file,
        )
