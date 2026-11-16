"""What the Mac end refuses, said straight in the protocol:   python refusals.py   (on the Mac, with its PyTorch)

A container is not trusted. Whatever it sends, the Mac end must run nothing but the operators
of its list on plain data, read no file, read no memory outside a message, and keep its door
open. Each case here sends what a program of the bridge never would and expects a refusal; a
few send what a program does, and expect it to work (a refusal of everything would pass
otherwise).

The server is this folder's, started here on a Unix socket in a temporary directory and stopped
at the end: nothing else on the Mac is spoken to. The one file read is made here, in that
directory, and the case that names it expects not to get its bytes.
"""

import importlib.metadata
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from bridge import wire  # noqa: E402

TORCH = importlib.metadata.version("torch").split("+")[0]
MARKER = b"varco-refusals: these bytes are a file of the Mac's"


class Door:
    """A server of this folder's bridge, on a socket of its own."""

    def __init__(self, folder, executor=True):
        self.path = os.path.join(folder, "gpu.sock")
        if os.path.exists(self.path):  # left by a server that ended badly
            os.unlink(self.path)
        self.log = open(os.path.join(folder, "server.log"), "w+")
        # BRIDGE_EXECUTOR=0: the Mac end without its C++ (executor.cpp), which Python then does the work of.
        self.process = subprocess.Popen([sys.executable, "-m", "bridge.server", self.path], cwd=HERE, stdout=self.log,
                                        stderr=subprocess.STDOUT, env={**os.environ, "BRIDGE_EXECUTOR": "1" if executor else "0"})
        deadline = time.time() + 60
        while True:  # until it listens
            probe = socket.socket(socket.AF_UNIX)
            try:
                probe.connect(self.path)
                break
            except OSError:
                if self.process.poll() is not None or time.time() > deadline:
                    raise RuntimeError("the server did not start: " + self.said()) from None
                time.sleep(0.05)
            finally:
                probe.close()

    def alive(self):
        return self.process.poll() is None

    def said(self):
        self.log.flush()
        self.log.seek(0)
        return self.log.read()

    def stop(self):
        if self.alive():
            self.process.terminate()
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.log.close()


class Writer:
    def __init__(self, sock):
        self.sock = sock

    def write(self, *pieces):
        self.sock.sendall(b"".join(bytes(p) for p in pieces))


class Peer:
    """A connection that says what it likes."""

    def __init__(self, door, greet=True, env=None):
        self.sock = socket.socket(socket.AF_UNIX)
        self.sock.settimeout(60)
        self.sock.connect(door.path)
        self.w = Writer(self.sock)
        self.r = self.sock.makefile("rb")
        self.asked = 0
        if greet:
            hello = {"k": "hello", "torch": TORCH, "pid": os.getpid(), "program": "refusals.py", "env": env or {}}
            if hasattr(wire, "PROTOCOL"):
                hello["protocol"] = wire.PROTOCOL
            self.greeting = self.ask(hello)[0]

    def send(self, header, blobs=()):
        wire.send(self.w, header, blobs)

    def ask(self, header, blobs=()):
        """A question and its answer: (header, blobs), or (None, None) if the Mac end closed."""
        if hasattr(wire, "PROTOCOL") and header.get("k") != "hello":
            self.asked += 1
            header = {**header, "i": self.asked}
        self.send(header, blobs)
        try:
            return wire.recv(self.r)
        except (OSError, ValueError, struct.error):
            return None, None

    def put(self, handle, values, cpu=False, dtype="float32", code="f"):
        self.send({"k": "put", "t": handle, "cpu": cpu, "dtype": dtype, "shape": [len(values)]},
                  [struct.pack(f"<{len(values)}{code}", *values)])

    def op(self, name, overload, a, kw=None, base=1000, room=8):
        """An operation the program waits for: what came of it. Its tensors are stored from `base`, `room` of them at most."""
        header = {"k": "op", "op": name, "ov": overload, "a": a, "kw": kw or {}, "sync": True, "base": base, "room": room}
        return self.ask({key: value for key, value in header.items() if value is not None})

    def define(self, signature, spec):
        wire.define(self.w, signature, spec)

    def run(self, signature, handles, reals=(), after=b""):
        self.w.write(struct.pack(f"<cI{len(handles)}I{len(reals)}d", b"O", signature, *handles, *reals) + after)

    def get(self, handle):
        return self.ask({"k": "get", "t": handle})

    def ping(self):
        """The first failure since the last answer, or None."""
        reply, _ = self.ask({"k": "call", "fn": "ping", "a": []})
        return "the Mac end closed the connection" if reply is None else reply.get("err")

    def values(self, handle, code="f"):
        reply, blobs = self.get(handle)
        if reply is None or reply.get("err") or not blobs:
            return None
        return list(struct.unpack(f"<{len(blobs[0]) // struct.calcsize(code)}{code}", blobs[0]))

    def close(self):
        try:
            self.r.close()
            self.sock.close()
        except OSError:
            pass


results = []


def case(kind, name, ok, detail):
    results.append((kind, ok))
    said = ("refused " if ok else "ACCEPTED") if kind == "refusal" else ("works   " if ok else "FAILS   ")
    print(f"{said} {name}: {str(detail)[:150]}", flush=True)


def refused(name, error, also=True):
    """`error`: what the Mac end answered. A refusal is an error; `also` is whatever else must hold."""
    case("refusal", name, bool(error) and also, error if error else "no error came back")


def works(name, ok, detail=""):
    case("control", name, ok, detail)


def operators(door, folder):
    """C-1: only the operators of the list, on plain data, and no file of the Mac's."""
    secret = os.path.join(folder, "a-file-of-the-macs.bin")
    with open(secret, "wb") as f:
        f.write(MARKER)

    p = Peer(door)
    # What a program does: these must go on working.
    p.put(1, [1.0, 2.0, 3.0])
    p.put(2, [10.0, 20.0, 30.0])
    reply, _ = p.op("aten::add", "Tensor", [{"t": 1}, {"t": 2}], {"alpha": 1})
    works("an addition, waited for", bool(reply) and not reply.get("err") and p.values(1000) == [11.0, 22.0, 33.0], reply)
    p.define(1, {"op": "aten::mul", "ov": "Tensor", "a": [{"t": 0}, {"t": 1}], "kw": {}, "nin": 2, "nout": 1, "nf": 0})
    p.run(1, [1, 2, 3])
    works("a multiplication, as a signature", p.values(3) == [10.0, 40.0, 90.0])
    reply, _ = p.op("aten::div", "Tensor_mode", [{"t": 2}, {"t": 1}], {"rounding_mode": "floor"})
    works("a division with its rounding mode, a string", bool(reply) and not reply.get("err"), reply)
    reply, _ = p.op("aten::_reshape_alias", "default", [{"t": 1}, [3, 1], [1, 1]])
    works("a reshape inside its tensor", bool(reply) and not reply.get("err") and reply["res"][0]["t"][0] == [3, 1], reply)
    reply, _ = p.op("aten::empty", "memory_format", [[2, 3]], {"dtype": {"dtype": "float32"}, "device": {"device": "mps"}})
    works("an empty tensor on the GPU", bool(reply) and not reply.get("err") and reply["res"][0]["t"][0] == [2, 3], reply)
    works("no failure left behind by the above", p.ping() is None)

    # A file of the Mac's, by name: as a question...
    reply, blobs = p.op("aten::from_file", "default", [secret, False, len(MARKER)], {"dtype": {"dtype": "uint8"}})
    got = b"".join(bytes(b) for b in blobs or ())
    refused("from_file, asked for", reply and reply.get("err"), MARKER not in got)
    if MARKER in got:
        print(f"         the Mac end read {secret} and sent back its {len(got)} bytes")
    # ...and as a signature, which the C++ of the Mac end runs.
    p.define(2, {"op": "aten::from_file", "ov": "default", "a": [secret, False, len(MARKER)], "kw": {"dtype": {"dtype": "uint8"}},
                 "nin": 0, "nout": 1, "nf": 0})
    p.run(2, [50])
    reply, blobs = p.get(50)
    got = b"".join(bytes(b) for b in blobs or ())
    refused("from_file, as a signature", reply and reply.get("err"), MARKER not in got)
    if MARKER in got:
        print(f"         the Mac end read {secret} and sent back its {len(got)} bytes")
    p.ping()

    # An operator that is not PyTorch's own tensor library.
    p.define(3, {"op": "prim::Uninitialized", "ov": "default", "a": [], "kw": {}, "nin": 0, "nout": 1, "nf": 0})
    refused("a signature of an operator outside aten", p.ping())
    reply, _ = p.op("quantized::add", "default", [{"t": 1}, {"t": 2}, 1.0, 0])
    refused("an operator outside aten, asked for", reply and reply.get("err"))
    reply, _ = p.op("aten::_print", "default", ["said by a container"])
    refused("an operator that is not in the list (_print)", reply and reply.get("err"))
    reply, _ = p.op("aten::add", "no_such_overload", [{"t": 1}, {"t": 2}])
    refused("an overload that does not exist", reply and reply.get("err"))
    reply, _ = p.op("aten::add.Tensor", "Tensor", [{"t": 1}, {"t": 2}])
    refused("a name that is not one operator's", reply and reply.get("err"))
    reply, _ = p.op("aten::set_", "source_Storage", [{"t": 1}, {"t": 2}])
    refused("an operator that takes a storage", reply and reply.get("err"))

    # Devices: the GPU and the CPU, and no other.
    for device in ("meta", "cuda", "cuda:0", "mps:1", "cpu:3", "xla", "", 7, None, ["mps"]):
        reply, _ = p.op("aten::empty", "memory_format", [[2]], {"dtype": {"dtype": "float32"}, "device": {"device": device}})
        refused(f"a tensor made on the device {device!r}", reply and reply.get("err"))

    # Arguments: plain data of the kind the operator takes.
    for label, a, kw in (
        ("a string where a tensor goes", ["x", {"t": 2}], {}),
        ("a dictionary of an unknown kind", [{"t": 1}, {"callable": "os.system"}], {}),
        ("a type that is not one", [[2]], {"dtype": {"dtype": "save"}}),
        ("a type given as an attribute path", [[2]], {"dtype": {"dtype": "float32.__class__"}}),
        ("an argument the operator has not", [{"t": 1}, {"t": 2}], {"filename": "/etc/hosts"}),
        ("more arguments than the operator takes", [{"t": 1}, {"t": 2}, 1, 2, 3, 4, 5, 6], {}),
        ("a list where a number goes", [{"t": 1}, {"t": 2}], {"alpha": [1, 2]}),
    ):
        name = "aten::empty" if "dtype" in kw else "aten::add"
        reply, _ = p.op(name, "memory_format" if "dtype" in kw else "Tensor", a, kw)
        refused(label, reply and reply.get("err"))
    reply, _ = p.op("aten::div", "Tensor_mode", [{"t": 2}, {"t": 1}], {"rounding_mode": "../../etc/hosts"})
    refused("a path in a string argument", reply and reply.get("err"))

    # Sizes, strides and offsets: checked before PyTorch is given them.
    reply, _ = p.op("aten::_reshape_alias", "default", [{"t": 1}, [4096], [1]])
    refused("a reshape that reaches past its tensor's memory", reply and reply.get("err"))
    reply, _ = p.op("aten::as_strided", "default", [{"t": 1}, [4096], [1], 0])
    refused("as_strided past its tensor's memory", reply and reply.get("err"))
    for label, name, overload, a, kw in (
        ("a negative size", "aten::empty", "memory_format", [[-3]], {}),
        ("sizes whose product overflows", "aten::empty", "memory_format", [[1 << 40, 1 << 40]], {}),
        ("a size beyond 64 bits", "aten::empty", "memory_format", [[1 << 70]], {}),
        ("a negative stride", "aten::empty_strided", "default", [[4], [-1]], {}),
        ("strides that do not match the sizes", "aten::empty_strided", "default", [[4, 4], [1]], {}),
        ("a negative offset", "aten::as_strided", "default", [{"t": 1}, [1], [1], -8], {}),
        ("a resize whose size overflows", "aten::resize_", "default", [{"t": 1}, [1 << 40, 1 << 40]], {}),
    ):
        reply, _ = p.op(name, overload, a, {"dtype": {"dtype": "float32"}, **kw} if "empty" in name else kw)
        refused(label, reply and reply.get("err"))

    # An operation that fails half way: PyTorch's resize_ takes the new shape, then fails to get the memory
    # for it, and leaves a tensor of 2^50 numbers with the memory of one. It is not to be used after that
    # (asked through a signature, so that nothing is read back).
    p.put(70, [1.0])
    reply, _ = p.op("aten::resize_", "default", [{"t": 70}, [1 << 50]])
    works("a resize to more memory than there is fails", bool(reply) and bool(reply.get("err")), reply)
    p.define(7, {"op": "aten::slice", "ov": "Tensor", "a": [{"t": 0}, 0, 0, 4096, 1], "kw": {}, "nin": 1, "nout": 1, "nf": 0})
    p.run(7, [70, 71])
    refused("a view of a tensor whose resize failed half way", p.ping())

    # Data without bytes.
    p.send({"k": "put", "t": 60, "dtype": "float32", "shape": [1 << 16]}, [b""])
    refused("a tensor of 65536 numbers sent with no bytes", p.ping())
    p.send({"k": "put", "t": 61, "dtype": "float32", "shape": [3]}, [b"\0" * 8])
    refused("a tensor whose bytes are not its shape's", p.ping())
    p.send({"k": "put", "t": 62, "dtype": "float32.__class__", "shape": [1]}, [b"\0" * 4])
    refused("data of a type that is not one", p.ping())

    # Named calls: those of the table, with the arguments they take.
    for label, fn, a in (
        ("a call that is not in the table", "__import__", ["os"]),
        ("a seed that is not a number", "manual_seed", ["x"]),
        ("a memory fraction that is a list", "set_per_process_memory_fraction", [[1.0]]),
        ("a call with too many arguments", "synchronize", [1, 2]),
        ("a generator that was never made", "generator.initial_seed", [12345]),
    ):
        reply, _ = p.ask({"k": "call", "fn": fn, "a": a})
        refused(label, reply is None or reply.get("err"))
    works("after all that, the connection still computes", p.values(3) == [10.0, 40.0, 90.0])
    p.close()


def unchecked(door, native=True):
    """D-1: no operator that takes on trust what it does not check, and no kernel of the CPU's on a container's data.

    Every case gives the operator arguments that are in order, so that where it is still run
    nothing is read or written out of place: what is looked at is whether the Mac end runs it
    at all, and with which kernel.
    """
    listed = "not in the list of operators"
    p = Peer(door)
    p.put(1, [1.0, 2.0, 3.0])
    p.put(2, [1, 0, 1], dtype="bool", code="?")
    p.put(3, [0, 2, 1], dtype="int64", code="q")
    p.put(4, [5.0, 6.0, 7.0])
    # PyTorch's own account of these two: "the indices are expected to be in bounds and is not checked", and
    # they walk the list of indices against the tensor's sizes without asking how many of each there are.
    reply, _ = p.op("aten::_unsafe_masked_index", "default", [{"t": 1}, {"t": 2}, [{"t": 3}], 0.0])
    refused("_unsafe_masked_index, with indices in order", reply and reply.get("err"), bool(reply) and listed in str(reply.get("err")))
    reply, _ = p.op("aten::_unsafe_masked_index_put_accumulate", "default", [{"t": 1}, {"t": 2}, [{"t": 3}], {"t": 4}])
    refused("_unsafe_masked_index_put_accumulate, with indices in order", reply and reply.get("err"),
            bool(reply) and listed in str(reply.get("err")))
    p.define(1, {"op": "aten::_unsafe_masked_index", "ov": "default", "a": [{"t": 0}, {"t": 1}, {"tl": [2, 1]}, 0.0], "kw": {},
                 "nin": 3, "nout": 1, "nf": 0})
    p.run(1, [1, 2, 3, 9])
    error = p.ping()
    refused("_unsafe_masked_index, as a signature", error, listed in str(error))
    # Those named "unsafe" that were read and found to hand their arguments to an operator that checks them.
    reply, _ = p.op("aten::_unsafe_index", "Tensor", [{"t": 1}, [{"t": 3}]])
    works("_unsafe_index, which is index", bool(reply) and not reply.get("err") and p.values(1000) == [1.0, 3.0, 2.0], reply)
    reply, _ = p.op("aten::_unsafe_view", "default", [{"t": 1}, [3, 1]])
    works("_unsafe_view, which is view", bool(reply) and not reply.get("err") and reply["res"][0]["t"][0] == [3, 1], reply)
    reply, _ = p.op("aten::_unsafe_view", "default", [{"t": 1}, [4096]])
    refused("_unsafe_view to more elements than there are", reply and reply.get("err"))
    reply, _ = p.op("aten::_unsafe_index", "Tensor", [{"t": 1}, [{"t": 3}, {"t": 3}, {"t": 3}]])
    refused("_unsafe_index with more indices than dimensions", reply and reply.get("err"))
    # An argument that says "do not check" (autograd sets it, for the gradient of an index): the operator
    # is run, and checks all the same.
    reply, _ = p.op("aten::_index_put_impl_", "default", [{"t": 1}, [{"t": 3}], {"t": 4}, False, True])
    works("_index_put_impl_ told not to check (unsafe=True), with indices in order",
          bool(reply) and not reply.get("err") and p.values(1) == [5.0, 7.0, 6.0], reply)
    # (The GPU's kernel looks at the indices while it runs: a Mac reports one out of place when the tensor is next read.)
    p.put(5, [0, 4096, 1], dtype="int64", code="q")
    reply, _ = p.op("aten::_index_put_impl_", "default", [{"t": 1}, [{"t": 5}], {"t": 4}, False, True])
    reply, _ = p.get(1)
    refused("_index_put_impl_ told not to check, with an index past the tensor", reply and reply.get("err"),
            bool(reply) and "out of bounds" in str(reply.get("err")))
    p.put(1, [5.0, 7.0, 6.0])
    if native:  # (a list of tensors in a signature is the C++'s: Python in its place takes none)
        p.define(3, {"op": "aten::_index_put_impl_", "ov": "default", "a": [{"t": 0}, {"tl": [1, 1]}, {"t": 2}, False, True],
                     "kw": {}, "nin": 3, "nout": 1, "nf": 0})
        p.run(3, [1, 5, 4, 1])
        reply, _ = p.get(1)
        refused("the same, as a signature", reply and reply.get("err"), bool(reply) and "out of bounds" in str(reply.get("err")))
        p.put(1, [5.0, 7.0, 6.0])

    # The CPU's kernels are not the GPU's: some write where a tensor of indices says without looking
    # (max_pool2d_with_indices_backward on the CPU, cpu/MaxPoolKernel.cpp). A container can have tensors
    # kept on the Mac's CPU (those it lends for one operation), and must not get an operation run on those alone.
    p.put(10, [1.0, 2.0], cpu=True)
    p.put(11, [10.0, 20.0], cpu=True)
    reply, _ = p.op("aten::add", "Tensor", [{"t": 10}, {"t": 11}], {"alpha": 1})
    refused("an addition of two tensors of the CPU's", reply and reply.get("err"), bool(reply) and "GPU" in str(reply.get("err")))
    p.send({"k": "put", "t": 12, "cpu": True, "dtype": "float32", "shape": [1, 1, 2, 2]}, [struct.pack("<4f", 1, 2, 3, 4)])
    p.send({"k": "put", "t": 13, "cpu": True, "dtype": "float32", "shape": [1, 1, 1, 1]}, [struct.pack("<f", 1)])
    p.send({"k": "put", "t": 14, "cpu": True, "dtype": "int64", "shape": [1, 1, 1, 1]}, [struct.pack("<q", 3)])
    reply, _ = p.op("aten::max_pool2d_with_indices_backward", "default",
                    [{"t": 13}, {"t": 12}, [2, 2], [2, 2], [0, 0], [1, 1], False, {"t": 14}])
    refused("a pooling's gradient on the CPU, where the indices are not checked (given indices in order)",
            reply and reply.get("err"), bool(reply) and "GPU" in str(reply.get("err")))
    p.define(2, {"op": "aten::add", "ov": "Tensor", "a": [{"t": 0}, {"t": 1}], "kw": {}, "nin": 2, "nout": 1, "nf": 0})
    p.run(2, [10, 11, 15])
    error = p.ping()
    refused("an addition of two tensors of the CPU's, as a signature", error, "GPU" in str(error))
    # What a program does with a tensor of the CPU's: gives it to an operator with one of the GPU's.
    p.send({"k": "put", "t": 20, "dtype": "float32", "shape": [2, 2, 1]}, [struct.pack("<4f", 1, 2, 3, 4)])
    p.put(21, [2, 1], cpu=True, dtype="int64", code="q")
    reply, blobs = p.op("aten::_pack_padded_sequence", "default", [{"t": 20}, {"t": 21}, False])
    works("a sequence on the GPU packed by its lengths, which are on the CPU", bool(reply) and not reply.get("err")
          and len(reply["res"]) == 2 and "c" in reply["res"][1], reply)
    reply, _ = p.op("aten::empty", "memory_format", [[2]], {"dtype": {"dtype": "float32"}, "device": {"device": "cpu"}})
    works("an empty tensor made on the CPU, which reads nothing", bool(reply) and not reply.get("err"), reply)
    works("after all that, the connection still computes", p.values(1) == [5.0, 7.0, 6.0])
    p.close()

    # With PYTORCH_ENABLE_MPS_FALLBACK=1 a Mac runs on its CPU what it has no GPU kernel for. Of those, the ones
    # whose kernel does not check what it reads by are not run: segment_reduce ("TODO: add checks").
    p = Peer(door, env={"PYTORCH_ENABLE_MPS_FALLBACK": "1"})
    p.put(1, [1.0, 2.0, 3.0, 4.0])
    p.put(2, [1, 3], dtype="int64", code="q")
    reply, _ = p.op("aten::segment_reduce", "default", [{"t": 1}, "sum"], {"lengths": {"t": 2}})
    refused("segment_reduce, with lengths in order, for a program that asked for the CPU", reply and reply.get("err"),
            bool(reply) and listed in str(reply.get("err")))
    reply, _ = p.op("aten::logit_", "default", [{"t": 1}, 0.25])  # a CPU-only operator that is in the list
    works("an operator a Mac runs on its CPU, for a program that asked for it", bool(reply) and not reply.get("err"), reply)
    p.close()


def signatures(door):
    """C-4: a signature names places in an operation's frame, and only there."""
    p = Peer(door)
    p.put(1, [1.0])
    p.put(326, [5.0])  # 326: the first four bytes of the frame `F 1 ...`, read as a number
    spec = {"op": "aten::neg", "ov": "default", "kw": {}, "nin": 1, "nout": 1, "nf": 0}
    # The tensor is "the third number" of a frame that has two: what follows the frame.
    p.define(1, {**spec, "a": [{"t": 2}]})
    p.run(1, [1, 9], after=struct.pack("<cII", b"F", 1, 77777))
    error = p.ping()
    refused("a tensor's place beyond the frame", error)
    if not error:
        print(f"         the result is {p.values(9)}: the negative of tensor 326, whose number was read past the operation's frame")
    for label, changes in (
        ("a tensor's place among the results", {"a": [{"t": 1}]}),
        ("a negative place", {"a": [{"t": -1}]}),
        ("a place that is not a number", {"a": [{"t": "0"}]}),
        ("a list of tensors that runs past the frame", {"op": "aten::cat", "a": [{"tl": [0, 4096]}, 0]}),
        ("a list of tensors that starts past the frame", {"op": "aten::cat", "a": [{"tl": [7, 1]}, 0]}),
        ("a real number that was not sent", {"op": "aten::mul", "ov": "Scalar", "a": [{"t": 0}, {"f": 3}], "nf": 1}),
        ("real numbers in a signature that has none", {"op": "aten::mul", "ov": "Scalar", "a": [{"t": 0}, {"f": 0}]}),
        ("more results than the operator gives", {"a": [{"t": 0}], "nout": 3}),
        ("fewer results than the operator gives", {"a": [{"t": 0}], "nout": 0}),
        ("a tensor given twice by name", {"op": "aten::add", "ov": "Tensor", "a": [{"t": 0}, {"t": 0}], "kw": {"other": {"t": 0}}}),
    ):
        p.define(2, {**spec, **changes})
        refused(label, p.ping())
    # An operator that gives a list of tensors gives as many as it gives: the container made tensors for a
    # number of them, and with another number the tensors it goes on to use are not the operator's.
    p.put(30, [1.0, 2.0, 3.0])
    unbind = {"op": "aten::unbind", "ov": "int", "a": [{"t": 0}, 0], "kw": {}, "nin": 1, "nf": 0}
    p.define(5, {**unbind, "nout": 3})
    p.run(5, [30, 31, 32, 33])
    works("a tensor of three taken apart into three", p.ping() is None and p.values(33) == [3.0], p.values(33))
    for label, count in (("two", 2), ("four", 4)):
        p.define(6, {**unbind, "nout": count})
        p.run(6, [30] + [40 + i for i in range(count)])
        refused(f"a tensor of three taken apart into {label}, as a signature", p.ping())
        works("  and none of them was kept", p.values(40) is None)
    p.send({"k": "op", "op": "aten::unbind", "ov": "int", "a": [{"t": 30}, 0], "kw": {}, "out": [50, 51]})
    refused("a tensor of three taken apart into two, asked for without waiting", p.ping())
    works("  and none of them was kept", p.values(50) is None)
    # Waited for, its tensors are stored from a number on, as many as the container set numbers aside for.
    p.put(2001, [9.0])
    reply, _ = p.op("aten::unbind", "int", [{"t": 30}, 0], base=2000, room=1)
    refused("three results where numbers were set aside for one", reply and reply.get("err"), p.values(2001) == [9.0])
    reply, _ = p.op("aten::unbind", "int", [{"t": 30}, 0], base=2000, room=None)
    refused("results with no numbers set aside for them", reply and reply.get("err"), p.values(2001) == [9.0])
    reply, _ = p.op("aten::unbind", "int", [{"t": 30}, 0], base=3000, room=3)
    works("three results where numbers were set aside for three", bool(reply) and not reply.get("err")
          and [item["n"] for item in reply["res"]] == [0, 1, 2] and p.values(3002) == [3.0], reply)
    # A signature that was refused is still frames of a known length: what follows them is read.
    p.define(3, {**spec, "a": [{"t": 5}]})
    p.run(3, [1, 9])
    p.define(4, {**spec, "a": [{"t": 0}]})
    p.run(4, [1, 10])
    first = p.ping()
    works("operations after a refused signature keep their place", bool(first) and p.values(10) == [-1.0], first)
    p.close()
    # One whose length cannot be believed leaves nothing after it that can be read: the connection ends,
    # which is that program's own process on the Mac and nobody else's.
    for label, changes in (
        ("a count of tensors that is not a number", {"nin": "1"}),
        ("four thousand million tensors", {"nin": (1 << 32) - 1}),
        ("a signature that is a list", None),
    ):
        p = Peer(door)
        p.put(1, [1.0])
        p.define(2, [1, 2] if changes is None else {**spec, "a": [{"t": 0}], **changes})
        error = p.ping()
        refused(label, error)
        if not error:
            p.run(2, [1, 9])
            print(f"         its operations ran: {p.values(9)}")
        p.close()
    p = Peer(door)
    p.put(1, [2.0])
    p.define(1, {**spec, "a": [{"t": 0}]})
    p.run(1, [1, 2])
    works("after those, a new program is served", p.values(2) == [-2.0])
    p.close()


def door_stays(door):
    """C-5: nothing a connection says first closes the door to the others."""

    def frame(body, kind=b"J", length=None):
        return struct.pack("<cI", kind, len(body) if length is None else length) + body

    hellos = (
        ("a list for a greeting", frame(b"[]")),
        ("a string for a greeting", frame(b'"hello"')),
        ("a number for a greeting", frame(b"1")),
        ("null for a greeting", frame(b"null")),
        ("settings that are a list", frame(b'{"k":"hello","torch":"%s","env":[1,2]}' % TORCH.encode())),
        ("settings that are a number", frame(b'{"k":"hello","torch":"%s","env":7}' % TORCH.encode())),
        ("a version that is a list", frame(b'{"k":"hello","torch":["x"],"env":{}}')),
        ("a program that is a dictionary", frame(b'{"k":"hello","torch":"%s","program":{"a":1},"pid":[1],"env":{}}' % TORCH.encode())),
        ("brackets nested ten thousand deep", frame(b"[" * 10000 + b"]" * 10000)),
        ("bytes that are not text", frame(b"\xff\xfe\x00{")),
        ("a frame that is not a greeting", struct.pack("<cI", b"O", 1) + b"\0" * 8),
        ("a greeting longer than it says it is allowed", frame(b"{}", length=1 << 30)),
        ("half a frame", b"J\x10"),
        ("nothing at all", b""),
    )
    # The other end of another version of the protocol, or of another PyTorch: told so, and not served.
    # Told so that it can be acted on: the message names the container's PyTorch and the Mac's, both,
    # says that the image and the Mac end are built for one version and which one this repository is
    # built and tested for, and no longer sends the user to a server of the container's version,
    # which the repository does not hold.
    built = ("have to be built for the same version", "PyTorch 2.15.0 is the one this repository is built and tested for")
    for label, changes, words in (("a container end of the protocol's first version", {"protocol": None}, ("version",)),
                                  ("a container end of a later version", {"protocol": 99}, ("version",)),
                                  ("a container end with another PyTorch", {"torch": "0.0.1"}, ("PyTorch 0.0.1", f"server {TORCH}", *built)),
                                  ("a container end with a later PyTorch", {"torch": "99.0.0"}, ("PyTorch 99.0.0", f"server {TORCH}", *built))):
        p = Peer(door, greet=False)
        hello = {"k": "hello", "torch": TORCH, "protocol": getattr(wire, "PROTOCOL", None), "pid": 1, "program": "refusals.py", "env": {}}
        hello.update(changes)
        reply, _ = p.ask({key: value for key, value in hello.items() if value is not None})
        said = str(reply.get("err")) if reply else ""
        refused(label, reply and reply.get("err"), bool(reply) and all(word in said for word in words) and "server.sh" not in said)
        if "torch" in changes:  # refused, and not served: the next question finds nobody
            try:
                again, _ = p.ask({"k": "sync"})
            except OSError:  # the Mac end had already closed when the question was written
                again = None
            refused(label + ": it is not served after that", "the Mac end closed the connection" if again is None else None)
        p.close()
    # One that says nothing keeps its place, and nobody else waiting.
    silent = socket.socket(socket.AF_UNIX)
    silent.connect(door.path)
    started = time.time()
    p = Peer(door)
    took = time.time() - started
    case("refusal", "a connection that says nothing: the next one does not wait for it", took < 3 and p.greeting.get("torch") == TORCH,
         f"greeted after {took:.1f} s")
    p.close()
    silent.close()
    for label, data in hellos:
        s = socket.socket(socket.AF_UNIX)
        s.settimeout(15)
        try:
            s.connect(door.path)
            s.sendall(data)
            s.shutdown(socket.SHUT_WR)
            while s.recv(4096):
                pass
        except OSError:
            pass
        finally:
            s.close()
        time.sleep(0.1)
        alive = door.alive()
        answer = None
        if alive:
            try:
                p = Peer(door)
                answer = p.greeting
                p.close()
            except OSError as e:
                answer = e
        ok = alive and isinstance(answer, dict) and answer.get("torch") == TORCH
        case("refusal", f"{label}: the door stays open", ok, "the next program is greeted" if ok else
             ("the server ended: " + door.said().strip().splitlines()[-1] if not alive else f"the next program got {answer!r}"))
        if not alive:
            return False
    return True


def main():
    folder = tempfile.mkdtemp(prefix="varco-refusals-")
    try:
        # Twice: with the Mac end's C++ (executor.cpp), as it runs, and with Python doing its work.
        for executor in (True, False):
            door = Door(folder, executor)
            try:
                how = "the Mac end's C++" if executor else "Python in its place"
                print(f"== operators and arguments (PyTorch {TORCH}, {how})")
                operators(door, folder)
                print(f"== operators that do not check, and the CPU's kernels ({how})")
                unchecked(door, executor)
                print(f"== signatures ({how})")
                signatures(door)
                if executor:
                    print("== the door")
                    door_stays(door)
            finally:
                door.stop()
    finally:
        for name in os.listdir(folder):
            os.unlink(os.path.join(folder, name))
        os.rmdir(folder)
    for kind, what in (("refusal", "refusals"), ("control", "things a program does")):
        of = [ok for k, ok in results if k == kind]
        print(f"{sum(of)} of {len(of)} {what}" + ("" if all(of) else "   <-- NOT ALL"))
    sys.exit(0 if all(ok for _, ok in results) else 1)


if __name__ == "__main__":
    main()
