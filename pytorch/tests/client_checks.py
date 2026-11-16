"""What the container end does with the Mac's answers, without a Mac:   python client_checks.py   (PyTorch on the CPU)

Three things that must hold whatever the Mac answers, tried on the real client.py and session.py
with only what needs a GPU or a container stood in for (native.cpp's module, the GPU's events):

- a failure on the Mac reaches the program with the next answer it reads, also when that answer
  is to a copy it asked for without waiting, and reading it leaves every later answer with its
  own question;
- an answer that is not the one to the question asked ends the connection, loudly;
- a result's shape is remembered only when it does not depend on the values.
"""

import collections
import io
import itertools
import os
import socket
import struct
import sys
import threading
import types
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ["BRIDGE_COMMIT"] = "0"  # nothing here is on the GPU

import torch  # noqa: E402

# native.cpp's module, as far as these checks go: it records what client.py asks it to remember.
remembered = []
numbers = itertools.count(100, 8)
native = types.ModuleType("bridge_native")
native.fake = lambda shape, stride, dtype, handle=0, offset=0: torch.empty_strided(shape, stride or [1] * len(shape), dtype=dtype, device="meta")
native.handle = lambda tensor: 1
native.write = native.flush = native.caught_up = native.stop = lambda *a: None
native.handles = lambda n: next(numbers)
native.signature = lambda: 1
native.remember = lambda signature, outputs, as_list: remembered.append([list(t.shape) for t in outputs])
sys.modules["bridge_native"] = native

from bridge import client, session, wire  # noqa: E402
from bridge.client import Bridge  # noqa: E402

failures = 0


def check(name, ok, detail=""):
    global failures
    failures += not ok
    print(f"{'ok      ' if ok else 'WRONG   '}{name}" + (f": {detail}" if detail else ""), flush=True)


class Frames:
    """Answers as the Mac would write them."""

    def __init__(self):
        self.data = io.BytesIO()

    def write(self, *pieces):
        for piece in pieces:
            self.data.write(piece)

    def flush(self):
        pass

    def reader(self):
        return io.BytesIO(self.data.getvalue())


def a_bridge(answers):
    """A Bridge that is connected, as far as reading goes, to those answers."""
    bridge = Bridge.__new__(Bridge)
    bridge.lock = threading.RLock()
    bridge.w = Frames()
    bridge.r = answers.reader()
    bridge.sock = object()  # connected
    bridge.pending = collections.deque()
    bridge.asked = bridge.landed = bridge.waits = bridge.questions = 0
    bridge.waited = 0.0
    bridge._drop = lambda: setattr(bridge, "sock", None)
    return bridge


def later(bridge, into):
    """A copy asked for without waiting, as Bridge._later leaves it."""
    numbered = hasattr(bridge, "_number")
    bridge.pending.append((into, bridge._number()) if numbered else into)
    bridge.asked += 1


def value(x):
    return struct.pack("<f", x)


def copies_keep_their_answers():
    """C-7: an answer that reports a failure is still one copy done with."""
    answers = Frames()
    wire.send(answers, {"shape": [1], "dtype": "float32", "err": "aten::add_: RuntimeError: sizes differ", "i": 1}, [value(11)])
    wire.send(answers, {"shape": [1], "dtype": "float32", "err": None, "i": 2}, [value(22)])
    bridge = a_bridge(answers)
    first, second = torch.tensor([-1.0]), torch.tensor([-2.0])
    later(bridge, first)
    later(bridge, second)
    try:
        bridge._land(1)
        raised = None
    except RuntimeError as e:
        raised = str(e)
    check("a copy whose answer reports a failure raises it", raised is not None and "sizes differ" in raised, raised)
    check("and is done with", (bridge.landed, len(bridge.pending)) == (1, 1), f"landed={bridge.landed}, pending={len(bridge.pending)}")
    try:
        bridge._land(2)
    except RuntimeError as e:
        check("the next copy is read", False, str(e))
    check("the next answer goes to the next copy", (first.item(), second.item()) == (11.0, 22.0), f"first={first.tolist()}, second={second.tolist()}")


def a_question_after_a_failed_copy():
    """C-7: a question reads the copies before it, then its own answer, and only then raises what a copy reported."""
    answers = Frames()
    wire.send(answers, {"err": "get: RuntimeError: tensor 7 is not here", "i": 1})
    wire.send(answers, {"v": 41, "err": None, "i": 2})
    wire.send(answers, {"v": 42, "err": None, "i": 3})
    bridge = a_bridge(answers)
    into = torch.tensor([-1.0])
    later(bridge, into)
    try:
        bridge._ask({"k": "call", "fn": "current_allocated_memory", "a": []})
        raised = None
    except RuntimeError as e:
        raised = str(e)
    check("a question after a failed copy raises the copy's failure", raised is not None and "tensor 7" in raised, raised)
    try:
        reply, _ = bridge._ask({"k": "call", "fn": "current_allocated_memory", "a": []})
        check("the question after that gets its own answer", reply.get("v") == 42, reply)
    except RuntimeError as e:
        check("the question after that gets its own answer", False, str(e))


def an_answer_to_another_question():
    """An answer out of step is not taken for a value."""
    if not hasattr(Bridge, "_mine"):
        check("an answer to another question ends the connection", False, "this client does not number its questions")
        return
    answers = Frames()
    wire.send(answers, {"v": 1, "err": None, "i": 5})
    bridge = a_bridge(answers)
    try:
        bridge._ask({"k": "call", "fn": "current_allocated_memory", "a": []})
        raised = None
    except RuntimeError as e:
        raised = str(e)
    check("an answer to another question ends the connection", raised is not None and "lost its place" in raised and bridge.sock is None, raised)


class DoneEvent:
    """The GPU's event, for copies that are of CPU tensors here: done as soon as asked."""

    def __init__(self, *a, **kw):
        pass

    def record(self):
        pass

    def query(self):
        return True

    def synchronize(self):
        pass


def the_mac_reports_a_failure_with_a_copy():
    """C-7, the Mac's side: the real session.serve, on CPU tensors lent to it."""
    ours, theirs = socket.socketpair()
    ours.settimeout(20)
    ended = []

    def serve():
        try:
            ended.append(session.serve(theirs))
        except BaseException as e:  # noqa: BLE001 - shown below
            ended.append(e)
        finally:
            theirs.close()

    frames = Frames()
    for handle, values in ((1, [1.0, 2.0]), (2, [3.0, 4.0, 5.0])):
        wire.send(frames, {"k": "put", "cpu": True, "t": handle, "dtype": "float32", "shape": [len(values)]},
                  [struct.pack(f"<{len(values)}f", *values)])
    wire.send(frames, {"k": "op", "op": "aten::add_", "ov": "Tensor", "a": [{"t": 1}, {"t": 2}], "kw": {}, "out": [1]})  # fails: 2 and 3 elements (and, with no GPU here, tensors of the CPU's alone)
    wire.send(frames, {"k": "get", "t": 1, "later": True, "i": 1})
    wire.send(frames, {"k": "call", "fn": "ping", "a": [], "i": 2})
    with patch.object(torch, "Event", DoneEvent):
        worker = threading.Thread(target=serve, daemon=True)
        worker.start()
        ours.sendall(frames.data.getvalue())
        reading = ours.makefile("rb")
        first, blobs = wire.recv(reading)
        second, _ = wire.recv(reading)
        ours.shutdown(socket.SHUT_RDWR)
        reading.close()
        ours.close()
        worker.join(20)
    check("the answer to a copy asked for without waiting carries the failure before it",
          bool(first and first.get("err")) and "add_" in str(first.get("err")), first)
    check("and carries the question's number", bool(first) and first.get("i") == 1 and bool(second) and second.get("i") == 2,
          f"{first and first.get('i')}, {second and second.get('i')}")
    check("the failure is reported once", bool(second) and second.get("err") is None, second)
    check("the session ended when its program did", len(ended) == 1 and isinstance(ended[0], int), ended)


def shapes_in_the_values_are_not_remembered():
    """C-6: what the Mac answers for unique_consecutive is the shape for those values, not for the next."""
    bridge = Bridge.__new__(Bridge)
    bridge.positions, bridge.signatures, bridge.operators, bridge.in_values = {}, {}, {}, {}
    bridge.w = Frames()
    bridge.sock = object()  # connected
    bridge.slow = None
    bridge.every = False
    first = torch.tensor([1, 1, 2])
    op = torch.ops.aten.unique_consecutive.default
    results = op(first)

    def answer(header, blobs=()):
        return {"seq": True, "res": [{"t": [list(t.shape), str(t.dtype)[6:], list(t.stride()), t.storage_offset()], "n": n}
                                     for n, t in enumerate(results)]}, []

    bridge._ask = answer
    bridge.upload = lambda t: t
    bridge.handle_of = lambda t: 1 if isinstance(t, torch.Tensor) else None
    del remembered[:]
    # As native.cpp hands it over the first time it meets the operation: "routine", to be remembered.
    stand_in = torch.empty(3, dtype=torch.int64, device="meta")
    # (The tensor is on the CPU here and stands for one on the GPU: _on_cpu would send it the way of the mixed ones.)
    with patch.object(client, "_handle", lambda t: 1), patch.object(Bridge, "_on_cpu", staticmethod(lambda *values: False)), \
            patch.object(Bridge, "_prepare", lambda self, x, keep=None: (
                ([stand_in, False, False, None], [{"t": 1}, False, False, None]) if isinstance(x, list) else (x, x))):
        out = bridge._general("aten::unique_consecutive", "", [first, False, False, None], {}, True)
    check("unique_consecutive is answered by the Mac", [list(t.shape) for t in out] == [[2], [0], [0]], [list(t.shape) for t in out])
    check("and its shapes are not remembered", remembered == [], f"asked native.cpp to remember {remembered}")
    tagged = [name for name in ("unique_consecutive", "unique_dim_consecutive", "_unique2", "unique_dim", "bincount", "nonzero", "masked_select")
              if not bridge._in_values(getattr(torch.ops.aten, name).default)] if hasattr(bridge, "_in_values") else ["(this client has no such list)"]
    check("the operators whose shapes are in the values are known as such", not tagged, tagged)
    check("so is repeat_interleave by a tensor", hasattr(bridge, "_in_values") and bridge._in_values(torch.ops.aten.repeat_interleave.Tensor))
    listed = sorted(getattr(client, "SHAPE_FROM_SHAPES", ["(none)"]))
    wrong = []
    for name in listed:
        packet = getattr(torch.ops.aten, name[6:], None)
        overloads = [getattr(packet, overload) for overload in packet.overloads()] if packet is not None else []
        if not overloads or any(torch.Tag.dynamic_output_shape in overload.tags for overload in overloads):
            wrong.append(name)
    check("none of the operators remembered without a shape worked out here is one of those", not wrong, wrong)


def main():
    print(f"PyTorch {torch.__version__}, on the CPU")
    for part in (copies_keep_their_answers, a_question_after_a_failed_copy, an_answer_to_another_question,
                 the_mac_reports_a_failure_with_a_copy, shapes_in_the_values_are_not_remembered):
        try:
            part()
        except Exception as e:  # noqa: BLE001 - a check that cannot run is a check that failed
            check(part.__name__, False, f"{type(e).__name__}: {e}")
    print("all as it should be" if not failures else f"{failures} WRONG")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
