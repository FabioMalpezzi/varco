"""What the Mac server answers ahead of its queue, and what it leaves in its turn.

Speaks the wire protocol itself (mlx/mlx/backend/remote/protocol.h), as a
container that marks urgent whatever it likes. An urgent graph is answered
early, by the cpu, only when that gives what the GPU would give and reads no
memory but that of its inputs: anything else takes its turn behind the long
graph sent before it, and nothing a container sends ends the server.

    python urgent_refusals.py PORT
"""

import struct
import sys

from load_refusals import GRAPH, STATS, SYNC, UPLOAD, Server

BOOL, UINT8, UINT32, INT32, FLOAT32 = (0, 1), (1, 1), (3, 4), (7, 4), (10, 4)
EVAL, TELL, URGENT = 1, 2, 8
IS_INPUT, WITH_BYTES = 0x80000000, 0x80000000
DONE, EARLY = 2, 4


def spec(dtype, shape):
    return struct.pack("<IBI", dtype[0], dtype[1], len(shape)) + b"".join(struct.pack("<i", d) for d in shape)


def ints(values):
    """A vector of integers as mlx saves it."""
    return struct.pack("<Q", len(values)) + b"".join(struct.pack("<i", v) for v in values)


def primitive(name, state=b""):
    """A primitive of the gpu as mlx saves it: stream, name, state."""
    return struct.pack("<iii", 0, 1, 0) + struct.pack("<Q", len(name)) + name.encode() + state


def slice_(start, end, strides):
    return primitive("Slice", ints(start) + ints(end) + ints(strides))


def gather(axes, slice_sizes):
    return primitive("Gather", ints(axes) + ints(slice_sizes))


class Container(Server):
    def __init__(self, port):
        super().__init__(port)
        self.seq = 0

    def upload(self, dtype, shape, data):
        self.handle += 1
        self.send(UPLOAD, struct.pack("<Q", self.handle) + spec(dtype, shape) + data)
        return self.handle

    def graph(self, inputs, nodes, keep, flags=EVAL | TELL):
        """inputs: (handle, dtype, shape); nodes: (primitive, values, (dtype, shape));
        keep: values, whose bytes come back. Returns the handles of the kept."""
        message = struct.pack("<BI", flags, len(inputs))
        for handle, dtype, shape in inputs:
            message += struct.pack("<Q", handle) + spec(dtype, shape)
        handles = []
        message += struct.pack("<I", len(keep))
        for value in keep:
            self.handle += 1
            handles.append(self.handle)
            message += struct.pack("<IQ", value | WITH_BYTES, self.handle)
        message += struct.pack("<I", len(nodes))
        for saved, values, (dtype, shape) in nodes:
            message += struct.pack("<I", len(saved)) + saved
            message += struct.pack("<H", len(values)) + b"".join(struct.pack("<I", v) for v in values)
            message += struct.pack("<H", 1) + spec(dtype, shape)
        self.send(GRAPH, message)
        self.seq += 1
        return self.seq, handles

    def frame(self):
        """The next frame the server sends by itself: (status, seq, error, {handle: bytes})."""
        status, n = struct.unpack("<BQ", self.read(9))
        body = self.read(n)
        assert status in (DONE, EARLY), (status, body[:200])
        seq, error_len = struct.unpack_from("<QI", body)
        # (An error is a byte for its kind, kept in self.error_kind, and its text.)
        self.error_kind = body[12] if error_len else None
        error = body[13:12 + error_len].decode(errors="replace")
        at = 12 + error_len
        count = struct.unpack_from("<I", body, at)[0] & 0x7FFFFFFF
        at += 4
        arrays = {}
        for _ in range(count):
            handle, nbytes = struct.unpack_from("<QQ", body, at)
            arrays[handle] = body[at + 16:at + 16 + nbytes]
            at += 16 + nbytes
        return status, seq, error, arrays

    def long_graph(self, matrix, n=400):
        """Work for the GPU that takes a while: a chain of matrix products."""
        matmul = primitive("Matmul")
        shape = (FLOAT32, [512, 512])
        nodes = [(matmul, [IS_INPUT, IS_INPUT], shape)]
        nodes += [(matmul, [i, IS_INPUT], shape) for i in range(n - 1)]
        # Only its corner comes back.
        nodes.append((slice_([0, 0], [1, 1], [1, 1]), [n - 1], (FLOAT32, [1, 1])))
        return self.graph([(matrix, FLOAT32, [512, 512])], nodes, [n])


def main():
    port = int(sys.argv[1])
    c = Container(port)
    failures = []

    def check(what, ok, detail=""):
        print(("ok      " if ok else "FAILED  ") + what + (f": {detail}" if detail and not ok else ""))
        if not ok:
            failures.append(what)

    numbers = c.upload(INT32, [10], struct.pack("<10i", *range(10)))
    floats = c.upload(FLOAT32, [4], struct.pack("<4f", 1.5, 2.5, 3.5, 4.5))
    matrix = c.upload(FLOAT32, [512, 512], bytes(4 * 512 * 512))
    seven = c.upload(UINT32, [1], struct.pack("<I", 7))
    last = c.upload(INT32, [1], struct.pack("<i", -1))
    far = c.upload(UINT32, [1], struct.pack("<I", 1000))
    before = c.upload(INT32, [1], struct.pack("<i", -11))
    three = c.upload(INT32, [10], struct.pack("<10i", *([3] * 10)))
    status, _ = c.sync()
    assert status == 0
    c.seq = 0

    N, I32, U32, F32 = [(numbers, INT32, [10])], INT32, UINT32, FLOAT32

    def behind_long_work(what, inputs, nodes, keep, early, want=None, flags=EVAL | TELL | URGENT):
        """Send long work, then the graph: which is answered first, and with what."""
        long_seq, _ = c.long_graph(matrix)
        seq, handles = c.graph(inputs, nodes, keep, flags)
        first = c.frame()
        second = c.frame()
        if early:
            ok = first[0] == EARLY and first[1] == seq and second[0] == DONE and second[1] == long_seq
            answer = first
        else:
            ok = first[0] == DONE and first[1] == long_seq and second[0] == DONE and second[1] == seq
            answer = second
        detail = f"frames {[(f[0], f[1], f[2][:80]) for f in (first, second)]}"
        if ok and want is not None:
            got = answer[3].get(handles[0])
            ok = got == want
            detail = f"got {got!r}, wanted {want!r}"
        check(("answered early: " if early else "takes its turn: ") + what, ok, detail)

    take = [(numbers, I32, [10]), (seven, U32, [1])]
    one = (I32, [1])
    # What the cpu answers.
    behind_long_work("a slice of an array", N, [(slice_([3], [5], [1]), [IS_INPUT], (I32, [2]))], [0], True,
                     struct.pack("<2i", 3, 4))
    behind_long_work("a slice backwards", N, [(slice_([9], [5], [-2]), [IS_INPUT], (I32, [2]))], [0], True,
                     struct.pack("<2i", 9, 7))
    behind_long_work("the element at an index", take,
                     [(gather([0], [1]), [IS_INPUT, IS_INPUT | 1], (I32, [1, 1]))], [0], True, struct.pack("<i", 7))
    behind_long_work("the element at a negative index", [(numbers, I32, [10]), (last, I32, [1])],
                     [(gather([0], [1]), [IS_INPUT, IS_INPUT | 1], (I32, [1, 1]))], [0], True, struct.pack("<i", 9))
    behind_long_work("a comparison of integers", [(numbers, I32, [10]), (three, I32, [10])],
                     [(primitive("Equal", b"\0"), [IS_INPUT, IS_INPUT | 1], (BOOL, [10]))], [0], True,
                     bytes(int(i == 3) for i in range(10)))
    behind_long_work("an element of floating point, moved and not computed", [(floats, F32, [4])],
                     [(slice_([2], [3], [1]), [IS_INPUT], (F32, [1]))], [0], True, struct.pack("<f", 3.5))
    # What it leaves to the GPU, in its turn.
    behind_long_work("an addition of floating point", [(floats, F32, [4])],
                     [(primitive("Add"), [IS_INPUT, IS_INPUT], (F32, [4]))], [0], False,
                     struct.pack("<4f", 3.0, 5.0, 7.0, 9.0))
    behind_long_work("a comparison of floating point", [(floats, F32, [4])],
                     [(primitive("Equal", b"\0"), [IS_INPUT, IS_INPUT], (BOOL, [4]))], [0], False, b"\1\1\1\1")
    behind_long_work("a graph that is not marked urgent", N, [(slice_([3], [5], [1]), [IS_INPUT], (I32, [2]))], [0],
                     False, struct.pack("<2i", 3, 4), flags=EVAL | TELL)
    # What would read memory that is not of its inputs: never on the cpu.
    behind_long_work("an index past the end", [(numbers, I32, [10]), (far, U32, [1])],
                     [(gather([0], [1]), [IS_INPUT, IS_INPUT | 1], (I32, [1, 1]))], [0], False)
    behind_long_work("an index before the start", [(numbers, I32, [10]), (before, I32, [1])],
                     [(gather([0], [1]), [IS_INPUT, IS_INPUT | 1], (I32, [1, 1]))], [0], False)
    behind_long_work("a slice past the end of its input", N, [(slice_([8], [20], [1]), [IS_INPUT], (I32, [12]))], [0],
                     False)
    behind_long_work("a slice that says it is longer than it is", N,
                     [(slice_([3], [5], [1]), [IS_INPUT], (I32, [6]))], [0], False)
    behind_long_work("a gather of slices larger than the input", take,
                     [(gather([0], [4]), [IS_INPUT, IS_INPUT | 1], (I32, [1, 4]))], [0], False)
    behind_long_work("a gather along an axis that is not there", take,
                     [(gather([3], [1]), [IS_INPUT, IS_INPUT | 1], (I32, [1, 1]))], [0], False)
    behind_long_work("a gather that says its elements are smaller", take,
                     [(gather([0], [1]), [IS_INPUT, IS_INPUT | 1], (UINT8, [1, 1]))], [0], False)
    behind_long_work("a reshape to more elements", N,
                     [(primitive("Reshape", ints([20])), [IS_INPUT], (I32, [20]))], [0], False)
    behind_long_work("a comparison of arrays of different types", [(numbers, I32, [10]), (seven, U32, [1])],
                     [(primitive("Equal", b"\0"), [IS_INPUT, IS_INPUT | 1], (BOOL, [10]))], [0], False)
    # (Not here: a node with fewer inputs than its primitive takes. The cpu
    # refuses it, but in its turn it ends the server, as it did before there
    # were urgent graphs.)
    # An input that the long work is still computing: its turn comes after.
    long_seq, kept = c.long_graph(matrix)
    seq, handles = c.graph([(kept[0], F32, [1, 1])], [(slice_([0, 0], [1, 1], [1, 1]), [IS_INPUT], (F32, [1, 1]))],
                           [0], EVAL | TELL | URGENT)
    first, second = c.frame(), c.frame()
    check("takes its turn: a value of what is still being computed",
          first[:2] == (DONE, long_seq) and second[:2] == (DONE, seq) and second[3].get(handles[0]) == bytes(4),
          f"frames {[(f[0], f[1], f[2][:80]) for f in (first, second)]}")

    # The server is still there, and counts what it did.
    status, answer = c.sync()
    check("the server still answers", status == 0, repr(answer[:200]))
    status, text = c.ask(STATS, b"")
    text = text.decode()
    counts = text[text.index("("):text.index(")") + 1] if "(" in text else text[:80]
    check("it counts 18 urgent graphs, 6 of them answered early", "(urgent 18, early 6)" in text, counts)
    print(f"{len(failures)} of the checks failed" if failures else "all the checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
