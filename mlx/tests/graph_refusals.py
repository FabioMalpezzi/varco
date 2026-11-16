"""What the Mac server refuses of a container's graphs, and what it tells it.

Speaks the wire protocol itself (mlx/mlx/backend/remote/protocol.h), as a
container that sends whatever it likes. The server must greet only its own
protocol and its own MLX, read no memory of its own beyond an array's, say
that a message failed before it says that what followed is done, keep
nothing of a container once the last one has gone, and stay up throughout.

    python graph_refusals.py PORT

The server at PORT is this test's alone (run-all.sh starts one for it): the
test counts on being its only connection.
"""

import socket
import struct
import sys

from load_refusals import FREE, GRAPH, HELLO, KERNEL, READ, STATS, SYNC, UPLOAD, VERSION, Server, hello, mlx_version
from urgent_refusals import (DONE, EARLY, EVAL, FLOAT32, INT32, IS_INPUT, TELL, UINT8, URGENT, Container, gather, ints, primitive,
                             slice_, spec)

CPU, GPU = 0, 1


def on(device, name, state=b""):
    """A primitive as mlx saves it, on the stream of that device."""
    return struct.pack("<iii", 0, device, 0) + struct.pack("<Q", len(name)) + name.encode() + state


def as_strided(shape, strides, offset):
    """(Its state as mlx saves it: the shape, the strides, 64 bits each, and where it starts.)"""
    return primitive("AsStrided", ints(shape) + struct.pack("<Q", len(strides))
                     + b"".join(struct.pack("<q", s) for s in strides) + struct.pack("<Q", offset))


def greeting(port, payload):
    """What the server answers to a first message: (status, text), or None if it closed."""
    sock = socket.create_connection(("127.0.0.1", port))
    sock.settimeout(30)
    try:
        sock.sendall(struct.pack("<BQ", HELLO, len(payload)) + payload)
        head = b""
        while len(head) < 9:
            piece = sock.recv(9 - len(head))
            if not piece:
                return None
            head += piece
        status, n = struct.unpack("<BQ", head)
        text = b""
        while len(text) < n:
            text += sock.recv(n - len(text))
        return status, text
    finally:
        sock.close()


def counters(c):
    status, text = c.ask(STATS, b"")
    assert status == 0, text
    return text.decode()


def number(text, name):
    return int(text.split(name + "=")[1].split()[0])


def main():
    port = int(sys.argv[1])
    failures = []

    def check(what, ok, detail=""):
        print(("ok      " if ok else "FAILED  ") + what + (f": {detail}" if detail and not ok else ""), flush=True)
        if not ok:
            failures.append(what)

    def alive():
        try:
            return Server(port).ask(STATS, b"")[0] == 0
        except (OSError, AssertionError):
            return False

    # --- the greeting: the same protocol and the same MLX, or nothing
    ours = mlx_version()
    answer = greeting(port, hello())
    check("greeted: this protocol and this MLX", answer is not None and answer[0] == 0
          and answer[1][8:8 + struct.unpack("<I", answer[1][4:8])[0]].decode() == ours, repr(answer))
    for what, payload, word in (
        ("another MLX", hello(mlx="0.0.1"), b"MLX"),
        ("the same MLX with something after it", hello(mlx=ours + ".dev1"), b"MLX"),
        ("no MLX version at all, as protocol 9 greeted", struct.pack("<I", VERSION), b""),
        ("protocol 9", struct.pack("<I", 9), b"protocol"),
        ("a later protocol", hello(protocol=VERSION + 1), b"protocol"),
        ("an MLX version of a megabyte", struct.pack("<II", VERSION, 1 << 20) + b"x" * (1 << 20), b""),
    ):
        answer = greeting(port, payload)
        check(f"not greeted: {what}", answer is not None and answer[0] == 1 and word in answer[1], repr(answer)[:200])
    quiet = Server(port, greet=False)
    quiet.send(UPLOAD, struct.pack("<Q", 1) + spec(UINT8, [1]) + b"x")
    status, text = quiet.ask(STATS, b"")
    check("nothing is taken from a container that has not greeted", status == 1 and b"greeting" in text, text[:200])
    quiet.sock.close()

    # --- what a graph may be
    c = Container(port)
    four = c.upload(FLOAT32, [4], struct.pack("<4f", 1, 2, 3, 4))
    ten = c.upload(FLOAT32, [10], struct.pack("<10f", *range(10)))
    status, _ = c.sync()
    c.seq = 0
    seq, (out,) = c.graph([(four, FLOAT32, [4])], [(primitive("Add"), [IS_INPUT, IS_INPUT], (FLOAT32, [4]))], [0])
    _, done, error, arrays = c.frame()
    check("a sum of two arrays comes back", not error and arrays.get(out) == struct.pack("<4f", 2, 4, 6, 8), error)
    seq, (out,) = c.graph([(ten, FLOAT32, [10])], [(as_strided([3], [2], 1), [IS_INPUT], (FLOAT32, [3]))], [0])
    _, done, error, arrays = c.frame()
    status, got = c.bytes_of(out)
    check("a view by strides inside its array comes back", not error and status == 0 and got == struct.pack("<3f", 1, 3, 5),
          error or repr(got))

    def refused(what, inputs, nodes, word=""):
        """The graph fails, with an error that says so, and leaves no array; the server goes on."""
        seq, (out,) = c.graph(inputs, nodes, [len(nodes) - 1])
        try:
            _, done, error, arrays = c.frame()
        except AssertionError as e:  # the server closed the connection: it ended
            check(f"refused: {what}", False, f"the server ended ({e})")
            return False
        status, got = c.bytes_of(out)
        check(f"refused: {what}", bool(error) and word in error and out not in arrays and status == 1,
              f"error {error!r}, {len(arrays.get(out, b''))} bytes with the answer, read: status {status}, {len(got)} bytes")
        return True

    # A node with fewer arrays than its primitive reads: this ended the server, for every container.
    if not refused("a sum of one array", [(four, FLOAT32, [4])], [(primitive("Add"), [IS_INPUT], (FLOAT32, [4]))], "Add"):
        check("the server is still there", alive())
        print(f"{len(failures)} FAILED")
        return 1
    refused("a selection of two arrays", [(four, FLOAT32, [4])],
            [(primitive("Select"), [IS_INPUT, IS_INPUT], (FLOAT32, [4]))], "Select")
    refused("a node on the cpu's stream", [(four, FLOAT32, [4])],
            [(on(CPU, "Add"), [IS_INPUT, IS_INPUT], (FLOAT32, [4]))], "gpu")
    refused("a primitive the server does not know", [(four, FLOAT32, [4])],
            [(primitive("Bogus"), [IS_INPUT], (FLOAT32, [4]))])
    refused("a primitive with no name", [(four, FLOAT32, [4])], [(struct.pack("<iii", 0, 1, 0), [IS_INPUT], (FLOAT32, [4]))])
    refused("a view by strides past the end of its array", [(ten, FLOAT32, [10])],
            [(as_strided([4096], [1], 0), [IS_INPUT], (FLOAT32, [4096]))], "strides")
    refused("a view by strides that starts past its array", [(ten, FLOAT32, [10])],
            [(as_strided([1], [1], 4096), [IS_INPUT], (FLOAT32, [1]))], "strides")
    refused("a view by strides that goes backwards out of its array", [(ten, FLOAT32, [10])],
            [(as_strided([4], [-1], 2), [IS_INPUT], (FLOAT32, [4]))], "strides")
    refused("an output of a type of the wrong width", [(four, FLOAT32, [4])],
            [(primitive("Add"), [IS_INPUT, IS_INPUT], ((10, 1), [4]))], "dtype")
    refused("an output too large to count", [(four, FLOAT32, [4])],
            [(primitive("Add"), [IS_INPUT, IS_INPUT], (FLOAT32, [2**31 - 1] * 8))], "large")

    # --- a node's state and shapes are those mlx's own operations give it. What runs a primitive is first of
    # all the server's own code (it fills in the strides of a transposition, walks a slice's starts, takes an
    # axis out of a shape), which takes the node on trust: an axis that is not there has it write outside its
    # vectors. So is a shape or a type that the primitive does not make of its inputs: a kernel of the GPU
    # then reads or writes outside its arrays, and an output that shares its input's memory is larger than it.
    def state(name, *parts):
        return primitive(name, b"".join(parts))

    def i32(*values):
        return struct.pack(f"<{len(values)}i", *values)

    table = c.upload(FLOAT32, [2, 3], struct.pack("<6f", *range(6)))
    whole = c.upload(INT32, [4], struct.pack("<4i", 1, 2, 3, 4))
    c.sync()
    F4, T23 = [(four, FLOAT32, [4])], [(table, FLOAT32, [2, 3])]

    def works(what, inputs, nodes, want):
        seq, (out,) = c.graph(inputs, nodes, [len(nodes) - 1])
        _, done, error, arrays = c.frame()
        status, got = c.bytes_of(out)
        check(f"works: {what}", not error and status == 0 and got == want, error or repr(got))

    works("a transposition", T23, [(state("Transpose", ints([1, 0])), [IS_INPUT], (FLOAT32, [3, 2]))], struct.pack("<6f", 0, 3, 1, 4, 2, 5))
    works("a sum along an axis", T23, [(state("Reduce", i32(2), ints([0])), [IS_INPUT], (FLOAT32, [1, 3]))], struct.pack("<3f", 3, 5, 7))
    works("a reshape that leaves one size to be worked out", T23, [(state("Reshape", ints([3, -1])), [IS_INPUT], (FLOAT32, [3, 2]))],
          struct.pack("<6f", *range(6)))
    works("a slice backwards to before the start, as mlx writes x[5::-1]", [(ten, FLOAT32, [10])],
          [(slice_([5], [-11], [-1]), [IS_INPUT], (FLOAT32, [6]))], struct.pack("<6f", 5, 4, 3, 2, 1, 0))
    works("a new axis, and an axis of one taken out", T23,
          [(state("ExpandDims", ints([0])), [IS_INPUT], (FLOAT32, [1, 2, 3])), (state("Squeeze", ints([0])), [0], (FLOAT32, [2, 3]))],
          struct.pack("<6f", *range(6)))
    for what, inputs, node, word in (
        ("a transposition by more axes than the array has", F4, (state("Transpose", ints([0, 0, 0])), [IS_INPUT], (FLOAT32, [4])), "axes"),
        ("a transposition by an axis that is not there", F4, (state("Transpose", ints([5])), [IS_INPUT], (FLOAT32, [4])), "axes"),
        ("a transposition by the same axis twice", T23, (state("Transpose", ints([0, 0])), [IS_INPUT], (FLOAT32, [2, 2])), "axes"),
        ("a transposition declared of another shape", T23, (state("Transpose", ints([1, 0])), [IS_INPUT], (FLOAT32, [2, 3])), "shape"),
        ("a sum along an axis that is not there", F4, (state("Reduce", i32(2), ints([1])), [IS_INPUT], (FLOAT32, [4])), "axes"),
        ("a sum declared of a narrower type than it makes", [(whole, INT32, [4])],
         (state("Reduce", i32(2), ints([0])), [IS_INPUT], (UINT8, [1])), "types"),
        ("the place of the largest along an axis that is not there", F4, (state("ArgReduce", i32(1, 3)), [IS_INPUT], ((3, 4), [4])), "axes"),
        ("a sort along an axis that is not there", F4, (state("Sort", i32(7)), [IS_INPUT], (FLOAT32, [4])), "axes"),
        ("a new axis far past the last", F4, (state("ExpandDims", ints([40])), [IS_INPUT], (FLOAT32, [4, 1])), "axes"),
        ("an axis taken out that is not of one element", F4, (state("Squeeze", ints([0])), [IS_INPUT], (FLOAT32, [])), "axes"),
        ("a concatenation along an axis that is not there", F4, (state("Concatenate", i32(3)), [IS_INPUT, IS_INPUT], (FLOAT32, [8])), "axes"),
        ("a slice with fewer starts than the array has dimensions", T23, (slice_([0], [1], [1]), [IS_INPUT], (FLOAT32, [1, 3])), "axes"),
        ("a slice past the end of its input", [(ten, FLOAT32, [10])], (slice_([8], [20], [1]), [IS_INPUT], (FLOAT32, [12])), "slice"),
        ("a slice that never moves", [(ten, FLOAT32, [10])], (slice_([0], [4], [0]), [IS_INPUT], (FLOAT32, [4])), "axes"),
        ("an array broadcast to fewer dimensions", T23, (state("Broadcast", ints([3])), [IS_INPUT], (FLOAT32, [3])), "shapes"),
        ("padding along an axis that is not there", [(four, FLOAT32, [4]), (four, FLOAT32, [4])],
         (state("Pad", ints([6]), ints([1]), ints([1])), [IS_INPUT, IS_INPUT | 1], (FLOAT32, [6])), "shapes"),
        ("a sum of arrays of different shapes", [(four, FLOAT32, [4]), (ten, FLOAT32, [10])],
         (primitive("Add"), [IS_INPUT, IS_INPUT | 1], (FLOAT32, [10])), "shapes"),
        ("a sum of arrays of different types", [(four, FLOAT32, [4]), (whole, INT32, [4])],
         (primitive("Add"), [IS_INPUT, IS_INPUT | 1], (FLOAT32, [4])), "types"),
        ("a sum declared of another type than its arrays", F4, (primitive("Add"), [IS_INPUT, IS_INPUT], (INT32, [4])), "types"),
        ("a sum declared larger than its arrays", F4, (primitive("Add"), [IS_INPUT, IS_INPUT], (FLOAT32, [4096])), "shape"),
        ("a reshape to more elements", F4, (state("Reshape", ints([8])), [IS_INPUT], (FLOAT32, [8])), "shape"),
        ("a copy declared of a wider type", F4, (primitive("Copy"), [IS_INPUT], ((8, 8), [4])), "types"),
        ("a matrix product of matrices that do not go together", T23 + T23,
         (primitive("Matmul"), [IS_INPUT, IS_INPUT | 1], (FLOAT32, [2, 3])), "shapes"),
    ):
        refused(what, inputs, [node], word)

    # A copy shares its input's memory and keeps the shape the container declares for it. Declared larger, but no
    # larger than the buffer mlx gave its input, it would be sent back with what that buffer held before: mlx
    # hands out buffers it has used, as they were left. Here that is an array of this connection's, freed just before.
    kept = 0
    for _ in range(8):
        marker = c.upload(UINT8, [256], b"\x5a" * 256)
        c.sync()
        c.send(FREE, struct.pack("<IQ", 1, marker))
        small = c.upload(FLOAT32, [4], struct.pack("<4f", 1, 2, 3, 4))
        seq, (out,) = c.graph([(small, FLOAT32, [4])], [(primitive("Copy"), [IS_INPUT], (FLOAT32, [64]))], [0])
        _, done, error, arrays = c.frame()
        status, got = c.bytes_of(out)
        kept = max(kept, (arrays.get(out, b"") + (got if status == 0 else b"")).count(b"\x5a"))
        if not error or status == 0:
            kept = max(kept, 1)
    check("a copy declared as large as its input's buffer is refused, and sends nothing of what the buffer held before", kept == 0,
          f"{kept} bytes of an array freed before came back, or the copy was not refused")
    c.sync()

    # A copy shares its input's memory and keeps the shape the container declares for it: 4096 numbers
    # of a copy of 4. Neither the answer nor a read may carry what lies after those 4.
    seq, (out,) = c.graph([(four, FLOAT32, [4])], [(primitive("Copy"), [IS_INPUT], (FLOAT32, [4096]))], [0])
    _, done, error, arrays = c.frame()
    status, got = c.bytes_of(out)
    leaked = max(len(arrays.get(out, b"")), len(got) if status == 0 else 0)
    check("a copy declared larger than its input sends none of the memory after it", leaked == 0 and status == 1,
          f"{len(arrays.get(out, b''))} bytes came with the answer and {len(got) if status == 0 else 0} when read, "
          "of an array that has 16")
    c.sync()

    # Uploads: a type of its own width, a size that can be counted.
    for what, dtype, shape, data in (
        ("an upload of a type of the wrong width", (10, 1), [4], bytes(4)),
        ("an upload of an unknown type", (99, 4), [1], bytes(4)),
        ("an upload whose size overflows", UINT8, [2**31 - 1] * 8, b""),
        ("an upload with a negative size", UINT8, [-1], b""),
    ):
        handle = c.upload(dtype, shape, data)
        status, text = c.sync()
        check(f"refused: {what}", status == 1, text[:120])
        check("  and it left no array", c.bytes_of(handle)[0] == 1)

    # --- a failure is told before what follows it is said to be done (DONE frames)
    c.seq = 0
    c.send(UPLOAD, struct.pack("<Q", 9999) + spec(FLOAT32, [4]) + b"only three")  # fails: 10 bytes for 16
    seq, (out,) = c.graph([(four, FLOAT32, [4])], [(primitive("Add"), [IS_INPUT, IS_INPUT], (FLOAT32, [4]))], [0])
    kind, done, error, arrays = c.frame()
    check("the answer to a graph carries the failure of the message before it", kind == DONE and "upload" in error,
          f"the frame says graph {done} is done, with the error {error!r}")
    status, text = c.sync()
    check("and the failure is told once", status == 0, text[:200])

    # --- an error says what kind of error it was on the Mac, so that the program in the container can catch
    # what it would catch there: mlx raises an invalid argument (a ValueError in Python) for an array seen
    # with a shape it does not have, and what this server refuses of its own is a runtime error.
    INVALID_ARGUMENT, RUNTIME = 1, 0
    c.seq = 0
    seq, (out,) = c.graph([(four, FLOAT32, [5])], [(primitive("Add"), [IS_INPUT, IS_INPUT], (FLOAT32, [5]))], [0])
    kind, done, error, arrays = c.frame()
    check("an error of mlx's on the Mac is told as the kind of error it was", bool(error) and c.error_kind == INVALID_ARGUMENT,
          f"kind {c.error_kind}, error {error!r}")
    seq, (out,) = c.graph([(four, FLOAT32, [4])], [(primitive("Bogus"), [IS_INPUT], (FLOAT32, [4]))], [0])
    kind, done, error, arrays = c.frame()
    status, text = c.bytes_of(out)
    check("and a refusal of the server's as a runtime error, in a frame and in a reply",
          bool(error) and c.error_kind == RUNTIME and status == 1 and c.error_kind == RUNTIME, f"kind {c.error_kind}, error {error!r}")

    # --- an urgent graph is answered ahead of its turn (an EARLY frame) only when nothing is owed to the
    # container: the frame says "this graph is done", and a container that has only such answers takes
    # everything for done. So not while a failure waits to be told, not ahead of a message that could
    # still fail without a reply, and never to a connection that has not greeted.
    numbers = c.upload(INT32, [10], struct.pack("<10i", *range(10)))
    c.sync()
    c.seq = 0
    take = [(slice_([3], [5], [1]), [IS_INPUT], (INT32, [2]))]
    seq, (out,) = c.graph([(numbers, INT32, [10])], take, [0], EVAL | TELL | URGENT)
    kind, done, error, arrays = c.frame()
    check("an urgent graph with nothing before it is answered early", kind == EARLY and not error
          and arrays.get(out) == struct.pack("<2i", 3, 4), f"frame {kind}, error {error!r}")
    c.send(UPLOAD, struct.pack("<Q", 9998) + spec(FLOAT32, [4]) + b"only three")  # fails: 10 bytes for 16
    seq, (out,) = c.graph([(numbers, INT32, [10])], take, [0], EVAL | TELL | URGENT)
    kind, done, error, arrays = c.frame()
    check("an urgent graph after a message that failed takes its turn, and its answer carries the failure",
          kind == DONE and "upload" in error, f"frame {kind} (4 is EARLY), error {error!r}")
    status, text = c.sync()
    check("and the failure is told once", status == 0, text[:200])
    # (One write: the message and the graph arrive together, whichever the server takes first.)
    seq, _ = 0, None
    bad = struct.pack("<I", 1) + struct.pack("<I", 1) + b"k" + struct.pack("<II", 0, 0)  # a kernel: this server takes none
    message = struct.pack("<BI", EVAL | TELL | URGENT, 1) + struct.pack("<Q", numbers) + spec(INT32, [10])
    c.handle += 1
    message += struct.pack("<I", 1) + struct.pack("<IQ", 0 | 0x80000000, c.handle)
    message += struct.pack("<I", 1) + struct.pack("<I", len(take[0][0])) + take[0][0] + struct.pack("<HI", 1, IS_INPUT)
    message += struct.pack("<H", 1) + spec(INT32, [2])
    c.sock.sendall(struct.pack("<BQ", KERNEL, len(bad)) + bad + struct.pack("<BQ", GRAPH, len(message)) + message)
    kind, done, error, arrays = c.frame()
    check("an urgent graph behind a message that fails without a reply is not answered ahead of it",
          kind == DONE and "kernel" in error, f"frame {kind} (4 is EARLY), error {error!r}")
    status, text = c.sync()
    check("and that failure is told once", status == 0, text[:200])
    quiet = Server(port, greet=False)
    status, text = quiet.ask(HELLO, hello(protocol=VERSION + 1))
    empty = struct.pack("<BIII", EVAL | TELL | URGENT, 0, 0, 0)  # no input, no result, no node
    quiet.send(GRAPH, empty)
    status, n = struct.unpack("<BQ", quiet.read(9))
    body = quiet.read(n)
    error = body[13:12 + struct.unpack_from("<I", body, 8)[0]].decode(errors="replace")
    check("an urgent graph from a container that was not greeted is not answered early", status == DONE and "greeting" in error,
          f"frame {status} (4 is EARLY), error {error!r}")
    quiet.sock.close()

    # --- what is not refused, said here so that the refusals above are not taken for more than they are.
    # MLX does not check an index on the GPU: one past the end of an array reads whatever the GPU has
    # there, as it does for a program on the Mac. This server is one process, so what is there can be
    # another connection's arrays: here, the 64 arrays that the GPU computes for a second connection.
    # A server for each group of containers that trust each other is the only separation there is.
    other = Container(port)
    zero = other.upload(FLOAT32, [4096], bytes(4 * 4096))
    add = primitive("Add")
    for i in range(64):  # each of them 1000 + i throughout
        value = other.upload(FLOAT32, [1], struct.pack("<f", 1000.0 + i))
        other.graph([(zero, FLOAT32, [4096]), (value, FLOAT32, [1])], [(add, [IS_INPUT, IS_INPUT | 1], (FLOAT32, [4096]))], [0],
                    flags=EVAL)
    other.sync()
    small = c.upload(FLOAT32, [8], struct.pack("<8f", *([-1.0] * 8)))
    seq, (mine,) = c.graph([(small, FLOAT32, [8])], [(add, [IS_INPUT, IS_INPUT], (FLOAT32, [8]))], [0], flags=EVAL)
    count = 1 << 20
    far = c.upload(INT32, [count], struct.pack(f"<{count}i", *range(8, 8 + count)))
    c.sync()
    seq, (out,) = c.graph([(mine, FLOAT32, [8]), (far, INT32, [count])],
                          [(gather([0], [1]), [IS_INPUT, IS_INPUT | 1], (FLOAT32, [count, 1]))], [0], flags=EVAL)
    status, got = c.bytes_of(out)
    found = set()
    if status == 0:
        for (value,) in struct.iter_unpack("<f", got):
            if 1000.0 <= value < 1064.0:
                found.add(int(value))
    print(f"known limit: indices past the end of an array, on the GPU, read {len(found)} of the 64 arrays "
          "of another connection to this server", flush=True)
    other.sock.close()

    # --- nothing of a container outlives the last one
    first = counters(c)
    built = number(first, "imports")
    c.sock.close()
    c = Container(port)
    for _ in range(50):  # until the server has seen the first connection close
        if "live_arrays=0 " in counters(c) + " ":
            break
    four = c.upload(FLOAT32, [4], struct.pack("<4f", 1, 2, 3, 4))
    start = counters(c)
    c.seq = 0
    seq, (out,) = c.graph([(four, FLOAT32, [4])], [(primitive("Add"), [IS_INPUT, IS_INPUT], (FLOAT32, [4]))], [0])
    _, done, error, arrays = c.frame()
    again = number(counters(c), "imports")
    check("the first connection had the primitives of its graphs built", built > 0, first)
    check("the next connection, alone on the server, has them built again: none was kept",
          not error and again > 0, f"imports={again} for a graph the server had seen")
    check("and finds no memory in use but its own upload", number(start, "mac_active_memory") <= 16384, start)
    check("the server is still there", alive())
    print("all as it should be" if not failures else f"{len(failures)} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
