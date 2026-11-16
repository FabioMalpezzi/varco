"""What the Mac server takes of custom kernels, and what it refuses.

Speaks the wire protocol itself (mlx/mlx/backend/remote/protocol.h), as a
container that sends whatever it likes.

    python kernel_refusals.py PORT on|off

on: the server was started with --custom-kernels; off: without.
"""

import struct
import sys

from load_refusals import GRAPH, KERNEL, READ, STATS, SYNC, UPLOAD, Server

FLOAT32, INT32 = (10, 4), (7, 4)
EVAL, HAS_KERNELS = 1, 4
CALL, IS_INPUT = 0x40000000, 0x80000000
CALL_KERNEL = 3
MAX_SOURCE, MAX_NAME, MAX_ARRAYS, MAX_TEMPLATE, MAX_KERNELS = 256 << 10, 128, 64, 64, 256
SOURCE = "uint i = thread_position_in_grid.x; o[i] = a[i] + 1.0f;"


def text(s, length="<I"):
    s = s.encode() if isinstance(s, str) else s
    return struct.pack(length, len(s)) + s


def spec(dtype, shape):
    return struct.pack("<IBI", dtype[0], dtype[1], len(shape)) + b"".join(struct.pack("<i", d) for d in shape)


def kernel(name="add", inputs=("a",), outputs=("o",), source=SOURCE, header="", contiguous=1, atomic=0,
           options=0, tail=b""):
    """A KERNEL message after the number."""
    return (text(name) + struct.pack("<I", len(inputs)) + b"".join(text(n) for n in inputs)
            + struct.pack("<I", len(outputs)) + b"".join(text(n) for n in outputs)
            + text(source) + text(header) + struct.pack("<BBi", contiguous, atomic, options) + tail)


def call(number, grid=(4, 1, 1), threadgroup=(4, 1, 1), template=b"", n_template=0, tail=b""):
    """The node that runs a kernel."""
    return (struct.pack("<II6iBfH", CALL_KERNEL, number, *grid, *threadgroup, 0, 0.0, n_template)
            + template + tail)


class Container(Server):
    def upload(self, values):
        self.handle += 1
        self.send(UPLOAD, struct.pack("<Q", self.handle) + spec(FLOAT32, [len(values)])
                  + struct.pack(f"<{len(values)}f", *values))
        return self.handle

    def add(self, number, message):
        self.send(KERNEL, struct.pack("<I", number) + message)

    def error(self):
        """What the messages sent so far left to say, or ''."""
        status, answer = self.ask(SYNC, struct.pack("<Q", 0))
        return answer.decode(errors="replace") if status else ""

    def run(self, node, inputs, n=4, flags=EVAL | HAS_KERNELS, kind=CALL):
        """A graph of one node on the arrays `inputs`; its output, or the error."""
        self.handle += 1
        out = self.handle
        graph = struct.pack("<BI", flags, len(inputs))
        for h in inputs:
            graph += struct.pack("<Q", h) + spec(FLOAT32, [n])
        graph += struct.pack("<IIQ", 1, 0, out)
        graph += struct.pack("<II", 1, len(node) | kind) + node
        graph += struct.pack("<H", len(inputs)) + b"".join(struct.pack("<I", IS_INPUT | i) for i in range(len(inputs)))
        graph += struct.pack("<H", 1) + spec(FLOAT32, [n])
        self.send(GRAPH, graph)
        why = self.error()
        if why:
            return why
        status, data = self.ask(READ, struct.pack("<Q", out))
        return list(struct.unpack(f"<{n}f", data)) if status == 0 else data.decode(errors="replace")


def main():
    port, mode = int(sys.argv[1]), sys.argv[2]
    failed = []

    def check(what, ok, detail=""):
        print(f"  {'ok    ' if ok else 'FAILED'} {what}{': ' + str(detail) if detail and not ok else ''}")
        if not ok:
            failed.append(what)

    def refused(what, got, saying=""):
        check(f"refused: {what}", isinstance(got, str) and got != "" and saying in got, got)

    s = Container(port)
    a = s.upload([1.0, 2.0, 3.0, 4.0])
    # A saved CustomKernel primitive, source and all, is never taken.
    refused("a saved CustomKernel primitive in a graph", s.run(b"\0" * 8 + text("CustomKernel", "<Q"), [a], kind=0),
            "saved CustomKernel")

    if mode == "off":
        check("the server says it takes no kernels", s.features & 1 == 0)
        s.add(1, kernel())
        refused("a kernel", s.error(), "without --custom-kernels")
        refused("a node that runs it", s.run(call(1), [a]), "without --custom-kernels")
        check("and the connection goes on", s.error() == "")
        print("all as expected" if not failed else f"{len(failed)} FAILED")
        return 1 if failed else 0

    check("the server says it takes kernels", s.features & 1 == 1)
    s.add(1, kernel())
    check("a kernel is taken", s.error() == "")
    check("and runs", s.run(call(1), [a]) == [2.0, 3.0, 4.0, 5.0])
    check("a second time, from the cached program", s.run(call(1), [a]) == [2.0, 3.0, 4.0, 5.0])

    # The same numbers and the same graph on another connection, another kernel.
    other = Container(port)
    b = other.upload([1.0, 2.0, 3.0, 4.0])
    other.add(1, kernel(source=SOURCE.replace("1.0f", "2.0f")))
    check("the same graph of another connection runs that connection's kernel",
          other.run(call(1), [b]) == [3.0, 4.0, 5.0, 6.0], other.error())
    check("and the first connection still runs its own", s.run(call(1), [a]) == [2.0, 3.0, 4.0, 5.0])
    third = Container(port)
    c = third.upload([1.0, 2.0, 3.0, 4.0])
    refused("a kernel of another connection, by its number", third.run(call(1), [c]), "was not sent on this connection")
    third.add(7, kernel())
    check("the same kernel sent by a third connection, under another number",
          third.run(call(7), [c]) == [2.0, 3.0, 4.0, 5.0])

    # Nodes.
    refused("a kernel in a graph that does not say it has one", s.run(call(1), [a], flags=EVAL), "does not say")
    refused("a kernel that was never sent", s.run(call(99), [a]), "was not sent")
    refused("the wrong number of inputs", s.run(call(1), [a, a]), "it takes 1 arrays")
    for what, node in (
        ("a grid of size zero", call(1, grid=(0, 1, 1))),
        ("a negative grid", call(1, grid=(4, -1, 1))),
        ("a threadgroup of size zero", call(1, threadgroup=(4, 0, 1))),
        ("a threadgroup whose size overflows", call(1, threadgroup=(65536, 65536, 1))),
        ("a template argument of an unknown kind", call(1, template=b"\x09" + text("T", "<H") + b"\0\0\0\0", n_template=1)),
        ("a template argument that is not a name", call(1, template=b"\x00" + text("a b", "<H") + b"\1\0\0\0", n_template=1)),
        ("a template type that does not exist", call(1, template=b"\x02" + text("T", "<H") + struct.pack("<IB", 99, 4), n_template=1)),
        ("too many template arguments",
         call(1, template=(b"\x00" + text("N", "<H") + b"\1\0\0\0") * (MAX_TEMPLATE + 1),
              n_template=MAX_TEMPLATE + 1)),
        ("bytes after the node", call(1, tail=b"\0")),
        ("a node cut short", call(1)[:-3]),
    ):
        refused(what, s.run(node, [a]))
    check("and the kernel still runs", s.run(call(1), [a]) == [2.0, 3.0, 4.0, 5.0])

    # Kernels.
    n = 1
    for what, message in (
        ("a name that is not an identifier", kernel(name="not a name")),
        ("an empty name", kernel(name="")),
        ("a name too long", kernel(name="k" * (MAX_NAME + 1))),
        ("an input that is not an identifier", kernel(inputs=("a b",))),
        ("an output that starts with a digit", kernel(outputs=("1o",))),
        ("too many inputs", kernel(inputs=[f"a{i}" for i in range(MAX_ARRAYS + 1)])),
        ("no output", kernel(outputs=())),
        ("a source too large", kernel(source="/" * (MAX_SOURCE + 1))),
        ("a source and a header too large together", kernel(source="/" * MAX_SOURCE, header="//")),
        ("unknown compile options", kernel(options=7)),
        ("bytes after the kernel", kernel(tail=b"\0")),
        ("a kernel cut short", kernel()[:-2]),
        ("#include", kernel(header='#include "/etc/passwd"\n')),
        ("#include in the source", kernel(source='\n#include "/etc/passwd"\n' + SOURCE)),
        ("#include spelled %:include", kernel(header='%:include "/etc/passwd"\n')),
        ("#include over two lines", kernel(header='#inc\\\nlude "/etc/passwd"\n')),
        ("#include over two lines, with blanks", kernel(header='#inc\\  \r\nlude "/etc/passwd"\n')),
        ("#include with a comment inside", kernel(header='# /* a\n b */ include "/etc/passwd"\n')),
        ("#include_next", kernel(header='#include_next "/etc/passwd"\n')),
        ("#import", kernel(header='#import "/etc/passwd"\n')),
        ("__has_include", kernel(header='#if __has_include("/etc/passwd")\n#endif\n')),
        ("#pragma", kernel(header='#pragma GCC dependency "/etc/passwd"\n')),
        ("_Pragma", kernel(header='_Pragma("GCC dependency \\"/etc/passwd\\"")\n')),
        ("the word after a #define", kernel(header="#define A 1\n// nothing to include\n")),
    ):
        n += 1
        s.add(n, message)
        refused(what, s.error())
        got = s.run(call(n), [a])
        refused("  a node that runs it", got, "refused" if "cut short" not in what else "")
    s.add(0, kernel())
    refused("the number zero", s.error())
    s.add(1, kernel(source=SOURCE.replace("1.0f", "3.0f")))
    refused("a number used before", s.error())
    check("  whose kernel is the first one still", s.run(call(1), [a]) == [2.0, 3.0, 4.0, 5.0])

    # What is taken.
    n += 1
    s.add(n, kernel(name="defines", header="#define ONE 1.0f\n#if 1\n#define TWO 2.0f\n#endif\n",
                    source="uint i = thread_position_in_grid.x; o[i] = a[i] * TWO + ONE;"))
    check("taken: #define and #if", s.error() == "" and s.run(call(n), [a]) == [3.0, 5.0, 7.0, 9.0])
    n += 1
    s.add(n, kernel(name="wordy", source="// does not include the bias, nor import it\n" + SOURCE))
    check("taken: the words without a directive", s.error() == "" and s.run(call(n), [a]) == [2.0, 3.0, 4.0, 5.0])
    n += 1
    s.add(n, kernel(name="largest", source=SOURCE + "/" * (MAX_SOURCE - len(SOURCE) - 2) + "\n", header="\n"))
    check("taken: a source of the largest size", s.error() == "" and s.run(call(n), [a]) == [2.0, 3.0, 4.0, 5.0])

    # As many as a connection may send, and one more.
    many = Container(port)
    d = many.upload([1.0, 2.0, 3.0, 4.0])
    for i in range(1, MAX_KERNELS + 2):
        many.add(i, kernel(name=f"k{i}", source=SOURCE.replace("1.0f", f"{i}.0f")))
    refused(f"kernel number {MAX_KERNELS + 1} of a connection", many.error(), "kernels already")
    check(f"  number {MAX_KERNELS} runs", many.run(call(MAX_KERNELS), [d]) == [1.0 + MAX_KERNELS + i for i in range(4)])
    refused(f"  number {MAX_KERNELS + 1} does not", many.run(call(MAX_KERNELS + 1), [d]), "kernels already")

    # A message larger than any kernel may be closes the connection.
    big = Container(port)
    try:
        big.send(KERNEL, struct.pack("<I", 1) + kernel(source="/" * (MAX_SOURCE + (128 << 10))))
        closed = big.sock.recv(1) == b""
    except OSError:
        closed = True
    check("a message too large for a kernel: the connection is closed", closed)
    check("and the others go on", s.run(call(1), [a]) == [2.0, 3.0, 4.0, 5.0])

    # --- nothing of a container's kernels outlives the containers that sent them. The server keeps so many
    # kernels at a time (4096) and refuses more: containers that have gone must not count among them, or a
    # server that has run long enough refuses every program that has a kernel, with nobody connected.
    def counters(connection):
        status, answer = connection.ask(STATS, b"")
        text = answer.decode(errors="replace")
        return {name: int(text.split(name + "=")[1].split()[0]) if name + "=" in text else None
                for name in ("server_kernels", "kernel_libraries")}

    def alone():
        """A new connection, once the server has seen every other one close."""
        connection = Container(port)
        for _ in range(200):
            if counters(connection)["server_kernels"] in (0, None):
                break
        return connection

    for connection in (s, other, third, many, big):
        connection.sock.close()
    s = alone()
    check("with every container gone the server keeps no kernel, and no library mlx compiled of one",
          counters(s) == {"server_kernels": 0, "kernel_libraries": 0}, counters(s))
    s.sock.close()
    total = 0
    for round_ in range(4096 // MAX_KERNELS + 1):  # more kernels than the server keeps at a time, from containers that leave
        visitor = Container(port)
        for i in range(1, MAX_KERNELS + 1):
            visitor.add(i, kernel(name=f"visitor_{round_}_{i}"))
        why = visitor.error()
        total += MAX_KERNELS
        visitor.sock.close()
        if why:
            break
    check(f"{total} kernels from containers that came and went are all taken", why == "", why[:200])
    s = alone()
    a = s.upload([1.0, 2.0, 3.0, 4.0])
    s.add(1, kernel())
    check("and the next container's kernel is taken, and runs", s.error() == "" and s.run(call(1), [a]) == [2.0, 3.0, 4.0, 5.0],
          s.error())
    check("it is the only one the server keeps, with its one library", counters(s) == {"server_kernels": 1, "kernel_libraries": 1},
          counters(s))
    print("all as expected" if not failed else f"{len(failed)} FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
