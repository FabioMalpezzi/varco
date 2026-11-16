"""What the Mac server reads for a container, and what it refuses.

Speaks the wire protocol itself (mlx/mlx/backend/remote/protocol.h), as a
container that asks for whatever it likes: the server must only ever read
regular files under the directories it was started with.

    python load_refusals.py PORT DIR CONTAINER_DIR

DIR is the tree made by `load_files.py make`; the server was started with
--map CONTAINER_DIR/allowed=DIR/allowed and nothing else.
"""

import os
import socket
import struct
import sys

HELLO, UPLOAD, GRAPH, OP, FREE, READ, SYNC, STATS, OPEN, LOAD, CLOSE, KERNEL = range(1, 13)
VERSION = 11
UINT8, UINT32 = (1, 1), (3, 4)
HEAD, SAMPLE, SAMPLES = 64 << 10, 4 << 10, 16


def digest(data):
    """What OPEN says of a file's contents (file_digest in protocol.h): its first and last bytes and
    sixteen pieces in between, or all of it when it is no longer than its first and last together."""
    size = len(data)
    if size <= 2 * HEAD:
        parts = [data]
    else:
        between = size - 2 * HEAD
        parts = [data[:HEAD]]
        for k in range(1, SAMPLES + 1):
            at = HEAD + between // (SAMPLES + 1) * k
            parts.append(data[at:at + SAMPLE])
        parts.append(data[size - HEAD:])
    h = 0xCBF29CE484222325
    for part in parts:
        for byte in part:
            h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def mlx_version():
    """The MLX of this Python: the server's, when the tests are run as run-all.sh runs them."""
    import mlx.core as mx

    return mx.__version__


def hello(protocol=VERSION, mlx=None):
    """The greeting's payload: the protocol's version, and this end's MLX version as text."""
    mlx = mlx_version() if mlx is None else mlx
    return struct.pack("<II", protocol, len(mlx)) + mlx.encode()


class Server:
    def __init__(self, port, greet=True):
        self.sock = socket.create_connection(("127.0.0.1", port))
        self.sock.settimeout(120)
        self.handle = 0
        self.file = 0
        if greet:
            status, text = self.ask(HELLO, hello())
            assert status == 0, text
            # What the server says it does (bit 0: it takes custom kernels), and its MLX version.
            self.features, length = struct.unpack("<II", text[:8])
            self.mlx = text[8:8 + length].decode()

    def send(self, kind, payload):
        self.sock.sendall(struct.pack("<BQ", kind, len(payload)) + payload)

    def ask(self, kind, payload):
        """A message and its reply: (status, payload). The payload of an error is its text: the byte before it,
        which says what kind of error it was on the Mac (protocol.h, ErrorKind), is kept in self.error_kind.
        (The reply to a greeting is text alone.)"""
        self.send(kind, payload)
        head = self.read(9)
        status, n = struct.unpack("<BQ", head)
        answer = self.read(n)
        if status == 1 and kind != HELLO and answer:
            self.error_kind, answer = answer[0], answer[1:]
        return status, answer

    def read(self, n):
        data = b""
        while len(data) < n:
            piece = self.sock.recv(n - len(data))
            assert piece, "the server closed the connection"
            data += piece
        return data

    def open(self, name, size, mtime, dig):
        """(file id or 0, the reason given)."""
        self.file += 1
        status, answer = self.ask(
            OPEN, struct.pack("<IQqQ", self.file, size, mtime, dig) + name.encode())
        assert status == 0, answer
        return (self.file if answer[:1] == b"\x01" else 0), answer[1:].decode()

    def load(self, file, offset, shape, dtype=UINT8, swap=0):
        self.handle += 1
        self.send(LOAD, struct.pack("<QIBI", self.handle, dtype[0], dtype[1], len(shape))
                  + b"".join(struct.pack("<i", d) for d in shape)
                  + struct.pack("<IQB", file, offset, swap))
        return self.handle

    def bytes_of(self, handle):
        return self.ask(READ, struct.pack("<Q", handle))

    def sync(self):
        return self.ask(SYNC, struct.pack("<Q", 0))


def main():
    port, root, container = int(sys.argv[1]), os.path.abspath(sys.argv[2]), sys.argv[3]
    good = os.path.join(root, "allowed/w.safetensors")
    data = open(good, "rb").read()
    size, time, dig = len(data), os.stat(good).st_mtime_ns, digest(data)
    outside = open(os.path.join(root, "outside/s.safetensors"), "rb").read()
    osize, otime, odig = len(outside), os.stat(os.path.join(root, "outside/s.safetensors")).st_mtime_ns, digest(outside)
    s = Server(port)
    failed = []

    def check(what, ok, detail=""):
        print(f"  {'ok    ' if ok else 'FAILED'} {what}{': ' + detail if detail and not ok else ''}")
        if not ok:
            failed.append(what)

    # Names that must be refused, with the size, time and digest of the file
    # they would reach, so that only where the file is can be the reason.
    refused = [
        ("a file outside the allowed directories", f"{container}/outside/s.safetensors", osize, otime, odig),
        ("a symlink that leaves them", f"{container}/allowed/link-out.safetensors", osize, otime, odig),
        ("'..' that leaves them", f"{container}/allowed/../outside/s.safetensors", osize, otime, odig),
        ("'..' deeper", f"{container}/allowed/sub/../../outside/s.safetensors", osize, otime, odig),
        ("the same file by its path on the Mac", os.path.join(root, "outside/s.safetensors"), osize, otime, odig),
        ("an allowed file by its path on the Mac, which no mapping covers", good, size, time, dig),
        ("a file of the system", "/etc/passwd", os.path.getsize("/etc/passwd"), os.stat("/etc/passwd").st_mtime_ns,
         digest(open("/etc/passwd", "rb").read())),
        ("a missing file", f"{container}/allowed/missing.safetensors", size, time, dig),
        ("a symlink to a missing file outside", f"{container}/allowed/link-nowhere.safetensors", osize, otime, odig),
        ("a directory", f"{container}/allowed/sub", 0, os.stat(os.path.join(root, "allowed/sub")).st_mtime_ns, digest(b"")),
        ("a named pipe", f"{container}/allowed/fifo", 0, os.stat(os.path.join(root, "allowed/fifo")).st_mtime_ns, digest(b"")),
        ("a relative path", f"{container.lstrip('/')}/allowed/w.safetensors", size, time, dig),
        ("a name with a zero byte", f"{container}/allowed/w.safetensors\0x", size, time, dig),
        ("an empty name", "", size, time, dig),
        ("the right file with another size", f"{container}/allowed/w.safetensors", size - 1, time, dig),
        ("the right file with other contents", f"{container}/allowed/w.safetensors", size, time, dig ^ 1),
        ("the right file, last modified at another time", f"{container}/allowed/w.safetensors", size, time + 1_000_000_000, dig),
    ]
    reasons = {}
    for what, name, n, t, d in refused:
        file, why = s.open(name, n, t, d)
        reasons[what] = why
        check(f"refused: {what}", file == 0 and why != "", f"opened as {file}")
    # A container cannot tell a file that exists outside from one that does not.
    check("the same answer for an existing and a missing file outside",
          reasons["a symlink that leaves them"] == reasons["a symlink to a missing file outside"]
          == reasons["a missing file"])

    file, why = s.open(f"{container}/allowed/w.safetensors", size, time, dig)
    check("opened: an allowed file", file != 0, why)
    link, why = s.open(f"{container}/allowed/link-in.safetensors", size, time, dig)
    check("opened: a symlink that stays inside", link != 0, why)
    file2, why = s.open(f"{container}/allowed/w.safetensors", size, time // 1_000_000_000 * 1_000_000_000, dig)
    check("opened: an allowed file seen through a file system that keeps its time to the second", file2 != 0, why)

    # Another file under the same name: what the Mac finds must be the container's file, not one
    # that only starts like it. Made here, a megabyte of weights and three files that pass for it.
    mine = bytes((i * 131 + (i >> 8) * 7) & 0xFF for i in range(1 << 20))
    there = os.path.join(root, "allowed", "identity.bin")

    def put(contents, mtime=None):
        with open(there, "wb") as f:
            f.write(contents)
        if mtime is not None:
            os.utime(there, ns=(mtime, mtime))
        return os.stat(there).st_mtime_ns

    when = put(mine)
    name = f"{container}/allowed/identity.bin"
    file3, why = s.open(name, len(mine), when, digest(mine))
    check("opened: the container's own file", file3 != 0, why)
    s.send(CLOSE, struct.pack("<I", file3))
    # The same model trained further: the same size, the same header (the first 64 KiB), the same time.
    put(mine[:HEAD] + bytes(b ^ 0x5A for b in mine[HEAD:]), when)
    got, why = s.open(name, len(mine), when, digest(mine))
    check("refused: a file of the same size, start and time whose weights are others", got == 0 and "contents" in why, why or f"opened as {got}")
    # The same bytes, written at another time.
    put(mine, when + 5_000_000_000)
    got, why = s.open(name, len(mine), when, digest(mine))
    check("refused: the same bytes in a file written at another time", got == 0 and "time" in why, why or f"opened as {got}")
    # What the check does not see, said here so that it is not taken for more than it is: one byte,
    # where the digest does not look, in a file with the same time.
    odd = bytearray(mine)
    odd[HEAD + 100] ^= 1
    put(bytes(odd), when)
    got, why = s.open(name, len(mine), when, digest(mine))
    print(f"  known limit: a file that differs by one byte the digest does not read, with the same time, is "
          f"{'taken for the same' if got else 'refused: ' + why}")
    if got:
        s.send(CLOSE, struct.pack("<I", got))
    # A file that changes after it was opened is not read.
    when = put(mine)
    file4, why = s.open(name, len(mine), when, digest(mine))
    check("opened: the file again", file4 != 0, why)
    put(bytes(b ^ 0xFF for b in mine), when + 7_000_000_000)
    h = s.load(file4, HEAD, [64])
    status, text = s.sync()
    check("load refused: a file that was changed after it was opened", status == 1 and b"changed" in text, text.decode(errors="replace"))
    status, _ = s.bytes_of(h)
    check("  and it left no array", status == 1)
    os.unlink(there)

    h = s.load(file, 100, [50])
    status, got = s.bytes_of(h)
    check("the bytes are those of the file", status == 0 and got == data[100:150])
    h = s.load(file, size - 7, [7])
    status, got = s.bytes_of(h)
    check("up to the last byte", status == 0 and got == data[-7:])
    h = s.load(link, 8, [3, 2], UINT32, swap=1)
    status, got = s.bytes_of(h)
    want = b"".join(data[8 + 4 * i:12 + 4 * i][::-1] for i in range(6))
    check("with the byte order swapped", status == 0 and got == want)
    h = s.load(file, 0, [0])
    status, got = s.bytes_of(h)
    check("an empty array", status == 0 and got == b"")

    for what, args in (
        ("past the end of the file", (file, size - 7, [8])),
        ("an offset past the end", (file, size + 1, [0])),
        ("a shape whose size overflows", (file, 0, [2**31 - 1] * 8)),
        ("a huge offset", (file, 2**64 - 4, [8])),
        ("a file that was refused", (file + 100, 0, [8])),
    ):
        h = s.load(*args)
        status, text = s.sync()
        check(f"load refused: {what}", status == 1, text.decode(errors="replace"))
        status, text = s.bytes_of(h)
        check(f"  and it left no array", status == 1)

    s.send(CLOSE, struct.pack("<I", file))
    h = s.load(file, 0, [8])
    status, text = s.sync()
    check("load refused: a closed file", status == 1)
    print("all refusals and reads as expected" if not failed else f"{len(failed)} FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
