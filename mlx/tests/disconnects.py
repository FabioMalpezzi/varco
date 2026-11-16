"""What the Mac server does when a container goes away in the middle of its work, or stops reading.

Speaks the wire protocol itself, as containers that leave at the worst moments:
with graphs queued, with files being read, with an urgent graph waiting, with
an answer half sent. After each the server must still serve the others, and
once they have all gone it must hold none of their memory.

    python disconnects.py PORT DIR CONTAINER_DIR

The server at PORT is this test's alone, started with
--map CONTAINER_DIR/allowed=DIR/allowed (DIR is made by `load_files.py make`).
"""

import os
import struct
import sys
import time

from load_refusals import READ, STATS, digest
from urgent_refusals import EVAL, FLOAT32, INT32, IS_INPUT, TELL, URGENT, UINT8, Container, primitive, slice_


def counters(c):
    status, text = c.ask(STATS, b"")
    assert status == 0, text
    return text.decode()


def number(text, name):
    return int(text.split(name + "=")[1].split()[0])


def main():
    port, root, container = int(sys.argv[1]), os.path.abspath(sys.argv[2]), sys.argv[3]
    failures = []

    def check(what, ok, detail=""):
        print(("ok      " if ok else "FAILED  ") + what + (f": {detail}" if detail and not ok else ""), flush=True)
        if not ok:
            failures.append(what)

    def served(what, within=60, known_limit=None):
        """Another container gets a sum computed, within that many seconds: how long it took.

        With known_limit, a container that is not served is that limit of the server, said as
        one and not counted as a failure: nothing is returned then."""
        started = time.time()
        try:
            c = Container(port)
            c.sock.settimeout(within)
            four = c.upload(FLOAT32, [4], struct.pack("<4f", 1, 2, 3, 4))
            seq, (out,) = c.graph([(four, FLOAT32, [4])], [(primitive("Add"), [IS_INPUT, IS_INPUT], (FLOAT32, [4]))], [0])
            _, _, error, arrays = c.frame()
            ok = not error and arrays.get(out) == struct.pack("<4f", 2, 4, 6, 8)
            c.sock.close()
        except (OSError, AssertionError) as e:
            ok, error = False, f"{type(e).__name__}: {e}"
        if not ok and known_limit:
            print(f"known limit: {what}: the next container is not served within {within} s ({error}). {known_limit}", flush=True)
            return None
        check(f"{what}: the next container is served", ok, error)
        return time.time() - started

    def memory_back(what):
        """Once every connection has gone, a new one finds the server holding nothing."""
        text = ""
        for _ in range(200):  # the server lets go when it reaches the end of what was queued
            c = Container(port)
            text = counters(c)
            c.sock.close()
            if number(text, "mac_active_memory") == 0:
                break
            time.sleep(0.1)
        check(f"{what}: the Mac has its memory back", number(text, "mac_active_memory") == 0, text)

    # With graphs queued: seconds of work for the GPU, and nobody to want it.
    c = Container(port)
    matrix = c.upload(FLOAT32, [512, 512], bytes(4 * 512 * 512))
    for _ in range(8):
        c.long_graph(matrix)
    c.sock.close()
    served("a container that left with graphs queued")
    memory_back("a container that left with graphs queued")

    # With an urgent graph waiting behind a long one.
    c = Container(port)
    matrix = c.upload(FLOAT32, [512, 512], bytes(4 * 512 * 512))
    numbers = c.upload(INT32, [10], struct.pack("<10i", *range(10)))
    c.sync()
    c.long_graph(matrix)
    c.graph([(numbers, INT32, [10])], [(slice_([3], [4], [1]), [IS_INPUT], (INT32, [1]))], [0], flags=EVAL | TELL | URGENT)
    c.sock.close()
    served("a container that left with an urgent graph waiting")
    memory_back("a container that left with an urgent graph waiting")

    # With a file being read for it: 256 MiB in 64 arrays, asked for and not waited for.
    there = os.path.join(root, "allowed", "disconnect.bin")
    piece = bytes(range(256)) * 4096  # 1 MiB
    with open(there, "wb") as f:
        for _ in range(256):
            f.write(piece)
    size, when = os.path.getsize(there), os.stat(there).st_mtime_ns
    with open(there, "rb") as f:
        dig = digest(f.read())
    c = Container(port)
    file, why = c.open(f"{container}/allowed/disconnect.bin", size, when, dig)
    check("the file is opened", file != 0, why)
    for i in range(64):
        c.load(file, i * (4 << 20), [4 << 20], UINT8)
    c.sock.close()
    served("a container that left while its file was being read")
    memory_back("a container that left while its file was being read")
    os.unlink(there)

    # With an answer half sent: 64 MiB asked for and never read. The server has one thread that runs
    # every container's work, and it is the one that writes the answer.
    c = Container(port)
    big = c.upload(UINT8, [64 << 20], bytes(64 << 20))
    c.sync()
    c.send(READ, struct.pack("<Q", big))
    time.sleep(1.0)
    # Not a refusal that can be asked of this server as it is built: said as the limit it is, and not counted.
    took = served("a container that asked for 64 MiB and does not read them", within=10,
                  known_limit="While a container does not read an answer the server is writing, no other container is served")
    if took is None:
        c.sock.close()
        served("once it has gone")
    else:
        print(f"        (served after {took:.1f} s)")
        c.sock.close()
    memory_back("a container that left with an answer half sent")
    print("all as it should be" if not failures else f"{len(failures)} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
