"""A connection that is closed before the server's greeting arrives:   python no_server.py   (in the container)

Kiri accepts a container's connection to any port of host.docker.internal and closes it at once
when nothing listens there on the Mac, which is what a container started without --gpus finds:
it has no bridge. The bridge used to say of that "the bridge lost the Mac: its process for this
program ended (it crashed ...)", of a Mac it had never spoken to. This checks, on the real
bridge, what the program is told instead:

- that no bridge server answered at the address, which it names;
- that with Kiri the container must be started with --gpus all;
- nothing of a process that ended, and the connection is not kept;
- torch.backends.mps.is_available() is False, as when the Mac cannot be reached at all;
- with the server's own address back, the same process connects and computes.

The listener that closes without a greeting is made here, in the container, for both ways the
program's end can find out (the end of what there is to read, and a connection reset): nothing
on the Mac is changed for it. Not a program of the batteries: on a Mac there is no connection.
"""

import os
import socket
import struct
import sys
import threading

import torch

from bridge import install

device = torch.device("mps")
bridge = install.bridge
failures = []


def check(what, ok, detail=""):
    print(("ok      " if ok else "WRONG   ") + what + (f": {detail}" if detail and not ok else ""), flush=True)
    if not ok:
        failures.append(what)


def closes_at_once(reset):
    """A listener on a port of this container that accepts every connection and closes it: its port."""
    door = socket.socket()
    door.bind(("127.0.0.1", 0))
    door.listen(8)

    def serve():
        while True:
            conn, _ = door.accept()
            if reset:  # closed with what the program sent unread: the program's end reads an error
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            else:  # closed in order, once the program has spoken: the program's end reads the end
                conn.recv(1 << 16)
            conn.close()

    threading.Thread(target=serve, daemon=True).start()
    return door.getsockname()[1]


check("the bridge has not spoken to the Mac yet", bridge.sock is None)
given = {name: os.environ.get(name) for name in ("BRIDGE_HOST", "BRIDGE_PORT", "KIRI_GPU_BRIDGE")}
for how, reset in (("closed without an answer", False), ("reset at once", True)):
    port = closes_at_once(reset)
    os.environ.update(BRIDGE_HOST="127.0.0.1", BRIDGE_PORT=str(port))
    os.environ.pop("KIRI_GPU_BRIDGE", None)
    said = None
    try:
        value = float(torch.arange(6.0, device=device).sum())
    except RuntimeError as e:
        said = str(e)
    except OSError as e:
        check(f"{how}: the program is told by the bridge", False, f"{type(e).__name__}: {e}")
        continue
    else:
        check(f"{how}: the program is told", False, f"no error, and {value!r}")
        continue
    check(f"{how}: no bridge server answered, at the address it tried", f"no bridge server answered at 127.0.0.1:{port}" in said, said[:400])
    check(f"{how}: before the server's greeting", "before the server's greeting arrived" in said, said[:400])
    check(f"{how}: with Kiri the container must be started with --gpus all", "With Kiri, the container must be started with --gpus all" in said, said[:400])
    check(f"{how}: nothing of a process of the Mac's that ended", "lost the Mac" not in said and "crashed" not in said, said[:400])
    check(f"{how}: the connection is not kept", bridge.sock is None)
    check(f"{how}: torch.backends.mps.is_available() is False", torch.backends.mps.is_available() is False)
    if not reset:
        print("        " + said[:400], flush=True)
    # A container that Kiri did start with --gpus has the variable: it is told of the server, not of --gpus.
    os.environ["KIRI_GPU_BRIDGE"] = f"127.0.0.1:{port}"
    try:
        float(torch.arange(6.0, device=device).sum())
        said = "no error"
    except RuntimeError as e:
        said = str(e)
    check(f"{how}, in a container Kiri started with --gpus: told to look for the server on the socket",
          "no bridge server answered" in said and "is the bridge's server listening there" in said and "must be started" not in said, said[:400])

for name, value in given.items():
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
total = float(torch.arange(6.0, device=device).sum())
check("with the server's own address the same process connects", bridge.sock is not None)
check("and computes", total == 15.0, total)
print("all as it should be" if not failures else f"{len(failures)} WRONG")
sys.exit(1 if failures else 0)
