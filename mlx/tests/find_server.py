"""Where the container end looks for the Mac's server, and what it says when none answers.

    python find_server.py        (in the container, with a server it can reach)

The address is MLX_REMOTE when that is set; without it KIRI_GPU_BRIDGE, which Kiri sets in a
container started with --gpus to the address of its channel to the Mac; without either,
host.docker.internal:47800. And a connection that is closed before the server's greeting
arrives was never one to a server: Kiri accepts a container's connection to any port of
host.docker.internal and closes it at once when nothing listens there on the Mac, which is
what a container started without --gpus finds. The program is then told that no MLX server
answered at the address, and that with Kiri the container must be started with --gpus all,
not that "the MLX server on the Mac closed the connection".

Each case is a process of its own, with the two variables as the case gives them: the address
is read when a process first connects. The listener that closes without a greeting is made
here, in the container, for both ways the program's end can find out (the end of what there
is to read, and a connection reset): nothing on the Mac is changed for it.
"""

import os
import socket
import struct
import subprocess
import sys
import threading

EXPRESSION = "import mlx.core as mx; a = mx.arange(6.0).reshape(2, 3); print((a @ a.T + 1).sum().item(), mx.default_device())"
VALUE = "87.0 Device(gpu, 0)"
failures = []


def check(what, ok, detail=""):
    print(("ok      " if ok else "WRONG   ") + what + (f": {detail}" if detail and not ok else ""), flush=True)
    if not ok:
        failures.append(what)


def closes(reset):
    """A listener on a port of this container that takes every connection and closes it without a word: its address."""
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
    return f"127.0.0.1:{door.getsockname()[1]}"


def run(remote, kiri):
    """The expression in a process with those two variables (None: not set): what it printed, all of it."""
    env = {name: value for name, value in os.environ.items() if name not in ("MLX_REMOTE", "KIRI_GPU_BRIDGE")}
    if remote is not None:
        env["MLX_REMOTE"] = remote
    if kiri is not None:
        env["KIRI_GPU_BRIDGE"] = kiri
    done = subprocess.run([sys.executable, "-c", EXPRESSION], env=env, text=True, capture_output=True, timeout=120)
    return (done.stdout + done.stderr).strip()


def computes(what, remote, kiri):
    said = run(remote, kiri)
    check(what, said.splitlines()[-1:] == [VALUE], said[-400:])


def told(what, remote, kiri, *words, without=()):
    said = run(remote, kiri)
    check(what, all(word in said for word in words) and not any(word in said for word in without), said[-500:])
    return said


def main():
    # The server this container was given: by MLX_REMOTE, by Kiri, or where the image looks by itself.
    server = os.environ.get("MLX_REMOTE") or os.environ.get("KIRI_GPU_BRIDGE") or "host.docker.internal:47800"
    nobody = closes(reset=False)
    computes("as the container was started, it computes on the Mac", os.environ.get("MLX_REMOTE"), os.environ.get("KIRI_GPU_BRIDGE"))
    computes("KIRI_GPU_BRIDGE alone is where the server is looked for", None, server)
    computes("an empty MLX_REMOTE is no address: KIRI_GPU_BRIDGE is used", "", server)
    computes("MLX_REMOTE and KIRI_GPU_BRIDGE: MLX_REMOTE is the one used", server, nobody)
    told("MLX_REMOTE to nobody and KIRI_GPU_BRIDGE to the server: MLX_REMOTE is the one used, and fails", nobody, server,
         f"no MLX server answered at {nobody}")
    for how, reset in (("closed without an answer", False), ("reset at once", True)):
        address = nobody if not reset else closes(reset=True)
        said = told(f"{how}: no MLX server answered, at the address it tried, before the greeting; with Kiri, --gpus all", address, None,
                    f"no MLX server answered at {address}", "before the server's greeting arrived",
                    "With Kiri, the container must be started with --gpus all",
                    without=("closed the connection", "lost while"))
        if not reset:
            print("        " + " ".join(line for line in said.splitlines() if "no MLX server" in line)[-420:], flush=True)
        told(f"{how}, in a container Kiri started with --gpus: told to look for the server on the socket", None, address,
             f"no MLX server answered at {address}", "is the MLX server listening there", without=("must be started",))
    print("all as it should be" if not failures else f"{len(failures)} WRONG")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
