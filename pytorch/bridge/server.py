"""Mac end of the bridge: the door. Gives every program that connects a process of its own (session.py).

    python -m bridge.server [PORT | PATH]

A loopback TCP port (47700 if none), or a Unix socket at PATH. The socket is the one to use behind
Kiri's own channel: only the user's processes can open it, where every container can reach a
loopback port. A prototype, not a service.

Why a process each, and not one process for all:
- several programs can use the GPU at once (two containers, or a program and the processes it
  starts), each with its own tensors, generators and settings, as on a Mac;
- a program that ends, however it ends, gives back all its GPU memory, because its process ends;
- a program that takes the GPU down with it (out of memory) takes down its own process only.

A process is kept ready before there is a program for it, so that connecting does not wait for
PyTorch to load. This process itself never loads PyTorch.
"""

import json
import os
import select
import signal
import socket
import subprocess
import sys
import time

from .wire import ENVIRONMENT, HEAD

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# How long a connection has to say who it is, how long what it says may be, and how many may be
# saying it at once: beyond that the oldest is let go.
PATIENCE = 5.0
LONGEST = 1 << 16
WAITING = 64


def start(settings=None):
    """A session process, waiting to be handed a connection: (the process, our end of the socket to it)."""
    ours, theirs = socket.socketpair()
    environment = {k: v for k, v in os.environ.items() if k not in ENVIRONMENT}
    environment.update(settings or {})
    process = subprocess.Popen([sys.executable, "-m", "bridge.session"], stdin=theirs, cwd=HERE, env=environment)
    theirs.close()
    return process, ours


def text(value, longest):
    """What a container says of itself, as text short enough and plain enough to print and to pass on."""
    return "".join(c if c.isprintable() else "?" for c in str(value))[:longest]


def greeting(body):
    """The first frame a container sends, as what the session is told of it: its PyTorch version and MPS settings (wire.py).

    Whatever was sent, what comes out is a dictionary of plain values of known kinds, or a ValueError.
    """
    hello = json.loads(body)
    if not isinstance(hello, dict) or hello.get("k") != "hello":
        raise ValueError("not a greeting")
    given = hello.get("env") or {}
    if not isinstance(given, dict):
        raise ValueError("settings that are not a dictionary")
    settings = {}
    for name in ENVIRONMENT:  # no other variable is taken from a container
        if name in given:
            value = given[name]
            if type(value) not in (str, int, float) or len(str(value)) > 64 or not str(value).isprintable():
                raise ValueError(f"a value for {name} that is not a short text or a number")
            settings[name] = str(value)
    return {"k": "hello", "torch": text(hello.get("torch"), 40), "program": text(hello.get("program") or "", 200),
            "protocol": hello["protocol"] if type(hello.get("protocol")) is int else None,
            "pid": hello["pid"] if type(hello.get("pid")) is int else None, "env": settings}


class Arriving:
    """A connection that has not yet said who it is. It is read as its bytes come, so that one that says
    nothing, or says it slowly, keeps nobody else waiting."""

    def __init__(self, conn):
        conn.setblocking(False)
        self.conn = conn
        self.data = bytearray()
        self.until = time.monotonic() + PATIENCE

    def fileno(self):
        return self.conn.fileno()

    def read(self):
        """Takes what has arrived: the greeting, once it is all here, else None. Raises if it never will be one."""
        want = 5 if len(self.data) < 5 else 5 + HEAD.unpack_from(self.data)[1]
        try:
            chunk = self.conn.recv(want - len(self.data))
        except BlockingIOError:
            return None
        if not chunk:
            # A connection that sent nothing is somebody checking the port: not worth a line in the log.
            raise EOFError("" if not self.data else "it closed in the middle of its greeting")
        self.data += chunk
        if len(self.data) < 5:
            return None
        kind, length = HEAD.unpack_from(self.data)
        if kind != b"J" or length > LONGEST:
            raise ValueError("not a greeting")
        if len(self.data) < 5 + length:
            return None
        return greeting(bytes(self.data[5:]))


def main():
    where = sys.argv[1] if len(sys.argv) > 1 else "47700"
    path = None
    if where.isdigit():
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", int(where)))
        where = f"127.0.0.1:{where}"
    else:
        if os.path.exists(where):
            os.unlink(where)
        listener = socket.socket(socket.AF_UNIX)
        listener.bind(where)
        os.chmod(where, 0o600)
        path = where
    listener.listen(16)
    running = []  # the session processes that have a program
    spare = start()

    def stop(*_):
        for process in running + [spare[0]]:
            if process.poll() is None:
                process.terminate()
        if path and os.path.exists(path):
            os.unlink(path)
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    version = subprocess.run([sys.executable, "-c", "import importlib.metadata as m; print(m.version('torch'))"],
                             capture_output=True, text=True).stdout.strip()
    print(f"bridge server: torch {version}, MPS on {where}, a process for each program", flush=True)
    handing = {}  # our end of the socket to a session -> the connection it is being handed
    arriving = []  # the connections that have not yet said who they are, oldest first

    def turn_away(one, why):
        arriving.remove(one)
        one.conn.close()
        if why:
            print(f"bridge server: a connection that did not introduce itself: {why}", flush=True)

    try:
        while True:
            ready, _, _ = select.select([listener, *handing, *arriving], [], [], 1.0)
            running = [p for p in running if p.poll() is None]  # a session that has ended is let go of at once
            for one in [a for a in arriving if a.until < time.monotonic()]:
                turn_away(one, "it said nothing for too long")
            for item in ready:
                if item is listener:
                    try:
                        conn, _ = listener.accept()
                    except OSError:
                        continue
                    if len(arriving) >= WAITING:
                        turn_away(arriving[0], "too many are waiting to")
                    arriving.append(Arriving(conn))
                    continue
                if item in handing:
                    # The session has the connection (it says so with a byte), or has died: only now may our
                    # copies be closed. On macOS a descriptor still on its way is lost if its sender closes first.
                    handing.pop(item).close()
                    item.close()
                    continue
                if item not in arriving:  # turned away a moment ago
                    continue
                # Whatever a connection says first, and however wrong, is that connection's business alone:
                # nothing it sends may end this process, which is every program's way in.
                try:
                    hello = item.read()
                    if hello is None:
                        continue
                    arriving.remove(item)
                    conn = item.conn
                    conn.setblocking(True)  # the session's, from here on
                    settings = hello["env"]
                    if settings or spare[0].poll() is not None:  # MPS settings are read when PyTorch loads: a process started with them
                        process, control = start(settings)
                    else:
                        (process, control), spare = spare, start()
                    try:
                        socket.send_fds(control, [json.dumps(hello).encode()], [conn.fileno()])
                        handing[control] = conn
                    except OSError as e:
                        print(f"bridge server: could not hand a connection over: {e}", flush=True)
                        control.close()
                        conn.close()
                    running = [p for p in running if p.poll() is None] + [process]
                except Exception as e:  # noqa: BLE001 - see above
                    if item in arriving:
                        turn_away(item, "" if isinstance(e, EOFError) and not str(e) else f"{type(e).__name__}: {str(e)[:200]}")
                    else:
                        print(f"bridge server: could not start a program's process: {type(e).__name__}: {str(e)[:200]}", flush=True)
                        item.conn.close()
    finally:
        # However this ends, the process kept ready goes with it, and the socket's name: the programs
        # that are running keep their own processes until they end.
        if spare[0].poll() is None:
            spare[0].terminate()
        listener.close()
        if path and os.path.exists(path):
            os.unlink(path)


if __name__ == "__main__":
    main()
