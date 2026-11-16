"""After the bridge gave a connection up, the program goes on with a new one:   python reconnect.py   (in the container)

The bridge closes its connection to the Mac when an answer is not the one to the question it
asked (client.py, _mine): every value read after that would be another question's. What the
program had on the GPU goes with the connection, and the Mac's process for it ends. This checks
what comes after, on the real bridge:

- a tensor of the closed connection says so when it is used, read or saved, and is never taken
  for a tensor of the new connection (their numbers start again on the Mac);
- operations the bridge had learnt on the closed connection are learnt again on the new one:
  the Mac's new process was never told their signatures;
- a copy asked for without waiting, on the closed connection, is not waited for on the new one;
- the new connection computes, with the right values.

The answer out of step is made here, by sending a question and not reading its answer: nothing
on the Mac is changed for it. Not a program of the batteries: on a Mac there is no connection to lose.
"""

import io
import sys

import torch

from bridge import install

device = torch.device("mps")
bridge = install.bridge
failures = []


def check(what, ok, detail=""):
    print(("ok      " if ok else "WRONG   ") + what + (f": {detail}" if detail and not ok else ""), flush=True)
    if not ok:
        failures.append(what)


def refused(what, action, word):
    try:
        value = action()
    except RuntimeError as e:
        check(what, word in str(e), str(e)[:200])
    else:
        check(what, False, f"no error, and {value!r:.100}")


def learnt(x):
    """Operations with the same shapes each time: after the first they are a few integers, and no Python."""
    y = x
    for _ in range(4):
        y = (y * 2.0 + x).tanh()
    return y


old = torch.arange(6, dtype=torch.float32, device=device)
before = [learnt(old).cpu().tolist() for _ in range(3)]
check("before: the same operations give the same values each time", before[0] == before[1] == before[2])
copied = torch.empty(6)
copied.copy_(old * 3, non_blocking=True)  # asked for without waiting
event = torch.Event(device=device)
event.record()

bridge._send({"k": "sync", "i": 1 << 40})  # a question whose answer nobody reads: the next question will be given it
bridge.w.flush()
refused("an answer out of step closes the connection, and says so", lambda: float(old.sum()), "lost its place")
check("the connection is closed", bridge.sock is None)

refused("a tensor of the closed connection, read", lambda: old.cpu(), "has been closed")
refused("a tensor of the closed connection, computed with", lambda: float((old + 1).sum()), "has been closed")
refused("a tensor of the closed connection, saved", lambda: torch.save(old, io.BytesIO()), "has been closed")
refused("a copy asked of the closed connection is not waited for on another", event.synchronize, "lost the Mac")
check("and its event is not taken for done", event.query() is False)

new = torch.arange(6, dtype=torch.float32, device=device)
check("the next operation opens a new connection", bridge.sock is not None)
after = [learnt(new).cpu().tolist() for _ in range(3)]
check("after: the operations learnt before are learnt again, and give the values they gave",
      after[0] == after[1] == after[2] == before[0], f"{after[0]} against {before[0]}")
refused("a tensor of the closed connection next to one of the new", lambda: float((old + new).sum()), "has been closed")
buffer = io.BytesIO()
torch.save({"new": new, "view": new[2:]}, buffer)
buffer.seek(0)
check("the new connection saves and loads", torch.load(buffer)["view"].cpu().tolist() == [2.0, 3.0, 4.0, 5.0])
again = torch.empty(6)
again.copy_(new * 3, non_blocking=True)
event = torch.Event(device=device)
event.record()
event.synchronize()
check("and copies asked of it without waiting arrive", again.tolist() == [0.0, 3.0, 6.0, 9.0, 12.0, 15.0], again.tolist())
print("all as it should be" if not failures else f"{len(failures)} WRONG")
sys.exit(1 if failures else 0)
