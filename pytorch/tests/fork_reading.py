"""A process made by fork() while a thread of its parent is waiting for the Mac:   python fork_reading.py   (in the container)

The child inherits the parent's connection and must leave it alone: it closes its own copy and
opens a connection of its own when it first needs the GPU (client.py, _forked). The parent's
thread was reading an answer when the child was made, so in the child the reader of that
connection is locked by a thread that does not exist: the child must not wait for it. On its
own connection the child computes, reads back and saves; it ends at once or as a program
does, and either way the parent's connection and tensors are as they were.

Not a program of the batteries, which compare with the Mac: on a Mac a process made by fork()
cannot use the GPU its parent used at all.
"""

import os
import sys
import threading
import time

import torch

from bridge import install

device = torch.device("mps")
x = torch.ones(3000, 3000, device=device)
answer = []


def wait_for_the_mac():
    y = x
    for _ in range(200):  # seconds of work for the GPU, sent without waiting
        y = (y @ x) / 3000
    answer.append(float(y[0, 0]))  # waits for all of it: the thread sits in the read of the answer


thread = threading.Thread(target=wait_for_the_mac)
thread.start()
# The bridge's lock is held for as long as a question is out, which is nearly all of it the read of
# its answer: the process is forked at a moment the thread holds it.
lock = install.bridge.lock
while thread.is_alive():
    if not lock.acquire(blocking=False):
        break
    lock.release()
    time.sleep(0.001)
else:
    sys.exit("the thread was never found waiting for the Mac: nothing was checked")


def in_child():
    """What a child does with the GPU: 0 if all of it went as it should."""
    value = float(torch.full((3,), 2.0, device=device).sum())
    if value != 6.0:
        return 2
    # Its own tensors, on its own connection: computed with, read back, saved and loaded.
    mine = torch.arange(6, dtype=torch.float32, device=device)
    for _ in range(3):  # (the same operations again: the second time they are the few integers the bridge has learnt)
        total = float((mine * 2 + 1).sum())
        if total != 36.0:
            return 4
    import io
    buffer = io.BytesIO()
    torch.save({"mine": mine, "view": mine[2:]}, buffer)
    buffer.seek(0)
    if torch.load(buffer)["view"].cpu().tolist() != [2.0, 3.0, 4.0, 5.0]:
        return 5
    # The parent's tensor is not this process's to use.
    try:
        float(x[0, 0])
        return 6
    except RuntimeError as e:
        if "fork" not in str(e):
            return 7
    return 0


def ended(child):
    """The child's exit status, or None if it never ended (and was killed)."""
    deadline = time.time() + 60
    while time.time() < deadline:
        done, status = os.waitpid(child, os.WNOHANG)
        if done:
            return os.waitstatus_to_exitcode(status)
        time.sleep(0.1)
    os.kill(child, 9)
    os.waitpid(child, 0)
    return None


# Twice: a child that leaves at once (os._exit), as a worker of multiprocessing does, and one that
# ends as a program does (sys.exit: the interpreter is shut down, the bridge's exit handlers run and
# every tensor the child inherited is let go of). Neither may touch the parent's connection.
statuses = []
for leave in (os._exit, sys.exit):
    child = os.fork()
    if child == 0:
        try:
            code = in_child()
        except BaseException:  # noqa: BLE001 - only the exit code leaves this process
            code = 3
        sys.stdout.flush()
        leave(code)
    statuses.append(ended(child))
thread.join()
# After both children have gone, the parent's connection and tensors are as they were.
still = float((x[0, :3] * 2).sum())
ok = statuses == [0, 0] and answer == [1.0] and still == 6.0
print(f"{'ok' if ok else 'WRONG'}: the child that left at once " + ("never ended (it was killed after 60 s)" if statuses[0] is None else
      f"ended with status {statuses[0]}") + ", the child that ended as a program does " + ("never ended" if statuses[1] is None else
      f"ended with status {statuses[1]}") + f"; the parent's thread read {answer}, and the parent then computed {still}")
sys.exit(0 if ok else 1)
