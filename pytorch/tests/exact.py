"""What a battery keeps of each program, in full: for compat.py and compat_train.py.

A program's number (PASS name number) is one figure to read: two different
tensors can give the same one. What compat_compare.py compares is what this
module keeps: for every tensor a program hands to num() or note(), its type,
its shape and a SHA-256 of all its bytes, and the program's number bit for
bit. They are printed on the line after PASS:

    DATA name {"value": "0x1.8p+1", "tensors": [["torch.float32", [3, 4], "9f86d0..."], ...]}

With DUMP=directory in the environment the tensors themselves are written there as well
(name.NNN.npy, in the order of the DATA line), so that scripts/tensor-diff.py can say how far
two runs that differ are from each other: how many elements, and by how much.
"""

import hashlib
import json
import os
import sys
import threading

import torch

DUMP = os.environ.get("DUMP")  # a directory: the tensors themselves are kept there too, for scripts/tensor-diff.py
_main = []  # what the program noted, in order
_threads = {}  # thread -> what it noted, in order, for the threads the program started


def note(t):
    """Keeps a tensor's type, its shape and a digest of all its bytes in row-major order. Returns the tensor."""
    whole = t.detach().cpu().contiguous()
    flat = whole.reshape(-1)
    raw = flat.view(torch.uint8).numpy().tobytes() if flat.numel() else b""
    entry = [str(whole.dtype), list(whole.shape), hashlib.sha256(raw).hexdigest()]
    thread = threading.current_thread()
    # A thread the program started with threading runs beside the others, and which of them
    # gets to the GPU first changes from run to run: its tensors are kept apart. Any other
    # thread that is not the main one is PyTorch's own (a backward pass can call a hook on one,
    # and does on a Mac) at work for the program while it waits: its tensors are the program's.
    apart = thread is not threading.main_thread() and not isinstance(thread, threading._DummyThread)
    (_threads.setdefault(thread, []) if apart else _main).append((entry, whole if DUMP else None))
    return t


def begin():
    """Before a program runs: nothing is noted yet."""
    _main.clear()
    _threads.clear()


def data(value, name=None):
    """What the program that just ran gave, for its DATA line: its number bit for bit, and the tensors it noted.

    The program's own tensors come first, in the order it noted them. Those of the threads it
    started follow thread by thread, the threads ordered by what they noted (see note).
    """
    noted = _main + [item for thread in sorted(_threads.values(), key=lambda items: [e for e, _ in items]) for item in thread]
    if DUMP and name:
        import numpy as np
        os.makedirs(DUMP, exist_ok=True)
        for i, (_, whole) in enumerate(noted):
            # numpy has no bfloat16: as float32, which holds every bfloat16 exactly.
            np.save(os.path.join(DUMP, f"{name}.{i:03d}.npy"), (whole.float() if whole.dtype == torch.bfloat16 else whole).numpy())
    return json.dumps({"value": float(value).hex(), "tensors": [entry for entry, _ in noted]}, separators=(",", ":"))


def info(battery, device):
    """The INFO line: which battery this is and what ran it. compat_compare.py reads from it which
    inventory the run is to be held against, and refuses two files that do not belong together."""
    only = os.environ.get("ONLY")
    return (f"INFO battery={battery} torch={torch.__version__.split('+')[0]} device={device.type} platform={sys.platform}"
            + (f" only={only}" if only else ""))
