"""A program that holds GPU memory and then ends in the way it is told to: for lifetime.sh.

    python lifetime.py clean|exception|exit|hang [GIGABYTES]

Whatever the ending, the Mac must have the memory back: the bridge gives each program a process
of its own on the Mac (bridge/server.py), which ends when the program's connection does.
"""
import os
import sys
import time

import torch

how, gigabytes = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4
held = [torch.ones(256, 1024, 1024, device="mps") for _ in range(gigabytes)]  # 1 GB each
total = sum(float(t[0, 0, :8].sum()) for t in held)  # the GPU has made them
torch.mps.synchronize()
print(f"ready {torch.mps.current_allocated_memory() >> 20} MB allocated, {torch.mps.driver_allocated_memory() >> 20} MB driver, sum {total}", flush=True)
if how == "clean":
    sys.exit(0)
if how == "exception":
    raise RuntimeError("the program fails")
if how == "exit":
    os._exit(3)  # no cleanup of any kind
if how == "busy":  # killed while the GPU is at work for it
    while True:
        held[0] = held[0] @ held[0][0]
time.sleep(600)  # hang: killed from outside
