"""Times the first training steps of a program that trains with PyTorch Lightning, as it is.

The program is run from its own directory, with every Lightning Trainer it makes changed to
stop after a number of steps, skip validation and checkpoints, and time each step (waiting for
the GPU at both ends). Used to compare PyTorch versions on the Mac, and the Mac with a container.
    cd PROJECT && python step_timer.py --steps 30 --out DIR TRAIN.py [the program's own options]
Prints one line per step and a summary, also written to DIR/summary.json.

This is the harness the field reports of docs/benchmarks.md were timed with; there it named the
private project's training program, here the program is an argument.
"""
import argparse
import json
import os
import runpy
import statistics
import sys
import time

import lightning.pytorch as pl
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--warmup", type=int, default=5, help="steps left out of the figures")
parser.add_argument("--out", required=True)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("program")
parser.add_argument("rest", nargs=argparse.REMAINDER)
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)


def wait_for_gpu():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer(pl.Callback):
    def __init__(self):
        self.times, self.losses = [], []

    def on_train_batch_start(self, trainer, module, batch, index):
        wait_for_gpu()
        self.start = time.perf_counter()

    def on_train_batch_end(self, trainer, module, outputs, batch, index):
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        self.losses.append(float(loss))  # waits for the step's values
        wait_for_gpu()
        self.times.append(time.perf_counter() - self.start)
        print(f"step {len(self.times):3d}: {self.times[-1]:6.3f} s, loss {self.losses[-1]:.6f}", flush=True)


timer = Timer()
trainer_init = pl.Trainer.__init__


def bounded(self, *a, **kw):
    kw.update(max_steps=args.steps, max_epochs=-1, limit_val_batches=0, num_sanity_val_steps=0,
              enable_progress_bar=False, enable_checkpointing=False, log_every_n_steps=1)
    kw["callbacks"] = [timer]  # the program's own may save a checkpoint on validation, which is off
    trainer_init(self, *a, **kw)


pl.Trainer.__init__ = bounded
pl.seed_everything(args.seed, workers=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(args.program)))
sys.argv = [args.program, *args.rest]
started = time.perf_counter()
runpy.run_path(args.program, run_name="__main__")
if len(timer.times) <= args.warmup:
    sys.exit(f"the program ran {len(timer.times)} training steps: not more than the {args.warmup} of --warmup, nothing to report")
steady = timer.times[args.warmup:]
if torch.backends.mps.is_available():
    held = torch.mps.driver_allocated_memory() / 2**30
elif torch.cuda.is_available():
    held = torch.cuda.memory_reserved() / 2**30
else:
    held = None
summary = {
    "torch": torch.__version__,
    "platform": sys.platform,
    "steps": len(timer.times),
    "seconds_per_step_median": statistics.median(steady),
    "seconds_per_step_min": min(steady),
    "seconds_per_step_max": max(steady),
    "first_step_seconds": timer.times[0],
    "whole_run_seconds": time.perf_counter() - started,
    "losses": timer.losses,
    "driver_memory_gib": held,
}
print(f"torch {summary['torch']} on {summary['platform']}: median {summary['seconds_per_step_median']:.3f} s per step "
      f"(min {summary['seconds_per_step_min']:.3f}, max {summary['seconds_per_step_max']:.3f}) over {len(steady)} steps; "
      f"first step {summary['first_step_seconds']:.1f} s; GPU memory held "
      f"{held if held is None else round(held, 1)} GiB", flush=True)
with open(os.path.join(args.out, "summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
