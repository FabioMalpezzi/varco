"""An ordinary PyTorch program: it knows nothing of the bridge.

It picks the GPU if PyTorch says there is one, as any program does, and
times two kinds of work: a few heavy operations, and many light ones.
Inputs come from formulas, not random numbers, so the checks compare
across machines.
"""

import time

import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {device}")


def timed(label, fn):
    fn()  # once to warm up
    start = time.perf_counter()
    check = fn()
    print(f"{label:<28} {(time.perf_counter() - start) * 1000:9.1f} ms   check {check:.6g}")


def few_heavy():
    """10 products of 4096x4096 matrices: 20 operations."""
    n = 4096
    a = torch.sin(torch.linspace(-3, 3, n * n, device=device).reshape(n, n))
    for _ in range(10):
        a = a @ a
        a = a / a.std()  # keeps the values from growing or vanishing
    return float(a.abs().mean())


def many_light():
    """2000 passes of a small three-layer network: 18000 operations."""
    grid = lambda rows, cols: torch.cos(torch.linspace(0, 50, rows * cols, device=device).reshape(rows, cols))
    x0, w1, w2, w3 = grid(64, 256), grid(256, 256) / 16, grid(256, 256) / 16, grid(256, 256) / 16
    x = x0
    for _ in range(2000):
        h = torch.tanh(x @ w1)
        h = torch.tanh(h @ w2)
        y = h @ w3
        x = x0 + y / (y.abs().mean() + 1)
    return float(x.abs().max())


timed("few heavy operations", few_heavy)
timed("many light operations", many_light)
