"""What an operation whose answer is in the values costs, in a tight loop: the program waits for each.

    python values_cost.py        prints microseconds per call, on the Mac's GPU directly or through the bridge

These are the operations the bridge cannot send and forget: .item(), a comparison used by an
`if`, nonzero, an index by a mask, unique. Each is a round trip to the Mac. Next to them, for
scale, an operation that is sent and not waited for.
"""
import time

import torch

device = torch.device("mps")
x = torch.randn(4096, generator=torch.Generator().manual_seed(0)).to(device)
m = torch.randn(64, 64, generator=torch.Generator().manual_seed(1)).to(device)


def timed(name, function, repeat=300):
    for _ in range(20):
        function()
    torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        function()
    torch.mps.synchronize()
    print(f"COST {name:34} {(time.perf_counter() - start) / repeat * 1e6:9.1f} us", flush=True)


timed("add, not waited for", lambda: x + 1.0, 3000)
timed("matmul 64x64, not waited for", lambda: m @ m, 3000)
timed("sum().item()", lambda: x.sum().item())
timed("float(x[0])", lambda: float(x[0]))
timed("bool(x.max() > 0) for an if", lambda: bool(x.max() > 0))
timed("nonzero", lambda: torch.nonzero(x > 1.0))
timed("x[x > 1] (mask index)", lambda: x[x > 1.0])
timed("masked_select", lambda: torch.masked_select(x, x > 1.0))
timed("unique", lambda: torch.unique((x * 4).round()))
timed("tolist of 16 values", lambda: x[:16].tolist())
timed("cpu() of 4096 values", lambda: x.cpu())
timed("cpu() of 1M values", lambda: torch.ones(1024, 1024, device=device).cpu(), 50)
timed("to(mps) of 1M values", lambda: torch.ones(1024, 1024).to(device), 50)
timed("argmax().item()", lambda: x.argmax().item())
timed("torch.equal", lambda: torch.equal(x, x))
