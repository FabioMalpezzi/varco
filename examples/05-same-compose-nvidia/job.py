"""A training step loop that runs on whatever GPU the container was given.

On an NVIDIA server PyTorch finds "cuda"; on a Mac, in a container of the
image varco-torch, it finds "mps", which is the Mac's GPU through the bridge.
The program asks PyTorch and takes what it finds: that is all it takes for
one program and one Compose file to serve both.
    python job.py
"""
import time

import torch
import torch.nn as nn

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"torch {torch.__version__.split('+')[0]}, device {device}", flush=True)

# Data from torch.rand, whose values are the same on every machine.
generator = torch.Generator().manual_seed(0)
x = (torch.rand((4096, 64), generator=generator) - 0.5).to(device)
y = (x[:, :8].sum(1, keepdim=True) > 0).float()
model = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
for layer in model:
    if isinstance(layer, nn.Linear):  # uniform weights by hand, the same on every machine
        nn.init.uniform_(layer.weight, -0.05, 0.05, generator=generator)
        nn.init.zeros_(layer.bias)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

start = time.perf_counter()
for step in range(300):
    batch = torch.arange(step * 128, step * 128 + 128, device=device) % len(x)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if device.type == "cpu" else torch.float16):
        loss = nn.functional.binary_cross_entropy_with_logits(model(x[batch]), y[batch])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 99:
        print(f"step {step + 1}: loss {loss.item():.4f}", flush=True)
with torch.no_grad():
    accuracy = ((model(x) > 0).float() == y).float().mean().item()
print(f"300 steps in {time.perf_counter() - start:.2f} s on {device}; accuracy {accuracy:.3f}")
