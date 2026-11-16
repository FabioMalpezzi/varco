"""Where the time of one BGE-M3 forward goes: building the graph, evaluating
it, reading the result. Usage: python bge_phases.py [--model DIR] [--runs R] [N sentences ...]"""
import sys, time
import numpy as np
import mlx.core as mx
from mlx_embeddings.utils import load
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from bge_m3 import SENTENCES

model_dir = "/models/mlx-community/bge-m3-mlx-fp16"
args = sys.argv[1:]
if "--model" in args:
    i = args.index("--model"); model_dir = args[i + 1]; del args[i:i + 2]
runs = 15
if "--runs" in args:
    i = args.index("--runs"); runs = int(args[i + 1]); del args[i:i + 2]
counts = [int(a) for a in args] or [1]
model, tok = load(model_dir)

def once(sent):
    t0 = time.perf_counter()
    inputs = tok.batch_encode_plus(sent, return_tensors="mlx", padding=True, truncation=True, max_length=512)
    t1 = time.perf_counter()
    out = model(inputs["input_ids"], attention_mask=inputs["attention_mask"]).text_embeds
    t2 = time.perf_counter()
    mx.eval(out)
    t3 = time.perf_counter()
    a = np.array(out)
    t4 = time.perf_counter()
    return np.array([t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0]) * 1000

for n in counts:
    sent = (SENTENCES * 8)[:n]
    for _ in range(3):
        once(sent)
    r = np.array([once(sent) for _ in range(runs)])
    best = r[r[:, 4].argmin()]
    print(f"{n} sentence(s) on {mx.default_device()} ({sys.platform}): tokenize {best[0]:.2f} ms, "
          f"build graph {best[1]:.2f} ms, eval {best[2]:.2f} ms, read {best[3]:.2f} ms, total {best[4]:.2f} ms"
          f" (median of {runs}: {np.median(r[:, 4]):.2f} ms)", flush=True)
