"""An ordinary MLX program: sentence embeddings with BGE-M3 (mlx-embeddings).

It knows nothing of the bridge. Run on a Mac it computes on the Mac's GPU;
run in a container of the image varco-mlx-embeddings it computes on the same
GPU, through the bridge. The numbers it prints should be the same.
    python embed.py MODEL_DIR
"""
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_embeddings.utils import load

QUERY = "How do I run a Linux container on a Mac?"
PASSAGES = [
    "On macOS, containers run inside a lightweight Linux virtual machine.",
    "A container shares the kernel of the Linux system it runs on.",
    "MLX is an array framework for machine learning on Apple silicon.",
    "Carbonara is made with guanciale, eggs, pecorino and black pepper.",
    "The Giro d'Italia is a cycling stage race held in May.",
]


def embed(model, tokenizer, sentences):
    inputs = tokenizer.batch_encode_plus(sentences, return_tensors="mlx", padding=True, truncation=True, max_length=512)
    out = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    # The embeddings as an application gets them: bytes in this process.
    return np.array(out.text_embeds.astype(mx.float32))


start = time.perf_counter()
model, tokenizer = load(sys.argv[1])
print(f"device {mx.default_device()}; model loaded in {time.perf_counter() - start:.2f} s")

vectors = embed(model, tokenizer, [QUERY] + PASSAGES)
scores = vectors[1:] @ vectors[0]
print(f"query: {QUERY}")
for score, passage in sorted(zip(scores.tolist(), PASSAGES), reverse=True):
    print(f"  {score:.4f}  {passage}")

for count in (1, 8, 32):
    batch = (PASSAGES * 7)[:count]
    times = []
    for _ in range(15):
        start = time.perf_counter()
        embed(model, tokenizer, batch)
        times.append(time.perf_counter() - start)
    print(f"{count:2d} sentence(s): median {1000 * sorted(times)[len(times) // 2]:.1f} ms")
print(f"checksum of the embeddings: {float(np.abs(vectors).sum()):.6f}")
