"""Sentence embeddings with BGE-M3 (XLM-RoBERTa large) through mlx-embeddings.

The same script runs natively on the Mac and in the container; compare the
two result files with compare_bge.py. What it compares is the fingerprint
kept here of each array of embeddings: its type, its shape and the SHA-256
of its bytes. The elements are written too, to say how far two results that
differ are: as numbers they do not tell the bytes (0.0 == -0.0).

Usage: python bge_m3.py OUTPUT.json [--model DIR] [--cpu] [--runs N]
"""

import argparse
import hashlib
import json
import sys
import time

import numpy as np

import mlx.core as mx
from mlx_embeddings.utils import load

SENTENCES = [
    "Il gatto dorme sul divano.",
    "A cat is sleeping on the sofa.",
    "La borsa di Milano ha chiuso in rialzo dello 0,8 per cento.",
    "Stock markets closed higher today after the central bank's decision.",
    "Per fare la carbonara servono guanciale, uova, pecorino e pepe nero.",
    "MLX is an array framework for machine learning on Apple silicon.",
    "Quanto costa un biglietto del treno da Bologna a Firenze?",
    "The quick brown fox jumps over the lazy dog, again and again, until the "
    "dog finally gets up, stretches, and walks away to find a quieter place.",
]


def embed(model, tokenizer, sentences):
    inputs = tokenizer.batch_encode_plus(
        sentences, return_tensors="mlx", padding=True, truncation=True,
        max_length=512,
    )
    out = model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    cls = out.last_hidden_state[:, 0]
    cls = cls / mx.linalg.norm(cls, axis=-1, keepdims=True)
    return out.text_embeds, cls


def fingerprint(a):
    """The type of an array of embeddings, its shape and a digest of all its bytes, row-major and little-endian."""
    raw = np.ascontiguousarray(a, dtype=a.dtype.newbyteorder("<")).tobytes()
    return {"dtype": str(a.dtype), "shape": list(a.shape), "sha256": hashlib.sha256(raw).hexdigest()}


def timed(fn, runs):
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return {"best_ms": 1000 * min(times), "median_ms": 1000 * float(np.median(times))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output")
    p.add_argument("--model", default="/models/mlx-community/bge-m3-mlx-fp16")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--runs", type=int, default=10)
    args = p.parse_args()
    if args.cpu:
        mx.set_default_device(mx.cpu)

    t = time.perf_counter()
    model, tokenizer = load(args.model)
    load_s = time.perf_counter() - t

    def run(sentences):
        mean, cls = embed(model, tokenizer, sentences)
        # The embeddings as the application gets them: bytes in this process.
        return np.array(mean.astype(mx.float32)), np.array(cls.astype(mx.float32))

    t = time.perf_counter()
    mean, cls = run(SENTENCES)
    first_s = time.perf_counter() - t

    run(SENTENCES)
    run(SENTENCES)
    batch8 = timed(lambda: run(SENTENCES), args.runs)
    run(SENTENCES[:1])
    single = timed(lambda: run(SENTENCES[:1]), args.runs)
    big = SENTENCES * 4
    run(big)
    batch32 = timed(lambda: run(big), max(3, args.runs // 2))

    result = {
        "info": {
            "device": str(mx.default_device()),
            "platform": sys.platform,
            "mlx": mx.__version__,
            "dtype": str(mean.dtype),
            "shape": list(mean.shape),
        },
        "timings": {
            "load_s": load_s,
            "first_forward_s": first_s,
            "batch8": batch8,
            "single": single,
            "batch32": batch32,
        },
        # The bytes the application got, before they are written as numbers below.
        "fingerprints": {"mean_pooled": fingerprint(mean), "cls": fingerprint(cls)},
        "mean_pooled": mean.tolist(),
        "cls": cls.tolist(),
    }
    with open(args.output, "w") as f:
        json.dump(result, f)
    print(json.dumps({"info": result["info"], "timings": result["timings"]}, indent=1))
    sim = mean @ mean.T
    print("cosine (sentence 0 vs 1, italian/english cat):", round(float(sim[0, 1]), 4),
          " (0 vs 3, cat/stock market):", round(float(sim[0, 3]), 4))


if __name__ == "__main__":
    main()
