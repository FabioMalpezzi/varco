"""The shape of token generation: a decoder-only transformer with a key/value
cache, random weights, and the loop mlx-lm uses (async_eval of the next step
before reading the current token with .item()).

No trained model is needed: what is measured is the cost per step of the
machinery, natively and through the remote gpu. What each run generated is
kept too (every token, and a fingerprint of the last step's distribution):
`compare.py gen` holds the container's to the Mac's.

Usage: python genloop.py OUTPUT.json [--layers N] [--dim D] [--tokens T] [--cpu]
"""

import argparse
import hashlib
import json
import sys
import time

import numpy as np

import mlx.core as mx
import mlx.nn as nn


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.rope = nn.RoPE(dim // heads)

    def __call__(self, x, cache):
        B, L, D = x.shape
        q = self.q_proj(x).reshape(B, L, self.heads, -1).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.heads, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            offset = cache[0].shape[2]
            q = self.rope(q, offset=offset)
            k = self.rope(k, offset=offset)
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)
        else:
            q = self.rope(q)
            k = self.rope(k)
        mask = "causal" if L > 1 else None
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.o_proj(out), (k, v)


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = Attention(dim, heads)
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.gate = nn.Linear(dim, 4 * dim, bias=False)
        self.up = nn.Linear(dim, 4 * dim, bias=False)
        self.down = nn.Linear(4 * dim, dim, bias=False)

    def __call__(self, x, cache):
        h, cache = self.attn(self.norm1(x), cache)
        x = x + h
        y = self.norm2(x)
        return x + self.down(nn.silu(self.gate(y)) * self.up(y)), cache


class LM(nn.Module):
    def __init__(self, vocab, dim, layers, heads):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.blocks = [Block(dim, heads) for _ in range(layers)]
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def __call__(self, tokens, cache):
        x = self.embed(tokens)
        new_cache = []
        for i, block in enumerate(self.blocks):
            x, c = block(x, cache[i] if cache else None)
            new_cache.append(c)
        return self.head(self.norm(x)), new_cache


def generate(model, prompt, n):
    """The loop of mlx_lm.generate.generate_step, with greedy sampling."""
    cache = None

    def step(y):
        nonlocal cache
        logits, cache = model(y[None], cache)
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return mx.argmax(logprobs, axis=-1), logprobs.squeeze(0)

    y, logprobs = step(prompt)
    mx.async_eval(y, logprobs)
    tokens, times = [], []
    t = time.perf_counter()
    for _ in range(n):
        next_y, next_logprobs = step(y)
        mx.async_eval(next_y, next_logprobs)
        tokens.append(y.item())
        now = time.perf_counter()
        times.append(now - t)
        t = now
        y, logprobs = next_y, next_logprobs
    return tokens, times, logprobs


def fingerprint(a):
    """An array in full: its type, its shape and a digest of all its bytes."""
    return {"dtype": str(a.dtype), "shape": list(a.shape), "sha256": hashlib.sha256(np.array(a).tobytes()).hexdigest()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output")
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--heads", type=int, default=12)
    p.add_argument("--vocab", type=int, default=32000)
    p.add_argument("--tokens", type=int, default=200)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    if args.cpu:
        mx.set_default_device(mx.cpu)

    mx.random.seed(0)
    model = LM(args.vocab, args.dim, args.layers, args.heads)
    model.set_dtype(mx.float16)
    mx.eval(model.parameters())
    prompt = mx.array([1, 17, 4096, 250, 31999, 7, 7, 1024])

    # Warm-up with another prompt length: the kernels get compiled, but every
    # step of the first measured run is still a graph never seen before (the
    # cache grows by one token per step), as in a real generation.
    generate(model, prompt[:5], 20)
    runs, generated = [], []
    for _ in range(4):
        tokens, times, logprobs = generate(model, prompt, args.tokens)
        # What each run generated, for `compare.py gen`: every token, and the
        # distribution the last step left, to the bit.
        generated.append({"tokens": tokens, "logprobs": fingerprint(logprobs)})
        # The first token carries the prompt; steady state is the rest.
        steady = sorted(times[1:])
        runs.append({
            "tokens_per_s": (len(times) - 1) / sum(times[1:]),
            "median_ms": 1000 * steady[len(steady) // 2],
            "best_ms": 1000 * steady[0],
            "p90_ms": 1000 * steady[int(len(steady) * 0.9)],
            "first_token_ms": 1000 * times[0],
        })
    best = {"first_run": runs[0], "best_repeated_run": max(runs[1:], key=lambda r: r["tokens_per_s"]),
            "all_runs_tokens_per_s": [round(r["tokens_per_s"], 1) for r in runs]}
    result = {
        "info": {"device": str(mx.default_device()), "platform": sys.platform,
                 "layers": args.layers, "dim": args.dim, "tokens": args.tokens},
        "timing": best,
        "tokens": tokens,
        "runs": generated,
    }
    with open(args.output, "w") as f:
        json.dump(result, f)
    print(json.dumps({"info": result["info"], "timing": best, "first_tokens": tokens[:12]}))


if __name__ == "__main__":
    main()
