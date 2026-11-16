"""Scaled dot product attention with the shapes of a generation step of
Qwen3.6-35B, on the same inputs natively and in a container: the bytes of the
result, hashed. Tells whether the container's attention is native's.
    python sdpa_probe.py OUT.json
"""
import hashlib
import json
import sys

import numpy as np

import mlx.core as mx


def values(shape, seed):
    n = int(np.prod(shape))
    # The same numbers everywhere: integer arithmetic only.
    x = ((np.arange(n, dtype=np.int64) * 2654435761 + seed * 40503) % 2001 - 1000) / 1000.0
    return x.astype(np.float32).reshape(shape)


results = {}
for dtype in (mx.bfloat16, mx.float16, mx.float32):
    for hq, hkv, d in ((16, 2, 256), (16, 16, 64), (8, 8, 128)):
        for lq, lk, mask in ((26, 26, "causal"), (1, 27, None), (1, 300, None), (9, 40, "causal"), (64, 64, "causal"), (26, 26, None)):
            q = mx.array(values((1, hq, lq, d), 1)).astype(dtype)
            k = mx.array(values((1, hkv, lk, d), 2)).astype(dtype)
            v = mx.array(values((1, hkv, lk, d), 3)).astype(dtype)
            name = f"{dtype} heads {hq}/{hkv} dim {d} q {lq} k {lk} mask {mask}"
            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=d**-0.5, mask=mask)
            data = np.array(out.astype(mx.float32)).tobytes()
            results[name] = hashlib.sha256(data).hexdigest()[:16]
            # As a model gives them: heads moved in front by a transpose, keys
            # and values the first part of a longer cache.
            qt = mx.array(values((1, lq, hq, d), 1)).astype(dtype).transpose(0, 2, 1, 3)
            kc = mx.concatenate([k, mx.zeros((1, hkv, 256, d), dtype)], axis=2)[..., :lk, :]
            vc = mx.concatenate([v, mx.zeros((1, hkv, 256, d), dtype)], axis=2)[..., :lk, :]
            out = mx.fast.scaled_dot_product_attention(qt, kc, vc, scale=d**-0.5, mask=mask)
            data = np.array(out.astype(mx.float32)).tobytes()
            results[name + ", views"] = hashlib.sha256(data).hexdigest()[:16]
            mx.eval(qt, kc, vc)
            out = mx.fast.scaled_dot_product_attention(qt, kc, vc, scale=d**-0.5, mask=mask)
            data = np.array(out.astype(mx.float32)).tobytes()
            results[name + ", views evaluated before"] = hashlib.sha256(data).hexdigest()[:16]
json.dump(results, open(sys.argv[1], "w"), indent=1)
print(len(results), "cases on", mx.default_device())
