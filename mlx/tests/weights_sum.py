"""A checksum of every array of a model's weight files, computed on the gpu.

Natively the file is read by MLX; in a container the Mac reads it (or the
container reads and uploads): the checksums must be the same.
    python weights_sum.py MODEL_DIR OUT.json [FILE_PATTERN]
(FILE_PATTERN, e.g. "model-0002*", takes some of the files of a model too
large to hold twice.)
    python weights_sum.py compare A.json B.json
"""

import glob
import json
import os
import sys
import time

import mlx.core as mx


def checksum(a):
    """Two sums of the bytes, modulo 2**32: plain and weighted by position."""
    b = a.view(mx.uint8).reshape(-1).astype(mx.uint32) if a.dtype != mx.bool_ else a.reshape(-1).astype(mx.uint32)
    weights = (mx.arange(b.size, dtype=mx.uint32) % 65521) + 1
    return [int(b.sum().item()), int((b * weights).sum().item())]


def main(model, out, pattern="*"):
    start = time.perf_counter()
    sums = {}
    nbytes = 0
    for path in sorted(glob.glob(os.path.join(model, pattern + ".safetensors"))):
        arrays = mx.load(path)
        for name in sorted(arrays):
            a = arrays[name]
            nbytes += a.nbytes
            sums[f"{os.path.basename(path)}:{name}"] = [str(a.dtype), list(a.shape)] + checksum(a)
        del arrays
    json.dump(sums, open(out, "w"))
    print(f"{len(sums)} arrays, {nbytes / 1e9:.1f} GB, on {mx.default_device()}, {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        a, b = (json.load(open(p)) for p in sys.argv[2:4])
        bad = [k for k in a if a[k] != b.get(k)]
        print(f"{len(a) - len(bad)} of {len(a)} arrays have the same checksum" + (f"; differ: {bad[:5]}" if bad else ""))
        sys.exit(1 if bad or len(a) != len(b) else 0)
    main(*sys.argv[1:4])
