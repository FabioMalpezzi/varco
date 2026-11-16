"""Arrays loaded from files, which the Mac reads by itself when it may.

    python load_files.py make DIR             (on the Mac) makes the files
    python load_files.py run DIR OUT.json     loads them and writes what they hold
    python load_files.py compare A.json B.json

DIR/allowed is a directory the server is started with, DIR/outside is not:
in a container the files of the first are read by the Mac, those of the
second (and the compressed archive, and the link that leaves the directory)
are read in the container and uploaded. Either way the values must be those
native MLX gives.
"""

import hashlib
import json
import os
import sys
import tempfile

import numpy as np

import mlx.core as mx


def make(root):
    allowed = os.path.join(root, "allowed")
    outside = os.path.join(root, "outside")
    os.makedirs(os.path.join(allowed, "sub"), exist_ok=True)
    os.makedirs(outside, exist_ok=True)
    rng = np.random.default_rng(7)
    arrays = {
        "f32": rng.standard_normal((3, 5)).astype(np.float32),
        "f16": rng.standard_normal((4, 2, 3)).astype(np.float16),
        "i32": rng.integers(-1000, 1000, (7,)).astype(np.int32),
        "i64": rng.integers(-(2**40), 2**40, (2, 2)).astype(np.int64),
        "u8": rng.integers(0, 255, (11, 3)).astype(np.uint8),
        "u32": rng.integers(0, 2**31, (5,)).astype(np.uint32),
        "bool": rng.integers(0, 2, (6,)).astype(np.bool_),
        "scalar": np.float32(3.25),
        "empty": np.zeros((0, 4), np.float32),
        # Larger than one piece of the Mac's reads (32 MB).
        "large": rng.standard_normal((20, 1024, 1024)).astype(np.float32),
    }
    mlx_arrays = {k: mx.array(v) for k, v in arrays.items()}
    mlx_arrays["bf16"] = mx.array(arrays["f32"]).astype(mx.bfloat16)
    mx.save_safetensors(os.path.join(allowed, "w.safetensors"), mlx_arrays)
    small = {k: v for k, v in mlx_arrays.items() if k not in ("large", "bf16")}
    mx.savez(os.path.join(allowed, "w.npz"), **small)
    mx.savez_compressed(os.path.join(allowed, "wc.npz"), **small)
    mx.save(os.path.join(allowed, "sub", "a.npy"), mlx_arrays["f32"])
    # Bytes to be swapped, and columns first.
    np.save(os.path.join(allowed, "big-endian.npy"), arrays["f32"].astype(">f4"))
    np.save(os.path.join(allowed, "fortran.npy"), np.asfortranarray(arrays["f16"]))
    mx.save_safetensors(
        os.path.join(outside, "s.safetensors"),
        {k: mlx_arrays[k] for k in ("f32", "i32", "u8", "bool")},
    )
    for name, target in (
        ("link-in.safetensors", "w.safetensors"),
        ("link-out.safetensors", "../outside/s.safetensors"),
        ("link-nowhere.safetensors", "../outside/missing.safetensors"),
    ):
        path = os.path.join(allowed, name)
        # Left alone when already there: a container that has the directory
        # mounted may not see a link that was just made again.
        if not os.path.islink(path) or os.readlink(path) != target:
            if os.path.lexists(path):
                os.remove(path)
            os.symlink(target, path)
    fifo = os.path.join(allowed, "fifo")
    if not os.path.exists(fifo):
        os.mkfifo(fifo)


FILES = [
    "allowed/w.safetensors",
    "allowed/w.npz",
    "allowed/wc.npz",
    "allowed/sub/a.npy",
    "allowed/big-endian.npy",
    "allowed/fortran.npy",
    "allowed/link-in.safetensors",
    "allowed/link-out.safetensors",
    "outside/s.safetensors",
]


def describe(a):
    """What an array holds: a number computed on the gpu from it, before its
    bytes are looked at, and then the bytes themselves."""
    out = {"dtype": str(a.dtype), "shape": list(a.shape)}
    if a.size > 0:
        x = a.astype(mx.float32)
        out["sum"] = float((x * 2 + 1).sum().item())
        out["max"] = float(x.max().item())
    out["sha"] = hashlib.sha256(np.array(a.astype(mx.float32) if a.dtype == mx.bfloat16 else a).tobytes()).hexdigest()
    return out


def run(root, out_path):
    results = {}
    for name in FILES:
        loaded = mx.load(os.path.join(root, name))
        if not isinstance(loaded, dict):
            loaded = {"": loaded}
        results[name] = {k: describe(loaded[k]) for k in sorted(loaded)}
    # Saving what was loaded: the bytes come back from the Mac.
    loaded = mx.load(os.path.join(root, "allowed/w.safetensors"))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "saved.safetensors")
        mx.save_safetensors(path, loaded)
        again = mx.load(path)
        results["saved again"] = {k: describe(again[k]) for k in sorted(again)}
    # A lazy array used in a graph without being evaluated first, and one
    # read on the cpu.
    lazy = mx.load(os.path.join(root, "allowed/w.safetensors"))
    results["lazy in a graph"] = float((lazy["large"][3] @ lazy["large"][4]).sum().item())
    cpu = mx.load(os.path.join(root, "allowed/sub/a.npy"), stream=mx.cpu)
    results["on the cpu"] = describe(cpu)
    json.dump({"device": str(mx.default_device()), "results": results}, open(out_path, "w"), indent=1)
    print(f"{sum(len(v) for v in results.values() if isinstance(v, dict))} arrays from {len(FILES)} files, device {mx.default_device()}")


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(sys.argv[2])
    elif sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3])
    else:
        # In compare.py with the other stages' comparisons: every field of
        # every array must be the Mac's, exactly.
        import compare
        sys.exit(compare.main(["load", sys.argv[2], sys.argv[3]]))
