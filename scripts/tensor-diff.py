#!/usr/bin/env python3
"""How far apart two runs are, tensor by tensor: for a comparison that failed.

    scripts/tensor-diff.py DIR_A DIR_B [NAME ...]

DIR_A and DIR_B hold the tensors two runs of the same programs kept with
DUMP=directory (mlx/tests/coverage.py, pytorch/tests/compat.py and
compat_train.py): files NAME.NNN.npy. For every tensor that is not the same
bit for bit it says how many elements differ, the largest difference, and for
floating point how many representable values apart the two are at most
(ULPs). An element that is a NaN on either side is counted apart: its
distance from anything is not a number, and must not pass for no distance.
NAME limits it to the programs of that name.

The comparators say whether two runs are the same; this says by how much they
are not. Two native runs compared this way tell whether the Mac repeats
itself. Needs numpy: run it with the Python of either bridge's environment.
"""
import os
import sys

import numpy as np


def ulps(a, b):
    """For two floating-point arrays, how many representable values lie between each pair of elements."""
    integer = {2: np.int16, 4: np.int32, 8: np.int64}[a.itemsize]

    def ordered(x):
        # The bits of a float as an integer that grows with the float: negative ones are mirrored.
        i = x.view(integer).astype(np.int64)
        return np.where(i < 0, np.iinfo(integer).min - i, i)

    return np.abs(ordered(a) - ordered(b))


def main(argv):
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    first, second, names = argv[0], argv[1], argv[2:]
    files = sorted(set(os.listdir(first)) | set(os.listdir(second)))
    files = [f for f in files if f.endswith(".npy") and (not names or f.rsplit(".", 2)[0] in names)]
    same = different = 0
    for name in files:
        paths = [os.path.join(d, name) for d in (first, second)]
        if not all(map(os.path.exists, paths)):
            different += 1
            print(f"{name}: only in {first if os.path.exists(paths[0]) else second}")
            continue
        a, b = (np.load(p) for p in paths)
        if a.dtype != b.dtype or a.shape != b.shape:
            different += 1
            print(f"{name}: {a.dtype} {list(a.shape)} against {b.dtype} {list(b.shape)}")
            continue
        if a.tobytes() == b.tobytes():
            same += 1
            continue
        different += 1
        a, b = np.ascontiguousarray(a).reshape(-1), np.ascontiguousarray(b).reshape(-1)
        raw = a.view(np.uint8).reshape(a.size, -1) != b.view(np.uint8).reshape(b.size, -1)
        differing = raw.any(axis=1)
        line = f"{name}: {a.dtype} {list(np.load(paths[0], mmap_mode='r').shape)}, {int(differing.sum())} of {a.size} elements differ"
        if a.dtype.kind == "f":
            # A NaN has no distance from anything: those elements are told apart from the ones
            # that are measured. (An infinity has one, from a number or from the other infinity.)
            nans = differing & (np.isnan(a) | np.isnan(b))
            measured = differing & ~nans
            if measured.any():
                with np.errstate(over="ignore", invalid="ignore"):  # inf - inf, where the two are the same
                    gap = np.abs(a.astype(np.float64) - b.astype(np.float64))
                worst = int(np.argmax(np.where(measured, gap, -1.0)))
                line += f"; largest difference {gap[worst]:.3g} ({a[worst]!r} against {b[worst]!r})"
                if np.isfinite(gap[worst]):
                    line += f", relative {gap[worst] / max(abs(float(a[worst])), abs(float(b[worst])), 1e-300):.3g}"
                line += f"; at most {int(ulps(a, b)[measured].max())} ULP apart"
            if nans.any():
                at = int(np.argmax(nans))
                line += (f"; a NaN on one side or on both in {int(nans.sum())} of them ({a[at]!r} against {b[at]!r}): "
                         "no difference to measure")
        elif a.dtype.kind in "iub":
            gap = np.abs(a.astype(np.int64) - b.astype(np.int64))
            line += f"; largest difference {int(gap.max())}"
        print(line)
    print(f"{same} tensors identical, {different} not, of {len(files)}")
    return 1 if different or not files else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
