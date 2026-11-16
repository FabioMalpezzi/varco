"""Compare result files of bge_m3.py: [--report-only] NATIVE.json OTHER.json [OTHER2.json ...]

The embeddings of each OTHER must be those of NATIVE, exactly: the same type,
the same shape and the same bytes. The exit status is 0 only then. What says
so is the fingerprint bge_m3.py keeps of each array of embeddings: its type,
its shape and the SHA-256 of its bytes. The elements, which the files hold
too, are numbers, and equal numbers are not the same bytes (0.0 == -0.0, and
a NaN is in the file without its bits): they say how far a result that
differs is (the cosines, the largest difference, how many elements), and each
file's digest must be that of its own elements. A file without the
fingerprints, as bge_m3.py wrote them before it kept them, is never called
identical: the comparison says that it could not be made, and fails.

The timings are printed and never compared. With --report-only a difference
does not fail, and neither does a file without fingerprints: for a result
that is not meant to be the Mac's, such as the container's CPU. A file that
is not there, or holds no embeddings, is not a difference, and fails.

Needs nothing but the standard library.
"""

import hashlib
import json
import math
import re
import struct
import sys

KEYS = ("mean_pooled", "cls")
# The types whose bytes can be made again from the elements, for struct.
PACK = {"float16": "e", "float32": "f", "float64": "d"}


def rows(result, key):
    """The embeddings under that key as rows of floats, or None when they are not there."""
    value = result.get(key)
    if (not isinstance(value, list) or not value
            or not all(isinstance(row, list) and row and all(isinstance(x, float) for x in row) for row in value)):
        return None
    return value


def fingerprint(result, key):
    """What bge_m3.py kept of those embeddings besides their elements: the type, the shape and
    the SHA-256 of the bytes. None when the file has none, or one that is not a fingerprint."""
    kept = result.get("fingerprints")
    kept = kept.get(key) if isinstance(kept, dict) else None
    if (isinstance(kept, dict) and isinstance(kept.get("dtype"), str) and isinstance(kept.get("shape"), list)
            and re.fullmatch(r"[0-9a-f]{64}", str(kept.get("sha256"))) is not None):
        return kept
    return None


def digest(values, dtype):
    """The SHA-256 of the bytes those elements have in that type (row-major, little-endian, as
    bge_m3.py takes it). None when the elements do not tell the bytes: a NaN is in the file
    without its bits, and so is every element of a type this does not know."""
    flat = [x for row in values for x in row]
    if dtype not in PACK or any(x != x for x in flat):
        return None
    try:
        return hashlib.sha256(struct.pack(f"<{len(flat)}{PACK[dtype]}", *flat)).hexdigest()
    except (struct.error, OverflowError):
        return "not elements of that type"


def missing(result):
    """The kinds of embeddings a file lacks."""
    return [key for key in KEYS if rows(result, key) is None]


def compare(native, other):
    """Print how the embeddings compare. True when they are the same, in full."""
    same = True
    for field in ("dtype", "shape"):
        a, b = native["info"].get(field), other["info"].get(field)
        if a is None or a != b:
            same = False
            print(f"  DIFFERENT {field}: {a} in the reference, {b} here")
    for key in KEYS:
        a, b = rows(native, key), rows(other, key)
        if a is None or b is None:
            same = False
            print(f"  {key}: MISSING in {'the reference' if a is None else 'this file'}")
            continue
        if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
            same = False
            print(f"  {key}: DIFFERENT shape, {len(a)} x {len(a[0])} in the reference, {len(b)} x {len(b[0])} here")
            continue
        cosines = []
        for x, y in zip(a, b):
            norms = math.sqrt(sum(v * v for v in x)) * math.sqrt(sum(v * v for v in y))
            cosines.append(sum(v * w for v, w in zip(x, y)) / norms if norms else float("nan"))
        # How the elements differ, as far as numbers tell. float.hex holds the sign of a zero and
        # not which NaN: a NaN on both sides is left to the digests.
        pairs = [(v, w) for x, y in zip(a, b) for v, w in zip(x, y) if v.hex() != w.hex()]
        zeros = sum(v == w for v, w in pairs)
        odd = sum(v != w and not (math.isfinite(v) and math.isfinite(w)) for v, w in pairs)
        apart = [abs(v - w) for v, w in pairs if math.isfinite(v) and math.isfinite(w)]
        total = sum(len(x) for x in a)
        print(f"  {key}: cosine per sentence min {min(cosines):.7f} max {max(cosines):.7f}, "
              f"max abs difference {max(apart, default=0.0):.2e}")
        if pairs:
            same = False
            print(f"  {key}: {len(pairs)} of {total} elements DIFFER"
                  + (f"; {zeros} only in the sign of a zero (0.0 against -0.0: equal numbers, other bytes)" if zeros else "")
                  + (f"; {odd} with an infinity or a NaN against another value" if odd else ""))
        kept = fingerprint(native, key), fingerprint(other, key)
        if None in kept:
            same = False
            where = " and ".join(side for side, f in zip(("the reference", "this file"), kept) if f is None)
            print(f"  {key}: NOT COMPARED bit for bit: no digest of the bytes in {where} (bge_m3.py kept none before)"
                  + ("" if pairs else f"; none of the {total} elements differs as written, which does not say the bytes are the same"))
            continue
        wrong = [f"DIFFERENT {label} of the bytes: {kept[0][field]} in the reference, {kept[1][field]} here"
                 for field, label in (("dtype", "type"), ("shape", "shape")) if kept[0][field] != kept[1][field]]
        if kept[0]["sha256"] != kept[1]["sha256"]:
            wrong.append(f"the bytes DIFFER: SHA-256 {kept[0]['sha256'][:16]}... in the reference, {kept[1]['sha256'][:16]}... here")
        for side, f, values in (("the reference", kept[0], a), ("this file", kept[1], b)):
            if f["shape"] != [len(values), len(values[0])]:
                wrong.append(f"the shape of the bytes in {side}, {f['shape']}, is NOT that of its elements")
            elif digest(values, f["dtype"]) not in (None, f["sha256"]):
                wrong.append(f"the digest in {side} is NOT that of its elements as {f['dtype']}")
        for line in wrong:
            same = False
            print(f"  {key}: {line}")
        if not wrong and not pairs:
            print(f"  {key}: all {total} elements identical: {kept[0]['dtype']} {kept[0]['shape']}, "
                  f"the same bytes (SHA-256 {kept[0]['sha256'][:16]}...)")
    return same


def main(argv):
    report_only = argv[:1] == ["--report-only"]
    paths = argv[1:] if report_only else argv
    if len(paths) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        native = json.load(open(paths[0]))
        print("reference:", native["info"])
        if missing(native):
            raise ValueError(f"no embeddings under {' and '.join(missing(native))}")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        print(f"FAILED: {paths[0]} is not a result of bge_m3.py: {type(e).__name__}: {e}")
        return 1
    failed = absent = 0
    for path in paths[1:]:
        try:
            other = json.load(open(path))
            print("\ncompared: ", other["info"], path)
            same = compare(native, other)
            for name in ("batch8", "single", "batch32"):
                n, o = native["timings"][name], other["timings"][name]
                print(f"  {name}: best {o['best_ms']:.1f} ms (reference {n['best_ms']:.1f} ms, "
                      f"x{o['best_ms'] / n['best_ms']:.2f}), median {o['median_ms']:.1f} ms "
                      f"(reference {n['median_ms']:.1f} ms)")
            print(f"  load {other['timings']['load_s']:.2f} s, first forward "
                  f"{other['timings']['first_forward_s']:.2f} s (reference "
                  f"{native['timings']['load_s']:.2f} s, {native['timings']['first_forward_s']:.2f} s)")
            if missing(other):
                absent += 1
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            print(f"FAILED: {path} is not a result of bge_m3.py: {type(e).__name__}: {e}")
            same = False
            absent += 1
        if not same:
            failed += 1
    if absent:
        # Not a difference to report: there is nothing to tell the difference of.
        print(f"\nFAILED: {absent} of {len(paths) - 1} files are not results of bge_m3.py with both kinds of embeddings")
        return 1
    if failed and report_only:
        print(f"\n{failed} of {len(paths) - 1} results are not shown to be the reference's (reported only)")
        return 0
    if failed:
        print(f"\nFAILED: {failed} of {len(paths) - 1} results are not shown to be the reference's embeddings, bit for bit")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
