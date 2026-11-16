"""Lines up two runs of compat.py or compat_train.py: the Mac's and the container's.

    compat_compare.py MAC.txt CONTAINER.txt
    compat_compare.py --inventory compat|compat_train > expected/BATTERY.txt

The exit status is 0 only when both files hold exactly the programs of the battery's inventory
(expected/BATTERY.txt: a program that is missing on either side, or one that is not in the
inventory, is a failure, and so is an empty file) and every program ended as expected.

As expected means: it works on both sides and gives the same thing in full. That is the number
it returns, bit for bit, and for every tensor behind that number the type, the shape and a
digest of all the bytes (the DATA lines, see exact.py). Nothing is compared with a tolerance,
and nothing may fail, except the programs listed in DECLARED below for that battery and that
PyTorch, each with what it must do instead and why. A program that fails on the Mac (an
operator its GPU has not, in that PyTorch version) must fail in the container too, with the
same kind of error: giving a result there is a difference as well.

Numbers that are not finite. The number is written as float.hex writes it, which holds every
bit of a number, of an infinity and of a zero (0.0 is not -0.0), and not those of a NaN: every
NaN is "nan". So an infinity is the same as the same infinity, and a number that is a NaN is
never called the same, on both sides either: what the line does not hold cannot be compared
(NAN below; the tensors hold their NaNs in their bytes, which the digests cover). Where a
tolerance is declared it is for numbers: an infinity or a NaN is within no tolerance, of a
number or of another infinity or NaN, so such a program passes only if it comes out identical.

Needs nothing but the standard library: no PyTorch and no GPU, so that
scripts/check-comparators.py can run it anywhere.
"""
import ast
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BATTERIES = ("compat", "compat_train")

# The only programs that are not held to the Mac's result bit for bit, by battery and PyTorch.
#   ("close", TOLERANCE)       works on both sides; the two numbers are finite and differ by at most
#                              that part of the larger of the two, and the tensors are as many, of
#                              the same types and shapes. No element of a tensor is compared, with
#                              the tolerance or otherwise: the number (for a training loop the last
#                              loss plus a weighted sum of every weight) is all that says how far
#                              apart the two sides are
#   ("fails on both", ERROR)   fails with that error on both sides
#   ("container only", ERROR)  fails with that error on the Mac and works in the container
# A program that is here and no longer does what is written fails the comparison too (a program
# declared close may still come out identical): the entry is then to be removed, not left behind.
NOT_REPRODUCIBLE = ("a transformer training loop that two runs on the Mac itself do not give bit for bit "
                    "(docs/compatibility.md, PyTorch bridge: what works)")
# (PyTorch says so itself of the operator: "Nondeterministic due to atomic_add", in
# aten/src/ATen/native/mps/operations/Embedding.mm and GridSampler.mm, where torch.use_deterministic_algorithms refuses it.)
ADDED_IN_ANY_ORDER = ("a gradient that the Mac's GPU adds up in an order that changes from one run to the next (%s): "
                      "two runs on the Mac itself do not always give it bit for bit")
NO_KERNEL = "PyTorch has no MPS kernel for it in this version: the container raises the Mac's error (docs/compatibility.md)"
DECLARED = {
    ("compat_train", "2.15.0"): {
        "loop_transformer_adamw_cosine_warmup_clip": ("close", 1e-3, NOT_REPRODUCIBLE),
        "loop_adamw_fused_onecycle": ("close", 1e-3, NOT_REPRODUCIBLE),
        "loop_autocast_gradscaler_fused_adam": ("close", 1e-3, ADDED_IN_ANY_ORDER % "the gradient of an embedding whose "
                                                "indices repeat, embedding_dense_backward: 10 runs of 10 differ on the Mac"),
        "gradient_grid_sample": ("close", 1e-3, ADDED_IN_ANY_ORDER % "grid_sampler_2d_backward: 1 run of 6 differed on the "
                                 "Mac, with the GPU busy"),
        "random_rrelu": ("fails on both", "NotImplementedError", NO_KERNEL),
        "attention_with_dropout": ("fails on both", "NotImplementedError", NO_KERNEL),
    },
}
# The lines a program prints besides its number, which must be the same on both sides too.
# (TIME is a speed: printed, never compared.)
SAID = ("TEXT", "KINDS", "OOM")
# A float as float.hex writes it: all the bits of a number, an infinity, or "nan" for every NaN.
HEX = re.compile(r"-?(0x[0-9a-f]+(\.[0-9a-f]*)?p[+-][0-9]+|inf)|nan")

HEADER = """\
# The programs of tests/{battery}.py, one name per line, in the order they run.
# tests/compat_compare.py fails when a run lacks one of them or holds another.
# After adding or removing a program, from pytorch/:
#   python3 tests/compat_compare.py --inventory {battery} > tests/expected/{battery}.txt
# (it reads the names from the source; scripts/check.sh fails when this file
# is not what that command gives).
"""


def programs_of(battery):
    """The programs of a battery, from its source: the functions marked @test, in order."""
    with open(os.path.join(HERE, battery + ".py"), encoding="utf-8") as f:
        tree = ast.parse(f.read())
    return [node.name for node in tree.body if isinstance(node, ast.FunctionDef)
            and any(isinstance(d, ast.Name) and d.id == "test" for d in node.decorator_list)]


def inventory(battery):
    with open(os.path.join(HERE, "expected", battery + ".txt"), encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def is_tensor(t):
    """What exact.py keeps of a tensor: its type, its shape and the SHA-256 of its bytes."""
    return (isinstance(t, list) and len(t) == 3 and isinstance(t[0], str) and t[0] != ""
            and isinstance(t[1], list) and all(type(n) is int and n >= 0 for n in t[1])
            and isinstance(t[2], str) and re.fullmatch(r"[0-9a-f]{64}", t[2]) is not None)


def read(path):
    """What a battery printed: (the INFO line's fields, {program: what it gave}, {TEXT...: line}, what is wrong with the file)."""
    info, programs, said, wrong = {}, {}, {}, []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            kind, _, rest = line.rstrip("\n").partition(" ")
            name, _, value = rest.partition(" ")
            if kind == "INFO":
                info = dict(item.split("=", 1) for item in rest.split() if "=" in item)
            elif kind in ("PASS", "FAIL"):
                if name in programs:
                    wrong.append(f"{name} is there twice")
                programs[name] = {"kind": kind, "text": value}
            elif kind == "DATA":
                try:
                    data = json.loads(value)
                    # As exact.py writes them, or the line is not one: the number as float.hex does
                    # (text that only reads as a number is not it), and of every tensor the type,
                    # the shape and the digest. A tensor without its digest is not compared in full.
                    assert HEX.fullmatch(data["value"])
                    assert isinstance(data["tensors"], list) and all(is_tensor(t) for t in data["tensors"])
                    programs[name]["data"] = data
                except (ValueError, KeyError, TypeError, AssertionError):
                    wrong.append(f"the DATA line of {name} cannot be read, or has no PASS line before it")
            elif kind in SAID or kind == "TIME":
                said[kind] = rest
    return info, programs, said, wrong


def number(data):
    return float.fromhex(data["value"])


def difference(a, b):
    """Where two results that should be the same part ways."""
    parts = []
    x, y = number(a), number(b)
    if a["value"] != b["value"]:
        # The distance of an infinity or a NaN from anything is not a number to print.
        apart = f"they differ by {abs(x - y):.3g}" if math.isfinite(x) and math.isfinite(y) else "not both are finite"
        parts.append(f"the number is {x!r} on the Mac and {y!r} in the container ({apart})")
    ta, tb = a["tensors"], b["tensors"]
    if len(ta) != len(tb):
        parts.append(f"{len(ta)} tensors on the Mac, {len(tb)} in the container")
    else:
        where = [i for i, (p, q) in enumerate(zip(ta, tb)) if p != q]
        if where:
            i = where[0]
            what = [f"{label} {ta[i][k]} on the Mac and {tb[i][k]} in the container"
                    for k, label in ((0, "type"), (1, "shape")) if ta[i][k] != tb[i][k]] or [f"the bytes differ ({ta[i][0]} {ta[i][1]})"]
            parts.append(f"{len(where)} of {len(ta)} tensors differ; tensor {i}: " + ", ".join(what))
    return "; ".join(parts)


def close(a, b, tolerance):
    """Whether two results agree as a program declared not reproducible to the bit must: as many
    tensors, of the same types and shapes, and two finite numbers that differ by at most the
    tolerance times the larger of them. The tensors' bytes and elements are not looked at.

    The tolerance is relative and nothing else: there is no difference small enough to pass by
    itself, so 0 is close to 0 only. An infinity or a NaN is close to nothing: inf - inf is no
    distance, and `inf <= tolerance * inf` would hold for an infinity against any number."""
    x, y = number(a), number(b)
    same_layout = [t[:2] for t in a["tensors"]] == [t[:2] for t in b["tensors"]]
    if not (math.isfinite(x) and math.isfinite(y)):
        return False
    return same_layout and abs(x - y) <= tolerance * max(abs(x), abs(y))


def error(program):
    return program["text"].partition(":")[0]


def compare(mac_path, box_path):
    mac_info, mac, mac_said, mac_wrong = read(mac_path)
    box_info, box, box_said, box_wrong = read(box_path)
    problems = [f"{mac_path}: {w}" for w in mac_wrong] + [f"{box_path}: {w}" for w in box_wrong]
    battery, torch = mac_info.get("battery"), mac_info.get("torch")
    for path, info in ((mac_path, mac_info), (box_path, box_info)):
        if info.get("battery") not in BATTERIES:
            print(f"FAILED: {path} has no INFO line of a battery: it is not the output of compat.py or compat_train.py, or the program never started")
            return 1
    for field in ("battery", "torch", "only"):
        if mac_info.get(field) != box_info.get(field):
            print(f"FAILED: the two files are not of the same run: {field} is {mac_info.get(field)} on the Mac and {box_info.get(field)} in the container")
            return 1
    names = inventory(battery)
    only = mac_info.get("only")
    if only:
        # ONLY=name in the environment of both sides: a part of the battery, said to be one.
        names = [name for name in names if only in name]
        print(f"a part of the battery only: the {len(names)} programs whose name contains {only}")
    if not names:
        problems.append(f"NOTHING TO COMPARE: the inventory of {battery} has no program" + (f" whose name contains {only}" if only else ""))
    declared = DECLARED.get((battery, torch), {})
    same = near = fails_alike = lenient = 0
    for name in names:
        a, b = mac.get(name), box.get(name)
        kind, detail, _ = declared.get(name, ("same", None, None))
        if a is None or b is None:
            sides = [side for side, x in (("on the Mac", a), ("in the container", b)) if x is None]
            problems.append(f"MISSING  {name}: did not run " + " and ".join(sides))
            continue
        both_work = a["kind"] == b["kind"] == "PASS"
        if both_work and ("data" not in a or "data" not in b):
            problems.append(f"NO DATA  {name}: a PASS line without its DATA line cannot be compared in full")
        elif kind == "fails on both":
            if a["kind"] == b["kind"] == "FAIL" and error(a) == error(b) == detail:
                fails_alike += 1
                print(f"fails on both  {name}: {detail}, as declared")
            else:
                problems.append(f"NOT AS DECLARED  {name}: declared to fail with {detail} on both sides; "
                                f"on the Mac {a['kind']} {a['text'][:90]}, in the container {b['kind']} {b['text'][:90]}")
        elif kind == "container only":
            if a["kind"] == "FAIL" and error(a) == detail and b["kind"] == "PASS":
                lenient += 1
                print(f"works in the container only  {name}: {a['text'][:110]} on the Mac, as declared")
            else:
                problems.append(f"NOT AS DECLARED  {name}: declared to fail with {detail} on the Mac and to work in the container; "
                                f"on the Mac {a['kind']} {a['text'][:90]}, in the container {b['kind']} {b['text'][:90]}")
        elif both_work and math.isnan(number(a["data"])) and math.isnan(number(b["data"])):
            problems.append(f"NAN      {name}: the number is a NaN on both sides, and the DATA line does not hold which: "
                            "it cannot be called the same bit for bit (the program is to return a number)")
        elif both_work and a["data"] == b["data"]:
            same += 1
        elif both_work and kind == "close" and close(a["data"], b["data"], detail):
            near += 1
            print(f"close    {name}: {a['text']} on the Mac, {b['text']} in the container, as declared: the two numbers differ by at most "
                  f"{detail:g} of the larger; the {len(a['data']['tensors'])} tensors have the same types and shapes, and their elements are not compared")
        elif both_work:
            problems.append(f"DIFFERS  {name}: {difference(a['data'], b['data'])}"
                            + (f"; declared close, which asks for two finite numbers within {detail:g} of the larger and the same "
                               "types and shapes" if kind == "close" else ""))
        elif a["kind"] == b["kind"] == "FAIL":
            how = "alike" if error(a) == error(b) else f"and with {b['text'][:90]} in the container"
            problems.append(f"FAILS ON BOTH  {name}: {a['text'][:110]} on the Mac, {how}; not declared for {battery} on PyTorch {torch}")
        elif a["kind"] == "PASS":
            problems.append(f"FAILS    {name}: {b['text'][:150]} in the container, works on the Mac")
        else:
            problems.append(f"WORKS IN THE CONTAINER ONLY  {name}: {a['text'][:110]} on the Mac; not declared for {battery} on PyTorch {torch}")
    for name in sorted((set(mac) | set(box)) - set(names)):
        problems.append(f"NOT IN THE INVENTORY  {name} (tests/expected/{battery}.txt)")
    for key in SAID:
        if mac_said.get(key) != box_said.get(key):
            problems.append(f"DIFFERS  the {key} line: {mac_said.get(key, 'none')} on the Mac, {box_said.get(key, 'none')} in the container")
    for problem in problems:
        print(problem)
    if any(problem.startswith("DIFFERS") for problem in problems):
        print("(by how much: run the program on both sides with DUMP=directory, see tests/exact.py, and give the two directories to scripts/tensor-diff.py)")
    # "Equal to 3 digits" is said of the number alone, and the line says so: see DECLARED.
    print(f"{len(names)} programs of {battery} on PyTorch {torch}: {same} identical, {near} equal to 3 digits in their number as declared"
          + (f", {fails_alike} fail on both alike as declared" if fails_alike else "")
          + (f", {lenient} work in the container only as declared" if lenient else "")
          + f"; {len(problems)} not as expected")
    for key in ("TIME",) + SAID:
        if key in mac_said or key in box_said:
            print(f"{key} Mac: {mac_said.get(key, '-')} | container: {box_said.get(key, '-')}")
    return 1 if problems else 0


def main(argv):
    if len(argv) == 2 and argv[0] == "--inventory" and argv[1] in BATTERIES:
        sys.stdout.write(HEADER.format(battery=argv[1]))
        for name in programs_of(argv[1]):
            print(name)
        return 0
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        return compare(argv[0], argv[1])
    except OSError as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
