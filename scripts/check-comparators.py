#!/usr/bin/env python3
"""The comparators of the test suites, held to what they must refuse.

The tests of the bridges run every program on the Mac and in a container and
compare the two results. A comparison that passes when a result is wrong or
is not there proves nothing, so this gives the comparators results made up
for the purpose and looks at their exit status:

  - the two sides as each stage expects them        must pass
  - one value, shape or type that differs           must fail
  - a test missing on one side, or on both          must fail
  - a test that is not in the stage's inventory     must fail
  - empty results, and files that are no results    must fail
  - a difference that is not a declared one         must fail
  - +0.0 for -0.0, 1 for 1.0, true for 1            must fail
  - an infinity or a NaN where a tolerance is
    declared, against a number or another           must fail
  - a NaN written without its bits, a result
    without the digest of its bytes                 must fail

    python3 scripts/check-comparators.py

It needs nothing but the standard library, no GPU and no container:
scripts/check.sh runs it, on the hosted runner too. With numpy or PyTorch in
the Python that runs it (the bridges' own environments have them) it also
checks the functions that take the fingerprints of arrays and tensors, and
scripts/tensor-diff.py.
"""
import ast
import copy
import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MLX = os.path.join(ROOT, "mlx", "tests")
TORCH = os.path.join(ROOT, "pytorch", "tests")
failures = []
checks = 0


def check(what, ok, detail=""):
    global checks
    checks += 1
    if not ok:
        failures.append(what)
        print(f"  FAILED {what}" + (f"\n         {detail}" if detail else ""))


def run(script, *args):
    """The exit status of a comparator and the last line it printed."""
    done = subprocess.run([sys.executable, "-B", script, *args], text=True, capture_output=True)
    lines = (done.stdout + done.stderr).strip().splitlines()
    return done.returncode, lines[-1][:200] if lines else "(nothing printed)"


def printed(script, *args):
    """Everything a comparator printed."""
    done = subprocess.run([sys.executable, "-B", script, *args], text=True, capture_output=True)
    return done.stdout + done.stderr


def judged(what, passes, script, *args):
    """The comparator must exit 0 for results that are as expected, and not 0 for any other."""
    code, last = run(script, *args)
    check(f"{'passes' if passes else 'fails'}: {what}", (code == 0) == passes,
          f"{os.path.relpath(script, ROOT)} ended with status {code}: {last}")


def module(path):
    """A comparator as a module, for its inventory and its declared cases. None when it cannot be one."""
    argv, sys.argv = sys.argv, [path]
    try:
        spec = importlib.util.spec_from_file_location("comparator_" + os.path.basename(path)[:-3], path)
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        loaded.main, loaded.inventory, loaded.DECLARED
        return loaded
    except BaseException as e:  # noqa: BLE001 - a script that runs at import, or one without the names used here
        check(f"{os.path.relpath(path, ROOT)} has an inventory and a list of declared cases", False, f"{type(e).__name__}: {e}")
        return None
    finally:
        sys.argv = argv


def write(folder, name, content):
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content if isinstance(content, str) else json.dumps(content))
    return path


ARRAY = {"dtype": "mlx.core.float32", "shape": [2, 3], "sha256": "0" * 64}
OTHER_BYTES = dict(ARRAY, sha256="1" * 64)


# --- the MLX bridge: mlx/tests/compare.py, by stage ---------------------------------------------


def mlx_side(compare, stage, side):
    """A result file of that stage as the comparator expects that side to be."""
    declared = compare.DECLARED.get(stage, {})
    names = compare.inventory(stage)
    outcome = {}
    for i, name in enumerate(names):
        kind, detail, _ = declared.get(name, ("same", None, None))
        if kind == "raises":
            outcome[name] = f"{detail}: it must"
        elif kind == "container raises" and side == "container":
            outcome[name] = f"RuntimeError: [remote gpu] {detail}"
        else:
            outcome[name] = None
    if compare.STAGES[stage][0] is compare.read_coverage:
        # coverage, and any stage added later that writes its results the same way. A test declared to have
        # values of each side's CPU has them: arrays of one type and shape, with other bytes on each side.
        def values(name):
            kind, detail, _ = declared.get(name, ("same", None, None))
            given = [ARRAY, {"python": "float", "value": "0x1.8p+1"}]
            if kind == "cpu of each side":
                given += [ARRAY] * (max(detail) + 1 - len(given))
                for i in detail:
                    given[i] = ARRAY if side == "native" else OTHER_BYTES
            return given

        return {"info": {}, "results": {n: {"raised": e} if e else {"checksum": 1.0, "values": values(n)} for n, e in outcome.items()}}
    if stage == "edge":
        return {"info": {}, "results": {n: {"raised": e} if e else {"ok": "12.0"} for n, e in outcome.items()}}
    if stage == "kernels":
        return {"info": {}, "results": {n: e if e else [ARRAY] for n, e in outcome.items()}}
    if stage == "load":
        # Flat: one value by name, which is what the comparator makes of the arrays of each file.
        return {"device": "gpu", "results": {n: 1.5 for n in outcome}}
    if stage == "gen":
        return {"info": {}, "runs": [{"tokens": [1, 2, 3], "logprobs": ARRAY} for _ in range(len(names) // 2)]}
    if stage == "urgent":
        # The first half are reads the cpu answers: early in the container, and only there.
        cases = {n: {"value": [1, 2], "cpu": i < len(names) // 2, "work_ms": 100.0,
                     "read_ms": 1.0 if side == "container" and i < len(names) // 2 else 99.0} for i, n in enumerate(names)}
        waited = {"value": 21, "waited_ms": 100.0, "work_ms": 100.0}
        return {"cases": cases, "synchronize": waited, "eval": dict(waited, value=[21, 0.5]), "first": {"read_ms": 50.0, "work_ms": 100.0}}
    if stage == "upstream":
        return upstream_side(compare, side, kernels=False)
    return None


def upstream_side(compare, side, kernels):
    """A result of MLX's own tests as the comparator expects that side to be, the Mac's server taking custom kernels or not."""
    doc = {n: {"ran": 3, "skipped": ["TestIt.test_skipped"], "bad": {}} for n in compare.inventory("upstream")}
    fails, skips = compare.upstream_declared(side, kernels)
    for module_name, method in fails:
        doc[module_name]["bad"][method + " (case=1)"] = "AssertionError"
    for module_name, method in skips:
        doc[module_name]["skipped"].append(method)
    # What mx.metal.is_available() answers on that side: on the Mac yes, in a container what the server takes.
    doc["info"] = {"metal": side == "native" or kernels}
    return doc


def mlx_items(stage, doc):
    """The dictionary of a result file that holds its tests by name."""
    return doc["cases"] if stage == "urgent" else doc if stage == "upstream" else doc.get("results")


def check_mlx(folder):
    script = os.path.join(MLX, "compare.py")
    # What the review found: a wrong number, and nothing at all, both passed.
    old = [write(folder, f"old-{side}.json", {"info": "made up", "results": {"arithmetic": value}})
           for side, value in (("native", 1.0), ("container", 99.0))]
    judged("coverage, one test whose result is 1 on the Mac and 99 in the container", False, script, *old)
    judged("coverage, numbers alone, even equal ones: they are not the arrays", False, script, old[0], old[0])
    empty = write(folder, "empty.json", "")
    judged("coverage, two empty files", False, script, empty, empty)
    nothing = write(folder, "nothing.json", {"info": {}, "results": {}})
    judged("coverage, two results with no test in them", False, script, nothing, nothing)
    judged("coverage, a file that is not there", False, script, os.path.join(folder, "absent.json"), nothing)

    compare = module(script)
    if compare is None:
        return
    for stage in compare.STAGES:
        sides = {side: mlx_side(compare, stage, side) for side in ("native", "container")}
        if sides["native"] is None:
            check(f"{stage}: this script has results made up for the stage (mlx_side)", False,
                  "a stage of compare.py that is not checked here: give it results in mlx_side, or the format of coverage")
            continue

        def judge(what, passes, native=None, container=None):
            paths = [write(folder, f"{stage}-{side}.json", doc if doc is not None else sides[side])
                     for side, doc in (("native", native), ("container", container))]
            judged(f"{stage}, {what}", passes, script, stage, *paths)

        judge("the two sides as expected", True)
        if stage == "gen":
            for what, change in (("another token in a run", lambda d: d["runs"][1]["tokens"].append(7)),
                                 ("other bytes in the last logprobs of a run", lambda d: d["runs"][0].update(logprobs=OTHER_BYTES)),
                                 ("a run missing in the container", lambda d: d["runs"].pop())):
                changed = copy.deepcopy(sides["container"])
                change(changed)
                judge(what, False, container=changed)
            both = copy.deepcopy(sides["native"])
            both["runs"].pop()
            judge("a run missing on both sides", False, native=both, container=both)
            judge("no runs at all", False, native={"info": {}, "runs": []}, container={"info": {}, "runs": []})
            continue
        names = compare.inventory(stage)
        plain = [n for n in names if n not in compare.DECLARED.get(stage, {})]
        victim = plain[len(plain) // 2]
        changed = copy.deepcopy(sides["container"])
        item = mlx_items(stage, changed)[victim]
        if compare.STAGES[stage][0] is compare.read_coverage:
            item["values"][0] = OTHER_BYTES
        elif stage == "edge":
            item["ok"] = "12.000001"
        elif stage == "kernels":
            item[0] = OTHER_BYTES
        elif stage == "load":
            mlx_items(stage, changed)[victim] = 1.5000000000000002
        elif stage == "urgent":
            item["value"] = [1, 3]
        elif stage == "upstream":
            item["ran"] = 4
        judge("one result that differs in the container", False, container=changed)
        for what, sides_changed in (("a test missing in the container", ("container",)), ("a test missing on the Mac", ("native",)),
                                    ("a test missing on both sides", ("native", "container"))):
            docs = {side: copy.deepcopy(sides[side]) for side in sides}
            for side in sides_changed:
                del mlx_items(stage, docs[side])[victim]
            judge(what, False, **docs)
        docs = {side: copy.deepcopy(sides[side]) for side in sides}
        for side in docs:
            mlx_items(stage, docs[side])["not_in_the_inventory"] = copy.deepcopy(mlx_items(stage, docs[side])[victim])
        judge("a test that is not in the inventory, the same on both sides", False, **docs)
        docs = {side: copy.deepcopy(sides[side]) for side in sides}
        for side in docs:
            mlx_items(stage, docs[side]).clear()
        judge("no tests at all", False, **docs)

    # What each stage adds to "the same on both sides".
    sides = {side: mlx_side(compare, "coverage", side) for side in ("native", "container")}
    victim = compare.inventory("coverage")[3]

    def coverage(what, change, both=False):
        docs = {side: copy.deepcopy(sides[side]) for side in sides}
        for side in ("native", "container") if both else ("container",):
            change(docs[side]["results"][victim])
        paths = [write(folder, f"coverage-{side}.json", docs[side]) for side in ("native", "container")]
        judged(f"coverage, {what}", False, script, "coverage", *paths)

    coverage("the same bytes in another shape", lambda r: r["values"][0].update(shape=[3, 2]))
    coverage("the same bytes as another type", lambda r: r["values"][0].update(dtype="mlx.core.int32"))
    coverage("one array fewer", lambda r: r["values"].pop())
    coverage("a value read that differs in its last bit", lambda r: r["values"][1].update(value="0x1.8000000000001p+1"))
    coverage("an error in the container where the Mac has a result", lambda r: (r.clear(), r.update(raised="ValueError: no")))
    coverage("the same error on both sides, which is not declared", lambda r: (r.clear(), r.update(raised="ValueError: no")), both=True)

    # A test declared to have values that the CPU of each side computes: those values may differ in their
    # bytes and in nothing else, and no other value of the test may differ at all.
    for name, (kind, places, _) in compare.DECLARED.get("coverage", {}).items():
        if kind != "cpu of each side":
            continue

        def by_cpu(what, passes, change):
            docs = {side: copy.deepcopy(sides[side]) for side in sides}
            change(docs["container"]["results"][name]["values"])
            paths = [write(folder, f"coverage-{side}.json", docs[side]) for side in ("native", "container")]
            judged(f"coverage, {name}, {what}", passes, script, "coverage", *paths)

        other = next(i for i in range(len(sides["native"]["results"][name]["values"])) if i not in places)
        by_cpu("a value of each side's CPU with other bytes, as declared", True, lambda v: None)
        by_cpu("the values of each side's CPU the same after all", True, lambda v: [v.__setitem__(i, dict(ARRAY)) for i in places])
        by_cpu("a value of each side's CPU of another shape", False, lambda v: v.__setitem__(places[0], dict(OTHER_BYTES, shape=[3, 2])))
        by_cpu("a value of each side's CPU of another type", False, lambda v: v.__setitem__(places[0], dict(OTHER_BYTES, dtype="mlx.core.int32")))
        by_cpu("another value of the test with other bytes", False, lambda v: v.__setitem__(other, dict(OTHER_BYTES))
               if isinstance(v[other], dict) and "sha256" in v[other] else v[other].update(value="0x1.8000000000001p+1"))
        by_cpu("one value fewer", False, lambda v: v.pop())

    # What "the same" is held to for the values a program read: their type and all their bits,
    # and a NaN, which the files hold without its bits, is the same as nothing.
    def read(stage, what, passes, native, container, key=None):
        docs = {side: mlx_side(compare, stage, side) for side in ("native", "container")}
        name = next(n for n in compare.inventory(stage) if n not in compare.DECLARED.get(stage, {}))
        for side, value in (("native", native), ("container", container)):
            if key:
                docs[side][key]["value"] = value
            elif stage == "coverage":
                docs[side]["results"][name]["values"][1] = value
            elif stage == "urgent":
                docs[side]["cases"][name]["value"] = value
            elif stage == "gen":
                docs[side]["runs"][0]["tokens"] = value
            elif stage == "upstream":
                docs[side][name]["ran"] = value
            else:
                docs[side]["results"][name] = value
        paths = [write(folder, f"{stage}-{side}.json", docs[side]) for side in ("native", "container")]
        judged(f"{stage}, {what}", passes, script, stage, *paths)

    nan, inf = float("nan"), float("inf")
    read("load", "0.0 on the Mac and -0.0 in the container", False, 0.0, -0.0)
    read("load", "-0.0 on both sides", True, -0.0, -0.0)
    read("load", "2.0 on the Mac and the integer 2 in the container", False, 2.0, 2)
    read("load", "true on the Mac and 1 in the container", False, True, 1)
    read("load", "a number on the Mac and the same number as text in the container", False, 1.5, "1.5")
    read("load", "nothing (null) for a value, the same on both sides", False, None, None)
    read("load", "nothing (null) inside a value, the same on both sides", False, [1.5, None], [1.5, None])
    read("load", "a NaN on both sides: the file does not hold its bits", False, nan, nan)
    read("load", "a NaN inside a value, on both sides", False, [1.5, nan], [1.5, nan])
    read("load", "a NaN on the Mac and a number in the container", False, nan, 1.5)
    read("load", "an infinity on both sides", True, inf, inf)
    read("load", "an infinity on the Mac and minus infinity in the container", False, inf, -inf)
    read("load", "a number on the Mac and an infinity in the container", False, 1.5, inf)
    read("urgent", "[1, 2] on the Mac and [1.0, 2.0] in the container", False, [1, 2], [1.0, 2.0])
    read("urgent", "0.0 in a list on the Mac and -0.0 in the container", False, [0.0, 1.0], [-0.0, 1.0])
    read("urgent", "true on the Mac and 1 in the container", False, True, 1)
    read("urgent", "a NaN on both sides", False, nan, nan)
    read("urgent", "nothing (null) on both sides", False, None, None)
    read("urgent", "0.0 after eval() on the Mac and -0.0 in the container", False, [21, 0.0], [21, -0.0], key="eval")
    read("urgent", "21 after synchronize() on the Mac and 21.0 in the container", False, 21, 21.0, key="synchronize")
    read("urgent", "a NaN after eval(), on both sides", False, [21, nan], [21, nan], key="eval")
    read("gen", "the tokens as integers on the Mac and as floating point in the container", False, [1, 2, 3], [1.0, 2.0, 3.0])
    read("gen", "tokens that are not integers, the same on both sides", False, [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    read("upstream", "3 tests ran on the Mac and 3.0 in the container", False, 3, 3.0)
    read("upstream", "a count of tests that is not an integer, the same on both sides", False, None, None)

    def value(kind, what):
        return {"python": kind, "value": what}

    read("coverage", "a value read written as a number: 0.0 on the Mac and -0.0 in the container", False, value("float", 0.0), value("float", -0.0))
    read("coverage", "a value read: 0x0.0p+0 on the Mac and -0x0.0p+0 in the container", False, value("float", (0.0).hex()), value("float", (-0.0).hex()))
    read("coverage", "a value read that is true on the Mac and 1 in the container", False, value("bool", True), value("bool", 1))
    read("coverage", "a value read that is a NaN on both sides: float.hex does not hold its bits", False, value("float", "nan"), value("float", "nan"))
    read("coverage", "a value read that is a NaN written as a number, on both sides", False, value("float", nan), value("float", nan))
    read("coverage", "a value read that is an infinity on both sides", True, value("float", "inf"), value("float", "inf"))
    read("coverage", "a value read that is not as float.hex writes it, the same on both sides", False, value("float", "1.5"), value("float", "1.5"))
    read("coverage", "a value read that is nothing (None) on both sides", True, value("NoneType", None), value("NoneType", None))

    for stage in ("edge", "kernels"):
        declared = compare.DECLARED[stage]
        raises = next(n for n, d in declared.items() if d[0] == "raises")
        refused = next(n for n, d in declared.items() if d[0] == "container raises")
        encode = (lambda e: {"raised": e}) if stage == "edge" else (lambda e: e)
        fine = {"ok": "1.0"} if stage == "edge" else [ARRAY]
        for what, name, value in (("a declared error that is of another type in the container", raises, encode("OtherError: it must")),
                                  ("a declared error that does not come in the container", raises, fine),
                                  ("what the container must refuse, done there", refused, fine),
                                  ("what the container must refuse, refused for another reason", refused, encode("RuntimeError: something else"))):
            changed = mlx_side(compare, stage, "container")
            changed["results"][name] = value
            paths = [write(folder, f"{stage}-native.json", mlx_side(compare, stage, "native")), write(folder, f"{stage}-container.json", changed)]
            judged(f"{stage}, {what}", False, script, stage, *paths)

    native = mlx_side(compare, "urgent", "native")
    for what, change in (("a read the GPU computes answered early in the container", lambda d: list(d["cases"].values())[-1].update(read_ms=1.0)),
                         ("a read the cpu can answer that waits in the container", lambda d: list(d["cases"].values())[0].update(read_ms=99.0)),
                         ("synchronize() that does not wait in the container", lambda d: d["synchronize"].update(waited_ms=10.0)),
                         ("another value after eval()", lambda d: d["eval"].update(value=[21, 0.75]))):
        changed = mlx_side(compare, "urgent", "container")
        change(changed)
        judged(f"urgent, {what}", False, script, "urgent", write(folder, "urgent-native.json", native), write(folder, "urgent-container.json", changed))

    native = mlx_side(compare, "upstream", "native")
    first = compare.inventory("upstream")[0]
    declared_module, declared_method = sorted(compare.upstream_declared("container", False)[0])[0]
    skipped_module, skipped_method = sorted(compare.upstream_declared("container", False)[1])[0]
    for what, change in (("a test that fails in the container and is not declared to", lambda d: d[first]["bad"].update({"TestIt.test_new": "AssertionError"})),
                         ("a test declared to fail in the container that passes", lambda d: d[declared_module]["bad"].pop(declared_method + " (case=1)")),
                         ("a test the container skips and the Mac runs, not declared", lambda d: d[first]["skipped"].append("TestIt.test_not_run")),
                         ("a test declared to be skipped in the container that it runs", lambda d: d[skipped_module]["skipped"].remove(skipped_method)),
                         ("a test declared to be skipped in the container that the Mac skips too", None)):
        changed = mlx_side(compare, "upstream", "container")
        mac = native
        if change is None:  # (the Mac's side is the one changed here)
            mac = mlx_side(compare, "upstream", "native")
            mac[skipped_module]["skipped"].append(skipped_method)
        else:
            change(changed)
        judged(f"upstream, {what}", False, script, "upstream", write(folder, "upstream-native.json", mac), write(folder, "upstream-container.json", changed))

    # What is declared of the container depends on whether the Mac's server takes custom kernels, which the
    # container's result says: each set of declarations holds for its own server and not for the other.
    def upstream(what, passes, container, mac=None):
        judged(f"upstream, {what}", passes, script, "upstream", write(folder, "upstream-native.json", mac or native),
               write(folder, "upstream-container.json", container))

    with_kernels = upstream_side(compare, "container", kernels=True)
    upstream("a server with custom kernels, the container as declared for one", True, with_kernels)
    upstream("a server with custom kernels, the container as declared for a server without", False,
             dict(upstream_side(compare, "container", kernels=False), info={"metal": True}))
    upstream("a server without custom kernels, the container as declared for a server with", False, dict(with_kernels, info={"metal": False}))
    kernel_module, kernel_method = sorted(compare.UPSTREAM_FAILS_WITH_KERNELS)[0]
    changed = copy.deepcopy(with_kernels)
    changed[kernel_module]["bad"].pop(kernel_method + " (case=1)")
    upstream("a server with custom kernels, a test declared to fail with them that passes", False, changed)
    changed = copy.deepcopy(with_kernels)
    changed[first]["bad"]["TestIt.test_new"] = "AssertionError"
    upstream("a server with custom kernels, a test that fails in the container and is not declared to", False, changed)
    for what, info in (("that does not say whether the server takes custom kernels", None), ("that says it with a number", {"metal": 1}),
                       ("that says it with nothing", {"metal": None})):
        changed = mlx_side(compare, "upstream", "container")
        if info is None:
            del changed["info"]
        else:
            changed["info"] = info
        upstream(f"a container's result {what}", False, changed)

    # A test that may fail on either side: its known failure is not a difference, and hides no other.
    for (may_module, may_method), (words, _) in compare.UPSTREAM_MAY_FAIL.items():
        known = f"AssertionError: False is not true : Shapes (3,) (3,), {words}"

        def failing(side, text=known, kernels=False):
            doc = upstream_side(compare, side, kernels)
            doc[may_module]["bad"][may_method] = text
            return doc

        name = f"{may_module} {may_method}"
        upstream(f"{name} fails in the container as it may", True, failing("container"))
        upstream(f"{name} fails on the Mac as it may", True, mlx_side(compare, "upstream", "container"), mac=failing("native"))
        upstream(f"{name} fails on both sides as it may", True, failing("container"), mac=failing("native"))
        upstream(f"{name} fails in the container as it may, a server with custom kernels", True, failing("container", kernels=True))
        upstream(f"{name} fails in the container with other words", False, failing("container", "RuntimeError: [remote gpu] something else"))
        upstream(f"{name} fails on the Mac with other words", False, mlx_side(compare, "upstream", "container"),
                 mac=failing("native", "AssertionError: Lists differ"))
        changed = failing("container")
        changed[may_module]["bad"][may_method + " (case=2)"] = "RuntimeError: [remote gpu] something else"
        upstream(f"{name} fails in the container as it may and, in another case, with other words", False, changed)
        changed = failing("container")
        changed[first]["bad"]["TestIt.test_new"] = "AssertionError"
        upstream(f"{name} fails as it may, and another test fails that is not declared to", False, changed)
        changed = failing("container")
        changed[declared_module]["bad"].pop(declared_method + " (case=1)")
        upstream(f"{name} fails as it may, and a test declared to fail passes", False, changed)

    # The inventory the comparator prints from a result is the one it then holds results to.
    for stage in compare.STAGES:
        if mlx_side(compare, stage, "native") is None:
            continue
        path = write(folder, f"{stage}-native.json", mlx_side(compare, stage, "native"))
        done = subprocess.run([sys.executable, "-B", script, "--inventory", stage, path], text=True, capture_output=True)
        with open(os.path.join(MLX, "expected", stage + ".txt"), encoding="utf-8") as f:
            check(f"mlx/tests/expected/{stage}.txt is what `compare.py --inventory {stage}` writes", done.stdout == f.read())


def bge_fingerprints(result):
    """What bge_m3.py keeps of the embeddings besides their elements: the type, the shape and a digest of the bytes."""
    out = {}
    for key in ("mean_pooled", "cls"):
        flat = [x for row in result[key] for x in row]
        out[key] = {"dtype": "float32", "shape": [len(result[key]), len(result[key][0])],
                    "sha256": hashlib.sha256(struct.pack(f"<{len(flat)}f", *flat)).hexdigest()}
    return out


def check_bge(folder):
    script = os.path.join(MLX, "compare_bge.py")
    rows = [[0.25, -0.5, 0.125], [0.75, 0.5, -0.25]]
    result = {"info": {"dtype": "float32", "shape": [2, 3]}, "mean_pooled": rows, "cls": rows,
              "timings": {"load_s": 1.0, "first_forward_s": 0.1, **{k: {"best_ms": 1.0, "median_ms": 1.0} for k in ("batch8", "single", "batch32")}}}
    result["fingerprints"] = bge_fingerprints(result)

    def judge(what, passes, change, *options, native=None, says=(), never=()):
        """change is made to the container's file, native to the Mac's; says and never are words of what is printed."""
        mac, changed = copy.deepcopy(result), copy.deepcopy(result)
        if native:
            native(mac)
        change(changed)
        paths = [write(folder, "bge-native.json", mac), write(folder, "bge-container.json", changed)]
        judged(f"bge, {what}", passes, script, *options, *paths)
        if says or never:
            text = printed(script, *options, *paths)
            check(f"bge, {what}: says {', '.join(says) or 'nothing'}" + (f" and never {', '.join(never)}" if never else ""),
                  all(word in text for word in says) and not any(word in text for word in never), text[-400:])

    def kept(change):
        """A change to the embeddings as bge_m3.py would write it: with the digest of the new bytes."""
        def both(r):
            change(r)
            r["fingerprints"] = bge_fingerprints(r)
        return both

    def element(value, key="cls"):
        return lambda r: r[key][1].__setitem__(2, value)

    def no_digests(r):
        del r["fingerprints"]

    def digest(key, **changes):
        return lambda r: r["fingerprints"][key].update(changes)

    judge("the same embeddings", True, lambda r: None, says=("identical", "batch8: best", "first forward"))
    judge("one element that differs in its last bit", False, lambda r: r["cls"][1].__setitem__(2, -0.25000000000000006))
    judge("a sentence fewer", False, lambda r: r["mean_pooled"].pop())
    judge("no embeddings of one kind", False, lambda r: r.pop("cls"))
    judge("empty embeddings", False, lambda r: r.update(mean_pooled=[], cls=[]))
    judge("another type", False, lambda r: r["info"].update(dtype="float16"))
    judge("embeddings that differ, compared with --report-only", True, lambda r: r["cls"][1].__setitem__(2, 0.5), "--report-only",
          says=("batch8: best",))
    empty = write(folder, "empty.json", "")
    judged("bge, two empty files", False, script, empty, empty)

    # The same numbers are not the same bytes: 0.0 and -0.0 are equal, and are not one value of a float32.
    judge("0.0 on the Mac and -0.0 in the container, each with the digest of its bytes", False, kept(element(-0.0)),
          native=kept(element(0.0)), says=("bytes",), never=("identical",))
    judge("0.0 on the Mac and -0.0 in the container, in files without digests", False, lambda r: (element(-0.0)(r), no_digests(r)),
          native=lambda r: (element(0.0)(r), no_digests(r)), never=("identical",))
    judge("-0.0 on both sides", True, kept(element(-0.0)), native=kept(element(-0.0)), says=("identical",))
    judge("the same embeddings in files without digests, as bge_m3.py wrote them before it kept them", False, no_digests,
          native=no_digests, says=("no digest", "batch8: best"), never=("identical",))
    judge("the same embeddings, with the digests on the Mac's side only", False, no_digests, never=("identical",))
    judge("the same embeddings, with the digests in the container's file only", False, lambda r: None, native=no_digests, never=("identical",))
    judge("the same numbers with the digest of other bytes", False, digest("cls", sha256="1" * 64), never=("cls: all 6 elements identical",))
    judge("the same numbers and digest, of another type", False, digest("mean_pooled", dtype="float16"))
    judge("the same numbers and digest, in another shape", False, digest("mean_pooled", shape=[3, 2]))
    judge("a digest that is not one, the same on both sides", False, digest("cls", sha256="the same"), native=digest("cls", sha256="the same"),
          says=("no digest",), never=("cls: all 6 elements identical",))
    judge("the same digest on both sides, which is not that of the elements", False, digest("cls", sha256="1" * 64),
          native=digest("cls", sha256="1" * 64), never=("cls: all 6 elements identical",))
    judge("a NaN on the Mac and a number in the container", False, lambda r: None, native=kept(element(float("nan"))))
    judge("a number on the Mac and a NaN in the container", False, kept(element(float("nan"))))
    judge("a number on the Mac and an infinity in the container", False, kept(element(float("inf"))))
    judge("an infinity on the Mac and minus infinity in the container", False, kept(element(float("-inf"))), native=kept(element(float("inf"))))
    judge("an infinity on both sides", True, kept(element(float("inf"))), native=kept(element(float("inf"))))
    judge("a NaN on both sides, with the same digest of the bytes", True, kept(element(float("nan"))), native=kept(element(float("nan"))))
    judge("a NaN on both sides, with digests that differ", False, lambda r: (kept(element(float("nan")))(r), digest("cls", sha256="1" * 64)(r)),
          native=kept(element(float("nan"))))
    judge("an element written as text, the same on both sides", False, element("-0.25"), native=element("-0.25"))
    judge("an element written as float.hex does, the same on both sides", False, element((-0.25).hex()), native=element((-0.25).hex()))
    judge("an element that is nothing (null), the same on both sides", False, element(None), native=element(None))
    judge("an element that is an integer, the same on both sides", False, element(1), native=element(1))
    # --report-only: a difference is told and does not fail; a result that is not there is not a difference.
    judge("0.0 against -0.0, compared with --report-only", True, kept(element(-0.0)), "--report-only", native=kept(element(0.0)),
          says=("bytes",), never=("identical",))
    judge("files without digests, compared with --report-only", True, no_digests, "--report-only", native=no_digests,
          says=("no digest", "batch8: best"), never=("identical",))
    judge("no embeddings of one kind, compared with --report-only", False, lambda r: r.pop("cls"), "--report-only")
    judged("bge, an empty file compared with --report-only", False, script, "--report-only",
           write(folder, "bge-native.json", result), empty)
    judged("bge, a file that is not there compared with --report-only", False, script, "--report-only",
           write(folder, "bge-native.json", result), os.path.join(folder, "absent.json"))


# --- the PyTorch bridge: pytorch/tests/compat_compare.py ----------------------------------------

TENSOR = ["torch.float32", [2, 3], "0" * 64]


def torch_side(compare, battery, torch, side, only=None):
    """What a battery prints on that side when everything is as the comparator expects."""
    declared = compare.DECLARED.get((battery, torch), {})
    lines = [f"INFO battery={battery} torch={torch} device=mps platform={'darwin' if side == 'mac' else 'linux'}" + (f" only={only}" if only else "")]
    for name in compare.inventory(battery):
        if only and only not in name:
            continue
        kind, detail, _ = declared.get(name, ("same", None, None))
        if kind == "fails on both" or (kind == "container only" and side == "mac"):
            lines.append(f"FAIL {name} {detail}: it must")
        else:
            lines += [f"PASS {name} 1.5", "DATA " + name + " " + json.dumps({"value": (1.5).hex(), "tensors": [TENSOR, TENSOR]})]
    return lines + ["TEXT the same words", "KINDS TypeError RuntimeError", "TIME 100 tokens per second"]


def data(line, **changes):
    """A DATA line with some of what it holds changed."""
    kind, name, text = line.split(" ", 2)
    return f"{kind} {name} " + json.dumps({**json.loads(text), **changes})


def check_torch(folder):
    script = os.path.join(TORCH, "compat_compare.py")
    # What the review found: a wrong number, and nothing at all, both passed.
    judged("compat, a program that gives 1 on the Mac and 99 in the container", False, script,
           write(folder, "mac.txt", "PASS arithmetic 1\n"), write(folder, "box.txt", "PASS arithmetic 99\n"))
    judged("compat, two empty files", False, script, write(folder, "mac.txt", ""), write(folder, "box.txt", ""))
    judged("compat, a file that is not there", False, script, os.path.join(folder, "absent.txt"), write(folder, "box.txt", ""))

    compare = module(script)
    if compare is None:
        return
    for battery in compare.BATTERIES:
        done = subprocess.run([sys.executable, "-B", script, "--inventory", battery], text=True, capture_output=True)
        with open(os.path.join(TORCH, "expected", battery + ".txt"), encoding="utf-8") as f:
            check(f"pytorch/tests/expected/{battery}.txt lists the programs of {battery}.py (compat_compare.py --inventory {battery})",
                  done.returncode == 0 and done.stdout == f.read())

    def judge(what, passes, mac, box):
        judged(what, passes, script, write(folder, "mac.txt", "\n".join(mac) + "\n"), write(folder, "box.txt", "\n".join(box) + "\n"))

    for battery, torch in sorted({(b, t) for b in compare.BATTERIES for _, t in compare.DECLARED}):
        mac, box = (torch_side(compare, battery, torch, side) for side in ("mac", "box"))
        label = f"{battery} on PyTorch {torch}"
        judge(f"{label}, the two sides as expected", True, mac, box)
        declared = compare.DECLARED.get((battery, torch), {})
        plain = [n for n in compare.inventory(battery) if n not in declared]
        victim = plain[len(plain) // 2]
        at = box.index(f"PASS {victim} 1.5")
        other = list(TENSOR[:2]) + ["1" * 64]

        def changed(line, lines=box, where=at + 1):
            return lines[:where] + ([line] if line else []) + lines[where + 1:]

        judge(f"{label}, a number that differs in its last bit", False, mac, changed(data(box[at + 1], value=(1.5000000000000002).hex())))
        judge(f"{label}, the same number from a tensor with other bytes", False, mac, changed(data(box[at + 1], tensors=[TENSOR, other])))
        judge(f"{label}, the same bytes in another shape", False, mac, changed(data(box[at + 1], tensors=[TENSOR, ["torch.float32", [3, 2], "0" * 64]])))
        judge(f"{label}, the same bytes as another type", False, mac, changed(data(box[at + 1], tensors=[TENSOR, ["torch.float16", [2, 3], "0" * 64]])))
        judge(f"{label}, a tensor fewer", False, mac, changed(data(box[at + 1], tensors=[TENSOR])))
        judge(f"{label}, a number within three digits, for a program not declared so", False, mac,
              changed(data(box[at + 1], value=(1.5001).hex(), tensors=[TENSOR, other])))
        judge(f"{label}, a PASS line without its DATA line", False, mac, changed(None))
        without, mac_without = ([x for x in lines if x.split(" ")[1:2] != [victim]] for lines in (box, mac))
        judge(f"{label}, a program missing in the container", False, mac, without)
        judge(f"{label}, a program missing on the Mac", False, mac_without, box)
        judge(f"{label}, a program missing on both sides", False, mac_without, without)
        extra = ["PASS not_in_the_inventory 1.5", data(box[at + 1]).replace(victim, "not_in_the_inventory")]
        judge(f"{label}, a program that is not in the inventory, the same on both sides", False, mac + extra, box + extra)
        judge(f"{label}, a program that fails in the container", False, mac, box[:at] + [f"FAIL {victim} RuntimeError: no"] + box[at + 2:])
        fails = [f"FAIL {victim} RuntimeError: no"]
        judge(f"{label}, a program that fails on both sides alike, which is not declared", False, mac_without + fails, without + fails)
        judge(f"{label}, the text a model generated differs", False, mac, [x.replace("the same words", "other words") for x in box])
        judge(f"{label}, the kinds of error differ", False, mac, [x.replace("KINDS TypeError", "KINDS ValueError") for x in box])
        judge(f"{label}, the speed differs", True, mac, [x.replace("TIME 100", "TIME 50") for x in box])
        judge(f"{label}, only the first line of each side", False, mac[:1], box[:1])
        judge(f"{label}, the container ran another PyTorch", False, mac, [box[0].replace(f"torch={torch}", "torch=9.9.9")] + box[1:])

        # What a DATA line must hold to be compared in full, the same on both sides or not.
        def both(what, passes, **changes):
            line = data(box[at + 1], **changes)
            judge(f"{label}, {what}, the same on both sides", passes, changed(line, lines=mac, where=mac.index(box[at + 1])), changed(line))

        both("a number that is a NaN: float.hex does not hold its bits", False, value=float("nan").hex())
        both("a number that is an infinity", True, value=float("inf").hex())
        both("a number that is not as float.hex writes it", False, value="1.5")
        both("a number that is not text", False, value=1.5)
        both("a number that is nothing (null)", False, value=None)
        both("a tensor without the digest of its bytes (null)", False, tensors=[TENSOR, TENSOR[:2] + [None]])
        both("a tensor with a digest that is not one", False, tensors=[TENSOR, TENSOR[:2] + ["the same"]])
        both("a tensor without its type", False, tensors=[TENSOR, [None] + TENSOR[1:]])
        both("a tensor with a shape that is not one", False, tensors=[TENSOR, [TENSOR[0], "2x3", TENSOR[2]]])
        judge(f"{label}, 0.0 on the Mac and -0.0 in the container", False, changed(data(box[at + 1], value=(0.0).hex()), lines=mac, where=mac.index(box[at + 1])),
              changed(data(box[at + 1], value=(-0.0).hex())))
        for name, (kind, detail, _) in declared.items():
            where = box.index(next(x for x in box if x.split(" ")[1:2] == [name]))
            if kind == "close":
                near = data(box[where + 1], value=(1.5005).hex(), tensors=[TENSOR, other])
                judge(f"{label}, {name} (declared close): other bytes and a number within the tolerance", True, mac, changed(near, where=where + 1))
                far = data(box[where + 1], value=(1.6).hex(), tensors=[TENSOR, other])
                judge(f"{label}, {name} (declared close): a number beyond the tolerance", False, mac, changed(far, where=where + 1))
                shape = data(box[where + 1], tensors=[TENSOR, ["torch.float32", [3, 2], "1" * 64]])
                judge(f"{label}, {name} (declared close): another shape", False, mac, changed(shape, where=where + 1))
                # A tolerance is for numbers: an infinity or a NaN is within none, of a number or of another.
                inf, nan = float("inf"), float("nan")
                for what, passes, x, y in (("a number on the Mac and an infinity in the container", False, 1.5, inf),
                                           ("a number on the Mac and minus infinity in the container", False, 1.5, -inf),
                                           ("an infinity on the Mac and a number in the container", False, inf, 1.5),
                                           ("an infinity on the Mac and minus infinity in the container", False, inf, -inf),
                                           ("a number on the Mac and a NaN in the container", False, 1.5, nan),
                                           ("a NaN on the Mac and a number in the container", False, nan, 1.5),
                                           ("an infinity on both sides, and other bytes", False, inf, inf),
                                           ("a NaN on both sides, and other bytes", False, nan, nan),
                                           ("0 on the Mac and 5e-10 in the container: no digit in common", False, 0.0, 5e-10),
                                           ("0.0 on the Mac and -0.0 in the container, and other bytes", True, 0.0, -0.0)):
                    judge(f"{label}, {name} (declared close): {what}", passes,
                          changed(data(mac[mac.index(box[where + 1])], value=x.hex()), lines=mac, where=mac.index(box[where + 1])),
                          changed(data(box[where + 1], value=y.hex(), tensors=[TENSOR, other]), where=where + 1))
            elif kind == "fails on both":
                works = ["PASS " + name + " 1.5", "DATA " + name + " " + json.dumps({"value": (1.5).hex(), "tensors": [TENSOR]})]
                judge(f"{label}, {name} (declared to fail on both sides): works in the container", False, mac, box[:where] + works + box[where + 1:])
                judge(f"{label}, {name} (declared to fail on both sides): another error in the container", False, mac,
                      changed(f"FAIL {name} OtherError: no", where=where))
            elif kind == "container only":
                judge(f"{label}, {name} (declared to work in the container only): fails there too", False, mac,
                      box[:where] + [f"FAIL {name} {detail}: it must"] + box[where + 2:])
    battery, torch = "compat", "2.15.0"
    judge("compat against compat_train", False, torch_side(compare, "compat", torch, "mac"), torch_side(compare, "compat_train", torch, "box"))
    part = [torch_side(compare, battery, torch, side, only="layers") for side in ("mac", "box")]
    judge("a part of a battery (ONLY=layers on both sides), as expected", True, *part)
    judge("a part of a battery, with one of its programs missing", False, part[0], part[1][:1] + part[1][3:])
    judge("a part of a battery against the whole of it", False, part[0], torch_side(compare, battery, torch, "box"))
    judge("a part of a battery that holds no program", False, *[torch_side(compare, battery, torch, side, only="no_such_program") for side in ("mac", "box")])


# --- the fingerprints: what the comparators are given ------------------------------------------


def function(path, names, namespace):
    """Functions of a test program by name, without running the program: it would ask for a GPU."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), namespace)
    return [namespace.get(name) for name in names]


def check_fingerprints():
    try:
        import numpy as np
    except ImportError:
        print("  (no numpy in this Python: the fingerprint of mlx/tests/coverage.py is not checked)")
    else:
        mx = types.SimpleNamespace(array=type("NoArray", (), {}), bfloat16=object(), float32=object())
        _, fingerprint, checksum = function(os.path.join(MLX, "coverage.py"), ["leaves", "fingerprint", "checksum"],
                                            {"np": np, "mx": mx, "hashlib": hashlib})
        # The two arrays of the review: unequal, with the same checksum.
        a, b = np.array([2.0, 0.0]), np.array([0.0, 1.0])
        if fingerprint is None:
            check("mlx/tests/coverage.py has a fingerprint of the whole of a result", False, "no function fingerprint: only the checksum, which is the same for [2, 0] and [0, 1]")
        else:
            check("the checksum of coverage.py is the same for [2, 0] and [0, 1] (why it is not what is compared)", checksum(a) == checksum(b))
            check("the fingerprint of coverage.py differs for [2, 0] and [0, 1]", fingerprint(a) != fingerprint(b))
            check("and is the same for equal arrays", fingerprint([a, 1.5]) == fingerprint([a.copy(), 1.5]))
            check("and differs for the same bytes in another shape", fingerprint(np.zeros((2, 3))) != fingerprint(np.zeros((3, 2))))
            check("and differs for the same bytes as another type", fingerprint(np.zeros(4, np.float32)) != fingerprint(np.zeros(4, np.int32)))
            check("and differs for a value read that moves by one bit", fingerprint([a, 1.5]) != fingerprint([a, 1.5000000000000002]))
            check("and does not depend on how the array lies in memory",
                  fingerprint(np.arange(6.0).reshape(2, 3).T) == fingerprint(np.ascontiguousarray(np.arange(6.0).reshape(2, 3).T)))
    try:
        import torch
    except ImportError:
        print("  (no PyTorch in this Python: the fingerprint of pytorch/tests/exact.py is not checked)")
        return
    sys.path.insert(0, TORCH)
    try:
        import exact
    except ImportError as e:
        check("pytorch/tests/exact.py keeps the whole of a tensor", False, f"{type(e).__name__}: {e}")
        return
    finally:
        sys.path.pop(0)
    num, = function(os.path.join(TORCH, "compat.py"), ["num"], {"torch": torch, "note": exact.note})
    a, b = torch.tensor([2.0, 0.0]), torch.tensor([0.0, 1.0])

    def noted(value, *tensors):
        exact.begin()
        for t in tensors:
            num(t)
        return json.loads(exact.data(value))

    check("the number of compat.py is the same for [2, 0] and [0, 1] (why it is not what is compared)", num(a) == num(b))
    check("what exact.py keeps differs for [2, 0] and [0, 1]", noted(1.0, a) != noted(1.0, b))
    check("and is the same for equal tensors", noted(1.0, a, b) == noted(1.0, a.clone(), b.clone()))
    check("and differs for the same bytes in another shape", noted(1.0, torch.zeros(2, 3)) != noted(1.0, torch.zeros(3, 2)))
    check("and differs for the same bytes as another type", noted(1.0, torch.zeros(4)) != noted(1.0, torch.zeros(4, dtype=torch.int32)))
    check("and differs for a number that moves by one bit", noted(1.5, a) != noted(1.5000000000000002, a))
    check("and takes the tensors in order", noted(1.0, a, b) != noted(1.0, b, a))
    check("and does not depend on how the tensor lies in memory",
          noted(1.0, torch.arange(6.0).reshape(2, 3).t()) == noted(1.0, torch.arange(6.0).reshape(2, 3).t().contiguous()))


def check_tensor_diff(folder):
    """scripts/tensor-diff.py says by how much two runs differ: also where a side is not a number."""
    try:
        import numpy as np
    except ImportError:
        print("  (no numpy in this Python: scripts/tensor-diff.py is not checked)")
        return
    script = os.path.join(ROOT, "scripts", "tensor-diff.py")

    def differ(what, first, second, status, says=(), never=()):
        dirs = [os.path.join(folder, "dump-" + side) for side in ("a", "b")]
        for path, array in zip(dirs, (first, second)):
            os.makedirs(path, exist_ok=True)
            np.save(os.path.join(path, "program.000.npy"), np.array(array, np.float32))
            np.save(os.path.join(path, "program.001.npy"), np.arange(4, dtype=np.int32))  # one that follows, and is the same
        done = subprocess.run([sys.executable, "-B", script, *dirs], text=True, capture_output=True)
        text = done.stdout + done.stderr
        check(f"tensor-diff, {what}", done.returncode == status and all(word in text for word in says) and not any(word in text for word in never),
              f"status {done.returncode}: {text[-400:]}")

    nan, inf = float("nan"), float("inf")
    differ("the same tensors", [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 0, says=("2 tensors identical, 0 not",))
    differ("one element that differs", [1.0, 2.0, 3.0], [1.0, 2.5, 3.0], 1, says=("1 of 3 elements differ", "largest difference 0.5"))
    differ("a NaN against a number is not a difference of 0", [1.0, 2.0, 3.0], [1.0, nan, 3.0], 1, says=("NaN", "1 tensors identical, 1 not"),
           never=("largest difference 0", "Traceback"))
    differ("NaNs against numbers in every element", [1.0, 2.0], [nan, nan], 1, says=("NaN",), never=("Traceback",))
    differ("a NaN against a number beside a difference of 0.5", [1.0, 2.0, 3.0], [1.5, nan, 3.0], 1, says=("NaN", "largest difference 0.5"), never=("Traceback",))
    differ("an infinity against a number", [1.0, 2.0], [1.0, inf], 1, says=("largest difference inf",), never=("nan", "Warning", "Traceback"))
    differ("an infinity against minus infinity", [inf, 2.0], [-inf, 2.0], 1, says=("largest difference inf",), never=("nan", "Warning", "Traceback"))
    differ("the same infinity beside a difference of 0.5", [inf, 2.0], [inf, 2.5], 1, says=("largest difference 0.5",), never=("nan", "Warning", "Traceback"))
    differ("0.0 against -0.0 is a difference", [0.0, 2.0], [-0.0, 2.0], 1, says=("1 of 2 elements differ",))


def main():
    with tempfile.TemporaryDirectory(prefix="comparators-") as folder:
        for title, part in (("MLX: mlx/tests/compare.py", check_mlx), ("MLX: mlx/tests/compare_bge.py", check_bge),
                            ("PyTorch: pytorch/tests/compat_compare.py", check_torch)):
            print(title)
            part(folder)
        print("scripts/tensor-diff.py")
        check_tensor_diff(folder)
    print("fingerprints")
    check_fingerprints()
    if failures:
        print(f"{len(failures)} of {checks} checks FAILED")
        return 1
    print(f"all {checks} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
