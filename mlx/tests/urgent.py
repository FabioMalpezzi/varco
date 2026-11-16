"""Values a program reads while a long evaluation is in flight.

A model's step is started with async_eval and the program goes on: it reads
an element of an array it already has, compares two tokens, takes a few
items of a list. Natively each of these reads waits for the GPU to finish
what was started. Through the bridge those that only select, move or compare
elements are answered by the Mac's cpu at once (the urgent graphs of
mlx/mlx/backend/remote/protocol.h): the values must be the ones the GPU
gives, and the reads that the GPU has to compute must still wait their turn.
    python urgent.py RESULT.json
    python urgent.py compare NATIVE.json CONTAINER.json
"""
import json
import sys
import time

import mlx.core as mx


def main(out):
    ints = mx.arange(1000).reshape(10, 100) * 3
    floats = (mx.arange(1000) * 0.5).reshape(10, 100)
    wide = mx.arange(40000).reshape(200, 200) + 1
    matrix = mx.zeros((1024, 1024)) + 0.001
    mx.eval(ints, floats, wide, matrix)

    def work(n):
        a = matrix
        for _ in range(n):
            a = (a @ matrix) * 0.5 + 0.25
        return a

    # About 150 ms of work for the GPU.
    mx.eval(work(50))
    start = time.perf_counter()
    mx.eval(work(200))
    steps = max(20, int(200 * 0.15 / (time.perf_counter() - start)))

    # (name, what the program reads, whether the cpu can answer it)
    cases = [
        ("an element", lambda w: ints[3, 7].item(), True),
        ("an element from the end", lambda w: ints[-1, -1].item(), True),
        ("a slice", lambda w: ints[2, 10:15].tolist(), True),
        ("a slice with a step", lambda w: ints[5, 90:10:-20].tolist(), True),
        ("an element of floating point", lambda w: floats[4, 4].item(), True),
        ("the items of a list", lambda w: floats[2][[1, 5, 9]].tolist(), True),
        ("rows by an array of indices", lambda w: mx.take(ints, mx.array([3, 1]), axis=0)[:, 0].tolist(), True),
        ("an element of the transpose", lambda w: ints.T[5, 2].item(), True),
        ("an element after a reshape", lambda w: ints.reshape(100, 10)[99, 9].item(), True),
        ("a comparison of integers", lambda w: (ints[0, :4] == 6).tolist(), True),
        ("any of a comparison", lambda w: mx.any(ints == 2997).item(), True),
        ("all of a comparison", lambda w: mx.all(ints >= 0).item(), True),
        ("a concatenation", lambda w: mx.concatenate([ints[0, :2], ints[9, -2:]]).tolist(), True),
        ("integers of another size", lambda w: ints[1, :3].astype(mx.int64).tolist(), True),
        ("a choice between two rows", lambda w: mx.where(ints[0, :4] > 3, ints[0, :4], ints[1, :4]).tolist(), True),
        ("a sum", lambda w: floats.sum().item(), False),
        ("a comparison of floating point", lambda w: (floats > 100).sum().item(), False),
        ("the place of the largest", lambda w: mx.argmax(floats).item(), False),
        ("floating point made integer", lambda w: floats[0, :3].astype(mx.int32).tolist(), False),
        ("a large result", lambda w: sum(sum(row) for row in wide[:150].tolist()), False),
        ("a value of the work in flight", lambda w: round(w[0, 0].item(), 3), False),
    ]
    report = {"device": str(mx.default_device()), "platform": sys.platform, "steps": steps, "cases": {}}
    # The first read of a program waits like any other evaluation: it is what
    # tells the Mac that this program reads values while its work runs.
    w = work(steps)
    start = time.perf_counter()
    mx.async_eval(w)
    ints[0, 0].item()
    took = time.perf_counter() - start
    mx.eval(w)
    report["first"] = {"read_ms": 1e3 * took, "work_ms": 1e3 * (time.perf_counter() - start)}
    for name, read, cpu in cases:
        w = work(steps)
        # (Natively async_eval itself waits for the GPU when the work is long.)
        start = time.perf_counter()
        mx.async_eval(w)
        value = read(w)
        took = time.perf_counter() - start
        mx.eval(w)
        total = time.perf_counter() - start
        report["cases"][name] = {"value": value, "read_ms": 1e3 * took, "work_ms": 1e3 * total, "cpu": cpu}

    # Many reads in a row, each of another element.
    w = work(steps)
    start = time.perf_counter()
    mx.async_eval(w)
    values = [ints[i % 10, (7 * i) % 100].item() for i in range(40)]
    took = time.perf_counter() - start
    mx.eval(w)
    total = time.perf_counter() - start
    report["cases"]["40 elements, one by one"] = {"value": values, "read_ms": 1e3 * took, "work_ms": 1e3 * total, "cpu": True}

    # synchronize() waits for everything, also after a read that did not.
    w = work(steps)
    start = time.perf_counter()
    mx.async_eval(w)
    value = ints[3, 7].item()
    mx.synchronize()
    waited = time.perf_counter() - start
    mx.eval(w)
    total = time.perf_counter() - start
    report["synchronize"] = {"value": value, "waited_ms": 1e3 * waited, "work_ms": 1e3 * total}

    # eval() of what is in flight waits for it, also after such a read.
    w = work(steps)
    start = time.perf_counter()
    mx.async_eval(w)
    value = ints[3, 7].item()
    mx.eval(w)
    waited = time.perf_counter() - start
    corner = w[0, 0].item()
    total = time.perf_counter() - start
    report["eval"] = {"value": [value, round(corner, 3)], "waited_ms": 1e3 * waited, "work_ms": 1e3 * total}

    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    early = [c["read_ms"] < 0.5 * c["work_ms"] for c in report["cases"].values()]
    print(f"{len(early)} reads on {report['device']} ({sys.platform}) during {steps} steps of work "
          f"(about {report['synchronize']['work_ms']:.0f} ms): {sum(early)} came before the work was half done")


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        # In compare.py with the other stages' comparisons.
        import compare
        sys.exit(compare.main(["urgent", sys.argv[2], sys.argv[3]]))
    main(sys.argv[1])
