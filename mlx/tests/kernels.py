"""Custom Metal kernels (mx.fast.metal_kernel) through the remote gpu backend.

Each case runs kernels and gives the bytes of what they compute: natively and
in a container whose server takes kernels they must be the same bytes.

    python kernels.py OUTPUT.json                 the cases
    python kernels.py compare NATIVE.json CONTAINER.json
    python kernels.py off                         in a container whose server
                                                  does not take kernels: what
                                                  a program is told
"""

import hashlib
import json
import os
import platform
import sys
import tempfile

import numpy as np

import mlx.core as mx

CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def kernel(name, inputs, outputs, source, **options):
    return mx.fast.metal_kernel(name=name, input_names=inputs, output_names=outputs, source=source, **options)


def run(k, inputs, n, shapes=None, dtypes=None, threadgroup=None, **options):
    """The kernel over a grid of n threads; one float32 output of n elements unless said."""
    return k(inputs=inputs, grid=(n, 1, 1), threadgroup=threadgroup or (min(n, 32), 1, 1),
             output_shapes=shapes or [(n,)], output_dtypes=dtypes or [mx.float32], **options)


def ramp(n, dtype=mx.float32):
    return (mx.arange(n) * 0.37 + 0.11).astype(dtype)


I = "uint i = thread_position_in_grid.x;\n"


@case
def basic():
    k = kernel("basic", ["a"], ["o"], I + "o[i] = a[i] * 2.0f + 1.0f;")
    return run(k, [ramp(16)], 16)


@case
def no_inputs():
    k = kernel("iota", [], ["o"], I + "o[i] = float(i) * 0.25f;")
    return run(k, [], 16)


@case
def two_outputs():
    k = kernel("two_outputs", ["a"], ["root", "count"],
               I + "root[i] = metal::sqrt(a[i]); count[i] = int(a[i] * 10.0f) * 3 - 7;")
    return run(k, [ramp(24)], 24, shapes=[(24,), (2, 12)], dtypes=[mx.float32, mx.int32])


@case
def template_arguments():
    k = kernel("templated", ["a"], ["o"],
               I + "T x = static_cast<T>(a[i]); o[i] = FLAG ? static_cast<T>(x * N) : x;")
    out = []
    for dtype, n, flag in ((mx.float16, 3, True), (mx.float32, -2, True), (mx.float16, 3, False)):
        out += run(k, [ramp(16)], 16, dtypes=[dtype], template=[("T", dtype), ("N", n), ("FLAG", flag)])
    return out


@case
def init_value():
    k = kernel("half_written", ["a"], ["o", "n"], I + "if (i % 2 == 0) { o[i] = a[i]; n[i] = int(i); }")
    return run(k, [ramp(16)], 16, shapes=[(16,), (16,)], dtypes=[mx.float32, mx.int32], init_value=7.5)


@case
def atomic_outputs():
    k = kernel("histogram", ["bins"], ["hist"],
               I + "atomic_fetch_add_explicit(&hist[bins[i] % 8u], 1u, memory_order_relaxed);",
               atomic_outputs=True)
    bins = (mx.arange(1000) * 7 % 13).astype(mx.uint32)
    return run(k, [bins], 1000, shapes=[(8,)], dtypes=[mx.uint32], init_value=0)


@case
def row_contiguous():
    # The transposed input is copied before the kernel sees it.
    k = kernel("in_order", ["x"], ["o"], I + "o[i] = x[i];")
    a = ramp(12).reshape(3, 4)
    return run(k, [a.T], 12) + run(k, [a[:, ::2]], 6)


@case
def strides_and_shape():
    # The input as it lies in memory, with its shape and strides.
    k = kernel("by_strides", ["x"], ["o"],
               I + "uint r = i / x_shape[1], c = i % x_shape[1];\n"
               "o[i] = x[r * x_strides[0] + c * x_strides[1]] + x_ndim;",
               ensure_row_contiguous=False)
    a = ramp(12).reshape(3, 4)
    b = mx.array(np.arange(20, dtype=np.float32).reshape(4, 5))
    return run(k, [a.T[::2]], 6) + run(k, [a], 12) + run(k, [b.T], 20)


@case
def as_it_lies():
    # Not copied and read without the strides: the order of the buffer.
    k = kernel("raw", ["x"], ["o"], I + "o[i] = x[i];", ensure_row_contiguous=False)
    a = ramp(12).reshape(3, 4)
    return run(k, [a.T], 12)


@case
def contiguous_first():
    # mx.contiguous makes the copy that the kernel then relies on.
    k = kernel("raw_after_contiguous", ["x"], ["o"], I + "o[i] = x[i];", ensure_row_contiguous=False)
    a = ramp(12).reshape(3, 4)
    lazy = run(k, [mx.contiguous(a.T)], 12)
    b = mx.contiguous(a.T)
    mx.eval(b)
    return lazy + run(k, [b], 12) + [mx.contiguous(a.T[::2])]


@case
def scalars():
    k = kernel("scalars", ["a", "f", "n", "flag", "w"], ["o"],
               I + "o[i] = a[i] * f + n + (flag ? w[i % 3] : 0.0f);")
    w = mx.array([1.0, 2.0, 3.0])
    return (run(k, [ramp(16), 2.5, 3, True, w], 16) + run(k, [ramp(16), -1.0, 40, False, w], 16)
            + run(k, [ramp(16), mx.array(0.5), mx.array(7), mx.array(True), w], 16))


@case
def grid_3d():
    k = kernel("grid_3d", ["a"], ["o"],
               "uint3 p = thread_position_in_grid;\n"
               "uint idx = (p.z * 3 + p.y) * 4 + p.x;\n"
               "o[idx] = int(p.x + 10 * p.y + 100 * p.z) + int(threads_per_grid.x * threads_per_grid.y"
               " * threads_per_grid.z) + int(a[idx]);")
    return k(inputs=[ramp(24)], grid=(4, 3, 2), threadgroup=(2, 1, 1), output_shapes=[(2, 3, 4)],
             output_dtypes=[mx.int32])


@case
def threadgroups():
    k = kernel("in_threadgroup", ["a"], ["o"],
               I + "o[i] = int(thread_position_in_threadgroup.x) + 1000 * int(threads_per_threadgroup.x)"
               " + int(a[i]);")
    a = ramp(64)
    return (run(k, [a], 64, dtypes=[mx.int32], threadgroup=(16, 1, 1))
            + run(k, [a], 64, dtypes=[mx.int32], threadgroup=(64, 1, 1))
            # Larger than the grid: mlx cuts it down.
            + run(k, [a[:4]], 4, dtypes=[mx.int32], threadgroup=(64, 1, 1)))


@case
def threadgroup_too_large():
    k = kernel("too_large", ["a"], ["o"], I + "o[i] = a[i];")
    return run(k, [ramp(4096)], 4096, threadgroup=(2048, 1, 1))


def math_mode(mode):
    k = kernel("math_" + mode, ["a", "b", "c"], ["o", "p"],
               I + "float x = a[i], y = b[i];\n"
               "o[i] = x * y + c[i];\n"
               "p[i] = (x + 1.0e8f) - 1.0e8f + metal::sin(x) / y;",
               compile_options={"math_mode": mode})
    n = 1024
    return run(k, [ramp(n), ramp(n) * 1.7 + 0.3, ramp(n) * -0.9], n, shapes=[(n,), (n,)],
               dtypes=[mx.float32, mx.float32])


@case
def math_safe():
    return math_mode("safe")


@case
def math_relaxed():
    return math_mode("relaxed")


@case
def math_fast():
    return math_mode("fast")


@case
def header():
    k = kernel("with_header", ["a"], ["o"], I + "o[i] = twice(a[i]) + OFFSET;",
               header="inline float twice(float x) { return 2.0f * x; }\n#define OFFSET 3.0f\n")
    return run(k, [ramp(16)], 16)


@case
def types():
    k = kernel("typed", ["a"], ["o"], I + "o[i] = a[i];")
    out = []
    for dtype in (mx.float16, mx.bfloat16, mx.int8, mx.uint8, mx.int16, mx.uint16, mx.int32,
                  mx.uint32, mx.int64, mx.uint64, mx.bool_):
        out += run(k, [(mx.arange(16) * 3 - 5).astype(dtype)], 16, dtypes=[dtype])
    return out


@case
def sizes():
    # Fewer than 8 elements are passed another way, and compiled apart.
    k = kernel("sized", ["a"], ["o"], I + "o[i] = a[i] + 1.0f;")
    return run(k, [ramp(16)], 16) + run(k, [ramp(4)], 4) + run(k, [ramp(1)], 1)


@case
def compiled():
    k = kernel("in_compiled", ["a"], ["o"], I + "o[i] = a[i] * a[i];")

    @mx.compile
    def f(x):
        return mx.exp(run(k, [x * 2], 16)[0] * 0.01) + 1

    a = ramp(16)
    return [f(a), f(a + 1), f(a + 2)]


@case
def chained():
    k1 = kernel("chain_a", ["a"], ["o"], I + "o[i] = a[i] + float(i);")
    k2 = kernel("chain_b", ["a", "b"], ["o", "p"], I + "o[i] = a[i] - b[i]; p[i] = a[i] * b[i];")
    a = ramp(64)
    y = run(k1, [a], 64)[0]
    z = (y.reshape(8, 8) @ a.reshape(8, 8)).reshape(64)
    o, p = run(k2, [z, y], 64, shapes=[(64,), (64,)], dtypes=[mx.float32, mx.float32])
    return [o, p, run(k1, [mx.tanh(o + p)], 64)[0]]


@case
def repeated():
    # The same kernel many times in one graph, then once per evaluation.
    k = kernel("step", ["a", "s"], ["o"], I + "o[i] = a[i] * 0.99f + s;")
    lazy = ramp(32)
    for step in range(50):
        lazy = run(k, [lazy, float(step)], 32)[0]
    each = ramp(32)
    for step in range(50):
        each = run(k, [each, float(step)], 32)[0]
        mx.eval(each)
    return [lazy, each]


@case
def made_again():
    # A program that makes its kernel at every call: sent once all the same.
    a = ramp(16)
    for step in range(300):
        k = kernel("made_again", ["a"], ["o"], I + "o[i] = a[i] + 0.5f;")
        a = run(k, [a], 16)[0]
        mx.eval(a)
    return [a]


@case
def same_name():
    k1 = kernel("same_name", ["a"], ["o"], I + "o[i] = a[i] + 1.0f;")
    k2 = kernel("same_name", ["a"], ["o"], I + "o[i] = a[i] + 2.0f;")
    a = ramp(16)
    return run(k1, [a], 16) + run(k2, [a], 16) + run(k1, [a], 16)


@case
def large():
    n = 1 << 20
    k = kernel("large", ["a"], ["o"], I + "o[i] = a[i] * 0.5f + float(i % 7u);")
    return run(k, [mx.arange(n).astype(mx.float32)], n, threadgroup=(256, 1, 1))


@case
def on_the_gpu_stream():
    k = kernel("on_stream", ["a"], ["o"], I + "o[i] = a[i] - 1.0f;")
    mx.set_default_device(mx.cpu)
    try:
        return run(k, [ramp(16)], 16) + run(k, [ramp(16)], 16, stream=mx.gpu)
    finally:
        mx.set_default_device(mx.gpu)


@case
def verbose():
    # What metal_kernel prints of the source it generated.
    k = kernel("spoken", ["a", "s"], ["o"], I + "o[i] = T(a[i]) + s;", header="// a header\n")
    sys.stdout.flush()
    with tempfile.TemporaryFile() as f:
        saved = os.dup(1)
        os.dup2(f.fileno(), 1)
        try:
            out = run(k, [ramp(16), 2.0], 16, template=[("T", mx.float32)], verbose=True)
            mx.eval(out)
        finally:
            os.dup2(saved, 1)
            os.close(saved)
        f.seek(0)
        text = f.read()
    assert b"custom_kernel_spoken" in text, text
    return out + [text]


@case
def syntax_error():
    k = kernel("broken", ["a"], ["o"], I + "o[i] = ;")
    return run(k, [ramp(16)], 16)


@case
def after_syntax_error():
    k = kernel("mended", ["a"], ["o"], I + "o[i] = a[i];")
    return run(k, [ramp(16)], 16)


@case
def bad_name():
    k = kernel("not a name", ["a"], ["o"], I + "o[i] = a[i];")
    return run(k, [ramp(16)], 16)


@case
def wrong_number_of_inputs():
    k = kernel("one_input", ["a"], ["o"], I + "o[i] = a[i];")
    return run(k, [ramp(16), ramp(16)], 16)


@case
def on_the_cpu():
    k = kernel("on_cpu", ["a"], ["o"], I + "o[i] = a[i];")
    return run(k, [ramp(16)], 16, stream=mx.cpu)


@case
def include_refused():
    k = kernel("includes", ["a"], ["o"], I + "o[i] = a[i];", header='#include "/etc/hosts"\n')
    return run(k, [ramp(16)], 16)


@case
def pragma_refused():
    k = kernel("asks", ["a"], ["o"], I + "o[i] = a[i];", header='_Pragma("GCC dependency \\"/etc/hosts\\"")\n')
    return run(k, [ramp(16)], 16)


@case
def source_too_large():
    k = kernel("too_long", ["a"], ["o"], I + "o[i] = a[i];" + "/" * (300 << 10))
    return run(k, [ramp(16)], 16)


@case
def imported_function():
    k = kernel("exported", ["a"], ["o"], I + "o[i] = a[i] * 3.0f;")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "f.mlxfn")
        mx.export_function(path, lambda x: run(k, [x], 16)[0], ramp(16))
        return [mx.import_function(path)(ramp(16))[0]]


@case
def after_everything():
    k = kernel("last", ["a"], ["o"], I + "o[i] = a[i] + 100.0f;")
    return run(k, [ramp(16)], 16)


def digest(x):
    if isinstance(x, bytes):
        return {"text": hashlib.sha256(x).hexdigest(), "bytes": len(x)}
    mx.eval(x)
    # The bytes as they are, whatever the type (numpy has no bfloat16).
    raw = np.array(mx.view(mx.contiguous(x).reshape(-1), mx.uint8)).tobytes() if x.size else b""
    return {"dtype": str(x.dtype), "shape": list(x.shape), "sha256": hashlib.sha256(raw).hexdigest()}


def run_cases(path):
    results = {}
    for fn in CASES:
        try:
            results[fn.__name__] = [digest(x) for x in fn()]
        except Exception as e:  # noqa: BLE001
            results[fn.__name__] = f"{type(e).__name__}: {str(e).strip().splitlines()[0][:300]}"
    info = {"device": str(mx.default_device()), "mlx": mx.__version__, "platform": sys.platform,
            "machine": platform.machine(), "metal.is_available": mx.metal.is_available()}
    json.dump({"info": info, "results": results}, open(path, "w"), indent=1)
    raised = [name for name, r in results.items() if isinstance(r, str)]
    print(f"{len(results)} cases on {info['device']} ({info['platform']}), mx.metal.is_available() "
          f"{info['metal.is_available']}, {len(raised)} raised: {raised}")


def off():
    """With a server that does not take kernels a library takes its other path.
    0 when that is how it went: no kernels on offer, a kernel refused in so
    many words, and ordinary operations as ever."""
    available = mx.metal.is_available()
    print("mx.metal.is_available():", available)
    ok = not available
    try:
        print("a kernel all the same: ok", basic()[0].sum().item())
        ok = False
    except Exception as e:  # noqa: BLE001
        print(f"a kernel all the same: {type(e).__name__}: {e}")
        ok = ok and "Custom Metal kernels are off" in str(e)
    print("ordinary operations: ok", (ramp(16) * 2.0 + 1.0).sum().item())
    if not ok:
        print("NOT as expected of a server started without --custom-kernels")
    return 0 if ok else 1


if __name__ == "__main__":
    if sys.argv[1] == "off":
        sys.exit(off())
    elif sys.argv[1] == "compare":
        # In compare.py with the other stages' comparisons: the inventory of
        # the cases, and the ones that are meant to raise or to be refused.
        import compare
        sys.exit(compare.main(["kernels", sys.argv[2], sys.argv[3]]))
    else:
        run_cases(sys.argv[1])
