"""Edge cases of the remote gpu backend: what is refused and how errors show.

Prints one line per case, and with OUTPUT.json writes them down too; run it
natively and in the container and compare the two files with
`compare.py edge`.

Usage: python edge.py [OUTPUT.json]
"""

import json
import sys

import mlx.core as mx

RESULTS = {}


def case(name, fn):
    try:
        r = fn()
        print(f"{name}: ok {r}")
        RESULTS[name] = {"ok": repr(r)}
    except Exception as e:  # noqa: BLE001
        text = f"{type(e).__name__}: {str(e).splitlines()[0][:230]}"
        print(f"{name}: {text}")
        RESULTS[name] = {"raised": text}


def custom_function():
    @mx.custom_function
    def f(x):
        return mx.sin(x) * 2

    @f.vjp
    def f_vjp(x, out, cot):
        return cot * 5

    x = mx.arange(4.0)
    return [f(x).sum().item(), mx.grad(lambda x: f(x).sum())(x).sum().item()]


def float64_on_gpu():
    a = mx.arange(4.0).astype(mx.float64)
    return (a * 2).sum().item()


def after_error():
    return (mx.arange(4.0) * 2).sum().item()


def huge_allocation():
    a = mx.zeros((300000, 300000)) + 1
    return a.sum().item()


def after_huge_allocation():
    return (mx.arange(5.0) * 3).sum().item()


def segmented_mm():
    a = mx.ones((4, 8))
    b = mx.ones((8, 3))
    s = mx.array([[0, 4], [4, 8]], dtype=mx.uint32)
    return mx.segmented_mm(a, b, s).sum().item() if hasattr(mx, "segmented_mm") else "absent"


def fp8():
    if not hasattr(mx, "to_fp8"):
        return "absent"
    return mx.from_fp8(mx.to_fp8(mx.arange(4.0))).sum().item()


def metal_kernel():
    k = mx.fast.metal_kernel(name="k", input_names=["a"], output_names=["o"],
                             source="uint elem = thread_position_in_grid.x; o[elem] = a[elem] * 2;")
    a = mx.arange(4.0)
    return k(inputs=[a], grid=(4, 1, 1), threadgroup=(4, 1, 1), output_shapes=[(4,)],
             output_dtypes=[mx.float32])[0].sum().item()


def compiled_on_cpu_default():
    mx.set_default_device(mx.cpu)
    try:
        f = mx.compile(lambda x: mx.exp(x) + 1)
    finally:
        mx.set_default_device(mx.gpu)
    return f(mx.arange(3.0)).sum().item()


def load_on_gpu_stream():
    import os, tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.safetensors")
        mx.save_safetensors(p, {"a": mx.arange(5.0)})
        return (mx.load(p, stream=mx.gpu)["a"] * 2).sum().item()


def threads():
    import threading
    out = []

    def work(i):
        out.append(((mx.arange(1000.0) + i).sum().item(), i))

    ts = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    return sorted(out, key=lambda t: t[1])


def memory_info():
    a = mx.ones((1000, 1000)) * 2
    mx.eval(a)
    return [mx.get_active_memory() >= 0, mx.device_info().get("device_name", "?")[:20] != ""]


for name, fn in list(globals().items()):
    if callable(fn) and fn.__module__ == "__main__" and name != "case":
        case(name, fn)
if len(sys.argv) > 1:
    info = {"device": str(mx.default_device()), "mlx": mx.__version__, "platform": sys.platform}
    with open(sys.argv[1], "w") as f:
        json.dump({"info": info, "results": RESULTS}, f, indent=1)
