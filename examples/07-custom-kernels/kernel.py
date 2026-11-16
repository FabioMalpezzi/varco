"""An MLX program with a Metal kernel of its own (mx.fast.metal_kernel).

Libraries do this for speed: mlx-lm computes some layers of some models with
kernels of its own when mx.metal.is_available() says it can, and with
ordinary operations otherwise. This program does the same with a small one.
    python kernel.py
In a container the answer depends on the bridge's server: it takes custom
kernels only if it was started with --custom-kernels.
"""
import mlx.core as mx

print(f"device {mx.default_device()}; mx.metal.is_available(): {mx.metal.is_available()}")
a = (mx.arange(4096) * 0.01 - 20.0).astype(mx.float32)
expected = a / (1 + mx.abs(a))

if mx.metal.is_available():
    kernel = mx.fast.metal_kernel(
        name="softsign",
        input_names=["a"],
        output_names=["out"],
        source="""
            uint i = thread_position_in_grid.x;
            T x = a[i];
            out[i] = x / (T(1) + metal::abs(x));
        """,
    )
    out = kernel(inputs=[a], template=[("T", mx.float32)], grid=(a.size, 1, 1), threadgroup=(256, 1, 1),
                 output_shapes=[a.shape], output_dtypes=[a.dtype])[0]
    print("computed by the custom kernel")
else:
    out = expected
    print("custom kernels are not available here: computed with ordinary operations")
    try:
        mx.fast.metal_kernel(name="softsign", input_names=["a"], output_names=["out"], source="out[0] = a[0];")(
            inputs=[a], grid=(1, 1, 1), threadgroup=(1, 1, 1), output_shapes=[(1,)], output_dtypes=[mx.float32])
    except Exception as error:  # what a program that insists is told
        print(f"a kernel all the same: {type(error).__name__}: {error}")

print(f"sum {out.sum().item():.6f}; largest difference from the ordinary operations {mx.abs(out - expected).max().item():.2e}")
