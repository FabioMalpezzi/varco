"""Coverage of the remote gpu backend: what each test computes, whole.

Run it natively on the Mac and in the container, then compare the two JSON
files with compare.py. Each test builds a result with mlx on the default
device. What is kept of it is a fingerprint of the bytes that come back: for
every array its type, its shape and a SHA-256 of all its bytes, so that a
wrong value, a wrong order, a wrong shape and a wrong dtype all show, each
by itself. The checksum next to it is one number to read, and is not what is
compared: two different arrays can give the same one.

Usage: python coverage.py OUTPUT.json
With DUMP=directory in the environment the arrays themselves are written
there too (test.NNN.npy), for scripts/tensor-diff.py.
"""

import hashlib
import json
import os
import sys
import tempfile

import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def leaves(x):
    """A result as a flat list of what it holds: (type, numpy array) for every
    array, (None, value) for every Python value."""
    if isinstance(x, (list, tuple)):
        return [leaf for v in x for leaf in leaves(v)]
    if isinstance(x, mx.array):
        # numpy has no bfloat16: as float32, which holds every bfloat16 exactly.
        return [(str(x.dtype), np.array(x.astype(mx.float32) if x.dtype == mx.bfloat16 else x))]
    if isinstance(x, (np.ndarray, np.generic)):
        return [("numpy." + str(x.dtype), np.asarray(x))]
    return [(None, x)]


def fingerprint(x):
    """What a result holds, in full: for every array its type, its shape and
    a digest of all its bytes in row-major order, and every Python value as
    it is (a float in hexadecimal, bit for bit)."""
    return [{"dtype": dtype, "shape": list(a.shape), "sha256": hashlib.sha256(a.tobytes()).hexdigest()} if dtype
            else {"python": type(a).__name__, "value": a.hex() if isinstance(a, float) else a}
            for dtype, a in leaves(x)]


def checksum(x):
    """A float that depends on every element and on its position: one number
    to read, not a proof that two results are the same."""
    if isinstance(x, (list, tuple)):
        return float(sum((i + 1) * checksum(v) for i, v in enumerate(x)))
    if isinstance(x, mx.array):
        if x.dtype == mx.bfloat16:
            x = x.astype(mx.float32)
        x = np.array(x)
    a = np.asarray(x)
    if a.dtype.kind == "c":
        a = np.stack([a.real, a.imag])
    a = a.astype(np.float64).reshape(-1)
    w = 1.0 + (np.arange(a.size) % 7)
    return float((a * w).sum() + 0.001 * a.size)


def base(shape=(4, 6), seed=1):
    mx.random.seed(seed)
    return mx.random.normal(shape)


# --- arithmetic -------------------------------------------------------------


@test
def arith_basic():
    a, b = base(), base(seed=2)
    return [a + b, a - b, a * b, a / (mx.abs(b) + 1.0), -a]


@test
def arith_unary_math():
    a = base()
    p = mx.abs(a) + 0.5
    return [mx.exp(a), mx.log(p), mx.sqrt(p), mx.rsqrt(p), mx.square(a),
            mx.sin(a), mx.cos(a), mx.tanh(a), mx.sigmoid(a), mx.erf(a),
            mx.log1p(p), mx.expm1(a), mx.arctan(a), mx.sinh(a)]


@test
def arith_power_remainder():
    a = mx.abs(base()) + 0.1
    b = mx.arange(1, 25).reshape(4, 6)
    return [a ** 2.5, mx.power(a, a), b % 5, b // 4, mx.remainder(a, 0.3),
            mx.divmod(b, 7)[0], mx.divmod(b, 7)[1]]


@test
def arith_minmax_clip_round():
    a, b = base(), base(seed=3)
    return [mx.maximum(a, b), mx.minimum(a, b), mx.clip(a, -0.5, 0.5),
            mx.floor(a * 3), mx.ceil(a * 3), mx.round(a * 3), mx.sign(a)]


@test
def arith_scalars_python():
    a = base()
    return [a * 2.0 + 1, 3 - a, 1.0 / (a * a + 1), a > 0.25]


@test
def arith_logaddexp():
    a, b = base(), base(seed=4)
    return mx.logaddexp(a, b)


# --- broadcasting -----------------------------------------------------------


@test
def broadcast_shapes():
    a = base((3, 1, 4))
    b = base((5, 1), seed=5)
    c = base((4,), seed=6)
    return [a + b, a * c, (a + b) * c]


@test
def broadcast_to_and_full():
    a = base((1, 6))
    return [mx.broadcast_to(a, (4, 6)) + 0, mx.full((2, 3), 7.5), mx.ones((2, 2)),
            mx.zeros((3,)), mx.eye(4), mx.tri(4), mx.linspace(0, 1, 9),
            mx.arange(0, 10, 3)]


# --- reductions -------------------------------------------------------------


@test
def reduce_basic():
    a = base((3, 4, 5))
    return [a.sum(), a.sum(axis=1), a.mean(axis=(0, 2)), a.max(axis=-1),
            a.min(axis=0, keepdims=True), a.prod(axis=2), a.var(axis=1),
            mx.std(a, axis=0)]


@test
def reduce_arg_and_logical():
    a = base((3, 4, 5))
    return [mx.argmax(a, axis=1), mx.argmin(a, axis=-1), mx.argmax(a),
            (a > 0).all(axis=0), (a > 1.5).any(axis=2), mx.logsumexp(a, axis=1)]


@test
def reduce_cumulative():
    a = base((4, 6))
    return [mx.cumsum(a, axis=1), mx.cumprod(a, axis=0), mx.cummax(a, axis=1),
            mx.cummin(a, axis=0), mx.cumsum(a, axis=1, reverse=True)]


# --- indexing and slicing ---------------------------------------------------


@test
def index_slices():
    a = base((5, 6, 4))
    return [a[1:3], a[:, ::2], a[..., -1], a[2], a[1:4, 2:5, 1], a[::-1],
            a[None, 0], a[:, :, 1:-1]]


@test
def index_arrays():
    a = base((5, 6, 4))
    i = mx.array([4, 0, 2])
    j = mx.array([1, 5, 3])
    return [a[i], a[i, j], a[:, j], mx.take(a, i, axis=2),
            mx.take_along_axis(a, mx.argsort(a, axis=1), axis=1)]


@test
def index_updates():
    a = base((5, 6))
    b = mx.array(a)
    b[1:3, ::2] = 9.0
    c = mx.array(a)
    c[mx.array([0, 4]), mx.array([1, 2])] = mx.array([-7.0, 7.0])
    d = a.at[mx.array([0, 0, 1])].add(1.0)
    return [b, c, d]


@test
def index_where_and_masks():
    a = base((5, 6))
    return [mx.where(a > 0, a, -1.0), mx.where(a > 0, 1, 0), a * (a < 0)]


# --- shapes -----------------------------------------------------------------


@test
def shape_concat_split_stack():
    a, b = base((3, 4)), base((3, 4), seed=7)
    parts = mx.split(mx.concatenate([a, b], axis=1), 4, axis=1)
    uneven = mx.split(a, [1, 3], axis=1)
    return [mx.concatenate([a, b], axis=0), mx.stack([a, b], axis=1), *parts,
            *uneven]


@test
def shape_reshape_transpose():
    a = base((2, 3, 4))
    return [a.reshape(4, 6), a.transpose(2, 0, 1), mx.swapaxes(a, 0, 2), a.T,
            mx.expand_dims(a, 1), mx.squeeze(a[:1]), a.flatten(),
            mx.moveaxis(a, 0, -1)]


@test
def shape_pad_tile_repeat():
    a = base((2, 3))
    return [mx.pad(a, ((1, 2), (0, 1))), mx.tile(a, (2, 2)),
            mx.repeat(a, 3, axis=1), mx.tril(base((4, 4))), mx.triu(base((4, 4))),
            mx.diag(base((4, 4))), mx.roll(a, 1, axis=1)]


@test
def shape_contiguous_and_views():
    a = base((4, 6))
    return [mx.contiguous(a.T), mx.as_strided(a, (2, 3), (6, 2), 1),
            mx.view(mx.arange(8, dtype=mx.int32), mx.int16)]


# --- linear algebra ---------------------------------------------------------


@test
def matmul_basic():
    a, b = base((4, 6)), base((6, 5), seed=8)
    return [a @ b, mx.matmul(a, a.T), a @ b[:, 0]]


@test
def matmul_batched_and_addmm():
    a, b = base((2, 3, 4, 6)), base((3, 6, 5), seed=9)
    c = base((4, 5), seed=10)
    return [a @ b, mx.addmm(c, a[0, 0], b[0])]


@test
def matmul_einsum_tensordot():
    a, b = base((4, 6)), base((6, 5), seed=11)
    return [mx.einsum("ij,jk->ik", a, b), mx.tensordot(a, b, axes=1),
            mx.outer(a[0], b[:, 0]), mx.inner(a, a), mx.kron(a[:2, :2], b[:2, :2])]


@test
def matmul_large_float16():
    a = base((64, 128)).astype(mx.float16)
    b = base((128, 32), seed=12).astype(mx.float16)
    return (a @ b).astype(mx.float32)


# --- softmax, comparisons, casts -------------------------------------------


@test
def softmax_family():
    a = base((4, 6))
    return [mx.softmax(a, axis=-1), mx.softmax(a, axis=0), nn.log_softmax(a),
            mx.softmax(a * 30, axis=-1, precise=True)]


@test
def comparisons():
    a, b = base(), base(seed=13)
    return [a == b, a != b, a < b, a <= b, a > b, a >= b,
            mx.logical_and(a > 0, b > 0), mx.logical_or(a > 0, b > 0),
            mx.logical_not(a > 0), mx.isnan(a / (a - a)), mx.array_equal(a, a)]


@test
def casts():
    a = base() * 100
    return [a.astype(mx.int32), a.astype(mx.int8), a.astype(mx.uint8),
            a.astype(mx.float16), a.astype(mx.bool_), a.astype(mx.int64),
            mx.arange(5).astype(mx.float32) / 2, a.astype(mx.uint32)]


@test
def casts_bfloat16():
    a = base()
    return (a.astype(mx.bfloat16) * 3).astype(mx.float32)


@test
def bitwise():
    a = mx.arange(24, dtype=mx.int32).reshape(4, 6)
    b = mx.arange(24, 0, -1, dtype=mx.int32).reshape(4, 6)
    return [a & b, a | b, a ^ b, a << 2, b >> 1]


@test
def bitwise_invert():
    a = mx.arange(24, dtype=mx.int32).reshape(4, 6)
    return [~a, mx.bitwise_invert(a > 5), ~a.astype(mx.uint8)]


@test
def sorting():
    a = base((5, 7))
    return [mx.sort(a, axis=1), mx.argsort(a, axis=0), mx.topk(a, 3, axis=1),
            mx.partition(a, 2, axis=1)[:, 2], mx.argpartition(a, 2, axis=1)[:, 2]]


# --- random -----------------------------------------------------------------


@test
def random_seeded():
    mx.random.seed(42)
    return [mx.random.uniform(shape=(3, 4)), mx.random.normal((3, 4)),
            mx.random.randint(0, 100, (3, 4)), mx.random.bernoulli(0.3, (5, 5)),
            mx.random.truncated_normal(-1, 1, (3, 3)), mx.random.gumbel((2, 3))]


@test
def random_keys_and_categorical():
    key = mx.random.key(7)
    k1, k2 = mx.random.split(key)
    logits = base((4, 10))
    return [mx.random.uniform(shape=(4,), key=k1),
            mx.random.categorical(logits, key=k2),
            mx.random.permutation(10, key=k1)]


# --- nn layers --------------------------------------------------------------


@test
def nn_linear():
    mx.random.seed(0)
    layer = nn.Linear(16, 8)
    return layer(base((3, 16)))


@test
def nn_layernorm():
    mx.random.seed(0)
    layer = nn.LayerNorm(16)
    return [layer(base((3, 5, 16))), mx.fast.layer_norm(base((2, 16)), None, None, 1e-5)]


@test
def nn_rmsnorm():
    mx.random.seed(0)
    return nn.RMSNorm(16)(base((3, 5, 16)))


@test
def nn_embedding():
    mx.random.seed(0)
    layer = nn.Embedding(50, 12)
    return layer(mx.array([[1, 4, 49], [0, 7, 7]]))


@test
def nn_multihead_attention():
    mx.random.seed(0)
    layer = nn.MultiHeadAttention(32, 4)
    x = base((2, 7, 32))
    mask = nn.MultiHeadAttention.create_additive_causal_mask(7)
    return [layer(x, x, x), layer(x, x, x, mask)]


@test
def nn_transformer_block():
    mx.random.seed(0)
    layer = nn.TransformerEncoderLayer(32, 4, 64)
    return layer(base((2, 7, 32)), None)


@test
def nn_activations():
    a = base()
    return [nn.gelu(a), nn.relu(a), nn.silu(a), nn.gelu_approx(a),
            nn.leaky_relu(a), nn.softplus(a), nn.elu(a)]


@test
def nn_conv():
    mx.random.seed(0)
    c1 = nn.Conv1d(4, 6, 3, padding=1)
    c2 = nn.Conv2d(3, 5, 3, stride=2)
    return [c1(base((2, 10, 4))), c2(base((2, 9, 9, 3)))]


@test
def nn_rope_and_sdpa():
    mx.random.seed(0)
    q = base((1, 2, 5, 8))
    return [nn.RoPE(8)(q), mx.fast.scaled_dot_product_attention(q, q, q, scale=0.35),
            mx.fast.scaled_dot_product_attention(q, q, q, scale=0.35, mask="causal")]


@test
def nn_losses():
    logits = base((6, 10))
    targets = mx.array([1, 0, 9, 3, 3, 7])
    return [nn.losses.cross_entropy(logits, targets),
            nn.losses.mse_loss(logits, logits * 0.5)]


@test
def nn_losses_gradient():
    """The cross entropy is one kernel of the GPU's in mlx, and so is its gradient: the Mac must be sent those,
    not the operations they stand for (which give other last bits)."""
    logits = base((6, 10))
    targets = mx.array([1, 0, 9, 3, 3, 7])
    loss, gradient = mx.value_and_grad(lambda x: nn.losses.cross_entropy(x, targets).sum())(logits)
    return [loss, gradient, mx.fast.cross_entropy(logits.astype(mx.float16), targets)]


# --- transforms -------------------------------------------------------------


@test
def transform_grad():
    w = base((6,))

    def loss(w, x):
        return mx.tanh(x @ w).sum()

    value, grad = mx.value_and_grad(loss)(w, base((4, 6), seed=14))
    return [value, grad]


@test
def transform_grad_model():
    mx.random.seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.LayerNorm(16), nn.Linear(16, 2))
    x = base((5, 8))

    def loss(model):
        return (model(x) ** 2).mean()

    value, grads = nn.value_and_grad(model, loss)(model)
    flat = [v for _, v in sorted(tree_flatten(grads))]
    return [value, *flat]


@test
def transform_compile():
    @mx.compile
    def fun(x, y):
        return mx.exp(-x) * y + mx.sin(x)

    a, b = base(), base(seed=15)
    return [fun(a, b), fun(b, a)]


@test
def transform_vmap():
    a, b = base((5, 4)), base((5, 4), seed=16)
    return mx.vmap(lambda x, y: (x * y).sum())(a, b)


# --- evaluation paths and the cpu ------------------------------------------


@test
def eval_async_then_read():
    a = base((4, 6))
    b = a @ a.T
    mx.async_eval(b)
    c = b + 1
    mx.async_eval(c)
    return [c, b.sum().item()]


@test
def eval_chain_reuse():
    a = base((4, 6))
    b = mx.exp(a)
    mx.eval(b)
    c = b * 2
    mx.eval(c)
    d = (c - b).sum()
    return [d.item(), np.array(c).sum()]


@test
def cpu_stream_mix():
    a = base((4, 6))
    b = mx.add(a, 1.0, stream=mx.cpu)           # gpu data read on the cpu
    c = (b * a).sum(axis=0)                      # cpu data used on the gpu
    d = mx.sort(c, stream=mx.cpu)
    return [b, c, d, d * 2]


@test
def cpu_views_to_gpu():
    a = mx.multiply(base((6, 8)), 1.0, stream=mx.cpu)
    v = mx.transpose(a, stream=mx.cpu)[1:5, ::2]  # a view made on the cpu
    r = mx.reshape(a, (8, 6), stream=mx.cpu)      # the same bytes, another shape
    mx.eval(v, r)
    return [v * 2, r + 1, a.sum()]


@test
def cpu_linalg():
    a = base((5, 5))
    q, r = mx.linalg.qr(a, stream=mx.cpu)
    return [mx.abs(q @ r - a).max() < 1e-4, mx.linalg.norm(a), mx.linalg.inv(a, stream=mx.cpu) @ a]


@test
def numpy_roundtrip():
    x = np.arange(24, dtype=np.float32).reshape(4, 6)
    a = mx.array(x)
    b = a * 2
    return [np.array(b), np.asarray(b[1:3]), b.tolist()[2][3], memoryview(b).nbytes]


@test
def save_and_load():
    a, b = base((4, 6)), mx.arange(10)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.safetensors")
        mx.save_safetensors(path, {"a": a, "b": b})
        loaded = mx.load(path)
        npz = os.path.join(d, "t.npz")
        mx.savez(npz, a=a)
        again = mx.load(npz)
        return [loaded["a"] * 2, loaded["b"] + 1, again["a"] - a]


@test
def fft_ops():
    a = base((4, 16))
    return [mx.fft.fft(a), mx.abs(mx.fft.rfft(a)), mx.fft.irfft(mx.fft.rfft(a))]


@test
def quantized_matmul():
    mx.random.seed(0)
    w = mx.random.normal((16, 32))
    wq, scales, biases = mx.quantize(w, group_size=32, bits=4)
    x = base((3, 32))
    return [mx.dequantize(wq, scales, biases, group_size=32, bits=4),
            mx.quantized_matmul(x, wq, scales, biases, transpose=True, group_size=32, bits=4)]


@test
def big_array_roundtrip():
    a = mx.arange(3_000_000, dtype=mx.float32).reshape(1000, 3000)
    b = mx.sqrt(a) + 1
    return np.array(b)[::97, ::89]


@test
def zero_size_and_scalars():
    e = mx.zeros((0, 4))
    return [e.sum(), mx.concatenate([e, mx.ones((2, 4))]), mx.array(3.5) * 2,
            mx.array(7) + mx.array(1)]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "coverage.json"
    results = {}
    # DUMP=directory keeps the arrays themselves there, for scripts/tensor-diff.py
    # to say how far two runs that differ are from each other.
    dump = os.environ.get("DUMP")
    if dump:
        os.makedirs(dump, exist_ok=True)
    for fn in TESTS:
        try:
            value = fn()
            results[fn.__name__] = {"checksum": checksum(value), "values": fingerprint(value)}
            if dump:
                for i, (dtype, a) in enumerate(leaves(value)):
                    if dtype:
                        np.save(os.path.join(dump, f"{fn.__name__}.{i:03d}.npy"), a)
        except Exception as e:  # noqa: BLE001
            results[fn.__name__] = {"raised": f"{type(e).__name__}: " + str(e).split("\n")[0][:300]}
    info = {
        "device": str(mx.default_device()),
        "mlx": mx.__version__,
        "platform": sys.platform,
    }
    with open(out, "w") as f:
        json.dump({"info": info, "results": results}, f, indent=1)
    errors = [k for k, v in results.items() if "raised" in v]
    print(f"{len(results)} tests on {info['device']} ({info['platform']}), "
          f"{len(errors)} raised: {errors}")


if __name__ == "__main__":
    main()
