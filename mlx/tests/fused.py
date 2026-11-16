"""Compiled functions and attention in the forms the coverage does not have:
prints one number per case; the output on the Mac and in the container must
be the same. Usage: python fused.py"""
import math
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

mx.random.seed(3)
cases = {}


def case(f):
    try:
        out = f()
        out = out if isinstance(out, (list, tuple)) else [out]
        cases[f.__name__] = round(sum(float(mx.sum(o.astype(mx.float32)).item()) for o in out), 4)
    except Exception as e:
        cases[f.__name__] = f"{type(e).__name__}: {str(e)[:90]}"
    return f


q = mx.random.normal((2, 4, 6, 16))
k = mx.random.normal((2, 4, 9, 16))
v = mx.random.normal((2, 4, 9, 16))


@case
def attention_additive_mask():
    mask = mx.where(mx.random.uniform(shape=(6, 9)) > 0.3, 0.0, -1e9)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=0.25, mask=mask)


@case
def attention_bool_mask():
    mask = mx.random.uniform(shape=(2, 1, 6, 9)) > 0.3
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=0.25, mask=mask)


@case
def attention_grouped_heads():
    return mx.fast.scaled_dot_product_attention(q, k[:, :2], v[:, :2], scale=0.25)


@case
def attention_half_precision_long():
    a = mx.random.normal((1, 8, 300, 64)).astype(mx.float16)
    return mx.fast.scaled_dot_product_attention(a, a, a, scale=0.125, mask="causal")


@case
def attention_head_192():
    a = mx.random.normal((1, 2, 12, 192))
    return mx.fast.scaled_dot_product_attention(a, a, a, scale=0.07)


@case
def attention_grad():
    f = lambda q: mx.fast.scaled_dot_product_attention(q, k, v, scale=0.25).sum()
    return mx.grad(f)(q)


@case
def attention_on_cpu():
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=0.25, stream=mx.cpu)


@case
def compiled_two_outputs():
    @mx.compile
    def f(a, b):
        return mx.exp(a) * b + 1.5, mx.sin(a) - b / 3

    return list(f(q, q * 2)) + list(f(q[:1], q[:1]))


@case
def compiled_shapeless():
    @partial(mx.compile, shapeless=True)
    def f(a):
        return a * mx.sigmoid(a) + mx.tanh(a) * 0.5

    return [f(q), f(k), f(q.astype(mx.float16))]


@case
def compiled_mixed():
    @mx.compile
    def f(a, b):
        c = mx.maximum(a, 0.1) * 2
        return (c @ b.swapaxes(-1, -2)).sum(axis=-1) + mx.where(a[..., 0] > 0, 1.0, -1.0)

    return f(q, k)


@case
def compiled_scalar_from_gpu():
    s = (q * q).mean()
    mx.eval(s)

    @mx.compile
    def f(a):
        return a * s + s

    return f(q)


@case
def compiled_integer_ops():
    @mx.compile
    def f(a):
        return (~a) & (a + 3) | (a >> 1)

    return f(mx.arange(24).reshape(4, 6))


@case
def compiled_grad():
    @mx.compile
    def f(a):
        return mx.grad(lambda x: (nn.gelu(x) * nn.silu(x)).sum())(a)

    return f(q)


@case
def compiled_training_step():
    model = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 4))
    mx.eval(model.parameters())
    opt = optim.Adam(learning_rate=1e-2)
    x = mx.random.normal((64, 16))
    y = mx.random.randint(0, 4, (64,))

    def loss_fn(x, y):
        return nn.losses.cross_entropy(model(x), y, reduction="mean")

    state = [model.state, opt.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y):
        loss, grads = nn.value_and_grad(model, loss_fn)(x, y)
        opt.update(model, grads)
        return loss

    losses = [step(x, y) for _ in range(5)]
    mx.eval(state)
    return losses


@case
def compiled_on_cpu():
    @mx.compile
    def f(a):
        return mx.exp(a, stream=mx.cpu) * mx.cos(a, stream=mx.cpu) + 1

    return f(q)


for name, value in cases.items():
    print(f"{name}: {value}")
