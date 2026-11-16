"""How much of ordinary PyTorch runs through the bridge: a battery of small programs.

Each computes something on the GPU and reduces it to a number. Run on the
Mac and in the container (through the bridge), the two must agree;
compat_compare.py lines the two runs up. Inputs are made on the CPU from a
seed, so both runs start from the same values.

The number is what a person reads (PASS name number). What is compared is on
the line after it (DATA, see exact.py): the number bit for bit and, of every
tensor that went into it, the type, the shape and a digest of all the bytes.
"""

import copy
import io
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import exact
from exact import note

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
HUB = os.environ.get("MODELS", os.path.expanduser("~/.cache/huggingface/hub"))
tests = []


def test(fn):
    tests.append(fn)
    return fn


def rnd(*shape, seed=0):
    """Random values that are the same on every machine, with mean 0 and variance 1.

    Not torch.randn: on the CPU its values differ in the last bit between macOS and Linux (about
    one in fifty; see rnd_cpu in compat_train.py), and the two runs would then start from
    different values: their results agreed to the six digits printed, not to the bit.
    """
    return ((torch.rand(shape, generator=torch.Generator().manual_seed(seed)) - 0.5) * 3.4641016).to(device)


def num(t):
    """A tensor as one number to read, and the whole of it noted for the comparison (exact.note):
    the number alone can be the same for two different tensors."""
    note(t)
    t = t.detach().float()
    weights = torch.arange(1, t.numel() + 1, device=t.device, dtype=torch.float32).reshape(t.shape) / t.numel()
    return float((t * weights).sum().cpu())


def module(make, seed=0):
    torch.manual_seed(seed)
    return make().to(device).eval()


# --- making tensors ------------------------------------------------------------

@test
def tensor_from_python_list():
    return num(torch.tensor([[1.0, 2.0], [3.0, 4.5]], device=device))


@test
def factories():
    parts = [torch.zeros(3, 4, device=device), torch.ones(3, 4, device=device), torch.full((3, 4), 2.5, device=device),
             torch.arange(12, device=device).reshape(3, 4).float(), torch.linspace(0, 1, 12, device=device).reshape(3, 4),
             torch.eye(3, 4, device=device)]
    return sum(num(p) for p in parts)


@test
def like_factories():
    x = rnd(3, 4)
    return num(torch.zeros_like(x) + torch.ones_like(x) * 2 + x.new_full((3, 4), 0.5) + x.new_zeros(3, 4) + torch.full_like(x, 3.0))


@test
def from_numpy_and_back():
    import numpy as np
    x = torch.from_numpy(np.arange(12, dtype=np.float32).reshape(3, 4)).to(device)
    return float(note((x * 2).cpu()).numpy().sum())


# --- arithmetic, shapes, indexing ----------------------------------------------

@test
def arithmetic():
    x, y = rnd(8, 16), rnd(8, 16, seed=1)
    return num(x * 2 + 1 - y / 3 + x ** 2 - torch.sqrt(x.abs()) + torch.exp(-y.abs()) + torch.log1p(x.abs()) % 0.7)


@test
def broadcasting():
    return num(rnd(8, 1, 5) + rnd(1, 7, 5, seed=1) * rnd(5, seed=2))


@test
def reductions():
    x = rnd(6, 7, 8)
    values, indices = x.max(dim=1)
    return (float(x.sum()) + float(x.mean()) + num(x.std(dim=0)) + num(x.var(dim=-1)) + num(values) + num(indices)
            + num(x.argmax(dim=2)) + num(x.norm(dim=1)) + num(x.cumsum(dim=0)) + float(x.abs().amin()) + num(x.logsumexp(dim=1)))


@test
def indexing_basic():
    x = rnd(6, 7, 8)
    return num(x[1]) + num(x[:, 2:5]) + num(x[..., -1]) + num(x[::2, 1::3]) + num(x[None, 0, :, None])


@test
def indexing_with_tensors():
    x = rnd(6, 7)
    rows = torch.tensor([0, 2, 5], device=device)
    return (num(x[rows]) + num(x[[1, 3]]) + num(x[rows, rows]) + num(x.index_select(1, rows))
            + num(x.gather(1, rows[None].expand(6, 3))) + num(x.take(rows)))


@test
def boolean_masks():
    x = rnd(6, 7)
    return num(x[x > 0]) + num(x.masked_select(x < -0.5)) + num(torch.nonzero(x > 1.0)) + num(x.masked_fill(x > 0, 0.25))


@test
def assignment_in_place():
    x = rnd(6, 7)
    x.add_(1).mul_(0.5)
    x[0] = 0
    x[:, 1] = rnd(6, seed=3)
    x[x > 1] = 1
    x.clamp_(-1, 0.9)
    x[2:4, 3:5].zero_()
    return num(x)


@test
def reshaping_in_place():
    x = rnd(4, 5)
    x.unsqueeze_(0)
    x.transpose_(1, 2)
    x.squeeze_(0)
    x.t_()
    return num(x) + sum(x.shape)


@test
def views():
    x = rnd(4, 6, 8)
    parts = [x.view(24, 8), x.reshape(4, 48), x.permute(2, 0, 1), x.transpose(0, 2).contiguous(), x.flatten(1),
             x[:, :1].expand(4, 6, 8), x.repeat(2, 1, 1), x.unsqueeze(1).squeeze(1)]
    return sum(num(p) for p in parts) + sum(num(c) for c in x.chunk(3, dim=1)) + sum(num(s) for s in x.split(3, dim=2)) + num(x.unbind(0)[2])


@test
def joining():
    a, b = rnd(3, 4), rnd(3, 4, seed=1)
    return num(torch.cat([a, b])) + num(torch.cat([a, b], dim=1)) + num(torch.stack([a, b], dim=1)) + num(torch.hstack([a, b]))


@test
def sorting():
    x = rnd(6, 9)
    values, order = x.sort(dim=1, descending=True)
    top, where = x.topk(3, dim=1)
    return num(values) + num(order) + num(top) + num(where) + num(x.argsort(dim=0)) + num(torch.unique((x * 2).round()))


@test
def comparisons():
    x, y = rnd(6, 7), rnd(6, 7, seed=1)
    return (num(torch.where(x > y, x, y)) + num((x == x).float()) + float((x < y).any()) + float((x.abs() >= 0).all())
            + num(torch.isnan(x / x).float()) + num(x.clamp(-0.5, 0.5)) + num(torch.maximum(x, y)) + float(torch.equal(x, x.clone())))


@test
def products():
    a, b = rnd(5, 6), rnd(6, 7, seed=1)
    return (num(a @ b) + num(torch.bmm(rnd(3, 5, 6), rnd(3, 6, 2, seed=1))) + num(torch.einsum("ij,jk->ik", a, b))
            + num(torch.einsum("bij,bkj->bik", rnd(2, 3, 4), rnd(2, 5, 4, seed=1))) + num(torch.outer(a[0], b[0])) + float(a[0].dot(a[1])))


@test
def integer_and_boolean_types():
    x = (rnd(6, 7) * 10).long()
    return num(x + 3) + num(x // 2) + num(x % 3) + num((x > 0) & (x < 5)) + num(x.float() / 2) + num(x.bool()) + num(x.to(torch.int32) * 2)


@test
def half_precision():
    x = rnd(6, 7).half()
    return num((x * 2 + 1).float())


# --- reading values back -----------------------------------------------------------

@test
def reading_values():
    x = note(rnd(3, 4))
    return float(x[0, 0]) + x.sum().item() + int((x > 0).sum()) + sum(sum(row) for row in x.tolist()) + len(x) + bool(x.abs().sum() > 0)


@test
def control_flow_on_values():
    x, steps = rnd(64), 0
    while float(x.abs().max()) > 0.1 and steps < 200:
        x = x * 0.9
        steps += 1
    return steps + float(note(x).sum())


@test
def moving_between_devices():
    x = rnd(5, 6)
    return num(x.cpu().to(device) + x.to("cpu").to(device=device, dtype=torch.float32)) + num(x.clone().detach())


# --- layers ---------------------------------------------------------------------

@test
def layers_linear_and_activations():
    net = module(lambda: nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 8), nn.Mish(), nn.LayerNorm(8)))
    x = rnd(4, 16)
    with torch.no_grad():
        y = net(x)
    return num(y) + num(F.softmax(y, dim=-1)) + num(F.log_softmax(y, dim=-1)) + num(torch.sigmoid(y)) + num(F.leaky_relu(y)) + num(F.elu(y))


@test
def layers_convolution_2d():
    net = module(lambda: nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(), nn.MaxPool2d(2),
                                       nn.Conv2d(8, 16, 3, stride=2), nn.GroupNorm(4, 16), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 5)))
    with torch.no_grad():
        return num(net(rnd(2, 3, 32, 32)))


@test
def layers_convolution_1d_and_resize():
    net = module(lambda: nn.Sequential(nn.Conv1d(4, 8, 5, padding=2), nn.ReLU(), nn.AvgPool1d(2), nn.ConvTranspose1d(8, 4, 2, stride=2)))
    with torch.no_grad():
        y = net(rnd(2, 4, 40))
        return num(y) + num(F.interpolate(rnd(1, 2, 8, 8), scale_factor=2, mode="bilinear")) + num(F.pad(y, (1, 2)))


@test
def layers_embedding():
    net = module(lambda: nn.Embedding(50, 12))
    nn.init.uniform_(net.weight, -1.0, 1.0)  # its own initial weights are normal ones: see rnd
    with torch.no_grad():
        return num(net(torch.tensor([[1, 5, 9], [3, 0, 49]], device=device)))


@test
def layers_recurrent():
    lstm, gru = module(lambda: nn.LSTM(8, 16, num_layers=2, batch_first=True)), module(lambda: nn.GRU(8, 16, batch_first=True), seed=1)
    with torch.no_grad():
        out, (h, c) = lstm(rnd(3, 10, 8))
        out2, h2 = gru(rnd(3, 10, 8, seed=1))
    return num(out) + num(h) + num(c) + num(out2) + num(h2)


@test
def layers_attention():
    attention = module(lambda: nn.MultiheadAttention(32, 4, batch_first=True))
    encoder = module(lambda: nn.TransformerEncoderLayer(32, 4, 64, dropout=0.0, batch_first=True), seed=1)
    x = rnd(2, 9, 32)
    with torch.no_grad():
        y, weights = attention(x, x, x)
        return num(y) + num(weights) + num(encoder(x)) + num(F.scaled_dot_product_attention(x, x, x, is_causal=True))


@test
def losses():
    logits, target = rnd(8, 5), torch.tensor([0, 1, 2, 3, 4, 0, 1, 2], device=device)
    return (float(F.cross_entropy(logits, target)) + float(F.mse_loss(logits, rnd(8, 5, seed=1))) + float(F.nll_loss(F.log_softmax(logits, 1), target))
            + float(F.binary_cross_entropy_with_logits(logits, (rnd(8, 5, seed=2) > 0).float())))


# --- training ---------------------------------------------------------------------

@test
def gradients():
    w = rnd(5, 6).requires_grad_()
    x = rnd(4, 5, seed=1)
    loss = (torch.tanh(x @ w) ** 2).mean()
    loss.backward()
    return float(loss) + num(w.grad)


def train(make, optimizer, data, target, steps):
    torch.manual_seed(0)
    net = make().to(device).train()
    opt = optimizer(net.parameters())
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(net(data), target)
        loss.backward()
        opt.step()
    for parameter in net.parameters():
        note(parameter)  # every weight is compared, not only the one that gives the number
    return float(loss) + num(next(net.parameters()))


@test
def training_with_sgd():
    target = torch.tensor([0, 1, 2, 3] * 8, device=device)
    return train(lambda: nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 4)), lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9), rnd(32, 20), target, 20)


@test
def training_with_adam():
    target = torch.tensor([0, 1, 2, 3] * 8, device=device)
    return train(lambda: nn.Sequential(nn.Linear(20, 64), nn.Tanh(), nn.Dropout(0.0), nn.Linear(64, 4)), lambda p: torch.optim.AdamW(p, lr=0.01), rnd(32, 20), target, 20)


@test
def training_a_convolution():
    target = torch.tensor([0, 1] * 4, device=device)
    make = lambda: nn.Sequential(nn.Conv2d(1, 4, 3), nn.BatchNorm2d(4), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 2))
    return train(make, lambda p: torch.optim.SGD(p, lr=0.05), rnd(8, 1, 12, 12), target, 6)


# --- around the edges ----------------------------------------------------------------

@test
def random_numbers_from_a_seed():
    torch.manual_seed(0)
    return num(torch.randn(4, 5, device=device)) + num(torch.rand(4, 5, device=device))


@test
def random_numbers_shape_only():
    x = torch.randn(4, 5, device=device) + torch.rand(4, 5, device=device) + torch.randint(0, 5, (4, 5), device=device)
    return float(x.numel()) + float(torch.isfinite(x).all())


@test
def memory_and_synchronize_calls():
    torch.mps.synchronize()
    torch.mps.empty_cache()
    return float(torch.mps.current_allocated_memory() >= 0)


@test
def copying_and_saving_a_model():
    net = module(lambda: nn.Linear(6, 3))
    twin = copy.deepcopy(net)
    buffer = io.BytesIO()
    torch.save(net.state_dict(), buffer)
    buffer.seek(0)
    twin.load_state_dict(torch.load(buffer, map_location=device))
    with torch.no_grad():
        return num(twin(rnd(2, 6))) + num(net.weight)


@test
def inference_mode():
    net = module(lambda: nn.Linear(6, 3))
    with torch.inference_mode():
        return num(net(rnd(2, 6)))


@test
def inference_mode_attention_and_lstm():
    """Inside torch.inference_mode() an operation skips autograd on its way to the GPU: attention and an LSTM over a
    sequence, which a Mac's GPU runs whole, must reach it whole from there too."""
    q, k, v = rnd(2, 4, 6, 8), rnd(2, 4, 6, 8, seed=1), rnd(2, 4, 6, 8, seed=2)
    lstm = module(lambda: nn.LSTM(8, 16, num_layers=2, batch_first=True, bidirectional=True))
    x = rnd(3, 10, 8, seed=3)
    with torch.inference_mode():
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out, (h, c) = lstm(x)
    with torch.no_grad():
        again, _ = lstm(x)
    return num(attended) + num(out) + num(h) + num(c) + num(again)


# --- results whose shape is in the values, and views saved with their base -----------------

@test
def shapes_that_depend_on_the_values():
    """One operation, on tensors of one shape and other values: each result has the length its own values give it.
    (A length remembered from the first call would be given to the others.)"""
    total = 0.0
    for values in ([1, 1, 2, 3, 3, 3], [5, 5, 5, 5, 5, 5], [1, 2, 3, 4, 5, 6], [2, 2, 1, 1, 2, 2]):
        x = torch.tensor(values, device=device)
        runs, counts = torch.unique_consecutive(x, return_counts=True)
        along = torch.unique_consecutive(x.reshape(2, 3), dim=1)
        repeated = torch.repeat_interleave(x, x)
        counted = torch.bincount(x)
        picked = x[x > 2]
        lengths = [t.shape[-1] for t in (runs, counts, along, repeated, counted, picked)]
        total += sum(num(t) for t in (runs, counts, along, repeated, counted, picked)) + sum(n * 10 ** i for i, n in enumerate(lengths))
    return total


@test
def saving_a_tensor_with_its_views():
    """A tensor and two views of it are one storage in the file, and share it when loaded: a change through one shows in the others."""
    import zipfile
    base = torch.arange(12, dtype=torch.float32, device=device)
    view, table = base[4:8], base.view(3, 4)
    buffer = io.BytesIO()
    torch.save({"base": base, "view": view, "table": table}, buffer)
    storages = len([name for name in zipfile.ZipFile(buffer).namelist() if "/data/" in name])
    buffer.seek(0)
    loaded = torch.load(buffer)
    loaded["base"][5] = 100.0
    loaded["view"][0] = -7.0
    again = io.BytesIO()
    torch.save(loaded, again)  # loaded tensors are views of one storage too: saved again, it is still one
    storages_again = len([name for name in zipfile.ZipFile(again).namelist() if "/data/" in name])
    return num(loaded["base"]) + num(loaded["view"]) + num(loaded["table"]) + 1000 * storages + 100000 * storages_again


def saved_storages(buffer):
    """The storages in a file torch.save wrote, each as a tensor of its bytes, in the file's order."""
    import zipfile
    archive = zipfile.ZipFile(buffer)
    names = sorted((name for name in archive.namelist() if "/data/" in name), key=lambda name: int(name.rsplit("/", 1)[1]))
    return [torch.frombuffer(bytearray(archive.read(name)), dtype=torch.uint8) for name in names]


def saved_and_loaded(things):
    """Saves, and loads on the GPU and on the CPU: (the file's storages, what was loaded on each)."""
    buffer = io.BytesIO()
    torch.save(things, buffer)
    storages = saved_storages(buffer)
    buffer.seek(0)
    on_gpu = torch.load(buffer)
    buffer.seek(0)
    return storages, on_gpu, torch.load(buffer, map_location="cpu")


@test
def saving_a_view_after_its_base_shrank():
    """A tensor made smaller keeps its memory, and a view made before still reads it: the file holds that memory whole."""
    base = torch.arange(6, dtype=torch.float32, device=device) + 1
    view = base[2:]
    base.resize_(2)
    storages, on_gpu, on_cpu = saved_and_loaded({"base": base, "view": view})
    on_gpu["view"][0] = -7.0  # loaded, they are still one memory: this is the third number of it
    on_gpu["base"].resize_(6)
    return (sum(num(t) for t in storages) + num(view) + num(on_gpu["base"]) + num(on_gpu["view"]) + num(on_cpu["base"])
            + num(on_cpu["view"]) + 1000 * len(storages))


@test
def saving_a_view_after_its_base_grew():
    """A tensor made larger takes its views with it to the new memory: they still share it, in the program and in the file."""
    base = torch.arange(4, dtype=torch.float32, device=device) + 1
    view = base[1:3]
    base.resize_(8)
    base[4:] = 9.0
    view[0] = 50.0
    storages, on_gpu, on_cpu = saved_and_loaded({"view": view, "base": base})
    on_gpu["base"][2] = -3.0
    return (sum(num(t) for t in storages) + num(base) + num(on_gpu["base"]) + num(on_gpu["view"]) + num(on_cpu["base"])
            + num(on_cpu["view"]) + 1000 * len(storages))


@test
def saving_a_tensor_given_the_memory_of_another():
    """x.set_(y) makes x a tensor of y's memory. Saved, the two are one storage; a view of x made before keeps x's first memory."""
    first = torch.tensor([7.0, 8.0, 9.0], device=device)
    before = first[1:]
    other = torch.arange(6, dtype=torch.float32, device=device)
    first.set_(other)
    first[0] = 100.0  # shows in `other`
    part = torch.empty(0, device=device).set_(other, 2, (3,), (1,))  # three numbers of the same memory, from its third
    part[0] = -5.0
    storages, on_gpu, on_cpu = saved_and_loaded({"first": first, "other": other, "before": before, "part": part})
    on_gpu["other"][1] = 55.0  # shows in the loaded `first`
    emptied = torch.ones(3, device=device).set_()
    return (sum(num(t) for t in storages) + num(first) + num(other) + num(before) + num(part)
            + sum(num(on_gpu[name]) + num(on_cpu[name]) for name in ("first", "other", "before", "part"))
            + 1000 * len(storages) + 100000 * emptied.numel())


@test
def saving_views_of_other_types():
    """The same memory read as another type: a view of it is saved by itself with all of the memory, and PyTorch
    refuses to save the two types of one memory together."""
    base = torch.arange(8, dtype=torch.float32, device=device) + 0.5
    as_integers = base.view(torch.int32)[2:6]
    storages, on_gpu, on_cpu = saved_and_loaded({"integers": as_integers})
    try:
        torch.save({"base": base, "integers": as_integers}, io.BytesIO())
        refused = 0
    except RuntimeError as e:
        refused = 1 if "different types" in str(e) else 2
    return (sum(num(t) for t in storages) + num(on_gpu["integers"]) + num(on_cpu["integers"]) + 1000 * len(storages)
            + 100000 * refused)


# --- whole models ---------------------------------------------------------------------

def snapshot(name):
    folder = f"{HUB}/models--{name.replace('/', '--')}/snapshots"
    return f"{folder}/{sorted(os.listdir(folder))[0]}"


@test
def model_minilm_sentence_embeddings():
    from transformers import AutoModel, AutoTokenizer
    path = snapshot("sentence-transformers/all-MiniLM-L6-v2")
    tokenizer, model = AutoTokenizer.from_pretrained(path), AutoModel.from_pretrained(path).eval().to(device)
    batch = tokenizer(["un container Linux su un Mac", "la ricetta della carbonara"], padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).float()
    return num(F.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1))


@test
def model_gpt2_text_generation():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = snapshot("openai-community/gpt2")
    tokenizer, model = AutoTokenizer.from_pretrained(path), AutoModelForCausalLM.from_pretrained(path).eval().to(device)
    prompt = tokenizer("The container runs on the Mac and", return_tensors="pt").to(device)
    with torch.no_grad():
        model.generate(**prompt, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)  # warm up
        start = time.perf_counter()
        out = model.generate(**prompt, max_new_tokens=60, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        seconds = time.perf_counter() - start
    new = out.shape[1] - prompt["input_ids"].shape[1]
    print(f"TIME gpt2 {new / seconds:.1f} tokens per second")
    print("TEXT " + tokenizer.decode(out[0][-12:]).replace("\n", " "))
    return float(note(out).sum())


only = os.environ.get("ONLY")
print(exact.info("compat", device), flush=True)
for fn in tests:
    if only and only not in fn.__name__:
        continue
    exact.begin()
    try:
        value = fn()
        print(f"PASS {fn.__name__} {value:.6g}", flush=True)
        print(f"DATA {fn.__name__} {exact.data(value, fn.__name__)}", flush=True)
    except Exception as e:  # noqa: BLE001 - the point is to list what does not work
        print(f"FAIL {fn.__name__} {type(e).__name__}: {str(e).splitlines()[0][:150]}", flush=True)
        if os.environ.get("TRACE"):
            import traceback
            traceback.print_exc()
