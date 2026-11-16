"""What a real training uses, through the bridge: a second battery, next to compat.py.

Same idea and same output (PASS name number and DATA name ..., lined up by compat_compare.py):
each program computes something on the GPU and reduces it to a number to read, and what is
compared is that number bit for bit and every tensor behind it, in full (exact.py). Here the
programs are whole pieces of a training: loops with the optimizers and schedulers people use,
mixed precision, data loaders with worker processes, saving and resuming, Lightning, random
numbers from a seed, gradients of the less ordinary operators, threads and processes.

    python compat_train.py            ONLY=name runs the programs whose name contains it

It is a module with a main guard: data-loader workers and child processes started afresh
(spawn, the default on a Mac) import it again.
"""

import copy
import io
import math
import multiprocessing
import os
import subprocess
import sys
import tempfile
import threading

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


def rnd_cpu(*shape, seed=0):
    """Random values that are the same on every machine, with mean 0 and variance 1.

    Not torch.randn: on the CPU its values differ in the last bit between macOS and Linux
    (about one in fifty; the normal distribution goes through the C library's log and cos, and
    the two have their own). torch.rand's are the same, and so is this arithmetic on them. The
    same goes for a layer's initial weights: uniform ones are the same on both, normal ones
    (nn.Embedding, a transformers model's new head) are not, so the programs here that build
    such layers on the CPU give them uniform weights by hand.
    """
    return (torch.rand(shape, generator=torch.Generator().manual_seed(seed)) - 0.5) * 3.4641016


def rnd(*shape, seed=0):
    return rnd_cpu(*shape, seed=seed).to(device)


def num(t):
    """A tensor as one number to read, and the whole of it noted for the comparison (exact.note):
    the number alone can be the same for two different tensors."""
    note(t)
    t = t.detach().float()
    weights = torch.arange(1, t.numel() + 1, device=t.device, dtype=torch.float32).reshape(t.shape) / t.numel()
    return float((t * weights).sum().cpu())


def weights_of(net):
    return sum(num(p) for p in net.parameters())


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.1), nn.Linear(32, 10))

    def forward(self, x):
        return self.head(self.features(x))


class SmallTransformer(nn.Module):
    def __init__(self, vocabulary=100, width=64, classes=4):
        super().__init__()
        self.embed = nn.Embedding(vocabulary, width)
        nn.init.uniform_(self.embed.weight, -1.0, 1.0)  # see rnd_cpu
        self.position = nn.Parameter(torch.zeros(1, 32, width))
        layer = nn.TransformerEncoderLayer(width, 4, 128, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.head = nn.Linear(width, classes)

    def forward(self, tokens):
        x = self.embed(tokens) + self.position[:, :tokens.shape[1]]
        return self.head(self.encoder(x).mean(1))


def images(n=16, seed=0):
    return rnd(n, 3, 16, 16, seed=seed), (torch.arange(n) % 10).to(device)


def tokens(n=8, length=20, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 100, (n, length), generator=generator).to(device), (torch.arange(n) % 4).to(device)


def fit(net, optimizer, batch, steps, scheduler=None, clip=None, scaler=None, autocast=False, accumulate=1):
    """An ordinary training loop. Returns the last loss and the weights as one number."""
    x, y = batch
    net.train()
    for step in range(steps):
        with torch.autocast(device.type, dtype=torch.float16, enabled=autocast):
            loss = F.cross_entropy(net(x), y) / accumulate
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (step + 1) % accumulate == 0:
            if clip:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(net.parameters(), clip)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
    return float(loss) + weights_of(net)


def made(make, seed=0):
    torch.manual_seed(seed)
    return make().to(device)


# --- training loops ----------------------------------------------------------------------------

@test
def loop_cnn_sgd_momentum_steplr():
    net = made(SmallCNN)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9, nesterov=True, weight_decay=1e-4)
    return fit(net, optimizer, images(), 12, scheduler=torch.optim.lr_scheduler.StepLR(optimizer, 4, 0.5))


@test
def loop_transformer_adamw_cosine_warmup_clip():
    net = made(SmallTransformer)
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / 4) * 0.5 * (1 + math.cos(math.pi * s / 12)))
    return fit(net, optimizer, tokens(), 12, scheduler=schedule, clip=1.0)


@test
def loop_adam_fused():
    net = made(SmallCNN)
    return fit(net, torch.optim.Adam(net.parameters(), lr=1e-3, fused=True), images(), 10)


@test
def loop_adamw_fused_onecycle():
    net = made(SmallTransformer)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, fused=True)
    return fit(net, optimizer, tokens(), 10, scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer, 5e-3, total_steps=10))


@test
def loop_sgd_fused():
    net = made(SmallCNN)
    return fit(net, torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9, fused=True), images(), 10)


@test
def loop_adam_foreach_amsgrad():
    net = made(SmallCNN)
    return fit(net, torch.optim.Adam(net.parameters(), lr=1e-3, foreach=True, amsgrad=True, weight_decay=1e-5), images(), 10)


@test
def loop_other_optimizers():
    total = 0.0
    for make in (lambda p: torch.optim.RMSprop(p, lr=1e-3, momentum=0.5), lambda p: torch.optim.Adagrad(p, lr=1e-2),
                 lambda p: torch.optim.NAdam(p, lr=1e-3), lambda p: torch.optim.RAdam(p, lr=1e-3),
                 lambda p: torch.optim.Adamax(p, lr=1e-3), lambda p: torch.optim.Adadelta(p, lr=0.5)):
        net = made(SmallCNN)
        total += fit(net, make(net.parameters()), images(), 5)
    return total


@test
def loop_autocast_and_gradscaler():
    net = made(SmallCNN)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    return fit(net, optimizer, images(), 10, scaler=torch.amp.GradScaler(device.type), autocast=True, clip=1.0)


@test
def loop_autocast_gradscaler_fused_adam():
    net = made(SmallTransformer)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, fused=True)
    return fit(net, optimizer, tokens(), 8, scaler=torch.amp.GradScaler(device.type), autocast=True)


@test
def loop_gradient_accumulation_and_plateau():
    net = made(SmallCNN)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, factor=0.5)
    total = fit(net, optimizer, images(), 8, accumulate=2)
    for value in (1.0, 1.1, 1.2):
        plateau.step(value)
    return total + optimizer.param_groups[0]["lr"]


@test
def loop_ema_weights_and_eval():
    net = made(SmallCNN)
    average = copy.deepcopy(net)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
    x, y = images()
    for _ in range(6):
        optimizer.zero_grad()
        F.cross_entropy(net(x), y, label_smoothing=0.1).backward()
        optimizer.step()
        with torch.no_grad():
            for a, p in zip(average.parameters(), net.parameters()):
                a.lerp_(p, 0.1)
    average.eval()
    with torch.no_grad():
        return num(average(x)) + num(net.features[1].running_mean) + float(net.features[1].num_batches_tracked)


@test
def loop_swa_and_parameter_groups():
    net = made(SmallCNN)
    optimizer = torch.optim.SGD([{"params": net.features.parameters(), "lr": 0.01}, {"params": net.head.parameters()}], lr=0.1, momentum=0.9)
    averaged = torch.optim.swa_utils.AveragedModel(net)
    x, y = images()
    for _ in range(5):
        optimizer.zero_grad()
        F.cross_entropy(net(x), y).backward()
        optimizer.step()
        averaged.update_parameters(net)
    return weights_of(averaged) + weights_of(net)


@test
def finetune_transformers_model():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    folder = f"{HUB}/models--sentence-transformers--all-MiniLM-L6-v2/snapshots"
    path = f"{folder}/{sorted(os.listdir(folder))[0]}"
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=2)
    nn.init.uniform_(model.classifier.weight, -0.05, 0.05)  # the new head: see rnd_cpu
    model = model.to(device).train()
    batch = tokenizer(["un container Linux su un Mac", "la ricetta della carbonara", "il ponte verso la GPU", "pasta e fagioli"],
                      padding=True, return_tensors="pt").to(device)
    labels = torch.tensor([0, 1, 0, 1], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    for parameter in model.bert.embeddings.parameters():
        parameter.requires_grad_(False)
    for _ in range(4):
        loss = model(**batch, labels=labels).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    model.eval()
    with torch.no_grad():
        return float(loss) + num(model(**batch).logits)


# --- data loaders ------------------------------------------------------------------------------

class Squares(torch.utils.data.Dataset):
    """Items made in the worker, on the CPU, from their index."""

    def __len__(self):
        return 64

    def __getitem__(self, index):
        return rnd_cpu(3, 16, 16, seed=index), index % 10


def with_loader(**options):
    net = made(SmallCNN)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9)
    loader = torch.utils.data.DataLoader(Squares(), batch_size=16, shuffle=False, **options)
    for _ in range(2):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(net(x), y)
            loss.backward()
            optimizer.step()
    return float(loss) + weights_of(net)


@test
def loader_in_process():
    return with_loader(num_workers=0)


@test
def loader_workers_default_start():
    return with_loader(num_workers=2)


@test
def loader_workers_pin_memory_persistent():
    return with_loader(num_workers=2, pin_memory=True, persistent_workers=True)


@test
def loader_workers_forked_after_gpu_use():
    if not hasattr(os, "fork"):
        return with_loader(num_workers=2)
    x = rnd(4, 4)  # the GPU is in use, with operations not yet sent, when the workers are forked
    y = x @ x
    total = with_loader(num_workers=3, multiprocessing_context="fork")
    return total + num(y @ x)


@test
def loader_workers_spawned():
    return with_loader(num_workers=2, multiprocessing_context="spawn")


@test
def pin_memory_by_hand():
    x = torch.arange(12.0).reshape(3, 4).pin_memory()
    return num(x.to(device, non_blocking=True) * 2)


# --- what the bridge decides here, and must decide as PyTorch does ------------------------------

@test
def numbers_and_the_type_of_the_result():
    """A Python number counts for less than a tensor when a result's type is decided, a 0-dim CPU tensor for as much."""
    x, zero = torch.ones(3, device=device), torch.ones((), device=device)
    half, whole, halves = zero.half(), torch.ones((), device=device, dtype=torch.long), x.half()
    cases = [lambda: x + 1.5, lambda: zero + 1e-6, lambda: half * 0.5, lambda: whole + 0.5, lambda: whole * 3, lambda: halves * 2.0,
             lambda: half + torch.tensor(2.0), lambda: halves + torch.tensor(2.0), lambda: 1.0 / zero, lambda: zero > 0.5, lambda: whole / 2,
             lambda: 2 ** zero, lambda: zero / (zero + 1e-6), lambda: x * torch.tensor(3), lambda: whole + True]
    codes = {torch.float32: 1, torch.float16: 2, torch.int64: 3, torch.bool: 4}
    total = 0.0
    for _ in range(2):  # the second time each is an operation the bridge has remembered
        for i, case in enumerate(cases):
            result = case()
            total += (i + 1) * codes[note(result).dtype] + float(result.float().sum())
    return total


@test
def inference_mode_and_views():
    """A view of an ordinary tensor made in inference mode is an ordinary tensor; what is computed there is not."""
    x = torch.zeros(3, device=device)
    kinds = []
    with torch.inference_mode():
        for make in (lambda: x.detach(), lambda: x.view(3, 1), lambda: x + 1, lambda: (x + 1).detach(), lambda: x.detach().clone(), lambda: (x + 1)[1:]):
            kinds.append(make().is_inference())
        y = x + 1
        y.add_(1)
    try:
        y.add_(1)
        kinds.append(True)
    except RuntimeError:
        kinds.append(False)
    return sum(2 ** i for i, kind in enumerate(kinds) if kind) + float(note(y * 2).sum())


@test
def results_that_are_the_argument_for_some_values():
    """native_dropout hands its argument back when p is 0 and makes a tensor otherwise: the argument must stay what it was."""
    x = torch.arange(6.0, device=device)
    total = 0.0
    for p in (0.0, 0.5, 0.0, 0.7):
        torch.manual_seed(0)
        y, _ = torch.native_dropout(x, p, True)
        total += num(y) + num(x)
    z = x.clone()
    for alpha in (0.5, 1.5, 2.5):
        z.add_(x, alpha=alpha)
    return total + num(z) + num(torch.lerp(x, z, 0.25)) + num(torch.lerp(x, z, 0.75))


# --- tensors that stay on the CPU ---------------------------------------------------------------

@test
def indexing_with_indices_on_the_cpu():
    x = rnd(6, 7)
    rows, mask = torch.tensor([0, 2, 5]), torch.tensor([True, False, True, False, True, True])
    y = x.clone()
    y[rows] = 0
    y[mask, 1] = 1
    y.index_put_((rows, rows), torch.ones(3, device=device))
    y[0] = torch.arange(7.0)  # values from the CPU, copied in
    return num(x[rows]) + num(x[mask]) + num(x[rows, rows.to(device)]) + num(x[[1, 3]][:, rows]) + num(y)


@test
def recurrent_layers_with_packed_sequences():
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    lstm = made(lambda: nn.LSTM(4, 8, batch_first=True, bidirectional=True))
    gru = made(lambda: nn.GRU(4, 8, batch_first=True), seed=1)
    x, lengths = rnd(3, 6, 4), torch.tensor([6, 4, 2])  # the lengths are a CPU tensor, as PyTorch wants them
    packed = pack_padded_sequence(x, lengths, batch_first=True)
    out, (h, c) = lstm(packed)
    padded, back = pad_packed_sequence(out, batch_first=True)
    out2, h2 = gru(packed)
    loss = padded.pow(2).mean() + h2.pow(2).mean()
    loss.backward()
    return (num(padded) + num(h) + num(c) + num(out2.data) + num(lstm.weight_ih_l0.grad) + num(gru.weight_hh_l0.grad)
            + float(note(back).sum()) + (packed.batch_sizes.device.type == "cpu") + (back.device.type == "cpu"))


# --- saving and resuming -----------------------------------------------------------------------

@test
def save_load_state_dict_and_optimizer():
    net = made(SmallCNN)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    fit(net, optimizer, images(), 4)
    with tempfile.TemporaryDirectory() as folder:
        torch.save({"model": net.state_dict(), "optimizer": optimizer.state_dict(), "step": 4}, f"{folder}/checkpoint.pt")
        expected = fit(net, optimizer, images(), 4)  # the run that was not interrupted
        total = 0.0
        for location in ("cpu", device, None, lambda storage, where: storage):
            state = torch.load(f"{folder}/checkpoint.pt", map_location=location)
            twin = SmallCNN().to(device)
            again = torch.optim.AdamW(twin.parameters(), lr=1e-3)
            twin.load_state_dict(state["model"])
            again.load_state_dict(state["optimizer"])
            resumed = fit(twin, again, images(), 4)
            total += resumed + (0.0 if resumed == expected else 1000.0)  # resuming must give the uninterrupted run
    return total


@test
def save_load_tensors_of_every_kind():
    x = torch.arange(24.0, device=device).reshape(2, 3, 4)
    things = {"plain": x, "view": x[:, 1], "transposed": x.transpose(0, 2), "strided": x[..., ::2], "half": x.half(), "long": x.long(),
              "bool": x > 5, "empty": x[:0], "scalar": x.sum(), "list": [x + 1, (x * 2, "text", 3)], "parameter": nn.Parameter(x.clone())}
    buffer = io.BytesIO()
    torch.save(things, buffer)
    total = 0.0
    for location in (None, "cpu", "mps" if device.type == "mps" else "cpu"):
        buffer.seek(0)
        back = torch.load(buffer, map_location=location, weights_only=False)
        for name in ("plain", "view", "transposed", "strided", "half", "long", "bool", "scalar", "parameter"):
            assert back[name].dtype == things[name].dtype and back[name].shape == things[name].shape, name
            assert torch.equal(back[name].cpu(), things[name].detach().cpu()), name
            total += num(back[name]) + (back[name].device.type == device.type)
        total += num(back["list"][1][0]) + back["empty"].numel() + back["parameter"].requires_grad
    return total


@test
def save_whole_module_and_pickle():
    import pickle
    net = made(SmallCNN).eval()
    buffer = io.BytesIO()
    torch.save(net, buffer)
    buffer.seek(0)
    twin = torch.load(buffer, weights_only=False)
    third = pickle.loads(pickle.dumps(net))
    x, _ = images()
    with torch.no_grad():
        return num(twin(x)) + num(third(x)) + (next(twin.parameters()).device.type == device.type)


@test
def save_on_cpu_load_on_gpu_and_back():
    net = SmallCNN()
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as folder:
        torch.save(net.state_dict(), f"{folder}/cpu.pt")
        on_gpu = torch.load(f"{folder}/cpu.pt", map_location=device)
        torch.save(on_gpu, f"{folder}/gpu.pt")
        on_cpu = torch.load(f"{folder}/gpu.pt", map_location="cpu")
        size = os.path.getsize(f"{folder}/gpu.pt")
    return sum(num(v) for v in on_gpu.values()) + sum(num(v) for v in on_cpu.values()) + size + all(v.device.type == "cpu" for v in on_cpu.values())


@test
def safetensors_round_trip():
    from safetensors.torch import load_file, save_file
    net = made(SmallCNN)
    with tempfile.TemporaryDirectory() as folder:
        save_file(net.state_dict(), f"{folder}/model.safetensors")
        back = load_file(f"{folder}/model.safetensors", device=str(device))
    return sum(num(v) for v in back.values()) + all(v.device.type == device.type for v in back.values())


# --- Lightning ---------------------------------------------------------------------------------

def lightning_module():
    import lightning.pytorch as pl
    import torchmetrics

    class Lit(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.net = SmallCNN()
            self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
            self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=10, average="macro")
            self.confusion = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=10)

        def training_step(self, batch, index):
            x, y = batch
            loss = F.cross_entropy(self.net(x), y)
            self.log("train_loss", loss, on_step=True, on_epoch=True)
            return loss

        def validation_step(self, batch, index):
            x, y = batch
            logits = self.net(x)
            self.accuracy(logits, y)
            self.f1(logits, y)
            self.confusion.update(logits, y)
            self.log("val_loss", F.cross_entropy(logits, y))
            self.log("val_acc", self.accuracy)
            self.log("val_f1", self.f1)

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
            return {"optimizer": optimizer, "lr_scheduler": torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.9)}

    return Lit


def lightning_fit(folder, epochs, checkpoint=None, precision="32-true", workers=0):
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import TensorBoardLogger
    pl.seed_everything(0, workers=True, verbose=False)
    module = lightning_module()()
    loader = torch.utils.data.DataLoader(Squares(), batch_size=16, num_workers=workers, persistent_workers=workers > 0)
    saver = ModelCheckpoint(dirpath=folder, save_last=True, monitor="val_acc", mode="max")
    trainer = pl.Trainer(max_epochs=epochs, accelerator="mps" if device.type == "mps" else "cpu", devices=1, precision=precision,
                         logger=TensorBoardLogger(folder, name="tb"), callbacks=[saver], enable_progress_bar=False,
                         enable_model_summary=False, log_every_n_steps=1, num_sanity_val_steps=1)
    trainer.fit(module, loader, loader, ckpt_path=checkpoint)
    metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
    events = [f for root, _, files in os.walk(f"{folder}/tb") for f in files if "tfevents" in f]
    return module, metrics, saver, len(events) > 0


@test
def lightning_trainer_validation_metrics_tensorboard():
    with tempfile.TemporaryDirectory() as folder:
        module, metrics, _, logged = lightning_fit(folder, 2)
        return sum(metrics.values()) + weights_of(module) + num(module.confusion.compute()) + logged


@test
def lightning_checkpoint_resumed():
    with tempfile.TemporaryDirectory() as folder:
        whole, _, _, _ = lightning_fit(f"{folder}/whole", 3)
        _, _, saver, _ = lightning_fit(f"{folder}/parts", 2)
        resumed, metrics, _, _ = lightning_fit(f"{folder}/parts", 3, checkpoint=saver.last_model_path)
        same = weights_of(whole) == weights_of(resumed)
        return weights_of(resumed) + sum(metrics.values()) + (0.0 if same else 1000.0)


@test
def lightning_mixed_precision_and_workers():
    with tempfile.TemporaryDirectory() as folder:
        module, metrics, _, _ = lightning_fit(folder, 2, precision="16-mixed", workers=2)
        return sum(metrics.values()) + weights_of(module)


# --- random numbers ----------------------------------------------------------------------------

@test
def random_every_distribution_from_a_seed():
    torch.manual_seed(123)
    parts = [torch.randn(5, 7, device=device), torch.rand(5, 7, device=device), torch.randint(0, 100, (5, 7), device=device).float(),
             torch.randperm(35, device=device).float().reshape(5, 7), torch.empty(5, 7, device=device).normal_(1.0, 2.0),
             torch.empty(5, 7, device=device).uniform_(-1, 1), torch.bernoulli(torch.full((5, 7), 0.3, device=device)),
             torch.empty(5, 7, device=device).exponential_(2.0), torch.normal(torch.zeros(5, 7, device=device), torch.ones(5, 7, device=device)),
             torch.multinomial(torch.ones(5, 20, device=device), 7).float(), torch.randn_like(torch.empty(5, 7, device=device)),
             torch.empty(5, 7, device=device).bernoulli_(0.5), torch.empty(5, 7, device=device).random_(0, 10)]
    return sum(num(p) for p in parts)


@test
def random_seed_repeats_and_differs():
    torch.manual_seed(7)
    a = torch.randn(64, device=device)
    b = torch.randn(64, device=device)
    torch.manual_seed(7)
    c = torch.randn(64, device=device)
    torch.mps.manual_seed(7) if device.type == "mps" else torch.manual_seed(7)
    d = torch.randn(64, device=device)
    return num(a) + float(torch.equal(a, c)) + 2 * float(torch.equal(a, d)) + 4 * float(torch.equal(a, b))


@test
def random_generator_of_its_own():
    generator = torch.Generator(device=device).manual_seed(99)
    a = torch.randn(4, 5, device=device, generator=generator)
    b = torch.randint(0, 50, (4, 5), device=device, generator=generator)
    state = generator.get_state()
    c = torch.rand(4, 5, device=device, generator=generator)
    generator.set_state(state)
    d = torch.rand(4, 5, device=device, generator=generator)
    torch.manual_seed(1)
    untouched = torch.randn(3, device=device)  # the default generator is another one
    e = torch.randperm(20, device=device, generator=generator)
    return num(a) + num(b) + num(c) + float(torch.equal(c, d)) + num(untouched) + num(e) + generator.initial_seed()


@test
def random_dropout_layers():
    torch.manual_seed(5)
    x = torch.ones(8, 16, 4, 4, device=device)
    parts = [F.dropout(x, 0.3, training=True), nn.Dropout2d(0.4)(x), F.alpha_dropout(x, 0.2, training=True),
             F.dropout1d(x.flatten(2), 0.5, training=True), nn.Dropout(0.9)(x)]
    return sum(num(p) for p in parts)


@test
def random_rrelu():  # not on the GPU before PyTorch 2.13: there it must fail as on the Mac
    torch.manual_seed(5)
    return num(F.rrelu(torch.ones(8, 16, device=device) - 1.5, training=True))


@test
def random_initializations_on_the_gpu():
    torch.manual_seed(11)
    total = 0.0
    for init in (nn.init.kaiming_uniform_, nn.init.kaiming_normal_, nn.init.xavier_uniform_, nn.init.xavier_normal_,
                 lambda w: nn.init.trunc_normal_(w, std=0.02), lambda w: nn.init.normal_(w, 0, 0.1), lambda w: nn.init.uniform_(w, -0.1, 0.1),
                 lambda w: nn.init.constant_(w, 0.5), nn.init.zeros_, nn.init.ones_, nn.init.eye_, nn.init.orthogonal_):
        weight = torch.empty(12, 12, device=device)
        init(weight)
        total += num(weight)
    conv = nn.Conv2d(3, 8, 3).to(device)
    conv.reset_parameters()
    return total + num(conv.weight)


@test
def random_module_built_on_the_gpu():
    torch.manual_seed(3)
    net = nn.Sequential(nn.Linear(8, 16, device=device), nn.LayerNorm(16, device=device), nn.Embedding(10, 4, device=device))
    return weights_of(net)


# --- gradients of the less ordinary operators --------------------------------------------------

def grad_of(function, *inputs):
    inputs = [i.clone().requires_grad_() for i in inputs]
    function(*inputs).sum().backward()
    return sum(num(i.grad) for i in inputs)


@test
def gradient_lstm_over_a_sequence():
    """An LSTM over a whole sequence is one operator of the GPU's on a Mac, with a gradient of its own: two steps of
    training one, its weights' gradients (a list of tensors) and what they do to the weights."""
    torch.manual_seed(0)
    lstm = nn.LSTM(6, 12, num_layers=2, batch_first=True, bidirectional=True).to(device)
    head = nn.Linear(24, 3).to(device)
    optimizer = torch.optim.SGD(list(lstm.parameters()) + list(head.parameters()), lr=0.05)
    x, target = rnd(4, 9, 6), rnd(4, 3, seed=1)
    total = 0.0
    for _ in range(2):
        optimizer.zero_grad()
        out, (h, c) = lstm(x)
        loss = F.mse_loss(head(out[:, -1]), target) + 0.01 * (h.pow(2).mean() + c.pow(2).mean())
        loss.backward()
        total += float(loss) + sum(num(p.grad) for p in lstm.parameters())
        optimizer.step()
    return total + weights_of(lstm) + grad_of(lambda i: lstm(i)[0], x)


@test
def deterministic_algorithms_asked_for():
    """torch.use_deterministic_algorithms(True) has PyTorch refuse the operators that do not give the same bits
    twice. The gradient of an embedding is one, on the GPU (its repeated indices are added up in any order): asked
    for where it is computed, it is refused there, and comes back once the setting is off again."""
    weight = rnd(10, 4).requires_grad_()
    index = torch.tensor([1, 1, 1, 3], device=device)
    torch.use_deterministic_algorithms(True)
    try:
        F.embedding(index, weight).sum().backward()
        torch.mps.synchronize()
        refused = 0
    except RuntimeError as e:
        refused = 1 if "deterministic" in str(e) else 2
    finally:
        torch.use_deterministic_algorithms(False)
    weight.grad = None
    F.embedding(index, weight).sum().backward()
    return 1000 * refused + num(weight.grad)


@test
def gradient_cdist_large_and_norms():
    a, b = rnd(2, 40, 3), rnd(2, 50, 3, seed=1)
    return (grad_of(lambda x, y: torch.cdist(x, y), a, b) + grad_of(lambda x: torch.cdist(x, x).topk(5, largest=False).values, a)
            + grad_of(lambda x: x.norm(dim=-1), a))


@test
def gradient_cdist_small_and_other_norms():  # its backward is not on the GPU before PyTorch 2.13: there it must fail as on the Mac
    a, b = rnd(2, 40, 3), rnd(2, 50, 3, seed=1)
    return grad_of(lambda x, y: torch.cdist(x, y, p=1), a, b) + grad_of(lambda x, y: torch.cdist(x[:, :8], y[:, :8]), a, b)


@test
def gradient_gather_scatter_index():
    x = rnd(6, 8)
    index = torch.tensor([[0, 2, 2, 5], [1, 1, 7, 3]] * 3, device=device)
    rows = torch.tensor([0, 3, 3, 5], device=device)
    total = grad_of(lambda t: t.gather(1, index) ** 2, x)
    total += grad_of(lambda t: torch.zeros(6, 8, device=device).scatter_add(1, index, t[:, :4]) ** 2, x)
    total += grad_of(lambda t: torch.zeros(6, 8, device=device).scatter(1, index[:, :3], t[:, :3]) ** 2, x)
    total += grad_of(lambda t: t[rows] ** 2 + t[rows, rows].unsqueeze(1), x)
    total += grad_of(lambda t: torch.zeros(6, 8, device=device).index_put((rows,), t[:4], accumulate=True) ** 2, x)
    total += grad_of(lambda t: torch.zeros(6, 8, device=device).index_add(0, rows, t[:4]) ** 2, x)
    total += grad_of(lambda t: t.index_select(1, rows) * t.take_along_dim(index, 1), x)
    total += grad_of(lambda t: t.masked_fill(t > 0, 0.0) + torch.where(t > 0, t, t * 0.1) + t[t > 0.5].sum(), x)
    return total


@test
def gradient_topk_sort_max_cumulative():
    x = rnd(5, 9)
    return (grad_of(lambda t: t.topk(3, dim=1).values ** 2, x) + grad_of(lambda t: t.sort(dim=1).values * torch.arange(9.0, device=device), x)
            + grad_of(lambda t: t.max(dim=1).values + t.min(dim=0).values.sum() + t.amax(1), x) + grad_of(lambda t: t.cumsum(1) ** 2 + t.cumprod(0), x)
            + grad_of(lambda t: t.kthvalue(2, dim=1).values + t.median(dim=1).values, x) + grad_of(lambda t: t.logsumexp(1) + t.softmax(1).max(1).values, x))


@test
def gradient_grid_sample_and_resizing():
    image, grid = rnd(2, 3, 8, 8), torch.tanh(rnd(2, 6, 6, 2, seed=1))
    return (num(F.grid_sample(image, grid, align_corners=False))
            + grad_of(lambda i: F.interpolate(i, scale_factor=2, mode="bilinear", align_corners=False), image)
            + grad_of(lambda i: F.interpolate(i, size=(5, 7), mode="nearest") + F.adaptive_avg_pool2d(i, 4).sum(), image)
            + grad_of(lambda i: F.max_pool2d(i, 2) ** 2 + F.avg_pool2d(i, 2), image) + grad_of(lambda i: F.unfold(i, 3).sum(1) + F.pad(i, (1, 1, 1, 1), mode="reflect").mean(), image)
            + grad_of(lambda i: F.pixel_shuffle(i.repeat(1, 4, 1, 1)[:, :12], 2) ** 2, image))


@test
def gradient_grid_sample():  # its backward is not on the GPU before PyTorch 2.13: there it must fail as on the Mac
    image, grid = rnd(2, 3, 8, 8), torch.tanh(rnd(2, 6, 6, 2, seed=1))
    return grad_of(lambda i, g: F.grid_sample(i, g, align_corners=False), image, grid)


@test
def gradient_second_derivative():
    x = rnd(4, 6).requires_grad_()
    w = rnd(6, 6, seed=1).requires_grad_()
    y = torch.tanh(x @ w).pow(2).sum()
    first, = torch.autograd.grad(y, x, create_graph=True)
    penalty = first.pow(2).sum()
    second_x, second_w = torch.autograd.grad(penalty, (x, w))
    return float(y) + num(first) + num(second_x) + num(second_w)


@test
def gradient_penalty_training():
    net = made(lambda: nn.Sequential(nn.Linear(8, 32), nn.Tanh(), nn.Linear(32, 1)))
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    x = rnd(16, 8).requires_grad_()
    for _ in range(4):
        out = net(x)
        gradient, = torch.autograd.grad(out.sum(), x, create_graph=True)
        loss = out.mean() + ((gradient.norm(dim=1) - 1) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return float(loss) + weights_of(net)


@test
def gradient_torch_func():
    from torch.func import functional_call, grad, hessian, jacrev, vmap
    net = made(lambda: nn.Sequential(nn.Linear(6, 12), nn.Tanh(), nn.Linear(12, 1)))
    parameters = {k: v.detach() for k, v in net.named_parameters()}
    x, y = rnd(10, 6), rnd(10, seed=1)

    def loss(p, sample, target):
        return (functional_call(net, p, (sample.unsqueeze(0),)).squeeze() - target) ** 2

    per_sample = vmap(grad(loss), in_dims=(None, 0, 0))(parameters, x, y)
    jacobian = jacrev(lambda v: torch.tanh(v) * v.sum())(x[0])
    curvature = hessian(lambda v: (v ** 3).sum())(x[0])
    return sum(num(g) for g in per_sample.values()) + num(jacobian) + num(curvature) + num(vmap(torch.dot)(x, x))


@test
def gradient_custom_function_hooks_checkpoint():
    from torch.utils.checkpoint import checkpoint

    class Clamp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            ctx.save_for_backward(value)
            return value.clamp(-0.5, 0.5)

        @staticmethod
        def backward(ctx, upstream):
            value, = ctx.saved_tensors
            return upstream * (value.abs() < 1.0).float()

    seen = []
    x = rnd(4, 6).requires_grad_()
    hidden = Clamp.apply(x * 2)
    hidden.register_hook(lambda g: seen.append(num(g)) or g * 2)
    block = made(lambda: nn.Sequential(nn.Linear(6, 6), nn.GELU()))
    out = checkpoint(block, hidden, use_reentrant=False).sum()
    out.backward(retain_graph=True)
    first = num(x.grad)
    out.backward()
    return first + num(x.grad) + sum(seen)


@test
def gradient_detect_anomaly_and_no_grad_islands():
    x = rnd(4, 6).requires_grad_()
    with torch.autograd.detect_anomaly(check_nan=True):
        y = (x.exp() + 1).log()
        with torch.no_grad():
            scale = y.mean()
        with torch.enable_grad():
            z = (y / scale).sum()
        z.backward()
    frozen = x.detach().clone()
    frozen.requires_grad_(False)
    return num(x.grad) + float(z) + float(frozen.is_leaf)


@test
def gradient_through_in_place_and_views():
    x = rnd(4, 6).requires_grad_()
    y = x * 2
    y[:, 0] = 0
    y.mul_(3)
    y[1:3].add_(x[1:3])
    z = y.t().contiguous().view(-1)[::2]
    z.sum().backward()
    return num(x.grad) + num(y)


@test
def attention_in_every_setting():
    q, k, v = rnd(2, 4, 9, 16), rnd(2, 4, 11, 16, seed=1), rnd(2, 4, 11, 16, seed=2)
    keep, bias = rnd(2, 1, 9, 11, seed=3) > -0.5, rnd(9, 11, seed=4)
    cases = [lambda: F.scaled_dot_product_attention(q, k, v), lambda: F.scaled_dot_product_attention(q, q, q, is_causal=True),
             lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=keep), lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=bias),
             lambda: F.scaled_dot_product_attention(q, k, v, scale=0.3), lambda: F.scaled_dot_product_attention(q[0], k[0], v[0]),
             lambda: F.scaled_dot_product_attention(q[:, :, ::2], k[:, :, ::2], v[:, :, ::2]),
             lambda: F.scaled_dot_product_attention(q.permute(1, 0, 2, 3), k.permute(1, 0, 2, 3), v.permute(1, 0, 2, 3)),
             lambda: F.scaled_dot_product_attention(q.half(), k.half(), v.half()).float(),
             lambda: F.scaled_dot_product_attention(q, k[:, :2], v[:, :2], enable_gqa=True)]
    total = 0.0
    for case in cases:
        with torch.no_grad():
            total += num(case())
        q.requires_grad_(True)
        q.grad = None
        out = case()
        out.sum().backward()
        total += num(out) + num(q.grad)
        q.requires_grad_(False)
    return total


@test
def attention_with_dropout():  # PyTorch 2.15 refuses it on the GPU outside a backward pass: there it must fail as on the Mac
    q, k, v = rnd(2, 4, 9, 16), rnd(2, 4, 11, 16, seed=1), rnd(2, 4, 11, 16, seed=2)
    torch.manual_seed(3)
    with torch.no_grad():
        return num(F.scaled_dot_product_attention(q, k, v, dropout_p=0.3))


# --- values in a tight loop --------------------------------------------------------------------

@test
def values_item_nonzero_masks_unique_in_a_loop():
    x = rnd(256)
    total = 0.0
    for step in range(60):
        y = x * (1 + step / 60)
        total += y.max().item() + int((y > 1).sum()) + float(y[y > 1.5].sum()) + torch.nonzero(y > 2).numel()
        total += torch.unique((y * 2).round()).numel() + bool((y > 3).any()) + len(y[y < -1].tolist())
        if y.mean() > 0.01:
            total += 1
    return total


@test
def values_shapes_that_depend_on_data():
    x = (rnd(40, 6) * 3).round()
    kept = x[x[:, 0] > 0]
    values, inverse, counts = torch.unique(x[:, 0], return_inverse=True, return_counts=True)
    rows = torch.unique(x[:, :2], dim=0)
    where = torch.where(x > 2)
    selected = torch.masked_select(x, x < -1)
    consecutive = torch.unique_consecutive(x[:, 1].sort().values)
    counted = torch.bincount(x[:, 2].abs().long(), minlength=3)
    return (num(kept) + num(values) + num(inverse) + num(counts) + num(rows) + sum(num(w) for w in where) + num(selected) + num(consecutive)
            + num(counted) + kept.shape[0] + float(torch.count_nonzero(x)) + num(x.argwhere()[:5]))


# --- threads and processes ---------------------------------------------------------------------

def small_net(seed):
    """A small network whose weights depend on nothing but `seed`: not on the global generator, which threads share."""
    net = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 3))
    for i, parameter in enumerate(net.parameters()):
        parameter.data = rnd_cpu(*parameter.shape, seed=100 * seed + i) * 0.2
    return net.to(device)


def train_small(seed, out, index):
    net = small_net(seed)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    x, y = rnd(32, 10, seed=seed), (torch.arange(32) % 3).to(device)
    for step in range(40):
        optimizer.zero_grad()
        loss = F.cross_entropy(net(x * (1 + step % 3)), y)  # shapes repeat, constants change
        loss.backward()
        optimizer.step()
        if step % 5 == 0:
            float(loss)
    out[index] = float(loss) + weights_of(net)


@test
def threads_training_at_once():
    alone = [None] * 4
    for i in range(4):
        train_small(i, alone, i)
    together = [None] * 4
    threads = [threading.Thread(target=train_small, args=(i, together, i)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return sum(together) + (0.0 if together == alone else 1000.0)  # each thread's training is the one it is alone


@test
def threads_reading_values_at_once():
    x = rnd(64, 64)
    results = [0.0] * 8

    def work(index):
        total = 0.0
        for step in range(30):
            y = (x + index) @ x
            total += y.sum().item() + int((y > step).sum()) + float(y[index].cpu().sum())
        results[index] = total

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return sum(results)


def child_training(seed, queue):
    out = [None]
    train_small(seed, out, 0)
    queue.put(out[0])


def infer_small(seed):
    net = small_net(seed).eval()
    with torch.no_grad():
        return num(net(rnd(32, 10, seed=seed)).softmax(-1)) + float(torch.mps.current_allocated_memory() > 0 if device.type == "mps" else 1)


def child_inference(seed, queue):
    queue.put(infer_small(seed))


def with_processes(method, target, alone):
    context = multiprocessing.get_context(method)
    queue = context.Queue()
    before = rnd(8, 8)
    pending = before @ before  # operations of the parent's not yet sent when the children start
    processes = [context.Process(target=target, args=(seed, queue)) for seed in range(3)]
    for process in processes:
        process.start()
    try:
        values = sorted(queue.get(timeout=120) for _ in processes)
    finally:
        for process in processes:
            process.join(5)
            if process.is_alive():
                process.kill()
    mine = sorted(alone(seed) for seed in range(3))  # the same work, in this process
    return sum(values) + num(pending) + (0.0 if values == mine else 1000.0)


def trained_alone(seed):
    out = [None]
    train_small(seed, out, 0)
    return out[0]


@test
def processes_spawned_each_training_on_the_gpu():
    return with_processes("spawn", child_training, trained_alone)


@test
def processes_forked_each_on_the_gpu():
    # A forked child gets a connection of its own to the Mac (on a Mac a forked child cannot use
    # the GPU at all: there this runs with spawn). It does not train: PyTorch itself refuses a
    # backward pass in a process forked from one that has run a backward pass with a GPU present.
    return with_processes("fork" if sys.platform == "linux" else "spawn", child_inference, infer_small)


@test
def processes_run_as_programs():
    code = ("import torch; d = 'mps' if torch.backends.mps.is_available() else 'cpu'; torch.manual_seed(0); "
            "print(float((torch.randn(100, device=d) ** 2).sum()))")
    runs = [subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True) for _ in range(3)]
    values = [float(run.communicate()[0].strip().splitlines()[-1]) for run in runs]
    return values[0] + (0.0 if len(set(values)) == 1 else 1000.0)


# --- memory ------------------------------------------------------------------------------------

@test
def memory_follows_tensors():
    if device.type != "mps":
        return 0.0
    torch.mps.synchronize()
    torch.mps.empty_cache()
    start = torch.mps.current_allocated_memory()
    big = torch.zeros(64, 1024, 1024, device=device)  # 256 MB
    big += 1
    torch.mps.synchronize()
    held = torch.mps.current_allocated_memory() - start
    driver = torch.mps.driver_allocated_memory()
    del big
    torch.mps.synchronize()
    freed = torch.mps.current_allocated_memory() - start
    torch.mps.empty_cache()
    after = torch.mps.driver_allocated_memory()
    megabyte = 1 << 20
    return (float(250 * megabyte <= held <= 300 * megabyte) + 2 * float(freed < 8 * megabyte) + 4 * float(driver >= held)
            + 8 * float(after < driver) + 16 * float(torch.mps.recommended_max_memory() > 1 << 30) + 32 * torch.mps.device_count())


@test
def memory_of_a_large_model():
    if device.type != "mps":
        return 0.0
    torch.mps.empty_cache()
    start = torch.mps.current_allocated_memory()
    torch.manual_seed(0)
    net = nn.Sequential(nn.Embedding(200_000, 1024), nn.Linear(1024, 4096), nn.GELU(), nn.Linear(4096, 4096), nn.GELU(), nn.Linear(4096, 1024))
    nn.init.uniform_(net[0].weight, -1.0, 1.0)  # see rnd_cpu
    net = net.to(device)  # about 1.1 GB of weights
    torch.mps.synchronize()
    held = torch.mps.current_allocated_memory() - start
    tokens_ = (torch.arange(64 * 128) % 200_000).reshape(64, 128).to(device)
    loss = net(tokens_).pow(2).mean()
    loss.backward()
    value = float(loss) + num(net[1].weight.grad[:4, :4])
    del net, loss
    torch.mps.empty_cache()
    return value + float(1000 << 20 <= held <= 1300 << 20)


@test
def memory_given_back_after_operations_that_wait():
    """A tensor the program has let go of is no longer on the GPU, whichever operation it went through last.

    These are the operations a container cannot send ahead and forget: their result depends on the
    data (an index with a mask, and its backward pass), or they move data to or from the CPU. Each
    is 64 MB; what is still in use after each, in bytes, is part of the number: none on a Mac.
    """
    if device.type != "mps":
        return 0.0
    import gc

    def used():
        gc.collect()
        torch.mps.synchronize()
        return torch.mps.current_allocated_memory()

    def masked():
        big = torch.ones(16, 1024, 1024, device=device)
        return float(big[big[:, 0, 0] > 0].sum())

    def put_by_mask():
        big = torch.zeros(16, 1024, 1024, device=device)
        mask = torch.arange(16, device=device) % 2 == 0
        return float(big.index_put((mask,), torch.ones(8, 1024, 1024, device=device)).sum())

    def gradient_through_a_mask():
        big = torch.ones(16, 1024, 1024, device=device, requires_grad=True)
        big[torch.arange(16, device=device) % 2 == 0].sum().backward()
        return float(big.grad.sum())

    def indexed_from_the_cpu():
        big = torch.ones(16, 1024, 1024, device=device)
        return float(big[torch.arange(0, 16, 2)].sum())

    def sent_and_fetched():
        big = torch.ones(16, 1024, 1024).to(device) * 2
        return float(big.cpu().sum()) + float(torch.unique(big).numel())

    start, total, left = used(), 0.0, 0
    for piece in (masked, put_by_mask, gradient_through_a_mask, indexed_from_the_cpu, sent_and_fetched):
        total += piece()
        left += abs(used() - start)
    return total / (1 << 20) + left


@test
def errors_are_the_macs():
    """What fails on a Mac fails here, with the same kind of error, where it happens."""
    seen = []

    def kind(function):
        try:
            function()
            seen.append("nothing")
        except Exception as e:  # noqa: BLE001
            seen.append(type(e).__name__)

    x = rnd(4, 5)
    kind(lambda: x.double())
    kind(lambda: torch.zeros(3, dtype=torch.float64, device=device))
    kind(lambda: x @ rnd(4, 5))
    kind(lambda: x.view(3, 7))
    kind(lambda: x[10])
    kind(lambda: torch.cat([x, rnd(3, 3)]))
    kind(lambda: x + torch.zeros(4, 5, device="cpu"))
    kind(lambda: torch.zeros(4, 5) * x)
    kind(lambda: torch.where(x > 0, x, torch.zeros(4, 5)))
    kind(lambda: F.linear(x, torch.ones(3, 5)))
    kind(lambda: F.cross_entropy(x, torch.tensor([0, 1, 2, 3])))
    kind(lambda: x.masked_fill(torch.ones(4, 5) > 0, 1.0))
    kind(lambda: torch.equal(x, torch.zeros(4, 5)))
    kind(lambda: x.clone().index_put_((torch.tensor([0, 1], device=device),), torch.zeros(2, 5)))
    kind(lambda: torch.take(x, torch.tensor([0, 1], device=device)).sum().item())
    kind(lambda: x.numpy())
    kind(lambda: torch.randn(3, device=device, generator=torch.Generator().manual_seed(0)) if device.type == "mps" else x.double().numpy().foo)
    print("KINDS " + " ".join(seen))
    return float(len(seen))


OUT_OF_MEMORY = """
import torch
torch.mps.set_per_process_memory_fraction(0.01)  # of what the Mac recommends: about a gigabyte here
try:
    held = [torch.ones(256, 1024, 1024, device="mps") for _ in range(8)]  # 8 GB
    print("allocated", float(sum(t[0, 0, 0] for t in held)))
except Exception as e:
    print(type(e).__name__, "out of memory" in str(e))
"""


@test
def memory_running_out_is_an_error():
    if device.type != "mps":
        return 0.0
    run = subprocess.run([sys.executable, "-c", OUT_OF_MEMORY], capture_output=True, text=True, timeout=120)
    last = (run.stdout.strip().splitlines() or ["nothing printed"])[-1]
    print("OOM " + last)
    return float(last == "RuntimeError True")


@test
def accelerator_and_event_calls():
    if device.type != "mps":
        return 0.0
    # (Not Event.elapsed_time: on the Mac itself it can wait for ever, after a copy from the CPU.)
    x = rnd(256, 256)
    event = torch.mps.event.Event()
    for _ in range(20):
        x = torch.tanh(x @ x) * 0.1
    event.record()
    event.synchronize()
    elapsed = 0.0
    second = event
    torch.accelerator.synchronize()
    return (float(torch.accelerator.current_accelerator().type == "mps") + torch.accelerator.device_count() + float(torch.accelerator.is_available())
            + float(torch.backends.mps.is_macos_or_newer(13, 0)) + float(torch.backends.mps.is_built()) + float(elapsed >= 0) + float(second.query()) + num(x))


def main():
    only = os.environ.get("ONLY")
    print(exact.info("compat_train", device), flush=True)
    for fn in tests:
        if only and only not in fn.__name__:
            continue
        exact.begin()
        try:
            value = fn()
            print(f"PASS {fn.__name__} {value:.9g}", flush=True)
            print(f"DATA {fn.__name__} {exact.data(value, fn.__name__)}", flush=True)
        except Exception as e:  # noqa: BLE001 - the point is to list what does not work
            print(f"FAIL {fn.__name__} {type(e).__name__}: {str(e).splitlines()[0][:150]}", flush=True)
            if os.environ.get("TRACE"):
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
