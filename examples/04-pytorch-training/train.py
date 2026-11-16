"""An ordinary PyTorch Lightning training: mixed precision, checkpoints, resuming.

It knows nothing of the bridge. A small convolutional network learns to tell
ten kinds of synthetic 32x32 images apart; the data is made from a seed, so
no dataset is downloaded and two machines train on the same values.
    python train.py --out DIR [--epochs 3] [--resume]
The first run trains for the given epochs and saves a checkpoint after each.
With --resume it goes on from the last checkpoint in DIR for one more epoch.
"""
import argparse
import logging
import os
import statistics
import time
import warnings

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

if torch.cuda.is_available():
    accelerator = "cuda"
elif torch.backends.mps.is_available():
    accelerator = "mps"
else:
    accelerator = "cpu"


class Patterns(torch.utils.data.Dataset):
    """Images of ten classes: class k is a grating of its own direction and pitch, under noise.

    Made with torch.rand, whose values are the same on macOS and on Linux.
    """

    def __init__(self, count=2048, seed=0):
        generator = torch.Generator().manual_seed(seed)
        self.labels = torch.randint(0, 10, (count,), generator=generator)
        y, x = torch.meshgrid(torch.arange(32.0), torch.arange(32.0), indexing="ij")
        angle = self.labels.float().reshape(-1, 1, 1) * 0.314
        wave = torch.sin((x * torch.cos(angle) + y * torch.sin(angle)) * (0.3 + 0.05 * self.labels.float().reshape(-1, 1, 1)))
        self.images = (wave + 1.5 * (torch.rand((count, 32, 32), generator=generator) - 0.5)).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


class Net(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, 10)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))

    def training_step(self, batch, index):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, index):
        x, y = batch
        logits = self(x)
        self.log("val_loss", F.cross_entropy(logits, y))
        self.log("val_acc", (logits.argmax(1) == y).float().mean())

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-3)


class Report(pl.Callback):
    """One line per epoch. The time of a step is the median over the epoch, from the end of
    one step to the end of the next: the first step, which waits for the data loader's workers
    to start, does not weigh on it."""

    def on_train_epoch_start(self, trainer, module):
        self.times, self.last = [], None

    def on_train_batch_end(self, trainer, module, outputs, batch, index):
        now = time.perf_counter()
        if self.last is not None:
            self.times.append(now - self.last)
        self.last = now

    def on_train_epoch_end(self, trainer, module):
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
        print(f"epoch {trainer.current_epoch}: train loss {metrics.get('train_loss', float('nan')):.4f}, "
              f"validation loss {metrics.get('val_loss', float('nan')):.4f}, accuracy {metrics.get('val_acc', float('nan')):.3f}, "
              f"{1000 * statistics.median(self.times):.1f} ms per step", flush=True)


def main():
    # Lightning's advice and notices: not what this example is about.
    warnings.filterwarnings("ignore")
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    pl.seed_everything(0, workers=True, verbose=False)
    train = torch.utils.data.DataLoader(Patterns(2048, seed=0), batch_size=64, shuffle=True, num_workers=2, persistent_workers=True)
    validation = torch.utils.data.DataLoader(Patterns(512, seed=1), batch_size=128)
    last = os.path.join(args.out, "last.ckpt")
    if args.resume and not os.path.exists(last):
        raise SystemExit(f"no checkpoint to resume from at {last}: run without --resume first")
    epochs = args.epochs + 1 if args.resume else args.epochs
    trainer = pl.Trainer(
        max_epochs=epochs, accelerator=accelerator, devices=1, precision="16-mixed",
        callbacks=[ModelCheckpoint(dirpath=args.out, save_last=True), Report()],
        logger=False, enable_progress_bar=False, enable_model_summary=False, num_sanity_val_steps=0,
    )
    print(f"torch {torch.__version__.split('+')[0]}, accelerator {accelerator}, precision 16-mixed"
          + (f", resuming from {last}" if args.resume else ""), flush=True)
    trainer.fit(Net(), train, validation, ckpt_path=last if args.resume else None)
    print(f"finished at epoch {trainer.current_epoch}, step {trainer.global_step}; checkpoint in {last}")


if __name__ == "__main__":  # the data loader's workers import this module again
    main()
