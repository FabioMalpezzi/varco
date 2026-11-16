# Example 4: PyTorch training

A small PyTorch Lightning training with mixed precision, data loader
workers, a checkpoint per epoch and a resume, in a container, computed on
the Mac's GPU through the PyTorch bridge. The data is synthetic, made from a
seed: nothing is downloaded.

```sh
./run.sh            # 3 epochs in a container, then a resume for one more
./run.sh native     # the same on the Mac, to compare
```

The checkpoints are written to `out/`. What it shows, the measured figures
and the limits:
[docs/use-cases/pytorch-training.md](../../docs/use-cases/pytorch-training.md).

| File | |
|---|---|
| `train.py` | The training: `python train.py --out DIR [--epochs N] [--resume]` |
| `compose.yaml` | The service: the image `varco-torch`, shared memory for the workers, `out/` |
| `compose.gpu.yaml` | The GPU request, which `run.sh` adds with `ENGINE=kiri` |
| `run.sh` | Starts the bridge's server, trains, resumes, stops the server |
