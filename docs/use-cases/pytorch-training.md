# Use case 4: PyTorch training

**What you get**

- Training code runs unchanged and gives the Mac's numbers: of 68 training
  programs, 63 or 62 were identical to the Mac by the SHA-256 of every
  tensor, depending on the run, 3 or 4 were within the declared tolerance,
  where two runs on the Mac differ too, and 2 fail on both sides alike
  (PyTorch 2.15.0, Docker Desktop and Kiri, protocol 3).
- The Mac's speed on long steps: 2.463 s per step in a Kiri container for a
  private segmentation project trained with Lightning and 16-bit autocast,
  over four runs a side in mirrored order (a field report).
- Checkpoints cross between the two. One made on the Mac was resumed in the
  container and the reverse, with an identical first loss (the same
  project, Docker Desktop, PyTorch 2.12.1).

**Compared with**

- The same training installed on the Mac: 2.477 s per step for that
  project, a ratio of 0.995. A second project with short steps took 0.026 s
  per step on the Mac, 0.030 in a Kiri container (ratio 1.136) and
  0.044–0.046 on Docker Desktop.

**When it is not the right tool:** for training whose steps last a few tens
of milliseconds. A 26 ms step is 13% slower in Kiri and about 1.7 times
slower on Docker Desktop.

**Who it is for.** Anyone who trains with PyTorch on a Mac's GPU and wants
the training to run in a container, for a reproducible environment, to run
alongside other services, or to use the same image on a Mac and elsewhere.

## What runs

[`examples/04-pytorch-training/train.py`](../../examples/04-pytorch-training/train.py)
is a small, self-contained PyTorch Lightning training:

- a convolutional network that learns to tell ten kinds of synthetic 32×32
  images apart; the data is made from a seed, so nothing is downloaded and
  two machines train on the same values;
- `precision="16-mixed"` (autocast);
- a `DataLoader` with two worker processes;
- a checkpoint after every epoch, and `--resume`, which goes on from the
  last checkpoint for one more epoch.

It imports nothing from the bridge.

## Commands

Once: `scripts/setup-mac.sh pytorch` ([setup](../setup.md)).

```sh
examples/04-pytorch-training/run.sh            # 3 epochs in a container, then a resume
examples/04-pytorch-training/run.sh native     # the same on the Mac
ENGINE=kiri examples/04-pytorch-training/run.sh   # the container in Kiri
```

The checkpoints are written to `examples/04-pytorch-training/out/`.

## What you see

In the container:

```
torch 2.15.0, accelerator mps, precision 16-mixed
epoch 0: train loss 0.7559, validation loss 0.1934, accuracy 1.000, 6.7 ms per step
epoch 1: train loss 0.1050, validation loss 0.0402, accuracy 1.000, 6.8 ms per step
epoch 2: train loss 0.0279, validation loss 0.0130, accuracy 1.000, 7.2 ms per step
finished at epoch 3, step 96; checkpoint in /out/container/last.ckpt
torch 2.15.0, accelerator mps, precision 16-mixed, resuming from /out/container/last.ckpt
epoch 3: train loss 0.0119, validation loss 0.0079, accuracy 1.000, 6.8 ms per step
finished at epoch 4, step 128; checkpoint in /out/container/last.ckpt
```

Natively the losses printed were the same to the four digits shown, and the
steps took 3.8 to 5.3 ms. Steps this short show the bridge's fixed cost,
which is the point of the example (see the limits below). This is one run,
shown as an illustration.

## Measured

- In the training battery (`pytorch/tests/compat_train.py`, 68 programs),
  63 were identical in the container, 3 were within the declared tolerance
  and 2 fail on both sides alike, in one run with Docker Desktop and in one
  through Kiri, on the released code. A second run with Docker Desktop
  gave 62, 4 and 2. The battery covers optimizers and schedulers, autocast
  and `GradScaler`, data loader workers started in every way, saving and
  resuming, Lightning with checkpoints and TensorBoard, gradients, threads
  and processes. The three that are close in every run are two training
  loops of a transformer and a loop whose embedding has repeated indices.
  The fourth is the gradient of `grid_sample`, which is identical in some
  runs
  ([benchmarks](../benchmarks.md#measured-again-on-the-released-code)).
- The first field report has long steps. A private point-cloud segmentation
  project that trains with PyTorch Lightning and 16-bit autocast at batch 2
  ran unchanged in the container. It took 2.477 s per step on the Mac and
  2.463 in a Kiri container (ratio 0.995), over four runs a side in mirrored
  order. The first step is identical in 8 runs of 8; later steps differ as
  two Mac runs do. Container memory 732–785 MB. A checkpoint made on the Mac
  was resumed in the container and the reverse, with an identical first
  loss.
- The second field report has short steps. A second training project, a
  point-network pilot with a fixed seed, took 0.026 s per step on the Mac,
  0.030 in a Kiri container (ratio 1.136) and 0.044–0.046 on Docker Desktop.
  The loss was identical to the bit at every step and the weights were
  identical, in every run.

Neither field report can be reproduced by a reader. Both were taken at
protocol 1, before the review fixes.

## Limits

- Short steps pay the bridge's fixed cost: 13% on a 26 ms step in Kiri, more
  on Docker Desktop, and nothing measurable on a 2.5 s step. What costs is
  every read of a value (a logged loss, a metric) and every batch moved to
  the device.
- Training is not always reproducible on the Mac itself. The gradient of
  rows taken by a repeated index is summed by Metal in varying order: 18
  gradient tensors of 82 changed between two computations of the same step
  in the segmentation project. The container then differs from the Mac as
  the Mac differs from itself. `torch.use_deterministic_algorithms(True)`
  reaches the Mac's process, where PyTorch refuses the operators that do
  not give the same bits twice.
- CPU work is done by the virtual machine. A stage of the segmentation
  project that works in numpy was 50% slower in the container (4.7 s over 40
  steps, where the Mac took 1.9 s). Its PyTorch part took 2.5 s in the
  container and 2.4 s on the Mac. Give the virtual machine enough CPUs for
  your data loading.
- A checkpoint's file hash differs between Mac and container although the
  weights in it are identical. The cause was not verified. It matters if you
  record checkpoint hashes.
- PyTorch versions differ a lot on MPS. On the Mac, the segmentation
  project's step was 5 to 20% faster with 2.15.0 than with 2.12.1, in half
  the GPU memory (21.4–22.3 GiB instead of 43.7–44.7)
  ([benchmarks](../benchmarks.md#pytorch-2121-and-2150-compared-on-the-mac)).
- There are no CUDA extensions and no float64.
- Data loader workers pass batches through `/dev/shm`, so give the container
  `shm_size` (the example sets 2 GB).

Further measurements will be published as they are taken.
