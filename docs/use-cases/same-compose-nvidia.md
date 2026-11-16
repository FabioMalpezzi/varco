# Use case 5: the same Compose file on a Mac and on an NVIDIA server

**What you get**

- One Compose file and one program for both machines. With the patches for
  Kiri, the GPU request an NVIDIA server understands (`--gpus all`, or
  Compose's `capabilities: [gpu]`) gives a container its bridge on a Mac.
  On the released code the PyTorch batteries were run in Kiri in
  containers started that way: 47 of 47 inference programs were identical
  to the Mac by the SHA-256 of every tensor, and of 68 training programs 63
  were identical, 3 were within the declared tolerance and 2 fail on both
  sides alike (PyTorch 2.15.0, protocol 3).
- The request is also the permission. A container started with `--gpus all`
  in Kiri has `KIRI_GPU_BRIDGE=127.0.0.1:47700` and a listener at that
  address. One started without it has neither:
  `torch.backends.mps.is_available()` is `False` in it, and a program that
  asks for `"mps"` regardless is told to start the container with
  `--gpus all`.

**Compared with**

- The same program installed on the Mac, which is what the batteries
  compare each container with: the results above. A training with 26 ms
  steps took 0.026 s per step on the Mac and 0.029–0.030 in a Kiri
  container.

**When it is not the right tool:** for programs that need float64 on the
GPU or CUDA extensions such as `torch-scatter`. Those run on the NVIDIA
server and not on the Mac's GPU, with or without the bridge.

**Who it is for.** A team that develops on Macs and deploys on Linux servers
with NVIDIA GPUs, and wants one Compose file and one program for both,
instead of a GPU path that exists only in production.

## The idea

On a Linux server, `docker run --gpus all` (or Compose's
`deploy.resources.reservations.devices` with `capabilities: [gpu]`) gives a
container the GPU through the NVIDIA container toolkit. On a Mac the same
request has no meaning to Docker Desktop, which refuses it.

With [the patches for Kiri](../../kiri/README.md), the same request on an
Apple-silicon Mac gives the container its end of the bridge at
`127.0.0.1:47700`, and only that container. A program that asks PyTorch
which GPU it has then runs on both, unchanged.

## What runs

[`examples/05-same-compose-nvidia/`](../../examples/05-same-compose-nvidia/):

- [`job.py`](../../examples/05-same-compose-nvidia/job.py) takes `cuda` if
  PyTorch has it, else `mps`, else `cpu`, and trains a small network for 300
  steps with autocast;
- [`compose.yaml`](../../examples/05-same-compose-nvidia/compose.yaml) is
  the service;
- [`compose.gpu.yaml`](../../examples/05-same-compose-nvidia/compose.gpu.yaml)
  adds the GPU request.

## Commands

On a Mac with Docker Desktop (run for this page): no GPU request, the bridge
is reached on a TCP port.

```sh
examples/05-same-compose-nvidia/run.sh
```

On a Mac with Kiri and the patches (run for this page): the GPU request,
and the bridge's server on Kiri's socket.

```sh
ENGINE=kiri scripts/setup-mac.sh                          # once: builds the images in Kiri
ENGINE=kiri examples/05-same-compose-nvidia/run.sh gpu
```

On an NVIDIA server (**not run for this page**: there was none), with any
CUDA build of PyTorch as the image:

```sh
cd examples/05-same-compose-nvidia
IMAGE=<a CUDA build of PyTorch> docker compose -f compose.yaml -f compose.gpu.yaml run --rm job
```

## What you see

On Docker Desktop:

```
torch 2.15.0, device mps
step 100: loss 0.0523
step 200: loss 0.0074
step 300: loss 0.0033
300 steps in 0.26 s on mps; accuracy 1.000
```

In Kiri, with the GPU request, the last line was `300 steps in 0.23 s on
mps; accuracy 1.000`.

With the GPU request, Docker Desktop on a Mac answers (run for this page):

```
Error response from daemon: could not select device driver "nvidia" with capabilities: [[gpu]]
```

Upstream Kiri refuses it too, with its own message. With the patches, on an
Apple-silicon Mac, the request is accepted (`--gpus all`, and Compose's
reservation with `driver: nvidia`).

## What stays the same, and what does not

| | Mac (Kiri with the patches) | NVIDIA server |
|---|---|---|
| The Compose files | the same | the same |
| The GPU request | `--gpus all` / `capabilities: [gpu]` | the same |
| The program | the same | the same |
| The image | `varco-torch`: CPU build of PyTorch for Linux arm64, with the bridge | a CUDA build of PyTorch, x86-64 or arm64 |
| The device PyTorch reports | `mps` | `cuda` |
| Where the work runs | a process on the Mac, outside the container | the GPU, from inside the container |
| What must also be running | the bridge's server on the Mac | nothing |
| float64, CUDA extensions (`torch-scatter` and the like) | no | yes |
| Operators | those MPS has | those CUDA has |
| Numbers | the Mac's | the card's |
| GPU memory | shared with the Mac's own | the card's |
| Isolation between containers | each program has its own process on the Mac ([security](../security.md#what-a-container-can-do-through-the-pytorch-bridge)) | by the driver |

So a program must be written for both devices: ask PyTorch what there is,
avoid float64 on the GPU, avoid CUDA-only packages or guard them. That is
what a program that already runs natively on a Mac and on a server does.

With MLX there is nothing to share on the NVIDIA side by this route: this
bridge is for the Mac. (Upstream MLX has its own CUDA backend, which was not
tried here.)

## Measured

On the released code the PyTorch batteries were run in Kiri in containers
started with `--gpus`, on a daemon built from a fresh clone with the
patches: 47 of 47 inference programs and 63 of 68 training programs were
identical to the Mac, 3 were within the declared tolerance and 2 fail on
both sides alike
([benchmarks](../benchmarks.md#measured-again-on-the-released-code)). On
that daemon a container started with `--gpus all` had
`KIRI_GPU_BRIDGE=127.0.0.1:47700` and a listener there, and a container
started without `--gpus` had neither: Kiri closed its connection at once,
and the bridge's error said that the container must be started with
`--gpus all`.

## Limits

- Kiri's channel is a patch against one commit of Kiri, not part of Kiri
  ([`kiri/README.md`](../../kiri/README.md)).
- Only one bridge can be up at a time in Kiri, because the channel leads to
  one socket.
- `--network host` cannot be combined with `--gpus` in Kiri, and no
  container can join the network of a container that has `--gpus`.
- Docker Desktop has no equivalent. There the GPU request must be left out
  and the bridge is reachable by every container.
- Images are per architecture and per PyTorch version. The Mac's image is
  `linux/arm64`.

Further measurements will be published as they are taken.
