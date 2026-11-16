# Use case 1: embeddings in a container

**What you get**

- The embedding code stays in its image and computes on the Mac's GPU, with
  the Mac's results. BGE-M3 (`bge-m3-mlx-fp16`) on 8 sentences gave
  embeddings identical to the Mac's, by the SHA-256 of their bytes (Docker
  Desktop and Kiri, protocol 11).
- Close to the Mac's speed: 9.68, 25.24 and 73.61 ms for batches of 1, 8
  and 32 sentences in a Kiri container, and 10.72, 27.74 and 80.73 ms in a
  Docker Desktop container (medians of 15 runs, protocol 11).
- The model loaded in 0.85 s in the container and in 0.82 s on the Mac
  (Kiri, protocol 8).

**Compared with**

- The same program installed on the Mac: 9.61, 24.80 and 72.37 ms for the
  three batches in the run with Kiri, and 9.64, 24.80 and 72.65 ms in the
  run with Docker Desktop.

**When it is not the right tool:** when every fraction of a millisecond of
a single sentence counts. The round trip weighs most there: with Docker
Desktop one sentence takes 10.72 ms where the Mac takes 9.64.

**Who it is for.** Anyone whose application runs in containers (a retrieval
pipeline, an indexing job, a service with a vector store next to it) and
needs sentence embeddings computed on the Mac's GPU instead of the
container's CPU, without moving the embedding code out of its image.

## What runs

[`examples/01-embeddings/embed.py`](../../examples/01-embeddings/embed.py) is
an ordinary program for [mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings):
it loads `mlx-community/bge-m3-mlx-fp16`, embeds a query and five passages,
prints the passages by similarity, and times batches of 1, 8 and 32
sentences. It knows nothing of the bridge.

## Commands

Once, for the whole repository: `scripts/setup-mac.sh mlx` and
`scripts/fetch-models.sh` ([setup](../setup.md)).

```sh
examples/01-embeddings/run.sh            # in a container, through the bridge
examples/01-embeddings/run.sh native     # the same program on the Mac
ENGINE=kiri examples/01-embeddings/run.sh   # the container in Kiri
```

`run.sh` starts the bridge's server with `--map /models=$MODELS`, runs the
service of [`compose.yaml`](../../examples/01-embeddings/compose.yaml), and
stops the server. The service is three lines of substance: the image
`varco-mlx-embeddings`, the models mounted read-only at `/models`, and the
program.

## What you see

```
device Device(gpu, 0); model loaded in 1.09 s
query: How do I run a Linux container on a Mac?
  0.8739  On macOS, containers run inside a lightweight Linux virtual machine.
  0.8186  A container shares the kernel of the Linux system it runs on.
  0.6841  MLX is an array framework for machine learning on Apple silicon.
  0.5652  Carbonara is made with guanciale, eggs, pecorino and black pepper.
  0.5578  The Giro d'Italia is a cycling stage race held in May.
 1 sentence(s): median 11.4 ms
 8 sentence(s): median 22.5 ms
32 sentence(s): median 53.4 ms
checksum of the embeddings: 118.969635
```

The device is the GPU, inside a Linux container. The scores and the checksum
are the same in the container and natively. The timings above are from one
run of the example on Docker Desktop and are an illustration. The measured
figures are below.

## Measured

With `mlx/tests/bge_m3.py` on 8 sentences, the embeddings are identical to
the Mac's with both engines: the same type, shape and SHA-256 of the bytes
(protocol 11, 16 November 2026).

Medians of 15 runs, in milliseconds, each container next to the Mac's
figures of the same run. Kiri's virtual machine had 8 CPUs and 16 GiB
([benchmarks](../benchmarks.md#bge-m3-timings-on-the-released-code)):

| Sentences | Mac | Docker Desktop container | Ratio | Mac | Kiri container | Ratio |
|---|---|---|---|---|---|---|
| 1 | 9.64 | 10.72 | 1.11 | 9.61 | 9.68 | 1.01 |
| 8 | 24.80 | 27.74 | 1.12 | 24.80 | 25.24 | 1.02 |
| 32 | 72.65 | 80.73 | 1.11 | 72.37 | 73.61 | 1.02 |

At protocol 8, in Kiri, loading took 0.85 s in the container and 0.82 s on
the Mac, and the first forward pass 0.05 s on both.

## Limits

- An embedding is a short computation that ends with a read: the bridge's
  cost per round trip shows most on single sentences. Batch where you can.
- The weights are read by the Mac only if the server was started with a
  `--map` or `--models` that covers them. Otherwise they are uploaded
  through the container, which is correct and slower to load, and they pass
  through the virtual machine's memory.
- A small request slows 5 to 8 times while another program runs a long job
  on the same GPU, on the Mac itself: a BGE-M3 sentence went from 13.5 ms to
  70–73 ms
  ([apple-gpu-limits](../apple-gpu-limits.md#sharing-it-between-programs)).
- The same test through the PyTorch bridge (BGE-M3 with transformers) is in
  [use case 3](pytorch-inference.md).

Further measurements will be published as they are taken.
