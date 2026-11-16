# Use case 3: PyTorch inference

**What you get**

- A PyTorch program that takes `"mps"` finds it in a Linux container and
  gives the Mac's results: 47 of 47 ordinary inference programs were
  identical to the Mac, by the SHA-256 of every tensor (PyTorch 2.15.0,
  Docker Desktop and Kiri, protocol 3).
- The Mac's speed where steps are heavy. The evaluation program of a
  private point-cloud segmentation project took 1.085 s per scan in a Kiri
  container, identical to the bit over 8 scans (a field report, protocol
  1).

**Compared with**

- The same program installed on the Mac: 1.088 s per scan for that
  evaluation. GPT-2 generation, which reads a value per token, ran at 579
  tokens per second on the Mac and at 411 in a Docker Desktop container,
  and at 581 and 475 in the run with Kiri (one run a side, protocol 3).
  `sum().item()` takes 171 µs on the Mac and 318 µs in a Docker Desktop
  container.

**When it is not the right tool:** for loops that read a value at every
step, such as token-by-token generation with a small model. Each read costs
one round trip, 150–200 µs more than on the Mac.

**Who it is for.** Anyone with a PyTorch program in a container, typically
a model from `transformers`, that takes `"mps"` when it is there and would
otherwise run on the virtual machine's CPU.

## What runs

[`examples/03-pytorch-inference/infer.py`](../../examples/03-pytorch-inference/infer.py)
picks the device PyTorch offers (`cuda`, then `mps`, then `cpu`), computes
sentence embeddings with `sentence-transformers/all-MiniLM-L6-v2`, and
generates 100 tokens with `openai-community/gpt2`, greedily. It imports
nothing from the bridge.

The image `varco-torch:2.15.0` is the stock CPU build of PyTorch for Linux
with the bridge added. In it, when the bridge's server can be reached,
`torch.backends.mps.is_available()` is `True` and a tensor moved to `"mps"`
lives on the Mac.

## Commands

Once: `scripts/setup-mac.sh pytorch` and `scripts/fetch-models.sh` ([setup](../setup.md)).

```sh
examples/03-pytorch-inference/run.sh            # in a container, through the bridge
examples/03-pytorch-inference/run.sh native     # the same program on the Mac
examples/03-pytorch-inference/run.sh cpu        # the same container with the bridge off
ENGINE=kiri examples/03-pytorch-inference/run.sh   # the container in Kiri
```

The Hugging Face cache is mounted read-only; the program finds the two
models there.

## What you see

```
torch 2.15.0, device mps
query: How do I run a Linux container on a Mac?
  0.7716  On macOS, containers run inside a lightweight Linux virtual machine.
  0.5756  A container shares the kernel of the Linux system it runs on.
  0.0526  Carbonara is made with guanciale, eggs, pecorino and black pepper.
generated 100 tokens in 0.26 s, 379 tokens per second
A Linux container is a container that is used to store and retrieve data from a Linux system. ...
```

The scores and the generated text are the same natively and in the
container. In single runs of the example the generation ran at 379 tokens
per second in a Docker Desktop container, at 462 in a Kiri container and at
561 on the Mac. With the bridge off (`run.sh cpu`) the container's CPU gave
52 tokens per second with Docker Desktop and 81 in Kiri.

## Measured

- Of 47 ordinary inference programs (`pytorch/tests/compat.py`), 47 were
  identical to the Mac on the released code, with Docker Desktop and
  through Kiri: every tensor has the same type, shape and SHA-256
  ([benchmarks](../benchmarks.md#measured-again-on-the-released-code)).
- GPT-2 generation, which reads a value per token, is timed inside that
  battery, in one run a side. On the released code it gave 411 tokens per
  second in a Docker Desktop container where the Mac gave 579, and 475 in a
  Kiri container where the Mac gave 581. At protocol 1 the same single run
  gave 398 with Docker Desktop and 448 in Kiri, and an alternated
  measurement on Docker Desktop, as reported, gave 509–520 in the container
  and 581 on the Mac.
- Per operation ([benchmarks](../benchmarks.md#cost-of-small-operations)),
  an operation that is not waited for costs about the same (`add` takes
  5.0 µs on the Mac and 7.2 µs in the container). One whose answer is in the
  values costs a round trip (`sum().item()` takes 171 µs on the Mac and
  318 µs in the container). A transfer of a million values to the CPU takes
  4.8 ms in the container and 0.13 ms on the Mac.
- Earlier measurements, taken from the notes and not read back from raw
  files, put BGE-M3 on 17 texts at 0.88 of native, a reranker on 16 pairs at
  0.91, and heavy loads at 0.995–1.05.
- In a field report, the evaluation program of a private point-cloud
  segmentation project took 1.088 s per scan on the Mac and 1.085 in the
  container over 8 scans, with results identical to the bit.

## Limits

- Reading values costs. A loop that calls `.item()`, compares, or indexes by
  a mask at every step pays 150–200 µs each time. Generation token by token
  is the common case. On heavy steps the cost does not show.
- Large transfers cost. Moving big tensors to and from the CPU is tens of
  times slower than on the Mac. Keep data on the device.
- Both ends need PyTorch 2.15.0. The server refuses a container with
  another version.
- An operator that is not on the Mac end's list is refused
  ([security](../security.md#what-a-container-can-do-through-the-pytorch-bridge)).
- What MPS lacks, the container lacks too: float64, and the operators with
  no MPS kernel, which fail with the Mac's own error
  ([compatibility](../compatibility.md#what-fails-on-the-mac-fails-in-the-container)).
- `torch.compile`, MPS streams other than the default, and custom Metal
  shaders were not tried.

Further measurements will be published as they are taken.
