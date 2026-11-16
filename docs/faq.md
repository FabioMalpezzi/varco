# Frequently asked questions

**Does the computation run on the real GPU?**
Yes. Nothing is emulated. Each operation a program issues in the container
is computed by the Mac's own MLX or PyTorch on the Apple GPU, through Metal,
as it is for a program started on the Mac, which is why almost every
program tested gives the Mac's result to the bit. What those frameworks
compute on the CPU on a Mac (a matrix inverse in MLX, for one) is computed
on a CPU here as well, and the MLX server answers some small reads of a
value on the Mac's CPU, with the same bytes
([architecture](architecture.md)).

**Is this GPU passthrough?**
No. No device is given to the virtual machine, and the GPU stays with
macOS. The framework inside the container is given a backend that describes
each array operation to a process on the Mac, which computes it there and
sends back the values the program reads.

**Do I have to change my program?**
No. An MLX program sees `mx.default_device()` as the GPU. A PyTorch program
sees `torch.backends.mps.is_available()` as `True` and uses `"mps"`. The
tests are ordinary programs run unchanged on both sides.

**Do I have to change my image?**
Yes. The container needs the bridge's build of the framework: the patched
MLX wheel, or stock PyTorch with the bridge's module added. Build your
image `FROM varco-mlx` or `FROM varco-torch:2.15.0`, or copy what their
Dockerfiles do ([`mlx/docker/`](../mlx/docker/),
[`pytorch/docker/`](../pytorch/docker/)).

**Is it as fast as running on the Mac?** For work that is mostly GPU time,
yes. Qwen3.6-35B-A3B-4bit with mlx-lm generated the same text in a
container as on the Mac, at 130 tokens per second in the first generation
of a run and 141 in the second, with Docker Desktop and with Kiri alike; on
the Mac the same program gave 125 to 135
([the runs](benchmarks.md#qwen36-35b-generation-on-the-released-code)).
Programs that read a value back after every short step are slower, because
each read is a round trip of 150–200 µs: BGE-M3 embeddings by 1 to 2% in
Kiri and 11 to 12% with Docker Desktop, a 26 ms PyTorch training step by
13% in Kiri. GPT-2 generation with PyTorch, which reads a value per token,
ran at 411 tokens per second with Docker Desktop and 475 in Kiri, where the
Mac gave 579 and 581 (one run a side). The tables are in
[benchmarks](benchmarks.md).

**Are the results the same as on the Mac?**
In almost every program tested, to the bit: the results are compared by the
SHA-256 of their bytes. The exceptions are listed: training loops that are
not reproducible on the Mac itself, a value that each side computes on its
own CPU, requests batched by a model server, and libraries that take
another path when custom kernels are off. See
[compatibility](compatibility.md).

**Which Macs?**
It was built and measured on one Apple M4 Max with 128 GB. It needs Apple
silicon. Nothing else is known.

**Does it work with Docker Desktop, OrbStack, Podman, Colima?**
It was run with Docker Desktop and with Kiri. The TCP path needs only that a
container can reach a port on the Mac (`host.docker.internal`), which other
engines also offer, but none was tried.

**What is Kiri and do I need it?**
[Kiri](https://github.com/libracoder/kiri) is a container engine for macOS.
You do not need it. With the patches in this repository it adds two things:
`--gpus all` gives a container, and only that container, its end of the
bridge; and its channel is faster than a TCP port for short steps (one
BGE-M3 sentence took 1.01 times the Mac's time through it, and 1.11 times
with Docker Desktop).

**Why two bridges instead of one for the GPU?**
Because the bridge is at the level of a framework's operations, not of the
GPU's API. That keeps the transport small (most operations are a few bytes
and need no reply) and the results the framework's own, at the cost of one
bridge per framework. A general path exists elsewhere, through Vulkan
([related work](related-work.md#podman-with-libkrun)).

**Can several containers use the GPU at once?**
Yes. What to expect is in
[use-cases/several-containers.md](use-cases/several-containers.md): on one
MLX server they share one stream of work evenly, and in total separate
programs on an Apple GPU reach about 1.4 times the throughput of one.

**How much memory does the container need?**
Little, because the arrays live on the Mac: 456 MiB at its peak for a model
server with a 19.5 GB model, 541–785 MB for a training that holds tens of
gigabytes on the GPU. The Mac needs what the program would need natively.

**Is it safe?**
A container that has the channel computes on the Mac's GPU and uses the
Mac's memory, as a program you start on the Mac does. The Mac end runs only
the operations on its list and refuses the rest. Give the channel to
containers you run yourself, or trust as you would a program on the Mac.
[Security](security.md) says what is checked and what is left to you.

**What about TensorFlow, JAX, llama.cpp, vLLM?**
There is no bridge for
them. llama.cpp has its own path with Podman ([related work](related-work.md#api-remoting-for-llamacpp)). vLLM's Metal plugin needs
more than this bridge carries
([compatibility](compatibility.md#not-supported-and-open-measurements)).

**Why is MLX pinned to 0.32.4?**
The container's MLX is a patch against that tag, and nodes travel in MLX's
own export format, which the Mac's MLX must read. Moving to another version
means regenerating the patch and running the tests
([development](development.md#changing-the-patches)).

**Can I rely on it?**
It is a prototype written by one person and measured on one machine. Its
known limits are in [compatibility](compatibility.md#known-faults) and
[security](security.md). Measure on your own machine.

**Why are some test prompts in Italian?**
They are the inputs the measurements were taken with. Changing them would
change the figures (the "93 tokens" of the generation tests is the answer to
one of them).

**Where does the name come from?**
*Varco* is Italian for a gap opened in a barrier: here, the one between a
Linux container and the Mac's GPU.
