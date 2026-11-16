# Related work

Other ways to get a Mac's GPU to programs in containers, and how engines
that own the Apple GPU share it between requests. **None of the projects on
this page was run for this repository.** What is said of them was read in
their own documents and sources in November 2026, and each entry names where.
Their figures are their own, taken on their own machines, and are not
comparable with those of [benchmarks.md](benchmarks.md).

## Where the bridge can sit

A program in a container can be given the Mac's GPU at three levels. They
differ in what stays inside the container.

| Level | What crosses to the Mac | What stays in the container | Examples |
|---|---|---|---|
| Model | "load this model", "complete this prompt", over HTTP | The client only. The model and the framework run on the Mac. | A model server on the host reached at `host.docker.internal`; Docker Model Runner; docker_mlx_cpp |
| Operation | Array operations of one framework | The program and the framework, with a backend that forwards | **varco** (MLX graphs, PyTorch operator calls); llama.cpp's API remoting (GGML calls) |
| GPU API | Graphics or compute API calls | Everything, including the GPU backend of the framework | Podman with libkrun (Vulkan through Venus and MoltenVK) |

The lower the level, the more programs run unchanged and the more is
carried per operation. The operation level keeps the program and its
libraries in the container (so a container image is still the unit that is
built, shipped and versioned) and needs one bridge per framework.

## Containers and the Apple GPU

### A model server on the host

The common arrangement: run Ollama, LM Studio, mlx-lm's server or vLLM on the
Mac and call it from containers over HTTP. It needs no bridge and gives
native speed. What it gives up is the container: the model code, its
dependencies and its versions live on the Mac, outside the image, and a
program that computes with arrays itself (training, embeddings inside a
pipeline, a custom model) is not served by it.

varco can run that same model server *inside* a container
([use case 2](use-cases/openai-server.md)).

### vllm-metal and Docker Model Runner

[vllm-metal](https://docs.vllm.ai/projects/vllm-metal) is a plugin that lets
vLLM run on Apple-silicon Macs with MLX as its compute backend. Docker
announced it as a backend of [Docker Model Runner](https://www.docker.com/blog/docker-model-runner-vllm-metal-macos/),
where it runs on the host, next to the containers and not in them.

Read in the plugin's source (0.33.0, not run): it requires a native
extension of about 10,500 lines of C++ and Metal shaders that defines eight
MLX primitives and runs them on Metal directly. It is pinned to MLX 0.32.4,
the version of this bridge. Those primitives do not go through
`mx.fast.metal_kernel`, so the bridge as it is cannot carry them; what it
would take is in the [roadmap](roadmap.md#other-programs-to-carry).

### docker_mlx_cpp

Announced in a
[discussion of the MLX repository](https://github.com/ml-explore/mlx/discussions/3349)
in April 2026 as a way to give Docker containers access to MLX. As its
author describes it: a daemon on the host with access to Metal, proxied to
containers over HTTP (container, gateway, daemon, GPU), with an
OpenAI-compatible API in front, 107 GPU operations, and support for mlx-lm,
mlx-vlm, mlx-audio and others; MIT licence; benchmarks on an M5 with 24 GB
given without their method.

By that description it sits at the model level, with a set of operations
exposed through a service, where varco replaces the GPU backend inside the
framework so that the unmodified MLX program runs in the container. The
project was not installed or read beyond that announcement.

### Podman with libkrun

Podman's virtual machine on macOS can use libkrun, which gives the Linux
guest a virtio-gpu device. Vulkan calls made in the container are serialised
by Mesa's Venus driver, carried over virtio-gpu, and executed on the host by
virglrenderer through MoltenVK, on Metal. Any program with a Vulkan backend
can use it. Red Hat's write-up of the work
([June 2025](https://developers.redhat.com/articles/2025/06/05/how-we-improved-ai-inference-macos-podman-containers))
reports llama.cpp at 75–80% of native speed on this path.

This is the GPU-API level: general, at the cost of a translation layer for
every call, and of use only to programs that have a Vulkan backend. MLX has
none, and PyTorch's Vulkan support is not a training backend.

### API remoting for llama.cpp

This is the closest relative of this work. Instead of forwarding Vulkan, it
forwards the calls of GGML, llama.cpp's tensor library: a frontend library
in the container, virtio-gpu as the transport, and on the host a backend
library loaded by virglrenderer that calls `ggml-metal`. Red Hat's
[September 2025 article](https://developers.redhat.com/articles/2025/09/18/reach-native-speed-macos-llamacpp-container-inference)
reports prompt processing at native speed and generation near native (about
95% at the lowest, on the smallest model) on an M4 Pro with 48 GB; the
integration into llama.cpp was proposed in its pull request 17072.

Its authors state its limits plainly: the backend runs in the hypervisor's
address space, so a crash there can take the hypervisor down; the backends
of all containers share that address space, so one container can crash
another or reach its GPU data; the design "isn't multitenant safe"; and the
preview allows one container at a time.

How varco differs:

| | API remoting for llama.cpp | varco |
|---|---|---|
| Framework | GGML (llama.cpp) | MLX and PyTorch |
| Transport | virtio-gpu, through a modified virglrenderer; needs libkrun | A socket: a TCP port, or Kiri's relay to a Unix socket. No change to the hypervisor. |
| Where the Mac end runs | Inside the hypervisor's process | In ordinary processes of the user: one server (MLX) or one process per program (PyTorch) |
| Several containers | One at a time in the preview | Several; MLX containers share one stream of work, PyTorch programs each have a process |
| Programs | llama.cpp | Ordinary MLX and PyTorch programs, including training |

The isolation concern is shared in part: an MLX server with custom kernels
enabled lets one container reach another's arrays
([security](security.md#custom-metal-kernels)).

## How other engines share the Apple GPU

The bridge's MLX server is, like a model server, one process through which
several clients reach one GPU. How engines that drive Metal themselves
handle that was read in their sources and documents on 15 November 2026;
none was run. It matters to the
[roadmap](roadmap.md#a-small-request-alongside-a-long-job-who-decides-the-order).

- **ds4**, an inference engine with its own Metal backend: one process, one
  Metal command queue, and its own scheduler. `--batched-session N` gives N
  slots with independent KV states; decoding is batched natively where the
  model allows and otherwise run row by row ("concurrency and scheduling
  fairness, not the aggregate speedup"). A long prefill yields to active
  decoders every 128 tokens, and runs 2,048 at a time when nobody is
  generating.
- **Kapitsa**: up to four conversation slots; requests sent at once each
  take a slot and "take turns on the GPU (a first-come-first-served turn
  lock)", cycle by cycle; the engine owns the scheduling.
- **Strata** ([repository](https://github.com/Niko1221/Strata), commit
  `13eedf5`): on a Mac `--parallel N` keeps N sequences side by side in one
  engine, each with its own state. Its own figures on an M5 Max are
  85 tokens per second in total with one request, 95–102 with two, 115 with
  three, 119–125 with four, each request slower as they grow (from 96–100
  alone to 32–40 at four). A long text is read in chunks of up to
  8,192 tokens. On PCs with several cards it runs replicas, whole engines
  behind one server: a request goes to the replica where it costs least, the
  cost being the prompt tokens that replica would have to read again (it
  tracks what each holds as hashes of 64-token blocks) plus 3,000 tokens for
  every request already running there; and the replicas can share one copy
  of the weights in RAM through one shared memory object (Linux and Windows
  only).

What they have in common:

- None uses a priority of the GPU's. There is none to use
  ([apple-gpu-limits](apple-gpu-limits.md#sharing-it-between-programs)).
- Each keeps one owner of the GPU. ds4 and Kapitsa take turns in bounded
  chunks, batching what can be batched.
- The ceiling agrees. Strata's own figures put several sequences on one
  Apple GPU at about 1.45 times one, close to the 1.4 measured here on
  another chip.

Two of Strata's ideas apply to the paths in the roadmap: its routing rule is
a ready design for a balancer that keeps a conversation where its cache is
until that server is too busy; and its shared memory object is a third way
for the MLX server's executor, a process per container with one copy of the
weights.
