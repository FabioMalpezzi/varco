# Roadmap

This is where the work stands at the first release: what was decided, the
paths that the measurements suggest, and what was decided against. None of
the paths below is planned or being built. Each one names the measurement it
rests on and what would confirm or drop it, so that whoever picks one up
knows where to start and when to stop.

The figures are those of [benchmarks.md](benchmarks.md); the findings about
the GPU itself are in [apple-gpu-limits.md](apple-gpu-limits.md).

## What was decided, and why

The architecture stays as it is: one MLX server shared by all containers,
with one executor thread, and for PyTorch one process per program. What is
left to gain without changing it is inside MLX itself.

The release keeps the checks and the limits of the
[security](security.md) page. The project's owner judges them sufficient
as long as whoever uses the channel knows them. No further security work
is planned.

| Not done | Why |
|---|---|
| A private worker per container behind one entry point, with reduced permissions for the Mac end | Drawn up, reviewed and set aside on 16 November 2026. It is recorded [below](#the-servers-one-executor) with what it would give. |
| A parallel executor in the MLX server (a thread of evaluation per connection) | It would add speed for several containers (129 tokens/s in total today, where two processes on the Mac reach 176) at the price of a server that no longer decides the order of work, and of evaluating from several threads on an MLX version that predates its thread-safety fixes. A first attempt was stopped before it changed anything that runs. It is recorded [below](#the-servers-one-executor) as what was learnt, not as a plan. |
| A load balancer and replicas of a model server on one Mac | On one GPU separate programs reach about 1.4 times one, while one server that batches reached 215 tokens/s with 4 requests. Replicas add no throughput on one Mac. |
| A larger MLX command buffer in the server (`MLX_MAX_MB_PER_BUFFER=2048`) | Through the bridge the gain in generation is 1 to 3%, within the noise, the model server shows no clear gain, and a larger buffer is a longer stretch of GPU work that another container cannot get in between. It remains an environment variable of the server for anyone who wants to try it. |
| Custom kernels on by default | A kernel from one container can read and change the arrays of another on the same server ([custom kernels](use-cases/custom-kernels.md#what-it-implies)). |
| A bridge for TensorFlow | It would be a third bridge; Keras 3 on its PyTorch backend is the candidate to try first. |
| `GLIBC_TUNABLES` for malloc in the MLX image | It made a loop that only builds BGE-M3's graph faster (0.60 to 0.48 ms) and a whole embedding 0.3 ms slower. |

## Paths that could be taken

### The server's one executor

The MLX server runs the work of every container one job at a time, so
containers on one server add up to what one does alone. There are two ways
out, and a third borrowed from elsewhere:

1. **Work in parallel inside one process, with the weights shared.** A
   connection would get its own thread of evaluation, and containers that
   load the same weight file would get the same arrays, read once (the
   server reads the files itself, so it can tell). This would lift both
   limits, speed and memory, which separate processes on a Mac cannot do.
2. **A private worker per container behind one entry point**, as the
   PyTorch bridge has: one process on the Mac for each container, started
   by a small server that all containers reach at one address, and run with
   reduced permissions (a macOS sandbox). It would separate containers from
   each other on MLX, give a container's memory back as soon as it leaves,
   and reach about 170 tokens per second in total with two containers,
   where the shared server gives 129. Every container would keep its own
   copy of the weights, and the GPU would be shared by the system, not
   evenly. The design was drawn up, reviewed and set aside on 16 November
   2026, and is not planned. A server per container started by hand gives
   the separation today
   ([security](security.md#containers-that-share-one-mlx-server)).
3. **A process per container that maps one copy of the weights**, if the
   Mac's server can map a weight file instead of reading it into its own
   memory (the idea is Strata's, see [related work](related-work.md)).

*Rests on:* two and three threads on one copy of the weights, on the Mac in
Python, gave the right text every time and 177 and 186 tokens per second, as
much as separate processes, in 18.2 GiB instead of two or three times that
([several threads in one process](benchmarks.md#several-threads-in-one-process-one-copy-of-the-weights)).
A server per container gave about 170–180 tokens per second in total, with
one container finishing in 6.9 s and the other in 10.6 s ([several containers at once](benchmarks.md#several-containers-at-once-and-where-it-stops-scaling)).

*Would confirm it:* the same thread test inside the server, in C++, with a
thread of evaluation per connection, each on its own stream, giving
identical text and a total near 176. If it holds there, path 1 is the one
to take, and otherwise path 2 is what remains.

The container end has its own limit, found by the same test. Several threads
of one program share one connection, and three or four of them together do
less than one alone (93 and 88 tokens per second, where one thread gives
128). A program with a thread per request would meet it. The likely remedy
goes with the one above: a connection, or a channel within it, per thread.

### More requests on one Mac: a batching server, not replicas

For throughput on one Mac the unit is one batching server per model
(mlx-lm's, oMLX, vLLM), in one container. A load balancer in front of
several containers is for something else:

- continuity: when one container is down or being replaced, the others
  answer;
- several models behind one address, a container each;
- several Macs, where each brings its own GPU, which is the one case where
  replicas add speed.

If one is built, it keeps a conversation on the server that has its cache. A
server finds a cache by the tokens of the prompt, so the balancer needs its
own key: a header from the client (or the request's `prompt_cache_key`), or,
with nothing asked of the client, a digest of the conversation's start. A
server that goes takes its caches with it: the conversation moves and is
read again once.

The rule can be stateless: a hash of the start of the prompt, plus the load
of each server, and no table of sessions. OpenAI's prompt-caching guide
(read on 16 November 2026) describes that rule for its own service: a
request is routed by the load of the machines and a hash of the first
tokens of the prompt, a `prompt_cache_key` influences the routing without
pinning a request to a machine, and a miss costs one reading of the prompt.
The same would hold for a balancer in front of several containers or
several Macs.

*Rests on:* mlx-lm's server with 4 requests at once, 216.0 tokens per second
in total on the Mac and 215.0 in the container; two and three separate
processes, 176 and 166. Sending a conversation to two servers in turn cost
one more full reading of the prompt (4.4 s on 6,400 tokens), no more
([keeping a conversation on one server](benchmarks.md#keeping-a-conversation-on-one-server)).

*Would confirm it:* nothing more is needed for one Mac. For several Macs,
a measurement with two machines.

### A small request alongside a long job: who decides the order

On the Mac itself, with two processes and no bridge, a small request slows 5
to 8 times while another process reads long prompts in a loop: a BGE-M3
sentence from 13.5 ms to 70–73 ms, a short answer's first token from 55 ms
to 390–470 ms (as reported, from a reference run of 15 November). That is
the shared GPU.

What can be driven:

- Metal has no public priority between command queues or processes for
  compute, and the unit the GPU switches on is the command buffer. MLX's
  maintainers closed a request for priorities as not feasible (issue 4716,
  October 2026): its buffers are committed one after the other, "there is
  simply no scheduling in MLX". The only mitigation its author had found is
  to pace the heavy stream, at a cost in throughput.
- The size of a command buffer is set by MLX: one is committed every 50
  operations or 50 MB on this chip (`mlx/backend/metal/device.cpp`), and
  `MLX_MAX_OPS_PER_BUFFER` and `MLX_MAX_MB_PER_BUFFER` change it. Smaller
  buffers should let a small request in sooner and cost the long job some
  speed. This was not tried for that purpose.
- A long prompt can be read in chunks (mlx-lm's prefill step), which others
  report keeps a long prefill from stalling running decodes.
- The bridge's server is in a place where no program on a Mac is: it sees
  the work of every container before the GPU does. It can hand a long step
  over in parts (it does, for urgent graphs) and choose whose part
  goes next, putting the small request's ahead of the long job's next part.
  Through the bridge a small request could then wait for one part instead of
  for what the GPU's own sharing gives, and do better than on the Mac.

Engines that drive Metal themselves (ds4, Kapitsa, Strata: see [related work](related-work.md#how-other-engines-share-the-apple-gpu)) do not use a
priority of the GPU's either. Each keeps one owner of the GPU and takes
turns in bounded chunks, batching what can be batched.

This pulls against the parallel executor above: threads of evaluation leave
the order to the GPU, as on the Mac (a higher total, with the Mac's
latencies), while one executor that chooses the order gives short waits and
no higher total. A class of priority per container, declared when it starts,
with the heavy stream paced while a latency-sensitive one is active, is the
candidate to have both.

*Rests on:* the 5 to 8 times above; the urgent reads, which already let a
small graph wait 0.25–0.3 ms while a long step runs.

*Would confirm it:* a mixed load through the bridge (a stream of short
embeddings alongside a loop of long prompts) with the server choosing the
order, compared with the same two programs on the Mac. A small request that
waits less than the 70–73 ms measured natively would show it.

### What MLX itself is doing

No roadmap of MLX's was found; this is from its repository's recent issues
and pull requests, read on 15 November 2026.

- **Several threads**: supported and still being made safe. Thread-local
  streams and builds for free-threaded Python were merged in August; a
  report on threads and streams was closed in October; two pull requests
  that fix data races in the scheduler and in the Metal allocator's memory
  counts are open (4844, 4885), and one on random numbers across threads
  (4857). A server here that evaluates from several threads runs on 0.32.4,
  before those fixes.
- **Order on the GPU**: nothing planned. The request for priorities between
  command queues was closed as not feasible (4716).
- **Command-buffer limits**: an issue measured 5–8% of decode speed lost to
  the 50 MB rule on a model with large weight matrices, which closes a
  buffer about once a layer (4521). The limits stay as they are by default,
  for the memory they save; they are documented environment variables, and a
  general way to change settings while running is being discussed (4562).
  Raising the limit was tried here and gave 8 to 10% on the Mac directly and
  1 to 3% through the bridge; it was not adopted
  ([a larger command buffer](benchmarks.md#a-larger-command-buffer-mlx_max_mb_per_buffer)).
- **Speed of single operations**: most open pull requests on Metal are about
  kernels (quantized products, sorting, fusing reductions in `mx.compile`).
  The bridge gets those for nothing: it runs the Mac's MLX.

*What follows:* moving the bridge to a newer MLX is the cheapest way to
gain thread safety and speed. It means regenerating the patch against the
new tag and running `mlx/tests/run-all.sh`.

### Why the container is ahead in generation

At protocols 7 to 9, generating with Qwen3.6-35B, the container was 4 to 5%
faster than the same program on the Mac, in every alternated pair. The
leading explanation is
that the work is split in two processes, the container building the next
step while the Mac computes; it is unverified
([benchmarks](benchmarks.md#why-the-container-can-be-faster-than-the-mac)).

*Would confirm it:* the same program on the Mac itself, talking to the
server through the bridge, with no virtual machine and no Linux. If that is
also faster than plain native, the gain belongs to the split, and is an
argument for the design on its own. A profile of the native run, comparing
the time per token spent in Python with the time spent on the GPU, would say
how large that gain can be.

### What is still slower, by size

1. **PyTorch, operations that read a value**: one round trip each, 150–200
   µs; 13% on a 26 ms step, and GPT-2 at 411 tokens per second with Docker
   Desktop and 475 in Kiri, where the Mac gave 579 and 581. The MLX
   bridge's urgent answers are the model to follow.
2. **PyTorch, large transfers**: 4.8 ms for a million values to the CPU,
   where the Mac takes 0.13 ms. Shared memory between the virtual machine
   and the Mac is the candidate.
3. **MLX, `top_logprobs`**: 88–90 tokens per second, where the Mac gives
   100–108. Its `argpartition` stays on the GPU.
4. **MLX, the first prompt of a process**, about 310 ms more per connection,
   in the first hand-overs to the GPU after the weights are loaded.
5. **MLX on Docker Desktop, reading a long prompt**: 3–6% slower there;
   level in Kiri.

*Would confirm each:* the measurement that found it, run again after a
change: `pytorch/tests/values_cost.py` for 1 and 2, `mlx/tests/serve.sh`
for 3 and 5, `mlx/tests/qwen.py` for 4.

### Solidity

What the first release leaves open:

- **Containers on one MLX server.** They are not separated from each other,
  and the way round is a server per container, started by hand
  ([security](security.md#containers-that-share-one-mlx-server)). The
  private worker per container [above](#the-servers-one-executor) would
  make that the default, and was set aside.
- **Unexplained**: the text of requests sent at once differs from the Mac's
  in some pairs; a checkpoint file's hash differs with identical weights.
- **One command for everything in Kiri**, so that a regression shows
  immediately. `ENGINE=kiri mlx/tests/run-all.sh` runs five of the ten
  default stages; the other five need a TCP port or a second server, and
  `mlx/tests/serve.sh` and `mlx/tests/bench.sh` have no mode for Kiri.
- **Images ready on a registry.** The first release builds every image on
  the user's Mac ([setup](setup.md#install)), which takes about four
  minutes and compiles MLX for Linux. With an automated build in place the
  images could be published on GitHub's container registry, so that
  installing is a `docker pull` and a tag says which state of the
  repository an image belongs to. The Mac ends would still be built on the
  Mac, against its own MLX and PyTorch.
- **Kiri's channel**: builds for Linux, Intel and Windows are not checked;
  the step down of the memory headroom after ten calm minutes was not
  observed; Kiri's other tests were not run with the patches.
- **Other machines**: every figure is from one M4 Max.

### Other programs to carry

- **oMLX**: its kernels go through `mx.fast.metal_kernel`; the server must
  first admit the three standard Metal headers, which the refusal of
  `#include` now rules out.
- **vLLM**: needs "native MLX extensions run on the Mac". The `vllm-metal`
  plugin defines eight MLX primitives in a native extension
  ([compatibility](compatibility.md#not-supported-and-open-measurements)).
  The container would keep only the shape of those operations, and the
  server on the Mac would load the plugin's real extension. vLLM and the
  plugin's Python already install and import on Linux; the next step is one
  operation, sampling, through the bridge. The same version of MLX and of
  the plugin is then needed on both ends.
- **Reaching the others through PyTorch or MLX**: Keras 3 on its PyTorch
  backend and `jax-mps` (JAX over MLX) are untried candidates.

*Would confirm each:* the program run in a container and compared with the
same program on the Mac, with the same result. Nothing here was tried beyond
the import check of vLLM.

### Proposing it upstream

This is the order that was thought through, and where it stands on 16
November 2026:

1. a working demonstration and this repository, with its measurements:
   published;
2. a proposal to MLX for the remote backend, which is a build option and one
   directory next to the existing backends: asked in MLX's
   [discussion 4900](https://github.com/ml-explore/mlx/discussions/4900);
3. Kiri's channel, proposed in Kiri's
   [issue 356](https://github.com/libracoder/kiri/issues/356), and the
   memory change, reported in its
   [issue 210](https://github.com/libracoder/kiri/issues/210);
4. a discussion, not a pull request, with vllm-metal on splitting what its
   operations are from how Metal runs them: not started.

The benefit that can be promised is containers. Speed is shown for MLX and
PyTorch programs here; it is not shown for vLLM. Any proposal upstream goes
by that project's contribution rules.
