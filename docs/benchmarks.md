# Benchmarks and tests

Every figure on this page says which code it was taken on. The tables were
measured between 12 and 15 November 2026 on the code before the review fixes
(MLX protocol 9, PyTorch protocol 1), except where a table or a row says
otherwise. That code is not in this repository, whose history begins with
release 0.1.0. [Measured again on the released code](#measured-again-on-the-released-code)
gives what was run on 16 November 2026 on the code as released (MLX protocol
11, PyTorch protocol 3, PyTorch 2.15.0), from a fresh clone, with Docker
Desktop and through Kiri. The [native server comparison](#native-server-priority-in-varco-020) covers the changes from 0.1.2 to 0.2.0.

These are measurements of a prototype on a single Mac, and were not repeated
on other hardware. Many rows are one run; the method below says how the
others were taken. Rows marked "as reported" come from the notes of the work
and were not read back from raw files. Two training projects used as test
beds are private: their rows are field reports, kept because they are the
heaviest real programs the bridges ran, and a reader cannot reproduce them.
For the bridge tests and benchmarks in this repository, see [Running them again](#running-them-again).

Related pages: [what the Apple GPU allows](apple-gpu-limits.md),
[what is and is not supported](compatibility.md), [roadmap](roadmap.md),
[glossary](glossary.md).

## What is being measured

A program in a Linux container computes on the Mac's Apple GPU: its operations
are intercepted in the container and executed by a process on the Mac. There
are two bridges, one per framework:

- **MLX bridge** ([`mlx/`](../mlx/)): MLX 0.32.4 for Linux with a remote
  backend (a patch against the upstream tag), and a C++ server on the Mac
  built on the stock MLX library. Protocol version 11 in the release; each
  table names the version it was taken with.
- **PyTorch bridge** ([`pytorch/`](../pytorch/)): an MPS backend for
  PyTorch on Linux whose one kernel sends every operation to the Mac, where a
  process per program runs them on real MPS. The release is for PyTorch
  2.15.0. Some of the earlier rows were taken with 2.12.1, and say so.

In Kiri a container started with `--gpus all` finds its bridge at
`127.0.0.1:47700` inside its own network; Kiri forwards those connections to
the Unix socket `~/.kiri/run/gpu.sock` on the Mac (a nine-file patch in
[`kiri/patches/`](../kiri/)).

## Setup

| | |
|---|---|
| Machine | MacBook Pro (Mac16,6) with an Apple M4 Max: 16 CPU cores (12 performance, 4 efficiency), 40 GPU cores, 128 GB of memory; macOS 26.6.2 (build 25G83), Metal 4 |
| Kiri | upstream at `baa2d41`, the commit the patches were against when the measurements were taken (they are now against `69dc0b7`), with the two patches of [`kiri/patches/`](../kiri/) (the GPU channel and the memory-headroom change below); VM of 2 CPUs and 8 GB unless stated, and of 8 CPUs and 16 GiB for the measurements on the released code |
| Docker Desktop | engine 29.2.1, 16 CPUs |
| MLX | 0.32.4 on both ends; mlx-lm 0.32.2; the patch and the server at the protocol version each table names |
| PyTorch | 2.15.0 (Python 3.13 in the container), the same version on both ends; the rows marked 2.12.1 were taken with that version (Python 3.11) |

## Method

- Each test is an ordinary program, unchanged, run on the Mac directly
  ("native") and in a container through the bridge.
- The GPU slows as it warms up, by more than the differences being measured,
  so timings are taken in alternated runs, in mirrored order (Mac,
  container, container, Mac), and compared pair by pair. The resolution is
  about 3%.
- Correctness is checked before speed. On the released code a result is
  identical when every array or tensor has the same type, the same shape
  and the same SHA-256 of its bytes on both sides, and every number a
  program read is the same bit for bit. Nothing is compared with a
  tolerance except the programs declared in the comparators, each with its
  reason. Where two runs on the Mac do not give the same bits, the
  difference between container and Mac is compared with the difference
  between two Mac runs.
- The earlier tables compared less: one checksum per program for the MLX
  coverage, and for the PyTorch batteries the one number each program
  prints, to six digits. Two different arrays can share such a number, so
  those rows say that the numbers agreed, not that the bytes did.
- Language-model speed is given in tokens per second.
- The test programs are in [`mlx/tests/`](../mlx/tests/) and
  [`pytorch/tests/`](../pytorch/tests/). Several prompts are in Italian.
  They are the inputs the figures were taken with (the 93-token answer below
  is the answer to one of them), so they are left as they are.
- The JSON results the MLX tables cite are in
  [`mlx/results/reference/`](../mlx/results/reference/). The logs, and every
  raw output of the PyTorch bridge, are not published, because they carry
  paths and names of the machine and of the private projects.

## MLX bridge

### Correctness

| Test | Result | Where | Raw output |
|---|---|---|---|
| Coverage, 57 small programs | 55 of 57 give the same number (one checksum per program); `nn_losses` and `cpu_linalg` differ | Kiri, protocol 8 | `coverage-{native,kiri}-k8.json` |
| BGE-M3 embeddings (`bge-m3-mlx-fp16`), 8 sentences | cosine 1.0000000, largest difference 0, on the elements as the result files hold them | Kiri, protocol 8 | `bge-{native,kiri}-k8.json` |
| Custom Metal kernels, 38 synthetic cases | 29 of 29 that run give the same bytes; 5 raise on both sides; 4 refused on purpose (`#include`, `#pragma`, source over 256 KB, kernel of an imported function) | Kiri, protocol 8, `--custom-kernels` | `kernels-{native,kiri}-k8.json` |
| Qwen3.6-35B-A3B-4bit, generated text | identical to the Mac's, 93 tokens, in 3 rounds of 3 | Kiri, protocol 8, `--custom-kernels` | log, not published |
| Same, without `--custom-kernels` | differs: 105 tokens, not 93 | Kiri, protocol 7 | log, not published |
| Weights read by the Mac: 68 values (safetensors, npz, npy, swapped byte order, empty, 80 MB) | 68 of 68 equal to native, both with the Mac reading and with the fallback | Docker Desktop, protocol 7, as reported | not published; the `load` stage runs it again |
| Refusals of the weight reader, 34 checks made by a test that speaks the protocol directly (file outside the directories, symlink out, `..`, missing, directory, pipe, wrong size or content, read past the end) | 34 of 34 | Docker Desktop, protocol 7, as reported | the `load` stage runs them again (`tests/load_refusals.py`) |
| Checksums of every weight of Qwen3.6-35B, 2090 arrays, 20.4 GB | all equal natively and in the container | Docker Desktop, as reported | not published (`tests/weights_sum.py`) |

The two coverage checksums that differ are those of `nn_losses`
(49.023107154846194 on the Mac, 49.023109300613406 in Kiri) and of
`cpu_linalg` (80.51065736781636 and 80.51066145259712). The files hold one
checksum per program, so they do not show whether the bytes of the arrays
were equal. The released code is compared by bytes, and there `nn_losses`
is identical and `cpu_linalg` is declared
([below](#collections-on-the-released-code)).

The generated text differed because mlx-lm uses custom Metal kernels
(`gated_delta.py`) when `mx.metal.is_available()` is true, and other
operations otherwise. This was verified by running the Mac with that
function forced to False. It then gives the container's 105-token text word
for word.

### Speed: BGE-M3 in Kiri

Medians of 15 runs, in milliseconds (`bge-{native,kiri}-k8.json`):

| Sentences | Mac | Kiri container | Ratio of the best times |
|---|---|---|---|
| 1 | 9.6 | 10.3 | 1.05 |
| 8 | 24.8 | 25.6 | 1.00 |
| 32 | 72.6 | 74.1 | 1.02 |

Loading took 0.85 s in the container and 0.82 s on the Mac, and the first
forward pass 0.05 s on both. On Docker Desktop the same test at protocol 8
gave ratios of 1.20, 1.10 and 1.07, so Kiri's channel is the faster of the
two here.

### Speed: Qwen3.6-35B-A3B-4bit with mlx-lm 0.32.2

`tests/qwen.py` loads the model and then generates twice from the same
26-token prompt. The table gives the tokens per second of the second
generation, in Kiri, in runs alternated with the Mac.

| Configuration | Generation | Prompt, second run | Load | Tokens |
|---|---|---|---|---|
| Mac native | 130–136 | 480–493 | 1.0 s | 93 |
| Kiri, protocol 8, `--custom-kernels` | 141 (two runs of six at 104–105: the stall below) | 510–519 | 1.1–1.3 s | 93 |
| Kiri, protocol 7 (no custom kernels) | 118–119 | 298–299 | 1.2 s | 105 |
| Mac native without its custom kernels | 108–110 | 276–280 | 1.0–1.1 s | 105 |
| Kiri, protocol 4, weights through the container | 96.7 (one run) | 230 | 31.6 s | 93 |

The logs of these runs are not published.

- With the weights read by the Mac (protocol 7), the load went from 31.6 s
  to 1.2 s and the VM from 40 GB to the default 8 GB. The server read 1757
  arrays, 19.5 GB, in 0.36–0.43 s, while 608–652 bytes went up from the
  container. The container's memory peaked at 566–592 MiB (cgroup
  `memory.peak`).
- With custom kernels (protocol 8), the server registered one kernel per
  connection and compiled nothing again.
- The first prompt of a process stays slower in the container, at 155–170
  tokens per second where the Mac gives 229–239. The cause was not found.
- Over a longer series in one process, 60 generations in Kiri and 20 on the
  Mac (`tests/token_gaps.py`), the median was 136.1 tokens per second in
  Kiri and 130.8 on the Mac, and the longest gap between two tokens was 17.3
  ms in Kiri and 14.8 ms on the Mac.

### A model server in a container: mlx-lm's own

`python -m mlx_lm server` (mlx-lm 0.32.2) runs unchanged in a container and
is compared with the same server on the Mac. The same requests are sent to
both over HTTP in mirrored order, in two blocks (`mlx/tests/serve.sh`,
`mlx/tests/serve_client.py`). The weights are read by the Mac.

At protocol 8 the container was slower. A whole request of 120 tokens took
1.40 s in the container and 1.15 s on the Mac, and streaming gave 66 tokens
per second where the Mac gave 95–101. The server reads one value per token
(`logprobs[token].item()`), and that small graph queued on the Mac behind
the next step, which had already been sent. It waited 6.8 ms where the Mac
itself waits 1.6 ms. The network's round trip was 0.15–0.3 ms of it.
`tests/qwen.py` reads no such value, which is why it was level.

From protocol 9 on, a small graph whose result the program is about to read
is marked urgent. It is answered ahead of the
queue, on the CPU, when it only selects, moves or compares elements, which
gives the same bytes as the GPU. A long step is handed to the GPU in parts
so that an urgent graph waits 0.25–0.3 ms. mlx-lm is not changed.

Qwen3.6-35B-A3B-4bit in Kiri at protocol 9, medians of four pairs
(`serve-Qwen3.6-35B-A3B-4bit-kiri-p9.json`):

| Case | Mac | Kiri container |
|---|---|---|
| One request, 120 tokens | 1.120 s | 0.884 s |
| Streaming | 101.1 tokens/s, first token 0.368 s | 125.6 tokens/s, first token 0.365 s |
| Long prompt (about 2,300 tokens) | first token 1.455 s, prompt at 1,569 tokens/s | first token 1.445 s, prompt at 1,586 tokens/s |
| Same long start again | 0.211 s, 2,262 of 2,289 tokens from the cache | 0.183 s, the same tokens from the cache |
| Next turn of the conversation | 0.212 s | 0.188 s |
| 2 requests at once | 152.9 tokens/s in total | 160.4 (two pairs at 186–187, two at 133–134) |
| 4 requests at once | 216.0 tokens/s in total | 215.0 |
| First request after 3 s idle | first token 0.437 s | 0.322 s |

Container memory at its peak: 456 MiB.

- The text is identical to the Mac's in every pair with one request at a
  time. With several requests at once each answer is the one to its own
  question, but the text differs from the Mac's in 2 of 4 pairs with 2
  requests and in 1 of 4 with 4, and those are also the slower pairs. This
  was not looked into. The server groups requests by when they arrive.
- On Docker Desktop, as reported, the ratio of container to Mac changed from
  protocol 8 to protocol 9 (`serve-*-{p8,p9}.json`): one request went from
  1.23 to 0.77, the streaming rate from 0.69 to 1.21, 2 at once from 0.78 to
  1.20, and 4 at once from 0.84 to 1.05. With a second model,
  Qwen3.5-0.8B-Base, streaming went from 0.70 to 1.22 and 4 at once from
  0.77 to 1.27.
- Two things are still slower in the container. One is a request with
  `top_logprobs`, at 88–90 tokens per second where the Mac gives 100–108.
  Its `argpartition` stays on the GPU. The other is the first prompt of a
  process.
- `/v1/models` fails in the container because of mlx-lm itself, which scans
  a Hugging Face cache directory that is not there.

### Several containers at once, and where it stops scaling

Each copy runs an ordinary mlx-lm loop on its own copy of Qwen3.6-35B, 8
generations of 93 tokens, and all copies start at the same moment
(`mlx/tests/multi.sh`, `mlx/tests/multi_gen.py`). A rate in the table is
the number of tokens a copy generated divided by the time its 8 generations
took, prompts included, and the rates of the copies are added up. Kiri had
6 CPUs. Each configuration was run
once.

| Copies | Mac, processes | Kiri, containers on one server | Docker Desktop, a server per container |
|---|---|---|---|
| 1 | 122 | 122 | – |
| 2 | 176 (8.1 and 8.9 s) | 129 (11.5 s each) | about 170 while both ran (6.9 and 10.6 s) |
| 3 | 166 | 128 (17.4 s each) | 179 |
| 4 | – | – | 140 on two servers, with the Mac swapping |

- The text is the same everywhere: one digest in every run, alone or with
  other copies running.
- The limit of the bridge is its one executor. The server has a thread per
  connection that only queues the work, and one thread that runs it, a job
  at a time (`mlx/server/mlx_remote_server.cpp`, the loop at the end of
  `main`). Two or three containers on one server share one stream of work
  evenly and add up to what one does alone. Each finished its 8 generations
  in 11.5 s at two and in 17.4 s at three, which is about 65 and 43 tokens
  per second each over the whole loop, the figure the table adds up. The
  rate mlx-lm reports for each generation is another measure and is higher:
  71.1 tokens per second in both containers at two, and 47.0 to 47.1 at
  three. It counts the time from a generation's first token to its last,
  and so leaves out the reading of the prompt and what the loop does
  between two generations. On Docker Desktop, 2 containers on one server
  gave the same 127.
- With a server per container the total is the Mac's, about 170–180, as with
  two or three processes on the Mac. The GPU is then shared by the system,
  and not evenly: one container finished in 6.9 s and the other in 10.6 s.
- The limit of the GPU for separate programs is about 1.4 times what one
  program gets. Two processes reach it and a third adds nothing. One server
  that batches does better on the same GPU, with 215 tokens per second from
  4 requests at once.
- Memory is the next limit. Every connection has its own copy of the weights
  on the Mac, which made 36.7 GB resident with two containers. Four copies
  (78 GB) took 8–9 s to load instead of 2 and left 3.4 GB of swap in use.
- When one of two running containers was killed half way (`docker kill`),
  the other went from 71 to 140 tokens per second within one generation,
  as mlx-lm reports the rate of each generation, with the right text, and
  the server stayed up. The server's resident
  memory was still 36.7 GB four seconds later, and was given back when the
  last connection closed.

### Several threads in one process, one copy of the weights

In `mlx/tests/threads_gen.py` one process loads Qwen3.6-35B once, and each
thread asks its own question six times through mlx-lm's `stream_generate`,
which runs on the calling thread's own stream
(`mx.new_thread_local_stream`). Every answer is first taken alone, on one
thread, and an answer given while other threads run must be the same text. A
rate is the number of tokens generated divided by the time the threads ran,
and the threads' rates are added up. Each configuration was run once.

| Threads | Mac | Kiri container |
|---|---|---|
| 1 | 124.5 | 128.4 |
| 2 | 177.1 | 130.8 |
| 3 | 186.0 | 92.9 |
| 4 | 155.4 | 88.3 (89.6 with 2 CPUs) |
| Answers that differ from the one given alone | 0 of 60 | 0 of 84 |
| Memory | 18.2 GiB on the GPU and 18.8 GiB for the process, with any number of threads | 0.4 GiB for the container's process |

- MLX computes correctly from several threads sharing the weights. On the
  Mac two and three threads reach what two and three processes do (176 and
  166), with one copy of the weights instead of two or three. This is the
  test that the first working hypothesis of the [roadmap](roadmap.md) asked
  for, done in Python. A server in C++ has no interpreter lock, which may be
  what holds four threads back here.
- Through the bridge the answers are right too, with the threads of one
  process on one connection, but the total does not grow with two threads
  and falls below a single thread's with three and four. The VM's CPUs are
  not the reason, since 6 instead of 2 changed nothing. Three containers do
  better than three threads (128 tokens per second and 93). The cause was
  not looked into. The threads share one connection, whose work the Mac
  takes in order.
- mlx-lm's own server is not affected: it generates on one thread and
  batches (215 tokens per second with 4 requests).

### Keeping a conversation on one server

Two mlx-lm servers ran in two Kiri containers and were sent one conversation
of six turns over a context of about 6,400 tokens, with 40 tokens generated
per turn (`mlx/tests/affinity.py`). It was run once.

| | First turn | Second turn | Later turns | Six turns |
|---|---|---|---|---|
| Every turn to the same server | 5.37 s, nothing cached | 0.42 s, 6,436 of 6,453 tokens cached | 0.42–0.43 s | 7.49 s |
| Turns sent to the two servers alternately | 4.37 s | 4.39 s, nothing cached | 0.43–0.57 s, 99–100% cached | 10.73 s |

Sent to the two servers alternately, the conversation costs one more full
reading of the prompt, which is the second server's first. After that each
server finds its own cache from two turns earlier. The server finds a cache
by the tokens of the prompt, not by an identifier, so affinity is a matter
of which server holds it.

### A larger command buffer: `MLX_MAX_MB_PER_BUFFER`

MLX commits a command buffer to the GPU every 50 operations or 50 MB of
distinct inputs on this chip. The test raises the second limit to 2,048 MB,
on the Mac directly and, for the container, on the bridge's server (the
container's own setting does not matter). It used Qwen3.6-35B on Docker
Desktop at protocol 9.

**The generation loop** (`threads_gen.py` with one thread, 8 generations;
order: default, 2,048, 2,048, default):

| | Mac, default | Mac, 2,048 MB | Container, default | Container, 2,048 MB |
|---|---|---|---|---|
| Tokens per second while generating | 131–135 (one run 115–135) | 144–146 | 137–141 | 140–143 |
| Over the whole run | 117.9 and 120.8 | 130.8 and 131.2 | 125.4 and 129.1 | 130.6 and 131.6 |
| Memory | 18.3 GiB at its peak on the GPU | 18.4 GiB | 18.5 GiB resident in the server | 18.5 GiB |

The texts are the same in every run. The larger buffer is worth 8 to 10% on
the Mac and 1 to 3% through the bridge. With it the Mac (144–146) is ahead
of the container (140–143), so part of what the container had over the Mac
at the default setting was the cost of the Mac's own buffer boundaries.

**The model server** (`tests/serve.sh`, stages single, stream and long, two
blocks; the default session first and the 2,048 one after it, so the two are
not alternated and the GPU was warmer for the second;
`serve-mb-{default,mb2048}.json`):

| | Mac, default | Mac, 2,048 MB | Container, default | Container, 2,048 MB |
|---|---|---|---|---|
| One request | 1.342 s | 1.231 s | 0.999 s | 0.892 s |
| Streaming | 95.2 tokens/s | 95.7 | 112.5 | 114.2 |
| Long prompt, read at | 1,225 tokens/s | 1,149 | 1,233 | 1,118 |

One request is 8 to 11% faster on both sides, streaming is the same, and the
reading of a long prompt is 6 to 9% slower. Without alternation these three
differences cannot be told from the GPU warming up.

The larger buffer was not adopted, and the server keeps MLX's default.
Through the bridge the gain in generation is within the noise, the model
server shows no clear gain, and a larger buffer is a longer stretch of GPU
work that another container cannot get in between. It remains an environment
variable of the server for anyone who wants to try it.

### Why the container can be faster than the Mac

At protocols 7 to 9, generating with Qwen3.6-35B, the container was 4 to 5%
faster than the same program on the Mac. The difference showed in every
alternated pair on both engines (141 tokens per second in Kiri and 139–141
on Docker Desktop, where the Mac gave 130–136), over a long series (a
median of 136.1 where the Mac gave 130.8), and on the path without custom
kernels as well (118–119 where the Mac gave 108–110). The generated text
was identical on the two sides. On the released code the container's
second generation of a run gave 141 tokens per second on both engines and
its first 130, where the Mac gave 129–135, and 138 in one later run
([below](#measured-again-on-the-released-code)).

It showed only there. Where a step is long GPU work the two sides are level
(0.995 and 0.997 for the segmentation project), and where steps are short
and each waits for a value the container is slower (BGE-M3 1.00–1.05, the
point-network pilot 1.13).

None of the explanations below has been tested; they are ordered by how well
they fit what was measured.

1. **Two processes instead of one.** Natively one Python process builds the
   next token's graph (model code, sampler, detokenizer) and drives the GPU.
   Through the bridge the container builds the next graph while the server
   on the Mac is still computing the current one, so the Python time of a
   token overlaps the GPU time instead of adding to it. This fits the gain
   being the same with and without custom kernels, and being absent where
   GPU time dominates a step.
2. **The process that computes has no Python in it.** The server is C++: it
   rebuilds each graph from a compact description and evaluates it, with no
   interpreter, no interpreter lock and none of Python's allocations between
   two evaluations.
3. **Memory settings differ.** mlx-lm sets a limit on wired memory when it
   runs natively; the server does not. This was ruled out as the cause of
   the slower first prompt, not for the steady state.
4. **Neither the clock nor the order.** Each side times itself, but the
   whole-run times agree with the rates (0.7 s and 0.7–0.8 s for the second
   run), and the alternation rules out heat.

What would tell them apart is the same program on the Mac itself, talking to
the server through the bridge, with no VM and no Linux. If that is also
faster than plain native, the gain comes from splitting the work between two
processes (1 and 2) and not from anything in the container. A profile of the
native run, comparing the time per token spent in Python with the time spent
on the GPU, would say how large that gain can be.

### Earlier protocol versions

These figures are from the notes taken during the work and were not read
back from raw files.

| Protocol | Test | Container and Mac |
|---|---|---|
| 4 | BGE-M3 in Kiri, 1 / 8 / 32 sentences | 10.7 / 25.9–26.3 / 75.1–75.6 ms in the container, 9.9 / 25.1 / 73.3 on the Mac |
| 6 | synthetic generation loop, Docker Desktop | 98–99% of native |
| 7 | synthetic generation loop, Docker Desktop | 977–993 tokens per second in the container, 988–994 on the Mac |

## A stall in Kiri, and its fix

The stall was found while measuring Qwen in Kiri, where one generation in
six ran at 104 tokens per second instead of 140.

- **What it was.** It was not a slow run but a single gap of 213–226 ms
  between two tokens, every 4 seconds (`mlx/tests/token_gaps.py`). On Docker
  Desktop, 36 generations showed none.
- **Cause.** Kiri's balloon shrank the VM by 256 MiB a second down to the
  guest's counted use plus 256 MiB (599 MiB here), where Linux ran short and
  stalled until its pressure event got it 1 GiB back. Then the shrinking
  began again. In the log of the balloon's target, which is not published,
  every stall falls on the 599 MiB reading.
- **Evidence.** With `KIRI_NO_BALLOON=1` there were 0 stalls in 40
  generations, and with the balloon 5 to 7 in 40.
- **Fix**
  ([`kiri/patches/0002-memory-headroom-after-shortage.patch`](../kiri/patches/0002-memory-headroom-after-shortage.patch)):
  after a shortage the balloon keeps more headroom (what was not enough,
  plus the usual 256 MiB, at most 2 GiB), and gives it back a step every ten
  calm minutes.
- **After.** There was one stall in the first 40 generations (the one it
  learns from; the target then settles at 857 MiB), then 0 in 60. A later
  45-second probe under a PyTorch load showed no gap over 5 ms.
- **Not verified.** The step down after ten minutes, Kiri's other tests
  (only the new unit test was run), and whether the released Kiri 0.2.0 has
  the same stall.

## PyTorch bridge

### Compatibility batteries

`compat.py` (inference programs) and `compat_train.py` (training loops,
optimizers, autocast and `GradScaler`, `DataLoader` workers, saving and
resuming, Lightning, random numbers, gradients, threads and processes, memory,
errors). A program counts when it works on the Mac.

The table is of the batteries as they were at PyTorch protocol 1, with 40
and 64 programs, compared by the number each program prints. "Identical"
means that the printed numbers were the same, and "equal to 3 digits" that
they agreed to one part in a thousand; the elements of the tensors were not
compared. The batteries of the released code, with 47 and 68 programs
compared by their bytes, are
[below](#measured-again-on-the-released-code).

| Engine | PyTorch | Battery | Work on the Mac | Identical | Equal to 3 digits | Different or failing | Fail on both alike |
|---|---|---|---|---|---|---|---|
| Kiri | 2.15.0 | compat | 40 | 40 | 0 | 0 | 0 |
| Kiri | 2.15.0 | compat_train | 64 | 62 | 2 | 0 | 2 |
| Docker Desktop | 2.15.0 | compat | 40 | 40 | 0 | 0 | 0 |
| Docker Desktop | 2.15.0 | compat_train | 63 | 61 | 2 | 0 | 2 |
| Docker Desktop | 2.12.1 | compat | 39 | 39 | 0 | 0 | 1 |
| Docker Desktop | 2.12.1 | compat_train | 61 | 58 | 3 | 0 | 3 |

The raw outputs are not published. The Kiri rows were run again after the
last change to the bridge at protocol 1.

- The programs "equal to 3 digits" are training loops of a transformer. When
  two Mac runs were compared (2.12.1), the same three were the only ones not
  reproducible to the bit.
- The programs that "fail on both alike" use operations PyTorch has no MPS
  kernel for (`rrelu`, attention with dropout; on 2.12.1 also
  `grid_sample`'s gradient and a `cdist` case). The container raises the
  Mac's `NotImplementedError`.
- Of 17 kinds of error, all 17 are raised as on the Mac, and running out of
  GPU memory is a `RuntimeError` on both.
- In a Kiri container started without `--gpus`,
  `torch.backends.mps.is_available()` is False.

### Field report: a point-cloud segmentation project

The first test bed is a point-cloud segmentation project that trains with
PyTorch Lightning and 16-bit autocast, on the Mac with PyTorch MPS, at batch 2.
It is private, so a reader cannot reproduce these rows. Its programs run
in the container unchanged, with the project mounted read-only.

| Program | Engine, PyTorch | Mac | Container | Ratio | Same numbers | Container memory |
|---|---|---|---|---|---|---|
| training, 20 steps | Kiri, 2.15.0 | 2.477 s/step | 2.463 | 0.995 | first step identical in 8 runs of 8; later steps differ as two Mac runs do | 732–785 MB |
| evaluation, 8 scans | Kiri, 2.15.0 | 1.088 s/scan | 1.085 | 0.997 | identical to the bit, every scan, 8 runs | 541–557 MB |
| training, 20 steps | Docker Desktop, 2.12.1 | 2.932 s/step | 2.818 | 0.961 | as above | 663–674 MB |
| evaluation, 8 scans | Docker Desktop, 2.12.1 | 1.262 s/scan | 1.251 | 0.991 | identical to the bit | 507–509 MB |
| two-stage prediction, one raw scan | Docker Desktop, 2.12.1 | 4.290 s | 4.438 | 1.035 | identical given the same first stage; else 1.78% of faces differ | 640–658 MB |
| training of the second stage, 40 steps | Docker Desktop, 2.12.1 | 0.166 s/step | 0.249 | 1.498 | weights identical to the bit after each step | 576–582 MB |
| resume from a checkpoint | Docker Desktop, 2.12.1 | – | – | – | a Mac checkpoint resumed in the container and the reverse: first loss identical | – |

Each side was run four times, in mirrored order. The Docker Desktop rows
beyond the first two are as reported. The raw outputs are not published.

- Training is not reproducible on the Mac itself. The gradient of rows taken
  by a repeated index (`x[idx]`) is summed by Metal in varying order: 18
  gradient tensors of 82 change between two computations of the same step.
  At step 19 the loss differs by 0.026 between two Mac runs and by 0.023
  between Mac and Kiri container.
- Pair by pair, the training ratios in Kiri were 1.093, 0.995, 0.979 and
  0.994, so heat weighs more than the side.
- The 1.78% comes from `pyfqmr`, which simplifies the mesh on the CPU and
  gives a different mesh on Linux and on macOS, the same each time on each.
- Stage two is 50% slower because of numpy on the container's CPU, which
  takes 4.7 s over 40 steps where the Mac takes 1.9 s. Its PyTorch part
  takes 2.5 s in the container and 2.4 s on the Mac.
- One further training program of the project was not run, because it binds
  signatures to the paths of its own sources and writes to its cache.

### Field report: a point-network pilot

The second test bed is a point-network pilot with a fixed seed. It was run
unchanged with `--device mps` on PyTorch 2.15.0, with the project and the
dataset mounted read-only. It is private too, so a reader cannot reproduce
these rows.

| Run | Engine | Mac | Container | Ratio | Same numbers | Container memory |
|---|---|---|---|---|---|---|
| 40 steps, 4 runs a side | Kiri | 0.026 s/step | 0.030 | 1.136 | loss identical to the bit at every step, weights identical, in every run | 1612–1616 MB (Mac process 1657–1672 MB) |
| 320 steps, 2 runs a side | Kiri | 0.026 s/step | 0.029–0.030 | 1.125 | as above | 1613 MB |
| 320 steps, 2 container runs | Docker Desktop | 0.026 s/step | 0.044–0.046 | about 1.7 | first and last loss equal | 1612–1626 MB |

The 40-step row was run again after the last change to the bridge. The raw
outputs are not published.

- Steps this short (26 ms) show the bridge's fixed cost, which is 13% in
  Kiri. On the segmentation project's 2.5-second steps it does not show.
- The GPU memory left in use at the end is the Mac's (1,617,408 bytes) after
  a fix to the bridge. Before the fix the container left 2,158,336.
- The checkpoint file's SHA-256 differs between Mac and container although
  the weights are identical. The cause was not verified. It matters to this
  project, which records checkpoint hashes.
- The project's full network was not hosted. It needs CUDA extensions (its
  own CUDA sources and `torch-scatter`) that do not run on the Mac natively
  either.
- The project's other programs were not tried (a verifier, two benchmarks,
  two probes, its tests).

### PyTorch 2.12.1 and 2.15.0 compared on the Mac

The segmentation project pins 2.12.1 because 2.13.0 took twice as long per
step. The table gives its own training program over 25 steps, in three
alternated pairs with the other packages unchanged (timed with
`pytorch/tests/step_timer.py`; a field report, as above).

| Pair | 2.12.1 | 2.15.0 |
|---|---|---|
| 1 | 2.015 s/step | 1.907 |
| 2 | 2.413 | 1.918 |
| 3 | 2.827 | 2.542 |
| GPU memory held | 43.7–44.7 GiB | 21.4–22.3 GiB |

2.15.0 is faster in every pair, by 5 to 20%, in half the memory. The quality of
the trained model was not compared.

### Cost of small operations

Time per call on the Mac and in a Docker Desktop container, PyTorch 2.15.0
(`pytorch/tests/values_cost.py`):

| Operation | Mac | Container |
|---|---|---|
| `add`, not waited for | 5.0 µs | 7.2 µs |
| `matmul` 64×64, not waited for | 13.3 µs | 12.6 µs |
| `sum().item()` | 171 µs | 318 µs |
| `nonzero` | 136 µs | 317 µs |
| `x[x > 1]` | 143 µs | 372 µs |
| `.cpu()` of 1M values | 0.13 ms | 4.8 ms |
| `.to("mps")` of 1M values | 0.22 ms | 2.0 ms |

An operation whose answer is in the values costs the container one round
trip, 150–200 µs more. GPT-2 generation, which reads a value per token, was
measured within the batteries at protocol 1, in one run each and not
alternated. It gave 448 tokens per second in a Kiri container and 566 on
the Mac (79%). On Docker Desktop it gave 398 in the container and 587 on
the Mac with 2.15.0, and 367 and 459 with 2.12.1. An alternated measurement
on Docker Desktop, as reported, gave 509–520 in the container and 581 on
the Mac (89%). On the released code, again within the batteries and in one
run each, it gave 411 in a Docker Desktop container where the Mac gave 579,
and 475 in a Kiri container where the Mac gave 581
([below](#measured-again-on-the-released-code)).

### Earlier measurements

These figures are from the notes taken during the work (PyTorch 2.15.0,
Kiri's channel) and were not read back from raw files: light operations at
0.91 of native, BGE-M3 on 17 texts at 0.88, a reranker on 16 pairs at 0.91,
heavy loads at 0.995–1.05, and GPT-2 at about 500 tokens per second where
the Mac gave 555.

## Kiri and Docker Desktop compared

| | Kiri | Docker Desktop |
|---|---|---|
| BGE-M3 through the MLX bridge, ratio to native (1 / 8 / 32 sentences), released code | 1.01 / 1.02 / 1.02 | 1.11 / 1.12 / 1.11 |
| The same at protocol 8 | 1.05 / 1.00 / 1.02 | 1.20 / 1.10 / 1.07 |
| Qwen3.6-35B, second generation of a run, tokens per second, released code | 141–142 | 141 |
| GPT-2 within `compat`, tokens per second, one run, released code | 475 | 411 |
| The point-network pilot, seconds per step, protocol 1 | 0.029–0.030 | 0.044–0.046 |
| Starting a container (from the notes, not read back) | 14 ms | 266 ms |

## Measured again on the released code

On 16 November 2026 the collections and two timed models were run again
from a fresh clone of the code as released: MLX protocol 11, PyTorch
protocol 3, PyTorch 2.15.0, on the same machine, with Docker Desktop and
through Kiri. Kiri's virtual machine had 8 CPUs and 16 GiB. One measurement
ran at a time and nothing else used the GPU. Every comparison is by the
type, the shape and the SHA-256 of the bytes.

Changes followed these first runs, before release 0.1.0. They touched where
the MLX container end looks for the server, the wording of some errors, the
tests and the installation scripts, and no wire format. `scripts/test-all.sh`
and the stage `upstream` were then run once more, the same day, from a fresh
clone with Docker Desktop. They gave the counts of the tables below, and
their timings appear there as those of the last run.

### Collections on the released code

| What was run | Docker Desktop | Kiri |
|---|---|---|
| `mlx/tests/run-all.sh` | Exit status 0 with the ten default stages | Exit status 0 with the five stages that run there, in 22 s |
| Stage `coverage`, 58 programs | 57 of 57 identical. `cpu_linalg` is declared: its matrix inverse is computed by Accelerate on the Mac's CPU and by OpenBLAS in the container, and the GPU gives the same bytes when the inverses are swapped | The same |
| Stage `edge`, 12 cases | 6 of 6 the same result; 3 raise on both sides; 3 the container refuses | The same |
| Stage `bge`, BGE-M3 on 8 sentences | Identical by bytes, 8,192 elements in each of the two embeddings | Identical by bytes, with the same two digests |
| Stage `gen`, a synthetic generation loop | 8 of 8 results the same | The same |
| The checks of `pytorch/compat.sh` | 168 of 168 refusals at the Mac end; 44 of 44 things a program does at the container end | The same |
| Battery `compat`, 47 programs | 47 identical | 47 identical |
| Battery `compat_train`, 68 programs | 63 identical, 3 within the declared tolerance and 2 failing on both sides alike in one run; 62, 4 and 2 in another; 63, 3 and 2 in the last run | 63, 3 and 2 in its one run |
| GPT-2 generation within `compat`, one run a side | 410.8 tokens/s in the container, 579.4 on the Mac; 398.4 and 581.2 in the last run | 475.1 in the container, 581.1 on the Mac |
| Duration | `scripts/test-all.sh`, both bridges: 4 min 42 s and 4 min 34 s, and 4 min 49 s in the last run | `pytorch/compat.sh`: 1 min 53 s |

The stages `load`, `kernels`, `urgent`, `graphs` and `disconnects` speak to
the server on a TCP port or start a second server, and with Kiri one server
listens on the socket alone. `run-all.sh` refuses them under `ENGINE=kiri`,
so their figures are from Docker Desktop:

| What was run, Docker Desktop | Outcome |
|---|---|
| Stage `load` | 68 of 68 loaded arrays the same, with the Mac reading the files and again with everything uploaded; every refusal of the weight reader as expected |
| Stage `kernels`, 38 cases | 29 of 29 that run give the same bytes; 5 raise the same kind of error on both sides; 4 refused on purpose |
| Stage `urgent` | 22 of 22 reads give the same value |
| Stages `graphs` and `disconnects` | All as expected, with the one limit `disconnects` declares: while a container does not read an answer the server is writing, no other container is served |
| `mlx/tests/run-all.sh upstream`, MLX's own Python tests | 926 tests on each side; 1 test method fails on the Mac and 19 in the container, and 30 are skipped in the container only, all declared. Against a server started with `--custom-kernels` none is skipped on one side and 23 fail in the container, all declared. Five runs of six passed, in 1 min 4 s to 1 min 17 s; the sixth failed on MLX's `test_divmod`, which fails on the Mac too and which the stage now reports without counting it ([known faults](compatibility.md#known-faults)) |
| `mlx/tests/run-all.sh bge-cpu`, BGE-M3 on the container's CPU | Cosine 1.0000000 with the Mac's GPU, largest differences 1.57e-05 and 1.97e-05; 36 to 54 times slower than the Mac's GPU |
| `pytorch/lifetime.sh` | 4,096 MB allocated and gone within 1 to 4 s after each of five endings (clean, exception, exit, hang, busy) |

The three training programs that are close in every run are two training
loops of a transformer and `loop_autocast_gradscaler_fused_adam`. A fourth,
`gradient_grid_sample`, is declared close and came out identical in some
runs, which is why the count of identical programs is 63 or 62. Neither
`loop_autocast_gradscaler_fused_adam` nor `gradient_grid_sample` is
repeatable on the Mac itself: the gradient of an embedding whose indices
repeat differed in 10 native runs of 10, the gradient of `grid_sample` in 1
native run of 6, and PyTorch marks both operators as non-deterministic on
MPS. The two programs that fail on both sides use operators with no MPS
kernel (`rrelu`, attention with dropout).

Three differences that the comparison by bytes showed in the code before the
fixes were closed in the bridges: MLX's fused cross entropy and its gradient
now go to the Mac as one node (`nn_losses` in the coverage stage), an LSTM
over a sequence goes to the Mac whole (`layers_recurrent` in `compat`, where
557 elements of 672 had other last bits), and an error is raised in the
container as the kind it was on the Mac.

### BGE-M3 timings on the released code

Medians of 15 runs, in milliseconds, from the stage `bge`. Each engine's
container is compared with the Mac's figures of the same run of the stage.

| Sentences | Mac | Docker Desktop container | Ratio | Mac | Kiri container | Ratio |
|---|---|---|---|---|---|---|
| 1 | 9.64 | 10.72 | 1.11 | 9.61 | 9.68 | 1.01 |
| 8 | 24.80 | 27.74 | 1.12 | 24.80 | 25.24 | 1.02 |
| 32 | 72.65 | 80.73 | 1.11 | 72.37 | 73.61 | 1.02 |

A second run with Docker Desktop gave 10.92, 27.08 and 78.22 ms in the
container and 9.70, 25.07 and 72.51 ms on the Mac. The last run gave 10.80,
27.95 and 79.09 ms in the container and 9.70, 25.03 and 72.29 ms on the Mac.

### Qwen3.6-35B generation on the released code

`SERVER_ARGS=--custom-kernels mlx/tests/lm.sh mlx-community/Qwen3.6-35B-A3B-4bit native container`,
then the same with `container native`, and both again with `ENGINE=kiri` in
front: the order is Mac, container, container, Mac. Each run loads the
model once and generates twice, 93 tokens each time. The text is identical
in every run. The figures are tokens per second.

| Engine | Run | Side | First generation | Second generation |
|---|---|---|---|---|
| Docker Desktop | 1 | Mac | 132.1 | 133.1 |
| Docker Desktop | 2 | container | 129.9 | 141.1 |
| Docker Desktop | 3 | container | 129.5 | 140.8 |
| Docker Desktop | 4 | Mac | 129.0 | 124.9 |
| Docker Desktop | 4, repeated once after a wait of 150 s | Mac | 132.5 | 135.1 |
| Kiri | 1 | Mac | 133.4 | 132.5 |
| Kiri | 2 | container | 130.1 | 141.2 |
| Kiri | 3 | container | 130.6 | 141.9 |
| Kiri | 4 | Mac | 131.3 | 133.1 |

With both engines the container's second generation is the faster one, at
141, and its first is at the Mac's level or just under, at 130. The
container took 328 MiB at its peak with Docker Desktop, and 406 and 330 MiB
in Kiri (`docker stats`). The Mac read the weights itself: with Docker
Desktop, 1,757 arrays, 19.5 GB, in 461 ms, while 608 bytes went up from the
container.

After the ten commits the first command was run once more with Docker
Desktop, in the order Mac, container. The container gave 131.0 and 141.4
tokens per second and the Mac 138.5 and 137.8, with identical text. The
Mac's two figures are above those of its four earlier runs, and that run
was not repeated.

## Native server priority in Varco 0.2.0

The 0.1.2 Mac environment provided the standard MLX-LM server. Version 0.2.0
adds an opt-in priority server on the same MLX 0.32.4 and MLX-LM 0.32.2.
The baseline below was the standard server in the release test environment;
it uses the same dependency versions as 0.1.2. The comparison measures the
native HTTP service on a Mac. The Docker and Kiri bridge protocols stayed
the same across these two releases.

On 20 November 2026, [the HTTP burst test](../mlx/tests/priority_server_http.py)
sent three short requests during a long prompt on one M4 Max with 128 GB
and Qwen3.6-35B-A3B-4bit. Each value is the median of four paired rounds
with prefix caching disabled and alternating server order.

| Client measurement | Standard server available in 0.1.2 | Priority server in 0.2.0 | Time saved |
|---|---:|---:|---:|
| Short client 1, first visible text | 21.75 s | 0.66 s | 21.10 s |
| Short client 2, first visible text | 21.70 s | 0.61 s | 21.10 s |
| Short client 3, first visible text | 21.66 s | 0.56 s | 21.10 s |
| Long request, completion | 23.24 s | 9.15 s | 14.10 s |

Savings are differences of the unrounded medians. Each client received the
word it requested. The [four individual rounds and reproduction instructions](native-priority.md#release-measurements)
give the scope of this comparison.

### Why the scheduler uses one stream and 512-token blocks

A short request can spend most of its time waiting for a long prompt's
current block to finish. In four instrumented prototype rounds, the first
short client waited a median 297 ms for that block. Once the short batch
got its turn, its first GPU compute pass started within 3.4 ms (median).
Reducing the wait between turns offered more room for improvement than
optimizing the CPU instructions used to submit work.

Three prototype comparisons on the same Mac guided the design. Their
timings are from separate experiments and should be compared within each
experiment:

- **One model, one GPU stream and two queues.** With 512-token long blocks,
  a single stream with explicit priority gave the three clients first text
  in 0.35–0.45 s (per-client medians). Two independent streams sharing the
  same weights took 1.88–2.06 s. The single stream lets the scheduler give
  the next turn to the waiting short batch.
- **512 tokens per long block.** In the generator test, reducing the block
  from 2,048 to 512 tokens brought the short client's median first-token
  time from 1.88 to 0.44 s, at a median paired cost of 0.07 s for the long
  request. At 256 tokens the short client reached 0.28 s, but the long
  request took 1.40 s longer than at 2,048.
- **A fixed block size.** A policy that grew blocks after a quiet period
  often left the first unexpected arrival behind a 2,048-token block. Over
  72 short requests per policy in synthetic bursts and irregular traffic,
  the 95th percentile to first text was 1.13 s with fixed 512-token blocks
  and 1.77 s with the adaptive policy.

The released scheduler also gives a waiting long batch a turn after at most
four consecutive short turns, so a continuous stream of short requests
can coexist with a long document. The release comparison measures the
combined scheduling changes. The results apply to this model, machine and
workload.

## Running them again

Install first ([setup](setup.md)): `scripts/setup-mac.sh`. The
commands below were run from a fresh clone of this repository on the machine
above with Docker Desktop, unless the line says otherwise. `MODELS` is the
directory that holds the MLX models (default `$HOME/models`).

MLX bridge, from [`mlx/`](../mlx/):

```sh
tests/run-all.sh                                   # smoke, coverage, edge cases, BGE-M3, generation loop, load, kernels, urgent reads, refused graphs, disconnections
tests/run-all.sh upstream                          # MLX's own Python tests, natively and in the container
tests/lm.sh Qwen/Qwen3.5-0.8B-Base                 # a language model natively and in a container: text, speed, memory
SERVER_ARGS=--custom-kernels tests/lm.sh mlx-community/Qwen3.6-35B-A3B-4bit
SERVER_ARGS=--custom-kernels tests/serve.sh Qwen/Qwen3.5-0.8B-Base   # mlx-lm's server natively and in a container, the same requests to both
```

PyTorch bridge, from [`pytorch/`](../pytorch/):

```sh
./compat.sh                                        # the checks of both ends, then the two batteries, Mac and container, lined up
./lifetime.sh 1                                    # is the GPU memory given back however a program ends
./server.sh start && ./in-container.sh python values_cost.py; ./server.sh stop   # cost of small operations
```

With Kiri: build it with the patches
([`kiri/README.md`](../kiri/README.md)), start its daemon, put the images
into it (`ENGINE=kiri scripts/setup-mac.sh`), then

```sh
ENGINE=kiri mlx/tests/run-all.sh                   # smoke, coverage, edge cases, BGE-M3, generation loop
ENGINE=kiri SERVER_ARGS=--custom-kernels mlx/tests/lm.sh mlx-community/Qwen3.6-35B-A3B-4bit
ENGINE=kiri pytorch/compat.sh                      # the checks and the two batteries through Kiri's channel
```

Each of the three starts its own server on `~/.kiri/run/gpu.sock` and stops
it at the end. The socket has one server, so run them one after the other.

A comparison that fails says which program differs. To see by how much,
run the program on both sides with `DUMP=directory` and give the two
directories to `scripts/tensor-diff.py`.

Further measurements will be published as they are taken.
