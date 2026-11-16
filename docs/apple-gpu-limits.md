# What the Apple GPU allows: the evidence

What the [measurements](benchmarks.md) say about the GPU itself (M4 Max, 40
GPU cores, 128 GB), with or without the bridge. Each line points to the
section that has the figures. They are the findings of one machine: a
different chip, or a different amount of memory, may move the numbers.

This page matters to a reader who does not use the bridge at all: most of
it is about what any program on a Mac meets when it shares the GPU.

## Sharing it between programs

| Finding | Evidence | Section |
|---|---|---|
| Separate streams of work add up to about 1.4 times one, and no more | Qwen3.6-35B: one process 122 tokens/s, two 176, three 166; threads of one process 124.5, 177.1, 186.0, 155.4 at four. Strata's own figures on an M5 Max: 85, 95–102, 115, 119–125 | [Several containers at once](benchmarks.md#several-containers-at-once-and-where-it-stops-scaling), [Several threads in one process](benchmarks.md#several-threads-in-one-process-one-copy-of-the-weights), [A small request alongside a long job](roadmap.md#a-small-request-alongside-a-long-job-who-decides-the-order) |
| One program that batches does better than several that share | mlx-lm's server, 4 requests at once: 216 tokens/s in total on the Mac, 215 in the container | [A model server in a container](benchmarks.md#a-model-server-in-a-container-mlx-lms-own) |
| A small request alongside a long job slows 5 to 8 times, on the Mac itself | two processes: a BGE-M3 sentence 13.5 ms alone, 70–73 ms while another process loops over long prompts; a short answer's first token 55 ms, then 390–470 ms | [A small request alongside a long job](roadmap.md#a-small-request-alongside-a-long-job-who-decides-the-order) |
| There is nothing to set an order with | no public priority between command queues or processes for compute; MLX's maintainers: "there is simply no scheduling in MLX" (issue 4716). ds4, Kapitsa and Strata each keep one owner of the GPU and take turns or batch | same |
| Sharing by separate programs is not even | two servers, one container each: one finished in 6.9 s, the other in 10.6 s; on one server of the bridge, two containers finished in 11.5 s each, and mlx-lm reported the same generation rate in both, 71.1 tokens/s | [Several containers at once](benchmarks.md#several-containers-at-once-and-where-it-stops-scaling) |

## One stream of work

| Finding | Evidence | Section |
|---|---|---|
| The command buffer is the unit, and its boundaries cost | MLX commits one every 50 operations or 50 MB here; raising the second limit to 2,048 MB took generation on the Mac from 131–135 to 144–146 tokens/s, with the same text and the same memory | [A larger command buffer](benchmarks.md#a-larger-command-buffer-mlx_max_mb_per_buffer) |
| Speed falls as the GPU warms, by more than most differences measured | Qwen3.6-35B over one long series: median 136 tokens/s, lowest 78.5 (Kiri) and 130.8, lowest 73.3 (Mac); the segmentation project's step from 1.84 s to 2.70 s over eight runs, 2.02 to 2.83 s over three pairs | [Speed: Qwen3.6-35B](benchmarks.md#speed-qwen36-35b-a3b-4bit-with-mlx-lm-0322), [the segmentation project](benchmarks.md#field-report-a-point-cloud-segmentation-project), [PyTorch 2.12.1 and 2.15.0 compared](benchmarks.md#pytorch-2121-and-2150-compared-on-the-mac) |
| A GPU left idle answers late | after 3 s idle the first token came after 0.44 s on the Mac and 0.32 s in the container, compared with 0.08 s when busy | [A model server in a container](benchmarks.md#a-model-server-in-a-container-mlx-lms-own) |
| The first work of a process is slower | first prompt of a process 229–239 tokens/s on the Mac and 480–493 afterwards; about 310 ms more per connection through the bridge | [Speed: Qwen3.6-35B](benchmarks.md#speed-qwen36-35b-a3b-4bit-with-mlx-lm-0322), [What is still slower](roadmap.md#what-is-still-slower-by-size) |
| Results are not always repeatable on the Mac itself | PyTorch MPS: the gradient of rows taken by a repeated index is summed in varying order, 18 gradient tensors of 82 change between two computations of one step; the gradient of an embedding whose indices repeat differed in 10 native runs of 10; requests batched by when they arrive give different text in some pairs | [the segmentation project](benchmarks.md#field-report-a-point-cloud-segmentation-project), [A model server in a container](benchmarks.md#a-model-server-in-a-container-mlx-lms-own) |

## Memory

| Finding | Evidence | Section |
|---|---|---|
| Memory is shared with everything else, and copies of weights fill it | four copies of a 19.5 GB model (78 GB): 8–9 s to load instead of 2, and 3.4 GB of swap in use; a 99 GB model was stopped while loading when swap grew | [Several containers at once](benchmarks.md#several-containers-at-once-and-where-it-stops-scaling), [Not supported](compatibility.md#not-supported-and-open-measurements) |
| Threads of one process share one copy | 18.2 GiB on the GPU for one to four threads; two containers on the bridge's server, 36.7 GB | [Several threads in one process](benchmarks.md#several-threads-in-one-process-one-copy-of-the-weights), [Several containers at once](benchmarks.md#several-containers-at-once-and-where-it-stops-scaling) |
| What a framework holds varies with its version | the segmentation project's training holds 43.7–44.7 GiB with PyTorch 2.12.1 and 21.4–22.3 GiB with 2.15.0 | [PyTorch 2.12.1 and 2.15.0 compared](benchmarks.md#pytorch-2121-and-2150-compared-on-the-mac) |

## What it does not do

- PyTorch on the Apple GPU has no float64, and no kernel for some
  operations (`rrelu`, attention with dropout): the same
  `NotImplementedError` on the Mac and in the container
  ([Compatibility batteries](benchmarks.md#compatibility-batteries)).
- **No CUDA**: code with CUDA extensions (the full network of the second
  training project) does not run on the Mac, with or without the bridge
  ([the point-network pilot](benchmarks.md#field-report-a-point-network-pilot)).
- **Reported by others, not checked here**: macOS fails a command buffer
  that runs for more than about 5 seconds.
