# Compatibility

What is supported, what is not, and with which versions. "Supported" here
means that it was run on the machine of the [benchmarks](benchmarks.md) and
gave the Mac's result. Nothing was run anywhere else.

## Versions

| Component | Version | Notes |
|---|---|---|
| Hardware | Apple M4 Max, 128 GB | The only machine. Other Apple-silicon chips are untested. Intel Macs have no bridge. |
| macOS | 26.6.2 | The only version. |
| MLX | 0.32.4, exactly, on both ends | The container's is upstream `v0.32.4` (commit `36969c0`) with [the patch](../mlx/patches/); the Mac's is the stock wheel. Another version on either end is untested and likely to fail: nodes travel in MLX's own export format. |
| mlx-lm | 0.32.2 | The version measured. |
| mlx-embeddings | 0.1.0 | The version measured. |
| PyTorch | 2.15.0 (Python 3.13 in the container) | The same version on both ends, or the server refuses the connection. |
| Bridge protocols | MLX: 11. PyTorch: 3 | Each end refuses another version. See [protocol-mlx.md](protocol-mlx.md), [protocol-pytorch.md](protocol-pytorch.md). |
| Docker Desktop | engine 29.2.1 | Every collection of the released code was run on it. |
| Kiri | upstream `baa2d41` with [the patches](../kiri/README.md), which are now against `69dc0b7` | The collections that need one server were run through it on the released code, with its virtual machine at 8 CPUs and 16 GiB. Most of the earlier measurements were taken on it. |

The bridge is for PyTorch 2.15.0, the one version it was built for and
tested with. Another version needs its own operator lists, made on a Mac
with that version: `TORCH=x.y.z pytorch/build-mac.sh` runs
[`gen_ops.py`](../pytorch/gen_ops.py), which generates them, and
`TORCH=x.y.z pytorch/build-image.sh` builds the image. Nothing about another
version is in the repository.

## MLX bridge: what works

Run on both sides and compared by the type, the shape and the SHA-256 of
the bytes of every array
([benchmarks](benchmarks.md#measured-again-on-the-released-code)). On the
released code the first three gave the same with Docker Desktop and through
Kiri:

- 58 small programs covering MLX's operations: 57 identical. The other,
  `cpu_linalg`, is declared: it asks for a matrix inverse on the CPU stream,
  which Accelerate computes on the Mac and OpenBLAS in the container. The
  two inverses differ in 13 elements of 25, by up to 384 units in the last
  place, and each is right to 1e-6. The GPU gives the same bytes on both
  sides when it is given the same inverse.
- BGE-M3 embeddings with mlx-embeddings: identical.
- Text generation with mlx-lm (Qwen3.6-35B-A3B-4bit, Qwen3.5-0.8B-Base):
  identical text with `--custom-kernels`; see below without.
- mlx-lm's OpenAI-compatible server, with its prompt cache and batched
  requests.
- `mx.compile`, scaled dot-product attention, custom functions and their
  gradients, saving and loading safetensors, npz and npy files.
- Custom Metal kernels, with the server's `--custom-kernels`: 29 of 29 cases
  that run give the same bytes, 5 raise the same kind of error on both
  sides, and 4 are refused on purpose.
- Several threads of one process sharing the weights: right answers, but
  slower than one thread from three threads up
  ([benchmarks](benchmarks.md#several-threads-in-one-process-one-copy-of-the-weights)).

MLX's own Python tests (`mlx/tests/run-all.sh upstream`) run 926 tests on
each side. One test method fails on the Mac, `test_circular_leaks` in
`test_nn`, and 19 fail in the container. The comparison holds each side to
that list: another failure, or one of these passing, fails the stage. The
19 are five custom-kernel tests of `test_fast` (the server of that stage
runs without `--custom-kernels`), `test_segmented_mm` and
`test_gather_mm_sorted_vjp` in `test_blas`, `test_to_from_fp8` and
`test_load_f8_e4m3` (the two primitives below), five tests of
`test_quantized` (`qqmm`, `qqmv`, `gather_qqmm`, its global-scale variant,
`gather_qmm_grad`), `test_buffer_protocol` and `test_memoryless_copy` in
`test_array`, `test_array_buffer_size` in `test_memory`,
`test_circular_leaks` in `test_nn`, and `test_threadlocal_stream` in
`test_threads`. Those without a reason given here were not analysed
further. The container also skips 30 tests that the Mac runs: MLX marks
them for `mx.metal.is_available()`, which is `False` when the server runs
without `--custom-kernels`. They are declared too, and any other skip on
one side fails the stage.

Against a server started with `--custom-kernels`
(`SERVER_ARGS=--custom-kernels mlx/tests/run-all.sh upstream`) the
container skips none of the 30 and the five custom-kernel tests pass. Then
23 tests fail in the container: the other 14 of the 19, and nine that pass
on the Mac. None of the nine is a wrong value computed through the bridge.
They fail for five reasons, declared in `mlx/tests/compare.py`:

- `test_dlx_device_type` expects an array to be in Metal's memory wherever
  `mx.metal.is_available()`. A container's array has its bytes in the
  container's own memory, and DLPack reports the CPU.
- Two convolution tests choose the Mac's kernel with environment variables
  (`MLX_CONV_WINOGRAD` and two others) set in their own process. MLX reads
  them in the process that computes, which for a container is the server.
- Two attention tests do the same with `MLX_SDPA_D512_MIN_KL`. One of them
  also expects an error when the operation is made, and in a container the
  Mac's error arrives when the graph is evaluated.
- Two memory tests count on Metal's cache of freed buffers, which is in the
  server, and on `mx.set_wired_limit()`, which does nothing in a container.
- Two tests of `test_quantized` use `mx.qqmm`, the primitive that
  `test_qqmm` and `test_qqmv` above fail for.

The stage `edge` has 12 cases. Six give the same result on both sides, and
three raise the same error on both: float64 on the GPU, an allocation
larger than Metal makes, and a load on a gpu stream that the program names.
Three fail in the container only, with an error that says why:

- two primitives that MLX's `import_function` cannot rebuild on the Mac,
  `SegmentedMM` and `ToFP8`: "primitive ... is not supported by the remote
  gpu ... Run it on the cpu stream";
- a custom kernel, on a server started without `--custom-kernels`.

Two things that MLX accepts are refused by the server's rules, with an
error that begins `malformed graph:`: a `pad` whose value is an array of
more than one element, and a bitwise operation under `vmap` whose inputs
are not broadcast to one shape.

## PyTorch bridge: what works

Two batteries of small programs, a program counting when it works on the
Mac ([benchmarks](benchmarks.md#compatibility-batteries)):

- Inference (`compat.py`, 47 programs): making tensors, arithmetic,
  indexing, views, sorting, half precision, reading values, moving between
  devices, layers (linear, convolution, embedding, recurrent, attention),
  losses, gradients, small trainings, random numbers, saving a model and
  its views, operations whose result's shape depends on the values, and two
  transformers models (MiniLM sentence embeddings, GPT-2 generation). All
  47 are identical with both engines: every tensor has the same type, shape
  and SHA-256.
- Training (`compat_train.py`, 68 programs): optimizers and schedulers,
  autocast and `GradScaler`, `DataLoader` workers, saving and resuming,
  Lightning with checkpoints and TensorBoard, random numbers from a seed,
  gradients, threads and processes, memory figures, errors. 63 or 62 are
  identical, depending on the run. Four are declared close: two training
  loops of a transformer, `loop_autocast_gradscaler_fused_adam` (the
  gradient of an embedding whose indices repeat) and
  `gradient_grid_sample`. The first three are close in every run. The
  fourth is identical in some runs and within the tolerance in others: it
  was identical in three runs of four, two of the three with Docker Desktop
  and the one in Kiri. Two runs on the Mac itself do not give the four bit
  for bit, and PyTorch marks the two gradients as non-deterministic on MPS.
  For these four the comparison holds the program's printed number to one
  part in a thousand and the tensors to the same types and shapes; it
  compares no element. Two programs fail on both sides alike (`rrelu`,
  attention with dropout).
- Errors: of 17 kinds of error, all 17 are raised as on the Mac, and running
  out of GPU memory is a `RuntimeError` on both sides.
- `torch.backends.mps.is_available()`, `torch.mps.*`, `torch.accelerator.*`,
  and Lightning's `accelerator="mps"`.

## What fails on the Mac fails in the container

The bridge reproduces the Mac's GPU, including what it lacks:

- PyTorch has no float64 on MPS, and no MPS kernel for some operators
  (`rrelu`, attention with dropout). The container raises the Mac's
  `NotImplementedError`.
- No CUDA. Code with CUDA extensions does not run, with or without the
  bridge.
- Training is not always reproducible on the Mac itself. The gradient of
  rows taken by a repeated index is summed by Metal in varying order: the
  container then differs from the Mac as two Mac runs differ from each
  other ([benchmarks](benchmarks.md#field-report-a-point-cloud-segmentation-project)).
- CPU code is Linux's, not macOS's. Whatever a program computes on the
  CPU runs in the virtual machine: a mesh simplifier gave a different mesh
  on Linux and on macOS, and numpy-heavy steps were 50% slower there.
  `torch.randn` on the CPU differs in the last bit between the two systems
  for about one value in fifty. An operation a program puts on MLX's CPU
  stream is computed by the container's CPU, with its libraries
  (`cpu_linalg` above).

## Not supported, and open measurements

- Custom kernels with `#include` are refused by the MLX server, also for
  the three standard Metal headers that mlx-vlm and oMLX use.
- Custom kernels are off unless the server is started with
  `--custom-kernels`. Without them, libraries that test
  `mx.metal.is_available()` take another path: mlx-lm gave a 105-token text
  instead of 93 for one model ([custom kernels](use-cases/custom-kernels.md)).
- A model of 99 GB (Qwen3.8-Flash-Next) was never run: the native load
  was stopped after 16 s when swap grew.
- oMLX, vMLX, vllm-mlx and vllm-metal in a container were never tried.
- vLLM needs more than the bridge. vLLM itself computes with PyTorch on
  CUDA; on Apple GPUs it runs through the `vllm-metal` plugin. Read in its
  source (0.33.0, commit `baf8a0c` of 14 November 2026, not run): the plugin
  requires a native extension, `_paged_ops`, of about 10,500 lines of C++
  and Metal shaders, which defines eight MLX primitives (paged attention,
  writing the KV cache, sampling, and others) and runs them on Metal
  directly, against MLX's private headers. It is pinned to MLX 0.32.4
  exactly, the version of the bridge. Those primitives do not go through
  `mx.fast.metal_kernel`, so the bridge as it is cannot carry them. A first
  check: in a `python:3.12-slim` container on Linux arm64, `pip install
  "vllm==0.33.*"` succeeds (vLLM 0.33.0, PyTorch 2.14.0), the plugin
  installs from source (its Mac-only dependencies carry a Darwin marker and
  are skipped), and `vllm`, `vllm_metal`, `vllm_metal.platform`,
  `vllm_metal.envs` and `vllm_metal.metal` import. Nothing was served: the
  native extension is loaded only on first use. The image was 14.8 GB.
- **PyTorch**: float64 on MPS, MPS streams other than the default, custom
  Metal shaders, `torch.compile`, versions other than 2.15.0. An operator
  that is not on the Mac end's list is refused
  ([security](security.md#what-a-container-can-do-through-the-pytorch-bridge)).
- **MLX**: `/v1/models` of mlx-lm's server fails in the container because of
  mlx-lm itself, which scans a Hugging Face cache directory that is not
  there.
- **Kiri's channel**: builds for Linux, Intel and Windows not checked; one
  connection left open in an edge case, not reproduced.
- **Other frameworks.** TensorFlow, JAX and anything that drives Metal
  itself have no bridge. Keras 3 on its PyTorch backend and `jax-mps` are
  untried candidates ([roadmap](roadmap.md#other-programs-to-carry)).

## Known faults

- Containers that share one MLX server are not separated from each other:
  an index out of range in a GPU operation can reach another connection's
  arrays, and a container that stops reading its answers delays the others
  ([security](security.md#containers-that-share-one-mlx-server)).
- The text of requests sent at once to mlx-lm's server differs from the
  Mac's in some pairs (2 of 4 with two requests, 1 of 4 with four). Each
  answer is the one to its own question. This was not looked into.
- A checkpoint file's SHA-256 differs between Mac and container although the
  weights in it are identical. The cause was not verified.
- The first prompt of an MLX process is slower in the container, at
  155–170 tokens per second where the Mac gives 229–239. The cause was not
  found.
- Several threads of one MLX program share one connection, and three or four
  together do less than one alone.
- A program that used several threads ended with `terminate called without
  an active exception` after printing its results
  (`mlx/tests/threads_gen.py` in a container, in the run for this
  repository). This was not looked into.
- `python -m mlx_lm generate` in a container ends its report with
  `Peak memory: 0.000 GB`.
- MLX's own test `test_divmod` (`test_ops`) draws its inputs at random
  without a seed and fails now and then on either side: in 32 of 1,000 runs
  of that test alone on the Mac and in 23 of 1,000 in a container, always
  for float16. For some pairs the quotient rounds up to the next integer in
  float16, which `mx.divmod` returns, and numpy's floor is one below (81.1
  by 9.016 gives 9 and 8). Of 1,000,000 seeded pairs 437 differ in this
  way, the same pairs with the same bytes on both sides. The stage
  `upstream` reports such a failure with the line `fails this time, as it
  may` and does not count it.
- After a graph fails on the Mac, a program in an MLX container that reads
  the result with `np.array(x)` gets `array #N does not exist (an earlier
  graph failed, or it was freed)` in place of the graph's own error.
  `mx.eval(x)`, `x.item()` and `x.tolist()` report the right one, and no
  wrong value is returned.

What could be done about each is in the [roadmap](roadmap.md).
