# Glossary

The words this repository uses, and the ones it avoids.

**Bridge** (GPU bridge). The whole mechanism by which a program in a Linux
container gets its work done on the Mac's Apple GPU. There are two, the MLX
bridge and the PyTorch bridge. Not "GPU passthrough": no device is passed
through.

**Operation level.** The level the bridges work at: array operations issued
inside the container are computed on the Mac's GPU. The alternative, not
built here, is the *model level*, where a container asks a service on the Mac
to load a model and generate text and never sees arrays.

**Container end.** The part of a bridge inside the container: a backend
inside the framework. For MLX, the remote gpu backend of the patched MLX;
for PyTorch, `bridge_native` and the `bridge` package.

**Mac end**, **bridge server**, **the server**. The process on the Mac that
computes on behalf of containers: `mlx-remote-server` for MLX,
`bridge.server` with its sessions for PyTorch. Kiri's glossary calls this the
*executor*. Not to be confused with a model server.

**Executor thread.** Inside the MLX server, the one thread that runs the
work of every connection, a job at a time.

**Session.** In the PyTorch bridge, the process on the Mac that belongs to
one program in a container.

**Model server.** A program that serves a model over HTTP, such as
`python -m mlx_lm server`. It is an ordinary program to the bridge, and can
run in a container.

**Native.** The same program run on the Mac directly, with no container and
no bridge. Every measurement compares the container with native.

**Channel.** What carries the bridge's bytes between the container and the
Mac: a TCP port of the Mac with Docker Desktop, Kiri's own relay (a
listener in the container's network, vsock, a Unix socket) with Kiri.

**Engine.** The program that runs containers: Docker Desktop or Kiri.

**Handle.** The number by which the container names an array or a tensor
that lives on the Mac.

**Graph.** In MLX, the description of a computation that is sent to the Mac
to be evaluated: nodes, their inputs, and which outputs to keep.

**Urgent graph**, **urgent read.** A small MLX graph whose result the
program is about to read, which the server may answer ahead of its queue.

**Custom kernel.** A Metal kernel a program writes itself with
`mx.fast.metal_kernel`: source code the Mac's GPU compiles and runs. Off
unless the MLX server is started with `--custom-kernels`.

**Rule.** What the MLX server checks of a node before it runs it: how many
arrays its primitive takes, and which axes, shapes and types go in and come
out. A primitive without a rule is refused.

**Allow-list.** The list of PyTorch operators the Mac end runs for a
container, by exact name and overload, made by rule from PyTorch's own
declarations (`pytorch/bridge/ops/allowed-2.15.0.txt`).

**Weights read by the Mac.** The MLX server reading a model's weight file
itself instead of receiving the arrays from the container.

**Alternated runs**, **mirrored order.** How timings are taken: Mac,
container, container, Mac, compared pair by pair, because the GPU slows as
it warms.

**Identical**, **to the bit.** The container's result has the same bytes as
the Mac's: for every array or tensor the same type, shape and SHA-256 of
its bytes, and every number a program read bit for bit.

**Declared.** A difference the comparators expect, listed in their source
with its reason: an error on both sides, a refusal on purpose, a value each
side computes on its own CPU, or a program that is **close**.

**Close.** Said of a training program that two runs on the Mac itself do
not give bit for bit. Its printed number differs from the Mac's by at most
one part in a thousand, and its tensors are as many, of the same types and
shapes. No element of a tensor is compared.

**Field report.** A measurement on a private project, which a reader cannot
reproduce.

**As reported.** A figure taken from the notes of the work and not read
back from its raw files.
