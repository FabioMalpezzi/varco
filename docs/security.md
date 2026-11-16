# Security

varco opens a channel from a container to the Mac's GPU. That is what it is
for: a program in a container sends its array operations to a process on the
Mac, which runs them on the GPU as your user. A container that has the
channel can therefore compute on the GPU and use the Mac's memory, as a
program you start on the Mac can.

What the Mac end checks. The operations each bridge carries were
inventoried, and only those on the list run: 114 MLX primitives, each with a
rule on its inputs, and 1,972 of PyTorch 2.15.0's 3,120 operators. A request
for anything else, or a message that is not the protocol's, is refused. The
MLX server reads model files only under the directories you name when you
start it. GPU programs written by the container (custom Metal kernels) are
refused unless you start the server with `--custom-kernels`.

What is left to you. Which containers get the channel, which directories
the server may read, and whether custom kernels are on. The checks are those
of a prototype that was reviewed twice before publication, not of a hardened
service: give the channel to containers you run yourself, or trust as you
would a program you start on the Mac.

## Who can connect

| Setup | Who reaches the server |
|---|---|
| Docker Desktop, server on a TCP port | Every container of the engine, through `host.docker.internal`, and every process on the Mac. The servers listen on `127.0.0.1` only, so other machines on the network do not reach them. |
| Kiri with the patches, server on `~/.kiri/run/gpu.sock` and on no TCP port | Containers created with a GPU request (`--gpus all`), each through a listener in its own network namespace. The Unix socket is created with mode `0600`, so only your user's processes can open it. |

Neither setup authenticates a connection. Nothing identifies a container to
the server beyond the connection itself.

With Kiri, "only a `--gpus` container reaches the bridge" holds only while
no bridge server listens on a TCP port. Kiri forwards
`host.docker.internal:PORT` of every container to the Mac's loopback, so a
server on a port is reached by containers started without `--gpus` as well.
With `ENGINE=kiri` the scripts of this repository start the servers on the
socket and on no port, and `ENGINE=kiri scripts/doctor.sh` reports a bridge
server on a TCP port as a fault.

Kiri refuses the combinations that would share the listener. A container
with `--gpus` cannot use `--network host` or `--network container:...`, and
no container can join the network namespace of a container that has
`--gpus`: the daemon refuses it when the container is created and when it is
started, and the guest agent refuses to join a namespace that holds a bridge
listener
([`0001-gpu-bridge-channel.patch`](../kiri/patches/0001-gpu-bridge-channel.patch)).

## What a container can do through the MLX bridge

A container sends graphs. The server validates each graph before it runs
it: every node must be one of the 114 primitives the server has a rule for,
on the GPU's stream, with the number of inputs, the axes, the shapes and the
types that MLX's own operation would have given it. A primitive without a
rule is refused, and so is a node that MLX's operations would not have made.
Besides primitives a graph may hold a compiled function (made only of
primitives that `mx.compile` fuses), one of five operations the server
builds by calling MLX (attention, `contiguous`, cross entropy and its
gradient, a custom kernel), and one operation sent by number (bitwise
invert). The rules are in `check_shapes` in
[`mlx/server/mlx_remote_server.cpp`](../mlx/server/mlx_remote_server.cpp);
[`mlx/tests/graph_refusals.py`](../mlx/tests/graph_refusals.py) sends the
server what a container of the bridge never would and expects each refusal.

Handles are per connection: a container cannot name another connection's
array. The bytes the server sends back are bounded by the memory each array
was given, and the buffers it fills for a container are zeroed beyond their
data.

The rules cover the structure of a graph, not the values in its arrays. MLX
does not check an index on the GPU, so a gather with an index out of range
reads, and a scatter writes, whatever the GPU holds at that place, which can
be an array the server holds for another connection.

The server does not limit memory or GPU time. A connection's arrays are
freed when it closes, and the process gives its memory back to macOS when
the last connection closes.

### Containers that share one MLX server

One server process serves every container that connects to it, with one
executor thread that takes their work in arrival order. Containers on the
same server are therefore not separated from each other:

- an index out of range in a GPU operation can read or write arrays the
  server holds for another connection, as described above;
- a container that stops reading its answers delays the others, because
  nobody else is served while the server waits to write an answer
  ([`mlx/tests/disconnects.py`](../mlx/tests/disconnects.py) prints this as
  a known limit);
- a long job of one container is ahead of the others' work in the one
  queue;
- no container has a quota of memory or of GPU time.

The way round is a server per container, or per group of containers that
trust each other, started by hand on its own port. `SERVER_NAME` keeps a
second server's log and pid file apart, and the container is told the port
with `MLX_REMOTE`:

```sh
SERVER_NAME=team-a mlx/server/ctl.sh start --port 47810 --map /models=$HOME/models
docker run --rm -e MLX_REMOTE=host.docker.internal:47810 -v $HOME/models:/models:ro varco-mlx-lm:latest python your_program.py
SERVER_NAME=team-a mlx/server/ctl.sh stop
```

Each server then holds its own copy of the weights its containers load.
Kiri's channel leads to one socket, so with Kiri every `--gpus` container
shares the server that listens there.

### Weight files

The server reads files only when it is started with `--models DIR` or
`--map CONTAINER_DIR=DIR`, and then only regular files under those
directories. The path a container gives is mapped, symlinks and `..` are
resolved, the result must be under an allowed directory, the file is opened
read-only, and it is checked again once open.

The container must already know the file. It sends the file's size, the
time it was last modified, and a digest of its first 64 KiB, its last
64 KiB and sixteen 4 KiB parts in between, and the server opens the file
only if all three match. A `LOAD` from a file that changed after it was
opened fails. A different file with the same size, the same modification
time and the same bytes in the parts the digest covers is taken for the
container's file.

[`mlx/tests/load_refusals.py`](../mlx/tests/load_refusals.py) asks the
server, in the protocol itself, for a file outside the directories, through
a symlink, through `..`, for a missing file, a directory, a pipe, a file
with another size, time or content, and for a read past the end, and
expects a refusal each time (the stage `load` of `mlx/tests/run-all.sh`).

Without `--models` or `--map` the server reads no file at all.

### Custom Metal kernels

A kernel made with `mx.fast.metal_kernel` is source code that the Mac
compiles and its GPU runs. The server takes kernels only when it is started
with `--custom-kernels`. A kernel reads and writes the GPU's memory through
pointers that nothing bounds, so with the option on, the containers on that
server must trust each other: one container's kernel can read and change
the arrays of another. The server still refuses a source that could make
the compiler read a file of the Mac (`#include`, `#import`, `#pragma`).

The page [custom Metal kernels](use-cases/custom-kernels.md) has the whole
list of what is refused, the commands that turn the option on and off, and
how to see its state.

## What a container can do through the PyTorch bridge

A container names operators and gives them arguments. The Mac end runs an
operator only if it is on a list, by exact name and overload:
[`pytorch/bridge/ops/allowed-2.15.0.txt`](../pytorch/bridge/ops/allowed-2.15.0.txt)
holds 1,972 of the 3,120 `aten` operators of PyTorch 2.15.0. Of those, 89
have no GPU kernel on a Mac and are marked `cpu`: they run only for a
program that set `PYTORCH_ENABLE_MPS_FALLBACK=1`.

The list is made by rule
([`pytorch/gen_ops.py`](../pytorch/gen_ops.py),
[`pytorch/bridge/policy.py`](../pytorch/bridge/policy.py)). An operator is
on it if a Mac can run it for a GPU tensor and none of these holds:

- it belongs to a family that works on something other than a plain tensor
  (sparse, quantized and nested tensors, other backends, PyTorch's own
  tests) or that is not computation (`from_file`, printing, pinned memory);
- one of its arguments or results is not plain data, or is a string other
  than one of a few named options;
- it takes its arguments on trust. Five operators were read in PyTorch's
  source and found to do so (`_unsafe_masked_index`,
  `_unsafe_masked_index_put_accumulate`, `segment_reduce` and its gradient,
  `adaptive_max_pool3d_backward`). Every other operator whose name contains
  "unsafe" is left out too, except five that were read and found to hand
  their arguments to an operator that checks them.

Before an operator runs, each argument is checked against the operator's
schema: a tensor of the connection by its number, a number, a boolean, a
list of those, a type, a memory format, the GPU or the CPU as a device.
Sizes, strides and offsets are counted without overflow. No tensor a
container can name may reach outside its own memory. An operation whose
tensors are all on the Mac's CPU is refused, so that no CPU kernel runs on a
container's data unless the program asked for the fallback. The same checks
are in the Python path and in the C++ one
([`executor.cpp`](../pytorch/bridge/executor.cpp)).

Messages are JSON, not pickle, and the Mac end never runs code a container
chose. Besides operators a container can ask for one of 27 named calls
(seeds, generators, memory figures, synchronisation, profiler events), each
with arguments of a fixed kind.
[`pytorch/tests/refusals.py`](../pytorch/tests/refusals.py) sends the Mac
end what a program of the bridge never would and expects each refusal.

What the list does not do is make PyTorch's operators safe against hostile
arguments. It is made by rule; the ten operators named above were read in
PyTorch's sources, and the others were not read one by one. An operator on
the list that reads where its data tells it (an index beyond a tensor, on
the GPU) does for a container what it does for a program on the Mac. With
`PYTORCH_ENABLE_MPS_FALLBACK=1` the operators marked `cpu` run CPU kernels
on the container's data.

Every program has its own process on the Mac for that reason. Running out
of GPU memory, or a crash, ends that process and no other, and its memory
is given back when it ends, however the program ended
([`pytorch/lifetime.sh`](../pytorch/lifetime.sh) checks five ways of
ending). A container uses memory as a program on the Mac does, within
PyTorch's own MPS limits (`PYTORCH_MPS_HIGH_WATERMARK_RATIO` is passed from
the container's environment to its process on the Mac). Nothing limits how
many processes a container starts on the Mac, one per connection.

The server refuses a container whose PyTorch version or protocol version is
not its own. A connection has five seconds and 64 KiB to send its greeting,
and at most 64 connections may be waiting to greet.

## What is not covered

- The bytes are not encrypted. They travel on a loopback port or a local
  socket.
- The servers log connections, kernels and failures, not what was computed.
- Two rounds of adversarial review were done before publication. Their
  findings were fixed or are listed as limits on this page. No outside
  party has audited the code.
- Kiri's channel was built and checked on an Apple-silicon Mac only.

To report a problem, see [SECURITY.md](../SECURITY.md).
