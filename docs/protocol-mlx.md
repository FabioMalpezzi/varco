# The MLX bridge's wire protocol

Protocol version **11**. This page is a reading of
`mlx/backend/remote/protocol.h`, the header shared by both ends, which is
part of [the patch](../mlx/patches/0001-remote-gpu-backend.patch) and is the
authority: where this page and the header disagree, the header is right.
After `mlx/fetch-source.sh` the header is at
`mlx/upstream/mlx/backend/remote/protocol.h`.

The container end is `mlx/backend/remote/client.cpp` in the patch; the Mac
end is [`mlx/server/mlx_remote_server.cpp`](../mlx/server/mlx_remote_server.cpp).

## Transport

A byte stream: TCP to `host:port`, or a Unix socket. The container takes the
address from the environment variable `MLX_REMOTE` (`host:port`, or
`unix:PATH`) when it is set and not empty, else from `KIRI_GPU_BRIDGE`
(which Kiri sets to `127.0.0.1:47700` in a `--gpus` container), and uses
`host.docker.internal:47800` when neither is set. The server listens on
`127.0.0.1` (default port 47800) or on a Unix socket (`--unix PATH`).

All integers are little endian.

## Frames

Every message from the container is a frame:

| Field | Size |
|---|---|
| type | u8 |
| payload length | u64 |
| payload | that many bytes |

Only HELLO, READ, SYNC, STATS and OPEN are answered. A reply is a frame:

| Field | Size |
|---|---|
| status | u8: 0 ok, 1 error |
| length | u64 |
| payload | the answer, or the error |

The other messages are one way, so that the container never waits for the
Mac while it builds graphs. An error they cause is kept by the server and
returned by the next message that has a reply, or by the next DONE frame.

An error is one byte for its kind, then its text. The kind is the exception
MLX raised on the Mac, so that the container raises the same and a program
catches there what it would catch on the Mac:

| Kind | Exception on the Mac | In Python |
|---|---|---|
| 0 | any other | `RuntimeError` |
| 1 | `std::invalid_argument` | `ValueError` |
| 2 | `std::out_of_range` | `IndexError` |
| 3 | `std::length_error` | `ValueError` |
| 4 | `std::domain_error` | `ValueError` |
| 5 | `std::range_error` | `ValueError` |
| 6 | `std::overflow_error` | `OverflowError` |

The error of a HELLO is text alone, so that ends of two versions can tell
each other so.

The server closes a connection that sends a frame of an unknown type, or a
payload longer than 2^37 bytes.

The Mac also sends frames nobody asked for:

| Status | Name | When |
|---|---|---|
| 2 | DONE | One for each GRAPH that has the "tell" flag, in their order, as soon as the GPU has finished it. |
| 3 | MORE | Right after a DONE whose count has bit 31 set: the large outputs the container marked, in a separate frame so that they do not hold back the small ones. |
| 4 | EARLY | An urgent graph answered ahead of the queue. Says that this graph alone is done, nothing about the earlier ones. |

All three have the same layout: `u64 n` (the GRAPH, OP and LOAD messages
evaluated so far), `u32 length` and an error (its kind and text; nothing if
none), `u32 count`, then `count` times (`u64 handle`, `u64 nbytes`, the
bytes in row major order). The container meets them while it reads the
reply to something else, or reads them on purpose when it waits for that
graph.

A DONE frame carries the failure of any message before its graph that had
no reply to say it, so "done" is never taken for "went well". An EARLY
frame never carries an error ([urgent graphs](#urgent-graphs)).

## Common encodings

- **Handle**: u64, chosen by the container, never 0. It names an array on
  the Mac, within one connection.
- **dtype**: `u32 Dtype::Val`, `u8 size`.
- **shape**: `u32 ndim`, then `ndim` times `i32`. At most 64 dimensions.
- **Value**, inside a GRAPH: a u32 that is an input of the message (bit 31
  set, its index in the other bits) or an output of a node (numbered in
  order over all nodes).

## Messages

| Type | Name | Payload | Reply |
|---|---|---|---|
| 1 | HELLO | `u32` protocol version, `u32` length and text of the container's MLX version | `u32` features of the server (bit 0: it takes custom Metal kernels), `u32` length and text of the Mac's MLX version, then text describing it. Both versions must be the Mac's own, or the reply is an error; the container checks the Mac's answer in its turn. The server takes no other message before the greeting. |
| 2 | UPLOAD | handle, dtype, shape, raw bytes | none |
| 3 | GRAPH | see below | none (a DONE frame if "tell") |
| 4 | OP | `u32 op`, `u32 n_inputs` × (handle, dtype, shape), `u32 n_outputs` × handle. One public MLX operation, for primitives the export format cannot hold. The only one defined is 1, bitwise invert. | none |
| 5 | FREE | `u32 n`, n × handle | none |
| 6 | READ | handle | the array's bytes, row major |
| 7 | SYNC | `u64 n` | empty, once the first `n` GRAPH and OP messages of the connection are evaluated (0: all of them) |
| 8 | STATS | – | text with the connection's counters, [below](#stats) |
| 9 | OPEN | `u32 file` (chosen by the container, never 0), `u64` size of the file, `i64` time it was last modified (nanoseconds), `u64` digest of parts of it, then its path as the container sees it | `u8 1`, or `u8 0` and why not |
| 10 | LOAD | handle, dtype, shape, `u32 file`, `u64 offset`, `u8` swap byte order | none. Counts as a GRAPH for SYNC and DONE. Fails if the file changed since its OPEN. |
| 11 | CLOSE | `u32 file` | none |
| 12 | KERNEL | see below | none |

### GRAPH

```
u8  flags      bit 0: start evaluating      bit 1: tell when done
               bit 2: some node is a custom kernel      bit 3: urgent
u32 n_inputs,  n_inputs x (handle, dtype, shape)
u32 n_keep,    n_keep x (u32 value, handle)
               bit 31 of the value asks for the bytes in the DONE frame,
               bit 30 in the MORE frame
u32 n_nodes,   then for each node, in topological order:
    u32 length and the primitive: stream, name and state, in the bytes
        MLX's export_function writes and import_function reads
    u16 n_in,  n_in x u32 value
    u16 n_out, n_out x (dtype, shape)
```

The values listed in "keep" get a handle; the others are intermediates that
the Mac need not keep.

Two kinds of node are not a saved primitive, and say so in the top bits of
the length:

- **bit 31, a Compiled primitive** (what `mx.compile` fuses), sent as what
  its constructor takes: its inputs (dtype, shape, whether constant and the
  element if so), its tape of primitives, and its outputs.
- **bit 30, an operation the Mac builds by calling MLX**, since how it is
  computed depends on the GPU: `u32 call`, then
  - 1, attention (`scaled_dot_product_attention`): `f32 scale`, `u8 causal`,
    `u8` has a mask, `u8` has sinks, `u8 force_fused`;
  - 2, `contiguous`: `u8 allow_col_major`;
  - 3, a custom kernel sent before with KERNEL: `u32 kernel`, 3 × `i32`
    grid, 3 × `i32` threadgroup, `u8` has an init value, `f32` init value,
    `u16 n_template` × (`u8` kind, `u16` length and name, then an `i32`, a
    `u8` or a dtype by kind);
  - 4, MLX's fused cross entropy (`fast::cross_entropy`, of logits and
    targets), and 5, its gradient (of logits, targets, the loss and its
    cotangent): nothing more. Sent as the operations they stand for, they
    gave other last bits than the Mac's.

### What the server checks of a graph

The server validates every node before MLX is given it
(`check_shapes` and `check_node` in the server's source). A saved primitive
must be one of the 114 the server has a rule for, on the GPU's stream, with
as many inputs as its primitive reads. Its state must go with its arrays:
an axis is one of the array's, a list of axes, sizes or strides is as long
as the array has dimensions, a slice or a view by strides lies inside its
input. Its outputs must have the shape and the type that the primitive
makes of those inputs. A compiled function may hold only the primitives
that `mx.compile` fuses, and is given the arrays it was compiled for. A
node that fails a rule fails its graph with an error that begins
`malformed graph:`.

The rules are those of the operations that make each primitive in MLX
0.32.4. Two nodes that MLX itself can make are refused: a `Pad` whose value
has more than one element, and a bitwise operation whose two inputs do not
have the same shape, which MLX makes under `vmap` with inputs that are not
broadcast.

The values inside the arrays are not checked: an index out of range, on the
GPU, reads or writes what the GPU has there
([security](security.md#what-a-container-can-do-through-the-mlx-bridge)).

### Urgent graphs

A GRAPH with the urgent flag is a small one whose results the program is
about to read (the value of one element, a comparison). When all its inputs
are already computed and its nodes only select, move or compare elements,
which gives the same bytes on any device, the Mac runs it immediately on its
CPU, ahead of the messages queued before it and without waiting for the GPU
work in flight, and answers with an EARLY frame. Otherwise the graph takes
its turn and gets a DONE frame like any other.

Selecting and moving are an index, a slice, a reshape, a concatenation;
comparing is comparison and logic on integers and booleans. Nothing that
computes with floating point is answered early, and no comparison of
floating point either. An early graph has at most 64 nodes and no array
larger than 4 MiB. The CPU path goes through the same rules as the GPU's,
and the indices of a gather, which are data, are looked at before the graph
runs.

An EARLY frame carries no error, and a container whose graphs were all
answered early takes everything it sent for done. So an urgent graph takes
its turn, where its DONE frame can carry a failure, in three cases: while a
failure of an earlier message waits to be told; when a message ahead of it
on its connection could still fail without a reply (a kernel, an upload
that is not the graph's own operand); and for a connection that has not
greeted.

### OPEN, LOAD, CLOSE: weights read by the Mac

The Mac opens the file, read only, if the path is under one of the
directories its server was started with (after their mapping, and with
symlinks and `..` resolved) and the file it finds has that size, was last
modified at that time and has those bytes where the digest looks. Otherwise
the container reads the file and uploads, as for any other data.

The digest is FNV-1a over the first 64 KiB of the file (for safetensors, the
names and offsets of the tensors), sixteen parts of 4 KiB at even distances
after them, and the last 64 KiB; a file of 128 KiB or less is digested
whole. The two ends may see the modification time with different precision
(a file system between them can keep microseconds or seconds), and the
comparison allows for that. A file of the same size, modified at the same
instant and equal in the parts the digest covers is taken for the
container's.

A LOAD says that an array is those bytes of the file, which the Mac reads
when the message arrives. A file that changed since its OPEN is not read,
and the LOAD fails. A connection may have 256 files open.

### KERNEL: custom Metal kernels

Sent once per connection, and only to a server that says in its HELLO that
it takes them (it is started with `--custom-kernels`):

```
u32 kernel     chosen by the container, never 0
               then what MLX's metal_kernel takes:
u32 length and name
u32 n_inputs  and their names  (u32 length and text each)
u32 n_outputs and their names
u32 length and source
u32 length and header
u8  ensure_row_contiguous
u8  atomic_outputs
i32 compile options (the math mode)
```

The limits a server accepts: source and header together at most 256 KiB;
names at most 128 characters; at most 64 inputs, 64 outputs and 64 template
arguments; at most 256 kernels per connection. A kernel goes, with what MLX
compiled of it, when the last connection that sent it closes. The whole list
of what is refused is in
[custom Metal kernels](use-cases/custom-kernels.md#what-the-server-still-refuses).

### STATS

The reply is one line of text with the counters of the connection that
asks, which the server also writes to its log when a connection closes:

```
graphs=N (urgent N, early N) nodes=N imports=N compiles=N ops=N kernels=N
uploads=N (N B) reads=N (N B) files=N (refused N) loads=N (N B, N ms)
syncs=N frees=N live_arrays=N mac_active_memory=N server_kernels=N kernel_libraries=N
```

`urgent` and `early` are the graphs the container marked urgent and those
answered early; `files` and `refused` the files opened and refused; `loads`
the arrays read from them. The last three are the server's own: the memory
MLX holds, and the custom kernels and compiled kernel variants kept for all
the containers connected.

## Versions

The container sends its protocol version in HELLO, and the server refuses
any version but its own. The two ends of this repository are built from the
same header and both are 11. What each version added:

| Version | What it added |
|---|---|
| 4 | The first version measured in Kiri. Weights travelled through the container. |
| 6 | Compiled functions and attention as single nodes, made again on the Mac. |
| 7 | The Mac reads the weight files by itself (OPEN, LOAD, CLOSE). |
| 8 | Custom Metal kernels on a Mac that takes them (KERNEL, the features of HELLO); `contiguous` made by the Mac's own. |
| 9 | Urgent graphs and the EARLY frame. |
| 10 | The MLX version in HELLO and in its reply; the modification time in OPEN and a digest of more of the file than its first bytes; the failure of an earlier message in DONE. The server checks what it reads of a graph. |
| 11 | An error carries its kind; cross entropy and its gradient as operations the Mac builds. Every admitted primitive has a rule, and a primitive without one is refused. |

The measurements of [benchmarks](benchmarks.md) name the version each was
taken with.

## Environment variables of the container end

Read in `client.cpp`. They are diagnostic switches, apart from the first
two.

| Variable | Effect |
|---|---|
| `MLX_REMOTE` | Address of the server: `host:port` or `unix:PATH`. |
| `KIRI_GPU_BRIDGE` | Set by Kiri in a `--gpus` container: `127.0.0.1:47700`. The address used when `MLX_REMOTE` is not set. |
| `MLX_REMOTE_SPIN_US` | Keeps the waiting thread awake for that many microseconds while the Mac works. |
| `MLX_REMOTE_NO_LOAD` | The container reads weight files itself and uploads them. |
| `MLX_REMOTE_NO_URGENT` | No graph is marked urgent. |
| `MLX_REMOTE_NO_KERNELS` | Custom kernels are not used: `mx.metal.is_available()` is `False` without the server being asked. |
| `MLX_REMOTE_NO_COMPILE` | `mx.compile` returns the function as it is, not fused. |
| `MLX_REMOTE_NO_TELL`, `MLX_REMOTE_NO_CHUNKS`, `MLX_REMOTE_NO_INTERN`, `MLX_REMOTE_NO_SDPA` | Turn off, one each: DONE frames, sending a long evaluation in parts, reuse of small constants, attention as a single node. |
| `MLX_REMOTE_TRACE`, `MLX_REMOTE_STATS` | Print what is sent, and where this end's time goes. |

## Options of the server

```
mlx-remote-server [--port N | --unix PATH] [-v] [--timing]
                  [--compile-graphs] [--slow-build F] [--custom-kernels]
                  [--models DIR]... [--map CONTAINER_DIR=DIR]...
```

`--models DIR`: the files under DIR may be read by the server for a
container that sees them under the same path. `--map CONTAINER_DIR=DIR`: for
a container that sees DIR as CONTAINER_DIR. Without either no file is ever
read by the server. `--custom-kernels` is off unless given
([security](security.md#custom-metal-kernels)). `MLX_MAX_MB_PER_BUFFER` and
`MLX_MAX_OPS_PER_BUFFER` in the server's environment are MLX's own
([a larger command buffer](benchmarks.md#a-larger-command-buffer-mlx_max_mb_per_buffer)).
