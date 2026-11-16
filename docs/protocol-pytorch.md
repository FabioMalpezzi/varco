# The PyTorch bridge's wire protocol

This page is a reading of [`pytorch/bridge/wire.py`](../pytorch/bridge/wire.py)
(the framing, shared by both ends), [`client.py`](../pytorch/bridge/client.py)
and [`native.cpp`](../pytorch/bridge/native.cpp) (the container end),
[`server.py`](../pytorch/bridge/server.py), [`session.py`](../pytorch/bridge/session.py)
[`policy.py`](../pytorch/bridge/policy.py) and
[`executor.cpp`](../pytorch/bridge/executor.cpp) (the Mac end). The code is
the authority.

Protocol version **3** (`PROTOCOL` in `wire.py`). The first message checks
the two things that must match, the protocol version and the PyTorch
version, and each end refuses the other when either differs.

## Transport

A byte stream: TCP, or a Unix socket on the Mac's side.

- The server listens on `127.0.0.1:47700`, or on a Unix socket given as its
  argument (`python -m bridge.server [PORT | PATH]`), created with mode
  `0600`.
- The container connects when something first needs the GPU, not before: a
  process that imports PyTorch and never touches `"mps"` costs the Mac
  nothing. The address is `BRIDGE_HOST` and `BRIDGE_PORT` if set, else the
  value of `KIRI_GPU_BRIDGE` (which Kiri sets to `127.0.0.1:47700` in a
  `--gpus` container), else `host.docker.internal:47700`.
- A process made by `fork()` opens its own connection when it first needs
  one.

On the Mac every connection is handed to its own process.

## Frames

Every frame starts with a kind byte and a u32, little endian.

| Kind | Layout | Use |
|---|---|---|
| `J` | `u32` header length, header (JSON), blobs | Anything rare: data, questions, answers. The header's `"b"` lists its blobs' lengths. |
| `D` | `u32` signature, `u32` length, JSON | Defines an operation's signature, once. |
| `O` | `u32` signature, `u32` handles..., `f64`... | One operation: the handles of its tensors, then of its results, then the real numbers that change from call to call. |
| `F` | `u32` count, `u32` handles... | Tensors the container no longer needs. |

`O` and `F` are the hot path: they are written from C++ in the container
and read by C++ on the Mac, with no Python per operation and no reply.

JSON on purpose, not pickle: the Mac end must never run code the container
chose.

Every question carries a number, `"i"`, and its answer carries it back. The
container checks that the answer it reads is the one to the question it
asked; an answer out of step ends the connection, and the program goes on
with a new one, on which the tensors of the closed connection say so when
they are used.

## The greeting

The first frame on a connection is the container's:

```json
{"k": "hello", "protocol": 3, "torch": "2.15.0", "pid": 1234, "program": "train.py",
 "env": {"PYTORCH_ENABLE_MPS_FALLBACK": "1"}}
```

`env` carries the variables that change how MPS behaves and are read when
PyTorch loads (`PYTORCH_ENABLE_MPS_FALLBACK`,
`PYTORCH_MPS_HIGH_WATERMARK_RATIO`, `PYTORCH_MPS_LOW_WATERMARK_RATIO`,
`PYTORCH_MPS_FAST_MATH`, `PYTORCH_MPS_PREFER_METAL`): the Mac's process for
the program is started with the program's values of them. No other variable
is taken from the container, and a value must be a short text or a number.

The reply is `{"torch": "2.15.0", "protocol": 3, "pid": 5678, "err": null}`,
or an `err` that names the two PyTorch versions or the two protocol versions
that differ, after which the connection is closed. An operator's meaning,
and the list of those a Mac runs whole, change with the PyTorch version.

The server reads a greeting as its bytes arrive and takes nothing but a
dictionary of known values. A connection has 5 seconds and 64 KiB to greet,
and at most 64 connections may be waiting to; a malformed greeting or a
silent connection is closed and delays nobody else.

## Signatures and operations

An operation is known to the Mac by its **signature**: the operator and its
overload, its constant arguments, and how many tensors go in and come out.
Shapes are not in it, so an operation met again with new shapes (text
generation, where every token is one longer) keeps its signature.

The first time an operation is met, the container sends a `D` frame whose
JSON looks like this (an illustration, not a capture):

```json
{"op": "aten::add", "ov": "Tensor", "a": [{"t": 0}, {"t": 1}], "kw": {"alpha": 1},
 "nin": 2, "nout": 1, "nf": 0}
```

In `a` and `kw`, `{"t": i}` is the i-th tensor of the call, `{"tl": [first, n]}`
a list of n tensors, `{"f": i}` the i-th real number sent with each call (a
step size that an optimizer changes); anything else is a constant.

Every call after that is an `O` frame: the signature's number, `nin` handles,
`nout` handles for the results (chosen by the container), and `nf` f64
values. The container already knows the results' shapes: it worked them out
by running the operation on empty (meta) tensors, or asked the Mac the first
time when PyTorch cannot.

### Which operators run

Only an operator on the Mac end's list can be named, by exact name and
overload: [`bridge/ops/allowed-2.15.0.txt`](../pytorch/bridge/ops/allowed-2.15.0.txt),
1,972 of the 3,120 `aten` operators of PyTorch 2.15.0. The 89 marked `cpu`
have no GPU kernel on a Mac and run only for a program that set
`PYTORCH_ENABLE_MPS_FALLBACK=1`. [`gen_ops.py`](../pytorch/gen_ops.py) makes
the list on a Mac from PyTorch's own declarations, by the rules of
`policy.py` ([security](security.md#what-a-container-can-do-through-the-pytorch-bridge)).

`policy.check` is the one check that an operation asked for whole (`J op`),
a signature run by Python and a signature run by `executor.cpp` all go
through. Each argument must be plain data of the kind the operator's schema
takes at that place, and the places a signature names in a frame must be
inside the frame it declares (`nin`, `nout` and `nf` are at most 4,096). A
signature that is refused is remembered as broken: its `O` frames are
skipped, and the failure is reported at the next question. An operation
whose tensors are all on the Mac's CPU is refused.

## J messages

The header's `"k"` says which.

| `k` | From the container | Reply |
|---|---|---|
| `hello` | The greeting, above. | Yes. |
| `op` | One operation in full: `op`, `ov`, arguments `a` and `kw`. Not waited for, it names the result handles in `out`, as many as the operator gives results. With `"sync": true` the container waits for what came of it, and says where the results go: `base`, the first number it set aside, and `room`, how many. Used the first time, and for operations whose answer is in the values (`item`, `is_nonzero`, `equal`, `nonzero`, `masked_select`, `unique`). | Only with `sync`: `res`, the results' description, and their bytes where they are values. |
| `put` | A tensor's bytes, as one blob: handle `t`, `dtype`, `shape`. The blob must be exactly the bytes of that shape. With `"cpu": true` it stays a CPU tensor, lent for one operation. | No. |
| `get` | Asks for the bytes of tensor `t`. With `"later": true` the copy is started and not waited for, and its answer comes in the order of the questions. With `"storage": true` the answer is all of the memory the tensor is in, which is what saving writes: a tensor and its views are one storage in the file, as on a Mac. | `shape`, `dtype`, one blob. |
| `sync` | Wait for the GPU (`torch.mps.synchronize()`). | `{}` |
| `call` | One of a fixed table of 27 named calls (`fn`, arguments `a`): seeds and generators, memory figures, MPS settings and events, `use_deterministic_algorithms`. Each takes arguments of a fixed kind. With `"quiet": true` it is told, not asked. | `{"v": value}` unless quiet. |

In the answer to a waited operation, each item of `res` is one of:

| Item | Meaning |
|---|---|
| `{"t": [shape, dtype, stride, offset], "n": k}` | A tensor on the GPU, stored under the number `base + k`. The Mac stores no more than `room` of them. |
| `{"c": [shape, dtype, blob]}` | A tensor that is on the CPU on the Mac, sent back whole as that blob. |
| `{"l": [items]}` | A list of tensors, as the gradient of an LSTM's weights is. |
| `{"v": value}` | A number, a boolean or nothing. |

`"seq"` says whether the operator gave a sequence or a single result.

Every reply carries `"err"`: `null`, or the first error since the last
reply. Operations are not waited for, so an operation that fails on the Mac
is reported at the next question, as the text of the Mac's own exception;
the container raises it as the same kind of error. The answer to a copy
asked for with `later` carries it too.

`use_deterministic_algorithms` is a call because the setting belongs to the
process that computes: with it PyTorch refuses, on the Mac, the GPU
operators that do not give the same bits twice.

## Versions

| Version | What it added |
|---|---|
| 1 | No version number in the greeting: an end that sends none is taken for 1. |
| 2 | The questions' numbers; the first failure in the answers to copies asked for without waiting; the version itself. |
| 3 | `get` with `storage`; `room`, `n` and `l` in waited operations; the call `use_deterministic_algorithms`. |

## Environment variables

| Variable | Where | Effect |
|---|---|---|
| `BRIDGE` | container | `0` turns the bridge off: PyTorch is the plain CPU build of the image. |
| `BRIDGE_HOST`, `BRIDGE_PORT` | container | Where the server is, overriding `KIRI_GPU_BRIDGE` and the default. |
| `KIRI_GPU_BRIDGE` | container | Set by Kiri in a `--gpus` container: `127.0.0.1:47700`. |
| `BRIDGE_TIMEOUT` | container | Seconds to wait for the connection (10). |
| `BRIDGE_BUFFER` | container | Bytes of operations gathered before they are written (1024). |
| `BRIDGE_AHEAD` | container | How many operations the program may send before it waits for the Mac to have taken them (4096; 0 for no limit). |
| `BRIDGE_SYNC` | container | Waits after every operation: for finding which one fails. |
| `BRIDGE_REPORT`, `BRIDGE_DEBUG` | container | Print, when the program ends, what the bridge did and what left the fast path. |
| `BRIDGE_COMMIT` | Mac | `0` stops the session from handing the GPU what it has encoded whenever the container has sent nothing more. |
| `BRIDGE_EXECUTOR` | Mac | `0` makes the session run its Python path instead of `executor.cpp`: for the tests of that path. |
