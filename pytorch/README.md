# The PyTorch bridge

PyTorch in a Linux container whose `"mps"` device is the Mac's Apple GPU.

- **Container end**: the stock CPU build of PyTorch for Linux, plus
  [`bridge/native.cpp`](bridge/native.cpp) (an MPS backend whose one kernel
  forwards every operation) and the Python of [`bridge/`](bridge/). A `.pth`
  file loads it in every Python process that imports `torch`.
- **Mac end**: [`bridge/server.py`](bridge/server.py), which gives every
  program that connects a process of its own
  ([`bridge/session.py`](bridge/session.py), with its hot path in
  [`bridge/executor.cpp`](bridge/executor.cpp)) running the operations on
  the real MPS. It runs only the operators of a list
  ([`bridge/ops/allowed-2.15.0.txt`](bridge/ops/allowed-2.15.0.txt), 1,972
  of PyTorch's 3,120), on arguments checked against each operator's schema
  ([`bridge/policy.py`](bridge/policy.py)).
- **Version**: PyTorch 2.15.0, the same on both ends.
- **Protocol** version 3: [docs/protocol-pytorch.md](../docs/protocol-pytorch.md).
  Each end refuses another protocol version and another PyTorch version.

How it works is in [docs/architecture.md](../docs/architecture.md#the-pytorch-bridge).

## Build

```sh
./build-mac.sh           # .venv-mac-2.15.0, the operator lists, the Mac end's C++
./build-image.sh         # varco-torch:2.15.0
```

The bridge is for PyTorch 2.15.0. `gen_ops.py`, which `build-mac.sh` runs,
makes the operator lists from the Mac's PyTorch; `TORCH=x.y.z` in front of
both commands makes them, and the image, for another version, about which
nothing was tested.

## Run

```sh
./server.sh start        # listens on 127.0.0.1:47700
docker run --rm -v "$PWD/tests:/work" varco-torch:2.15.0 python program.py
./server.sh stop
```

Nothing is passed to the container: the bridge connects to
`host.docker.internal:47700` when the program first uses `"mps"`.
`BRIDGE=0` in the container turns it off. With Kiri:
`ENGINE=kiri ./server.sh start` listens on `~/.kiri/run/gpu.sock` and on no
TCP port, and the container is started with `--gpus all`. The script uses
a bridge server of the same PyTorch that is already listening there, on
the socket as on the port, and leaves it running; it refuses a socket that
any other process is listening on.

`./in-container.sh COMMAND` runs a command in the image with `tests/` at
`/work`, the Hugging Face cache at `/hub`, and this directory's `bridge/`
mounted over the image's:

```sh
./in-container.sh python models.py
./in-container.sh python -m bridge program.py     # the same, with a count of what the bridge did
```

## Test

```sh
./compat.sh                      # the checks of both ends, then both batteries, Mac and container, compared by bytes
./compat.sh compat               # one battery; checks runs the checks alone
ENGINE=kiri ./compat.sh          # the same through Kiri's channel
ONLY=lightning ./in-container.sh python compat_train.py   # the programs whose name contains it (server running)
./lifetime.sh 1                  # is GPU memory given back however a program ends
./compare.sh 1                   # three programs timed in mirrored order
```

The batteries need `sentence-transformers/all-MiniLM-L6-v2` and
`openai-community/gpt2` in the Hugging Face cache; `bench.sh` and
`compare.sh` need `BAAI/bge-m3` and `BAAI/bge-reranker-v2-m3` too. The
results are in [docs/benchmarks.md](../docs/benchmarks.md#pytorch-bridge).

## Files

| | |
|---|---|
| `versions.sh` | Which PyTorch (`TORCH`), which engine (`ENGINE`), image name, port |
| `bridge/` | Both ends of the bridge; `bridge/ops/` holds the two operator lists |
| `gen_ops.py` | Makes the operator lists of the Mac's PyTorch: those with an MPS kernel, and those the Mac end runs |
| `docker/` | The image and the packages both ends install |
| `tests/` | `compat.py`, `compat_train.py`, their comparator and inventories, the checks of both ends (`refusals.py`, `client_checks.py`, `fork_reading.py`, `reconnect.py`, `other_pytorch.py`, `no_server.py`), and the programs the timings were taken with |
