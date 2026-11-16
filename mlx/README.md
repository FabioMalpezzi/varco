# The MLX bridge

MLX in a Linux container whose gpu device is the Mac's Apple GPU.

- **Container end**: upstream MLX `v0.32.4` with
  [`patches/0001-remote-gpu-backend.patch`](patches/0001-remote-gpu-backend.patch)
  applied: a "remote" gpu backend (`mlx/backend/remote`) that records the
  nodes of a graph and sends them to the Mac. The patch changes 27 files
  and is MIT, as MLX is.
- **Mac end**: [`server/mlx_remote_server.cpp`](server/mlx_remote_server.cpp),
  one C++ file linked against the stock MLX library of the same version. It
  checks each node of a graph against a rule for its primitive, rebuilds
  the graph with MLX's own `import_function` and evaluates it.
- **Protocol** version 11: [docs/protocol-mlx.md](../docs/protocol-mlx.md).
  Each end refuses another protocol version and another MLX version.

How it works is in [docs/architecture.md](../docs/architecture.md#the-mlx-bridge).

## Build

```sh
./setup-mac.sh           # .venv: the stock mlx 0.32.4 wheel, mlx-lm 0.32.2, mlx-embeddings 0.1.0
./fetch-source.sh        # upstream/: MLX v0.32.4 with the patch applied
server/build.sh          # server/mlx-remote-server
./build-images.sh        # varco-mlx, varco-mlx-embeddings, varco-mlx-lm
```

## Run

```sh
server/ctl.sh start --map /models=$HOME/models       # listens on 127.0.0.1:47800
docker run --rm -v $HOME/models:/models:ro varco-mlx-lm:latest \
  python -m mlx_lm generate --model /models/Qwen/Qwen3.5-0.8B-Base --prompt "Hello" --max-tokens 30
server/ctl.sh stop
```

The container is given no address: it looks for the server at
`MLX_REMOTE` when that is set, else at `KIRI_GPU_BRIDGE`, which Kiri sets
in a container started with `--gpus all`, else at
`host.docker.internal:47800`. Server options:

| Option | Meaning |
|---|---|
| `--port N` | Listen on `127.0.0.1:N` (default 47800). |
| `--unix PATH` | Listen on a Unix socket instead (mode 0600). For Kiri: `~/.kiri/run/gpu.sock`. |
| `--models DIR` | The server may read weight files under DIR for containers that see them at the same path. |
| `--map CONTAINER_DIR=DIR` | The same, for containers that see DIR as CONTAINER_DIR. |
| `--custom-kernels` | Compile and run the Metal kernels containers send. Off unless given: [docs/use-cases/custom-kernels.md](../docs/use-cases/custom-kernels.md). |
| `-v`, `--timing` | Log every message; log where the Mac end's time goes. |
| `--compile-graphs`, `--slow-build F` | Experiments kept from the work: compile whole graphs; slow the building of graphs by a factor. |

`server/ctl.sh start [options]` runs it in the background with its log in
`logs/server.log` and its pid in `logs/server.pid`; `SERVER_NAME=x` keeps a
second server's files apart (`logs/server-x.*`). It refuses a port or a
socket that another process is listening on, and `stop` stops only a
process that is a server.

One server serves every container that connects to it, and those containers
are not separated from each other. A second server on its own port
(`SERVER_NAME=x server/ctl.sh start --port 47810 ...`) separates two groups
of containers ([docs/security.md](../docs/security.md#containers-that-share-one-mlx-server)).

## Test

```sh
tests/run-all.sh                    # every stage, natively and in a container, compared by bytes
tests/run-all.sh coverage kernels   # some stages
ENGINE=kiri tests/run-all.sh        # in Kiri: the five stages that need one server
tests/lm.sh Qwen/Qwen3.5-0.8B-Base  # a language model on both sides
SERVER_ARGS=--custom-kernels tests/serve.sh Qwen/Qwen3.5-0.8B-Base   # mlx-lm's server on both sides
```

`MODELS` (default `$HOME/models`) is the directory of the models. The
stages are described in [docs/development.md](../docs/development.md#testing),
the results in [docs/benchmarks.md](../docs/benchmarks.md#mlx-bridge). The
raw results the earlier tables cite are in [`results/reference/`](results/reference/).

## Files

| | |
|---|---|
| `versions.sh` | The pinned MLX tag and commit, mlx-lm and mlx-embeddings versions |
| `patches/` | The patch against MLX |
| `server/` | The Mac server, its build script, start and stop |
| `docker/` | The images, and the development container |
| `tests/` | Test programs (run on both sides), their comparators and inventories (`expected/`), and the scripts that drive them |
| `results/reference/` | Raw JSON results of the earlier measurements |
