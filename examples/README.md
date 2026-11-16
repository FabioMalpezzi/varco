# Examples

One runnable example per use case. Each directory has a `run.sh` that
starts the bridge's server it needs, runs an ordinary program in a
container, and stops the server; most take `native` to run the same program
on the Mac directly, so that you can compare.

Install first, once: `scripts/setup-mac.sh`, and get the small models:
`scripts/fetch-models.sh` ([setup](../docs/setup.md)).

| Example | Use case | Bridge | Needs |
|---|---|---|---|
| [`01-embeddings`](01-embeddings/) | [Embeddings in a container](../docs/use-cases/embeddings.md) | MLX | `mlx-community/bge-m3-mlx-fp16` under `MODELS` |
| [`02-openai-server`](02-openai-server/) | [An OpenAI-compatible model server](../docs/use-cases/openai-server.md) | MLX | `Qwen/Qwen3.5-0.8B-Base` under `MODELS` |
| [`03-pytorch-inference`](03-pytorch-inference/) | [PyTorch inference](../docs/use-cases/pytorch-inference.md) | PyTorch | `all-MiniLM-L6-v2` and `gpt2` in the Hugging Face cache |
| [`04-pytorch-training`](04-pytorch-training/) | [PyTorch training](../docs/use-cases/pytorch-training.md) | PyTorch | nothing |
| [`05-same-compose-nvidia`](05-same-compose-nvidia/) | [The same Compose file on a Mac and on an NVIDIA server](../docs/use-cases/same-compose-nvidia.md) | PyTorch | nothing |
| [`06-several-containers`](06-several-containers/) | [Several containers at once](../docs/use-cases/several-containers.md) | MLX | `Qwen/Qwen3.5-0.8B-Base` under `MODELS` |
| [`07-custom-kernels`](07-custom-kernels/) | [Custom Metal kernels](../docs/use-cases/custom-kernels.md) | MLX | nothing |

## Settings

All optional, read by [`common.sh`](common.sh):

| Variable | Default | Meaning |
|---|---|---|
| `MODELS` | `$HOME/models` | The directory of the MLX models, mounted at `/models` in the containers |
| `HF_CACHE` | `$HOME/.cache/huggingface` | The Hugging Face cache the PyTorch examples read |
| `MLX_PORT` | `47800` | With Docker Desktop, the port of the examples' MLX server, which their containers are given |
| `ENGINE` | `docker` | `kiri` runs the example in Kiri: `ENGINE=kiri examples/01-embeddings/run.sh` |
| `DOCKER_HOST` | not set | Left as the example finds it, so the `docker` command reaches the engine it reaches by itself. `ENGINE=kiri` sets it to Kiri's socket. Example 6 starts its containers with `mlx/tests/multi.sh`, which uses Docker Desktop's socket when the variable is not set |

With Docker Desktop the examples start their servers on TCP ports of the
Mac. With `ENGINE=kiri` they talk to a Kiri built with the patches of
[`kiri/`](../kiri/README.md), start the server on `~/.kiri/run/gpu.sock`
and on no port, and start their containers with a GPU request: the
`compose.gpu.yaml` next to each `compose.yaml` for examples 1 to 5, and
`--gpus all` for examples 6 and 7. The images must be in Kiri
(`ENGINE=kiri scripts/setup-mac.sh`). All seven were run with both engines
and gave the Mac's scores, checksums, losses and text with each.

The examples' MLX server keeps its log in `mlx/logs/server-example.log`, and
the PyTorch server in `pytorch/runs/server.log`. If an example is
interrupted, `SERVER_NAME=example mlx/server/ctl.sh stop` and
`pytorch/server.sh stop` stop what it left.

An example that finds a bridge server already listening on its port uses it
and leaves it running, and so does an MLX example in Kiri that finds an MLX
server on the socket. That is the case when the servers are
[kept running from login](../docs/setup.md#keep-the-servers-running-from-login-optional).
Example 7 starts servers in its own way and then needs a free port:
`MLX_PORT=47810 examples/07-custom-kernels/run.sh`.

The programs are ordinary: none imports anything of the bridge. The output
shown in the use-case pages is one run on one machine; expect your timings
to differ.
