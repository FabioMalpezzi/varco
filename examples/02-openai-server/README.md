# Example 2: an OpenAI-compatible model server in a container

mlx-lm's own server (`python -m mlx_lm server`), unchanged, in a container,
computing on the Mac's GPU through the MLX bridge, and a client that sends
it one request, a streamed request, a long prompt three times (the prompt
cache) and four requests at once.

```sh
./run.sh                                   # the small model, Qwen/Qwen3.5-0.8B-Base
./run.sh Qwen/Qwen3.5-0.8B-Base --keep     # leave it running on http://127.0.0.1:18080
./run.sh --stop
KERNELS=1 ./run.sh mlx-community/Qwen3.6-35B-A3B-4bit   # the large model of the measurements
```

`KERNELS=1`, and no other value, starts the bridge's server with
`--custom-kernels`, which the large model needs for the Mac's text and speed
([docs/use-cases/custom-kernels.md](../../docs/use-cases/custom-kernels.md)).
`LLM_PORT` changes the port. What it shows, the measured figures and the
limits: [docs/use-cases/openai-server.md](../../docs/use-cases/openai-server.md).

| File | |
|---|---|
| `compose.yaml` | The service: mlx-lm's server in the image `varco-mlx-lm` |
| `compose.gpu.yaml` | The GPU request, which `run.sh` adds with `ENGINE=kiri` |
| `client.py` | The client, standard library only |
| `run.sh` | Starts the bridge's server and the service, runs the client, stops both |
