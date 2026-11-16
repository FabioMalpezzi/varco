# Example 1: embeddings in a container

BGE-M3 sentence embeddings with mlx-embeddings, in a container, computed on
the Mac's GPU through the MLX bridge.

```sh
./run.sh            # in a container
./run.sh native     # the same program on the Mac, to compare
```

Needs `mlx-community/bge-m3-mlx-fp16` under `MODELS`
(`scripts/fetch-models.sh`). What it shows, the measured figures and the
limits: [docs/use-cases/embeddings.md](../../docs/use-cases/embeddings.md).

| File | |
|---|---|
| `embed.py` | The program: an ordinary mlx-embeddings program |
| `compose.yaml` | The service: the image, the models read-only, the program |
| `compose.gpu.yaml` | The GPU request, which `run.sh` adds with `ENGINE=kiri` |
| `run.sh` | Starts the bridge's server, runs the service, stops the server |
