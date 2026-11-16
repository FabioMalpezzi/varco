# Example 5: the same Compose file on a Mac and on an NVIDIA server

One service and one program that take the GPU they are given: `mps` through
the bridge on a Mac, `cuda` on an NVIDIA server.

```sh
./run.sh                 # Docker Desktop on a Mac: compose.yaml alone, no GPU request
ENGINE=kiri ./run.sh     # Kiri on a Mac: with compose.gpu.yaml, and the bridge's server on Kiri's socket
./run.sh gpu             # with compose.gpu.yaml and no server started: an NVIDIA server
```

For an NVIDIA server set `IMAGE` to a CUDA build of PyTorch (no NVIDIA
server was available to run it on). What stays the same and what does not:
[docs/use-cases/same-compose-nvidia.md](../../docs/use-cases/same-compose-nvidia.md).

| File | |
|---|---|
| `job.py` | The program: 300 training steps on whichever device PyTorch has |
| `compose.yaml` | The service |
| `compose.gpu.yaml` | The GPU request, as `--gpus all` sends it |
| `run.sh` | Runs the service, with the bridge's server where one is needed |
