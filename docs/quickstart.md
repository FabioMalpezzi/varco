# Quick start

From nothing to an MLX program and a PyTorch program running in a
container on the Mac's GPU. There are two paths:
[Docker Desktop](#docker-desktop), which is what most Macs have, and
[Kiri](#kiri), a container engine for which this repository carries two
patches that give `--gpus all` a meaning on a Mac.

Every command on this page was run from a fresh clone of this repository
on 16 November 2026, on an Apple M4 Max with Docker Desktop 29.2.1 and with
a Kiri built from a fresh clone with the patches.

## Docker Desktop

### 1. Install

```sh
git clone https://github.com/FabioMalpezzi/varco.git && cd varco
scripts/setup-mac.sh
```

The script checks what the Mac needs, builds both ends of both bridges and
runs one computation through each. It took about four minutes on the
machine above. What it needs, what it builds, how to install one bridge
only, and how to update or remove it are in [setup.md](setup.md).

### 2. Get the models

```sh
export MODELS="$HOME/models"     # or another directory of yours
scripts/fetch-models.sh
```

It downloads four small public models, about 3.3 GB, with the `hf` command
(`uv tool install huggingface_hub`): `mlx-community/bge-m3-mlx-fp16` and
`Qwen/Qwen3.5-0.8B-Base` into the directory `MODELS` names, and
`sentence-transformers/all-MiniLM-L6-v2` and `openai-community/gpt2` into
the Hugging Face cache. The scripts use `$HOME/models` when `MODELS` is not
set; the commands typed by hand further down this page read the variable,
so set it in every shell you run them in. With `MODELS` set to an empty
directory, a fast connection and three of the four models already in the
Hugging Face cache, it took 19 seconds on the machine above.

### 3. Run an MLX program in a container

```sh
examples/01-embeddings/run.sh
```

You should see the server start, two lines from Compose about the container
it creates, then the program's output from inside the container:

```
mlx-remote-server: mlx 0.32.4, listening on 127.0.0.1:47800
reads the files of weights under /path/to/your/models (/models in a container)
device Device(gpu, 0); model loaded in 1.09 s
query: How do I run a Linux container on a Mac?
  0.8739  On macOS, containers run inside a lightweight Linux virtual machine.
  ...
checksum of the embeddings: 118.969635
```

`Device(gpu, 0)` inside a Linux container is the bridge. Run the same
program on the Mac and compare the scores and the checksum:

```sh
examples/01-embeddings/run.sh native
```

### 4. Run a PyTorch program in a container

```sh
examples/03-pytorch-inference/run.sh
```

```
bridge server: torch 2.15.0, MPS on 127.0.0.1:47700, a process for each program
torch 2.15.0, device mps
query: How do I run a Linux container on a Mac?
  0.7716  On macOS, containers run inside a lightweight Linux virtual machine.
  ...
generated 100 tokens in 0.26 s, 379 tokens per second
```

`device mps` in a Linux container is the bridge. `run.sh native` runs it on
the Mac, and `run.sh cpu` runs the same container with the bridge turned off
(`BRIDGE=0`), which shows what the container does without it.

### 5. Do it by hand

The examples hide two steps you will want for your own programs: starting
the server, and running a container that points at it.

MLX:

```sh
mlx/server/ctl.sh start --map "/models=$MODELS"
docker run --rm -e MLX_REMOTE=host.docker.internal:47800 \
  -v "$MODELS:/models:ro" varco-mlx-lm:latest \
  python -m mlx_lm generate --model /models/Qwen/Qwen3.5-0.8B-Base --prompt "Hello" --max-tokens 30
mlx/server/ctl.sh stop
```

`--map "/models=$MODELS"` lets the server read, by itself, the weight
files that containers see under `/models`. Without it the container reads
them and uploads them, which works and is slower.
`host.docker.internal:47800` is where the container looks when
`MLX_REMOTE` is not set, so the variable is shown here only to be explicit.

PyTorch:

```sh
pytorch/server.sh start
docker run --rm -v "$PWD/examples/05-same-compose-nvidia:/app:ro" varco-torch:2.15.0 python /app/job.py
pytorch/server.sh stop
```

Nothing is passed to the container: the bridge in the image connects to
`host.docker.internal:47700` when the program first touches `"mps"`.

## Kiri

[Kiri](https://github.com/libracoder/kiri) is a container engine for macOS
that speaks the Docker API. Upstream Kiri refuses `--gpus`. With the
patches in [`kiri/`](../kiri/README.md), on an Apple-silicon Mac a container
started with `--gpus all` gets its end of the bridge at `127.0.0.1:47700` in
its own network, and nothing has to be configured in the container.

1. Build Kiri with the patches and start its daemon:
   [`kiri/README.md`](../kiri/README.md). The build took 211 seconds on the
   machine above, and 107 seconds a second time, with the kernel's build in
   Docker's cache.

2. Put the images into Kiri, which has its own image store. Build them
   there, which took 207 and 237 seconds in two runs on the machine above,
   with the Mac ends already built and Kiri's virtual machine at 8 CPUs and
   16 GiB, and 331 seconds from a fresh clone with Kiri's default virtual
   machine of 2 CPUs and 8 GiB:

   ```sh
   ENGINE=kiri scripts/setup-mac.sh
   ```

   Or copy the ones Docker Desktop built, which took 3 seconds for these
   two:

   ```sh
   docker save varco-mlx-lm:latest varco-torch:2.15.0 | DOCKER_HOST=unix://$HOME/.kiri/run/docker.sock docker load
   ```

3. Start one bridge's server on the socket Kiri's channel leads to. There is
   one socket, so one bridge is reachable at a time. To change bridge, stop
   the server that is there and start the other.

   ```sh
   # MLX
   mlx/server/ctl.sh start --unix ~/.kiri/run/gpu.sock --map "/models=$MODELS"
   # or PyTorch
   ENGINE=kiri pytorch/server.sh start
   ```

4. Run a container with a GPU request:

   ```sh
   export DOCKER_HOST=unix://$HOME/.kiri/run/docker.sock
   # MLX
   docker run --rm --gpus all -v "$MODELS:/models:ro" varco-mlx-lm:latest \
     python -m mlx_lm generate --model /models/Qwen/Qwen3.5-0.8B-Base --prompt "Hello" --max-tokens 30
   # PyTorch
   docker run --rm --gpus all -v "$PWD/examples/05-same-compose-nvidia:/app:ro" varco-torch:2.15.0 python /app/job.py
   ```

   A container started with `--gpus all` has
   `KIRI_GPU_BRIDGE=127.0.0.1:47700` in its environment and a listener at
   that address. Both bridges read the variable, so the container is given
   no address. A container started without `--gpus` has neither: its
   connection to `127.0.0.1:47700` is refused, and Kiri closes at once a
   connection to a port of `host.docker.internal` on which nothing listens
   on the Mac. A program in it that asks for the GPU is told that no server
   answered and that the container must be started with `--gpus all`
   ([troubleshooting](troubleshooting.md#in-the-container)), and in the
   PyTorch image `torch.backends.mps.is_available()` is `False`.

With `ENGINE=kiri` in front, the scripts do steps 3 and 4 themselves: they
start the server on the socket and their containers with `--gpus all`. The
seven examples were run that way (`ENGINE=kiri examples/01-embeddings/run.sh`),
as were `scripts/doctor.sh`, `scripts/autostart.sh`, `pytorch/compat.sh`,
`mlx/tests/lm.sh` and, for the stages that need one server,
`mlx/tests/run-all.sh` ([development](development.md#testing)).

## Check the whole installation (optional)

The steps above are the installation. The test collection goes further: it
runs every test program natively and in a container and compares the
results.

```sh
scripts/test-all.sh
```

With Docker Desktop it took 4 min 34 s, 4 min 42 s and 4 min 49 s in three
runs on the machine above. See [benchmarks.md](benchmarks.md#running-them-again)
for what each stage is, and [development.md](development.md#testing) for
the rest, the stages that run in Kiri included.

## Next

- The seven [use cases](../README.md#use-cases), each with an example.
- `scripts/doctor.sh` and [troubleshooting](troubleshooting.md) if something
  above did not print what this page says.
- [Keeping the servers running from login](setup.md#keep-the-servers-running-from-login-optional),
  so that a container finds its bridge without starting anything first.
- [Security](security.md): what the Mac end checks, and what is left to
  you.
- [Custom Metal kernels](use-cases/custom-kernels.md), if a library you run
  has its own.
