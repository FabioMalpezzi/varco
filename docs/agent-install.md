# Installing varco: instructions for a coding agent

A person asked you to install varco on their Mac. Work from the root of a
clone of this repository. Each step below ends on the result that shows it
is done: go on only when you have seen it. When a step fails, run
`scripts/doctor.sh`, find the message it prints in
[troubleshooting](troubleshooting.md), apply the remedy given there, and
repeat the step. If the same step fails twice, stop and show the person the
command and its output.

What each command does, and why, is in [setup](setup.md) and in the pages
the steps link to. Read a section of them when a step sends you there.

## 1. Ask the person

Four choices are theirs. Ask all four before you build anything, and keep to
the answers. The value in brackets is what to use when the person says
"the defaults".

1. **The container engine** [Docker Desktop]. With Docker Desktop both
   bridges work at once, and every container and local process can reach a
   bridge server while one runs. Kiri has to be built from source with this
   repository's patches (step 2); with it only a container started with
   `--gpus all` reaches a bridge, and one bridge is reachable at a time.
2. **The directory of models** [`$HOME/models`]. The MLX server reads weight
   files under this directory and nowhere else.
3. **Custom Metal kernels** [off]. Some libraries run GPU programs of their
   own; with the option on, the containers that share an MLX server must
   trust each other. What the person gains and gives up is in
   [custom kernels](use-cases/custom-kernels.md): show them its first
   section if they are unsure.
4. **Servers running from login** [no]. Without it each example starts and
   stops the server it needs, and the person starts a server by hand for
   their own containers.

Then set the directory of models. The scripts read it from the environment,
and without it they use `$HOME/models`:

```sh
export MODELS=<the directory the person chose>
mkdir -p "$MODELS"     # the MLX server does not start on a directory that is not there
```

`MODELS`, and with Kiri the `ENGINE=kiri` of step 2, must be set in every
shell a command of this file runs in. If your shell does not keep variables
from one command to the next, export them again at the start of each
command.

Tell the person what will be downloaded, and ask before each of these:
step 3 fetches Python packages into `uv`'s cache and base images into the
engine; step 5 fetches one model (1.1 GB). `setup-mac.sh` checks the
Mac's tools first, and when one is missing it builds nothing: it prints a
line `MISSING` with the command that provides the tool, and stops. Ask
before you run such a command.

## 2. Kiri only: build Kiri with the patches

Skip this step for Docker Desktop. It needs Rust (`cargo`) and, for
`cargo xtask kernel` and `cargo xtask disk`, a Docker engine that is
running. What each command builds is in
[`kiri/README.md`](../kiri/README.md#applying-and-building).

The patched daemon uses the same sockets under `~/.kiri/run` as any other
Kiri, so look first for one that is running. The `grep` itself is one of
the lines printed:

```sh
ps -axo pid,command | grep "kiri daemon"
```

If there is one, ask the person whether you may stop it, and build while it
runs:

```sh
kiri/apply.sh
cd kiri/kiri-src
cargo xtask kernel && cargo xtask disk && cargo xtask guest && cargo xtask host
```

The four build commands took 211 seconds on an Apple M4 Max. A second build
on the same machine took 107 seconds, because Docker had the kernel's build
in its cache: the kernel took 28 seconds where the first build took 123.

When the build is done and the person has agreed, stop the daemon that was
running by its pid (`kill PID`). Then start the new one:

```sh
KIRI_DATA_DIR=$PWD/target/gpu-data target/release/kiri daemon    # stays in the foreground: run it in a terminal of its own
```

Done when, from another terminal,
`DOCKER_HOST=unix://$HOME/.kiri/run/docker.sock docker version` prints a
`Server` section that reads `Server: Kiri`. The build commands were run in `kiri/kiri-src`: go back to
the root of the clone, where every step below runs, and tell the scripts of
this repository to use Kiri:

```sh
cd ../..
export ENGINE=kiri
```

## 3. Build both bridges

What the script needs, and what it does when something is missing, is in
[setup](setup.md#what-you-need).

```sh
scripts/setup-mac.sh
```

The images are built on this Mac. With none there yet, on an Apple M4 Max,
that took 244 seconds with Docker Desktop. In Kiri it took 331 seconds with
Kiri's default virtual machine of 2 CPUs and 8 GiB, 237 of them for the
three MLX images, and 207 seconds with the virtual machine at 8 CPUs and 16
GiB. With the images already built from the same files it takes seconds.

Done when the output has a line `== Ready, in N s` and, under it, one line
for the MLX bridge and one for the PyTorch bridge, each saying
`a container computed on this Mac's GPU`. After them come a line about the
servers, which begins `No server is left running` unless one was running
before the script started, and a block `Next:`.

## 4. Check the installation

```sh
scripts/doctor.sh
```

Done when the exit status is 0 and the summary says `No fault found`. Lines
that begin with `off` are expected here: no server is running yet.

## 5. Run one program through each bridge

```sh
hf download mlx-community/bge-m3-mlx-fp16 --local-dir "$MODELS/mlx-community/bge-m3-mlx-fp16"   # BGE-M3, 1.1 GB
examples/01-embeddings/run.sh               # MLX
examples/05-same-compose-nvidia/run.sh      # PyTorch; needs no model
```

The download needs the `hf` command (`uv tool install huggingface_hub`).
Each example starts the server it needs and stops it. Done when the first
prints `device Device(gpu, 0)` and `checksum of the embeddings: 118.969635`,
and the second prints `torch 2.15.0, device mps` and a line that ends with
`on mps; accuracy 1.000`. The other examples use three more models:
`scripts/fetch-models.sh` downloads all four, 3.3 GB, when the person wants
to run them.

If the person declines the download, leave the first example out and check
the MLX bridge with a server and the diagnosis instead:
`mlx/server/ctl.sh start --map "/models=$MODELS"` (with Kiri:
`--unix "$HOME/.kiri/run/gpu.sock"` as well), `scripts/doctor.sh mlx`,
`mlx/server/ctl.sh stop`. Done when `doctor.sh` has a line that says a
container computed through the server what the Mac gives.

With Kiri, check as well that a container that was not given the channel
does not have it:

```sh
DOCKER_HOST=unix://$HOME/.kiri/run/docker.sock docker run --rm varco-torch:2.15.0 python -c "import torch; print(torch.backends.mps.is_available())"
```

Done when it prints `False`.

## 6. Only what the person chose in step 1

Custom kernels on, and no server running from login: start the MLX server
by hand with the option
([custom kernels](use-cases/custom-kernels.md#turning-them-on-and-off)).
`mlx/server/ctl.sh` does not read `ENGINE`, so with Kiri the command names
the socket:

```sh
# Docker Desktop
mlx/server/ctl.sh start --map "/models=$MODELS" --custom-kernels
# Kiri
mlx/server/ctl.sh start --unix "$HOME/.kiri/run/gpu.sock" --map "/models=$MODELS" --custom-kernels
```

Servers running from login
([setup](setup.md#keep-the-servers-running-from-login-optional)). If the
person chose custom kernels as well, start no server by hand: the option is
given to `autostart.sh` with `KERNELS=1` in front of `install`, and the MLX
server it keeps is started with it. With Kiri one bridge can be kept
running, so ask which, and name it: `install mlx` or `install pytorch`.

```sh
scripts/autostart.sh install               # kernels off
KERNELS=1 scripts/autostart.sh install     # kernels on
scripts/autostart.sh status
```

Done when `scripts/doctor.sh` exits with status 0 and its lines for the
servers show what the person chose: `off` for a server left stopped, `ok`
for one that was asked to run. With Kiri the bridge whose server is not on
the socket reports `off`. With kernels on, the MLX server's line says
`custom Metal kernels ON`.

## 7. Report

Tell the person, in this order:

1. the engine and the choices you applied;
2. the last lines of `setup-mac.sh`;
3. the output of `doctor.sh`;
4. the lines of step 5;
5. whether a server is running now and how to stop it
   (`mlx/server/ctl.sh stop`, `pytorch/server.sh stop`,
   `scripts/autostart.sh uninstall`);
6. with Kiri, what is left on the Mac: the daemon, which is still running
   and is stopped by its pid, and buildx's builder container
   `buildx_buildkit_default`, which stays in Kiri after the build. If you
   replaced a Kiri that was running, say that its images do not appear in
   the new daemon and are still in the earlier one's data directory;
7. anything that did not print what this file says, with its output.

## What stays as it is unless the person asks

- Custom kernels stay off and nothing is installed to start at login.
- Images, containers and model files that were on the Mac before you
  started are kept. To remove varco, [setup](setup.md#uninstall) has the
  commands; they touch only what varco made.
- The patches under `mlx/patches/` and `kiri/patches/`, and the pinned
  versions in the `versions.sh` files, are used as they are. If one does not
  apply or a version is not available, report it.
- Servers are stopped with the repository's own commands, by the pid files
  they keep, and never by matching process names.
