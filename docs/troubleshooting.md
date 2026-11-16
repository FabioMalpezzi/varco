# Troubleshooting

Messages you may meet, what they mean, and what to do. The messages in
`code` are the ones the scripts and the bridges print.

Start with `scripts/doctor.sh`. It looks at the Mac, the engine, both ends
of each bridge and the servers that are running, and prints the remedy
under each fault it finds ([setup](setup.md#check-an-installation)).

## Building

**`uv is not installed`** (from `mlx/setup-mac.sh` or `pytorch/build-mac.sh`).
Install [uv](https://docs.astral.sh/uv/), or make the environment yourself:
the script names the packages.

**`no upstream/mlx/backend/remote/protocol.h: run ./fetch-source.sh first`**
(from `mlx/server/build.sh`). The server is built against one header of the
patched MLX tree. `mlx/fetch-source.sh` clones upstream MLX and applies the
patch.

**`the tag v0.32.4 is at ..., not at the pinned ...`**. Upstream moved the
tag, or the clone is not upstream's. The patch is against one commit and is
not applied to another.

**`ld: warning: building for macOS-..., but linking with dylib ... built for newer version`**
while building the MLX server. A warning from the linker about the MLX
wheel; the server works.

**`no bridge/ops/mps_ops-X.inc: run TORCH=X ./build-mac.sh first`**. The
PyTorch image needs the list of operators that PyTorch version runs on a
Mac's GPU. The lists for 2.15.0 are in the repository; another version's are
made on the Mac by `build-mac.sh`, and nothing about another version was
tested.

**`the image ... is not built: run scripts/setup-mac.sh ...`** (from an
example). The message names the command to run.

**The image build cannot reach the network.** The MLX image clones MLX from
GitHub and its build fetches MLX's own dependencies; both images install
from PyPI.

**`failed to solve: varco-mlx:latest: ... pull access denied`** from a
`docker build` in Kiri of an image that begins `FROM varco-mlx`. Kiri's
builder runs in a container and keeps what it builds in its own cache, so
it does not find an image by its name in Kiri. For the repository's images
`mlx/build-images.sh` passes `--load` and gives the base image to the build
as an OCI layout (`--build-context`). An image of your own can be built
with Docker Desktop and copied into Kiri with `docker save | docker load`,
as the repository's can ([setup](setup.md#docker-desktop-or-kiri)).

## Starting a server

**`port 47800 is already taken by process N (...)`** from
`mlx/server/ctl.sh start`, followed in an example by
`the MLX server did not start: see mlx/logs/server-example.log`. Something
else is listening on the port, and the message names it. If it is another
MLX server, stop it (`mlx/server/ctl.sh stop`,
`SERVER_NAME=example mlx/server/ctl.sh stop` for one an example left, or
`scripts/autostart.sh uninstall mlx` for the one kept from login). Otherwise
move the bridge: set `MLX_PORT` for the examples and `PORT` for the tests.

**`this example starts a server of its own, and an MLX server is already
listening on port 47800`**. The example needs a server with its own options
and will not replace one it did not start. Run it on another port, as the
message says: `MLX_PORT=47810 examples/07-custom-kernels/run.sh`.

**`port 47700 is held by a bridge server for another PyTorch`** from
`pytorch/server.sh start`. A server for one PyTorch version is running and
the script was asked for another. Stop the first
(`TORCH=X pytorch/server.sh stop`, or `scripts/autostart.sh uninstall pytorch`).

**`the socket ... is held by process N (...)`** from `mlx/server/ctl.sh
start --unix` or `ENGINE=kiri pytorch/server.sh start`. With Kiri both
bridges' servers listen on the one socket the channel leads to, and a
server does not take the place of one that is running. Stop the process the
message names, then start the bridge you want. `pytorch/server.sh start`
does not say this of a PyTorch server of the same version: it uses that
one, as it does on the port.

**`with ENGINE=kiri, name the one bridge to keep running`** from
`scripts/autostart.sh install`. One socket has one server: run
`ENGINE=kiri scripts/autostart.sh install mlx`, or `install pytorch`.

**`not a stage for ENGINE=kiri`** from `mlx/tests/run-all.sh`. The stages
`load`, `kernels`, `urgent`, `graphs` and `disconnects` need a TCP port or
a second server. Run them with Docker Desktop
([development](development.md#testing)).

**`FAULT a server of the ... bridge listens on TCP port ...`** from
`ENGINE=kiri scripts/doctor.sh`. With Kiri every container reaches a TCP
port of the Mac, started with `--gpus` or not. Stop that server with the
command on the line under the fault, and start it on the socket
([setup](setup.md#start-a-server-by-hand)).

**`--models and --map take absolute paths`**. Give the server full paths.

**`no .venv-mac-X: run TORCH=X ./build-mac.sh first`** (from
`pytorch/server.sh`). The Mac end is built for one PyTorch version, 2.15.0
unless `TORCH` names another.

**The PyTorch server is running and the program still runs on the CPU.**
Look at `pytorch/runs/server.log`. A line that begins
`bridge server: refused job.py (...)` gives the reason: the container's
PyTorch is not the server's, or the two ends speak different versions of
the protocol. Build the image and the Mac end from the same state of the
repository (`scripts/setup-mac.sh pytorch`), and restart the server.

## In the container

**`cannot connect to the MLX server on the Mac at host.docker.internal:47800 (...). Start mlx-remote-server there, or set MLX_REMOTE=host:port.`**
With Docker Desktop, no server is listening where the container looks.
Start it (`mlx/server/ctl.sh start --map /models=$HOME/models`), or give
the container the server's port with `MLX_REMOTE`.

**`no MLX server answered at host.docker.internal:47800: the connection was closed before the server's greeting arrived. With Kiri, the container must be started with --gpus all (...)`**
from an MLX program, or **`no bridge server answered at host.docker.internal:47700: ...`**
with the same ending from a PyTorch program. In Kiri the container was
started without a GPU request and has no bridge: Kiri accepts its
connection to any port of `host.docker.internal` and closes it at once when
nothing listens on that port of the Mac. Start the container with
`--gpus all`, or with `capabilities: [gpu]` in Compose. With another
engine, the server's log on the Mac says why it closed the connection
(`mlx/logs/server.log`, `pytorch/runs/server.log`).

**`no MLX server answered at 127.0.0.1:47700: ... This container was started with --gpus in Kiri (KIRI_GPU_BRIDGE is set), and Kiri's channel leads to a socket on the Mac: is the MLX server listening there (mlx/server/ctl.sh start --unix ~/.kiri/run/gpu.sock)?`**,
or **`no bridge server answered at 127.0.0.1:47700: ... is the bridge's server listening there (ENGINE=kiri pytorch/server.sh start)?`**
The container has Kiri's channel, and the server of its bridge is not on
the socket. Start it with the command in the message. The socket has one
server, and `ENGINE=kiri scripts/doctor.sh` names the bridge whose server
is there.

**`torch.backends.mps.is_available()` is `False` in the PyTorch image.**
The bridge answers that question by reaching the Mac, so it is `False`
whenever it cannot: no server is running, the server is of another PyTorch
version, `BRIDGE=0` is set, or, in Kiri, the container was started without
`--gpus`. A program written to fall back then runs on the CPU without a
word: print the device. A program that asks for `"mps"` regardless gets the
error of its case. With Docker Desktop and no server that is
`the bridge cannot reach the Mac's GPU at host.docker.internal:47700 (...): is python -m bridge.server running on the Mac?`,
and the remedy is `pytorch/server.sh start`. In Kiri it is one of the two
entries above.

**`the container has PyTorch X, the Mac's bridge server Y: they must be the same. The image and the Mac end have to be built for the same version, and PyTorch 2.15.0 is the one this repository is built and tested for`**,
from a program's first operation on `"mps"`. The image and the Mac end were
built for different PyTorch versions. `scripts/setup-mac.sh pytorch` builds
both for 2.15.0; restart the server after it.

**`mx.metal.is_available()` is `False`, or `Custom Metal kernels are off:
the MLX server on the Mac was started without --custom-kernels.`** The
server does not take custom kernels unless it is started with the option.
[use-cases/custom-kernels.md](use-cases/custom-kernels.md) says what the
option gives, what it implies, and the commands that turn it on and off.

**`malformed graph: ...`**, raised in the container as an MLX error. The
server checks every node of a graph before it runs it, and refuses one that
MLX's own operations would not have made. Two such nodes come from ordinary
programs: a `pad` whose value is an array with more than one element, and a
bitwise operation under `vmap` whose inputs are not broadcast to one shape
([compatibility](compatibility.md#not-supported-and-open-measurements)).

**`protocol version N not supported, this server speaks 11`**, or **`the
container runs MLX X, this server Y: the two must be the same version`**,
when the container connects. The image and the server are from different
states of the repository, or their MLX versions differ. Each end refuses
the other. Run `scripts/setup-mac.sh mlx`, and restart the server.

**`[remote gpu] the Mac does not read FILE itself (...): the file ... read
here and their arrays uploaded.`** This is not an error. The server could
not or may not read that weight file (no `--map` covers the path, the file
is compressed, it changed), so the container read it and sent the arrays.
The result is the same; loading is slower and passes through the virtual
machine's memory. To have the Mac read it, start the server with a `--map`
that covers the directory as the container sees it: `--map
/models=$HOME/models` for a container that mounts `$HOME/models` at
`/models`.

**`NotImplementedError` for an operator on MPS.** The Mac's PyTorch has no
MPS kernel for it, and neither has the container: the error is the Mac's
own. `PYTORCH_ENABLE_MPS_FALLBACK=1` in the container's environment is
passed to the Mac's process, as on a Mac.

**`could not select device driver "nvidia" with capabilities: [[gpu]]`**.
Docker Desktop on a Mac has no GPU to give. Leave the GPU request out there
([use-cases/same-compose-nvidia.md](use-cases/same-compose-nvidia.md)).

**A data loader with workers fails or hangs.** Give the container shared
memory: `--shm-size=2g`, or `shm_size: 2g` in Compose.

**`port is already allocated`** when starting the model server example. Port
18080 of the Mac is taken: set `LLM_PORT`.

## Results and speed

**The text differs from the Mac's.** For a language model with mlx-lm, most
likely the server runs without `--custom-kernels` and mlx-lm took its other
path ([custom kernels](use-cases/custom-kernels.md#who-needs-them-and-how-you-notice-they-are-off)).
With requests sent at once to a model server, some difference is expected
and unexplained ([compatibility](compatibility.md#known-faults)).

**Training losses differ after the first step.** They may differ between
two runs on the Mac too: check that first
([compatibility](compatibility.md#what-fails-on-the-mac-fails-in-the-container)).

**It is slower than on the Mac.** Expected where steps are short and each
reads a value, and on Docker Desktop more than in Kiri. Check also that the
weights are read by the Mac (no "does not read" message), that the two ends
are built from the same state of the repository, and that nothing else is
using the GPU: a second program on the same GPU slows a small request 5 to
8 times.

**Timings jump around.** The GPU slows as it warms, by more than most
differences worth measuring. Compare in alternated runs
([benchmarks](benchmarks.md#method)).

**One generation in six is slow, in Kiri.** The memory balloon stall that
[`0002-memory-headroom-after-shortage.patch`](../kiri/patches/0002-memory-headroom-after-shortage.patch)
is for ([benchmarks](benchmarks.md#a-stall-in-kiri-and-its-fix)).

## Cleaning up

```sh
scripts/autostart.sh uninstall                  # the servers kept running from login, if installed
mlx/server/ctl.sh stop                          # the MLX server started by hand or by the tests
SERVER_NAME=example mlx/server/ctl.sh stop      # the one an example started
pytorch/server.sh stop                          # the PyTorch server
docker image rm varco-mlx varco-mlx-embeddings varco-mlx-lm varco-torch:2.15.0
```

The servers are stopped by the pid their scripts kept, and nothing else is
touched. `mlx/logs/`, `mlx/results/` (except `reference/`) and
`pytorch/runs/` hold what the scripts wrote and can be deleted. What a full
removal takes is in [setup](setup.md#uninstall).
