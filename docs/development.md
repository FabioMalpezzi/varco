# Development

How to build both ends, where the code is, how the tests run, and how to
change the patches. The commands were run from a fresh clone on the machine
of the [benchmarks](benchmarks.md) with Docker Desktop, unless a line says
otherwise.

## Layout

```
mlx/
  versions.sh             the pinned MLX tag and commit, mlx-lm, mlx-embeddings
  patches/                the remote gpu backend, as a patch against MLX v0.32.4
  fetch-source.sh         clones upstream MLX at the tag into upstream/ and applies patches/
  setup-mac.sh            the Mac's Python environment (.venv) with the stock MLX wheel
  server/                 mlx_remote_server.cpp, build.sh, ctl.sh (start and stop)
  docker/                 Dockerfile (clones and patches MLX), Dockerfile.embeddings, Dockerfile.lm,
                          and the development container (Dockerfile.dev, dev-up.sh, dev-build.sh)
  build-images.sh         builds the three images
  tests/                  the test programs and run-all.sh, lm.sh, serve.sh, multi.sh, bench.sh
  results/reference/      the raw results the benchmarks cite
pytorch/
  versions.sh             which PyTorch (TORCH), which engine (ENGINE), names and ports
  bridge/                 both ends: native.cpp, client.py, install.py, auto.py (container);
                          server.py, session.py, policy.py, executor.cpp (Mac); wire.py (both);
                          ops/ (the operator lists for PyTorch 2.15.0)
  gen_ops.py              makes the operator lists of the Mac's PyTorch
  docker/                 Dockerfile and the packages of both ends
  build-mac.sh            the Mac's environment, the operator lists, executor.cpp
  build-image.sh          the image
  server.sh               start, stop and log of the Mac end
  in-container.sh         run a command in the image with tests/ and the working tree's bridge mounted
  compat.sh               the checks of both ends, then the two batteries on both sides, compared
  bench.sh, compare.sh, lifetime.sh
  tests/                  compat.py, compat_train.py, their comparator and inventories (expected/),
                          and the other test programs
kiri/
  patches/                two patches against Kiri 69dc0b7
  apply.sh                clones Kiri at that commit and applies them
examples/                 one directory per use case, each with a run.sh
scripts/
  setup-mac.sh            checks the Mac, builds what is missing or stale, tries each bridge
  doctor.sh               what is installed, what is running, what is wrong
  autostart.sh            launchd agents that keep the servers running from login
  lib.sh                  the names and functions those three share
  test-all.sh             mlx/tests/run-all.sh and pytorch/compat.sh
  fetch-models.sh         the small public models
  check.sh                what the CI runs: syntax, comparators, links, autostart, patches
  check-links.py          the links between the documents
  check-comparators.py    gives the tests' comparators made-up results that they must refuse
  check-autostart.sh      where autostart.sh makes the servers listen, for each engine
  tensor-diff.py          how far apart two runs kept with DUMP=directory are
```

What the scripts write is ignored by git: `mlx/.venv`, `mlx/upstream`,
`mlx/logs`, `mlx/results` (except `reference/`), `pytorch/.venv-mac-*`,
`pytorch/bridge/_build`, `pytorch/runs`,
`examples/04-pytorch-training/out` and the `__pycache__` directories.

## Building

```sh
scripts/setup-mac.sh            # checks, builds what is missing or stale, tests; see setup.md
scripts/setup-mac.sh --rebuild  # the images again from nothing; or: setup-mac.sh mlx | pytorch
```

`setup-mac.sh` decides whether an image is stale by a label it writes on it,
`varco.inputs`: a digest of the files the image is built from
(`mlx_inputs` and `torch_inputs` in `scripts/lib.sh`). An image built by
the bridges' own scripts has no label, and `setup-mac.sh` builds it again,
from Docker's cache. `ENGINE=kiri scripts/setup-mac.sh` builds in Kiri,
where the build scripts load each image from Kiri's builder and the label
is read from the image itself ([setup](setup.md#docker-desktop-or-kiri)).

Step by step, for the MLX bridge:

```sh
mlx/setup-mac.sh                # .venv with mlx 0.32.4, mlx-lm 0.32.2, mlx-embeddings 0.1.0
mlx/fetch-source.sh             # upstream/ = MLX v0.32.4 + patches/
mlx/server/build.sh             # server/mlx-remote-server, against the wheel's headers and library
mlx/build-images.sh             # varco-mlx, varco-mlx-embeddings, varco-mlx-lm
```

and for the PyTorch bridge, per version:

```sh
pytorch/build-mac.sh            # .venv-mac-2.15.0, the two lists of bridge/ops/, bridge/_build/2.15.0/
pytorch/build-image.sh          # varco-torch:2.15.0
```

`build-mac.sh` runs [`gen_ops.py`](../pytorch/gen_ops.py), which writes two
files for the Mac's PyTorch: `mps_ops-2.15.0.inc`, the operators that have
an MPS kernel, which the container's backend is built with, and
`allowed-2.15.0.txt`, the operators the Mac end runs for a container. Both
are kept in the repository. `TORCH=x.y.z` in front of the two commands
makes them for another PyTorch version, about which nothing was tested.

The MLX image compiles MLX for Linux inside the build; it took about a
minute and a half of compilation on the machine above.

## Testing

```sh
scripts/test-all.sh             # both bridges; 4 min 34 s to 4 min 49 s in three runs
```

**MLX**: `mlx/tests/run-all.sh [stage ...]` starts the server, runs each
stage natively and in a container, and compares.

| Stage | What it checks |
|---|---|
| `smoke` | one expression on both sides, on the GPU; then `tests/find_server.py` in the container: where the container end looks for the server (`MLX_REMOTE`, else `KIRI_GPU_BRIDGE`), and what it says when a connection is closed before the server's greeting |
| `coverage` | 58 small programs: the type, the shape and the SHA-256 of every array, the same on both sides |
| `edge` | custom functions, errors, odd shapes |
| `bge` | BGE-M3 embeddings: the same bytes, and timings |
| `bge-cpu` | the same embeddings on the container's CPU, set against the Mac's GPU as a report: the bytes differ, as they should (not in the default set; run it after `bge`) |
| `gen` | a synthetic generation loop |
| `load` | weights read by the Mac, the fallback, and what the server refuses to read |
| `kernels` | custom kernels with and without `--custom-kernels`, and what is refused |
| `urgent` | values read while a long evaluation is in flight, and what the server answers early |
| `graphs` | what the server refuses of a container's graphs and greetings, sent in the protocol itself to a server of the stage's own |
| `disconnects` | containers that leave in the middle of their work: the others are served, and the Mac has its memory back. Its output has one line that begins with `known limit:` and says that, while a container does not read an answer the server is writing, no other container is served |
| `upstream` | MLX's own Python tests on both sides (not in the default set; 1 min 4 s to 1 min 17 s). `SERVER_ARGS=--custom-kernels mlx/tests/run-all.sh upstream` runs them against a server that takes custom kernels, with the failures declared for such a server. MLX's `test_divmod` may fail on either side, and is then reported and not counted ([compatibility](compatibility.md#known-faults)) |

Every stage has an inventory in `mlx/tests/expected/`. `tests/compare.py`
fails when a test is missing on either side or is not listed, when a result
differs, and when a difference is not one of those declared in its source
with their reasons. `run-all.sh` goes on after a stage that fails, lists
what failed at the end, and exits with a status that is 0 only when every
stage passed.

With `ENGINE=kiri` in front, the server listens on Kiri's socket and the
containers are started with `--gpus all`. The default stages are then
`smoke`, `coverage`, `edge`, `bge` and `gen`, which took 22 s on the machine
above, and `bge-cpu` and `upstream` can be named after the script. The
stages `load`, `kernels`, `urgent`, `graphs` and `disconnects` speak to the
server from the Mac on a TCP port or start a second server, and with Kiri
one server listens on the socket alone: the script fails each of them with
`not a stage for ENGINE=kiri`. The five default stages were run in Kiri;
`bge-cpu` and `upstream` were run with Docker Desktop.

`mlx/tests/lm.sh MODEL` and `mlx/tests/serve.sh MODEL` run a language model
and mlx-lm's server on both sides. `mlx/tests/bench.sh` alternates the two
sides for timings and needs the development containers (below). `lm.sh`
takes `ENGINE=kiri`; `serve.sh` and `bench.sh` work with Docker Desktop
only.

**PyTorch**: `pytorch/compat.sh [checks|compat|compat_train]` runs three
parts, all by default. With `ENGINE=kiri` in front it runs them through
Kiri's channel, in 1 min 53 s on the machine above.

| Part | What it checks |
|---|---|
| `checks` | `tests/refusals.py`: what the Mac end refuses, sent in the protocol itself to a server of its own. `tests/client_checks.py`: what the container end does with the Mac's answers, on the CPU. `tests/fork_reading.py`, `tests/reconnect.py`, `tests/other_pytorch.py` and `tests/no_server.py`, in the container: a fork while a thread waits for the Mac, a program that goes on after the bridge gave a connection up, a container whose PyTorch is not the Mac's, which the server refuses with both versions named, and a connection closed before the server's greeting, which is what a container without a bridge meets in Kiri |
| `compat` | 47 inference programs on the Mac and in the container, compared |
| `compat_train` | 68 training programs, compared |

`tests/compat_compare.py` holds each program to the Mac's result in full:
the number it prints bit for bit, and the type, the shape and the SHA-256
of every tensor behind it. The programs that are not held to that are
declared in its source, each with its reason. `ONLY=name ./in-container.sh
python compat_train.py` runs the programs whose name contains `name`;
`TRACE=1` prints tracebacks. `pytorch/lifetime.sh` checks that the GPU
memory is given back however a program ends.

When a comparison fails, run the program on both sides with
`DUMP=directory` and give the two directories to `scripts/tensor-diff.py`:
it says how many elements differ and by how much.

**Without a GPU**: `scripts/check.sh` checks the syntax of the scripts, that
the comparators fail on results that are wrong, missing or not declared
(`scripts/check-comparators.py`), the links between the documents, where
`scripts/autostart.sh` makes the servers listen for each engine
(`scripts/check-autostart.sh`, with stand-ins for launchd and the servers;
on a Mac only), and that the patches apply to their upstream. That is all
the CI runs ([`.github/workflows/check.yml`](../.github/workflows/check.yml)):
a hosted runner has no Apple GPU, so no bridge is exercised there. A pull
request must come with the output of the tests run on a Mac.

## Working on the MLX backend

The container images are built from a clean clone, which is slow to iterate
on. For development there are two long-lived containers:

```sh
cd mlx
docker build -f docker/Dockerfile.dev -t varco-mlx-dev:latest docker
docker/dev-up.sh                                  # starts varco-mlx-dev and varco-mlx-test
docker exec varco-mlx-dev sh /docker/dev-build.sh # incremental build of ./upstream into ./site
BOX=varco-mlx-test tests/run-all.sh coverage      # the tests against that build
docker/dev-up.sh down
```

`varco-mlx-dev` builds `mlx/upstream` incrementally; `varco-mlx-test` is the
final image with that build ahead of its own MLX.

## Working on the PyTorch bridge

`pytorch/in-container.sh` mounts the working tree's `bridge/` over the
image's, so a change to the Python needs no rebuild. A change to
`native.cpp` needs `./build-image.sh`; a change to `executor.cpp` needs
`./build-mac.sh`. `python -m bridge PROGRAM` runs a program and prints, at
the end, how many operations took the fast path and what left it;
`BRIDGE_DEBUG=1` says which.

## Changing the patches

The upstream sources are not in this repository. A change to MLX or to Kiri
is a change to a patch.

**MLX.** Edit the tree in `mlx/upstream` (it is a git clone with the patch
applied and staged), then regenerate the patch from it:

```sh
cd mlx/upstream
git add -A
git diff --cached v0.32.4 > ../patches/0001-remote-gpu-backend.patch
```

Then check that it applies to a clean clone (`scripts/check.sh`), rebuild
the server if `protocol.h` changed (`mlx/server/build.sh`) and the images
(`mlx/build-images.sh`), and run `mlx/tests/run-all.sh`. A change to the
wire format changes `kProtocolVersion` in `protocol.h` and
[protocol-mlx.md](protocol-mlx.md).

To move to another MLX version: change `mlx/versions.sh`, rebase the patch
on the new tag, and expect the Mac's wheel, the server and both
Dockerfiles' defaults to follow. This was not done here.

**Kiri.** `kiri/apply.sh DIR` gives a clone of Kiri at the pinned commit
with both patches staged. Edit there and regenerate each patch with
`git diff --cached` over the files it covers; the two patches touch
`crates/kiri/src/daemon/engine.rs` in different places and must keep
applying each alone and both together (`scripts/check.sh` checks that).
The unit tests the patches add or change are named in
[`kiri/README.md`](../kiri/README.md#what-was-verified-and-what-was-not).

## Conventions

- Scripts are POSIX `sh`, print in English, say what is missing and which
  command provides it, and stop servers by the pid they kept.
- Existing identifiers are kept: `MLX_REMOTE`, `mlx-remote-server`,
  `BRIDGE_*`, `KIRI_GPU_BRIDGE`. The repository's name appears in image
  names and in prose only.
- A figure in a document names the program that produced it, and says how
  many runs it is.
