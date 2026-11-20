# Changelog

## 0.2.0 (20 November 2026)

- Added an opt-in native Mac HTTP server for MLX-LM. Two queues share one
  model and one GPU stream. Short prompts get priority between blocks of
  up to 512 tokens. A long request gets a turn after at most four short
  turns when both queues have work.
- Added an HTTP check for streaming, client disconnects and sustained short
  traffic, and a repeatable burst comparison. See the [native server guide](docs/native-priority.md)
  for the command and measurements on an M4 Max.

## 0.1.2 (18 November 2026)

- The two Kiri patches are against Kiri's commit `69dc0b7`, its `main` of
  18 November 2026 (they were against `260cf66`). On a Kiri built from that
  commit the channel, the refusals, the diagnosis and the seven examples
  were run again.
- Kiri does not plan to include the channel (its issue 356, closed as not
  planned): the patch stays in this repository.

## 0.1.1 (16 November 2026)

- The two Kiri patches are against Kiri's commit `260cf66` (they were
  against `baa2d41`). On a Kiri built from that commit the channel, the
  installation, the diagnosis and one example per bridge were run again.

## 0.1.0 (16 November 2026)

The first release.

varco lets MLX and PyTorch programs in Linux containers on a Mac compute on
the Mac's Apple GPU, unchanged. It is an experimental prototype, written by
one person and measured on one machine (Apple M4 Max, 128 GB).

- MLX bridge: a remote gpu backend for MLX 0.32.4 on Linux, as a patch
  against the upstream tag, and a C++ server for the Mac. Protocol version
  11. The server checks every node of a graph against a rule for its
  primitive before it runs it, reads weight files itself under the
  directories it is started with, answers small reads ahead of its queue,
  and takes custom Metal kernels only when started with `--custom-kernels`.
- PyTorch bridge: an MPS backend for PyTorch 2.15.0 on Linux, and a server
  for the Mac with a process per program. Protocol version 3. The Mac end
  runs only the operators of a list made from PyTorch's own declarations.
- Kiri: two patches against `baa2d41`, the GPU channel for `--gpus`
  containers and a change to the memory balloon's headroom. With
  `ENGINE=kiri` the installation, the diagnosis, the examples and the
  tests that need one server run through that channel.
- Seven runnable examples, the tests of both bridges with comparisons by
  the SHA-256 of the results' bytes, and the measurements with their method.
- Installation in one command (`scripts/setup-mac.sh`), a diagnosis
  (`scripts/doctor.sh`), and optional launchd agents that keep the servers
  running from login (`scripts/autostart.sh`). The container images are
  built on the user's Mac.

The two wire protocols are described in
[docs/protocol-mlx.md](docs/protocol-mlx.md) and
[docs/protocol-pytorch.md](docs/protocol-pytorch.md), and what the release
runs and does not run in [docs/compatibility.md](docs/compatibility.md).
