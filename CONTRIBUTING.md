# Contributing

This is an experimental prototype measured on one machine, so the most
useful contributions are of three kinds:

- **Measurements on other hardware.** Run `scripts/test-all.sh` and any of
  the examples on your Mac and open an issue with the output, the chip, the
  memory and the macOS version. A result that disagrees with
  [docs/benchmarks.md](docs/benchmarks.md) is as welcome as one that agrees.
- **Programs that do not work.** An ordinary MLX or PyTorch program that
  gives another result in the container than on the Mac, or fails only
  there, is a bug. The smaller the program, the better.
- **Fixes** to what [docs/roadmap.md](docs/roadmap.md) lists under solidity.

## Before you open a pull request

1. Read [docs/development.md](docs/development.md): how to build both ends,
   where the code is, and how the tests are run.
2. Run `scripts/check.sh`. It is what the CI runs: syntax of the scripts,
   the tests' comparators, links between the documents, where autostart
   makes the servers listen (on a Mac), and the patches against their
   upstream.
3. Run the tests of the bridge you changed on a Mac (`scripts/test-all.sh mlx`
   or `scripts/test-all.sh pytorch`) and paste their summary in the pull
   request. The CI has no Apple GPU and cannot run them for you.
4. A change of speed needs a measurement taken as the others were: the same
   program on both sides, in mirrored order (Mac, container, container, Mac),
   compared pair by pair. See the method in
   [docs/benchmarks.md](docs/benchmarks.md#method).

## Conventions

- Plain, full sentences in comments, messages and documents, in English. No
  promotional tone. Say what was measured and what was not.
- A number in a document names the program that produced it.
- A change to MLX goes into `mlx/patches/` as a patch against the pinned
  tag; a change to Kiri goes into `kiri/patches/`. The upstream sources are
  not copied into this repository. See
  [docs/development.md](docs/development.md#changing-the-patches).
- The two ends of a bridge must agree on the protocol: a change to a wire
  format changes its version (`kProtocolVersion` in MLX's `protocol.h`,
  `PROTOCOL` in `pytorch/bridge/wire.py`) and its page under `docs/`.
- A test that is added or removed changes its inventory
  (`mlx/tests/expected/`, `pytorch/tests/expected/`), and a result that is
  meant to differ from the Mac's is declared in the comparator with its
  reason.
- Existing identifiers (`MLX_REMOTE`, `BRIDGE_*`, `KIRI_GPU_BRIDGE`,
  `mlx-remote-server`) are not renamed for style.

## Licence of contributions

By contributing you agree that your contribution is licensed as the part of
the repository it changes: MIT for everything except `kiri/patches/`, which
is Apache-2.0 (see [NOTICE](NOTICE)).

Changes meant for MLX, PyTorch or Kiri themselves belong upstream, under
those projects' own contribution rules: read them before proposing anything
there.
