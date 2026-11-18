# Kiri patches

[Kiri](https://github.com/libracoder/kiri) is a container engine for macOS,
Windows and Linux that speaks the Docker API. These are two patches against
its commit `69dc0b7`, licensed under Apache-2.0 as Kiri is
([LICENSE-APACHE-2.0](LICENSE-APACHE-2.0)). They are not part of Kiri.

The channel was proposed to Kiri on 16 November 2026 in its
[issue 356](https://github.com/libracoder/kiri/issues/356). Kiri's
maintainer closed it the next day as not planned: Kiri works on the
container engine itself, and the channel is outside that. The patch stays
here for those who build Kiri themselves, and is moved to a newer commit of
Kiri when someone needs it: on 18 November 2026 it was moved to `69dc0b7`,
Kiri's `main` of that day. With Docker Desktop the bridges need no patch.

The memory change was reported in Kiri's
[issue 210](https://github.com/libracoder/kiri/issues/210), which tracks the
same cause and for which Kiri's maintainer announced a fix in Kiri 0.2.2.

| Patch | What it does |
|---|---|
| [`0001-gpu-bridge-channel.patch`](patches/0001-gpu-bridge-channel.patch) | The channel. On an Apple-silicon Mac, a container created with a GPU request gets its end of the bridge. Nine files. |
| [`0002-memory-headroom-after-shortage.patch`](patches/0002-memory-headroom-after-shortage.patch) | A fix to the memory balloon, found while measuring. One file. Independent of the first. |

Each applies alone to `69dc0b7`, and both apply together.

## The channel

Upstream Kiri refuses `--gpus`: it has no GPU to give. With the patch, on an
Apple-silicon Mac:

- A container created with a device request whose capabilities name `gpu`
  (`docker run --gpus all`, or Compose's
  `deploy.resources.reservations.devices` with `capabilities: [gpu]`, with
  any driver name) is accepted. Any other device request is still refused,
  and so is a GPU request on any other host.
- Kiri's guest agent opens a listener at `127.0.0.1:47700` inside that
  container's network namespace, before the program starts.
- Each connection to it is relayed over vsock (port 1041) to the daemon on
  the Mac, which connects it to the Unix socket `run/gpu.sock` in Kiri's
  state directory (`~/.kiri/run/gpu.sock`). That is where a bridge's server
  listens. With no server there the container sees its connection close.
- The container's bytes travel framed (`u32` length, big endian, then the
  bytes; an empty frame for the end), because a plain half-close after a
  large write was not seen by the Mac.
- The variable `KIRI_GPU_BRIDGE=127.0.0.1:47700` is set in the container,
  unless the container sets it itself.
- `docker inspect` shows the request under `HostConfig.DeviceRequests`.

The listener is for that container alone, and Kiri refuses what would share
it:

- `--gpus` together with `--network host` or `--network container:...`:
  "conflicting options: --gpus and --network host: the GPU bridge needs a
  network namespace of the container's own".
- A container without `--gpus` that asks for the network of one that has
  it, `--network container:A`: "conflicting options: --network container:A
  and the --gpus of A: the GPU bridge listens in that network namespace,
  for A alone". The daemon follows a chain of `container:` to the container
  the namespace belongs to and refuses if any container on the way has the
  bridge, when the container is created and when it is started. The guest
  agent refuses as well to join a namespace that holds a bridge listener.

The channel leads to one socket, so one bridge's server is reachable at a
time. Kiri also forwards `host.docker.internal:PORT` of every container to
the Mac's loopback: a bridge server that listens on a TCP port is reached
by containers started without `--gpus`. With `ENGINE=kiri` the scripts of
this repository start the servers on the socket and on no port
([setup](../docs/setup.md#docker-desktop-or-kiri)).

The files: `crates/kiri-proto/src/lib.rs` (the two port numbers),
`crates/kiri/src/daemon/hostopts.rs` and `api.rs` (accepting the request,
the variable), `engine.rs` and `ports.rs` (the daemon's end of the relay,
the refusal of a joined network), `crates/kiri-init/src/linux/hostopts.rs`,
`container.rs`, `agent.rs` and `relay.rs` (the guest's end).

## The memory headroom

Kiri gives memory back to macOS with a balloon that shrinks the virtual
machine to what the guest counts as used plus 256 MiB. A program that keeps
reading files (a language model's weights are not in that count) ran short
at that point, stalled until the guest's pressure event got it 1 GiB back,
and then the shrinking began again: one stall of 213–226 ms every 4
seconds. The patch keeps more headroom after a shortage (what was not
enough, plus the usual 256 MiB, at most 2 GiB) and gives it back a step
every ten calm minutes.

Measured: 5 to 7 stalls in 40 generations before; one stall in the first 40
after (the one it learns from), then 0 in 60
([benchmarks](../docs/benchmarks.md#a-stall-in-kiri-and-its-fix)).

## Applying and building

```sh
kiri/apply.sh             # clones Kiri at 69dc0b7 into kiri/kiri-src and applies both patches
```

Then build Kiri as its own `docs/development.md` says, from that directory.
The times are from a fresh clone on an Apple M4 Max; the first two steps
need a Docker engine that is running:

```sh
cargo xtask kernel        # the guest kernel: 123 s
cargo xtask disk          # the data disk template: 4 s
cargo xtask guest         # the guest's initramfs, with the patched agent: 13 s
cargo xtask host          # target/release/kiri, signed with the virtualization entitlement: 71 s
```

A second build on the same machine took 107 s where the first took 211 s,
because Docker had the kernel's build in its cache: 28 s for the kernel, 3
for the disk, 11 for the guest and 65 for the host.

Start the daemon it builds. It finds its guest files by itself, and
`KIRI_DATA_DIR` keeps its data apart from that of an installed Kiri:

```sh
KIRI_DATA_DIR=$PWD/target/gpu-data target/release/kiri daemon
```

Its Docker socket is `~/.kiri/run/docker.sock`, where `docker version`
reports `Server: Kiri` (with `Version: 27.5.1`, the Docker engine version
Kiri gives itself, for a build from `69dc0b7`). How
to use it from there is in the [quick start](../docs/quickstart.md#kiri).

The daemon keeps its images in its data directory. One started from a new
build, with a `KIRI_DATA_DIR` of its own, does not show the images of an
earlier Kiri, which are still in the earlier directory.

When port 80 is in use on the Mac the daemon prints at start
`WARN container domains off error=port 80 is taken; pick another in Settings → Domains`.
It does not affect the bridge.

`docker build` against this daemon runs its builder in a container of the
engine, and the image stays in the builder unless the build is given
`--load`. The build scripts of this repository pass it
([setup](../docs/setup.md#docker-desktop-or-kiri)).

## What was verified, and what was not

On 18 November 2026 the patches were moved from `260cf66` to `69dc0b7`,
31 commits later. In the channel, two places had changed around it and
keep both changes: the guest agent now waits for the container's root
before it reports the start, and opens the bridge's listener after that,
while the program still waits for its network; and the daemon's startup
keeps the files of containers missing from a damaged container list, and
starts the bridge's listener just before. The memory patch applied as it
was. On that commit, on an Apple M4 Max, with Docker's cache: each patch
applies alone and both together; the four build steps complete in 123 s;
`cargo test -p kiri` passes (428 tests, 6 ignored), with the ten named
below; the listener, the variable and the refusals behave as described;
and through that daemon, with its default virtual machine and the images
copied from Docker Desktop, `scripts/doctor.sh` with each bridge's server
alone on the socket and the seven examples completed with status 0.

The patches were written against Kiri's commit `baa2d41` and moved to
`260cf66` on 16 November 2026, 20 commits later: two hunks were placed
again by hand, and the unit test of the refusals was given the two fields
Kiri had added to a container. On a fresh clone at `260cf66` with both
patches, on an Apple M4 Max: the four build steps complete (103 s with
Docker's cache), the ten unit tests named below pass, the channel and the
four refusals behave as described, and through that daemon, with its
default virtual machine and the images copied from Docker Desktop,
`ENGINE=kiri scripts/setup-mac.sh`, `scripts/doctor.sh` with each bridge's
server alone on the socket, and examples 1 and 5 gave the results of the
earlier commit.

Verified on a fresh clone of upstream Kiri at `baa2d41`, on an Apple M4 Max:

- each patch applies alone, and both apply together (`scripts/check.sh`);
- with both applied, the four build steps above complete;
- the unit tests the patches add or change in the `kiri` crate pass:
  `cargo test -p kiri hostopts` (7 tests), `cargo test -p kiri joined_network`
  (2 tests) and `cargo test -p kiri a_shortage_leaves_more_headroom`
  (1 test);
- on the daemon started as above, a container started with `--gpus all` has
  `KIRI_GPU_BRIDGE=127.0.0.1:47700` and a listener at that address, and a
  container started without `--gpus` has neither: its connection to
  `127.0.0.1:47700` is refused, and Kiri closes at once a connection to a
  port of `host.docker.internal` on which nothing listens on the Mac. Both
  bridges then tell the program that no server answered and that the
  container must be started with `--gpus all`;
- `docker run --network container:A`, where A has `--gpus`, and `--gpus`
  with `--network host` are refused with the two messages above;
- on the same daemon, on 16 November 2026 and with the virtual machine at 8
  CPUs and 16 GiB: the images built in Kiri
  (`ENGINE=kiri scripts/setup-mac.sh`), `scripts/doctor.sh` with each
  bridge's server alone on the socket, the seven examples, the stages of
  `mlx/tests/run-all.sh` that need one server, Qwen3.6-35B-A3B-4bit with
  `mlx/tests/lm.sh`, `pytorch/compat.sh`, and `scripts/autostart.sh` with
  one bridge. The results are in the
  [benchmarks](../docs/benchmarks.md#measured-again-on-the-released-code);
- on a daemon from a second build, with the default virtual machine of 2
  CPUs and 8 GiB: the images built in Kiri in 331 s, of which 237 for the
  three MLX images, `scripts/doctor.sh` with no server running, examples 1
  and 5, and `torch.backends.mps.is_available()` printing `False` in a
  container started without `--gpus`.

Not verified:

- the changed unit test of `kiri-init`, which is a Linux program, and
  Kiri's other tests;
- builds for Linux, Intel Macs and Windows;
- the headroom's step down after ten calm minutes;
- the same stall on a released Kiri;
- the test collections and the timed models of the released code with
  Kiri's default virtual machine of 2 CPUs and 8 GiB.

One fault is known: a connection was left open in an edge case, which was
not reproduced.
