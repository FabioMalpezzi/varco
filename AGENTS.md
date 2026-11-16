# For agents working on this repository

To install varco for a person, follow
[docs/agent-install.md](docs/agent-install.md) and nothing else in this
file. What follows is for changing the code.

## Build, test, check

```sh
scripts/setup-mac.sh        # builds what is missing or stale, then tries each bridge
scripts/check.sh            # what the CI runs: syntax, comparators, links, autostart, patches; no GPU needed
scripts/test-all.sh         # both bridges on the Mac's GPU against the Mac itself; needs Docker Desktop running
mlx/tests/run-all.sh STAGE  # one MLX stage; pytorch/compat.sh checks|compat|compat_train for PyTorch
```

Run `scripts/check.sh` before every commit. Run the tests of the bridge you
changed before saying a change works: the CI has no Apple GPU and runs none
of them. The layout, every stage and the way to change a patch are in
[docs/development.md](docs/development.md).

## What the code does not tell you

- **Two ends, one state.** A bridge's container end and Mac end must come
  from the same commit. After a change to either, rebuild both
  (`scripts/setup-mac.sh`); a change to a wire format raises the protocol
  number in both ends and in its protocol page.
- **The patches are the source.** `mlx/upstream` and `kiri/kiri-src` are
  fetched and patched copies, ignored by git. Change the code there, then
  regenerate the patch as `docs/development.md` describes; `check.sh`
  applies each patch to a fresh clone.
- **A result is the Mac's, or declared.** The comparators accept a container
  result only when its bytes are the Mac's. A difference with a known cause
  is declared in the comparator's source with its reason; widening a
  tolerance to make a test pass is the wrong fix.
- **Only listed operations run on the Mac.** A new MLX primitive needs a
  rule in `check_shapes`; a PyTorch operator enters the list through
  `pytorch/gen_ops.py` and `policy.py`. Both have a refusal test to extend.
- **One user of the GPU.** Timings are taken with nothing else on the GPU,
  Mac and container in mirrored order, and a figure goes into a document
  with the code it was taken on.
- **Servers are stopped by their pid files** (`mlx/server/ctl.sh stop`,
  `pytorch/server.sh stop`, `scripts/autostart.sh uninstall`), never by
  matching process names: the tests and a person's own servers can run side
  by side.
- **Documents state what was run.** Write a command in a document after
  running it, and a number with its unit and conditions.
  [CONTRIBUTING.md](CONTRIBUTING.md) has the rules for prose.
