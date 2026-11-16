---
name: A measurement
about: Results of the tests or the examples on your machine, agreeing or not with docs/benchmarks.md
labels: measurement
---

**Machine**: chip, memory, macOS version.

**Engine**: Docker Desktop or Kiri, and its version; CPUs and memory of the
virtual machine.

**What you ran** (the exact commands) and the commit of this repository.

**Output.** The summary lines of `scripts/test-all.sh`, or the output of
the example, for both sides (native and container).

**How it was taken.** One run, or alternated runs (Mac, container,
container, Mac)? Was anything else using the GPU?

**What differs** from docs/benchmarks.md, if anything.
