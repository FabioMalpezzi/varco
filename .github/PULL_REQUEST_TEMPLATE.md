**What this changes**, and why.

**Which part**: MLX bridge / PyTorch bridge / Kiri patches / examples / documents

**Checks**

- [ ] `scripts/check.sh` passes.
- [ ] The tests of the bridge I changed were run on a Mac, and their summary
      is pasted below (`scripts/test-all.sh mlx` or `scripts/test-all.sh pytorch`).
      The CI has no Apple GPU and does not run them.
- [ ] A change to a wire format changes its version and its page under `docs/`.
- [ ] A change of speed comes with a measurement in alternated runs
      (docs/benchmarks.md, "Method"), and the machine it was taken on.
- [ ] Documents and numbers that the change makes stale are updated.

**Test output**

```
```
