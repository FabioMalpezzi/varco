---
name: A program that fails or gives another result
about: An MLX or PyTorch program that behaves differently in the container than on the Mac
labels: bug
---

**Which bridge**: MLX / PyTorch

**The program.** The smallest program that shows it, in full.

**On the Mac directly** (`native`): what it prints.

**In the container**: what it prints, with any message of the bridge
(`[remote gpu] ...`, `bridge server: ...`) and the server's log
(`mlx/logs/server.log` or `pytorch/runs/server.log`).

**Setup**

- Mac (chip, memory) and macOS version:
- Engine and version (Docker Desktop, Kiri with the patches, other):
- Commit of this repository:
- The output of `scripts/doctor.sh`:
- MLX server options (`--map`, `--custom-kernels`, ...):

**Does it fail on the Mac too?** A program that fails the same way natively
is a limit of the Mac's GPU, not of the bridge
(docs/compatibility.md).
