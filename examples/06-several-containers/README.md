# Example 6: several containers at once

N containers generating text at the same moment through one server of the
MLX bridge, each with its own copy of the model: first one alone, then N
together, with each copy's rate, the total, and whether the text is the
same.

```sh
./run.sh            # 1, then 2 containers, with Qwen/Qwen3.5-0.8B-Base
./run.sh 3          # 1, then 3
./run.sh mac        # 1, then 2 processes on the Mac directly, to compare
./run.sh 2 mlx-community/Qwen3.6-35B-A3B-4bit     # a large model: mind the memory, N copies are loaded
```

It drives [`mlx/tests/multi.sh`](../../mlx/tests/multi.sh). What to expect,
the measured figures and the limits:
[docs/use-cases/several-containers.md](../../docs/use-cases/several-containers.md).
