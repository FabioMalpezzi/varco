# Reference results

Raw JSON output of the MLX bridge's tests, as the earlier measurements of
[docs/benchmarks.md](../../../docs/benchmarks.md) cite them. They were
produced on one machine (Apple M4 Max, 128 GB, macOS 26.6.2) on 15 November
2026 and are copied here unchanged, except for one file name
(`serve-mb-default.json`, whose `out` field still carries its original
name).

| Files | Test | Where |
|---|---|---|
| `coverage-{native,kiri}-k8.json` | `tests/coverage.py`, 57 programs | Mac, and a Kiri container, protocol 8 |
| `bge-{native,kiri}-k8.json` | `tests/bge_m3.py`, BGE-M3 embeddings and timings | Mac, and a Kiri container, protocol 8 |
| `kernels-{native,kiri}-k8.json` | `tests/kernels.py`, 38 custom-kernel cases | Mac, and a Kiri container with `--custom-kernels`, protocol 8 |
| `serve-Qwen3.6-35B-A3B-4bit-kiri-p9.json` | `tests/serve.sh`, mlx-lm's server on both sides | Kiri, protocol 9 |
| `serve-Qwen3.6-35B-A3B-4bit-{p8,p9}.json`, `serve-Qwen3.5-0.8B-Base-{p8,p9}.json` | the same, before and after protocol 9 | Docker Desktop |
| `serve-mb-{default,mb2048}.json` | the same, with MLX's default command buffer and with 2,048 MB | Docker Desktop, protocol 9 |

These files were written at protocols 8 and 9, before the tests kept a
digest of every array, and are published as the record of the earlier
tables. The comparators of the released code hold a result to the type, the
shape and the SHA-256 of its bytes, so they print what the files hold and
then fail on them:

```sh
python3 tests/compare.py results/reference/coverage-native-k8.json results/reference/coverage-kiri-k8.json
python3 tests/compare_bge.py --report-only results/reference/bge-native-k8.json results/reference/bge-kiri-k8.json
python3 tests/compare.py kernels results/reference/kernels-native-k8.json results/reference/kernels-kiri-k8.json
```

- The coverage files hold one checksum per program, the same on both sides
  for 55 of the 57. The two that differ are `nn_losses`
  (49.023107154846194 on the Mac, 49.023109300613406 in Kiri) and
  `cpu_linalg` (80.51065736781636 and 80.51066145259712). A checksum does
  not show whether the bytes of the arrays were equal. `compare.py` shows
  each pair and calls it "not a result that can be compared in full". On
  the released code, which is compared by bytes, `nn_losses` is identical
  and `cpu_linalg` is declared
  ([benchmarks](../../../docs/benchmarks.md#collections-on-the-released-code)).
- The BGE-M3 files hold the embeddings as numbers and no digest.
  `compare_bge.py` prints the cosines (1.0000000), the largest difference
  (0) and the timings of the tables, and says that the two files are not
  compared bit for bit. Without `--report-only` it exits with an error for
  that reason.
- The kernel files hold the bytes of each case. `compare.py` finds 29 cases
  with the same bytes, and one difference that protocol 11 closed: a
  threadgroup that is too large raised a `ValueError` on the Mac and a
  `RuntimeError` in the container.

The comparators need only Python's standard library.

The logs of the runs and the outputs of the PyTorch bridge are not
published.
