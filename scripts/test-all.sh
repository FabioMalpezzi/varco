#!/bin/sh
# Run the tests of both bridges: every program natively on the Mac and in a
# container through the bridge, compared.
#   scripts/test-all.sh            both
#   scripts/test-all.sh mlx        mlx/tests/run-all.sh (needs the model
#                                  mlx-community/bge-m3-mlx-fp16 under MODELS)
#   scripts/test-all.sh pytorch    pytorch/compat.sh (needs gpt2 and
#                                  all-MiniLM-L6-v2 in the Hugging Face cache)
# The tests use the Mac's GPU and take a few minutes: run nothing else on it
# meanwhile if you mean to read the timings.
cd "$(dirname "$0")/.."
what=${*:-mlx pytorch}
status=0
for part in $what; do
  case $part in
    mlx)
      echo "== MLX bridge: mlx/tests/run-all.sh"
      mlx/tests/run-all.sh || status=1
      ;;
    pytorch)
      echo "== PyTorch bridge: pytorch/compat.sh, PyTorch ${TORCH:-2.15.0}"
      pytorch/compat.sh || status=1
      ;;
    *)
      echo "usage: test-all.sh [mlx] [pytorch]" >&2
      exit 2
      ;;
  esac
done
exit $status
