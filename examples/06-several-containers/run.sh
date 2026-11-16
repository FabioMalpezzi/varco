#!/bin/sh
# Several containers generating text at the same time, on one server of the
# MLX bridge, each with its own copy of the model.
#   ./run.sh [N] [MODEL] [GENERATIONS]      (default 2, Qwen/Qwen3.5-0.8B-Base, 8)
#   ./run.sh mac [N] [MODEL] [GENERATIONS]  N processes on the Mac directly, to compare
# ENGINE=kiri runs the containers in Kiri, each started with --gpus.
# Every copy loads the weights: N copies of the model are in the Mac's
# memory at once. Mind that before raising N with a large model.
cd "$(dirname "$0")" || exit 1
. ../common.sh
where=$ENGINE
if [ "$1" = mac ]; then
  where=mac
  shift
fi
n=${1:-2}
MODEL=${2:-Qwen/Qwen3.5-0.8B-Base}
generations=${3:-8}
need_model "$MODEL"
if [ $where = mac ]; then
  [ -x "$MLX_PY" ] || die "no Mac environment: run mlx/setup-mac.sh"
else
  need_image varco-mlx-lm:latest "scripts/setup-mac.sh mlx"
  start_mlx_server
  trap stop_mlx_server EXIT
fi
# The status is 0 only when every round ended as tests/multi.sh expects: every
# copy without an error, and the same text from all of them.
status=0
for copies in 1 "$n"; do
  echo "=== $copies at once"
  PORT=$MLX_PORT WAIT=${WAIT:-10} OUT=${OUT:-/tmp/varco-multi-gen} "$REPO/mlx/tests/multi.sh" $where "$copies" "$generations" "$MODEL" || status=1
  [ "$n" = 1 ] && break
done
exit $status
