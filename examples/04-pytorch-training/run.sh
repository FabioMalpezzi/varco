#!/bin/sh
# A Lightning training with mixed precision, a checkpoint and a resume, in a
# container on the Mac's GPU.
#   ./run.sh            trains 3 epochs in the container, then resumes from
#                       the checkpoint for one more epoch
#   ./run.sh native     the same on the Mac directly, to compare
# The checkpoints are in ./out (container/ and native/).
# ENGINE=kiri runs the container in Kiri, with the GPU request of compose.gpu.yaml.
cd "$(dirname "$0")" || exit 1
. ../common.sh
mkdir -p out
if [ "$1" = native ]; then
  [ -x "$TORCH_PY" ] || die "no Mac environment: run pytorch/build-mac.sh"
  rm -rf out/native
  "$TORCH_PY" train.py --out out/native && "$TORCH_PY" train.py --out out/native --resume
  exit $?
fi
need_image "varco-torch:$TORCH" "scripts/setup-mac.sh pytorch"
start_torch_server
trap stop_torch_server EXIT
rm -rf out/container
compose run --rm train && compose run --rm train python /app/train.py --out /out/container --resume
