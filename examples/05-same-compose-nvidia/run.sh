#!/bin/sh
# The same Compose service on a Mac and on an NVIDIA server.
#   ./run.sh            Docker Desktop on a Mac: compose.yaml alone
#   ./run.sh gpu        with the GPU request of compose.gpu.yaml: for Kiri on
#                       a Mac (ENGINE=kiri ./run.sh gpu, or ENGINE=kiri
#                       ./run.sh: in Kiri the request is what gives the
#                       container its bridge) or an NVIDIA server (IMAGE=a
#                       CUDA build of PyTorch, and no server is started here)
cd "$(dirname "$0")" || exit 1
. ../common.sh
if [ "$1" = gpu ] || [ "$ENGINE" = kiri ]; then
  if [ "$ENGINE" = kiri ]; then
    need_image "varco-torch:$TORCH" "scripts/setup-mac.sh pytorch"
    start_torch_server
    trap stop_torch_server EXIT
  fi
  docker compose -f compose.yaml -f compose.gpu.yaml run --rm job
  exit $?
fi
need_image "varco-torch:$TORCH" "scripts/setup-mac.sh pytorch"
start_torch_server
trap stop_torch_server EXIT
docker compose run --rm job
