#!/bin/sh
# Embeddings with BGE-M3 in a container, on the Mac's GPU.
#   ./run.sh            the container, through the bridge
#   ./run.sh native     the same program on the Mac directly, to compare
# ENGINE=kiri runs the container in Kiri, with the GPU request of compose.gpu.yaml.
cd "$(dirname "$0")" || exit 1
. ../common.sh
MODEL=mlx-community/bge-m3-mlx-fp16
need_model "$MODEL"
if [ "$1" = native ]; then
  [ -x "$MLX_PY" ] || die "no Mac environment: run mlx/setup-mac.sh"
  exec "$MLX_PY" embed.py "$MODELS/$MODEL"
fi
need_image varco-mlx-embeddings:latest "scripts/setup-mac.sh mlx"
start_mlx_server
trap stop_mlx_server EXIT
compose run --rm embed
