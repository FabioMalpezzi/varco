#!/bin/sh
# A program with a Metal kernel of its own, in a container.
#   ./run.sh            twice: against a server started as it is by default,
#                       which refuses the kernel, and against one started
#                       with --custom-kernels, which compiles and runs it
#   ./run.sh native     the same program on the Mac directly
# --custom-kernels lets a container run code of its own on the Mac's GPU.
# Read docs/security.md before using it with containers you do not trust.
# ENGINE=kiri runs the containers in Kiri, started with --gpus.
cd "$(dirname "$0")" || exit 1
. ../common.sh
if [ "$1" = native ]; then
  [ -x "$MLX_PY" ] || die "no Mac environment: run mlx/setup-mac.sh"
  exec "$MLX_PY" kernel.py
fi
need_image varco-mlx:latest "scripts/setup-mac.sh mlx"
box() {
  docker run --rm $GPUS ${MLX_REMOTE:+-e MLX_REMOTE="$MLX_REMOTE"} -v "$PWD:/app:ro" varco-mlx:latest python /app/kernel.py
}
MLX_OWN_SERVER=1   # the example is about how the server was started
trap stop_mlx_server EXIT
echo "=== the server as it starts by default"
start_mlx_server
box
echo "=== the server started with --custom-kernels"
start_mlx_server --custom-kernels
box
