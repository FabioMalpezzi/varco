#!/bin/sh
# Build the Mac server against the stock MLX wheel of ./.venv (its headers
# and library; no Xcode project needed) and the protocol header of the
# patched tree in ./upstream.
#   ./setup-mac.sh && ./fetch-source.sh && server/build.sh
set -e
cd "$(dirname "$0")/.."
PY=${PY:-$PWD/.venv/bin/python}
if [ ! -x "$PY" ]; then
  echo "no $PY: run ./setup-mac.sh first" >&2
  exit 1
fi
if [ ! -f upstream/mlx/backend/remote/protocol.h ]; then
  echo "no upstream/mlx/backend/remote/protocol.h: run ./fetch-source.sh first" >&2
  exit 1
fi
SP=$("$PY" -c "import mlx.core, os; print(os.path.dirname(mlx.core.__file__))")
# (metal_cpp: mlx's own header of its Metal device, which the server asks to let
# go of the libraries of custom kernels, is written against it.)
clang++ -std=c++20 -O2 -Wall -Wno-unused-function -I"$SP/include" -I"$SP/include/metal_cpp" \
  server/mlx_remote_server.cpp \
  -L"$SP/lib" -lmlx -Wl,-rpath,"$SP/lib" \
  -o server/mlx-remote-server
echo "built server/mlx-remote-server"
