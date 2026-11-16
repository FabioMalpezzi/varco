#!/bin/sh
# Incremental build of the patched MLX inside the dev container (source at
# /src, build tree at /build, python package assembled in /opt/mlx-site).
set -e
mkdir -p /build /opt/mlx-site
if [ ! -f /build/build.ninja ]; then
  cmake -S /src -B /build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DMLX_BUILD_REMOTE=ON \
    -DMLX_BUILD_PYTHON_BINDINGS=ON \
    -DMLX_BUILD_PYTHON_STUBS=OFF \
    -DMLX_BUILD_TESTS=OFF -DMLX_BUILD_EXAMPLES=OFF -DMLX_BUILD_BENCHMARKS=OFF \
    -DMLX_PYTHON_BINDINGS_OUTPUT_DIRECTORY=/build/pyout
fi
if ! cmake --build /build -j 14 > /build/build.log 2>&1; then
  grep -E "error|FAILED" -A6 /build/build.log | head -${TAIL:-60}
  exit 1
fi
rm -rf /opt/mlx-site/mlx
cp -r /src/python/mlx /opt/mlx-site/mlx
cp /build/pyout/core*.so /opt/mlx-site/mlx/
echo "build ok: $(ls /opt/mlx-site/mlx/core*.so)"
