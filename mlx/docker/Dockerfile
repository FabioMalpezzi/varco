# syntax=docker/dockerfile:1
# MLX for Linux with the remote gpu backend: the gpu device is a server on the
# Mac (server/mlx-remote-server). A container finds it by itself: at
# host.docker.internal:47800, or in Kiri at the address Kiri gives a container
# started with --gpus (KIRI_GPU_BRIDGE). MLX_REMOTE names another address, and
# is not set here: set, it would be used in Kiri too.
# The source is upstream MLX at a pinned tag with ../patches applied. Build
# from the directory that holds patches/ and docker/ (see ../build-images.sh):
#   docker build -f docker/Dockerfile -t varco-mlx:latest .
FROM python:3.12-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake ninja-build git ca-certificates \
      libopenblas-dev liblapacke-dev \
    && rm -rf /var/lib/apt/lists/*
ARG MLX_REPO=https://github.com/ml-explore/mlx.git
ARG MLX_TAG=v0.32.4
ARG MLX_COMMIT=36969c094354086e15d2a327a9c50feb20b9cf11
RUN git clone --depth 1 --branch "$MLX_TAG" "$MLX_REPO" /src/mlx \
    && head=$(git -C /src/mlx rev-parse HEAD) \
    && if [ "$head" != "$MLX_COMMIT" ]; then echo "the tag $MLX_TAG is at $head, not at the pinned $MLX_COMMIT" >&2; exit 1; fi
COPY patches /patches
WORKDIR /src/mlx
RUN for patch in /patches/*.patch; do git apply "$patch" && echo "applied $patch"; done
# PYPI_RELEASE gives the wheel the plain version of the base (0.32.4), so
# that packages requiring mlx accept it.
ENV PYPI_RELEASE=1 \
    CMAKE_GENERATOR=Ninja \
    CMAKE_ARGS="-DMLX_BUILD_REMOTE=ON -DMLX_BUILD_PYTHON_STUBS=OFF"
RUN pip wheel --no-deps -w /wheels .

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
      libopenblas0 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/mlx-*.whl numpy
# (GLIBC_TUNABLES=glibc.malloc.tcache_count=65535 was tried here: a loop that
# only builds the graph of BGE-M3 goes from 0.60 to 0.48 ms, as on macOS, but
# a whole embedding, which is what matters, gets 0.3 ms slower. Not set.)
WORKDIR /work
CMD ["python"]
