#!/bin/sh
# Build the three container images of the MLX bridge:
#   varco-mlx             patched MLX for Linux, the remote gpu backend
#   varco-mlx-embeddings  the same with mlx-embeddings (the tests' image)
#   varco-mlx-lm          the same with mlx-lm (text generation, the server)
#   ./build-images.sh [docker build options, e.g. --no-cache]
# TAG names another tag than latest for all three (TAG=dev ./build-images.sh
# builds varco-mlx:dev and the two others on top of it), so that a second
# working tree can build its own without replacing the ones in use.
# The first one clones upstream MLX at the pinned tag and applies patches/;
# it compiles MLX and takes several minutes the first time.
# DOCKER_HOST names the engine to build in (Docker Desktop's by default).
set -e
cd "$(dirname "$0")"
. ./versions.sh
export DOCKER_HOST=${DOCKER_HOST:-unix://$HOME/.docker/run/docker.sock}
TAG=${TAG:-latest}
# Where the builder leaves what it builds. Docker Desktop's (buildx's docker
# driver) builds into the engine's own images: an image is there when its
# build ends, and the next build finds it by name. A builder that runs in a
# container of the engine (the docker-container driver, which is what
# "docker build" uses with Kiri) does neither. So with such a builder each
# image is loaded into the engine (--load), and the two images that are built
# on the first are given it as an OCI layout, written from the builder's cache.
driver=$(docker buildx inspect 2> /dev/null | sed -n 's/^Driver: *//p' | head -1)
load=
base=
if [ -n "$driver" ] && [ "$driver" != docker ]; then
  load=--load
  work=$(mktemp -d)
  trap 'rm -rf "$work"' EXIT
fi
docker build "$@" $load -f docker/Dockerfile -t "varco-mlx:$TAG" \
  --build-arg MLX_TAG="$MLX_TAG" --build-arg MLX_COMMIT="$MLX_COMMIT" --build-arg MLX_REPO="$MLX_REPO" .
if [ -n "$load" ]; then
  # The same build again, with the same options but --no-cache: it is the one
  # just made, taken from the cache.
  layout() {
    for arg in "$@"; do
      shift
      [ "$arg" = --no-cache ] || set -- "$@" "$arg"
    done
    docker build "$@" -f docker/Dockerfile --output "type=oci,dest=$work/base.tar" \
      --build-arg MLX_TAG="$MLX_TAG" --build-arg MLX_COMMIT="$MLX_COMMIT" --build-arg MLX_REPO="$MLX_REPO" .
  }
  layout "$@"
  mkdir "$work/base"
  tar -xf "$work/base.tar" -C "$work/base"
  base="--build-context varco-mlx:$TAG=oci-layout://$work/base"
fi
docker build "$@" $load $base -f docker/Dockerfile.embeddings -t "varco-mlx-embeddings:$TAG" \
  --build-arg BASE="varco-mlx:$TAG" --build-arg MLX_EMBEDDINGS_VERSION="$MLX_EMBEDDINGS_VERSION" docker
docker build "$@" $load $base -f docker/Dockerfile.lm -t "varco-mlx-lm:$TAG" \
  --build-arg BASE="varco-mlx:$TAG" --build-arg MLX_LM_VERSION="$MLX_LM_VERSION" docker
echo "built varco-mlx:$TAG, varco-mlx-embeddings:$TAG and varco-mlx-lm:$TAG"
