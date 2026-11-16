#!/bin/sh
# Fetch upstream MLX at the pinned tag into ./upstream and apply patches/.
#   ./fetch-source.sh          (does nothing if ./upstream is already patched)
#   ./fetch-source.sh --force  (removes ./upstream and starts again)
# The Mac server is built against one header of the patched tree
# (mlx/backend/remote/protocol.h), the stage "upstream" of tests/run-all.sh
# runs MLX's own tests from it, and docker/dev-up.sh builds it incrementally.
# The container images do not use this copy: their Dockerfile clones and
# patches by itself.
set -e
cd "$(dirname "$0")"
. ./versions.sh
if [ "$1" = --force ]; then
  rm -rf upstream
fi
if [ -f upstream/mlx/backend/remote/protocol.h ]; then
  echo "upstream/ is already there and patched (use --force to fetch it again)"
  exit 0
fi
if [ -e upstream ]; then
  echo "upstream/ exists but is not a patched MLX tree: remove it, or run ./fetch-source.sh --force" >&2
  exit 1
fi
git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$MLX_TAG" "$MLX_REPO" upstream
head=$(git -C upstream rev-parse HEAD)
if [ "$head" != "$MLX_COMMIT" ]; then
  echo "the tag $MLX_TAG is at $head, not at the pinned $MLX_COMMIT: refusing to patch it" >&2
  exit 1
fi
for patch in patches/*.patch; do
  git -C upstream apply --index "../$patch"
  echo "applied $patch"
done
echo "upstream/ is MLX $MLX_TAG ($MLX_COMMIT) with patches/ applied"
