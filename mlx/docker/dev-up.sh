#!/bin/sh
# Start the two development containers (Docker Desktop):
#   varco-mlx-dev    builds the patched MLX of ./upstream incrementally
#                    (docker/dev-build.sh), build tree in the volume
#                    varco-mlx-build, python package assembled in ./site
#   varco-mlx-test   the final image with ./site ahead of its own mlx, for
#                    tests/bench.sh and BOX=varco-mlx-test tests/run-all.sh
# MODELS is the directory of the models (default $HOME/models).
# docker/dev-up.sh down removes them (the volume and ./site stay).
cd "$(dirname "$0")/.."
export DOCKER_HOST=${DOCKER_HOST:-unix://$HOME/.docker/run/docker.sock}
MODELS=${MODELS:-$HOME/models}
docker rm -f varco-mlx-dev varco-mlx-test > /dev/null 2>&1
[ "$1" = down ] && exit 0
mkdir -p site results
docker run -d --name varco-mlx-dev -v "$PWD/upstream:/src" -v "$PWD/docker:/docker:ro" \
  -v varco-mlx-build:/build -v "$PWD/site:/opt/mlx-site" varco-mlx-dev:latest sleep infinity
docker run -d --name varco-mlx-test -e PYTHONPATH=/opt/mlx-site \
  -v "$PWD/site:/opt/mlx-site:ro" -v "$PWD/tests:/tests:ro" \
  -v "$MODELS:/models:ro" -v "$PWD/results:/out" \
  varco-mlx-embeddings:latest sleep infinity
echo "then: docker exec varco-mlx-dev sh /docker/dev-build.sh"
