#!/bin/sh
# Run a test script in the dev container against the Mac server and copy its
# JSON result back: run-container.sh SCRIPT RESULT.json [extra docker exec args]
SCRIPT=$1; OUT=$2; shift 2
export DOCKER_HOST=${DOCKER_HOST:-unix://$HOME/.docker/run/docker.sock}
docker exec "$@" varco-mlx-dev python /tests/$SCRIPT /tmp/out.json && docker exec varco-mlx-dev cat /tmp/out.json > "$OUT"
