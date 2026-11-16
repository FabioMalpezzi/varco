#!/bin/sh
# Container against native, back to back and alternating (the Mac is shared
# and its GPU slows as it warms). Needs the server running, started with
# --map /models=$MODELS so that it reads the model's weights
# itself, and the test container (docker/dev-up.sh).
#   tests/bench.sh [rounds] [bge|gen|all]
# BOXENV adds options to the container's side, e.g.
#   BOXENV="-e MLX_REMOTE_SPIN_US=200000" tests/bench.sh
cd "$(dirname "$0")/.."
export DOCKER_HOST=${DOCKER_HOST:-unix://$HOME/.docker/run/docker.sock}
MODELS=${MODELS:-$HOME/models}
MODEL=$MODELS/mlx-community/bge-m3-mlx-fp16
PY=${PY:-$PWD/.venv/bin/python}
BOX=${BOX:-varco-mlx-test}
rounds=${1:-3}
what=${2:-all}
i=0
while [ $i -lt $rounds ]; do
  i=$((i + 1))
  if [ $what != gen ]; then
    "$PY" tests/bge_phases.py --model "$MODEL" --runs 30 1 8 32
    docker exec $BOXENV $BOX python /tests/bge_phases.py --runs 30 1 8 32
  fi
  if [ $what != bge ]; then
    "$PY" tests/genloop.py results/gen-native.json | python3 -c "import sys,json; d=json.load(sys.stdin); print('gen native   ', d['timing']['all_runs_tokens_per_s'])"
    docker exec $BOXENV $BOX python /tests/genloop.py /out/gen-container.json | python3 -c "import sys,json; d=json.load(sys.stdin); print('gen container', d['timing']['all_runs_tokens_per_s'])"
  fi
done
