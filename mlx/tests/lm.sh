#!/bin/sh
# A language model natively and in a container, whose weights the Mac reads
# from the files itself: load time, first answer, tokens per second, and the
# memory the container takes at its peak (docker stats).
#   tests/lm.sh MODEL [native|container ...]      (default: native container)
# MODEL is a directory under MODELS (default $HOME/models), e.g.
#   tests/lm.sh mlx-community/Qwen3.6-35B-A3B-4bit
# PROGRAM is the script to run (default tests/qwen.py, an mlx-lm program),
# IMAGE the container image (default varco-mlx-lm:latest), LMPY the native
# python (default .venv/bin/python, which setup-mac.sh makes with mlx 0.32.4
# and mlx-lm 0.32.2), PORT and
# SERVER_BIN as in tests/run-all.sh, SERVER_ARGS more options for the server
# (e.g. -v, or --custom-kernels for the models whose layers mlx-lm computes
# with Metal kernels of its own). The logs are in logs/lm-*.log.
# ENGINE=kiri runs the container in Kiri: the server listens on the Unix
# socket Kiri's channel leads to (WHERE, default ~/.kiri/run/gpu.sock) and on
# no port, and the container is started with --gpus and finds it by itself.
# The exit status is 0 only when every side that was run ended without an
# error and, with both logs there, the text is the same and is not empty.
cd "$(dirname "$0")/.."
ROOT=$PWD
ENGINE=${ENGINE:-docker}
case $ENGINE in
  kiri) export DOCKER_HOST=unix://$HOME/.kiri/run/docker.sock ;;
  docker) export DOCKER_HOST=${DOCKER_HOST:-unix://$HOME/.docker/run/docker.sock} ;;
  *) echo "ENGINE is docker or kiri, not $ENGINE" >&2; exit 2 ;;
esac
MODELS=${MODELS:-$HOME/models}
MODEL=$1; shift
PROGRAM=${PROGRAM:-qwen.py}
IMAGE=${IMAGE:-varco-mlx-lm:latest}
LMPY=${LMPY:-$ROOT/.venv/bin/python}
PORT=${PORT:-47800}
NAME=varco-mlx-lm-run
mkdir -p logs
status=0
# The status of the program of a side, which the pipe that shows its output
# as it comes would hide: kept in a file, read here.
ended() {
  code=$(cat "logs/lm-$1.status" 2>/dev/null)
  if [ "${code:-none}" != 0 ]; then
    echo "FAILED: the program ended with status ${code:-unknown} on the $1 side: see logs/lm-$1.log"
    status=1
  fi
}
# The answers of a side: its log without the lines of timings.
text() {
  grep -v -e '^---' -e '^device' -e 'remote gpu' "logs/lm-$1.log"
}
for side in ${*:-native container}; do
  echo "=== $side"
  rm -f "logs/lm-$side.status"
  case $side in
    native)
      { "$LMPY" tests/$PROGRAM "$MODELS/$MODEL" $ARGS 2>&1; echo $? > logs/lm-native.status; } | tee logs/lm-native.log | cut -c1-400
      ended native
      ;;
    container)
      docker rm -f $NAME > /dev/null 2>&1
      if [ "$ENGINE" = kiri ]; then
        ./server/ctl.sh start --unix "${WHERE:-$HOME/.kiri/run/gpu.sock}" --map /models="$MODELS" $SERVER_ARGS || exit 1
        docker run -d --name $NAME --gpus all $BOXENV \
          -v "$ROOT/tests:/tests:ro" -v "$MODELS:/models:ro" $IMAGE sleep infinity > /dev/null
      else
        ./server/ctl.sh start --port $PORT --map /models="$MODELS" $SERVER_ARGS || exit 1
        docker run -d --name $NAME -e MLX_REMOTE=host.docker.internal:$PORT $BOXENV \
          -v "$ROOT/tests:/tests:ro" -v "$MODELS:/models:ro" $IMAGE sleep infinity > /dev/null
      fi
      docker stats --format '{{.MemUsage}}' $NAME > logs/lm-stats.log 2>&1 &
      stats=$!
      sleep 2
      { docker exec $NAME python /tests/$PROGRAM /models/$MODEL $ARGS 2>&1; echo $? > logs/lm-container.status; } | tee logs/lm-container.log | cut -c1-400
      ended container
      sleep 2
      kill $stats
      echo "container memory at its peak, docker stats: $(tr '\r' '\n' < logs/lm-stats.log | grep -o '[0-9.]*[KMG]iB /' | sed 's/GiB/ 1024/; s/MiB/ 1/; s/KiB/ 0.001/; s/ \/$//' | LC_ALL=C awk '{ v = $1 * $2; if (v > max) max = v } END { printf "%.0f MiB", max }'); with the files it has in cache (the cgroup's memory.peak): $(docker exec $NAME sh -c 'echo $(($(cat /sys/fs/cgroup/memory.peak) >> 20))') MiB"
      docker rm -f $NAME > /dev/null
      ./server/ctl.sh stop > /dev/null
      grep closed logs/server.log | tail -1 | grep -o "uploads=.*loads=[^)]*)"
      ;;
  esac
done
if [ -f logs/lm-native.log ] && [ -f logs/lm-container.log ]; then
  # The answers: what follows each line of timings.
  if [ -z "$(text native)" ] || [ -z "$(text container)" ]; then
    echo "=== NO TEXT to compare: a side generated nothing"
    status=1
  elif [ "$(text native)" = "$(text container)" ]; then
    echo "=== the text is identical"
  else
    echo "=== the text DIFFERS"
    status=1
  fi
fi
exit $status
