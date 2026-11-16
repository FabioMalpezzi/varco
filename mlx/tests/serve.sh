#!/bin/sh
# A model server natively and in a container: mlx-lm's own (python -m mlx_lm
# server), unchanged, with the same model on both sides, and the same requests
# sent to both over HTTP in mirrored order (tests/serve_client.py): one
# request, streaming, a long prompt, the prompt cache, 2 and 4 requests at once.
#   tests/serve.sh MODEL [options of tests/serve_client.py]
# MODEL is a directory under MODELS (default $HOME/models), e.g.
#   SERVER_ARGS=--custom-kernels tests/serve.sh mlx-community/Qwen3.6-35B-A3B-4bit
#   tests/serve.sh mlx-community/Qwen3.6-35B-A3B-4bit --stages single,stream --blocks 4
# for a model of the Hugging Face cache:
#   MODELS=$HOME/.cache/huggingface/hub tests/serve.sh models--X--Y/snapshots/HASH
# IMAGE, LMPY, PORT, SERVER_BIN, SERVER_ARGS and BOXENV as in tests/lm.sh;
# NPORT and CPORT are where the two model servers listen on this Mac (18081
# and 18082), LM_ARGS more options for both of them, SITE=1 puts the
# development build of ./site ahead of the image's mlx, LABEL names the files:
# results/serve-LABEL.json, logs/serve-LABEL-{native,container,bridge}.log,
# CLIENT is another program to run in the place of tests/serve_client.py.
cd "$(dirname "$0")/.."
ROOT=$PWD
export DOCKER_HOST=${DOCKER_HOST:-unix://$HOME/.docker/run/docker.sock}
MODELS=${MODELS:-$HOME/models}
MODEL=$1; shift
IMAGE=${IMAGE:-varco-mlx-lm:latest}
LMPY=${LMPY:-$ROOT/.venv/bin/python}
PORT=${PORT:-47800}
NPORT=${NPORT:-18081}
CPORT=${CPORT:-18082}
LABEL=${LABEL:-$(basename "$MODEL")}
NAME=varco-mlx-serve
mkdir -p logs results
native=
stop() {
  [ -n "$native" ] && kill "$native" 2>/dev/null
  docker logs $NAME > "logs/serve-$LABEL-container.log" 2>&1
  docker rm -f $NAME > /dev/null 2>&1
  ./server/ctl.sh stop > /dev/null
  cp logs/server.log "logs/serve-$LABEL-bridge.log"
}
trap 'stop; exit 1' INT TERM

./server/ctl.sh start --port $PORT --map /models="$MODELS" $SERVER_ARGS || exit 1
"$LMPY" -m mlx_lm server --model "$MODELS/$MODEL" --host 127.0.0.1 --port $NPORT $LM_ARGS \
  > "logs/serve-$LABEL-native.log" 2>&1 &
native=$!
docker rm -f $NAME > /dev/null 2>&1
docker run -d --name $NAME -p 127.0.0.1:$CPORT:8080 -e MLX_REMOTE=host.docker.internal:$PORT $BOXENV \
  ${SITE:+-e PYTHONPATH=/opt/mlx-site -v "$ROOT/site:/opt/mlx-site:ro"} \
  -v "$MODELS:/models:ro" $IMAGE \
  python -m mlx_lm server --model "/models/$MODEL" --host 0.0.0.0 --port 8080 $LM_ARGS > /dev/null
# Until both answer their health check with success (curl -f: an answer with
# an error status is not one).
for port in $NPORT $CPORT; do
  i=0
  until curl -sf -o /dev/null "http://127.0.0.1:$port/health"; do
    i=$((i + 1))
    if [ $i -gt 600 ]; then
      echo "the server on port $port does not answer: see logs/serve-$LABEL-*.log"
      stop
      exit 1
    fi
    sleep 0.2
  done
done
"$LMPY" ${CLIENT:-tests/serve_client.py} --mac http://127.0.0.1:$NPORT --container http://127.0.0.1:$CPORT \
  --out "results/serve-$LABEL.json" "$@"
status=$?
echo "container memory at its peak (the cgroup's memory.peak): $(docker exec $NAME sh -c 'echo $(($(cat /sys/fs/cgroup/memory.peak) >> 20))') MiB"
stop
echo "=== bridge server counters"
grep closed logs/server.log | cut -c1-400
exit $status
