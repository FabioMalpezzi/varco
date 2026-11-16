#!/bin/sh
# An OpenAI-compatible model server in a container, on the Mac's GPU.
#   ./run.sh [MODEL]         starts the bridge's server and the model server,
#                            runs client.py against it, and stops both
#   ./run.sh MODEL --keep    leaves both running, for your own requests:
#                            curl http://127.0.0.1:18080/v1/chat/completions ...
#   ./run.sh --stop          stops what --keep left running
# MODEL is a directory under MODELS (default Qwen/Qwen3.5-0.8B-Base, a small
# model; mlx-community/Qwen3.6-35B-A3B-4bit is the large one the measurements
# were taken with). KERNELS=1 starts the bridge's server with
# --custom-kernels, which the large model needs for the Mac's own text and
# speed: read docs/security.md first (any other value, KERNELS=0 too, leaves
# them off). LLM_PORT is the port on the Mac (18080).
# ENGINE=kiri runs the model server in Kiri, with the GPU request of
# compose.gpu.yaml.
cd "$(dirname "$0")" || exit 1
. ../common.sh
if [ "$1" = --stop ]; then
  compose down
  stop_mlx_server
  exit 0
fi
export MODEL=${1:-Qwen/Qwen3.5-0.8B-Base}
export LLM_PORT=${LLM_PORT:-18080}
need_model "$MODEL"
need_image varco-mlx-lm:latest "scripts/setup-mac.sh mlx"
# KERNELS=1 and nothing else turns custom kernels on, as in scripts/autostart.sh.
kernels=
[ "${KERNELS:-}" = 1 ] && kernels=--custom-kernels
start_mlx_server $kernels
stop() { compose down; stop_mlx_server; }
[ "$2" = --keep ] || trap stop EXIT
compose up -d || exit 1
i=0
# Until the model server answers its health check with success: an answer
# with an error status is not one (curl -f).
until curl -sf -o /dev/null "http://127.0.0.1:$LLM_PORT/health"; do
  i=$((i + 1))
  if [ $i -gt 600 ]; then
    compose logs --tail 20 llm
    die "the model server does not answer on port $LLM_PORT"
  fi
  sleep 0.2
done
python3 client.py "http://127.0.0.1:$LLM_PORT"
if [ "$2" = --keep ]; then
  echo "left running: the model server on http://127.0.0.1:$LLM_PORT and the bridge's server; ./run.sh --stop stops them"
fi
