# Sourced by the run.sh of every example. Sets where things are, and gives
# the functions that start and stop the Mac end of each bridge. A server
# started here is stopped by its pid, kept in a file by the bridge's own
# script: nothing else on the Mac is touched.
# The caller has moved to its own directory, examples/NN-name.
REPO=$(cd ../.. && pwd)
die() { echo "$*" >&2; exit 1; }
# The engine: the one the docker command reaches by itself (Docker Desktop
# on a Mac; DOCKER_HOST or the current docker context names another, and is
# left as it is: on an NVIDIA server it is that machine's own engine), or
# with ENGINE=kiri a Kiri built with the patches of kiri/. In Kiri a
# container is started with a GPU request and finds its bridge at
# 127.0.0.1:47700 in its own network, and the bridge's server listens on the
# Unix socket Kiri's channel leads to (WHERE, default ~/.kiri/run/gpu.sock)
# and on no TCP port: one socket, so one bridge's server at a time.
export ENGINE="${ENGINE:-docker}"
case $ENGINE in
  kiri)
    export DOCKER_HOST="unix://$HOME/.kiri/run/docker.sock"
    GPU_SOCKET=${WHERE:-$HOME/.kiri/run/gpu.sock}
    GPUS="--gpus all"
    ;;
  docker)
    GPU_SOCKET=
    GPUS=
    ;;
  *) die "ENGINE is docker or kiri, not $ENGINE" ;;
esac
# docker compose for the example's service. In Kiri the GPU request of the
# example's compose.gpu.yaml is added: what docker run --gpus all sends.
compose() {
  if [ "$ENGINE" = kiri ]; then
    docker compose -f compose.yaml -f compose.gpu.yaml "$@"
  else
    docker compose "$@"
  fi
}
# The directory of the MLX models, seen as /models in the containers.
export MODELS="${MODELS:-$HOME/models}"
# The Hugging Face cache, where the PyTorch examples read their models.
export HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
# The PyTorch version of the image and of the Mac's environment.
export TORCH="${TORCH:-2.15.0}"
MLX_PY=$REPO/mlx/.venv/bin/python
TORCH_PY=$REPO/pytorch/.venv-mac-$TORCH/bin/python

# In front of the commands the examples print, so that a command pasted from
# one builds for the engine the example was run with.
[ "$ENGINE" = kiri ] && WITH_ENGINE="ENGINE=kiri " || WITH_ENGINE=

need_image() {
  docker image inspect "$1" > /dev/null 2>&1 || die "the image $1 is not built: run $WITH_ENGINE$2"
}

need_model() {
  [ -d "$MODELS/$1" ] || die "no model at $MODELS/$1: download it with
  hf download $1 --local-dir \"\$MODELS/$1\"
(MODELS is $MODELS; set it to the directory that holds your models)"
}

# The MLX bridge's server, on port MLX_PORT (default 47800), or in Kiri on
# the socket, reading the weights under MODELS by itself. More options may
# follow (--custom-kernels). The containers of the examples look for it
# there unless MLX_REMOTE says otherwise: on Docker Desktop they are told
# the port, and in Kiri nothing, since a container started with a GPU
# request finds the server by the address Kiri gives it (KIRI_GPU_BRIDGE).
MLX_PORT=${MLX_PORT:-47800}
if [ "$ENGINE" != kiri ]; then
  export MLX_REMOTE="${MLX_REMOTE:-host.docker.internal:$MLX_PORT}"
fi
# A server that is already listening there and was not started by an example
# (scripts/autostart.sh keeps one) is used as it is and left running, unless
# the example needs a server with options of its own (MLX_OWN_SERVER=1 says
# so for an example that needs one even without options).
MLX_BORROWED=
start_mlx_server() {
  [ -x "$REPO/mlx/server/mlx-remote-server" ] || die "the MLX server is not built: run ${WITH_ENGINE}scripts/setup-mac.sh mlx"
  SERVER_NAME=example "$REPO/mlx/server/ctl.sh" stop > /dev/null
  if [ "$ENGINE" = kiri ]; then
    holder=$(lsof -nP -U -Fpn 2> /dev/null | awk -v s="n$GPU_SOCKET" '/^p/ { pid = substr($0, 2) } $0 == s { print pid; exit }')
    listens="on $GPU_SOCKET"
    elsewhere="Stop that server first (ENGINE=kiri scripts/autostart.sh uninstall mlx, if it is the one kept at login): in Kiri one socket has one server."
  else
    holder=$(lsof -nP -iTCP:"$MLX_PORT" -sTCP:LISTEN -t 2> /dev/null | head -1)
    listens="on port $MLX_PORT"
    # The example as it was called, as one word when its path has a space.
    case $0 in *[!A-Za-z0-9_./-]*) again="\"$0\"" ;; *) again=$0 ;; esac
    elsewhere="Run the example on another port:  MLX_PORT=$((MLX_PORT + 10)) $again
or stop that server first (scripts/autostart.sh uninstall mlx, if it is the one kept at login)."
  fi
  if [ -n "$holder" ]; then
    case "$(ps -o command= -p "$holder")" in
      *mlx-remote-server*)
        if [ $# -gt 0 ] || [ -n "$MLX_OWN_SERVER" ]; then
          die "this example starts a server of its own${*:+ with $*}, and an MLX server is already listening $listens (process $holder).
$elsewhere"
        fi
        echo "using the MLX server already listening $listens (process $holder); it is left running"
        MLX_BORROWED=1
        return
        ;;
    esac
  fi
  if [ "$ENGINE" = kiri ]; then
    SERVER_NAME=example "$REPO/mlx/server/ctl.sh" start --unix "$GPU_SOCKET" --map /models="$MODELS" "$@" \
      || die "the MLX server did not start: see mlx/logs/server-example.log"
  else
    SERVER_NAME=example "$REPO/mlx/server/ctl.sh" start --port "$MLX_PORT" --map /models="$MODELS" "$@" \
      || die "the MLX server did not start: see mlx/logs/server-example.log"
  fi
}
stop_mlx_server() {
  [ -n "$MLX_BORROWED" ] || SERVER_NAME=example "$REPO/mlx/server/ctl.sh" stop > /dev/null
}

# The PyTorch bridge's server, on port 47700, or in Kiri on the socket
# (pytorch/server.sh reads ENGINE).
start_torch_server() {
  [ -x "$TORCH_PY" ] || die "the PyTorch bridge's Mac end is not built: run ${WITH_ENGINE}scripts/setup-mac.sh pytorch"
  "$REPO/pytorch/server.sh" start || die "the PyTorch bridge's server did not start: see pytorch/runs/server.log"
}
stop_torch_server() {
  "$REPO/pytorch/server.sh" stop
}
