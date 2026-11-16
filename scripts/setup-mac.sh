#!/bin/sh
# Install varco on this Mac, in one command: check what it needs, make the
# Python environments, build the two servers and the container images, run
# one small computation through each bridge, and say what is ready.
#   scripts/setup-mac.sh              both bridges
#   scripts/setup-mac.sh mlx          the MLX bridge only
#   scripts/setup-mac.sh pytorch      the PyTorch bridge only (TORCH=2.15.0 by default)
#   scripts/setup-mac.sh --rebuild    build the images again from nothing (docker build --no-cache)
#   scripts/setup-mac.sh --no-test    build, and leave out the computation at the end
# It can be run again at any time: a step whose result is already there and
# was made from the files as they are now is skipped, so a second run only
# checks. After "git pull", run it again: it rebuilds what the pull changed.
# The images are built on this Mac; none is downloaded from a registry.
# ENGINE=kiri builds and tests in Kiri instead of Docker Desktop: a Kiri
# built with the patches of kiri/, its daemon running (docs/setup.md).
cd "$(dirname "$0")/.." || exit 1
. scripts/lib.sh
export ENGINE TORCH
# In front of the commands this script prints, so that a command pasted from
# it looks at the engine this run was for.
[ "$ENGINE" = kiri ] && with_engine="ENGINE=kiri " || with_engine=

parts=
rebuild=
run_test=yes
for arg in "$@"; do
  case $arg in
    mlx | pytorch) parts="$parts $arg" ;;
    --rebuild) rebuild=--no-cache ;;
    --no-test) run_test= ;;
    *)
      echo "usage: setup-mac.sh [mlx] [pytorch] [--rebuild] [--no-test]" >&2
      exit 2
      ;;
  esac
done
parts=${parts:-mlx pytorch}

ok() { printf 'ok       %s\n' "$*"; }
note() { printf 'note     %s\n' "$*"; }
missing=0
miss() {
  printf 'MISSING  %s\n' "$1"
  printf '         -> %s\n' "$2"
  missing=$((missing + 1))
}
began=$(date +%s)
mkdir -p mlx/logs pytorch/runs
# Run a step with its output kept in a file: the step's name and how long it
# took are printed, and the end of the output if it fails.
step() {
  what=$1
  log=$2
  shift 2
  printf '...      %s\n' "$what"
  start=$(date +%s)
  if "$@" > "$log" 2>&1; then
    printf 'done     %s (%s s)\n' "$what" $(($(date +%s) - start))
  else
    printf 'FAILED   %s\n' "$what"
    tail -25 "$log"
    echo "The whole output is in $log. ${with_engine}scripts/doctor.sh says what is wrong with what is installed; docs/troubleshooting.md lists the messages."
    exit 1
  fi
}

echo "== What varco needs"
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  if [ "$(sysctl -n sysctl.proc_translated 2> /dev/null)" = 1 ]; then
    miss "a native shell: this one runs under Rosetta" "open a terminal that is not set to open with Rosetta, and run this again"
  else
    miss "a Mac with Apple silicon: this is $(uname -s) on $(uname -m)" "varco computes on the Apple GPU through Metal; there is nothing to install on this machine"
  fi
else
  macos=$(sw_vers -productVersion)
  if [ "${macos%%.*}" -lt 14 ]; then
    miss "macOS 14 or later: this is macOS $macos" "update macOS: the MLX $MLX_VERSION and PyTorch $TORCH packages for the Mac are built for macOS 14 and later"
  else
    ok "$(sysctl -n machdep.cpu.brand_string), macOS $macos"
    [ "${macos%%.*}" = 26 ] || note "everything was built and measured on macOS 26.6.2 only"
  fi
fi
if xcode-select -p > /dev/null 2>&1 && command -v clang++ > /dev/null; then
  ok "the compiler: $(clang++ --version 2> /dev/null | head -1)"
else
  miss "the Xcode command line tools (clang++), to build the two servers" "xcode-select --install"
fi
if xcode-select -p > /dev/null 2>&1 && command -v git > /dev/null; then
  ok "$(git --version)"
else
  miss "git, to fetch the MLX source the patch applies to" "xcode-select --install   (git comes with the Xcode command line tools)"
fi
if command -v uv > /dev/null; then
  if uv python find 3.12 > /dev/null 2>&1; then
    ok "$(uv --version | cut -d' ' -f1-2), with Python 3.12 for the environments"
  else
    ok "$(uv --version | cut -d' ' -f1-2); it will download Python 3.12 for the environments"
  fi
else
  miss "uv, which makes the Python environments on the Mac" "curl -LsSf https://astral.sh/uv/install.sh | sh   (or: brew install uv), then open a new terminal"
fi
if ! command -v docker > /dev/null; then
  miss "the docker command" "install Docker Desktop (https://www.docker.com/products/docker-desktop/) and open it"
elif engine_up; then
  ok "the container engine: $ENGINE_NAME, engine $ENGINE_VERSION, at $DOCKER_HOST"
  docker compose version > /dev/null 2>&1 || note "docker compose is not there: the examples need it"
elif [ "$ENGINE" = kiri ]; then
  miss "Kiri's daemon: nothing answers at $DOCKER_HOST" "build Kiri with the patches and start its daemon (kiri/README.md)"
else
  miss "a container engine that answers at $DOCKER_HOST" "open Docker Desktop (open -a Docker) and wait until it is running; for another engine set DOCKER_HOST"
fi
free=$(df -k "$REPO" | awk 'NR == 2 { print int($4 / 1048576) }')
[ "$free" -ge 10 ] || note "$free GB free on this disk: the images take about 4 GB and the two environments 1.7 GB"
command -v hf > /dev/null || note "the hf command is not installed: scripts/fetch-models.sh needs it to download the examples' models (uv tool install huggingface_hub)"
if [ $missing -gt 0 ]; then
  echo "$missing thing(s) missing: nothing was built. Do what the arrows say and run ${with_engine}scripts/setup-mac.sh again."
  exit 1
fi

# Stop on the way out whatever server a test started.
started_mlx=
started_torch=
cleanup() {
  [ -n "$started_mlx" ] && SERVER_NAME=setup mlx/server/ctl.sh stop > /dev/null
  [ -n "$started_torch" ] && pytorch/server.sh stop
  started_mlx=
  started_torch=
}
trap cleanup EXIT
trap 'exit 130' INT TERM

setup_mlx() {
  echo "== MLX bridge (MLX $MLX_VERSION)"
  # mlx/setup-mac.sh writes packages-installed when uv has installed every
  # package: an environment without it was left by a run that was interrupted.
  if [ -x "$MLX_PY" ] && [ -f mlx/.venv/packages-installed ] && [ "$(version_in "$MLX_PY" mlx)" = "$MLX_VERSION" ] \
    && [ "$(version_in "$MLX_PY" mlx-lm)" = "$MLX_LM_VERSION" ] \
    && [ "$(version_in "$MLX_PY" mlx-embeddings)" = "$MLX_EMBEDDINGS_VERSION" ]; then
    ok "the Mac's environment is there: mlx/.venv, mlx $MLX_VERSION"
  else
    step "the Mac's environment: mlx/.venv with mlx $MLX_VERSION, mlx-lm and mlx-embeddings" mlx/logs/setup-env.log mlx/setup-mac.sh
  fi

  if [ ! -e mlx/upstream ]; then
    step "the MLX source at $MLX_TAG with the patch applied: mlx/upstream" mlx/logs/setup-source.log mlx/fetch-source.sh
  elif git -C mlx/upstream apply --cached --reverse --check "$REPO"/mlx/patches/*.patch 2> /dev/null; then
    ok "the patched MLX source is there: mlx/upstream"
  else
    echo "FAILED   mlx/upstream is not MLX $MLX_TAG with mlx/patches/ applied as they are now."
    echo "         -> if you are not working on that tree: mlx/fetch-source.sh --force   (it removes mlx/upstream and fetches it again), then run this again"
    exit 1
  fi

  if [ -x "$MLX_SERVER" ] && [ "$MLX_SERVER" -nt mlx/server/mlx_remote_server.cpp ] \
    && [ "$MLX_SERVER" -nt mlx/upstream/mlx/backend/remote/protocol.h ] \
    && [ "$MLX_SERVER" -nt mlx/.venv/pyvenv.cfg ]; then
    ok "the server is built: mlx/server/mlx-remote-server"
  else
    step "the server: mlx/server/mlx-remote-server" mlx/logs/setup-server.log mlx/server/build.sh
  fi

  inputs=$(mlx_inputs)
  stale=$rebuild
  for image in $MLX_IMAGES; do
    [ "$(label_of "$image")" = "$inputs" ] || stale=yes
  done
  if [ -z "$stale" ]; then
    ok "the images are built from the files as they are now: $MLX_IMAGES"
  else
    step "the three images, built here (about two and a half minutes from nothing; they compile MLX for Linux)" mlx/logs/setup-images.log \
      mlx/build-images.sh $rebuild --label "varco.inputs=$inputs"
  fi

  [ -n "$run_test" ] || return 0
  if [ "$ENGINE" = kiri ]; then
    # On the socket Kiri's channel leads to, and on no port. A server that
    # starts there removes the socket it finds: one that is listening is
    # used if it is an MLX server, and is in the way if it is anything else.
    holder=$(socket_listener "$GPU_SOCKET")
    if [ -n "$holder" ] && is_mlx_server "$holder"; then
      note "an MLX server is already listening on $GPU_SOCKET (process $holder): the test uses it"
    elif [ -n "$holder" ]; then
      echo "FAILED   $GPU_SOCKET, where Kiri's channel leads, is taken by something that is not an MLX server (process $holder: $(short_command "$holder")), and one socket has one server."
      echo "         -> stop it (ENGINE=kiri scripts/doctor.sh says how), then run this again"
      exit 1
    else
      SERVER_NAME=setup mlx/server/ctl.sh start --unix "$GPU_SOCKET" > mlx/logs/setup-test.log 2>&1 || {
        echo "FAILED   the MLX server did not start on $GPU_SOCKET:"
        cat mlx/logs/setup-test.log
        exit 1
      }
      started_mlx=yes
    fi
  else
    holder=$(listener "$MLX_PORT")
    if [ -n "$holder" ] && is_mlx_server "$holder"; then
      note "an MLX server is already listening on port $MLX_PORT (process $holder): the test uses it"
    elif [ -n "$holder" ] || port_open "$MLX_PORT"; then
      echo "FAILED   port $MLX_PORT is taken by something that is not an MLX server${holder:+ (process $holder: $(short_command "$holder"))}."
      echo "         -> stop it, or give the bridge another port: MLX_PORT=$((MLX_PORT + 10)) scripts/setup-mac.sh mlx"
      exit 1
    else
      SERVER_NAME=setup mlx/server/ctl.sh start --port "$MLX_PORT" > mlx/logs/setup-test.log 2>&1 || {
        echo "FAILED   the MLX server did not start:"
        cat mlx/logs/setup-test.log
        exit 1
      }
      started_mlx=yes
    fi
  fi
  native=$("$MLX_PY" -c "$MLX_SMOKE" 2>&1)
  box=$(mlx_in_container varco-mlx:latest)
  cleanup
  if [ -n "$native" ] && [ "$box" = "$native" ]; then
    ok "a computation in a container gave what the Mac gives: $box"
    ready="$ready mlx"
  else
    echo "FAILED   the test of the MLX bridge. On the Mac:"
    echo "$native"
    echo "         in a container of varco-mlx:latest:"
    echo "$box"
    echo "         -> ${with_engine}scripts/doctor.sh mlx looks at each part"
    exit 1
  fi
}

setup_pytorch() {
  echo "== PyTorch bridge (PyTorch $TORCH)"
  if [ ! -f "pytorch/bridge/ops/mps_ops-$TORCH.inc" ]; then
    note "PyTorch $TORCH is not the version the bridge was built for (2.15.0): its operator list is made here, and nothing about it was tested"
  fi
  executor=$(ls "pytorch/bridge/_build/$TORCH"/_executor*.so 2> /dev/null | head -1)
  # build-mac.sh copies the list of packages into the environment when uv has
  # installed them all: without the copy the environment was left by a run
  # that was interrupted, and with another list the packages have changed.
  if [ -x "$TORCH_PY" ] && [ "$(version_in "$TORCH_PY" torch)" = "$TORCH" ] \
    && cmp -s pytorch/docker/requirements-extra.txt "$TORCH_VENV/requirements-extra.txt" \
    && [ -n "$executor" ] && [ "$executor" -nt pytorch/bridge/executor.cpp ]; then
    ok "the Mac end is there: pytorch/.venv-mac-$TORCH, torch $TORCH, and the compiled executor"
  else
    step "the Mac end: pytorch/.venv-mac-$TORCH with torch $TORCH, the operator list, the compiled executor" pytorch/runs/setup-mac.log pytorch/build-mac.sh
  fi

  inputs=$(torch_inputs)
  if [ -z "$rebuild" ] && [ "$(label_of "$TORCH_IMAGE")" = "$inputs" ]; then
    ok "the image is built from the files as they are now: $TORCH_IMAGE"
  else
    step "the image $TORCH_IMAGE, built here (about a minute from nothing, most of it downloading PyTorch)" pytorch/runs/setup-image.log \
      pytorch/build-image.sh $rebuild --label "varco.inputs=$inputs"
  fi

  [ -n "$run_test" ] || return 0
  # With Kiri pytorch/server.sh starts the server on the socket Kiri's
  # channel leads to (ENGINE is exported), and removes the socket it finds:
  # a server that is listening there is used if it is this PyTorch's, and is
  # in the way if it is anything else.
  holder=
  [ "$ENGINE" != kiri ] || holder=$(socket_listener "$GPU_SOCKET")
  if [ -n "$holder" ] && is_torch_server "$holder" && [ "$(torch_of_server "$holder")" = "$TORCH" ]; then
    note "a bridge server for PyTorch $TORCH is already listening on $GPU_SOCKET (process $holder): the test uses it"
  elif [ -n "$holder" ]; then
    echo "FAILED   $GPU_SOCKET, where Kiri's channel leads, is taken by something that is not a bridge server for PyTorch $TORCH (process $holder: $(short_command "$holder")), and one socket has one server."
    echo "         -> stop it (ENGINE=kiri scripts/doctor.sh says how), then run this again"
    exit 1
  else
    if ! pytorch/server.sh start > pytorch/runs/setup-test.log 2>&1; then
      echo "FAILED   the PyTorch bridge's server did not start:"
      cat pytorch/runs/setup-test.log
      exit 1
    fi
    if grep -q "not started by this script" pytorch/runs/setup-test.log; then
      note "a bridge server for PyTorch $TORCH is already listening on port $TORCH_PORT: the test uses it"
    else
      started_torch=yes
    fi
  fi
  native=$("$TORCH_PY" -c "$TORCH_SMOKE" 2>&1)
  box=$(torch_in_container "$TORCH_IMAGE")
  cleanup
  if [ -n "$native" ] && [ "$box" = "$native" ]; then
    ok "a computation in a container gave what the Mac gives: $box"
    ready="$ready pytorch"
  else
    echo "FAILED   the test of the PyTorch bridge. On the Mac:"
    echo "$native"
    echo "         in a container of $TORCH_IMAGE:"
    echo "$box"
    echo "         -> ${with_engine}scripts/doctor.sh pytorch looks at each part"
    exit 1
  fi
}

# Where server $1 listens, for the lines at the end.
listens_on() {
  open=$(ports_of "$1")
  if [ -n "$open" ]; then
    echo "port $open"
    return
  fi
  open=$(sockets_of "$1")
  echo "${open:+socket }${open:-no port or socket}"
}

ready=
for part in $parts; do
  setup_$part
done

echo "== Ready, in $(($(date +%s) - began)) s"
for part in $parts; do
  case "$part,$run_test" in
    mlx,yes) echo "The MLX bridge: the server and the images $MLX_IMAGES; a container computed on this Mac's GPU." ;;
    mlx,) echo "The MLX bridge is built: the server and the images $MLX_IMAGES. Not tested (--no-test)." ;;
    pytorch,yes) echo "The PyTorch bridge: the Mac end and the image $TORCH_IMAGE; a container computed on this Mac's GPU." ;;
    pytorch,) echo "The PyTorch bridge is built: the Mac end and the image $TORCH_IMAGE. Not tested (--no-test)." ;;
  esac
done
# The servers that are running now. Those the tests started are stopped, so
# each of these was running before: one kept from login or started by hand,
# which a test used if it listened where the test looks.
found=$(
  for pid in $(mlx_servers); do echo "the MLX server (process $pid, $(listens_on "$pid"))"; done
  for pid in $(torch_servers); do echo "the PyTorch bridge's server (process $pid, $(listens_on "$pid"))"; done
)
if [ -z "$found" ]; then
  echo "No server is left running: each example starts and stops the one it needs."
else
  echo "$found" | while read -r line; do
    echo "Left running, as it was found: $line."
  done
fi
echo "Next:"
echo "  scripts/fetch-models.sh              the examples' small public models, about 3.3 GB"
if [ "$ENGINE" = kiri ]; then
  case " $parts " in *" mlx "*) echo "  ENGINE=kiri examples/01-embeddings/run.sh        an MLX program in a container, on the GPU" ;; esac
  case " $parts " in *" pytorch "*) echo "  ENGINE=kiri examples/03-pytorch-inference/run.sh a PyTorch program in a container, on the GPU" ;; esac
  echo "  ENGINE=kiri scripts/doctor.sh                    what is installed, what is running, what is wrong"
  echo "  ENGINE=kiri scripts/autostart.sh install mlx     keep one bridge's server (mlx or pytorch) running from login on (optional)"
else
  case " $parts " in *" mlx "*) echo "  examples/01-embeddings/run.sh        an MLX program in a container, on the GPU" ;; esac
  case " $parts " in *" pytorch "*) echo "  examples/03-pytorch-inference/run.sh a PyTorch program in a container, on the GPU" ;; esac
  echo "  scripts/doctor.sh                    what is installed, what is running, what is wrong"
  echo "  scripts/autostart.sh install         keep the servers running from login on (optional)"
fi
