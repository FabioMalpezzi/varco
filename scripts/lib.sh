# Sourced by scripts/setup-mac.sh, scripts/doctor.sh and scripts/autostart.sh,
# which have moved to the root of the repository first. It sets the names
# those three share and the functions they use to look at this Mac. Nothing
# here starts, stops or changes anything.
REPO=$PWD
. mlx/versions.sh

# The engine. Docker Desktop's socket unless DOCKER_HOST names another;
# ENGINE=kiri is Kiri's socket, with the containers started with --gpus and
# a server on the Unix socket Kiri's channel leads to (the same setting
# pytorch/versions.sh reads). The channel leads to that one socket, so one
# bridge's server at a time; and a server must not listen on a TCP port
# there, which every container of Kiri reaches (tcp_servers, below).
ENGINE=${ENGINE:-docker}
case $ENGINE in
  kiri)
    DOCKER_HOST=unix://$HOME/.kiri/run/docker.sock
    GPU_SOCKET=${WHERE:-$HOME/.kiri/run/gpu.sock}
    ;;
  docker)
    DOCKER_HOST=${DOCKER_HOST:-unix://$HOME/.docker/run/docker.sock}
    GPU_SOCKET=
    ;;
  *)
    echo "ENGINE is docker or kiri, not $ENGINE" >&2
    exit 2
    ;;
esac
export DOCKER_HOST

# The MLX bridge.
MLX_PORT=${MLX_PORT:-47800}
MLX_PY=$REPO/mlx/.venv/bin/python
MLX_SERVER=$REPO/mlx/server/mlx-remote-server
MLX_IMAGES="varco-mlx:latest varco-mlx-embeddings:latest varco-mlx-lm:latest"
MODELS=${MODELS:-$HOME/models}

# The PyTorch bridge, for the version TORCH names. The names come from
# pytorch/versions.sh, which expects to be read from its own directory.
TORCH=${TORCH:-2.15.0}
TORCH_PORT=${BRIDGE_PORT:-47700}
TORCH_VENV=$REPO/pytorch/.venv-mac-$TORCH
TORCH_PY=$TORCH_VENV/bin/python
TORCH_IMAGE=${IMAGE:-varco-torch:$TORCH}
TORCH_PYTHON=$(cd pytorch && ENGINE=docker . ./versions.sh > /dev/null 2>&1 && echo "$PYTHON")

# One small computation for each bridge, which gives the same line natively
# and through the bridge: a value and the device it was computed on.
MLX_SMOKE='import mlx.core as mx; a = mx.arange(6.0).reshape(2, 3); print((a @ a.T + 1).sum().item(), mx.default_device())'
TORCH_SMOKE='import torch; a = torch.arange(6.0, device="mps").reshape(2, 3); print((a @ a.T + 1).sum().item(), a.device)'

# The launchd agents of scripts/autostart.sh.
AGENTS_DIR=$HOME/Library/LaunchAgents
MLX_AGENT=varco.mlx-server
TORCH_AGENT=varco.pytorch-server

# The process that listens on TCP port $1 of this Mac: its pid, or nothing.
# lsof shows the processes of this user only; port_open says whether anything
# at all answers.
listener() {
  lsof -nP -iTCP:"$1" -sTCP:LISTEN -t 2> /dev/null | head -1
}
port_open() {
  nc -z 127.0.0.1 "$1" > /dev/null 2>&1
}
# The full command line of process $1, and a short form of it for messages:
# the program without its directory, cut at 80 characters.
command_of() {
  ps -o command= -p "$1" 2> /dev/null
}
short_command() {
  command_of "$1" | sed 's#^[^ ]*/##' | cut -c1-80
}
# The TCP ports process $1 listens on, on one line.
ports_of() {
  lsof -nP -a -p "$1" -iTCP -sTCP:LISTEN -Fn 2> /dev/null | sed -n 's/^n.*:\([0-9]*\)$/\1/p' | sort -u | tr '\n' ' ' | sed 's/ $//'
}
# The Unix sockets process $1 is bound to, on one line. lsof names alike the
# socket a process listens on and each connection it accepted there.
sockets_of() {
  lsof -nP -a -p "$1" -U -Fn 2> /dev/null | sed -n 's/^n\(\/.*\)$/\1/p' | sort -u | tr '\n' ' ' | sed 's/ $//'
}
# Is process $1 bound to the Unix socket $2?
holds_socket() {
  lsof -nP -a -p "$1" -U -Fn 2> /dev/null | grep -qxF "n$2"
}
# The process of this user that listens on the Unix socket $1: its pid, or
# nothing. A program's process of the PyTorch bridge (bridge.session) holds a
# connection that was accepted there, not the socket, and is passed over. A
# socket's file that no process listens on was left by a server that was
# killed: the next server replaces it.
socket_listener() {
  for holder in $(lsof -nP -U -Fpn 2> /dev/null | awk -v name="n$1" '/^p/ { pid = substr($0, 2) } $0 == name && pid != last { print pid; last = pid }'); do
    case "$(command_of "$holder")" in
      *" -m bridge.session"*) continue ;;
    esac
    echo "$holder"
    return 0
  done
}
# Is process $1 an MLX server, or a PyTorch bridge server? Told by the
# program that runs, not by a name anywhere in a command line: a shell or an
# editor that only mentions the server is not one. A server started from a
# copy of the binary with another name (SERVER_BIN of mlx/server/ctl.sh) is
# known by the pid file its script kept.
is_mlx_server() {
  case "$(ps -o comm= -p "$1" 2> /dev/null)" in
    *mlx-remote-server) return 0 ;;
  esac
  for file in "$REPO"/mlx/logs/*.pid; do
    [ -f "$file" ] && [ "$(cat "$file")" = "$1" ] && return 0
  done
  return 1
}
is_torch_server() {
  case "$(ps -o comm= -p "$1" 2> /dev/null)" in
    *[Pp]ython*) ;;
    *) return 1 ;;
  esac
  case "$(command_of "$1")" in
    *" -m bridge.server"*) return 0 ;;
  esac
  return 1
}
# The PyTorch version a bridge server runs, read from the environment its
# command line names (.venv-mac-VERSION).
torch_of_server() {
  command_of "$1" | sed -n 's/.*\.venv-mac-\([0-9.]*\)\/.*/\1/p'
}
# The pids of every MLX server and of every PyTorch bridge server of this user.
mlx_servers() {
  {
    ps -xo pid=,comm= | awk '/mlx-remote-server$/ { print $1 }'
    for file in "$REPO"/mlx/logs/*.pid; do
      [ -f "$file" ] && kill -0 "$(cat "$file")" 2> /dev/null && cat "$file"
    done
  } | sort -un
}
torch_servers() {
  for pid in $(ps -xo pid=,command= | awk '/ -m bridge\.server/ { print $1 }'); do
    is_torch_server "$pid" && echo "$pid"
  done
}

# The bridge servers of this user that listen on a TCP port, one on each
# line: the pid, MLX or PyTorch, and the ports. With Kiri each is a way
# around the channel: Kiri forwards host.docker.internal:PORT of every
# container to the Mac's loopback, whether it was started with --gpus or not.
tcp_servers() {
  for server in $(mlx_servers); do
    open=$(ports_of "$server")
    [ -z "$open" ] || echo "$server MLX $open"
  done
  for server in $(torch_servers); do
    open=$(ports_of "$server")
    [ -z "$open" ] || echo "$server PyTorch $open"
  done
}

# $1 as one word of a command that is printed to be pasted: as it is when
# every character is one the shell takes as written, between double quotes
# otherwise (a path with a space in it).
quoted() {
  case $1 in
    *[!A-Za-z0-9_./:=@%+,-]*) printf '"%s"' "$(printf '%s' "$1" | sed 's/[\\"$`]/\\&/g')" ;;
    *) printf '%s' "$1" ;;
  esac
}

# The version of package $2 in the Python $1, without importing the package.
version_in() {
  "$1" -c "import importlib.metadata as m, sys; print(m.version(sys.argv[1]))" "$2" 2> /dev/null
}
# The same in the image $1.
version_in_image() {
  docker run --rm -e BRIDGE=0 "$1" python -c "import importlib.metadata as m, sys; print(m.version(sys.argv[1]))" "$2" 2> /dev/null
}

# What an image is built from, reduced to sixteen hex digits. setup-mac.sh
# writes it in the label varco.inputs of the images it builds, so that an
# image built from another state of the repository can be told.
mlx_inputs() {
  (cd "$REPO/mlx" && cat versions.sh patches/*.patch docker/Dockerfile docker/Dockerfile.embeddings docker/Dockerfile.lm | shasum -a 256 | cut -c1-16)
}
torch_inputs() {
  (cd "$REPO/pytorch" && { echo "$TORCH $TORCH_PYTHON"; cat docker/Dockerfile docker/requirements-extra.txt bridge/native.cpp "bridge/ops/mps_ops-$TORCH.inc" bridge/*.py; } 2> /dev/null | shasum -a 256 | cut -c1-16)
}
have_image() {
  docker image inspect "$1" > /dev/null 2>&1
}
# Kiri's "docker image inspect" leaves an image's labels out. There the
# label is read from the image's own configuration, as "docker save" writes
# it (a third of a second for an image of 360 MB).
label_of() {
  label=$(docker image inspect --format '{{ index .Config.Labels "varco.inputs" }}' "$1" 2> /dev/null)
  case $label in
    "" | "<no value>")
      label=
      if [ "$ENGINE" = kiri ] && have_image "$1"; then
        label=$(docker save "$1" 2> /dev/null | python3 -c '
import json, sys, tarfile
found = ""
with tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as archive:
    for member in archive:
        if not member.isfile() or member.size > 1 << 20:
            continue
        try:
            doc = json.load(archive.extractfile(member))
        except ValueError:
            continue
        labels = doc.get("config") if isinstance(doc, dict) else None
        labels = labels.get("Labels") if isinstance(labels, dict) else None
        if isinstance(labels, dict) and isinstance(labels.get("varco.inputs"), str):
            found = labels["varco.inputs"]
print(found)' 2> /dev/null)
      fi
      ;;
  esac
  echo "$label"
}

# Does the engine answer? Sets ENGINE_NAME and ENGINE_VERSION when it does.
engine_up() {
  ENGINE_VERSION=$(docker version --format '{{.Server.Version}}' 2> /dev/null) || return 1
  [ -n "$ENGINE_VERSION" ] || return 1
  ENGINE_NAME=$(docker info --format '{{.OperatingSystem}}' 2> /dev/null)
  [ "$ENGINE" = kiri ] && ENGINE_NAME=Kiri
  return 0
}

# Run the small computation in a container of image $1, through the bridge.
# Docker Desktop: the container reaches the server's TCP port. Kiri: the
# container is started with a GPU request and finds its bridge by itself, at
# the address Kiri gives it (KIRI_GPU_BRIDGE, 127.0.0.1:47700).
mlx_in_container() {
  if [ "$ENGINE" = kiri ]; then
    docker run --rm --gpus all "$1" python -c "$MLX_SMOKE" 2>&1
  else
    docker run --rm -e MLX_REMOTE=host.docker.internal:"$MLX_PORT" "$1" python -c "$MLX_SMOKE" 2>&1
  fi
}
torch_in_container() {
  if [ "$ENGINE" = kiri ]; then
    docker run --rm --gpus all "$1" python -c "$TORCH_SMOKE" 2>&1
  else
    docker run --rm -e BRIDGE_PORT="$TORCH_PORT" "$1" python -c "$TORCH_SMOKE" 2>&1
  fi
}

# The launchd agent $1 of this user: "absent" (no file, not loaded),
# "unloaded" (a file that launchd has not loaded), "loaded" and the pid of
# its process, or "loaded -" when launchd has it and no process is running.
agent_file() {
  echo "$AGENTS_DIR/$1.plist"
}
agent_state() {
  if launchctl print "gui/$(id -u)/$1" > /dev/null 2>&1; then
    pid=$(launchctl print "gui/$(id -u)/$1" 2> /dev/null | sed -n 's/^[[:space:]]*pid = \([0-9]*\)$/\1/p' | head -1)
    echo "loaded ${pid:--}"
  elif [ -f "$(agent_file "$1")" ]; then
    echo unloaded
  else
    echo absent
  fi
}
# The program an agent's file names: the first string of its ProgramArguments.
agent_program() {
  [ -f "$(agent_file "$1")" ] || return 0
  /usr/libexec/PlistBuddy -c "Print :ProgramArguments:0" "$(agent_file "$1")" 2> /dev/null
}
# Where an agent's file tells its server to listen: "port N" or "socket PATH",
# read from its ProgramArguments (--port N or --unix PATH for the MLX server,
# the word after bridge.server for the PyTorch one); nothing without a file.
agent_where() {
  [ -f "$(agent_file "$1")" ] || return 0
  index=1
  previous=
  told=
  while word=$(/usr/libexec/PlistBuddy -c "Print :ProgramArguments:$index" "$(agent_file "$1")" 2> /dev/null); do
    case $previous in
      --port) told="port $word" ;;
      --unix) told="socket $word" ;;
      bridge.server)
        case $word in
          *[!0-9]*) told="socket $word" ;;
          *) told="port $word" ;;
        esac
        ;;
    esac
    previous=$word
    index=$((index + 1))
  done
  echo "$told"
}
