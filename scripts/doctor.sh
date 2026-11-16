#!/bin/sh
# Say what is installed, what is running and what is wrong, with one line of
# remedy under each finding. It changes nothing.
#   scripts/doctor.sh              both bridges
#   scripts/doctor.sh mlx          the MLX bridge only
#   scripts/doctor.sh pytorch      the PyTorch bridge only (TORCH=2.15.0 by default)
#   scripts/doctor.sh --quick      leave out the checks that run a container
# Each line begins with ok, FAULT, off or note. A FAULT is something that
# keeps a container from the GPU or gives it another result than the Mac's;
# "off" is a server that is not running, which is how setup-mac.sh leaves
# them. The exit status is 1 if there is a FAULT, 0 otherwise.
# It reads the same settings as the other scripts: MLX_PORT (47800),
# BRIDGE_PORT (47700), TORCH, DOCKER_HOST, and ENGINE=kiri for Kiri. With
# ENGINE=kiri the servers are
# looked for on the Unix socket Kiri's channel leads to, and a bridge server
# that listens on a TCP port is a FAULT: every container of Kiri reaches a
# port of the Mac, started with --gpus or not.
cd "$(dirname "$0")/.." || exit 1
. scripts/lib.sh
# The option that maps the directory of models and the socket of Kiri's
# channel, as the remedies print them, to be pasted: the person's own MODELS,
# or the default written so that it holds on any Mac, and each as one word
# when its path has a space.
if [ "$MODELS" = "$HOME/models" ]; then models_map='--map "/models=$HOME/models"'; else models_map="--map $(quoted "/models=$MODELS")"; fi
socket_shown=$(quoted "$GPU_SOCKET")

parts=
quick=
for arg in "$@"; do
  case $arg in
    mlx | pytorch) parts="$parts $arg" ;;
    --quick) quick=yes ;;
    *)
      echo "usage: doctor.sh [mlx] [pytorch] [--quick]" >&2
      exit 2
      ;;
  esac
done
parts=${parts:-mlx pytorch}

faults=0
offs=0
ok() { printf 'ok       %s\n' "$*"; }
note() { printf 'note     %s\n' "$*"; }
fault() {
  printf 'FAULT    %s\n' "$1"
  printf '         -> %s\n' "$2"
  faults=$((faults + 1))
}
off() {
  printf 'off      %s\n' "$1"
  printf '         -> %s\n' "$2"
  offs=$((offs + 1))
}
# The settings this run was given, in front of the commands the remedies
# print: pasted without them, a command would look at another engine, image,
# version, port or directory than the one that was diagnosed.
[ "$TORCH" = 2.15.0 ] && with_torch= || with_torch="TORCH=$TORCH "
with_engine=
if [ "$ENGINE" = kiri ]; then
  with_engine="ENGINE=kiri "
  [ "$GPU_SOCKET" = "$HOME/.kiri/run/gpu.sock" ] || with_engine="ENGINE=kiri WHERE=$socket_shown "
fi
[ "$TORCH_IMAGE" = "varco-torch:$TORCH" ] && with_image= || with_image="IMAGE=$(quoted "$TORCH_IMAGE") "
[ "$MODELS" = "$HOME/models" ] && with_models= || with_models="MODELS=$(quoted "$MODELS") "
# A port is a setting of Docker Desktop's servers: with Kiri they listen on
# the socket.
with_mlx_port=
with_torch_port=
if [ "$ENGINE" != kiri ]; then
  [ "$MLX_PORT" = 47800 ] || with_mlx_port="MLX_PORT=$MLX_PORT "
  [ "$TORCH_PORT" = 47700 ] || with_torch_port="BRIDGE_PORT=$TORCH_PORT "
fi
# The commands of the remedies: build, start a server where the containers of
# this engine look for it, keep one running from login.
mlx_setup="${with_engine}${with_mlx_port}scripts/setup-mac.sh mlx"
torch_setup="${with_engine}${with_torch}${with_image}${with_torch_port}scripts/setup-mac.sh pytorch"
if [ "$ENGINE" = kiri ]; then
  mlx_start="mlx/server/ctl.sh start --unix $socket_shown $models_map"
else
  mlx_start="mlx/server/ctl.sh start ${with_mlx_port:+--port $MLX_PORT }$models_map"
fi
torch_start="${with_engine}${with_torch}${with_torch_port}pytorch/server.sh start"
mlx_keep="${with_engine}${with_models}${with_mlx_port}scripts/autostart.sh install mlx"
torch_keep="${with_engine}${with_torch}${with_torch_port}scripts/autostart.sh install pytorch"

# Who started server $1, and the command that stops it: sets $by and $stop.
# $2 is the launchd agent that may keep it, $3 the directory of pid files.
started_by() {
  by="not by this repository's scripts"
  stop="kill $1"
  if [ "$(agent_state "$2")" = "loaded $1" ]; then
    by="kept by autostart"
    case $2 in
      "$MLX_AGENT") stop="scripts/autostart.sh uninstall mlx" ;;
      *) stop="scripts/autostart.sh uninstall pytorch" ;;
    esac
    return
  fi
  for file in "$3"/*.pid; do
    [ -f "$file" ] && [ "$(cat "$file")" = "$1" ] || continue
    by="started by a script (${file#"$REPO"/})"
    case $file in
      */mlx/logs/server.pid) stop="mlx/server/ctl.sh stop" ;;
      */mlx/logs/server-*.pid)
        name=${file##*/server-}
        stop="SERVER_NAME=${name%.pid} mlx/server/ctl.sh stop"
        ;;
      *) stop="pytorch/server.sh stop" ;;
    esac
  done
}

# What the options --map and --models of the command line $1 name, on one
# line. ps joins the arguments with spaces, so a value is read up to the next
# option: a directory with a space in its name is shown whole.
maps_of() {
  echo "$1" | awk '{
    n = split($0, part, / --/)
    out = ""
    for (i = 2; i <= n; i++)
      if (part[i] ~ /^(map|models) /) {
        sub(/^(map|models) /, "", part[i])
        out = out (out == "" ? "" : " ") part[i]
      }
    print out
  }'
}

echo "== This Mac"
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  fault "this is $(uname -s) on $(uname -m), not a Mac with Apple silicon (or the shell runs under Rosetta)" "varco needs the Apple GPU: use a native terminal on an Apple-silicon Mac"
else
  macos=$(sw_vers -productVersion)
  if [ "${macos%%.*}" -lt 14 ]; then
    fault "macOS $macos: the MLX and PyTorch packages for the Mac are built for macOS 14 and later" "update macOS"
  else
    ok "$(sysctl -n machdep.cpu.brand_string), macOS $macos"
  fi
fi
if xcode-select -p > /dev/null 2>&1 && command -v clang++ > /dev/null; then
  ok "the compiler: $(clang++ --version 2> /dev/null | head -1)"
else
  fault "the Xcode command line tools (clang++) are not installed: the servers cannot be built" "xcode-select --install"
fi
if command -v uv > /dev/null; then
  ok "$(uv --version | cut -d' ' -f1-2)"
else
  fault "uv is not installed: the Python environments cannot be made" "curl -LsSf https://astral.sh/uv/install.sh | sh   (or: brew install uv)"
fi

echo "== Container engine"
engine=
if ! command -v docker > /dev/null; then
  fault "the docker command is not installed" "install Docker Desktop (https://www.docker.com/products/docker-desktop/) and open it"
elif engine_up; then
  engine=yes
  if [ "$ENGINE" = kiri ]; then
    ok "Kiri answers at $DOCKER_HOST"
  else
    ok "$ENGINE_NAME, engine $ENGINE_VERSION, answers at $DOCKER_HOST: every container of this engine can reach a server's port"
  fi
elif [ "$ENGINE" = kiri ]; then
  fault "Kiri's daemon does not answer at $DOCKER_HOST" "start the daemon of a Kiri built with kiri/patches/ (kiri/README.md)"
else
  fault "no container engine answers at $DOCKER_HOST" "open Docker Desktop (open -a Docker) and wait until it is running; for another engine set DOCKER_HOST"
fi
[ -n "$engine" ] || note "without the engine, the images and the containers are not checked"

# With Kiri the bridge is for the containers started with --gpus: they reach
# it through the channel, which leads to one Unix socket on the Mac. This
# looks at the servers of both bridges, whichever was asked about, and
# whether Kiri is running now or not.
#   - One socket has one server. Both servers remove the socket they find
#     and bind their own: a second one takes the channel from the first, and
#     the first that stops removes the socket's file.
#   - A server on a TCP port is a way around the channel: Kiri forwards
#     host.docker.internal:PORT of every container to the Mac's loopback.
# It sets $channel_server and $channel_kind: the server the channel leads to.
channel_server=
channel_kind=
doctor_channel() {
  echo "== Kiri's channel"
  bound=
  count=0
  for pid in $(mlx_servers); do
    holds_socket "$pid" "$GPU_SOCKET" || continue
    bound="$bound $pid"
    count=$((count + 1))
    channel_server=$pid
    channel_kind=MLX
  done
  for pid in $(torch_servers); do
    holds_socket "$pid" "$GPU_SOCKET" || continue
    bound="$bound $pid"
    count=$((count + 1))
    channel_server=$pid
    channel_kind=PyTorch
  done
  holder=$(socket_listener "$GPU_SOCKET")
  if [ $count -gt 1 ]; then
    fault "$count servers are bound to $GPU_SOCKET (processes$bound), and one socket has one server: the channel leads to the one that bound it last, and the first that stops removes the socket's file" "stop them all (the lines below say how), then start the one bridge to use with Kiri"
  elif [ $count = 1 ] && [ ! -S "$GPU_SOCKET" ]; then
    fault "the $channel_kind bridge's server (process $channel_server) is bound to $GPU_SOCKET and the socket's file is gone (another server on the same path removed it when it stopped): the channel finds no server" "stop that server and start it again (the lines below say how)"
  elif [ $count = 1 ]; then
    ok "one server listens on $GPU_SOCKET, where the channel leads: the $channel_kind bridge's (process $channel_server)"
  elif [ -n "$holder" ]; then
    fault "$GPU_SOCKET is taken by process $holder ($(short_command "$holder")), which is not a bridge server: the containers started with --gpus do not find the bridge" "stop that process, then start a server of the bridge to use with Kiri"
  elif [ -S "$GPU_SOCKET" ]; then
    note "$GPU_SOCKET is there and no server listens on it: a server that was killed left it, and the next one replaces it"
  else
    note "no server listens on $GPU_SOCKET"
  fi
  exposed=$(tcp_servers)
  if [ -z "$exposed" ]; then
    ok "no bridge server of this user listens on a TCP port: only a container started with --gpus reaches a bridge"
    return
  fi
  while read -r server kind ports; do
    case $kind in
      MLX)
        started_by "$server" "$MLX_AGENT" "$REPO/mlx/logs"
        start="$mlx_start   (or: $mlx_keep)"
        ;;
      *)
        started_by "$server" "$TORCH_AGENT" "$REPO/pytorch/runs"
        start="$torch_start   (or: $torch_keep)"
        ;;
    esac
    fault "a server of the $kind bridge listens on TCP port $ports (process $server, $by): every container of Kiri reaches it as host.docker.internal:$ports, started with --gpus or not" "stop it ($stop); with Kiri a server listens on the socket only: $start"
  done << LIST
$exposed
LIST
}
[ "$ENGINE" != kiri ] || doctor_channel

# Run the small computation natively and in a container and compare.
# $1 is the bridge's name, $2 the Mac's python, $3 its program, $4 the
# function that runs it in a container, $5 the image, $6 where the server is.
round_trip() {
  [ -z "$quick" ] && [ -n "$engine" ] && have_image "$5" && [ -x "$2" ] || return 0
  native=$("$2" -c "$3" 2>&1)
  box=$($4 "$5")
  if [ -n "$native" ] && [ "$box" = "$native" ]; then
    ok "a container of $5 computed through it what the Mac gives: $box"
  else
    fault "a container of $5 did not compute through the server $6; it said: $(echo "$box" | tail -1 | cut -c1-200)" "see the server's log, and docs/troubleshooting.md for that message"
  fi
}

doctor_mlx() {
  echo "== MLX bridge"
  mac=
  if [ ! -x "$MLX_PY" ]; then
    fault "the Mac's environment is not there (mlx/.venv)" "$mlx_setup"
  else
    mac=$(version_in "$MLX_PY" mlx)
    if [ ! -f mlx/.venv/packages-installed ]; then
      fault "the Mac's environment is not complete: mlx/.venv has no packages-installed, which is written when every package is installed (a run of setup-mac.sh was interrupted)" "$mlx_setup"
    elif [ "$mac" = "$MLX_VERSION" ]; then
      ok "the Mac's environment: mlx $mac"
    else
      fault "mlx/.venv has mlx ${mac:-missing}, and the bridge is pinned to $MLX_VERSION" "$mlx_setup"
    fi
  fi
  if [ ! -x "$MLX_SERVER" ]; then
    fault "the server is not built (mlx/server/mlx-remote-server)" "$mlx_setup"
  elif [ mlx/server/mlx_remote_server.cpp -nt "$MLX_SERVER" ] || [ mlx/upstream/mlx/backend/remote/protocol.h -nt "$MLX_SERVER" ]; then
    fault "the server was built before its source last changed" "$mlx_setup"
  else
    ok "the server is built: mlx/server/mlx-remote-server"
  fi

  if [ -n "$engine" ]; then
    inputs=$(mlx_inputs)
    for image in $MLX_IMAGES; do
      if ! have_image "$image"; then
        fault "the image $image is not built" "$mlx_setup"
        continue
      fi
      label=$(label_of "$image")
      if [ -z "$label" ]; then
        note "the image $image was not built by setup-mac.sh: whether it matches the files as they are now is not known ($mlx_setup settles it)"
      elif [ "$label" != "$inputs" ]; then
        fault "the image $image was built from another state of mlx/patches/ and mlx/docker/ than the one here" "$mlx_setup"
        continue
      fi
      [ -z "$quick" ] || { ok "the image $image is built"; continue; }
      inside=$(version_in_image "$image" mlx)
      if [ -n "$mac" ] && [ "$inside" = "$mac" ]; then
        ok "the image $image has the Mac's MLX: $inside"
      elif [ -n "$mac" ]; then
        fault "MLX is ${inside:-missing} in the image $image and $mac on the Mac: the two ends must run the same version" "$mlx_setup --rebuild"
      else
        ok "the image $image is built, with mlx $inside"
      fi
    done
  fi

  # The servers that are running, and the one containers find.
  found=
  elsewhere=
  for pid in $(mlx_servers); do
    line=$(command_of "$pid")
    started_by "$pid" "$MLX_AGENT" "$REPO/mlx/logs"
    ports=$(ports_of "$pid")
    sockets=$(sockets_of "$pid")
    if [ -n "$ports" ]; then
      where="port $ports"
    elif [ -n "$sockets" ]; then
      where="the socket $sockets"
    else
      where="no port or socket yet"
    fi
    case $line in
      *--custom-kernels*) kernels="custom Metal kernels ON" ;;
      *) kernels="custom Metal kernels off" ;;
    esac
    case $line in
      *"--map "* | *"--models "*) reads="reads weights under $(maps_of "$line")" ;;
      *) reads="reads no weight files itself (no --map or --models): containers upload them, which is slower to load" ;;
    esac
    if [ "$ENGINE" = kiri ]; then
      [ -z "$ports" ] && holds_socket "$pid" "$GPU_SOCKET" && found=$pid
    else
      [ "$where" = "port $MLX_PORT" ] && found=$pid
    fi
    if [ "$found" = "$pid" ]; then
      ok "a server is running on $where (process $pid, $by): $kernels; it $reads"
      case $kernels in
        *ON) note "with custom kernels on, a container can read and change the arrays of the others on this server: turn it on only for containers that trust each other (docs/security.md)" ;;
      esac
    else
      elsewhere="$elsewhere $pid"
      note "another MLX server is running on $where (process $pid, $by): $kernels"
    fi
  done
  if [ -n "$found" ]; then
    if [ "$ENGINE" = kiri ]; then
      round_trip MLX "$MLX_PY" "$MLX_SMOKE" mlx_in_container varco-mlx:latest "on $GPU_SOCKET"
    else
      round_trip MLX "$MLX_PY" "$MLX_SMOKE" mlx_in_container varco-mlx:latest "on port $MLX_PORT"
    fi
  elif [ "$ENGINE" = kiri ] && [ "$channel_kind" = PyTorch ]; then
    off "no MLX server is listening on $GPU_SOCKET, where Kiri's channel leads: the PyTorch bridge's server has it (process $channel_server), and one socket has one server" "to use the MLX bridge with Kiri instead, stop that server and then: $mlx_start"
  elif [ "$ENGINE" = kiri ]; then
    off "no MLX server is listening on $GPU_SOCKET, where Kiri's channel leads" "$mlx_start   (or keep one running: $mlx_keep)"
  else
    holder=$(listener "$MLX_PORT")
    if [ -n "$holder" ]; then
      fault "port $MLX_PORT is taken by process $holder ($(short_command "$holder")), which is not an MLX server: containers that look for the bridge there do not find it" "stop that process, or run the MLX server on another port and tell the containers: mlx/server/ctl.sh start --port $((MLX_PORT + 10)) $models_map, then -e MLX_REMOTE=host.docker.internal:$((MLX_PORT + 10))"
    elif port_open "$MLX_PORT"; then
      fault "port $MLX_PORT is taken by a process of another user, which is not this user's MLX server" "free the port, or run the MLX server on another one and give containers -e MLX_REMOTE=host.docker.internal:PORT"
    elif [ -n "$elsewhere" ]; then
      pid=${elsewhere# }
      pid=${pid%% *}
      port=$(ports_of "$pid")
      started_by "$pid" "$MLX_AGENT" "$REPO/mlx/logs"
      fault "no MLX server is listening on port $MLX_PORT, where the images look unless told otherwise; the one that is running listens on port $port" "give the containers -e MLX_REMOTE=host.docker.internal:$port (the examples: MLX_PORT=$port), or stop that server ($stop) and start one on port $MLX_PORT"
    else
      off "no MLX server is running: a container would say \"cannot connect to the MLX server on the Mac\"" "$mlx_start   (or keep one running: $mlx_keep)"
    fi
  fi
}

doctor_pytorch() {
  echo "== PyTorch bridge (PyTorch $TORCH)"
  others=$(ls -d pytorch/.venv-mac-* 2> /dev/null | sed 's/.*\.venv-mac-//' | grep -v "^$TORCH\$" | tr '\n' ' ' | sed 's/ $//')
  mac=
  if [ ! -x "$TORCH_PY" ]; then
    fault "the Mac end for PyTorch $TORCH is not there (pytorch/.venv-mac-$TORCH)" "$torch_setup"
  else
    mac=$(version_in "$TORCH_PY" torch)
    if [ ! -f "$TORCH_VENV/requirements-extra.txt" ]; then
      fault "the Mac end's environment is not complete: pytorch/.venv-mac-$TORCH has no requirements-extra.txt, which is copied there when every package is installed (a run of setup-mac.sh was interrupted)" "$torch_setup"
    elif [ "$mac" != "$TORCH" ]; then
      fault "pytorch/.venv-mac-$TORCH has torch ${mac:-missing}, not $TORCH" "remove pytorch/.venv-mac-$TORCH and run $torch_setup"
    else
      executor=$(ls "pytorch/bridge/_build/$TORCH"/_executor*.so 2> /dev/null | head -1)
      if [ -z "$executor" ]; then
        note "the Mac end's compiled executor is not built: the server runs its Python equivalent, which is slower (${with_torch}pytorch/build-mac.sh builds it)"
      elif [ pytorch/bridge/executor.cpp -nt "$executor" ]; then
        fault "the Mac end's compiled executor was built before its source last changed" "$torch_setup"
      else
        ok "the Mac end: torch $mac, with the compiled executor"
      fi
    fi
  fi
  [ -z "$others" ] || note "also built on this Mac, for PyTorch $others (${with_engine}TORCH=VERSION scripts/doctor.sh pytorch looks at one)"

  inside=
  if [ -n "$engine" ]; then
    if ! have_image "$TORCH_IMAGE"; then
      fault "the image $TORCH_IMAGE is not built" "$torch_setup"
    else
      label=$(label_of "$TORCH_IMAGE")
      if [ -z "$label" ]; then
        note "the image $TORCH_IMAGE was not built by setup-mac.sh: whether it matches the files as they are now is not known ($torch_setup settles it)"
      elif [ "$label" != "$(torch_inputs)" ]; then
        fault "the image $TORCH_IMAGE was built from another state of pytorch/bridge/ and pytorch/docker/ than the one here" "$torch_setup"
      fi
      if [ -n "$quick" ]; then
        ok "the image $TORCH_IMAGE is built"
      else
        inside=$(version_in_image "$TORCH_IMAGE" torch)
        if [ "${inside%%+*}" = "$TORCH" ]; then
          ok "the image $TORCH_IMAGE has PyTorch $inside"
        else
          fault "PyTorch is ${inside:-missing} in the image $TORCH_IMAGE, not $TORCH: the server refuses a container with another version than its own" "$torch_setup --rebuild"
        fi
      fi
    fi
  fi

  found=
  elsewhere=
  for pid in $(torch_servers); do
    version=$(torch_of_server "$pid")
    started_by "$pid" "$TORCH_AGENT" "$REPO/pytorch/runs"
    ports=$(ports_of "$pid")
    sockets=$(sockets_of "$pid")
    if [ -n "$ports" ]; then
      where="port $ports"
    elif [ -n "$sockets" ]; then
      where="the socket $sockets"
    else
      where="no port or socket yet"
    fi
    if [ "$ENGINE" = kiri ]; then
      [ -z "$ports" ] && holds_socket "$pid" "$GPU_SOCKET" && found=$pid
    else
      [ "$where" = "port $TORCH_PORT" ] && found=$pid
    fi
    if [ "$found" != "$pid" ]; then
      elsewhere="$elsewhere $pid"
      note "another bridge server is running on $where (process $pid, PyTorch ${version:-unknown}, $by)"
    elif [ "$version" = "$TORCH" ]; then
      ok "a server for PyTorch $version is running on $where (process $pid, $by)"
    else
      fault "the server on $where is for PyTorch ${version:-unknown} (process $pid, $by), not for $TORCH: it refuses a container of $TORCH_IMAGE, whose program then falls back to the CPU or fails" "stop it ($stop) and start the one for PyTorch $TORCH: $torch_start"
      found=mismatch
    fi
  done
  if [ "$found" = mismatch ]; then
    :
  elif [ -n "$found" ]; then
    if [ "$ENGINE" = kiri ]; then
      round_trip PyTorch "$TORCH_PY" "$TORCH_SMOKE" torch_in_container "$TORCH_IMAGE" "on $GPU_SOCKET"
    else
      round_trip PyTorch "$TORCH_PY" "$TORCH_SMOKE" torch_in_container "$TORCH_IMAGE" "on port $TORCH_PORT"
    fi
  elif [ "$ENGINE" = kiri ] && [ "$channel_kind" = MLX ]; then
    off "no bridge server is listening on $GPU_SOCKET, where Kiri's channel leads: the MLX server has it (process $channel_server), and one socket has one server" "to use the PyTorch bridge with Kiri instead, stop that server and then: $torch_start"
  elif [ "$ENGINE" = kiri ]; then
    off "no bridge server is listening on $GPU_SOCKET, where Kiri's channel leads" "$torch_start   (or keep one running: $torch_keep)"
  else
    holder=$(listener "$TORCH_PORT")
    if [ -n "$holder" ]; then
      fault "port $TORCH_PORT is taken by process $holder ($(short_command "$holder")), which is not a bridge server: torch.backends.mps.is_available() is False in the containers" "stop that process, or move the bridge: ${with_torch}BRIDGE_PORT=$((TORCH_PORT + 10)) pytorch/server.sh start, then -e BRIDGE_PORT=$((TORCH_PORT + 10)) for the containers"
    elif port_open "$TORCH_PORT"; then
      fault "port $TORCH_PORT is taken by a process of another user, which is not this user's bridge server" "free the port, or move the bridge: ${with_torch}BRIDGE_PORT=$((TORCH_PORT + 10)) pytorch/server.sh start, then -e BRIDGE_PORT=$((TORCH_PORT + 10)) for the containers"
    elif [ -n "$elsewhere" ]; then
      pid=${elsewhere# }
      pid=${pid%% *}
      port=$(ports_of "$pid")
      started_by "$pid" "$TORCH_AGENT" "$REPO/pytorch/runs"
      fault "no bridge server is listening on port $TORCH_PORT, where the image looks unless told otherwise; the one that is running listens on port $port" "give the containers -e BRIDGE_PORT=$port, or stop that server ($stop) and start one on port $TORCH_PORT"
    else
      off "no bridge server is running: torch.backends.mps.is_available() is False in the containers, and a program that falls back runs on the CPU" "$torch_start   (or keep one running: $torch_keep)"
    fi
  fi
}

for part in $parts; do
  doctor_$part
done

echo "== Autostart (optional)"
agent() {
  state=$(agent_state "$1")
  program=$(agent_program "$1")
  if [ -n "$program" ] && [ ! -x "$program" ]; then
    fault "the agent $1 names a program that is not there ($program): the repository was moved, or that end was removed" "$5"
    return
  fi
  told=$(agent_where "$1")
  case $state in
    absent) note "$3: not kept running from login ($5 does that)" ;;
    unloaded) note "$3: $(agent_file "$1") is there and launchd has not loaded it; it will at the next login" ;;
    "loaded -") fault "$3: the agent $1 is loaded and its server does not stay up" "read its log ($4), then $5" ;;
    *) ok "$3: kept running from login by the agent $1 (process ${state#loaded })" ;;
  esac
  # Where the agent tells its server to listen, against this engine. A server
  # that is running on a port has its FAULT under "Kiri's channel" already.
  case "$ENGINE,$told,$state" in
    kiri,port*,unloaded | kiri,port*,"loaded -")
      fault "$3: the agent $1 starts its server on TCP ${told}, which every container of Kiri reaches, started with --gpus or not" "$5   (on $GPU_SOCKET and no port; or: scripts/autostart.sh uninstall $2)"
      ;;
    kiri,port*)
      note "$3: that is the server on TCP ${told} above; $5 puts it on $GPU_SOCKET and on no port"
      ;;
    "kiri,socket $GPU_SOCKET",*) ;;
    kiri,socket*)
      note "$3: the agent $1 starts its server on ${told#socket }, and Kiri's channel leads to $GPU_SOCKET"
      ;;
    docker,socket*)
      note "$3: the agent $1 starts its server on the ${told}, for Kiri (ENGINE=kiri): the containers of Docker Desktop do not reach it"
      ;;
  esac
}
for part in $parts; do
  case $part in
    mlx) agent "$MLX_AGENT" mlx "the MLX server" mlx/logs/autostart.log "$mlx_keep" ;;
    pytorch) agent "$TORCH_AGENT" pytorch "the PyTorch server" pytorch/runs/autostart.log "$torch_keep" ;;
  esac
done

echo "== Summary"
if [ $faults -gt 0 ]; then
  echo "$faults fault(s): each has its remedy on the line under it."
  exit 1
fi
if [ $offs -gt 0 ]; then
  echo "No fault found. $offs server(s) not running: the examples start their own, and the line under each says how to start one by hand."
else
  echo "No fault found."
fi
