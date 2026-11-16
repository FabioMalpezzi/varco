#!/bin/sh
# Keep the bridges' servers running from login on, with launchd agents of
# this user. Optional: without it, each example and each test starts the
# server it needs and stops it.
#   scripts/autostart.sh install [mlx] [pytorch]     (both by default)
#   scripts/autostart.sh status
#   scripts/autostart.sh uninstall [mlx] [pytorch]   (both by default)
# install writes one file per server in ~/Library/LaunchAgents
# (varco.mlx-server.plist, varco.pytorch-server.plist) and loads it: launchd
# starts the server now and at every login, and starts it again if it ends.
# uninstall unloads the agents, which stops their servers, and removes the
# files. Nothing else is written outside the repository; the servers' logs
# are mlx/logs/autostart.log and pytorch/runs/autostart.log.
# What install reads:
#   MODELS     the directory of the MLX models (default $HOME/models), which
#              the MLX server reads for containers that see it at /models
#   MLX_PORT   the MLX server's port (default 47800)
#   TORCH      the PyTorch version of the PyTorch server (default 2.15.0)
#   KERNELS=1  starts the MLX server with --custom-kernels: every container
#              that reaches it must then be trusted (docs/security.md)
#   ENGINE     docker (the default) or kiri: where the servers listen
# With Docker Desktop each server listens on its TCP port of the Mac's
# loopback. With ENGINE=kiri a server listens on the Unix socket Kiri's
# channel leads to ($HOME/.kiri/run/gpu.sock, or WHERE) and on no TCP port:
# Kiri forwards host.docker.internal:PORT of every container to the Mac's
# loopback, so a port would let in the containers started without --gpus.
# One socket has one server, so with ENGINE=kiri install takes one bridge
# (install mlx, or install pytorch) and refuses the socket of the other.
# The agents name this copy of the repository by its path: after moving it,
# install again. The tests need the ports for servers of their own: uninstall
# before running them.
cd "$(dirname "$0")/.." || exit 1
. scripts/lib.sh
domain=gui/$(id -u)

die() {
  echo "$*" >&2
  exit 1
}
# Text made safe for the XML of a property list.
xml() {
  printf '%s' "$1" | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g'
}
# Unload agent $1 and wait until its process is gone.
unload() {
  state=$(agent_state "$1")
  case $state in
    loaded*)
      pid=${state#loaded }
      launchctl bootout "$domain/$1" 2> /dev/null
      i=0
      while [ $i -lt 100 ]; do
        gone=yes
        launchctl print "$domain/$1" > /dev/null 2>&1 && gone=
        [ "$pid" != - ] && kill -0 "$pid" 2> /dev/null && gone=
        [ -n "$gone" ] && break
        sleep 0.1
        i=$((i + 1))
      done
      ;;
  esac
}
# Write the file of agent $1 (log $2, working directory $3, then the
# program and its arguments) and load it.
load() {
  label=$1
  log=$2
  directory=$3
  shift 3
  mkdir -p "$AGENTS_DIR" "$(dirname "$log")"
  {
    echo '<?xml version="1.0" encoding="UTF-8"?>'
    echo '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">'
    echo '<plist version="1.0">'
    echo '<dict>'
    echo "  <key>Label</key><string>$label</string>"
    echo '  <key>ProgramArguments</key>'
    echo '  <array>'
    for word in "$@"; do
      echo "    <string>$(xml "$word")</string>"
    done
    echo '  </array>'
    echo "  <key>WorkingDirectory</key><string>$(xml "$directory")</string>"
    if [ "$label" = "$TORCH_AGENT" ]; then # what pytorch/server.sh sets for the same server
      echo '  <key>EnvironmentVariables</key>'
      echo '  <dict>'
      echo '    <key>HF_HUB_OFFLINE</key><string>1</string>'
      echo '    <key>TRANSFORMERS_VERBOSITY</key><string>error</string>'
      echo '  </dict>'
    fi
    echo '  <key>RunAtLoad</key><true/>'
    echo '  <key>KeepAlive</key><true/>'
    echo "  <key>StandardOutPath</key><string>$(xml "$log")</string>"
    echo "  <key>StandardErrorPath</key><string>$(xml "$log")</string>"
    echo '</dict>'
    echo '</plist>'
  } > "$(agent_file "$label")"
  : > "$log"
  launchctl bootstrap "$domain" "$(agent_file "$label")" || die "launchd did not load $(agent_file "$label")"
}
# Wait until agent $1 has a process listening where the containers of this
# engine find it: on port $2, or with Kiri on the socket. Print its pid.
wait_listening() {
  i=0
  while [ $i -lt 150 ]; do
    state=$(agent_state "$1")
    pid=${state#loaded }
    if [ "$pid" != "$state" ] && [ "$pid" != - ]; then
      if [ "$ENGINE" = kiri ]; then
        [ -S "$GPU_SOCKET" ] && holds_socket "$pid" "$GPU_SOCKET" && echo "$pid" && return 0
      elif [ "$(listener "$2")" = "$pid" ]; then
        echo "$pid"
        return 0
      fi
    fi
    sleep 0.1
    i=$((i + 1))
  done
  return 1
}
# Refuse to install on a port that something else holds.
port_free() {
  holder=$(listener "$1")
  if [ -n "$holder" ]; then
    die "port $1 is taken by process $holder ($(short_command "$holder")): stop it first ($2), then install again"
  fi
  port_open "$1" && die "port $1 is taken by a process of another user: free it, or set $3"
  return 0
}
# With Kiri, before anything is changed: the socket is one a server can bind,
# and the agent of the other bridge ($1, its server $2, its name $3) is not
# installed on it. Both servers remove the socket they find and bind their
# own, so a second one would take the channel from the first without a word.
socket_usable() {
  case $GPU_SOCKET in
    /*) ;;
    *) die "WHERE is $GPU_SOCKET: the socket's path must be absolute, since the agent runs in another directory" ;;
  esac
  [ ${#GPU_SOCKET} -lt 104 ] || die "the socket's path is ${#GPU_SOCKET} characters long and macOS takes 103 at most: $GPU_SOCKET"
  [ -d "$(dirname "$GPU_SOCKET")" ] || die "the directory of $GPU_SOCKET is not there: Kiri's daemon makes it when it starts (kiri/README.md). Start it once, then install again"
  if [ "$(agent_where "$1")" = "socket $GPU_SOCKET" ]; then
    die "$2 is installed on $GPU_SOCKET (the agent $1), and one socket has one server: Kiri's channel leads to one bridge at a time. To change bridge: scripts/autostart.sh uninstall $3, then install again"
  fi
}
# With Kiri, once this bridge's own agent is unloaded: refuse to install on a
# socket that something else listens on.
socket_free() {
  holder=$(socket_listener "$GPU_SOCKET")
  [ -n "$holder" ] || return 0
  if is_mlx_server "$holder"; then
    hint="mlx/server/ctl.sh stop, or scripts/autostart.sh uninstall mlx"
  elif is_torch_server "$holder"; then
    hint="ENGINE=kiri pytorch/server.sh stop, or scripts/autostart.sh uninstall pytorch"
  else
    hint="kill $holder"
  fi
  die "the socket $GPU_SOCKET is taken by process $holder ($(short_command "$holder")), and one socket has one server: stop it first ($hint), then install again"
}
# With Kiri, the server of agent $1 (process $2) that has just started must
# listen on no TCP port: one that does is taken down again.
socket_only() {
  [ "$ENGINE" = kiri ] || return 0
  open=$(ports_of "$2")
  [ -n "$open" ] || return 0
  unload "$1"
  rm -f "$(agent_file "$1")"
  die "the server (process $2) opened TCP port $open besides $GPU_SOCKET, and every container of Kiri reaches a port of the Mac: the agent $1 was removed again"
}
# With Kiri, name the bridge servers that listen on a TCP port all the same
# (started by hand, or kept by an agent that was installed for Docker Desktop).
tcp_warning() {
  [ "$ENGINE" = kiri ] || return 0
  tcp_servers | while read -r server kind open; do
    echo "warning: a server of the $kind bridge (process $server) listens on TCP port $open, which every container of Kiri reaches as host.docker.internal:$open, started with --gpus or not. Stop it: ENGINE=kiri scripts/doctor.sh says how" >&2
  done
}

install_mlx() {
  [ -x "$MLX_SERVER" ] || die "the MLX server is not built: run scripts/setup-mac.sh mlx first"
  if [ "$ENGINE" = kiri ]; then
    socket_usable "$TORCH_AGENT" "the PyTorch server" pytorch
    unload "$MLX_AGENT"
    socket_free
    set -- "$MLX_SERVER" --unix "$GPU_SOCKET"
    where="socket $GPU_SOCKET"
    reached="every container started with --gpus"
  else
    unload "$MLX_AGENT"
    port_free "$MLX_PORT" "mlx/server/ctl.sh stop, or SERVER_NAME=example mlx/server/ctl.sh stop" MLX_PORT
    set -- "$MLX_SERVER" --port "$MLX_PORT"
    where="port $MLX_PORT"
    reached="every container that reaches port $MLX_PORT"
  fi
  if [ -d "$MODELS" ]; then
    set -- "$@" --map "/models=$(cd "$MODELS" && pwd)"
  else
    echo "note: $MODELS is not a directory, so the server is given no weights to read: containers will upload them (set MODELS and install again)"
  fi
  if [ "${KERNELS:-}" = 1 ]; then
    set -- "$@" --custom-kernels
    echo "note: custom Metal kernels are ON for this server: $reached must be trusted by every other (docs/security.md)"
  fi
  load "$MLX_AGENT" "$REPO/mlx/logs/autostart.log" "$REPO/mlx" "$@"
  if pid=$(wait_listening "$MLX_AGENT" "$MLX_PORT"); then
    socket_only "$MLX_AGENT" "$pid"
    [ "$ENGINE" = kiri ] && where="$where, no TCP port"
    echo "installed: the MLX server runs now and from every login on (process $pid, $where; log mlx/logs/autostart.log)"
    tcp_warning
  else
    echo "the agent $MLX_AGENT is loaded and its server is not listening on $where; its log, mlx/logs/autostart.log, says:" >&2
    tail -5 mlx/logs/autostart.log >&2
    exit 1
  fi
}

install_pytorch() {
  [ -x "$TORCH_PY" ] || die "the PyTorch bridge's Mac end for PyTorch $TORCH is not built: run scripts/setup-mac.sh pytorch first"
  # The server takes one word: a port, or the path of a socket (which is
  # absolute here, so never a number).
  if [ "$ENGINE" = kiri ]; then
    socket_usable "$MLX_AGENT" "the MLX server" mlx
    unload "$TORCH_AGENT"
    socket_free
    set -- "$GPU_SOCKET"
    where="socket $GPU_SOCKET"
  else
    unload "$TORCH_AGENT"
    port_free "$TORCH_PORT" "pytorch/server.sh stop" BRIDGE_PORT
    set -- "$TORCH_PORT"
    where="port $TORCH_PORT"
  fi
  load "$TORCH_AGENT" "$REPO/pytorch/runs/autostart.log" "$REPO/pytorch" "$TORCH_PY" -m bridge.server "$@"
  if pid=$(wait_listening "$TORCH_AGENT" "$TORCH_PORT"); then
    socket_only "$TORCH_AGENT" "$pid"
    [ "$ENGINE" = kiri ] && where="$where, no TCP port"
    echo "installed: the PyTorch bridge's server for PyTorch $TORCH runs now and from every login on (process $pid, $where; log pytorch/runs/autostart.log)"
    tcp_warning
  else
    echo "the agent $TORCH_AGENT is loaded and its server is not listening on $where; its log, pytorch/runs/autostart.log, says:" >&2
    tail -5 pytorch/runs/autostart.log >&2
    exit 1
  fi
}

uninstall() {
  label=$1
  name=$2
  state=$(agent_state "$label")
  if [ "$state" = absent ]; then
    echo "$name: not installed"
    return
  fi
  pid=${state#loaded }
  unload "$label"
  rm -f "$(agent_file "$label")"
  if [ "$(agent_state "$label")" != absent ]; then
    die "$name: launchd still has the agent $label; try again: launchctl bootout $domain/$label"
  fi
  case $state in
    loaded*) [ "$pid" = - ] || ! kill -0 "$pid" 2> /dev/null || die "$name: the agent is removed and process $pid is still running: kill $pid" ;;
  esac
  echo "$name: uninstalled (the agent is unloaded, its server is stopped, and $(agent_file "$label") is removed)"
}

# Agent $1, the server $2 of the bridge $3 that listens on port $4 with Docker
# Desktop: a line about it, and under it what is wrong for this engine.
status() {
  label=$1
  name=$2
  part=$3
  port=$4
  state=$(agent_state "$label")
  told=$(agent_where "$label")
  case $state in
    absent)
      echo "$name: not installed"
      ;;
    unloaded)
      echo "$name: $(agent_file "$label") is there and launchd has not loaded it (it will at the next login; scripts/autostart.sh install loads it now)"
      ;;
    "loaded -")
      echo "$name: installed, and its server is not running: launchd keeps starting it again. Its log says:"
      tail -3 "$(/usr/libexec/PlistBuddy -c 'Print :StandardErrorPath' "$(agent_file "$label")" 2> /dev/null)" 2> /dev/null | sed 's/^/    /'
      ;;
    *)
      pid=${state#loaded }
      ports=$(ports_of "$pid")
      sockets=$(sockets_of "$pid")
      if [ -n "$ports" ]; then
        where="port $ports${sockets:+, socket $sockets}"
        told="port $ports"
      elif [ -n "$sockets" ]; then
        where="socket $sockets, no TCP port"
        told="socket $sockets"
      else
        where="no port or socket yet"
        told=
      fi
      case "$(command_of "$pid")" in
        *--custom-kernels*) extra="; custom Metal kernels ON" ;;
        *) extra= ;;
      esac
      echo "$name: installed and running (process $pid, $where$extra)"
      ;;
  esac
  # Where it listens (or, when it is not running, where its file tells it
  # to) against where the containers of this engine look.
  case "$ENGINE,$told" in
    kiri,port*)
      echo "    a fault with ENGINE=kiri: every container of Kiri reaches a TCP port of the Mac (host.docker.internal:${told#port }), started with --gpus or not. ENGINE=kiri scripts/autostart.sh install $part puts this server on $GPU_SOCKET and on no port; scripts/autostart.sh uninstall $part stops it."
      ;;
    "kiri,socket $GPU_SOCKET") ;;
    kiri,socket*)
      echo "    Kiri's channel leads to $GPU_SOCKET, not to that socket: ENGINE=kiri scripts/autostart.sh install $part"
      ;;
    docker,socket*)
      echo "    it was installed for Kiri (ENGINE=kiri): the containers of Docker Desktop look on port $port, and scripts/autostart.sh install $part puts it there."
      ;;
  esac
  program=$(agent_program "$label")
  if [ -n "$program" ] && [ ! -x "$program" ]; then
    echo "    its file names a program that is not there ($program): was the repository moved? Install again."
  fi
}

verb=$1
[ $# -gt 0 ] && shift
for part in "$@"; do
  case $part in
    mlx | pytorch) ;;
    *) verb=usage ;;
  esac
done
parts=${*:-mlx pytorch}
case $verb in
  install)
    if [ "$ENGINE" = kiri ] && [ "$parts" != mlx ] && [ "$parts" != pytorch ]; then
      die "with ENGINE=kiri, name the one bridge to keep running: scripts/autostart.sh install mlx, or install pytorch. Kiri's channel leads to one socket ($GPU_SOCKET) and one socket has one server; nothing was installed"
    fi
    for part in $parts; do
      install_$part
    done
    ;;
  uninstall)
    for part in $parts; do
      case $part in
        mlx) uninstall "$MLX_AGENT" "the MLX server" ;;
        pytorch) uninstall "$TORCH_AGENT" "the PyTorch server" ;;
      esac
    done
    ;;
  status)
    status "$MLX_AGENT" "the MLX server" mlx "$MLX_PORT"
    status "$TORCH_AGENT" "the PyTorch server" pytorch "$TORCH_PORT"
    ;;
  *)
    echo "usage: autostart.sh install [mlx] [pytorch] | status | uninstall [mlx] [pytorch]" >&2
    exit 2
    ;;
esac
