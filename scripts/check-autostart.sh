#!/bin/sh
# Check where scripts/autostart.sh makes the two servers listen, for each
# engine, what "autostart.sh status" and doctor.sh then say of them, and that
# pytorch/server.sh uses the PyTorch server it finds on Kiri's socket.
#   Docker Desktop   each server on its TCP port of the Mac's loopback
#   ENGINE=kiri      one server, on the Unix socket Kiri's channel leads to,
#                    and on no TCP port: Kiri forwards host.docker.internal:PORT
#                    of every container to the Mac's loopback, so a port lets
#                    in the containers started without --gpus
# It needs no GPU, and reaches neither launchd nor a container engine:
#   - the scripts run from a copy in a temporary directory, with a HOME of
#     their own there: the agents' files and the socket are under it, not in
#     ~/Library/LaunchAgents or ~/.kiri;
#   - a stand-in for launchctl, first in PATH, starts the program an agent's
#     file names, tells its pid, and stops it;
#   - stand-ins for the two servers listen where their arguments say, as the
#     servers do, and compute nothing.
# So it is autostart.sh that chooses the arguments, waits for the server and
# refuses, and what the started process listens on is read with lsof.
#   scripts/check-autostart.sh          with the stand-ins
#   scripts/check-autostart.sh --live   with the two servers built on this Mac
#                                       in place of their stand-ins (these
#                                       need the GPU, and a second or two)
# SCRIPTS=DIR checks the autostart.sh, lib.sh and doctor.sh of another
# directory: an earlier state of them, to see a check fail there.
# It runs on a Mac only (launchd agents are a Mac's): elsewhere it says so.
# The exit status is 0 when every check passed.
here=$(cd "$(dirname "$0")" && pwd) || exit 1
repo=$(dirname "$here")
scripts=${SCRIPTS:-$here}
live=
case ${1:-} in
  "") ;;
  --live) live=yes ;;
  *)
    echo "usage: check-autostart.sh [--live]" >&2
    exit 2
    ;;
esac
if [ "$(uname -s)" != Darwin ]; then
  echo "skipped: launchd agents are a Mac's, and this is $(uname -s)"
  exit 0
fi
for tool in python3 lsof; do
  command -v "$tool" > /dev/null || {
    echo "FAILED   $tool is not installed: the check needs it"
    exit 1
  }
done
# Nothing of the caller's settings chooses where the servers go.
unset WHERE KERNELS DOCKER_HOST IMAGE PYTHON
TORCH=${TORCH:-2.15.0}
mlx_agent=varco.mlx-server
torch_agent=varco.pytorch-server

# A short path: a Unix socket's is 103 characters at most.
work=$(mktemp -d /tmp/varco-autostart.XXXXXX) || exit 1
tree=$work/repo
home=$work/home
bin=$work/bin
state=$work/launchd
agents=$home/Library/LaunchAgents
socket=$home/.kiri/run/gpu.sock
other=
# Stop, by the pids kept, whatever is still running, and remove the directory.
cleanup() {
  for file in "$state"/*.pid; do
    [ -f "$file" ] && kill "$(cat "$file")" 2> /dev/null
  done
  [ -z "$other" ] || kill "$other" 2> /dev/null
  rm -rf "$work"
}
trap cleanup EXIT
trap 'exit 130' INT TERM
mkdir -p "$tree/scripts" "$tree/mlx/server" "$tree/mlx/logs" "$tree/pytorch" "$bin" "$state" "$work/models" "$(dirname "$socket")"
for file in autostart.sh lib.sh doctor.sh; do
  cp "$scripts/$file" "$tree/scripts/" || exit 1
done
cp "$repo/mlx/versions.sh" "$tree/mlx/" || exit 1
cp "$repo/pytorch/versions.sh" "$repo/pytorch/server.sh" "$tree/pytorch/" || exit 1

# The stand-in for a server: the MLX server's --port N or --unix PATH, or the
# PyTorch server's one word after "-m bridge.server", a port or a path. Like
# both, it removes the socket it finds, and removes its own when it is stopped.
cat > "$work/standin.py" << 'PY'
import os
import signal
import socket
import sys

args = sys.argv[1:]
if "--unix" in args:
    where = args[args.index("--unix") + 1]
elif "--port" in args:
    where = args[args.index("--port") + 1]
elif args[:2] == ["-m", "bridge.server"] and len(args) == 3:
    where = args[2]
else:
    sys.exit("stand-in for a server: no --port N, --unix PATH or -m bridge.server WORD")
path = None
if where.isdigit():
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", int(where)))
else:
    if os.path.exists(where):
        os.unlink(where)
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(where)
    os.chmod(where, 0o600)
    path = where
listener.listen(16)


def stop(*_):
    if path and os.path.exists(path):
        os.unlink(path)
    os._exit(0)


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
print("stand-in server: listening on", where, flush=True)
while True:
    connection, _ = listener.accept()
    connection.close()
PY

# The stand-in for launchctl, for the three verbs the scripts use.
cat > "$bin/launchctl" << 'PY'
#!/usr/bin/env python3
import os
import plistlib
import signal
import subprocess
import sys
import time

state = os.environ["FAKE_LAUNCHD"]
verb = sys.argv[1]


def kept(label):
    return os.path.join(state, label + ".pid")


def alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


if verb == "bootstrap":  # bootstrap DOMAIN FILE
    with open(sys.argv[3], "rb") as file:
        agent = plistlib.load(file)
    log = open(agent["StandardOutPath"], "ab")
    process = subprocess.Popen(
        agent["ProgramArguments"], cwd=agent["WorkingDirectory"], stdin=subprocess.DEVNULL, stdout=log, stderr=log,
        env=dict(os.environ, **agent.get("EnvironmentVariables", {})), start_new_session=True)
    with open(kept(agent["Label"]), "w") as file:
        file.write(str(process.pid))
elif verb == "print":  # print DOMAIN/LABEL
    label = sys.argv[2].rsplit("/", 1)[1]
    if not os.path.exists(kept(label)):
        sys.exit(113)
    pid = int(open(kept(label)).read())
    print(sys.argv[2], "= {")
    if alive(pid):
        print(f"\tpid = {pid}")
    print("}")
elif verb == "bootout":  # bootout DOMAIN/LABEL
    label = sys.argv[2].rsplit("/", 1)[1]
    if not os.path.exists(kept(label)):
        sys.exit(113)
    pid = int(open(kept(label)).read())
    if alive(pid):
        os.kill(pid, signal.SIGTERM)
    limit = time.time() + 10
    while alive(pid) and time.time() < limit:
        time.sleep(0.05)
    os.unlink(kept(label))
else:
    sys.exit(f"stand-in for launchctl: no {verb}")
PY
# No engine answers: doctor.sh then runs no container.
printf '#!/bin/sh\nexit 1\n' > "$bin/docker"
chmod +x "$bin/launchctl" "$bin/docker"

if [ -n "$live" ]; then
  if [ ! -x "$repo/mlx/server/mlx-remote-server" ] || [ ! -x "$repo/pytorch/.venv-mac-$TORCH/bin/python" ]; then
    echo "FAILED   --live needs the two servers built on this Mac: scripts/setup-mac.sh"
    exit 1
  fi
  # A copy of the MLX server, and the PyTorch server's files where they are;
  # Python is told to write nothing next to them.
  cp "$repo/mlx/server/mlx-remote-server" "$tree/mlx/server/" || exit 1
  ln -s "$repo/pytorch/.venv-mac-$TORCH" "$tree/pytorch/.venv-mac-$TORCH"
  ln -s "$repo/pytorch/bridge" "$tree/pytorch/bridge"
  export PYTHONDONTWRITEBYTECODE=1
  echo "== the servers built on this Mac (--live), PyTorch $TORCH"
else
  # Each stand-in is where the scripts look for the server's program, and
  # its command line names that place, as the server's does. The Python of
  # the PyTorch end is the stand-in when it is asked for the server, and a
  # Python (without torch) for anything else doctor.sh asks of it.
  mkdir -p "$tree/pytorch/.venv-mac-$TORCH/bin"
  for program in "$tree/mlx/server/mlx-remote-server" "$tree/pytorch/.venv-mac-$TORCH/bin/python"; do
    cp "$work/standin.py" "$program.py"
    printf '#!/bin/sh\ncase $1 in\n  -c) exec python3 "$@" ;;\nesac\nexec python3 "$0.py" "$@"\n' > "$program"
    chmod +x "$program"
  done
fi

# Two free ports of the loopback, for the servers of Docker Desktop.
ports=$(python3 -c '
import socket
held = [socket.socket() for _ in range(2)]
for s in held:
    s.bind(("127.0.0.1", 0))
print(*(s.getsockname()[1] for s in held))') || exit 1
mlx_port=${ports% *}
torch_port=${ports#* }

status=0
ok() { echo "ok       $*"; }
failed() {
  echo "FAILED   $*"
  status=1
}
# autostart.sh and doctor.sh of the copy, for the engine $1.
run() {
  engine=$1
  shift
  HOME="$home" PATH="$bin:$PATH" FAKE_LAUNCHD="$state" ENGINE="$engine" MLX_PORT="$mlx_port" BRIDGE_PORT="$torch_port" \
    MODELS="$work/models" TORCH="$TORCH" sh "$tree/scripts/autostart.sh" "$@" 2>&1
}
doctor() {
  # doctor.sh knows a stand-in of the MLX server as it knows a copy of the
  # server with another name: by a pid file under mlx/logs.
  rm -f "$tree/mlx/logs/server-standin.pid"
  [ -n "$live" ] || [ -z "$(agent_pid "$mlx_agent")" ] || agent_pid "$mlx_agent" > "$tree/mlx/logs/server-standin.pid"
  HOME="$home" PATH="$bin:$PATH" FAKE_LAUNCHD="$state" ENGINE="$1" MLX_PORT="$mlx_port" BRIDGE_PORT="$torch_port" \
    MODELS="$work/models" TORCH="$TORCH" sh "$tree/scripts/doctor.sh" --quick 2>&1
}
# pytorch/server.sh of the copy, which the examples and the tests start and
# stop the PyTorch server with, for the engine $1.
server() {
  engine=$1
  shift
  HOME="$home" PATH="$bin:$PATH" ENGINE="$engine" BRIDGE_PORT="$torch_port" TORCH="$TORCH" sh "$tree/pytorch/server.sh" "$@" 2>&1
}
# The arguments an agent's file gives its server, one on each line.
arguments() {
  python3 -c 'import plistlib, sys; print("\n".join(plistlib.load(open(sys.argv[1], "rb"))["ProgramArguments"]))' "$agents/$1.plist" 2> /dev/null
}
# The same on one line, without the temporary directory, to print.
shown() {
  arguments "$1" | sed "s|$work/||g" | tr '\n' ' ' | sed 's/ $//'
}
agent_pid() {
  cat "$state/$1.pid" 2> /dev/null
}
# What process $1 listens on: its TCP ports, and whether the socket $2.
tcp_of() {
  [ -n "$1" ] || return 0
  lsof -nP -a -p "$1" -iTCP -sTCP:LISTEN -Fn 2> /dev/null | sed -n 's/^n.*:\([0-9]*\)$/\1/p' | sort -u | tr '\n' ' ' | sed 's/ $//'
}
holds() {
  [ -n "$1" ] && [ -S "$2" ] && lsof -nP -a -p "$1" -U -Fn 2> /dev/null | grep -qxF "n$2"
}
running() {
  [ -n "$1" ] && kill -0 "$1" 2> /dev/null
}

# "ENGINE=kiri autostart.sh install $1": the agent $2 of the server $3 is told
# the socket and no port, and its process listens on the socket only.
kiri_install() {
  part=$1
  agent=$2
  name=$3
  out=$(run kiri install "$part")
  code=$?
  if [ ! -f "$agents/$agent.plist" ]; then
    failed "ENGINE=kiri autostart.sh install $part wrote no agent (status $code): $out"
    return
  fi
  words=$(arguments "$agent")
  echo "         $name, ENGINE=kiri, is started with: $(shown "$agent")"
  if echo "$words" | grep -qxF -- "$socket"; then
    ok "$name is told the socket Kiri's channel leads to"
  else
    failed "$name is not told the socket Kiri's channel leads to (.kiri/run/gpu.sock)"
  fi
  if echo "$words" | grep -qE -- '^(--port|[0-9]+)$'; then
    failed "$name is told a TCP port with ENGINE=kiri: every container of Kiri reaches a port of the Mac, started with --gpus or not"
  else
    ok "$name is told no TCP port"
  fi
  [ $code = 0 ] || failed "ENGINE=kiri autostart.sh install $part ended with status $code: $out"
  pid=$(agent_pid "$agent")
  if ! running "$pid"; then
    failed "$name is not running"
    return
  fi
  if holds "$pid" "$socket"; then
    ok "$name (process $pid) listens on the socket"
  else
    failed "$name (process $pid) does not listen on the socket"
  fi
  open=$(tcp_of "$pid")
  if [ -z "$open" ]; then
    ok "$name (process $pid) listens on no TCP port"
  else
    failed "$name (process $pid) listens on TCP port $open"
  fi
  line=$(run kiri status | grep "^$name:")
  case $line in
    *"socket $socket, no TCP port"*) ok "autostart.sh status says so" ;;
    *) failed "autostart.sh status does not say the socket and no TCP port: $line" ;;
  esac
}

echo "== Kiri: a server on the socket, and on no TCP port"
kiri_install mlx "$mlx_agent" "the MLX server"
# The MLX server is on the socket: pytorch/server.sh does not take it.
pid=$(agent_pid "$mlx_agent")
out=$(server kiri start)
code=$?
if [ $code != 0 ] && running "$pid" && holds "$pid" "$socket"; then
  ok "pytorch/server.sh start is refused while the MLX server listens on the socket, which keeps it"
else
  failed "pytorch/server.sh start was not refused while the MLX server listens on the socket (status $code): $(echo "$out" | cut -c1-240)"
fi
server kiri stop > /dev/null
run kiri uninstall mlx > /dev/null
kiri_install pytorch "$torch_agent" "the PyTorch server"
pid=$(agent_pid "$torch_agent")
# The PyTorch server of the same version is on the socket: pytorch/server.sh
# uses it, and its stop leaves it running.
out=$(server kiri start)
code=$?
case "$code,$out" in
  0,*"process $pid, not started by this script"*) ok "pytorch/server.sh start uses the PyTorch server kept on the socket (process $pid)" ;;
  *) failed "pytorch/server.sh start does not use the PyTorch server kept on the socket (status $code): $(echo "$out" | cut -c1-240)" ;;
esac
server kiri stop > /dev/null
if running "$pid" && holds "$pid" "$socket"; then
  ok "pytorch/server.sh stop leaves that server running, on the socket"
else
  failed "after pytorch/server.sh stop the PyTorch server kept on the socket (process $pid) is not listening there"
fi
out=$(doctor kiri)
if echo "$out" | grep -q "^ok .* is running on the socket $socket (process $pid,"; then
  ok "doctor.sh finds the PyTorch server on the socket"
else
  failed "doctor.sh does not find the PyTorch server (process $pid) on the socket; it says: $(echo "$out" | grep -E "bridge server|$socket" | cut -c1-240)"
fi
if echo "$out" | grep '^FAULT' | grep -q "process $pid[,)]"; then
  failed "doctor.sh has a fault for the server on the socket: $(echo "$out" | grep '^FAULT' | grep "process $pid[,)]" | cut -c1-240)"
fi

echo "== Kiri: one socket has one server"
# The PyTorch server is on the socket: the MLX server is refused it.
out=$(run kiri install mlx)
code=$?
if [ $code != 0 ] && [ ! -f "$agents/$mlx_agent.plist" ]; then
  ok "install mlx is refused while the PyTorch server is installed on the socket"
else
  failed "install mlx was not refused while the PyTorch server is installed on the socket (status $code): $(echo "$out" | cut -c1-240)"
fi
if running "$pid" && holds "$pid" "$socket"; then
  ok "the PyTorch server (process $pid) kept the socket"
else
  failed "the PyTorch server (process $pid) does not hold the socket any more"
fi
run kiri uninstall > /dev/null
# Both at once.
out=$(run kiri install)
code=$?
if [ $code != 0 ] && [ ! -f "$agents/$mlx_agent.plist" ] && [ ! -f "$agents/$torch_agent.plist" ]; then
  ok "install of both bridges is refused, and nothing is installed"
else
  failed "install of both bridges was not refused with ENGINE=kiri (status $code): the MLX server is started with: $(shown "$mlx_agent"); the PyTorch server with: $(shown "$torch_agent")"
fi
run kiri uninstall > /dev/null
# A process that no agent keeps listens on the socket.
python3 "$work/standin.py" --unix "$socket" > /dev/null 2>&1 &
other=$!
i=0
while [ $i -lt 50 ] && ! holds "$other" "$socket"; do
  sleep 0.1
  i=$((i + 1))
done
holds "$other" "$socket" || failed "the check's own process did not come to listen on the socket"
out=$(run kiri install mlx)
code=$?
if [ $code != 0 ] && holds "$other" "$socket"; then
  ok "install is refused while another process listens on the socket, which keeps it"
else
  failed "install was not refused while another process listens on the socket (status $code): $(echo "$out" | cut -c1-240)"
fi
run kiri uninstall > /dev/null
kill "$other" 2> /dev/null
wait "$other" 2> /dev/null
other=

echo "== Docker Desktop: each server on its port"
out=$(run docker install)
code=$?
[ $code = 0 ] || failed "autostart.sh install ended with status $code: $out"
echo "         the MLX server is started with: $(shown "$mlx_agent")"
echo "         the PyTorch server is started with: $(shown "$torch_agent")"
mlx_pid=$(agent_pid "$mlx_agent")
torch_pid=$(agent_pid "$torch_agent")
if arguments "$mlx_agent" | grep -qxF -- "$mlx_port" && [ "$(tcp_of "$mlx_pid")" = "$mlx_port" ]; then
  ok "the MLX server (process $mlx_pid) is told its port, $mlx_port, and listens on it"
else
  failed "the MLX server is not listening on its port, $mlx_port"
fi
if arguments "$torch_agent" | grep -qxF -- "$torch_port" && [ "$(tcp_of "$torch_pid")" = "$torch_port" ]; then
  ok "the PyTorch server (process $torch_pid) is told its port, $torch_port, and listens on it"
else
  failed "the PyTorch server is not listening on its port, $torch_port"
fi

echo "== Kiri: a server on a TCP port is a fault"
out=$(run kiri status)
if [ "$(echo "$out" | grep -c "a fault with ENGINE=kiri")" = 2 ]; then
  ok "ENGINE=kiri autostart.sh status tells both servers on a port as a fault"
else
  failed "ENGINE=kiri autostart.sh status does not tell the two servers on a port as a fault; it says: $out"
fi
out=$(doctor kiri)
for found in "$mlx_pid $mlx_port MLX" "$torch_pid $torch_port PyTorch"; do
  set -- $found
  if echo "$out" | grep -q "^FAULT .*TCP port $2 (process $1,"; then
    ok "ENGINE=kiri doctor.sh has a FAULT for the $3 server on port $2"
  else
    failed "ENGINE=kiri doctor.sh has no FAULT for the $3 server on port $2 (process $1); of it, it says: $(echo "$out" | grep "process $1[,)]" | cut -c1-240)"
  fi
done

echo "== Uninstall"
run docker uninstall > /dev/null
if [ -z "$(ls "$state" "$agents" 2> /dev/null | grep -E '\.(pid|plist)$')" ] && ! running "$mlx_pid" && ! running "$torch_pid"; then
  ok "the agents' files are removed and their servers stopped"
else
  failed "after uninstall something is left: $(ls "$state" "$agents" 2> /dev/null | tr '\n' ' ')"
fi

if [ $status = 0 ]; then
  echo "autostart: every check passed"
else
  echo "autostart: FAILED"
fi
exit $status
