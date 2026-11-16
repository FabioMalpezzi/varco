#!/bin/sh
# Start or stop the Mac server in the background: ctl.sh start [server args] | stop
# Log in logs/server.log, pid in logs/server.pid; with SERVER_NAME=x in
# logs/server-x.log and logs/server-x.pid, for a second server next to the
# first (on another port).
# SERVER_BIN names another copy of the binary to run (one whose name a
# "pkill -f mlx-remote-server" issued elsewhere on this Mac does not match).
cd "$(dirname "$0")/.."
mkdir -p logs
name=logs/server${SERVER_NAME:+-$SERVER_NAME}
# The pid is read from a file that may have outlived its process (a server that
# was killed, a restart of the Mac): the number can since be another process's,
# which is not this script's to stop. Only a server is.
is_server() {
  case "$(ps -o command= -p "$1" 2>/dev/null)" in
    "${SERVER_BIN:-./server/mlx-remote-server}"*|*mlx-remote-server*) return 0 ;;
  esac
  return 1
}
stop() {
  if [ -f $name.pid ]; then
    pid=$(cat $name.pid)
    if is_server "$pid"; then
      kill "$pid" 2>/dev/null
      i=0
      while kill -0 "$pid" 2>/dev/null && [ $i -lt 100 ]; do sleep 0.1; i=$((i + 1)); done
      if kill -0 "$pid" 2>/dev/null; then
        echo "the server (process $pid) did not stop in 10 seconds: kill -9 $pid" >&2
      fi
    elif kill -0 "$pid" 2>/dev/null; then
      echo "$name.pid names process $pid, which is not a server (the file is older than that process): left alone" >&2
    fi
    rm -f $name.pid
  fi
}
case "$1" in
  start)
    shift
    stop
    # The TCP port this server is asked for (none with --unix), and who has
    # it already: a server that cannot bind would only say so in its log.
    port=47800
    socket=
    previous=
    for arg in "$@"; do
      [ "$previous" = --port ] && port=$arg
      [ "$previous" = --unix ] && socket=$arg
      [ "$arg" = --unix ] && port=
      previous=$arg
    done
    # A Unix socket that a process is listening on is that process's: the
    # server removes the socket it finds and binds its own, which would take
    # the place of a server that is running without a word.
    if [ -n "$socket" ]; then
      holder=$(lsof -nP -U -Fpn 2>/dev/null | awk -v s="n$socket" '/^p/ { pid = substr($0, 2) } $0 == s { print pid; exit }')
      if [ -n "$holder" ]; then
        echo "the socket $socket is held by process $holder ($(ps -o command= -p "$holder" | sed 's#^[^ ]*/##' | cut -c1-80)): stop it, or give this server another --unix" >&2
        exit 1
      fi
    fi
    if [ -n "$port" ]; then
      holder=$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | head -1)
      if [ -n "$holder" ]; then
        echo "port $port is already taken by process $holder ($(ps -o command= -p "$holder" | sed 's#^[^ ]*/##' | cut -c1-80)): stop it, or give this server another --port" >&2
        exit 1
      fi
    fi
    nohup "${SERVER_BIN:-./server/mlx-remote-server}" "$@" > $name.log 2>&1 &
    echo $! > $name.pid
    # Until it listens (a new binary takes a moment the first time it runs).
    i=0
    while [ $i -lt 100 ] && ! grep -q "listening on" $name.log && kill -0 $! 2>/dev/null; do
      sleep 0.1
      i=$((i + 1))
    done
    head -5 $name.log
    if ! grep -q "listening on" $name.log; then
      kill -0 $! 2>/dev/null || rm -f $name.pid
      exit 1
    fi
    ;;
  stop)
    stop
    echo "server stopped"
    ;;
  *)
    echo "usage: ctl.sh start [server args] | stop"; exit 2 ;;
esac
