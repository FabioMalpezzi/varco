#!/bin/sh
# The Mac end, for the scripts:   [TORCH=2.15.0] ./server.sh start|stop|log
# On the bridge's loopback port (47700), or on the Unix socket WHERE (ENGINE=kiri sets it:
# versions.sh). One server at a time, on that port. Its pid is kept in a file, so that stopping it stops
# that process and its sessions and nothing else.
cd "$(dirname "$0")" || exit 1
. ./versions.sh
[ -x "$VENV/bin/python" ] || { echo "no $VENV: run TORCH=$TORCH ./build-mac.sh first" >&2; exit 1; }
mkdir -p runs
pidfile=runs/server.pid
log=runs/server.log
# The pid is read from a file that may have outlived its process (a server that was killed, a
# restart of the Mac): the number can since be another process's, which is not this script's to
# stop. Only a bridge server is.
stop() {
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        case "$(ps -o command= -p "$pid" 2>/dev/null)" in
            *"-m bridge.server"*)
                kill "$pid" 2>/dev/null
                for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$pid" 2>/dev/null || break; sleep 0.2; done ;;
            "") ;;
            *) echo "$pidfile names process $pid, which is not a bridge server (the file is older than that process): left alone" >&2 ;;
        esac
        rm -f "$pidfile"
    fi
}
case "${1:-start}" in
    start)
        stop
        # Who listens where this server is to listen: on the socket, the process that is bound to it
        # (a program's own process, bridge.session, holds a connection accepted there and is passed
        # over); on the port, its listener. A server that this script did not start
        # (scripts/autostart.sh keeps one) is used as it is when it is for the same PyTorch, and left
        # running by stop; anything else is in the way, and on the socket this server would remove
        # it and bind its own, and take that process's place without a word.
        holder=
        if [ -n "${WHERE:-}" ]; then
            place="the socket $WHERE"
            for pid in $(lsof -nP -U -Fpn 2>/dev/null | awk -v s="n$WHERE" '/^p/ { pid = substr($0, 2) } $0 == s && pid != last { print pid; last = pid }'); do
                case "$(ps -o command= -p "$pid")" in *" -m bridge.session"*) continue ;; esac
                holder=$pid
                break
            done
        else
            place="port $PORT"
            holder=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | head -1)
        fi
        if [ -n "$holder" ]; then
            command=$(ps -o command= -p "$holder")
            case "$command" in
                *".venv-mac-$TORCH/"*"-m bridge.server"*)
                    echo "bridge server: torch $TORCH, already running on ${WHERE:-127.0.0.1:$PORT} (process $holder, not started by this script): using it"
                    exit 0 ;;
                *"-m bridge.server"*)
                    echo "$place is held by a bridge server for another PyTorch (process $holder: $(echo "$command" | sed 's#^[^ ]*/##' | cut -c1-80)), and this is PyTorch $TORCH: stop that one first" >&2
                    exit 1 ;;
                *)
                    if [ -n "${WHERE:-}" ]; then
                        echo "the socket $WHERE is held by process $holder ($(echo "$command" | sed 's#^[^ ]*/##' | cut -c1-80)): stop it first" >&2
                    else
                        echo "port $PORT is already taken by process $holder ($(echo "$command" | sed 's#^[^ ]*/##' | cut -c1-80)): stop it, or set BRIDGE_PORT" >&2
                    fi
                    exit 1 ;;
            esac
        fi
        [ -S "${WHERE:-}" ] && rm -f "$WHERE"  # a socket left by a server that was killed: not a sign that this one is up
        HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error nohup "$VENV/bin/python" -m bridge.server "${WHERE:-$PORT}" >"$log" 2>&1 &
        echo $! >"$pidfile"
        until nc -z 127.0.0.1 "$PORT" 2>/dev/null || [ -S "${WHERE:-}" ]; do
            kill -0 "$(cat "$pidfile")" 2>/dev/null || { cat "$log"; exit 1; }
            sleep 0.2
        done
        head -1 "$log" ;;
    stop) stop ;;
    log) cat "$log" ;;
    *) echo "usage: server.sh start|stop|log" >&2; exit 2 ;;
esac
