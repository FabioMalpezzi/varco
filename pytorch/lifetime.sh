#!/bin/sh
# Does the Mac get its GPU memory back when a program in a container ends, however it ends?
#   [TORCH=...] ./lifetime.sh [GIGABYTES]        (starts and stops the bridge's server)
# For each way of ending: the program takes GIGABYTES of GPU memory, then ends or is killed.
# Printed: the memory of the Mac's process for that program while it runs, and whether that
# process is gone afterwards (a process that is gone holds nothing).
# The exit status is 0 only when, for every way of ending, the program got its memory and the
# Mac's process for it is gone afterwards.
cd "$(dirname "$0")" || exit 1
. ./versions.sh
gigabytes="${1:-4}"
./server.sh start >/dev/null || exit 1
[ -f runs/server.pid ] || { echo "lifetime.sh reads the log of a server it started itself: stop the one already listening on ${WHERE:-port $PORT} first (scripts/autostart.sh uninstall pytorch, if it is the one kept at login)" >&2; exit 1; }
status=0
for how in clean exception exit hang busy; do
    name="varco-lifetime-$how"
    docker rm -f "$name" >/dev/null 2>&1
    : >runs/lifetime.out
    NAME="$name" ./in-container.sh python lifetime.py "$how" "$gigabytes" >runs/lifetime.out 2>&1 &
    runner=$!
    until grep -q "^ready" runs/lifetime.out || ! kill -0 $runner 2>/dev/null; do sleep 0.2; done
    session=$(grep "lifetime.py" runs/server.log | grep connected | tail -1 | sed 's/.*, \([0-9]*\) here.*/\1/')
    during=$(ps -o rss= -p "$session" 2>/dev/null | awk '{printf "%d", $1/1024}')
    footprint=$(footprint "$session" 2>/dev/null | awk '/phys_footprint:/ {print $2 $3; exit}')
    case "$how" in hang|busy) sleep 1; docker kill "$name" >/dev/null ;; esac
    wait $runner 2>/dev/null
    gone=no
    start=$(date +%s)
    for _ in $(seq 80); do kill -0 "$session" 2>/dev/null || { gone="yes, within $(( $(date +%s) - start + 1 )) s"; break; }; sleep 0.25; done
    echo "$how: $(grep '^ready' runs/lifetime.out | cut -c7-); Mac process $session: ${during:-?} MB resident, footprint ${footprint:-?} while it ran; gone afterwards: $gone"
    # Without its "ready" line the program never held the memory, and without a process of its
    # own on the Mac there was nothing to see go: neither is a memory given back. (The memory
    # while it ran is not asked for: a program that ends at once may be gone before it is read.)
    grep -q "^ready" runs/lifetime.out || { echo "FAILED: $how: the program did not get as far as holding the memory: see runs/lifetime.out"; status=1; }
    [ -n "$session" ] || { echo "FAILED: $how: the server's log names no process of the Mac's for the program"; status=1; }
    [ "$gone" != no ] || { echo "FAILED: $how: the Mac's process $session is still there, with the memory"; status=1; }
done
./server.sh stop
exit $status
