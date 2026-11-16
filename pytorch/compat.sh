#!/bin/sh
# The two batteries (tests/compat.py, tests/compat_train.py) on the Mac's GPU directly and in the
# container through the bridge, lined up, after the checks of the Mac end and of the container end
# that are not comparisons:   [TORCH=2.15.0] [ENGINE=kiri] ./compat.sh [checks|compat|compat_train]...
# Starts the bridge's server for that PyTorch, and stops it at the end. The outputs are in runs/.
# tests/compat_compare.py says what "lined up" asks for: every program of the battery's inventory
# (tests/expected/), on both sides, the same in full.
# Needs sentence-transformers/all-MiniLM-L6-v2 and openai-community/gpt2 in the Hugging Face
# cache (HUB): the programs that use a model fail on both sides alike without them.
cd "$(dirname "$0")" || exit 1
. ./versions.sh
[ -x "$VENV/bin/python" ] || { echo "no $VENV: run TORCH=$TORCH ./build-mac.sh first" >&2; exit 1; }
mkdir -p runs
./server.sh start || exit 1
trap './server.sh stop' EXIT
# 0 only when every battery ran to its end on both sides and compared as expected. A battery
# that does not goes on to the next one all the same, and the end tells.
status=0
for battery in ${*:-checks compat compat_train}; do
    if [ "$battery" = checks ]; then
        # Not programs to line up with the Mac's: what the Mac end refuses of a container that sends what
        # it likes (with a server of its own, on a socket in a temporary directory), what the container
        # end does with the Mac's answers (on the CPU), a fork while a thread waits for the Mac, what a
        # program is left with when the bridge gave its connection up, what a container whose PyTorch
        # is not the Mac's is told, and what one whose connection is closed before the greeting is.
        echo "== checks, PyTorch $TORCH"
        for check in refusals client_checks; do
            "$VENV/bin/python" "tests/$check.py" >"runs/$check-$TORCH.txt" 2>&1 \
                || { grep -E "ACCEPTED|FAILS|WRONG|Error" "runs/$check-$TORCH.txt" | head -20; echo "FAILED: tests/$check.py: see runs/$check-$TORCH.txt"; status=1; }
            tail -2 "runs/$check-$TORCH.txt"
        done
        for check in fork_reading reconnect other_pytorch no_server; do
            ./in-container.sh python "$check.py" >"runs/$check-$BOX-$TORCH.txt" 2>&1 \
                || { grep WRONG "runs/$check-$BOX-$TORCH.txt" | head -20; echo "FAILED: tests/$check.py in the container: see runs/$check-$BOX-$TORCH.txt"; status=1; }
            tail -1 "runs/$check-$BOX-$TORCH.txt"
        done
        continue
    fi
    echo "== $battery, PyTorch $TORCH"
    mac="runs/$battery-mac-$TORCH.txt" box="runs/$battery-$BOX-$TORCH.txt"
    # Removed first: a side that dies before it prints must not be compared through an earlier run's file.
    rm -f "$mac" "$box"
    (cd tests && MODELS="$HUB" HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error "$VENV/bin/python" "$battery.py") >"$mac" 2>"runs/$battery-mac-$TORCH.err" \
        || { echo "FAILED: $battery ended with status $? on the Mac: see runs/$battery-mac-$TORCH.err"; status=1; }
    ./in-container.sh python "$battery.py" >"$box" 2>"runs/$battery-$BOX-$TORCH.err" \
        || { echo "FAILED: $battery ended with status $? in the container: see runs/$battery-$BOX-$TORCH.err"; status=1; }
    "$VENV/bin/python" tests/compat_compare.py "$mac" "$box" || status=1
done
[ $status = 0 ] && echo "== as expected: ${*:-checks compat compat_train}, PyTorch $TORCH" || echo "== NOT as expected, PyTorch $TORCH: see the lines above"
exit $status
