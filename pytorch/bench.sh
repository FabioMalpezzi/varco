#!/bin/sh
# Times three ordinary programs (tests/program.py, tests/models.py) on the Mac's GPU directly and from the container
# through the bridge:   [TORCH=...] [ENGINE=kiri] ./bench.sh [native] [bridge]   (both by default)
cd "$(dirname "$0")" || exit 1
. ./versions.sh
what="${*:-native bridge}"
# ENGINE=kiri runs the container on Kiri (its daemon must be up) with --gpus, and the Mac's end
# listens on the socket Kiri's own channel leads to (versions.sh). Docker Desktop otherwise.
mac() { shift; (cd tests && MODELS="$HUB" HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error "$VENV/bin/python" "$@"); }
box() { docker run --rm ${GPUS:+--gpus all} -e LONGER -e MORE -e BRIDGE_PORT -e BRIDGE_BUFFER -e BRIDGE_FREE_AT -e MODELS=/hub -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_VERBOSITY=error \
    -v "$HUB:/hub:ro" -v "$ROOT/tests:/work" -v "$ROOT/bridge:$SITE:ro" "$IMAGE" "$@"; }
keep() { grep -E "operations  |texts in|pairs in|similarity|scores|\[bridge\]|Error|error" | sed "s/^ */$1  /"; }

case "$what" in *native*)
    mac python program.py 2>&1 | keep native
    mac python models.py 2>&1 | keep native
    LONGER=8 MORE=8 mac python models.py 2>&1 | keep "native heavy" | grep -v -E "similarity|scores"
esac
case "$what" in *bridge*)
    ./server.sh start >/dev/null || exit 1
    trap './server.sh stop' EXIT
    box python -m bridge program.py 2>&1 | keep bridge
    box python -m bridge models.py 2>&1 | keep bridge
    LONGER=8 MORE=8 box python -m bridge models.py 2>&1 | keep "bridge heavy" | grep -v -E "similarity|scores"
esac
