#!/bin/sh
# Runs a command in the bridge's container, in tests/ (mounted at /work), with the Hugging Face
# cache read-only at /hub.
#   ./in-container.sh python models.py              (on the Mac's GPU: every Python process gets the bridge)
#   BRIDGE=0 ./in-container.sh python models.py     (CPU only)
#   ./in-container.sh python -m bridge models.py    (the same as the first, and a count of what the bridge did)
# The bridge's Python is taken from this folder, not from the image, so that a change to it needs
# no rebuild; native.cpp is in the image (./build-image.sh). MOUNTS adds docker run options.
# DUMP=directory, as the container sees it, keeps the tensors of a battery there (tests/exact.py):
#   MOUNTS="-v $PWD/runs/dump-box:/dump" DUMP=/dump ONLY=name ./in-container.sh python compat_train.py
cd "$(dirname "$0")" || exit 1
. ./versions.sh
# GPUS=1 starts the container with --gpus (Kiri's own channel to the Mac).
exec docker run --rm ${GPUS:+--gpus all} ${NAME:+--name "$NAME"} --shm-size=2g \
    -e TRACE -e ONLY -e DUMP -e BRIDGE -e BRIDGE_REPORT -e BRIDGE_AHEAD -e BRIDGE_DEBUG -e BRIDGE_SYNC -e BRIDGE_PORT -e BRIDGE_BUFFER -e LONGER -e MORE \
    -e PYTORCH_ENABLE_MPS_FALLBACK -e PYTORCH_MPS_HIGH_WATERMARK_RATIO \
    -e MODELS=/hub -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_VERBOSITY=error \
    -v "$HUB:/hub:ro" -v "$ROOT/tests:/work" -v "$ROOT/bridge:$SITE:ro" $MOUNTS "$IMAGE" "$@"
