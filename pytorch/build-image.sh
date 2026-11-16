#!/bin/sh
# The container end for one PyTorch version:   [TORCH=2.15.0] ./build-image.sh [docker build options]
# Needs bridge/ops/mps_ops-$TORCH.inc, which build-mac.sh makes on the Mac with that version
# (the lists for 2.15.0 are in the repository).
cd "$(dirname "$0")" || exit 1
. ./versions.sh
[ -f "bridge/ops/mps_ops-$TORCH.inc" ] || { echo "no bridge/ops/mps_ops-$TORCH.inc: run TORCH=$TORCH ./build-mac.sh first" >&2; exit 1; }
# A builder that runs in a container of the engine (buildx's docker-container driver, which is what
# "docker build" uses with Kiri) leaves the image in its own cache: --load puts it in the engine.
driver=$(docker buildx inspect 2>/dev/null | sed -n 's/^Driver: *//p' | head -1)
load=
[ -n "$driver" ] && [ "$driver" != docker ] && load=--load
exec docker build "$@" $load -f docker/Dockerfile --build-arg "TORCH=$TORCH" --build-arg "PYTHON=$PYTHON" -t "$IMAGE" .
