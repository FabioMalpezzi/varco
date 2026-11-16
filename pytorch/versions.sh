# Sourced by the scripts under pytorch/: which PyTorch they are about. The bridge is for PyTorch
# 2.15.0, the one version it was built for and tested with.
# The Mac's environment, the list of operators a Mac runs on its GPU, the Mac end's C++ and the
# image are all per version: the two ends must run the same PyTorch (the server refuses another).
# TORCH names another version only for whoever makes its lists (./build-mac.sh runs gen_ops.py)
# and tries it: nothing about another version is in the repository.
ROOT=$PWD
TORCH="${TORCH:-2.15.0}"
PYTHON="${PYTHON:-3.13}"
VENV="$ROOT/.venv-mac-$TORCH"
IMAGE="${IMAGE:-varco-torch:$TORCH}"
PORT="${BRIDGE_PORT:-47700}"
# The Hugging Face cache the test programs read their models from, on the Mac and (mounted
# read-only at /hub) in the container.
HUB="${HUB:-$HOME/.cache/huggingface/hub}"
# Which engine runs the container: Docker Desktop (the default) or ENGINE=kiri.
# In Kiri the container is started with --gpus and reaches the Mac through Kiri's own channel:
# 127.0.0.1:47700 inside the container's own network, which Kiri forwards to a Unix socket on
# the Mac, so that is where the server listens (server.sh). Kiri sets KIRI_GPU_BRIDGE in the
# container and the bridge reads it (client.py): nothing else is told to the program.
# BOX names the container's side in the files the scripts write, so that a run in Kiri does not
# replace one in Docker Desktop.
ENGINE="${ENGINE:-docker}"
BOX=box
case "$ENGINE" in
    kiri)
        export DOCKER_HOST="unix://$HOME/.kiri/run/docker.sock"
        GPUS=1
        WHERE="${WHERE:-$HOME/.kiri/run/gpu.sock}"
        BOX=kiri ;;
    docker)
        export DOCKER_HOST="${DOCKER_HOST:-unix://$HOME/.docker/run/docker.sock}" ;;
    *) echo "ENGINE is docker or kiri, not $ENGINE" >&2; exit 1 ;;
esac
# The bridge's Python in the image, for the scripts that mount this folder's over it: a change to
# it then needs no rebuild (native.cpp is in the image: ./build-image.sh).
SITE="/usr/local/lib/python$PYTHON/site-packages/bridge"
