#!/bin/sh
# The Mac end for one PyTorch version:   [TORCH=2.15.0] ./build-mac.sh
# Makes the environment .venv-mac-$TORCH if it is not there with all its packages (with uv), lists the operators that
# PyTorch runs on the GPU (bridge/ops/, which the image is built with), and builds the C++ hot
# path (bridge/executor.cpp) against it.
cd "$(dirname "$0")" || exit 1
. ./versions.sh
# The environment is complete when it holds the list of packages that was installed into it, which
# is copied there once uv has installed them all. Without that copy there is no environment yet,
# or one that a run which was interrupted left: a Python without its pyvenv.cfg, which uv takes
# for the Python it links to, or a package with its metadata and not all of its files, which uv
# takes for installed and cannot install again. Such an environment is removed and made anew.
# With a copy of another list than docker/requirements-extra.txt, the packages are installed again.
if ! cmp -s docker/requirements-extra.txt "$VENV/requirements-extra.txt"; then
    if command -v uv >/dev/null; then
        if [ ! -f "$VENV/requirements-extra.txt" ]; then
            rm -rf "$VENV"
            uv venv --python "${MAC_PYTHON:-3.12}" "$VENV" || exit 1
        fi
        rm -f "$VENV/requirements-extra.txt"
        uv pip install --python "$VENV/bin/python" "torch==$TORCH" -r docker/requirements-extra.txt || exit 1
        cp docker/requirements-extra.txt "$VENV/requirements-extra.txt" || exit 1
    elif [ ! -x "$VENV/bin/python" ]; then
        echo "uv is not installed: see https://docs.astral.sh/uv/ (or make $VENV yourself with torch==$TORCH and docker/requirements-extra.txt)" >&2
        exit 1
    fi
fi
py=$VENV/bin/python
"$py" gen_ops.py || exit 1
torch=$("$py" -c "import torch, os; print(os.path.dirname(torch.__file__))")
include=$("$py" -c "import sysconfig; print(sysconfig.get_paths()['include'])")
suffix=$("$py" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
mkdir -p "bridge/_build/$TORCH"
clang++ -O2 -std=c++20 -shared -fPIC -undefined dynamic_lookup -Wno-deprecated-declarations \
    -I"$torch/include" -I"$torch/include/torch/csrc/api/include" -I"$include" \
    bridge/executor.cpp -o "bridge/_build/$TORCH/_executor$suffix" \
    -L"$torch/lib" -ltorch -ltorch_cpu -lc10 -ltorch_python -Wl,-rpath,"$torch/lib" || exit 1
echo "built bridge/_build/$TORCH/_executor$suffix"
