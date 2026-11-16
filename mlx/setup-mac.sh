#!/bin/sh
# The Mac's Python environment for the MLX bridge: ./.venv with the stock MLX
# wheel of the pinned version (its headers and library are what the server is
# built against), and the libraries the tests run natively.
#   ./setup-mac.sh
# Needs uv (https://docs.astral.sh/uv/). PYTHON picks the interpreter (3.12).
set -e
cd "$(dirname "$0")"
. ./versions.sh
if ! command -v uv > /dev/null; then
  echo "uv is not installed: see https://docs.astral.sh/uv/ (or make ./.venv yourself with the packages listed in this script)" >&2
  exit 1
fi
# The environment is complete when it has the mark .venv/packages-installed,
# which is removed before uv installs and written when it has installed every
# package. Without the mark there is no environment yet, or one that a run
# which was interrupted left: a Python without its pyvenv.cfg, which uv takes
# for the Python it links to, or a package with its metadata and not all of
# its files, which uv takes for installed and cannot install again. Such an
# environment is removed and made anew.
if [ ! -f .venv/packages-installed ]; then
  rm -rf .venv
  uv venv --quiet --python "${PYTHON:-3.12}" .venv
fi
rm -f .venv/packages-installed
uv pip install --quiet --python .venv/bin/python \
  "mlx==$MLX_VERSION" "mlx-lm==$MLX_LM_VERSION" "mlx-embeddings==$MLX_EMBEDDINGS_VERSION" numpy
: > .venv/packages-installed
.venv/bin/python -c "import mlx.core as mx; print('mlx', mx.__version__, 'on', mx.default_device())"
