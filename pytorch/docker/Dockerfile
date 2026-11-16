# Container end of the PyTorch bridge: PyTorch for Linux, CPU only, same version as the Mac's.
# Build from the directory that holds bridge/ and docker/:
#   ./build-image.sh   (docker build -f docker/Dockerfile --build-arg TORCH=... --build-arg PYTHON=... -t varco-torch:... .)
ARG PYTHON=3.13
FROM python:${PYTHON}-slim AS base
ARG TORCH=2.15.0
RUN pip install --no-cache-dir torch==${TORCH} --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
COPY docker/requirements-extra.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-extra.txt

# The container end's C++ (bridge/native.cpp): tensors PyTorch takes for MPS tensors. Built with
# the list of the operators this PyTorch version runs on a Mac's GPU (bridge/ops/, gen_ops.py).
FROM base AS native
ARG TORCH=2.15.0
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends g++ >/dev/null
COPY bridge/native.cpp /src/
COPY bridge/ops/mps_ops-${TORCH}.inc /src/mps_ops.inc
RUN T=$(python -c "import torch, os; print(os.path.dirname(torch.__file__))") && \
    g++ -shared -fPIC -O2 -std=c++20 -D_GLIBCXX_USE_CXX11_ABI=$(python -c "import torch; print(int(torch.compiled_with_cxx11_abi()))") \
        -I"$T/include" -I"$T/include/torch/csrc/api/include" -I"$(python -c "import sysconfig; print(sysconfig.get_paths()['include'])")" \
        /src/native.cpp -o "/bridge_native$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")" \
        -L"$T/lib" -ltorch -ltorch_cpu -lc10 -ltorch_python -Wl,-rpath,"$T/lib"

# Every Python process in the image gets the bridge when it imports torch (bridge/auto.py): the
# .pth file is what Python reads at startup. BRIDGE=0 in the environment turns it off.
FROM base
COPY --from=native /bridge_native*.so /site/
COPY bridge/*.py /site/bridge/
RUN S=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") && \
    mv /site/bridge_native*.so "$S/" && mv /site/bridge "$S/bridge" && rmdir /site && \
    echo "import bridge.auto" > "$S/bridge.pth"
WORKDIR /work
