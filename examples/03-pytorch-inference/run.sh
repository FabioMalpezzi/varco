#!/bin/sh
# PyTorch inference (transformers) in a container, on the Mac's GPU.
#   ./run.sh            the container, through the bridge
#   ./run.sh native     the same program on the Mac directly, to compare
#   ./run.sh cpu        the same container with the bridge off (BRIDGE=0)
# ENGINE=kiri runs the container in Kiri, with the GPU request of compose.gpu.yaml.
# The two models are read from the Hugging Face cache (HF_CACHE):
#   hf download sentence-transformers/all-MiniLM-L6-v2
#   hf download openai-community/gpt2
cd "$(dirname "$0")" || exit 1
. ../common.sh
for model in sentence-transformers--all-MiniLM-L6-v2 openai-community--gpt2; do
  [ -d "$HF_CACHE/hub/models--$model" ] || die "no $model in $HF_CACHE/hub: download it with hf download $(echo $model | sed 's/--/\//')"
done
case "$1" in
  native)
    [ -x "$TORCH_PY" ] || die "no Mac environment: run pytorch/build-mac.sh"
    HF_HOME="$HF_CACHE" HF_HUB_OFFLINE=1 TRANSFORMERS_VERBOSITY=error exec "$TORCH_PY" infer.py ;;
  cpu)
    need_image "varco-torch:$TORCH" "scripts/setup-mac.sh pytorch"
    BRIDGE=0 exec docker compose run --rm infer ;;
esac
need_image "varco-torch:$TORCH" "scripts/setup-mac.sh pytorch"
start_torch_server
trap stop_torch_server EXIT
compose run --rm infer
