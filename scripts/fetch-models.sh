#!/bin/sh
# Download the small public models the tests and the examples use.
#   scripts/fetch-models.sh
# MLX models go to MODELS (default $HOME/models) as plain directories; the
# PyTorch ones go to the Hugging Face cache. About 3.3 GB in total. Needs the
# hf command (uv tool install huggingface_hub).
set -e
MODELS=${MODELS:-$HOME/models}
command -v hf > /dev/null || { echo "the hf command is not installed: uv tool install huggingface_hub" >&2; exit 1; }
for model in mlx-community/bge-m3-mlx-fp16 Qwen/Qwen3.5-0.8B-Base; do
  hf download "$model" --local-dir "$MODELS/$model" > /dev/null
  echo "$MODELS/$model"
done
for model in sentence-transformers/all-MiniLM-L6-v2 openai-community/gpt2; do
  hf download "$model" > /dev/null
  echo "Hugging Face cache: $model"
done
