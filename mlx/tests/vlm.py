"""An ordinary mlx-vlm program with a text prompt: loads a model and generates
text, twice, with timings (as tests/qwen.py does with mlx-lm).
    python vlm.py MODEL_DIR [MAX_TOKENS]
"""
import sys
import time

import mlx.core as mx
from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template

start = time.perf_counter()
model, processor = load(sys.argv[1])
print(f"device {mx.default_device()}; loaded in {time.perf_counter() - start:.1f} s", flush=True)
question = "Spiega in tre frasi che cos'è un container Linux."
try:
    prompt = apply_chat_template(processor, model.config, question, enable_thinking=False)
except TypeError:
    prompt = apply_chat_template(processor, model.config, question)
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 120
for attempt in ("first", "second"):
    text, last = "", None
    start = time.perf_counter()
    for response in stream_generate(model, processor, prompt, max_tokens=limit, temperature=0.0):
        text += response.text
        last = response
    print(f"--- {attempt} run, {time.perf_counter() - start:.1f} s: prompt {last.prompt_tokens} tokens at {last.prompt_tps:.1f} per second; "
          f"generation {last.generation_tokens} tokens at {last.generation_tps:.1f} per second", flush=True)
    print(text.strip(), flush=True)
