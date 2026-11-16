"""An ordinary mlx-lm program: loads a model and generates text, twice, with timings.

Run on the Mac and in a container (where MLX's GPU is the Mac's, through the bridge):
the text should be the same, and the tokens per second comparable.
    python qwen.py MODEL_DIR [MAX_TOKENS]
"""
import sys
import time

import mlx.core as mx
from mlx_lm import load, stream_generate

start = time.perf_counter()
model, tokenizer = load(sys.argv[1])
print(f"device {mx.default_device()}; loaded in {time.perf_counter() - start:.1f} s", flush=True)
messages = [{"role": "user", "content": "Spiega in tre frasi che cos'è un container Linux."}]
try:
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
except TypeError:
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 120
for attempt in ("first", "second"):
    text, last = "", None
    start = time.perf_counter()
    for response in stream_generate(model, tokenizer, prompt, max_tokens=limit):
        text += response.text
        last = response
    print(f"--- {attempt} run, {time.perf_counter() - start:.1f} s: prompt {last.prompt_tokens} tokens at {last.prompt_tps:.1f} per second; "
          f"generation {last.generation_tokens} tokens at {last.generation_tps:.1f} per second", flush=True)
    print(text.strip(), flush=True)
