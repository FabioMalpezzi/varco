"""Where a slow generation loses its time: the gap before each token, over several generations.

One line per generation: tokens per second, the median gap, the three longest gaps, the
tokens they come before and the wall-clock time the longest one ended at. A run that is slow all along has a longer median; one that
stalled once has a single long gap.
    python token_gaps.py MODEL_DIR [GENERATIONS]
"""
import statistics
import sys
import time

from mlx_lm import load, stream_generate

model, tokenizer = load(sys.argv[1])
messages = [{"role": "user", "content": "Spiega in tre frasi che cos'è un container Linux."}]
try:
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
except TypeError:
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
for generation in range(int(sys.argv[2]) if len(sys.argv) > 2 else 10):
    gaps, at, last = [], [], None
    before = time.perf_counter()
    for response in stream_generate(model, tokenizer, prompt, max_tokens=120):
        now = time.perf_counter()
        gaps.append((now - before) * 1000)
        at.append(time.time())
        before, last = now, response
    rest = gaps[1:]  # the first gap is the prompt
    longest = sorted(range(len(rest)), key=lambda i: -rest[i])[:3]
    print(f"{generation:2d}: {last.generation_tps:6.1f} tokens per second; first {gaps[0]:6.1f} ms; median gap {statistics.median(rest):5.2f} ms; "
          + "longest " + ", ".join(f"{rest[i]:6.1f} ms before token {i + 2}" for i in longest)
          + f"; the longest ended at {time.strftime('%H:%M:%S', time.localtime(at[longest[0] + 1]))}.{int(at[longest[0] + 1] % 1 * 1000):03d}", flush=True)
