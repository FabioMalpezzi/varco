"""One of several programs generating at the same time: an ordinary mlx-lm loop that starts at an agreed moment.

Loads the model, waits for START (seconds since the epoch, the same for every copy),
then generates the same answer several times. One line per generation, with the
tokens per second and a digest of the text, and a last line that sums the run up.
Run alone and beside others: the digest should not change, and the rates tell what
sharing the GPU costs.
    python multi_gen.py MODEL_DIR START [GENERATIONS] [NAME]
"""
import hashlib
import statistics
import sys
import time

from mlx_lm import load, stream_generate

model_dir, start_at = sys.argv[1], float(sys.argv[2])
generations = int(sys.argv[3]) if len(sys.argv) > 3 else 8
name = sys.argv[4] if len(sys.argv) > 4 else "one"

began = time.time()
model, tokenizer = load(model_dir)
loaded = time.time()
messages = [{"role": "user", "content": "Spiega in tre frasi che cos'è un container Linux."}]
try:
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
except TypeError:
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
late = loaded - start_at
if late > 0:
    print(f"{name}: loaded {late:.1f} s after the start: this copy began late", flush=True)
else:
    time.sleep(-late)

rates, digests, tokens, first = [], set(), 0, time.time()
for generation in range(generations):
    text, last = "", None
    for response in stream_generate(model, tokenizer, prompt, max_tokens=120):
        text += response.text
        last = response
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    rates.append(last.generation_tps)
    tokens += last.generation_tokens
    digests.add(digest)
    print(f"{name}: generation {generation}: {last.generation_tokens} tokens at {last.generation_tps:.1f} per second, text {digest}", flush=True)
elapsed = time.time() - first
print(f"SUMMARY {name}: load {loaded - began:.1f} s; {generations} generations, {tokens} tokens in {elapsed:.1f} s, {tokens / elapsed:.1f} per second; "
      f"tokens per second median {statistics.median(rates):.1f}, min {min(rates):.1f}, max {max(rates):.1f}; "
      f"texts {sorted(digests)}", flush=True)
