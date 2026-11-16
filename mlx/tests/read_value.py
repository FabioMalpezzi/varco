"""A generation whose program reads a value of every token, as a model server does.

mlx-lm's server asks for the log-probability of each token it sends
(logprobs[token].item()) and, on request, for the most likely alternatives:
values that depend on the token, computed from an array the step already
made. Timed here without the server, natively and in a container:
    python read_value.py MODEL_DIR [MAX_TOKENS] [MODES]
MODES, comma separated (default all): none (the plain loop of tests/qwen.py),
item (one value per token), top (the 5 most likely tokens and their values),
both. Each mode prints tokens per second and a digest of the text and of the
values read: the digests must be the same on both sides.
"""
import hashlib
import struct
import sys
import time

import mlx.core as mx
from mlx_lm import load, stream_generate

start = time.perf_counter()
model, tokenizer = load(sys.argv[1])
print(f"device {mx.default_device()}; loaded in {time.perf_counter() - start:.1f} s", flush=True)
messages = [{"role": "user", "content": "Scrivi un racconto di almeno trecento parole su un faro e il suo guardiano."}]
try:
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
except TypeError:
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 120
modes = sys.argv[3].split(",") if len(sys.argv) > 3 else ["none", "item", "top", "both"]


def run(mode):
    text, last = "", None
    values = hashlib.sha256()
    for response in stream_generate(model, tokenizer, prompt, max_tokens=limit):
        text += response.text
        last = response
        if mode in ("item", "both"):
            values.update(struct.pack("<f", response.logprobs[response.token].item()))
        if mode in ("top", "both"):
            order = mx.argpartition(-response.logprobs, kth=4)
            top = order[:5].tolist()
            values.update(struct.pack("<5i", *sorted(top)))
            values.update(struct.pack("<5f", *sorted(response.logprobs[top].tolist())))
    return text, last, values.hexdigest()[:16]


for mode in modes:
    run(mode)
    rates = []
    for _ in range(3):
        text, last, values = run(mode)
        rates.append(last.generation_tps)
    print(f"--- {mode}: {last.generation_tokens} tokens at " + " ".join(f"{r:.1f}" for r in rates)
          + f" per second; text {hashlib.sha256(text.encode()).hexdigest()[:16]}, values {values}", flush=True)
