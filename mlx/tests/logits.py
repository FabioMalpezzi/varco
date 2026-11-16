"""The token and a hash of the log-probabilities of each step of a generation
with mlx-lm: where a container first differs from native, bit for bit.
    python logits.py MODEL_DIR [MAX_TOKENS]
"""
import hashlib
import sys

import numpy as np

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step

model, tokenizer = load(sys.argv[1])
messages = [{"role": "user", "content": "Spiega in tre frasi che cos'è un container Linux."}]
try:
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
except TypeError:
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 60
for n, (token, logprobs) in enumerate(generate_step(mx.array(prompt), model, max_tokens=limit)):
    data = np.array(logprobs.astype(mx.float32))
    print(f"step {n} token {int(token)} {hashlib.sha256(data.tobytes()).hexdigest()[:12]} max {float(data.max()):.6f}", flush=True)
