"""GPT-2 generating 200 tokens, timed: the workload of a text-generation server in miniature."""
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HUB = os.environ.get("MODELS", os.path.expanduser("~/.cache/huggingface/hub"))
folder = f"{HUB}/models--openai-community--gpt2/snapshots"
path = f"{folder}/{sorted(os.listdir(folder))[0]}"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tokenizer, model = AutoTokenizer.from_pretrained(path), AutoModelForCausalLM.from_pretrained(path).eval().to(device)
prompt = tokenizer("The container runs on the Mac and", return_tensors="pt").to(device)
tokens = int(os.environ.get("TOKENS", "200"))
with torch.no_grad():
    model.generate(**prompt, max_new_tokens=tokens, min_new_tokens=tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)  # warm up
    times = []
    for _ in range(int(os.environ.get("REPEATS", "3"))):
        start = time.perf_counter()
        out = model.generate(**prompt, max_new_tokens=tokens, min_new_tokens=tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        times.append(time.perf_counter() - start)
print(f"gpt2 {tokens / min(times):.1f} tokens per second (best of {len(times)}), {1000 * min(times) / tokens:.2f} ms per token, check {int(out.sum())}")
