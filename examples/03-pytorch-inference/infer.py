"""An ordinary PyTorch program: sentence embeddings and text generation with transformers.

It knows nothing of the bridge. It takes the GPU PyTorch says there is
("cuda" on an NVIDIA machine, "mps" on a Mac and in a container of the image
varco-torch, where "mps" is the Mac's GPU through the bridge), and prints
results that should be the same on a Mac and in the container.
    python infer.py
The models come from the Hugging Face cache: sentence-transformers/all-MiniLM-L6-v2
and openai-community/gpt2.
"""
import time

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging

logging.disable_progress_bar()

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"torch {torch.__version__.split('+')[0]}, device {device}")

# Sentence embeddings.
name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer, model = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval().to(device)
sentences = [
    "How do I run a Linux container on a Mac?",
    "On macOS, containers run inside a lightweight Linux virtual machine.",
    "A container shares the kernel of the Linux system it runs on.",
    "Carbonara is made with guanciale, eggs, pecorino and black pepper.",
]
with torch.no_grad():
    batch = tokenizer(sentences, padding=True, return_tensors="pt").to(device)
    hidden = model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1)
    vectors = torch.nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1)
    scores = (vectors[1:] @ vectors[0]).cpu()
print(f"query: {sentences[0]}")
for score, sentence in zip(scores.tolist(), sentences[1:]):
    print(f"  {score:.4f}  {sentence}")

# Text generation, greedy: the same tokens every time.
name = "openai-community/gpt2"
tokenizer, model = AutoTokenizer.from_pretrained(name), AutoModelForCausalLM.from_pretrained(name).eval().to(device)
prompt = tokenizer("A Linux container is", return_tensors="pt").to(device)
with torch.no_grad():
    model.generate(**prompt, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)  # warm up
    start = time.perf_counter()
    out = model.generate(**prompt, max_new_tokens=100, min_new_tokens=100, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    seconds = time.perf_counter() - start
print(f"generated 100 tokens in {seconds:.2f} s, {100 / seconds:.0f} tokens per second")
print(" ".join(tokenizer.decode(out[0], skip_special_tokens=True).split())[:240])
