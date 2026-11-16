"""Loads BGE-M3 and the BGE reranker, then runs one heavy pass for each line on stdin
("embed" or "rerank") and prints its time in ms. An ordinary program: heavy_ab.py
runs one copy on the Mac and one in the container, and alternates them pass by pass."""
import os
import sys
import time

import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

HUB = os.environ.get("MODELS", os.path.expanduser("~/.cache/huggingface/hub"))
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
QUERY = "Come si esegue un container Linux su un Mac?"
PASSAGES = [
    "Kiri esegue container Linux su macOS dentro una macchina virtuale leggera.",
    "Docker Desktop avvia una macchina virtuale Linux e vi fa girare i container.",
    "La ricetta della carbonara prevede guanciale, uova, pecorino e pepe.",
    "Apple's Virtualization framework boots Linux guests on Apple silicon.",
    "Il Giro d'Italia è una corsa ciclistica a tappe che si disputa a maggio.",
    "A container shares the kernel of the Linux system it runs on.",
    "MLX is an array framework for machine learning on Apple silicon.",
    "Per installare un'applicazione su Mac si trascina l'icona in Applicazioni.",
]
PASSAGES = [p * 8 for p in PASSAGES] * 16  # 128 long passages


def load(cls, name):
    folder = f"{HUB}/models--{name.replace('/', '--')}/snapshots"
    path = f"{folder}/{sorted(os.listdir(folder))[0]}"
    return AutoTokenizer.from_pretrained(path), cls.from_pretrained(path).eval().to(device)


embed_tokenizer, embed_model = load(AutoModel, "BAAI/bge-m3")
rerank_tokenizer, rerank_model = load(AutoModelForSequenceClassification, "BAAI/bge-reranker-v2-m3")


@torch.no_grad()
def embed():
    batch = embed_tokenizer([QUERY] + PASSAGES, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    return torch.nn.functional.normalize(embed_model(**batch).last_hidden_state[:, 0], dim=-1).cpu()


@torch.no_grad()
def rerank():
    batch = rerank_tokenizer([[QUERY, p] for p in PASSAGES], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    return rerank_model(**batch).logits.view(-1).float().cpu()


print("ready", flush=True)
for line in sys.stdin:
    start = time.perf_counter()
    result = embed() if line.strip() == "embed" else rerank()
    print(f"{(time.perf_counter() - start) * 1000:.1f} {float(result.abs().sum()):.4f}", flush=True)
