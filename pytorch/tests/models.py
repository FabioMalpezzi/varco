"""An ordinary program: BGE-M3 embeddings and the BGE reranker, with transformers.

It knows nothing of the bridge. It takes the GPU if PyTorch says there is
one, loads the two models from the Hugging Face cache, and prints scores
and timings that compare across machines.
"""

import os
import time

import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

HUB = os.environ.get("MODELS", os.path.expanduser("~/.cache/huggingface/hub"))
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {device}")

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
# A heavier load on request: longer passages (LONGER) and more of them (MORE).
PASSAGES = [p * int(os.environ.get("LONGER", "1")) for p in PASSAGES] * (2 * int(os.environ.get("MORE", "1")))


def snapshot(name):
    folder = f"{HUB}/models--{name.replace('/', '--')}/snapshots"
    return f"{folder}/{sorted(os.listdir(folder))[0]}"


def load(cls, name):
    start = time.perf_counter()
    path = snapshot(name)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = cls.from_pretrained(path).eval()
    read = time.perf_counter()
    model = model.to(device)
    float(next(model.parameters()).detach().flatten()[0])  # wait until the weights are really there
    print(f"  loaded {name}: {read - start:.1f} s from disk, {time.perf_counter() - read:.1f} s onto {device}")
    return tokenizer, model


def timed(fn, repeats=int(os.environ.get("REPEATS", "5"))):
    fn()  # once to warm up
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - start) * 1000)
    return result, f"{sum(times) / repeats:.1f} ms (best {min(times):.1f})"


print("BGE-M3 (embeddings)")
tokenizer, model = load(AutoModel, "BAAI/bge-m3")


@torch.no_grad()
def embed(texts):
    batch = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    vectors = model(**batch).last_hidden_state[:, 0]
    return torch.nn.functional.normalize(vectors, dim=-1).cpu()


vectors, ms = timed(lambda: embed([QUERY] + PASSAGES))
similarity = vectors[1:9] @ vectors[0]
print(f"  {len(PASSAGES) + 1} texts in {ms}")
print("  similarity to the query: " + " ".join(f"{s:.4f}" for s in similarity.tolist()))
del model

print("BGE reranker v2 M3")
tokenizer, model = load(AutoModelForSequenceClassification, "BAAI/bge-reranker-v2-m3")


@torch.no_grad()
def rerank(query, passages):
    batch = tokenizer([[query, p] for p in passages], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    return model(**batch).logits.view(-1).float().cpu()


scores, ms = timed(lambda: rerank(QUERY, PASSAGES))
print(f"  {len(PASSAGES)} pairs in {ms}")
print("  scores: " + " ".join(f"{s:.3f}" for s in scores[:8].tolist()))
order = sorted(range(8), key=lambda i: -scores[i])
print(f"  best passage: {PASSAGES[order[0]]}")
