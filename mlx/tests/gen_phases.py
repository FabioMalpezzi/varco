"""Where the time of one step of tests/genloop.py goes (building the graph,
async_eval, reading the token) and how many nodes a step and a BGE-M3 forward
have. Usage: python gen_phases.py [--bge DIR] [--tokens T]"""
import collections, io, re, sys, tempfile, time
import numpy as np
import mlx.core as mx
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import genloop


def count_nodes(*arrays):
    with tempfile.NamedTemporaryFile(suffix=".dot", mode="r") as f:
        mx.export_to_dot(f.name, *arrays)
        text = f.read()
    return collections.Counter(re.findall(r'label ="([^"]+)"', text)), text


args = sys.argv[1:]
tokens = 200
if "--tokens" in args:
    i = args.index("--tokens"); tokens = int(args[i + 1]); del args[i:i + 2]

mx.random.seed(0)
model = genloop.LM(32000, 768, 12, 12)
model.set_dtype(mx.float16)
mx.eval(model.parameters())
prompt = mx.array([1, 17, 4096, 250, 31999, 7, 7, 1024])


def generate(n, count=False):
    cache = None

    def step(y):
        nonlocal cache
        logits, cache = model(y[None], cache)
        logits = logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return mx.argmax(logprobs, axis=-1), logprobs.squeeze(0)

    y, logprobs = step(prompt)
    mx.async_eval(y, logprobs)
    rows = []
    for i in range(n):
        if count and i == 5:
            # Only the nodes of this step: nothing older is still lazy.
            mx.eval(y, logprobs, cache)
        t0 = time.perf_counter()
        next_y, next_logprobs = step(y)
        t1 = time.perf_counter()
        if count and i == 5:
            c, text = count_nodes(next_y, next_logprobs)
            print("nodes of one step:", sum(c.values()), dict(c.most_common()))
        mx.async_eval(next_y, next_logprobs)
        t2 = time.perf_counter()
        y.item()
        t3 = time.perf_counter()
        rows.append((t1 - t0, t2 - t1, t3 - t2, t3 - t0))
        y, logprobs = next_y, next_logprobs
    return np.array(rows[1:]) * 1e6


generate(20, count=True)
generate(tokens)
best = None
for _ in range(4):
    r = generate(tokens)
    if best is None or r[:, 3].sum() < best[:, 3].sum():
        best = r
m = np.median(best, axis=0)
a = best.mean(axis=0)
print(f"step on {mx.default_device()} ({sys.platform}), microseconds per token, best run of 4: "
      f"build graph {a[0]:.0f}, async_eval {a[1]:.0f}, item {a[2]:.0f}, total {a[3]:.0f} "
      f"(medians {m[0]:.0f}, {m[1]:.0f}, {m[2]:.0f}, {m[3]:.0f}); {1e6 / a[3]:.0f} tokens/s")

if "--bge" in args:
    from mlx_embeddings.utils import load
    from bge_m3 import SENTENCES
    model, tok = load(args[args.index("--bge") + 1])
    inputs = tok.batch_encode_plus(SENTENCES[:1], return_tensors="mlx", padding=True, truncation=True, max_length=512)
    out = model(inputs["input_ids"], attention_mask=inputs["attention_mask"]).text_embeds
    c, text = count_nodes(out)
    print("nodes of one BGE-M3 forward:", sum(c.values()), dict(c.most_common()))
