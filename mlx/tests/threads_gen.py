"""Several threads of one process generating at the same time from one copy of a model.

mlx-lm's stream_generate runs on a stream of the calling thread's own
(mx.new_thread_local_stream), so calling it from several threads with one model is
its own pattern. Each thread asks its own question, several times; the answers are
first taken one at a time, on one thread, and every answer given beside others must
be the same text. Prints each thread's rate, the rates added up (tokens over the
time the threads ran), and the memory held: one copy of the weights, however many
threads.
    python threads_gen.py MODEL_DIR THREADS [GENERATIONS]
"""
import hashlib
import resource
import sys
import threading
import time

import mlx.core as mx
from mlx_lm import load, stream_generate

model_dir, threads = sys.argv[1], int(sys.argv[2])
generations = int(sys.argv[3]) if len(sys.argv) > 3 else 6
QUESTIONS = [
    "Spiega in tre frasi che cos'è un container Linux.",
    "Spiega in tre frasi che cos'è una macchina virtuale.",
    "Spiega in tre frasi che cos'è una GPU.",
    "Spiega in tre frasi che cos'è un sistema operativo.",
]

model, tokenizer = load(model_dir)


def prompt_of(question):
    messages = [{"role": "user", "content": question}]
    try:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True)


def generate(prompt):
    text, last = "", None
    for response in stream_generate(model, tokenizer, prompt, max_tokens=120):
        text += response.text
        last = response
    return hashlib.sha256(text.encode()).hexdigest()[:12], last.generation_tokens, last.generation_tps


prompts = [prompt_of(q) for q in QUESTIONS]
# Each answer alone first (and once more, to tell a model that does not repeat itself).
alone = [generate(p) for p in prompts]
again = [generate(p) for p in prompts]
print("alone: " + ", ".join(f"q{i} {d} {n} tokens at {r:.1f}/s" for i, (d, n, r) in enumerate(alone))
      + ("" if [a[0] for a in alone] == [a[0] for a in again] else "  (NOT the same text twice, even alone)"), flush=True)

results = [None] * threads
barrier = threading.Barrier(threads)


def work(index):
    prompt, expected = prompts[index % len(prompts)], alone[index % len(prompts)][0]
    barrier.wait()
    start, tokens, rates, wrong = time.perf_counter(), 0, [], 0
    for _ in range(generations):
        digest, n, rate = generate(prompt)
        tokens += n
        rates.append(rate)
        wrong += digest != expected
    results[index] = (tokens, time.perf_counter() - start, sorted(rates)[len(rates) // 2], wrong)


pool = [threading.Thread(target=work, args=(i,)) for i in range(threads)]
began = time.perf_counter()
for t in pool:
    t.start()
for t in pool:
    t.join()
elapsed = time.perf_counter() - began
total = sum(r[0] for r in results)
for i, (tokens, seconds, median, wrong) in enumerate(results):
    print(f"thread {i} (q{i % len(prompts)}): {tokens} tokens in {seconds:.1f} s, {tokens / seconds:.1f} per second "
          f"(median while generating {median:.1f}); answers that differ from the one given alone: {wrong} of {generations}", flush=True)
print(f"TOTAL {threads} thread(s): {total / elapsed:.1f} tokens per second added up; wrong answers {sum(r[3] for r in results)} "
      f"of {threads * generations}; GPU memory in use {mx.get_active_memory() / 2**30:.1f} GiB, at its peak {mx.get_peak_memory() / 2**30:.1f} GiB; "
      f"process at its peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**30 if sys.platform == 'darwin' else 2**20):.1f} GiB", flush=True)
