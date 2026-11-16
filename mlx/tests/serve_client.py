"""The same requests to two model servers, one on the Mac and one in a container.

Both speak the chat completions API of mlx-lm's server (python -m mlx_lm
server). The GPU slows as it warms, so every measure is taken in mirrored
order (Mac, container, container, Mac, and again) and compared pair by pair:
the two requests of a pair are the same, byte for byte.
A server that was left alone for a second or more answers late once, on
either side (the stage "idle" measures it): so each turn of a server starts
with a short request that is not measured.
    python serve_client.py --mac URL --container URL [--stages a,b] [--blocks N]
                           [--max-tokens N] [--long-words N] [--out FILE.json]
Stages (default: all):
    single      one request, answered whole, the same every time (so that the
                server has been at those positions of its cache before):
                seconds per request
    stream      a streamed request with a prompt of another length each time
                (positions the server has not been at): time to the first
                token, tokens per second
    long        a prompt of a few thousand tokens, new each time: tokens of
                prompt per second, time to the first token; then the server's
                prompt cache, with a second request that starts the same
                (another question) and a third that goes on from the answer
    concurrent  2 and 4 requests at once: tokens per second in total, and
                whether each answer is the one to its own question (the
                server computes them together, as they happen to arrive: on
                a busy machine two runs of the same server can differ)
    idle        a streamed request after 3 seconds in which nothing ran:
                time to the first token
The exit status is 0 only when every request was answered with a text, the
text of every pair is the same on the two servers, and every answer to
requests sent at once is the one to its own question. The text of those is
not held to the Mac's (NOT_BITWISE below says why).
Needs nothing but the standard library.
"""
import argparse
import hashlib
import http.client
import json
import random
import statistics
import sys
import threading
import time
from urllib.parse import urlparse

QUESTION = "Scrivi un racconto di almeno trecento parole su un faro e il suo guardiano."
# What the concurrent requests ask: each answer must name its own subject.
SUBJECTS = ["gatto", "treno", "vulcano", "violino"]
# The kinds of request whose text is not held to the Mac's, byte for byte:
# requests sent at once are computed together as they happen to arrive, and
# the texts differ from the Mac's in some pairs (docs/compatibility.md, Known
# faults). They are held to what is said of them there: each answer is the
# one to its own question. Every other kind must give the Mac's text.
NOT_BITWISE = {"2 at once", "4 at once"}
WORDS = (
    "casa mare vento luce strada ponte fiume monte campo porta libro tempo "
    "notte giorno acqua fuoco terra cielo pietra legno ferro vetro carta voce "
    "mano occhio passo viaggio città paese giardino finestra tavolo sedia "
    "lettera numero colore suono silenzio ombra sole luna stella nuvola pioggia"
).split()


def document(nonce, words):
    """A text of about that many words, different for every nonce from its start."""
    rng = random.Random(nonce)
    lines = [f"Documento numero {nonce}."]
    count = 0
    n = 0
    while count < words:
        n += 1
        length = rng.randint(8, 16)
        lines.append(f"{n}. " + " ".join(rng.choice(WORDS) for _ in range(length)) + ".")
        count += length
    return "\n".join(lines)


class Server:
    def __init__(self, name, url):
        self.name = name
        u = urlparse(url)
        self.host, self.port = u.hostname, u.port

    def chat(self, messages, max_tokens, stream=False):
        """One request. Returns its timings, what it used and the text."""
        body = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        data = json.dumps(body).encode()
        conn = http.client.HTTPConnection(self.host, self.port, timeout=600)
        start = time.perf_counter()
        conn.request("POST", "/v1/chat/completions", data, {"Content-Type": "application/json"})
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f"{self.name}: HTTP {response.status}: {response.read()[:300]!r}")
        r = {"first": None, "tps": None}
        text, usage, finish = "", {}, None
        if not stream:
            answer = json.loads(response.read())
            end = time.perf_counter()
            message = answer["choices"][0]["message"]
            text = (message.get("reasoning") or "") + (message.get("content") or "")
            usage = answer.get("usage", {})
            finish = answer["choices"][0].get("finish_reason")
        else:
            first = last = None
            for line in response:
                if not line.startswith(b"data: "):
                    continue  # blank lines, and the keepalives of a long prompt
                payload = line[6:].strip()
                if payload == b"[DONE]":
                    break
                chunk = json.loads(payload)
                now = time.perf_counter()
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    piece = (delta.get("reasoning") or "") + (delta.get("content") or "")
                    if piece:
                        text += piece
                        first = first or now
                        last = now
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
                        last = now
            end = time.perf_counter()
            if first is not None:
                r["first"] = first - start
                tokens = usage.get("completion_tokens", 0)
                if tokens > 1 and last > first:
                    r["tps"] = (tokens - 1) / (last - first)
        conn.close()
        r["seconds"] = end - start
        r["prompt_tokens"] = usage.get("prompt_tokens")
        r["completion_tokens"] = usage.get("completion_tokens")
        r["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        r["finish"] = finish
        r["text"] = text
        r["sha"] = hashlib.sha256(text.encode()).hexdigest()[:16]
        return r


def single(server, pair, args):
    return {"single": server.chat([{"role": "user", "content": QUESTION}], args.max_tokens)}


def stream(server, pair, args):
    # Some 200 tokens longer for each pair: no answer is at the positions of
    # another.
    content = document(args.nonce + 50 + pair, 150 * (pair + 1)) + "\n\n" + QUESTION
    return {"stream": server.chat([{"role": "user", "content": content}], args.max_tokens, stream=True)}


def long_prompt(server, pair, args):
    doc = document(args.nonce + pair, args.long_words)
    system = {"role": "system", "content": doc}
    ask = {"role": "user", "content": "Quante righe numerate ha il documento? Rispondi in una frase."}
    out = {}
    out["long"] = a = server.chat([system, ask], args.max_tokens, stream=True)
    if a["first"] and a["prompt_tokens"]:
        a["prompt_tps"] = (a["prompt_tokens"] - (a["cached_tokens"] or 0)) / a["first"]
    other = {"role": "user", "content": "Qual è la prima parola della riga 3? Rispondi in una frase."}
    out["same start"] = server.chat([system, other], args.max_tokens, stream=True)
    more = {"role": "user", "content": "E la seconda parola della stessa riga?"}
    out["follow up"] = server.chat(
        [system, other, {"role": "assistant", "content": out["same start"]["text"]}, more],
        args.max_tokens,
        stream=True,
    )
    return out


def concurrent(n):
    def run(server, pair, args):
        results = [None] * n
        barrier = threading.Barrier(n + 1)

        def one(i):
            subject = SUBJECTS[i]
            content = (
                f"Richiesta {args.nonce + pair}. Scrivi un racconto di almeno trecento parole "
                f"il cui protagonista è un {subject}. Usa la parola {subject} già nella prima frase."
            )
            barrier.wait()
            try:
                results[i] = server.chat([{"role": "user", "content": content}], args.max_tokens)
            except Exception as e:  # reported below as a wrong answer
                results[i] = {"error": str(e), "text": "", "sha": "", "completion_tokens": 0}

        threads = [threading.Thread(target=one, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        barrier.wait()
        start = time.perf_counter()
        for t in threads:
            t.join()
        seconds = time.perf_counter() - start
        tokens = sum(r["completion_tokens"] or 0 for r in results)
        # Each answer is about the subject of its own question.
        right = sum(SUBJECTS[i] in r["text"].lower() for i, r in enumerate(results))
        return {
            f"{n} at once": {
                "seconds": seconds,
                "completion_tokens": tokens,
                "tps": tokens / seconds,
                "right": right,
                "of": n,
                "cached_tokens": [r.get("cached_tokens") for r in results],
                "sha": hashlib.sha256("".join(r["sha"] for r in results).encode()).hexdigest()[:16],
                "text": "\n---\n".join(r["text"] for r in results),
                "errors": [r["error"] for r in results if "error" in r],
            }
        }

    return run


def idle(server, pair, args):
    time.sleep(3)
    r = server.chat([{"role": "user", "content": QUESTION}], 20, stream=True)
    return {"after 3 s idle": {k: r[k] for k in ("first", "seconds", "sha", "text")}}


def settle(server):
    """The short request that starts a turn."""
    server.chat([{"role": "user", "content": "Ciao."}], 4)


STAGES = {
    "single": [single],
    "stream": [stream],
    "long": [long_prompt],
    "concurrent": [concurrent(2), concurrent(4)],
    "idle": [idle],
}

# What is printed of each kind of request: (key, heading, format).
COLUMNS = [
    ("seconds", "seconds", "{:.3f}"),
    ("first", "first token s", "{:.3f}"),
    ("tps", "tokens/s", "{:.1f}"),
    ("prompt_tps", "prompt tokens/s", "{:.0f}"),
    ("prompt_tokens", "prompt", "{}"),
    ("cached_tokens", "cached", "{}"),
    ("completion_tokens", "generated", "{}"),
    ("right", "right answers", "{}"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mac", required=True)
    p.add_argument("--container", required=True)
    p.add_argument("--stages", default="single,stream,long,concurrent,idle")
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=120)
    p.add_argument("--long-words", type=int, default=1500)
    p.add_argument("--nonce", type=int, default=int(time.time()) % 100000 * 100)
    p.add_argument("--out")
    args = p.parse_args()
    servers = {"mac": Server("mac", args.mac), "container": Server("container", args.container)}

    report = {"args": vars(args), "first": {}, "runs": {}}
    failed = []
    # The first request a process answers is slower: on its own line.
    for name in ("mac", "container"):
        r = servers[name].chat([{"role": "user", "content": QUESTION}], args.max_tokens)
        report["first"][name] = r
        print(
            f"first request, {name}: {r['seconds']:.3f} s, {r['prompt_tokens']} + "
            f"{r['completion_tokens']} tokens, {r['sha']}",
            flush=True,
        )
    first = report["first"]
    if not first["mac"]["text"] or first["mac"]["sha"] != first["container"]["sha"]:
        failed.append("first request: the two texts are not the same, or there is none")

    for stage in args.stages.split(","):
        for unit in STAGES[stage]:
            runs = {"mac": [], "container": []}
            for block in range(args.blocks):
                # Mac, container, container, Mac: the two of a pair get the
                # same request.
                for name, pair in (("mac", 0), ("container", 0), ("container", 1), ("mac", 1)):
                    args_pair = 2 * block + pair
                    settle(servers[name])
                    result = unit(servers[name], args_pair, args)
                    result["_pair"] = args_pair
                    runs[name].append(result)
            for kind in [k for k in runs["mac"][0] if not k.startswith("_")]:
                rows = {name: sorted(runs[name], key=lambda r: r["_pair"]) for name in runs}
                mac = [r[kind] for r in rows["mac"]]
                box = [r[kind] for r in rows["container"]]
                same = [a["sha"] == b["sha"] for a, b in zip(mac, box)]
                print(f"=== {kind}: text identical in {sum(same)} of {len(same)} pairs")
                for key, heading, fmt in COLUMNS:
                    if mac[0].get(key) is None:
                        continue
                    line = f"  {heading:16}"
                    for name, values in (("mac", mac), ("container", box)):
                        line += f" {name} " + " ".join(fmt.format(v[key]) for v in values)
                    if isinstance(mac[0][key], float):
                        ratios = [b[key] / a[key] for a, b in zip(mac, box)]
                        line += "   container/mac " + " ".join(f"{x:.3f}" for x in ratios)
                        line += f"  median {statistics.median(ratios):.3f}"
                    print(line, flush=True)
                for r in mac + box:
                    if r.get("errors"):
                        print("  ERRORS:", r["errors"])
                if any(not r["text"] or r.get("errors") for r in mac + box):
                    failed.append(f"{kind}: a request was not answered with a text")
                if kind in NOT_BITWISE:
                    wrong = sum(r["of"] - r["right"] for r in mac + box)
                    if wrong:
                        failed.append(f"{kind}: {wrong} answers are not the ones to their own question")
                elif not all(same):
                    failed.append(f"{kind}: the text differs in {len(same) - sum(same)} of {len(same)} pairs")
                report["runs"][kind] = {"mac": mac, "container": box, "identical": same}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1, ensure_ascii=False)
    for what in failed:
        print("FAILED", what)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
