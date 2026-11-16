"""What keeping a conversation on one server is worth: the same conversation of several turns,
first with every turn sent to the same model server, then with the turns sent to two servers in
turn. Prints, per turn, the prompt's tokens, how many the server took from its cache and the time.
    python affinity.py URL_A URL_B [TURNS] [CONTEXT_WORDS]
"""
import json
import sys
import time
import urllib.request

urls = sys.argv[1:3]
turns = int(sys.argv[3]) if len(sys.argv) > 3 else 6
words = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
questions = ["Che cos'è un container Linux?", "E una macchina virtuale?", "Quale dei due parte prima?", "Perché?",
             "Fai un esempio pratico.", "Riassumi in una frase.", "E in tre parole?", "Grazie, e i rischi?"]


def ask(url, messages):
    body = json.dumps({"model": "default_model", "messages": messages, "max_tokens": 40, "temperature": 0}).encode()
    start = time.perf_counter()
    with urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=300) as r:
        reply = json.load(r)
    usage = reply.get("usage", {})
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    message = reply["choices"][0]["message"]
    return time.perf_counter() - start, usage.get("prompt_tokens"), cached, message.get("content") or ""


for label, pick in (("one server", lambda turn: urls[0]), ("two servers in turn", lambda turn: urls[turn % 2])):
    # A different long context each time, so that nothing is in a cache from the pass before.
    context = f"Conversazione {label} {time.time()}. " + " ".join(f"parola{i}" for i in range(words))
    messages = [{"role": "system", "content": context}]
    total = 0.0
    print(f"=== {label}")
    for turn in range(turns):
        messages.append({"role": "user", "content": questions[turn % len(questions)]})
        seconds, prompt, cached, answer = ask(pick(turn), messages)
        messages.append({"role": "assistant", "content": answer})
        total += seconds
        print(f"  turn {turn + 1}: prompt {prompt} tokens, {cached} from the cache ({100 * cached / prompt:.0f}%), {seconds:.2f} s")
    print(f"  {turns} turns in {total:.2f} s")
