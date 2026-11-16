"""A client of an OpenAI-compatible server, with nothing but the standard library.

Sends, to the chat completions endpoint of the server at URL:
  1. one request, answered whole;
  2. one streamed request: time to the first token, tokens per second;
  3. a long document three times: with a question, with another question,
     and with the next turn of that conversation; the server takes the
     common start from its prompt cache;
  4. four requests at once: the server computes them together.
    python client.py [URL] [MAX_TOKENS]      (default http://127.0.0.1:18080, 120)
"""
import json
import sys
import threading
import time
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18080"
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 120


def post(messages, stream=False, max_tokens=MAX_TOKENS):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": stream,
            "chat_template_kwargs": {"enable_thinking": False}}
    if stream:
        body["stream_options"] = {"include_usage": True}
    request = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=600)


def text_of(message):
    return message.get("content") or message.get("reasoning") or ""


def ask(messages, **options):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    start = time.perf_counter()
    with post(messages, **options) as response:
        reply = json.load(response)
    return time.perf_counter() - start, reply


def cached(usage):
    return (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)


# 1. One request.
ask("Say hello.", max_tokens=8)  # the first request of a server is slower: not measured
seconds, reply = ask("Explain in three sentences what a Linux container is.")
usage = reply["usage"]
print(f"1. one request: {usage['completion_tokens']} tokens in {seconds:.2f} s")
print("   " + " ".join(text_of(reply["choices"][0]["message"]).split())[:300])

# 2. Streaming.
start = time.perf_counter()
first, tokens = None, 0
with post([{"role": "user", "content": "Write a short story about a lighthouse and its keeper."}], stream=True) as response:
    for line in response:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        event = json.loads(line[6:])
        if event.get("choices") and text_of(event["choices"][0].get("delta", {})) and first is None:
            first = time.perf_counter()
        if event.get("usage"):
            tokens = event["usage"]["completion_tokens"]
end = time.perf_counter()
print(f"2. streaming: first token after {first - start:.3f} s, then {tokens / (end - first):.1f} tokens per second ({tokens} tokens)")

# 3. The prompt cache: a long document, the same document with another question, then the
# next turn of that conversation.
document = {"role": "system", "content": f"Document {time.time()}.\n" + "\n".join(f"{i}. The value of item {i} is {i * 7 % 1000}." for i in range(1, 251))}
first = {"role": "user", "content": "How many numbered lines does the document have? Answer in one sentence."}
other = {"role": "user", "content": "What is the value of item 3? Answer in one sentence."}
more = {"role": "user", "content": "And the value of item 5?"}
answer = None
for label, messages in (("first time", [document, first]), ("same start, another question", [document, other]),
                        ("next turn of that conversation", None)):
    if messages is None:
        messages = [document, other, {"role": "assistant", "content": answer}, more]
    seconds, reply = ask(messages, max_tokens=24)
    usage, answer = reply["usage"], text_of(reply["choices"][0]["message"])
    print(f"3. long prompt, {label}: {usage['prompt_tokens']} prompt tokens, {cached(usage)} from the cache, {seconds:.2f} s")

# 4. Four requests at once.
subjects = ["a cat", "a train", "a volcano", "a violin"]
replies = [None] * len(subjects)


def one(i):
    replies[i] = ask(f"Write a few sentences about {subjects[i]}.")


threads = [threading.Thread(target=one, args=(i,)) for i in range(len(subjects))]
start = time.perf_counter()
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
elapsed = time.perf_counter() - start
total = sum(reply["usage"]["completion_tokens"] for _, reply in replies)
print(f"4. {len(subjects)} requests at once: {total} tokens in {elapsed:.2f} s, {total / elapsed:.1f} tokens per second in total")
