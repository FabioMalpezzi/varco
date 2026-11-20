"""Check the native priority server with Qwen3.6-35B on a local port."""
import argparse
import concurrent.futures
import http.client
import json
import random
import statistics
import threading
import time


def chat(port, prompt, on_progress=None, disconnect=False):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": 12, "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=90)
    start = time.monotonic()
    pieces = []
    try:
        conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        with conn.getresponse() as response:
            if response.status != 200:
                raise RuntimeError(response.read().decode())
            for line in response:
                if line.startswith(b": keepalive "):
                    if disconnect:
                        return
                    if on_progress:
                        processed, total = line.split()[-1].split(b"/")
                        on_progress(int(processed), int(total))
                if line.startswith(b"data: "):
                    data = line[6:].strip()
                    if data == b"[DONE]":
                        assert pieces, "No text in completed response"
                        return "".join(text for _, text in pieces), pieces[0][0] - start
                    text = json.loads(data)["choices"][0]["delta"].get("content")
                    if text:
                        pieces.append((time.monotonic(), text))
            raise RuntimeError("Streaming response ended before [DONE]")
    finally:
        conn.close()


def run(port):
    rng = random.Random(time.time_ns())
    words = "amber lake fern ash clay gold stone wind".split()
    def document():
        return " ".join(rng.choices(words, k=8100)) + "\n\nReply with only the word JADE."

    short = "Reply with only the word OPAL."
    assert chat(port, short)[0] == "OPAL"
    chat(port, document(), disconnect=True)
    assert chat(port, short)[0] == "OPAL"

    started = threading.Event()
    finished = threading.Event()
    prompt = document()

    def long_client():
        try:
            assert chat(port, prompt, lambda *_: started.set())[0] == "JADE"
        finally:
            finished.set()

    def short_client():
        times = []
        while not finished.is_set():
            text, first = chat(port, short)
            assert text == "OPAL"
            if not finished.is_set():
                times.append(first)
        return times

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        long = pool.submit(long_client)
        assert started.wait(90), "Long request did not report prefill progress"
        shorts = [pool.submit(short_client) for _ in range(4)]
        try:
            long.result(timeout=120)
        finally:
            finished.set()
        batches = [client.result() for client in shorts]
    assert all(batches), "Each short client must receive text before the long request completes"
    times = [dt for batch in batches for dt in batch]
    print(f"{len(times)} short replies completed during the long request; median first text {statistics.median(times):.3f} s")
    print("The long request and the request following a disconnected client returned their expected text.")


def burst(port, round_number):
    """Send the same prompts to either server after a reported long-prompt boundary."""
    words = "amber lake fern ash clay gold stone wind".split()
    prompt = " ".join(random.Random(round_number).choices(words, k=8100))
    prompt += "\n\nReply with only the word JADE."
    boundary = threading.Event()
    def progress(processed, total):
        if 2048 <= processed < total:
            boundary.set()

    def short_client(index, word):
        assert boundary.wait(90), "No prefill boundary reported"
        time.sleep(index * 0.05)
        text, latency = chat(port, f"Reply with only the word {word}.")
        assert text == word
        return round(latency, 3)

    assert chat(port, "Reply with only the word RUBY.")[0] == "RUBY"
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        shorts = [pool.submit(short_client, i, word) for i, word in enumerate(("OPAL", "AMBER", "ONYX"), 1)]
        begin = time.monotonic()
        assert chat(port, prompt, progress)[0] == "JADE"
        long_seconds = time.monotonic() - begin
        print(json.dumps({"round": round_number, "short_first_text_s": [s.result() for s in shorts],
                          "long_completion_s": round(long_seconds, 3)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--burst-round", type=int, help="Run a reproducible three-client timing sample")
    args = parser.parse_args()
    if args.burst_round is None:
        run(args.port)
    else:
        burst(args.port, args.burst_round)
