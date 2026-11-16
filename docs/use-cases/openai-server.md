# Use case 2: an OpenAI-compatible model server in a container

**What you get**

- mlx-lm's server, unchanged, as one more service of a Compose stack, with
  the Mac's text: identical to the Mac's in every pair with one request at
  a time (Qwen3.6-35B-A3B-4bit, Kiri, protocol 9).
- The weights stay out of the container. Its memory peaked at 456 MiB while
  it served a 19.5 GB model.
- One request of 120 tokens in 0.884 s, 125.6 tokens per second when
  streaming, and 215.0 tokens per second in total with 4 requests at once
  (medians of four pairs in mirrored order).

**Compared with**

- The same server installed on the Mac and called over HTTP, which was sent
  the same requests: 1.120 s for one request, 101.1 tokens per second when
  streaming, and 216.0 tokens per second in total with 4 requests at once.

**When it is not the right tool:** when answers to concurrent requests must
be the Mac's word for word. With 2 and with 4 requests at once the text
differed from the Mac's in 2 of 4 pairs and in 1 of 4.

**Who it is for.** Anyone who deploys with Compose and wants the language
model to be one more service of the stack, with its code and its versions in
an image, instead of a program installed on the Mac that the containers call
out to.

## What runs

mlx-lm's own server, `python -m mlx_lm server`, version 0.32.2, unchanged,
in a container of the image `varco-mlx-lm`. It serves the chat completions
API, streams, keeps a prompt cache, and batches requests that arrive
together. The weights are read by the bridge's server on the Mac; the
container holds none of them.

[`examples/02-openai-server/client.py`](../../examples/02-openai-server/client.py)
sends it four kinds of request with nothing but Python's standard library.

## Commands

```sh
examples/02-openai-server/run.sh                    # the small model, Qwen/Qwen3.5-0.8B-Base
examples/02-openai-server/run.sh Qwen/Qwen3.5-0.8B-Base --keep   # leave it running
curl -s http://127.0.0.1:18080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 20}'
examples/02-openai-server/run.sh --stop
ENGINE=kiri examples/02-openai-server/run.sh        # the model server in Kiri
```

The large model the measurements were taken with:

```sh
hf download mlx-community/Qwen3.6-35B-A3B-4bit \
  --local-dir "${MODELS:-$HOME/models}/mlx-community/Qwen3.6-35B-A3B-4bit"   # about 19 GB; not run for this page
KERNELS=1 examples/02-openai-server/run.sh mlx-community/Qwen3.6-35B-A3B-4bit
```

`KERNELS=1` starts the bridge's server with `--custom-kernels`. mlx-lm
computes some layers of that model with its own Metal kernels. Without the
option it takes another path, which gives another (correct) text and is
slower. [Custom kernels](custom-kernels.md) says what the option gives and
what it implies for the containers on that server.

The service listens on port 18080 of the Mac (`LLM_PORT` changes it).

## What you see

With the small model, on Docker Desktop:

```
1. one request: 106 tokens in 0.59 s
   A Linux container is a lightweight, isolated virtual environment that runs ...
2. streaming: first token after 0.059 s, then 196.9 tokens per second (120 tokens)
3. long prompt, first time: 4554 prompt tokens, 0 from the cache, 6.57 s
3. long prompt, same start, another question: 4554 prompt tokens, 4531 from the cache, 0.31 s
3. long prompt, next turn of that conversation: 4584 prompt tokens, 4550 from the cache, 0.14 s
4. 4 requests at once: 286 tokens in 0.96 s, 298.6 tokens per second in total
```

This is one run of the example, shown as an illustration. Two things are
worth looking at: the second and third long prompts are answered from the
cache, and four requests at once produce more tokens per second than one.
One run in Kiri ended with `4 requests at once: 283 tokens in 0.60 s, 471.0
tokens per second in total`.

## Measured

Qwen3.6-35B-A3B-4bit in Kiri at protocol 9, the same requests to the server
on the Mac and to the one in the container, medians of four pairs in
mirrored order ([benchmarks](../benchmarks.md#a-model-server-in-a-container-mlx-lms-own)):

| Case | Mac | Kiri container |
|---|---|---|
| One request, 120 tokens | 1.120 s | 0.884 s |
| Streaming | 101.1 tokens/s, first token 0.368 s | 125.6 tokens/s, first token 0.365 s |
| Long prompt (about 2,300 tokens) | first token 1.455 s, prompt at 1,569 tokens/s | first token 1.445 s, prompt at 1,586 tokens/s |
| Same long start again | 0.211 s, 2,262 of 2,289 tokens from the cache | 0.183 s, the same tokens from the cache |
| Next turn of the conversation | 0.212 s | 0.188 s |
| 2 requests at once | 152.9 tokens/s in total | 160.4 (two pairs at 186–187, two at 133–134) |
| 4 requests at once | 216.0 tokens/s in total | 215.0 |
| First request after 3 s idle | first token 0.437 s | 0.322 s |

Container memory at its peak: 456 MiB, for a model of 19.5 GB.

On Docker Desktop, as reported, container over Mac: one request 0.77,
streaming rate 1.21, 2 at once 1.20, 4 at once 1.05. With the small model,
Qwen3.5-0.8B-Base: streaming 1.22, 4 at once 1.27.

The text is identical to the Mac's in every pair with one request at a
time.

The prompt cache belongs to the server, so a conversation must stay on the
server that holds it. With two servers in two containers, sending the turns
of one conversation to the two alternately cost one more full reading of the
prompt, 10.73 s over six turns instead of 7.49 s
([benchmarks](../benchmarks.md#keeping-a-conversation-on-one-server)).

## Limits

- With several requests at once, each answer is the one to its own question,
  but the text differs from the Mac's in 2 of 4 pairs with two requests and
  in 1 of 4 with four. The server groups requests by when they arrive. This
  was not looked into.
- A request with `top_logprobs` is slower in the container, at 88–90 tokens
  per second where the Mac gives 100–108.
- The first prompt of a process is slower in the container.
- `/v1/models` fails in the container because of mlx-lm itself, which scans
  a Hugging Face cache directory that is not there.
- Without `--custom-kernels`, models for which mlx-lm has its own kernels
  run on its other path. Qwen3.6-35B gave 105 tokens instead of 93, at
  118–119 tokens per second instead of 141. With mlx-lm's server,
  Qwen3.5-0.8B-Base gave the Mac's text in 2 of 2 pairs with the option and
  in none without it (`mlx/tests/serve.sh`, one request at a time, 16
  November 2026).
- For throughput use one server, not several. One server that batches
  reached 215 tokens per second with 4 requests, while containers that each
  have their own copy of the model share the GPU at about 1.4 times the
  throughput of one ([several containers](several-containers.md)).

Further measurements will be published as they are taken.
