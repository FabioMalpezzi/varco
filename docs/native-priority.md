# Short-prompt priority on a Mac

This experimental mode runs an MLX-LM HTTP server directly on an Apple silicon
Mac. It is for an interactive client sharing a model with a client that sends
long documents. Start it explicitly when you want the two clients to share
one loaded model on one GPU stream.

## Start the server

From the root of the repository, install the pinned Mac environment with
`mlx/setup-mac.sh`. Then start the server with a local MLX language model.
The example uses the directory of the Qwen model used in the tests. Set the
model path to your own directory.

```sh
mlx/setup-mac.sh
mlx/.venv/bin/python mlx/priority_server.py \
  --model "$HOME/models/mlx-community/Qwen3.6-35B-A3B-4bit"
```

The server listens on loopback port 8080 by default. MLX-LM handles the
usual chat completions API, including streaming. A client can send a request
from another terminal with:

```sh
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of Italy?"}],"stream":true}'
```

Press Ctrl-C in the server terminal to stop it. This command uses MLX 0.32.4
and MLX-LM 0.32.2. MLX-LM describes its HTTP server as suitable for local use
and warns that it implements only basic security checks.

## How requests get a turn

After tokenization and prefix-cache lookup, a batchable request with at most
64 prompt tokens left to process joins the short queue. A long conversation
with a cached prefix can qualify. The other queue processes long prompts
in blocks of at most 512 tokens. The current block finishes before the next
queue gets a turn. Each turn advances one batch by one generator step.

While both queues have work, the scheduler allows at most four consecutive
short turns before one long turn. The worker also yields to generation after
at most eight consecutive admissions while a batch is active. These limits
let the long request advance while new short requests keep arriving. They
bound turns, and elapsed time varies with the model and batch size. Short
prompts that generate long replies keep their original queue. Requests
that MLX-LM serves sequentially, including those with an explicit seed,
follow its sequential path.

The scheduler runs inside the native Mac server process. The Docker and Kiri
bridges use their existing scheduling and wire protocols.

## Check the behavior on your Mac

With the Qwen model above running, this check disconnects a streaming client
during prefill, sends a follow-up request, then runs four short clients
continuously while a long prompt is processed. It checks the text each
client receives and that the long request completes.

```sh
python3 mlx/tests/priority_server_http.py
```

For a repeatable three-client burst, start the server with
`--prompt-cache-size 0` and run the same script with `--burst-round 1`.
It warms the server with one short request and sends three more requests
50, 100 and 150 ms after the client sees a prefill progress event at or above
2,048 tokens. It reports first visible text for each client and completion
time for the long request. Repeat with the same round number against the
standard server (`mlx/.venv/bin/python -m mlx_lm server`) using the same
model and cache setting.

## Release measurements

On 20 November 2026, four paired rounds of the burst test ran on one M4 Max
with 128 GB, macOS 26.6.2 and Qwen3.6-35B-A3B-4bit. Each server started
fresh for each round and prefix caching was disabled. The standard server
ran first in rounds 1 and 3, the priority server first in rounds 2 and 4.
Each short reply contained the requested word. The first three columns
are seconds from the HTTP client sending its request to receiving its first
visible text. The last is seconds for the long request to complete.

| Round and server | Short 1 | Short 2 | Short 3 | Long completion |
|---|---:|---:|---:|---:|
| 1, stock | 20.387 | 20.337 | 20.289 | 22.343 |
| 1, priority | 0.579 | 0.531 | 0.483 | 8.285 |
| 2, stock | 21.350 | 21.295 | 21.250 | 22.777 |
| 2, priority | 0.568 | 0.516 | 0.472 | 8.574 |
| 3, stock | 22.382 | 22.332 | 22.286 | 23.971 |
| 3, priority | 0.923 | 0.877 | 0.827 | 9.719 |
| 4, stock | 22.155 | 22.108 | 22.064 | 23.707 |
| 4, priority | 0.736 | 0.681 | 0.634 | 9.741 |

These measurements cover short greedy replies on one Mac over loopback.
The benchmark compares the complete scheduling policies, including polling,
batch grouping and chunk size. The separate sustained-load check uses
the default prefix cache.
