# Use case 7: custom Metal kernels

**What you get**

- With `--custom-kernels`, a library's own Metal kernels run from the
  container and give the Mac's result. Qwen3.6-35B-A3B-4bit with mlx-lm
  generated the Mac's text, 93 tokens, at 130 tokens per second in the
  first generation of a run and 141 in the second, with Docker Desktop and
  in Kiri (protocol 11), and 29 of 29 synthetic cases that run gave the
  same bytes.
- Without it, no container can run code of its own on the Mac's GPU, and
  libraries that test `mx.metal.is_available()` still work on their other
  path: the same model gave a correct text of 105 tokens at 118–119 tokens
  per second (Kiri, protocol 7).

**Compared with**

- The same program installed on the Mac: 129–135 tokens per second in the
  same runs, and 93 tokens. With its kernels turned off, the Mac gives the
  container's 105-token text word for word, at 108–110 tokens per second.

**When it is not the right tool:** on a server shared by containers that do
not trust each other. A kernel from one container can read and change the
arrays of another.

**Who it is for.** Anyone who runs in a container a library that writes its
own Metal kernels, and anyone who starts an MLX server and must decide
whether to turn the option on.

## What they are

A custom kernel is a GPU program that a library writes itself with
`mx.fast.metal_kernel`: Metal source code, which MLX compiles and the GPU
runs. Through the bridge the source travels from the container to the Mac,
and the Mac's MLX compiles and runs it there.

They are off unless the person who starts the MLX server chooses them. A
server started without `--custom-kernels` refuses every kernel, and tells
the container so when it connects.

## Who needs them, and how you notice they are off

mlx-lm computes some layers of some models with kernels of its own
(`gated_delta.py`; Qwen3.6-35B-A3B is one such model). It asks
`mx.metal.is_available()` when it is imported, uses its kernels if the
answer is `True`, and ordinary operations if it is `False`. mlx-vlm and oMLX
write kernels too; theirs include standard Metal headers, which the server
refuses with the option on as well ([limits](#limits)).

With the option off:

- `mx.metal.is_available()` is `False` in the container. It is `True` on a
  Mac, and in a container whose server takes kernels.
- A library that asks takes its path without kernels and says nothing. The
  result is correct and can differ from the Mac's. With Qwen3.6-35B the
  text was another one (105 tokens where the Mac gives 93) and generation
  ran at about 89% of the Mac's speed (118–119 tokens per second where the
  Mac gives 130–136).
- A program that makes a kernel without asking gets a `RuntimeError`:
  `[metal_kernel] Custom Metal kernels are off: the MLX server on the Mac
  was started without --custom-kernels.`

The different text is mlx-lm's other path, not an error of the bridge: the
Mac gives the same 105-token text, word for word, when its own kernels are
turned off.

A container asks the server once, the first time a library calls
`mx.metal.is_available()` or makes a kernel, and keeps the answer for the
life of the process. If the server cannot be reached at that moment the
answer is `False`, and the container says so once on its standard error.

## What turning them on gives

| | Without `--custom-kernels` (default) | With `--custom-kernels` |
|---|---|---|
| `mx.metal.is_available()` in the container | `False` | `True` |
| A library that tests it | takes its path without kernels | uses its kernels, as on the Mac |
| A program that makes a kernel regardless | gets a `RuntimeError` that says why | runs |
| Results | correct; can differ from the Mac's | the Mac's |
| Qwen3.6-35B-A3B-4bit with mlx-lm | 105 tokens, 118–119 tokens/s | 93 tokens, identical text, 130 tokens/s in the first generation of a run and 141 in the second |

Each kernel is sent once per connection. The server compiles it with MLX's
own `metal_kernel`, keeps it for every connection that sends the same one,
and lets go of it, with what MLX compiled, when the last connection that
sent it closes.

## What it implies

The server's source says what a kernel is to the process that runs it:

> Whatever is checked here, that code reads and writes the GPU's memory
> through pointers that nothing bounds: it can read or change arrays of this
> process that it was not given, those of other connections included, make
> the process fail, and keep the GPU or the compiler busy for as long as it
> likes.

So with the option on, every container connected to that server must be
trusted by every other. Containers that do not trust each other need a
server each, and that holds with the option off as well: without custom
kernels a GPU operation with an index out of range can still reach another
connection's arrays
([security](../security.md#containers-that-share-one-mlx-server)).

Every kernel the server takes is written to its log, with the connection
that sent it, its name and the size of its source.

## What the server still refuses

With `--custom-kernels` on, the server refuses a kernel, or a node that runs
one, in these cases
([`mlx/server/mlx_remote_server.cpp`](../../mlx/server/mlx_remote_server.cpp),
`add_kernel` and `call_kernel`):

| Refused | Limit |
|---|---|
| A source or header that could make the compiler read a file of the Mac | `#include`, `#import` or `#pragma` in any spelling the compiler takes (`%:include`, a directive split over lines, `#include_next`, `__has_include`), and `_Pragma`. The check is on the text: any of the three words anywhere after the first `#` is refused, in a comment too. `#define` and `#if` are taken. |
| A source that is too large | Source and header together over 256 KiB. A message over 320 KiB closes the connection. |
| Names that are not identifiers | The kernel's name, its arrays' names and its template arguments' names: letters, digits and `_`, not starting with a digit, at most 128 characters. |
| Too many arrays or arguments | More than 64 inputs, 64 outputs or 64 template arguments; a kernel with no output. |
| Unknown compile options | Anything but MLX's three math modes. |
| Too many kernels | More than 256 per connection. More than 4,096 different kernels, 64 MiB of them, or 16,384 compiled variants kept at a time for the containers connected. |
| A grid or a threadgroup of an impossible size | A size below 1, or a threadgroup of more than 1,048,576 threads. |
| A node that does not match its kernel | A kernel that was not sent on that connection, another number of arrays than the kernel takes, or a kernel in a graph that does not declare one. |
| A saved kernel | A `CustomKernel` primitive inside a graph. A kernel arrives only through its own message. |

The container end applies the limits on size, names and arrays itself
before it sends anything, and does not send the kernel of a function loaded
with `mx.import_function`.

[`mlx/tests/kernel_refusals.py`](../../mlx/tests/kernel_refusals.py) sends
each of these to a server in the protocol itself, and the stage `kernels` of
`mlx/tests/run-all.sh` runs it against a server with the option and one
without.

## Turning them on and off

The option belongs to the server, so the change is a restart of the server.
The programs that were connected lose their connection and have to be
started again; a program that goes on running keeps the answer it got when
it first asked.

For a server started by hand, `ctl.sh start` stops the server it started
before and starts a new one with the options given. `MODELS` is the
directory of your models in the commands of this page
(`export MODELS="$HOME/models"` for the one the scripts use when it is not
set):

```sh
mlx/server/ctl.sh start --map "/models=$MODELS" --custom-kernels   # on
mlx/server/ctl.sh start --map "/models=$MODELS"                    # off
```

With Kiri the server listens on the channel's socket instead of a port:

```sh
mlx/server/ctl.sh start --unix ~/.kiri/run/gpu.sock --map "/models=$MODELS" --custom-kernels
```

For a server kept running from login, `install` writes the agent again and
launchd restarts the server with the new options. It reads `MODELS` and
`MLX_PORT` again as well, so give the ones the agent was installed with:

```sh
KERNELS=1 scripts/autostart.sh install mlx     # on
scripts/autostart.sh install mlx               # off
```

With Kiri, put `ENGINE=kiri` in front of either line. With `KERNELS=1` the
script prints a note that says which containers reach that server.

The examples and tests that start a server of their own take the option
from the environment: `KERNELS=1 examples/02-openai-server/run.sh MODEL`,
and `SERVER_ARGS=--custom-kernels` for `mlx/tests/lm.sh`,
`mlx/tests/serve.sh` and the stage `upstream` of `mlx/tests/run-all.sh`.

## Seeing the state

`scripts/doctor.sh mlx` reads the option from the running server's command
line and prints it in the server's line, with a note under it when it is
on:

```
ok       a server is running on port 47800 (process 85531, started by a script (mlx/logs/server.pid)): custom Metal kernels ON; it reads weights under /models=/path/to/your/models
note     with custom kernels on, a container can read and change the arrays of the others on this server: turn it on only for containers that trust each other (docs/security.md)
```

With the option off the same line says `custom Metal kernels off`.
`scripts/autostart.sh status` adds `custom Metal kernels ON` to the line of
a server kept from login. A server started with the option says so in the
second line of its log (`mlx/logs/server.log`), which `ctl.sh start` prints:

```
compiles and runs on this Mac's GPU the Metal kernels the containers send (--custom-kernels)
```

From a container, ask what a library would ask:

```sh
docker run --rm varco-mlx:latest python -c "import mlx.core as mx; print(mx.metal.is_available())"
```

With Kiri the container needs `--gpus all`.

## Keeping them off for one container

A container started with `MLX_REMOTE_NO_KERNELS` set to any value does not
use kernels although its server takes them:

```sh
docker run --rm -e MLX_REMOTE_NO_KERNELS=1 -v "$MODELS:/models:ro" varco-mlx-lm:latest python your_program.py
```

In that container `mx.metal.is_available()` is `False` without the server
being asked, libraries take their path without kernels, and a kernel made
regardless raises `Custom Metal kernels are off: MLX_REMOTE_NO_KERNELS is
set.` The variable is read by the bridge inside the container. It is for a
container that should give the same result on a server with kernels and on
one without; it does not restrain a container that sets its own
environment.

## The example

[`examples/07-custom-kernels/kernel.py`](../../examples/07-custom-kernels/kernel.py)
does what a library does: it asks `mx.metal.is_available()`, computes with a
small kernel of its own if it can and with ordinary operations if it
cannot, and compares the two.

```sh
examples/07-custom-kernels/run.sh            # against a default server, then one with --custom-kernels
examples/07-custom-kernels/run.sh native     # on the Mac
ENGINE=kiri examples/07-custom-kernels/run.sh   # the same, with the containers in Kiri
```

```
=== the server as it starts by default
device Device(gpu, 0); mx.metal.is_available(): False
custom kernels are not available here: computed with ordinary operations
a kernel all the same: RuntimeError: [metal_kernel] Custom Metal kernels are off: the MLX server on the Mac was started without --custom-kernels.
sum 90.576538; largest difference from the ordinary operations 0.00e+00
=== the server started with --custom-kernels
compiles and runs on this Mac's GPU the Metal kernels the containers send (--custom-kernels)
device Device(gpu, 0); mx.metal.is_available(): True
computed by the custom kernel
sum 90.576538; largest difference from the ordinary operations 0.00e+00
```

The example starts servers of its own. If an MLX server is already
listening on port 47800, run it on another port:
`MLX_PORT=47810 examples/07-custom-kernels/run.sh`.

## Measured

- Of 38 synthetic cases (`mlx/tests/kernels.py`), the 29 that run give the
  same bytes as on the Mac, 5 raise the same kind of error on both sides,
  and 4 are refused on purpose (`#include`, `#pragma`, a source over
  256 KiB, the kernel of an imported function).
- Qwen3.6-35B-A3B-4bit with mlx-lm 0.32.2
  ([benchmarks](../benchmarks.md#speed-qwen36-35b-a3b-4bit-with-mlx-lm-0322),
  and [on the released code](../benchmarks.md#qwen36-35b-generation-on-the-released-code)):

  | Configuration | Generation, tokens/s | Tokens of the answer |
  |---|---|---|
  | Mac native, in the runs of protocol 11 | 129–135, and 138 in one later run | 93 |
  | Docker Desktop container, `--custom-kernels`, protocol 11 | 130 in the first generation of a run, 141 in the second | 93, identical text |
  | Kiri container, `--custom-kernels`, protocol 11 | 130 and 141–142 | 93, identical text |
  | Kiri container, without, protocol 7 | 118–119 | 105 |
  | Mac native with its custom kernels turned off | 108–110 | 105, the container's text word for word |

- The server registered one kernel per connection and compiled nothing
  again. After six containers of 120 kernels each had come and gone, the
  server held 73 MB.

## Limits

- `#include` is refused, even for the three standard Metal headers that
  mlx-vlm and oMLX use, so those libraries' kernels do not pass.
- Kernels of an imported function (`mx.import_function`) are not sent.
- Operations a library runs on Metal directly, not through
  `mx.fast.metal_kernel`, are not carried at all, which is why vLLM's Metal
  plugin does not work through the bridge
  ([compatibility](../compatibility.md#not-supported-and-open-measurements)).

Further measurements will be published as they are taken.
