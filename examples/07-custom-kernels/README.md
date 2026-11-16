# Example 7: custom Metal kernels

A program with its own Metal kernel (`mx.fast.metal_kernel`), run in a
container twice: with a bridge server started as it is by default, which
refuses the kernel, and with one started with `--custom-kernels`, which
compiles and runs it.

```sh
./run.sh            # both servers, one after the other
./run.sh native     # the same program on the Mac
```

`--custom-kernels` lets a container run its own code on the Mac's GPU, where
it can reach the arrays of other containers on the same server. What the
option gives, what it implies and the commands that turn it on and off:
[docs/use-cases/custom-kernels.md](../../docs/use-cases/custom-kernels.md).

| File | |
|---|---|
| `kernel.py` | The program: it asks `mx.metal.is_available()` and acts on the answer |
| `run.sh` | Runs it with each of the two servers |
