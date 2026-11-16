# Example 3: PyTorch inference

Sentence embeddings and text generation with `transformers`, in a container,
computed on the Mac's GPU through the PyTorch bridge.

```sh
./run.sh            # in a container
./run.sh native     # the same program on the Mac, to compare
./run.sh cpu        # the same container with the bridge off (BRIDGE=0)
```

Needs `sentence-transformers/all-MiniLM-L6-v2` and `openai-community/gpt2`
in the Hugging Face cache (`scripts/fetch-models.sh`). What it shows, the
measured figures and the limits:
[docs/use-cases/pytorch-inference.md](../../docs/use-cases/pytorch-inference.md).

| File | |
|---|---|
| `infer.py` | The program: it takes `cuda`, `mps` or `cpu`, whichever PyTorch has |
| `compose.yaml` | The service: the image `varco-torch`, the Hugging Face cache read-only |
| `compose.gpu.yaml` | The GPU request, which `run.sh` adds with `ENGINE=kiri` |
| `run.sh` | Starts the bridge's server, runs the service, stops the server |
