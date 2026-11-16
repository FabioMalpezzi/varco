"""The heavy load, native against bridge, one pass at a time, turn and turn about.

Both programs stay loaded; a pass of one is followed at once by a pass of the
other, so the GPU's temperature and whatever else the Mac is doing weigh on
both alike. The bridge server must be running (./server.sh start; on Kiri's socket for ENGINE=kiri):
    [TORCH=2.15.0] [ENGINE=kiri] .venv-mac-2.15.0/bin/python tests/heavy_ab.py [passes]
"""
import os
import statistics
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
passes = int(sys.argv[1]) if len(sys.argv) > 1 else 12
env = {"DOCKER_HOST": f"unix://{os.path.expanduser('~')}/.docker/run/docker.sock", **os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_VERBOSITY": "error"}
engine = os.environ.get("ENGINE", "")
if engine == "kiri":
    env["DOCKER_HOST"] = f"unix://{os.path.expanduser('~')}/.kiri/run/docker.sock"
gpus = ["--gpus", "all"] if engine == "kiri" else []  # Kiri's own channel to the Mac
hub = os.environ.get("HUB", os.path.expanduser("~/.cache/huggingface/hub"))
pipes = dict(stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env, cwd=here)
torch_version = os.environ.get("TORCH", "2.15.0")
env["MODELS"] = hub
native = subprocess.Popen([f"{here}/../.venv-mac-{torch_version}/bin/python", "pass_loop.py"], **pipes)
bridge = subprocess.Popen(
    ["docker", "run", "--rm", "-i", *gpus, "-e", "BRIDGE_FREE_AT", "-e", "MODELS=/hub", "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_VERBOSITY=error",
     "-v", f"{hub}:/hub:ro", "-v", f"{here}:/work", f"varco-torch:{torch_version}", "python", "-m", "bridge", "pass_loop.py"], **pipes)
for p in (native, bridge):
    while (line := p.stdout.readline()).strip() != "ready":
        if not line:
            sys.exit("a program did not start")


def one(p, which):
    p.stdin.write(which + "\n")
    p.stdin.flush()
    ms, check = p.stdout.readline().split()
    return float(ms), check


only = os.environ.get("ONLY")  # "embed" or "rerank", to measure just one
for which, label in (("embed", "BGE-M3, 129 long texts"), ("rerank", "reranker, 128 long pairs")):
    if only and which != only:
        continue
    for p in (native, bridge):
        one(p, which)  # warm up
    n, b, same = [], [], True
    for i in range(passes):
        for p in ((native, bridge) if i % 2 == 0 else (bridge, native)):
            ms, check = one(p, which)
            (n if p is native else b).append(ms)
            checks = {**locals().get("checks", {}), p is native: check}
        same = same and checks[True] == checks[False]
    pairs = sorted(y / x for x, y in zip(n, b))
    print(f"{label:26} native {statistics.median(n):7.1f} ms  bridge {statistics.median(b):7.1f} ms  ratio {statistics.median(b) / statistics.median(n):.3f}"
          f"   pass by pass: median {statistics.median(pairs):.3f}, middle half {pairs[len(pairs) // 4]:.3f} to {pairs[-len(pairs) // 4 - 1]:.3f}"
          f"   results {'identical' if same else 'DIFFER'}")
for p in (native, bridge):
    p.stdin.close()
    p.wait()
