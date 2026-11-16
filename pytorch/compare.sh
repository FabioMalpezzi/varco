#!/bin/sh
# Native against bridge, in mirrored order (native, bridge, bridge, native, ...)
# so that whatever else the Mac is doing, and the GPU warming up, weighs on
# both alike:   [ENGINE=kiri] ./compare.sh [blocks]     (a block is four runs)
cd "$(dirname "$0")" || exit 1
. ./versions.sh
log=$(mktemp)
for block in $(seq "${1:-2}"); do
    for which in native bridge bridge native; do
        echo "== $which" >>"$log"
        ./bench.sh $which >>"$log" 2>&1
    done
done
"$VENV/bin/python" -I - "$log" <<'PY'
import re, statistics, sys
runs = {"native": [], "bridge": []}  # each run: {case: ms}
for line in open(sys.argv[1]):
    if line.startswith("== "):
        runs[line.split()[1]].append({})
        continue
    m = re.match(r"(native|bridge)( heavy)?\s+(.*?)\s+([\d.]+) ms", line)
    if m:
        runs[m[1]][-1][((m[2] or "") + " " + m[3].replace(" in", "")).strip()] = float(m[4])
    elif "Error" in line:
        print("!", line.strip()[:120])
cases = list(runs["native"][0])
print(f"{len(runs['native'])} native runs, {len(runs['bridge'])} bridge runs; times are medians in ms")
print(f"{'':28}{'native':>9}{'bridge':>9}{'ratio':>7}   ratio run by run (each bridge run compared with the native run next to it)")
for case in cases:
    n = [r[case] for r in runs["native"] if case in r]
    b = [r[case] for r in runs["bridge"] if case in r]
    pairs = sorted(y / x for x, y in zip(n, b))
    print(f"{case:28}{statistics.median(n):9.1f}{statistics.median(b):9.1f}{statistics.median(b) / statistics.median(n):7.2f}"
          f"   median {statistics.median(pairs):.2f}, from {pairs[0]:.2f} to {pairs[-1]:.2f}")
PY
rm -f "$log"
