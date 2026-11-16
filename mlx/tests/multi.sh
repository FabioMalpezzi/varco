#!/bin/sh
# Several programs generating at the same time on the Mac's GPU: N containers through the MLX
# bridge, or N processes on the Mac directly, each with its own copy of the model.
#   tests/multi.sh kiri|docker|mac N [GENERATIONS] [MODEL under MODELS]
# MODELS is the directory of the models (default $HOME/models), IMAGE the container image
# (default varco-mlx-lm:latest), LMPY the Mac's python (default .venv/bin/python).
# kiri and docker expect the engine up, the image in it, and the bridge's server listening
# (for kiri on ~/.kiri/run/gpu.sock, which a container started with --gpus finds by itself;
# for docker on the port PORT, default 47800), started with --map /models=$MODELS. The copies load, then all start at the same moment. Prints
# each copy's summary, then the rates added up (tokens generated over the time the generations
# took, for each copy). The logs go to OUT (default /tmp/multi-gen).
# The exit status is 0 only when all N copies ended without an error and summed their run up,
# and every generation of every copy gave the same text.
here=$(cd "$(dirname "$0")" && pwd)
where=$1 n=$2 generations=${3:-8} model=${4:-mlx-community/Qwen3.6-35B-A3B-4bit}
models=${MODELS:-$HOME/models}
image=${IMAGE:-varco-mlx-lm:latest}
out=${OUT:-/tmp/multi-gen}
case "$where:$n" in
    mac:[1-9]*|kiri:[1-9]*|docker:[1-9]*) ;;
    *) echo "usage: multi.sh kiri|docker|mac N [GENERATIONS] [MODEL]"; exit 2 ;;
esac
mkdir -p "$out"
rm -f "$out"/"$where"-"$n"-*.log
# Late enough for every copy to have loaded: the Mac reads N copies of the weights.
start=$(( $(date +%s) + ${WAIT:-20} + 5 * n ))
pids=
for i in $(seq "$n"); do
    case $where in
        mac)
            "${LMPY:-$here/../.venv/bin/python}" "$here/multi_gen.py" "$models/$model" "$start" "$generations" "mac$i" \
                > "$out/$where-$n-$i.log" 2>&1 & ;;
        kiri)
            DOCKER_HOST=unix://$HOME/.kiri/run/docker.sock docker run --rm --name "multi-gen-$i" --gpus all \
                -v "$here:/tests:ro" -v "$models:/models:ro" "$image" \
                python /tests/multi_gen.py "/models/$model" "$start" "$generations" "kiri$i" > "$out/$where-$n-$i.log" 2>&1 & ;;
        docker)
            DOCKER_HOST=${DOCKER_HOST:-unix://$HOME/.docker/run/docker.sock} docker run --rm --name "multi-gen-$i" \
                -e MLX_REMOTE=host.docker.internal:${PORT:-47800} -v "$here:/tests:ro" -v "$models:/models:ro" "$image" \
                python /tests/multi_gen.py "/models/$model" "$start" "$generations" "docker$i" > "$out/$where-$n-$i.log" 2>&1 & ;;
    esac
    pids="$pids $!"
done
# Each copy by itself: a plain wait would not say how any of them ended.
status=0
i=0
for pid in $pids; do
    i=$((i + 1))
    wait "$pid" || { echo "FAILED: copy $i ended with status $?: see $out/$where-$n-$i.log"; status=1; }
done
grep -h -E "^SUMMARY|began late|Error|error" "$out"/"$where"-"$n"-*.log | cut -c1-260
# Added up: each copy's tokens over the time its generations took.
grep -h "^SUMMARY" "$out"/"$where"-"$n"-*.log | sed -E 's/.* s, ([0-9.]+) per second;.*texts (.*)/\1 \2/' | LC_ALL=C awk -v w="$where" -v n="$n" \
    '{ total += $1; texts[$2] = 1 } END { k = 0; for (t in texts) k++; printf "TOTAL %s x%d: %.1f tokens per second added up; %d distinct text(s)\n", w, n, total, k }'
summaries=$(grep -h -c "^SUMMARY" "$out"/"$where"-"$n"-*.log | LC_ALL=C awk '{ s += $1 } END { print s + 0 }')
[ "$summaries" = "$n" ] || { echo "FAILED: $summaries of $n copies summed their run up"; status=1; }
# A copy's summary lists the digests of its texts: one digest, the same in every copy.
texts=$(grep -h "^SUMMARY" "$out"/"$where"-"$n"-*.log | sed -E 's/.*texts //' | sort -u | wc -l | tr -d ' ')
[ "$texts" = 1 ] && ! grep -h "^SUMMARY" "$out"/"$where"-"$n"-*.log | grep -q "texts \[.*,.*\]" \
    || { echo "FAILED: the generations did not all give the same text"; status=1; }
exit $status
