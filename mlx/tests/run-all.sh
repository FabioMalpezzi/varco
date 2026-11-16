#!/bin/sh
# Run every stage natively on the Mac and in a container of the final image
# (varco-mlx-embeddings), then compare. Starts and stops the Mac server.
#   tests/run-all.sh [smoke|coverage|edge|bge|gen|load|kernels|urgent|graphs|disconnects|upstream ...]   (default: all but upstream)
# MODELS is the directory that holds mlx-community/bge-m3-mlx-fp16 (default
# $HOME/models), PY the Mac's python (default .venv/bin/python, see
# setup-mac.sh), IMAGE the container image.
# PORT is where the server listens (default 47800), SERVER_BIN the copy of it
# to run (see server/ctl.sh), BOXENV more options for the container's side.
# SERVER_ARGS is more options for the server the stages share, as in
# tests/lm.sh:
#   SERVER_ARGS=--custom-kernels tests/run-all.sh upstream
# runs MLX's own tests against a server that takes custom Metal kernels,
# where tests/compare.py holds the container to what is declared for such a
# server. The stages edge and kernels count on a server started without
# --custom-kernels, and say so instead of running.
# The stage "kernels" starts a second server, with --custom-kernels, on KPORT
# (default PORT + 1); "graphs" and "disconnects" each start one of their own
# there too, since they count on being its only connection.
# ENGINE=kiri runs the container's side in Kiri: the server listens on the
# Unix socket Kiri's channel leads to (WHERE, default ~/.kiri/run/gpu.sock)
# and on no port, and the containers are started with --gpus and find it by
# themselves, at the address Kiri gives them (KIRI_GPU_BRIDGE, which the
# container end reads when MLX_REMOTE is not set). The stages that speak to
# the server from the Mac, or need a second server next to the first, want a
# TCP port and are not for Kiri, where one socket has one server: smoke,
# coverage, edge, bge, bge-cpu, gen and upstream are (the default there: all
# of those but bge-cpu and upstream).
# The exit status is 0 only when every stage passed. A stage that fails does
# not stop the others, so that one run tells everything that is wrong: what
# failed is said where it happens (FAILED) and listed again at the end. A
# stage fails when one of its programs ends with an error on either side,
# when a result is missing, or when tests/compare.py finds the two sides
# different from each other or from the stage's inventory (tests/expected/).
cd "$(dirname "$0")/.."
ROOT=$PWD
ENGINE=${ENGINE:-docker}
case $ENGINE in
  kiri)
    export DOCKER_HOST=unix://$HOME/.kiri/run/docker.sock
    SOCKET=${WHERE:-$HOME/.kiri/run/gpu.sock}
    ;;
  docker) export DOCKER_HOST=${DOCKER_HOST:-unix://$HOME/.docker/run/docker.sock} ;;
  *) echo "ENGINE is docker or kiri, not $ENGINE" >&2; exit 2 ;;
esac
MODELS=${MODELS:-$HOME/models}
MODEL=$MODELS/mlx-community/bge-m3-mlx-fp16
PY=${PY:-$ROOT/.venv/bin/python}
IMAGE=${IMAGE:-varco-mlx-embeddings:latest}
PORT=${PORT:-47800}
KPORT=${KPORT:-$((PORT + 1))}
# Where the containers are told the server is: its port, on Docker Desktop.
# In Kiri nothing is told: a container started with --gpus finds its bridge.
WHERE_SERVER="-e MLX_REMOTE=host.docker.internal:$PORT"
GPUS=
if [ "$ENGINE" = kiri ]; then
  WHERE_SERVER=
  GPUS="--gpus all"
fi
# The files the stage "load" makes: the server may read those of allowed/.
FILES=$ROOT/results/loadtest
mkdir -p results "$FILES/allowed"
# With BOX=name the container side runs in that running container instead
# (the development one, see tests/bench.sh).
box() {
  if [ -n "$BOX" ]; then
    docker exec $WHERE_SERVER $BOXENV "$BOX" "$@"
    return
  fi
  docker run --rm $GPUS $WHERE_SERVER $BOXENV -v "$ROOT/tests:/tests:ro" -v "$ROOT/upstream/python/tests:/upstream:ro" \
    -v "$MODELS:/models:ro" -v "$ROOT/results:/out" "$IMAGE" "$@"
}
if [ "$ENGINE" = kiri ]; then
  stages=${*:-smoke coverage edge bge gen}
else
  stages=${*:-smoke coverage edge bge gen load kernels urgent graphs disconnects}
fi
status=0
bad=
# What went wrong, remembered: the stages go on, and the end tells.
failed() {
  echo "FAILED ($stage): $*"
  status=1
  case " $bad " in *" $stage "*) ;; *) bad="$bad $stage" ;; esac
}
# A command of a stage: when it fails, the stage has.
try() {
  "$@" || failed "status $? from: $(echo "$*" | cut -c1-200)"
}
# The same, with what the command printed kept in $out for the stage to cut.
capture() {
  out=$("$@") || failed "status $? from: $(echo "$*" | cut -c1-200)"
}
# The results of a stage, removed before it runs: a side that fails before
# it writes its own must not be compared through the file of an earlier run.
fresh() {
  for name in "$@"; do rm -f "results/$name"; done
}
# The containers see the models at /models and the test files at /out/loadtest.
if [ "$ENGINE" = kiri ]; then
  ./server/ctl.sh start --unix "$SOCKET" --map /models="$MODELS" \
    --map /out/loadtest/allowed="$FILES/allowed" $SERVER_ARGS || exit 1
else
  ./server/ctl.sh start --port $PORT --map /models="$MODELS" \
    --map /out/loadtest/allowed="$FILES/allowed" $SERVER_ARGS || exit 1
fi
for stage in $stages; do
  echo "=== $stage"
  case "$ENGINE,$stage" in
    kiri,load | kiri,kernels | kiri,urgent | kiri,graphs | kiri,disconnects)
      failed "not a stage for ENGINE=kiri: it speaks to the server on a TCP port, or starts a second server, and in Kiri the one server listens on the socket alone"
      continue
      ;;
  esac
  case "$stage, $SERVER_ARGS " in
    edge,*" --custom-kernels "* | kernels,*" --custom-kernels "*)
      failed "not a stage for a server started with --custom-kernels (SERVER_ARGS): it checks what a server without them refuses"
      continue
      ;;
  esac
  case $stage in
    smoke)
      # One expression on both sides: the same value, on the gpu device.
      expression="import mlx.core as mx; a = mx.arange(6.0).reshape(2,3); print((a @ a.T + 1).sum().item(), mx.default_device())"
      capture "$PY" -c "$expression"; native=$out
      capture box python -c "$expression"; container=$out
      echo "native:    $native"
      echo "container: $container"
      if [ -z "$native" ] || [ "$native" != "$container" ]; then
        failed "the two sides do not print the same"
      fi
      echo "--- where the container end looks for the server, and what it says when none answers"
      try box python /tests/find_server.py
      ;;
    coverage)
      fresh coverage-native.json coverage-container.json
      try "$PY" tests/coverage.py results/coverage-native.json
      try box python /tests/coverage.py /out/coverage-container.json
      try "$PY" tests/compare.py coverage results/coverage-native.json results/coverage-container.json
      ;;
    edge)
      fresh edge-native.json edge-container.json
      echo "--- native"; try "$PY" tests/edge.py results/edge-native.json
      echo "--- container"; try box python /tests/edge.py /out/edge-container.json
      echo "--- compared"
      try "$PY" tests/compare.py edge results/edge-native.json results/edge-container.json
      ;;
    bge)
      fresh bge-native.json bge-container.json
      capture "$PY" tests/bge_m3.py results/bge-native.json --model "$MODEL"
      capture box python /tests/bge_m3.py /out/bge-container.json
      try "$PY" tests/compare_bge.py results/bge-native.json results/bge-container.json
      for n in 1 8 32; do
        capture "$PY" tests/bge_phases.py --model "$MODEL" $n; echo "$out" | tail -1
        capture box python /tests/bge_phases.py $n; echo "$out" | tail -1
      done
      ;;
    bge-cpu)
      # The container's CPU against the Mac's GPU: not the same numbers, and
      # not meant to be. Reported, and only the programs' failures count.
      fresh bge-container-cpu.json
      capture box python /tests/bge_m3.py /out/bge-container-cpu.json --cpu --runs 3
      try "$PY" tests/compare_bge.py --report-only results/bge-native.json results/bge-container-cpu.json
      ;;
    gen)
      fresh gen-native.json gen-container.json
      capture "$PY" tests/genloop.py results/gen-native.json; echo "$out" | cut -c1-400
      capture box python /tests/genloop.py /out/gen-container.json; echo "$out" | cut -c1-400
      try "$PY" tests/compare.py gen results/gen-native.json results/gen-container.json
      ;;
    load)
      fresh load-native.json load-container.json load-container-upload.json load-container.err
      try "$PY" tests/load_files.py make "$FILES"
      try "$PY" tests/load_files.py run "$FILES" results/load-native.json
      echo "--- container, the Mac reads the allowed files"
      box python /tests/load_files.py run /out/loadtest /out/load-container.json 2> results/load-container.err \
        || failed "status $? from load_files.py in the container: see results/load-container.err"
      try "$PY" tests/compare.py load results/load-native.json results/load-container.json
      said=$(grep -c 'does not read' results/load-container.err)
      echo "said in the container, $said times (the compressed archive, the directory outside, the file saved in /tmp):"
      cut -c1-300 results/load-container.err
      # Those three and no other: any more would be a file the Mac was to
      # read itself and did not.
      [ "$said" = 3 ] || failed "the container read $said kinds of file itself, not 3"
      grep closed logs/server.log | tail -1 | grep -o "uploads=.*loads=[^)]*)"
      echo "--- container, everything read there and uploaded (MLX_REMOTE_NO_LOAD=1)"
      (BOXENV="$BOXENV -e MLX_REMOTE_NO_LOAD=1"; box python /tests/load_files.py run /out/loadtest /out/load-container-upload.json) \
        || failed "status $? from load_files.py in the container, with MLX_REMOTE_NO_LOAD=1"
      try "$PY" tests/compare.py load results/load-native.json results/load-container-upload.json
      grep closed logs/server.log | tail -1 | grep -o "uploads=.*loads=[^)]*)"
      echo "--- what the server refuses to read"
      try "$PY" tests/load_refusals.py $PORT "$FILES" /out/loadtest
      ;;
    kernels)
      fresh kernels-native.json kernels-container.json
      echo "--- the server above, started without --custom-kernels"
      try box python /tests/kernels.py off
      try "$PY" tests/kernel_refusals.py $PORT off
      echo "--- no server at all when a library asks"
      out=$(BOXENV="$BOXENV -e MLX_REMOTE=127.0.0.1:9"; box python -c "import mlx.core as mx; print('mx.metal.is_available():', mx.metal.is_available())" 2>&1) \
        || failed "status $? from asking mx.metal.is_available() with no server"
      echo "$out" | cut -c1-300
      case $out in
        *"mx.metal.is_available(): False"*) ;;
        *) failed "mx.metal.is_available() is not False with no server" ;;
      esac
      echo "--- a second server, with --custom-kernels"
      if ! SERVER_NAME=kernels ./server/ctl.sh start --port $KPORT --custom-kernels; then
        failed "the second server did not start on port $KPORT"
        continue
      fi
      try "$PY" tests/kernels.py results/kernels-native.json
      (BOXENV="$BOXENV -e MLX_REMOTE=host.docker.internal:$KPORT"; box python /tests/kernels.py /out/kernels-container.json) \
        || failed "status $? from kernels.py in the container"
      try "$PY" tests/compare.py kernels results/kernels-native.json results/kernels-container.json
      echo "the container sent $(grep -c "kernel #.*new to this server" logs/server-kernels.log) kernels, of which mlx compiled $(grep -c "compiles kernel" logs/server-kernels.log) variants"
      echo "--- what that server takes and what it refuses"
      try "$PY" tests/kernel_refusals.py $KPORT on
      SERVER_NAME=kernels ./server/ctl.sh stop > /dev/null
      ;;
    urgent)
      fresh urgent-native.json urgent-container.json
      try "$PY" tests/urgent.py results/urgent-native.json
      try box python /tests/urgent.py /out/urgent-container.json
      try "$PY" tests/compare.py urgent results/urgent-native.json results/urgent-container.json
      grep closed logs/server.log | tail -1 | grep -o "graphs=[^)]*)"
      echo "--- what the server answers early and what it leaves to take its turn"
      try "$PY" tests/urgent_refusals.py $PORT
      ;;
    graphs)
      # What the server refuses of a container's graphs and greetings, and what
      # it tells it, spoken straight in the protocol to a server of the stage's own.
      if ! SERVER_NAME=graphs ./server/ctl.sh start --port $KPORT > /dev/null; then
        failed "the stage's own server did not start"
        continue
      fi
      try "$PY" tests/graph_refusals.py $KPORT
      SERVER_NAME=graphs ./server/ctl.sh stop > /dev/null
      ;;
    disconnects)
      # Containers that leave in the middle of their work: the others are
      # served, and the Mac has its memory back.
      try "$PY" tests/load_files.py make "$FILES"
      if ! SERVER_NAME=disconnects ./server/ctl.sh start --port $KPORT --map /out/loadtest/allowed="$FILES/allowed" > /dev/null; then
        failed "the stage's own server did not start"
        continue
      fi
      try "$PY" tests/disconnects.py $KPORT "$FILES" /out/loadtest
      SERVER_NAME=disconnects ./server/ctl.sh stop > /dev/null
      ;;
    upstream)
      # What MLX's tests print is long: kept in a file by side, with the
      # status of the two runs looked at here.
      fresh upstream-native.json upstream-container.json
      "$PY" tests/upstream.py "$ROOT/upstream/python/tests" results/upstream-native.json > results/upstream-native.log 2>&1 \
        || failed "status $? from MLX's tests on the Mac: see results/upstream-native.log"
      box python /tests/upstream.py /upstream /out/upstream-container.json > results/upstream-container.log 2>&1 \
        || failed "status $? from MLX's tests in the container: see results/upstream-container.log"
      try "$PY" tests/compare.py upstream results/upstream-native.json results/upstream-container.json
      ;;
    *)
      failed "there is no such stage"
      ;;
  esac
done
echo "=== server counters"
grep closed logs/server.log | cut -c1-240
./server/ctl.sh stop
if [ $status = 0 ]; then
  echo "=== passed: $stages"
else
  echo "=== FAILED:$bad (of: $stages)"
fi
exit $status
