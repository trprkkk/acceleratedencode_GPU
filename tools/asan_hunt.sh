#!/bin/bash
# Catch the intermittent api_check hang in the act and dump every thread's
# backtrace.
set -u
R=/mnt/q/Downloads/HybridEncode/repo
D="$R/build-asan"
export LD_LIBRARY_PATH="$D:/usr/local/cuda-12.1/targets/x86_64-linux/lib"
export ASAN_OPTIONS=detect_leaks=0
found=""
for i in 1 2 3 4 5 6; do
  "$D/api_check" > /tmp/hang_run.log 2>&1 &
  pid=$!
  waited=0
  while kill -0 $pid 2>/dev/null && [ $waited -lt 120 ]; do
    sleep 5
    waited=$((waited + 5))
  done
  if kill -0 $pid 2>/dev/null; then
    echo "run $i: STILL RUNNING after ${waited}s (pid $pid) -- dumping"
    gdb -p "$pid" -batch -ex 'thread apply all bt' 2>&1 \
        | head -80 > /tmp/hang_bt.txt
    cat /tmp/hang_bt.txt
    kill -9 $pid 2>/dev/null
    wait $pid 2>/dev/null
    found=1
    break
  fi
  wait $pid
  echo "run $i: rc=$? after ${waited}s"
done
[ -n "$found" ] || echo "no hang in six runs"
