#!/bin/bash
# Per-operation device cost for both backends, on the same machine: the
# redesign is measured here, not assumed.
set -u
R=/mnt/q/Downloads/HybridEncode/repo
B="$R/build-work"
gcc -O2 -g -I"$R/include" -I"$R/src" -o /tmp/aegmicro \
    /mnt/q/Downloads/HybridEncode/micro_probe.c "$B/libaeg.a" -ldl -lpthread -lm
export LD_LIBRARY_PATH="$B:/usr/local/cuda-12.1/targets/x86_64-linux/lib"
for be in opencl cuda; do
  echo "=== $be, one lane ==="
  /tmp/aegmicro $be 2>/dev/null | grep -E 'tq_iq|\(cpu\)|satd_batch|agree'
  echo "=== $be, eight lanes ==="
  AEG_LANES=8 /tmp/aegmicro $be 2>/dev/null | grep -E 'tq_iq|satd_batch|agree'
done
