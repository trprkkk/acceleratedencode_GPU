#!/bin/bash
set -u
R=/mnt/q/Downloads/HybridEncode/repo
B="$R/build-work"
gcc -O2 -I"$R/include" -o /tmp/aegstats /mnt/q/Downloads/HybridEncode/stats_tool.c \
    "$B/libaeg.a" -ldl -lpthread -lm
export LD_LIBRARY_PATH="$B:/usr/local/cuda-12.1/targets/x86_64-linux/lib"
SRC=/tmp/aegbench/src_1280x720_30.yuv
for p in 2 3 4; do
  echo "=== preset $p (720p, 30f, QP 27) ==="
  /tmp/aegstats "$SRC" 1280 720 30 $p 27
done
