#!/bin/bash
# Where does the encode time go, per preset?  (aegenc's own stage accounting,
# cross-checked with the wall clock.)
set -u
R=/mnt/q/Downloads/HybridEncode/repo
B="$R/build-work"
export LD_LIBRARY_PATH="$B:/usr/local/cuda-12.1/targets/x86_64-linux/lib"
W=1280; H=720; N=30
SRC=/tmp/aegbench/src_1280x720_30.yuv
echo "### ${W}x${H} ${N} frames, preset sweep (aegenc's own stage numbers)"
printf '%-8s %10s %8s %8s %8s %8s %8s\n' preset wall fps est transform deblock psnr
for p in 1 2 3 4 5 9; do
  t0=$(date +%s.%N)
  log=$("$B/aegenc" -w "$W" -h "$H" -n "$N" -i "$SRC" -o /tmp/pre_$p.264 \
        -p "$p" -q 27 2>&1)
  t1=$(date +%s.%N)
  wall=$(echo "$t1 - $t0" | bc)
  fps=$(echo "$log" | grep -o '[0-9.]* fps' | head -1 | cut -d' ' -f1)
  est=$(echo "$log" | grep 'motion-est' | grep -o '[0-9.]*' | head -1)
  tr=$(echo "$log" | grep 'transform' | grep -o '[0-9.]*' | head -1)
  de=$(echo "$log" | grep 'deblock' | grep -o '[0-9.]*' | head -1)
  ps=$(echo "$log" | grep 'psnr' | grep -o '[0-9.]*' | head -1)
  printf '%-8s %10s %8s %8s %8s %8s %8s\n' "$p" "$wall" "$fps" "$est" "$tr" "$de" "$ps"
done
