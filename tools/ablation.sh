#!/bin/bash
# Ablation at a fixed preset: how much does each search feature cost?  The
# result says where the remaining factor of nine against x264 lives.
set -u
R=/mnt/q/Downloads/HybridEncode/repo
B="$R/build-work"
export LD_LIBRARY_PATH="$B:/usr/local/cuda-12.1/targets/x86_64-linux/lib"
W=1280; H=720; N=30
SRC=/tmp/aegbench/src_1280x720_30.yuv
P=${1:-4}
Q=27
run() {
  local label="$1"; shift
  local log t
  log=$("$B/aegenc" -w "$W" -h "$H" -n "$N" -i "$SRC" -o /tmp/abl.264 \
        -p "$P" -q "$Q" "$@" 2>&1)
  local fps est
  fps=$(printf '%s\n' "$log" | grep -o '[0-9.]* fps' | head -1 | cut -d' ' -f1)
  est=$(printf '%s\n' "$log" | grep 'motion-est' | grep -o '[0-9.]*' | head -1)
  printf '  %-28s %7s fps   est %8s ms\n' "$label" "$fps" "$est"
}
echo "### preset $P, $P ${W}x${H} ${N} frames, ablated"
for extra in "" "--subme 0" "--qp-rd 0" "--trellis 0" "--bframes 0" \
             "--mbtree 0" "--partitions 1" "--ref 1" "--no-cabac" \
             "--subme 0 --qp-rd 0 --trellis 0"; do
  run "${extra:-(defaults)}" $extra
done
