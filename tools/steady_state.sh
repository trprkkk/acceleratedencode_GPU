#!/bin/bash
# Steady state: a 300-frame clip, so the device's ~1.5 s of startup is a small
# part of the run.  This is the number that says how fast the encoder is.
set -u
R=/mnt/q/Downloads/HybridEncode/repo
B="$R/build-work"
export LD_LIBRARY_PATH="$B:/usr/local/cuda-12.1/targets/x86_64-linux/lib"
W=1280; H=720; N=300
SRC=/tmp/aegbench/src_1280x720_300.yuv
[ -s "$SRC" ] || ffmpeg -v error -y -f lavfi -i "testsrc2=size=${W}x${H}:rate=30" \
    -frames:v "$N" -pix_fmt yuv420p -f rawvideo "$SRC"
echo "### ${W}x${H} ${N} frames (10 s of 30 fps content), aegenc preset $1"
P=${1:-4}
for be in cpu cuda auto-cuda opencl; do
  b="$be"; [ "$be" = auto-cuda ] && b=cuda
  t0=$(date +%s.%N)
  log=$("$B/aegenc" -w "$W" -h "$H" -n "$N" -i "$SRC" -o "/tmp/ss_$be.264" \
        -p "$P" -q 27 -b "$b" 2>&1)
  t1=$(date +%s.%N)
  printf '  %-12s %7.2f s wall   %s   %s\n' "$be" "$(echo "$t1 - $t0" | bc)" \
      "$(printf '%s\n' "$log" | grep -o '[0-9.]* fps' | head -1)" \
      "$(md5sum "/tmp/ss_$be.264" | cut -c1-16)"
done
echo "  same bytes in all: $(md5sum /tmp/ss_*.264 | awk '{print $1}' | sort -u | wc -l) distinct"
echo
echo "### and the stage breakdown at 300 frames (cuda)"
"$B/aegenc" -w "$W" -h "$H" -n "$N" -i "$SRC" -o /tmp/ss_stage.264 -p "$P" \
    -q 27 -b cuda 2>&1 | grep -E 'stage|encoded'
