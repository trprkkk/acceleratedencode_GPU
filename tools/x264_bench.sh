#!/bin/sh
# Measure the MinGW x264 fork. Run from the repo root in Git Bash.
# Usage: tools/x264_bench.sh <csv> [extra x264 args...]
set -e
csv=$1
shift
root=$(pwd)
x264=$root/forks/x264/x264.exe
[ -x "$x264" ] || { echo "missing $x264"; exit 1; }
: > "$csv"
for clip in cockL vtestL megaL treeL; do
  y4m="$root/build/work/${clip}_1080_120.y4m"
  [ -f "$y4m" ] || { echo "missing $y4m"; exit 1; }
  for preset in medium slow; do
    for rep in 1 2 3; do
      out="$root/build/work/x264_bench.264"
      t0=$(date +%s%N)
      env -u X264_MV_HINT -u X264_MV_DUMP "$x264" \
        --input-res 1920x1080 --fps 30 --frames 120 \
        --preset "$preset" --crf 27 "$@" -o "$out" "$y4m" \
        >/dev/null 2>&1
      t1=$(date +%s%N)
      sec=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.4f", (b-a)/1e9}')
      size=$(stat -c%s "$out")
      echo "$clip,$preset,$rep,$sec,$size" >> "$csv"
      echo "$clip $preset rep$rep $sec s $size B"
    done
  done
done
