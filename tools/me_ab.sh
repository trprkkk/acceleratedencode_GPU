#!/bin/bash
# Does the device actually make the motion search faster?  The same whole-frame
# search (AEG_SEED=1 AEG_PARFULL=1) on the device and on the CPU, with the
# stage times the encoder itself reports.
set -u
R=/mnt/q/Downloads/HybridEncode/repo
B="$R/build-work"
export LD_LIBRARY_PATH="$B:/usr/local/cuda-12.1/targets/x86_64-linux/lib"
W=1280; H=720; N=60
SRC=/tmp/aegbench/src_1280x720_60.yuv
[ -s "$SRC" ] || ffmpeg -v error -y -f lavfi -i "testsrc2=size=${W}x${H}:rate=30" \
    -frames:v "$N" -pix_fmt yuv420p -f rawvideo "$SRC"
run() {
  local tag="$1"; shift
  local log
  log=$("$B/aegenc" -w "$W" -h "$H" -n "$N" -i "$SRC" -o "/tmp/me_$tag.264" \
        -p 4 -q 27 -b cuda "$@" 2>&1)
  printf '  %-10s %-11s %s\n' "$tag" \
      "$(printf '%s\n' "$log" | grep -o '[0-9.]* fps' | head -1)" \
      "$(printf '%s\n' "$log" | grep 'motion-est' | tail -1)"
}
echo "### ${W}x${H} ${N} frames, the motion search's own time"
echo "--- per-macroblock search (the default) ---"
run default
echo "--- whole-frame search on the CPU (AEG_SEED=1 AEG_PARFULL=1) ---"
AEG_SEED=1 AEG_PARFULL=1 run searchcpu
echo "--- whole-frame search on the device ---"
run search -s est=cuda
echo "--- and the two give the same stream? ---"
md5sum /tmp/me_default.264 /tmp/me_search.264 | cut -c1-16,33-
