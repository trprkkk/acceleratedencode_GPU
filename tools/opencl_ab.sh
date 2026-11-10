#!/bin/bash
# A/B on the OpenCL data flow: the same forced device path, the old backend
# (baseline-repo, commit 5929971) against the redesigned one, same machine,
# same clip, same flags.  This is the number the redesign is worth here.
set -u
T=/mnt/q/Downloads/HybridEncode/repo
O=/mnt/q/Downloads/HybridEncode/baseline-repo/build-base
N="$T/build-work"
CLIP=/tmp/aegbench/src_1280x720_60.yuv
[ -s "$CLIP" ] || ffmpeg -v error -y -f lavfi -i "testsrc2=size=1280x720:rate=30" \
    -frames:v 60 -pix_fmt yuv420p -f rawvideo "$CLIP"
run() {
  local tag="$1" B="$2"; shift 2
  local t0 t1
  export LD_LIBRARY_PATH="$B:/usr/local/cuda-12.1/targets/x86_64-linux/lib"
  t0=$(date +%s.%N)
  "$B/aegenc" -w 1280 -h 720 -n 60 -i "$CLIP" -o "/tmp/ab_$tag.264" \
      -p 4 -q 27 "$@" > "/tmp/ab_$tag.log" 2>&1
  t1=$(date +%s.%N)
  printf '  %-22s %7.2f s   %s   %s B\n' "$tag" "$(echo "$t1 - $t0" | bc)" \
      "$(md5sum "/tmp/ab_$tag.264" | cut -c1-16)" "$(stat -c %s "/tmp/ab_$tag.264")"
}
echo "### 720p 60 frames, preprint 4, the coefficient stages forced to OpenCL"
echo "--- the old backend (blocking transfers) ---"
run old-ocl "$O" -s tq=opencl -s iq=opencl
echo "--- the redesigned one (non-blocking, one clFinish per operation) ---"
run new-ocl "$N" -s tq=opencl -s iq=opencl
echo "--- CPU reference (identical bytes, no device) ---"
run cpu "$O" -b cpu
echo
echo "same bytes in all three runs: $(md5sum /tmp/ab_*.264 | awk '{print $1}' | sort -u | wc -l) distinct stream(s)"
