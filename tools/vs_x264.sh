#!/bin/bash
# Quality and speed against x264 and x265, on the same clips, with the same
# rate control.  This is the evidence the objective asks for ("x264/x265 と
# 同等以上"): the same source, the same setting, and what each encoder reached.
#
#   usage: vs_x264.sh [clip] [frames] [w] [h] [preset]
#
# PSNR is measured the same way for every encoder: decode the stream with
# ffmpeg, compare the decoded samples with the source through the psnr filter.
# Nothing an encoder claims about itself is used.
set -u
CLIP="${1:-/tmp/aegbench/src_1280x720_30.yuv}"
N="${2:-30}"
W="${3:-1280}"
H="${4:-720}"
PRESET="${5:-4}"
WORK=/tmp/aegvs
AEG=/mnt/q/Downloads/HybridEncode/repo/build-work/aegenc
export LD_LIBRARY_PATH="/mnt/q/Downloads/HybridEncode/repo/build-work:/usr/local/cuda-12.1/targets/x86_64-linux/lib"
rm -rf "$WORK"; mkdir -p "$WORK"
[ -s "$CLIP" ] || ffmpeg -v error -y -f lavfi -i "testsrc2=size=${W}x${H}:rate=30" \
    -frames:v "$N" -pix_fmt yuv420p -f rawvideo "$CLIP"
want=$((W * H * 3 / 2 * N))
got=$(stat -c %s "$CLIP")
if [ "$got" -ne "$want" ]; then
    echo "$CLIP is $got bytes, but ${W}x${H} $N frames of yuv420p is $want" >&2
    exit 2
fi

# psnr of a decoded stream against the clip, measured outside every encoder
psnr_of() {
  local mp4="$1"
  local dec="$WORK/dec.yuv"
  rm -f "$dec"
  ffmpeg -v error -y -vsync 0 -i "$mp4" -f rawvideo -pix_fmt yuv420p "$dec" 2>/dev/null
  [ -s "$dec" ] || { echo "n/a"; return; }
  ffmpeg -hide_banner -f rawvideo -s ${W}x${H} -pix_fmt yuv420p -i "$CLIP" \
      -f rawvideo -s ${W}x${H} -pix_fmt yuv420p -i "$dec" -lavfi psnr \
      -f null - 2>&1 | grep -o 'average:[0-9.]*' | head -1 | cut -d: -f2
}

kbps() {
  python3 -c "print('%.0f' % ($(stat -c %s "$1") * 8 / 1000.0 / $N * 30))"
}

timeit() {
  local t0 t1
  t0=$(date +%s.%N)
  "$@" >"$WORK/last.log" 2>&1
  t1=$(date +%s.%N)
  echo "$t0 $t1" | awk '{printf "%.2f", $2 - $1}'
}

row() {
  local tag="$1" f="$2" t="$3" q="$4"
  local sz=0 kb=0
  [ -s "$f" ] && sz=$(stat -c %s "$f") && kb=$(kbps "$f")
  printf '  %-16s %9d B %8s kbit/s  %7s dB  %6s s\n' "$tag" "$sz" "$kb" "$q" "$t"
}

echo "### $(basename "$CLIP")  ${W}x${H} ${N} frames, preset $PRESET"
echo "--- constant QP 26 (no rate control in the way) ---"
t=$(timeit "$AEG" -w "$W" -h "$H" -n "$N" -i "$CLIP" -o "$WORK/aeg_qp.264" -p "$PRESET" -q 26)
row "aegenc QP26" "$WORK/aeg_qp.264" "$t" "$(psnr_of "$WORK/aeg_qp.264")"
t=$(timeit x264 --qp 26 --preset medium --input-res ${W}x${H} --fps 30 \
    --keyint 250 --min-keyint 25 --bframes 3 --ref 4 -o "$WORK/x264_qp.264" "$CLIP")
row "x264 QP26" "$WORK/x264_qp.264" "$t" "$(psnr_of "$WORK/x264_qp.264")"
t=$(timeit x265 --qp 26 -p 4 --input-res ${W}x${H} --fps 30 --keyint 250 --min-keyint 25 \
    --bframes 3 --ref 4 -o "$WORK/x265_qp.hevc" "$CLIP")
row "x265 QP26" "$WORK/x265_qp.hevc" "$t" "$(psnr_of "$WORK/x265_qp.hevc")"
echo "--- CRF 24 (rate control on, the setting users compare) ---"
t=$(timeit "$AEG" -w "$W" -h "$H" -n "$N" -i "$CLIP" -o "$WORK/aeg_crf.264" -p "$PRESET" --crf 24)
row "aegenc CRF24" "$WORK/aeg_crf.264" "$t" "$(psnr_of "$WORK/aeg_crf.264")"
t=$(timeit x264 --crf 24 --preset medium --input-res ${W}x${H} --fps 30 \
    --keyint 250 --min-keyint 25 --bframes 3 --ref 4 -o "$WORK/x264_crf.264" "$CLIP")
row "x264 CRF24" "$WORK/x264_crf.264" "$t" "$(psnr_of "$WORK/x264_crf.264")"
t=$(timeit x265 --crf 24 -p 4 --input-res ${W}x${H} --fps 30 --keyint 250 --min-keyint 25 \
    --bframes 3 --ref 4 -o "$WORK/x265_crf.hevc" "$CLIP")
row "x265 CRF24" "$WORK/x265_crf.hevc" "$t" "$(psnr_of "$WORK/x265_crf.hevc")"
