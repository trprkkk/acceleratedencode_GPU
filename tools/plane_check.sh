#!/bin/bash
# Per-plane PSNR/SSIM, aegenc against x264, on the same clip at a matched rate.
# A chroma plane that lags badly is a bug (the chroma transform, the offsets),
# not a rate-distortion gap, so this is the check that says which one it is.
set -u
R=/mnt/q/Downloads/HybridEncode/repo
B="$R/build-work"
export LD_LIBRARY_PATH="$B:/usr/local/cuda-12.1/targets/x86_64-linux/lib"
W=${W:-1280}; H=${H:-720}; N=${N:-60}
WORK=/tmp/aegplane
rm -rf "$WORK"; mkdir -p "$WORK"
SRC=${SRC:-$WORK/clip.yuv}
case "$SRC" in
  *mandelbrot*) FILTER="mandelbrot=size=${W}x${H}";;
  *smpte*)      FILTER="smptebars=size=${W}x${H}";;
  *rgb*)        FILTER="rgbtestsrc=size=${W}x${H}";;
  *)            FILTER="testsrc2=size=${W}x${H}:rate=30";;
esac
[ -s "$SRC" ] || ffmpeg -v error -y -f lavfi -i "$FILTER" -frames:v "$N" \
    -pix_fmt yuv420p -f rawvideo "$SRC"
dec() {  # stream -> y u v ssim
  local f="$1" o="$WORK/d.yuv"
  rm -f "$o"
  ffmpeg -v error -y -vsync 0 -i "$f" -f rawvideo -pix_fmt yuv420p "$o" 2>/dev/null
  ffmpeg -hide_banner -f rawvideo -s ${W}x${H} -pix_fmt yuv420p -i "$SRC" \
      -f rawvideo -s ${W}x${H} -pix_fmt yuv420p -i "$o" \
      -lavfi psnr -f null - 2>&1 \
      | grep -oE 'y:[0-9.]* u:[0-9.]* v:[0-9.]* average:[0-9.]*' | head -1
}
echo "### $(basename "$SRC") ${W}x${H} ${N} frames"
echo "  aegenc preset 4 at several QPs (its report, per plane):"
for q in 20 27 34; do
  "$B/aegenc" -w "$W" -h "$H" -n "$N" -i "$SRC" -o "$WORK/a.264" -p 4 -q "$q" \
      > "$WORK/a.log" 2>&1
  printf '    QP %-3s %s\n' "$q" \
      "$(grep -oE 'PSNR Y [0-9.]* dB.*' "$WORK/a.log" | head -1)"
done
echo "  and the same, measured outside every encoder (decode + psnr filter):"
for q in 26 32; do
  for enc in aeg x264; do
    if [ "$enc" = aeg ]; then
      "$B/aegenc" -w "$W" -h "$H" -n "$N" -i "$SRC" -o "$WORK/e.264" -p 4 -q "$q" \
          >/dev/null 2>&1
    else
      x264 --qp "$q" --preset medium --input-res ${W}x${H} --fps 30 \
           --keyint 250 --bframes 3 --ref 4 -o "$WORK/e.264" "$SRC" >/dev/null 2>&1
    fi
    printf '    QP %-3s %-6s %-52s %s\n' "$q" "$enc" "$(dec "$WORK/e.264")" \
        "$(stat -c %s "$WORK/e.264") B"
  done
done
