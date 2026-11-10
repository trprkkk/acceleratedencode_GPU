#!/bin/bash
# 10-bit lossless: the pcm samples are BitDepthY bits wide, so a 10-bit encoder
# must write ten bits a sample and put them back exactly.
set -u
R=/mnt/q/Downloads/HybridEncode/repo
B="$R/build-work"
export LD_LIBRARY_PATH="$B:/usr/local/cuda-12.1/targets/x86_64-linux/lib"
T=/tmp/aegll10
rm -rf "$T"; mkdir -p "$T"
ffmpeg -v error -y -f lavfi -i "testsrc2=size=320x192:rate=30" -frames:v 8 \
    -pix_fmt yuv420p10le -f rawvideo "$T/c10.yuv" 2>/dev/null
echo "  source $(stat -c %s "$T/c10.yuv") B"
"$B/aegenc" -w 320 -h 192 -n 8 --input-depth 10 --bit-depth 10 -i "$T/c10.yuv" -o "$T/l10.264" \
    --lossless --dump-recon "$T/l10.recon" >/dev/null 2>&1
echo "  encode rc=$?  stream $(stat -c %s "$T/l10.264") B"
cmp "$T/c10.yuv" "$T/l10.recon" && echo "  recon == source (bit-exact, 10-bit)" \
  || echo "  recon differs from the source"
ffmpeg -v error -y -i "$T/l10.264" -f rawvideo -pix_fmt yuv420p10le "$T/l10.dec" \
    > "$T/l10.decerr" 2>&1
echo "  decoder rc=$?, stderr $(wc -c < "$T/l10.decerr") B"
cmp "$T/c10.yuv" "$T/l10.dec" && echo "  decode == source (bit-exact, 10-bit)" \
  || echo "  decode differs from the source"
