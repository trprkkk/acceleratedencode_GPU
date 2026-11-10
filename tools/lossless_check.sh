#!/bin/bash
# Does --lossless actually give the source back, byte for byte?
set -u
R=/mnt/q/Downloads/HybridEncode/repo
B="$R/build-work"
export LD_LIBRARY_PATH="$B:/usr/local/cuda-12.1/targets/x86_64-linux/lib"
T=/tmp/aeglossless
rm -rf "$T"; mkdir -p "$T"
ffmpeg -v error -y -f lavfi -i "testsrc2=size=320x192:rate=30" -frames:v 12 \
    -pix_fmt yuv420p -f rawvideo "$T/c.yuv" 2>/dev/null
echo "=== the stream ==="
"$B/aegenc" -w 320 -h 192 -n 12 -i "$T/c.yuv" -o "$T/l.264" --lossless \
    -v 2 > "$T/l.log" 2>&1
echo "rc=$?"
grep -iE 'lossless|cavac|cab|error' "$T/l.log" | head -3
tail -3 "$T/l.log"
echo "=== the encoder's own reconstruction ==="
"$B/aegenc" -w 320 -h 192 -n 12 -i "$T/c.yuv" -o /dev/null --lossless \
    --dump-recon "$T/l.recon" > /dev/null 2>&1
cmp "$T/c.yuv" "$T/l.recon" && echo "  recon == source (bit-exact)" \
  || echo "  recon differs from the source"
echo "=== and an independent decoder's output ==="
ffmpeg -v error -y -i "$T/l.264" -f rawvideo -pix_fmt yuv420p "$T/l.dec" \
    > "$T/l.decerr" 2>&1
echo "decoder rc=$?, stderr: $(wc -c < "$T/l.decerr") bytes"
cmp "$T/c.yuv" "$T/l.dec" && echo "  decode == source (bit-exact)" \
  || echo "  decode differs from the source"
echo "=== sizes and the rate ==="
printf '  source %s B, stream %s B (%.2f bits/sample)\n' \
    "$(stat -c %s "$T/c.yuv")" "$(stat -c %s "$T/l.264")" \
    "$(python3 -c "print($(stat -c %s "$T/l.264")*8/$(stat -c %s "$T/c.yuv"))")"
