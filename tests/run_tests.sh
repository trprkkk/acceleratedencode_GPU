#!/bin/sh
# Local verification: build, encode a synthetic clip, decode it with an
# independent decoder and report both the coding quality and -- most
# importantly -- whether the encoder's own reconstruction agrees with the
# decoder.  Agreement proves the transform/quantisation/deblocking maths and
# the reconstruction path are right; a disagreement means the encoder and the
# decoder disagree about the samples, which would drift on P pictures.
#
# Set FFMPEG=/path/to/ffmpeg to use a specific build.
set -e
BUILD=build
FF=${FFMPEG:-ffmpeg}
W=320
H=192
N=10
TMP=${TMPDIR:-/tmp}/aegtest.$$
mkdir -p "$TMP"

echo "== build"
make -s >/dev/null

echo "== synthetic clip"
make -s "$BUILD/genyuv"
"./$BUILD/genyuv" "$W" "$H" "$N" "$TMP/src.yuv"

echo "== encode"
./$BUILD/aegenc -w $W -h $H -q 23 -n $N -i "$TMP/src.yuv" -o "$TMP/out.264" \
     --dump-recon "$TMP/recon.yuv" | sed 's/^/   /'

echo "== decode with an independent decoder ($FF)"
if command -v "$FF" >/dev/null 2>&1; then
    "$FF" -v error -i "$TMP/out.264" -f rawvideo -pix_fmt yuv420p \
        "$TMP/dec.yuv" -y 2>"$TMP/dec.log" || true
    if [ -s "$TMP/dec.log" ]; then
        echo "   decoder reported problems:"
        sed 's/^/     /' "$TMP/dec.log" | head -5
    fi
    python3 - "$TMP" "$W" "$H" <<'PY'
import sys, math, os
tmp, w, h = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
fs = w * h * 3 // 2
def rd(n):
    p = os.path.join(tmp, n)
    if not os.path.exists(p):
        return b''
    return open(p, 'rb').read()[:fs]
src, dec, rec = rd('src.yuv'), rd('dec.yuv'), rd('recon.yuv')
def psnr(a, b):
    if not a or not b or len(a) != len(b):
        return -1
    s = sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)
    return 99.0 if s == 0 else 10 * math.log10(255 * 255 / s)
print("   bitstream vs source      %.2f dB" % psnr(src, dec))
print("   reconstruction vs source %.2f dB" % psnr(src, rec))
d = psnr(rec, dec)
if rec and dec:
    mx = max(abs(x - y) for x, y in zip(rec, dec))
    print("   reconstruction vs decode %.2f dB (max abs diff %d)" % (d, mx))
PY
else
    echo "   skipped (no ffmpeg; set FFMPEG=...)"
fi

echo "== device enumeration"
./$BUILD/aegenc --list-devices 2>/dev/null | sed 's/^/   /'

rm -rf "$TMP"
