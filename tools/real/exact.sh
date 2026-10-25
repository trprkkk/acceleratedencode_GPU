#!/bin/bash
# Bit-exactness on the real clips: encoder reconstruction vs FFmpeg's decode,
# every frame, through the coding order the encoder reports.
# usage: tools/real/exact.sh "<aegenc opts>" [bframes]   (QPS="22 30 38")
REAL=${REAL:-/tmp/real}
cd "$(dirname "$0")/../.."
FF=${FFMPEG:-/tmp/dl/ext/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2}
T=${TMPDIR:-/tmp}/aeg_exact; mkdir -p $T
opts="$1"; b=${2:-3}; ok=0; tot=0
for c in cock:640:352 vtest:384:288 mega:352:256 short:320:240 tree:320:240; do
 IFS=: read n w h <<<"$c"
 for q in ${QPS:-22 30 38}; do
  t=$T/${n}_$q; tot=$((tot+1))
  ./build/aegenc -w $w -h $h -n 30 -i $REAL/$n.yuv -o $t.264 --dump-recon $t.rec \
     --dump-order $t.ord --bframes $b -q $q $opts >$t.log 2>&1 || { echo "enc fail $n q$q"; continue; }
  "$FF" -v error -y -i $t.264 -f rawvideo -pix_fmt yuv420p $t.dec
  python3 - $t.rec $t.dec $w $h $t.ord ${n}q$q <<'PY' && ok=$((ok+1))
import sys
rec, dec, w, h, order, tag = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
fs = w * h * 3 // 2
A = open(rec, 'rb').read(); B = open(dec, 'rb').read()
o = [int(l) for l in open(order)]
n = len(A) // fs
if n != len(o) or len(B) // fs != n:
    print(tag, "COUNT MISMATCH", n, len(o), len(B) // fs); sys.exit(1)
for k in range(n):
    if A[k*fs:(k+1)*fs] != B[o[k]*fs:(o[k]+1)*fs]:
        print(tag, "MISMATCH coded", k, "display", o[k]); sys.exit(1)
PY
 done; done
echo "$ok/$tot exact ($opts, b=$b)"
