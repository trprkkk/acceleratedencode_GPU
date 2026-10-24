#!/bin/sh
# Conformance sweep: encode a set of clips over the option matrix, decode each
# stream with an independent decoder and require the encoder's own
# reconstruction to be bit-identical.  Any difference, encoder failure or
# decoder complaint is a failure.
#
#   tools/sweep.sh clip1 clip2 ...        (clips are /tmp/<name>.yuv)
#   FFMPEG=/path/to/ffmpeg tools/sweep.sh ...
cd "$(dirname "$0")/.." || exit 1
FF=${FFMPEG:-/tmp/dl/ext/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2}
TMP=${TMPDIR:-/tmp}
tot=0; fail=0
for c in "$@"; do
  [ -f "$TMP/$c.yuv" ] || continue
  W=$(python3 -c "
import os, sys
n = os.path.getsize('$TMP/$c.yuv')
for w, h in [(320,192),(352,288),(176,144),(160,120),(96,96),(64,48),(32,32),(32,16),(16,16)]:
    if n % (w * h * 3 // 2) == 0:
        print(w, h); break
else:
    print(0, 0)")
  set -- $W; w=$1; h=$2
  [ "$w" = "0" ] && { echo "size? $c"; continue; }
  for q in 20 26 34; do for p in 8 15; do for g in 250 3; do
    tot=$((tot+1))
    if ! ./build/aegenc -w "$w" -h "$h" -q "$q" -n 6 -g "$g" --partitions "$p" \
         -i "$TMP/$c.yuv" -o "$TMP/sw.264" --dump-recon "$TMP/swr.yuv" \
         -v 0 2>"$TMP/sw.err"; then
      echo "ENC-FAIL $c q$q p$p g$g"; fail=$((fail+1)); continue
    fi
    "$FF" -y -v error -i "$TMP/sw.264" -f rawvideo -pix_fmt yuv420p \
        "$TMP/swd.yuv" 2>"$TMP/swderr"
    if [ -s "$TMP/swderr" ]; then
      echo "DEC-ERR $c q$q p$p g$g: $(head -c 100 "$TMP/swderr")"
      fail=$((fail+1)); continue
    fi
    cmp -s "$TMP/swr.yuv" "$TMP/swd.yuv" || {
      echo "DIFF $c q$q p$p g$g w$w"; fail=$((fail+1)); }
  done; done; done
done
echo "sweep: $tot cases, $fail failures"
[ "$fail" = 0 ]
