#!/bin/sh
# x264_hint_autoskip.sh - estimate hint generation cost and fall back to base
# when generation is likely to lose end-to-end.  Opt-in driver; x264 itself is
# unchanged.
#
# usage: x264_hint_autoskip.sh <raw.yuv> <w> <h> <frames> <preset> <crf> \
#                              <out264> <hintbin> [threads]
set -eu
IN=$1; W=$2; H=$3; NF=$4; PRESET=$5; CRF=$6; OUT=$7; HINT=$8; TH=${9:-4}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
X264="$ROOT/forks/x264/x264.exe"
GH="$ROOT/build/gpu_hints"
TMP="$ROOT/build/work/autoskip"
mkdir -p "$TMP"

ms() { python3 -c "import time;print(time.time())"; }

# 10-frame base estimate
t0=$(ms)
env -u X264_MV_HINT "$X264" --input-res ${W}x${H} --fps 30 --frames 10 \
    --threads "$TH" --preset "$PRESET" --crf "$CRF" --me hex --subme 7 \
    -o "$TMP/base10.264" "$IN" >/dev/null 2>&1
t1=$(ms)

# 10-frame hint estimate
t2=$(ms)
"$GH" -i "$IN" -w "$W" -h "$H" -n 10 -o "$TMP/hint10.bin" -b cuda >/dev/null 2>&1
t3=$(ms)

BASE=$(python3 -c "print(($t1-$t0)/10.0)")
GEN=$(python3 -c "print(($t3-$t2)/10.0)")
SKIP=$(python3 -c "print(1 if $GEN > 0.3*$BASE else 0)")
echo "base_ms/frame=$BASE gen_ms/frame=$GEN skip=$SKIP" >&2

if [ "$SKIP" = 1 ]; then
    env -u X264_MV_HINT "$X264" --input-res ${W}x${H} --fps 30 --frames "$NF" \
        --threads "$TH" --preset "$PRESET" --crf "$CRF" --me hex --subme 7 \
        -o "$OUT" "$IN" >/dev/null 2>&1
    exit 0
fi

"$GH" -i "$IN" -w "$W" -h "$H" -n "$NF" -o "$HINT" -b cuda >/dev/null 2>&1
X264_MV_HINT="$HINT" X264_MV_HINT_MODE=limit:4 "$X264" \
    --input-res ${W}x${H} --fps 30 --frames "$NF" --threads "$TH" \
    --preset "$PRESET" --crf "$CRF" --me hex --subme 7 \
    -o "$OUT" "$IN" >/dev/null 2>&1
