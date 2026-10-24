#!/bin/sh
# Local verification: build, encode synthetic clips, decode them with an
# independent decoder and check that the encoder's own reconstruction is
# *bit-exact* with the decoder's output on every frame.
#
# Conformance (encoder recon == decoder output) is the property that matters:
# it proves the transform/quantisation/prediction/deblocking maths and the
# bitstream syntax agree with the standard, because any disagreement would
# drift on the following P pictures.
#
# Set FFMPEG=/path/to/ffmpeg to use a specific build; without one the clip
# checks are skipped (the table/CAVLC checks in `make test` still run).
set -e
BUILD=build
FF=${FFMPEG:-ffmpeg}
TMP=${TMPDIR:-/tmp}/aegtest.$$
mkdir -p "$TMP"
FAIL=0

echo "== build"
make -s >/dev/null
make -s "$BUILD/genyuv"

# compare BOTH files fully, over every frame
cat > "$TMP/cmp.py" <<'PY'
import sys, os, math
rec, dec, w, h, n, tag = (sys.argv[1], sys.argv[2], int(sys.argv[3]),
                          int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
fs = w * h * 3 // 2
a = open(rec, 'rb').read() if os.path.exists(rec) else b''
b = open(dec, 'rb').read() if os.path.exists(dec) else b''
if len(a) != len(b) or len(a) < fs * n:
    print("   %-22s SIZE MISMATCH recon=%d dec=%d (expected %d)"
          % (tag, len(a), len(b), fs * n))
    sys.exit(1)
bad = 0
first = None
for i in range(len(a)):
    if a[i] != b[i]:
        bad += 1
        if first is None:
            f, o = divmod(i, fs)
            first = (f, o // w, o % w)
mx = max(abs(x - y) for x, y in zip(a, b))
if bad:
    print("   %-22s FAIL %d/%d samples differ, first at frame %d (x=%d y=%d), "
          "max abs diff %d" % (tag, bad, len(a), first[0], first[2], first[1], mx))
    sys.exit(1)
print("   %-22s bit-exact (%d frames, max abs diff 0)" % (tag, n))
PY

# run_case <name> <w> <h> <frames> <genyuv mode> <extra aegenc args...>
run_case() {
    name=$1; w=$2; h=$3; n=$4; mode=$5; shift 5
    "$BUILD/genyuv" "$w" "$h" "$n" "$TMP/$name.yuv" "$mode" >/dev/null
    ./$BUILD/aegenc -w "$w" -h "$h" -n "$n" -i "$TMP/$name.yuv" \
        -o "$TMP/$name.264" --dump-recon "$TMP/$name.recon" "$@" \
        >"$TMP/$name.log" 2>&1 || { echo "   $name: encode failed"; FAIL=1; return; }
    if ! command -v "$FF" >/dev/null 2>&1; then
        echo "   $name: skipped (no ffmpeg; set FFMPEG=...)"
        return
    fi
    "$FF" -v error -y -i "$TMP/$name.264" -f rawvideo -pix_fmt yuv420p \
        "$TMP/$name.dec" 2>"$TMP/$name.decerr"
    if [ -s "$TMP/$name.decerr" ]; then
        echo "   $name: decoder reported problems:"
        sed 's/^/     /' "$TMP/$name.decerr" | head -3
        FAIL=1
    fi
    python3 "$TMP/cmp.py" "$TMP/$name.recon" "$TMP/$name.dec" "$w" "$h" "$n" "$name" || FAIL=1
}

echo "== inter conformance (encoder recon vs independent decoder, every frame)"
run_case h264_320x192 320 192 10 0 -q 23
run_case lowqp        320 192  6 0 -q 12
run_case highqp       320 192  6 0 -q 40
run_case nodeblock    320 192  6 0 -q 23 --no-deblock
run_case preset0      320 192  6 0 -q 23 -p 0
run_case preset4      320 192  6 0 -q 23 -p 4
# Preset 3/4 choose the Intra_4x4 mode by exact rate-distortion (quantise,
# reconstruct, count the real bits), which is a different code path from the
# SATD approximation the faster presets use.
run_case preset3_intra 320 192  4 1 -q 26 -p 3 -g 1
run_case odd_208      320 208  4 0 -q 26
run_case small_64     64  64   6 0 -q 23
run_case cpu_only     320 192  4 0 -q 23 -b cpu
# inter partitioning and P_Skip: motion that a single 16x16 vector cannot
# describe, then a still picture (which must collapse into skip runs).
run_case part16x8     320 192  8 1 -q 26 --partitions 7
run_case part8x16     320 192  8 2 -q 26 --partitions 7
run_case part_none    320 192  8 1 -q 26 --partitions 1
run_case pskip        320 192  8 3 -q 26
# P_8x8: four quadrants moving in four directions, so the macroblock needs one
# motion vector per 8x8 block; part8x8only makes that the only partition shape.
run_case part8x8      320 192  8 4 -q 26 --partitions 15
run_case part8x8only  320 192  8 4 -q 26 --partitions 8
# Mixed sub-macroblock shapes across P pictures: the predictor of a 4x4/8x4
# partition may only use the sub-blocks the decoder has already filled in, so
# this case exercises the availability rule (it desynced before that fix).
run_case p8x8_gop2    320 192  8 4 -q 26 --partitions 15 -g 2
run_case p8x8_q12     320 192  6 4 -q 12 --partitions 15
# The quarter-pel stage is what most of the inter efficiency rides on, so both
# it and the integer-only path get their own case: the two use different
# motion compensation code (integer copy vs. the 6-tap filter chain), and the
# half-pel-in-both-axes position is the one that is easy to round twice.
run_case subpel       320 192  8 1 -q 26 --subme 1
run_case integer_me   320 192  8 1 -q 26 --subme 0
# The whole-frame search seed (a fixed candidate grid evaluated as one batch,
# the motion-estimation work the GPU actually runs) is switched on by the
# dispatcher only when that stage runs on the GPU; AEG_SEED=1 forces it here so
# the extra candidate start can never break the stream when it is enabled.
AEG_SEED=1 run_case seed_search  320 192  8 1 -q 26
# The second level of that path refines all macroblocks of a picture together
# in quarter-pel rounds (AEG_PARFULL=1) instead of the per-block walk; same
# conformance requirement.
AEG_SEED=1 AEG_PARFULL=1 run_case par_subpel 320 192 8 1 -q 26
# The batched coefficient pass (tq_batch + iq_idct_batch over the sixteen luma
# blocks of one inter candidate) is normally only taken when the dispatcher
# puts both coefficient stages on a device; AEG_COEF_BATCH=1 runs the same code
# path against the CPU reference ops, which has to produce the same stream as
# the per-block path it replaces.
AEG_COEF_BATCH=1 run_case coef_batch  320 192  8 1 -q 26 --partitions 15
AEG_COEF_BATCH=1 run_case coef_batch4 320 192  6 4 -q 22
# The provisional RD batching puts *all* inter candidates of a macroblock in one
# pair of kernel launches and decides afterwards.  Forced here against the CPU
# reference ops it has to produce the same stream as the serial candidate loop,
# on an all-inter clip (p0/partitions) and on one with intra macroblocks too.
AEG_RD_BATCH=1 run_case rd_batch    320 192  8 1 -q 26 --partitions 15
AEG_RD_BATCH=1 run_case rd_batch4   320 192  6 4 -q 22 -g 3
# Rate control and adaptive quantisation both change the QP per picture (and,
# for AQ, per macroblock), which is exactly where an mb_qp_delta could desync:
# a macroblock without a coded residual must not change the decoder's QP.
run_case abr          320 192 16 1 -q 24 --bitrate 300
run_case abr_gop4     320 192 16 4 -q 24 --bitrate 400 -g 4
run_case aq           320 192  8 4 -q 26 --aq 12
# Intra_4x4 in every picture, and IDR pictures after P pictures (gop > 1),
# which is where the mode prediction and the reference handling meet.
run_case all_intra    320 192  6 1 -q 26 -g 1
run_case gop3         320 192 10 1 -q 26 -g 3
# Slices: every slice is its own NAL and start code, so a macroblock in the top
# row of a non-first slice must not predict from anything above it, and each
# slice's data has to end exactly on its last macroblock (a slice that ends in
# the middle of a picture is what tells the decoder to stop).  The decoder is
# strict about both, so these run every slice count that the encoder supports
# end to end, plus a picture whose rows do not divide evenly.
run_case slices2      320 192  8 1 -q 26 --slices 2
run_case slices3      320 192  8 4 -q 26 --slices 3
run_case slices4      320 192  8 1 -q 26 --slices 4 -g 4
run_case slices6      320 192  8 4 -q 22 --slices 6 --partitions 15
run_case slices12     320 192  6 0 -q 26 --slices 12
run_case slices_odd   320 208  4 1 -q 24 --slices 5
run_case slices_small  64  64  6 0 -q 23 --slices 2
run_case slices_pskip 320 192  8 3 -q 26 --slices 3
# The same slices coded serially (-t 1) and by two threads have to produce the
# same bytes -- a slice's bits are a function of its own rows only, so the
# thread count must not be visible in the bitstream.
run_case slices_serial 320 192 8 1 -q 26 --slices 4 -t 1
run_case slices_thr    320 192 8 1 -q 26 --slices 4 -t 2
# -t without --slices: one slice per thread, which is the shape the option is
# meant for.
run_case threads_only  320 192 8 4 -q 26 -t 2
# Threads and the batched coefficient path together: the batch buffers live in
# the encoder struct, which the threads copy, so each thread must end up with
# its own.
AEG_COEF_BATCH=1 run_case slices_coef 320 192 6 1 -q 26 --slices 4 -t 2 --partitions 15

echo "== PSNR"
python3 - "$TMP" "320" "192" "10" <<'PY'
import sys, os, math
tmp, w, h, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
fs = w * h * 3 // 2
src = open(os.path.join(tmp, 'h264_320x192.yuv'), 'rb').read()[:fs * n]
for tag in ('h264_320x192.recon', 'h264_320x192.dec'):
    p = os.path.join(tmp, tag)
    if not os.path.exists(p):
        continue
    d = open(p, 'rb').read()[:len(src)]
    s = sum((x - y) ** 2 for x, y in zip(src, d)) / len(src)
    print("   %-22s %.2f dB" % (tag, 99.0 if s == 0 else 10 * math.log10(255 * 255 / s)))
PY

echo "== rate"
sed -n 's/^bitrate /   bitrate /p' "$TMP/h264_320x192.log"

echo "== rate control"
"$BUILD/genyuv" 320 192 40 "$TMP/abr.yuv" 0 >/dev/null
./$BUILD/aegenc -w 320 -h 192 -n 40 -q 24 -i "$TMP/abr.yuv" \
    -o "$TMP/abr.264" --bitrate 400 -v 0
python3 -c '
import os, sys
size = os.path.getsize(sys.argv[1])
kbps = size * 8 * 30 / 40 / 1000.0
ok = abs(kbps - 400) <= 40
print("   ABR target 400 kb/s -> %.1f kb/s%s" % (kbps, "" if ok else "  FAIL"))
sys.exit(0 if ok else 1)
' "$TMP/abr.264" || FAIL=1

echo "== device enumeration"
./$BUILD/aegenc --list-devices 2>/dev/null | sed 's/^/   /'

rm -rf "$TMP"
if [ "$FAIL" != 0 ]; then
    echo "== FAILED"
    exit 1
fi
echo "== all conformance cases bit-exact"
