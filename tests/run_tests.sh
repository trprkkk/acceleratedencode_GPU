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

# B pictures are coded out of display order, so the reconstruction dump (coding
# order) and the decoder's output (display order) can only be compared after
# undoing that permutation.  It is the one the encoder schedules: the anchor
# first, then the group's B pictures, then the leftovers of the last, incomplete
# group -- which is what the loop below rebuilds from the frame and B counts.
cat > "$TMP/cmp_b.py" <<'PY'
import sys, os
rec, dec, w, h, n, b, tag = (sys.argv[1], sys.argv[2], int(sys.argv[3]),
                             int(sys.argv[4]), int(sys.argv[5]),
                             int(sys.argv[6]), sys.argv[7])
fs = w * h * 3 // 2
A = open(rec, 'rb').read() if os.path.exists(rec) else b''
B = open(dec, 'rb').read() if os.path.exists(dec) else b''
if len(A) != len(B) or len(A) < fs * n:
    print("   %-22s SIZE MISMATCH recon=%d dec=%d (expected %d)"
          % (tag, len(A), len(B), fs * n))
    sys.exit(1)
G = b + 1
gop = int(sys.argv[8]) if len(sys.argv) > 8 else 0
order = [0]
a = 0
last_idr = 0
while a + G < n:
    nxt = a + G
    if gop > 0 and nxt - last_idr >= gop:
        # no B picture may precede an IDR: the group's B pictures go out as P
        # pictures in display order and the IDR comes last
        order.extend(range(a + 1, nxt))
        order.append(nxt)
        last_idr = nxt
    else:
        order.append(nxt)
        order.extend(range(a + 1, nxt))
    a = nxt
order.extend(range(a + 1, n))
if len(order) != n:
    print("   %-22s BAD ORDER MAP (%d != %d)" % (tag, len(order), n))
    sys.exit(1)
bad = 0
first = None
mx = 0
for k in range(n):
    x = A[k * fs:(k + 1) * fs]
    y = B[order[k] * fs:(order[k] + 1) * fs]
    for i in range(fs):
        d = abs(x[i] - y[i])
        if d:
            bad += 1
            mx = max(mx, d)
            if first is None:
                first = (k, order[k], i // w, i % w)
if bad:
    print("   %-22s FAIL %d/%d samples differ, first at coding frame %d "
          "(display %d, x=%d y=%d), max abs diff %d"
          % (tag, bad, fs * n, first[0], first[1], first[3], first[2], mx))
    sys.exit(1)
print("   %-22s bit-exact (%d frames, coding order, max abs diff 0)" % (tag, n))
PY

# run_bcase <name> <w> <h> <frames> <bframes> <genyuv mode> <extra aegenc args...>
run_bcase() {
    name=$1; w=$2; h=$3; n=$4; b=$5; mode=$6; shift 6
    "$BUILD/genyuv" "$w" "$h" "$n" "$TMP/$name.yuv" "$mode" >/dev/null
    ./$BUILD/aegenc -w "$w" -h "$h" -n "$n" -i "$TMP/$name.yuv" \
        -o "$TMP/$name.264" --dump-recon "$TMP/$name.recon" --bframes "$b" "$@" \
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
    gop=0
    for a in "$@"; do
        if [ "$prev" = "-g" ]; then gop=$a; fi
        prev=$a
    done
    python3 "$TMP/cmp_b.py" "$TMP/$name.recon" "$TMP/$name.dec" "$w" "$h" \
        "$n" "$b" "$name" "$gop" || FAIL=1
}

# run_case <name> <w> <h> <frames> <genyuv mode> <extra aegenc args...>
run_case() {
    name=$1; w=$2; h=$3; n=$4; mode=$5; shift 5
    "$BUILD/genyuv" "$w" "$h" "$n" "$TMP/$name.yuv" "$mode" >/dev/null
    # --bframes 0: the cases below compare the dump against the decoded stream
    # frame by frame in *display* order, which is only the coding order when
    # the stream has no B pictures.  The B path has its own cases (run_bcase)
    # that rebuild the coding order before comparing.
    ./$BUILD/aegenc -w "$w" -h "$h" -n "$n" -i "$TMP/$name.yuv" \
        -o "$TMP/$name.264" --dump-recon "$TMP/$name.recon" --bframes 0 "$@" \
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
# The slow presets refine the winning inter vector by rate-distortion, and the
# refinement reads the neighbours' mvd and writes its own, so it has to be run
# through the slice threads as well: the threaded path codes a slice on a copy
# of the encoder, and a refinement that kept its state outside that copy would
# produce a different stream there.
run_case refine_thr    320 192  6 1 -q 26 -p 4 --slices 2 -t 2
# A 16x8 or 8x16 winner has two vectors and each is refined on its own,
# so the refinement now runs once per partition instead of once per
# macroblock; mode 1 moves the halves of the picture in opposite
# directions, which is what makes the encoder pick those shapes.
run_case refine_part   320 192  6 1 -q 26 -p 4 --slices 2 -t 2
# Level selection (--trellis) changes the coefficients *after* the quantiser,
# so everything the macroblock layer derives from them -- the coded block
# pattern, the counts the neighbours' contexts are built from, the
# reconstruction the next picture predicts from -- has to follow.  A stream
# that is off by one level still decodes, just not into what the encoder
# reconstructed, so it is checked like any other path.  Preset 3 covers the
# 4x4 blocks, preset 4 the 8x8 transform, and the threaded/ batched variants
# cover the copies of the shadow coder the slices and the batch path use.
run_case trellis4      320 192  8 1 -q 26 -p 3 --trellis 1
run_case trellis8      320 192  8 4 -q 24 -p 4 --trellis 1
run_case trellis_thr   320 192  6 1 -q 26 -p 4 --trellis 1 --slices 2 -t 2
AEG_COEF_BATCH=1 run_case trellis_batch 320 192 6 1 -q 26 -p 4 --trellis 1 --partitions 15
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
# A buffer model changes the QP per picture (that is the point), so it goes
# through the same conformance check as the plain ABR case.
run_case vbv          320 192 12 1 -q 24 --vbv-maxrate 400 --vbv-bufsize 60
run_case abr_gop4     320 192 16 4 -q 24 --bitrate 400 -g 4
run_case aq           320 192  8 4 -q 26 --aq 12
# AQ is what puts non-zero mb_qp_delta values inside a picture, and the CABAC
# context that codes the *first* bin of one is selected by the previous
# macroblock's delta -- which a skipped macroblock clears (9.3.2.7).  Both the
# unary code of the delta and that reset were wrong until this case existed:
# mode 3 gives long P_Skip runs for the delta to be forgotten in, and the two
# QPs give the AQ a reason to move the value both ways.
run_case aq_skip      320 192 10 3 -q 26 --aq 12 -g 4
run_case aq_skip_lo   320 192 10 3 -q 16 --aq 16 -g 4
# Intra_4x4 in every picture, and IDR pictures after P pictures (gop > 1),
# which is where the mode prediction and the reference handling meet.
run_case all_intra    320 192  6 1 -q 26 -g 1
run_case gop3         320 192 10 1 -q 26 -g 3
# B pictures.  Their macroblocks are direct (vectors derived from the co-located
# picture, no mvd at all), list 0, list 1 or bi-predictive, and the direct ones
# are where a wrong derivation shows up: it reconstructs a different picture
# without breaking the syntax, so only this comparison sees it.
#   one and three B pictures between anchors (different group sizes, and with
#   three a B picture that is not adjacent to the anchor it predicts from)
run_bcase bframes1     320 192 12 1 1 -q 24 --scenecut 0
run_bcase bframes3     320 192 12 3 4 -q 24 --scenecut 0
#   a still clip: every B macroblock collapses into direct-skip, which is the
#   skip flag and direct prediction with no residual at all
run_bcase bframes_still 320 192  8 2 3 -q 24 --scenecut 0
#   the lowest and highest rate: at q 51 every macroblock is a skip, at q 12
#   all of them carry residuals and the bi-predictive average is at its most
#   visible
run_bcase bframes_hi    320 192  9 1 2 -q 12 --scenecut 0
run_bcase bframes_lo    320 192  9 1 2 -q 51 --scenecut 0
#   a stream whose length is not a multiple of the group size (the tail is
#   coded as P pictures) and one with an IDR picture in the middle
run_bcase bframes_odd   320 192 11 2 1 -q 26 --scenecut 0
run_bcase bframes_idr   320 192 12 1 1 -q 26 --scenecut 0 -g 4
#   the 8x8 transform and the deblocking filter both treat B macroblocks
#   differently from P ones: the transform flag is coded for a direct
#   macroblock, and the boundary strength compares both prediction lists
run_bcase bframes_t8    320 192 10 2 4 -q 22 --scenecut 0
run_bcase bframes_nodeb 320 192 10 2 1 -q 24 --scenecut 0 --no-deblock
run_bcase bframes_q8x8  320 192 10 1 1 -q 24 --scenecut 0 --no-8x8dct
#   the rate controller gives a picture its QP from the running bit count, and a
#   B picture is coded coarser than the anchors around it: the mb_qp_delta that
#   follows has to survive both, which is what this case is for
run_bcase bframes_abr   320 192 12 2 1 -q 24 --scenecut 0 --bitrate 300
#   B_8x8: four 8x8 groups, each picking direct / list 0 / list 1 / both.  It
#   follows the preset (3 and up), not the partition mask, so the mask here
#   keeps the P pictures at 16x16 -- the point of the case is the B macroblock
#   layer, including the sub_mb_type tree and the per-group mvd suppression
run_bcase bframes_8x8   320 192 12 2 4 -q 24 --scenecut 0 -p 3 --partitions 1
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

echo "== skip coding"
# A still clip has to be coded almost entirely of skips: P_Skip is one
# context-coded flag, and it is only reachable because the search adds the
# vector the skip rule derives as a candidate of its own.  Before that
# the encoder skips 74% of this clip's macroblocks (85% of the P ones -- the
# rest is the IDR picture, which cannot skip at all); the guard is 65%, well
# below that, so it fails only if the skip path breaks rather than when the
# mode decision shifts.
"$BUILD/genyuv" 320 192 8 "$TMP/still.yuv" 3 >/dev/null
./$BUILD/aegenc -w 320 -h 192 -n 8 -i "$TMP/still.yuv" -o "$TMP/still.264" \
    -q 26 >"$TMP/still.log" 2>&1
python3 -c '
import re, sys
log = open(sys.argv[1]).read()
m = re.search(r"(\d+) skip of (\d+)", log)
if not m:
    print("   skip coding: no macroblock statistics in the encoder output")
    sys.exit(1)
skip, total = int(m.group(1)), int(m.group(2))
frac = 100.0 * skip / total
ok = frac >= 65.0
print("   still clip: %.0f%% skip (%d of %d)%s" % (frac, skip, total,
      "" if ok else "  FAIL"))
sys.exit(0 if ok else 1)
' "$TMP/still.log" || FAIL=1

echo "== scene cuts"
# Two unrelated clips back to back: the picture at the join has nothing to
# predict from, so the encoder has to spend an IDR on it -- and only on it.  The
# NAL count is checked directly, because a scene-cut detector that never fires
# and one that fires on everything both "work" otherwise.
"$BUILD/genyuv" 320 192 30 "$TMP/cutA.yuv" 0 >/dev/null
"$BUILD/genyuv" 320 192 30 "$TMP/cutB.yuv" 4 >/dev/null
cat "$TMP/cutA.yuv" "$TMP/cutB.yuv" > "$TMP/cut.yuv"
# --bframes 0 for the reconstruction comparison (the dump is in coding order
# when there are B pictures, see run_case), and the same clip once more with B
# pictures for the IDR count: a cut still has to become an IDR when the group
# structure puts B pictures around it.
./$BUILD/aegenc -w 320 -h 192 -n 60 -q 26 -i "$TMP/cut.yuv" \
    -o "$TMP/cut.264" -g 250 --bframes 0 --dump-recon "$TMP/cut.recon" \
    >"$TMP/cut.log" 2>&1 \
    || { echo "   cut clip: encode failed"; FAIL=1; }
./$BUILD/aegenc -w 320 -h 192 -n 60 -q 26 -i "$TMP/cut.yuv" \
    -o "$TMP/cutb.264" -g 250 --bframes 3 >"$TMP/cutb.log" 2>&1 \
    || { echo "   cut clip with B pictures: encode failed"; FAIL=1; }
./$BUILD/aegenc -w 320 -h 192 -n 120 -q 26 -i "$TMP/seed_search.yuv" \
    -o "$TMP/nocut.264" -g 250 --bframes 0 >/dev/null 2>&1
./$BUILD/aegenc -w 320 -h 192 -n 120 -q 26 -i "$TMP/seed_search.yuv" \
    -o "$TMP/nocutb.264" -g 250 --bframes 3 >/dev/null 2>&1
if ! command -v "$FF" >/dev/null 2>&1; then
    echo "   scenecut: skipped (no ffmpeg)"
else
    "$FF" -v error -y -i "$TMP/cut.264" -f rawvideo -pix_fmt yuv420p \
        "$TMP/cut.dec" 2>"$TMP/cut.decerr"
    if [ -s "$TMP/cut.decerr" ]; then
        echo "   cut clip: decoder reported problems:"
        sed 's/^/     /' "$TMP/cut.decerr" | head -3
        FAIL=1
    fi
    python3 "$TMP/cmp.py" "$TMP/cut.recon" "$TMP/cut.dec" 320 192 60 cut || FAIL=1
fi
python3 - "$TMP/cut.264" "$TMP/nocut.264" "$TMP/cutb.264" "$TMP/nocutb.264" \
    <<'PY2' || FAIL=1
import sys

def idr_count(path):
    d = open(path, 'rb').read()
    n, i = 0, 0
    while True:
        j = d.find(b'\x00\x00\x01', i)
        if j < 0:
            return n
        if j + 3 < len(d) and (d[j + 3] & 0x1f) == 5:
            n += 1
        i = j + 3

cut = idr_count(sys.argv[1])
plain = idr_count(sys.argv[2])
cutb = idr_count(sys.argv[3])
plainb = idr_count(sys.argv[4])
ok = True
if cut != 2:
    print("   60 frame clip with one scene cut: %d IDR NALs, wanted 2  FAIL" % cut)
    ok = False
if plain != 1:
    print("   120 frame clip of ordinary motion: %d IDR NALs, wanted 1  FAIL"
          % plain)
    ok = False
if cutb != 2:
    print("   with B pictures: %d IDR NALs, wanted 2  FAIL" % cutb)
    ok = False
if plainb != 1:
    print("   with B pictures, ordinary motion: %d IDR NALs, wanted 1  FAIL"
          % plainb)
    ok = False
if ok:
    print("   one cut -> 2 IDR NALs, ordinary motion -> 1 (no false positives),"
          " with and without B pictures")
sys.exit(0 if ok else 1)
PY2

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

echo "== VBV / HRD"
# The buffer model has to do two things: leave a stream that fits its channel
# alone (the bytes must be exactly what the same ABR configuration without a
# buffer produces), and raise the QP when a picture would not fit -- including
# saying so when even QP 51 cannot make it fit, because a model that can never
# report an underflow is not a model.
"$BUILD/genyuv" 320 192 40 "$TMP/vbv.yuv" 0 >/dev/null
./$BUILD/aegenc -w 320 -h 192 -n 40 -q 24 -i "$TMP/vbv.yuv" \
    -o "$TMP/vbv_off.264" --bitrate 400 -v 0
./$BUILD/aegenc -w 320 -h 192 -n 40 -q 24 -i "$TMP/vbv.yuv" \
    -o "$TMP/vbv_wide.264" --bitrate 400 --vbv-maxrate 400 --vbv-bufsize 120 \
    >"$TMP/vbv_wide.log" 2>&1
if cmp -s "$TMP/vbv_off.264" "$TMP/vbv_wide.264"; then
    echo "   generous buffer (120 kbit): stream unchanged"
else
    echo "   generous buffer (120 kbit): stream differs from the ABR stream  FAIL"
    FAIL=1
fi
./$BUILD/aegenc -w 320 -h 192 -n 40 -q 24 -i "$TMP/vbv.yuv" \
    -o "$TMP/vbv_tight.264" --bitrate 400 --vbv-maxrate 400 --vbv-bufsize 30 \
    >"$TMP/vbv_tight.log" 2>&1
./$BUILD/aegenc -w 320 -h 192 -n 40 -q 24 -i "$TMP/vbv.yuv" \
    -o "$TMP/vbv_alone.264" --vbv-maxrate 400 --vbv-bufsize 120 \
    >"$TMP/vbv_alone.log" 2>&1
python3 - "$TMP" <<'PY2' || FAIL=1
import re, sys, os
tmp = sys.argv[1]

def read(name):
    return open(os.path.join(tmp, name)).read()

def vbv_line(text):
    m = re.search(r'vbv: buffer low water ([0-9.]+) kbit, (\d+) underflow\(s\), '
                  r'(\d+) picture\(s\) raised to fit', text)
    return tuple(float(m.group(1)) if i == 0 else int(m.group(i + 1))
                 for i in range(3)) if m else None

wide = vbv_line(read('vbv_wide.log'))
tight = vbv_line(read('vbv_tight.log'))
alone = vbv_line(read('vbv_alone.log'))
ok = True

if wide is None or tight is None or alone is None:
    print("   VBV: the model's report is missing")
    ok = False
else:
    if wide[1] != 0 or wide[2] != 0:
        print("   byte-identical stream reported %d underflows, %d clamps  FAIL"
              % (wide[1], wide[2]))
        ok = False
    if tight[1] + tight[2] == 0:
        print("   a 30 kbit buffer fit every picture unchanged  FAIL")
        ok = False
    if wide[0] <= tight[0]:
        print("   the tight buffer did not get closer to empty  FAIL")
        ok = False

# --vbv-maxrate/--vbv-bufsize alone must imply ABR (a buffer model without a
# rate target cannot squeeze anything), and the rate must still land.
kbps = os.path.getsize(os.path.join(tmp, 'vbv_alone.264')) * 8 * 30 / 40 / 1000.0
if abs(kbps - 400) > 40:
    print("   --vbv-* alone: %.1f kb/s, wanted 400  FAIL" % kbps)
    ok = False
print("   generous buffer: low water %.1f kbit, 0 clamps" % wide[0]
      if ok and wide else "   VBV check failed")
print("   30 kbit buffer:  low water %.1f kbit, %d underflow(s), %d clamp(s)"
      % tight[:3] if ok and tight else "")
print("   --vbv-* alone (no --bitrate): %.1f kb/s" % kbps)
sys.exit(0 if ok else 1)
PY2

echo "== two-pass rate control"
# Two-pass on a clip whose two halves have very different complexity: the point
# of the first pass is to *measure* that difference and spend accordingly.  The
# rate controller stays on in pass 2, so the stream still lands on target; the
# first pass only decides how the bits are shared out.
./$BUILD/aegenc -w 320 -h 192 -n 60 -q 26 -i "$TMP/cut.yuv" \
    -o "$TMP/2p1.264" --pass 1 --stats "$TMP/2p.stats" >"$TMP/2p1.log" 2>&1 \
    || { echo "   pass 1 failed"; FAIL=1; }
./$BUILD/aegenc -w 320 -h 192 -n 60 -i "$TMP/cut.yuv" \
    -o "$TMP/2p2.264" --pass 2 --stats "$TMP/2p.stats" --bitrate 250 \
    --dump-recon "$TMP/2p.recon" >"$TMP/2p2.log" 2>&1 \
    || { echo "   pass 2 failed"; FAIL=1; }
# The reference for "two-pass beats one-pass" has to be coded the same way the
# two-pass run is: --pass 1 measures one packet per picture in display order, so
# a two-pass run cannot use B pictures at all (the CLI turns them off), and the
# one-pass stream has to be P-only for the comparison to mean anything.
./$BUILD/aegenc -w 320 -h 192 -n 60 -i "$TMP/cut.yuv" \
    -o "$TMP/1p.264" --bitrate 250 --bframes 0 >"$TMP/1p.log" 2>&1 \
    || { echo "   one-pass control failed"; FAIL=1; }
python3 - "$TMP/2p2.log" "$TMP/1p.log" <<'PY2' || FAIL=1
import re, sys

def measure(path):
    t = open(path).read()
    m = re.search(r'bitrate ([0-9.]+) kb/s at \d+ fps, PSNR ([0-9.]+) dB', t)
    return (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)

two, one = measure(sys.argv[1]), measure(sys.argv[2])
ok = True
if abs(two[0] - 250) > 15:
    print("   two-pass rate %.1f kb/s, target 250  FAIL" % two[0])
    ok = False
if two[1] <= one[1]:
    print("   two-pass %.2f dB did not beat one-pass %.2f dB  FAIL"
          % (two[1], one[1]))
    ok = False
print("   1-pass %5.1f kb/s %.2f dB  ->  2-pass %5.1f kb/s %.2f dB"
      % (one[0], one[1], two[0], two[1]))
sys.exit(0 if ok else 1)
PY2
if command -v "$FF" >/dev/null 2>&1; then
    "$FF" -v error -y -i "$TMP/2p2.264" -f rawvideo -pix_fmt yuv420p \
        "$TMP/2p.dec" 2>"$TMP/2p.decerr"
    if [ -s "$TMP/2p.decerr" ]; then
        echo "   two-pass: decoder reported problems:"
        sed 's/^/     /' "$TMP/2p.decerr" | head -3
        FAIL=1
    fi
    python3 "$TMP/cmp.py" "$TMP/2p.recon" "$TMP/2p.dec" 320 192 60 two_pass \
        || FAIL=1
fi

echo "== device enumeration"
./$BUILD/aegenc --list-devices 2>/dev/null | sed 's/^/   /'

rm -rf "$TMP"
if [ "$FAIL" != 0 ]; then
    echo "== FAILED"
    exit 1
fi
echo "== all conformance cases bit-exact"
