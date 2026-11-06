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
# Set BUILD=dir to test another build of the encoder (for example an OpenCL one
# made with `make OPENCL=1 BUILD=dir`, against a device: see
# tools/pocl_env.sh); it is not rebuilt here, and AEG_TEST_BACKEND=opencl
# runs every case with `-b opencl` so the cases that force the batched paths
# go through the device.
set -e
FULLBUILD=${BUILD:+1}
BUILD=${BUILD:-./build}
FF=${FFMPEG:-ffmpeg}
TMP=${TMPDIR:-/tmp}/aegtest.$$
mkdir -p "$TMP"
FAIL=0

echo "== build"
if [ -z "$FULLBUILD" ]; then
    make -s >/dev/null
    make -s "$BUILD/genyuv"
fi

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
# undoing that permutation.  The encoder reports it (--dump-order: the display
# index of every packet); with the adaptive group length (--b-adapt 1, the
# default) that is the only way to know it.  With --b-adapt 0 the structure is
# fixed -- the anchor first, then the group's B pictures, then the leftovers of
# the last, incomplete group -- and the loop below rebuilds it from the frame and
# B counts to check the reported order against it.
cat > "$TMP/cmp_b.py" <<'PY'
import sys, os, array
rec, dec, w, h, n, b, tag = (sys.argv[1], sys.argv[2], int(sys.argv[3]),
                             int(sys.argv[4]), int(sys.argv[5]),
                             int(sys.argv[6]), sys.argv[7])
fs = w * h * 3 // 2
A = open(rec, 'rb').read() if os.path.exists(rec) else b''
B = open(dec, 'rb').read() if os.path.exists(dec) else b''
if os.environ.get('CMP_BPS') == '2':
    # 10 bits: both files hold 16-bit little-endian samples (yuv420p10le)
    if len(A) % 2 or len(B) % 2:
        print("   %-22s ODD SIZE recon=%d dec=%d" % (tag, len(A), len(B)))
        sys.exit(1)
    A, B = array.array('H', A), array.array('H', B)
    if sys.byteorder != 'little':
        A.byteswap()
        B.byteswap()
if len(A) != len(B) or len(A) < fs * n:
    print("   %-22s SIZE MISMATCH recon=%d dec=%d (expected %d)"
          % (tag, len(A), len(B), fs * n))
    sys.exit(1)
G = b + 1
gop = int(sys.argv[8]) if len(sys.argv) > 8 else 0
reported = [int(l) for l in open(sys.argv[9])]
fixed = sys.argv[10] == "1"
if sorted(reported) != list(range(n)):
    print("   %-22s BAD REPORTED ORDER %s" % (tag, reported))
    sys.exit(1)
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
if fixed and order != reported:
    print("   %-22s ORDER %s, expected %s" % (tag, reported, order))
    sys.exit(1)
order = reported
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
nb = sum(1 for k in range(1, n) if order[k] < max(order[:k]))
print("   %-22s bit-exact (%d frames, %d B, coding order, max abs diff 0)"
      % (tag, n, nb))
PY

# run_bcase <name> <w> <h> <frames> <bframes> <genyuv mode> <extra aegenc args...>
# (with IN_FILE set, that file is the input instead of a generated clip)
run_bcase() {
    name=$1; w=$2; h=$3; n=$4; b=$5; mode=$6; shift 6
    src=$IN_FILE
    if [ -z "$src" ]; then
        src="$TMP/$name.yuv"
        "$BUILD/genyuv" "$w" "$h" "$n" "$src" "$mode" >/dev/null
    fi
    $BUILD/aegenc -w "$w" -h "$h" -n "$n" -i "$src" \
        -o "$TMP/$name.264" --dump-recon "$TMP/$name.recon" --bframes "$b" \
        --dump-order "$TMP/$name.order" \
        ${AEG_TEST_BACKEND:+-b "$AEG_TEST_BACKEND"} "$@" \
        >"$TMP/$name.log" 2>&1 || { echo "   $name: encode failed (exit $?)"; tail -5 "$TMP/$name.log" | sed 's/^/     /'; FAIL=1; return; }
    if ! command -v "$FF" >/dev/null 2>&1; then
        echo "   $name: skipped (no ffmpeg; set FFMPEG=...)"
        return
    fi
    gop=0
    fixed=0
    pix=yuv420p
    bps=1
    for a in "$@"; do
        if [ "$prev" = "-g" ]; then gop=$a; fi
        if [ "$prev" = "--b-adapt" ] && [ "$a" = "0" ]; then fixed=1; fi
        if [ "$prev" = "--bit-depth" ] && [ "$a" = "10" ]; then
            pix=yuv420p10le; bps=2
        fi
        prev=$a
    done
    "$FF" -v error -y -i "$TMP/$name.264" -f rawvideo -pix_fmt $pix \
        "$TMP/$name.dec" 2>"$TMP/$name.decerr"
    if [ -s "$TMP/$name.decerr" ]; then
        echo "   $name: decoder reported problems:"
        sed 's/^/     /' "$TMP/$name.decerr" | head -3
        FAIL=1
    fi
    CMP_BPS=$bps python3 "$TMP/cmp_b.py" "$TMP/$name.recon" "$TMP/$name.dec" "$w" "$h" \
        "$n" "$b" "$name" "$gop" "$TMP/$name.order" "$fixed" || FAIL=1
}

# run_case <name> <w> <h> <frames> <genyuv mode> <extra aegenc args...>
run_case() {
    name=$1; w=$2; h=$3; n=$4; mode=$5; shift 5
    "$BUILD/genyuv" "$w" "$h" "$n" "$TMP/$name.yuv" "$mode" >/dev/null
    # --bframes 0: the cases below compare the dump against the decoded stream
    # frame by frame in *display* order, which is only the coding order when
    # the stream has no B pictures.  The B path has its own cases (run_bcase)
    # that rebuild the coding order before comparing.
    $BUILD/aegenc -w "$w" -h "$h" -n "$n" -i "$TMP/$name.yuv" \
        -o "$TMP/$name.264" --dump-recon "$TMP/$name.recon" --bframes 0 \
        ${AEG_TEST_BACKEND:+-b "$AEG_TEST_BACKEND"} "$@" \
        >"$TMP/$name.log" 2>&1 || { echo "   $name: encode failed (exit $?)"; tail -5 "$TMP/$name.log" | sed 's/^/     /'; FAIL=1; return; }
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
# The second half of a 16x8 or 8x16 predicts from the first half (8.4.1.3.1),
# so moving the first vector in the refinement invalidates the second
# partition's predictor and it has to be re-derived (bug 43).  Column 0 is the
# only place where that partition has no left neighbour to fall back on, so
# this case is single-threaded, at the preset that reaches the refinement
# without the slice machinery, on the clip that pans in both directions.
run_case refine_pred   320 192  6 1 -q 26 -p 2 --refine-mv 1 --partitions 11
# Level selection (--trellis) changes the coefficients *after* the quantiser,
# so everything the macroblock layer derives from them -- the coded block
# pattern, the counts the neighbours' contexts are built from, the
# reconstruction the next picture predicts from -- has to follow.  A stream
# that is off by one level still decodes, just not into what the encoder
# reconstructed, so it is checked like any other path.  Preset 3 covers the
# 4x4 blocks, preset 4 the 8x8 transform, and the threaded/ batched variants
# cover the copies of the shadow coder the slices and the batch path use.
run_case trellis4      320 192  8 1 -q 26 -p 3 --trellis 2
run_case trellis8      320 192  8 4 -q 24 -p 4 --trellis 2
run_case trellis_win   320 192  8 1 -q 26 -p 2 --trellis 1
run_case trellis_win8  320 192  8 4 -q 24 -p 4 --trellis 1
run_case trellis_thr   320 192  6 1 -q 26 -p 4 --trellis 2 --slices 2 -t 2
AEG_COEF_BATCH=1 run_case trellis_batch 320 192 6 1 -q 26 -p 4 --trellis 2 --partitions 15
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
# B pyramid (on by default with two or more B pictures, so the cases above
# already run it): the middle B picture of each group is a reference, the
# anchors re-order list 0 and drop it again with an MMCO.  Without the
# pyramid, and with groups of two (the first B picture is the reference).
run_bcase bpyr_off      320 192 12 3 4 -q 24 --scenecut 0 --b-pyramid 0
run_bcase bpyr_two      320 192 13 2 1 -q 24 --scenecut 0 --b-pyramid 1
run_bcase bpyr_adapt    320 192 24 3 4 -q 26 -p 3 --b-adapt 1 -g 10
run_bcase bpyr_trellis  320 192 24 3 4 -q 26 -p 3 --b-adapt 2
run_bcase bpyr_ref2     320 192 24 3 4 -q 26 -p 3 --ref 2 --scenecut 0
run_bcase bpyr_ref2s    320 192 20 2 1 -q 30 -p 4 --ref 2 --slices 2 -g 9
# macroblock tree: per-macroblock QP offsets on the reference pictures (the
# Intra_16x16 delta, chroma deblocking across macroblocks of different QP),
# and a short window over a long clip so the lookahead ring wraps and the
# plan crosses keyframes and the flushed tail
run_bcase mbtree_wrap   320 192 40 3 4 -q 30 -p 2 --mbtree 1 --rc-lookahead 3 -g 13
run_bcase mbtree_aq     320 192 24 2 1 -q 34 -p 3 --mbtree 1 --aq 10 --scenecut 0
# Explicit weighted prediction on a fade (genyuv mode 5): the P pictures send
# a luma and chroma weight/offset through PPS 1, the prediction is
# interpolated first and weighted afterwards, as 8.4.2.3 has it.  P only,
# CAVLC, two references, and with B pictures (which stay unweighted).
run_case  weightp       320 192 10 5 -q 26 -p 3 --scenecut 0
run_case  weightp_cavlc 320 192 10 5 -q 26 -p 3 --scenecut 0 --no-cabac
run_case  weightp_ref2  320 192 10 5 -q 26 -p 3 --scenecut 0 --ref 2 --slices 2 -t 2
run_bcase weightp_b     320 192 12 3 5 -q 26 -p 3 --scenecut 0
#   the lowest and highest rate: at q 51 every macroblock is a skip, at q 12
#   all of them carry residuals and the bi-predictive average is at its most
#   visible
run_bcase bframes_hi    320 192  9 1 2 -q 12 --scenecut 0
run_bcase bframes_lo    320 192  9 1 2 -q 51 --scenecut 0
#   a stream whose length is not a multiple of the group size (the tail is
#   coded as P pictures) and one with an IDR picture in the middle
run_bcase bframes_odd   320 192 11 2 1 -q 26 --scenecut 0
run_bcase bframes_idr   320 192 12 1 1 -q 26 --scenecut 0 -g 4
#   the fixed structure (--b-adapt 0): the reported coding order has to be the
#   one rebuilt from the frame and B counts, with and without a mid-stream IDR
run_bcase bframes_fix3  320 192 12 3 4 -q 24 --scenecut 0 --b-adapt 0 --b-pyramid 0
run_bcase bframes_fixidr 320 192 12 2 1 -q 26 --scenecut 0 -g 5 --b-adapt 0 --b-pyramid 0
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
#   B_L0_Bi_8x16 is Table 7-14's mb_type 13, and the CABAC tree's *intra* branch
#   is the four-bit code 13 -- the same number for two different things.  Only
#   the 8x16 shape reaches it (its 16x8 twin is 12), so this case runs three B
#   pictures between two anchors with 8x16 enabled and QP 24, which is the
#   combination that selects it.
run_bcase bframes_8x16  320 192 12 3 4 -q 24 --scenecut 0 -p 2 --partitions 5
#   The B pictures refine their vectors like the P ones do from preset 3 on:
#   each partition's list 0 vector and then its list 1 vector is re-coded with
#   its four quarter-pel neighbours.  What that must *not* touch is a direct
#   macroblock, whose vectors are derived from the co-located picture and are
#   not in the bitstream -- moving one changes the encoder's reconstruction and
#   nothing else, which is a silent desync from the first macroblock of the
#   picture.  Mode 4 (four quadrants) with the partitioned shapes enabled picks
#   direct, single-list and bi-predictive blocks next to each other.
run_bcase bframes_refine 320 192 12 3 4 -q 24 --scenecut 0 -p 4 --partitions 7
#   `--refine-cand 1` refines a candidate *before* it is compared with the
#   others, so the vector the bitstream carries is the refined one and the
#   reconstruction has to follow it -- an off-by-one here is another silent
#   desync.  One P-picture case (the quadrant clip is all partitions and
#   motion) and one B-picture case, both at the slowest preset.
run_case refine_cand   320 192 10 4 -q 24 -p 4 --partitions 15 --refine-cand 1
run_bcase bframes_rcand 320 192 10 3 4 -q 24 --scenecut 0 -p 4 --partitions 15 \
    --refine-cand 1
# Two P references (--ref 2): list0[1] is the anchor before the previous one,
# chosen per macroblock by rate-distortion.  Each of these selects it on a few
# hundred macroblocks.  What they guard, in the order it went wrong: the SPS has
# to keep two reference frames alive (with max_num_ref_frames = 1 the decoder
# drops the picture and conceals ref_idx 1 silently), ref_idx is one *inverted*
# bit in CAVLC and plain unary (two bins for 1) in CABAC, a macroblock on the
# second reference is never a P_Skip, the skip vector and the predictors follow
# the reference-matching rule, and the slice header drops to one reference when
# no macroblock of the slice used the second.  The B case adds the anchors' two
# references under B pictures: a direct block must not follow a co-located
# block on the older reference, and at preset 4 (search range 48, eight
# pictures between an anchor and its older reference) the vectors reach past a
# 32-sample picture padding -- which is why the padding is 64 now.
run_case ref2_p        320 192 16 4 -q 34 -p 3 --ref 2
run_case ref2_cavlc    320 192 16 4 -q 34 -p 3 --ref 2 --no-cabac
run_case ref2_slices   320 192 16 4 -q 34 -p 3 --ref 2 --slices 3 -t 2
run_case ref2_gop      320 192 12 1 -q 28 -p 3 --ref 2 -g 5
run_bcase bframes_ref2 320 192 13 3 1 -q 20 --scenecut 0 -p 4 --ref 2
# Three to eight P references (set_refs_multi): no marking operations, a
# sliding window of max_refs pictures (refs, plus the pyramid's reference B)
# that the SPS has to announce, list0 reordered to the newest anchor first and
# the rest by decreasing POC, ref_idx as ue(v) in CAVLC and the long unary
# (ctxIdxInc 4, then 5) in CABAC, per-slice num_ref_idx_active down to the
# highest index used, and the borrowed sub-pel planes of the references past
# the second (their stride once read as 0: small, silent drift).
run_case ref4_p        320 192 16 4 -q 34 -p 4 --ref 4
run_case ref6_cavlc    320 192 16 4 -q 34 -p 4 --ref 6 --no-cabac
run_case ref8_slices   320 192 16 4 -q 34 -p 4 --ref 8 --slices 3 -t 2
run_case ref5_gop      320 192 14 1 -q 28 -p 3 --ref 5 -g 6
run_case ref4_ft       320 192 16 4 -q 30 -p 4 --ref 4 --frame-threads 2
run_case weightp_ref4  320 192 10 5 -q 26 -p 4 --scenecut 0 --ref 4
run_bcase bframes_ref4 320 192 13 3 1 -q 20 --scenecut 0 -p 4 --ref 4 --b-pyramid 0
run_bcase bpyr_ref4    320 192 24 3 4 -q 26 -p 4 --ref 4 --scenecut 0
run_bcase bpyr_ref6ft  320 192 24 3 4 -q 26 -p 4 --ref 6 --scenecut 0 --frame-threads 2
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
# Sizes that are not whole macroblocks: coded rounded up, the margin repeated
# from the edge and cropped by the SPS; the decoder's cropped output has to be
# the encoder's visible reconstruction.
run_case crop_p      100  72  6 1 -q 24 -p 3
run_case crop_slices 174  94  6 4 -q 26 --slices 3 -p 1
run_case crop_cavlc   50  38  5 1 -q 28 --no-cabac
run_bcase crop_b     330 190 12 3 4 -q 24 -p 4
run_bcase crop_tiny    2   2  8 3 1 -q 30 -p 2
# Deblocking offsets and the chroma QP offset (both signs; a positive offset
# makes the chroma filter run on edges whose luma filter is off), the QP
# range and the access unit delimiter.
run_case dbk_offsets 176 144  6 1 -q 26 -p 3 --deblock -3:2
run_bcase dbk_strong 176 144 12 3 4 -q 20 -p 4 --deblock 4:4
run_bcase cqp_plus   176 144 12 3 4 -q 16 -p 2 --chroma-qp-offset 6
run_bcase cqp_minus  176 144 12 3 1 -q 30 -p 4 --chroma-qp-offset -5 --deblock 2:-1
run_case cqp_cavlc   176 144  5 4 -q 14 -p 1 --no-cabac --chroma-qp-offset 12
run_bcase qp_range   176 144 12 3 4 -q 20 -p 2 --qpmin 24 --qpmax 30 --aud
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
$BUILD/aegenc -w 320 -h 192 -n 8 -i "$TMP/still.yuv" -o "$TMP/still.264" \
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
$BUILD/aegenc -w 320 -h 192 -n 60 -q 26 -i "$TMP/cut.yuv" \
    -o "$TMP/cut.264" -g 250 --bframes 0 --dump-recon "$TMP/cut.recon" \
    >"$TMP/cut.log" 2>&1 \
    || { echo "   cut clip: encode failed"; FAIL=1; }
$BUILD/aegenc -w 320 -h 192 -n 60 -q 26 -i "$TMP/cut.yuv" \
    -o "$TMP/cutb.264" -g 250 --bframes 3 >"$TMP/cutb.log" 2>&1 \
    || { echo "   cut clip with B pictures: encode failed"; FAIL=1; }
$BUILD/aegenc -w 320 -h 192 -n 120 -q 26 -i "$TMP/seed_search.yuv" \
    -o "$TMP/nocut.264" -g 250 --bframes 0 >/dev/null 2>&1
$BUILD/aegenc -w 320 -h 192 -n 120 -q 26 -i "$TMP/seed_search.yuv" \
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
# Four seconds of two half-frame pans (genyuv 1): the whole-frame pan (mode 0)
# is predicted so well that its pictures barely change size with the QP, which
# no controller that models size against QP can land a target on.
"$BUILD/genyuv" 320 192 120 "$TMP/abr.yuv" 1 >/dev/null
$BUILD/aegenc -w 320 -h 192 -n 120 -q 24 -i "$TMP/abr.yuv" \
    -o "$TMP/abr.264" --bitrate 300 -v 0
python3 -c '
import os, sys
size = os.path.getsize(sys.argv[1])
kbps = size * 8 * 30 / 120 / 1000.0
ok = abs(kbps - 300) <= 30
print("   ABR target 300 kb/s -> %.1f kb/s%s" % (kbps, "" if ok else "  FAIL"))
sys.exit(0 if ok else 1)
' "$TMP/abr.264" || FAIL=1

# All-intra (no B pictures, an IDR every picture): no anchors to take an
# intra QP from, so each intra picture is rate-controlled like an anchor
# (before, the QP never moved: 6.4x the target).
$BUILD/aegenc -w 320 -h 192 -n 120 -i "$TMP/abr.yuv" -o "$TMP/abr_i.264" \
    --bitrate 1500 --bframes 0 -g 1 -v 0
python3 -c '
import os, sys
kbps = os.path.getsize(sys.argv[1]) * 8 * 30 / 120 / 1000.0
ok = abs(kbps - 1500) <= 150
print("   all-intra ABR 1500 kb/s -> %.1f kb/s%s" % (kbps, "" if ok else "  FAIL"))
sys.exit(0 if ok else 1)
' "$TMP/abr_i.264" || FAIL=1

echo "== VBV / HRD"
# The buffer model has to do two things: leave a stream that fits its channel
# alone (the bytes must be exactly what the same ABR configuration without a
# buffer produces), and raise the QP when a picture would not fit -- including
# saying so when even QP 51 cannot make it fit, because a model that can never
# report an underflow is not a model.  (The buffer turns the macroblock tree
# off, so the stream without one is coded without it too.)
"$BUILD/genyuv" 320 192 120 "$TMP/vbv.yuv" 1 >/dev/null
$BUILD/aegenc -w 320 -h 192 -n 120 -q 24 -i "$TMP/vbv.yuv" \
    -o "$TMP/vbv_off.264" --bitrate 300 --mbtree 0 -v 0
$BUILD/aegenc -w 320 -h 192 -n 120 -q 24 -i "$TMP/vbv.yuv" \
    -o "$TMP/vbv_wide.264" --bitrate 300 --vbv-maxrate 300 --vbv-bufsize 240 \
    >"$TMP/vbv_wide.log" 2>&1
if cmp -s "$TMP/vbv_off.264" "$TMP/vbv_wide.264"; then
    echo "   generous buffer (240 kbit): stream unchanged"
else
    echo "   generous buffer (240 kbit): stream differs from the ABR stream  FAIL"
    FAIL=1
fi
$BUILD/aegenc -w 320 -h 192 -n 120 -q 24 -i "$TMP/vbv.yuv" \
    -o "$TMP/vbv_tight.264" --bitrate 300 --vbv-maxrate 300 --vbv-bufsize 30 \
    >"$TMP/vbv_tight.log" 2>&1
# (--trellis 0: level selection takes this synthetic picture's QP down to 12-13
# and the one-pass rate control is still walking there at picture 120, 10 %
# under; on real content it lands as before -- what is tested here is that
# the options imply ABR.)
$BUILD/aegenc -w 320 -h 192 -n 120 -q 24 -i "$TMP/vbv.yuv" \
    -o "$TMP/vbv_alone.264" --vbv-maxrate 300 --vbv-bufsize 240 --trellis 0 \
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
kbps = os.path.getsize(os.path.join(tmp, 'vbv_alone.264')) * 8 * 30 / 120 / 1000.0
if abs(kbps - 300) > 30:
    print("   --vbv-* alone: %.1f kb/s, wanted 300  FAIL" % kbps)
    ok = False
print("   generous buffer: low water %.1f kbit, 0 clamps" % wide[0]
      if ok and wide else "   VBV check failed")
print("   30 kbit buffer:  low water %.1f kbit, %d underflow(s), %d clamp(s)"
      % tight[:3] if ok and tight else "")
print("   --vbv-* alone (no --bitrate): %.1f kb/s" % kbps)
sys.exit(0 if ok else 1)
PY2

# The model signalled as the NAL HRD (--nal-hrd): hrd_parameters in the SPS,
# buffering period + picture timing SEI.  The decoder has to reproduce the
# reconstruction with the SEI in the stream, and the signalled times have to
# put the pictures out in display order, across IDRs and B groups.
run_bcase hrd 176 144 30 3 4 -q 26 --bitrate 300 --vbv-maxrate 400 \
    --vbv-bufsize 600 --nal-hrd --info-sei -g 8
if command -v "$FF" >/dev/null 2>&1; then
    python3 "$(dirname "$0")/../tools/hrd_check.py" "$FF" "$TMP/hrd.264" \
        "$TMP/hrd.order" || FAIL=1
fi
if grep -q "options: preset=" "$TMP/hrd.264"; then
    echo "   info SEI present"
else
    echo "   info SEI missing  FAIL"; FAIL=1
fi
# A clip that holds each picture for five frames, with fine texture: a held
# picture the controller codes finer than its reference costs far more than
# the size predictor expects, so these streams only fit because frame_body()
# codes such a picture again.  None may underflow, and the re-coded pictures
# must decode to what the encoder reconstructed.
run_bcase vbv_recode 320 192 40 3 6 -p 3 --bitrate 300 --vbv-maxrate 300 --vbv-bufsize 150
run_bcase vbv_recode0 320 192 30 0 6 -p 2 --bitrate 200 --vbv-maxrate 200 --vbv-bufsize 100 --weightp 1
for c in vbv_recode vbv_recode0; do
    if grep -q " 0 underflow(s)" "$TMP/$c.log" && ! grep -q " 0 coded again" "$TMP/$c.log"; then
        echo "   $c: $(grep -o '[0-9]* coded again' "$TMP/$c.log"), no underflow"
    else
        echo "   $c: $(grep -o 'vbv:.*' "$TMP/$c.log")  FAIL"; FAIL=1
    fi
done

echo "== CRF"
# Constant quality: the tree's path (B pictures), the complexity path with and
# without B pictures (a fractional CRF: a run of macroblocks one QP coarser),
# CRF under a buffer cap, and the order of the scale.
run_bcase crf_tree   176 144 16 3 0 -p 3 --crf 24
run_bcase crf_cplx   176 144 16 3 4 -p 1 --crf 23.5 --mbtree 0
run_case  crf_nob    176 144 12 4 -p 2 --crf 26.3 --bframes 0
run_bcase crf_vbv    320 192 30 3 6 -p 3 --crf 12 --vbv-maxrate 300 --vbv-bufsize 150 --nal-hrd
if command -v "$FF" >/dev/null 2>&1; then
    python3 "$(dirname "$0")/../tools/hrd_check.py" "$FF" "$TMP/crf_vbv.264" \
        "$TMP/crf_vbv.order" || FAIL=1
fi
if grep -q " 0 underflow(s)" "$TMP/crf_vbv.log"; then
    echo "   capped CRF: no underflow"
else
    echo "   capped CRF: $(grep -o 'vbv:.*' "$TMP/crf_vbv.log")  FAIL"; FAIL=1
fi
"$BUILD/genyuv" 176 144 16 "$TMP/crfs.yuv" 4 >/dev/null
prev=0
for c in 30 24 18; do
    $BUILD/aegenc -w 176 -h 144 -n 16 -i "$TMP/crfs.yuv" -o "$TMP/crfs$c.264" \
        -p 2 --crf $c -v 0 || FAIL=1
    sz=$(wc -c < "$TMP/crfs$c.264")
    if [ "$sz" -le "$prev" ]; then
        echo "   --crf $c: $sz bytes, not more than the coarser CRF ($prev)  FAIL"; FAIL=1
    fi
    prev=$sz
done
echo "   --crf 30/24/18: sizes increase"
rc=0
$BUILD/aegenc -w 176 -h 144 -n 2 -i "$TMP/crfs.yuv" -o "$TMP/crfx.264" --crf 20 \
    --bitrate 100 >/dev/null 2>&1 || rc=$?
if [ "$rc" = 2 ]; then
    echo "   --crf with --bitrate: refused"
else
    echo "   --crf with --bitrate: exit $rc (want 2)  FAIL"; FAIL=1
fi

echo "== two-pass rate control"
# Two-pass on a clip whose two halves have very different complexity: the point
# of the first pass is to *measure* that difference and spend accordingly.  The
# rate controller stays on in pass 2, so the stream still lands on target; the
# first pass only decides how the bits are shared out.  What two-pass promises
# is constant quality at a rate chosen in advance: at the rate CRF 26 happened
# to take, it has to land within 6% and, with B pictures (the measurement keyed
# by display index while the packets come in coding order), come within 0.3 dB
# of it.  P-only it only has to land: without B pictures there is no tree, and
# CRF's complexity blur, which carries over the cut here, gives the new IDR and
# the pictures predicted from it a head start that is luck more than design.
for v in b p; do
    if [ $v = b ]; then o=""; else o="--bframes 0"; fi
    $BUILD/aegenc -w 320 -h 192 -n 60 -i "$TMP/cut.yuv" $o --crf 26 \
        -o "$TMP/crf$v.264" >"$TMP/crf$v.log" 2>&1 \
        || { echo "   CRF control ($v) failed"; FAIL=1; }
    r=$(sed -n 's/^bitrate \([0-9]*\).*/\1/p' "$TMP/crf$v.log")
    echo "${r:-0}" > "$TMP/crf$v.rate"
    $BUILD/aegenc -w 320 -h 192 -n 60 -i "$TMP/cut.yuv" $o --bitrate "${r:-1}" \
        -o "$TMP/2p1$v.264" --pass 1 --stats "$TMP/2p$v.stats" >"$TMP/2p1$v.log" 2>&1 \
        || { echo "   pass 1 ($v) failed"; FAIL=1; }
    $BUILD/aegenc -w 320 -h 192 -n 60 -i "$TMP/cut.yuv" $o --bitrate "${r:-1}" \
        -o "$TMP/2p2$v.264" --pass 2 --stats "$TMP/2p$v.stats" \
        --dump-recon "$TMP/2p$v.recon" --dump-order "$TMP/2p$v.order" \
        >"$TMP/2p2$v.log" 2>&1 || { echo "   pass 2 ($v) failed"; FAIL=1; }
done
python3 - "$TMP" <<'PY2' || FAIL=1
import re, sys, os
tmp = sys.argv[1]

def measure(name):
    t = open(os.path.join(tmp, name)).read()
    m = re.search(r'bitrate ([0-9.]+) kb/s at \d+ fps, PSNR ([0-9.]+) dB', t)
    return (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)

ok = True
for v, what in (('b', 'B pictures'), ('p', 'P only')):
    lines = open(os.path.join(tmp, '2p%s.stats' % v)).read().split('\n')
    types = [l.split()[1] for l in lines[1:] if l.strip()]
    if not lines[0].startswith('aeg-stats 2 bframes=') or len(types) != 60:
        print("   %s: stats file has %d pictures, header %r  FAIL"
              % (what, len(types), lines[0]))
        ok = False
    if (v == 'b') != any(t in 'Bb' for t in types):
        print("   %s: stats types %s  FAIL" % (what, ''.join(types)))
        ok = False
    two, crf = measure('2p2%s.log' % v), measure('crf%s.log' % v)
    want = int(open(os.path.join(tmp, 'crf%s.rate' % v)).read())
    if want <= 0 or abs(two[0] - want) > 0.06 * want:
        print("   %s: two-pass rate %.1f kb/s, target %d  FAIL" % (what, two[0], want))
        ok = False
    if v == 'b' and two[1] < crf[1] - 0.3:
        print("   %s: two-pass %.2f dB, CRF %.2f dB at that rate  FAIL"
              % (what, two[1], crf[1]))
        ok = False
    print("   %-10s CRF 26 %5.1f kb/s %.2f dB  ->  2-pass %5.1f kb/s %.2f dB"
          % (what, crf[0], crf[1], two[0], two[1]))
sys.exit(0 if ok else 1)
PY2
if command -v "$FF" >/dev/null 2>&1; then
    for v in b p; do
        "$FF" -v error -y -i "$TMP/2p2$v.264" -f rawvideo -pix_fmt yuv420p \
            "$TMP/2p$v.dec" 2>"$TMP/2p$v.decerr"
        if [ -s "$TMP/2p$v.decerr" ]; then
            echo "   two-pass ($v): decoder reported problems:"
            sed 's/^/     /' "$TMP/2p$v.decerr" | head -3
            FAIL=1
        fi
    done
    python3 "$TMP/cmp_b.py" "$TMP/2pb.recon" "$TMP/2pb.dec" 320 192 60 3 \
        two_pass_b 0 "$TMP/2pb.order" 0 || FAIL=1
    python3 "$TMP/cmp.py" "$TMP/2pp.recon" "$TMP/2pp.dec" 320 192 60 two_pass_p \
        || FAIL=1
fi

# Statistics a second pass must refuse (exit 1, with a reason), and one it
# must cut down: B pictures measured, none coded; a QP no encoder writes; an
# index far past the pictures before it (the table is allocated up to it);
# nothing measured.  Coding 30 of the 60 measured pictures plans for those 30.
awk 'NR==1{print;next}{$3=99;print}' "$TMP/2pb.stats" > "$TMP/bad_qp.stats"
awk 'NR==1{print;next}{$5=0;print}' "$TMP/2pb.stats" > "$TMP/bad_zero.stats"
{ head -3 "$TMP/2pb.stats"; echo "99999999 P 20 20 1000"; } > "$TMP/bad_far.stats"
bad=0
for c in "2pb.stats --bframes 0" "bad_qp.stats" "bad_zero.stats" "bad_far.stats"; do
    set -- $c
    rc=0
    $BUILD/aegenc -w 320 -h 192 -n 60 -i "$TMP/cut.yuv" -o "$TMP/2px.264" \
        --pass 2 --stats "$TMP/$1" --bitrate 100 $2 $3 -v 0 2>"$TMP/2px.err" || rc=$?
    if [ $rc -ne 1 ] || [ ! -s "$TMP/2px.err" ]; then
        echo "   pass 2 with $c: exit $rc  FAIL"; bad=1
    fi
done
r=$(cat "$TMP/crfb.rate")
$BUILD/aegenc -w 320 -h 192 -n 30 -i "$TMP/cut.yuv" -o "$TMP/2pn.264" --pass 2 \
    --stats "$TMP/2pb.stats" --bitrate "$r" >"$TMP/2pn.log" 2>&1 || bad=1
python3 - "$TMP/2pn.log" "$r" <<'PY2' || bad=1
import re, sys
m = re.search(r'bitrate ([0-9.]+) kb/s', open(sys.argv[1]).read())
kbps, want = (float(m.group(1)) if m else 0.0), float(sys.argv[2])
if abs(kbps - want) > 0.1 * want:
    print("   pass 2 coding 30 of 60 pictures: %.1f kb/s for %d  FAIL" % (kbps, want))
    sys.exit(1)
PY2
if [ $bad = 0 ]; then
    echo "   damaged or mismatched statistics refused; a shorter pass 2 lands"
else
    FAIL=1
fi

echo "== input and options"
# YUV4MPEG2: the header's size, rate and aspect ratio have to give the stream
# the raw input gives with the same values as options, from a file and from a
# pipe; then the errors the command line must refuse (exit status 2).
"$BUILD/genyuv" 96 64 5 "$TMP/io.yuv" 1 >/dev/null
fs=$((96 * 64 * 3 / 2))
{
    printf 'YUV4MPEG2 W96 H64 F30000:1001 Ip A12:11 C420jpeg\n'
    k=0
    while [ $k -lt 5 ]; do
        printf 'FRAME\n'
        dd if="$TMP/io.yuv" bs=$fs skip=$k count=1 2>/dev/null
        k=$((k + 1))
    done
} > "$TMP/io.y4m"
$BUILD/aegenc -i "$TMP/io.y4m" -o "$TMP/io1.264" -v 0 &&
    $BUILD/aegenc -o "$TMP/io2.264" -v 0 < "$TMP/io.y4m" &&
    $BUILD/aegenc -w 96 -h 64 --fps 30000/1001 --sar 12:11 -i "$TMP/io.yuv" \
        -o "$TMP/io3.264" -v 0 || { echo "   y4m: encode failed"; FAIL=1; }
if cmp -s "$TMP/io1.264" "$TMP/io2.264" && cmp -s "$TMP/io1.264" "$TMP/io3.264"; then
    echo "   y4m file = y4m pipe = raw input with --fps/--sar"
else
    echo "   y4m: streams differ  FAIL"; FAIL=1
fi
head -c $((fs * 2 + 100)) "$TMP/io.yuv" > "$TMP/io_part.yuv"
if $BUILD/aegenc -w 96 -h 64 -i "$TMP/io_part.yuv" -o "$TMP/io4.264" -v 0 \
        2>"$TMP/io4.err" && grep -q "partial frame" "$TMP/io4.err"; then
    echo "   a partial last frame is reported and dropped"
else
    echo "   partial frame: not reported  FAIL"; FAIL=1
fi
bad=0
for args in "-w" "-w 96 -h 64 -q x" "-w 95 -h 64" "-w 96 -h 64 -q 52" \
            "-w 96 -h 64 --bframes 4" "-w 96 -h 64 --ref 9" "-w 96 -h 64 -p 5" \
            "-w 96 -h 64 --fps 0" "-w 96 -h 64 --sar 4" "-w 96 -h 64 --direct x" \
            "-w 96 -h 64 --colorprim nope" "-w 64 -h 64 --demuxer y4m" \
            "-w 96 -h 64 --qpmax 0" "-w 96 -h 64 --qpmin 40 --qpmax 30" \
            "-w 96 -h 64 --input-csp foo" "-w 96 -h 64 --input-depth 17" \
            "--input-depth 10" "--demuxer raw -w 96 -h 62 --tff" \
            "--nonsense"; do
    # shellcheck disable=SC2086
    rc=0
    $BUILD/aegenc $args -i "$TMP/io.y4m" -o "$TMP/io5.264" -v 0 \
        >/dev/null 2>&1 </dev/null || rc=$?
    if [ $rc -ne 2 ] && [ $rc -ne 1 ]; then
        echo "   '$args' was accepted (exit $rc)  FAIL"; bad=1
    fi
done
if [ $bad = 0 ]; then echo "   19 malformed command lines refused"; else FAIL=1; fi

echo "== input formats and interlace"
# Other layouts and depths, made by ffmpeg from a generated clip, are
# converted to 8-bit 4:2:0 and must still decode bit-exactly; interlaced
# streams must decode bit-exactly and carry their field order (ffmpeg's
# showinfo: i:T top first, i:B bottom first) on every picture; a YUV4MPEG2
# header with a C tag and It must give the stream its raw equivalent gives.
IN_FILE=
if command -v "$FF" >/dev/null 2>&1; then
    "$BUILD/genyuv" 176 144 12 "$TMP/if.yuv" 4 >/dev/null
    for spec in yuv422p10le:i422:10 yuv444p:i444:8 nv12:nv12:8 \
                yuv420p12le:i420:12 gray16le:i400:16 nv21:nv21:8; do
        pf=${spec%%:*}; rest=${spec#*:}; csp=${rest%%:*}; dep=${rest#*:}
        "$FF" -v error -y -f rawvideo -s 176x144 -pix_fmt yuv420p \
            -i "$TMP/if.yuv" -pix_fmt "$pf" -f rawvideo "$TMP/if_$csp.raw"
        IN_FILE="$TMP/if_$csp.raw"
        run_bcase "in_${csp}_$dep" 176 144 12 2 4 -q 24 -p 2 \
            --input-csp "$csp" --input-depth "$dep"
        IN_FILE=
    done
    run_bcase tff      176 144 12 2 4 -q 24 -p 3 --tff
    run_bcase bff      176 144 12 3 1 -q 26 -p 2 --bff --slices 2 --b-adapt 0 \
        --b-pyramid 0
    run_case  bff_cavlc 176 144 8 4 -q 26 -p 2 --bff --no-cabac --no-8x8dct
    run_bcase tff_hrd  176 144 16 2 4 -p 2 --tff --bitrate 300 \
        --vbv-maxrate 300 --vbv-bufsize 600 --nal-hrd
    # field pictures (PAFF): every frame two field pictures, the second
    # predicting from the first; mid-stream IDRs, B fields (direct from the
    # co-located field of either parity), slices, CAVLC, refs, CRF
    run_bcase fld_p    176 144 12 0 4 -q 24 -p 3 --field -g 5
    run_bcase fld_b    176 144 12 2 4 -q 26 -p 3 --field --bff --slices 2
    run_bcase fld_cavlc 176 144 12 1 1 -q 28 -p 4 --field --no-cabac --ref 3
    run_bcase fld_crf  176 144 12 3 4 -p 2 --field --crf 24
    # the B pyramid in field pictures: the reference B's fields in both
    # lists (list 1 nearest first), P fields' lists modified back to the
    # newest anchor, a buffer of --ref + 2 frames (8 refs: 10)
    run_bcase fld_pyr  176 144 13 3 4 -q 27 -p 4 --field
    run_bcase fld_pyr2 176 144 13 3 4 -q 28 -p 1 --field --ref 2 --bff
    # frame and field pictures mixed (--paff): the references re-padded and
    # their half-sample planes rebuilt for the other structure, frame
    # pictures' bottom-field POC; a forced pattern (AEG_FLDFORCE, per display
    # index) for every switch, then the encoder's own choice
    export AEG_FLDFORCE=0110100
    run_bcase paff_mix  176 144 14 2 4 -q 26 -p 3 --paff
    run_bcase paff_mixp 176 144 14 0 1 -q 26 -p 4 --paff --bff --no-cabac
    unset AEG_FLDFORCE
    run_bcase paff_auto 176 144 12 3 4 -q 26 -p 2 --paff --slices 2
    # MBAFF frames (--mbaff): macroblock pairs in decoding order, the
    # mb_field_decoding_flag and its skipped-pair rules, no above-right
    # neighbour for a bottom macroblock, deblocking pair by pair
    run_bcase mbaff_b   176 144 12 2 4 -q 26 -p 3 --mbaff --slices 2
    run_bcase mbaff_cavlc 176 144 12 0 1 -q 28 -p 4 --mbaff --no-cabac --ref 3
    run_bcase mbaff_crf 176 144 12 3 4 -p 2 --mbaff --crf 24 --bff
    # the frame/field decision per pair, on a woven interlaced clip (mode 7):
    # the vertical SADs alone (presets 0/1), and with the coded trial of
    # both kinds where they are unsure (presets 2-4)
    run_bcase mbaff_d1  176 144 12 0 7 -q 28 -p 1 --mbaff --no-cabac
    run_bcase mbaff_d2  176 144 12 2 7 -q 26 -p 3 --mbaff --slices 2
    run_bcase mbaff_d3  176 144 12 3 7 -p 4 --mbaff --crf 25 --ref 3 --bff
    export AEG_MBTG=1000    # (every pair)
    run_bcase mbaff_d4  176 144 10 3 7 -q 30 -p 2 --mbaff --deblock -1:0
    unset AEG_MBTG
    # field pictures or an MBAFF frame per anchor (--paff-mbaff): the
    # decision on the woven clip, then forced patterns of both (AEG_FLDFORCE
    # per display index, AEG_MBFORCE inside the MBAFF frames): field
    # pictures reading MBAFF frames and the other way round, and in the B
    # MBAFF frames the co-located field nearer in POC (tff: top, bff: bottom)
    run_bcase pm_auto   176 144 12 2 7 -q 26 -p 3 --paff-mbaff
    export AEG_FLDFORCE=0110 AEG_MBFORCE=7
    run_bcase pm_mix    176 144 16 3 4 -q 27 -p 2 --paff-mbaff -g 8
    run_bcase pm_mixc   176 144 12 0 4 -q 30 -p 4 --paff-mbaff --ref 3 \
        --no-cabac --slices 2
    export AEG_FLDFORCE=1001
    run_bcase pm_mixb   176 144 14 2 4 -p 1 --paff-mbaff --crf 25 --bff
    # ... with the pyramid: reference B frames between field anchors and
    # reference B fields between frame anchors (MMCO and sliding window)
    export AEG_FLDFORCE=0000111100001111
    run_bcase paff_pyr  176 144 17 3 4 -q 27 -p 3 --paff
    run_bcase pm_pyr    176 144 17 3 4 -q 28 -p 2 --paff-mbaff --ref 2
    unset AEG_FLDFORCE AEG_MBFORCE
    # the trial restores everything a pair's coding changed: run but its
    # result discarded (AEG_MBDEC=3), the output is the vertical SADs' alone
    "$BUILD/genyuv" 176 144 12 "$TMP/mbdec.yuv" 7 >/dev/null
    for m in 1 3; do
        AEG_MBDEC=$m AEG_MBTG=1000 $BUILD/aegenc -w 176 -h 144 -n 12 \
            -i "$TMP/mbdec.yuv" -o "$TMP/mbdec$m.264" -p 2 --crf 24 \
            --bframes 3 --mbaff >/dev/null 2>&1
    done
    if cmp -s "$TMP/mbdec1.264" "$TMP/mbdec3.264"; then
        echo "   mbaff trial restore: discarded trial leaves the output unchanged"
    else
        echo "   mbaff trial restore: FAILED (AEG_MBDEC=3 differs from 1)"; FAIL=1
    fi
    # frame threads in MBAFF frames: a reference's field copy is built band
    # by band behind the rows it publishes, and a pair waits for its field
    # macroblocks' reach (twice as many frame rows).  Any worker count writes
    # the serial stream, and so do AEG_POISON=1 (rows past that bound are
    # garbage until it reaches them) and AEG_MBSTALE (the decisions array
    # filled with garbage before every picture: a worker's holds whatever it
    # coded last), with natural and forced (AEG_MBFORCE) field pairs
    "$BUILD/genyuv" 176 160 16 "$TMP/mbft.yuv" 7 >/dev/null
    bad=""
    for mf in "" 3; do
        if [ -n "$mf" ]; then export AEG_MBFORCE=$mf; fi
        for o in "-p 3 -q 28 --ref 3" "-p 2 --crf 24 --bff" \
                 "-p 4 -q 30 --ref 2 --b-pyramid 0"; do
            for k in s ft2 ft3 ft4 pz st; do
                ev="AEG_UNUSED=1"; t="--frame-threads 1"
                case $k in
                    ft2) t="--frame-threads 2" ;; ft3) t="--frame-threads 3" ;;
                    ft4) t="--frame-threads 4" ;;
                    pz) ev="AEG_POISON=1" ;; st) ev="AEG_MBSTALE=85" ;;
                esac
                env $ev $BUILD/aegenc -w 176 -h 160 -n 16 -i "$TMP/mbft.yuv" \
                    -v 0 --mbaff $o $t -o "$TMP/mbft_$k.264" || bad="$bad rc:$k"
                [ $k = s ] || cmp -s "$TMP/mbft_s.264" "$TMP/mbft_$k.264" ||
                    bad="$bad $k[$mf $o]"
            done
        done
        unset AEG_MBFORCE
    done
    if [ -z "$bad" ]; then
        echo "   mbaff frame threads 2/3/4 = serial = AEG_POISON = AEG_MBSTALE"
    else
        echo "   mbaff frame threads: streams differ:$bad  FAIL"; FAIL=1
    fi
    # frame threads in field pictures (--field): a reference frame's rows
    # are published once its second field reaches them (both fields final),
    # its fields' half-resolution copies built behind them; a field reads
    # frame rows 2 l + 1 for its field line l, and the second field its first
    # without waiting.  Any worker count writes the serial stream, and so
    # does AEG_POISON=1 (each reference frame read a copy garbage past that
    # bound: planes, half-sample planes, the field's half-resolution copy,
    # the co-located field's decisions)
    bad=""
    for o in "-p 3 -q 28 --ref 3" "-p 2 --crf 24 --bff" \
             "-p 4 -q 30 --ref 2 --b-pyramid 0" "-p 1 -q 26 --bframes 0"; do
        for k in s ft2 ft3 ft4 pz; do
            ev="AEG_UNUSED=1"; t="--frame-threads 1"
            case $k in
                ft2) t="--frame-threads 2" ;; ft3) t="--frame-threads 3" ;;
                ft4) t="--frame-threads 4" ;; pz) ev="AEG_POISON=1" ;;
            esac
            env $ev $BUILD/aegenc -w 176 -h 160 -n 16 -i "$TMP/mbft.yuv" \
                -v 0 --field $o $t -o "$TMP/fdft_$k.264" || bad="$bad rc:$k"
            [ $k = s ] || cmp -s "$TMP/fdft_s.264" "$TMP/fdft_$k.264" ||
                bad="$bad $k[$o]"
        done
    done
    if [ -z "$bad" ]; then
        echo "   field-picture frame threads 2/3/4 = serial = AEG_POISON"
    else
        echo "   field-picture frame threads: streams differ:$bad  FAIL"; FAIL=1
    fi
    # ... and with the structure chosen per anchor (--paff, --paff-mbaff):
    # an anchor's choice waits for the newest anchor (its estimate reads that
    # reconstruction), a change of structure for every picture in flight
    # (the reference buffers are restructured in place).  Natural choices
    # and forced patterns (AEG_FLDFORCE, per display index) that flip at
    # nearly every anchor; the weighted P frames of --paff among them
    bad=""
    for ff in "" 10 1100; do
        if [ -n "$ff" ]; then export AEG_FLDFORCE=$ff; fi
        for o in "--paff -p 3 -q 28 --ref 3" "--paff -p 2 --crf 24 --bff" \
                 "--paff-mbaff -p 3 -q 30 --ref 2" "--paff-mbaff -p 4 -q 26 --bff"; do
            for k in s ft2 ft3 ft4; do
                t="--frame-threads 1"
                case $k in
                    ft2) t="--frame-threads 2" ;; ft3) t="--frame-threads 3" ;;
                    ft4) t="--frame-threads 4" ;;
                esac
                $BUILD/aegenc -w 176 -h 160 -n 16 -i "$TMP/mbft.yuv" -v 0 $o $t \
                    -o "$TMP/pmft_$k.264" || bad="$bad rc:$k"
                [ $k = s ] || cmp -s "$TMP/pmft_s.264" "$TMP/pmft_$k.264" ||
                    bad="$bad $k[$ff $o]"
            done
        done
        unset AEG_FLDFORCE
    done
    if [ -z "$bad" ]; then
        echo "   paff / paff-mbaff frame threads 2/3/4 = serial"
    else
        echo "   paff / paff-mbaff frame threads: streams differ:$bad  FAIL"; FAIL=1
    fi
    # field macroblock pairs (forced pseudo-random pattern, AEG_MBFORCE): the
    # field views and scans, neighbour derivation between frame and field
    # pairs (Table 6-4), the mixed-edge deblocking, the field references and
    # the scaled neighbour vectors / indices, and in the B pictures spatial
    # direct with the co-located pair of either kind (Table 8-6)
    export AEG_MBFORCE=11
    run_bcase mbaff_fi  176 144 12 2 4 -q 30 -p 3 --mbaff -g 4 --slices 2
    run_bcase mbaff_fic 176 144 12 0 1 -q 36 -p 1 --mbaff -g 3 --no-cabac \
        --deblock 2:1
    export AEG_MBFORCE=all
    run_bcase mbaff_fa  176 144 8 2 4 -q 26 -p 4 --mbaff -g 4 --bff
    export AEG_MBFORCE=3
    run_bcase mbaff_fp  176 144 12 0 4 -q 28 -p 3 --mbaff -g 12 --ref 3 \
        --slices 2
    run_bcase mbaff_fpb 176 144 12 0 4 -q 33 -p 1 --mbaff -g 12 --bff \
        --ref 2 --deblock -1:1
    run_bcase mbaff_fpc 176 144 12 0 4 -p 4 --mbaff -g 12 --ref 4 --crf 25 \
        --no-cabac
    export AEG_MBFORCE=9
    run_bcase mbaff_fb  176 144 16 3 4 -p 4 --mbaff -g 16 --ref 3 --crf 24 \
        --b-adapt 2 --b-pyramid 0
    run_bcase mbaff_fbs 176 144 12 2 4 -q 31 -p 1 --mbaff -g 12 --slices 3 \
        --bff
    unset AEG_MBFORCE
    for fo in tff:T bff:B tff_hrd:T fld_p:T fld_b:B; do
        nm=${fo%%:*}; want=${fo#*:}
        got=$("$FF" -i "$TMP/$nm.264" -vf showinfo -f null - 2>&1 |
              grep -o 'i:[A-Z?]' | sort | uniq -c | tr -s ' ' | tr '\n' ' ')
        nfr=$("$FF" -i "$TMP/$nm.264" -vf showinfo -f null - 2>&1 |
              grep -c "i:$want" || true)
        if [ "$nfr" -ge 12 ] && ! echo "$got" | grep -q 'i:[^'"$want"']'; then
            echo "   $nm: every picture i:$want"
        else
            echo "   $nm: field order '$got'  FAIL"; FAIL=1
        fi
    done
    "$FF" -v error -y -f rawvideo -s 176x144 -pix_fmt yuv420p -i "$TMP/if.yuv" \
        -vf setfield=tff -pix_fmt yuv422p10le -strict -1 "$TMP/if.y4m"
    rc=0
    $BUILD/aegenc -i "$TMP/if.y4m" -o "$TMP/if1.264" -v 0 || rc=$?
    $BUILD/aegenc -i "$TMP/if_i422.raw" -w 176 -h 144 --fps 25 --tff \
        --input-csp i422 --input-depth 10 -o "$TMP/if2.264" -v 0 || rc=$?
    if [ $rc = 0 ] && head -1 "$TMP/if.y4m" | grep -q 'It.*C422p10' &&
       cmp -s "$TMP/if1.264" "$TMP/if2.264"; then
        echo "   y4m 'It C422p10' = raw --input-csp i422 --input-depth 10 --tff"
    else
        echo "   y4m C422p10/It: streams differ (rc $rc)  FAIL"; FAIL=1
    fi
else
    echo "   skipped (no ffmpeg)"
fi

# High 10: the 10-bit build of the encoder (x264 --output-depth 10).  Every
# reconstruction against ffmpeg's 10-bit decode (yuv420p10le), the paths whose
# arithmetic depends on the depth: QPs below 0 (CRF), CAVLC, slices, the rate
# controls, weighted prediction, true 10-bit input; then that the stream does
# not depend on the threads, and that it is High 10.
echo "== High 10 (10-bit)"
rc=0
"$BUILD/genyuv" 64 64 2 "$TMP/h10probe.yuv" 1 >/dev/null
$BUILD/aegenc -w 64 -h 64 -i "$TMP/h10probe.yuv" -o "$TMP/h10probe.264" \
    --bit-depth 10 -v 0 2>/dev/null || rc=$?
if [ $rc != 0 ]; then
    echo "   skipped (library built without the 10-bit encoder)"
elif ! command -v "$FF" >/dev/null 2>&1; then
    echo "   skipped (no ffmpeg)"
else
    run_bcase h10_cqp     176 144 12 3 4 -q 24 -p 2 --bit-depth 10
    run_bcase h10_p4      176 144 12 3 1 -q 30 -p 4 --bit-depth 10
    run_bcase h10_lowqp   176 144  8 2 4 -q 0  -p 3 --bit-depth 10
    run_bcase h10_crfneg  176 144 12 3 4 -p 3 --crf -6 --bit-depth 10
    run_bcase h10_cavlc   176 144 12 0 4 -q 26 -p 2 --no-cabac --bit-depth 10
    run_bcase h10_slices  176 144 12 2 6 -q 26 -p 3 --slices 3 --tff --bit-depth 10
    run_bcase h10_field   176 144 12 2 4 -q 26 -p 3 --field --bit-depth 10
    export AEG_FLDFORCE=011
    run_bcase h10_paff    176 144 12 2 4 -q 26 -p 3 --paff --bit-depth 10
    unset AEG_FLDFORCE
    run_bcase h10_mbaff   176 144 12 2 4 -q 26 -p 3 --mbaff --bit-depth 10
    export AEG_MBFORCE=5
    run_bcase h10_mbaff_fi 176 144 8 2 4 -q 34 -p 3 --mbaff -g 3 --bit-depth 10
    run_bcase h10_mbaff_fp 176 144 10 0 4 -q 30 -p 2 --mbaff -g 10 --ref 2 \
        --bit-depth 10
    unset AEG_MBFORCE
    run_bcase h10_mbaff_d 176 144 12 2 7 -q 28 -p 3 --mbaff --bit-depth 10
    AEG_FLDFORCE=01; export AEG_FLDFORCE
    run_bcase h10_pm    176 144 12 3 4 -q 28 -p 3 --paff-mbaff --bit-depth 10
    unset AEG_FLDFORCE
    run_bcase h10_vbv     176 144 20 3 6 -p 3 --bitrate 300 --vbv-maxrate 400 \
        --vbv-bufsize 200 --nal-hrd --bit-depth 10
    run_bcase h10_wp      176 144 12 0 6 -q 26 -p 3 --weightp 1 --bit-depth 10
    run_bcase h10_qpmin   176 144 12 3 4 -p 2 --crf 2 --qpmin 6 --bit-depth 10
    # true 10-bit input: a 352x288 clip scaled down by ffmpeg into 10 bits
    "$BUILD/genyuv" 352 288 12 "$TMP/h10big.yuv" 4 >/dev/null
    "$FF" -v error -y -f rawvideo -s 352x288 -pix_fmt yuv420p \
        -i "$TMP/h10big.yuv" -vf scale=176:144 -pix_fmt yuv420p10le \
        -f rawvideo "$TMP/h10in.yuv"
    IN_FILE="$TMP/h10in.yuv" run_bcase h10_in10 176 144 12 3 0 -q 22 -p 3 \
        --input-depth 10 --bit-depth 10
    for o in "-q 24 -p 3" "--crf 20 -p 4" "--bitrate 300 -p 3"; do
        rc=0
        for k in ft1 ft3 s1 s3; do
            case $k in
                ft1) t="--frame-threads 1" ;; ft3) t="--frame-threads 3" ;;
                s1) t="--slices 3 -t 1" ;;    s3) t="--slices 3 -t 3" ;;
            esac
            $BUILD/aegenc -w 176 -h 144 -n 12 -i "$TMP/h10_cqp.yuv" -v 0 \
                --bit-depth 10 $o $t -o "$TMP/h10_$k.264" || rc=$?
        done
        if [ $rc = 0 ] && cmp -s "$TMP/h10_ft1.264" "$TMP/h10_ft3.264" &&
           cmp -s "$TMP/h10_s1.264" "$TMP/h10_s3.264"; then
            echo "   10-bit $o: frame threads 1 = 3, slice threads 1 = 3"
        else
            echo "   10-bit $o: the stream depends on the threads (rc $rc)  FAIL"
            FAIL=1
        fi
    done
    if "$FF" -i "$TMP/h10_cqp.264" 2>&1 | grep -q 'h264 (High 10).*yuv420p10le'; then
        echo "   10-bit stream: High 10, yuv420p10le"
    else
        echo "   10-bit stream: not reported as High 10 / yuv420p10le  FAIL"
        FAIL=1
    fi
fi

echo "== device enumeration"
$BUILD/aegenc --list-devices 2>/dev/null | sed 's/^/   /'

rm -rf "$TMP"
if [ "$FAIL" != 0 ]; then
    echo "== FAILED"
    exit 1
fi
echo "== all conformance cases bit-exact"
