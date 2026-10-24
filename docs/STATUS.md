# Status and engineering notes

Date: 2026-10-25.  Branch: `arena/589d1e45-acceleratedencode-gpu`.

## Where the project stands

The encoder is **conformant on I *and* P pictures**: an independent decoder
(FFmpeg 7.0.2) reproduces the encoder's reconstruction *bit for bit*
(`max abs diff 0`) on every frame, not only the IDR.

| content | sizes | QP | presets | frames |
|---|---|---|---|---|
| 1 MB probe, grids, smooth gradients | 16x16 … 320x192 | 10 … 51 | 0 … 4 | 1 … 40 |
| moving synthetic texture (inter, heavy AC) | 320x192, 320x176, 320x208 | 12, 18, 23, 26, 28, 32, 40 | 0 … 4 | 1 … 60 |
| translated noise / gradient clips | 96x96 | 23, 32 | 0, 2 | 3 |
| odd frame sizes (13 MB rows, 11 MB rows) | 320x208, 320x176 | 23 … 28 | 2, 3 | 3 … 4 |
| Intra_4x4 probes (random 16x16, 32x16 gradient) | 16x16, 32x16 | 24 … 33 | 2, 4 | 1 … 2 |

with and without the in-loop deblocking filter.  `make test` asserts this and
fails (non-zero exit) on any mismatch; it checks all nine clips frame by frame.

Implemented and building without warnings:

* bit writer with emulation prevention, NAL/SPS/PPS/slice headers, IDR +
  `frame_num` handling (IDR resets `frame_num`, `log2_max_frame_num_minus4`
  sized to the GOP)
* CAVLC residual coding (coeff_token, levels, total_zeros, run_before) —
  round-tripped by `tests/cavlc_check.c` (12 000 random blocks, 4 block kinds)
* forward/inverse 4x4 integer transform, 4x4 and 2x2 DC Hadamard, quantisation
  and dequantisation, including the reference decoder's `int16_t` truncation
* Intra_16x16 (all four modes), Intra_4x4 (all nine modes, the Figure 6-8
  block order, `prev_intra4x4_pred_mode_flag`/`rem_intra4x4_pred_mode` and the
  per-block mode decision), chroma (all four modes)
* half/quarter-pel motion estimation and compensation, MV prediction, skip runs
* P inter partitioning: 16x16, 16x8 and 8x16, each partition with its own
  motion search, its own MV predictor and its own `mvd` (8x8 sub-partitions are
  still missing)
* P_Skip macroblocks (`mb_skip_run`), including the run that reaches the end of
  the picture
* presets scale the integer search range (8/16/24/32/48) and the subpel
  refinement
* deblocking filter (all edges, normative order, alpha/beta/tc0 tables)
* CPU + OpenCL backends behind a measured dispatcher, device query, calibration
* CLI with `--list-devices`, `--calibrate`, `--bench`, `--dump-recon`

## Bugs found by the bit-exactness work (do not reintroduce)

1. **Transform transpose.**  The decoder's 4x4 inverse transform is the
   *transpose* of ours: a coefficient in slot (row 1, column 0) comes out as a
   horizontal ramp there and as a vertical one here.  `aeg_quant_4x4` therefore
   stores `coef[x*4 + y]` (the transposed block) and `aeg_dequant_4x4` /
   `aeg_idct_add_block` read `level[x*4 + y]`.  The per-position scale factors
   are symmetric in x and y, so the quantiser is unchanged.
2. **Intra_16x16 AC order.**  The sixteen AC blocks are transmitted in H.264
   block numbering (`i4x4 + 4*i8x8`, i.e. quadrant by quadrant), not in 4x4
   raster order; the neighbour contexts (`nC`) must follow the same numbering.
3. **Intra_16x16 luma DC context.**  The DC block is context-coded with the
   `nC` of luma block (0, 0) — the reference decoder computes
   `pred_non_zero_count(0)` for it (`n >= LUMA_DC_BLOCK_INDEX` maps to
   `(n - 48) * 16`).  Coding it with `nC = 0` desynchronises the first block of
   every macroblock that has neighbours.
4. **`mb_skip_run` in P slices.**  Every coded macroblock is preceded by
   `mb_skip_run`, including the zeros.
5. **`frame_num`.**  An IDR must carry `frame_num == 0` and the counter is
   reset at every IDR; otherwise FFmpeg synthesises "non-existing" frames and
   warns `number of reference frames exceeds max`.
6. **Deblocking.**  (a) the normative order is one macroblock at a time — all
   its vertical edges then all its horizontal edges — not two frame-wide
   passes and not macroblock anti-diagonals; (b) a luma edge is 16 samples
   (four per bS entry) and a chroma edge is 8 samples (two per bS entry);
   (c) `tc0 == -1` means "skip this sample", `tc0 == 0` still filters with
   `tc = 0`; (d) the strong intra filter is chosen per edge from `bS[0]`;
   (e) internal edges use the macroblock's own QP, the macroblock edge the
   average of the two, and chroma uses that value through the QPc table.
7. **IDCT rounding.**  `block[0] += 1 << 5` before the first pass and `>> 6`
   (truncating) in the second — not `+32` in both passes.
8. **`int16_t` truncation** of dequantised DC blocks must be reproduced, or
   high QPs drift (`>> 7`/`>> 8` DC dequant wrap-around).
9. **QPc table** for 4:2:0: QP 47..49 map to 38 (not 39) — the tail is
   `…, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39`.
10. **Inter residual block order.**  The sixteen 4x4 luma blocks of an inter
    macroblock are transmitted quadrant by quadrant (`i4x4 + 4*i8x8`, i.e.
    blocks `{0,1,4,5}, {2,3,6,7}, {8,9,12,13}, {10,11,14,15}`), and **every**
    block of a coded 8x8 group is present — including the ones whose
    `coeff_token` ends up `TotalCoeff == 0`.  Writing a coded group in raster
    order, or skipping the empty blocks, both desynchronise the decoder.
11. **`coded_block_pattern` <-> codeNum table (the last inter-only bug).**
    Both directions of the table had five entries rotated:
    `aeg_cbp_code_num` at codeNum 41…45 and `aeg_cbp_codenum` at cbp
    22, 25, 27, 29, 30.  The symptom is nasty because the decoder reads a
    *legal* pattern with the *same* number of coded 8x8 groups, so there is no
    parse error and no drift — the residual simply lands in the wrong
    quadrants (`0b1011` instead of `0b1101`, say), which looks like "the
    decoder dropped the residual in some blocks of a few P macroblocks" and
    moved around with the content.  `tests/syntax_check.c` now asserts that
    the two tables are exact inverses.
12. **MV predictor.**  The third candidate is the macroblock *above and to the
    right* (its block 12), falling back to block 15 of the above-left
    macroblock when the above-right one does not exist (H.264 8.4.1.3.1).
    Using the above macroblock's own MV makes every right-border macroblock
    predict differently from the reference decoder.
13. **Deblocking, p side of an internal edge.**  The samples on the p side of
    an internal (non-macroblock) edge belong to the macroblock being filtered,
    not to the neighbour; passing the neighbour's `aeg_mb` mis-evaluates
    `bs = 2` (non-zero coefficients) and the MV comparison.
14. **`aeg_params.partitions` default.**  `analyse_mb()` gates the whole inter
    path on this bitmask; a default of 0 silently turned every picture
    all-intra (correct, but ~2x the bitrate).  Bit 0 (16x16) is the only
    partitioning implemented and is now set in `aeg_params_default()`.
15. **MV prediction for partitioned macroblocks is not the median.**  For
    16x8 and 8x16 the reference decoder short-circuits: the *top* half of a
    16x8 macroblock predicts from the macroblock above (block 12) as soon as
    that one is inter, the *bottom* half from the macroblock to the left
    (block 11), the *left* half of an 8x16 from the macroblock to the left
    (block 3) and the *right* half from the above-right block — each with the
    fallback median only when the short-circuit does not apply.  The third
    neighbour also depends on the partition width: a half-width partition
    looks at block `12+bx` of the macroblock above, a full-width one at the
    above-right macroblock.  When a full-width partition is *not* in the top
    row, its above-right block lies outside everything the decoder tracks and
    the slot it reads stays `LIST_NOT_USED`.  Getting any of this wrong keeps
    the stream decodable (same bit count) but moves the residual onto a
    different prediction, so the picture drifts slowly.
16. **P_Skip vector.**  `pred_pskip_motion()` is not the median either: as soon
    as the left or the top neighbour is missing, intra, or sits at a zero
    vector, the skipped macroblock's vector is `(0,0)`; otherwise it is the
    median with the "matching reference" count.  Using the plain predictor made
    every skip in a moving picture reconstruct shifted.
17. **A skip run that reaches the end of the picture still has to be written.**
    Decoders count macroblocks by consuming `mb_skip_run`, so a dropped tail
    run silently ends the slice early: FFmpeg stops at the previous row without
    logging anything (`get_bits_left() == 0` is accepted as a clean slice end),
    which looks like "the last macroblock row is wrong".
18. **`gop_size == 0` means *one IDR then all P*, so it must not gate the motion
    search.**  `analyse_mb()` also tested `e->p.gop_size`, which turned
    `-g 0` into an all-intra encode.
19. **Picture padding geometry.**  Every plane occupies `(h + 2*PAD)` rows of
    `w + 2*PAD` bytes and the bottom/right padding is edge-replicated.  A row
    copy that starts at the *sample* origin instead of the row origin runs
    `PAD` bytes past the plane; the chroma plane offset must be
    `ls*lh + cs*ch`, not `plane[1] + cs*ch` (which also drops its top
    padding).
20. **Intra_4x4 top-right samples.**  `p[4..7][-1]` is only available when the
    4x4 block that holds it was *already coded*.  Inside a macroblock that is
    the block one column to the right in the block row above, so it is missing
    for the blocks at `(rx, ry) = (1, odd)` — i.e. the third and the eleventh
    block of the Figure 6-8 order — and for the whole last block column; in
    the first block row it comes from the macroblock above (or from the one
    above-right for `bx == 12`).  Replicating `p[3][-1]` in all those cases is
    correct; *not* replicating it in the `(1, odd)` case reads the stale
    picture buffer (the 4x4 block there has not been reconstructed yet) and
    gives the modes 3 and 7 a prediction the decoder cannot reproduce.  The
    symptom is small: a one-sample error in the last row of the block above,
    which the modes that read `t4..t7` then spread over the rest of the
    macroblock.
21. **Intra_4x4 `coded_block_pattern` codeNum.**  Table 9-4 of the spec has
    two different orderings of the same 48 patterns: the Intra_4x4 one (which
    is what `I_NxN` uses) differs from the inter / Intra_16x16 one at codeNum
    1 … 15.  Reusing `aeg_cbp_codenum` writes a pattern with the *same number*
    of coded groups, exactly like bug 11, but the wrong quadrant.
22. **Intra_4x4 mode 4 (diagonal down-right) taps.**  The spec indexes the top
    row two samples to the left of the block, so for `x > y` the taps are
    `p[x-y-2][-1], p[x-y-1][-1], p[x-y][-1]`; using `x-y-1 … x-y+1` shifts the
    predictor by one sample.
23. **A 4x4 mode whose side is unavailable must not be signalled.**  The
    reference decoder rewrites the modes of the macroblocks on the picture
    edges (DC becomes one-sided, the directional modes the missing side needs
    are rejected as an error), so the encoder has to restrict the candidate
    list to the modes it can actually predict — see `intra4_mode_list()`.
24. **IDR pictures must not use inter candidates.**  `analyse_mb()` gated the
    motion search on *"is there a reference picture"* rather than *"is this a P
    slice"*.  From the second IDR on (i.e. at every GOP boundary, and for every
    picture with `-g 1`) the analysis could pick an inter `mb_type` in a
    picture whose slice header says *I*, which the decoder reads as `I_NxN`
    plus garbage — FFmpeg reports `top block unavailable for requested intra
    mode` and the frame is lost.
25. **Two rows of the forward quantiser table.**  `aeg_quant_mf[4][2]` and
    `[5][2]` were `4559` and `3355` instead of `3355` and `2893` (the last two
    entries had been shifted up by one).  Only `QP % 6 == 4` (and, mildly, 5)
    is affected, so it looked like an RD problem: QP 16/22/28 lost 6 … 12 dB
    against their neighbours while QP 18/24/30 were fine.  Bit-exactness does
    not catch it — the decoder does not see the table, only our levels.

## Verification

```sh
make -j2
FFMPEG=/path/to/ffmpeg make test      # tables + cavlc round-trip + 9 clips
```

`make test` runs three things:

1. `tests/syntax_check.c` — the two `coded_block_pattern` tables must be exact
   inverses (see bug 11; this is the check that would have caught it).
2. `tests/cavlc_check.c 2000` — random residual blocks through the writer and
   a transcription of the reference decoder.
3. `tests/run_tests.sh` — thirteen clips (QP 12/23/26/40, presets 0 and 4,
   deblocking on/off, 64x64 … 320x208, forced CPU backend, 16x8/8x16
   partitioning, P_Skip) encoded, decoded with ffmpeg and compared **frame by
   frame**: the encoder's `--dump-recon` must be bit-identical to the decoder's
   output.  Any mismatch fails the script; a decoder that logs an error also
   fails it.  `tests/genyuv.c` takes a mode argument so that the partitioning
   cases really move the two halves of the picture in opposite directions —
   the default clip pans as one block and can never make a partition pay.

`reconstruction vs decode` must be `max abs diff 0`.  A nonzero diff means the
encoder and the decoder disagree about samples, which drifts on every
following P picture.

Manual single-macroblock probe (used to isolate transform/quantisation bugs):

```sh
./build/aegenc -w 16 -h 16 -q 33 -n 1 -i mb0.yuv -o v.264 --dump-recon vr.yuv
ffmpeg -v error -i v.264 -f rawvideo -pix_fmt yuv420p vd.yuv -y
# vr.yuv and vd.yuv must be identical
```

## Next

1. Quality/RD: per-block rate-distortion level selection (trellis, or at least
   the last non-zero coefficient), then 8x8 sub-partitions and 8x8 DCT (the
   latter needs a High-profile SPS).  That is where the remaining ~0.6-0.9 dB
   of inter and ~1.5-2 dB of intra-only efficiency sits.
2. GPU: the profile above says where to spend the next kernel — the coefficient
   pipeline (forward transform + quantisation + inverse quantisation + inverse
   transform + reconstruction) is now ~80% of the encode time, and the
   Intra_4x4 mode analysis is a per-block batch that only needs the
   neighbours.  `src/backend_cuda.cu` still has to be written for `CUDA=1`
   (see item 2 of the previous list), and the dispatcher's calibration has
   never run against real hardware in this environment.
3. Multi-slice / rate control are not implemented yet.

## Quality against a reference CPU encoder

`x264` (master, `--profile baseline --qp N --keyint 250`, built without asm) on
the same 60-frame 320x192 synthetic texture, both streams decoded and compared
with the source:

| QP | ours (kbps / PSNR Y, all) | x264 (kbps / PSNR Y, all) |
|---|---|---|
| 24 | 4776 / 37.10, 38.14 dB | 4825 / 38.09, 38.99 dB |
| 32 | 2621 / 29.61, 31.08 dB | 2672 / 30.36, 31.73 dB |

At the same *QP* we spend about 1% fewer bits and get ~1 dB less PSNR, i.e.
x264 is ~0.6–0.9 dB ahead on the rate-distortion curve.  Comparing streams at
the same QP is misleading though (the QP is only an index into a quantiser, and
x264 spends 20% more bits at the same index): on the IDR frame, at matched
*rate*, we are level with it —

| IDR frame, Y-PSNR | ours | x264 baseline |
|---|---|---|
| ~45 kB | 42.82 dB (QP 20) | 41.63 dB (QP 24) |
| ~38 kB | 38.78 dB (QP 24) | 39.52 dB (QP 26) |

The remaining inter gap is RD efficiency, not correctness: the mode decision
compares inter against Intra_16x16 / Intra_4x4 with a lambda estimate and a
SATD-based per-block mode choice, the motion vectors come from a single
predictor, and the quantiser has no trellis or last-coefficient optimisation.

The quantiser now retains coefficients 1.5x more eagerly than the reference
rounding offset (`quant_bias()` in `src/transform.c`).  Without trellis, the
standard deadzone (`f = 2^qbits/3` intra, `/6` inter) is too wide: widening it
by 3/2 moved the curve by 0.2-0.3 dB per percent of rate against a QP slope of
~0.07 dB/%, i.e. a real gain, at the cost of ~4% more bits for the same QP.

The `QP % 6 == 4` rows of the forward quantiser table were wrong until bug 25
was fixed, which made half a dozen QPs look like an RD problem.  Same clip, one
IDR frame, Y-PSNR:

| QP | before | after |
|---|---|---|
| 16 (`%6 == 4`) | 33.79 dB | 45.69 dB |
| 22 (`%6 == 4`) | 33.38 dB | 39.73 dB |
| 28 (`%6 == 4`) | 31.63 dB | 33.84 dB |

Intra only (`-g 1`, the ten frames of a fast-panning clip, QP 24): 38.78 dB at
382.0 kB against x264's 41.64 dB at 443.8 kB, i.e. ~1.5-2 dB behind once the
rate is matched — the IDR of a static scene is level (table above), the frames
with a lot of motion behind an intra-only decision are where the per-block RD
(nothing but SATD) shows.

Inter prediction is worth roughly a 2x bitrate cut on this clip (the earlier
all-intra rows — 8930 kbps at QP 24 — came from `partitions == 0`, bug 14).

Partitioning pays on content a single vector cannot describe.  With
`--partitions 1` (16x16 only) versus `--partitions 7` (16x16 + 16x8 + 8x16),
same QP 26, 8 frames:

| clip (320x192) | 16x16 only | with 16x8/8x16 | change |
|---|---|---|---|
| halves moving in opposite directions vertically (`genyuv … 1`) | 18556 B | 15086 B | −18.7% |
| … horizontally (`genyuv … 2`) | 13839 B | 12066 B | −12.8% |
| noisy clip with a vertical split, 10 frames | 236591 B | 139556 B | −41.0% |

P_Skip collapses a still picture into a handful of bits per frame: the first
frame of the `genyuv … 3` clip costs 4148 bytes, each following frame 8 bytes.

## Speed

60 frames of 320x192 at QP 24, single thread, CPU backend (2 cores, no GPU in
this sandbox):

| encoder | fps | bytes |
|---|---|---|
| ours, preset 0 | 246 | 1194196 |
| ours, preset 2 / 4 | 51 / 55 | 1194023 |
| x264 baseline (no asm) | 109 | 1206202 |

Intra_4x4 costs ~22% (preset 2: 75.5 → 58.5 fps with the previous quantiser
bias) because a macroblock candidate now runs sixteen predictors, sixteen
transforms and sixteen quantisers.  That is
also why the stage profile has shifted: `transform` (forward/inverse transform,
quantisation and reconstruction of every candidate) is now ~80% of the encode
time, motion estimation ~12%, deblocking ~2% — i.e. the stage that a GPU should
take over is no longer the motion search but the coefficient pipeline.
