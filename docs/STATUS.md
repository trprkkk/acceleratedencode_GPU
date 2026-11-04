# Status and engineering notes

Date: 2026-11-01.  Branch: `arena/589d1e45-acceleratedencode-gpu`.

## Where the project stands

The encoder is **conformant on I, P *and* B pictures**: an independent decoder
(FFmpeg 7.0.2) reproduces the encoder's reconstruction *bit for bit*
(`max abs diff 0`) on every frame, not only the IDR.

| content | sizes | QP | presets | frames |
|---|---|---|---|---|
| 1 MB probe, grids, smooth gradients | 16x16 … 320x192 | 10 … 51 | 0 … 4 | 1 … 40 |
| moving synthetic texture (inter, heavy AC) | 320x192, 320x176, 320x208 | 12, 18, 23, 26, 28, 32, 40 | 0 … 4 | 1 … 60 |
| translated noise / gradient clips | 96x96 | 23, 32 | 0, 2 | 3 |
| odd frame sizes (13 MB rows, 11 MB rows) | 320x208, 320x176 | 23 … 28 | 2, 3 | 3 … 4 |
| Intra_4x4 probes (random 16x16, 32x16 gradient) | 16x16, 32x16 | 24 … 33 | 2, 4 | 1 … 2 |
| P_8x8 (four quadrants moving apart, all four `sub_mb_type` shapes) | 320x192, 96x64, 48x32 | 12, 20, 26, 34 | 2 | 6 … 8 |
| IDR pictures after P pictures (`-g 3`, `-g 1`) | 320x192 | 20, 26, 34 | 2 | 6 … 10 |
| slices (`--slices 2 … 12`, including uneven row counts) | 320x192, 320x208, 64x64 | 22 … 26 | 0, 1, 4 | 4 … 8 |
| 8x8 transform + Intra_8x8 (every clip above re-run with it on; 30-case I-picture matrix over four ramp/noise clips) | 32x16 … 320x192 | 12 … 40 | 0 … 4 | 1 … 16 |
| two P references (`--ref 2`), P-only and under B pictures, with IDRs and scene cuts (162-case B matrix x2) | 320x192 | 20 … 36 | 0 … 4 | 12 … 60 |

with and without the in-loop deblocking filter.  `make test` asserts this and
fails (non-zero exit) on any mismatch; it runs seventy-eight encoded clip cases
frame by frame, all of them bit-exact, eighteen of them with B pictures compared
through the coding order the encoder reports (`--dump-order`; with the adaptive
group length it is the only way to know it, and the two `--b-adapt 0` cases
check it against the fixed structure rebuilt from the frame and B counts).  It also
checks that the ABR controller hits its target, plus a sweep of
the partition, QP and GOP options (`tools/sweep.sh`) and a 240-case matrix over
`--partitions 1|3|7|15`.

Every case above is coded with **CABAC**, which is the default entropy coder;
`--no-cabac` codes the same decisions with CAVLC (34-36% more bytes on the
panning clip at equal PSNR) and is kept because the two coders must produce the
same reconstruction.

Implemented and building without warnings:

* bit writer with emulation prevention, NAL/SPS/PPS/slice headers, IDR +
  `frame_num` handling (IDR resets `frame_num`, `log2_max_frame_num_minus4`
  sized to the GOP)
* CAVLC residual coding (coeff_token, levels, total_zeros, run_before) —
  round-tripped by `tests/cavlc_check.c` (12 000 random blocks, 4 block kinds)
* the CABAC arithmetic coder (`src/cabac.c`, `tests/cabac_check.c`: 1.6 M bins,
  every probability state, bypass bins, both end-of-slice flags) **and the
  syntax layer that uses it** (`write_mb_cabac()` and the `cabac_*` helpers in
  `src/encoder.c`): mb_skip_flag, the I and P mb_type trees, sub_mb_type, mvd,
  the intra 4x4/8x8 and chroma prediction modes, coded_block_pattern,
  transform_size_8x8_flag, mb_qp_delta, and the residual block of all six
  categories with its significance/last/level-prefix contexts.  `make test`
  runs the whole conformance suite through it (see *CABAC* below)
* the rate-distortion pass priced by that same coder: every candidate is run
  through `write_mb_cabac()` in price-only mode on a copy of the slice's
  context state (`rd_cabac_bits()`), so the mode decision optimises for the
  coder that emits the stream (see *Pricing the decision pass with the
  coder*), and a `P_Skip` candidate carrying the vector the skip rule derives,
  so the cheapest macroblock the standard has is always among the candidates
  (see *The skip candidate*)
* forward/inverse 4x4 integer transform, 4x4 and 2x2 DC Hadamard, quantisation
  and dequantisation, including the reference decoder's `int16_t` truncation
* Intra_16x16 (all four modes), Intra_4x4 (all nine modes, the Figure 6-8
  block order, `prev_intra4x4_pred_mode_flag`/`rem_intra4x4_pred_mode` and the
  per-block mode decision), chroma (all four modes)
* the 8x8 luma transform and **Intra_8x8** prediction (High profile): 8x8
  integer DCT/quantisation/dequantisation/IDCT matching the reference
  decoder's rounding, 8x8 CAVLC (four interleaved 16-coefficient lists per
  block with the group-end non-zero rule), all nine Intra_8x8 modes chosen by
  exact rate-distortion, `transform_size_8x8_flag` in the SPS/PPS syntax and
  the deblocking rules that go with it, and `--no-8x8dct` to code the same clip
  in the Baseline profile for comparison (see *The 8x8 transform and
  Intra_8x8* below)
* half/quarter-pel motion estimation and compensation, MV prediction, skip runs
* P inter partitioning: 16x16, 16x8, 8x16 and **P_8x8** (four 8x8 sub-macroblock
  partitions, one motion vector and one `mvd` each).  Every partition gets its
  own motion search and its own MV predictor, including the sub-macroblock
  rules of 8.4.1.3.1 and the per-partition availability rule (bug 27).  Each 8x8
  sub-macroblock picks its own `sub_mb_type`: 8x8, 8x4, 4x8 or 4x4.
* P_Skip macroblocks (`mb_skip_run`), including the run that reaches the end of
  the picture
* scene-cut detection (`--scenecut`, on by default): an IDR picture is inserted
  where the inter prediction stops being better than the intra one, instead of
  spending a whole frame's bits predicting a scene that is no longer there (see
  *Scene cuts* below)
* multiple slices per picture (`--slices N`): the picture is cut into N row
  ranges, each an independent NAL with its own slice header and
  `first_mb_in_slice`, and every neighbour outside the current slice is
  unavailable (see *Slices* below)
* slice threads (`-t N`): the slices of a picture are coded concurrently, with
  a bitstream identical to the serial one (see *Slice threads* below)
* two-pass rate control (`--pass 1/2 --stats <file>`, `--qcomp`, see *Two-pass
  rate control* below)
* one-pass average-bitrate rate control (`--bitrate`, `AEG_RC_ABR`) and
  variance-based adaptive quantisation (`--aq`, off by default, see below)
* presets scale the integer search range (8/16/24/32/48) and the subpel
  refinement (off for ultrafast).  The quarter-pel stage runs once for the
  macroblock's 16x16 vector and once per 8x8 partition; the smaller partitions
  inherit the macroblock's fractional offset and only choose between it and the
  integer position, which is within 0.05 dB of refining each of them and twice
  as fast
* deblocking filter (all edges, normative order, alpha/beta/tc0 tables)
* CPU + OpenCL + CUDA backends behind a measured dispatcher, device query,
  calibration.  The CUDA backend is plain C: `libcuda.so.1` (driver API) and
  `libnvrtc.so` are both `dlopen()`ed and the kernels come from the same
  `src/kernels.h` string, compiled at run time for the device's compute
  capability -- so `make CUDA=1` needs no toolkit and no `nvcc`.
* `tests/kernel_check.c` compiles the kernel source string as ordinary C
  (with `__kernel`/`__global`/`get_global_id` macros) and runs every kernel
  work item against the CPU reference, so kernel regressions are caught on a
  machine with no GPU.
* SSE2 fast paths for the seven hottest inner loops, each kept bit-identical to
  the scalar code it replaces (so none of them can change a decision) and each
  with the scalar loop left in place as the non-x86 fallback and as the
  reference the tests compare against:
  * `aeg_sad()` -- `pmpsadbw`, 13x faster per 16x16 call, the motion search's
    innermost loop;
  * `aeg_satd_4x4()` -- a register-resident Hadamard (butterfly across rows,
    transpose, butterfly across columns) on int16 lanes, 2.2x faster; exact
    because a block of byte differences cannot leave the int16 range
    (|coef| <= 255 * 16);
  * `aeg_quant_4x4()` -- abs/sign/multiply through `pmaddwd` with the three
    (x&1)+(y&1) scale factors and the rounding offset hoisted out of the block,
    2.5x faster; the block is stored transposed, which is the layout the
    decoder expects;
  * `aeg_idct_add_block()` -- dequantise, both inverse passes and the
    prediction add in int32 lanes with saturating packs for the clip, 3x
    faster.  This one keeps the scalar loop's *pass order* and its two int16
    truncation points (bug 8), because the reference decoder truncates there
    too and a stream that leaves int16 in an intermediate must reconstruct the
    same way.  The three bugs the first version of it had are worth recording:
    the DC rounding offset has to be added *after* the dequantised DC has been
    truncated to int16 (not before), the "no AC coefficients" mask has to
    select the AC lanes (an `andnot` on the DC lane's own register zeroes the
    wrong half), and the DC-only shortcut is only valid when there is a
    prediction to add -- without one, a very large DC has to go through the
    int16 truncation that the shortcut would skip.
  * `aeg_transform_4x4()` -- the forward transform in int16 lanes, 3.2x faster.
    It computes the two passes in the other order than the scalar code (columns
    then rows, because a register holds a row), which is the same matrix product
    for a separable transform, and stores through a second transpose so the
    coefficients come out in the order the scan expects;
  * `idct_core()` -- the inverse transform the rate-distortion pass uses for
    every candidate, 2x faster: four values at a time in int32 lanes, with the
    two int16 truncation points of the reference arithmetic applied explicitly
    (`_mm_srai_epi32(_mm_slli_epi32(v, 16), 16)`), because those truncations are
    part of what the decoder does, not an artefact of the scalar code.
  * `aeg_hpel_plane_fill()` -- the three fractional-pel planes, eight samples at
    a time (0.19/0.22/0.65 -> 0.05/0.04/0.23 ms per 320x192 picture).  The
    two-dimensional plane is the interesting one: it needs the *unshifted*
    horizontal intermediate (H.264 8.4.2.2.1), whose values reach +-13 260, and
    the vertical sum over six of those reaches +-690 000, so the horizontal
    stage stays in int16 lanes and the vertical one is widened to int32 --
    rounding the horizontal stage first, as the clip of a real sample would,
    gives different samples (bug 28 in reverse).  This function is also the CPU
    reference the `interp_plane` device kernel is checked against, so its
    output has to stay byte-identical; `kernel_check` still passes.
* CLI with `--list-devices`, `--calibrate`, `--bench`, `--dump-recon`,
  `--slices N`, `-t N` (slice threads), `--scenecut`, `--vbv-maxrate`/
  `--vbv-bufsize`

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
26. **The top-right samples of an Intra_4x4 block are only available if they
    were *coded* already.**  Modes 3 (diagonal down-left) and 7 (vertical-left)
    read `p[4..7][-1]`.  For the blocks at `(x,y) = (4,4)` and `(4,12)` those
    samples belong to the blocks at `(8,0)` and `(8,8)`, which Figure 6-8 codes
    *later*, so the reference decoder replaces them with `p[3][-1]`; using the
    real pixels desynchronises the reconstruction.  Verified by forcing mode 3
    there with the real samples: the stream still parses, the frames no longer
    match.  The blocks in the last block column (`x = 12`) do take their
    samples from the macroblock above/to the above-right, and that case *is*
    accepted by the decoder (also verified by forcing it), so
    `code_luma_i4x4()` distinguishes the two instead of rejecting all of them.
27. **P_8x8 has no MV-predictor short-circuit, and a 4x4 block only counts as
    a neighbour once its own partition has been decoded.**  16x8 and 8x16
    partitions short-circuit to a single neighbour in the reference decoder, so
    they need their own rule; the 8x8 sub-macroblock partitions use the plain
    three-neighbour process.  Their "above-right" neighbour ("C") sits at
    block `(x+pw, y-1)`; when it is unavailable the decoder substitutes the
    block above-left of the partition, `(x-1, y-1)` — and when this crosses the
    macroblock edge it is `(3, y-1)` of the left macroblock, or for the top row
    of blocks `(x-1, 3)` of the macroblock above (`(3,3)` of the above-left
    macroblock for `x == 0`).  The subtle part: the decoder fills its 4x4
    neighbour cache partition by partition, so a block that belongs to a
    *later* sub-macroblock or a later sub-partition is `PART_NOT_AVAILABLE`, not
    the vector of the previous frame's decision.  For a 4x4 sub-partition the
    above-right block of the sub-macroblock's bottom-right corner is in the
    *next* sub-macroblock, so this fires on ordinary content — that was the
    last remaining desync (320x192 quadrants, from the third frame on, one
    macroblock, 38 luma samples).  Found by instrumenting the reference
    decoder to print the per-partition MV predictor and diffing it against
    `AEG_MBDBG=1 ./build/aegenc ...`; the encoder now keeps an `avail[16]`
    map that `avail_publish()` fills exactly where `me_publish()` fills the
    MV, and `predict_mv_at()` consults both.

28. **The two-dimensional half-pel sample was rounded twice.**  H.264 8.4.2.2
    builds the `j` position (half pel in both axes) from *unshifted* horizontal
    6-tap intermediates: the vertical 6-tap runs over 16-bit sums and the whole
    thing is shifted by 10 once (`+512 >> 10`).  Our `hpel_hv()` computed each
    horizontal intermediate with the full `+16 >> 5` shift and *clipped* it to
    8 bits, then ran the vertical pass with another `+16 >> 5`.  The gain is the
    same (32), so the error is only the double rounding plus the intermediate
    clipping -- a few levels on steep gradients -- but it is a real mismatch
    between our reconstruction and the decoder's.  It went unnoticed because
    the subpel refinement was never enabled by default (bug 29); the first
    `--subme 1` sweep desynchronised on the very first P picture.  Fixed by
    keeping the horizontal stage unshifted and shifting once at the end.
29. **`aeg_params_default()` left `subme` at zero**, so every preset ran with
    the quarter-pel stage disabled even though the CLI help and the preset
    table both say otherwise (`aeg_encoder_open()` only substitutes the preset
    value when the caller passes a negative `subme`).  Quarter-pel motion
    compensation is worth far more than any other single feature measured in
    this project: at the same QP it takes the panning-texture clip from 35 157 B
    / Y 43.64 dB to 28 676 B / Y 44.50 dB, and it is what closes most of the
    gap to x264.  It is now on for every preset except ultrafast.

30. **`sad_batch` candidates are two block *positions*, not a position and a
    vector.**  Every `sad_batch()` candidate is four `int16_t`,
    `(cx, cy, ref_dx, ref_dy)`, where the reference fields are the block origin
    *in the reference picture*, not a displacement.  The whole-frame search in
    `me_seed_frame()` filled in the vector (`nx`, `ny`) and read the winner back
    without subtracting the macroblock origin, so round 0 asked for the block at
    `(3, 2)` for *every* macroblock -- i.e. macroblock 0's neighbourhood -- and
    took round 0's cost as that macroblock's cost.  The first macroblock was
    then consistent (cost == its own SAD) and every other one was garbage
    (`mb 1 cost 312 at (19,2) direct 19670`), which is why the symptom looked
    like an indexing bug: with `AEG_PARDBG` the per-macroblock cost and its
    vector disagreed, and the fractional rounds then optimised that bogus
    baseline, running the vectors out to the clamp limits (parallel SATD total
    2 034 507 against the serial walk's 153 410).  Fixed by writing
    `ox + nx, oy + ny` and subtracting the origin again when reading the winner
    back; the integer rounds now total 195 346 SAD over 240 macroblocks (zero-MV
    baseline 877 272, ~814/macroblock) and the fractional rounds improve it
    monotonically.  The lesson is the same as bug 22's: a batch interface whose
    arguments are *positions* can be silently fed displacements and the CPU
    reference implementation will happily agree.

31. **The batched inverse transform was checked against a wrong reference.**
    `cpu_iq_idct_batch()` -- the C fallback of the `iq_idct_batch` device
    kernel, and the thing `tests/kernel_check.c` compares the kernel against --
    carried its own copy of the inverse transform, and that copy had lost all
    three properties of the real one: it skipped the int16 truncation of the
    dequantised coefficients (bug 8), it skipped the `+ 1 << 5` DC rounding
    before the first pass (bug 7), and it rounded the second pass with
    `(x + 32) >> 6` instead of truncating.  Comparing a kernel against it is
    comparing against nothing: 31% of the samples of a random block came out
    different from the encoder's own reconstruction, by 5 or more at QP 20, and
    `kernel_check` would have called that a pass.  There is now one definition
    of the arithmetic (`aeg_idct_residual()`), the scalar add path is a wrapper
    around it, and `tests/coef_check.c` drives the batched entry points against
    a *third* transcription of the reference.  The rule this is an instance of:
    a fast path must never be validated against another copy of itself.

32. **A slice ended with an extra `0x80` byte, so the decoder walked out of
    it.**  `code_slice()` aligned the bit writer and then called
    `aeg_bw_finish()`, which aligns *again* -- a second, all-zero-padded
    `0x80` byte at the end of every slice.  A single-slice picture could not
    show this: the decoder stops at the last macroblock of the picture and the
    tail is never read, so 30 conformance cases and a bit-exact reconstruction
    stayed green for a whole afternoon of debugging.  A slice that ends in the
    middle of a picture is decoded differently: FFmpeg's `decode_slice()` (the
    non-threaded path) has no idea where the next slice starts, it only knows
    that the slice is over when the bit reader lands *exactly* on the end of
    the RBSP -- `get_bits_left(&sl->gb) <= 0`.  With the extra byte the reader
    had 2 bits left after the last macroblock, fell through to the next
    `ff_h264_decode_mb_cavlc()`, and decoded the following slice's header as
    macroblock data ("left block unavailable for requested intra4x4 mode -1",
    "error while decoding MB 0 6").  The lesson: *a bitstream bug that the
    last macroblock of a picture hides is not a small bug* -- `decode_slice()`
    consumes the very bits that a picture end makes unreachable, so
    multi-slice is a stricter conformance test than the whole picture is.

33. **An Intra_8x8 neighbour is not a DC neighbour.**  H.264 derives the
    Intra_4x4 prediction mode of a block from `min(left, top)` of the
    *neighbouring blocks' modes*, and a decoder keeps those modes in one cache
    of 4x4 slots.  An Intra_8x8 macroblock has no 4x4 modes, but a decoder
    still fills the cache: it replicates each 8x8 mode over the 2x2 group of
    slots it covers, so the block to the left of block row `by` reports the
    mode of the 8x8 group in the right-hand group column of that block row.
    The encoder treated every non-Intra_4x4 neighbour as "no mode" and used the
    DC fallback 2 instead, which changes `min(left, top)` and therefore the
    *predicted* mode, and therefore which of `prev_intra4x4_pred_mode_flag` and
    `rem_intra4x4_pred_mode` the block sends.  The decoder then reads a
    different mode from the same bits -- the macroblock's reconstruction
    diverges while the syntax still parses, so nothing reports an error and the
    mismatch only shows up as a slightly wrong picture.  `intra4_pred_mode()`
    now maps an Intra_8x8 neighbour the way the cache does, and
    `intra8_pred_mode()` the other way round (an Intra_4x4 neighbour
    contributes the 4x4 mode of its last block column/row).  Found by
    comparing the encoder's chosen modes against FFmpeg's `DEMODE` dump
    macroblock by macroblock on a 32x16 ramp at QP 22.

34. **`transform_size_8x8_flag` is not always present for inter
    macroblocks.**  For an inter macroblock the flag follows the coded block
    pattern, but only when the luma pattern is non-zero *and* no partition is
    smaller than 8x8: `P_8x8` with an 8x4, 4x8 or 4x4 `sub_mb_type` may not
    signal it, and the decoder does not read a bit there at all.  Writing one
    anyway shifts every following bit, which showed up as
    "mb_type 689 in P slice too large", "negative number of zero coeffs" and
    "cbp too large" from the second P picture on -- 15 of the conformance
    cases failed at once.  `t8_allowed()` is now the single place that decides
    it, and it gates the decision, the bit count and the write.

35. **Motion-vector prediction has to know that Intra_8x8 is intra.**
    `is_intra_mb()` in `mc.c` listed I4x4, I16x16 and IPCM but not the new
    I8x8 type, so an Intra_8x8 neighbour was taken as an inter block with
    motion vector (0,0) and counted as an available candidate in the
    A/B/C/median rule.  With one reference picture that changes the predictor
    from "the single matching neighbour" to a median that includes a zero --
    the encoder then writes a different `mvd` than it meant, and the picture
    drifts from the first macroblock whose top-right neighbour happened to be
    Intra_8x8.  The type is added, which also fixes the same test for skip
    detection.

36. **The deblocking filter needs one non-zero count per 8x8 group, not per
    4x4 block.**  With the 8x8 transform the CAVLC lists carry a quarter of an
    entire 8x8 block each, so the 4x4-block non-zero counts an 8x8 macroblock
    holds are *sub-list* totals, not block totals; a decoder broadcasts one
    count per 8x8 group.  The boundary strength therefore has to compare
    groups.  Comparing sub-list totals made bS 2 instead of 0 (or 1) on an
    edge, i.e. it filtered an edge the decoder left alone, and with the
    deblocking filter enabled the picture drifted after a few frames in exactly
    the macroblocks whose 8x8 neighbour was coded.  `aeg_mb.nz8[4]` now keeps
    the group flags and `boundary_strength()` uses them for the side that was
    coded with the 8x8 transform.  With the filter off
    (`--no-deblock`) this bug is invisible, which is why the 30-case I-picture
    matrix was green while `make test` was not.

37. **The end-of-slice `1` is not a normal decision.**  The last
    `end_of_slice_flag` of a slice is coded against a *fixed two-unit* LPS
    interval, and the decoder's test is `codIOffset >= range` with the range
    already reduced by two.  A first version of `aeg_cabac_flush()` did what
    the bin looks like (`range -= 2; low += range`, i.e. move to the top of the
    reduced range) and every stream then failed to end: the value it picks is
    two units *below* the interval the decoder tests for.  The value has to be
    chosen at `low + range - 2` measured against the range **before** the
    subtraction, and the last nine bits have to be written on top of that,
    because the decoder pre-loads nine bits it has not been given yet.  Without
    them the last few bins of a slice decode from bits that were never written
    -- which `tests/cabac_check.c` reports as "end_of_slice_flag did not decode
    as 1", two bins from the end.

## The coefficient stages as one batch per macroblock

The forward transform + quantisation and the inverse transform of an inter
macroblock are 1.9 M block calls per hundred pictures at preset 2, which is
what makes them a stage worth moving: not that any single block is expensive
(it is 50 ns on the CPU now) but that there are thirty-two independent calls
per macroblock and 240 macroblocks per picture.  The sixteen luma blocks of an
inter candidate are independent of each other -- their prediction comes from
the reference picture, not from the block above or to the left -- so
`code_luma_inter()` gathers the sixteen residuals and issues

```c
tq_batch(res, 16, qp, 0, level, nz);        /* forward  + quantise  */
iq_idct_batch(level, 16, qp, rec);          /* dequant + inverse   */
```

one call each instead of sixteen, and then does the prediction add, the clip
and the SSE per block (those are cheap and they are what the RD cost needs).
`coef_batch` is set when the dispatcher has *both* `AEG_STAGE_TRANSFORM` and
`AEG_STAGE_IQUANT_IDCT` on the device: an inter candidate cannot transform on
one and reconstruct on the other without a round trip in the middle of its RD
cost.  `AEG_COEF_BATCH=1` forces the code path against the CPU reference ops,
which is how the conformance suite covers it without a GPU -- and it has to
produce the same bitstream as the per-block path, which it does on every case
in `tests/run_tests.sh`.  On the CPU the forced path is ~20% slower in the
transform stage (scalar add/clip/SSE instead of the SSE2 `aeg_idct_add_block`,
plus the gather); on a device it is two launches instead of thirty-two calls.

Two things first stayed per block, because their DC coefficients go through
a Hadamard of their own and the batch API had no per-block override:
Intra_16x16 (the 4x4 Hadamard of the sixteen DCs) and chroma (the 2x2 one).
Both are batched now with a DC-split pair of operations (`tq_dc_batch`,
`iq_dc_batch`; see *GPU groundwork without hardware, continued*).

## Rate-distortion batching (provisional)

The per-candidate batch above is one launch pair per candidate.  The next step is
to put *every* inter candidate of a macroblock in one launch pair:

```
for each candidate k:                 /* 1. produce */
    predict candidate k (plane MC)    -> pred_l
    residual = src - pred_l           -> rd_res[16 blocks * k]
    save pred_l                       -> rd_pred[k]
tq_batch(rd_res, 16 * n)              /* one launch for all candidates */
level selection (trellis), on the host
iq_idct_batch(rd_lev, 16 * n)         /* only without trellis: with it the
                                         host reconstructs (see *A lighter
                                         host side for the batch*) */
for each candidate k:                 /* 2. decide */
    finish(k) -> nz, cbp, reconstruction, SSE, CAVLC bits -> cost
```

`AEG_RD_BATCH=1` selects it; by default it is chosen when the dispatcher has
both coefficient stages on a device.  It cannot be lifted from a macroblock to
a picture without changing the bitstream, and that is worth stating precisely
because it is the obvious next question:

* the motion search predicts every macroblock's vectors from its neighbours'
  *decided* vectors (H.264 8.4.1.3), so candidate generation is serial in
  raster order;
* intra prediction needs the reconstructed samples of the macroblock to the
  left and above, which only exist once those macroblocks are decided;
* the rate term of the RD cost comes from `aeg_cavlc_block_cost()` with the
  neighbour non-zero counts as context, which is serial again.

So "produce every candidate for the whole picture" would have to guess the
neighbours' decisions, and the stream would no longer be the one the serial
encoder produces.  A picture-wide version is possible only as a *different
encoder* (slice-per-thread, or a lookahead/two-pass decision that accepts a
bounded worst case), not as a bit-exact accelerator.  The batch therefore stops
at the macroblock, which is where the coefficient work of a decision lives:
64 blocks per launch instead of 16, two launches per macroblock instead of two
per candidate.

State of the implementation (this is the provisional part):

* **Verified here**: the stream is byte-identical to the serial candidate loop
  -- with the CPU reference ops behind the batch entry points *and* on a real
  OpenCL device (PoCL): every preset on the five real clips at two QPs (50/50),
  CAVLC (25/25), slice threads, the weighted-prediction fade clip, and the
  conformance cases `rd_batch` / `rd_batch4`.  (This was true when the batch
  was written and had silently stopped being true: the serial loop had since
  grown the level selection, the CABAC price with chroma, and the candidate
  refinement, and the batch decided by the CAVLC estimate without them --
  4-10% more bits, found the first time it ran on a device.  The decide step
  now *is* the serial loop with the luma coefficients taken from the batch;
  see *GPU groundwork on a real OpenCL device*.)
* **Not verifiable here**: whether the batch actually beats thirty-two calls on
  a GPU.  On the CPU the forced path used to cost ~10% (a scalar add/clip/SSE
  loop, every candidate's 8x8 decision made twice, 8x8 candidates sent through
  the launches for nothing); since *A lighter host side for the batch* it costs
  the same as the serial loop within noise, so the batch's host side is no
  longer what a device has to beat.  Whether it wins on a GPU depends on the
  launch round trip, which the dispatcher's trial measures.
* **To be measured on hardware** with `tools/gpu_validate.sh clip.yuv W H
  FRAMES QP`: it encodes with the CPU backend, with the dispatcher free and
  with each batched path forced, then checks that the CPU/auto/coef/rd streams
  are the same bytes and that every stream's reconstruction matches an
  independent decode.  `search`/`all` are checked by reconstruction instead of
  by bytes, because the whole-frame search rounds are a different search, not a
  faster rendering of the same one.

Enabling it on a device, in order of increasing bluntness:

```
-s tq=opencl -s iq=opencl    # the dispatcher owns both coefficient stages
                             # (or =cuda), and the paths follow it
AEG_COEF_BATCH=1             # per-candidate batch, dispatcher ignored
AEG_RD_BATCH=1               # whole-macroblock batch, dispatcher ignored
```

The flags are resolved at the top of every picture, not once at open time, so a
`-s` override takes effect from the first `aeg_dispatch_update()` on.  Without
an override the dispatcher now places the two stages by trial
(`aeg_dispatch_coef_pick`): the first pictures of each kind alternate between
the placements and the faster one per macroblock stays -- see *GPU groundwork
on a real OpenCL device*.

## Verification


```sh
make -j2
FFMPEG=/path/to/ffmpeg make test      # tables + cavlc + cabac + 76 clip cases
```

Every push also runs `.github/workflows/ci.yml` on GitHub Actions
(ubuntu-24.04, no GPU), where items 8 and 9 do *not* skip:

* **cpu** — `make test`, plus `--frame-threads 4` against `--frame-threads 1`
  (`cmp`) at presets 1, 3 and 4.
* **opencl-pocl** — Ubuntu's PoCL (`pocl-opencl-icd`) as a real OpenCL
  device: `--list-devices`
  must show it, `backend_check opencl` (106 checks), the whole conformance
  suite with `AEG_TEST_BACKEND=opencl`, and `gpu_validate.sh` at `-p 3 -t 1`
  and `-p 2 -t 2` (every device path byte-identical to the CPU stream).
* **cuda-nvrtc** — `nvrtc_check` with NVRTC from pip (no toolkit, no driver).
* **sanitizers** — ASan + UBSan (`-fno-sanitize=alignment,shift`, no
  recovery) on the `OPENCL=1` build: `kernel_check`, `coef_check`,
  `backend_check` on PoCL, and encodes at presets 0/2/4 through the CPU, slice
  threads, frame threads, the forced RD batch and the device search, with the
  batch and frame-thread streams `cmp`'d against their CPU counterparts.

Each step asserts its success line (`checks agree`, `bit-exact`,
`kernels compile`), so a step that silently skipped (no ffmpeg, no device)
fails instead of passing.  The device check (`tools/ci_device_check.sh`)
reports a missing device as `::error::` annotations -- CPU model, the
encoder's device list, the ICD setup, a `POCL_DEBUG` tail -- which the checks
API returns even where the log archive cannot be fetched.  CI first used the
pip PoCL that `tools/pocl_env.sh` installs (PoCL 3.0-rc2, LLVM 14); its device
failed to enumerate in two of ten job runs, never locally, so CI now uses the
distribution's PoCL; `pocl_env.sh` remains the no-root way to get a device.

`make test` runs nine things (8 and 9 skip, and pass, without a device or
NVRTC; see *GPU groundwork on a real OpenCL device* for running them on one):

1. `tests/syntax_check.c` — the two `coded_block_pattern` tables must be exact
   inverses (see bug 11; this is the check that would have caught it).
2. `tests/cavlc_check.c 2000` — random residual blocks through the writer and
   a transcription of the reference decoder.
3. `tests/coef_check.c 200000` — the coefficient path against an independently
   written reference, including the batched entry points the device backends
   implement (`tq_batch`, `iq_idct_batch`, `aeg_idct_residual`): random blocks (uniform, sparse, single-coefficient,
   DC-only, and amplitudes at both int16/dispatch extremes) at QPs 0…51 and
   block strides 4, 8, 16, 20, 24 and 384, through the scalar quantiser and
   inverse transform, the SSE2 dispatcher, and the in-place case
   (`pred == block` is only defined when `pred_stride == stride`).  This is the
   test that catches a SIMD path which is self-consistently wrong — the reason
   the earlier throw-away harnesses in `/tmp` were useless: they re-implemented
   the reference with the same transposing store as the fast path, so both
   agreed while the bitstream changed.  Every failure the SSE2 rebuild had
   (three real ones, listed under bug 8) was found here.
4. `tests/cabac_check.c 200000 8` — the arithmetic coder against a decoder
   written from the decoding process (9.3.3) and from the bit definition of the
   output: nine-bit `codIOffset`, bit-by-bit refill, no access to the encoder's
   `low`/`range`/carry machinery.  1.6 M bins across eight seeds, over all 460
   contexts (initialized from every table at QPs 0..51, then perturbed so that
   all 128 states and the 0.5 state are visited), mixing context-coded bins,
   bypass bins, runs of either bin and `end_of_slice_flag = 0` bins, and
   requiring the decoder's context states to end up identical to the encoder's.
5. `tests/cabac_mb_check.c 20000 1` — the CABAC *residual block* syntax
   against a second decoder, transcribed from 9.3.2.2 and reading the byte
   stream bit by bit: 20 000 random blocks over all six `ctxBlockCat`
   categories, with levels that reach the Exp-Golomb escape (and beyond the
   unary prefix), the coded_block_flag of the five categories that have one,
   both signs, and the terminate flag after every block.  `cabac_check.c`
   proves the coder; this proves the context derivation and the binarization
   on top of it, which is where the three CABAC bugs in the encoder were.
6. `tests/kernel_check.c` — the device kernel source (`src/kernels.h`) compiled
   as C with the OpenCL qualifiers defined away and `get_global_id()` driven by
   a counter, so the kernels are checked on a machine with no GPU at all.  The
   inverse transform is fed four amplitude classes — the full int16 range,
   sparse extremes, DC-only at the int16 bounds, and +-3 — because the parts of
   it that are easy to get wrong (the int16 wrap of a dequantised coefficient,
   the DC rounding in the first pass only, the truncating second pass) do not
   show up at the amplitudes a quantiser normally produces.  Removing any of
   the three from the kernel makes this test fail; it is the check that would
   have caught bug 31 before the encoder ever ran on a device.
7. `tests/backend_check.c` — the device backends' *host side* against the CPU
   reference, on a real device: every operation through the guarded entry
   points the encoder calls, on padded pictures allocated exactly as the
   encoder allocates them, whole-picture candidate lists, vectors into the
   padding, then two threads at once (lock-step reference registrations and
   free-running coefficient batches).  106 checks.
8. `tests/nvrtc_check.c` — the CUDA kernels compiled by NVRTC with the
   backend's own preamble and options, for every architecture the NVRTC at
   hand supports, and every kernel the backend looks up present in the PTX.
   No GPU or driver needed, only `libnvrtc` (`AEG_NVRTC_LIB=`).
9. `tests/run_tests.sh` — sixty-nine cases (QP 12/23/26/40, presets 0-4,
   deblocking on/off, 64x64 … 320x208, forced CPU backend, 16x8/8x16/P_8x8
   partitioning with all four `sub_mb_type` shapes, both the integer and the
   quarter-pel motion search, the forced whole-frame search seed *and* the
   forced whole-frame fractional refinement, all-intra and 3-frame GOPs, the RD
   Intra_4x4 decision of the slow presets, P_Skip, ABR, AQ, AQ over
   P_Skip runs at two QPs, and level selection on both transforms, through the
   slice threads and through the batched coefficient path, the vector
   refinement of both a 16x16 and a 16x8 winner -- all of them with the 8x8
   transform and CABAC enabled, which are the defaults; thirteen of them with B
   pictures, one, two and three between two anchors, at QP 12 and 51, on a
   still clip, with a short GOP, without deblocking, without the 8x8 transform,
   under the rate controller, one with B_8x8 at preset 3 and one with a
   `B_L0_Bi_8x16` macroblock, whose mb_type is 13 -- the same number the CABAC
   tree uses for its intra branch)
   encoded, decoded with ffmpeg and compared **frame by frame**, then an ABR
   accuracy check: the encoder's `--dump-recon` must be bit-identical to the decoder's
   output.  Any mismatch fails the script; a decoder that logs an error also
   fails it.  `tests/genyuv.c` takes a mode argument so that the partitioning
   cases really move the two halves of the picture in opposite directions —
   the default clip pans as one block and can never make a partition pay.
   The script also asserts the skip statistics on a still clip (74% of its
   macroblocks, 85% of the P ones, must be skips), because a mode decision that
   stops finding `P_Skip` produces a conformant stream that is 45% larger, and
   no bit-exactness check can see that.  The encoder reports the full type
   breakdown (`skip`, `P16x16`, `P16x8`, `P8x16`, `P_8x8`, the three intra
   types), which is what the gap analysis below is read from: on the
   two-quadrant clip at QP 28 it codes 64% P16x16, 24% skip, 5.5% Intra_16x16,
   3.5% 8x16 and 2.0% P_8x8 -- i.e. the content is genuinely mostly 16x16 (the
   picture translates as two halves of 160 px each, so only the two columns at
   the boundary need a split), and the 20% of rate the clip still costs against
   x264 is in the residual and in the B pictures x264 codes and this encoder
   does not, not in the partition decision.

`reconstruction vs decode` must be `max abs diff 0`.

The B cases cannot compare the two files frame by frame as they are: the encoder
hands its pictures out, and dumps them, in *coding* order, while a decoder
outputs in display order.  `run_bcase` therefore rebuilds the coding order the
scheduler produced -- `[0]`, then per group either `[anchor, B...,]` or, when the
group ends in an IDR, `[B... (coded as P), anchor]`, then the tail in display
order -- and compares each dumped frame against the decoded frame at that display
index.  A B picture whose direct derivation is wrong reconstructs a different
picture *without* breaking the syntax, so this re-ordering is the only thing in
the suite that sees the four B bugs listed below.

The B cases run the default partition mask, so they exercise the partitioned
B types as well: at QP 24 on the panning clip the type breakdown reports
`B16x8`, `B8x16` and `B_8x8` macroblocks, and the two-quadrant clip selects
`B8x16`.
That is how the neighbour-mask bug in *The partitioned types* was narrowed down
-- the first differing picture was an anchor, not a B one, so the partitioning
in the B picture was working and the failure came from the *P* path using the
same mask.

Six cases failed from `db80149` on and were not a B-picture regression:
`refine_thr`, `refine_part`, `trellis4`, `trellis8`, `trellis_thr` and
`trellis_batch`.  They all have `--refine-mv` on, and the drift started at a
16x8/8x16 macroblock: the refinement moved one partition's vector, which changes
the *predictor* of the next one, so the mvd sent no longer matched what the
decoder derived (bug 43).  `--partitions 1` with the refinement was bit-exact
and `--partitions 5` was not, which is what pointed at the second partition
rather than at the coder.  Found by comparing the encoder's own block table with
the decoder's exported vectors: at the first bad macroblock the coded mvd was
the same on both sides but the predictor it was added to differed by exactly the
refinement step, and the first bad macroblock was always in column 0 -- the only
place where a 16x8's second partition has no left neighbour and falls back to
the first partition's vector.  With `refresh_pmvs()` in place every case is
exact: the whole gate (sixty-nine cases) is bit-exact, plus a 216-case sweep of
`--refine-mv 1/2` x `--partitions` 5/7/9/11/13/15 x preset 2/4 x QP 20/26/32 on
three clips.

The same refinement is also why preset 3 used to fail at `--partitions 3/5/15`
with B pictures: `-p 3` sets `--refine-mv 1`.  Preset 3 decodes bit-exactly now
for `--bframes` 1/2/3, QP 12/24/37 and every `--partitions` value, with the
refinement left on -- that is how B_8x8 was verified.

One more failure was found and fixed while isolating the refinement bug, and it
was a *syntax* bug of the nastiest kind: **mb_type 13 is B_L0_Bi_8x16, and the
four-bit value 13 in the CABAC mb_type tree means intra**.  The writer's
mb_type table is indexed by Table 7-14, where every partitioned shape is a
pair (`12`/`13` = B_L0_Bi_16x8/B_L0_Bi_8x16), and it had a special case that
sent the index 13 to the intra branch -- correct-looking, because the intra
branch *is* reached by the value 13, but the value is a code, not an index.  So
a B_L0_Bi_8x16 macroblock went out as an intra one.  Only the 8x16 shape can
hit it (its 16x8 twin is 12), which is why `--partitions 5` and `13`
desynchronised while `--partitions 7` did not, and why the thing looked like a
deblocking problem: the decoder did not reject the slice, it decoded garbage
from that macroblock on, and `--no-deblock`/`--no-8x8dct` merely changed the
decisions enough to avoid selecting the type.  The decoder's own log says what
happened ("left block unavailable for requested intra4x4 mode", then "error
while decoding MB 0 5"), and the way to read that message is: the *syntax* is
wrong, not the filter.  The sentinel is now `CAB_B_TYPE_INTRA = 23`, an index
the writer can never produce, and the gate has a case of its own
(`bframes_8x16`, three B pictures at QP 24 with 8x16 enabled).

With that fixed, the matrix that used to fail is exact: 630/630 bit-exact
decode cases -- three clips x `--bframes` 1/2/3 x QP 12/20/24/28/37 x
`--partitions` 1/3/5/7/9/13/15 x preset 2/3 with `--refine-mv 0` -- and no
decoder errors anywhere.

With a GPU present, `tools/gpu_validate.sh clip.yuv W H FRAMES QP` runs the same
comparison over every device code path at once (CPU stream against dispatcher
stream against each forced batched path, byte for byte where the path is the
same algorithm, by reconstruction where it is not).  A nonzero diff means the
encoder and the decoder disagree about samples, which drifts on every
following P picture.

When a diff *is* nonzero, the fastest way to localise it is to compare motion
vectors, not pixels: `AEG_MBDBG=1 ./build/aegenc ...` prints one line per coded
P_8x8 macroblock (`F<frame> MB x,y st <sub_mb_type x4> [b<block> mv dx,dy pmv
dx,dy]...`), and a throw-away FFmpeg build with two lines of `fprintf` in
`decode_mb_cavlc()` prints the predictor/length pair it derived for the same
sub-partitions.  Diffing the two listings pinpoints the neighbour that is being
read differently (bug 27 was found exactly that way, in one run, after the
pixel-level symptoms had been narrowed to a single macroblock).  The FFmpeg
instrumentation is deliberately *not* in this repository; only the encoder-side
side `AEG_MBDBG` hook is.  `tools/mvdec_probe.md` records the exact recipe.

Manual single-macroblock probe (used to isolate transform/quantisation bugs):

```sh
./build/aegenc -w 16 -h 16 -q 33 -n 1 -i mb0.yuv -o v.264 --dump-recon vr.yuv
ffmpeg -v error -i v.264 -f rawvideo -pix_fmt yuv420p vd.yuv -y
# vr.yuv and vd.yuv must be identical
```

## CABAC

CABAC is the default entropy coder (`--no-cabac` switches to CAVLC) and it codes
the whole macroblock layer: the skip flag, both mb_type trees, every
sub_mb_type, mvd, the intra prediction modes, the chroma mode, the coded block
pattern (luma and chroma), transform_size_8x8_flag, mb_qp_delta, and the
residual block in all six categories (Intra_16x16 DC and AC, luma 4x4, chroma
DC and AC, luma 8x8) with its significant/last/level-prefix contexts.  On the
panning clip at equal PSNR it costs 34-36% fewer bytes than this encoder's
CAVLC path.

`src/cabac.c` is the arithmetic coder (`tests/cabac_check.c` checks it against
a decoder written from 9.3.3 rather than from the encoder); the binarizations
and context assignments live with the syntax writer in `src/encoder.c`
(`write_mb_cabac()` and the `cabac_*` helpers above it), and
`tests/cabac_mb_check.c` round-trips the residual block syntax through a second
independent decoder.

The parts of the *syntax* layer that are easy to get wrong, in the order they
bite:

* **Context indices are read from the neighbours, so the encoder has to store
  what it decided, not what it did.**  Three bugs (38, 39, 40) came from
  storing the internal value instead of the coded one: the chroma pattern of a
  neighbour is the syntax element's 0/1/2 and not a packed pattern field, the
  chroma prediction mode context adds *one* per available neighbour, and
  transform_size_8x8_flag is only non-zero for the macroblock types that
  actually code it.  In each case the arithmetic coder was right and the
  macroblock decoded to something else.
* **`mb_qp_delta` is unary on an unsigned value, and the terminating zero of
  the unary code is read with context 2 only when the value is 1** (9.3.2.7);
  every later bin uses context 3.  Coding `2|d| - 1` for a positive delta and
  `2|d|` for a negative one is the mapping, not `2|d|` and `2|d| - 1`.
* **A skipped macroblock clears the previous delta's context.**  The decoder
  resets its "last mb_qp_delta was non-zero" flag on every skipped macroblock,
  because the syntax element is absent; an encoder that keeps the flag
  desynchronises the *next* macroblock's first bin.  This was invisible until a
  clip combined AQ (so the delta is non-zero somewhere) with P_Skip runs -- the
  `aq_skip` case in `tests/run_tests.sh` exists for it.

The way to find a divergence is not to stare at pixels but to compare the two
decoders' bin streams: `AEG_CABAC_TRACE=1` makes the encoder print every bin it
codes (`B <ctx> <bin> <range>`, `Y <bit>` for a bypass bin, `T <flag>` for
end_of_slice_flag) and one `ENCMB x,y ...` line per macroblock with the
decisions the contexts are derived from, and a reference decoder instrumented
at `get_cabac` prints the same `B` lines with the context it read.  Diffing the
two lists gives the exact bin, the syntax element and the macroblock; bug 40
was a single context index, 107 173 bins into a 320x240 picture.

Four things about the coder are worth recording, because each of them is
invisible until a whole slice decodes:

* **The state is 0..127, not 0..63.**  The stored byte is the spec's
  preCtxState compressed: bit 0 is the MPS and `state >> 1` is the `rangeTabLPS`
  row.  That is what makes one table (`aeg_cabac_trans[state][bin]`) serve both
  MPS and LPS transitions, and it is why the initialization clamps the
  pre-state to 1..126: the values 0 and 1 (the exactly-0.5 state, whose LPS
  range is 2) are unreachable and the transition table keeps them that way.
  The coder still handles them -- `aeg_cabac_renorm()` loops -- which is what
  lets the test drive all 128 states.
* **Carry handling is the part that has to be right.**  A decided byte can
  still be changed by a carry out of the next one, and a run of 0xff bytes can
  turn into 0x00.  The newest byte is therefore held in `prev` and written only
  when the next extraction shows whether it needs a carry; 0xff bytes are
  counted in `outstanding` and written (as 0x00 if the carry comes, 0xff if it
  does not) at the same moment.
* **The encoder runs nine bits ahead of its own output.**  A decoder pre-loads
  nine bits before its first bin, so a byte is only fully decided once nine
  more bits have gone into the interval.  That is the `queue = -9` of
  `aeg_cabac_open()`, and it is also why `aeg_cabac_flush()` writes nine extra
  bits and a stop bit at the end.
* **`end_of_slice_flag` is present after every macroblock**, 0 for all but the
  last (9.3.4.4), so the syntax writer needs `aeg_cabac_terminate(c, 0)` per
  macroblock and exactly one `aeg_cabac_flush()` per slice.

**Pricing the decision pass with the coder.**  `write_mb_cabac()` is now the
rate model as well as the writer: `rd_cabac_bits()` runs it into a copy of the
slice's coder state (`e->cab_shadow`) with the output switched off
(`aeg_cabac.cost`), so every bin a candidate would emit advances the real
context state and is charged what that state charges -- `aeg_cabac_bin_cost
[state][bin]`, in 1/256 bit, derived from the same `rangeTabLPS` the coder uses
(`256 * -log2` of the mean LPS probability of each range class, so a fresh
context is close to one bit and a settled one costs a fraction).  A bypass bin
is exactly one bit; `end_of_slice_flag` is charged a constant, since it cannot
change a decision.  The decision compares `SSD + lambda * cost` where CAVLC
still compares its `aeg_cavlc_block_cost()` estimate, which is what that
estimate models.

Three details make the shadow coder a faithful copy rather than an
approximation:

* **The state that `write_mb_cabac()` mutates outside the coder has to travel
  with it.**  `mb_qp_delta` needs the previous macroblock's QP and its "last
  delta was non-zero" flag, and the motion-vector predictor of the *next*
  macroblock reads the mvd field the writer stores.  The pass therefore keeps
  its own `rd_last_qp`/`rd_last_dqp` alongside the shadow, and a candidate that
  does not win restores the mvd array (`mbs[idx].mvd`) before the next
  candidate is priced.
* **A candidate is priced exactly as the bitstream pass will code it**, so
  `cbp == 0` reuses the previous QP (there is no delta to code) and the skip
  test is the same `P16x16 && cbp == 0 && mv == pskip_mv()` the writer uses.
  The bitstream pass then starts from the last committed decision -- the
  winner of the macroblock -- which is why it resets `last_qp`/`last_dqp` to the
  slice's QP at its own start: the decision pass's history is not the
  bitstream's.
* **The winner of the macroblock is committed once** (`commit = 1`), not once
  per candidate: the candidates all price against the same incoming contexts,
  which is what makes their costs comparable.

`tests/cabac_check.c` encodes 1.6 M bins per `make test` -- context-coded bins
over all 460 contexts from every initialization table at QPs 0..51, bypass bins,
runs in both directions, and `end_of_slice_flag = 0` bins between them --
decodes them with the spec's decoder, requires every bin and every context
state to match, and then requires the final terminate to decode as 1.
`tests/cabac_mb_check.c` does the same for one residual block at a time: 20 000
random blocks over the six categories with levels that reach the Exp-Golomb
escape, decoded by a second transcription of 9.3.2.2 that reads the byte stream
itself.  Neither needs a GPU or an external decoder, which is the point: a
CABAC bug produces a legal stream that reconstructs a different picture, so it
has to be caught before an external decoder is the only witness.

38. **The chroma coded block pattern a neighbour contributes is the syntax
    element, not the packed pattern.**  `coded_block_pattern_chroma` is 0, 1 or
    2, and the context of the next macroblock's chroma pattern is built from
    those two values.  The macroblock struct stores the *packed* pattern
    instead (0x10 = chroma DC only, 0x20 = chroma AC), so `(cbp >> 4) & 3` gave
    3 for a neighbour with AC coefficients: a context the standard never uses,
    i.e. a probability state that is not the one the decoder is holding.
    `packed_chroma_cbp()` maps the field back to the element (0x20 -> 2,
    0x10 -> 1) and the neighbours go through it.

39. **The chroma prediction mode context uses the top neighbour.**  The
    condition is `(left available && left mode != DC) + (top available &&
    top mode != DC)` (9.3.2.4), i.e. both neighbours add *one*, giving the
    three contexts 0..2.  A first version added 2 for the top neighbour (as the
    luma 4x4 mode coding does for its own context), which walked into context
    64+3 -- the context of the second bin -- and desynchronised every
    macroblock that had a coded top neighbour.

40. **`transform_size_8x8_flag` is only non-zero for the macroblock types that
    code it.**  A macroblock's residual may be transformed in 8x8 blocks
    internally (Intra_8x8 and inter), but the flag is a syntax element of
    Intra_NxN and of an inter macroblock with a luma pattern; the neighbour
    context of the *next* macroblock is the coded flag, not the internal
    choice.  Keeping the internal value made P16x16 macroblocks look like they
    had set the flag, which shifted context 399 + 2 entries and flipped a bin
    107 173 bins into a 320x240 picture.  `mb_t8_coded()` is now the only
    producer of the value the neighbours see.

41. **A coder field added for the rate model has to be initialised by
    `aeg_cabac_open()`.**  The shadow coder sets `cost = 1` by hand, but the
    *bitstream* coder is opened by `aeg_cabac_open()`, and when `cost` was added
    to the struct without being initialised there, the slice stack frame that
    happened to inherit a non-zero byte priced a macroblock instead of writing
    it.  Serial encoding was unaffected (the first slice's frame was zeroed),
    so the whole thing only showed up as `--slices N -t N` writing a five-byte
    second slice -- the decision pass was identical, byte for byte, which is
    what pointed at the writer rather than at the rate model.  `aeg_cabac_open()`
    now clears both price fields.

## B pictures

42. **mb_type 13 is B_L0_Bi_8x16 and the CABAC intra branch is the four-bit
    code 13.**  The writer's mb_type table is indexed by Table 7-14, where a
    partitioned B shape is a pair of indices (`12`/`13` = B_L0_Bi_16x8 /
    B_L0_Bi_8x16); the tree's intra branch, on the other hand, is reached by
    the four-bit *value* 13, which is not an index at all.  A special case on
    the index 13 sent the macroblock down the intra branch, so a
    B_L0_Bi_8x16 went out as an intra macroblock: the decoder parsed an intra
    mode list where the residual of a bi-predictive 8x16 partition should have
    been and gave up ninety macroblocks later ("left block unavailable for
    requested intra4x4 mode", "error while decoding MB 0 5").  Two things made
    this hard to see: only the 8x16 member of every pair can collide (its 16x8
    twin is 12), so it needed `--partitions 5`/`13` to appear at all; and
    turning the deblocking filter or the 8x8 transform off made the case pass,
    because that changed the mode decisions enough to stop selecting the type
    -- a "the filter is wrong" signature over a syntax bug.  Intra now has the
    sentinel `CAB_B_TYPE_INTRA = 23`, and the gate runs a case that selects the
    type (`bframes_8x16`).
43. **A predictor derived from a vector that later changes has to be
    re-derived.**  The motion-vector refinement re-codes the winning inter
    macroblock with each of the quarter-pel neighbours of the winner's vector,
    one partition at a time.  A 16x8 or 8x16 macroblock's *second* partition
    predicts from the first one (8.4.1.3.1: the decoder has just written those
    vectors into its block cache), and that predictor had been derived when the
    search found the first vector -- so as soon as the refinement moved it, the
    stored predictor was off by exactly the refinement step while the decoder
    derived the new one, and the two reconstructions diverged from that
    macroblock on.  Two properties kept this hidden: it needs a partitioned
    macroblock (16x16 has no neighbour inside itself, so `--partitions 1` is
    clean and `--partitions 5` is not), and it needs the refinement to *accept*
    a move, which is why `--no-8x8dct` and `--no-deblock` made it vanish -- they
    changed the residual enough that no move paid.  Six gate cases had failed
    from `db80149` on because of it (`refine_thr`, `refine_part`, `trellis4`,
    `trellis8`, `trellis_thr`, `trellis_batch` -- trellis is reached through the
    same presets, so its cases inherited the failure).  `refresh_pmvs()`
    re-derives every partition's predictor from the vectors the macroblock
    finally carries, after the refinement has run.
44. **A refinement must not move what the bitstream does not carry.**  The
    same refinement applied to B pictures initially included `B_Direct_16x16`,
    whose vectors are *derived* from the co-located picture rather than coded.
    Moving one changed the encoder's reconstruction and nothing else, so every
    B picture desynchronised from its first macroblock on -- and it looked like
    a predictor bug (the decoder derived 12,8 where the encoder predicted with
    11,7) even though the predictors were right.  The rule: a refinement may
    only touch syntax that exists.  Direct macroblocks are excluded by type, and
    the case that keeps them out is `bframes_refine`.

`--bframes N` (0..3) codes N B pictures between two anchors; the default is
**three**, with `--bqp 8` (see *Making them pay* below) -- this is the one place
where a default changed because a measurement said so.
The stream has `pic_order_cnt_type 0` with an 8-bit `pic_order_cnt_lsb` per
slice, `max_num_ref_frames 2`, and `direct_spatial_mv_pred_flag = 0` (temporal
direct, so a direct macroblock's vectors come from the co-located picture and
never from its neighbours).  The encoder holds `bframes + 3` input pictures of
lookahead, codes a group as *anchor, then its B pictures*, and emits each
picture as its own packet; `aeg_encoder_flush()` drains the tail, where the
group's remaining pictures can only be coded as P pictures because their
anchor never arrives.  B macroblocks are `B_Direct_16x16`, `B_L0_16x16`,
`B_L1_16x16`, `B_Bi_16x16` and the two partitioned shapes `B_16x8` and
`B_8x16`, whose partitions pick their lists independently (codes 4-21, see
*The partitioned types* below), and `B_8x8` (code 22, see *B_8x8* below); the
bi-predictive average is `(a + b + 1) >> 1`, rounded once on the interpolated
samples.  That is the whole B macroblock layer.

Four bugs stood between "it writes plausible B slices" and "the decoder
reconstructs the same picture", and all four were *syntax*-level in the sense
that the stream decoded without an error report while the picture was wrong --
which is why the two-pass md5 comparison in `tests/run_tests.sh` is what found
them and nothing else did.

1. **The B `mb_type` context counted unavailable neighbours as non-direct.**
   The context is `!IS_DIRECT(left - 1) + !IS_DIRECT(top - 1)`, in which the
   reference decoder's *unavailable* neighbour (stored as 0) satisfies
   `IS_DIRECT(-1)`: only a neighbour that is there and is not direct increments
   the context.  The encoder counted both missing neighbours, so every
   macroblock of the first row/column used the wrong context and from the first
   bin onwards the two sides read different symbols.
2. **The B `mb_skip_flag` context was derived from the wrong state.**  It is
   the *same* rule as in a P slice (`+13` for the B slice's own block of
   contexts, then `!skipped` of the left and top neighbours), not a comparison
   of the neighbours' types against direct mode.  With the wrong context the
   skip flag decoded as the wrong symbol, which inverts everything that
   follows.
3. **The deblocking filter ignored the second prediction list.**  For an edge
   between two B macroblocks, 8.7.2.1 compares the reference indices of *both*
   lists and the motion vectors of the lists that are used; the filter is on as
   soon as one macroblock predicts only backwards and the other only forwards,
   even with both vectors at zero.  The filter was derived from list 0 alone.
4. **Direct mode read the wrong co-located vector.**  `B_Direct_16x16`'s own
   type entry carries no 16x16 shape bit, and the reference decoder's
   partitioned path takes one vector per 8x8 group from the co-located
   picture's 4x4 blocks 0, 3, 12 and 15 -- not from the macroblock's first
   block, and not from each group's bottom-right corner.  The two readings only
   disagree when the co-located macroblock was partitioned, which is most of
   them on moving content; the encoder reconstructed a different picture than
   the decoder without desynchronising the stream.

### The partitioned types

`B_16x8` and `B_8x16` are not types of their own in the standard.  Table 7-14
assigns a code to each *pair* of partition list masks, so the B `mb_type`
codes 4-21 are the two shapes crossed with the nine ways two partitions can
pick from {L0, L1, bi}.  The encoder keeps one `AEG_MB_*` for both shapes
(`AEG_MB_B16x8`, `AEG_MB_B8x16`) with the two masks in `blist[0..1]`, and
`b_mb_type_code()` turns them into the code:

|              | part 1 L0 | part 1 L1 | part 1 bi |
|--------------|-----------|-----------|-----------|
| **part 0 L0**|  4 / 5    |  8 / 9    | 12 / 13   |
| **part 0 L1**| 10 / 11   |  6 / 7    | 14 / 15   |
| **part 0 bi**| 16 / 17   | 18 / 19   | 20 / 21   |

(the second number of each pair is the 8x16 shape; codes 12-21 are the ones the
CABAC binarization spends a fifth bin on).  Each partition picks its lists on
its own -- list 0, list 1 and bi-prediction are priced per partition by the
partition's own SATD plus eight times its own mvd bits -- and the macroblock as
a whole still goes through the exact RD comparison, which is what decides
whether the finer motion field pays for the extra `mb_type` and mvd bins.

Two properties of the *syntax* decide whether the reconstruction matches, and
both are per list rather than per macroblock:

* the differences go out list-major: every partition's list-0 mvd first, then
  every partition's list-1 mvd (the partition order is the bitstream's, top to
  bottom or left to right);
* a partition that does not predict from a list sends nothing for that list,
  and the decoder leaves that partition's vector *and* its difference at zero --
  so `cabac_mvd()` skips the pair but `cabac_mb_mvd()` still publishes the zeros
  in the per-block mvd table, because the *next* macroblock's context is built
  from it, and `nb_from_mb()` has to treat that block as LIST_NOT_USED rather
  than as a neighbour whose vector happens to be zero.

**The one bug this cost was in the neighbour mask, not in any of that.**  The
per-block list mask `aeg_mb_block_lists()` was written for the deblocking
filter, whose only requirement is that two macroblocks of the *same* picture be
comparable, and its `default:` returned 0 -- "this block predicts from no
list".  P macroblocks, `P_Skip` and the whole P path fall into that default, so
in a *P* picture `nb_from_mb()` asked the mask about a P neighbour, got 0, and
concluded that the neighbour used no list at all: every partitioned P
macroblock's predictor came out as (0, 0) while the decoder's was the real
median, and the two pictures diverged by the mvd.  In a B picture nothing
showed, because there every neighbour is a B type.

It was found the way these things are found: the first differing frame of a
`--bframes 2 --partitions 3` encode was *not* a B picture at all but the anchor
in front of the first B group, the differing macroblocks sat under the few
`P16x8` macroblocks of that frame, and dumping `mv`/`pmv` per partition showed
a partition whose predictor was (0, 0) with a top neighbour at (36, 24) and a
left neighbour at (-36, -24).  With the P types mapped to list 0 in the mask,
`--partitions 1 3 5 7 15` are all bit-exact with one, two and three B pictures,
at every QP and every preset, with slices, and on the 64x64 clip.

### B_8x8

The last B macroblock shape is code 22 plus one `sub_mb_type` per 8x8 group,
read through contexts 36-39.  Thirteen sub-types, and the tree is not shaped
the way the numbering suggests: after "is it direct" (36) and "is it one of the
two 8x8 list predictions" (37) the third bin (38) separates the 4x4 values
(11, 12) from everything else, and the fourth (39) then splits *the bi 8x8 and
the 8x4 values* (3-6) from the 4x8 ones (7-10) -- so `code_b_sub_mb_type()`
carries a comment about each branch, because the numbering that comes out is
0, 1/2, 3, 4-10, 11/12 rather than anything monotone.

Only the four 8x8 sub-types (direct, L0, L1, both) are *searched*.  The 8x4,
4x8 and 4x4 splits would triple the work for a shape whose value a B picture
has already spent: the two lists absorb most of the freedom an extra split
would buy, and the splits only matter where the two references disagree inside
8x8, which the 16x8/8x16 shapes already cover.  The writer emits all thirteen
values, so adding them later is a search change and not a syntax change.

Each group picks its own prediction out of direct, list 0, list 1 and both, and
the groups are searched in raster order so that a following group's predictor
sees the vectors already decided (the same per-list cache the partitioned
shapes use).  A group coded as `B_Direct_8x8` sends no mvd at all, which is why
`mb_part_sends_mvd()` exists and why `cabac_mb_mvd()` still publishes the zeros
in the per-block table: the *next* macroblock builds its contexts from it.

**It measured neutral, and that is why it is behind the preset.**  With
`--bframes 3 --bqp 12` on five clips at six QPs, enabling B_8x8 moves the
BD-rate by -0.01, -0.10, -0.01, -0.00 and -0.07 points -- i.e. nothing -- while
costing eight extra integer searches per B macroblock and only eight of 14 400
macroblocks select it.  Re-measured after the B pictures' vectors started being
refined (*The same refinement in a B picture*) it is *exactly* neutral -- the
BD-rate of preset 3 with and without it agrees to two decimals on four clips --
because two macroblocks in 2400 choose it, and the cost is then 1-3 % of the
encode time rather than a tenth.  It is therefore
enabled from *preset 3*, where the encoder is already spending its time on
refinement, and the gate has a case of its own (`bframes_8x8`, preset 3 with
the P partitions held at 16x16 so that the case is about the B macroblock layer
rather than about the motion search).

**The two bugs it cost were both in what the encoder publishes, not in the
syntax.**  The first was the copy the decision pass makes into the `aeg_mb` grid
that the deblocking filter and the following macroblocks read: it had been
written for the partitioned shapes and moved `blist[0..1]` only, so the bottom
half of every B_8x8 was published with "no list", which
`aeg_mb_block_lists()`'s fallback then read as *both* lists.  Ten samples of one
frame differed and `--no-deblock` was bit-exact: the filter alone disagreed
about the boundary strength of two 8x8 groups.  A ten-sample mismatch with a
maximum difference of 2 is what a boundary-strength disagreement looks like; a
motion vector or residual error moves whole blocks by much more.

The second was the per-list motion-vector cache a B_8x8 builds across its own
four groups, and it is the more interesting one because the *state* it was
missing is a third state.  `predict_mv_at()` takes the vectors of the groups
already coded in this macroblock from `cur[]`, with an `avail[]` flag saying
which of them the decoder has filled in yet.  That flag was a bit per list,
and `avail = 0` meant "not decoded yet" -- but a group that is coded and simply
does not predict from this list (an 8x8 group coded list 0 only, asked about
list 1) is *also* zero in that table, and the two are not the same thing to the
decoder: the first is `PART_NOT_AVAILABLE` and is skipped by both the fallback
rule of 8.4.1.3.1 and the median, while the second is `LIST_NOT_USED`, which
counts as available and contributes its zero vector.  So as soon as a group was
coded from one list and the next group was searched for the other, our predictor
came from a different neighbour set than the decoder's and the mvd no longer
matched -- 528 wrong samples, maximum 48, on the first B_8x8 that mixed its
lists.  `avail[]` is now a three-state field (`AEG_MVBLK_AVAIL` /
`AEG_MVBLK_NOTUSED` / `AEG_MVBLK_UNAVAIL`) and `nb_from_cur()` maps it to the
predictor's `NB_OK` / `NB_NOTUSED` / `NB_UNAVAIL`.  The lesson generalises: any
per-block table shared with the decoder needs the decoder's *three* cache states,
not a boolean, and "coded but not from this list" is the one that gets lost.

Verification: 630/630 decode cases bit-exact (three clips x `--bframes` 1/2/3 x
QP 12/20/24/28/37 x `--partitions` 1/3/5/7/9/13/15 x preset 2/3), all with
`--refine-mv 0` so that the matrix isolates the B layer from the P-side
refinement; the refinement itself is exact (bug 43) and was swept separately at
`--refine-mv 1/2`.  That matrix is also what found bug 42.

### Making them pay

B pictures on their own were a *loss*: four 16x16 shapes with no partitions and
no `B_8x8` save less in prediction than they spend in syntax, and the anchors sit
three frames further apart.  Two things turned that around, and neither needed a
new syntax element.

**A coarser quantiser for the non-reference pictures.**  A B picture is never
used as a reference, so its quantisation error cannot propagate: rounding it
harder costs nothing to the pictures that follow, which is why x264 spends a
`bframes`-dependent offset here by default.  `--bqp N` sets it; the default is
**12**, not the +2 the standard advice suggests.  The first sweep, with
`--bframes 2`, `--scenecut 0`, `tools/bdrate.py` against libx264 medium on the
panning clip / the quadrant clip / the eight-frame one (negative = we spend fewer
bits than x264), found 8, and predicted that the offset would have to *come
down* once B partitions existed -- the reasoning being that the two 16x16
shapes were what the offset was paying for.

| `--bqp` | g1_60 | p2 | q4 | (first sweep, `--bframes 2`) |
|---|---|---|---|---|
| 0 | +19.7 % | +17.2 % | +17.2 % |
| 2 | +7.4 % | +5.6 % | +8.5 % |
| 4 | +3.3 % | +1.9 % | +5.9 % |
| 6 | +1.1 % | -0.0 % | +4.6 % |
| 8 | -1.1 % | -1.6 % | +4.0 % |

The prediction was wrong in the interesting way: with the partitions in place
the optimum moved *up*.  Better prediction makes the residual cheaper, so a
still coarser quantiser costs less in bits than it did, and the second sweep
(with `--bframes 3`, the partitioned types, six QPs, and the offsets compared
against each other over the range the two curves actually share) has a flat top
at 12-14:

| `--bqp` vs 8 | g1_60 | p2 | p0 | p3 | q4 |
|---|---|---|---|---|---|
| 10 | -1.25 % | -0.85 % | -- | -- | -- |
| **12** | **-2.07 %** | **-1.34 %** | **-3.20 %** | +0.14 % | +0.06 % |
| 14 | -2.19 % | -1.00 % | -- | -- | -- |
| 16 | -1.14 % | +0.30 % | -- | -- | -- |

Twelve sits inside the flat top of both clips and costs nothing on the two
static ones (a static picture has no B residual to quantise at all).

**The offset fades out towards QP 0.**  A QP step is a *relative* change in
quality, and a fixed number of steps is only worth paying while there is room to
pay for it.  At QP 4 an offset of twelve puts the B pictures at the quality of
QP 16 -- plainly visible -- and the bits it saves cannot be spent on anything,
because the anchors are already at QP 0.  This is not a theory: it is what broke
the rate controller.  At 400 kb/s on the ABR clip the B pictures became cheap
enough that the whole stream landed 17 % *under* the target with every anchor at
QP 0 and nothing left to spend the budget on.  The offset is now scaled by
`min(1, qp / 12)`, which leaves every measurement above untouched (they are all
taken at QP 22-37) and puts that stream back at 419.7 kb/s.

**The rate controller, the buffer model and the scene-cut detector had to learn
what a B picture is.**  All three were written for a stream of interchangeable
pictures:

* the controller gave every picture a full frame's worth of budget.  A B picture
  cannot spend that (it is quantised eight coarser on purpose), so the controller
  kept lowering the QP to chase a target the picture type cannot reach, and the
  stream landed 14 % under the requested rate while the anchors got finer than
  they should.  Each picture's share is now a *measured* weight: the running ratio
  of B-picture bits to anchor bits (an EWMA, seeded from the QP offset's
  `2^(-bqp/6)`), so one group is worth exactly the pictures in it.  400 kb/s at
  `--bframes 3` now lands at 419.7 kb/s rather than 344;
* the debt clamp and the per-picture QP step are measured in *this picture's*
  budget, and the step cap grows with the outstanding debt (up to 4 QP).  With a
  group of four, the whole group's error is repaid through the anchor alone, and
  1.5 QP per picture would not converge inside a short clip;
* the buffer model predicted each picture from whatever was coded last.  A B
  picture is far cheaper than the anchor before it and far cheaper than the one
  after, so the prediction swung by a factor of several and the clamp fired on a
  buffer that was nowhere near empty; a B picture now predicts from the last
  *anchor*.

**The scene-cut detector was measuring the wrong distance.**  It compared the
picture being decided against its reference.  Without B pictures those are one
frame apart, which is the whole point of the metric; with three B pictures the
reference is the anchor four frames back, and a fast pan over four frames has the
same `|cur - ref|` as a cut between two unrelated clips -- the ratio test stopped
being able to tell them apart (2.2 for ordinary motion where it used to be 0.7).
It now compares two consecutive *source* pictures from the lookahead ring, which
exists whether or not the stream has B pictures.  And because the decision moved
to `schedule()`, it no longer has to fire on the anchor alone: every picture of
the group is in the ring by then, so the group is cut short at the picture that
actually is the cut and becomes its IDR.  Testing only the anchors missed a cut
that fell between two of them -- the next anchor is already past it, and compares
two pictures on the far side of the cut.  The gate now checks the IDR count with
and without B pictures.

**`--bench` counted packets instead of pictures**, so a B-picture stream reported
one "frame" per configuration (the first packets are `AEG_EAGAIN` and the rest
trail the input); it feeds a picture per iteration and drains the lookahead.

### The result

`tools/bdrate.py` against libx264 medium, 6 QPs, negative = we spend fewer bits
than x264 at matched PSNR:

| clip | B pictures off | `--bframes 3 --bqp 8`, 16x16 B | **default (`--bframes 3 --bqp 12`, partitioned)** |
|---|---|---|---|
| g1_60 (60 frames, panning texture) | +27.8 % | -8.2 % | **-10.3 %** |
| p2 (30 frames, two quadrants apart) | +24.7 % | -10.3 % | **-11.8 %** |
| q4 (8 frames, four quadrants) | +18.6 % | +4.4 % | **+4.4 %** |
| p3 (30 frames, still after the first) | -10.2 % | -21.5 % | **-21.4 %** |
| p0 (30 frames, whole frame pans) | +9.6 % | -19.1 % | **-21.9 %** |

Four of the five clips are ahead of x264's default preset, where P-only coding
was behind on three of them.  q4 is the exception and it is the measurement
rather than the encoder: its curve tops out at 46.9 dB where x264's reaches
47.4, so the BD-rate is integrated over 59 % of the reference's span and the
number moves by several points with the method.  Against the same reference
*at the same QPs* the partitioned types are worth 0.3 points on it (4.69 % ->
4.36 %), and the offset is worth nothing (a ten-frame clip at 400 kb/s has
nothing to spend it on).

What is left of the shape gap is `B_8x8` (code 22, sub-types 0-12, contexts
36-39): four 8x8 sub-macroblocks that each pick direct/L0/L1/bi *and* split
into 8x4, 4x8 and 4x4.  It is worth less than the 16x8/8x16 pair was -- the
pair is where most of a B picture's gain from partitioning lives -- but it is
the last type the standard has and the last thing between this encoder and a
full B macroblock layer.

## Two P references

`--ref 2` gives a P picture a second list-0 entry: list0[1] is the reference
picture before the previous one (the anchor before the previous anchor when
there are B pictures).  The slice header overrides `num_ref_idx_l0_active` to
two, the SPS keeps two reference frames, and no reordering syntax is needed --
the decoder's default list 0 is descending `frame_num`, which is exactly
[newest, older] for a two-entry sliding window.

What the encoder does with it is deliberately narrow: a **16x16 candidate on
the second reference**, with its own predictor and its own full quarter-pel
search, competes in the ordinary rate-distortion comparison.  The partitioned
shapes stay on reference 0.  They can do otherwise (the search, the predictors
and the syntax all handle a reference per partition, `part_ref_search()`), but
that pick is a SATD + 8 * bits decision that *replaces* the ref-0 vector rather
than adding a candidate, and it measured worse than not doing it (-0.5 %
against -1.2 % mean BD-rate, for more time).  When no macroblock of a slice
ends up on the second reference, the slice is written with one active
reference after all -- the header is now written after the decision pass, every
decision stays valid, and the ref_idx syntax disappears from every partition
(that alone moved the panning clip from +1.2 % to +0.6 %).

`tools/abbd.py` (ours against ours, no reference encoder needed), preset 3,
QP 22..37, 30 frames, negative = the second reference saves bits:

| clip | P-only (`--bframes 0`) | default (`--bframes 3`) |
|---|---|---|
| mode 0, whole frame pans | +0.57 % | -0.50 % |
| mode 1, halves apart vertically | **-2.87 %** | -0.44 % |
| mode 2, halves apart horizontally | +0.13 % | -0.81 % |
| mode 4, four quadrants | **-3.52 %** | +0.96 % |
| mean / encode time | **-1.42 %** / +13-18 % | -0.20 % / +5-9 % |

So it follows the preset (2 from preset 3 up) **only when there are no B
pictures**.  With three B pictures the anchors are four pictures apart and the
older reference eight, which is rarely the better prediction on any content,
and a B macroblock cannot use direct mode where the co-located block sat on
the older reference (see below), so the two effects cancel.  `--ref 2` forces
it either way and is conformant either way.  The synthetic clips are pure
translations, where an older picture has nothing a newer one lacks except a
different noise sample; real content with occlusion and repetitive motion is
where multiple references are known to pay, and this environment cannot show
that.

The feature had been "in" since the wiring commits, and the gate was green --
because the second reference was never actually coded.  Getting it to the point
where a macroblock on reference 1 reaches the bitstream and the decoder agrees
took six fixes, five of them silent (the stream parses, the picture differs):

45. **The candidate copy dropped the reference.**  `rd_luma()` copies a
    candidate's vectors and predictors into the macroblock it prices, but not
    its `ref[]`, so every candidate the search had put on reference 1 was
    priced -- and coded -- on reference 0, with the vector and the *predictor*
    derived for reference 1.  The predictor is what broke the reconstruction:
    a 16x8's top half predicts from the block above only when that block is
    on the same reference, so the decoder derived (20,0) where the encoder had
    stored the median (9,0).  The winner refinement had the same hole (it
    rebuilt its trial candidate from candidate 0 and overwrote the vectors
    only).
46. **`max_num_ref_frames` was 1 without B pictures.**  The SPS asked for two
    reference frames only when B pictures were on, so in a P-only stream the
    decoder's sliding window dropped the older picture as soon as the next one
    arrived, and ref_idx 1 pointed at a picture it no longer had.  FFmpeg
    conceals that with the picture it does have -- the decoded block was the
    newer reference at exactly the coded vector -- and logs nothing at
    `-v error`.
47. **ref_idx is not ue(v) and not truncated.**  With two active references,
    CAVLC codes ref_idx as te(v) with cMax 1, which is a single *inverted* bit
    (`!read_bits(1)`), and CABAC binarises it as plain unary whatever the
    number of references, so 1 is the two bins 1, 0 (the second at
    ctxIdxInc 4).  Both writers agreed with the decoder for 0 -- which is all a
    stream that never picked reference 1 contains.
48. **A macroblock on reference 1 was coded as P_Skip.**  The skip test was
    "16x16, no residual, vector equal to the skip vector", and P_Skip always
    predicts from reference 0.  The skip vector derivation itself (8.4.1.1)
    also has to look at the neighbours' references: only a neighbour on
    reference 0 *with a zero vector* forces (0,0), and the median follows the
    reference-matching rule.
49. **`cur_is_p` was one picture late.**  `encode_frame()` assigned it after
    `set_current_refs()`, which reads it, so every picture was set up with the
    previous picture's value: a B picture after a P anchor became a
    two-reference P picture with its two lists swapped, and the P anchor after
    a B picture got one reference.  P-only streams hid it -- the stale value
    was the right one -- and it is what made every preset-3 B case fail the
    moment two references were enabled with B pictures.
50. **Reference state survived an IDR.**  `picture_rotate()` kept both
    references across an IDR picture, so the P picture after a mid-stream IDR
    (`-g`, or a scene cut) took the picture before the IDR as list0[1] -- one
    an IDR has removed from the decoder.  An IDR now leaves itself as the only
    reference.

Two more were not about the second reference itself but only became visible
through it.  **Direct mode cannot follow a co-located block on the older
reference**: temporal direct takes the co-located block's reference picture as
its own list-0 reference, and the anchor's list0[1] is two anchors back, in
neither list of the B picture and no longer in the decoder's buffer;
`direct_group_ok()` keeps B_Skip, B_Direct_16x16 and B_Direct_8x8 away from
those 8x8 groups.  And **the picture padding was too small for the search
range**: the search bounds a vector by +-range around zero and knows nothing
about the picture edge, so preset 4's range of 48 read past the 32-sample
padding -- into the neighbouring row's padding instead of the clamped edge --
whenever a block at the border searched far enough out.  With one reference it
never happened on these clips; with an older reference the vectors double.
`AEG_PAD` is 64 and the search range is clamped to `AEG_PAD - 8`.

`AEG_MBDBG=2` prints one line per P macroblock with its type, each partition's
reference, vector and predictor, which is how all of the above were narrowed
to a single macroblock: diff the encoder's line against the decoded picture at
that macroblock, and the predictor or the reference that differs is the bug.

## Real clips (against libx264 medium)

Every earlier quality table in this file was measured on synthetic clips, and
two of the decisions taken on them did not survive real content.  Five real
clips (30 frames each, decoded from public sample videos: `cock` 640x352 with a
scene change and fast irregular motion, `vtest` 384x288 grainy surveillance,
`mega` 352x256, `short` and `tree` 320x240), QP 22..37, BD-rate of preset 3
against `libx264 -preset medium` (positive = more bits than x264 for the same
PSNR):

| configuration | cock | vtest | mega | short | tree | mean |
|---|---|---|---|---|---|---|
| defaults at `adf2ab9` (3 B, `--bqp 12`) | +51.74 | +10.61 | +11.40 | +13.42 | +12.04 | +19.84 |
| + rate term in the motion search | +24.66 | +13.44 | +3.59 | +4.26 | +11.22 | +11.43 (bqp 6) |
| + I offset 3, `--bqp 4` | +23.95 | +4.93 | +5.75 | +6.11 | +9.32 | +10.01 |
| + spatial direct | +22.77 | +5.11 | +6.80 | +5.30 | +8.95 | +9.79 |
| + level selection on intra blocks | +19.62 | +4.34 | +5.39 | +5.27 | +7.10 | +8.34 |
| + adaptive group length | +5.28 | +4.87 | +4.25 | -2.00 | +9.63 | +4.41 |
| + sub-pel search of the 16x8/8x16 partitions (P and B) | +2.81 | +2.71 | +1.71 | -5.29 | +9.01 | +2.19 |
| **+ level selection on chroma AC and Intra_16x16 AC (the defaults now)** | **+1.56** | **+0.10** | **-0.16** | **-8.22** | **+3.51** | **-0.64** |

x264's medium preset has its psychovisual options on (`psy-rd 1.0`), which
trade PSNR for detail, so the table flatters this encoder by about three
points: against `libx264 -preset medium -x264-params psy=0` (and both with
`--scenecut 0`) the last row is +4.25 / +2.05 / +3.37 / -2.21 / +6.97, mean
**+2.89%**.

What each step is:

* **Rate term in the motion search** (`me_rate()` in mc.c): the integer and
  sub-pel searches price a vector as SATD + lambda x the exp-Golomb length of
  its difference from the predictor, the way x264 does, instead of SATD alone.
  On synthetic clips this was inside the noise; on real ones (noise, many small
  vectors) it is the largest single gain.  `AEG_MELAMBDA=0` turns it off.
* **I-picture offset** (`--iqp`, default 3 = x264's ipratio 1.4) and **B offset**
  (`--bqp`, default 4, was 12).  The 12 was the optimum of the synthetic clips;
  on the real ones it costs 8-12 points, and the optimum is 4-6.
* **Spatial direct** (`--direct spatial`, the default; `temporal` keeps the old
  derivation).  Implemented after FFmpeg's `pred_spatial_direct_motion`
  (reference = min of the neighbours' positive references per list, vector =
  median, zeroed per 8x8 where the co-located block is still); bit-exact on the
  synthetic B matrix and on all fifteen real-clip encodes.  Worth little on its
  own (0.2 points) but it is what x264 uses and it raises the B skip rate from
  about 15% to 35-39% of the B macroblocks on `cock` (x264: 52%).
* **Level selection on intra blocks**: the coordinate-descent pass of
  *Rate-distortion level selection* now also runs on the Intra_4x4 and
  Intra_8x8 blocks, at half the inter lambda (an intra block's error is
  inherited by everything that predicts from it).  At the full lambda vtest
  lost 2.2 points; at half, -1.06 points on average and no clip loses.
  `AEG_TINTRA=0` turns it off, `=1` uses the full lambda.
* **Sub-pel search of the partitions**: the 16x8 and 8x16 candidates (P and
  B) used to search integer-pel and inherit the 16x16 search's fractional
  offset, so they competed with integer vectors against a quarter-pel 16x16
  one and lost (83 of ~1500 P macroblocks on `cock`; x264 splits about one in
  four).  Each partition now gets its own quarter-pel search from preset 2 up:
  -2.3% P-only, -1.55% with B pictures, a further -0.65% from the B ones, for
  6% of the time each (`AEG_PSUB=0|1|2`).
* **Level selection on chroma AC** (cat 4, against the already decided DC) and
  **Intra_16x16 AC** (cat 1): chroma is -2.65% on average (tree -5.05%) at no
  measurable cost; Intra_16x16 is neutral (-0.06%) and kept for completeness.
  The chroma lambda is the luma one (0.6x and 1.5x are both worse).
  `AEG_TCHROMA=0` / `AEG_TI16=0` turn them off.
* **The skipped macroblock priced as itself** (after the table: -1.80% on
  average, tree -6.13%, no clip loses, no measurable time).  The candidates
  are compared on luma alone, each with whatever residual its quantiser
  leaves, so a skip was only reached when every block of both planes
  quantised to nothing.  On a picture that repeats the previous one (tree has
  runs of identical frames) the re-coded quantisation error always won the
  luma comparison -- our PSNR kept creeping up over a run of identical frames
  while x264's stayed flat -- and cost more than it bought once header and
  chroma were added.  code_mb() now prices the winner, chroma included,
  against P_Skip / B_Skip (prediction only) and keeps the cheaper.
  `AEG_SKIPCHK=0` turns it off.
* **Adaptive group length** (`--b-adapt 1`, the default): `--bframes` becomes
  the most a group gets.  The lookahead (src/lookahead.c) estimates, on
  half-resolution source pictures, the cost of each structure -- per 8x8 block
  the minimum of an intra estimate (DC/H/V/plane SATD) and a half-pel hexagon
  search (P) or the best of list 0 / list 1 / average (B) -- and the scheduler
  applies x264's "fast" decision (b-adapt 1, including its 1/1.2 discount of
  B estimates and its distance-falling threshold) group by group.  The fixed
  3-B structure was costing `cock` +18 points against P only; the decision
  takes that back without giving up the B pictures where they pay.  The ring
  already held a whole group when its anchor was decided, so this needed no
  extra latency or memory beyond the lowres planes.  Not tuned further: the
  discount (`AEG_LABIAS`) and the threshold scale (`AEG_LATHRESH`) move single
  clips across a decision boundary in both directions, and the x264 values are
  as good as any tried.

Per-feature value, measured the same way for both encoders (P pictures only,
one reference; BD-rate of turning the feature *off*):

| turned off | x264 (mean of 5) | this encoder (mean of 5) |
|---|---|---|
| sub-pel refinement (x264 subme 2 / ours `--subme 0`) | +10.2 | +8.6 |
| partitions (x264 none / ours `--partitions 1`) | +9.0 | +4.4 |
| 8x8 transform | +5.4 | +4.8 |
| trellis | +5.4 | +1.1 (inter luma only at the time) |

(measured before the partition and level-selection rows of the table above,
which are the answers to the last two lines).  What remains: on `cock` at QP 28
with the fixed 3-B structure, against x264 with the same structure, each picture
type spends more bits for more PSNR -- I +4.5% for +0.6 dB, P +19% for +0.5 dB,
B +32% for +0.8 dB -- i.e. the B pictures are the least efficient part.

Tools for this: `tools/bdrate.py` (ours against x264) and `tools/abbd.py`
(ours against ours, any two option sets, with per-side environment).
`tools/real/` reproduces the five clips (`fetch.sh`: ffmpeg from pip, sources
via the GitHub API, 30-frame I420 in /tmp/real) and wraps the comparisons:
`bdx.sh` (against x264 medium, psy off), `ab.sh` (ours against ours, options
or `VAR=value` environment on either side) and `exact.sh` (bit-exactness
against FFmpeg over the clips).  The clip preparation differs slightly from
the table above, so its numbers start from a new baseline: at 818a739, preset
3, against x264 `psy=0`: cock +6.21, vtest -0.11, mega -2.32, short -3.07,
tree -1.48, mean -0.15%.

**More motion-search start points** (mean -0.15% -> -0.62%, cock +6.21 ->
+4.52, no measurable time).  The search started from the predictor, zero, its
rounding and the whole-frame seed only; on a motion boundary the median
predictor is none of the neighbours' vectors and the hexagon never reached the
right one.  Added, each measured with ab.sh against the previous step:

| start point | mean | cock | switch |
|---|---|---|---|
| P 16x16: left / top / top-right vectors (ref 0) | -0.13 | -0.60 | `AEG_MENB` |
| B 16x16: the same, per list | -0.19 | -0.87 | `AEG_MEBNB` |
| partitions: the 16x16 vector | -0.07 | -0.15 | `AEG_MEPART` |
| B list 1: list 0 vector scaled by -d1/d0 | -0.04 | -0.07 | `AEG_MEBSCALE` |

Tried and not adopted: evaluating the exact fractional predictor in the
quarter-pel stage (x264 does, at subme >= 3): +0.04%, `AEG_MEPMV=1`.

**Explicit weighted prediction of P pictures** (`--weightp`, on from preset
2).  wp_analyse() guesses, per P picture, x264's two candidates -- the gain
matching the standard deviations plus the offset matching the means, and the
offset alone -- for luma and (one denominator for both planes) chroma, and
measures each on a half-resolution picture: every 8x8 block's best SAD over
a +-4 integer search against the weighted reference, floored by a crude intra
estimate so that new content does not vote.  A candidate is kept when it saves
5% of that sum (`AEG_WPTH`, percent) *and* a quarter of a sample per pixel
(a near-black picture otherwise takes any offset).  The weighted pictures
refer to PPS 1, a twin of PPS 0 with weighted_pred_flag set, so the pictures
without weights do not carry a pred_weight_table.  With the luma weighted the
motion search runs on a weighted copy of the reference (refp[0] points at
it), while the prediction interpolates the real reference and weights the
result, which is the order of 8.4.2.3 and what keeps it bit-exact.  B pictures
stay unweighted (weighted_bipred_idc 0); list0[1] of a two-reference P
picture is never weighted.  Conformance cases `weightp`, `weightp_cavlc`,
`weightp_ref2` (2 slices, 2 threads) and `weightp_b` on genyuv mode 5 (a
fading pan).

| | fade (vtest x 0.5..1, 30 f) | cock | vtest | mega | short | tree | mean of 5 |
|---|---|---|---|---|---|---|---|
| vs x264 psy=0, `--weightp 0` | +137.9 | | | | | | |
| vs x264 psy=0, `--weightp 1` | +1.6 | | | | | | |
| vs x264, global PSNR, `--weightp 0` | | +5.64 | -0.25 | +0.24 | -2.69 | -0.07 | +0.57 |
| vs x264, global PSNR, `--weightp 1` | | +5.65 | -0.23 | +0.29 | -3.67 | -0.04 | +0.40 |

The per-picture-mean PSNR (what every other table here uses) reports mega at
+2.8% for this change, which is an artefact: mega opens with two black
pictures, at QP 34 the second one now gets a +1 luma offset that cancels the
I picture's DC error and is reconstructed *losslessly*, and a lossless picture
counts as ~170 dB in the per-picture mean -- one rate point jumps, the curve
stops being monotonic, and the BD integral goes wrong.  The sequence PSNR
moves 39.095 -> 39.116 dB for +0.5% bits there.  `tools/bdrate.py` (BD_GLOBAL=1)
and `tools/abbd.py` / `tools/real/ab.sh` (ABBD_GLOBAL=1) now offer the PSNR of
the sequence MSE for this reason; with it, ours against ours: short -0.94%,
the other four +0.01..+0.04% (the second PPS).

Not done yet: the weight guess is only refined by the two candidates; on the
synthetic fading pan (3 px/frame) the half-resolution integer search misses
the half-pel alignment and rejects the luma weight on half the pictures.

**B pyramid** (`--b-pyramid`, on whenever there are two or more B pictures,
spatial direct and one P reference).  The middle B picture of a group is
coded right after the anchor as a reference picture (nal_ref_idc 2) and the
other B pictures of the group predict from it: `P4 B2(ref) B1 B3`.  What it
took:

* four picture buffers and a mirror of the decoder's reference buffer
  (`dpb[]`, picture_rotate): sliding window with max_num_ref_frames = 3,
  or the picture's own MMCOs;
* reference selection by the decoder's default lists (set_current_refs): a B
  picture's list0[0]/list1[0] are the nearest references before/after it in
  display order, so the reference B is found without any list syntax;
* a P anchor's default list 0 starts with the reference B (largest
  frame_num), so it sends one ref_pic_list_modification (abs_diff_pic_num)
  that moves the previous anchor to the front, and an MMCO 1 per reference
  other than the previous anchor -- otherwise the sliding window would drop
  the previous anchor (smaller frame_num) while the next group still needs
  it;
* frame_num advances after every reference picture, including the B;
* the macroblock decisions are kept per buffer (`slot_mbs`), because the
  co-located picture of B1 is B2 and that of B3 is P4; colZero takes the
  co-located block's list 1 vector when its list 0 is unused (FFmpeg's
  `l1ref0 < 0 && l1ref1 == 0` branch).

The reference B's quantiser: the same offset as the other B pictures is best
(`AEG_BREFQP`, in halves of the offset: 0 = anchor QP +1.06%, 1 = half way
-0.18%, 2 = same -0.65% against no pyramid; an absolute offset of 3 or 5,
`AEG_BREFOFF`, is no better, nor is --bqp 3/5 with the pyramid).  Ours against
ours (global PSNR, `--b-pyramid 0` -> 1 at offset 2): cock -0.55, vtest
-1.12, mega -1.57, short 0.00 (never two B pictures), tree 0.00.

| vs x264 psy=0, global PSNR | cock | vtest | mega | short | tree | mean |
|---|---|---|---|---|---|---|
| weightp (above) | +5.65 | -0.23 | +0.29 | -3.67 | -0.04 | +0.40 |
| + B pyramid | +5.07 | -1.34 | -1.29 | -3.67 | -0.05 | **-0.26** |

Bit-exact on every real clip at presets 1/3/4, with slices and threads,
groups of two, mid-stream IDRs, and a 420-picture stream (POC lsb and
frame_num wrap) in CQP and ABR+VBV.  Conformance cases `bpyr_off`,
`bpyr_two`, `bpyr_adapt`; the fixed-structure order checks run with
`--b-pyramid 0` (their expected order is the flat one).

**Joint refinement of the bi-predictive vectors** (`AEG_BIDIR`, preset 3
up): the B_Bi_16x16 candidate used each list's own winner, which is the best
predictor alone, not the best half of an average.  bidir_refine() holds one
list's vector and walks the other's eight quarter-pel neighbours on the SATD
of the average plus the rate of both vectors, alternating (at most four
rounds) until neither moves (x264 does the same).  Ours against ours: cock
-0.30, vtest -0.02, mega -0.25, short -0.14, tree +0.02, mean -0.14% for
+1.8% time.  The same for the bi-predictive half of each 16x8/8x16 partition
(`AEG_BIDIRP`, b_part_search): cock -0.16, vtest -0.11, mega -0.13, short
-0.02, tree -0.01, mean -0.09% for +5% time.

**Half-resolution motion seeds** (lr_seed_frame, `AEG_LRSEED`, preset 1
up; `AEG_LRNB`).  Where cock lost to x264 was the P pictures at distance
3-4 (8 of them carry half the bits with three B pictures): our P pictures
went intra on 580 macroblocks where x264 found 404 inter ones.  An
exhaustive +-32 search (`AEG_MEFULL=32`, diagnostic only: +1000% time) was
-4.55% on cock's every third frame coded P only, i.e. the hexagon walk from
the predictor / zero / neighbours stalled in local minima.  x264 starts its
search from its lookahead's half-resolution vectors; now so do we: the
lookahead's block search (lookahead.c) over half-resolution copies of the
current source and of each list's reference reconstruction, one 8x8 block
per macroblock, +-16 half-resolution samples; its half-sample vectors are
whole samples at full resolution and become the search's seed (P and both B
lists), and the seeds of the right and lower macroblocks (not coded yet, so
not among the spatial candidates) join the neighbour candidates.  Ours
against ours: seed cock -1.74, vtest +0.03, mega -0.40, short +0.12, tree
-0.01 (mean -0.40%); the extra candidates cock -0.47, vtest -0.41, mega
-0.21, short -0.14, tree 0.00 (mean -0.25%); about +1% time together.
Exhaustive search still had -1.4% on cock after that.

**Multi-hexagon stage** (umh_search in mc.c, x264's UMH simplified): from
the best start, a cross of every other sample out to +-16 horizontally and
+-8 vertically, a 5x5 square, and 16-point hexagons at radii 4/8/12/16, then
the usual hexagon walk.  Per block class (`aeg_mc_ctx.umh[]`, umh_radii();
`AEG_UMH` / `AEG_UMHP` / `AEG_UMH8`), ours against ours:

| UMH on | cock | vtest | mega | short | tree | mean | time |
|---|---|---|---|---|---|---|---|
| 16x16 (preset 1 up) | -0.55 | -0.27 | -0.12 | +0.26 | 0.00 | -0.14 | +1.6% |
| + 16x8/8x16 (preset 3 up) | -0.74 | -0.50 | -0.72 | -0.41 | 0.00 | **-0.47** | +8% |
| + 8x8 and smaller (preset 4) | -0.89 | -0.81 | -0.64 | -0.31 | +0.02 | -0.53 | +26% |

| vs x264 psy=0, global PSNR | cock | vtest | mega | short | tree | mean |
|---|---|---|---|---|---|---|
| B pyramid | +5.07 | -1.34 | -1.29 | -3.67 | -0.05 | -0.26 |
| + bidir, lowres seeds | +2.31 | -1.83 | -2.24 | -3.84 | -0.05 | -1.13 |
| + multi-hexagon | +1.56 | -2.33 | -2.94 | -4.25 | -0.05 | **-1.60** |

**B-picture decision, measured again with the pyramid.**  Fixed structures
against the adaptive default (ours against ours, global PSNR):

| structure | cock | vtest | mega | short | tree | mean |
|---|---|---|---|---|---|---|
| --bframes 1 --b-adapt 0 | -0.80 | +2.05 | +2.91 | -0.11 | -0.03 | +0.81 |
| --bframes 2 --b-adapt 0 | +0.10 | -0.85 | +1.31 | +0.52 | +0.12 | +0.24 |
| --bframes 3 --b-adapt 0 | +1.55 | -3.40 | -3.35 | +1.95 | +0.12 | -0.63 |
| `AEG_LATHRESH=140` | +0.71 | -1.24 | -1.27 | +0.64 | +0.02 | -0.23 |
| --b-adapt 2 (trellis, bias 120) | +0.57 | -1.18 | -3.35 | +1.95 | +0.12 | -0.38 |
| --b-adapt 2, `AEG_LABIAS=112` | +0.83 | -0.69 | -1.41 | +1.95 | +0.12 | +0.16 |
| --b-adapt 2, `AEG_LABIAS=100` | -3.20 | +5.79 | +0.70 | -0.29 | -0.43 | +0.51 |

The best fixed structure per clip would be -1.5% on average, but every
adjustment of the estimate trades cock/short (which want short groups)
against vtest/mega (which want long ones): the half-resolution SATD estimate
does not tell them apart, and the trellis (`--b-adapt 2`, la_group_cost /
b_adapt_trellis: every split of the window priced with the B pictures
against their real references, pyramid included) flips between all-P and
all-3B within a few percent of the B discount.  It stays an option; the fast
rule stays the default.  For comparison x264 (b-adapt 1) puts three B
pictures in almost every group on all five clips; with three B pictures our
cock is +0.4% against x264's, with one -3.2%, so part of cock's remaining
gap is the coding of long groups (x264 gives the reference B picture QP+1
and the others +2, we +4 for both; lowering ours was measured worse on the
average).

**Update: the trellis is the default at presets 3 and 4.**  Re-measured with
everything since (macroblock tree in CQP, second reference, trellis
quantisation), against x264 medium psy=0, global PSNR, QP 22-37:

| preset 3 | cock | vtest | mega | short | tree | mean |
|---|---|---|---|---|---|---|
| fast rule (b-adapt 1) | -0.73 | -11.50 | -7.28 | -9.39 | +2.04 | -5.37 |
| trellis (b-adapt 2) | -0.75 | -11.39 | -11.30 | -6.22 | +2.11 | **-5.51** |
| trellis, `AEG_LABIAS=110` | -0.45 | -9.48 | -10.30 | -6.22 | +2.11 | -4.87 |
| trellis, `AEG_LABIAS=100` | -4.59 | -6.94 | -7.55 | -8.70 | +1.98 | -5.16 |

| preset 3, validation | cock | vtest | mega | tree | vtest2 | mega2 | mean |
|---|---|---|---|---|---|---|---|
| fast rule | -1.53 | -13.06 | -5.57 | +9.90 | -10.17 | -5.78 | -4.37 |
| trellis | -0.29 | -13.64 | -7.83 | +5.15 | -10.02 | -9.78 | **-6.07** |
| trellis, `AEG_LABIAS=110` | -2.21 | -12.73 | -7.16 | +5.15 | -7.85 | -2.06 | -4.48 |
| trellis, `AEG_LABIAS=100` | -3.76 | -6.61 | -4.41 | +5.15 | -6.07 | +9.95 | -0.96 |

Preset 4: -5.84 -> -5.91 (five clips), -4.66 -> -6.47 (validation).  Time
+3% at preset 3, not measurable at preset 4.  The case that decided it is
the validation tree segment (frames 30-59 of the same source: unlike the
first segment it has no repeated pictures, the foliage moves in every
frame).  The fast rule's absolute cost threshold (300 - 50/step SATD per
macroblock) is always exceeded there (a far P costs ~650 per macroblock
even though it is only ~9% dearer than a near one and a B picture costs
0.7 of a P), so it codes one B per group; the trellis and x264 code three.
The B discount is still chaotic (the 100/110 rows), so it stays at x264's
1.2.  Presets below 3 keep the fast rule, as x264 does below slow.

The price is at the high-quality end, and almost all of it is short (where
the trellis codes three B pictures per group instead of one, and our tree
loses with long groups -- see "Why 3-B groups lose on short" below):

| QP 16-25, preset 3 | cock | vtest | mega | short | tree | mean |
|---|---|---|---|---|---|---|
| fast rule | +0.64 | -12.99 | -6.57 | -13.38 | +2.40 | -5.98 |
| trellis | +0.93 | -12.69 | -8.06 | **-2.58** | +2.43 | -3.99 |

| QP 16-25, validation | cock | vtest | mega | tree | vtest2 | mega2 | mean |
|---|---|---|---|---|---|---|---|
| fast rule | -1.75 | -12.26 | -3.04 | +8.94 | -12.47 | -1.40 | -3.66 |
| trellis | +0.99 | -12.50 | -5.15 | +7.24 | -12.15 | -6.54 | -4.68 |

Over the eleven clips: QP 22-37 -4.82 -> -5.82 %, QP 16-25 -4.71 -> -4.37 %.
Against x264 at CRF with its tree (`aq-mode=0`): -0.99 -> **-1.63 %**
(cock -0.26, vtest -1.01, mega -4.71, short -4.04, tree +1.87).

**Two P references with the B pyramid** (`--ref 2` with B pictures; the
default stays one reference when there are B pictures).  x264's reference
count is worth 0.39 points from 1 to 2 and 0.29 from 2 to 3 on these clips
(global PSNR against x264 psy=0: ref=1 -2.28%, ref=2 -1.89%, ref=3 -1.60%).
The pyramid no longer forces one reference:

* list0[1] of a P picture is the newest other picture -- the reference B of
  the group just coded when there is one (`AEG_REF2ANCHOR=1`: the older
  anchor instead);
* list 0 modification is now general (refctl.n_mod entries, subtract/add):
  whenever the default list's first entries (decreasing frame_num) are not
  the wanted ones, every wanted entry is placed explicitly; a slice that ends
  up on one active reference writes only the first entry (n_mod_1ref) --
  writing both made FFmpeg reject the header ("reference count overflow");
* the P picture's MMCOs keep exactly its two references; the reference B
  picture's MMCOs drop the one picture that is neither of its group's
  anchors, so the buffer never exceeds max_num_ref_frames = 3;
* the second reference's search gets two more starting points: the first
  reference's vector scaled by the POC distances and the half-resolution
  pre-search's vector against it (`AEG_REF1C`).

Bit-exact on all real clips (CQP, ABR, slices/threads, -g 7, b-adapt 2,
preset 4, the fade clip, the 420-picture stream), conformance cases
`bpyr_ref2`, `bpyr_ref2s`.  But it does not pay (ours against ours):

| change | cock | vtest | mega | short | tree | mean |
|---|---|---|---|---|---|---|
| --ref 2 with pyramid (list0[1] = reference B) | -0.16 | -0.63 | +0.41 | +0.47 | +0.02 | +0.02 |
| same, list0[1] = older anchor | -0.18 | -0.83 | +0.55 | +0.47 | -0.01 | 0.00 |
| second-reference start points, P only | -0.08 | +0.06 | -0.21 | +0.10 | 0.00 | -0.02 |

Even P only, short is worse with two references (+0.21%) although only 1-5%
of its inter macroblocks pick the second one; a fixed penalty on those
candidates (4 or 16 lambda-bits) moves mega and short in opposite
directions.  The per-macroblock RD choice does not see what a reference
mismatch does to the neighbours' vector predictors (8.4.1.3.1) or to the
pictures that predict from this one.  Left as an option.

**Macroblock tree** (x264's mbtree; `--mbtree`, `--rc-lookahead`; on by
default at constant QP from preset 1, window 10/20/30/40 pictures at presets
1-4).  Measured first on x264 itself: on these clips x264 at CRF with its
mbtree is 4.4% cheaper than x264 at constant QP (the reference every number
above uses), and mbtree alone is worth 8.3% inside CRF -- by far the largest
x264 feature not yet in.

How it works here (src/encoder.c, `mbt_run()`; lookahead.c,
`aeg_lowres_mbcosts()`):

* every time schedule() queues a coding step it waits until `rc_lookahead`
  more pictures are in the input ring, then *plans* the rest of the window
  the way it will code it: the same b-adapt decisions on the same cached
  lowres costs (`la_cost_ci()`, so the plan and the real decision agree), the
  keyframe interval, the pyramid's reference B, the flushed tail as a P
  chain; scene cuts are not anticipated;
* per planned picture, per macroblock: lowres intra cost, best inter cost
  against the planned references (list 0, list 1 or both) and the vectors;
* from the newest picture back, each block passes (intra + propagate_in) x
  (1 - inter/intra) to the up to four macroblocks its vector lands on in its
  reference(s), split evenly between two references;
* the queued reference pictures (I, P, reference B) get
  `-1.3 * log2((intra + propagate_in) / intra)` added to the macroblock QP;
  non-reference B pictures get nothing, and the B offset drops from 4 to 2.

x264 uses strength 2.0 (5 x (1 - qcomp)), base QP + 5.4 and pb_factor 1;
measured here (fixed 3-B groups, offline whole-clip prototype first):

| variant (ours against ours) | cock | vtest | mega | short | tree | mean |
|---|---|---|---|---|---|---|
| P-chain prototype, --bqp 4 | +4.07 | -9.62 | -1.34 | +3.33 | +5.10 | +0.31 |
| P-chain, --bqp 1 | +2.06 | -9.10 | +0.28 | +0.69 | +5.12 | -0.19 |
| real group structure, bf3, strength 2.0, --bqp 1 | +1.07 | -9.03 | -4.12 | +1.26 | +3.44 | -1.48 |
| same, strength 1.3, --bqp 1 | +0.45 | -7.05 | -2.78 | -0.33 | +1.94 | -1.55 |
| same, strength 1.6, --bqp 0 | +0.88 | -7.50 | -2.34 | +0.91 | +3.66 | -0.88 |
| same, strength 1.3, --bqp 2 | +0.64 | -7.97 | -3.70 | -0.19 | +1.93 | **-1.86** |
| same, strength 1.3, --bqp 3 | +1.62 | -8.39 | -3.86 | +0.74 | +1.95 | -1.59 |
| **in-encoder, default b-adapt 1, vs --mbtree 0** | +0.87 | -9.13 | -2.55 | +0.20 | +1.93 | **-1.74** |
| strength 1.7 instead of 1.3 | +0.83 | -1.64 | -0.42 | +0.84 | +0.66 | +0.06 |
| strength 1.0 instead of 1.3 | -0.37 | +2.27 | +0.19 | -0.29 | -0.66 | +0.23 |
| --iqp 1 instead of 3 | -1.01 | +3.14 | -0.01 | -0.68 | -1.86 | -0.08 |
| no offsets on I pictures | -1.00 | +10.38 | +0.09 | -1.26 | -0.06 | +1.63 |

On the 420-picture tree/short stream (a scene cut every 30 pictures):
-3.55%, for +10% time (mostly from the lower QPs, i.e. more coefficients);
a 10-picture window instead of 30 is the same (-0.05).  Against x264:

| | cock | vtest | mega | short | tree | mean |
|---|---|---|---|---|---|---|
| vs x264 CQP psy=0 (the usual reference), mbtree off | +1.56 | -2.33 | -2.94 | -4.25 | -0.05 | -1.60 |
| vs x264 CQP psy=0, mbtree on | +2.64 | -10.47 | -4.48 | -4.33 | +3.38 | **-2.65** |
| vs x264 CRF psy=0 aq-mode=0 (its mbtree on), ours mbtree off | +2.23 | +9.47 | +3.47 | 0.00 | -0.01 | +3.03 |
| vs x264 CRF psy=0 aq-mode=0, ours mbtree on | +3.33 | +0.08 | +2.65 | +0.02 | +3.17 | **+1.85** |

(`XRC=crf tools/real/bdx.sh "" "psy=0:aq-mode=0"`; CRF at the same values as
the QPs.)  So against x264 as it is normally run, our tree recovers about
1.2 of the 3 points its tree is worth.  The clips that lose (cock, short,
tree) lose in every variant; vtest (a static surveillance camera) wants the
strongest offsets, including on the I picture.  Note that x264 never gives
its *first* IDR any tree offset (`slicetype_analyse()` returns before the
tree when there is no previous non-B picture): on tree x264's I picture sits
at exactly CRF - 3 while ours gets -5.6 on top of the I offset; taking the
offsets off our I pictures helps cock and short by 1 point and costs vtest
10, so they stay.  tree is a slide show (new picture at 10, 16 and 23, exact
repeats in between).

Off by default with a rate target or a buffer model: they predict a picture's
size from its QP; with `--mbtree 1` they use QP + the picture's mean offset
(`mbt_qoff`).  The ring holds bframes + 3 + rc_lookahead + bframes + 1
pictures: with the window the scheduler can run a group behind the input
(the first picture alone, a long group then a short one), and the first
version overwrote B pictures that were still waiting.

Per-macroblock QPs had never been exercised against a decoder before (AQ is
off by default and untested with FFmpeg), and they found two bugs that also
affected `--aq`:

51. **Chroma deblocking across macroblocks of different QP.**  8.7.2.2 takes
    each side's chroma QP from its own luma QP and averages the two; we took
    the chroma QP of the averaged luma QP, which differs once the table
    bends.  6 samples on an I picture at QP 38.
52. **An Intra_16x16 macroblock with no AC keeps its own QP.**  With
    cbp = 0 the bitstream passes replaced the QP by the previous one (no
    mb_qp_delta), but Intra_16x16 always sends the delta and its DC levels
    were quantised with its own QP.
53. **The lookahead ring was a group short** (above).

Conformance cases `mbtree_wrap` (40 pictures, window 3, -g 13) and
`mbtree_aq`; bit-exact on all real clips at presets 1-4, b-adapt 2, --ref 2
-g 7, slices/threads, ABR and VBV with --mbtree 1, and the 420-picture stream.

**Why 3-B groups lose on short, and the B mode lambda.**  With the tree on,
fixed 3-B groups are 4.57% worse than b-adapt 1 on short, while for x264 they
are 0.5% *better* than 1-B.  Taken apart:

* It is not the structure as such.  At constant QP x264's 3-B groups are just
  as bad on short (`b-adapt=0:bframes=3` vs `bframes=1`, QP mode: cock -0.83,
  vtest -1.87, mega -2.78, **short +4.19**, tree +0.17); ours at the same
  per-type offsets (I -3, Bref +1, B +2, no tree): +1.89 / -2.47 / -3.97 /
  +4.61 / +0.54.  x264's CQP pb-ratio offsets are not tuned for PSNR, which
  also inflates "what x264's tree is worth" (CQP -> CRF+tree at fixed 3-B:
  -3.09 / -10.54 / -6.77 / -4.36 / -0.31).
* It is not the tree's QP allocation.  x264's own per-macroblock QP maps
  (decoded with `ffmpeg -debug qp`, crf = q - 5, injected as our offsets with
  `--bqp 0 --mbtree 0`) give our encoder -0.03% against our own tree on short.
* It is not the motion search: a full search of radius 24 changes fixed 3-B
  short by -0.26% and the 3-B penalty by -0.4.
* Per picture type at *identical* per-macroblock QPs (the injected maps, crf 27
  / q 32): x264 32039 B at 37.83 dB, ours 35414 B at 38.10 dB, which on x264's
  own slope (crf 26: 36840 B, 38.44 dB) is 3.6% worse.  The I picture is
  better (-5.7% bits, +0.57 dB); P costs +12.5% bits for +0.24 dB and B
  +20.6% for +0.27 dB.  At constant QP and 0, 1 or 3 B pictures the pattern is
  the same (ours always spends ~11% more for +0.3-0.45 dB), so the weakness is
  in coding B (and less so P) pictures generally, not in 3-B groups.

Scaling the lambda of B pictures (`AEG_*LAM` experiment, fixed 3-B, cock +
short): 2x +6.7%, 1.4x +2.6%, 0.75x -1.2%, 0.5x -1.2%; P pictures 0.75x -0.1.
Only the *mode decision* lambda matters (0.6x on the trellis alone -0.08, on
the mode decision alone -1.38); both the non-reference B pictures (-0.54) and
the B references (-0.64) contribute.  The gain is the B *offset*: at
`--bqp 0` it is gone (+0.17), with `--bqp 4` (no tree) 0.6x is -1.00.  So a B
picture now chooses its modes with the lambda of the QP it would have without
its offset (`mode_lambda_off`, 0.63x at the default +2); the quantiser stays
coarser, but the decisions are priced at the slope the anchors are coded at.
Full compensation against half (0.79x): -0.34 against -0.27 on all clips;
the same shift of the motion-search lambda: +0.21.  With `--bqp 4` full
compensation (0.40x) is worse than 0.6x on short (+1.01), so the rule is only
measured near the default offset.

| default b-adapt 1, ab | cock | vtest | mega | short | tree | mean |
|---|---|---|---|---|---|---|
| B mode lambda at QP - offset | -0.93 | +0.41 | -0.31 | -0.85 | -0.01 | **-0.34** |

Against x264 (`bdx.sh`): CQP **-2.96%** (+1.71, -10.09, -4.67, -5.11, +3.37),
CRF with its tree **+1.57%** (+2.37, +0.49, +2.57, -0.72, +3.16).  B pictures
still spend more bits than x264's for their PSNR at the same QP; where that
goes (fewer skips: 135-170 per picture against x264's 150-186) is the open
question for the B search.

**What x264's features are worth here** (x264 against itself, CRF,
`psy=0:aq-mode=0`, scene cuts off, `/tmp/w/xab.py`-style global-PSNR BD-rate;
positive = turning the feature off costs that much):

| x264 change | cock | vtest | mega | short | tree | mean |
|---|---|---|---|---|---|---|
| subme=6 (no RD in B pictures) | +3.95 | +0.07 | +1.84 | +1.99 | +1.78 | +1.93 |
| subme=5 (no RD at all) | +7.75 | +2.34 | +5.81 | +4.82 | +3.84 | +4.91 |
| trellis=0 | +0.60 | +2.57 | +0.90 | +3.57 | +4.32 | +2.39 |
| 8x8dct=0 | +5.59 | +2.17 | +2.59 | +3.54 | +0.23 | +2.82 |
| weightp=0 | -0.10 | -0.14 | +0.49 | **+4.20** | -0.37 | +0.82 |
| weightp=1 (no duplicate refs) | -0.08 | -0.10 | +0.55 | +1.09 | -0.34 | +0.22 |
| dct-decimate=0 | +0.01 | +0.70 | +0.16 | +1.22 | +0.97 | +0.61 |
| mixed-refs=0 | +0.09 | +0.82 | +0.25 | +1.29 | -0.05 | +0.48 |
| me=dia | +0.28 | +2.08 | +0.04 | +0.26 | -0.05 | +0.52 |
| ref=1 | -0.27 | +1.45 | -0.39 | +0.72 | -0.40 | +0.22 |
| fast-pskip=0 | -0.24 | +0.46 | +0.02 | -0.20 | +0.35 | +0.08 |
| chroma-me=0 | -0.46 | -0.16 | -0.34 | +0.29 | -0.04 | -0.14 |

Ours with `--weightp 0` was only +0.45 on short, against x264's +4.20.

**Weighted prediction decision, x264's way.**  x264 weights every P picture
of short with a plain luma offset of -1 or -2 (denominator 0, weight 1: a slow
exposure drift; seen with `ffmpeg -bsf:v trace_headers`), and its smart mode
adds the same picture again as ref 1 with another offset.  Ours never weighted
short: the saving of an offset there is 1-2.5% of the half-resolution cost,
under the old 5% / quarter-sample-per-pixel rule.  wp_analyse() now follows
x264's weights_analyse(): exit early when the means differ by less than half
a level and the gain guess is within 1/128 of 1; otherwise try the gain
guess (sqrt of the variance ratio, /64) and one step either side, each with
the truncated mean-matching offset and one either side, and keep the best
when it saves 0.2%.  The guess is measured with the full +-4 search (on a
fade the unweighted vectors are wrong) and the other eight around its
vectors (`wp_cost_near`, +-1).

On its own that took weights on the synthetic pattern of the rate-control
tests (means moving by -3..+4 as content enters and leaves; 0.7-2% saving for
an offset), where they cost +36% bits; ABR overshot by 14% and two-pass lost
to one-pass.  SATD instead of SAD does not separate the two (short 0.7-1.2%,
synthetic 0.7-0.8%).  What does is the *sign consistency* of the level change:
among the blocks the unweighted search predicts better than intra, the DC
moved by half a level or more one way on 2.5-3.7 times as many blocks as the
other way on short, and on 1.1-1.2 times as many on the synthetic pattern.  A
picture now needs a 2:1 majority (`AEG_WPCONS`, tenths) before any luma
weight is tried.  The conformance fade (genyuv mode 5) now takes weights on
four of the last four P pictures instead of two.

| ab vs previous commit | cock | vtest | mega | short | tree | mean | fade |
|---|---|---|---|---|---|---|---|
| x264-style search, 0.2% | +0.01 | 0.00 | -0.19 | -1.46 | 0.00 | -0.33 | +3.81 |
| + full search for the guess | | | | -1.41 | | | -1.60 |
| + 2:1 sign majority (default) | +0.17 | 0.00 | -0.19 | -1.35 | 0.00 | **-0.28** | -1.60 |

Against x264 (`bdx.sh`): CQP **-3.30%** (+1.89, -10.09, -4.90, -6.78, +3.37),
CRF with its tree **+1.13%** (+2.55, +0.49, +2.30, -2.86, +3.16).  Still open
from this: the duplicate reference (x264's weightp=2 over 1 is another 1.1 on
short), which needs list0 to name one picture twice.  `tools/abbd.py` takes
`BIN=path` as a leading token so one side can run another build (a worktree
of the last commit).

**The mode lambda of P pictures.**  With identical per-type QPs (CQP, I -3,
Bref +1, B +2) and x264's own q28..q30 slope as the yardstick
(`/tmp/w/gap.sh`), our bit excess at equal PSNR on cock at q30 is 0.0% with
one B picture per group, 1.5% with two and 2.5-2.8% with three (pyramid or
not), so what we lose grows with the prediction distance; a full search of
radius 24 changes nothing.  At constant QP the B pictures take fewer skips
than x264's (45% against 53% on cock, 40% against 51% on short) and more
bi-predicted partitions.

Pricing the skipped macroblock higher in the skip check (x1.15) on P pictures
was -0.49% (cock + short), x0.85 +2.1%; on B pictures both ways within 0.3.
That pointed at the P lambda as a whole: the mode decision of P pictures now
uses 0.65x the lambda of the QP (`P_MODE_LAMBDA`; 0.55 * 2^((QP-12)/3), close
to HM's 0.57 for inter pictures, where 0.85 is the intra/JM constant).
Everything else -- trellis, motion search, I and B pictures -- keeps its
lambda.

| P mode lambda, cock + short | 0.8x | 0.65x | 0.5x | 0.65x, all lambdas incl. trellis (0.75x) |
|---|---|---|---|---|
| BD-rate | -0.90 | **-1.38** | -1.00 | -0.08 |

0.65x with `--mbtree 0`: -2.10; with `--bframes 0`: -2.56; on I pictures
too: +0.08; another 0.8x on B pictures: -0.08 on cock + short but +0.07 on
all five (vtest +0.40); P at 0.9x of the new value: +0.07.  Presets 1 and 4
(cock + short): -1.43 / -1.58.

| ab vs previous commit | cock | vtest | mega | short | tree | mean |
|---|---|---|---|---|---|---|
| P mode lambda 0.65x | -1.20 | -0.14 | -0.65 | -1.46 | +0.02 | **-0.69** |

Against x264 (`bdx.sh`): CQP **-3.98%** (+0.65, -9.76, -5.63, -8.62, +3.44),
CRF with its tree **+0.29%** (+0.94, +0.88, +1.47, -5.05, +3.23).

**Intra block modes by rate and distortion; a trellis bug.**  Coding only
every fourth picture of cock with P pictures (`cock4`, motion four times as
long) turned our -1.4% bit excess against x264 at equal PSNR into +1.8%, with
almost 60% of the macroblocks intra in both encoders -- and all-intra coding
(`-g 1`, x264 `keyint=1`) was behind on every clip: cock +1.9%, vtest +2.9%,
tree +2.2%, short +0.9%, mega +0.9% (`/tmp/w/gap2.py`, x264's q28..q30 slope;
the same in chroma).  Two causes:

* The Intra_8x8 modes were chosen by SATD plus lambda/2 per mode bit, which
  is several times x264's weight at high QP; x264 picks I8x8 for 49% of the
  cock macroblocks, we picked it for 40%.  The Intra_4x4 modes were already
  chosen by rate and distortion (preset 3), but with the CAVLC cost of the
  levels under CABAC, and the mode bits at lambda/2.
* `trel_descent()` copied only `n` levels in natural order -- 15 for the
  AC blocks of Intra_16x16 and chroma, leaving position 15 uninitialised --
  and took `last` as a raster position where the price uses the scan
  position.  The output depended on stack contents (a build with
  `-ftrivial-auto-var-init=pattern` coded 3% differently); now every build
  is identical.  On its own the fix measured +0.03 (noise).

Now, from preset 3 under CABAC, every legal mode of an Intra_4x4 or Intra_8x8
block is priced exactly -- quantised, reconstructed, SSE plus lambda times the
CABAC price in the current context plus lambda per mode bit -- and the best two
are priced again with the levels the trellis will choose
(`intra_rd_modes()`; `AEG_IRDK` = how many, default 2).

| all-intra, bit excess vs x264 | cock | vtest | tree |
|---|---|---|---|
| before | +1.90 | +2.91 | +2.18 |
| RD modes, no trellis look | -0.44 | +1.20 | +0.55 |
| + trellis look at the best 2 | -0.68 | +0.47 | +0.35 |
| + trellis look at the best 3 | -0.71 | +0.30 | +0.27 |
| + trellis look at all modes | -0.38 | +0.25 | +0.13 |

On the five clips (normal encode): RD modes -0.53% for +7% time; the trellis
look at two modes another -0.20% for +11%; at three -0.02% more; at all modes
(the first try) -0.28% for +47%.  Correcting the lambda of the mode bits
alone (the SATD paths) was within 0.3 everywhere.

| ab vs previous commit | cock | vtest | mega | short | tree | mean |
|---|---|---|---|---|---|---|
| intra RD modes + trellis fix | -0.75 | -1.00 | -0.47 | -0.06 | -1.33 | **-0.72** (+21% time) |

Against x264 (`bdx.sh`): CQP **-4.61%** (-0.03, -10.68, -6.09, -8.57, +2.31),
CRF with its tree **-0.30%** (+0.48, -0.15, +1.01, -4.95, +2.10) -- the first
time ahead of x264 as it is normally run.

**SSE2 for the 8x8 transform.**  After the intra RD the 8x8 inverse
transform was the largest single function of a preset-3 encode (14% in
gprof: the trellis prices every 8x8 candidate level with a full
reconstruction).  The forward transform, the dequantiser and the inverse
transform (with the scalar path's int16 wrap between the passes) now have
SSE2 versions, checked bit for bit against the portable ones by
`tests/coef_check.c` (int16-extreme levels included); every stream is
byte-identical to before, preset 3 is 5-7% faster.

Tools: `tools/real/gap.py` (our bit excess over x264 at equal PSNR at one QP,
from x264's own q-2..q slope; any clip, x264 and aegenc options, `PL=` plane)
and `tools/real/frames.py` (per-picture type, size, mean QP and PSNR of
several streams of one source).  On `tree` (a 7 fps clip with every picture
repeated 6-7 times, not a slideshow) our macroblock tree costs +1.35% while
x264's is neutral; the propagation through the repeated pictures is
legitimate, and the I picture ends up 3.9 QP below the refresh P pictures
where x264 has 1.5 -- not pursued further.

**Two P references with B pictures; three things that did not help.**

* `--ref 2` with the adaptive B decision on the five real clips: -0.27,
  -0.34, -0.30, -0.05, +0.06 = **-0.18%** for +3% time, so preset 3 and up
  now use the second P reference with B pictures as well (it was P-only).
  x264's own references are worth 0.77 points against us (`ref=1`: cock
  0.66, vtest 1.71, mega 1.05, short 0.73, tree -0.29); its list0[1] is the
  B-reference of the previous group (closer than our older anchor), which is
  what is left here.
* x264's dct-decimate on inter luma (8x8 groups scoring under 4 and the
  macroblock under 6 dropped, after the trellis): +1.53 cock, +2.37 short;
  the macroblock rule alone +0.08.  Our trellis and the coder-priced skip
  decision already make that choice by price.
* The b-adapt 1 of current x264 (greedy: picture j is a B picture when
  "B..BBP" costs less than "B..BPP" by the lookahead's path cost) instead of
  ours (the threshold rule of early x264): cock -1.90, vtest +1.36, mega
  -4.21, short +5.73, tree +0.09 = +0.22 -- the same split as a fixed three
  B pictures (and x264's own fixed 1B beats its 3B on short too, so this is
  the content, not our B coding: at fixed structures, mbtree off, we are
  ahead of x264 on every clip, 3B: cock -1.3, short -2.1, vtest -2.1, mega
  -5.1).  The path-cost ratios of short (0.86-0.96) and mega (0.89-1.0) look
  alike, so the lowres estimate cannot tell the two apart.  The B discount
  `AEG_LABIAS=150` instead of 120: vtest -0.96, the rest 0.00, -0.19 mean;
  not taken (one clip).

| vs x264 (`bdx.sh`) | cock | vtest | mega | short | tree | mean |
|---|---|---|---|---|---|---|
| CQP | -0.31 | -11.09 | -6.33 | -8.62 | +2.39 | **-4.79** |
| CRF + mbtree | +0.17 | -0.63 | +0.81 | -5.03 | +2.18 | **-0.50** |

**Chroma in the quarter-pel cost** (x264's chroma-me; `AEG_CHROMAME=1`):
the SATD of both chroma planes added to the luma SATD at every quarter-pel
candidate of blocks of 8x8 and up.  -0.02% for +18% time, so off by default
(x264 values it at 0.16 points, but our RD refinement of the vectors
already prices the chroma residual).

x264 feature values measured the same way (global PSNR, BD of ours against
x264 with the feature off, minus the baseline +0.40): weightb 0.11,
chroma-me 0.16, b-pyramid 0.46, ref 3 -> 1 0.70.  On cock alone: P only
(bframes 0, ref 1 both) ours +1.24%, intra only ours -27.5%.

## Next

1. The B *search*, now that the macroblock layer is complete and its vectors are
   refined (see *The same refinement in a B picture*).  Three ways of pricing
   the per-partition list choice and the partitioned shapes' vectors were
   measured and are inside the noise or worse, so what is left is the
   *candidates*, not their prices: B_8x8's groups still search integer-pel per
   list and inherit the macroblock's fractional offset, which is why the shape
   is selected on 2 macroblocks in 2400; giving each group the refinement the
   16x16 candidates now get (2 lists x 4 neighbours x 4 groups = 32 full
   candidates) is the one remaining quality-for-time trade here, and the 8x4/
   4x8/4x4 sub-types become worth searching only after that.  Spatial direct is
   in now (*Real clips*) and the default; if the 8x4/4x8 sub-types are ever
   enabled under it, note that the decoder then sees later direct groups of a
   B_8x8 as neighbours.
2. The motion-vector refinement is exact now (bug 43) and reaches the
   candidates as well as the winner (`--refine-cand`, see *Refining every
   candidate*), but its trials are still priced with the predictor the search
   happened to store.  Two cheap follow-ups: re-derive the predictors *inside*
   the trial loop so the mvd priced is the mvd coded (the difference is one or
   two bits per trial, and `--refine-cand` makes it four times as many trials),
   and refine the chroma vector with the luma one instead of copying it.  Both
   are pure quality-for-time trades and belong behind the same preset gate.
3. Quality/RD, the parts that are left.  Level selection is implemented and
   measured (*Rate-distortion level selection*): priced by the coder, exact
   distortion, worth 0.15-2.2 points (1.0-1.2 on average) for 12-14% of the
   time, and on from preset 3 up.
   What is left of the quality gap on P pictures is the search, not the
   coefficients:
   * the *partition shapes* (which of 16x16/16x8/8x16/P_8x8 to price, and the
     `sub_mb_type` inside P_8x8) are still chosen by SAD plus an 8-bit rate
     guess; the shapes that make it into the list are therefore a SATD decision
     even though the comparison between them is exact;
   * the sub-pel vector is chosen by SATD and then refined by the coder's
     price: the winner's from preset 3 (`--refine-mv`), every close candidate's
     with `--refine-cand 1`.  What neither reaches is the *search* itself -- the
     integer-pel point, the fractional offset and the partition shape are still
     SATD decisions, so a vector more than a quarter pel from the right one
     never becomes a candidate at all.
4. GPU: every stage that can be batched is now wired to its kernel.  The plane
   build is `interp_plane`, both halves of the search are rounds through
   `sad_batch` and `satd_batch`, and the coefficient stages of an inter
   macroblock are one `tq_batch` + one `iq_idct_batch` per candidate (see the
   section above); `tests/kernel_check.c` cross-checks all six kernels against
   the CPU reference.  What is *not* batched: the RD decision itself, which
   reads each candidate's cost and is therefore serial per macroblock, and
   chroma / Intra_16x16 (their DC Hadamards need a per-block override).  Going
   further means splitting the picture into "produce every candidate" and
   "decide" passes so a whole picture's candidates share one launch; that is
   the largest remaining restructure and it cannot be measured here.  The
   per-macroblock form of it is implemented (*Rate-distortion batching
   (provisional)*); the picture-wide form would change the bitstream, so it is
   a different encoder rather than an accelerator of this one.  Every
   device path now runs end to end on a real OpenCL device (PoCL, a CPU
   implementation) and is byte-identical to the CPU encoder; the dispatcher
   places the coefficient stages by trial.  What only GPU hardware can answer
   is whether that trial ever comes out for the device -- see *GPU groundwork
   on a real OpenCL device*.
4. Encoder features that need the profile to change: none left -- CABAC and the
   8x8 transform are both in (main profile and High profile respectively), and
   the encoder emits Main, High or Baseline depending on which tools a stream
   uses.  Of x264's *default* preset, multiple reference pictures are now in
   as far as two (*Two P references*); more than two would need a larger
   buffer and reference list reordering, and on these clips the second one is
   already worth little once B pictures are on.  B pictures are in, and only
   their partitioned shapes are left (item 1).  The macroblock
   tree is in at constant QP (*Macroblock tree*); making the rate controller
   and the buffer model aware of it beyond the mean offset is what is left
   for ABR/VBV.
5. Threads beyond slices: a per-thread device context (slice threads and
   device stages do run together now, but their device calls are serialised by
   the backend guard), and frame-level threading (code picture
   N+1's decision pass while picture N is being written), which needs the
   reconstruct-then-write split that the loop already has, plus one reference
   picture per thread.  Rate control is picture-level only (the macroblock
   tree is on at constant QP only), and the VBV model tracks the buffer level between pictures
   rather than the coded-picture removal times a full HRD also signals
   (`cpb_removal_delay` in the SPS VUI).

## Two-pass rate control

`--pass 1 --stats F` codes the stream at a fixed QP and records the bits each
picture took; `--pass 2 --stats F --bitrate N` uses those numbers to decide how
the bits are shared out.

The interesting part is what a first pass is *for*.  Scaling the measurements to
the target (`bits_i` proportional to `bits_i`) is exactly a constant QP shift --
it measures nothing.  The allocation is `bits_i ~ complexity_i^qcomp` with
`qcomp = 0.6` (x264's default), which gives the hard pictures slightly less than
proportional and the easy ones slightly more; that is the right direction because
distortion per bit falls faster in flat, well-predicted areas than in busy ones.
With `t_i ~ b_i^alpha` normalised to the same total, the QP of a picture relative
to a constant-QP stream is

    QP_i - QP = 6 * (1 - alpha) * (log2 b_i - mean_j log2 b_j)

so the result is a *mean-zero per-picture QP bias* that depends only on how
unusual a picture is, not on the target rate.  That is why pass 2 keeps the
normal rate controller: the controller owns the level (the stream still lands on
target), the bias owns the distribution
(`aeg_encoder_set_qp_offset()`).  With `--qcomp 100` the bias is identically
zero and pass 2 reproduces the one-pass stream bit for bit -- which is what the
`qcomp` check in the suite relies on.

Measured on a 60-picture clip made of two 30-picture clips of very different
complexity (140 kb/s against 311 kb/s at QP 26):

| mode | rate | PSNR |
|---|---|---|
| one-pass ABR, target 250 kb/s | 267.8 kb/s | 38.56 dB |
| two-pass, target 250 kb/s | 248.1 kb/s | **42.32 dB** |
| one-pass ABR, target 300 kb/s | 307.9 kb/s | 40.48 dB |
| two-pass, target 300 kb/s | 297.0 kb/s | **44.72 dB** |

Two-pass is 3.8 dB better at a 7% lower rate, and it is also more accurate: the
one-pass controller overspends by 7% here because the picture after the join
costs ten times a P picture, and a one-pass controller only finds that out by
coding it.  On a single-complexity clip the gain is the smaller, honest one:
300.6 kb/s / 46.37 dB against 300.5 / 46.10 for one pass (+0.27 dB at the same
rate).

The same round fixed the one-pass controller's response to a picture that costs
ten times the target (a keyframe, or any picture after a scene cut): the debt it
creates is now clamped to three pictures' worth, because letting a single
overshoot follow the controller for the whole window is what turns one expensive
picture into thirty bad ones -- on the clip above it cost 10 dB.

## Scene cuts

A fixed GOP puts a keyframe every N pictures, wherever the content happens to be.
A scene cut is the one place where that is measurably the wrong choice: the P
picture after a cut has nothing to predict from, so it costs a full frame of
residual at poor quality, while an I picture of the same frame is a normal,
well-predicted intra frame.  Detection runs before the picture is coded, using
only the current source picture and the previous reconstruction:

| measure | what it is |
|---|---|
| `p_sad` | `sum |cur - ref|` at zero motion -- an upper bound on the inter cost, because every real motion field is at least as good as "no motion" |
| `ac` | `sum |cur - block mean|` per 16x16 block -- what Intra_16x16 DC prediction has to code |

Both sums use every other sample in both directions (a quarter of the picture;
the ratio of two such sums is within a percent or two of the whole one).  A cut
is `p_sad > scenecut/100 * ac`, default `scenecut = 120`.

The comparison is between two *consecutive display-order source pictures*, taken
from the lookahead ring.  Without B pictures that is - and was - the previous
reconstruction, which is exactly the prediction the P picture will use.  With B
pictures it has to be the source ring: the reference standing behind an anchor is
the anchor of the *previous* group, three or four frames back, and over four
frames a fast pan and a cut look the same (measured ratio 2.2 for ordinary motion
on the panning clip, where one frame of motion is 0.7).  With B pictures the
decision is also made in `schedule()`, once the whole group is in the ring, so the
group can be cut short at whichever picture is the cut instead of only at the
anchor -- a cut that falls between two anchors is otherwise skipped entirely,
because the next anchor compares two pictures that are both past it.

Measured on the 60-frame clip that is two unrelated 30-frame clips back to back:
the ratio is 0.67-0.75 on every ordinary picture and 2.92 exactly at the join, so
the margin is wide in both directions, and the test suite checks both: the join
gets its IDR, 120 frames of ordinary motion get none.  The first version of this
compared `p_sad` against the picture's *vertical gradient*, which is not an intra
cost at all -- fast motion makes even an ordinary frame look uncorrelated -- and
fired on a quarter of the frames of the panning clip.  That is the bug this
section exists to record.

Inserting the IDR is rate-neutral on this synthetic pair (56 308 B with the cut
detected against 56 444 B predicting through it at the same QP, quality within
0.07 dB): both clips are texture, so coding one as intra is no cheaper than
coding it as a P picture that predicts badly.  On real content, where a cut is a
different picture entirely, the P picture is the expensive choice -- the
mechanism is the standard one and is what makes the feature worth having, but
this environment cannot demonstrate the gain, so the default threshold is
deliberately conservative.

## Slices

`--slices N` cuts a picture into `N` row ranges (`slice_rows = ceil(mb_h / N)`,
so a picture whose rows do not divide evenly gets fewer than `N` slices --
320x208 has 13 rows, `--slices 5` gives slices of 3 rows: 5 slices, the last
one with 1 row).  Each slice is a NAL of its own (`nal_ref_idc` 3 for IDR, 2
otherwise; type 5/1) with its own slice header, `first_mb_in_slice = first_row
* mb_w` and its own `rbsp_trailing_bits`.  The SPS and PPS still go out once per
IDR picture.

The interesting part is not the header, it is what the macroblocks may *not* do
next to a slice boundary.  H.264 7.4.1.2.4 and 8.4.1.3: a macroblock outside
the current slice is unavailable, which for row slices means that in the first
row of every non-top slice

* `intra4x4_pred_mode` and Intra_16x16/chroma prediction treat the whole row
  above as unavailable (`DC`/`DC_128` instead of `DDR`/`DDL`, no top-right
  samples, `prev_intra4x4_pred_mode_flag` predicted from the left only);
* the deblocking filter does not touch the edge above the slice (that edge is
  filtered by the slice above, from its side, where the samples are available);
* the coefficient contexts `top_nz[]`, `luma_nc[]`, `chroma_nc[]` and
  `luma_ac_nc[]` are zero, and `mb_qp_delta` chains restart from the slice QP;
* MV prediction loses the B/C candidates of the row above, so `P_Skip`'s
  implicit vector changes too.

All of that is one `mby > e->top_row` test (`e->top_row` is the first row of the
slice being coded) in each neighbour query, which is why the encoder only sets
`top_row` around `code_slice()` and leaves it at 0 everywhere else.  With
`n_slices == 1` the whole path is the code it was before slices existed.

What it costs, 320x192, 60 frames, preset 2, QP 26:

| slices | bytes | PSNR | encode |
|---|---|---|---|
| 1 | 67 906 | 45.32 dB | 82.9 fps |
| 2 | 67 484 | 45.28 dB | 84.3 fps |
| 3 | 70 076 | 45.30 dB | 91.2 fps |
| 4 | 69 277 | 45.34 dB | 86.2 fps |
| 6 | 71 415 | 45.36 dB | 95.1 fps |
| 12 | 77 343 | 45.06 dB | 95.8 fps |

The rate rises with the slice count (up to 14% at one slice per row) because
prediction and the deblocking filter lose their vertical reach at every
boundary; the picture quality at a fixed QP is flat, as it should be.  Slices
are therefore a parallelism knob, not a compression one -- the point of them is
that slices are what a decoder can decode in parallel, and what slice threads
in the encoder can code in parallel (not implemented yet: the decision pass of
every slice would have to run against the same reference picture and its own
per-macroblock context, see *Next*).

## Slice threads

`-t N` codes the slices of a picture with `N` threads (`0` = one per CPU, `1` =
serial, and `-t N` without `--slices` uses one slice per thread).  Nothing in
the bitstream depends on it: the bytes are identical for any thread count,
because a slice's bits are a function of its own rows only.

The obstacle is that the encoder keeps the mode decision's working state in the
encoder struct: `top_row`, the candidate list of the macroblock being analysed,
the pending skip run, the bit writer, the stage timers.  Threading a per-thread
context through a few hundred call sites of `analyse_mb()`/`code_mb()` would be
a rewrite; instead each thread runs against its own *shallow copy* of
`struct aeg_encoder`.  The copy carries the scratch, while every pointer in it
(the source and reference pictures, the half/quarter-pel planes, the
per-macroblock decisions, the slice bitstream buffers) still points at the
original's memory.  That is safe because of what slices mean: a thread only
writes the rows of its own slices (disjoint), and everything else it reads is
read-only for the whole picture -- the same reason a decoder can decode the
slices in parallel.

Two things a copy cannot do, both handled in `encode_frame()`:

* *grow a buffer*: a `realloc` in a copy would only grow that copy's pointer, so
  every slice's bitstream buffer is allocated before the threads start;
* *report a timer*: the stage timers and the dispatcher's counters are written
  through the struct, so each copy starts with them zeroed and they are summed
  after the join.  The bitstream is unaffected by that sum.

A device context -- one OpenCL command queue or one CUDA stream -- is not safe
to submit to from several threads.  The slice loop first stayed serial while
any stage was on a device; then the backend guard serialised the threads'
calls into one context; now each slice thread has a device *lane* of its own
(see "Device lanes" under *GPU groundwork without hardware, continued*).

320x192, 60 frames, preset 2, QP 26, on **two** cores:

| slices | threads | encode | bytes |
|---|---|---|---|
| 1 | 1 | 85.8 fps | 67 906 |
| 2 | 1 | 89.2 fps | 67 484 |
| 4 | 1 | 85.5 fps | 69 277 |
| 2 | 2 | 124.3 fps | 67 484 |
| 4 | 2 | 151.0 fps | 69 277 |
| 6 | 2 | 150.9 fps | 71 415 |

So four slices on two threads is 1.76x the serial single-slice encoder for 2%
more bytes.  The speedup is close to linear in the thread count while the slices
are large enough to keep the threads busy: each slice costs the prediction and
the deblocking filter its vertical reach, which is the price of the
parallelism, and it is paid in bytes, not in quality (the PSNR at a fixed QP is
flat, see the table in *Slices*).

## Frame threads

`--frame-threads N` (`0` = auto: CPU count + 1, at most 8, unless `-t 1`; `1` =
off; env `AEG_FTW=n` overrides) codes up to N pictures at once on a pool of
worker threads.  The stream is the same byte for byte as the serial one
(`tools/real/same.sh`, the conformance suite, and a frame-threads-versus-serial
matrix of 15 option sets x 5 clips with 2-8 workers), so unlike slices it costs
nothing in bytes.

The first version (`705d11e`) only moved each non-reference B picture to a
second thread -- the end of a dependency chain, nothing reads it.  That left the
P chain on the main thread as the critical path (about 75 % of the wall time),
and the present version overlaps consecutive reference pictures as a **row
wavefront**:

* **Prologue on the encoder, body on a worker.**  Each picture's prologue
  (slice type, reference lists, macroblock-tree offsets, adaptive
  quantisation) runs on the main thread in the serial order; the coding itself
  (`frame_body`) runs on a worker's copy of `struct aeg_encoder` whose every
  *written* buffer is private (macroblock decisions, coefficient/seed scratch,
  half-resolution pre-search, fractional and weighted planes, slice and packet
  buffers).  What coding the picture does to the encoder -- frame counters,
  zeroed seed vectors, reference marking, buffer rotation -- the main thread
  does at launch; statistics are merged at the join.
* **Rows, not pictures.**  A reference picture is reconstructed straight into
  the buffer its successors read and post-processed row by row behind the
  decision pass (deblocking row k once row k+1 is decided, padding, the
  half-resolution and half-pel planes a row later -- the same order as the
  whole-picture versions, so the same samples; `3c0d0af`).  Each finished row
  is published; a picture predicting from it waits per macroblock row for the
  rows its search can reach: macroblock row r reads reference lines up to
  16r + 15 + range + 4 (the vector is bounded by the search range, the 6-tap
  filter adds 3), the half-resolution pre-search a row further.  The samples
  read are therefore final and the decisions are the serial ones -- no
  vertical vector limit is needed beyond the one the search already has.
  Weighted prediction analyses the whole reference, so a weighted P picture
  waits for all of it.  `AEG_POISON=1` (serial) fills every row that would
  not be final yet with noise: the streams stay identical, which proves the
  bounds.
* **Buffers.**  A non-reference picture reconstructs into its worker's own
  buffer.  A buffer that reference marking drops stays in use while a picture
  in flight still reads it (`pic_busy`), hence NW + 2 more picture buffers;
  an input ring slot is not overwritten while a picture coded from it is in
  flight (`ft_ring_guard`).
* **Every picture starts in the same `aeg_encoder_encode()` call as it would
  serially.**  The macroblock tree refines a queued picture's offsets as more
  input arrives, so starting a picture one call early changes its QPs.  While
  input arrives each call launches exactly the serial job; when flushing,
  where nothing changes between calls any more, queued pictures start as soon
  as a worker is free.
* **Packets** leave in coding order through a queue that holds at most
  `FT_Q - 1` packets when a call returns (the oldest is finished first
  whenever a launch fills it, so every call can hand one out).  An earlier
  loop that waited for a free entry joined every worker and failed on a queue
  full of finished packets -- a rare truncated stream under load, reproduced
  every time with a 4-entry queue and fixed.  Each entry keeps its
  reconstruction and macroblock QPs for `aeg_encoder_recon()`.

Gate: B pictures, constant QP (rate control and VBV feed each picture's size
into the next picture's QP), one slice, and no whole-frame device search
(`AEG_SEED`, or the motion stage forced onto a device: its rounds read the
whole reference at once).  Device backends and the forced batch paths have
been allowed since the device lanes (*GPU groundwork without hardware,
continued*); the CPU backend was a condition before.  ThreadSanitizer is
clean (the environment-variable caches the workers would race to fill are
settled before the threads start).

Two cores, cock 640x352, 90 frames, `-p 3 -q 28`: serial 10.16 s, frame
threads 5.87 s (1.73x, about 15.3 fps; user time 10.23 s, so the threading
overhead is ~2 %).  With `--weightp 0` 5.28 s (1.89x, near the two-core
ceiling) -- the weighted P picture's whole-reference wait is what is left of
the P chain.  `same.sh` against `32e961c` (30 frames, five clips): 1.35x
overall, preset 1 1.61x, preset 3 1.33x, preset 4 1.31x.  The worker count
hardly matters on two cores (2: 6.04 s, 4: 5.85 s, 6: 5.78 s); on 4-8 cores
the bound is the P chain with its weighted-prediction waits, roughly 3.2 s
for these 90 frames (~28 fps).

**Running ahead of the weighted-prediction analysis.**  The analysis compares
the picture with the *whole* list-0 reconstruction (source-based analysis
loses 0.2 % BD-rate, see the dead ends), which serialised every P picture
behind its reference.  Now a frame-thread P picture whose reference is still
being coded starts as if the analysis had found no weights, and runs it as
soon as the reference is whole -- always before its own last row, which needs
the whole reference anyway.  No weights (most pictures): what it coded is
exactly the serial result.  Weights: it restarts from the copy of the encoder
it was launched with.  A restarting reference bumps a generation counter;
every picture reading it notices at its next macroblock row (or while waiting
for rows), stops, and restarts too, and the reference is rewritten only after
all of them have stopped -- so there is no read of a row being rewritten
(ThreadSanitizer clean, restarts included).  Rows are published only while
every reference read is still current, and a picture's last rows only after
its analysis; by induction over the coding order a picture whose last rows
are out never restarts, so a finished picture never read anything stale.  Whether
to run ahead is predicted by the same analysis on the reference's *source*
(`wp_predict`; the source is kept in the input ring while the picture is in
flight): it agrees with the reconstruction's on cock 27 of 28 P pictures,
fade 33/34, mega 24/26, vtest 26/26, short 25/32; a picture predicted to be
weighted waits for its reference (`AEG_NOWPSPEC=1` always waits).  Running
ahead *with* the predicted weights does not pay: on the fade clip they are
the exact final weights for only 3 of 17 pictures.  Streams identical to serial on
the 16 option sets of `tools/real/ftopts.txt` x six clips including the fade
clip (`tools/real/ftcmp.sh`), with 2-8 workers; cock 90 frames has about 10
restarts (one wrong guess also restarts the pictures that read it), short (an
exposure drift) about 6 per 30 frames.

cock 90 frames `-p 3 -q 28`, same machine state (this sandbox had become
~8 % slower than for the numbers above): serial 11.4 s, frame threads 7.4 s
waiting for the analysis, 6.4 s running ahead after every unweighted
analysis, **6.2 s** with the source prediction (-16 %), 5.7 s with
`--weightp 0` (the ceiling).  Preset 4, same clip: serial 15.2 s, frame threads 8.5 s
(10.6 fps; two cores make 7.6 s the ceiling).  On two cores both presets are
now within ~10 % of the CPU-bound limit, so more speed here means less work
per macroblock; on 4-8 cores the P chain no longer waits for whole
references.  (A vector-cost bound on the UMH positions -- skip a position
whose vector alone costs more than the best -- is exact but prunes too few
positions to measure.)  The single-thread time is unchanged (the
environment-variable caches became `_Atomic`, which costs nothing measurable).

Earlier numbers (B-only frame threads, against `6831122`): 30 frames, presets
0-4: 1.57 / 1.76 / 1.84 / 1.98 / 1.98x; 180 frames preset 3 48.8 -> 23.8 s.

## Early termination of the intra candidates

The first round of speed work after the frame threads (the ones that *do* move
decisions, each held to at most +0.1 % BD-rate on the five real clips).  Where
the time goes was measured with a cycle counter per section of `code_mb()`
(`AEG_CMTIME=1`; cock, 90 frames of the forward/backward loop, `-p 3 -q 28`,
one thread): in a P picture the motion search is 24 %, the inter candidates
20 %, Intra_16x16 16 %, Intra_8x8 16 %, Intra_4x4 14 %, the vector refinement
9 %; in a B picture the search is 43 % and the three intra shapes 34 % -- for
the one macroblock in sixty where intra wins in a B picture.

A per-macroblock log of every candidate's price (now removed again) showed
what separates the two cases:

* **B pictures:** the best inter price against the SATD of the best
  Intra_16x16 prediction.  Below 1.2 x, 65 % of the B macroblocks, there was
  not a single intra winner on cock.  So a B picture now prices its inter
  candidates *first* and skips every intra candidate under that gate
  (`AEG_GIB`, percent).
* **Intra_4x4/8x8 against Intra_16x16:** when the best inter price is below
  half the best Intra_16x16 one, the finer intra shapes practically never win
  (P: 24 % of their evaluations skipped, 44 of 9626 intra winners lost, with
  0.005 % of the price; B: 79 %, none lost).  `AEG_GI48P` / `AEG_GI48B`.
* **P pictures are different:** a quarter of cock's P macroblocks end up
  intra, a third of those where the inter prediction's SATD is the lower one.
  The B-style gate on P costs +0.24 % (mega +1.06), and no SATD rule for the
  inter partitions came close either (x264's "16x8/8x16 only if 8x8 beats
  16x16" loses 0.3-0.9 % of the price here: our partitions win on RD, not on
  SATD).
* **Three Intra_16x16 modes instead of four** (the SAD-worst one is never
  RD-priced; `AEG_I16N`): -0.01 %, -6 %.  Two modes: +0.04 %, -9 % (mega
  +0.25), held back.
* **Exact:** the luma distortion is a lower bound of a candidate's price, so a
  candidate whose SSE alone already loses is not given to the CABAC pricer --
  it was only done at preset 4 (with the chroma decision); now everywhere,
  byte-identical (a third of the Intra_4x4 candidates, a fifth of Intra_8x8).

Together, against the same binary with the gates off (`tools/real/ab.sh`,
`ABBD_GLOBAL=1`): preset 1 -0.02 % / -8.4 % time, preset 3 +0.01 % / -21.4 %,
preset 4 +0.02 % / -16.7 %.  cock 90 frames `-p 3 -q 28` with frame threads:
9.73 -> 8.47 s (10.6 fps); the main thread (P and reference-B pictures and the
lookahead) is still the critical path -- the B worker is busy 4.4 s of it.

Tried and rejected in the same round:

* x264's early skip *before* the analysis (a P_Skip whose residual quantises
  to zero everywhere): +0.67 % (cock +2.2 %); a SSE limit on top of it either
  never fires or loses as much.
* Skipping already-measured positions in the motion search (a per-thread
  visited table: 14 % of the integer and 24 % of the sub-pel evaluations are
  repeats) is exact but measured no faster -- the table costs what the SADs
  did.

`tools/prof/` has the sampling profiler used for this (`sprof.so`, an
`LD_PRELOAD` SIGPROF sampler that walks frame pointers, split by main/worker
thread; `an.py` prints self, inclusive, `--callers=` and `--children=`
tables), and `tools/real/spd.sh` times two binaries with alternating runs.

## Less motion work at presets 3 and 4

Preset 4 adds four things to preset 3, and each was measured against the
current preset 4 (ours against ours, `ABBD_GLOBAL=1`, five real clips, one
thread): the UMH stage on blocks below 16x8 (+0.18 % without it, -10 % time),
all eight neighbours in the vector refinement instead of the cross (+0.17 %,
-9 %), the chroma-aware candidate comparison (+0.72 %, -10 %), range 48
instead of 32 (+0.12 %, no time) and a longer lookahead (nothing either way).
None can simply go, so two of them were made cheaper instead:

* **The refinement ranks before it codes.**  Re-coding the macroblock with a
  neighbouring vector is a full candidate (residual, trellis, chroma, CABAC
  price); a quarter-pel SATD of the partition plus the mvd's bits is a small
  fraction of one.  `--refine-mv 2` now SATD-ranks all eight neighbours around
  the searched vector and re-codes the best K, in that order: K = 4 at preset 4,
  3 at preset 3 (which used the plain cross, now `--refine-mv 1`; the old
  eight-neighbour walk is `--refine-mv 3`; `AEG_MVREFK` overrides K).  Preset
  4: +0.09 % for -9 % time on the five clips, +0.05 % on six other 30-picture
  segments of the same sources (cock, vtest and mega from picture 60, tree
  from 30, vtest from 400, mega from 200); the fixed cross was +0.17 %; all
  eight ranked, not walked, +0.01 %.  Preset 3, against the cross: -0.11 % and
  -0.02 % for -2 to -3 % time (K = 2 is +0.02 % for -4.6 %).
* **The UMH stage per block class** (`umh_radii`): 8x8 blocks at radius 8 and
  none for 8x4, 4x8 and 4x4 (a fourth class, `AEG_UMHS`).  +0.00 % / -0.03 %
  on the two clip sets for -8.5 % time; no UMH below 8x8 at all is +0.05 % for
  -7 %, radius 8 on both +0.01 % for -4 %.  (Radius 8 for 16x8/8x16 at preset
  3: -0.03 % for -2 %, but +0.11 % at preset 4 -- not taken.)

Together at preset 4: -0.04 % / +0.07 % on the two sets for -18 % / -16 %
time.  Against x264 medium psy=0 at constant QP (`BD_GLOBAL=1
tools/real/bdx.sh "" psy=0 N`): preset 3 -0.01 / -11.30 / -6.17 / -8.98 /
+2.37, mean **-4.82 %**; preset 4 -1.78 / -11.74 / -8.05 / -9.24 / +2.11,
mean **-5.74 %** (cock, vtest, mega, short, tree).

Exact, alongside: the UMH stage's vector costs four positions at a time (the
Exp-Golomb length is the float exponent of 2|d| | 1, so SSE2 without a
leading-zero count; it was a third of `umh_search`'s own time), a two- or
three-position remainder of a SAD list goes through the four-way SAD, and a
candidate whose luma plus chroma distortion already reaches the best price is
not priced by the coder.  -3 % at preset 4, byte-identical (`same.sh` presets
1/3/4, 30/30).

cock 90 frames, frame threads, this (noisy: +-10 % between runs) sandbox:
preset 4 9.97 -> **7.73 s** (mean of 4; best 9.50 -> 7.23 s, ~12 fps), preset
3 unchanged within the noise (6.4-6.7 s; serial -2 %).  Decoder-exact
(`exact.sh` presets 3/4, `--refine-mv 3`, P-only, the fade clip), frame
threads identical to serial (`ftcmp.sh` 48/48 on cock, short and fade).

## Exact kernel work (second round)

Byte-identical again (`tools/real/same.sh` against 32e961c, presets 1/3/4,
30/30 streams identical; every kernel has a randomized differential test
against its scalar reference, the new ones in `tests/coef_check.c`):

* **SATD** the x264 way: vertical butterflies, a per-lane 4x4 transpose, one
  horizontal stage and `|a+b| + |a-b| = 2 max(|a|, |b|)` in place of the last
  one -- which is also the halving SATD applies, so no shift.  AVX2 for
  16-wide blocks (two 8x4 halves per register) and for 8-wide ones (rows y..y+3
  and y+4..y+7 in the two lanes), SSE2 otherwise.  16x16: 76 -> 27 ns.  The
  AVX2 code is compiled per function and picked at run time (`AEG_NOAVX2=1`
  forces SSE2), so the build stays baseline x86-64.
* **CAVLC bit estimates are no longer computed under CABAC** -- every CABAC
  decision goes through the real pricer, nothing read them.
* **8x8 quantiser** in SSE2 (it was the scalar loop with a 64-bit division
  per level for the int16 clamp; now a table): 207 -> 32 ns.  AVX2 8x8 inverse
  transform (a whole row of int32 per register), SAD and SSD loops with two
  rows (or four 4-byte rows) per register.
* **Trellis, 8x8:** a trial that moves one level redoes the dequantiser for
  that level, *one* row of the first inverse pass and the column pass, against
  a cache of the current levels' row pass (`aeg_idct8_cache_*`); same int16
  truncation points, so the SSE is the full reconstruction's to the unit.  The
  same idea for 4x4 blocks measured no faster (most of the 4x4 time is the
  initial and final pricing, not the trials) and was dropped.
* The CABAC residual pricer used by the RD pass keeps the context states and
  the running sum in locals instead of going through the generic coder per bin.
* **Motion search:** each stage of the hexagon / square / UMH patterns depends
  only on where the stage starts, so its positions are listed, their SADs
  measured four at a time with one load of the block per row (`aeg_sad_x4`)
  and compared in the listed order -- the same winner.  -3.8 %.
* **Quarter-pel SATD without storing the prediction** (`aeg_satd_avg`: the
  pavgb of the two planes inside the AVX2 SATD), also for the bi-predicted
  partition SATD of B pictures.  SSE2 chroma MC.
* **No measurement twice:** the trellis keeps the SSE of the levels it settles
  on (and the original price), and the intra block RD takes both from it
  instead of reconstructing and pricing the block again (-2.9 %); the
  coefficient price of the level search is inline; the quantiser's rounding
  offset is a shift, not a 64-bit division per block.

cock 90 frames, `-p 3 -q 28`, frame threads: 8.55 s (32e961c) -> about 7.0 s
(~12.8 fps; this machine's timings drift by +-3 % between runs, the
single-thread user time is about 16 % lower).  Tried: link-time optimisation
(`-flto`) is slower.

The main thread is still the critical path: P pictures are about 75 % of the
wall time (6.1 s of P, 1.1 s of reference B against 4.1 s of non-reference B on
the worker), so more B workers cannot help, and an exact speed-up beyond this
on two cores needs consecutive P pictures to overlap (row-wavefront frame
threads with a fixed vertical vector limit).  That is what *Frame threads*
above now does, exactly -- the search range already is the limit.

## Exact work, third round: nothing measured twice

Byte-identical to 40c1301 throughout (`tools/real/same.sh` presets 1/3/4 at
QP 24/34, 30/30; presets 0/2; CAVLC; `-g 1`; ABR + AQ; frame threads; `make
test`).  The kernels are mostly SIMD already, so this round removes work whose
result is already known:

* **Empty blocks and counts:** `aeg_nz_last()` (count and last index, eight
  values per compare, a SWAR bit count since baseline x86-64 has no popcnt) in
  the CABAC pricer and writer and in the trellis, which returns before
  gathering the scan when a block is empty; one SSD per reconstructed
  macroblock instead of one per block.
* **Lookahead:** Exp-Golomb length by leading-zero count, SSE2 half-pel
  averaging, and a hexagon walk that measures only the three new positions
  after a move (in the listed order, so ties still go the same way).
* **Motion search without revisits.**  A position measured earlier in the
  search lost to the best at the time, and the best only falls, so it can
  never be strictly cheaper later: the hexagon/square rounds skip what the
  previous round measured (a hexagon that moved by one step shares three
  points with the last one, and the first square shares four with the last
  hexagon), the sub-pel rounds skip their previous round's 3x3, the start
  candidates are de-duplicated and measured four at a time, and the hexagon
  walk takes the start's cost from the caller instead of measuring it again.
  (A per-search stamp table over all stages removed more SADs and was no
  faster -- the four-way SAD only gains when a group of four goes.)
* **Trellis converges early:** a trial depends only on the current state, so
  once a whole sweep's worth of trials has gone by without a move, the second
  sweep would repeat the first one's rejections and is skipped (-2.5 %).
* **Intra mode decision:** two modes that predict the same samples share one
  evaluation (only the mode bits differ); the trellis pass over the best two
  modes starts from the levels, SSE and price the plain pass already measured;
  and the winner's selected levels are coded as they are instead of being
  quantised and selected again (-8 % together).
* **Chroma memo:** a macroblock's chroma depends only on its prediction, the
  QP and intra/inter while `code_mb()` decides, so the result (levels, flags,
  SSE, reconstruction) is kept per macroblock and replayed -- every intra
  candidate under the chroma-aware comparison, the winner's final chroma pass
  (-2.6 % at preset 4).  The same memo for the luma of inter candidates hit
  13 % of them and saved nothing measurable, so it is not in.

cock 90 frames, QP 28, this two-core sandbox, the two builds run back to back:

| | 40c1301 | now |
|---|---|---|
| preset 3, one thread | 11.1 s (8.1 fps) | 8.9 s (10.1 fps) |
| preset 3, frame threads | 6.2 s (14.6 fps) | 5.1 s (17.7 fps) |
| preset 4, one thread | 12.6 s (7.2 fps) | 9.9 s (9.1 fps) |
| preset 4, frame threads | 6.9 s (13.0 fps) | 5.4 s (16.7 fps) |

The profile is flat now (preset 3, one thread: the four-way SAD 6 %, the
macroblock search's own code 5 %, quarter-pel SATD 5 %, MC 5 %, the 4x4
inverse transform 4 %, the CABAC pricer 4 %, nothing else above 3 %); what is
left is mostly decisions, which is the early-termination stage (each gated at
+0.1 % BD-rate).

## P_8x8 without the sub-8x8 shapes

The P_8x8 candidate chose, per 8x8 sub-macroblock, between 8x8, 8x4, 4x8 and
4x4 by SATD plus header bits -- nine partition searches per sub-macroblock
before the early exit.  Leaving the three smaller shapes out (they are now
`--partitions` bit 16, off by default; x264's medium preset has no p4x4
either), ours against ours, `ABBD_GLOBAL=1`, one thread:

| preset | 5 clips | validation set | time |
|---|---|---|---|
| 1 | -1.04 % | | -10.5 % |
| 2 | -0.27 % | | -11.0 % |
| 3 | -0.10 % | -0.12 % | -5.8 % / -4.8 % |
| 4 | +0.03 % | -0.07 % | -3.9 % / -2.3 % |

So the small shapes were not paying at any preset: the SATD shape decision
picks them, and the RD pass either rejects the macroblock or takes a motion
field that costs more than it predicts.  Decoder-exact (`exact.sh` presets 1
and 3), frame threads identical (`ftcmp.sh` 32/32), `--partitions 31` writes
exactly the old default's stream.

The same kind of check on the Intra_16x16 candidates: code_mb() fully coded
the three SAD-best modes; the best two now (`AEG_I16N=3` restores three).
Preset 3: -0.05 / -0.06 % (5 clips / validation set) for -5.6 / -5.1 % time;
preset 4: +0.01 / +0.07 % for -2.7 / -4.5 %; preset 2: +0.01 %, no time.
One mode is +0.10 % at preset 3 (-10.6 %), too close to the gate; ranking the
modes by SATD rather than SAD helped neither one nor two modes.  Tried and not
taken: fewer sub-pel walk steps (the walk is cheap -- -1.5 % time at most),
P_8x8 off entirely (+0.5 % at presets 3/4, +3.2 % at preset 1).

**No level selection on Intra_16x16 AC blocks** (`AEG_TI16=1` restores it):
off is +0.04 / -0.07 % at preset 3 (5 clips / validation set, -7.9 / -9.3 %
time), +0.04 / -0.05 % at preset 4 (-5.8 / -6.9 %), and +0.04 % on all-intra
streams (`-g 1`, preset 3, -15 %).  The other two switches of the same kind
stay: chroma level selection off is +2.2 %, the bi-predictive refinement of
B partitions off +0.14 % for -4 %.

**Early exit before the multi-hexagon stage** (x264's UMH has the same kind
of exit; `AEG_UMHET=0` turns it off, `AEG_UMHET=n` sets the threshold): when the
best integer predictor is the predicted MV itself and its SAD + MV cost is
below n/256 per sample, only the hexagon walk runs.  n = 1000 below preset 4,
500 at preset 4:

| preset | threshold | 5 clips | validation set | time |
|---|---|---|---|---|
| 1 | 1000 | +0.01 % | | -4.6 % |
| 2 | 1000 | +0.07 % | | -4.2 % |
| 3 | 1000 | +0.01 % | +0.11 % | -6.6 % / -6.1 % |
| 3 | 500 | +0.06 % | +0.05 % | -5.4 % / -3.7 % |
| 4 | 1000 | +0.08 % | +0.06 % | -7.6 % / -8.6 % |
| 4 | 500 | +0.03 % | +0.07 % | -6.1 % / -6.2 % |

Without the predictor condition the exit costs more (threshold 1000, preset
3: +0.09 % on the 5 clips).  `AEG_UMHET=0` writes the previous default's
stream at every preset; decoder-exact at presets 3 and 4, frame threads
identical 32/32.

**B_8x8 at preset 4 only** (`AEG_B8X8=1` restores it at preset 3, `=0`
turns it off at 4).  Once the other searches had become cheaper the shape's
share of the time grew; off at preset 3 measured +0.04 / +0.05 % (5 clips /
validation set) for -3.7 / -2.5 % of the time.  At preset 4 the same is
+0.08 / +0.06 % for -4.2 / -2.6 %, which the preset-4 budget no longer
covers, so it stays there.  `AEG_B8X8=1` writes the previous preset-3 stream;
decoder-exact at preset 3, frame threads identical 32/32.

**B 16x8 / 8x16 only where the 16x16 prediction is poor** (`AEG_BPGL=k`,
0 = always): a B macroblock whose better one-list 16x16 cost (SATD + vector
bits, from the search) is below k motion lambdas skips the two partitioned
shapes -- four searches and two joint refinements.  k = 96 below preset 4,
64 at preset 4, ours against ours:

| preset | k | 5 clips | validation set | time |
|---|---|---|---|---|
| 1 | 96 | +0.03 % | | -8.0 % |
| 2 | 96 | +0.09 % | | -9.4 % |
| 3 | 64 | +0.01 % | +0.01 % | -5.3 % / -2.8 % |
| 3 | 96 | +0.07 % | +0.04 % | -7.8 % / -7.3 % |
| 3 | 128 | +0.19 % | | -11.8 % |
| 4 | 64 | +0.03 % | -0.02 % | -6.2 % / -4.2 % |

At preset 4 the same gate also covers B_8x8 (`AEG_BPG8=0` lifts it): +0.01 /
+0.01 % for -1.3 / -1.5 %.  The same gate on the P partitions does not pay
(16x8 / 8x16 / P_8x8 at k = 64: +0.06 % for -1.7 %; k = 128: +0.40 % for
-4 %): a P macroblock's 16x16 cost says much less about its partitions.

A fixed threshold instead of one in lambdas is worse at every QP mix (512:
+0.08 % for -2.9 %, 1024: +0.26 % for -11 %), because the partitions matter
more at the low QPs where lambda is small.  `AEG_BPGL=0` writes the previous
stream at presets 2-4; decoder-exact at presets 3 and 4, frame threads
identical 32/32.

Also measured and not taken: pre-selecting the Intra_4x4 / Intra_8x8 modes
by SATD before the exact price (`AEG_I4K` / `AEG_I8K` 5 or 6: +0.05-0.10 %
for at most -1 % -- the exact price of a small block costs about what the SATD
does), and an AVX2 four-way SAD for 16-wide blocks (two rows per register:
no measurable change).  The SSE2 4x4 quantiser now does the sixteen
coefficients at once (identical output, ~1 %).

**The four before it together** against the defaults before them (57ebcb0; reproduced by
`AEG_I16N=3 AEG_TI16=1 AEG_UMHET=0 --partitions 31`), since each was gated
against the one before: preset 3 -0.10 / -0.13 % (5 clips / validation set)
for -22 / -21 % of the one-thread time over the ten clips; preset 4 +0.11 /
+0.03 % (+0.07 % over the ten) for -20 / -19 %.  cock 90 frames, QP 28, back to
back against 42f1f92 (the same streams as 57ebcb0), best of three:

| | 42f1f92 | now |
|---|---|---|
| preset 3, one thread | 8.84 s (10.2 fps) | 7.47 s (12.1 fps) |
| preset 3, frame threads | 5.22 s (17.2 fps) | 4.52 s (19.9 fps) |
| preset 4, one thread | 10.70 s (8.4 fps) | 8.45 s (10.7 fps) |
| preset 4, frame threads | 5.63 s (16.0 fps) | 4.89 s (18.4 fps) |

**All of this section together** (the seven gated changes up to 58dadaf:
sub-8x8 shapes, I16N, TI16, UMH early exit, B_8x8 at preset 4, the B
partition gate and its B_8x8 extension) against 57ebcb0, reproduced by
`AEG_I16N=3 AEG_TI16=1 AEG_UMHET=0 AEG_B8X8=1 AEG_BPGL=0 --partitions 31`:
preset 3 +0.01 / -0.04 % (5 clips / validation set) for -32 / -30 % of the
one-thread time; preset 4 +0.15 / +0.02 % (+0.085 % over the ten clips) for
-25 / -24 %.  Preset 4 is therefore at the edge of the 0.1 % budget and takes
no further lossy prunes; preset 3 has spent none of it net.  Timed on a
different (faster) sandbox from the table above, so both builds again back to
back, cock 90 frames, QP 28, best of three:

| | 42f1f92 | 58dadaf |
|---|---|---|
| preset 3, one thread | 6.98 s (12.9 fps) | 5.02 s (17.9 fps) |
| preset 3, frame threads | 3.75 s (24.0 fps) | 2.87 s (31.3 fps) |
| preset 4, one thread | 7.53 s (12.0 fps) | 5.89 s (15.3 fps) |
| preset 4, frame threads | 4.13 s (21.8 fps) | 3.24 s (27.8 fps) |

**One Intra_16x16 mode below preset 4** (`AEG_I16N=2` restores two): with
the Intra_16x16 AC level selection off, the second mode is worth less than it
was; one mode measured +0.08 / +0.02 % at preset 3 (5 clips / validation
set) for -4.9 / -2.0 % of the time, +0.07 % at preset 2 for -4.4 %, and +0.09 %
on all-intra streams (`-g 1`, preset 3) for -4.3 %.  Preset 4 keeps two.
With it, preset 3 has spent about +0.04 % of its budget net over the round.
Also tried for preset 3 alone: the UMH early exit at 1500 (+0.06 / -0.01 %
for -2.8 / 0 %), and skipping the coder's price of a candidate-refinement
neighbour whose luma distortion already loses (exact, no measurable time).

Also tried in this round and not taken (each against the then default,
preset 3): the lambda-scaled gate on the second-reference 16x16 search (k =
32: -0.01 % for -2.4 %, k = 64: +0.10 % for -1.1 % -- within the noise of the
timer), skipping Intra_4x4 / Intra_8x8 when the best inter price is below k
lambdas (k = 10-40: +0.02-0.11 % for at most -3 %, not monotone), the
`AEG_PSUB=1` / `AEG_REF1C=0` / `AEG_MENB=0` / `AEG_BIDIR=0` switches (+0.52 /
+0.07 / +0.18 / +0.09 % for -2.7 / -0.1 / -0.6 / -1.6 %), and skipping
measured positions in the joint bi-predictive walk (exact, but -0.2 %).
Pruning inter candidates on their luma SSE inside the luma coding is not worth
building: only 5 % of the inter candidates are ruled out by the luma
distortion at all, and 2.4 % of them repeat an earlier candidate's motion.

## Back to quality: the high-quality end

Against x264 medium (`psy=0`, constant QP, whole-sequence PSNR) after the
speed round, preset 3: -4.77 % (cock +0.46, vtest -10.98, mega -6.31, short
-9.31, tree +2.31); preset 4 -5.58 %.  Split by QP: over 16-25 cock +2.32,
vtest -12.11, mega -3.92, short -12.03, tree +2.75; over 28-37 -0.21, -12.90,
-7.78, -8.20, +0.34.  Where the gap is:

* not intra: all-intra (`-g 1 --bframes 0` against `keyint=1`) we are -1.3
  to -2.6 % at QP 16-25 on every clip;
* in P coding: P only (`--bframes 0` against `bframes=0`) we are -1.6 to
  -5.6 % at QP 28-37 but +0.8 % (cock) and +1.7 % (tree) at 16-25;
* not the motion refinement: `--refine-cand 1` is -0.17 % at QP 16-25 for
  +62 % time, `--refine-mv 3` -0.08 % for +29 % (x264's own subme 7 -> 9 is
  -1.9 % there, which our RD decisions already have most of);
* not the mode lambda of P pictures (0.5x: +0.57 %, 0.85x: +0.29 % at QP
  16-25, P only), nor the macroblock tree's strength (weaker below QP 28
  helps cock and tree by 0.3-0.5 % and costs vtest 6-14 %), nor the I offset
  (`--iqp 2`: -0.45 % at QP 16-25 but -0.02 / +0.53 % over the usual range);
* a third reference is worth -0.29 % to x264 itself (-0.18 % at QP 16-25).

**The level selection's lambda on inter blocks** was the plain lambda of the
QP, while the mode decision prices P pictures at 0.65x.  At 0.75x (both P and
B; `AEG_TLI` in percent): preset 3 -0.34 / -0.20 % (5 clips / validation set),
preset 4 -0.33 / -0.15 %, and -0.40 % over QP 16-25 (cock -0.58, tree -0.18),
for no time.  On P alone -0.33 %, on B alone -0.10 %; 0.6x -0.28 %, 0.45x
-0.35 % at QP 16-25 (vtest turns positive); 1.3x +0.89 %.  `AEG_TLI=100`
writes the previous stream; decoder-exact at presets 3 and 4, frame threads
identical 32/32.

**Chroma in the mode decision from preset 2** (`AEG_CDECIDE`; it was preset
4 only, at -0.34 % for 10 % of the time when it was added).  Pricing each
candidate with its chroma coded too measured, at QP 16-25 and preset 3,
-1.02 % (cock -1.13, mega -2.30, short -1.22): the chroma share of the rate
grows as the QP falls.  Over the usual range: preset 3 -0.51 / -0.73 % (5
clips / validation set) for +7 / +10 % of the time, preset 2 -0.49 / -0.58 %
for +9 %.  The speed round bought ten times that time for a tenth of the
loss, so this is the better place for it.  `AEG_CDECIDE=0` writes the
previous stream; decoder-exact at presets 2 and 3, frame threads identical.
After both (ab1405e), against x264 medium `psy=0` at constant QP: preset 3
**-5.37 %** (cock -0.73, vtest -11.50, mega -7.28, short -9.39, tree +2.04;
over QP 16-25 cock +0.64, tree +2.40), preset 4 -5.84 %, and -0.99 % against
x264 at CRF with its macroblock tree (`aq-mode=0`).  Timed back to back with
the end of the speed round (cock 90 frames, QP 28, best of three, this
session's sandbox): preset 3 14.8 -> 13.7 fps on one thread, 25.0 -> 23.6
with frame threads; preset 4 unchanged (11.8 / 21.0 -> 12.0 / 21.5); 42f1f92
on the same machine 10.5 / 17.6 and 9.0 / 17.0.  (Later, with the trellis
B-group decision by default at presets 3-4, `336a162`: preset 3 -5.51 %,
preset 4 -5.91 %, CRF -1.63 %; see *Quality and speed round after CI* below.)

tree is the clip still behind, and it is a peculiar one: each picture is
repeated six to ten times (four different pictures in thirty), so the result
is the I picture and three P pictures.  Without the macroblock tree we are at
parity with x264 there (QP 22: 69 038 B / 42.39 dB against 65 538 B / 42.07
dB); with it we lose about 1 % at QP 16-25.  Bounding the tree's offsets
(-4 QP: tree -1.27 %, vtest +3.99 %; -6: nothing) cannot tell tree's repeated
pictures from vtest's static background, so it is left alone.

Also measured at QP 16-25, preset 3, and not taken: chroma in the motion
search (`AEG_CHROMAME=1`, -0.20 %), the inter chroma level selection at 0.75x
/ 1.3x lambda (-0.03 / +0.15 %), the Intra_8x8 level selection at 0.75x /
0.65x lambda in P and B pictures (-0.04 / +0.03 %).

## Quality and speed round after CI (b-adapt 2, lookahead fields)

**tree, once more.**  The macroblock tree gives tree's I picture 3.9 QP more
than its refresh P pictures (x264 1.5), but nothing at picture level tells
tree's repeated pictures from vtest's static background or cock's busy P
pictures: tree strength 1.0 / 1.6 (`AEG_MBTS`) +1.03 / +3.40 against +2.04,
traded against vtest; per-picture flat offsets (the tree's mean only) -5.19
% mean; half the per-macroblock spread -5.28; `--iqp 1 / 2` -5.10 / -5.33
(all against -5.37); a "new information" share of each refresh picture
(0.24-0.40 on tree) overlaps cock's P pictures (0.23-0.61).  Left alone.
P-only at QP 16-25 tree is +1.30 (the others -0.6 to -6.0): at QP 20 our
refresh P pictures cost ~7% more than x264's at equal PSNR with the same
macroblock type counts (261 coded against ~281, no intra) -- the
P/B-coding gap described under "Why 3-B groups lose on short", no knob
(inter trellis lambda 0.5x / 1.33x -0.44 / +0.57 % on a 7-picture cut,
`--no-8x8dct`, `--partitions 1`, `--refine-mv 3`, `--range 32`, preset 4:
within +-0.4 %) moves it.  The validation tree segment was the bigger loss
(+9.90 %), and that was the group decision: see "B-picture decision" above
-- b-adapt 2 is now the default at presets 3-4.

A third reference was not added: x264's own ref 3 -> 1 costs it only +0.22 %
mean (table above) and ref 2 -> 3 is -0.18 % at QP 16-25.

**Lookahead fields computed once (exact).**  The search of one lowres
picture against one reference depends on the pair only (a block's
candidates and predictor are its neighbours' vectors in the same field),
yet the group decision priced it inside every (p0, p1, b) estimate and the
macroblock tree searched it again: on cock 663 searches for 427 distinct
pairs and 471 intra estimates for 87 pictures.  `la_field()` / `la_intra()`
keep the vectors and costs per ring slot (8 references each, entries of
pictures that left the ring reused first) and `aeg_lowres_cost_fields()` /
`aeg_lowres_mbcosts_fields()` assemble the estimates; only the bi-predicted
cost depends on the (p0, p1) pair.  Byte-identical on 70 cases (7 clips x
presets 1-4, b-adapt 1/2, `--mbtree 0`, frame threads, ABR, `-g 12
--rc-lookahead 8`).  cock 90 frames, QP 28, best of three: preset 3 7126 ->
6663 ms (-6.5 %), preset 4 8243 -> 7751 ms (-6.0 %) on one thread; preset 3
with frame threads 3792 -> 3520 ms (-7.2 %, 25.6 fps).

With frame threads the main thread (prologue: lookahead, tree, scheduling)
is now 6.3 % of all samples (95 of 1512), almost all of it the lowres
search; it runs beside the workers, so it bounds throughput only near 16x
one core.  The kernels left in the per-macroblock profile (SATD, SAD x4,
MC, quantisation, IDCT, the CABAC pricing) are all SSE2/AVX2 already and
none is above 6.5 % self time, so more exact SIMD work buys little.  The
wavefront lets a P picture's row r start once its reference has rows
r + 2-3 (range + 4 + 15 samples), and with three B pictures per group the B
pictures are independent once their anchors are done, so 4-8 workers have
work; scaling itself cannot be measured on this two-core sandbox.

## The skip candidate

`P_Skip` is the cheapest macroblock the standard has: one context-coded flag,
no motion vectors, no residual, and the decoder reconstructs it with a vector
derived from its neighbours (8.4.1.1).  Coding one therefore needs the
macroblock's own vector to be *exactly* that derived vector, and the motion
search never produces it: it refines sub-pel away from its seed, so a
macroblock whose true motion is "nothing moved" still ends up with a vector a
quarter pel off, is coded as a normal P16x16 with a residual, and costs a
hundred bits where a skip costs two.

The fix is one extra candidate: the search runs as usual, and then the same
16x16 candidate is added again with the vector `pskip_mv()` derives.  It is the
same prediction machinery, the same residual path and the same price, so the
two candidates are directly comparable; the skip test in `write_mb_cabac()`
then sees a candidate it can actually code as a skip, and the decision pass
prices it as one flag.  On a still clip (mode 3, 320x192, QP 32, 30 frames) the
stream drops from 6 165 to 4 201 bytes at the same PSNR (39.97 dB) -- 32% of
the bits, for one more candidate per macroblock.

One thing the extra candidate exposed: the candidate array was bounded by
four, exactly the number of inter partitionings, so the skip candidate silently
pushed P_8x8 out of the list.  The bound is now the real count (five) and the
macroblock carries a comment saying what they are; P_8x8 wins on none of the
clips here, so putting it back changes no measured number, but a dropped
candidate is not a decision the encoder made.

The median predictor as a candidate of its own (mv = the neighbour median, so
no mvd at all) was tried next and is *not* in: it is the same idea as the skip
vector, but the search already starts from that vector and only leaves it when
SATD says so, and pricing both costs 2.3% more bits on the two-quadrant clip at
a fixed QP (the extra candidate wins some macroblocks the estimate said were
worse, which means the mvd-free advantage is not what the price model thinks it
is).

This is also where the coder-priced decision pass pays off twice: the same
candidate priced by the CAVLC estimate is still worth coding (the estimate
charges *more* for the residual than the coder does, so it prefers the skip
less than it should), and it is the combination that shows in the BD-rate
table above (mode 3: 27.7% more bits before either change, 10.2% fewer with
both).

## Rate-distortion refinement of the motion vector

The motion search picks a vector by the SATD of the *prediction*, which cannot
see what the residual will cost or what the mvd will cost: a vector half a
pel away can have a marginally worse prediction and a much cheaper bitstream
(x264's `subme` scale is exactly this idea).  The slow presets therefore re-code
the winning P16x16 macroblock with each neighbour of its vector in quarter-pel
steps and keep whichever the coder's own price prefers -- `SSD + lambda * bits`
with `bits` from the shadow coder, so the mvd, the residual and the
macroblock's own syntax are all in the comparison.  Preset 3 tried the four
axial neighbours, preset 4 all eight (both now rank the eight by SATD and
re-code the best three or four -- *Less motion work at presets 3 and 4*);
`--refine-mv 0|1|2|3` overrides it, and
preset 2 and below do not refine at all.  B pictures refine both lists of every
partition the same way (*The same refinement in a B picture* below).

Measured as BD-rate against libx264 medium over QP 22..37, negative = we need
fewer bits; the middle column of each preset refines only the 16x16 winner, the
last one refines each partition of a 16x8 or 8x16 winner as well:

| clip | preset 2 (off) | preset 3, 16x16 | preset 3, per partition | preset 4, 16x16 | preset 4, per partition |
|---|---|---|---|---|---|
| mode 4, 320x240, 8 frames | +18.6% | +17.7% | +17.2% | +17.4% | +17.1% |
| mode 2, 320x192, 30 frames | +24.7% | +24.0% | +23.3% | +23.9% | +23.5% |
| mode 3 (still), 320x192, 30 frames | -10.2% | -12.0% | **-14.4%** | -15.4% | -13.8% |
| mode 1, 320x192, 60 frames | +27.8% | +26.1% | +23.6% | +25.7% | +24.3% |

The cost is one full candidate evaluation per neighbour: 320x192 QP 24, 60
frames, preset 2 runs at 69 fps, preset 3 at 50 and preset 4 at 40 on this
machine.  Refining the partitions as well costs a further 5% of the time and is
worth 0.3-0.7 points on every clip but the still one, where the measurement is
inside that clip's own instability (see the mode-3 paragraph above; its two
preset-4 columns differ by 1.6 points in the *wrong* direction, and the
preset-3 columns differ by 2.4 in the right one, which is what a curve that
crosses itself twice looks like).  The panning clip gains the most: it is the
one whose content is two regions moving apart, so 16x8 pays there, and a
partition's vector is where the search's quarter-pel error is.

The refinement runs before the chroma decision (the chroma prediction follows
the luma vector) and restores the luma plane afterwards, because every trial
wrote its own reconstruction there; the winner's reconstruction is
re-snapshotted each time it wins, so what the caller codes is the
reconstruction the price was computed from.  A partition is refined with its
sibling's vector held at the value the search returned, one partition after the
other, so what is priced is always a whole macroblock -- the encoder prices
macroblocks, not sub-regions.  P_8x8 is the one shape left out: four partitions
times eight neighbours is 32 full candidate evaluations, more than the rest of
the macroblock costs, and its four vectors come from a search that already ran
per 8x8 block.

### The same refinement in a B picture

A B macroblock had been the one place where the vectors were never revisited:
they come out of the per-list SATD search and go straight into the RD
comparison, so `--refine-mv` -- the thing that makes preset 3 and 4 what they
are on P pictures -- did nothing there.  It does now, per list and per
partition: each partition's list 0 vector is refined with the other vectors
held at the value the search returned, then its list 1 vector, and every trial
is a whole-macroblock re-code priced by the coder (`SSD + lambda * bits`).  The
predictors have to be re-derived after every accepted move, in *both* lists,
because a partition's vector and its list mask are both part of the state its
neighbours predict from -- the same rule as bug 43, and the reason
`refresh_pmvs()` now speaks both lists.

BD-rate against libx264 medium, preset 3, `--bframes 3 --scenecut 0`, QP
22..37, negative = we spend fewer bits at matched PSNR:

| clip | before | with the B refinement |
|---|---|---|
| mode 0, 320x192, 60 frames | -23.87% | **-24.78%** |
| mode 1, 320x192, 30 frames | -12.09% | **-12.83%** |
| mode 4, 320x192, 10 frames | -7.35% | **-8.02%** |
| mode 3 (still), 320x192, 30 frames | -22.70% | -22.70% |
| mode 0, 320x192, 30 frames | -22.64% | **-23.47%** |

That is 0.4-0.9 points on four of five clips for 9 % of the encode time at
preset 3 and 2 % at preset 4 (where the rest of the preset already costs more).
The still clip is unchanged, which is what it should be: nothing moves there,
so nothing has a quarter-pel neighbour worth trying.

What the refinement must *not* touch is a direct macroblock.  Its vectors are
derived from the co-located picture and are not in the bitstream at all, so
moving one changes the encoder's reconstruction and nothing else: the first
version refined direct blocks too and every B picture desynchronised from its
first macroblock (the decoder derives 12,8 where the encoder then predicted
with 11,7).  The gate is `best.type != AEG_MB_B_DIRECT`; B_8x8's *groups* are
left out separately, because four groups times two lists times four neighbours
is 32 full candidates per macroblock.

Three other ways to spend effort on the B side were measured and dropped,
which is worth recording because they were the obvious ones:

* **Pricing the per-partition list choice with lambda instead of a fixed eight
  bits per mvd bit.**  The search picks a partition's lists (and a B_8x8
  group's) by `SATD + 8 * bits`, a weight that is the same at QP 12 and QP 40.
  Replacing it with `lambda * bits` -- the same SSD-calibrated lambda the RD
  pass uses, on the argument that SATD and SSD differ by a constant factor --
  changed the BD-rate by 0.01 points on every clip.  The list choices are
  simply not near that boundary.
* **Refining the list choice exactly.**  Every partition was re-coded with
  each of the other two list masks and kept if the coder preferred it.  That
  is *worse* (-12.09% → -10.77% on the mode-1 clip), because the choice is
  made on luma SSD while the measurement is over Y+U+V and the chroma follows
  the luma decision: a bi-predictive partition averages its chroma too, and the
  trial cannot see that.
* **A per-partition subpel search.**  The partitioned B shapes search
  integer-pel and then inherit the macroblock's own quarter-pel offset; giving
  them the ordinary `subme` refinement instead is neutral (-23.81% vs -23.87%)
  and 3 % slower.

B_8x8 remains nearly unused (2 of 2400 macroblocks on the quadrant clip) even
with the refinement in place, so its problem is not the pricing of its groups
either: its vectors are searched per 8x8 with the macroblock's fractional
offset and, unlike the 16x16 candidates, they are not refined.

### Refining every candidate, not just the winner

The refinement above runs *after* the comparison: the candidate that won is
re-coded with the neighbours of its vector.  That leaves the rest of the list
untouched, and a candidate whose vector is a quarter pel off loses the
comparison by a margin the refinement never gets to see -- the search that
produced its vector was SATD-ranked, and SATD is the same proxy the whole
comparison is trying to beat.  `--refine-cand 1` moves the refinement *into*
the candidate loop: each inter candidate is re-coded with its four axial
quarter-pel neighbours per partition and list before it is priced against the
others, and the candidate that then wins carries the refined vector into the
bitstream.

The trial is the same shape as the winner's: the whole macroblock is re-coded,
priced by the coder (`SSD + lambda * bits` on the shadow coder), one partition
and one list at a time with the rest held at the searched values, so what is
priced is always a whole macroblock.  It needs `--refine-mv` on (the neighbour
set and the quarter-pel convention come from it), skips direct macroblocks for
the same reason the B refinement does, and skips shapes with more than two
partitions, which is the same budget the winner refinement works within.

BD-rate against libx264 medium, preset 4, `--bframes 3 --scenecut 0`, QP 22..37,
negative = we spend fewer bits at matched PSNR:

| clip | preset 4 | preset 4, `--refine-cand 1` |
|---|---|---|
| mode 0, 320x192, 60 frames | -24.38% | **-24.97%** |
| mode 1, 320x192, 30 frames | -12.58% | **-12.91%** |
| mode 4, 320x192, 10 frames | -7.52% | **-7.68%** |
| mode 3 (still), 320x192, 30 frames | -22.36% | **-22.81%** |
| mode 0, 320x192, 30 frames | -23.63% | **-23.86%** |

Every clip gains, 0.3 points on average, for 40 % of the encode time (0.84 s ->
1.18 s for 60 frames at 320x192 QP 28).  That is a worse exchange rate than the
winner refinement's (1.2-4.2 points for 30-45 %), which is why the winner
refinement follows the preset and this one does not: a candidate costs eleven
macroblock evaluations once its four neighbours are tried, and most of the
candidates are not close enough to the winner for a quarter pel to change the
outcome.

Which candidates are close enough is the one tuning decision in it.  Refining
*every* candidate costs 2.6x the encode time of the base and measured the same
as the gated version; refining the ones within 12 % of the best so far
(`cost <= best_cost + best_cost / 8`) or within 3 % was also the same, inside
each clip's noise (-13.13% vs -12.92% on the mode-1 clip, +0.1 on the mode-0
one).  The rule kept is the strictest and cheapest to explain: a candidate is
refined while it is at least as good as the best one before it, so the first
candidate of every macroblock is always refined and the ones that arrive
behind are priced as the search found them.

The conformance gate runs it both ways -- one P case and one B case with
`--refine-cand 1` at preset 4 and every partition shape enabled -- because the
risk here is specific: the refined vector is the one the bitstream carries, so
the predictor refresh after an accepted move has to happen exactly as it does
for the winner.  The 252-case matrix (three clips x bframes 0/1/3 x QP 20/26/32
x preset 3/4 x partitions 1/5/7/15) is bit-exact with it on.

## P_8x8 sub-macroblock partitions

P_8x8 (mb_type 3) splits the macroblock into four 8x8 partitions, each with its
own motion search, its own MV predictor and its own `mvd`, and each 8x8
sub-macroblock independently picks its `sub_mb_type`: 0 = 8x8, 1 = two 8x4,
2 = two 4x8, 3 = four 4x4 (table 7-17).  The shape is chosen per sub-macroblock
by `SATD + 8 * bits` over the shapes' own `sub_mb_type`, `mvd` and (for the
finer shapes) 4x4 vector costs; the macroblock as a whole is then compared
against 16x16/16x8/8x16 by the usual `SSD + lambda * bits` RD pass, so a finer
shape only reaches the bitstream if it pays for its header.  The decision uses
SATD rather than SAD: at 4x4 scale a small DC error can make a badly aligned
block look cheap in SAD, and the measurement agrees -- switching the decision
metric from SAD to SATD is worth +0.4 dB at 300 kb/s and +0.2 dB at 500 kb/s on
the quadrant clip and +0.2 dB at 300 kb/s on the panning one, for ~3 % of the
encode speed.  The `SAD + w * bits` weight was swept over 4, 8, 16 at three
rates; 8 is the best of the three.

Measured on the four synthetic 320x192 clips (`genyuv` modes 0 … 4, 24 frames,
QP 24), default mask `1|2|4|8` = 15:

| clip | mask 7 (16x16/16x8/8x16) | mask 15, 8x8 only\* | mask 15, all shapes |
|---|---|---|---|
| global pan | 27 487 B, Y 45.07 dB | 21 976 B, Y 45.36 dB | 22 723 B, Y 45.32 dB |
| halves moving apart | 51 344 B, Y 44.34 dB | 42 395 B, Y 44.80 dB | 41 183 B, Y 44.82 dB |
| left/right halves | 41 166 B, Y 44.91 dB | 36 269 B, Y 45.17 dB | 35 200 B, Y 45.18 dB |
| four quadrants moving apart | 52 320 B, Y 44.44 dB | 44 871 B, Y 44.82 dB | 43 707 B, Y 44.82 dB |

\* the middle column was measured before the sub-shapes landed; it cannot be
reproduced with the current binary.  8x4/4x8/4x4 add ~3 % on the clips whose
motion is uniform per 8x8 block (there the 8x8 shape already wins) and cost
nothing measurable in PSNR; on the multi-directional clips they take a further
1-3 %.  The per-sub-macroblock
weight `w` in `SAD + w * bits` was swept over 4, 8, 16, 32, 48, 64 at QP 18,
24 and 32 — 8 is the best or within 0.3 % of it everywhere, so it stays.

Lower rate *and* higher PSNR at the same QP: the extra candidate only wins where
the finer motion field pays for its header, because the decision is a real
`SSD + lambda * bits` comparison that includes the four `sub_mb_type` symbols.
It costs about 30 % of the encode speed over mask 7 (124 -> 87 fps on the
320x192 clips, two cores), because the 4x4 shape runs sixteen motion searches
per macroblock and all four shapes are searched per sub-macroblock.  `x264` (master, baseline, no asm) on the same clips at QP 24:
18 841 B / 46.14 dB, 33 003 B / 45.83 dB, 31 897 B / 45.83 dB, 35 074 B /
45.43 dB.  Those numbers were taken with the quarter-pel stage *off* (bug 29);
with it on, the gap at matched PSNR is 1-8 % (see "Quality against a reference
CPU encoder" below).

## Rate control and adaptive quantisation

`--bitrate <kbps>` switches from constant QP to one-pass average-bitrate rate
control.  It is a picture-level controller: the debt between the budget for the
pictures coded so far and the bits actually spent is spread over the next six
pictures, and the QP follows the logarithm of (actual / wanted) bits, because
the rate is roughly exponential in the QP.  The QP moves by at most 1.5 per
picture and stays inside [0, 51], so a scene cut cannot make the quality
collapse or explode.  Measured over 90 frames of two 320x192 clips:

| target | achieved (pan clip) | achieved (quadrant clip) |
|---|---|---|
| 250 kb/s | 250.4 (100.2 %) | 250.4 (100.2 %) |
| 400 kb/s | 400.8 (100.2 %) | 400.5 (100.1 %) |
| 600 kb/s | 600.7 (100.1 %) | 601.0 (100.2 %) |
| 900 kb/s | 901.2 (100.1 %) | 901.3 (100.1 %) |

Targets below the QP-51 floor of the content are not reachable: at 320x192 this
synthetic texture costs 244 kb/s at QP 51, so a 150 kb/s target ends at 244
kb/s (the controller sits at QP 51 and reports the overshoot).  Over 24-frame
streams the first IDR picture, which is ~7x a P picture, is absorbed by the
debt term and the average lands within 1-3 %.

Adaptive quantisation (`--aq <tenths>`, `aq_mode`/`aq_strength` in the API) is
the variance-AQ of x264: per-macroblock QP offsets proportional to
`log2(frame variance) - log2(block variance)`, normalised to a zero mean so the
picture QP is unchanged, clipped to +/-12, carried by `mb_qp_delta`.  It is
implemented and conformant but **off by default**, because measured at a fixed
rate it *costs* PSNR on the test set:

| clip, target | AQ off | AQ 1.0 |
|---|---|---|
| panning texture, 300 kb/s | Y 41.38 dB | Y 40.50 dB |
| panning texture, 500 kb/s | Y 46.05 dB | Y 45.84 dB |
| quadrants, 300 kb/s | Y 43.58 dB | Y 42.95 dB |
| quadrants, 500 kb/s | Y 49.66 dB | Y 49.45 dB |

The reason is convexity: a zero-mean *QP* offset still raises the total rate
(bits grow exponentially in QP), so the controller answers with a slightly
coarser average QP and every macroblock pays for it.  x264 accepts the same
PSNR loss for a perceptual gain on natural video; since this project's yardstick
is PSNR parity with x264, AQ stays opt-in until it has been re-tuned (a
rate-preserving normalisation, or aq-mode 2 style offsets, is the next step).

One bitstream detail that AQ forces into the open: `mb_qp_delta` is only present
when the macroblock carries a coded residual, so a macroblock with an all-zero
`coded_block_pattern` must keep the decoder's current QP.  The encoder therefore
falls back to the previous QP for such macroblocks (the reconstruction is the
prediction either way, but the deblocking filter reads the QP).  Getting that
wrong desynchronises the deblocking on the very first flat macroblock.

## Tried and rejected: per-block RD level tuning ("last non-zero coefficient")

The obvious cheap relative of trellis quantisation is to let each 4x4 block
drop its highest-frequency non-zero coefficient (or empty the block entirely)
when `sse + lambda * bits` says so, measuring both terms exactly: inverse
transform and SSE for the distortion, and the real CAVLC writer for the bits.
It is implemented, conformance-clean and *not worth it on this content* -- at a
fixed rate it is PSNR-neutral while costing 5-8 % of the encode speed:

| variant | 300 kb/s | 500 kb/s |
|---|---|---|
| no tuning | Y 41.38 dB | Y 46.05 dB |
| drop last coefficient, lambda x1 | 41.45 | 45.87 |
| drop last coefficient, lambda x0.25 | 41.39 | 46.04 |
| drop last coefficient, lambda x2 | 40.54 | 45.59 |
| allow emptying the block, lambda x1 | 41.20 | 45.83 |

The reason is that on this synthetic texture the last coefficient of a block is
rarely "free" enough: dropping it removes real detail, and the bits it saves are
already recovered by the macroblock-level RD pass choosing a coarser partition.
A full trellis (every coefficient, every level, with the cbp and neighbour
context costs carried through the whole 4x4 block) is the version that pays in
x264; it is implemented now, and *Rate-distortion level selection* below is the
measurement of it -- 1.5-2.8 points, which is more than this shortcut managed,
but not the several percent x264 gets.

## Rate-distortion level selection (`--trellis`)

The quantiser rounds into a deadzone widened on purpose (`aeg_quant_bias()`,
1.5x the rounding point) and never asks whether the bits a level costs are worth
the distortion it removes.  `--trellis 1` asks, block by block, with both terms
measured rather than modelled:

* the *price* is `aeg_cabac_residual()` run in price-only mode against the
  slice's shadow coder, i.e. the coder that decides the macroblock also decides
  the levels;
* the *distortion* is the inverse transform of the candidate levels, clipped and
  added to the prediction, SSE against the source -- not a coefficient-domain
  estimate of it.

The search is a coordinate descent over the levels of one block: start from what
the quantiser chose, consider emptying the block, then sweep the coefficients in
the order the coder sends them (last to first) trying each level one step towards
zero and one step away from it, keeping a move whenever it lowers
`sse + lambda * bits`.  A second sweep lets moves combine, which is the part a
per-coefficient rule cannot see: a level dropped here only pays for itself
together with a level raised there.  Two details make it affordable:

* a changed coefficient only changes its own significance bin, the last-coefficient
  flag (when it is the block's last), its level prefix and its sign, so the rate
  term of a candidate is `aeg_cabac_coef_price()`, O(1) instead of a walk over the
  whole list (the level-prefix context is taken as the block's first node, an
  approximation that only affects which candidates win);
* the block that comes out of the sweep is priced *exactly* and compared with the
  quantiser's own levels, so the approximation can cost choices but never make the
  result worse than not running the pass at all.

It covers the 4x4 inter blocks (both the per-block path and the batched one) and
the 8x8 inter blocks; chroma and intra keep the quantiser's levels.

| clip (BD-rate against x264 medium; - is better) | preset 2 off / on | preset 4 off / on |
|---|---|---|
| mode 4, 320x240, 8 frames | +18.6% / **+17.1%** | +17.1% / **+15.5%** |
| mode 2, 320x192, 30 frames | +24.7% / +24.6% | +23.5% / +23.3% |
| mode 3 (still), 320x192, 30 frames | -10.2% / **-13.0%** | -13.8% / **-14.9%** |
| mode 1, 320x192, 60 frames | +27.8% / +27.5% | +24.3% / **+22.3%** |

The verdict: worth 1.5 and 1.6 points on the four-quadrant clip, 2.8 and 1.1 on
the still one, and 0.1-2.0 on the two panning clips -- positive everywhere, and
the largest where the residual is largest.  That table is B-pictures-off and is
kept because it isolates the pass; the same measurement with the encoder as it
stands now, B pictures on and their vectors refined, moves the exchange rate in
its favour -- the rest of the encode got more expensive per picture, so the pass
now costs 12-14% of the time instead of 19-40% for the same absolute work:

| clip (BD-rate against x264 medium; - is better) | preset 3 off / on | preset 4 off / on |
|---|---|---|
| mode 0, 320x192, 60 frames | -24.78% / **-26.64%** | -24.38% / **-26.94%** |
| mode 1, 320x192, 30 frames | -12.83% / **-13.03%** | -12.58% / **-13.16%** |
| mode 4, 320x192, 10 frames | -8.02% / **-8.86%** | -7.52% / **-8.34%** |
| mode 3 (still), 320x192, 30 frames | -22.70% / **-22.81%** | -22.36% / **-22.52%** |
| mode 0, 320x192, 30 frames | -23.47% / **-25.50%** | -23.63% / **-25.61%** |

Every clip gains (0.15-2.2 points, 1.0-1.2 on average) for 14% of the encode
time at preset 3 (0.703 s -> 0.803 s for 60 frames of 320x192 at QP 28) and 12%
at preset 4 (0.821 -> 0.918 s), which is a better exchange rate than the
candidate refinement's 0.3-0.6 points for 40%.  It therefore **follows the
preset**: presets 3 and 4 run it, presets 0-2 do not, and `--trellis 0` turns it
off on a slow preset.  The reason it did not follow the preset before was the
cost, and the cost is what changed -- not the gain.

The earlier measurement of this pass was read in the wrong direction: it
reported the skip-heavy still clip as a 3-3.7 point *loss*, where the same six
QPs say it gains 2.8 and 1.1 (that table also carried the preset-4 column of the
still clip as the same number twice, which should have been the warning).  The
drift argument below is still real -- the block-local price cannot see what its
levels cost the next picture -- it is just smaller than the measurement noise it
was being read through.

Why the full search does not pay here, when it is worth several percent in x264:
this encoder's deadzone and lambda already sit near the RD optimum for these
decisions -- the same conclusion the "drop the last coefficient" shortcut below
reached from the other direction -- and, more fundamentally, the pass optimises
each block against a *fixed* reference picture while the levels it picks *change*
the reconstruction that later pictures predict from.  On a P-only stream that
drift is a real cost the block-local model never sees, and it is the whole of the
negative result on the still clip.  A version that pays would have to carry the
path across the whole picture's decisions -- the cbp, the neighbour contexts and
the effect on the next reference -- not just across one block.

The pieces of this pass that *are* reusable are the two pricing entry points it
needed: `aeg_cabac_residual()` in price-only mode (the whole block, exact) and
`aeg_cabac_coef_price()` (one coefficient, O(1)), both against a shadow coder.
Anything else that wants a rate term the coder would actually charge -- a
sub-pel vector refinement, a sub-partition shape, a B-picture's bi-prediction --
starts from those.

## The 8x8 transform and Intra_8x8

The High profile tools are in: the SPS signals `profile_idc = 100` with
`transform_8x8_mode_flag`, the PPS signals the 8x8 transform, and a macroblock
may use either the four 8x8 luma transforms or the sixteen 4x4 ones (chroma
keeps 4x4 throughout).  `--no-8x8dct` turns the whole thing off and codes
Baseline, which is what makes the cost measurable on this machine.

*Transform.*  `aeg_transform_8x8` is the H.264 integer 8x8 DCT; the quantiser
multiplier is x264's `quant8_scale`, and the dequantiser is the reference
decoder's `dequant8_coeff` (coefficient class `(x&1)+(y&1)`, `<< qp/6`, int16
truncation, `+32` on the DC before the first pass, rows then columns, `>>6` at
the end).  Quantise-then-dequantise reproduces each coefficient exactly at
every QP, which is what `tests/coef_check.c` checks (2.27 M coefficients).

*CAVLC.*  An 8x8 block is sent as four 16-coefficient lists: list `k` holds
`zigzag8x8[4n + k]`, i.e. every fourth entry of the 64-entry scan (the scan is
`ff_zigzag_direct`, and the decoder's `zigzag_scan8x8_cavlc` is its transpose,
so the transposition cancels out).  The lists are *not* independent: each one's
`nC` is the previous list's TotalCoeff, and the slot that survives into the
neighbour context is the sum of the four lists' TotalCoeff, written to the
group's first block when the group is finished -- exactly what
`h264_cavlc.c` does with `nnz[0] += nnz[1] + nnz[8] + nnz[9]`.  `luma8_costs()`
and `write_mb()` walk the same `group_blk[]` order with the same slot state,
because a cost pass and a write pass that disagree about `nC` produce a stream
that decodes to something else without any syntax error.

*Intra_8x8.*  All nine modes (`aeg_intra_predict_8x8`), chosen by exact
rate-distortion against the four 4x4 and four Intra_16x16 alternatives, with
`prev_intra8x8_pred_mode_flag`/`rem_intra8x8_pred_mode` and the
`min(left, top)` predictor.  The top-right samples follow 8.3.3.4.2: for a
block in the last block column they are unavailable unless the macroblock to
the right is alread decoded, and for the bottom-right block they are always
unavailable (the predictor clamps them to `p[3][-1]`).

*Deblocking.*  A macroblock coded with the 8x8 transform only has a transform
boundary at luma position 8, so only that internal edge (and the macroblock
edge) is filtered; the boundary strength compares 8x8 groups rather than 4x4
blocks (bug 36).

*What it buys.*  Measured with the same encoder at the same QP on three
320x192 synthetic clips (12 frames each), 8x8DCT off → on:

| clip | QP | bytes | PSNR (Y+U+V) |
|---|---|---|---|
| panning texture (mode 0) | 26 | 9 179 → 9 124 | 45.66 → 45.84 dB |
| half/half motion (mode 1) | 26 | 15 397 → 15 021 | 45.22 → 45.41 dB |
| four quadrants (mode 4) | 26 | 16 883 → 16 580 | 45.19 → 45.36 dB |
| panning texture | 22 | 11 411 → 11 654 | 47.96 → 48.18 dB |
| panning texture | 34 | 5 906 → 5 822 | 38.50 → 38.77 dB |

i.e. +0.2 dB at equal or fewer bits, the direction and size a synthetic
smooth-gradient clip should show (real content gains more from Intra_8x8 in
particular, which cannot be measured here).

## Quality against a reference CPU encoder

All the BD-rate numbers in this file are ours against the reference at matched
PSNR, integrated over the shared span of the six QPs 22/25/28/31/34/37 by
`tools/bdrate.py`: **negative means we need fewer bits**, so negative is better
and positive is the gap.

> **Where the numbers below were taken.**  Everything in this section except
> *The result* under *B pictures* was measured **with B pictures off**, which was
> the default when they were written.  B pictures are now on by default and are
> worth 20 to 36 points on these same clips (q4: +18.6 -> +0.7 %, p2: +24.7 ->
> -10.1 %, if you want two of them), so read the tables here as "what the P-only
> part of the encoder is worth" -- which is what they isolate -- rather than as
> the encoder's current standing.

> **A correction.**  Until this round those numbers came from a throwaway
> script that divided by the reference instead of by us and was then read with
> the sign of its own difference: the panning clip was written down as "21.7%
> more" where the same measurement says 27.8%.  The two are exact reciprocals
> (`ours/ref = 1/(1 + printed)`), so every number here is either re-measured
> with `tools/bdrate.py` or its old value passed through that identity.
> Nothing in the encoder changed because of it; the gaps were a wrong
> denominator away, which is why the largest ones moved the most, and one
> conclusion changed with them: adaptive quantisation is RD-neutral here (see
> below), not worth two points.

**With CABAC in, the reference to beat is x264's default preset, not its
baseline profile.**  Measured on the genyuv mode-4 clip (320x240, 8 frames),
our preset 2 against `libx264 -preset medium -qp N`:

| what is compared | BD-rate |
|---|---|
| ours against x264 medium with `cabac=0:8x8dct=0` (baseline profile) | **-32.1%** (32% fewer bits) |
| ours against x264 medium with `cabac=1:8x8dct=1:trellis=0:aq-mode=0` | +16.1% |
| ours against x264 medium, its defaults (B pictures, trellis, AQ, psy) | +18.6% |
| ours with `--no-cabac --no-8x8dct` against x264 medium | +118.3% |

CABAC and the 8x8 transform together are what used to be the whole gap against
x264 *baseline*: without them the encoder needs 118% more bits than x264's
medium preset, with them it needs 32% fewer than x264 restricted to the
baseline syntax.  Against the *default* preset what is left is 18.6% on this
clip, and the split is measurable: x264 with the same CABAC and 8x8 but
`trellis=0:aq-mode=0` is 16.1% ahead of us, so 2.5 points are x264's trellis
and psy-rd and the remaining ~16 are B pictures (the bframes=0 row below puts
that at 21 points on the clips with more motion).

Adaptive quantisation is RD-neutral in this encoder: `--aq 0` and the default
`--aq 10` both measure +18.6% on this clip (at a fixed QP, `--aq 10` spends 15%
more bits for a slightly higher PSNR, which is what an RD-neutral redistributor
does), and `--aq 20` -- twice x264's default strength -- is 45.5%, so the
strength that ships is the right one.

The four clips every other measurement here uses, against `libx264 -preset
medium -qp N` over QP 22/25/28/31/34/37 (negative = we need fewer bits):

| clip (`tests/genyuv W H N out MODE`) | preset 2 | preset 3 | preset 4 |
|---|---|---|---|
| 320x192, 60 frames, mode 1 (halves moving apart vertically) | +27.8% | +23.6% | +24.3% |
| 320x192, 30 frames, mode 2 (the same horizontally) | +24.7% | +23.3% | +23.5% |
| 320x192, 30 frames, mode 3 (still after the first frame) | **-10.2%** | -14.4% | -13.8% |
| 320x240, 8 frames, mode 4 (four quadrants, four directions) | +18.6% | +17.2% | +17.1% |

The spread is what the decision pass can and cannot see: on the still picture
it is ahead of x264's default preset by 10-14% (the skip rule and the coder's
price agree there), and on the clips that move it is 17-28% behind, of which
most is B pictures.  With B pictures off in the reference
(`-x264-params bframes=0`) the same four clips measure -2.4% (mode 4), -2.0%
(mode 2), -6.2% (mode 3) and +6.5% (mode 1): B pictures are worth 21-27 points
on the three clips with motion, and on two of the four the encoder is within
2.4% of x264's medium preset once the reference does not have them either.

The mode-3 clip has one property worth knowing before it is used to judge a
change: its rate is not monotone in QP.  At preset 2 it is 2.8 kB at QP 28,
2.4 kB at QP 29, 2.9 kB at QP 30 and 3.7 kB at QP 32, because at high QP the
intra picture is coarse enough that the following P pictures spend their bits
correcting it and the correction is itself quantised coarsely, so it is paid
again every frame.  A BD-rate over a non-monotone curve depends on where the
two curves cross; the still clip therefore moves by several points for changes
that improve all three other clips, which is why it is the outlier in every
table here.

What the decision pass prices a candidate *with* matters more than any of the
knobs tried before it.  Until this round the rate term was CAVLC-priced (the
estimate `aeg_cavlc_block_cost()` returns), which over-prices it by the 0.65
the CABAC coder actually spends, and a single global scale on lambda was tried
as a stand-in for the real model: it gained 0.9-0.5 points on two clips and lost
1.6 on the third (skip-heavy), so the bias is per syntax element and a single
factor cannot fix it.  The rate term is now the coder's own price -- see
*Pricing the decision pass with the coder* under *CABAC* -- and the three clips
below are the measurement of it (every clip in this file is
`tests/genyuv W H N out.yuv MODE`; the 320x240 row is the exception, generated
by an older `genyuv` whose mode 4 moves faster, so it does not reproduce from
the tree and the current mode-4 clip below is a few points further behind):

| clip | before | coder-priced | + the skip candidate | now (preset 2) | preset 3 | preset 4 |
|---|---|---|---|---|---|---|
| mode 4, 320x240, 8 frames (older `genyuv` output, see below) | +14.8% | +14.0% | +14.4% | +14.7% | +10.4% | +10.5% |
| mode 2, 320x192, 30 frames | +40.1% | +32.5% | +26.6% | **+24.7%** | +23.3% | +23.5% |
| mode 3 (still, skip-heavy), 320x192, 30 frames | +27.7% | +6.6% | -10.2% | -10.2% | -14.4% | **-13.8%** |

The last two columns are the slow presets, which also refine the winning
motion vector by rate-distortion (*Rate-distortion refinement of the motion
vector*): 1.2-1.5 points on modes 2 and 4, and 3.5-4.2 on the panning clip and
on the still one, for 30% / 45% more time than preset 2 (the slow presets also
widen the search range, 24 -> 32 -> 48, so part of that is not the refinement).  The "now" column is preset 2
with one more thing fixed: the candidate array was still sized four while the
macroblock generates five candidates, so P_8x8 was being dropped by the bound
whenever the skip vector was present (*P_8x8 sub-macroblock partitions*).  With
all five priced, the two-quadrant clip -- the one that exists to make partitions
pay -- gains 1.0-1.2 points at every preset.

The clip that gains most is the one whose decisions are dominated by skip and
by near-zero residuals, which is exactly where the CAVLC estimate is furthest
off: a skipped macroblock costs it one flag plus the estimate's guess, while the
coder charges one context-coded bin whose probability has adapted to the
neighbourhood, so the skip/inter boundary is finally in the right place.  The
last column is the second half of the story: with the price right, the *vector*
the skip rule reconstructs with is worth having as a candidate of its own (see
*The skip candidate*), and on the still clip that alone is worth 17 points of
BD-rate -- the encoder now spends 10% *fewer* bits than x264 medium there
instead of 6.6% more.

Lambda itself did not need to move once the price was right: the scale on
lambda that was tried earlier as a *stand-in* for the coder's model was
re-measured with the real one (0.7 / 0.85 / 1.0 of `0.85 * 2^((QP-12)/3)`) and
1.0 is the best of the three on all three clips -- 0.7 costs 1.3 points on one
and 11 points on the still clip.

The scale a *measurement* of the coder suggests was tried for the same reason
and is *not* in either.  The rate term of the candidate comparison is still
CAVLC-priced wherever it comes from `aeg_cavlc_block_cost()`, and CABAC spends
0.65 of that (55 271 bytes against 84 901 at QP 28 on the panning clip), so
lambda should arguably be 0.65 of the standard value.  It gains 1.5 points on
the four-quadrant clip and 0.4 on the two-quadrant one, and **loses 11.6 points
on the still clip**: a smaller lambda makes a macroblock with a near-zero
residual code that residual instead of skipping it, and on a still picture the
skip is the whole picture.  That is the same per-element bias as the earlier
experiment, now measured with the right rate on the other side of the
comparison, and it is the reason the honest fix is a per-syntax CABAC cost
model (a price per bin and per syntax element, `aeg_cabac_bin_cost[]` extended
into the candidate comparison) rather than any single factor.

The two changes together cost 15% of the encode time at this size (320x192,
30 frames, preset 2: 86 -> 73 fps), which is the price of running
`write_mb_cabac()` once per candidate instead of once per macroblock and of one
more 16x16 candidate; pruning the candidates the estimate dislikes was measured
and is *not* in (it recovered none of the time -- the bin walk is small next to
the transform and the search -- and cost 4% of the bits on the clip where the
decisions are closest).

The older comparison, against x264's baseline profile, and the history behind
it:

`x264` (master, `--profile baseline --keyint 250 --qp N`, built without asm) on
two 60-frame 320x192 clips: the panning synthetic texture (`genyuv` mode 1) and
the four-quadrants clip (mode 4).  Both streams are decoded and compared with
the source; the "rate at a given PSNR" numbers are interpolated on the
rate-PSNR curve, which is the only fair comparison (a QP index is not
comparable between encoders).

| Y-PSNR | ours | x264 baseline | x264 is |
|---|---|---|---|
| 45.03 dB (pan clip) | 46 395 B | 45 899 B | 1.1 % smaller |
| 45.56 dB (pan clip) | 50 083 B | 48 039 B | 4.1 % smaller |
| 44.89 dB (quadrants) | 50 966 B | 47 980 B | 5.9 % smaller |
| 45.25 dB (quadrants) | 53 435 B | 49 367 B | 7.6 % smaller |

So the rate penalty is now 1-8 %, against 25-27 % before the quarter-pel fix
(bug 28/29) and ~50 % before P_8x8.  For reference, x264 here uses three
reference pictures, trellis quantisation, adaptive quantisation and hand-written
SIMD; we use one reference, none of those, and C with two SSE2 kernels, and are
2.3x slower (74 fps against 170 fps on two cores at 320x192, QP 24, 60 frames).  At a matched QP x264 still
spends more bits for more PSNR: QP 26 gives 46 395 B / Y 45.03 dB against
x264's 48 039 B / Y 45.56 dB.

All-intra (`-g 1`, every picture an IDR) is at the same parity, on both clips:
at Y 42.4 dB we spend 96 150 B where x264 spends 100 480 B, and at Y 43.7 dB
109 638 B against 111 173 B.  Comparing the two encoders at a matched *QP*
suggests a ~1 dB deficit, but that is an artefact of the QP index: at QP 28 we
spend 109 496 B / Y 42.70 dB and x264 spends 127 287 B / Y 43.94 dB, i.e. it is
simply buying a better picture with more bits.  (This also explains why the
earlier "~1 dB behind" statements in this file were withdrawn: the rate-matched
comparison is the only one that means anything.)

An exact rate-distortion choice of the Intra_4x4 mode (quantise each candidate
residual, reconstruct, weigh the real SSE against the block's CAVLC bits)
replaces the SATD approximation at preset 3 and above.  It is worth ~1.5 % of
the bits on all-intra content and ~0.04 dB on inter content, for a 2x slowdown
on intra macroblocks, so it stays on the slow presets only.

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
this sandbox), best of three runs.  The clip is `genyuv 320 192 60 out.yuv 1`
(the two halves moving in opposite directions, i.e. the one where 16x8/8x16
partitions are worth coding).  The `ours` rows below predate B pictures being
the default and the level selection following the preset; re-measured on the
same clip with both (see the paragraph after the table):

| encoder | fps | bytes | PSNR |
|---|---|---|---|
| ours, preset 0 | 528 | 118 680 | 43.74 |
| ours, preset 2 | 69 | 47 309 | 46.51 |
| ours, preset 3 (four-neighbour vector refinement) | 53 | 46 871 | 46.50 |
| ours, preset 4 (eight-neighbour vector refinement) | 43 | 46 846 | 46.50 |
| ours, preset 2, `--trellis 1` | 52 | 46 556 | 46.31 |
| libx264, baseline profile (CABAC and 8x8 off), asm | 405 | 79 935 | 46.32 |
| libx264, medium, asm | 408 | 34 537 | 46.37 |

All seven rows are decoded and PSNR'd the same way (mean of the per-frame YUV
PSNR), so the numbers are comparable; absolute values move a few percent
between runs, the `-n 60`, single-thread, same-QP comparison is the one to
repeat.

Re-measured with the encoder as it stands (B pictures on by default at
`--bframes 3`, preset 3 and 4 running the level selection), same clip, same
method, best of three: preset 0 243 fps / 86 933 B / 42.02 dB, preset 2 97 fps
/ 28 645 B / 45.66 dB, preset 3 63 fps / 28 425 B / 45.72 dB, preset 4 49 fps /
28 669 B / 45.71 dB.  B pictures made the slow presets *faster* per frame in
spite of the work added to them, because a third of the pictures are now cheap
ones: preset 4 went from 43 to 49 fps while gaining the level selection.  The
cost of the pass on its own is visible in the two controls: preset 3 with
`--trellis 0` runs at 72 fps and preset 2 with `--trellis 1` at 84 fps.

The preset rows differ by more than "a wider search": the coder-priced decision
pass is the decision pass, and the vector refinement adds another 25% / 40% on
top of it at presets 3 and 4, which is why the slow presets are 1.3x and 1.6x
slower than preset 2 for 1-7 points of BD-rate.  Since the previous round the
fifth candidate (P_8x8, which an array bound was silently dropping) costs ~10%
and buys 1.0-1.2 points on the partitioned clip, and `--trellis 1` costs another
20% / 35% for nothing measurable, which is why it is off.

The honest reading against the reference: a *no-asm* x264 build is within 1.8x
of this encoder, a normal libx264 with assembly is 6x faster, and at matched
PSNR it spends 22% fewer bits than we do on this clip (18% fewer on the still
clip, 15-20% fewer on the partitioned ones) -- while
coding B pictures we do not have yet (see *Next*).

What the remaining time is, per stage (`-v 2`, 60 frames of 320x192 at QP 24,
preset 2 / preset 4, milliseconds per run): transform (the forward/inverse
transform, quantisation and reconstruction of every candidate, and the coder's
bin walk over them) 573 / 1100, motion estimation 266 / 268, inter prediction
17 / 17, deblocking 6 / 6.  The decision pass is now 68% of the encode at
preset 2 and 83% at preset 4, which is where a device would be pointed first.

Where the speed came from, each step measured on its own, all of them
byte-identical (they add, average and shift the same integers, so none of them
can move a decision):

| change | fps | bytes (40 fr, QP 26) |
|---|---|---|
| start of the round before this one | 54.5 | 46 395 |
| SSE2 SAD (13x per 16x16 call) | 65 | 46 395 |
| SSE2 4x4 SATD (2.2x per call, the subpel walk's only real cost) | 76 | 46 395 |
| `_mm_avg_epu8` for the two-plane average of a fractional block | 78 | 46 395 |
| `-O3` instead of `-O2` | 89 | 46 395 |
| SSE2 dequant + inverse transform + add (3x per block) | 90 | 46 395 |
| SSE2 quantiser (2.5x per block) | 91 | 46 395 |
| SSE2 4x4 forward transform (3.2x per call) | 92.6 | 117 974 |
| SSE2 inverse transform for the RD residual (`idct_core`, 784 k calls / 60 fr) | 97.4 | 117 974 |
| SSE2 half-pel planes (all three, 3.3x: 1.06 -> 0.32 ms per picture) | ~100 | 117 974 |
| the bit writer's inner loop inlined (7.5 M calls / 90 pictures) | 105 | 117 974 |

The two newest rows are 90-frame runs of the panning texture at QP 24 (the
older rows are 40-frame runs, hence the different byte count); both are
byte-identical to the stream before them, and both were found by profiling
rather than by guessing: a `-pg` profile put `idct_core` -- the *scalar*
inverse transform the rate-distortion pass reconstructs every candidate with,
784 k calls in 60 frames -- at the top with 15.9%, and the forward transform at
2.3%.  Halving the first and quartering the second is +9.5% of the encode
(88.9 -> 97.4 fps, medians of five runs).  Their arithmetic is the delicate
part, not their speed: both passes of the forward transform and both int16
truncation points of the inverse are part of the *reference* arithmetic (bug
8), so the vector versions reproduce them exactly and `tests/coef_check.c` now
checks the forward transform against its own transcription of the definition
(2.27 M cases), exactly as it already did for the inverse.

The earlier rows are worth less than the microbenchmarks suggest: instrumenting
the encoder with a cycle counter shows the three coefficient functions were 10%
of the encode (2.88 M calls per 150 frames, 40 + 101 + 47 cycles each), not the
33% a `-pg` profile had attributed to them, so a 2-3x faster block is ~3% of
the picture.  The machine is noisy (the same binary measures anywhere between
78 and 91 fps depending on the run), so the last rows are medians of five
alternating runs rather than a precision result.

Stage profile at QP 26, 40 frames, preset 2 (the default): motion estimation
178 ms, fractional-pel planes 43 ms, transform 214 ms, deblocking 4 ms of
444 ms (motion estimation was 433 ms of 783 ms before the SIMD kernels).
Motion estimation is 40% of the encode and the *transform* stage is the largest
single block at 48% -- but that stage is the whole of `code_mb()`, i.e. the
rate-distortion pass: forward transform, quantisation and reconstruction of
every candidate, plus the bitstream, not one 4x4 transform.  Its cost is
therefore a *call count*, and that is what the batch form fixes (see *The
coefficient stages as one batch per macroblock*): 1.9 M block calls per hundred
pictures become 32 calls per macroblock, launched as two kernels.

What the profile says is left on the CPU, and why it stays there:

| function | share | note |
|---|---|---|
| `aeg_mc_luma_p` | 14% | already `pavgb`, 16 samples an instruction; it is bandwidth, not instructions |
| `aeg_quant_4x4` | 11% | scalar int64 divisions; SSE2 path exists (2.5x) but the call count dominates |
| `code_luma_inter` | 11% | residual gather + SSE of sixteen blocks; the batch path replaces the middle of it |
| `code_luma_i16` | 11% | Intra_16x16 RD: serial by construction (the DC Hadamard) |
| `aeg_sad` | 8.6% | already SSE2; the search evaluates a fixed candidate grid |
| `aeg_idct_add_block` | 8.6% | SSE2, 3x the scalar loop |
| `aeg_hpel_plane_fill` | 8.6% | the interpolation pass; a kernel (`interp_plane`) when the stage is on a device |
| `aeg_transform_4x4` | 5.7% | scalar; the batch path moves it |
| `write_block` | 5.7% | CAVLC, serial by definition |

Trimming the RD search itself (fewer intra modes, SATD-first inter partitions)
was measured at +10-19% speed for -0.1 to -0.2 dB, and on this machine the
speed part is inside the noise; it was not taken.  Multi-threading is the
remaining structural option and it is not implemented.

## Fractional-pel planes (the interpolation cache)

A quarter-pel motion search interpolates enormous numbers of samples: every
candidate position re-runs the 6-tap filter chain for its whole block, so a
macroblock at preset 2 interpolates ~8000 samples per subpel pass, and the
picture repeats that 240 times.  The samples themselves are *not* a property of
the candidate: the four luma planes a fractional position is built from --
integer, horizontal, vertical and both (H.264 8.4.2.2.1's half-pel memory
optimisation) -- are a property of the *reference picture*.

`aeg_mc_ctx` (`src/mc.c`) therefore holds the three fractional planes, stored
with the same stride and the same origin as the integer picture, so a pointer
into the integer picture becomes a pointer into plane *k* by offsetting from
`base[0]`.  `aeg_mc_ctx_refresh()` fills them once per P picture (timed as
`AEG_STAGE_INTER_PRED`), and `aeg_mc_luma_p()` then computes a fractional block
as two array reads and an average per sample.  The samples are bit-identical to
what `aeg_mc_luma()` filters per block; the only change is how often the filter
runs (per picture instead of per candidate, ~30x fewer times at preset 2).

That one change is the largest speed win since the encoder became conformant,
and it also cleans up the GPU story: building the planes is one data-parallel
pass over the picture with no dependence between output samples, i.e. exactly
the shape of kernel `sad_batch`/`tq_batch` already have, whereas the old
per-block filtering was a scatter of tiny launches.

It is therefore also the fifth backend operation: `aeg_be_ops.interp_plane`
(`interp_plane` in `src/kernels.h`, one work item per output sample) fills a
plane on the device, and the encoder uses it only when the dispatcher has
assigned the `inter` stage to the GPU (`-s inter=cuda` / `-s inter=opencl`
forces it).  The CPU implementation of the operation *is* the code the encoder
runs when the stage stays on the CPU -- `aeg_hpel_plane_fill()` in `src/mc.c` --
so a stage that moves between the two cannot drift: `tests/kernel_check.c`
compiles the kernel source as C and compares the three planes sample by sample
against it (6 kernels, all bit-exact).

## Whole-frame parallel search

The motion search is the one stage that is completely data parallel across
macroblock boundaries, so `me_seed_frame()` runs it as rounds: round 0 evaluates
zero and the previous picture's vector at every macroblock, then four rounds
evaluate the eight neighbours of every macroblock's current best vector at step
8, 4, 2, 1 -- each round is *one* `sad_batch()` call over the whole picture, i.e.
five launches per picture instead of ~10 000 dependent 16x16 SADs.  The vectors
that come out are used as an extra candidate start by the per-macroblock search
(which keeps its serial hex walk, so the decision logic does not move).

Measured: with `--subme 0`, the rounds shorten the hex walks enough to be 8%
faster with byte-identical output; at `--subme 1` they are neutral in quality and
~3% slower on this CPU-only machine, so they follow the dispatcher's vote for
`AEG_STAGE_MOTION_EST` (a GPU that wins that stage gets them) and `AEG_SEED=1/0`
forces either way.  `tests/run_tests.sh` encodes one case with them forced on.

The fractional half of the search runs as rounds too (`me_par_subpel()`): the
integer winners are refined at step 2 and then step 1, each step evaluating the
eight neighbours of *every* macroblock through `satd_batch()` until a round
stops improving (at most `AEG_SUBPEL_ITERS` rounds, three).  The per-macroblock
walk is then replaced outright -- `analyse_mb()` takes `par_mv`/`par_cost`
instead of calling `aeg_me_block()` -- so the same two decisions (vector, SATD)
come out of a fixed number of launches instead of ~240 dependent walks.
`satd_batch()` carries the picture geometry (`pstride`, `pad`) instead of a
pointer so that the device side can index its own reference copy; the encoder
registers that copy once per picture through `ref_upload()`.

That variant follows the same dispatcher gate (`AEG_PARFULL=1/0` forces it, and
the integer rounds have to be on for it to have a baseline).  On this CPU it is
a *loss*: three rounds per step is 54 whole-picture SATD passes against the
walk's ~10-20 dependent evaluations per macroblock, so the 320x192 QP 26 clip
goes 51.7 -> 43.9 fps for 46 395 -> 46 634 bytes (Y 44.15 -> 44.09 dB) -- a wash
in quality for 15% more time.  It exists because on a device those 54 passes are
54 launches over 240 macroblocks each, which is what a GPU is for;
`tests/run_tests.sh` keeps the path conformant (`par_subpel` case) even though
this machine always chooses the walk.

### The trellis price of the last scan position

AddressSanitizer (run while chasing the frame-thread queue bug) found the
trellis' inline coefficient price reading `aeg_cabac_sig8[63]` and
`aeg_cabac_last8[63]` -- one past both tables -- whenever the coefficient at the
last scan position of an 8x8 block was priced; the 4x4 blocks read a
neighbouring category's context for their last position the same way.  That
position carries neither a significance nor a last flag (9.3.2.2), so its
price now has neither.  The old value came from whatever the linker put after
the tables, i.e. the stream could change with an unrelated edit.  BD-rate
against the previous binary (five clips, global PSNR): preset 2 0.00 %, preset 3
+0.04 %, preset 4 -0.03 % -- noise; the streams of presets 3-4 change, presets
0-2 do not.

## GPU groundwork on a real OpenCL device

Every device path used to be checked in two ways only: the kernel *source*
compiled as C (`tests/kernel_check.c`), and the batched paths run with the
*CPU* implementations behind the batch entry points.  Neither runs the host
side of a device backend -- what is transferred, from where, in which layout,
what comes back and where it is written -- and neither runs two threads into
one device context.  PoCL (a complete OpenCL implementation whose device is the
host CPU) runs all of it here; `tools/pocl_env.sh` installs and sets it up
without root.  The first run found five bugs, all now fixed; every one of them
would have shown up on the first GPU as a crash or as a stream different from
the CPU's.

| # | bug | effect | fix |
|---|---|---|---|
| 1 | the RD batch's decide step had drifted from the serial candidate loop: no level selection on the batched levels, candidates ranked by the CAVLC estimate instead of the CABAC price with chroma, no candidate refinement, a different `jinter` | `AEG_RD_BATCH=1` (and the dispatcher's choice whenever a device owned the coefficient stages) wrote a *different* stream, 4-10% larger | the decide step is now the serial loop step for step, with only the luma coefficients taken from the batch: level selection on every candidate's batched levels between the two launches, then one inverse launch on the selected levels; each candidate's chroma prediction saved alongside its luma one (the batch had left the *last* candidate's in `pred_c`) |
| 2 | one device context shared by slice threads with no locking (staging buffers, kernel arguments, a `static` scratch) | segfault on the first picture with `-t 2` and any device path | a guard around every device backend (`backend.c`) serialises the calls |
| 3 | the registered reference (`ref_upload` then `satd_batch`) is device state that outlives a call; the CPU version kept it in a process-wide static | with two pictures in flight, one picture's search could read the other's reference -- a stream that depends on thread timing (CPU: `AEG_PARFULL=1` with frame threads) | the guard remembers each thread's registration and re-uploads it when another thread replaced it; the CPU registration is thread-local |
| 4 | `sad_batch` on both devices transferred *one packed 16x16 block* and 48 reference rows, from an older one-block API; the encoder sends a whole picture's candidates with reference positions in the padding | out-of-bounds device reads: the whole-frame search was garbage and non-deterministic | the host transfers exactly the span the candidates read (`aeg_sad_span_rebase()`), positions rebased to it |
| 5 | `interp_plane` transferred from the picture *origin* while its kernel reads and writes the padding around it, and read the result back over the end of the host buffer | device out-of-bounds reads, host heap corruption | the kernel takes the whole padded allocations (the convention `satd_batch` already used) |

Also: a device call that fails now runs the CPU reference instead and keeps
the device off (the encoder ignores the batch operations' return codes, which
used to mean a failed call silently left garbage); the CUDA context is made
current on the calling thread in every operation (a driver-API context is
current only on its creating thread, so slice threads would have failed every
call); the slice loop no longer drops to one thread when a stage is on a
device; `-v 2` reports a stage as `gpu` when any of its work ran there
(`stage_gpu_ns` / `stage_cpu_ns` are now the per-stage totals of device and host
batch time, summed over the thread copies).

**Tests added.**  `tests/backend_check.c` (106 checks, every operation and two
threads, against the CPU reference; it is the one that fails on bugs 2-5 --
the old backend corrupts its heap within the first checks) and
`tests/nvrtc_check.c` (the CUDA kernels through NVRTC for every architecture it
supports; a kernel line that is valid C but not C++ fails it).  Both are in
`make test` and skip without a device / NVRTC.  `tools/gpu_validate.sh` was
itself broken (the forced run passed its `-s` options as environment
variables, the decimal stage times did not match its pattern, and the
reconstruction check compared the coding-order dump with the display-order
decode, so with B pictures every run "failed"); fixed, and it now also runs the
whole-frame search on the CPU backend and requires the device's to be the
same bytes, and forces `-s inter=` so the plane kernel is in an encode too.
`tests/run_tests.sh` takes `BUILD=dir` and `AEG_TEST_BACKEND=opencl`.

**State on the device (PoCL), all byte-identical to `-b cpu`:**

* `tools/gpu_validate.sh` PASS on vtest and cockatoo, `-t 1` and `-t 2`: cpu,
  auto, coefbatch, rdbatch, `-s tq= -s iq= -s inter=` the same stream; search
  and all the same stream as the same search on the CPU backend; every
  reconstruction equal to ffmpeg's decode.
* `tests/run_tests.sh` with `BUILD=<OpenCL build> AEG_TEST_BACKEND=opencl`: all
  conformance cases bit-exact, `seed_search`, `coef_batch(4)`, `rd_batch(4)`
  through the device.
* `-s inter=opencl` on vtest and the weighted fade clip, presets 1 and 3.
* With the CPU backend nothing changed: `tools/real/same.sh` against the
  previous commit 50/50 (presets 0-4, five clips, QP 24/32) plus 10/10 CAVLC,
  frame threads 32/32, `make test` bit-exact.

**Placement by trial.**  The dispatcher could not place the coefficient stages
on its own: it only measured the path it had chosen, and the CPU form of the
stages is not a batch with a per-call time at all -- sixteen SSE2 transforms
inline in each candidate.  It now compares whole pictures instead
(`aeg_dispatch_coef_pick` / `_done`): both placements code the same stream, so
everything else a picture does is the same work, and the difference in time per
macroblock is the placement's cost, host-side overhead of the batch included.
The first pictures of each kind (P, B) alternate; the first picture on each side
is not counted (kernel compilation at the first launch, buffer allocation);
the faster side stays after three counted pictures each, or after one each when
the two are a factor of two apart; the slower side is re-tried every 64
pictures.  The motion search is never placed by timing -- its device form is a
different search, so the stream would depend on the machine's speed -- only
`-s est=` (or `AEG_SEED` / `AEG_PARFULL`) puts it on a device.  On PoCL the
trial comes out at 0.15-0.45 ms per macroblock on the device against 0.02-0.06
on the CPU (the launch overhead of a CPU OpenCL runtime, ~1 ms per call, two
calls per macroblock), settles on the CPU after two device pictures per kind,
and the stream is the CPU's byte for byte.  On PoCL that is the right answer;
whether a GPU's launch cost ever makes it come out the other way is the one
thing still to be measured on hardware.  (The batch's host side is now as
cheap as the serial loop, and one launch per macroblock at presets 3-4; see
*A lighter host side for the batch*.)

**CUDA without a GPU.**  `pip install nvidia-cuda-nvrtc-cu12` gives NVRTC
12.9, which compiles the kernels for every architecture it supports.  The
backend now also falls back to the newest architecture NVRTC supports when the
device's own is newer than the NVRTC at hand (the driver JIT-compiles PTX
forward), instead of falling back to the CPU.  What remains unexercised is the
driver half (`libcuda`: context, transfers, launches), which needs an NVIDIA
driver.

**Loader overrides.**  `AEG_OPENCL_LIB`, `AEG_CUDA_LIB` and `AEG_NVRTC_LIB`
name the library to load before the usual sonames are tried.

```sh
. tools/pocl_env.sh                 # once per shell; installs into ~/.cache/aeg-pocl
make OPENCL=1 && ./build/backend_check
BE=opencl FFMPEG=ffmpeg tools/gpu_validate.sh clip.yuv W H FRAMES QP
BUILD=./build AEG_TEST_BACKEND=opencl FFMPEG=ffmpeg sh tests/run_tests.sh
AEG_NVRTC_LIB=.../nvidia/cuda_nvrtc/lib/libnvrtc.so.12 ./build/nvrtc_check   # CUDA=1 build
```

## A lighter host side for the batch

(2026-11-01.)  What a device has to beat in the placement trial is not only
its kernels but everything the batched path does on the host around them.
Profiling the batch with the CPU reference ops behind it (`AEG_RD_BATCH=1
-b cpu`, cock x3, 90 pictures, preset 3) showed three things that were the
batch's own and not the serial loop's:

* **The reconstruction from batched residuals** was a scalar add, clip and
  store per sample followed by a separate `aeg_ssd()` pass, where the serial
  path uses the SSE2 inverse-and-add.  It is now `aeg_add_res16_ssd()`: one
  SSE2 pass over the sixteen 4x4 residual blocks in batch layout, saturating
  add + unsigned pack (which clips exactly as the scalar form does), the SSE
  accumulated in the same pass.  `coef_check` compares it with its scalar
  reference, including residuals at both int16 extremes.
* **The 8x8 transform decision** (`use_t8()`: an SA8D and sixteen SATDs over
  the macroblock) was made twice per candidate, and the candidates that chose
  the 8x8 transform still went through both launches and the level-selection
  prep before being re-coded serially.  It is now made once, when the
  candidate is predicted, and such candidates stay out of the batch: fewer
  blocks transferred, gathered and launched.
* **The inverse launch under level selection.**  With trellis on (presets 3
  and 4) the levels the inverse needs are the ones the host has just selected,
  so the device inverse is an upload, a launch, a synchronisation and a
  download, for work the host does in one SIMD pass per block (the serial
  path's own `aeg_idct_add_block()`, so the bytes are the same).  The host now
  reconstructs there (`batch_host_inverse()`), and the batch is one launch per
  macroblock instead of two.  An explicitly placed inverse (`-s iq=...`) still
  runs where it was put, so `gpu_validate`'s `stageforce` still exercises the
  device inverse.

The CPU reference op behind `iq_idct_batch` also takes the shortcut the serial
path has for empty and DC-only blocks (most blocks of a batch).  Its DC value
is `((int16)(dc + 32)) >> 6`, the residual's own arithmetic, which wraps at the
int16 top end where the add path's shortcut does not; `coef_check` now feeds it
DC-only blocks across the whole int16 range, and fails on the unwrapped form.

Measured (minimum of three runs; everything byte-identical to the serial
encoder):

| | before | after |
|---|---|---|
| CPU ops, cock x3 p3 `-t 1`, serial loop ~6.9 s | 7.53 s | 6.98 s |
| CPU ops, vtest p3 30 pictures (serial 819-827 ms) | 920 ms | 836 ms |
| CPU ops, `same.sh` p2-4 five clips QP 24/32 | 32.80 s | 30.19 s |
| PoCL, vtest 12 pictures, p2 `-t 1` | 1858 ms | 1481 ms |
| PoCL, p3 | 3192 ms | 1854 ms |
| PoCL, p4 | 3490 ms | 2253 ms |
| PoCL placement trial, P pictures (device / CPU, us per MB) | 1147 / 80 | 770 / 78 |
| PoCL placement trial, B pictures | 496 / 49 | 301 / 48 |

The trial still keeps the stages on the CPU on PoCL, and should: what is left
of the device side is about one PoCL launch per macroblock.  On a GPU the same
trial now compares the CPU against one round trip per macroblock, plus a host
side that costs what the serial loop costs.

Verification: `make test` bit-exact; the conformance suite through PoCL
bit-exact; `backend_check` 106 checks; `gpu_validate.sh` PASS at `-p 3 -t 2`,
`-p 4 -t 1` and `-p 2 -t 2`; `same.sh` 25/25 against the previous commit with
the default path (`-t 2`) and 30/30 with `AEG_RD_BATCH=1`; ASan + UBSan clean
on CPU, coefficient-batch, RD-batch and forced-stage encodes at presets 2-4,
all writing the same stream.

## GPU groundwork without hardware, continued

**Fused forward + inverse (`tq_iq_batch`).**  Without level selection
(presets 0-2, and CAVLC at any preset) nothing runs on the host between the
two halves of the coefficient batch, yet they were two round trips: upload
the residuals, launch, download the levels; upload the same levels, launch,
download the residual.  A backend may now provide `tq_iq_batch()`: one upload,
one launch, three downloads (levels, counts, residual).  The 4x4 arithmetic
moved into two helper functions of the kernel source (`aeg_tq4`, `aeg_iq4`,
on private arrays; `AEG_DEVFN` is `__device__` under CUDA and `static` in the
C build of `kernel_check`), which `tq_batch`, `iq_idct_batch` and the new
`tq_iq_batch` all call, so the fused kernel is the two kernels' code.  The CPU
op is literally `tq_batch` then `iq_idct_batch`.  Both batch paths
(`code_luma_inter`, the RD batch) use it when the backend has it and no level
selection runs (`use_tq_iq()`); `AEG_TQIQ=0` makes the two calls, which keeps
the separate kernels in an encode.  The time is booked half to each stage.

Checks: `kernel_check` runs the fused kernel (as C) at every QP 0-51, intra
and inter, full-range and sparse residuals, against the CPU two-step
sequence, and the CPU fused op against the same (7 kernels now);
`backend_check` compares the device op with the CPU sequence for every QP /
batch size / intra case of the coefficient test (186 checks on PoCL);
`nvrtc_check` requires its PTX entry point for every architecture.
`gpu_validate.sh` PASS on PoCL at `-p 2 -t 2`, `-p 1 -t 1`, `-p 3 -t 2`; the
CPU batch modes (`AEG_COEF_BATCH=1`, `AEG_RD_BATCH=1`) equal the serial
encoder at presets 0-2 on cock and mega and with CAVLC at preset 3; the
conformance suite bit-exact.  PoCL, vtest 12 pictures, preset 2, `-t 1`
(best of three, both streams identical and equal to the CPU build's):

| | two calls (`AEG_TQIQ=0`) | fused |
|---|---|---|
| `AEG_RD_BATCH=1` | 1358 ms | 1162 ms |
| `AEG_COEF_BATCH=1` | 5320 ms | 4543 ms |

**Device lanes: a device instance per thread.**  Frame threads were off with
a device backend and with the forced batch paths, and slice threads queued on
one lock: every thread called the one backend instance through the guard,
which had to serialise them (a backend keeps its staging buffers and kernel
arguments in the instance, and `clSetKernelArg` is not thread-safe).  The
guard now has up to 16 *lanes*, each its own instance behind its own lock:

* A backend may provide `clone()`: another instance on the same device that
  shares what is expensive and read-only and owns everything a call writes.
  OpenCL: the context and the built program are retained (`clRetainContext`,
  `clRetainProgram`), the command queue, the seven kernel objects and the
  buffers are new.  CUDA: the context, the module and its functions are
  shared (a context may be current on several threads; launches take their
  arguments per call), the buffers, the registered reference and the
  `sad_batch` scratch are new; the clone uses the legacy default stream like
  the first instance, so the device orders their work -- correct, and the
  host-side lock is still gone.  A clone does not unload the library, the
  module or the context; the guard destroys the clones first.
* `aeg_be_lane(ops, k, &out)` gives lane k (k mod 16), creating its clone on
  first use under the guard's creation lock.  Lane 0 is the backend itself.
  Without `clone()`, or if it fails, the lane is lane 0 (never one instance
  under two locks); the CPU backend has nothing to split and is returned as
  it is.  The registered-reference bookkeeping (each thread's registration,
  re-uploaded when another thread on the lane replaced it) is per lane; a
  failing call still switches the device off for every lane.
* The encoder gives each thread that codes at the same time as others a lane
  of its own: frame-thread worker k uses lane 1 + k * (slice threads), the
  slice threads of a picture add their index; the encoder itself is lane 0.
  `AEG_LANES=0` puts every thread on lane 0 (the measurement below, and a way
  out should a driver mishandle several queues; PoCL 4.x-6.x does, and gets
  that automatically -- see below).
* The copies of the encoder the threads run on carry a copy of the
  dispatcher.  Their measurements (stage timings, the coefficient-placement
  trials) used to be lost with the copy -- a frame thread's picture time never
  reached the coefficient trial, a slice thread's stage timings were summed
  back by hand.  `aeg_dispatcher.root` now points the copies at the
  encoder's dispatcher, whose mutex every copy shares; the decisions stay
  with the thread that owns it, and a copy's `aeg_dispatch_update` is a
  no-op.  ThreadSanitizer found the one unlocked reader this left (the
  struct copy that launches a frame thread now holds the lock) and is clean
  with frame threads in the CPU, coefficient-batch and RD-batch modes, and
  with slice threads.
* Frame threads are now allowed with any backend and with `AEG_COEF_BATCH=1`
  / `AEG_RD_BATCH=1`.  Two exclusions remain: `AEG_SEED` (as before) and the
  motion stage forced onto a device (`-s est=...`): the whole-frame search
  rounds read the whole reference at once, which the row wavefront does not
  provide.  For the same reason a worker does not filter the fractional
  planes on the device (`interp_plane` reads the whole reference); it uses
  the planes built as the reference was coded, the same samples
  (`kernel_check` holds the kernel to them).

Checks: `backend_check` (192 on PoCL) adds the two-thread tests on two lanes,
both threads running every operation at once against their own registered
references, and that the lanes are separate instances (so a clone that
silently fell back would fail); `gpu_validate.sh` PASS on PoCL at `-p 2
--frame-threads 3`; on the OpenCL backend, presets 2-4, the default,
coefficient-batch, RD-batch and forced-stage (`-s tq/iq/inter=opencl`) paths
with 1 and 3 frame threads all write the CPU serial stream (24/24); the CPU
batch modes with 2 and 4 frame threads equal the serial encode; ASan + UBSan
clean on `backend_check` and on the batch paths with slice threads and frame
threads (PoCL); no leak of ours under LeakSanitizer; `nvrtc_check`; the
conformance suite bit-exact.  The CUDA clone compiles but, like the rest of
the CUDA backend, has not met a GPU.

PoCL on the two-core sandbox, vtest 384x288, 24 pictures, preset 3, QP 28,
best of three, every stream identical:

| | serial | one lane (`AEG_LANES=0`) | lanes |
|---|---|---|---|
| `AEG_COEF_BATCH=1`, `--frame-threads 3` | 10116 ms (1 thread, one run) | 6282 ms | 3555 ms |
| `AEG_RD_BATCH=1`, `--frame-threads 3` | 4160 ms (1 thread, one run) | 2325 ms | 1764 ms |
| `AEG_COEF_BATCH=1`, `-t 2` (slices) | | 7248 ms | 4845 ms |
| `AEG_RD_BATCH=1`, `-t 2` (slices) | | 2832 ms | 1947 ms |

On PoCL the "device" is the same two cores and every launch costs about a
millisecond of latency, so this mostly measures launches overlapping instead
of queueing; what a GPU gains depends on its queue count and launch cost.

*A driver that mishandles several queues: PoCL 4.x-6.x.*  CI went red with
the lanes, in its one job that runs the conformance suite through the device
(Ubuntu 24.04's PoCL 5.0), and only in its two slice-thread cases: the
encoder aborted inside PoCL, `pocl_release_dlhandle_cache: Assertion
'found->ref_count > 0' failed`.  Two queues launching the same kernel at the
same moment break PoCL's cache of compiled work-group functions; PoCL fixed
it in c9ac4c0c3 ("fix dlhandle cache": a multithreaded program running the
same kernel with different sizes aborts) and 69d0ae9ac (its locking), both
first released in 7.0.  The sandbox's PoCL 3.0 does not have the cache bug,
which is why nothing here failed.  The OpenCL backend now reads the device's
version: on PoCL 4.x-6.x `clone()` declines with the reason as the last
error, every lane is lane 0, and the device calls are serialised as before
the lanes -- the same stream, without the overlap.  `AEG_POCL_LANES=1`
clones on any PoCL (to try a fixed build), `=0` declines on any PoCL (the
fallback, testable here: the conformance suite through the device is
bit-exact both ways, and `backend_check` reports the shared lane instead of
failing "separate instances" -- any other collapse still fails).  The suite
now prints a failed encode's exit status and log tail, which is what found
this: the job's log itself is not reachable from the sandbox.

**Intra_16x16 and chroma in the batch (`tq_dc_batch`, `iq_dc_batch`).**  The
coefficient batch covered the sixteen luma blocks of inter candidates; the
chroma of every macroblock and the luma of Intra_16x16 candidates stayed per
block, because their block DCs leave the 4x4 transform for a Hadamard of
their own (2x2 for chroma, 4x4 for Intra_16x16), and the inverse takes the
dequantised Hadamard output as each block's DC.  A pair of operations splits
the DC off:

* `tq_dc_batch()`: the forward transform and quantisation of every block,
  with the block's transform DC handed out (on the device the sum of its
  residual, the same number) and its DC level forced to zero -- what the
  per-block code did with `coef[0] = 0`;
* the Hadamard, its quantisation and dequantisation stay on the host: four or
  sixteen values, sequential, nothing to launch;
* `iq_dc_batch()`: the inverse of every block with its dequantised DC given
  (the kernel helper `aeg_iq4` takes the DC override that
  `aeg_idct_add_block()`'s `dc` argument is, int16 truncation included).

So chroma is two calls per macroblock for both planes (eight blocks) instead
of sixteen block calls, and an Intra_16x16 candidate two calls instead of 32.
They run wherever the coefficient batch does (the dispatcher's trial, or
`AEG_COEF_BATCH` / `AEG_RD_BATCH`), and not where level selection runs between
the halves (trellis with CABAC: the chroma and Intra_16x16 AC selection is
sequential per block); `AEG_DCBATCH=0` keeps the per-block path.  The
dispatcher's coefficient trial times whole pictures, so these launches are in
the comparison it makes.

Checks: `kernel_check` runs both kernels (as C) at every QP, intra and inter,
full-range and sparse residuals, with given DCs past the int16 range, against
the CPU ops, and the CPU op's residual plus prediction against
`aeg_idct_add_block()` with the same DC (9 kernels); `backend_check` the
device ops against the CPU ones (312 checks on PoCL); `nvrtc_check` requires
the two entry points.  The CPU batch modes equal the serial encoder on vtest,
cock and mega at presets 0-2, all-intra (`-g 1`), CAVLC at presets 1, 3 and 4,
and preset 2 without trellis (48/48); on PoCL an all-intra encode writes the
same stream with and without the pair and as the CPU build;
`gpu_validate.sh` PASS at `-p 1` and `-p 2` with frame threads and on cock
`-p 2 -g 4 -t 2`; ASan + UBSan clean on the batch paths with frame threads
on both backends; the conformance suite bit-exact.  On PoCL the pair makes an
all-intra encode slower (1.46 s -> 4.49 s for 6 pictures: three times the
launches at a millisecond each), as every batch path is there; whether a GPU
wins is the dispatcher's trial to find out.

## Practical encoder features (no GPU needed)

With no GPU to tune for, everything an encoder needs that does not depend on
one: input formats, stream signalling, the options x264 users expect, and rate
control.  Each commit leaves CQP streams (and, from CRF on, CRF streams)
byte-identical unless it says otherwise; every new path has a conformance case
in `tests/run_tests.sh` (decoded by FFmpeg, compared with `--dump-recon`), and
both the CPU and the OpenCL (PoCL) suites pass.

* **Any even frame size** (`78daf21`): the coded size is rounded up to whole
  macroblocks, the margin repeats the last column/row and the SPS crops it;
  PSNR is measured on the visible picture.  Before, sizes were rounded down and
  the SPS signalled a negative crop.
* **Input and VUI** (`fb78a5c`): YUV4MPEG2 (file or pipe; W, H, F, A,
  XCOLORRANGE), fractional rates, SAR (Table E-1 or Extended_SAR), full range,
  H.273 colour description; a missing, malformed or out-of-range option value
  exits with status 2 instead of crashing or clamping silently.
* **NAL HRD and SEI** (`810aef3`): `--nal-hrd` writes hrd_parameters from the
  VBV model, a buffering period on every IDR and picture timing on every
  picture; `tools/hrd_check.py` checks that the signalled output times put the
  pictures in display order (B groups, several IDRs, and capped CRF).  The
  level now also covers the VBV max rate (a 400 kb/s VBV stream at QCIF had
  claimed level 1.2, MaxBR 384).  `--info-sei` writes the options.
* **Options** (`516a374`): `--deblock a:b`, `--chroma-qp-offset`,
  `--qpmin/--qpmax` (`--qpmin 40 --qpmax 30` is refused), `--aud`, `--level`,
  `--seek`.  Two deblocking bugs surfaced, both invisible at zero offsets: the
  slice header always carried offset 0 while the filter used the parameter,
  without the `<< 1` of FilterOffsetA/B; and an edge with luma alpha/beta 0
  skipped its chroma even when a chroma QP offset raised the chroma QP.
* **CRF and VBV** (`b9ab68c`): x264's CRF, calibrated against x264
  `--preset medium --tune psnr` at the same CRF (preset 3, four clips, CRF
  16-38: -0.03 dB at 0.99x the bits with the tree, +0.02 dB at 1.06x
  without).  VBV with x264's per-type size predictor and re-coding of a picture
  that would still underflow: 0 underflows over 36 configurations (3 clips, 3
  buffers, ABR/CRF, with and without B pictures) where 13 underflowed before.
* **Rate control** (`5b9663f`): below.

### One-pass ABR and two-pass

The previous ABR controller corrected each picture's QP from the running bit
debt.  On held pictures (a still repeated for several frames) it collapsed the
QP and overspent -- 744 kb/s for a 300 kb/s target -- and on 30-frame clips it
missed by up to +383 % (tree).  It was replaced by x264's `ratecontrol.c`
model:

* one pass: qscale = complexity^(1-qcomp) * (sum of bits * qscale) / (wanted
  bits), an overflow correction against the total, an asymmetric step limit
  (x264's lstep) per anchor, intra pictures from the following anchors, B
  pictures from the anchor before them;
* the macroblock tree on by default for ABR, as in x264 (VBV keeps it off);
* the first intra picture is sized from its lowres estimate at 12 pictures'
  worth of bits and is left out of the rate factor (counted in, its term is
  several anchors' worth against one picture's budget and drags the first
  seconds coarse: tree clip -0.75 dB counted in, -0.42 dB left out).  Later
  intra pictures count: left out, a clip of held pictures with a scene cut
  every 25 frames overspent 53 % (now 33 % -- each cut is ~15 pictures' worth
  arriving unannounced, with B pictures nearly free);
* far off target (overflow above 1.5 or below 0.67) the step limit is lifted
  in the direction the total needs.  On real clips this never fires (output
  identical); it shortens the climb out of a start the estimates got wrong.

Two-pass keeps B pictures.  Pass 1 writes `aeg-stats 2` (display index, type,
average QP, the picture's QP before macroblock offsets, bits);
`aeg_encoder_set_pass1()` plans every picture as x264's `init_pass2` does:
complexities blurred with a Gaussian that stops at scene cuts, held pictures
kept at their reference's level, intra pictures from the following P pictures,
the rate factor by bisection.  Pass 2 then corrects against the plan.

Measured: 180 frames of five real clips, preset 3, each against the CRF curve
of the same encoder (CRF + tree, six CRFs, interpolated in log rate); rate
error, then PSNR difference at that rate (`leval.sh`):

| clip, target | 1-pass ABR | 2-pass | 1-pass, no B | 2-pass, no B |
|---|---|---|---|---|
| vtest 250 | -4.4 % -0.48 dB | -0.8 % -0.10 | -2.4 % -3.18 | -0.4 % -2.10 |
| cock 500 | +5.8 % -0.25 | +0.2 % -0.00 | +3.0 % -0.30 | -0.2 % -0.16 |
| tree 400 | +4.8 % -0.44 | +0.2 % -0.15 | -13.7 % -1.54 | +0.0 % -0.53 |
| short 250 | +3.6 % -0.12 | +0.0 % -0.01 | +1.2 % -0.51 | +0.0 % -0.49 |
| mega 300 | -1.7 % -0.03 | +0.0 % +0.01 | -0.7 % -0.89 | -0.7 % -0.86 |

(The no-B columns are against the CRF curve *with* B pictures, so they include
what B pictures and the tree are worth.)  Two-pass is within 1 % of the target
and 0.15 dB of constant quality.  Without the tree, two-pass had undervalued
held repeats by up to 1.38 dB (vtest).

What decided the first intra picture's share, one pass with the tree (rate
error, PSNR difference; vtest | cock | tree | short | mega):

| share | vtest | cock | tree | short | mega |
|---|---|---|---|---|---|
| 10 pictures | -6.8 % -0.81 | +8.2 % -0.16 | +2.5 % -0.42 | +2.4 % -0.08 | -2.3 % -0.02 |
| 12 (chosen) | -4.4 % -0.48 | +9.6 % -0.22 | +4.8 % -0.44 | +3.6 % -0.12 | -2.3 % -0.02 |
| 14 | -2.0 % -0.26 | +11.0 % -0.29 | +6.7 % -0.55 | +4.4 % -0.17 | -2.3 % -0.02 |
| 20 | +3.2 % +0.19 | +14.4 % -0.48 | +12.0 % -0.76 | +7.6 % -0.32 | -2.3 % -0.02 |

vtest (a static camera: the first picture is referenced for six seconds)
wants more; the others want less.  12 has the smallest summed loss with every
rate within 10 %.  (These rows are before later intra pictures counted, which
moved cock to +5.8 %; the first picture is the only intra one in the others.)

Tried and rejected: subtracting the tree's mean offset from the first intra
estimate (vtest -1.67 dB); spreading the first picture's excess over the
following seconds, or decaying the rate factor's history (0.99, 0.97 per
picture) -- both only moved the rate, the quality within 0.05 dB; sizing the
first P picture from the coefficient the intra picture measured, and coding
it again when it came out more than twice its share (the lowres estimate
averages noise away, so the coefficient does not carry over: tree clip
+24 % -1.50 dB, 30-frame clips up to +197 %).

Limits: a stream of a second or two cannot absorb the first intra picture --
30-frame clips land between -12 % and +71 % with the tree.  genyuv's whole-frame pan
(mode 0) is predicted so well that its pictures barely change size with the QP
(20 kbit at QP 13 and at QP 9), which a model-based controller cannot land a
target on (it codes 274 kb/s for 400); the suite uses mode 1, 120 frames
(309 kb/s for 300).  P-only, CRF can beat two-pass across a scene cut (cut
clip: CRF 26 122.7 kb/s 44.17 dB, two-pass 121.0 kb/s 43.20 dB): CRF's blur
carries the easy half over the cut and gives the new IDR and the pictures
after it a lower QP, which pays through prediction -- the effect the tree
models when there are B pictures (with them two-pass is +0.10 dB over CRF on
that clip).

Tests: ABR within 10 %; a generous buffer leaves the stream byte-identical
(at equal tree setting) and reports no clamp; a 30 kbit buffer clamps; VBV
options alone imply ABR; two-pass with B pictures within 6 % of CRF 26's rate
and 0.3 dB of its PSNR, P-only within 6 %, both bit-exact against the decoder
(coding order with B pictures); the stats file's header and types.

## Robustness: extreme and hostile inputs

`tools/robust_sweep.sh <aegenc>` runs the rate control at the edges (1 kb/s
to 100 Mb/s, CRF 0 and 51, QP ranges pinned to one value, 1 kbit buffers,
one-picture streams, 2x2 and 16x16 pictures, slices, frame threads, CAVLC,
all-intra) and the second pass on damaged and mismatched statistics, on a
sanitizer build, and requires no sanitizer report, no stream the decoder
complains about, and a refused input to exit 1 or 2 with a reason.  What it
found (`194df06`):

* **All-intra ABR did not control the rate.**  An intra picture takes the
  anchors' recent average QP, and with `--bframes 0 -g 1` there are no
  anchors: the QP stayed where it started.  vtest at 2500 kb/s coded
  15932 kb/s (short: 11264).  An intra picture right after an intra picture
  is now rate-controlled like an anchor, its intra estimate standing in for
  the complexity (x264's rule: accum_p only when the last non-B picture was
  not intra): 2583 and 2580 kb/s, the QP settling within ten pictures.
* **Short GOPs** (not all-intra) still overshoot in the first seconds: the
  intra pictures follow the anchors' slowly decaying average, as in x264, and
  only the overflow correction makes the P pictures pay for them.  180
  frames, preset 3: `-g 1` (with B pictures: P pictures between the IDRs)
  vtest +10 %, cock +10 %, short +13 %; `-g 5` +13 / +7 / +10 %; `-g 15`
  +10 / +6 / +7 %.  Two passes are the answer there.
* **Two-pass statistics** were trusted.  A second pass with another
  B-picture setting followed a plan made for other picture types (+27 % with
  B pictures measured and none coded); `-g 1` in pass 2 coded intra pictures
  from a plan for P pictures (+100 %); QP 999 in the file coded +85 %; all
  bits 0 coded +72 %; one line with display index 99999999 asked for a 3 GB
  table; coding 10 of 40 measured pictures shared the target out over 40
  (+26 %).  Now: the plan covers a picture only if pass 2 codes it as the
  same kind (intra, P, B), the rest fall back to the one-pass controller
  (`-g 1`: +27 %, the short-GOP figure); pass 1 writes `aeg-stats 2
  bframes=N` (the resolved value, `aeg_stats.bframes`) and pass 2 refuses
  another setting, as x264 does; `aeg_encoder_set_pass1()` refuses a type or
  QP out of range and statistics without bits; the CLI refuses an index more
  than 65536 past the pictures before it and plans only the pictures `-n`
  lets it code (+3 %); the reason is printed.
* **`--qpmax 0`** was silently QP 51: 0 is the library's "unset" (a zeroed
  `aeg_params` must not pin every macroblock to QP 0).  The CLI refuses it.
* **Undefined behaviour**, harmless on x86 but undefined: 32-bit loads and
  stores through `int *` at unaligned pixel addresses in the SSE2 chroma
  motion compensation and inverse transform (now `memcpy`, the same `movd`),
  and left shifts of negative dequantised coefficients (now multiplications).
  CI had excluded both checks (`-fno-sanitize=alignment,shift`).  With the
  exclusion gone, the conformance suite under full UBSan found the rest: the
  same shift in the CPU reference of the coefficient batch, in the scalar
  forward transform and in the device kernels' source (`t3 << 1`, `ls <<
  sh`); and a signed overflow in `mb_variance()` (AQ), where `256 * sum2`
  passes 2^31 for a bright block -- the result was right only because the
  wrap cancels in the subtraction; now 64-bit.  Output unchanged (12
  configurations with and without AQ, and the device paths by
  `gpu_validate.sh`).  CI now builds with every UBSan check, non-recoverable,
  and runs the sweep.

CQP and CRF output is unchanged (14 configurations, two clips, presets 0-4,
CAVLC); ABR on the five long clips is unchanged.  Tests: all-intra ABR
within 10 %, four refused statistics files, a pass 2 coding 30 of 60
measured pictures within 10 %, `--qpmax 0` and `--qpmin 40 --qpmax 30`
refused.

## The library interface

What an application embedding the encoder needs beyond the CLI, and what was
wrong with the interface as it stood:

* **Logging** went to stderr only, through a global `aeg_log_level`.  Now
  `aeg_set_log_callback(fn, opaque)` takes every message as one line with its
  level (formatted once, newline stripped; the sink is swapped under a lock,
  the callback runs outside it so it may log itself).  The levels were also
  mixed: "level 1" held both real warnings (a device failing over to the CPU,
  CAVLC dropping B pictures, a clamped bitrate) and the encoder's summary
  (backend, CRF, size).  They are now `ERROR` / `WARNING` / `INFO` / `DEBUG`;
  the library default is `WARNING`, so a clean encode embedded in an
  application prints nothing.  `aegenc -v N` maps onto the new levels and
  prints exactly what it printed before (compared for `-v 0/1/2` on three
  configurations).
* **The last error** was one static pointer for the process; two encoders
  failing in two threads overwrote each other's message.  Now thread-local.
  Two other process-wide variables written on every picture (the scheduler
  and frame-thread timers behind `AEG_LATIME`) were data races between
  encoders in parallel threads; they are per encoder now.
* **Parameters**: `aeg_encoder_get_params()` returns what the encoder runs
  with -- every `-1` resolved, the level the SPS signals, slices, frame
  threads, the timebase.
* **Parameter sets**: `aeg_encoder_headers()` returns the SPS/PPS (with PPS 1
  under weighted prediction) for an avcC box.  The IDR pictures now copy the
  same buffer, written once at open, so the two cannot disagree.
* **Input**: `aeg_encoder_encode()` took only a packed I420 buffer, so every
  caller with a decoder frame or padded capture buffer had to repack it.
  `aeg_encoder_encode_planes()` takes three planes with their own strides;
  the packed call is now a wrapper of it.  Same bytes out.
* **Timestamps**: packets carried only `pts`, which with B pictures is not
  monotonic in decoding order, and an MP4/MKV muxer needs a `dts`.  `dts` =
  the submitted pts in submission order, popped one per packet, minus
  `bframes` frame durations (the reorder depth the HRD model already uses);
  `params.timebase_num/den` says how long a frame is in pts units (default
  one tick per frame).  Strictly increasing, never above pts for
  constant-rate input; a variable-rate source can break the second property,
  as with x264.
* **Packaging**: only a static archive and no install rule.  Now `make
  shared` (position-independent, `-fvisibility=hidden`, only the `AEG_API`
  symbols exported -- 26 of them; 27 with `aeg_encoder_encode_image()`), `make install` / `uninstall` with `PREFIX`,
  `DESTDIR`, `LIBDIR`, `INCLUDEDIR`, `BINDIR`, an `aeg.pc`, and a run-time
  `aeg_version()`.  The soname is `libaeg.so.0`: `aeg_params` and
  `aeg_packet` are plain structs the caller allocates, so a field added in a
  later release changes the ABI; that is accepted while the major version is 0.
* **A build bug found on the way**: the Makefile only included the
  dependency files of the library objects, so a changed `aeg.h` did not
  rebuild `cli/aegenc.c` -- the CLI kept the old `aeg_params` layout and read
  garbage (`-p -1 is out of range`).  All `.d` files are included now.

Checks: `tests/api_check.c` (in `make test`; also under ASan with leak
detection and under ThreadSanitizer in CI), `tests/install_check.sh` (`make
install-check` in CI, plus a real `sudo make install` and the example built
with the system's pkg-config).  CQP, CRF and ABR output byte-identical to the
previous commit: 13 configurations (presets 2-4, no B, CRF, CRF+AQ, ABR, VBV,
HRD, CAVLC, slices, slice threads, short GOP), the CPU and OpenCL conformance
suites.

## Input formats and interlace signalling

The encoder codes 8-bit 4:2:0 progressive pictures.  What was missing was
everything a real source hands over that is not exactly that: decoders and
capture cards give NV12, professional material is 4:2:2 or 10-bit, a YUV4MPEG2
pipe from ffmpeg says `C420p10` or `C422`, and broadcast material is
interlaced.  x264's 8-bit build accepts all of them by converting on input;
this does the same, in the library (`aeg_encoder_encode_image()`, `aeg_image`,
`src/input.c`) so an application gets it too, and in the CLI.

* **Layouts**: I420, YV12, NV12, NV21 (4:2:0); I422, YV16, NV16 (4:2:2);
  I444, YV24 (4:4:4); I400 (grey, chroma coded as 128).  Each 8 to 16 bits
  per sample; above 8, one native-endian `uint16_t` per sample holding
  0..2^depth-1 (P010/P016 are NV12 with depth 16).
* **Depth**: every sample goes to 16 bits (`s << (16 - depth)`, x264's input
  scaling) and is dithered back to 8 bits with x264's Sierra-2-4A error
  diffusion (`filters/video/depth.c`).  `tests/input_check.c` compares it bit
  for bit with a transcription of x264's loop on random planes of odd sizes,
  and checks its two properties: a plane that was scaled up from 8 bits comes
  back unchanged (so `yuv420p10le` made from an 8-bit source codes to the
  same bytes as the source), and a flat level between two codes keeps its
  mean (100.25 → mean 100.2500).
* **Chroma**: resampled to 4:2:0 in the 16-bit stage, for H.264's default
  chroma siting (`chroma_loc_type` 0): horizontally (1 2 1)/4 on the even
  columns, vertically (1 3 3 1)/8 between two rows; for interlaced material
  the vertical filter runs within each field, (3 1)/4 for the top field and
  (1 3)/4 for the bottom one, where the 4:2:0 chroma lines of the two fields
  sit -- a progressive filter would blend the fields and smear colour on
  motion.  `input_check` checks every output sample against the taps written
  out with explicit edge clamping (sizes 1x1 to 65x34) and that a 4:2:2
  source with different chroma in its two fields keeps them apart when
  interlaced and blends them when not.
* **Exact paths**: 8-bit I420 and YV12 go straight to the encoder (no copy),
  NV12/NV21 are only deinterleaved, the luma of any 8-bit picture is copied.
  `tests/api_check.c` encodes the same picture in every layout that must
  convert exactly (NV12, NV21, YV12, 10- and 16-bit scaled up, 4:4:4 / 4:2:2
  / NV16 / 16-bit 4:4:4 with flat chroma, grey against chroma 128), with
  junk in the row padding, progressive and interlaced, and requires the very
  bytes of the I420 encode.
* **Cost** (one 1920x1080 picture, one core): 10-bit 4:2:0 9.9 ms, 10-bit
  4:2:2 10.3 ms, 10-bit 4:4:4 14.6 ms, 8-bit 4:2:2 4.5 ms, NV12 0.5 ms.  The
  dither is serial by nature (each sample's error feeds the next); against
  the encode of the same picture this is a few per cent.
* **CLI**: `--input-csp`, `--input-depth` for raw input; YUV4MPEG2 `C` tags
  `420jpeg/420paldv/420mpeg2/420`, `422`, `444`, `mono`, with `p9`..`p16`
  (`mono9`..`mono16`); `444alpha` is refused.  The 4:2:0 siting variants are
  read as they are, as x264 does.  A header that contradicts
  `--input-csp`/`--input-depth` is an error.

**Interlace**: `--tff` / `--bff` (`params.interlaced` 1 / 2), or a `.y4m`
header's `It` / `Ib` (`Im`, mixed, is warned about and coded as progressive).
By default the pictures are coded as frames -- `--field` codes field
pictures instead (*Field pictures (PAFF)* below; no field macroblock pairs,
MBAFF, yet) -- and signalled as interlaced, what x264 calls
`--fake-interlaced`:

* SPS `frame_mbs_only_flag` 0 with `mb_adaptive_frame_field_flag` 0, the
  height in macroblock-row pairs (the coded height is rounded up to 32, the
  crop counts in units of 4 rows, so the height must be a multiple of 4 --
  x264's rule), every slice header `field_pic_flag` 0; a stream with neither
  CABAC nor the 8x8 transform says Main instead of Baseline (Baseline forbids
  `frame_mbs_only_flag` 0).
* VUI `pic_struct_present_flag` 1 and a picture-timing SEI on every picture
  with `pic_struct` 3 (top field, bottom field) or 4 (bottom, top), with the
  HRD delays when `--nal-hrd` is on.  x264 writes `pic_struct` only in
  MBAFF mode or with `--pic-struct`, and sets the bottom field's picture
  order count one above or below the top's; frames coded as frames carry no
  bottom-field offset here, so `pic_struct` is what tells the decoder the
  field order.
* Levels 2.1 to 4.1 are the only ones that allow `frame_mbs_only_flag` 0;
  the automatic level picks the lowest of them that fits, and a forced level
  outside that range is warned about.

ffmpeg reports `i:T` / `i:B` on every picture (`showinfo`), and the streams
decode bit-exactly with B pictures, slices, CAVLC and HRD
(`tests/run_tests.sh`, *input formats and interlace*).

**Level table fix** (found while adding the interlace column): level 2.2 had
MaxMBPS 40500 instead of 20250, level 3.2 was missing, and its limits were
filed under 4.0, 4.0's under 4.1, so 4.1's 50 Mb/s did not exist.  The first
error *understated* the level: a 640x352 stream at 30 fps (26400 macroblocks
per second) claimed level 2.2, whose limit is 20250, where 3.0 is needed.
The others overstated it (a stream that fits 3.2 claimed 4.0).  The table is
the standard's Table A-1 now; only the `level_idc` byte of the SPS changes,
and a CQP/CRF stream with no rate given, which is sized for 20 Mb/s, now
claims 3.2 instead of 4.0.  Every other byte of 8-bit 4:2:0 progressive
output is unchanged: 14 configurations compared with the level forced to the
same value, and the conformance suites.

## High 10 (10-bit), x264's way

`--bit-depth 10` / `params.bit_depth = 10` codes High 10 (`profile_idc` 110,
`bit_depth_luma/chroma_minus8` 2, 4:2:0).  It is built the way x264 builds
its high-depth encoder: no run-time sample-size switches in the inner loops,
the depth-dependent sources compiled twice instead.

**Build.** `src/bitdepth.h` gives one build's `pixel` (`uint8_t` /
`uint16_t`), `dctcoef` (`int16_t` / `int32_t`: 10-bit levels and transform
outputs outgrow 16 bits), `PIXEL_MAX`, `QP_BD_OFFSET` (6 per extra bit) and
`QP_MIN`.  transform, intra, mc, deblock, cavlc, cabac, backend, lowres,
input and encoder are compiled with `-DBIT_DEPTH=8` and `-DBIT_DEPTH=10`; the
Makefile renames every symbol the 10-bit objects define with a `_10` suffix
(`nm` lists them, `objcopy --redefine-syms` applies the list, references
included), so both sets live in one `libaeg.a`/`libaeg.so`.  tables,
bitstream, util, lookahead and the device backends are compiled once:
nothing in them handles a sample at the coded depth (the lookahead works on
its own 8-bit half-resolution planes, which `lowres.c` -- per depth -- builds
from the full-depth picture).  `src/api.c` is the public `aeg_encoder_*`
layer: open dispatches on `params.bit_depth`, every other call on the depth
stored in the encoder (`aeg_params` is the first member of the encoder's
struct).  `make BIT_DEPTHS=8` leaves the 10-bit set out; open then refuses
10 with a message.  The exported symbols are the same 27.

**QP.** Every QP the encoder holds -- per macroblock, the rate control, the
slice header, `mb_qp_delta`, the deblocking filter -- is QPY, -12..51 at 10
bits; the offset is added only where a quantiser or lambda table is indexed
(QP'Y = QPY + 12).  So `-q`, `--crf`, `qp_min`/`qp_max` keep the 8-bit scale
(the same quantiser step relative to the signal at both depths), and CRF and
the rate controls may go below zero.  x264 counts its 10-bit `--qp` from
QP'Y instead; its `--crf` has the 8-bit scale, as here.  The CABAC context
initialisation clips a negative SliceQPY to 0 (9.3.1.1).  Distortions are in
10-bit units (SSD x16, SAD/SATD x4 against 8 bits) and the lambdas follow:
the RD lambda through QP'Y, the motion-search lambda and the UMH early-exit
threshold `<< 2`, the AQ variance `>> 4`, the weighted-prediction
thresholds and offsets scaled.

**Input.** 10-bit I420/YV12 images go in as they are (after one OR pass that
sends a frame with a sample above 1023 -- the caller's error -- through the
conversion, which clips: coded as it is, the residual and the levels would
overflow); 8-bit input is scaled up exactly; anything else is resampled and
dithered to 10 bits.  The reconstruction (`aeg_encoder_recon`,
`--dump-recon`) is 16-bit samples.  10-bit runs on the CPU only; `-b opencl`
/ `-b cuda` falls back with a warning.

**Verification.**

* Every 10-bit configuration decodes in FFmpeg 7.0.2 (`yuv420p10le`) to
  exactly the encoder's reconstruction: presets 1-4, QP 0 and 51, CRF -12 and
  -6, ABR, VBV + NAL HRD, two-pass, CAVLC, slices, `--tff`, weighted P, chroma
  QP offsets +-12, true 10-bit input, and four synthetic extremes (full-range
  noise, a moving 0/1023 checkerboard, frames alternating 0 and 1023, ramps)
  at QP 0, CRF -12, QP 51 and 50 kb/s.  Ten of these are in
  `tests/run_tests.sh` (*High 10*, compared after undoing the coding order).
* The stream does not depend on the threads: frame threads 1 = 3 and slice
  threads 1 = 3 at constant QP, CRF and ABR (in the suite).
* `tests/api_check.c`: depth 8/0/10 open, 1/9/12/-8 are refused; the SPS is
  `profile_idc` 110; every input layout that converts exactly gives the same
  10-bit stream as I420 (as at 8 bits); a picture with detail below the 8-bit
  step, coded at QP 0, comes out with an SSD of 8074 at 10 bits against
  29000 for the 8-bit encoder (the rounding floor, MSE 1.25); samples above
  1023 give the stream of the clipped picture; an 8-bit stream is unchanged
  by a 10-bit encoder running beside it; 8- and 10-bit encoders in parallel
  threads give the bytes they give alone (also under ThreadSanitizer).
* ASan + UBSan: the extremes above at ten settings each, and api_check,
  clean.  The 10-bit quantisers take the C path with 64-bit products (the
  SSE2 ones assume 16-bit coefficients).
* 8-bit output is byte-identical to the previous commit
  (`tools/identity.sh`, fourteen configurations, and the suites), with and
  without the 10-bit build in the library; the 8-bit speed is unchanged.

**What 10 bits buy on 8-bit sources.**  The five test clips, 30 frames,
preset 3, QP 22/27/32/37, bytes and PSNR (decoded stream against the source,
all three planes, global MSE, 8-bit scale: a 10-bit decode is compared with
the source `<< 2`, peak 1023):

| clip | 8-bit | 10-bit | BD-rate |
|---|---|---|---|
| vtest | 100119/45.85 58658/42.62 35934/39.90 21091/37.02 | 99692/46.21 58307/42.71 35829/39.96 21199/37.06 | -2.29 % |
| cock | 223548/47.60 127303/44.88 70476/41.99 38594/39.09 | 219334/48.05 125802/45.11 70111/42.12 38645/39.17 | -4.76 % |
| mega | 58340/47.54 30562/45.00 16897/42.39 9812/39.62 | 56279/47.97 30155/45.30 16709/42.60 9646/39.78 | -7.35 % |
| short | 124928/45.50 63869/42.13 31319/39.18 16788/36.51 | 122559/45.83 62481/42.32 31167/39.31 16685/36.59 | -5.13 % |
| tree | 102643/46.65 66895/41.96 39015/37.84 19993/34.33 | 101787/46.76 66588/42.02 38927/37.93 19901/34.40 | -1.54 % |

Mean **-4.21 %**: the gain grows toward high quality (QP 22: +0.11 to +0.45
dB at 0.4-3.5 % fewer bytes), where the 8-bit prediction loop's rounding is
a larger share of the error.

**Speed** (at this commit; see *10-bit SIMD* below for the kernels that
followed).  No 10-bit SIMD yet: the same runs at QP 22, default threads,
2-core Xeon, median of three, 8-bit / 10-bit fps: vtest 65.4 / 35.7, cock
25.9 / 12.6, mega 81.3 / 39.2, short 55.6 / 30.4, tree 60.4 / 40.0 --
1.5-2.1x the time.  The
SSE2/AVX2 kernels (SAD, SATD, SSD, the quarter-pel filters, the quantiser,
the transforms) have 16-bit-sample counterparts to write, as x264 has.

## 10-bit SIMD

The vector kernels now have 16-bit-sample forms, so the 10-bit build runs at
about the 8-bit speed.  Every one returns the 10-bit C path's integers -- the
10-bit stream is byte-identical to the previous commit's -- and the 8-bit
code paths compile to the same arithmetic (8-bit output byte-identical too).

**What has a 10-bit form** (SSE2 unless noted; AVX2 chosen at run time as
before, `AEG_NOAVX2=1` forces SSE2; `AEG_SIMD` in `bitdepth.h` is "SSE2 at
either depth", `AEG_SSE2` stays "the 8-bit forms"):

| kernel | how it stays exact at 10 bits |
|---|---|
| SATD 4x4 / 8xN / 16xN, `satd_avg` (+AVX2) | same code, only the loads differ (`ld_px*` helpers); int16 intermediates reach 16 x 1023 = 16368, and the half-sum 32736 |
| SA8D 8x8 | the third butterfly stage would reach 64 x 1023; two stages stay in int16, and the third is replaced by \|a+b\|+\|a-b\| = 2 max(\|a\|,\|b\|) |
| SAD, `sad_x4` (+AVX2 16- and 8-wide) | \|a-b\| = subs(a,b) \| subs(b,a); at most 32 differences per uint16 lane before a multiply-add widens them |
| SSD, `idct_add_ssd4`, `add_res16_ssd` | `pmaddwd` of differences (+-1023); a 16x16 sum is below 2^28 |
| 4x4 forward transform, quantiser, inverse (+ add, DC-only path) | int32 lanes throughout, as the scalar code at this depth (no int16 truncation but the residual's final store, which wraps); the 32-bit multiply is `pmuludq` x2 (SSE2 has no `pmulld`); quantiser: \|coef\| <= 2^17 and mf < 2^14 keep product + offset below 2^32 (else the C loop) |
| 8x8 forward transform, quantiser, dequantiser, inverse, the trellis's column SSD (AVX2 only; SSE2-only machines run C) | one row of eight int32 per register, two transposes; quantiser: \|coef\| <= 2^17, mf < 2^15, product + offset < 2^32; dequantiser: levels beyond what an int32 product holds (per QP) go to C |
| `idct_8x8_add` | saturating int16 add then a clamp to [0, 1023]: the scalar clip of the exact sum |
| half-pel planes (h, v, hv) | 20(a+b) - 5(c+d) + (e+f) lies in [-10230, 42966]: not int16, but within 2^16, so wrapping 16-bit arithmetic plus a bias holds it as an unsigned lane; (v+16)>>5 = ((v+16+320x32)>>>5) - 320; the hv stage widens the biased sums and removes 32 x bias at the end |
| quarter-pel average, chroma filter | `pavgw`; the chroma weights add up to 64 and 64 x 1023 + 32 < 2^16 |
| `me_pick` rate lanes | no samples involved: now on at both depths |
| input conversion | for input depths up to the encoder's, x264's dither is an exact shift (the rounding term and every error term are zero), so 8-bit input is `<< 2` directly, without the 16-bit plane and the dither pass; 4:2:2/4:4:4 chroma (resampled) still dithers |

Not vectorised at either depth: deblocking, intra prediction, CABAC.

**Verification.**
* `tests/simd_check.c`, new, compiled twice (`build/simd_check`, and
  `build/simd_check10` with `-DBIT_DEPTH=10` and its library references
  renamed by the same symbol list as the library's 10-bit objects), run by
  `make test` with and without AVX2.  It compares every kernel above with a
  scalar reference -- written in the test (the metrics, the filters'
  formulas) or the library's `_c` functions (transforms; new:
  `aeg_transform_4x4_c`, `aeg_quant_4x4_c`, `aeg_idct_residual_c`) -- on
  random, extreme (0 / PIXEL_MAX only), nearly flat and ramp inputs, at 80
  block shapes, every QP from QP_MIN to 51, levels from the real
  transform + quantiser, coefficients over the whole legal range and past it
  (where a vector form must hand over), the half-pel planes and all sixteen
  quarter-pel positions against the scalar `aeg_mc_luma`, all 64 chroma
  fractions, and the input conversion of I420 and NV12 at depths 8 to 10 and
  12 against widening + dither.
* Mutations caught by it: a wrong butterfly sign, a disabled dequantiser
  guard, a wrong quantiser clamp, a wrong rounding constant in SA8D, the
  half-pel filters and the chroma filter, a too-small bias, a wrong input
  shift.  (Two that it did not catch turned out equivalent: an arithmetic
  instead of a logical shift in the 4x4 quantiser, and a 2^18 guard -- the
  largest mf is 13107, so neither product reaches 2^31.)
* It found two latent issues on the way: the 8-bit SSE2 `aeg_ssd` treated any
  width other than 8 and 16 as 4 (no caller passes one; other widths now
  take the C loop), and widening the AVX2 gate to both depths at first also
  enabled the 8-bit-coefficient AVX2 idct8 / column-SSD kernels in the 10-bit
  build (the encoder comparison caught that; they are gated to 8 bits again
  and the test now covers those functions at 10 bits).
* 10-bit streams byte-identical to the previous commit: `tools/identity.sh
  --bit-depth 10` (14 configurations, with and without AVX2); the five clips
  x presets 0/2/4 x `-q 0` / `-q 22` / `-q 51` / `--crf -12` (60); B-frames +
  weighted P, 2-pass, AQ 2, `--tff`, chroma QP offset +12 at QP 0, CAVLC
  without 8x8, VBV, frame threads, slices, partitions 31, temporal direct;
  10- and 12-bit, NV12 and 4:2:2 input (the direct conversion and the dither
  path).
* 8-bit: `tools/identity.sh` 14/14 against the previous 8-bit reference;
  `make test` (all conformance cases bit-exact, simd_check at both depths);
  the OpenCL conformance run (PoCL) bit-exact; ASan + UBSan on both
  simd_checks and on 8- and 10-bit encodes (sub-partitions, B-frames, 10-bit
  and 4:2:2 input); TSan api_check; the 8-bit-only build (no `_10` symbols,
  `simd_check10` not built); install-check.

**Speed.**  2-core Xeon, 30 frames, preset 3, median of three, interleaved
runs, fps:

| clip (QP 22, default threads) | 8-bit | 10-bit before | 10-bit now | 10-bit speed-up | 10-bit / 8-bit time |
|---|---|---|---|---|---|
| vtest | 59.22 | 33.25 | 65.61 | 1.97x | 0.90 |
| cock | 28.51 | 14.09 | 25.67 | 1.82x | 1.11 |
| mega | 75.54 | 39.89 | 69.72 | 1.75x | 1.08 |
| short | 57.18 | 30.28 | 54.08 | 1.79x | 1.06 |
| tree | 60.75 | 34.51 | 59.40 | 1.72x | 1.02 |
| cock QP 26, one thread | 17.47 | 8.16 | 15.21 | 1.86x | 1.15 |

(The machine is noisy at the 10 % level; the 0.90 of vtest is that noise
and thread scheduling, not a faster 10-bit encoder.)  The sampled profile of
the one-thread run went from 975 samples at 10 bits against 407 at 8 to 496
against about 500: what is left at 10 bits is the same work as at 8 (CABAC
pricing, the trellis, motion search), none of it above 6 % self time.

## The final comparison with x264 and x265

Every preset against every x264 preset (and three x265 presets), measured the
same way, in one sitting, on the encoder as of `f52e4af`.  This supersedes the
scattered x264 numbers earlier in this file, which were taken at different
stages of the encoder (several with B pictures off, some on synthetic clips or
against x264's baseline profile).

**How.**  `tools/x264cmp.py` (`run` / `report` / `speed`; the long clips come
from `tools/real/longclips.sh`).  References: libx264 core 164 r3191 and x265
3.5, both with their assembly (AVX2, and AVX-512 for x264) through FFmpeg
7.0.2, with `--tune psnr` (psy-rd and AQ off) and otherwise their defaults.
aegenc at its defaults (AQ only acts under rate control, so at constant QP
both sides are without AQ; x264 turns AQ and mbtree off at constant QP by
itself, which is why the old reference, medium with `psy=0`, measured identical
to medium `--tune psnr`).  QP 22/25/28/31/34/37 (CRF 18-33 for the rate
control table); global PSNR (MSE over every sample of all three planes of the
whole sequence) of the decoded stream against the source, and FFmpeg's SSIM
in dB; BD-rate over the shared span as in `tools/bdrate.py`.  **Negative =
aegenc needs fewer bits.**  `*` = the two curves share less than half of the
smaller span.  Quality runs single-threaded on both sides.  Clips: the five
of `clips.txt` (30 pictures each), the six validation segments (30 each) and
three long ones (cockL 640x352 240 pictures, vtestL 384x288 300, megaL
352x256 240): 14 clips.

### Quality at constant QP

PSNR BD-rate, mean over the 14 clips:

| aegenc \ ref | x264 ultrafast | superfast | veryfast | faster | fast | medium | slow | slower | veryslow | x265 fast | x265 medium | x265 slow |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p0 | -21.3% | +37.0% | +52.3% | +64.8% | +72.5% | +77.0% | +82.5% | +88.7% | +89.5% | +77.1% | +82.4% | +97.1% |
| p1 | -47.9% | -16.4% | -7.3% | +0.4% | +4.9% | +7.3% | +10.7% | +14.5% | +14.9% | +6.8% | +10.5% | +19.3% |
| p2 | -49.8%* | -20.0% | -11.3% | -4.0% | +0.2% | +2.6% | +5.8% | +9.4% | +9.8% | +2.0% | +5.5% | +14.0% |
| p3 | -54.0% | -26.7% | -18.7% | -11.9% | -8.0% | **-5.9%** | -2.9% | +0.4% | **+0.8%** | -6.4% | -3.2% | +4.7% |
| p4 | -54.3% | -27.1% | -19.2% | -12.4% | -8.5% | -6.4% | -3.5% | -0.2% | **+0.1%** | -7.0% | **-3.8%** | +4.1% |

SSIM BD-rate, same runs:

| aegenc \ ref | x264 ultrafast | superfast | veryfast | faster | fast | medium | slow | slower | veryslow | x265 fast | x265 medium | x265 slow |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p0 | -34.9% | +26.6% | +42.6% | +51.4% | +57.9% | +61.2% | +65.0% | +70.9% | +72.1% | +65.5% | +71.6% | +76.1% |
| p1 | -55.3%* | -21.4% | -11.9% | -6.5% | -2.5% | -0.6% | +1.8% | +5.4% | +6.1% | +1.9% | +5.6% | +8.9% |
| p2 | -57.1%* | -25.1% | -16.0% | -10.8% | -7.1% | -5.3% | -2.9% | +0.5% | +1.1% | -2.9% | +0.6% | +3.9% |
| p3 | -59.5%* | -29.3% | -20.7% | -15.9% | -12.3% | -10.7% | -8.5% | -5.2% | -4.7% | -8.5% | -5.2% | -2.2% |
| p4 | -59.6%* | -29.7% | -21.2% | -16.3% | -12.8% | -11.2% | -9.0% | -5.8% | -5.2% | -9.1% | -5.8% | -2.7% |

The key pairs hold on each clip set separately (PSNR):

| pair | main (5) | validation (6) | long (3) | all 14 |
|---|---|---|---|---|
| p3 vs x264 medium | -4.99% | -5.49% | -8.08% | -5.9% |
| p3 vs x264 veryslow | +1.00% | +0.72% | +0.40% | +0.8% |
| p4 vs x264 veryslow | +0.57% | +0.26% | -0.79% | +0.1% |
| p4 vs x265 medium | -7.60% | -1.72% | -1.50% | -3.8% |
| p4 vs x265 slow | +1.3% | +5.4% | +6.0% | +4.1% |

The spread between clips is wider than any of these means: against x264
veryslow, p4 is +4.9 / +7.3 / +8.9 % on the three cockatoo clips and +5.6 /
+7.7 % on the two tree clips, and -6.1 / -7.7 / -5.4 / -11.6 % on the four
vtest clips (static camera, people walking).  Why the cockatoo and tree
clips favour x264 has not been isolated (x264 uses three references here,
aegenc at most two P references and one per B list).

### Rate control: CRF, both with their macroblock tree

aegenc `--crf N --aq 0`, x264/x265 `-crf N --tune psnr` (mbtree / cutree on,
AQ off on all three):

| PSNR BD-rate | x264 medium | x264 veryslow | x265 medium |
|---|---|---|---|
| p3, 11 short clips (30 pictures) | -1.5% | +5.8% | -2.5% |
| p4, 11 short clips | -2.0% | +5.3% | -3.0% |
| p3, 3 long clips (240-300) | **+2.5%** | +13.2% | +7.6% |
| p4, 3 long clips | **+1.2%** | **+11.8%** | +6.2% |

| SSIM BD-rate | x264 medium | x264 veryslow | x265 medium |
|---|---|---|---|
| p3, long clips | +0.3% | +10.9% | +11.3% |
| p4, long clips | -1.0% | +9.4% | +9.8% |

(The short-clip table has an x264 slow column too: p3 +1.5 %, p4 +1.0 %.)
This is the largest remaining quality gap.  At constant QP p3 is 5-8 %
ahead of x264 medium; with both trees on it is 1.5 % ahead on 30 pictures and
2.5 % *behind* on 240-300, so x264's macroblock tree is worth about 4 (short
clips) to 11 (long clips) points more than aegenc's (vtestL: -19.4 % at constant QP,
+5.9 % at CRF).  Against veryslow at CRF on the long clips the gap is 12 %.

### High 10

Both encoders get the same 10-bit source (the 8-bit clip shifted left by two,
as yuv420p10le, so no up-conversion is part of the comparison); x264's 10-bit
QP is ours + 12 (it counts from QP'Y).  Five main clips, PSNR at 10 bits:

| aegenc \ ref | x264 medium 10-bit | x264 veryslow 10-bit |
|---|---|---|
| p3 | -4.9% (SSIM -9.5%) | +1.1% (SSIM -4.0%) |
| p4 | -5.4% (SSIM -10.1%) | +0.5% (SSIM -4.6%) |

The same as at 8 bits (p3 vs medium -5.0 %, p4 vs veryslow +0.6 % on these
clips): the High 10 path keeps the 8-bit encoder's standing.

### Speed

`tools/x264cmp.py speed`, QP 27, median of 3, wall clock on this 2-core Xeon
(AVX2).  For x264/x265, FFmpeg's own time (start-up and reading the raw file,
measured with `-c:v rawvideo -f null`) is subtracted, which makes the two
fastest x264 rows (a 240-picture encode in 0.2-0.6 s) approximate.  hd = the
cockatoo clip scaled to 1280x720, 120 pictures.

| config | 1 thread, cockL | 1 thread, hd | default threads, cockL | default threads, hd |
|---|---|---|---|---|
| aegenc p0 | 76.6 | 24.5 | 144.5 | 45.5 |
| aegenc p1 | 30.5 | 7.8 | 56.1 | 14.9 |
| aegenc p2 | 25.9 | 6.9 | 44.9 | 12.5 |
| aegenc p3 | 13.5 | 4.0 | 24.4 | 7.3 |
| aegenc p4 | 11.5 | 3.7 | 21.5 | 6.8 |
| x264 ultrafast | 1381.4 | 406.5 | 2102.6 | 660.8 |
| x264 superfast | 414.5 | 111.3 | 683.6 | 196.4 |
| x264 veryfast | 198.9 | 66.8 | 384.6 | 120.1 |
| x264 faster | 122.2 | 44.1 | 211.1 | 83.6 |
| x264 fast | 95.8 | 31.5 | 168.4 | 60.0 |
| x264 medium | 85.5 | 26.1 | 147.6 | 49.5 |
| x264 slow | 58.0 | 18.4 | 99.3 | 33.3 |
| x264 slower | 28.6 | 9.5 | 54.4 | 17.6 |
| x264 veryslow | 13.8 | 4.6 | 27.9 | 9.1 |
| x265 fast | 37.8 | 9.2 | 61.1 | 16.5 |
| x265 medium | 32.0 | 9.4 | 50.3 | 15.3 |
| x265 slow | 10.6 | 3.1 | 17.8 | 5.5 |

(fps; default threads = aegenc's frame threads, x264/x265's own pools.)

### Matched quality, matched speed

Each aegenc preset against the x264 preset of the same quality (constant-QP
PSNR BD-rate, 14 clips) and the one of the same single-thread speed (cockL):

| aegenc | fps | x264 at equal quality | its fps | x264 faster by | x264 at equal speed | aegenc's BD-rate against it |
|---|---|---|---|---|---|---|
| p0 | 76.6 | between ultrafast (-21.3%) and superfast (+37.0%) | 414.5 (superfast) | 5.4x | medium (85.5) | +77.0% |
| p1 | 30.5 | faster (+0.4%) | 122.2 | 4.0x | slower (28.6) | +14.5% |
| p2 | 25.9 | fast (+0.2%) | 95.8 | 3.7x | slower (28.6) | +9.4% |
| p3 | 13.5 | slower (+0.4%) | 28.6 | 2.1x | veryslow (13.8) | +0.8% |
| p4 | 11.5 | veryslow (+0.1%) | 13.8 | 1.2x | veryslow (13.8) | +0.1% |

### Where that leaves the encoder

* **Quality at constant QP: x264-equivalent at the slow end.**  p3/p4 equal
  x264 veryslow (+0.1 to +0.8 %, PSNR; 4.7-5.2 % fewer bits by SSIM) and are
  3-4 % ahead of x265 medium, 4-5 % behind x265 slow.  x265-equivalence in the
  sense of its slow presets is not reached, and an H.264 stream is not
  expected to reach it.
* **Rate control is the quality gap.**  With the trees on, long clips: +1.2 %
  against x264 medium at p4, +11.8 % against veryslow.  The next quality work
  is the macroblock tree over long GOPs, not the macroblock decision.
* **Speed per quality: 1.2-5.4x behind x264 on the CPU**, least at p4 and most at
  p0-p2.  p0 is not competitive at all (x264 superfast is 5x faster at 37 %
  fewer bits).  At matched *speed* aegenc ties x264 at p3/p4 and trails it by
  9-15 % at p1/p2.  This is the CPU-only standing; the GPU stages exist for
  exactly this gap and still need hardware to be measured.

Reproduce (a few hours on this machine, resumable: every point is cached
in `/tmp/x264cmp.json` and re-measured only when `build/aegenc` changes):

```
bash tools/real/fetch.sh && bash tools/real/longclips.sh
python3 tools/x264cmp.py run --clips main,val,long
python3 tools/x264cmp.py run --clips main,val --mode crf --only '^(aeg-p3|aeg-p4|x264-medium|x264-slow|x264-veryslow|x265-medium)$'
python3 tools/x264cmp.py run --clips long --mode crf --only '^(aeg-p3|aeg-p4|x264-medium|x264-veryslow|x265-medium)$'
python3 tools/x264cmp.py run --clips main --mode 10 --only '^(aeg-p3|aeg-p4|x264-medium|x264-veryslow)$'
python3 tools/x264cmp.py report --clips main,val,long [--metric ssim_db] [--detail aeg-p3:x264-medium]
python3 tools/x264cmp.py speed --threads 1 --long cockL:/tmp/real/cockL.yuv:640:352,hd:/tmp/real/hdL.yuv:1280:720
```

## The cockatoo / tree gap against x264 veryslow

The final comparison left p4 +4.9 / +7.3 % behind x264 veryslow on the two
cockatoo clips and +5.6 / +7.7 % on the two tree clips, while 5-12 % ahead on
vtest.  `tools/x264abl.py` (same measurement as `x264cmp.py`: constant QP
22-37, global PSNR, one thread; `NAME=aeg:P[:args]`, `NAME=x264:PRESET[:x264
params]`, an aegenc token `VAR=value` is an environment variable) took both
encoders apart one tool at a time on those four clips plus the three vtest
clips.

**The gap is not clip-specific; the vtest lead is.**  The constant-QP
macroblock tree, which x264 does not run at constant QP, is worth 7-11 points
on vtest and nothing on cock / tree (p4 `--mbtree 0`: cock +0.06, cock2 -2.97,
tree -1.19, tree2 +3.71, vtest +11.31 / +10.71 / +6.94).  Without it aegenc
trails veryslow on every clip (+0.3 to +11.1).

**With the same tools, aegenc is not behind.**  x264 veryslow cut down to
aegenc's toolset (`ref=2:partitions=p8x8,b8x8,i8x8,i4x4:subme=7:trellis=1:
bframes=3:b-adapt=0`) against p4 with the same fixed B structure and no tree
(`--b-adapt 0 --mbtree 0 --bqp 2`): cock -1.50, cock2 -0.61, tree +0.35,
tree2 +3.90, vtest -0.73 / -2.36 / -2.56.  The B-picture coding is fine; the
trellis-off pair (P only) even has aegenc 3.8-5.2 % ahead on cock.

**What veryslow adds over that toolset** (x264 against itself, b-adapt 2):

| x264 change | cock | cock2 | tree | tree2 | vtest | vtest2 | vtest3 | mean |
|---|---|---|---|---|---|---|---|---|
| subme 7 -> 9 | -2.05 | -2.48 | -1.16 | -1.10 | -2.06 | -1.69 | -1.91 | -1.78 |
| trellis 1 -> 2 | -1.45 | -1.39 | -0.75 | -1.42 | -0.60 | -0.51 | -0.61 | -0.96 |
| ref 2 -> 3 | +0.29 | -0.08 | -0.05 | -0.80 | -1.08 | -0.65 | -0.31 | -0.38 |
| ref 2 -> 16 | +0.04 | +0.36 | -0.46 | -2.77 | -2.53 | -2.38 | -1.14 | -1.27 |
| all of veryslow | -3.68 | -4.62 | -3.77 | -3.72 | -5.82 | -6.71 | -4.44 | -4.68 |

aegenc's trellis already runs inside the decision (x264's trellis 2) and its
`--refine-mv` is the quarter-pel RD refinement of subme 8, but that refinement
is worth 0.63 here (p4 `--refine-mv 0`: cock +1.67, vtest +0.13) against
x264's 1.8.  x264's subme >= 8 does two more things (`encoder/analyse.c`):
`mb_analyse_transform_rd` re-codes the winner with the other transform size,
and from subme 9 `b_deblock_rdo` measures every candidate's distortion after
deblocking the macroblock's internal edges (`rdo.c`).  Both are now in:

* **Transform size by RD** (`AEG_T8RD`, from preset 2): use_t8() picks 4x4 /
  8x8 by the SATD / SA8D of the residual; code_mb() codes the inter winner
  once more with the other size (`t8_force`) and keeps the cheaper price.
* **Deblock RDO** (`AEG_DBRDO`, from preset 3, with deblocking on):
  full_price() deblocks a copy of the candidate's luma over its internal
  edges (`aeg_deblock_mb_internal_luma()`: edges 4/8/12, or 8 with the 8x8
  transform, the macroblock's own QP and bS) and adds the change in SSE.
  Every candidate, intra included, is priced that way.

BD-rate against the encoder without them (negative = better):

| | cock | cock2 | tree | tree2 | vtest | vtest2 | vtest3 | mean |
|---|---|---|---|---|---|---|---|---|
| p4 + transform RD | -0.46 | -0.33 | +0.01 | +0.02 | -0.31 | -0.35 | -0.07 | -0.21 |
| p4 + deblock RDO | -0.47 | -0.61 | +0.07 | -0.17 | -0.07 | -0.13 | -0.15 | -0.22 |
| p4 + both | -1.01 | -1.18 | +0.04 | -0.34 | -0.33 | -0.52 | -0.26 | -0.51 |

| main + val (11 clips) | transform RD | both | time (cock, 1 thread, median of 5) |
|---|---|---|---|
| p2 | **-0.65** | -0.93 | +5.5 % / +19.8 % |
| p3 | -0.41 | **-0.71** | -1.0 % / +8.1 % |
| p4 | | **-0.66** | +1.8 % / +11.0 % |

Deblock RDO is not on at preset 2: there it adds -0.28 for another 14 %.
Frame / slice thread identity and every conformance case hold (the decision
only reads the macroblock itself).

Against x264 now (main + val, 11 clips, PSNR BD-rate):

| | x264 medium before / after | x264 veryslow before / after |
|---|---|---|
| p2 | +3.13 / **+2.47** | |
| p3 | -5.27 / **-5.95** | +0.85 / **+0.12** |
| p4 | -5.68 / **-6.30** | +0.40 / **-0.25** |

Tried and not kept (p4, ct + vt clips): iterating the RD refinement while the
vector moves (`AEG_RMIT=4`, -0.07), trying the predicted vector as well
(x264 does; 0.00), a 10 / 20 % penalty on intra candidates in P / B pictures
(aegenc picks intra for 12 % of tree's P macroblocks against x264's 2-6 %,
but it is RD-correct: +0.08 / +0.75), the inter trellis lambda at 1.0x / 0.6x
(+0.17 / +0.04), `--subme 2` (0.00).  Chroma AC level selection is already on
(`AEG_TCHROMA`; the *Rate-distortion level selection* paragraph predates it).

What is left of the gap: references beyond two (tree2 and vtest: x264's ref
16 is worth 2.8 and 1.1-2.5 points; aegenc has two for P and one per B list,
and the second only for 16x16), and the constant-QP tree costing cock2 3 and
tree 1.2 points while it gains 9-11 on vtest.

Speed after it (cockL, one thread, median of 3, `x264cmp.py speed`, same
session for both encoders): aegenc p0-p4 81.0 / 31.0 / 25.3 / 12.5 / 10.8 fps;
x264 ultrafast 1268.4, superfast 411.7, faster 127.4, fast 97.4, medium 78.1,
slower 29.3, veryslow 15.3; x265 medium 33.8.  (`tools/real/fetch.sh` now pins
imageio-ffmpeg 0.6.0: the current release ships FFmpeg 9, not the 7.0.2 / x264
core 164 every number here was measured with.)

## CRF on the long clips: where the gap to x264 veryslow is

Before: at CRF on the long clips (cockL, vtestL, megaL; aegenc `--crf N --aq
0`, x264 `-crf N --tune psnr`, CRF 18-33) p4 was +10.69 % against x264
veryslow and +0.22 against medium.  `tools/x264abl.py` now takes `--mode crf`
(and `AEGENC_ID=<id>` reuses the cache of a build known to give identical
output, e.g. one that only adds an environment knob that is off).  Ablations
(BD-rate against x264 veryslow; PSNR):

| long clips | cockL | vtestL | megaL | mean |
|---|---|---|---|---|
| CRF: p4 | +9.85 | +15.90 | +6.32 | +10.69 |
| CRF: p4, no tree (`--mbtree 0`) | +10.22 | +37.28 | +14.74 | +20.75 |
| CRF: x264 veryslow, `mbtree=0` | +2.06 | +23.30 | +8.99 | +11.45 |
| CRF: x264 veryslow, `rc-lookahead=40` | +0.22 | +2.20 | -0.01 | +0.81 |
| CQP: p4 (its tree on) | +7.59 | -12.01 | -0.91 | -1.78 |
| CQP: p4, no tree | +6.40 | +8.48 | +3.59 | +6.16 |
| CQP: x264 veryslow `ref=2` | -0.41 | +6.71 | +1.86 | +2.72 |
| CQP: x264 veryslow `bframes=3` | -0.14 | +0.71 | +1.00 | +0.52 |
| CQP: x264 veryslow `ref=2:bframes=3` | -0.55 | +8.01 | +3.06 | +3.51 |

(x264 never runs its tree at constant QP; aegenc does, which is why the CQP
comparisons elsewhere in this file favour aegenc on static content.)  So of
the 10.7 points the tree itself is only about 1.3 (it is worth 10.1 to aegenc
and 11.4 to x264); with neither tree aegenc trails by 6.2 at CQP, of which
x264's 16 references are 2.7-3.5 (vtestL, megaL) and cockL's 6.4 is the
coding itself (the cock gap of the section above, larger on the long clip).
x264's longer lookahead (60 against 40) is 0.8; its B offset of 0 under the
tree (pb_factor 1) is wrong for aegenc (`--bqp 0`: +2.6), and without the
tree `--bqp 2` is +3.9 against the default 4.

The tree's strength trades clips against each other (CRF long: 1.3 / 1.6 /
2.0 / 2.5 = +10.69 / +9.36 / +8.48 / +8.65, vtestL gaining 11 points and cockL
losing 4; CQP main + val: 1.6 / 2.0 = +0.18 / +0.70 against 1.3, cock, tree
and their second halves losing 1-3 points each).  One reason the tree
over-propagates on those clips: the lowres intra estimate had only DC, H, V
and plane, where x264's (subme > 1) adds the six directional Intra_8x8 modes,
so a textured block with a direction looked dearer to code anew than it is.
`lookahead.c` now tries them too (`AEG_LRI8=0` restores the four), and with
it the default strength goes from 1.3 to 1.6 (`AEG_MBTS`, tenths):

| BD-rate against x264 veryslow | before | 8x8 modes | + strength 1.6 | (+ strength 2.0) |
|---|---|---|---|---|
| CRF, long clips, p4 | +10.69 | +10.15 | **+8.65** | +7.39 |
| CRF, long clips, p3 | +12.05 | | **+9.88** | |
| CRF, main + val, p4 | +4.70 | | **+4.31** | |
| CQP, main + val, p4 | -0.25 | -0.40 | **-0.37** | -0.02 |

| CQP, main + val, against x264 medium | before | after |
|---|---|---|
| p1 | +7.25 | **+7.09** |
| p2 | +2.47 | **+2.34** |
| p3 | -5.95 | **-6.04** |
| p4 | -6.30 | **-6.40** |

x264's own 2.0 would take another 1.3 points at CRF on the long clips but
gives back 0.35 at constant QP, all of it on cock and tree; 1.6 makes no clip
set worse.  The extra six SATDs per lowres block cost nothing measurable
(p1 cockL 7.84 / 7.69 s, p3 19.90 / 19.77 s user time, before / after); frame
threads stay byte-identical and every conformance case holds.

What remains of the CRF gap, largest first: the core coding of cockL (about 6
points; the cock gap), references beyond two (about 3 on vtestL and megaL;
next), strength beyond 1.6 on static content (about 1.3) and the lookahead
length (0.8).

## More than two P references

`--ref` now goes to 8 (it stopped at 2).  The reference handling was the
part built for two: `refp`/`ref2p` and two motion-compensation contexts,
masks of `& 1` on every `ref_idx`, a one-bit `te(v)` in CAVLC and two bins
in CABAC, and memory-management operations that keep the pyramid's buffer
at three pictures.  With three or more references:

- the buffer is a plain sliding window of `max_refs` = refs (+1 with a B
  pyramid) pictures, no marking operations, and the SPS says so
  (`aeg_sps_max_refs`, shared by the encoder and the header writer);
- list0 is the newest anchor, then every other picture by decreasing POC
  (`set_refs_multi`), sent as a `ref_pic_list_modification` whenever the
  default order (decreasing frame_num) differs; the slice's
  `num_ref_idx_l0_active` drops to the highest index any partition used;
- `ref_idx` is `ue(v)` in CAVLC and the full unary in CABAC (bin 0 on
  ctxIdxInc 0-3 from the neighbours, bin 1 on 4, the rest on 5), and the
  bit estimates follow (`ref_te_bits`);
- references past the second borrow their buffer's sub-pel planes instead
  of owning a context (their `stride` was once left at 0: a small, silent
  drift that only showed when a block chose such a reference);
- the 16x16 candidate on an older reference searches every one of them and
  keeps the best by SATD + bits; the 16x8 / 8x16 halves search all
  references; P_8x8 picks a reference per 8x8 first and then its sub-shape
  on it (`me_candidate_p8x8_mref`, x264's mixed references), and x264's
  early termination applies: when reference 0 is the best whole-macroblock
  match, the partitions look no further back than the neighbours did
  (`AEG_PREF`: 0 off, 1 all references, 2 pruned = default).

B pictures keep one reference per list.

**The level now covers the decoded picture buffer.**  `max_dec_frame_buffering`
was `max_refs + bframes` with B pictures (above level 4.0's four frames at
1080p even with the old defaults) and 1 without them (below
`max_num_ref_frames` with `--ref 2`).  It is now the larger of
`max_num_ref_frames` and the reorder count (x264's rule: non-reference
pictures leave the buffer as they are decoded), the automatic level is the
smallest whose MaxDpbMbs holds it (`pick_level`), and an explicit `--level`
lowers `--ref` until it fits, with a warning, as x264 does.  At 1080p:
p3 stays level 4.0, p4 (`--ref 8`, pyramid) is level 5.0 as x264 veryslow
is.

Constant QP 22-37, the long, main and val clips, BD-rate against p4 with two
references (x264 veryslow with fewer references in the second table; it
uses 16):

| config | cockL | vtestL | megaL | mean of 14 |
|---|---|---|---|---|
| `--ref 3`, partitions on ref 0 | -0.07 | -1.25 | -0.37 | -0.34 |
| `--ref 4`, partitions on ref 0 | -0.15 | -2.17 | -0.51 | -0.55 |
| `--ref 6`, partitions on ref 0 | -0.13 | -3.22 | -0.70 | -0.74 |
| `--ref 4`, partitions search all | -0.21 | -4.20 | -1.26 | -1.11 |
| `--ref 4`, pruned | -0.24 | -3.83 | -1.08 | -1.03 |
| `--ref 6`, pruned | -0.27 | -5.03 | -1.33 | -1.27 |
| `--ref 8`, pruned | -0.30 | -5.59 | -1.77 | -1.45 |

| x264 veryslow, refs | 1 | 2 | 3 | 4 | 6 | (16) |
|---|---|---|---|---|---|---|
| BD-rate against 16 | +1.44 | +1.18 | +0.84 | +0.80 | +0.30 | 0 |

So aegenc gets more out of references 3-8 than x264 does (x264 2 -> 6 is
-0.9, aegenc 2 -> 6 is -1.3), most of it on vtestL (static camera, things
reappearing).  Time, user seconds, single thread, minimum of three
(cockL 120 frames / vtestL 200): p4 9.02 / 5.71, `--ref 4` pruned 9.70 /
5.29, `--ref 8` pruned 9.65 / 5.41; p3 6.71 / 4.84, p3 `--ref 4` pruned
7.89 / 4.93.  More references cost little because most macroblocks skip or
stop at reference 0, and the older references' 16x16 search starts from the
neighbours' vectors.

The defaults are now `--ref 8` at preset 4 and `--ref 4` at preset 3 (with
the pruned partition search; below preset 3 one reference as before).  On
the target of this step, CRF on the long clips against x264 veryslow:

| CRF, long clips, vs x264 veryslow | cockL | vtestL | megaL | mean |
|---|---|---|---|---|
| p4 before (two references) | +9.93 | +10.30 | +5.70 | +8.65 |
| **p4 now (eight)** | +9.76 | +3.27 | +3.66 | **+5.56** |
| p3 before (two) | | | | +9.88 |
| **p3 now (four)** | +11.33 | +6.87 | +4.94 | **+7.71** |

vtestL's gap is down from 10 points to 3; what remains is mostly cockL's
core coding (the cock gap, about 6 points beyond the 3-4 every clip shares
with x264 at CRF), then the tree strength on static content and the
lookahead length.  B pictures still predict from one picture per list.

With `--ref 2` or fewer the slices are byte-identical to before; only the
SPS changed (`max_dec_frame_buffering`, and the level where the buffer
did not fit).  Frame threads and slices stay byte-identical to the serial
encode at every reference count, and every new conformance case
(`ref4_p` ... `bpyr_ref6ft`: CAVLC, slices, GOP, frame threads, weighted
prediction, pyramid) decodes bit-exact.

## Presets 1 and 2

The target: presets 1 and 2 were 3-4 times slower than x264 at the same
quality (p1 +7.09 % BD-rate against x264 medium on the main and validation
clips, about where x264 faster is at +6.14 %; p2 +2.34 %, x264 fast +1.98 %).
Two halves: exact speed-ups first, then a retune of what the two presets
run, measured as exchange rate (BD-rate per per cent of time; going from
p1 to p2 buys about 0.2 % per 1 % of time, so a tool at p1 has to beat that).

**Exact speed-ups.**

* Intra prediction as row copies (`36483fd`): every 4x4 / 8x8 mode builds
  its block from the edge by copying or shifting whole rows instead of
  per-pixel table walks.  Byte-identical; microbenchmark 8x8 276 -> 127 ns
  and 4x4 128 -> 72 ns for all nine modes.
* Motion search without repeated SATDs (`c2ad632`): the partition search
  measured the fractional offset twice (once in me_partition_ref(), again in
  me_partition()), and the P_8x8 / multi-reference candidate paths measured
  the winning offset's SATD a third time; it is now kept from the search
  (`me_last_satd`), and the B partition paths keep theirs per list.
  Byte-identical at presets 1-4, 10-bit, CAVLC, `--ref 4`, no B pictures.

Together 3-5 % (cockL QP 28, single thread: p1 1.24 -> 1.20 s, p2 1.53 ->
1.48, p3 3.58 -> 3.39).  A profile of p1 is flat after that (code_mb 54 %,
analyse_mb 28 %, the largest single function under 7 %), and the cycle
counter puts a P macroblock at about 59 k cycles against x264 faster's 20 k
for everything; the gap is the amount of RD rather than any one kernel.

**What did not pay.**

* Skipping a candidate's RD when its SATD is far behind (`AEG_RDK`): the
  inter prediction plus SATD costs about as much as the CABAC RD it saves.
* x264's early skip (probe the P_Skip residual, stop when it quantises to
  zero).  Before the motion search: -0.12 dB on mega at p1 and -0.14 dB on
  cock at p2 at the same rate, i.e. 2-3 % there; only where the 16x16
  search returns the skip vector (x264 at subme >= 3): neutral, but 3-5 %
  faster even on vtest with 86 % skipped macroblocks -- the skipped
  macroblocks were already cheap and most of them are in B pictures.
* Knobs in either direction at p1/p2 (vs own preset, main + validation):
  `--b-adapt 2` +0.01 / -0.03, `--ref 2` -0.29 / -0.28 for 9-15 % time,
  `--refine-mv 1` -0.78 at p1 for about 15 % (0.05 % per %, below the
  curve), `--partitions 1` +4.4, `--range 8` +3.7 (cock +22.7).

**Level selection on the winner only.**  The trellis (`--trellis`) was
worth -4.35 / -4.14 % at p1 / p2 but cost 26 %, because it ran inside every
candidate of the decision.  x264's `--trellis 1` runs it on the winner only;
aegenc now does the same: the decision compares plainly quantised
candidates (luma and the chroma of `AEG_CDECIDE`), and the winner is coded
again with level selection on (the chroma memo is invalidated for that, or
the final chroma would be the decision's).  `--trellis` takes x264's values:
0 off, 1 winner only, 2 inside every candidate (presets 3-4, unchanged).

| vs own preset, main + validation | p1 | time | p2 | time |
|---|---|---|---|---|
| trellis in every candidate (2) | -4.35 | +26 % | -4.14 | +26 % |
| trellis on the winner (1) | -4.06 | +6 % | -3.65 | +8 % |

**The retune** (asked for and approved): preset 1 gets the winner-only
trellis, `--rc-lookahead 20` (was 10: -0.45 % at no measurable time; 30
adds -0.13 % for up to 3 %), weighted P prediction (-0.16 %, short -1.8 %,
no measurable time) and x264's partition order (P_8x8 first, 16x8 / 8x16
only when it beats 16x16: +0.10 % for -6..-8 % time; at p2 it costs
+0.55 %, so not there).  Preset 2 gets the winner-only trellis.  Presets 0,
3 and 4 are byte-identical to before.

| BD-rate vs x264 medium, main + validation (QP) | before | **now** |
|---|---|---|
| preset 1 | +7.09 | **+2.44** |
| preset 2 | +2.34 | **-1.31** |
| x264 faster / fast | +6.14 / +1.98 | |

| CRF, long clips, vs the old preset 1 | cockL | vtestL | megaL | mean |
|---|---|---|---|---|
| preset 1 now | -2.39 | -9.27 | -1.94 | -4.53 |
| preset 2 before | -8.43 | -8.94 | -5.63 | -7.66 |
| preset 2 now | -10.40 | -12.59 | -6.62 | -9.87 |

Time, user seconds, single thread, interleaved minimum of five (cockL 60
pictures / vtestL 100, QP 28): preset 1 1.28 -> 1.30 / 0.81 -> 0.81,
preset 2 1.54 -> 1.65 / 0.98 -> 1.07.  So preset 1 now codes at the old
preset 2's quality (+2.44 against +2.34) in 17-20 % less time, and about
where x264 fast is; at that quality aegenc is about 3.4 times x264's time
(it was 3.7).  Preset 2 now beats x264 medium.  The remaining gap is the
amount of RD per macroblock (every candidate of the decision priced by the
CABAC coder, where x264 fast decides by SATD and prices only the winner at
subme 6); the cockatoo and tree clips, where p1/p2 are still 6-17 points
behind x264, are the next place to look.

Frame threads and slices stay byte-identical to the serial encode, every
conformance case (new: `trellis_win` at p2, `trellis_win8` at p4 with the
8x8 transform) decodes bit-exact, and 10-bit streams decode to the
reconstruction.  The `--vbv-*`-alone test pins `--trellis 0`: on its
synthetic picture level selection takes the QP down to 12-13 and the
one-pass rate control is still walking there at picture 120 (10 % under);
on real content ABR and VBV land as before (cockL 800 kb/s: 882 / 861
against 886 / 863).

## The cockatoo / tree gap at presets 1 and 2

After the retune, p1 / p2 were still +17.3 / +5.6 % behind x264 medium on
cock, +11.8 / +5.0 on cock2, +5.4 / +5.3 on tree and +15.7 / +13.9 on tree2.
Adding each preset-3 tool to p2 alone on those four clips (BD-rate vs p2):

| added to p2 | cock | cock2 | tree | tree2 | mean | cpu |
|---|---|---|---|---|---|---|
| `--range 32` (was 24) | -3.14 | -1.12 | +0.00 | -0.01 | -1.07 | +-0 |
| exact-RD Intra_4x4/8x8 modes (`AEG_IRD=1`) | -1.22 | -2.25 | -1.18 | -0.18 | -1.21 | +30 % |
| `--ref 4` | -0.36 | -0.32 | +0.04 | -3.86 | -1.13 | +10 % |
| `--refine-mv 2` | -1.19 | -1.06 | -0.00 | -0.53 | -0.69 | +5 % |
| `--trellis 2` | -0.58 | -0.75 | -0.38 | -0.46 | -0.54 | +10 % |
| DBRDO | -0.26 | -0.86 | -0.02 | -0.13 | -0.32 | |
| BIDIR + BIDIRP | -0.59 | -0.14 | +0.00 | -0.54 | -0.32 | |
| `--b-adapt 2` | +0.95 | +2.83 | +0.16 | -2.73 | +0.30 | |

and on p1: `--range 24` (was 16) -5.61 / -2.98 on cock / cock2, chroma in
the decision -0.23, T8RD -0.49, `AEG_PSUB=2` -1.00, exact-RD intra -1.42.

**The search range.**  The motion search bounds every vector, predictors
included, by a box of +-range around the zero vector (x264's merange is a
window around the prediction).  With three B pictures a P vector spans four
picture intervals, and cockatoo's motion does not fit 16 or 24 samples.
The search walks patterns, so the box size costs nothing by itself:

| range, vs own preset (main + validation) | cock | cock2 | mean |
|---|---|---|---|
| p0 8 -> 48 | -22.28 | -8.51 | -3.05 |
| p1 16 -> 24 / 32 / 48 | -5.61 / -8.42 / -8.95 | -2.98 / -3.77 / -4.00 | -0.80 / -1.12 / -1.20 |
| p2 24 -> 32 / 48 | -3.14 / -3.59 | -1.12 / -1.61 | -0.39 / -0.47 |
| p3 32 -> 48 | -0.42 | -0.66 | -0.11 |

Time (interleaved minimum, single thread): cockL p1 1.57 -> 1.51 s, p2
1.92 -> 1.97, p3 4.03 -> 4.10, p0 (120 pictures) 1.18 -> 1.13; vtestL p2
1.17 / 1.17 (seven runs); p0 with two frame threads 1.37-1.38 s wall either
way (the wavefront waits range + 24 rows below the block, and from preset 1
the lowres pre-search already waits 63).  The default is now 48 at every
preset (the padding allows 56; 48 is where the gain stops).  Against x264
medium on the main and validation clips: p1 +2.44 -> **+0.99 %** (cock
+17.3 -> +6.4), p2 -1.31 -> **-1.83 %**.

**Not adopted** (exchange rate below the 0.2 % per % of time between p1 and
p2):

* Exact-RD intra modes: p1 -1.37 % for about +36 %, p2 -1.32 % for +24 %;
  with the SATD pre-selection of the best two modes (`AEG_I4K/I8K=2`) -0.99
  / -1.02 % for +16 %.
* Three references with a gate that only lets the older reference into the
  RD pass when its SATD + vector + index cost beats reference 0's: p1 -0.72
  %, p2 -0.85 % (tree2 -3.0 / -3.3), for +22 % (cockL) / +6 % (vtestL) at
  p1, +11 % / +10 % at p2; partitions on reference 0 only (`AEG_PREF=0`)
  +14 % / +6 %.  Two references: -0.31 / -0.22 %.  The gate at preset 3:
  +0.06 % and no time saved.

**What the tree clips need.**  x264 medium's own ablation on the four
clips (each tool off, BD-rate vs medium, QP): cock / cock2 lose most without
sub-pel refinement (subme 2: +18 / +18 %), partitions (+11 / +11) and the
8x8 transform (+6 / +10) -- all of which aegenc has -- while tree2 spreads
over B pictures (+6.9), one reference (+4.4), subme (+4.0), partitions (+3.5)
and trellis (+3.1); weightp, fast-pskip and the macroblock tree (at constant
QP) are flat.  Without B pictures on either side p2 is still +9.65 % behind
on tree2 (p3 +1.29 %): it is P-picture coding, and the references are about
four of those points (tree2 is foliage with a hand moving in front of it,
uncovering what an older picture still has).  `--subme 2` is identical to 1
in the encoder.

**x264's sub-pel threshold for the references** (`p_halfpel_thresh`): the
16x16 searches of a macroblock's references run in order, and one whose
integer-pel cost is more than 8/7 of the best so far stops before its
sub-pel stage and adds no candidate (aeg_me_block_t, `AEG_HPT`).  Preset 3
+0.07 %, preset 4 +0.04 % on the main and validation clips for -3 / -4 % of
the time: on by default (it only acts with more than one reference).  It
does not make more references pay below preset 3: p1 `--ref 3` is still
+11-17 % of time, p2 `--ref 3` (partitions on reference 0) -0.40 %.  The
remaining cost is the older references' integer search and their lowres
pre-search, about 7 k cycles per macroblock and reference.

**Preset 1 gets the partition sub-pel search and T8RD** (they were preset 2
and up; re-measured with the wider range): `AEG_PSUB=2` -1.33 %, T8RD -0.70
%, both -2.00 % on the main and validation clips, for +2 % (cockL) / +11 %
(vtestL) -- 0.3 % per % of time.  Chroma in the decision on top: -0.25 %
for another +10 %, not taken.

**Result** (this step against `4962b79`; the old presets reproduced
byte-identically with `--range 16 AEG_PSUB=0 AEG_T8RD=0` / `--range 24`):

| BD-rate vs x264 medium, main + validation (QP) | cock | cock2 | tree | tree2 | mean |
|---|---|---|---|---|---|
| p1 before | +17.34 | +11.75 | +5.43 | +15.72 | +2.44 |
| **p1 now** | +2.94 | +5.06 | +5.18 | +14.42 | **-1.01** |
| p2 before | +5.64 | +5.03 | +5.28 | +13.88 | -1.31 |
| **p2 now** | +1.63 | +3.24 | +5.28 | +13.86 | **-1.83** |
| x264 fast | +3.69 | +2.97 | +0.09 | +1.64 | +1.98 |

CRF on the long clips, against the old preset 1: p1 -4.89 %, p2 -6.06 %
(old p2 -5.53 %).  Time, single thread, interleaved minimum of five (cockL
60 / vtestL 100 pictures, QP 28): p1 1.61 -> 1.67 / 0.99 -> 1.15 s, p2 2.02
-> 2.04 / 1.31 -> 1.33.  So preset 1 now codes better than x264 medium did
and better than the old preset 2, 12-17 % faster than that one.

Against x264 at the same settings (ffmpeg's libx264, one thread, QP 28,
`--tune psnr`): medium 0.56 / 0.21 s, fast 0.51 / 0.20 s.  At x264
medium's quality aegenc preset 1 takes 3.0 times x264's time on cockL and
5.3 times on vtestL, where 86 % of the macroblocks are skipped: x264 is
cheapest exactly where little changes, and that is where to look next.
Trees (tree +5.2, tree2 +14.4 % at p1) stay behind on the references.

## Static content at presets 1 and 2: where the time goes

At x264 medium's quality p1 took 3.0x x264 medium's time on cockL and 5.3x
on vtestL (static, camera-noisy, 84-86 % skip macroblocks).  The cycle
counters (`AEG_CMTIME=1`, vtestL, p1, QP 28, Mcyc) put the B pictures at
about 1.1 G of the 2.2 G: analysis 455, the inter candidates' RD 450, the
re-encodes after the decision (T8RD + the trellis pass, the section labelled
`refine_mv`) 190, chroma + skip check 85.  P pictures: analysis 210, inter RD
155, re-encodes 148, Intra_16x16 RD 114.

**Exact (byte-identical at every preset, 10-bit, slices, frame threads):**

* The trellis pass re-encoded the winner's luma even when nothing in it
  quantised to a level; level selection only shrinks or removes existing
  levels, so that re-encode cannot change anything and is skipped (it must
  still stamp the macroblock QP: a residual-free winner may carry another).
* With the partition sub-pel search on (p1 since `dfff9e8`), the B 16x8 /
  8x16 (and p4's B_8x8) search already returns the SATD of the vector it
  picks; it was predicted and measured a second time.
* The inter residual gather runs row-wise, so it vectorises.

vtestL p1 1.02 -> 0.86 s (-16 %), p2 1.09 -> 1.00 (-8 %); cockL p1 1.61 ->
1.45 (-10 %), p2 1.80 -> 1.75 (interleaved minimum, single thread).  Against
x264 medium that is about 4.4x on vtestL and 2.7x on cockL.

**x264's B shortcuts, measured:**

* Skip by SSD (x264 with RD: the direct prediction's SSD below six bits'
  cost means B_SKIP before any search) almost never fires on vtest: at QP 30
  six bits are an SSD of about 330 over 384 samples, under one per sample,
  and camera noise is far above that.  Not kept.
* RD only for candidates within G/16 of the best search cost
  (`mb_analyse_b_rd`, G = 17): noisy static content puts every candidate's
  SATD close together (5.3 candidates per B macroblock, 0.3 dropped), so
  p1 / p2 gain nothing.  At p3 / p4 it pays with G = 20 (-0.00 / -0.01 % BD
  for -2.8 / -5.2 % time; G = 17 +0.08 / +0.08 %) and is on there
  (`AEG_BRDG`).
* Skipping the joint bi-prediction refinement when the plain average loses
  to one list: the average nearly always wins on noisy content, so it never
  triggers.

**The B partitions are a third of a B macroblock** on vtestL p1: without
them analysis drops 414 -> 170 Mcyc and the RD pass 436 -> 308 (two
candidates fewer), but they win 1214 of the 2200 coded B macroblocks and
removing them costs 3.5 % of the bits.  The gate on the best 16x16 cost
(96 lambda) does not hold up when raised (main + validation, p1): 128
+0.09 % for -3 %, 160 +0.19 %, 200 +0.33 % (cock +1.10), 300 +0.61 %.  B
partitions without their own sub-pel search (`AEG_PSUBB=0`) +1.8 % bits on
vtestL.  What is left is per-operation cost (x264's RD and searches are
hand-written assembly) and the number of candidates priced.

### Cheaper candidates: exact work, then the B partitions' searches

**Exact (byte-identical at every preset, 10-bit, slices, frame threads,
CAVLC, temporal direct):**

* The 8x8-transform choice (`use_t8`) measured the four 8x8 SA8Ds one call
  each; one 16x16 SA8D (AVX2 does two blocks per pass) gives the same sum.
* A 4x4 / 8x8 block without levels costs only its coded_block_flag in the
  CABAC price, and reconstructs to exactly its prediction (copied, no
  inverse transform).
* B direct predicts the whole macroblock in one motion compensation when the
  four 8x8 groups carry the same vectors.
* **Luma memo** (`AEG_LMEMO`, on): candidates that predict the same pixels
  code the same luma -- the residual coding reads only the source, the
  prediction, the QP, whether the 8x8 transform may (or must) be used and the
  trellis switch.  Static content's partitions mostly find the 16x16 vector
  and the direct vector is often the searched one: 37 % of the inter luma
  codings of vtestL repeat an earlier candidate of the same macroblock (8 % on
  cockatoo).  The result and the reconstruction are replayed; the header bits
  are priced anew.  vtestL p1 0.82 -> 0.75 s, p3 2.16 -> 1.84 s.

**The B partitions' sub-pel search, x264-style (BD-rate, main + validation):**

* x264's 4-point diamond instead of the 8-point square for the B 16x8 /
  8x16 / 8x8 searches (`AEG_SPDIA`, mask 1): p1 / p2 +0.04 %, p3 +0.06 %,
  p4 +0.04 %; on at presets 1 and 2 (B analysis 430 -> 389 Mcyc).  Also on the
  P partitions or the 16x16 searches: +0.33 % / +0.58 % at p1 -- out.
* Seeding the partitions' sub-pel stage with the macroblock's own quarter-pel
  vector (`AEG_BPFS`): when the integer search lands where that vector rounds
  to (92 % of the time on vtestL), skip the half-pel pass and refine at
  quarter pel only.  p1 +0.02 %, p2 +0.02 % for B analysis 367 -> 321 Mcyc
  (-2.7 % time); on at presets 1 and 2.  The same for the P partitions
  (`AEG_PPFS`) is -1.4 % for +0.09 % at p1 and +0.11 % together with the B
  one at p2: off.  At p3 / p4 neither shows in the time.

Against `96fa945` (interleaved minimum, single thread, QP 28): vtestL p1
-15 %, p2 -15 %, p3 -9 %; cockL p1 -7 %, p2 -6 %, p3 -6 %.  The profile
is now flat -- CABAC residual pricing 7 %, the SATD kernels 5 % each, the
UMH search 3 % -- so what remains is the number of candidates priced, not
one hot spot.

## Field pictures (PAFF)

`--field` (`params.field_coding` 1; implies `--tff` unless `--bff`) codes
every frame as two field pictures, first field in display order first:

* Each field is an access unit (delimiter, picture timing SEI with
  `pic_struct` 1 / 2, slices with `field_pic_flag` 1 and `bottom_field_flag`),
  both with the frame's `frame_num` and `nal_ref_idc`; POC type 0 with
  `delta_pic_order_cnt_bottom` absent, the fields' counts 2n and 2n+1.  The
  second field of an I frame is a P field predicting from the first (the
  first of an IDR frame is the IDR picture).
* The field is every other row of the frame's buffers (offset by its parity,
  stride doubled), so the lookahead, rate control, macroblock tree and
  packets stay per frame; the tree's and AQ's offsets average the two frame
  rows a field row covers.
* Reference lists by 8.2.4.2.2 / .2.4 / .2.5: frames by `frame_num` (P) or
  POC (B), then their fields alternately starting with the current parity.
  A P field uses `2 * --ref` fields (the second field of a frame reads its
  first), a B field one per list; marking is the sliding window only, with
  `max_num_ref_frames` one above the frame-coding value.
* Field coding tools: the field scans (4x4 and 8x8, CAVLC's interleave), the
  CABAC field contexts (significance / last of all categories, the field 8x8
  significance map), the chroma vector offset between fields of opposite
  parity (8.4.1.4), the deblocking filter's field rules (bS 3 on horizontal
  intra edges, vertical vector limit 2 quarter-rows), half-pel planes and
  padding built per field.
* Turned off with field pictures: frame threads, weighted prediction, the B
  pyramid, temporal direct, chroma ME, `--nal-hrd`.

Every stream decodes bit-exactly in ffmpeg (tff and bff, I/P/B, CABAC and
CAVLC, slices, 1..8 refs, CQP / CRF / ABR / VBV, mid-stream and scene-cut
IDRs, 10-bit; `tests/run_tests.sh` cases `fld_*`, `h10_field`), and ffmpeg
reports the field order on every frame.

PSNR BD-rate of `--field` against frame coding (`--tff`), preset 3, QP
22..37, 60 frames woven from progressive clips by ffmpeg's `interlace`
filter: cockatoo (fast motion) **-47.0 %**, vtest (mostly static) +4.3 %.
Which one wins depends on the motion: hence the per-picture choice below.
CPU time per frame is lower (fewer intra macroblocks); wall time is higher
for now because frame threads are off.

Fixed on the way: with B pictures the slice header's POC lsb was the
absolute display index, so after a mid-stream IDR the IDR's POC was not 0
(decoders rebase it; ffmpeg's output was identical) -- it is now counted
from the last IDR.  Only multi-GOP streams with B pictures change.

### Frame or field per picture (`--paff`, `params.field_coding` 2)

Each I/P frame is coded as a frame or as two fields; the B frames before it
follow it, so a B picture's co-located picture (direct mode) always has its
structure (the mixed cases of Table 8-6 never arise).  What the switch
needs:

* Frame pictures count their POC in fields: the top field's lsb, and
  `delta_pic_order_cnt_bottom` +1 (tff) / -1 (bff) with the PPS's
  `bottom_field_pic_order_in_frame_present_flag` -- the frame's fields get
  2n and 2n+1 as a field-coded frame's do.
* A reference read in the other structure is re-padded (a frame's edge rows
  replicated, or each field's) and its half-sample planes rebuilt (6-tap
  across the frame's rows or within each field) before the picture
  (`slot_restructure`); the samples themselves are the same.  The motion
  compensation contexts get the frame's strides back after a field picture.
* A field-coded reference frame gets its lowres plane built, for the frame
  pictures' lowres pre-search.
* Frame pictures keep weighted prediction and chroma ME (field pictures do
  without); the B pyramid stays off for both.

**The choice** (`fld_decide`): against the newest anchor's reconstruction,
a 16x16 integer diamond search (steps 8, 4, 2, 1 from the left, above and
zero vectors, inside the picture) for the frame's blocks and for each
field's against both reference fields -- the second field also against the
first, half a frame period away.  The cost is the SATD at the vector found,
capped by the block's SATD against its own mean (intra, roughly); the
frame's prediction first takes out the brightness change weighted
prediction would (x264's estimate: gain = ratio of standard deviations,
offset from the means), the fields' does not.  Fields when their total is
below 88 % of the frame's (`AEG_FLDBIAS`, default 12).  Frames without a
reference fall back to a comb count (rows resembling their field's
neighbours more than their own).  On the way:

* A comb count alone could not separate a fade (brightness changes between
  the fields; fields without weighted prediction, +105 % when chosen) from
  moderate motion (short, -47 % with fields): every threshold failed one.
* SAD instead of SATD put every clip below 93 % -- a field's residual is
  less correlated vertically, which only the transform sees.
* A one-step diamond from zero underestimated the frame at an anchor
  distance of 4 (B-adapt at preset 3): the fade's anchor went to fields at
  +60 % bits for it.

PSNR BD-rate against frame coding (`--tff`), 40 woven frames, QP 22..37:

| clip | p2 `--field` | p2 `--paff` | p3 `--field` | p3 `--paff` |
|---|---|---|---|---|
| cockatoo | -68.4 % | -68.4 % | | |
| short | -46.7 % | -46.7 % | -42.4 % | -41.2 % |
| megamind | -1.7 % | -2.4 % | -2.7 % | -1.2 % |
| fade | +105.5 % | +0.0 % | +43.8 % | -0.6 % |
| vtest | +12.4 % | +0.6 % | +15.9 % | +2.9 % |
| tree | +8.9 % | -0.4 % | +6.9 % | +3.1 % |

At preset 3 what `--paff` loses on vtest and tree is the B pyramid: frame
coding without it is +2.9 % / +3.7 % (and `--paff` forced to frames the
same), so the choices themselves gain 0.0..0.7 %.  Field pictures as
reference B pictures (marking by field picture numbers) would give it back.

Every combination decodes bit-exactly in ffmpeg: forced switch patterns
(`AEG_FLDFORCE`, a 0/1 string per display index) with and without B,
CABAC / CAVLC, 1..8 refs, bff, weighted prediction on a fade, slices, CRF /
VBV, presets 0..4, 10-bit (`tests/run_tests.sh` `paff_*`, `h10_paff`).

## Macroblock-adaptive frame/field (MBAFF, `--mbaff`, `params.field_coding` 3)

Built in stages. Each stage decodes bit-exactly in ffmpeg before the next
one starts:

1. **Pair order (done).** The SPS has `mb_adaptive_frame_field_flag`. Slices
   start on pair boundaries (`first_mb_in_slice` counts pairs), and every
   pass walks the macroblocks pair by pair: top, bottom, then the next
   pair. Field pairs are not used yet, but the frame-pair rules are already
   in place:
   * `mb_field_decoding_flag` is written for a top macroblock, and for a
     bottom one whose top was skipped. Its CABAC context comes from the
     left and above pairs. A skipped pair infers its flag from those
     neighbours.
   * `end_of_slice_flag` is written only after a bottom macroblock.
   * A bottom macroblock has no above-right neighbour.
   * Motion vectors are predicted through the pair-aware neighbour lookup.
   * Deblocking runs pair by pair.
2. **Field pairs in I slices (done).** A field macroblock sees only its
   field's lines (`mbf_enter`: the top macroblock gets the even lines of
   the pair, the bottom one the odd lines, at twice the stride). It uses
   the field scans and the field CABAC contexts for significance and last.
   Each neighbour (A/B/C/D, 4x4 block, sample row) is derived through
   Table 6-4 (`nb_left`, `nb_mbB`, `nb_top_ok`) for:
   * nC and coded_block_flag;
   * the intra-mode predictors;
   * the CABAC cbp, chroma-mode, transform-8x8 and mb_type contexts;
   * intra availability.

   For a frame macroblock beside a field pair, the left 4x4 blocks
   alternate between the two left macroblocks.

   Intra prediction samples always come from the current view, one row or
   column outside the macroblock.

   Deblocking (`aeg_deblock_mb_mbaff`) filters a field macroblock in its
   own view. Two edges get special treatment:
   * **Left edge beside the other kind of pair.** It gets eight bS values
     and one averaged QP per left macroblock. A field macroblock's lines
     0-7 meet the top left macroblock and lines 8-15 the bottom one. A
     frame macroblock's even lines meet the top one and its odd lines the
     bottom one.
   * **Top edge of a frame top macroblock under a field pair.** It is
     filtered twice, once per field. The even lines go against the top
     field macroblock and the odd lines against the bottom one. bS is at
     most 3.

   Elsewhere, a frame/field edge gets bS ≥ 1, intra edges get bS 4 only
   between two frame macroblocks, and the vertical motion-vector limit is
   2 inside field macroblocks.

   Field pairs are only chosen when `AEG_MBFORCE` sets a test pattern:
   `all`, or a number that seeds a pseudo-random pattern per pair and
   picture. This applies to I slices only.
3. P slices (to do): field reference lists (refIdx >> 1 with parity),
   scaled neighbour motion vectors, P_Skip and the mvd contexts across
   frame/field pairs.
4. B slices (to do): spatial and temporal direct with the co-located pair
   mapping of Table 8-6.
5. The frame/field decision per pair (to do).

Under the forced patterns, ffmpeg decodes bit-exactly in every
combination tested (`tests/run_tests.sh` `mbaff_*`, `h10_mbaff*`):
* entropy coding: CABAC and CAVLC;
* field pairs: all of them, or seeded mixed patterns;
* presets 0..4;
* slices and slice threads;
* tff and bff;
* rate control: CQP, CRF and VBV;
* deblocking: offsets, QP up to 45, and off;
* 10-bit;
* P/B pictures referencing I pictures with field pairs;
* real clips (short, vtest, megamind, cockatoo, fade).

Progressive streams are unchanged, and MBAFF without forcing is
byte-identical to stage 1.
