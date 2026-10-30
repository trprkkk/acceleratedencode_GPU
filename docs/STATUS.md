# Status and engineering notes

Date: 2026-10-26.  Branch: `arena/589d1e45-acceleratedencode-gpu`.

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

Two things deliberately stay per block:

* Intra_16x16, because its DC coefficient is replaced by the 4x4 Hadamard of
  the sixteen DCs, and the batch API has no per-block override;
* chroma, for the same reason (the 2x2 chroma DC Hadamard).

## Rate-distortion batching (provisional)

The per-candidate batch above is one launch pair per candidate.  The next step is
to put *every* inter candidate of a macroblock in one launch pair:

```
for each candidate k:                 /* 1. produce */
    predict candidate k (plane MC)    -> pred_l
    residual = src - pred_l           -> rd_res[16 blocks * k]
    save pred_l                       -> rd_pred[k]
tq_batch(rd_res, 16 * n)              /* one launch for all candidates */
iq_idct_batch(rd_lev, 16 * n)         /* one launch for all candidates */
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

* **Verified here**: with the CPU reference ops behind the batch entry points,
  the stream is byte-identical to the serial candidate loop -- the conformance
  cases `rd_batch` and `rd_batch4`, the whole 84-case sweep run with
  `AEG_RD_BATCH=1`, and six clip/preset/QP comparisons.  The arithmetic cannot
  differ, because `luma_inter_finish()` is the one implementation both paths
  use; only the order in which the kernels are called changes.
* **Not verifiable here**: whether two launches per macroblock actually beat
  thirty-two calls.  On the CPU the forced path costs ~10% of the transform
  stage (the prediction add, the clip and the SSE are scalar in the batched
  form, where the serial path uses the SSE2 `aeg_idct_add_block`), so it is
  correctly kept off there.  A device turns those thirty-two calls into two
  launches of 64 blocks each; whether that wins depends on the launch cost the
  dispatcher calibrates, which is what `aeg_backend_calibrate()` exists for.
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
`-s` override takes effect from the first `aeg_dispatch_update()` on.  What is
*not* implemented is the dispatcher discovering the coefficient stages by
itself: it only ever measures the path it already chose, so a stage that starts
on the CPU records no device time and never flips.  Probing both paths for the
first few pictures (running the loser's work and throwing it away) is the next
dispatcher change, and it is what would make the placement automatic on
hardware.

## Verification


```sh
make -j2
FFMPEG=/path/to/ffmpeg make test      # tables + cavlc + cabac + 76 clip cases
```

`make test` runs seven things:

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
7. `tests/run_tests.sh` — sixty-nine cases (QP 12/23/26/40, presets 0-4,
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
   a different encoder rather than an accelerator of this one.  The
   dispatcher's
   calibration has never run against real hardware in this environment (there
   is no GPU here), so the crossover points are still the CPU-model ones; the
   dispatcher also only ever measures the path it chose, so a stage is switched
   by its forced override (`-s stage=backend`) or, on hardware, after both
   paths have been probed.
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
5. Threads beyond slices: a per-thread device context (so that slice threads
   and GPU stages can run at the same time -- today a picture with any stage on
   a device is coded by one thread), and frame-level threading (code picture
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
to submit to from several threads, so the slice loop stays serial while any
stage is dispatched to a non-CPU backend (`slice_thread_count()`); per-thread
device contexts are the follow-up that lifts that restriction.

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
into the next picture's QP), one slice, the CPU backend.  ThreadSanitizer is
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
`--weightp 0` (the ceiling).  The single-thread time is unchanged (the
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
macroblock's own syntax are all in the comparison.  Preset 3 tries the four
axial neighbours, preset 4 all eight; `--refine-mv 0|1|2` overrides it, and
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
