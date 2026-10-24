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
| P_8x8 (four quadrants moving apart, all four `sub_mb_type` shapes) | 320x192, 96x64, 48x32 | 12, 20, 26, 34 | 2 | 6 … 8 |
| IDR pictures after P pictures (`-g 3`, `-g 1`) | 320x192 | 20, 26, 34 | 2 | 6 … 10 |
| slices (`--slices 2 … 12`, including uneven row counts) | 320x192, 320x208, 64x64 | 22 … 26 | 0, 1, 4 | 4 … 8 |
| 8x8 transform + Intra_8x8 (every clip above re-run with it on; 30-case I-picture matrix over four ramp/noise clips) | 32x16 … 320x192 | 12 … 40 | 0 … 4 | 1 … 16 |

with and without the in-loop deblocking filter.  `make test` asserts this and
fails (non-zero exit) on any mismatch; it runs forty-seven encoded clip cases
frame by frame, checks that the ABR controller hits its target, plus a sweep of
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
  coder*)
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
FFMPEG=/path/to/ffmpeg make test      # tables + cavlc + cabac + 47 clip cases
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
7. `tests/run_tests.sh` — forty-seven cases (QP 12/23/26/40, presets 0 and 4,
   deblocking on/off, 64x64 … 320x208, forced CPU backend, 16x8/8x16/P_8x8
   partitioning with all four `sub_mb_type` shapes, both the integer and the
   quarter-pel motion search, the forced whole-frame search seed *and* the
   forced whole-frame fractional refinement, all-intra and 3-frame GOPs, the RD
   Intra_4x4 decision of the slow presets, P_Skip, ABR, AQ, and AQ over
   P_Skip runs at two QPs -- all of them with the 8x8 transform and CABAC
   enabled, which are the defaults)
   encoded, decoded with ffmpeg and compared **frame by frame**, then an ABR
   accuracy check: the encoder's `--dump-recon` must be bit-identical to the decoder's
   output.  Any mismatch fails the script; a decoder that logs an error also
   fails it.  `tests/genyuv.c` takes a mode argument so that the partitioning
   cases really move the two halves of the picture in opposite directions —
   the default clip pans as one block and can never make a partition pay.

`reconstruction vs decode` must be `max abs diff 0`.

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

## Next

1. Quality/RD: per-block rate-distortion level selection (trellis, or at least
   the last non-zero coefficient).  The rate term is now the coder's own price,
   so the level decision has the model it needs.  Measured against x264 (which has trellis=1
   by default) on the panning clip at QP 24: trellis off 60 047 B / 45.95 dB,
   trellis 1 61 495 B / 46.26 dB, trellis 2 61 106 B / 46.24 dB -- i.e. it buys
   quality with bits at a fixed QP and is a wash in PSNR-per-bit here, which is
   why it has not been a priority.  The bigger item behind it is the 8x8 DCT
   (see 3).  The sub-partition *shape* choice
   is currently a SAD + 8-bit approximation; re-deciding it inside the RD pass
   would recover a little more (it picks a slightly too fine shape on some
   macroblocks).
2. GPU: every stage that can be batched is now wired to its kernel.  The plane
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
3. Encoder features that need the profile to change: none left -- CABAC and the
   8x8 transform are both in (main profile and High profile respectively), and
   the encoder emits Main, High or Baseline depending on which tools a stream
   uses.  The decision pass now prices a candidate with the coder that emits the
   stream (see *Pricing the decision pass with the coder*), which was the
   prerequisite for trellis-style level optimisation: with a real per-bin price
   the level choice can be a run over the coefficients of a block, and the 3.6
   points the trellis row of the table above is worth are the largest single
   item left on the quality side.  MB-tree was
   evaluated for the rate controller and is *not* in: at a fixed rate on the
   two-complexity clip it undershot the target by 17% and gave back 0.7 dB, and
   this environment has no real content to show the gain it is known for, so
   two-pass allocation (which is measurable here, +3.8 dB) is the mechanism that
   was kept.
4. Threads beyond slices: a per-thread device context (so that slice threads
   and GPU stages can run at the same time -- today a picture with any stage on
   a device is coded by one thread), and frame-level threading (code picture
   N+1's decision pass while picture N is being written), which needs the
   reconstruct-then-write split that the loop already has, plus one reference
   picture per thread.  Rate control is picture-level only (no macroblock-tree
   lookahead), and the VBV model tracks the buffer level between pictures
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
x264, and that is a bigger piece of work than this shortcut.

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

**With CABAC in, the reference to beat is x264's default preset, not its
baseline profile.**  Measured on the genyuv mode-4 clip (320x240, 8 frames),
BD-rate over QP 22/25/28/31/34/37, our stream against `libx264 -preset medium
-qp N`:

| x264 medium with | ours, bits at matched PSNR |
|---|---|
| `cabac=0:8x8dct=0:trellis=0:aq-mode=0` | 47.7% fewer |
| `cabac=1:8x8dct=1:trellis=0:aq-mode=0` | 8.7% more |
| `cabac=1:8x8dct=1:trellis=1:aq-mode=0` | 12.3% more |
| (preset defaults: + AQ, psy-rd) | 12.3% more |

So CABAC and the 8x8 transform together are what used to be the whole gap
against x264 *baseline* (1-8%, table below, which predates CABAC); against the
*default* preset what is left is 12% on this clip -- 3.6 points of it trellis
quantisation, the rest the motion search and the rate-distortion refinement of
it.  On the two-quadrant clip (mode 2, the one that exists to make 16x8 and
8x16 partitions pay) the gap is 24%, i.e. partitioning is where a faster search
and a real sub-partition RD pass would buy the most.

What the decision pass prices a candidate *with* matters more than any of the
knobs tried before it.  Until this round the rate term was CAVLC-priced (the
estimate `aeg_cavlc_block_cost()` returns), which over-prices it by the 0.65
the CABAC coder actually spends, and a single global scale on lambda was tried
as a stand-in for the real model: it gained 0.9-0.5 points on two clips and lost
1.6 on the third (skip-heavy), so the bias is per syntax element and a single
factor cannot fix it.  The rate term is now the coder's own price -- see
*Pricing the decision pass with the coder* under *CABAC* -- and the three clips
below are the measurement of it:

| clip | before | after |
|---|---|---|
| mode 4, 320x240, 8 frames | 12.9% more | **12.3% more** |
| mode 2, 320x192, 30 frames | 28.6% more | **24.5% more** |
| mode 3 (still, skip-heavy), 320x192, 30 frames | 21.7% more | **6.2% more** |

The clip that gains most is the one whose decisions are dominated by skip and
by near-zero residuals, which is exactly where the CAVLC estimate is furthest
off: a skipped macroblock costs it one flag plus the estimate's guess, while the
coder charges one context-coded bin whose probability has adapted to the
neighbourhood, so the skip/inter boundary is finally in the right place.  The
gain costs 15% of the encode time (320x192, 30 frames: 92 -> 78 fps), which is
the price of running `write_mb_cabac()` once per candidate instead of once per
macroblock; pruning the candidates the estimate dislikes was measured and is
*not* in (it recovered none of the time -- the bin walk is small next to the
transform and the search -- and cost 4% of the bits on the clip where the
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
partitions are worth coding):

| encoder | fps | bytes | PSNR |
|---|---|---|---|
| ours, preset 0, before / after the coder-priced decision pass | 584 / 537 | 118 680 / 118 680 | 43.74 / 43.74 (YUV) |
| ours, preset 2 | 86 / 73 | 49 335 / 47 687 | 46.33 / 46.54 (YUV) |
| ours, preset 4 | 82 / 68 | 49 537 / 47 665 | 46.29 / 46.55 (YUV) |
| x264 baseline (no asm), earlier round | 175 | 78 664 | 45.16 (Y) |
| libx264 baseline, this machine, asm | 385 | 78 692 | 46.30 (YUV) |
| libx264 medium, this machine, asm | 395 | 34 612 | 46.39 (YUV) |

The three x264 rows are on the same clip and QP but from different builds and
PSNR conventions (the first row predates this table's rewrite and averages Y
only), so the honest reading is: against a *no-asm* x264 this encoder is within
1.8x, against a normal libx264 with assembly it is 4-5x slower, and it now
spends 13-18% more time than it did before the decision pass priced candidates
with the coder (preset 0 is unaffected -- it takes the SAD path and codes each
macroblock once).  The bits it buys with that time are 3-5%, and 15 points of
BD-rate on the clip whose decisions are closest (see *Quality against a
reference CPU encoder*).  Absolute numbers move a few percent between runs; the
`-n 60`, single-thread, same-QP comparison is the one to repeat.

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
