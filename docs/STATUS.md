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

with and without the in-loop deblocking filter.  `make test` asserts this and
fails (non-zero exit) on any mismatch; it runs twenty-six encoded clip cases
frame by frame, checks that the ABR controller hits its target, plus a sweep of
the partition, QP and GOP options (`tools/sweep.sh`) and a 240-case matrix over
`--partitions 1|3|7|15`.

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
* P inter partitioning: 16x16, 16x8, 8x16 and **P_8x8** (four 8x8 sub-macroblock
  partitions, one motion vector and one `mvd` each).  Every partition gets its
  own motion search and its own MV predictor, including the sub-macroblock
  rules of 8.4.1.3.1 and the per-partition availability rule (bug 27).  Each 8x8
  sub-macroblock picks its own `sub_mb_type`: 8x8, 8x4, 4x8 or 4x4.
* P_Skip macroblocks (`mb_skip_run`), including the run that reaches the end of
  the picture
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
* SSE2 fast paths for the four hottest inner loops, each kept bit-identical to
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

## Verification


```sh
make -j2
FFMPEG=/path/to/ffmpeg make test      # tables + cavlc + 19 clip cases
```

`make test` runs five things:

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
4. `tests/kernel_check.c` — the OpenCL device kernels against the same scalar
   reference, when a device is present.
5. `tests/run_tests.sh` — twenty-eight cases (QP 12/23/26/40, presets 0 and 4,
   deblocking on/off, 64x64 … 320x208, forced CPU backend, 16x8/8x16/P_8x8
   partitioning with all four `sub_mb_type` shapes, both the integer and the
   quarter-pel motion search, the forced whole-frame search seed *and* the
   forced whole-frame fractional refinement, all-intra and 3-frame GOPs, the RD
   Intra_4x4 decision of the slow presets, P_Skip, ABR and AQ)
   encoded, decoded with ffmpeg and compared **frame by frame**, then an ABR
   accuracy check: the encoder's `--dump-recon` must be bit-identical to the decoder's
   output.  Any mismatch fails the script; a decoder that logs an error also
   fails it.  `tests/genyuv.c` takes a mode argument so that the partitioning
   cases really move the two halves of the picture in opposite directions —
   the default clip pans as one block and can never make a partition pay.

`reconstruction vs decode` must be `max abs diff 0`.  A nonzero diff means the
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

## Next

1. Quality/RD: per-block rate-distortion level selection (trellis, or at least
   the last non-zero coefficient), then the 8x8 DCT (the latter needs a
   High-profile SPS).  That is where the remaining ~0.6-1.0 dB of inter and
   ~1.5-2 dB of intra-only efficiency sits.  The sub-partition *shape* choice
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
   dispatcher's
   calibration has never run against real hardware in this environment (there
   is no GPU here), so the crossover points are still the CPU-model ones; the
   dispatcher also only ever measures the path it chose, so a stage is switched
   by its forced override (`-s stage=backend`) or, on hardware, after both
   paths have been probed.
3. Multi-slice and multi-threaded slice encoding are not implemented yet, and
   rate control is picture-level only (no VBV/HRD model, no macroblock-tree
   lookahead).

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

## Quality against a reference CPU encoder

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
this sandbox), best of three runs:

| encoder | fps | bytes | Y-PSNR |
|---|---|---|---|
| ours, preset 0 | 638 | 163386 | 42.30 |
| ours, preset 2 / 4 | 90.2 / 82.6 | 78502 / 78468 | 45.12 / 45.12 |
| x264 baseline (no asm) | 169.5 | 78 664 | 45.16 |

At a matched QP the two encoders land within 0.2% of each other in rate and
0.04 dB in quality, and x264 is 1.9x faster (it was 3.4x two rounds ago).
Absolute numbers move a few percent between runs; the `--frames 60`,
single-thread, same-QP comparison is the one to repeat.

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

The last two are worth less than the microbenchmarks suggest: instrumenting the
encoder with a cycle counter shows the three coefficient functions are 10% of
the encode (2.88 M calls per 150 frames, 40 + 101 + 47 cycles each), not the
33% a `-pg` profile had attributed to them, so a 2-3x faster block is ~3% of
the picture.  The machine is noisy (the same binary measures anywhere between
78 and 91 fps depending on the run), so the last rows are medians of five
alternating runs rather than a precision result.

Stage profile at QP 26, 40 frames, preset 2 (the default): motion estimation
221 ms, fractional-pel planes 42 ms, transform 260 ms, deblocking 4 ms of
538 ms (motion estimation was 433 ms of 783 ms before the SIMD kernels).
Motion estimation is 41% of the encode now and the *transform* stage is the
largest single block at 48% -- but that stage is the whole of `code_mb()`, i.e.
the rate-distortion pass: forward transform, quantisation and reconstruction of
every candidate, plus the bitstream, not one 4x4 transform.  The GPU story is
unchanged and sharper: the search's two halves are pure batches of independent
evaluations (`sad_batch`, `satd_batch`) and the plane build has its kernel
(`interp_plane`); the coefficient pipeline is the one that still needs its
batch form (see *Next*).

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
