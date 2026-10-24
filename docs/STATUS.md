# Status and engineering notes

Date: 2026-10-24.  Branch: `arena/589d1e45-acceleratedencode-gpu`.

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
* Intra_16x16 (all four modes), chroma (all four modes)
* half/quarter-pel motion estimation and compensation, MV prediction, skip runs
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
15. **Picture padding geometry.**  Every plane occupies `(h + 2*PAD)` rows of
    `w + 2*PAD` bytes and the bottom/right padding is edge-replicated.  A row
    copy that starts at the *sample* origin instead of the row origin runs
    `PAD` bytes past the plane; the chroma plane offset must be
    `ls*lh + cs*ch`, not `plane[1] + cs*ch` (which also drops its top
    padding).

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
3. `tests/run_tests.sh` — nine clips (QP 12/23/26/40, presets 0 and 4,
   deblocking on/off, 64x64 … 320x208, forced CPU backend) encoded, decoded
   with ffmpeg and compared **frame by frame**: the encoder's `--dump-recon`
   must be bit-identical to the decoder's output.  Any mismatch fails the
   script; a decoder that logs an error also fails it.

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

1. Quality/RD: the mode decision is still SAD + a lambda estimate; the next
   wins are 16x8/8x16/8x8 partitions, Intra_4x4 and a real `lambda * bits`
   term (bits are already counted by `aeg_cavlc_block_cost()`).
2. `src/backend_cuda.cu` for the `CUDA=1` build (the Makefile expects it, the
   tree does not have it yet), plus more GPU kernels (intra prediction, SAD,
   transform/quantisation) and dispatcher calibration on real hardware.
3. Multi-slice / rate control are not implemented yet.

## Quality against a reference CPU encoder

`x264` (master, `--profile baseline --qp N --keyint 250`, built without asm) on
the same 60-frame 320x192 synthetic texture, both streams decoded and compared
with the source:

| QP | ours (kbps / PSNR) | x264 (kbps / PSNR) |
|---|---|---|
| 24 | 4592 / 37.19 dB | 4825 / 38.99 dB |
| 32 | 2497 / 30.23 dB | 2672 / 31.73 dB |

Inter prediction is worth roughly a 2x bitrate cut on this clip (the earlier
all-intra rows — 8930 kbps at QP 24 — came from `partitions == 0`, bug 14).
Today we spend ~5% *fewer* bits than x264 at the same QP and get ~1.5–1.8 dB
less PSNR, i.e. x264 is still ~15–20% more efficient.  The gap is
rate-distortion efficiency, not correctness: the mode decision compares 16x16
inter against Intra_16x16 with a lambda estimate, only Intra_16x16 is
implemented (no Intra_4x4 / I_PCM / 8x8 partitions), and the quantiser has no
trellis or last-coefficient optimisation.  Those items are the next quality
work; they do not affect conformance (any mode decision is decodable, and the
recon still has to match the decoder exactly — re-run `make test` after
touching them).
