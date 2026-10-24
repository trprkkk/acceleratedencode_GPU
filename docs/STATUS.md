# Status and engineering notes

Date: 2026-10-24.  Branch: `arena/589d1e45-acceleratedencode-gpu`.

## Where the project stands

The encoder is **conformant**: an independent decoder (FFmpeg 7.0.2) reproduces
the encoder's reconstruction *bit for bit* (`max abs diff 0`, PSNR `inf`) on

| content | sizes | QP | GOP | frames |
|---|---|---|---|---|
| 1 MB probe, grids, smooth gradients | 16x16 … 320x192 | 10 … 51 | 1, 2, 3, 5, 10, 250 | 1 … 40 |
| busy synthetic texture (heavy AC) | 320x192 | 10, 18, 20, 26, 32, 34, 40, 42, 44, 47…51 | 1 … 250 | 1 … 40 |

both with and without the in-loop deblocking filter.  `make test` asserts this
(`reconstruction vs decode = 99.00 dB, max abs diff 0`).

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

## Verification

```sh
make -j2
FFMPEG=/path/to/ffmpeg make test      # cavlc round-trip + end-to-end clip
```

The end-to-end check prints `reconstruction vs decode`; it must be `99.00 dB
(max abs diff 0)`.  A nonzero diff means the encoder and the decoder disagree
about samples, which drifts on every following P picture.

Manual single-macroblock probe (used to isolate transform/quantisation bugs):

```sh
./build/aegenc -w 16 -h 16 -q 33 -n 1 -i mb0.yuv -o v.264 --dump-recon vr.yuv
ffmpeg -v error -i v.264 -f rawvideo -pix_fmt yuv420p vd.yuv -y
# vr.yuv and vd.yuv must be identical
```

## Next

1. Quality/RD: the mode decision is still SAD + a lambda estimate; compare
   BD-rate against x264 at matched QP on standard clips.
2. `src/backend_cuda.c` for the `CUDA=1` build, plus more GPU kernels
   (intra prediction, SAD, transform/quantisation) and dispatcher calibration
   on real hardware.
3. Multi-slice / rate control are not implemented yet.

## Quality against a reference CPU encoder

`x264` (master, `--profile baseline --qp N --keyint 250`, built without asm) on
the same 60-frame 320x192 synthetic texture, both streams decoded and compared
with the source:

| QP | ours (kbps / PSNR) | x264 (kbps / PSNR) |
|---|---|---|
| 24 | 8930 / 38.50 dB | 4825 / 38.99 dB |
| 32 | 5593 / 31.22 dB | 2672 / 31.73 dB |

So the *per-QP* quality is within ~0.5 dB of x264, but x264 reaches that quality
at roughly half the bitrate.  The gap is rate-distortion efficiency, not
correctness: the mode decision here is SAD-only (no `lambda * bits` term), only
Intra_16x16 is implemented (no Intra_4x4), and the quantiser has no trellis or
last-coefficient optimisation.  Those three items are the next quality work;
they do not affect conformance (any mode decision is decodable, and the recon
still has to match the decoder exactly — re-run `make test` after touching
them).
