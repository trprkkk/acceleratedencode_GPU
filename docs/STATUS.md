# Status and engineering notes

Date: 2026-10-24.  Branch: `arena/589d1e45-acceleratedencode-gpu`.

## Where the project stands

Implemented and building cleanly (`make`, no warnings in our own code):

* bit writer with emulation prevention, NAL/SPS/PPS/slice headers
* CAVLC residual coding (coeff_token, levels, total_zeros, run_before)
* forward/inverse 4x4 integer transform, 4x4 and 2x2 DC Hadamard, quantisation
  and dequantisation with the reference tables
* Intra_16x16 (all four modes), chroma (all four modes), Intra_4x4 predictors
* quarter-pel motion estimation/compensation, MV prediction
* deblocking filter with the 52-entry alpha/beta/tc0 tables
* CPU + OpenCL backends behind a measured dispatcher, device query, calibration
* CLI with `--list-devices`, `--calibrate`, `--bench`, `--dump-recon`

**Blocker:** the emitted bitstream is not yet decodable by an external decoder.
The encoder's own reconstruction is good (38.1 dB on a smooth 320x192 clip at
QP 23, 53.5 dB on a flat one), but an external decoder reports
`Invalid level prefix` / `cbp too large` inside the first macroblock, which
means the residual bits are still wrong for blocks with several non-zero
coefficients.  Single-coefficient blocks are verified correct.

## Reference-decoder facts that were established empirically

These were derived by writing a hand-built slice with one macroblock
(`tests/synth_probe.c`), decoding it, and reading the output pixel.  They must
not be "simplified" again — each number is measured, not assumed.

* Inverse quantisation of an ordinary 4x4 block (FFmpeg `dequant4_coeff`):
  `W' = Z * (LevelScale[qp%6][class] << (qp/6 + 2))` combined with the
  `>> 6` in the inverse transform, which is equivalent to
  `W' = Z * LevelScale[qp%6][class] << (qp/6)` followed by the butterfly with
  a final `>> 6`.  Class is `(x&1)+(y&1)`.  Verified: at QP 26 a level of 1 in
  class 1 reconstructs a residual of 4, which is exactly the integer basis.
* Intra_16x16 luma DC: the 4x4 DC block is a plain Hadamard (no shift), then
  `dc = (z_hadamard * (LevelScale[qp%6][0] << (qp/6 + 6)) + 128) >> 8`, then the
  per-4x4-block inverse gives `(dc + 32) >> 6` for a DC-only block.  Verified at
  QP 2/8/14/20/26 and levels 1..1000 (see the table below).
* 4:2:0 chroma DC: `dc = (z_hadamard * (LevelScale[qp%6][0] << (qp/6 + 6))) >> 7`
  and then `(dc + 32) >> 6` per 4x4 block.  Note the `>> 7` versus the luma
  `>> 8` — that single bit is the difference between working and not.
* The dequantiser for the DC paths therefore has the extra `<< 4` (spec
  `LevelScale << (qp/6 + 2 + 4)`), i.e. the DC coefficient is 16x the naive
  value; the inverse transform then normalises it away.
* CAVLC coefficient tokens: the 4x4 tables are indexed `[nC bucket][total*4 +
  trailing]` and the chroma-DC table uses the *same* convention
  (`[total*4 + trailing]`, with slot 0 holding the empty-block code).  The
  chroma-DC table is **not** `[(total-1)*4 + trailing]`; that off-by-one-row
  mistake makes every chroma-DC block decode as empty and desynchronises the
  whole slice.
* CAVLC levels: `levelCode = 2*|level| - 2 + (level < 0 ? 1 : 0)`, minus 2 for
  the first non-trailing coefficient when TrailingOnes < 3, prefix
  `levelCode >> suffixLength` with the escape origin `15 << suffixLength` for
  ordinary coefficients but **30** for the first coefficient of the block
  (prefix 14 + four bits is the first coefficient's own sub-escape).  Suffix
  length update: `if (suffix == 0) suffix = 1; else if (|level| > 3 << (suffix-1)
  && suffix < 6) suffix++;`.
* Intra_16x16 always codes the luma DC block and all sixteen AC blocks, and
  CodedBlockPatternChroma != 0 forces both chroma DC blocks (and AC_N forces
  all eight chroma AC blocks) — even when the levels are all zero.  Omitting
  the empty ones desynchronises the slice immediately.
* Chroma intra prediction mode is signalled as `0 = DC, 1 = Horizontal,
  2 = Vertical, 3 = Plane`; Intra_16x16 luma modes are `0 = Vertical,
  1 = Horizontal, 2 = DC, 3 = Plane`.  The two orders are different.
* `slice_qp_delta` must be `qp - 26` (`pic_init_qp_minus26` is zero in our PPS).
  Forgetting it decodes as QP 26 whatever `-q` says, which looks exactly like a
  dequantisation error.

### Luma-DC scale measurement (single-coefficient probe, QP 26)

| level | pixel | residual | predicted by the formula above |
|---|---|---|---|
| 1 | 129 | 1 | 1 |
| 2 | 130 | 2 | 2 |
| 3 | 130 | 2 | 2 |
| 5 | 132 | 4 | 4 |
| 10 | 136 | 8 | 8 |
| 16 | 141 | 13 | 13 |
| 20 | 144 | 16 | 16 |
| 40 | 161 | 33 | 33 |
| 50 | 169 | 41 | 41 |
| 100 | 209 | 81 | 81 |

Chroma-DC at the same QP: level 1 -> 130, 5 -> 136, 10 -> 144, 20 -> 161,
50 -> 209, i.e. the same 1.625-per-level slope with `>> 7` in place of `>> 8`.

## Remaining bug (what to fix next)

Symptom: a single 16x16 macroblock with several non-zero coefficients decodes
with `Invalid level prefix`; larger pictures fail with `cbp too large` at the
second macroblock.  Single-coefficient blocks (luma DC, chroma DC) are exact.

Prime suspect: the *multi-level* path of `write_block()` in `src/cavlc.c`.  The
one-coefficient cases exercise only the "first coefficient" escape branch, so a
mistake in the suffix-length ramp, in the escape origin for later coefficients,
or in `total_zeros`/`run_before` would only show up with several coefficients.

Recommended next step: add `tests/cavlc_check.c`, which encodes a block of
pseudo-random coefficients and decodes it with a transcription of the reference
level/run decoder (FFmpeg's `h264_cavlc.c:decode_residual()`), comparing
coefficient by coefficient.  That isolates the writer from the rest of the
bitstream in a few lines of output, instead of debugging through a decoder.

After that, in order:

1. Re-check the luma-DC coefficient *order* (scan).  The DC levels are currently
   transmitted in raster order; if the decoder's scan for the Intra_16x16 DC
   block is the zigzag, the parse stays valid but the reconstruction differs,
   which is masked today by the level bug.  Compare `--dump-recon` with a
   decoded frame on a smooth clip: `rec == dec` exactly when this is right.
2. Motion-compensated path: P pictures, `mvp` prediction, skip runs, and the
   deblocking filter ordering (MB QP averaging) all still need a conformance
   pass.
3. The quality harness: compare BD-rate against x264 at the same QP on a
   standard clip; the RD decision currently only uses SAD + a lambda estimate.
4. `src/backend_cuda.c` (the `CUDA=1` build is still missing) and the GPU
   kernels for the intra modes.

## Verification commands

```sh
make -j2
gcc -O2 tests/genyuv.c -o /tmp/genyuv -lm && /tmp/genyuv 320 192 10 /tmp/test.yuv
./build/aegenc -w 320 -h 192 -q 23 -n 10 -i /tmp/test.yuv -o /tmp/out.264 \
               --dump-recon /tmp/recon.yuv
ffmpeg -v error -i /tmp/out.264 -f rawvideo -pix_fmt yuv420p /tmp/dec.yuv -y
# probe: one macroblock, one known coefficient
gcc -O2 -std=c11 -Iinclude -Isrc -o /tmp/probe tests/synth_probe.c \
    src/bitstream.c src/cavlc.c src/util.c src/tables.c src/transform.c
/tmp/probe <luma_dc_level> <chroma_dc_level> <qp> out.264 <mb_type> <chroma_mode> <qp_delta>
```

`tests/run_tests.sh` wraps the first part and prints the three PSNR numbers; the
`reconstruction vs decode` number is the conformance indicator and must be
`inf` (identical) before any quality comparison is meaningful.
