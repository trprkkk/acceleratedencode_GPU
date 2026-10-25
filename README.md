# acceleratedencode_GPU

A from-scratch H.264 (AVC) encoder core written in portable C, whose compute
stages are dispatched between the CPU and a general-purpose GPU backend
(OpenCL, with a CUDA backend behind the same interface).  **No NVENC, no QSV,
no fixed-function hardware encoder is used anywhere**: every pixel operation is
ordinary SIMD/SIMT compute, so the picture quality is decided by our own
rate-distortion decisions and transforms, not by a vendor ASIC.

> **Status: conformant High-profile encoder core (I, P and B pictures,
> Intra_16x16, Intra_4x4 and Intra_8x8, the 8x8 luma transform, CABAC,
> 16x16/16x8/8x16/P_8x8 inter partitioning down to 4x4 sub-partitions, the same
> shapes in B pictures with a per-partition choice of list, `B_8x8`, P_Skip,
> temporal direct mode and bi-prediction -- the complete B macroblock layer),
> quality work in progress.**
> The bitstream decodes correctly: an independent decoder (FFmpeg 7.0.2)
> reproduces the encoder's reconstruction **exactly** (max abs diff 0, on every
> frame, not just the IDR) on 16x16 – 320x208 content, QP 10–51, presets 0–4,
> every partition mask, `-g 1` and `-g 3`, with and without deblocking.
> `make test` asserts it over 71 clip cases (fourteen of them B pictures, compared
> after undoing the coding-order permutation), two CABAC round-trips, a
> 200 000-block coefficient cross-check and the device kernels, and *fails* on
> any mismatch: every case is bit-exact, including the preset-3/4
> motion-vector refinement and the trellis quantiser.  What
> is *not* finished is speed.  B pictures are conformant and on by default (three
> between two anchors, quantised twelve coarser): at matched PSNR against libx264's
> default preset (`medium`: B pictures, CABAC, 8x8 transform, trellis
> quantisation, AQ, SIMD) the encoder now spends 10.3–21.9 % **fewer** bits on four
> of the five test clips and 4.4 % more on the fifth, where P-only coding was
> 9.6–27.8 % *more* on four of them.  What is left of the gap is the *search*
> rather than the syntax (partition shapes, per-partition lists and sub-pel
> vectors are chosen by SATD rather than by the coder's price; `B_8x8` measured
> neutral for exactly that reason and is enabled from preset 3).  Level selection priced by the coder is implemented and
> measured, and stays off by default: it is worth 1.5–2.8 % BD-rate on the
> clips with the most residual and 0.1–2 % on the others, for 19–40 % more
> time.  Speed is the
> real gap: 69 fps (preset 2) or 40 fps (preset 4) against x264's ~400 fps on
> two cores at 320x192 QP 24, and 68 % of the CPU time is now the
> transform/quantise/reconstruct work of the rate-distortion pass with 31 % in
> the motion search (80 %/19 % at preset 4), which is exactly what the GPU
> backends exist for.  The GPU backends sit behind a measured dispatcher, and
> the kernels are checked against the CPU reference by `tests/kernel_check.c`,
> but nothing has run on real GPU hardware in this environment (there is no GPU
> available).  See
> [`docs/STATUS.md`](docs/STATUS.md).

## Why this design

Porting x264/x265 to the GPU is attractive because the expensive stages
(motion search, forward/inverse transforms, quantisation, loop filter) are data
parallel, while the serial stages (entropy coding, bitstream assembly, rate
control) are not.  This project therefore splits the encoder into two classes:

| Stage | Where it runs | Why |
|---|---|---|
| Motion estimation (SAD/SATD over a search window) | GPU (batched kernel) | thousands of independent block comparisons |
| Whole-frame parallel search rounds (8 neighbours of every macroblock's best vector, four rounds) | GPU (`sad_batch`, one launch per round) | a whole picture's search as four batched calls instead of a serial hex walk per macroblock |
| Fractional refinement as rounds (8 quarter/half-pel neighbours of every macroblock, up to three rounds at each step) | GPU (`satd_batch`, one launch per round, reference registered once by `ref_upload`) | each round is independent across macroblocks; replaces the per-macroblock subpel walk outright |
| Fractional-pel planes (the four luma planes a quarter-pel position is built from) | GPU (`interp_plane`, one launch per plane) when the dispatcher assigns `inter`, CPU reference otherwise | every output sample independent; removes the per-candidate 6-tap filters |
| Inter partition search (16x16 / 16x8 / 8x16 / P_8x8 candidates) | GPU for the SAD batches, CPU for the MV prediction | the four 8x8 searches of one macroblock are independent |
| Forward transform (4x4 DCT) + forward quantisation | GPU (`tq_batch`: one launch per inter candidate, or one per macroblock over *all* its candidates, `AEG_RD_BATCH`) | per-block, no dependencies; the candidates of a macroblock are independent of each other |
| Inverse quantisation + inverse transform | GPU (`iq_idct_batch`, same blocks as above) | per-block, no dependencies |
| SSE / SATD cost accumulation | GPU (batched kernel) | pure reduction |
| Distortion metrics on the CPU reference (`aeg_sad`, 4x4 Hadamard `aeg_satd`) | CPU, SSE2 | exact integer sums, so a 13x / 2.2x faster vector path cannot change a decision |
| Chroma and Intra_16x16 coefficient blocks | CPU | their DC coefficients are replaced by the 4x4 / 2x2 Hadamard of the neighbouring DCs, so they are not independent blocks |
| The scalar fallbacks of the quantiser and the inverse transform (`aeg_quant_4x4`, `aeg_idct_add_block`) | CPU, SSE2 | same reason: integer in, integer out, bit-identical to the scalar loop the GPU kernels are checked against |
| Intra prediction | CPU | needs the just-reconstructed neighbours, serial per MB |
| Intra_4x4 mode analysis (9 predictors x 16 blocks per MB) | CPU for now; the 16 blocks of a macroblock are independent once the neighbours exist | the candidate that dominates I-picture cost |
| Deblocking filter | CPU | pinned: serial in MB order, communication cost dominates |
| Entropy coding (CABAC by default, CAVLC with `--no-cabac`) | CPU | inherently serial: every bin depends on the arithmetic coder's state and on context states the previous macroblocks left behind |
| Bitstream assembly / rate decisions | CPU | serial |

The dispatcher is *measured*, not guessed: each stage is timed per frame in both
places when `--bench` is used, and a stage only stays on the GPU when the
measured time (including transfer and launch overhead) is lower than the CPU
path, with 10% hysteresis to avoid oscillation.  `aeg_backend_calibrate()`
measures a real kernel launch + transfer round trip so the model knows the
fixed cost below which batching cannot pay off.

Quality goal: the encoder must be *decision equivalent* to a CPU implementation
of the same algorithm — same transforms, same quantiser, same integer
reconstruction.  The CPU and GPU paths are required to be bit-identical; that is
why every kernel is an integer operation and the reference implementation in
`src/backend.c` is the ground truth the GPU kernels are checked against.

## Building

```sh
make                 # CPU only (default, no OpenCL/CUDA required)
make OPENCL=1        # + OpenCL backend, loaded with dlopen() at run time
make CUDA=1          # + CUDA backend (driver API + NVRTC, both dlopen()ed)
make test            # table + CAVLC + coefficient checks, then bit-exactness
                     # of 28 clips
```

`make OPENCL=1` needs no headers or libraries at build time: the backend
resolves `libOpenCL.so` with `dlopen()` so a CPU-only machine can still build
the tree.  `make CUDA=1` works the same way: `src/backend_cuda.c` is plain C
that opens `libcuda.so.1` (driver API -- no `cuda.h` needed) and `libnvrtc.so`
(NVRTC compiles `src/kernels.h` at run time for the device's compute
capability), so no CUDA toolkit or `nvcc` is required to build, and a machine
without a GPU simply reports no CUDA device at run time.

## Usage

```sh
./build/aegenc --list-devices
./build/aegenc --calibrate
./build/aegenc -w 320 -h 192 -q 23 -n 300 -i input.yuv -o out.264 \
               --dump-recon recon.yuv --bench
```

`-b cpu|opencl|cuda|auto` selects the backend, `--bitrate <kbps>` switches from
constant QP to one-pass average-bitrate rate control (it lands within 0.2 % of
the target over 90 frames), `--aq <tenths>` turns on variance-based adaptive
quantisation (off by default: it is worth its PSNR cost only perceptually, see
`docs/STATUS.md`), `-p 0..4` is the speed/quality
preset (it scales the search range, the subpel refinement and, from 3 up, the
rate-distortion refinement of the winning motion vector and the level
selection), `--refine-mv 0|1|2`
controls that last one on its own (0 off, 1 the four axial quarter-pel
neighbours, 2 all eight; 1.2-4.2 points of BD-rate against x264 medium for
30-45 % more time, and from preset 3 up it also refines each partition of a
16x8 / 8x16 winner and both lists of every B-picture partition (worth 0.4-0.9
points there, see `docs/STATUS.md`), `--refine-cand 0|1`
refines *every* inter candidate's vectors rather than only the winner's, so a
candidate that lost on a quarter-pel error gets a second chance (0.3-0.6 points
of BD-rate for about 40 % more time at preset 4; off by default, see
*Refining every candidate* in `docs/STATUS.md`), `--trellis 0|1` controls rate-distortion
level selection on the luma residual (priced by the CABAC coder; on from preset
3 up, where it is worth 0.15-2.2 points for 12-14 % of the encode time, see
*Rate-distortion level selection* in `docs/STATUS.md`),
`--partitions`
selects which inter partition shapes may be used (`1` = 16x16 only, `7` =
16x16 + 16x8 + 8x16, `8` = P_8x8 only, `15` = everything, which is the
default), `--dump-recon` writes what the encoder itself
reconstructed (this is what a decoder must produce, so it is the primary
conformance check), `--no-8x8dct` drops the 8x8 transform and Intra_8x8 (4x4
luma transforms only, which is how the cost of the High-profile tools was
measured), `--no-cabac` replaces CABAC with CAVLC (34-36% more bytes at equal
PSNR; kept as the comparison path and because both coders have to reconstruct
the same picture), and `--no-deblock`, `--subme N`, `--range N` override the
preset.

## What is verified, and how

* `tests/synth_probe.c` writes a hand-built slice containing exactly one
  macroblock with a single known coefficient.  Decoding it with an independent
  H.264 decoder and measuring the output pixel is an exact measurement of the
  decoder's inverse quantisation — this is how every scale factor in
  `src/transform.c` was derived and checked (see `docs/STATUS.md`).
* `tests/syntax_check.c` asserts that the two `coded_block_pattern` tables are
  exact inverses of one another.  A single wrong entry does not desynchronise
  the stream — the decoder reads a legal but *different* pattern and applies
  residual in the wrong 8x8 quadrants — so it has to be checked explicitly.
* `tests/cavlc_check.c` round-trips 2 000 random residual blocks through the
  CAVLC writer and a transcription of the reference decoder.
* `tests/cabac_mb_check.c` round-trips the CABAC *residual block* syntax
  through a second transcription of 9.3.2.2: 20 000 random blocks over the six
  context categories, levels reaching the Exp-Golomb escape, both signs.  This
  is the check that covers the binarization and the context derivation, which
  sit above the arithmetic coder and are where all three encoder-side CABAC
  bugs were.
* `tests/cabac_check.c 200000 8` round-trips the CABAC arithmetic coder
  through a decoder written from the spec's decoding process (nine-bit
  `codIOffset`, bit-by-bit refill) rather than from the encoder: 1.6 M bins
  over all 460 contexts, every probability state, bypass bins and the
  end-of-slice flags, with the decoder's contexts required to end up identical
  to the encoder's.  Conformance cannot catch a CABAC bug -- a wrong coder
  produces a legal stream that reconstructs a different picture -- so it is
  checked on its own before any syntax uses it.
* `tests/coef_check.c 200000` drives the coefficient path (forward
  quantisation, dequantisation, inverse transform, prediction add) with random
  blocks -- uniform, sparse, single-coefficient, DC-only, and amplitudes at
  both int16 and dispatch extremes -- at every QP and at six block strides,
  against a reference written out independently in the test, and through the
  in-place case as well.  Vector kernels here are the hardest thing in the tree
  to get right, because a wrong one produces a *valid* stream that simply
  reconstructs differently; the reference has to be written twice, from the
  spec, to catch it.
* `tools/bdrate.py --clip clip.yuv W H FRAMES [-q 22,25,28,31,34,37]` is the
  quality meter: it runs both encoders at the same QPs on the same source,
  compares the decoded streams (never an encoder's own PSNR summary, which uses
  a different convention), and prints the BD-rate with the PSNR range it was
  integrated over, flagging the result when one curve does not span the other.
  The BD-rate is the standard one: negative means this encoder needs fewer bits
  at matched PSNR.
* `tools/gpu_validate.sh clip.yuv W H FRAMES QP` is the hardware check: it
  encodes with the CPU backend, with the dispatcher free, and with each batched
  device path forced on, then requires the coefficient paths to be the *same
  bytes* as the CPU stream and every stream's reconstruction to match an
  independent decode.  Run it on the machine that has the GPU; on a CPU-only
  machine everything passes trivially, because the batch entry points fall back
  to the reference ops.
* `tests/run_tests.sh` encodes fifty-four synthetic cases (QP 12/23/26/40,
  presets 0–4, deblocking on/off, 64x64 … 320x208, forced CPU backend, 16x8,
  8x16 and P_8x8 partitioning with all four `sub_mb_type` shapes, P_Skip,
  integer-only and quarter-pel search, the forced whole-frame search seed,
  all-intra and short GOPs, ABR and AQ), decodes them
  with ffmpeg and requires the encoder's own reconstruction to be
  **bit-identical** on every frame; it exits non-zero otherwise.  It also
  checks the skip statistics the encoder reports on a still clip, which is the
  cheapest macroblock the standard has and the one the mode decision used to
  miss entirely (see *The skip candidate* in `docs/STATUS.md`).
* `--bench` prints per-stage CPU/GPU timings and how the dispatcher voted.
* `-v 1` (the default) also prints how many macroblocks of the stream are
  skips and the full type breakdown (P16x16 / 16x8 / 8x16 / P_8x8 / the intra
  types), which is the quickest way to see whether the mode decision is finding
  the cheapest macroblock the standard has.

## Layout

```
include/aeg/aeg.h   public API
src/h264.h          internal data structures and prototypes
src/tables.c        all constant tables (quantiser, deblock, CAVLC, zigzag, QP map)
src/transform.c     forward/inverse 4x4 transform, DC Hadamard, quantisation
src/intra.c         Intra_4x4 (9 modes) / Intra_16x16 / chroma prediction
src/mc.c            motion estimation, quarter-pel MC, fractional-pel plane cache
src/deblock.c       in-loop deblocking filter
src/cavlc.c         CAVLC entropy coding of residual blocks
src/bitstream.c     NAL, SPS, PPS, slice headers, bit writer
src/encoder.c       the encoder itself (analysis, RD decision, reconstruction)
src/backend.c       CPU kernels, dispatch model, device query, calibration
src/backend_opencl.c OpenCL backend (dlopen, cached buffers, 6 kernels)
src/backend_cuda.c  CUDA backend (driver API + NVRTC, both dlopen()ed)
src/kernels.h       the shared kernel source string (OpenCL C / CUDA C)
tests/kernel_check.c cross-checks every GPU kernel against the CPU reference
cli/aegenc.c        command line front end
tests/              synthetic clip generator, single-MB probe, test runner
```

## Licence

MIT (see `LICENSE`).
