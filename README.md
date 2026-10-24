# acceleratedencode_GPU

A from-scratch H.264 (AVC) encoder core written in portable C, whose compute
stages are dispatched between the CPU and a general-purpose GPU backend
(OpenCL, with a CUDA backend behind the same interface).  **No NVENC, no QSV,
no fixed-function hardware encoder is used anywhere**: every pixel operation is
ordinary SIMD/SIMT compute, so the picture quality is decided by our own
rate-distortion decisions and transforms, not by a vendor ASIC.

> **Status: conformant encoder core (I and P pictures, Intra_16x16 and
> Intra_4x4, 16x16/16x8/8x16 inter partitioning and P_Skip), quality work in
> progress.**
> The bitstream decodes correctly: an independent decoder (FFmpeg 7.0.2)
> reproduces the encoder's reconstruction **exactly** (max abs diff 0, on every
> frame, not just the IDR) on 16x16 – 320x208 content, QP 12–40, presets
> 0–4, with and without deblocking.  `make test` asserts it and *fails* on any
> mismatch.  What is *not* finished is rate-distortion efficiency: at matched QP
> the bitrate is already below x264 baseline (~5% less) but the PSNR is ~1.5 dB
> lower, i.e. x264 is still ~15–20% more efficient (SATD+λ mode decision, no
> trellis quantisation, no 8x8 sub-partitions or 8x8 DCT).  The GPU backends sit
> behind a measured dispatcher but have only been compile-checked in this
> environment (no GPU available).  See [`docs/STATUS.md`](docs/STATUS.md).

## Why this design

Porting x264/x265 to the GPU is attractive because the expensive stages
(motion search, forward/inverse transforms, quantisation, loop filter) are data
parallel, while the serial stages (entropy coding, bitstream assembly, rate
control) are not.  This project therefore splits the encoder into two classes:

| Stage | Where it runs | Why |
|---|---|---|
| Motion estimation (SAD/SATD over a search window) | GPU (batched kernel) | thousands of independent block comparisons |
| Forward transform (4x4 DCT) + forward quantisation | GPU (batched kernel) | per-block, no dependencies |
| Inverse quantisation + inverse transform | GPU (batched kernel) | per-block, no dependencies |
| SSE / SATD cost accumulation | GPU (batched kernel) | pure reduction |
| Intra prediction | CPU | needs the just-reconstructed neighbours, serial per MB |
| Intra_4x4 mode analysis (9 predictors x 16 blocks per MB) | CPU for now; the 16 blocks of a macroblock are independent once the neighbours exist | the candidate that dominates I-picture cost |
| Deblocking filter | CPU | pinned: serial in MB order, communication cost dominates |
| CAVLC entropy coding | CPU | inherently serial (nC context depends on previous blocks) |
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
make CUDA=1          # + CUDA backend (needs the CUDA toolkit; not yet written)
make test            # table + CAVLC checks, then bit-exactness of 13 clips
```

`make OPENCL=1` needs no headers or libraries at build time: the backend
resolves `libOpenCL.so` with `dlopen()` so a CPU-only machine can still build
the tree.

## Usage

```sh
./build/aegenc --list-devices
./build/aegenc --calibrate
./build/aegenc -w 320 -h 192 -q 23 -n 300 -i input.yuv -o out.264 \
               --dump-recon recon.yuv --bench
```

`-b cpu|opencl|cuda|auto` selects the backend, `-p 0..4` is the speed/quality
preset (it scales the search range and the subpel refinement), `--partitions`
selects which inter partition shapes may be used (`1` = 16x16 only, `7` =
16x16 + 16x8 + 8x16), `--dump-recon` writes what the encoder itself
reconstructed (this is what a decoder must produce, so it is the primary
conformance check), and `--no-deblock`, `--subme N`, `--range N` override the
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
* `tests/run_tests.sh` encodes thirteen synthetic cases (QP 12/23/26/40,
  presets 0–4, deblocking on/off, 64x64 … 320x208, forced CPU backend, 16x8 and
  8x16 partitioning, P_Skip), decodes them with ffmpeg and requires the
  encoder's own reconstruction to be **bit-identical** on every frame; it exits
  non-zero otherwise.
* `--bench` prints per-stage CPU/GPU timings and how the dispatcher voted.

## Layout

```
include/aeg/aeg.h   public API
src/h264.h          internal data structures and prototypes
src/tables.c        all constant tables (quantiser, deblock, CAVLC, zigzag, QP map)
src/transform.c     forward/inverse 4x4 transform, DC Hadamard, quantisation
src/intra.c         Intra_4x4 (9 modes) / Intra_16x16 / chroma prediction
src/mc.c            motion estimation and quarter-pel motion compensation
src/deblock.c       in-loop deblocking filter
src/cavlc.c         CAVLC entropy coding of residual blocks
src/bitstream.c     NAL, SPS, PPS, slice headers, bit writer
src/encoder.c       the encoder itself (analysis, RD decision, reconstruction)
src/backend.c       CPU kernels, dispatch model, device query, calibration
src/backend_opencl.c OpenCL backend (dlopen, cached buffers, 4 kernels)
src/kernels.h       the shared kernel source string (OpenCL C / CUDA C)
cli/aegenc.c        command line front end
tests/              synthetic clip generator, single-MB probe, test runner
```

## Licence

MIT (see `LICENSE`).
