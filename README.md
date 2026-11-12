# acceleratedencode_GPU

A from-scratch H.264 (AVC) encoder core written in portable C, whose compute
stages are dispatched between the CPU and a general-purpose GPU backend
(OpenCL, with a CUDA backend behind the same interface).  **No NVENC, no QSV,
no fixed-function hardware encoder is used anywhere**: every pixel operation is
ordinary SIMD/SIMT compute, so the picture quality is decided by our own
rate-distortion decisions and transforms, not by a vendor ASIC.

**Scope and limits, plainly** (see [`docs/AUDIT_RESPONSE.md`](docs/AUDIT_RESPONSE.md)
for the point-by-point answer to an external review):

* **H.264 only.**  There is no HEVC/H.265 encoder here; "x265" below means
  x265 is one of the encoders this one is *compared* against.  Coded as
  4:2:0 only, 8- or 10-bit (4:2:2/4:4:4 input is converted to 4:2:0); no
  lossless mode.
* **Never run on a GPU.**  Every device path runs end to end on PoCL, an
  OpenCL runtime on the *CPU*, and must write the CPU encoder's stream byte
  for byte; the CUDA kernels are only compiled (NVRTC, every architecture).
  Whether any device placement is faster than the CPU is unmeasured.
* **What `-b auto` puts on a device:** a GPU only (a CUDA device, else an
  OpenCL GPU; `-b opencl` is needed for PoCL), and on it only the two
  coefficient stages of P/B pictures, by timing whole pictures both ways,
  and the lookahead's lowres search, by timing fields both ways.
  Everything else stays on the CPU unless `-s stage=backend` forces it.
  The device paths are per-macroblock batches with blocking transfers -- a
  correctness scaffold, not a GPU-resident design.
* **Quality** is at x264 veryslow at constant QP (presets 3-4) and, with
  CRF on long clips, preset 4 is now level with x264 veryslow (-0.7 %);
  preset 3 is +0.6 %.  x264 is still 1.2-4x faster on the CPU.
* Tested on Linux x86-64 (GCC 12, AVX2/SSE2/C paths); other OSes and
  architectures are not built or tested here.

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
> `make test` asserts it over 76 clip cases (fifteen of them B pictures, compared
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
> measured, and from preset 3 up it is on: 0.15–2.2 points of BD-rate
> (1.0–1.2 on average) for 12–14 % of the time.  Speed is the
> real gap: 97 fps (preset 2) or 49 fps (preset 4) against x264's ~400 fps on
> two cores at 320x192 QP 24, and 68 % of the CPU time is now the
> transform/quantise/reconstruct work of the rate-distortion pass with 31 % in
> the motion search (80 %/19 % at preset 4), which is exactly what the GPU
> backends exist for.  The GPU backends sit behind a measured dispatcher.  There
> is no GPU in this environment, but every device path runs end to end on a
> real OpenCL device (PoCL, a CPU implementation of OpenCL: `tools/pocl_env.sh`)
> and writes the CPU encoder's stream byte for byte, and the CUDA kernels are
> compiled by NVRTC for every architecture it supports (`tests/nvrtc_check.c`).
> What still needs GPU hardware is the speed: whether the device side ever wins.
> See *GPU groundwork on a real OpenCL device* in
> [`docs/STATUS.md`](docs/STATUS.md).

**Against x264 and x265** (the final comparison, `f52e4af`; full tables and
method under *The final comparison with x264 and x265* in `docs/STATUS.md`).
libx264 r3191 / x265 3.5 with their assembly and `--tune psnr`, 14 real clips,
constant QP 22-37, BD-rate by global PSNR (negative = aegenc needs fewer bits),
speed single-threaded on 640x352 (median of 3, this 2-core sandbox):

| aegenc | vs x264 medium | vs x264 veryslow | vs x265 medium | fps | x264 preset of equal quality (its fps) |
|---|---|---|---|---|---|
| p0 | +77.0 % | +89.5 % | +82.4 % | 76.6 | between ultrafast and superfast (414.5) |
| p1 | +7.3 % | +14.9 % | +10.5 % | 30.5 | faster (122.2) |
| p2 | +2.6 % | +9.8 % | +5.5 % | 25.9 | fast (95.8) |
| p3 | **-5.9 %** | +0.8 % | -3.2 % | 13.5 | slower (28.6) |
| p4 | -6.4 % | **+0.1 %** | **-3.8 %** | 11.5 | veryslow (13.8) |

So at constant QP presets 3-4 match x264 veryslow and beat x265 medium (x265
slow is still 4-5 % ahead), but on the CPU alone x264 reaches the same quality
1.2x (p4) to 4x (p1) faster.  With rate control (CRF, both macroblock trees on)
on the three 240-300-picture clips that comparison found p4 +11.8 % against
x264 veryslow; since then QP-RD (`5929971`) and x264's CABAC trellis
(`bb7d271`, `02a03e3`) brought it to **+0.03 %** (cockL +2.85, vtestL -4.19,
megaL +1.43; `--aq 0` against x264 `--tune psnr`), and p3 to +4.31 % (see
*CABAC trellis* in `docs/STATUS.md`); a longer lookahead window (40 / 40 /
60 / 60 at presets 1-4) then took p4 to **-0.50 %** and p3 to +3.21 % (see
*Preset 3's CRF gap* there).  A fast QP-RD walk, now preset 3's default
(`--qp-rd 2`, about +9 % time), and a coarser QP-RD lambda on P pictures
then took p4 to **-0.67 %** and p3 to **+0.64 %** (on the second long set,
long2: p4 +0.83 -> +0.19 %, p3 +1.59 -> +0.81 %; see *QP-RD on P pictures*
there).  The constant-QP table above predates both; preset 3 is now about
9 % slower than its fps column says.  High 10 keeps the same standing (p3
-4.9 % against x264 medium 10-bit).

Since the quoted status at the top was written the encoder gained B pyramid
(`--b-pyramid`), explicit weighted P prediction (`--weightp`), a macroblock
tree over a lookahead window (`--mbtree`, `--rc-lookahead`), two P
references, joint refinement of the bi-predictive vectors, a motion search
seeded by a half-resolution pre-search with a multi-hexagon stage, coder-priced
partition/sub-pel/skip decisions and trellis quantisation of luma, chroma and
intra blocks; the per-feature measurements are in `docs/STATUS.md`.

**Speed now** (640x352, 90 frames, QP 28, a two-core sandbox whose speed
varies between sessions, so the builds are timed back to back): preset 3
13.7 fps on one thread and 23.6 fps with frame threads (the default), preset
4 12.0 / 21.5 fps, against 10.5 / 17.6 and 9.0 / 17.0 for 42f1f92 before the
speed round.  The speed round itself cost +0.0 % BD-rate at preset 3 and
+0.085 % at preset 4; part of the time it bought at presets 2-3 was then spent
on chroma in the mode decision (-0.5 to -0.7 %, see `docs/STATUS.md`).
Caching the lookahead's per-picture search fields then took another 6-7 %
off with the same stream (`11366da`; in that session's sandbox preset 3
3.79 -> 3.52 s with frame threads, 25.6 fps).  The speed figures in the quoted
text above are from an earlier, smaller test clip.

**GPU groundwork without a GPU** (all on PoCL, every stream byte-identical to
the CPU encoder's): a fused forward+inverse coefficient kernel for presets 0-2
and CAVLC (`tq_iq_batch`, -14 % on the batch paths); Intra_16x16 and chroma
in the coefficient batch (`tq_dc_batch` / `iq_dc_batch`: two calls per
macroblock instead of 16-32); and *device lanes* -- a device instance per
frame thread and slice thread (OpenCL: own queue, kernels and buffers on a
shared context and program; CUDA: own buffers on a shared context and
module), so frame threads now run with a device backend and the threads'
device calls overlap instead of queueing (-24 to -43 % on the batch paths
with three frame threads; on PoCL 4.x-6.x, whose runtime aborts when
several queues launch at once, the lanes fall back to one shared instance).
Details under *GPU groundwork without hardware,
continued* in `docs/STATUS.md`.

## Why this design

Porting x264/x265 to the GPU is attractive because the expensive stages
(motion search, forward/inverse transforms, quantisation, loop filter) are data
parallel, while the serial stages (entropy coding, bitstream assembly, rate
control) are not.  This project therefore splits the encoder into two classes:

The middle column says where a stage *can* run.  Under `-b auto` only the
two coefficient rows and the lookahead row are placed on a device (by
measurement, below); the others run there only when `-s` forces them.

| Stage | Where it runs | Why |
|---|---|---|
| Lookahead: the lowres motion search of each field (picture pair) | GPU (`lowres_chains`: at presets 3-4 every field a scheduling step will price, in one launch, one work-group per picture and direction, the lowres pictures resident by ring slot; `lowres_search`, one field per launch, elsewhere; `-s la=`), CPU otherwise | a block predicts only from its left, top and top-right neighbours, so the blocks of one anti-diagonal are independent and the walk is the CPU's raster order: the same field, byte for byte |
| Lookahead: downsampling, intra estimates, cost estimates (presets 3-4) | GPU when the searches are placed there (`lowres_build` writes the picture into its ring slot and hands it back; `lowres_intra` for the window's pictures, `lowres_costs` for every group estimate the B-adapt trellis can ask for, into the cost cache), CPU otherwise; the macroblock tree's per-block costs stay on the CPU | one work-item per sample, per block and picture, per block and estimate; integer arithmetic in the CPU's order, so the same planes, estimates and decisions |
| Motion estimation (SAD/SATD over a search window) | GPU (batched kernel) | thousands of independent block comparisons |
| Whole-frame parallel search rounds (8 neighbours of every macroblock's best vector, four rounds) | GPU (`sad_batch`, one launch per round) | a whole picture's search as four batched calls instead of a serial hex walk per macroblock |
| Fractional refinement as rounds (8 quarter/half-pel neighbours of every macroblock, up to three rounds at each step) | GPU (`satd_batch`, one launch per round, reference registered once by `ref_upload`) | each round is independent across macroblocks; replaces the per-macroblock subpel walk outright |
| Fractional-pel planes (the four luma planes a quarter-pel position is built from) | GPU (`interp_plane`, one launch per plane) when the dispatcher assigns `inter`, CPU reference otherwise | every output sample independent; removes the per-candidate 6-tap filters |
| Inter partition search (16x16 / 16x8 / 8x16 / P_8x8 candidates) | GPU for the SAD batches, CPU for the MV prediction | the four 8x8 searches of one macroblock are independent |
| Forward transform (4x4 DCT) + forward quantisation | GPU (`tq_batch`: one launch per inter candidate, or one per macroblock over *all* its candidates, `AEG_RD_BATCH`) | per-block, no dependencies; the candidates of a macroblock are independent of each other |
| Inverse quantisation + inverse transform | GPU (`iq_idct_batch`, same blocks as above); in the batched paths with trellis (presets 3-4) the host reconstructs instead, from the levels it has just selected -- one launch per macroblock instead of two, unless `-s iq=` places it explicitly | per-block, no dependencies |
| SSE / SATD cost accumulation | GPU (batched kernel) | pure reduction |
| Distortion metrics on the CPU reference (`aeg_sad`, 4x4 Hadamard `aeg_satd`) | CPU, SSE2 | exact integer sums, so a 13x / 2.2x faster vector path cannot change a decision |
| Chroma and Intra_16x16 coefficient blocks | CPU | their DC coefficients are replaced by the 4x4 / 2x2 Hadamard of the neighbouring DCs, so they are not independent blocks |
| The scalar fallbacks of the quantiser and the inverse transform (`aeg_quant_4x4`, `aeg_idct_add_block`) | CPU, SSE2 | same reason: integer in, integer out, bit-identical to the scalar loop the GPU kernels are checked against |
| Intra prediction | CPU | needs the just-reconstructed neighbours, serial per MB |
| Intra_4x4 mode analysis (9 predictors x 16 blocks per MB) | CPU for now; the 16 blocks of a macroblock are independent once the neighbours exist | the candidate that dominates I-picture cost |
| Deblocking filter | CPU | pinned: serial in MB order, communication cost dominates |
| Entropy coding (CABAC by default, CAVLC with `--no-cabac`) | CPU | inherently serial: every bin depends on the arithmetic coder's state and on context states the previous macroblocks left behind |
| Bitstream assembly / rate decisions | CPU | serial |

The dispatcher is *measured*, not guessed.  The coefficient stages are placed
by trial: the first pictures of each kind alternate between the CPU and the
device, and since both placements write the same stream, the difference in a
picture's time per macroblock is exactly the cost of the placement (transfers,
launches and the batch's host work included); the faster side stays, with 10%
hysteresis, and the slower one is re-tried every 64 pictures.  The motion
search is never placed by timing, because its device form is a different
search and the stream would then depend on the machine's speed; `-s est=`
forces it.  The lookahead's lowres search is placed by timing too, in time
per block searched: per scheduling step where the fields come in batches
(presets 3-4; each side probed twice, the slower re-tried every 16 steps),
per field elsewhere (four probes, every 128 fields), the first call of each
side not counted.  Its kernels are the same search, so the choice cannot
change the stream.  At presets 3-4 most of those fields are searched
before any step asks for them: as each picture arrives, a helper thread
searches the pairs it ends (on the device's own lane, or on the CPU when
the encoder may use threads) while the encoder codes, and the encoder
waits for it only when it needs a result or reuses a lowres picture the
helper is reading.  `AEG_LATIME=1`
prints the lookahead's time and that wait.  No other stage is placed automatically: inter prediction is the
only one with a device form, and that form rebuilds every sub-sample plane of
a reference per call where the CPU keeps them, so it cannot win; `-s inter=`
forces it.  `--calibrate` (`aeg_calibrate()`) measures the fixed cost of a
batch call and the throughput of a large one and reports them in
`--list-devices` / `aeg_query_devices()`; the coefficient placement does not
need them, since its whole-picture timing already includes every launch and
copy.  `--device N` picks a device of the chosen backend (the `#N` that
`--list-devices` shows).

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
make test            # API, table, CAVLC and coefficient checks, then
                     # bit-exactness of the conformance clips
make shared          # + libaeg.so (soname libaeg.so.0, AEG_API symbols only)
make install         # aegenc, libaeg.a, libaeg.so, aeg/aeg.h, aeg.pc under
                     # PREFIX (/usr/local); DESTDIR, LIBDIR, INCLUDEDIR, BINDIR
make uninstall
make install-check   # install into a scratch DESTDIR and build the example
                     # against it through pkg-config, shared and static
make BIT_DEPTHS=8    # leave out the 10-bit (High 10) encoder
```

The library holds two encoders, 8-bit and 10-bit, as x264 does: the sources
that touch samples or coefficients are compiled twice (`-DBIT_DEPTH=8` and
`-DBIT_DEPTH=10`, the second set's symbols renamed with a `_10` suffix by
`objcopy`), and `aeg_encoder_open()` picks one from `params.bit_depth`.  Use a
separate `BUILD=` directory when switching `BIT_DEPTHS`.

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

`-b cpu|opencl|cuda|auto` selects the backend (rate control: see below), `--aq <tenths>` turns on variance-based adaptive
quantisation (off by default: it is worth its PSNR cost only perceptually, see
`docs/STATUS.md`), `--frame-threads 0|1|2` codes each non-reference B
picture on a second thread next to the following picture (0 = auto, on with
more than one CPU unless `-t 1`; the stream is byte-identical to the serial
one; constant QP, one slice), `-p 0..4` is the speed/quality
preset (it scales the search range, the subpel refinement and, from 3 up, the
rate-distortion refinement of the winning motion vector and the level
selection), `--refine-mv 0|1|2|3`
controls that last one on its own (0 off, 1 the four axial quarter-pel
neighbours, 2 the two or three of all eight with the lowest SATD -- the
preset 3/4 default --, 3 all eight; 1.2-4.2 points of BD-rate against x264 medium for
30-45 % more time, and from preset 3 up it also refines each partition of a
16x8 / 8x16 winner and both lists of every B-picture partition (worth 0.4-0.9
points there, see `docs/STATUS.md`), `--refine-cand 0|1`
refines *every* inter candidate's vectors rather than only the winner's, so a
candidate that lost on a quarter-pel error gets a second chance (0.3-0.6 points
of BD-rate for about 40 % more time at preset 4; off by default, see
*Refining every candidate* in `docs/STATUS.md`), `--trellis 0|1|2` controls rate-distortion
level selection (priced by the CABAC coder; x264's values: 1 codes only the
winning macroblock with it, which presets 1-2 do for -4 / -3.7 % BD-rate at
6-8 % of the time, 2 runs it inside every candidate too, as presets 3-4 do --
the candidates get its Viterbi pass and the winner the full selection, see
*Rate-distortion level selection* and *Presets 1 and 2* in `docs/STATUS.md`), `--ref 1..8` gives P
pictures more reference pictures (8 at preset 4, 4 at preset 3: -1.45 / -0.97 %
BD-rate against two for a few per cent of time, the partitions choosing
their own reference as x264's mixed references do; see *Two P references*
and *More than two P references* in `docs/STATUS.md`),
from preset 2 up the inter winner's transform size is chosen by rate and
distortion (re-coded with the other size) and from preset 3 up every
candidate's distortion is measured after deblocking its internal edges --
x264's subme 8/9 tools, -0.4 to -0.65 and another -0.3 % BD-rate (see *The
cockatoo / tree gap* in `docs/STATUS.md`; `AEG_T8RD=0` / `AEG_DBRDO=0` turn
them off),
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

### Rate control

* `-q <qp>` (default): constant QP, with x264's picture-type offsets.
* `--crf <f>`: constant quality in x264's form (complexity^(1-qcomp) / rate
  factor; with B pictures the macroblock tree carries the qcomp).  The scale is
  calibrated so that a CRF gives the PSNR x264 `--preset medium --tune psnr`
  gives at the same CRF (preset 3, four clips: -0.03 dB at 0.99x the bits).
* `--bitrate <kbps>`: one-pass average bitrate after x264's `ratecontrol.c`
  (rate factor from the bits spent, overflow correction, a QP step limit), with
  the macroblock tree.  Over 180 frames of five real clips it lands within
  -4 %..+6 % of the target at 0.03-0.48 dB below the CRF curve.  A stream of a
  second or two cannot absorb the first intra picture, and short GOPs (`-g 1`
  to `-g 15` with B pictures) land 6-13 % over in six seconds -- use two
  passes there.  All-intra (`--bframes 0 -g 1`) lands within 3 %.
* `--pass 1` / `--pass 2 --bitrate <kbps>` (`--stats <file>`): two-pass, B
  pictures included.  Pass 1 measures every picture; pass 2 plans every
  picture's QP from that (x264's `init_pass2`) and lands within 1 % of the
  target and 0.15 dB of the CRF curve on the same clips.  Pass 2 has to use
  pass 1's B-picture setting (it refuses another, and damaged statistics).
* `--vbv-maxrate <kbps> --vbv-bufsize <kbit>`: a buffer model on
  ABR or CRF (capped CRF), with x264's per-type size predictor; a picture that
  would still underflow is coded again, coarser.  A stream that fits its buffer
  is byte-identical to the one without.  `--nal-hrd` signals the model
  (hrd_parameters, buffering period and picture timing SEI).
* `--qpmin`, `--qpmax` bound every picture and macroblock QP; `--qcomp <pct>`.

Numbers and the experiments behind them: *Practical encoder features* in
`docs/STATUS.md`.

### Input and stream options

YUV4MPEG2 input (`.y4m`, from a file or a pipe; size, rate, aspect ratio and
range from its header) or raw I420 with `-w`/`-h`; any even frame size (coded
as whole macroblocks, cropped in the SPS); `--fps 30000/1001`, `--sar W:H`,
`--fullrange`, `--colorprim`/`--transfer`/`--colormatrix` in the VUI;
`--deblock a:b`, `--chroma-qp-offset`, `--aud`, `--level`, `--seek`,
`--info-sei` (the options in a user-data SEI), `-g`, `--scenecut`.  A missing,
malformed or out-of-range value is refused with exit status 2.

Other input formats are converted to the 4:2:0 at the coded depth, the way
x264 does: `--input-csp i420|yv12|nv12|nv21|i422|yv16|nv16|
i444|yv24|i400` and `--input-depth 8..16` for raw input (above 8 bits, two
bytes per sample, little-endian: `yuv422p10le` and the like), or a `.y4m`
header's `C` tag (`420p10`, `422`, `444p12`, `mono16`, ...).  The chroma is
resampled to 4:2:0 in 16-bit precision (per field for interlaced material),
then every plane is dithered to the coded depth with x264's error diffusion.
I420 at the coded depth goes through untouched; 8-bit input to a 10-bit
stream is scaled up exactly.

### High 10 (10-bit)

`--bit-depth 10` (alias `--output-depth 10`, x264's name) codes a High 10
profile stream (`profile_idc` 110, 4:2:0, 10 bits luma and chroma) with the
10-bit build of the same encoder: every tool, preset, rate control, the frame
and slice threads, interlace signalling.  The QPs keep the 8-bit scale
(`-q 22` is the same quantiser step relative to the signal at both depths,
unlike x264, whose 10-bit `--qp` counts from QP'Y); CRF and the rate
controls may go below zero, to -12, which is where 10 bits pay off.
`--dump-recon` writes 16-bit little-endian samples (`yuv420p10le`).  At 10
bits only the lookahead goes to a device (`-b opencl` / `-b cuda`, or a GPU
with `-b auto`): its lowres pictures are 8-bit at every depth, so the
search, intra and cost kernels are the 8-bit encoder's, and one more
kernel downsamples the 10-bit luma.  Every other stage stays on the CPU,
since those device kernels are 8-bit.  The SSE2/AVX2 kernels have 16-bit-sample forms (SAD,
SATD/SA8D, SSD, the 4x4 and 8x8 transforms and quantisers, the half-pel,
quarter-pel and chroma filters), each bit-exact with the C path, so 10 bits
cost about what 8 bits do: 0.90-1.11x the 8-bit time at default threads,
1.15x on one thread (it was 1.5-2.1x and 2.1x without them).

What 10 bits buy on 8-bit sources (the five test clips, preset 3, QP 22-37,
the decoded stream against the source, PSNR of all three planes in the 8-bit
scale): **-4.2 %** BD-rate on average against the same encoder at 8 bits
(vtest -2.3, cock -4.8, mega -7.4, short -5.1, tree -1.5) -- less rounding in
the prediction loop and finer quantisers.  Every 10-bit configuration in the
test suite decodes in FFmpeg to exactly the encoder's reconstruction.

`--tff` / `--bff` (or a `.y4m` header's `It` / `Ib`) mark the stream as
interlaced with that field order: the pictures are still coded as frames
(what x264 calls `--fake-interlaced`), and the SPS (`frame_mbs_only_flag 0`)
and a picture-timing SEI on every picture (`pic_struct` 3 or 4) tell a
decoder or player to treat them as two fields.  The height must be a
multiple of 4.

## Using the library

`include/aeg/aeg.h` is the whole interface; `examples/encode.c` is a complete
program that uses nothing else.  After `make install`:

```sh
cc app.c $(pkg-config --cflags --libs aeg)            # libaeg.so
cc app.c $(pkg-config --cflags --static --libs aeg)   # with only libaeg.a
```

```c
aeg_params p;
aeg_params_default(&p);
p.width = 1920; p.height = 1080; p.fps_num = 30; p.fps_den = 1;
p.rc_mode = AEG_RC_CRF; p.crf = 23; p.preset = AEG_PRESET_MEDIUM;
aeg_encoder *e = aeg_encoder_open(&p);         /* NULL: aeg_last_error() */
aeg_packet pkt;
aeg_encoder_headers(e, &pkt);                  /* SPS/PPS for avcC */
/* per picture: packed I420 ... */
int rc = aeg_encoder_encode(e, i420, size, pts, &pkt);
/* ... or three planes with their own strides (same bytes out) */
rc = aeg_encoder_encode_planes(e, plane, stride, pts, &pkt);
/* ... or anything else: NV12, 4:2:2, 10-bit, grey ... (converted) */
aeg_image img = { AEG_CSP_I422, 10, { y, u, v }, { ys, us, vs } };
rc = aeg_encoder_encode_image(e, &img, pts, &pkt);
/* 10-bit stream: p.bit_depth = 10 before open; the reconstruction
 * (aeg_encoder_recon) is then uint16_t samples, strides in bytes */
if (rc == AEG_OK) write(pkt.data, pkt.size);   /* AEG_EAGAIN: no packet yet */
while (aeg_encoder_flush(e, &pkt) == AEG_OK) write(pkt.data, pkt.size);
aeg_encoder_close(e);
```

* **Packets** are Annex-B access units, in decoding order, owned by the
  encoder until the next call on it.  `pts` is the caller's; `dts` is derived
  (the display time of the picture with that coding index, `bframes` frames
  early, as x264 does), strictly increasing and never above `pts` for
  constant-rate input.  `params.timebase_num/den` gives the pts unit (default:
  one tick per frame).
* **Parameter sets**: `aeg_encoder_headers()` returns the SPS and PPS that
  every IDR repeats in-band, right after open, for containers that keep them
  out of band.  `aeg_encoder_get_params()` returns the parameters the encoder
  actually runs with: every "-1 = as the preset says" resolved, incompatible
  options turned off (CAVLC drops B pictures), the level that was chosen.
* **Logging**: `aeg_set_log_callback(fn, opaque)` receives every message as
  one line with its level (`AEG_LOG_ERROR` / `WARNING` / `INFO` / `DEBUG`);
  `aeg_set_log_level()` filters them.  The library default is `WARNING`, so a
  clean encode prints nothing (`aegenc -v 1` shows `INFO`, `-v 2` every
  picture).
* **Threads**: an encoder is used by one thread at a time, but any number can
  run in parallel threads and each produces the bytes it produces alone
  (checked by `tests/api_check.c`, also under ThreadSanitizer in CI);
  `aeg_last_error()` is per thread.
* **Bit depth**: `params.bit_depth` 8 (or 0) or 10 picks the encoder build;
  anything else, or 10 in a `BIT_DEPTHS=8` library, fails in open with a
  message.  10-bit I420/YV12 images are coded as they are (a sample above
  1023 is clipped); everything else is converted.
* **ABI**: the shared library exports only the `AEG_API` functions (checked
  against the header by `tests/install_check.sh`).  While the soname is
  `libaeg.so.0` the struct layouts may still change between releases: rebuild
  against the new header.

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
  independent decode, and the device's whole-frame search to be the same bytes
  as that search on the CPU backend.  No GPU at hand: `. tools/pocl_env.sh`
  gives a real OpenCL device that runs on the CPU (installed with pip into
  `~/.cache/aeg-pocl`, no root), and `make OPENCL=1` then runs every device
  path for real.
* `tests/backend_check.c` checks every device operation's host side
  (transfers, layouts, two threads into one context) against the CPU reference
  on whatever device loads, and `tests/nvrtc_check.c` compiles the CUDA kernels
  with NVRTC for every architecture (`AEG_NVRTC_LIB=` pointing at a
  `libnvrtc`, e.g. from `pip install nvidia-cuda-nvrtc-cu12`); both are in
  `make test` and skip without a device / NVRTC.
* `tests/api_check.c` (in `make test`) uses only the public header: log
  callback and levels, refused arguments and the per-thread error message,
  resolved parameters, headers equal to the first IDR's, strided planes
  byte-identical to packed input, `dts` order, and six encoders in parallel
  threads byte-identical to each alone.  `make install-check` installs into a
  scratch directory and builds `examples/encode.c` against it through
  pkg-config, shared and static, compiles the header as C99 `-pedantic` and as
  C++, and compares the exported symbols with the header.
* `.github/workflows/ci.yml` runs all of it on every push with no GPU: the
  CPU suite and frame threads, the packaging checks and a real `make
  install`, the OpenCL backend on PoCL (`backend_check`, the conformance suite
  through the device, `gpu_validate.sh`), the CUDA kernels through NVRTC, an
  ASan + UBSan build of the CPU and OpenCL host paths, ThreadSanitizer on
  the parallel-encoder and frame/slice-thread paths, and the fuzzers.
* `tests/fuzz/` (`make fuzz`): fuzz targets for the library API with
  arbitrary parameters, `aeg_encoder_encode_image` with arbitrary plane
  layouts, and the CLI's Y4M, two-pass statistics and frame-rate parsers,
  all under ASan + UBSan. They build with clang's libFuzzer
  (`FUZZ_ENGINE=libfuzzer CC=clang`) or, with gcc, with a small standalone
  driver. Five bugs found so far, all fixed (integer overflows in the VUI
  and in QP offsets, an unbounded allocation in the statistics reader, a
  frame rate of 0 accepted); see `tests/fuzz/README.md`.
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
examples/encode.c   a complete program using only the public API
aeg.pc.in           pkg-config template (make install fills in the paths)
src/h264.h          internal data structures and prototypes
src/bitdepth.h      pixel/dctcoef types and constants of one bit-depth build
src/api.c           the public encoder calls: dispatch to the 8- or 10-bit build
src/api_impl.h      the names each build's encoder.c defines
src/input.c         input conversion (layouts, chroma resampling, dither)
src/lowres.c        the lookahead's half-resolution planes (per depth)
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
tests/backend_check.c every device operation end to end on a real device
tests/nvrtc_check.c the CUDA kernels through NVRTC (no GPU needed)
tests/api_check.c   the public API alone (see above)
tests/fuzz/         fuzz targets, standalone driver, seed corpora (README)
tests/install_check.sh  make install, pkg-config, exports, header as C/C++
tools/pocl_env.sh   a real OpenCL device without a GPU (PoCL), for the above
tools/identity.sh   byte-compare two aegenc builds over fourteen configurations
cli/aegenc.c        command line front end
tests/              synthetic clip generator, single-MB probe, test runner
```

## Licence

MIT (see `LICENSE`). Several algorithms follow x264 (GPL) closely: its
methods and its tuned constants, in this project's own code.
`docs/PROVENANCE.md` lists what follows which source, with a
token-similarity scan (`tools/simscan.py`).

The three code-level correspondences the scan found have been rewritten
from their definitions, with no change in output:
- the input dither;
- the motion-search lambda table;
- the deblocking edge filters.

The licence review is the owner's decision; it is not settled here.
