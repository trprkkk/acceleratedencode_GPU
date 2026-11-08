# Response to the external audit (DeepSeek review of `5929971`)

The review was written against commit `5929971`; this answer is against
`043f23d` and later. Each point is classified as one of:

- **fixed**: a defect, now corrected (with the commit).
- **outdated**: true at `5929971`, no longer true.
- **confirmed**: true, and documented as a limit rather than "fixed".
- **needs hardware**: cannot be settled without a GPU.
- **owner decision**: a scope, design or legal question for the project
  owner.

The invariants that every fix had to keep still hold:

- Every stream is byte-identical to `02a03e3`. This was checked on 15
  configurations: presets 0-4, frame threads 1/3/5, slices, CRF/ABR/VBV,
  PAFF/MBAFF/mode 4, 10-bit and CAVLC.
- `make test` passes.
- On PoCL, the OpenCL conformance suite, `backend_check opencl` (312 checks)
  and `tools/gpu_validate.sh` pass.
- ASan and LSan report nothing.

## Concrete defects

| # | Finding | Verdict | What changed |
|---|---|---|---|
| 1 | The encoder `mb_encs` allocation is not NULL-checked | **fixed** (`043f23d`) | It is added to the out-of-memory check in `aeg_encoder_open`. |
| 2 | The timed dispatcher can never put a stage on the GPU | **fixed** (`043f23d`) | See note 2. |
| 3 | `launch_ns` is added on top of a measurement that already contains it | **fixed** (`043f23d`) | The branch that did this is gone (note 2). |
| 4 | The calibration result is unused; the device-info overhead fields are never set | **fixed** (`1e39497`, `043f23d`) | See note 4. |
| 5 | `have_gpu` ignores `is_gpu`: `-b auto` took an OpenCL CPU device (PoCL) | **fixed** (`1e39497`) | See note 5. |
| 6 | Motion estimation is reported as "gpu" | **fixed** (`043f23d`) | See note 6. |
| 7 | PSNR statistics are incomplete or wrong | **fixed** (`043f23d`) | See note 7. |
| 8 | `get_params` reports `frame_threads` as `ft_on ? 2 : 1` | **fixed** (`043f23d`) | It now reports the number of workers in use. |
| 9 | CLI and header text contradict the code | **fixed** (`043f23d`) | See note 9. |
| 10 | Portability and style: `_mm_storeu_si32`, a variable named `main`, `verbose` unused | **fixed** / **confirmed** | See note 10. |
| 11 | Device transfers are unchecked; empty batches are not guarded | **fixed** (`1e39497`) | See note 11. |
| 12 | No way to choose among several devices | **fixed** (`1e39497`) | See note 12. |
| — | `libOpenCL` is opened and closed per context (LSan reports the ICD loader's platform list each time) | **fixed** (`043f23d`) | It is now loaded once per process with `pthread_once`. CUDA still opens per context. |

**Note 2: the dispatcher.** The generic branch could never fire, for two
reasons:

- Every stage starts on the CPU, and a stage is only timed on the side it
  ran on.
- The only stage with a device form (inter prediction) rebuilds every
  sub-sample plane per call, so it could not have won anyway.

It is replaced by an explicit rule. Under `auto`, only the TQ/IQ coefficient
stages of P and B pictures are placed, by timing whole pictures both ways.
Every other stage runs on the CPU unless `-s` forces it.

**Note 4: calibration.** Calibration is kept per backend:

- `aeg_query_devices()` reports `launch_overhead_ns` and `bytes_per_ns` for
  the device that was measured.
- `--calibrate` prints them.

The dispatcher does not use the calibration. Its one automatic decision
times whole pictures, so every launch and copy is already inside the
measurement.

**Note 5: GPU detection.** There was also a latent bug here:
`CL_DEVICE_TYPE` (a 64-bit `cl_bitfield`) was read into 4 bytes. That query
fails, so every device, including a real GPU, read as a non-GPU.
`clGetDeviceIDs` was also declared with a 32-bit `device_type`. Now:

- `-b auto` takes only a GPU: a CUDA device, else an OpenCL GPU.
- PoCL is used only with `-b opencl`.

**Note 6: the motion-estimation report.** The seed rounds do run through the
device when it is chosen, but each macroblock's search runs on the CPU. The
stage report now prints the device's share of a stage's time ("cpu, 67% of
it on the device") instead of one word.

**Note 7: PSNR.** At `5929971`:

- `psnr_avg` held the *last picture's* PSNR, and the CLI printed it as the
  summary.
- `psnr_y`, `psnr_u`, `psnr_v` and `ssim_y` were never set.
- The field path was not timed, and the frame path added 0.

Now one timed helper serves the frame, field and frame-thread paths:

- The PSNR is the encode's global PSNR, computed from the summed squared
  error (x264's "global" convention).
- The per-plane errors travel through the frame-thread queue.
- The values match an independent ffmpeg decode to 0.01 dB, on the frame
  path and on the PAFF path.
- SSIM (x264's 8x8 windows on a 4-sample grid) is opt-in via
  `--ssim` / `aeg_params.ssim`.

**Note 9: help and header text.**

- `--subme`: the help now says 0 = whole-sample, 1 = quarter-sample, and 2-9
  are accepted and act as 1.
- `aq_strength` is documented in tenths.

**Note 10: portability and style.**

- `st32(_mm_cvtsi128_si32())` replaces `_mm_storeu_si32`, which GCC < 9
  lacks.
- The variable `main` is renamed `is_main`.
- `aeg_params.verbose` remains documented as unused. The CLI's `-v` drives
  `aeg_set_log_level`.

**Note 11: device transfers.**

- Every OpenCL/CUDA host-device copy is checked. A failure returns
  `AEG_EBACKEND`, and the guard reruns the CPU reference.
- Zero-sized batches and empty planes return at once.
- `sse_batch` of a plane smaller than one 8x8 tile returns 0, as the CPU
  reference does.

**Note 12: device selection.**

- `--device N` / `aeg_params.device` picks a device:
  - OpenCL: every device of every platform, in order.
  - CUDA: the device ordinal.
- `--list-devices` shows `#N`.
- If an explicitly requested backend or device is missing, there is a
  warning instead of a silent fallback.

## Broader claims

**"It is an H.264 encoder only; there is no HEVC / x265-class encoder."**
*Confirmed.*

- x265 appears only as a comparison point. The README now says so in its
  first lines.
- Writing an HEVC encoder is a separate project of the same size as this one
  (CTUs/quadtrees, the HEVC CABAC and transforms, SAO, and so on).
  **Owner decision.**

**"The GPU code has never run on a GPU."** *Confirmed.*

- PoCL is an OpenCL runtime on the CPU. It proves the host side, the buffer
  layouts and the kernels' arithmetic: every device path writes the CPU
  encoder's stream byte for byte.
- The CUDA kernels are only compiled (NVRTC, every architecture).
- Speed, and whether any placement ever wins, are **needs hardware**.

**"The auto dispatcher never selects the GPU."** *Partly outdated, partly
confirmed.*

- On a machine with a GPU, the coefficient stages are tried both ways and
  kept on the faster side.
- On PoCL that side is always the CPU, as expected.
- The timed generic branch the review saw was dead and has been removed
  (defect 2).

**"Per-macroblock blocking transfers and launches; this cannot be fast on a
GPU."** *Confirmed.*

The device paths are batches inside the CPU encoder's macroblock loop. Each
call uploads, launches and reads back synchronously. Device lanes let frame
and slice threads overlap their calls, but nothing stays resident on the
device.

A design that could plausibly win would keep the references and the source
on the device and run the whole-frame stages there, batching a picture at a
time:

- lookahead / lowres,
- the motion-search rounds,
- the interpolation planes,
- possibly the RD of whole rows.

That is a redesign of the data flow, not a tuning pass. It should be
measured on real hardware first. **Owner decision + needs hardware.**

**"CRF on long clips is +11.8 % against x264 veryslow."** *Outdated.*

After QP-RD (`5929971`) and x264's CABAC trellis (`bb7d271`, `02a03e3`), the
same three clips give:

- preset 4: +0.03 % (cockL +2.85, vtestL -4.19, megaL +1.43);
- preset 3: +4.31 %.

The method is the same, with `--aq 0` against x264 `--tune psnr`.

**"Missing x264 features."** *Confirmed in part.*

Missing:

- psy-rd / psy-trellis (they lower PSNR by design; the project tunes for
  PSNR);
- 4:2:2 / 4:4:4 coding and lossless;
- open GOP;
- temporal direct and chroma ME in field pictures;
- weighted prediction in MBAFF frames;
- subme > 1 as distinct levels (x264's subme 8-10 tools exist but are tied
  to presets: RD refinement, `--qp-rd`).

Present since the review's baseline: CABAC trellis, including the DC blocks.

**"Portability: Windows / macOS / ARM, GCC 7."** *Confirmed.*

- Linux x86-64 with GCC 12 is the only platform built and tested. CI is
  Ubuntu.
- The GCC 7 build break (`_mm_storeu_si32`) is fixed.
- There is no NEON code. The C fallback is the portable path.
- Windows would need a CMake/MSVC build and replacements for `pthread`,
  `dlopen` and `sysconf`. **Owner decision.**

**"Licence and provenance."** **Owner decision.**

The code is MIT-licensed and written for this project. However, several
algorithms follow x264 (GPL-2.0) and FFmpeg (LGPL/GPL) closely, and the
comments name them:

- the trellis,
- the macroblock tree,
- the scene-cut test,
- weightp analysis,
- QP-RD.

Tables come from ITU-T H.264. Whether any part is close enough to the GPL
sources to be a derivative work needs a review by the owner (or counsel)
before an MIT release. This document does not claim it is clean.

**Since then: `docs/PROVENANCE.md`** lists, area by area, which file and
function follows which x264 or FFmpeg algorithm, and which x264-specific
constants it uses. It also runs a MOSS-style token scan
(`tools/simscan.py`) against x264, both current and a 2016 version, and
against FFmpeg n7.0.2. The findings:
- **One function is x264 code re-typed:** `aeg_dither_plane` (the
  high-bit-depth input dither, from x264's `filters/video/depth.c`). The
  statement above that all code was "written for this project" is
  therefore wrong for that function.
- **Two smaller correspondences:**
  - the ME lambda table equals `x264_lambda_tab`; it is computable from
    its formula;
  - the deblocking edge filters use the same idiom of the standard's
    equations as x264 and FFmpeg.
- **The algorithm cores** (trellis, mbtree, weightp, AQ, rate control,
  B-adapt, ME) follow x264's methods and often its tuned constants, but
  show no code correspondence.

The memo is not a legal judgement; it is meant to make the review quick.

**"Untested areas."** *Partly confirmed.*

- Not covered:
  - any real GPU, including machines with several;
  - non-Linux builds.
- Covered: CPU, frame and slice threads, PoCL, NVRTC, ASan/UBSan, TSan,
  install/pkg-config, and 76 conformance cases decoded by FFmpeg.
- **Since then:** fuzzing of the library API, the image layouts and the
  CLI's parsers (`tests/fuzz/`, `make fuzz`, and a CI job under libFuzzer).
  It found five bugs, all fixed with no in-range stream change:
  - the VUI `time_scale` overflowed;
  - the SAR reduction overflowed;
  - `--fps` accepted a rate of 0;
  - the B/I QP offsets were unbounded and overflowed the QP arithmetic;
  - the statistics reader made an unbounded allocation.

  See `tests/fuzz/README.md`.

## Open items for the owner

1. **HEVC**: in scope or not?
2. **GPU data flow redesign** (resident references, whole-picture stages)
   before or after hardware is available?
3. **Feature gaps** (4:2:2/4:4:4, lossless, open GOP, field-picture direct):
   which, if any?
4. **CMake / Windows / ARM** support.
5. ~~**Fuzzing**~~: done (`tests/fuzz/`; see above).
6. **Licence review** of the x264-derived algorithms. The facts are in
   `docs/PROVENANCE.md`; start with `aeg_dither_plane`.
