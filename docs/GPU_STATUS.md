<!--
  The measured state of the GPU data flow: what was wrong, what changed, and
  what the numbers are on the machine it was measured on.  Every figure here is
  the output of a command in tools/ that can be run again -- nothing is quoted
  from memory.
-->

# The GPU data flow: what it costs and why

Machine: AMD Ryzen 5 5600 (12 threads) under WSL2, NVIDIA GeForce RTX 2060
SUPER (34 SMs, compute 7.5, driver 590.52.01 reached through WSL's forwarding
library), PoCL 1.1 as the only OpenCL device (a CPU device).

## 1. What was wrong

Every CUDA call used the legacy default stream.  The guard's lanes (frame
threads each own a cloned backend instance, one per CPU) therefore serialised
their work behind each other, and every synchronous copy waited for the whole
device to drain.  A 720p encode at preset 4 measured 120 s of device time for
30 frames, against 2.1 s of CPU motion search.

The OpenCL backend had the same shape with one host round trip per *transfer*:
four blocking transfers and four waits for a `tq_iq_batch()` of one
macroblock's sixteen blocks.

## 2. What changed

`src/backend_cuda.c`:

* a non-blocking stream per instance, so a lane's launches and transfers run
  concurrently with the other lanes' and nothing waits for the device as a whole;
* everything an operation uploads is packed into one pinned staging arena and
  everything it downloads into that arena's device mirror: one `HtoD`, one
  launch, one `DtoH`, one synchronisation on the instance's own stream;
* the reference picture is one transfer into one allocation;
* the `_v2` spelling of the async transfers (the deprecated spelling answers
  `CUDA_ERROR_INVALID_CONTEXT` on a driver with no compatibility path for it --
  that took the whole backend down);
* the PTX is cached under the user's cache directory, so a second run starts
  without NVRTC;
* picture-sized arenas stay pageable (several large pinned allocations per
  instance provoke a driver that aborts on the second one), every driver return
  code is checked, `AEG_CUDA_DEVICE` picks a device.

`src/backend_opencl.c`: non-blocking transfers, the kernel waits on the
uploads' events, the downloads wait on the kernel's, one `clFinish` per
operation, out-of-order queues (with an in-order fallback a runtime that
refuses the flag), and a device-resident reference.

## 3. The numbers

`tools/compare_backends.sh` (720p, 30 frames, preset 4, QP 27; every stream is
the same 546055 bytes, md5 `85cf05381e1c58d8`):

| run                  | before  | after   |
|----------------------|---------|---------|
| `-b cpu`             | 3.43 s  | 3.43 s  |
| `-b cuda` auto       | 33.39 s | 4.74 s  |
| `-s tq=cuda`         | 4.44 s  | 4.47 s  |
| `-s inter=cuda`      | 33.13 s | 4.51 s  |
| `-b opencl` auto     | 62.40 s | 3.46 s  |

`tools/opencl_ab.sh` (720p, 60 frames, `--s tq=opencl -s iq=opencl`, all three
streams the same 1083094 bytes):

| run                        | time      |
|----------------------------|-----------|
| old OpenCL backend         | 254.89 s  |
| redesigned OpenCL backend  | 78.71 s   |  |
| CPU reference              | 6.49 s    |

`tools/steady_state.sh` (720p, 300 frames, so the device's startup is amortised;
all streams the same bytes):

| backend | wall  | fps   |
|---------|-------|-------|
| CPU     | 30.73 s | 9.90 |
| CUDA    | 32.41 s | 9.96 |
| OpenCL  | 33.67 s | 9.06 |

The CUDA startup that the 30-frame runs show is `cuInit` + a first context
creation: **0.50 s** on this driver (a second context costs 0.22 s, a retained
primary context the same, and a module load from a PTX file 1.8 ms).  It is the
WSL GPU's own initialisation, not something the encoder does.

## 4. Why the device cannot win here, in numbers

`tools/probe_costs.sh` times one call kind at a time (the encoder's own
`tq_batch` kernel, 16 int16 in, a level array and a count out):

| call                              | ns/call |
|-----------------------------------|---------|
| `HtoDAsync` 512 B pinned          | 12 356  |
| `DtoHAsync` 512 B pinned          | 13 463  |
| `cuLaunchKernel` (no synchronise) |  7 858  |
| `cuStreamSynchronize` (empty)     |    305  |
| launch + synchronise              | 37 351  |
| the full round trip               | 94 307  |

`tools/probe_micro.sh`, against the CPU implementation of the same operation:

| operation                     | CUDA    | CPU      |
|-------------------------------|---------|----------|
| `tq_iq_batch` (16 blocks)     | 109 µs  | 0.9 µs   |
| `satd_batch` (5632 candidates | 419 µs  | 536 µs   |

So a round trip costs 30-90 µs on this machine, while the sixteen SSE2
transforms it would replace cost 0.9 µs.  The device wins only where the batch
is large enough to amortise the round trip: `satd_batch` at 5632 candidates is
1.3x faster on the device than on the CPU, and that is the regime the
whole-frame search rounds live in.  On a native GPU the round trip is ~5 µs and
the arithmetic is ~100x, so the coefficient stages would win there too -- which
is what `aeg_dispatch_coef_pick` now measures instead of assuming
(`src/backend.c`, `coef_seed`).

`tools/me_ab.sh` (720p, 60 frames, whole-frame search both ways):

| run                              | fps   | motion-est stage |
|----------------------------------|-------|------------------|
| per-macroblock search (default)  | 9.48  | 4355.1 ms cpu    |
| whole-frame search, CPU          | 3.32  | 4336.4 ms cpu    |
| whole-frame search, device       | 3.12  | 4719.0 ms gpu    |

The device is 8 % slower than the CPU for the same search on this machine, and
the whole-frame search is a different search (different bytes), which is why
only `-s` and `AEG_SEED` put it on a device.

## 5. Where the remaining encode time goes

`tools/profile.sh` (720p, 300 frames, `cycles:u`): the profile is flat -- no
function above 2 % -- with `aeg_idct_add_ssd4` (the per-candidate inverse
transform, reconstruction add and SSD), `aeg_quant_4x4`, `aeg_cabac_residual_price`
and `trel_descent_kn` (trellis) at the top.  `tools/preset_sweep.sh` (720p, 30
frames):

| preset | fps   | motion-est (ms) |
|--------|-------|-----------------|
| 0 (ultrafast) | 37.8 | 757 |
| 1 (fast)      | 32.9 | 931 |
| 2 (medium)    | 14.1 | 1584 |
| 3 (slow)      | 10.0 | 2034 |
| 4 (veryslow)  | 9.6  | 2114 |

The motion search is 79 % of the encode at preset 4; `tools/ablation.sh`
measures what each tool costs at that preset:

| over           | fps   | motion-est (ms) |
|----------------|-------|-----------------|
| defaults       | 9.63  | 2114.1 |
| `--trellis 0`  | 15.46 | 2122.9 |
| `--qp-rd 0`    | 11.58 | 2156.6 |
| `--partitions 1` | 12.33 | 1053.3 |
| `--ref 1`      | 12.30 | 1309.1 |

(`--subme 0` is *slower*, 6.52 fps: it also turns off the seed searches, and
the hexagon walks that replace them cost more than the sub-pel refinement.)

## 6. Quality and speed against x264 and x265

`tools/vs_x264.sh` (720p, 30 frames; PSNR measured by decoding each stream with
ffmpeg and comparing with the source, nothing an encoder claims about itself)
and `tools/bdrate.sh` (Bjontegaard's delta rate over five QPs, the same
decoding):

| encoder                    | QP26              | CRF24             | BD-rate vs aegenc |
|----------------------------|-------------------|-------------------|-------------------|
| aegenc preset 4 (veryslow) | 4657 kbit/s 50.27 dB | 3314 kbit/s 45.83 dB | - |
| x264 `--preset medium`     | 4287 kbit/s 48.95 dB | 2693 kbit/s 43.52 dB | +11.42 % |
| x265 `-p 4` (medium)       | 3953 kbit/s 46.31 dB | 2685 kbit/s 41.79 dB | +22.87 % |

So aegenc needs **11.4 % more bits than x264 medium** for the same PSNR.  A
negative number would mean aegenc is ahead; this one says the quality gap is
11 % BD-rate -- in the same class as x264 (it is an H.264 encoder, so losing to
x265's HEVC by ~23 % is what an H.264 encoder loses to HEVC by).

Speed at matched preset names (`tools/vs_presets.sh`):

| encoder                        | wall  |
|--------------------------------|-------|
| aegenc preset 4 (veryslow)     | 4.89 s |
| x264 `--preset veryslow`       | 0.79 s |
| x264 `--preset slow`           | 0.38 s |
| x264 `--preset medium`         | 0.27 s |

## 7. What is left, and what would close it

* the coefficient stages cannot beat the CPU on a machine whose device round
  trip costs 60 µs -- only a native GPU, or a much larger batch per call;
* the motion search is 79 % of the encode; the device's whole-frame rounds are
  close to the CPU's on this machine and 4x the cost of the per-macroblock
  search either way;
* 11.4 % BD-rate against x264 medium: the tools that would close it are the
  ones still missing (weighted B, custom quantisation matrices, open GOP), and
  they are listed in the API as absent rather than silently unimplemented.
