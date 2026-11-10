<!--
  What this encoder has and what it does not, against x264's and x265's feature
  lists.  Every "yes" is something a test in tests/run_tests.sh exercises; every
  "no" is something the API does not pretend to have.
-->

# Features: what is here, what is not

x264's and x265's command lines are the honest yardstick, so this is written
against them.  A feature counts as present only when it is on the command line,
in `aeg_params`, and covered by a case in `tests/run_tests.sh`.

## Present and tested

| | |
|---|---|
| coding tools | I_PCM, I_NxN (Intra_4x4/Intra_8x8), I_16x16, P_16x16/16x8/8x16/P_8x8 with 8x4/4x8/4x4 sub-partitions, B_Direct, B_16x16 with list choice and B_8x8, weighted P prediction (explicit, per picture), CABAC and CAVLC, 4x4 and 8x8 transforms, the Intra_16x16 and chroma Hadamards, trellis level selection (x264's 0/1/2), in-loop deblocking with the offsets |
| rate control | CQP, CRF (with the macroblock tree's lookahead), ABR (one and two pass), VBV/HRD with recode, qp_min/qp_max/scenecut, adaptive quantisation, QP-RD (x264's subme 10), `--zones` and `--qpfile` |
| structure | B pictures with the pyramid and b-adapt, multiple P references, 8 GOP shapes with spatial direct, scene-cut detection, slices, frame and slice threading, cropping, interlaced field / PAFF / MBAFF coding |
| profiles | Constrained Baseline to High 10 at both depths; 4:2:0 at 8 and 10 bits, 4:2:2 and 4:4:4 *input* converted to 4:2:0 |
| devices | CPU, OpenCL, CUDA (general-purpose compute only, no fixed-function encoder) |

## Present, verified this branch

| | |
|---|---|
| lossless | `--lossless`: every macroblock is I_PCM, so the reconstruction is the source exactly.  An all-I CAVLC stream (I_PCM is an I-slice macroblock type), 8.05 bits a sample at 8 bits, exact at 10 bits too.  H.264's cheaper lossless (transform bypass) is not implemented, so this is the expensive kind of lossless.  `tests/run_tests.sh`'s lossless section checks the reconstruction is the source *and* that an independent decode is the source, at 8 and at 10 bits. |
| QP zones | `--zones b,e,q[,b,e,q]` and `--qpfile` (x264's `<picture> <qp>` format).  The offset is applied after the rate control has decided, so the controller still sees what the zone cost. |

## Absent

These are the ones a review would find missing.  They are listed here rather
than faked; each is a chunk of work with its own syntax, and none of them is
within a line of the encoder.

| | what it needs |
|---|---|
| weighted B prediction | `weighted_bipred_idc = 1` in the slice header, the two-list `pred_weight_table`, the weights applied to each list's prediction, and the bi-predictive combination's formula (H.264 8.4.2.3.2).  The single-list half exists and is tested (`weightp`), so the missing part is the B table and the bi-pred combination. |
| custom quantisation matrices | the `seq_scaling_matrix` / `pic_scaling_matrix` syntax in the SPS and the PPS, with the fallback rule, and the quantiser's per-position scale replaced by the matrix.  The 4x4 quantiser is a three-class lookup (`aeg_quant_mf[qp%6][cls]`), so the matrix is a substitution there and in the dequantiser. |
| open GOP | a non-IDR I picture at the keyframe interval, and the first B pictures of the group referencing the previous group's pictures.  The reference-list management is the work; the syntax is nothing. |
| 4:2:2 / 4:4:4 output | chroma_format_idc 2 and 3 in the SPS, the chroma at full horizontal or both resolutions (four 8x8 chroma blocks per macroblock instead of two), the chroma motion compensation and prediction at that resolution, and the Cabac/CAVLC chroma contexts.  The *input* is converted to 4:2:0 today. |
| 12-bit output | x265's territory; the library compiles at 8 and 10 bits only. |
| intra refresh, adaptive intra | not implemented. |
| lossless via transform bypass | the cheaper H.264 lossless; `--lossless` uses PCM because bypass needs the `qpprime_y_zero_transform_bypass` machinery and the residual path. |

## Known quality and speed gaps

`docs/GPU_STATUS.md` carries the measurements.  In short:

* 11.42 % BD-rate behind x264 medium, 22.87 % behind x265 -p 4 (aegenc needs
  the bits), at constant QP on a 720p clip.  The gap is uniform per plane, so
  it is a rate-distortion gap and not a broken chroma path.
* 6x slower than x264 veryslow at preset 4 (4.89 s against 0.79 s for 30
  frames of 720p), and the motion search is 79 % of the encode.
* on the machine the device numbers were measured, a device round trip costs
  30-90 µs, so the device loses for per-macroblock batches and wins for
  picture-sized ones.
