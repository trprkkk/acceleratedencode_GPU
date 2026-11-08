# Provenance notes: what follows which source

This memo lists which parts of the encoder follow which outside algorithm or
code, and how closely. It was written for the owner's licence review (open
item 6 in `docs/AUDIT_RESPONSE.md`).

**This is not a legal opinion.** It does not decide whether any part is a
derivative work of x264 (GPL-2.0-or-later) or FFmpeg (LGPL-2.1-or-later),
or whether the MIT licence in `LICENSE` can stand as is. That decision is
for the owner, with counsel if needed. What the memo can do is make the
facts easy to check:
- where each idea came from;
- what was taken (a formula, tuned constants, a code structure);
- what a mechanical similarity scan says about the code itself.

Patents are a separate question from copyright: H.264 is covered by patent
pools whatever the code's licence, and nothing here addresses them.

## How this was assembled

1. **The source comments.** Every place that follows x264 or FFmpeg says so
   by name (`grep -n -i "x264\|ffmpeg"` over `src/`, `cli/` and `include/`):
   151 mentions, 82 of them in `src/encoder.c`. The tables below are built from them.
2. **The development notes.** FFmpeg's H.264 decoder (n7.0.2) was read
   directly for the decoder-side derivations: MBAFF neighbours, spatial
   direct, field deblocking, and the CABAC context details listed in
   *FFmpeg* below. The notes do not record the x264 source being read during
   development. The x264 behaviour came from its documentation, its
   command-line semantics, and the developer's prior knowledge of x264. That
   developer is an AI model, and prior knowledge of this kind can reproduce
   code closely: `aeg_dither_plane` below shows it. So the absence of a
   record is not evidence of independent writing.
3. **A mechanical scan** (`tools/simscan.py`). Every C function here, after
   stripping comments and strings, is cut into runs of *k* consecutive tokens,
   and each run is looked up in the other code base. This is what MOSS-like
   tools measure. Two modes:
   - **normalised:** identifiers become `I`, numbers `N`; this compares the
     shape of the code;
   - **raw:** names and numbers kept; this finds code copied with its names.

   The scan was run against x264 at `c24e06c` (github.com/mirror/x264) and
   FFmpeg `n7.0.2` (`libavcodec/h264*`, `cabac*`, `golomb*`). Re-run with:

   ```sh
   git clone --depth 1 https://github.com/mirror/x264 /tmp/x264
   tools/simscan.py /tmp/x264 --raw --k 12 --min-tokens 40   # copied with names
   tools/simscan.py /tmp/x264 --k 15 --min-tokens 40         # same shape
   ```

   Some code follows an older x264 (b-adapt 1, below), so the scan was
   repeated against x264 `24f25b6` (2016-04-03, the last version with that
   rule). The result was the same: the deblock idiom, `aeg_dither_plane`
   and `la_group_cost`, and nothing else beyond noise. `b_adapt_decide`
   scores 0.8 % normalised and shares no 12-token run raw, so it follows
   that version's rule and constants in its own code.

   **How to read a score.** A high score is a reason to put the two
   functions side by side, not a finding. Butterflies, filters and table
   lookups that the standard fixes come out alike whoever writes them, and
   generic runs (`a = 0; b = 0; ...`) match unrelated code. A low score does
   not prove independent creation either: an algorithm can be followed
   closely in different code.

## Scan results in brief

**Against x264, raw tokens (k = 12).** The highest function is 25 %. The
only pairs above 15 % are:
- the four deblocking edge filters, 21–25 % against x264's
  `deblock_edge_*_c`. Those are the standard's equations with the
  standard's names (`p0`, `q1`, `alpha`, `tc`); x264 marks its own versions
  "From ffmpeg";
- `aeg_now_ns` (a `clock_gettime` wrapper), 16 %.

Everything else is under 15 %, and most of the list is noise.

**Against x264, normalised tokens (k = 15).**
- **Two pairs line up with the right x264 function:**
  - `aeg_dither_plane`: 28 % against `filters/video/depth.c` `DITHER_PLANE`;
  - `la_group_cost`: 19 % against `slicetype_path_cost`.
- **The algorithm cores that follow x264 score 0.3–6 %**, and their best
  match is an *unrelated* x264 function (noise):
  - the CABAC trellis (`trel_viterbi` 2.7 %, `tv_level_bits` 4.8 %);
  - mbtree (`mbt_propagate` 2.6 %, `mbt_run` 5.4 %);
  - weightp (`wp_analyse` 5.8 %);
  - AQ (`aq_compute` 4.2 %);
  - ABR/VBV (`abr_anchor_q` 1.4 %, `vbv_update` 4.7 %);
  - B-adapt (`b_adapt_decide` 0.8 %, `b_adapt_trellis` 8.0 %);
  - the UMH search (`umh_search` 7.9 %).

  In other words, these follow x264's *method* in code of a different shape.
- **Higher normalised scores on generic code are noise:**
  - the DC transforms, 57–62 % (butterflies fixed by the standard);
  - `aeg_bw_init` and `aeg_cabac_open`, 47–50 % against
    `x264_param_default` (runs of assignments);
  - the backend function tables.

**Against FFmpeg n7.0.2.** Raw, the highest is 18 %; normalised (k = 25),
34 %. The top pairs are the standard's dequantisation/IDCT, plane
prediction and loop-filter kernels, plus generic runs. Nothing points at a
copied decoder routine.

## Close code correspondence

These are the places worth the owner's attention first.

| where | corresponds to | what matches | notes |
|---|---|---|---|
| `src/input.c` `aeg_dither_plane` (the >8-bit → 8/10-bit input dither) | x264 `filters/video/depth.c`, `DITHER_PLANE` (Sierra-2-4A error diffusion) | Effectively the same code: same variable names (`lshift`, `rshift`, `half`, `pixel_max`, `err`, `errors`), same statements in the same order. The comment says so: "x264's dither_plane_1/_2". | The **clearest case** of x264 code in the tree. It runs only when the input has more bits than the encode. Options: rewrite it from a description of the Sierra-2-4A filter (the result may stay bit-identical; the arithmetic is the filter's); replace it with a different dither (input > encode depth would then change); or keep it and settle the licence question. |
| `src/mc.c` `aeg_me_lambda` table | x264 `x264_lambda_tab` | The 52 values are the same. | They are round(2^((qp−12)/6)), floored at 1, so they could be computed at start-up from the formula instead of written out. The output is identical either way. |
| `src/deblock.c` `filter_luma_edge`, `filter_luma_intra_edge`, `filter_chroma_edge`, `filter_chroma_intra_edge` | x264 `common/deblock.c` `deblock_edge_*_c` (marked "From ffmpeg" there) and FFmpeg `h264dsp_template.c` | The idiom of the standard's equations (8.7.2.3/8.7.2.4): `if (tc0) ... tc++`, the delta clip. | The arithmetic is normative. Rewriting it in another form would be cheap if wanted; it is verified bit-exact by the conformance suite. |
| `src/encoder.c` `la_group_cost` | x264 `slicetype_path_cost` | The B-pyramid pricing: anchor, middle B, then the Bs on each side from the nearer pair. | About 15 lines; the same cost structure in different code. |

## Algorithms that follow x264 (method and constants, own code)

The scan finds no code correspondence for these. They deliberately follow
x264's method, often with x264's tuned constants, because matching x264's
quality was the goal. "Constants" lists the x264-specific numbers used.

| area | where | follows | constants / specifics taken | differences |
|---|---|---|---|---|
| RD lambda | `encoder.c` `lambda_init`, `aeg_lambda_of` | x264's `lambda2` | 0.85·2^((qp−12)/3); high-bit-depth scaling by QP'Y | — |
| ME lambda | `mc.c` `aeg_me_lambda` | x264 `lambda_tab` | the table, see above | — |
| CABAC trellis | `encoder.c` `struct trel`, `trel_viterbi`, `tv_level_bits`, `trel_descent*`, `trellis_*_dc`; `transform.c` `aeg_trel_levels`, `aeg_trel_dc_levels` | x264 `quant_trellis_cabac` | Candidates round(x), round(x)−1, 0; contexts frozen inside the block; the node transitions `tv_next` (these are the standard's level-context state machine, 9.3.3.1.3, so x264's table holds the same values). `--trellis 0/1/2` uses x264's meaning. | Exact-pricing descent pass after the Viterbi (`trel_descent_kn`); own distortion model; DC/chroma DC variants; lambda scaling tuned here. |
| Macroblock tree | `encoder.c` `mbt_propagate`, `mbt_add`, `mbt_group`, `mbt_run`, `mbt_apply`, `mbt_strength` | x264 `macroblock_tree`, `mbtree_propagate_cost` | Propagate fraction 1 − inter/intra; offset −strength·log2((intra+prop)/intra); strength 5·(1−qcomp) | Strength 1.6 by default, measured here (x264's 5·(1−qcomp) is 2.0 at its default qcomp); own lowres estimator; runs on this encoder's group structure. |
| Adaptive quantisation | `encoder.c` `aq_compute` | x264 "variance AQ" (the idea) | `aq_strength` 10 = x264 1.0 | **Different formula:** offset = strength·(mean log2 var − log2 var), zero-mean per picture, clamped ±12. x264 uses a fixed log2 reference instead. |
| CRF / ABR / VBV | `encoder.c` `crf_*`, `abr_anchor_q`, `abr_intra_q`, `rc_update`, `vbv_clamp_qp`, `vbv_update` | x264 `ratecontrol.c`: `get_qscale`, `rate_estimate_qscale`, `predict_size` / `update_predictor`, one-pass ABR | qscale = blurred complexity^(1−qcomp)/rf; qcomp; qpstep 4; the size predictor (coef·satd + offset)/(qscale·count) with halving decay | Own calibration against x264 medium; own two-pass statistics format and pass-2 shaping. |
| Weighted prediction | `encoder.c` `wp_analyse`, `wp_guess`, `wp_cost*`, `wp_threshold`, `wp_finish` | x264 `weights_analyse` (subme 7) | The initial guess (gain from the deviations, offset from the means, truncated), a ±1 search around it, early exit | Measured on a half-resolution 8×8-block search with an intra-like exclusion (`wp_intra_est`); own threshold. |
| B-picture decision | `encoder.c` `b_adapt_decide` (b-adapt 1), `b_adapt_trellis`, `la_group_cost` (b-adapt 2) | b-adapt 1: x264's fast rule **as it was before 2016** (x264 `aa26e88`, 2016-04-10, replaced it with a path-cost comparison). b-adapt 2: `slicetype_path` / `slicetype_path_cost` | The old fast rule's "mostly intra" test, threshold `INTER_THRESH − P_SENS_BIAS·(distance−1)` = 300 − 50·(distance−1) floored at 30, and the one-third-intra stop; the 1/1.2 discount; pyramid path pricing | — |
| Lookahead estimate | `lookahead.c`, `lowres.c` (`aeg_lowres_*`) | x264's lowres lookahead | Half-resolution source, 8×8 per macroblock, min(intra, inter) SATD, border macroblocks excluded; lambda at QP 12 (`LA_LAMBDA`); the directional intra-8×8 modes (subme > 1) | Own search and own field variant. |
| Scene cut | `encoder.c` `scene_ratio`, `scene_cut_lowres` | The `scenecut` parameter's scale (x264) | 120 = 1×, as x264's default | **Different test:** a spike relative to both neighbours. x264 uses an absolute ratio threshold, which fired falsely on fast motion here. |
| Mode-decision gates | `encoder.c` `b_rd_gate` (`mb_analyse_b_rd`), `hpt_on` / `me_block_inner` (`p_halfpel_thresh`), `pref_nref` (mixed refs), partition order (P_8x8 first), P fast skip | x264 `analyse.c` | B early termination at G/16 of the best (x264 G = 17; here 20, measured); the order and the gate ideas | Thresholds re-measured here (see the BD-rate tables in the comments). |
| QP-RD | `encoder.c` `qprd_on`, `qprd_try`, `qprd` | x264 `mb_analyse_qp_rd` (subme 10) | Trying neighbouring QPs on the decided macroblock with the real coder's bits | Own lambda and direction rules (0.65× on P, etc.). |
| Transform-size RD, deblock-aware RD | `encoder.c` `t8_rd_on`, `write_mb`; `dbrdo_on`, `dbrdo_delta`; `deblock.c` `aeg_deblock_mb_internal_luma` | x264 `analyse_transform_rd` (subme 8+) and `b_deblock_rdo` (subme 9+: the macroblock deblocked before its distortion is measured, `rdo.c`) | The ideas and the subme levels | Here only the macroblock's internal luma edges are filtered for the decision. |
| Motion search | `mc.c` `hex_search`, `umh_search`, `subpel_refine`; `h264.h` sub-pel diamond, chroma ME | x264 `me.c` (`hex`, `umh`, `refine_subpel`, chroma-me) | The 16-point hexagon pattern of UMH; the 4-point sub-pel diamond. UMH itself comes from the JM reference software ("UMHexagonS", Chen et al.). | "Simplified" per the comments; own predictor sets and early exits. |
| Field/frame pair decision | `encoder.c` `mbaff_vsad_field` | x264 `x264_field_vsad` | The vertical-SAD rule | Plus own trial coding where the rule is unsure (`mbaff_pair_trial`). |
| SSIM / PSNR reporting | `encoder.c` `ssim_window`, `pic_ssim_y`, `qual_account` | x264 `ssim_end1`, `--ssim`; x264's "global" PSNR | The 8×8 window on a 4-sample grid; the constants (from Wang et al. 2004) in x264's scaled form | Statistics only; no effect on the stream. |
| 10-bit build | `bitdepth.h`, `Makefile` (`obj10`) | x264's compile-twice approach (`BIT_DEPTH`) | — | Structure only. |
| Option semantics | `cli/aegenc.c`, `include/aeg/aeg.h` | x264 CLI | Names and meanings of `--crf`, `--qcomp`, `--trellis`, `--aq-strength` scale, `--scenecut`, `--b-adapt`, presets as a concept | Own preset contents. |

## From FFmpeg (decoder behaviour; own code)

FFmpeg's decoder was read to get the decoder's side of the standard exactly
right where the text is easy to misread. The code here was written against
the standard's clauses; FFmpeg was the reference for checking them. The
scan finds no copied routine.

| area | where | FFmpeg reference |
|---|---|---|
| MBAFF neighbours, MV/ref scaling between field and frame macroblocks | `encoder.c` `nb_set*`, `predict_mv_at` and its diagonal neighbour (8.4.1.3) | `h264_mvpred.h` (`fill_decode_neighbors`, `FIX_MV_MBAFF`, `fetch_diagonal_mv`) |
| Spatial / temporal direct in MBAFF and field pictures | `encoder.c` `col_block`, the direct derivation (8.4.1.2) | `h264_direct.c` (`pred_spatial_direct_motion`, `ff_h264_direct_ref_list_init`) |
| MBAFF / field deblocking | `deblock.c` `dbk_left_mixed`, `dbk_top_fields` | `h264_loopfilter.c` (`filter_mb_mbaff_edgev`, the "special case in the norm") |
| Field scans, CABAC field context offsets | `tables.c`, `encoder.c` | `h264_slice.c` (field scan tables), `h264_cabac.c` (significance/last context offsets for field coding) |
| Concealment behaviour avoided | `encoder.c` `direct_group_ok` | FFmpeg's error concealment |

## From the standard (normative; no outside code)

`src/tables.c` holds the ITU-T H.264 tables:
- the zigzag and field scans (the 4x4/8x8 scans, Tables 8-13 and 8-14);
- the quantisation and dequantisation scales;
- the chroma QP mapping (8-15);
- the deblocking alpha/beta/tc0 tables (8-16, 8-17);
- the coded_block_pattern mapping (9-4);
- the CAVLC code tables (`coeff_token`, `total_zeros`, `run_before`; 9-5
  to 9-10);
- the CABAC context initialisation tables (9-12 onwards).

`src/cabac.c` holds Table 9-44 (`rangeTabLPS`) and the state transitions,
plus the context index layout of Tables 9-34 and 9-40. Being normative,
these are the same numbers in every conforming encoder and decoder. The arithmetic
coder, the transforms and the intra/inter prediction follow the clauses
named in their comments.

The CABAC bit-cost table `aeg_cabac_bin_cost` is derived here from Table
9-44 (the mean over the four range classes). It is not x264's
`x264_cabac_entropy`.

## Used as tools, not in the library

- **ffmpeg** (imageio-ffmpeg build) decodes the test streams in `make test`
  and CI.
- **x264** (an installed binary) is the comparison target in
  `tools/x264cmp.py` / `tools/x264abl.py`.
- **PoCL** and **NVRTC** (pip) run the device paths in CI.

None of them is linked into `libaeg` or `aegenc`, or shipped with it.

## Suggested order for the review

1. **`aeg_dither_plane`:** decide first. It is a re-typed x264 function and
   small enough to rewrite or replace in an hour.
2. **The x264 lambda table and the deblock idiom:** cheap to regenerate or
   restate, with the output byte-identical.
3. **The algorithm table above:** this is where the real question lies.
   These parts follow x264's methods and tuned constants on purpose, in
   different code. Whether that matters under the GPL is the legal question
   this memo does not answer.
