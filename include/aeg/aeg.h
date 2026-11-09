/*
 * aeg - accelerated encode GPU
 *
 * A from-scratch H.264/AVC (MPEG-4 Part 10) video encoder that keeps the
 * *software* (non-fixed-function) coding tools of x264/x265 -- full motion
 * search, subpel motion compensation, macroblock/sub-macroblock partitioning,
 * rate-distortion mode decision, adaptive quantization and the normative
 * in-loop deblocking filter -- but executes the numerically heavy stages on
 * CUDA/OpenCL compute hardware instead of NVENC/QSV style fixed-function
 * units.
 *
 * Because all coding decisions are made by software algorithms (never by the
 * GPU's dedicated video engine), the produced bitstream is a normal H.264
 * stream whose quality is defined by the same rate-distortion criteria x264
 * uses; the GPU only changes *how fast* those numbers are computed.
 *
 * Every stage can run on either the CPU or the GPU.  The dispatcher
 * (aeg_dispatch) measures both paths and routes each stage to whichever one
 * is actually faster for the current frame size, so small frames do not pay
 * GPU launch/transfer overhead while large frames get the full GPU speedup.
 *
 * License: MIT (see LICENSE)
 */
#ifndef AEG_AEG_H
#define AEG_AEG_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AEG_VERSION_MAJOR 0
#define AEG_VERSION_MINOR 1
#define AEG_VERSION_PATCH 0
#define AEG_VERSION_STRING "0.1.0"
/* (major << 16) | (minor << 8) | patch, comparable with aeg_version_number() */
#define AEG_VERSION_NUMBER \
    ((AEG_VERSION_MAJOR << 16) | (AEG_VERSION_MINOR << 8) | AEG_VERSION_PATCH)

/* Every public function and object carries AEG_API.  The shared library is
 * built with -fvisibility=hidden, so only these are exported; the encoder's
 * internal functions stay private to it. */
#ifndef AEG_API
# if defined(_WIN32) && defined(AEG_BUILDING_DLL)
#  define AEG_API __declspec(dllexport)
# elif defined(__GNUC__) && __GNUC__ >= 4
#  define AEG_API __attribute__((visibility("default")))
# else
#  define AEG_API
# endif
#endif

/* ------------------------------------------------------------------ */
/* Error codes                                                         */
/* ------------------------------------------------------------------ */
enum {
    AEG_OK          = 0,
    AEG_ERROR       = -1,
    AEG_EINVAL      = -2,
    AEG_ENOMEM      = -3,
    AEG_EBACKEND    = -4,   /* requested compute backend unavailable   */
    AEG_EAGAIN      = -5,   /* need more input before a packet exists  */
    AEG_EOF         = 1
};

/* ------------------------------------------------------------------ */
/* Compute backend selection                                           */
/* ------------------------------------------------------------------ */
typedef enum aeg_backend {
    AEG_BACKEND_AUTO   = 0, /* choose per stage from measured timings     */
    AEG_BACKEND_CPU    = 1, /* pure CPU (reference path)                  */
    AEG_BACKEND_OPENCL = 2,
    AEG_BACKEND_CUDA   = 3
} aeg_backend;

/* Individual accelerator stages.  AEG_STAGE_AUTO means "let the
 * dispatcher decide for every frame". */
enum aeg_stage {
    AEG_STAGE_MOTION_EST  = 0, /* integer + fractional motion estimation    */
    AEG_STAGE_SAD         = 1, /* distortion (SAD/SATD) for mode decisions   */
    AEG_STAGE_INTER_PRED  = 2, /* quarter-pel interpolation of predictors    */
    AEG_STAGE_TRANSFORM   = 3, /* 4x4 DCT + quantisation of residuals       */
    AEG_STAGE_IQUANT_IDCT = 4, /* inverse quant + inverse DCT               */
    AEG_STAGE_DEBLOCK     = 5, /* in-loop deblocking filter                 */
    AEG_STAGE_INTRA_PRED  = 6, /* intra prediction of all modes             */
    AEG_STAGE_RECON       = 7, /* residual add / reconstruction             */
    AEG_STAGE_PSNR        = 8, /* distortion statistics                      */
    AEG_STAGE_COUNT       = 9
};

AEG_API const char *aeg_stage_name(int stage);
AEG_API const char *aeg_backend_name(aeg_backend b);

/* ------------------------------------------------------------------ */
/* Encoder parameters                                                  */
/* ------------------------------------------------------------------ */
typedef enum aeg_preset {
    AEG_PRESET_ULTRAFAST = 0,
    AEG_PRESET_FAST      = 1,
    AEG_PRESET_MEDIUM    = 2,
    AEG_PRESET_SLOW      = 3,
    AEG_PRESET_VERYSLOW  = 4
} aeg_preset;

typedef enum aeg_rcmode {
    AEG_RC_CQP = 0,   /* constant QP                                */
    AEG_RC_ABR = 1,   /* average bitrate (1-pass QP feedback)       */
    AEG_RC_CRF = 2    /* constant rate factor (constant quality)    */
} aeg_rcmode;

/* One --zones entry: display pictures start..end inclusive get qp. */
typedef struct aeg_zone {
    int start;
    int end;
    int qp;
} aeg_zone;

/* One --roi entry: macroblocks touching the rectangle (pixel coordinates)
 * get qp_offset added to their QP (negative = finer). */
typedef struct aeg_roi {
    int x, y, w, h;
    int qp_offset;
} aeg_roi;

typedef struct aeg_params {
    int width, height;        /* luma dimensions, >= 16, any value      */
    int fps_num, fps_den;     /* frame rate (e.g. 30/1, 30000/1001)     */
    /* The unit of the pts passed in and of the pts/dts handed back, in
     * seconds as a fraction (e.g. 1/90000 for MPEG-TS, 1/1000 for ms).  0/0
     * (the default) = fps_den/fps_num, i.e. the pts counts frames.  It is used
     * only to derive the packets' dts (see aeg_packet) and changes no coding
     * decision. */
    int timebase_num, timebase_den;
    /* Signalled in the SPS (VUI) only; they change no coding decision.
     * sar_w:sar_h is the sample aspect ratio (0:0 = not signalled);
     * fullrange 1 = the samples use 0..255 (JPEG/"pc" range), 0 = 16..235;
     * colorprim / transfer / colormatrix are the H.273 code points (1 =
     * BT.709, 6 = SMPTE 170M, 9 = BT.2020, ...), -1 = not signalled. */
    int sar_w, sar_h;
    int fullrange;
    int colorprim, transfer, colormatrix;
    /* HDR10 static metadata (SEI, off by default): `master_display` signals
     * mastering_display_colour_volume with R/G/B and white-point x,y in
     * 0.00002 units (md_primaries[0..7]) and the luminance in 0.0001 cd/m2
     * (md_max_lum / md_min_lum); `max_cll` / `max_fall` signal
     * content_light_level_info in cd/m2 (-1 = absent). */
    int master_display;
    int md_primaries[8];
    unsigned md_max_lum, md_min_lum;
    int max_cll, max_fall;
    /* Supplemental information, off by default (neither changes a coding
     * decision).  nal_hrd: with the VBV model on (vbv_maxrate/vbv_bufsize),
     * signal it as the NAL HRD (hrd_parameters in the VUI) with a buffering
     * period SEI on every IDR and a picture timing SEI on every picture, as
     * broadcast and Blu-ray style receivers expect.  info_sei: a
     * user_data_unregistered SEI on the first picture naming the encoder and
     * the options that shaped the stream (x264 writes the same). */
    int nal_hrd;
    int info_sei;
    /* aud: an access unit delimiter NAL before every picture (transport
     * streams want them).  level_idc: force the level (10 * level, e.g. 41;
     * 0 = the smallest that fits).  qp_min / qp_max: no macroblock is coded
     * outside this range, whatever the rate control, AQ or macroblock tree
     * ask for (defaults 0 / 51; qp_max 0 means 51, unset).  chroma_qp_offset: chroma_qp_index_offset,
     * -12..12 (negative = finer chroma). */
    int aud;
    /* Interlaced material: 0 = progressive (the default), 1 = top field
     * first, 2 = bottom field first.  The pictures are still coded as frames
     * (no field or MBAFF coding); the stream says it is interlaced
     * (frame_mbs_only_flag 0) and in which order the fields are displayed
     * (pic_struct in a picture timing SEI on every picture), so a player
     * deinterlaces it correctly -- what x264 --fake-interlaced does for
     * Blu-ray, with the field order of --tff/--bff.  The height must be a
     * multiple of 4; the levels that allow it are 2.1 to 4.1. */
    int interlaced;
    /* How interlaced material is coded (with `interlaced` 1 or 2; 0 there
     * and field_coding > 0 means top field first):
     *   0 = as frames (the default, x264 --fake-interlaced, above);
     *   1 = as field pictures (PAFF): every frame is two pictures, one per
     *       field, each predicted from the fields before it -- the second
     *       from the first -- which follows motion between the fields
     *       instead of coding the combing it leaves in a frame;
     *   2 = frame or field pictures, chosen per frame (adaptive PAFF): an
     *       I/P frame goes to fields when a quick estimate of its
     *       prediction cost as two fields beats the cost as a frame by 12 %,
     *       the B frames before it follow it.
     *   3 = MBAFF frames (macroblock-adaptive frame/field): every frame
     *       one picture whose macroblocks are coded in vertical pairs, each
     *       pair as two frame macroblocks or as one per field: chosen by
     *       the pair's vertical SADs within the frame and within the fields
     *       (x264's rule) and, at presets 2-4 where those are close, by
     *       coding the pair both ways and keeping the cheaper; without
     *       weighted prediction; the NAL HRD stays;
     *   4 = 2 and 3 together: per anchor frame, two field pictures or an
     *       MBAFF frame (the B frames follow their anchor) -- field pictures
     *       where the whole picture moves (the second field predicts from
     *       the first, which no MBAFF frame can), MBAFF frames elsewhere.
     * Field coding runs without temporal direct and keeps the B pyramid
     * (in field pictures too: a reference B frame's two fields); field
     * pictures (1, 2, 4) run without chroma ME (with 2 the frame pictures
     * keep it).  Weighted prediction (weightp) runs in field pictures and
     * frame pictures, not in MBAFF frames (3, and those of 4).  The NAL
     * HRD runs in every mode: a field picture is an access unit of its own,
     * removed half a frame period after the first field of its frame. */
    int field_coding;
    /* Bits per coded sample: 8 (the default; 0 also means 8) or 10 (High 10
     * profile, as x264's 10-bit build: the library holds both, compiled
     * twice, and picks one per encoder).  The rate controls, qp, crf, qp_min,
     * qp_max and the per-picture QPs keep the 8-bit scale (QPY: a QP of 22
     * means the same quantiser step relative to the signal at both depths);
     * at 10 bits the rate control may go below 0, down to -12, unless qp_min
     * says otherwise.  10-bit runs on the CPU only (the GPU kernels are
     * 8-bit).  Input: aeg_encoder_encode_image() with 10-bit I420/YV12 goes
     * in unchanged, anything else is converted (8-bit input is scaled up
     * exactly, deeper input dithered to 10 bits); aeg_encoder_encode() and
     * _encode_planes() take 8-bit samples.  aeg_encoder_recon() returns
     * uint16_t samples (strides in bytes) for a 10-bit encoder. */
    int bit_depth;
    int level_idc;
    int qp_min, qp_max;
    int chroma_qp_offset;
    int qp;                   /* 0..51, used by CQP and as ABR start    */
    int bitrate_kbps;         /* ABR target (0 = CQP)                   */
    int rc_mode;              /* aeg_rcmode                             */
    /* AEG_RC_CRF: the quality to hold, on x264's scale (lower is better;
     * 0..51, fractions allowed; -12..51 at 10 bits).  qcomp: how far the quantiser follows the
     * picture's complexity, in percent (x264's qcomp; 60 by default): 100
     * holds the QP constant, 0 holds the bits per picture constant.  With the
     * macroblock tree the tree does that work per macroblock instead. */
    double crf;
    int qcomp;
    /* VBV/HRD (H.264 Annex C) buffer model: the decoder's buffer must never
     * run dry, so a picture may not take more bits than the buffer holds at
     * the moment it starts.  Both 0 = no constraint.  Setting them implies
     * ABR; a bitrate above the maximum rate is clamped to it. */
    int vbv_maxrate_kbps;     /* R: channel rate, 0 = off               */
    int vbv_bufsize_kbits;    /* B: buffer size, 0 = off                */
    int gop_size;             /* keyframe interval, 0 = only first      */
    /* B pictures between two anchors: 0 = every picture predicts from the one
     * before it (P only), 1..3 = that many B pictures are inserted between an
     * anchor and the next, and the stream is coded out of display order.  The
     * encoder holds `bframes` pictures of lookahead, so aeg_encoder_encode()
     * returns AEG_EAGAIN until it has them. */
    int bframes;
    /* Adaptive B pictures: 1 = `bframes` is the most a group can have, and the
     * lookahead decides group by group how many it gets (none on fast,
     * irregular motion); 0 = always `bframes`.  -1 = the default (1). */
    int b_adapt;
    /* QP added to every B picture (it is never a reference, so the coarser
     * quantiser cannot propagate; x264 spends +2 by default for the same
     * reason).  Only meaningful with bframes > 0.  -1 (the default) = 4, or
     * 2 with the macroblock tree. */
    int b_qp_offset;
    /* QP subtracted from every I/IDR picture at constant QP (x264's ipratio
     * 1.4 is the same thing, -3).  Everything after a keyframe predicts from
     * it, so its quality is inherited by every skipped or cheaply coded
     * macroblock that follows.  -1 = the default. */
    int i_qp_offset;
    /* Direct-mode prediction of B pictures (B_Skip, B_Direct_16x16,
     * B_Direct_8x8): 0 = temporal (the co-located anchor's vectors scaled by
     * the picture distances), 1 = spatial (the neighbours' vectors, zeroed
     * where the co-located block stands still).  -1 = the default. */
    int direct;
    /* Scene-cut detection: insert an IDR when a picture's inter prediction is
     * worse than its intra prediction by more than `scenecut` percent (0 = off).
     * A scene change is exactly where a P picture costs more than the IDR it
     * would replace, and it is the one place where a fixed GOP is measurable. */
    int scenecut;
    /* x264's --open-gop: a keyframe boundary is an I slice that does not
     * reset the decoder state (no IDR), so the B pictures before it in
     * display order may keep predicting from the previous GOP.  A recovery
     * point SEI marks each such I. */
    int open_gop;
    int preset;               /* aeg_preset                             */
    int threads;              /* 0 = auto                               */
    int slices;               /* slices per picture, 0/1 = one           */
    /* x264's --slice-max-mbs: no slice holds more macroblocks than this
     * (the row granularity of this encoder is kept; 0 = unlimited). */
    int slice_max_mbs;
    /* x264's --slice-max-size: no slice's payload may exceed this many
     * bytes.  The picture is coded again with half as many rows per slice
     * while it does (down to one row); 0 = unlimited. */
    int slice_max_size;
    /* x264's --intra-refresh: no IDR after the first picture; instead a
     * vertical band of macroblocks is coded intra, moving across the
     * picture so that every macroblock is refreshed once per this many
     * frames.  0 = off.  Implies --keyint infinite and no scene cuts. */
    int intra_refresh;
    /* Frame threads: 2 = a non-reference B picture is coded next to the
     * picture after it (same bitstream as 1), 1 = off, 0 = auto (2 with more
     * than one CPU unless threads == 1).  Constant QP or CRF (ABR and VBV
     * code serially), one slice per picture, every field_coding. */
    int frame_threads;
    int backend;              /* aeg_backend                            */
    int stage_backend[AEG_STAGE_COUNT]; /* aeg_backend per stage, -1 = auto */
    int aq_mode;              /* 0 off, 1 variance AQ                   */
    int aq_strength;          /* 0..3, default 1.0 (units of 0.1?)      */
    /* High-profile 8x8 luma transform.  One transform per 8x8 block instead of
     * four 4x4 ones, with the matching Intra_8x8 prediction modes; a stream that
     * uses it needs profile_idc 100 (High).  On by default: it is the larger of
     * the two quality tools this encoder implements. */
    int transform_8x8;
    /* CABAC (context-adaptive binary arithmetic coding).  The alternative is
     * CAVLC; CABAC codes the same decisions in ~10% fewer bits on typical
     * content, at a higher cost per macroblock.  On by default. */
    int cabac;
    int deblock;              /* 0 = off, 1 = on (default on)           */
    int deblock_alpha;        /* slice_alpha_c0_offset_div2, -6..6      */
    int deblock_beta;         /* slice_beta_offset_div2, -6..6          */
    int search_range;         /* integer pel search range, 0 = preset   */
    int subme;                /* 0..9: subpel refinement quality        */
    /* Rate-distortion refinement of the winning inter vector: 0 = off (the
     * vector is the one the SATD search returned), 1 = the four quarter-pel
     * neighbours, 2 = all eight.  Each step is one more full macroblock
     * candidate priced by the coder, so it is a slow-preset tool.  -1 means
     * "as the preset says" (3 -> 1, 4 -> 2, below that off). */
    int refine_mv;
    /* Refine every inter candidate's vectors before the candidates are
     * compared, not just the winner's afterwards: 0 = off (the default),
     * 1 = on, the four axial quarter-pel neighbours per partition and list,
     * with the candidates that are already too expensive to win skipped.
     * Costs about a third of the encode time at preset 4 for 0.1-0.6 points of
     * BD-rate (see docs/STATUS.md, "Refining every candidate"). */
    int refine_cand;
    /* QP-RD (x264's subme 10): the decided macroblock is coded again at the
     * neighbouring QPs and the full rate-distortion price picks its QP (with
     * the macroblock tree, distortion is weighted by how much later pictures
     * inherit it).  0 = off, 1 = on; -1 = as the preset says (on at preset
     * 4).  Needs CABAC.  About -2.2 points of BD-rate against x264 veryslow
     * at CRF on the long clips, for 10 % of the encode time at preset 4 and
     * 15-26 % at preset 3 (docs/STATUS.md, "QP-RD"). */
    int qp_rd;
    /* P-picture reference pictures: 1..8.  -1 = as the preset says (8 at
     * preset 4, 4 at preset 3, 1 below).  The
     * second reference is the anchor before the previous one (with a B
     * pyramid the reference B just coded); a whole-macroblock 16x16
     * candidate on it competes in the rate-distortion decision.  With three
     * or more, list0 holds the newest anchor and then the other pictures in
     * the buffer by decreasing display order, each older one gets its own
     * 16x16 search and the best of them is the extra candidate. */
    int refs;
    /* Explicit weighted prediction of P pictures (fades): 0 = off, 1 = a
     * weight and offset per picture for luma and chroma, chosen by a small
     * motion-compensated test of the reference against the picture (P field
     * pictures too: each field against its list 0 field).  -1 = as the
     * preset says (on from preset 1 up). */
    int weightp;
    /* B pyramid: the middle B picture of a group of two or more is a
     * reference picture for the others (x264's b-pyramid "normal").  -1 = on
     * whenever it can be (B pictures, spatial direct, one P reference). */
    int b_pyramid;
    /* Macroblock tree (x264's mbtree): per-macroblock QP offsets from how
     * much of each block later pictures predict from, estimated on the
     * lookahead's half-resolution pictures over `rc_lookahead` pictures past
     * the group being coded.  Needs B pictures (the lookahead ring).
     * -1 = as the preset says. */
    int mbtree;
    int rc_lookahead;
    /* Rate-distortion level selection ("trellis", see trellis_luma4() in
     * encoder.c), with x264's values: 0 = off, 1 = applied to the winning
     * macroblock only (the decision compares plainly quantised candidates),
     * 2 = inside every candidate of the decision too.  -1 means "as the
     * preset says" (1 at presets 1-2, 2 at 3-4). */
    int trellis;
    /* Psy-rd (x264's --psy-rd): weight the luma distortion of a candidate by
     * a sqrt(residual) energy mask.  `psy_rd` is the strength (0 = off,
     * 1.0 = x264's usual value, up to 2.0); `psy_trellis` is reserved for the
     * quantiser-side variant (M-5b), currently parsed but not applied. */
    double psy_rd;
    double psy_trellis;
    int partitions;           /* bitmask: 1=16x16,2=16x8,4=8x16,8=8x8, 16=8x8 split further */
    int chroma_me;            /* include chroma in motion SAD           */
    int verbose;              /* informational; logging is aeg_set_log_level() */
    int force_cpu_reference;  /* also compute CPU path for validation   */
    const char *dump_stats;   /* optional per-frame CSV path            */
    /* Per-picture QP overrides (x264's --qpfile): frame_qp[k] is the QP of
     * display picture k, -1 = the rate controller's choice.  The table is
     * borrowed, not copied, and must stay valid for the encoder's lifetime. */
    const int *frame_qp;
    int        n_frame_qp;
    /* x264's --zones: per-frame-range QP overrides.  A frame_qp entry wins
     * when both are given.  Borrowed for the encoder's lifetime. */
    const aeg_zone *zones;
    int             n_zones;
    /* ROI: per-rectangle QP offsets, applied after AQ/mbtree. */
    const aeg_roi *rois;
    int            n_rois;
} aeg_params;

AEG_API void aeg_params_default(aeg_params *p);

/* One compressed access unit: an Annex-B byte stream (every NAL unit is
 * preceded by a start code; the buffer starts with the first one).  The data
 * belongs to the encoder and stays valid until the next call on the same
 * encoder (encode, flush, headers or close); copy it to keep it.
 *
 * pts is the one the picture was submitted with.  dts is its decoding time in
 * the same unit: packets come out in decoding order, and with B pictures that
 * is not the display order, so a container (MP4, MKV) needs both.  dts
 * increases strictly from packet to packet and never exceeds pts as long as
 * the pts advance by one frame duration (params.timebase) per picture; it
 * starts `bframes` frame durations before the first pts, as x264's does. */
typedef struct aeg_packet {
    uint8_t *data;
    size_t   size;
    int64_t  pts;
    int64_t  dts;
    int      keyframe;
    int      qp;
    int      type;     /* 0 = I, 1 = P, 2 = B, 3 = B used as a reference */
    double   qp_avg;   /* the macroblocks' average QP (adaptive offsets in) */
    double   qp_base;  /* the picture's QP before the macroblock offsets   */
} aeg_packet;

/* Two-pass rate control: what the first pass measured for one picture. */
typedef struct aeg_pass1_frame {
    int     type;      /* aeg_packet.type */
    double  qp;        /* aeg_packet.qp_avg */
    double  qp_base;   /* aeg_packet.qp_base */
    int64_t bits;      /* the picture's size in bits; < 0 = not measured */
} aeg_pass1_frame;

typedef struct aeg_stats {
    int64_t frames;
    int64_t bytes;
    int64_t enc_ns;             /* total wall time inside encode calls    */
    int64_t stage_ns[AEG_STAGE_COUNT];   /* time spent per stage          */
    /* time inside the stage's batch operations, on a device and on the host
     * (only the batched paths are timed per call, so these are parts of
     * stage_ns, not a split of it) */
    int64_t stage_gpu_ns[AEG_STAGE_COUNT];
    int64_t stage_cpu_ns[AEG_STAGE_COUNT];
    int     stage_use_gpu[AEG_STAGE_COUNT]; /* any of its work on a device */
    double  psnr_y, psnr_u, psnr_v, psnr_avg; /* reconstructed vs source  */
    double  ssim_y;
    /* VBV/HRD accounting (zero when the model is off): the lowest buffer level
     * the model reached, in bits, and how many pictures would have underflowed
     * it -- which the controller is supposed to have avoided. */
    double  vbv_min_fill_bits;
    int64_t vbv_underflows;
    int64_t vbv_size_clamps;    /* pictures whose QP the buffer raised */
    int64_t scene_cuts;         /* IDR pictures inserted by scenecut  */
    /* Macroblock statistics, cumulative over the encode: how many macroblocks
     * were written and how many of them as a skip. */
    int64_t macroblocks;
    int64_t skipped;
    /* How many macroblocks were coded with each type, indexed by the internal
     * AEG_MB_* enumeration (0 = skip ... 12 = B_Bi_16x16). */
    int64_t mb_types[16];
    /* VBV: pictures coded a second (or third) time because the first attempt
     * did not fit in the buffer */
    int64_t vbv_recodes;
    /* the most B pictures between two anchors this encoder codes, after the
     * options were resolved (CAVLC turns them off): what a second pass has
     * to match its first pass with */
    int     bframes;
} aeg_stats;

typedef struct aeg_encoder aeg_encoder;

/* ------------------------------------------------------------------ */
/* Input pictures in other formats                                     */
/* ------------------------------------------------------------------ */
/* The layouts aeg_encoder_encode_image() accepts.  The planes are given in
 * memory order: plane[0] is always Y; I420 has U then V, YV12 V then U; NV12
 * has one plane of interleaved U,V pairs (NV21: V,U); I400 is luma only. */
typedef enum aeg_csp {
    AEG_CSP_I420 = 0,   /* 4:2:0 planar Y, U, V                          */
    AEG_CSP_YV12 = 1,   /* 4:2:0 planar Y, V, U                          */
    AEG_CSP_NV12 = 2,   /* 4:2:0 Y + interleaved UV (P010/P016: depth 16) */
    AEG_CSP_NV21 = 3,   /* 4:2:0 Y + interleaved VU                      */
    AEG_CSP_I422 = 4,   /* 4:2:2 planar Y, U, V                          */
    AEG_CSP_YV16 = 5,   /* 4:2:2 planar Y, V, U                          */
    AEG_CSP_NV16 = 6,   /* 4:2:2 Y + interleaved UV                      */
    AEG_CSP_I444 = 7,   /* 4:4:4 planar Y, U, V                          */
    AEG_CSP_YV24 = 8,   /* 4:4:4 planar Y, V, U                          */
    AEG_CSP_I400 = 9    /* luma only (grey); chroma is coded as 128      */
} aeg_csp;

/* A picture of the encoder's width x height in any aeg_csp, 8 to 16 bits per
 * sample.  depth 8: one byte per sample; depth 9..16: one uint16_t per
 * sample in the machine's byte order, holding a value 0..2^depth - 1 (larger
 * values are clipped).  stride[] is in bytes.  How it becomes 8-bit 4:2:0:
 * samples are scaled to 16 bits, the chroma is resampled to 4:2:0 in that
 * precision for MPEG-2/H.264 chroma siting (per field when params.interlaced
 * is set), and every plane is dithered to 8 bits by error diffusion, exactly
 * as x264's 8-bit build does with high bit depth input. */
typedef struct aeg_image {
    int csp;                  /* aeg_csp                                 */
    int depth;                /* bits per sample, 8..16                  */
    const void *plane[3];
    int stride[3];            /* bytes per row                           */
} aeg_image;

/* ------------------------------------------------------------------ */
/* API                                                                 */
/* ------------------------------------------------------------------ */
/* Thread safety: one encoder must not be used by two threads at once, but
 * any number of encoders can run in parallel threads, and they produce the
 * same bytes as when they run alone (the encoder keeps no global state; its
 * own worker threads are internal).  aeg_last_error() is per thread. */

/* Returns NULL on failure; aeg_last_error() says why. */
AEG_API aeg_encoder *aeg_encoder_open(const aeg_params *params);
/* Encode one picture given as packed planar 4:2:0 (I420): the Y plane of
 * width x height bytes followed by the U and V planes of
 * ((width + 1) / 2) x ((height + 1) / 2) bytes each, without row padding.
 * Returns AEG_OK with a packet, AEG_EAGAIN when the picture was taken but no
 * packet is ready yet (B pictures: the encoder holds a lookahead), or an
 * error (< 0). */
AEG_API int  aeg_encoder_encode(aeg_encoder *enc, const uint8_t *yuv420p,
                                size_t size, int64_t pts, aeg_packet *pkt);
/* The same from three separate planes with their own row strides in bytes
 * (stride[0] >= width, stride[1..2] >= (width + 1) / 2), e.g. a decoder's
 * frame or a padded capture buffer.  The output is byte-identical to
 * aeg_encoder_encode() on the same samples. */
AEG_API int  aeg_encoder_encode_planes(aeg_encoder *enc,
                                       const uint8_t *const plane[3],
                                       const int stride[3], int64_t pts,
                                       aeg_packet *pkt);
/* Any other input format: the picture is converted to the 8-bit 4:2:0 the
 * encoder codes (see aeg_image), then coded as by aeg_encoder_encode_planes().
 * 8-bit I420/YV12 is passed through untouched, so the bytes are the same as
 * from the calls above. */
AEG_API int  aeg_encoder_encode_image(aeg_encoder *enc, const aeg_image *img,
                                      int64_t pts, aeg_packet *pkt);
/* After the last picture: call until it returns AEG_EOF; each AEG_OK hands
 * back one more packet. */
AEG_API int  aeg_encoder_flush(aeg_encoder *enc, aeg_packet *pkt);
AEG_API void aeg_packet_free(aeg_packet *pkt);
/* The stream's parameter sets: the SPS and PPS NAL units (Annex-B, with
 * start codes) that every IDR picture repeats in-band, available right after
 * aeg_encoder_open().  Containers that keep them out of band (MP4/MKV avcC,
 * "global headers") take them from here.  pkt->data stays valid until the
 * next call on the encoder; pts/dts/qp are 0, keyframe 1. */
AEG_API int  aeg_encoder_headers(aeg_encoder *enc, aeg_packet *pkt);
/* The parameters the encoder actually runs with: the caller's, with every
 * "-1 = as the preset says" resolved, the options that cannot be combined
 * turned off (e.g. CAVLC drops B pictures) and the level filled in. */
AEG_API int  aeg_encoder_get_params(aeg_encoder *enc, aeg_params *out);
AEG_API int  aeg_encoder_stats(aeg_encoder *enc, aeg_stats *out);
/* Read-only access to the reconstruction of the most recently encoded frame.
 * The pointers stay valid until the next encode call.  Used by the test tools
 * to compare the encoder's own reconstruction against an independent decoder. */
AEG_API int  aeg_encoder_recon(aeg_encoder *enc, const uint8_t *plane[3],
                               int stride[3], int mb_qp[]);
AEG_API int  aeg_encoder_set_qp(aeg_encoder *enc, int qp);
/* Bias the QP of the pictures submitted from now on, in QP steps (positive =
 * coarser).  The bias belongs to the picture it was set for: with B pictures a
 * picture is coded later than it is submitted, and it is coded with the bias
 * that was in force when it was submitted.  The two-pass mode uses it to
 * redistribute a fixed budget: the rate controller keeps the stream on target,
 * the bias moves bits from the pictures that are easy to the ones that are
 * hard.  It is cleared by aeg_encoder_open(). */
AEG_API int  aeg_encoder_set_qp_offset(aeg_encoder *enc, int offset);
/* Second pass of a two-pass encode (ABR only, before the first picture): f[d]
 * is the first pass's measurement of the picture with display index d.  The
 * encoder plans every picture's QP from it -- x264's second pass: each
 * picture's complexity (bits * qscale), blurred over its neighbours and raised
 * to 1 - qcomp, scaled so that the predicted total meets the bitrate -- and
 * then follows the plan, pulled back towards it as far as the bits actually
 * spent differ from what it expected.  Pictures past n fall back to the
 * one-pass controller.  Returns AEG_OK, or AEG_EINVAL without a bitrate
 * target or once encoding has started. */
AEG_API int  aeg_encoder_set_pass1(aeg_encoder *enc, const aeg_pass1_frame *f,
                                   int n);
AEG_API void aeg_encoder_close(aeg_encoder *enc);
/* A description of the last error of a call made by this thread ("" if
 * none).  The string is static. */
AEG_API const char *aeg_last_error(void);

/* The library's version at run time (the macros above are the header's). */
AEG_API const char *aeg_version(void);
AEG_API int aeg_version_number(void);

/* ------------------------------------------------------------------ */
/* Logging                                                             */
/* ------------------------------------------------------------------ */
enum {
    AEG_LOG_ERROR   = 0,
    AEG_LOG_WARNING = 1,   /* the default threshold */
    AEG_LOG_INFO    = 2,
    AEG_LOG_DEBUG   = 3
};
/* A log message, one line without the newline.  It can be called from the
 * encoder's worker threads and from several encoders at once, so it has to be
 * thread-safe; msg is only valid during the call. */
typedef void (*aeg_log_fn)(void *opaque, int level, const char *msg);
/* Route the library's messages to fn (NULL = back to stderr, the default).
 * Process-wide, like the level.  Change it while no encoder is running: a
 * message already on its way may still reach the previous function. */
AEG_API void aeg_set_log_callback(aeg_log_fn fn, void *opaque);
/* Messages above `level` are dropped (-1 = none at all). */
AEG_API void aeg_set_log_level(int level);
AEG_API int  aeg_get_log_level(void);

/* Compute device information (for logging / CLI --list-devices). */
typedef struct aeg_device_info {
    int         backend;      /* aeg_backend                            */
    char        name[192];
    char        vendor[96];
    int         is_gpu;       /* heuristic: real GPU vs CPU emulation    */
    uint64_t    global_mem;
    int32_t     max_compute_units;
    int64_t     launch_overhead_ns; /* measured, if calibrated          */
    int64_t     bytes_per_ns;       /* measured upload bandwidth        */
} aeg_device_info;

AEG_API int aeg_query_devices(aeg_backend backend, aeg_device_info *out,
                              int max);
AEG_API int aeg_calibrate(aeg_backend backend); /* measure launch overhead */

AEG_API int64_t aeg_now_ns(void);

/* The log threshold as a variable, kept for existing callers; prefer
 * aeg_set_log_level(). */
AEG_API extern int aeg_log_level;

#ifdef __cplusplus
}
#endif
#endif /* AEG_AEG_H */
