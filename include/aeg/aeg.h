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

const char *aeg_stage_name(int stage);
const char *aeg_backend_name(aeg_backend b);

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
    AEG_RC_ABR = 1    /* average bitrate (1-pass QP feedback)       */
} aeg_rcmode;

typedef struct aeg_params {
    int width, height;        /* luma dimensions, >= 16, any value      */
    int fps_num, fps_den;     /* frame rate (e.g. 30/1)                 */
    int qp;                   /* 0..51, used by CQP and as ABR start    */
    int bitrate_kbps;         /* ABR target (0 = CQP)                   */
    int rc_mode;              /* aeg_rcmode                             */
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
    /* QP added to every B picture (it is never a reference, so the coarser
     * quantiser cannot propagate; x264 spends +2 by default for the same
     * reason).  Only meaningful with bframes > 0. */
    int b_qp_offset;
    /* Scene-cut detection: insert an IDR when a picture's inter prediction is
     * worse than its intra prediction by more than `scenecut` percent (0 = off).
     * A scene change is exactly where a P picture costs more than the IDR it
     * would replace, and it is the one place where a fixed GOP is measurable. */
    int scenecut;
    int preset;               /* aeg_preset                             */
    int threads;              /* 0 = auto                               */
    int slices;               /* slices per picture, 0/1 = one           */
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
    /* P-picture reference pictures: 1 or 2.  -1 = as the preset says (2 from
     * preset 3 up on streams without B pictures, 1 otherwise).  The second
     * reference is the anchor before the previous one; a whole-macroblock
     * 16x16 candidate on it competes in the rate-distortion decision. */
    int refs;
    /* Rate-distortion level selection ("trellis") for the luma residual: 0 =
     * off, 1 = the candidates that only remove or shrink coefficients (see
     * trellis_luma4() in encoder.c).  -1 means "as the preset says". */
    int trellis;
    int partitions;           /* bitmask: 1=16x16,2=16x8,4=8x16,8=8x8  */
    int chroma_me;            /* include chroma in motion SAD           */
    int verbose;              /* log verbosity                          */
    int force_cpu_reference;  /* also compute CPU path for validation   */
    const char *dump_stats;   /* optional per-frame CSV path            */
} aeg_params;

void aeg_params_default(aeg_params *p);

/* One compressed access unit (Annex-B byte stream, no start code prefix
 * added: the buffer starts at the first NAL unit's start code). */
typedef struct aeg_packet {
    uint8_t *data;
    size_t   size;
    int64_t  pts;
    int      keyframe;
    int      qp;
} aeg_packet;

typedef struct aeg_stats {
    int64_t frames;
    int64_t bytes;
    int64_t enc_ns;             /* total wall time inside encode calls    */
    int64_t stage_ns[AEG_STAGE_COUNT];   /* time spent per stage          */
    int64_t stage_gpu_ns[AEG_STAGE_COUNT];
    int64_t stage_cpu_ns[AEG_STAGE_COUNT];
    int     stage_use_gpu[AEG_STAGE_COUNT];
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
} aeg_stats;

typedef struct aeg_encoder aeg_encoder;

/* ------------------------------------------------------------------ */
/* API                                                                 */
/* ------------------------------------------------------------------ */
aeg_encoder *aeg_encoder_open(const aeg_params *params);
int  aeg_encoder_encode(aeg_encoder *enc, const uint8_t *yuv420p,
                        size_t size, int64_t pts, aeg_packet *pkt);
int  aeg_encoder_flush(aeg_encoder *enc, aeg_packet *pkt);
void aeg_packet_free(aeg_packet *pkt);
int  aeg_encoder_stats(aeg_encoder *enc, aeg_stats *out);
/* Read-only access to the reconstruction of the most recently encoded frame.
 * The pointers stay valid until the next encode call.  Used by the test tools
 * to compare the encoder's own reconstruction against an independent decoder. */
int  aeg_encoder_recon(aeg_encoder *enc, const uint8_t *plane[3],
                       int stride[3], int mb_qp[]);
int  aeg_encoder_set_qp(aeg_encoder *enc, int qp);
/* Bias the QP of the pictures that follow, in QP steps (positive = coarser).
 * The two-pass mode uses it to redistribute a fixed budget: the rate controller
 * keeps the stream on target, the bias moves bits from the pictures that are
 * easy to the ones that are hard.  It is cleared by aeg_encoder_open(). */
int  aeg_encoder_set_qp_offset(aeg_encoder *enc, int offset);
void aeg_encoder_close(aeg_encoder *enc);
const char *aeg_last_error(void);

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

int aeg_query_devices(aeg_backend backend, aeg_device_info *out, int max);
int aeg_calibrate(aeg_backend backend); /* measure launch overhead etc.   */

int64_t aeg_now_ns(void);

/* Library log verbosity (0 = quiet, 1 = normal, 2 = debug). */
extern int aeg_log_level;

#ifdef __cplusplus
}
#endif
#endif /* AEG_AEG_H */
