/*
 * aeg internal definitions: H.264 coding structures shared by the encoder,
 * the verification decoder and the accelerator backends.
 */
#ifndef AEG_H264_H
#define AEG_H264_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "aeg/aeg.h"

/* ------------------------------------------------------------------ */
/* Geometry                                                            */
/* ------------------------------------------------------------------ */
#define AEG_PAD 32              /* border padding around each picture      */
#define AEG_MAX_MB_W 512
#define AEG_MAX_MB_H 512
#define AEG_MAX_MBS  (AEG_MAX_MB_W * AEG_MAX_MB_H)

/* MB partition types */
enum {
    AEG_MB_SKIP = 0,
    AEG_MB_P16x16,
    AEG_MB_P16x8,
    AEG_MB_P8x16,
    AEG_MB_P8x8,
    AEG_MB_I4x4,
    AEG_MB_I16x16,
    AEG_MB_IPCM
};

enum { AEG_MB_I_V = 0, AEG_MB_I_H = 1, AEG_MB_I_DC = 2, AEG_MB_I_P = 3 };

/* block indices inside aeg_mb.coef[] */
enum {
    AEG_BLK_LUMA = 0,       /* 0..15  luma 4x4 (AC only for I16x16)      */
    AEG_BLK_LUMA_DC = 16,   /* 16     luma DC (I16x16 only)              */
    AEG_BLK_CHROMA_DC = 17, /* 17,18  chroma DC (2x2)                    */
    AEG_BLK_CHROMA_CB = 19, /* 19..22 chroma Cb AC (4x4)                 */
    AEG_BLK_CHROMA_CR = 23, /* 23..26 chroma Cr AC (4x4)                 */
    AEG_BLK_COUNT = 27
};

#define AEG_DCT_SIZE 16

/* Zigzag scan (frame coded order) */
extern const uint8_t aeg_zigzag4x4[16];

/* Constant tables (tables.c) */
extern const uint16_t aeg_quant_mf[6][3];
extern const uint8_t  aeg_dequant_ls[6][3];
extern const uint8_t  aeg_qp_chroma_420[52];
extern const uint8_t  aeg_alpha_tab[52];
extern const uint8_t  aeg_beta_tab[52];
extern const int8_t   aeg_tc0_tab[52][4];
extern const uint8_t  aeg_intra4x4_pred_modes[16][8];
extern const uint8_t  aeg_coeff_token_len[4][68];
extern const uint8_t  aeg_coeff_token_bits[4][68];
extern const uint8_t  aeg_chroma_dc_coeff_token_len[20];
extern const uint8_t  aeg_chroma_dc_coeff_token_bits[20];
extern const uint8_t  aeg_total_zeros_len[15][16];
extern const uint8_t  aeg_total_zeros_bits[15][16];
extern const uint8_t  aeg_chroma_dc_total_zeros_len[3][4];
extern const uint8_t  aeg_chroma_dc_total_zeros_bits[3][4];
extern const uint8_t  aeg_run_before_len[7][16];
extern const uint8_t  aeg_run_before_bits[7][16];
extern const uint8_t  aeg_cbp_code_num[48];
extern const uint8_t  aeg_cbp_codenum[48];
extern const uint8_t  aeg_cbp_codenum_i4x4[48];


/* Sub-block scan index helpers: raster position of 4x4 block n */
#define AEG_BLK_X(n) ((n) & 3)
#define AEG_BLK_Y(n) ((n) >> 2)

/* ------------------------------------------------------------------ */
/* Per-macroblock decisions                                            */
/* ------------------------------------------------------------------ */
typedef struct aeg_mb {
    uint8_t  type;            /* AEG_MB_*                                 */
    uint8_t  i16_mode;        /* intra 16x16: V/H/DC/Plane                */
    uint8_t  intra4[16];      /* intra 4x4 modes, raster order            */
    uint8_t  chroma_mode;     /* intra chroma V(0)/H(1)/DC(2)/Plane(3)    */
    uint8_t  cbp;             /* H.264 CodedBlockPattern: (c<<4)|luma     */
    int8_t   qp;              /* luma QP actually used by this MB         */
    int8_t   qp_delta;        /* mb_qp_delta                              */
    int8_t   sub_type[4];     /* P_8x8 sub partitions: 0=8x8,1=8x4,2=4x8,3=4x4 */
    int16_t  mv[16][2];       /* per 4x4 MV in 1/4 pel (luma)             */
    int8_t   nz[AEG_BLK_COUNT];
    int16_t  coef[AEG_BLK_COUNT][AEG_DCT_SIZE]; /* quantised coefficients  */
    int32_t  cost;            /* chosen RD cost (debug/stats)             */
} aeg_mb;

/* ------------------------------------------------------------------ */
/* Picture                                                             */
/* ------------------------------------------------------------------ */
typedef struct aeg_picture {
    uint8_t *plane[3];        /* pointer to the visible (0,0) sample       */
    int      w[3], h[3];      /* visible dimensions per plane             */
    int      stride[3];
    int      mb_w, mb_h;
    int      poc;             /* picture order count                      */
    int      frame_num;
    int      qp;
    int      is_idr;
    int      type;            /* 0=I, 1=P                                 */
    int64_t  pts;
    int      ref_count;       /* 0 free, >0 in use                        */
    void    *priv;
} aeg_picture;

/* ------------------------------------------------------------------ */
/* Bit writer                                                          */
/* ------------------------------------------------------------------ */
typedef struct aeg_bitwriter {
    uint8_t *buf;
    size_t   cap;
    size_t   len;             /* bytes used                               */
    uint32_t cache;           /* bit cache                                */
    int      bits;            /* bits currently in cache                  */
    int      err;
    int      enable_emu;      /* insert emulation prevention bytes        */
    int      count_only;      /* measure size without storing bytes       */
    int      zeros;           /* consecutive zero bytes written           */
} aeg_bitwriter;

void aeg_bw_init(aeg_bitwriter *bw, uint8_t *buf, size_t cap, int emu);
void aeg_bw_put(aeg_bitwriter *bw, int n, uint32_t val);
void aeg_bw_align_zero(aeg_bitwriter *bw);
void aeg_bw_align_one(aeg_bitwriter *bw);
void aeg_bw_ue(aeg_bitwriter *bw, uint32_t val);
void aeg_bw_se(aeg_bitwriter *bw, int32_t val);
size_t aeg_bw_finish(aeg_bitwriter *bw);

/* NAL framing (adds the start code and emulation prevention bytes) */
size_t aeg_nal_write(uint8_t *out, size_t out_cap, int nal_ref_idc,
                     int nal_type, const uint8_t *rbsp, size_t rbsp_len,
                     int long_start_code);
size_t aeg_write_sps(uint8_t *rbsp, size_t cap, const aeg_params *p,
                     int mb_w, int mb_h, int frame_num_bits, int level_idc);
size_t aeg_write_pps(uint8_t *rbsp, size_t cap);
void   aeg_write_slice_header(aeg_bitwriter *bw, int slice_type, int frame_num,
                              int frame_num_bits, int idr, int idr_pic_id,
                              int slice_qp_delta, int deblock_alpha,
                              int deblock_beta, int deblock_idc);
void   aeg_set_last_error(const char *msg);

/* CAVLC residual coding (cavlc.c); kind: 0 = 4x4, 1 = luma DC, 2 = chroma
 * DC, 3 = Intra_16x16 AC block. */
void aeg_cavlc_write_block(aeg_bitwriter *bw, const int16_t *coef, int nC, int kind);
int  aeg_cavlc_block_cost(const int16_t *coef, int nC, int kind);
int  aeg_cavlc_nc(int nz_left, int nz_top, int have_left, int have_top);

/* ------------------------------------------------------------------ */
/* Transform / quantisation                                            */
/* ------------------------------------------------------------------ */
void aeg_transform_4x4(const int16_t *res, int stride, int16_t *coef);
void aeg_transform_dc4x4(const int16_t dc[16], int16_t out[16]);
void aeg_itransform_dc4x4(const int16_t in[16], int16_t out[16]);
void aeg_transform_dc2x2(const int16_t dc[4], int16_t out[4]);
void aeg_itransform_dc2x2(const int16_t in[4], int16_t out[4]);
/* Inverse transform + prediction add, with the DC (for Intra_16x16) taken
 * from `dc`; when dc == NULL the coefficient at scan position 0 is used. */
void aeg_idct_add_block(uint8_t *dst, int stride, const int16_t *level,
                        int qp, const uint8_t *pred, int pred_stride,
                        const int32_t *dc_in);
void aeg_recon_add(uint8_t *dst, int stride, const uint8_t *pred,
                   int pred_stride, const int16_t *res, int w, int h);

/* forward quantisation of a 4x4 block, writes `level` (raster order) */
/* Rounding offset of the forward quantiser for `1 << qbits` and the reference
 * denominator (3 intra, 6 inter), shared with the GPU kernels so that both
 * paths produce the same levels. */
int aeg_quant_bias(int qbits, int den);

void aeg_quant_4x4(const int16_t *coef, int16_t *level, int qp, int intra);
void aeg_quant_4x4_dc(const int16_t *coef, int16_t *level, int qp, int intra);
void aeg_quant_2x2_dc(const int16_t *coef, int16_t *level, int qp, int intra);
void aeg_dequant_4x4(const int16_t *level, int32_t *coef, int qp);
void aeg_dequant_4x4_dc_out(const int16_t *level, int32_t *dc, int qp);
void aeg_dequant_2x2_dc_out(const int16_t *level, int32_t *dc, int qp);
void aeg_dc_block_unpermute(int32_t *dc);
void aeg_transform_dc4x4_fwd(const int16_t *dc, int16_t *level);
void aeg_transform_dc2x2_fwd(const int16_t *dc, int16_t *level);

int  aeg_qp_chroma(int qp, int offset);

/* distortion metrics */
int  aeg_sad(int w, int h, const uint8_t *a, int sa, const uint8_t *b, int sb);
int  aeg_satd_4x4(const uint8_t *a, int sa, const uint8_t *b, int sb);
int  aeg_satd(int w, int h, const uint8_t *a, int sa, const uint8_t *b, int sb);
int  aeg_satd_8x8(const uint8_t *a, int sa, const uint8_t *b, int sb);
int  aeg_satd_16x16(const uint8_t *a, int sa, const uint8_t *b, int sb);
int  aeg_satd_8x8(const uint8_t *a, int sa, const uint8_t *b, int sb);

/* ------------------------------------------------------------------ */
/* Prediction                                                          */
/* ------------------------------------------------------------------ */
/* Intra prediction of one block using already reconstructed neighbours.
 * `pred` receives the prediction, `stride` is the destination stride. */
void aeg_intra_predict_16x16(uint8_t *pred, int stride, const uint8_t *top,
                             const uint8_t *left, int top_avail, int left_avail,
                             int topleft_avail, int mode);
void aeg_intra_predict_4x4(uint8_t *pred, int stride, const uint8_t *top,
                           const uint8_t *left, int topleft, int mode,
                           int top_avail, int left_avail, int topleft_avail);
void aeg_intra_predict_chroma(uint8_t *pred, int stride, const uint8_t *top,
                              const uint8_t *left, int top_avail, int left_avail,
                              int topleft_avail, int mode);
int  aeg_intra4x4_pred_mode(const aeg_mb *mb_grid, int mb_stride, int mb_x,
                            int mb_y, int blk, int *available);

/* quarter-pel motion compensation of a luma block. `src` points at the
 * integer position of the block origin in the padded plane. */
void aeg_mc_luma(uint8_t *dst, int dstride, const uint8_t *src, int sstride,
                 int w, int h, int fx, int fy);

/* Precomputed half-pel planes (half-pel memory optimisation, H.264 8.4.2.2).
 *
 * A quarter-pel motion search interpolates the same reference samples over and
 * over: every candidate position re-runs the 6-tap filter chain for its whole
 * block. The four luma planes a fractional position is built from -- integer,
 * horizontal, vertical and both -- are properties of the *reference picture*,
 * not of the block, so they are computed once per picture and then a fractional
 * position is just two array reads and an average. The samples are identical to
 * the ones aeg_mc_luma() computes per block (same filters, same rounding); only
 * the number of times they are computed changes.
 *
 * base[0] is the integer picture itself, base[1..3] the H, V and HV planes.
 * All four use the same stride and the same origin as the integer picture, so a
 * pointer into one of them can be turned into a pointer into another by
 * subtracting base[0]. */
typedef struct aeg_mc_ctx {
    const uint8_t *base[4];
    uint8_t       *buf;
    int            stride;
    int            ready;
} aeg_mc_ctx;

void aeg_mc_ctx_init(aeg_mc_ctx *mc, int w, int h, int stride);
void aeg_mc_ctx_free(aeg_mc_ctx *mc);
/* Recompute the fractional planes from the reference picture `ref` (padded,
 * origin and stride as `mc->stride`).  Called once per picture, right after the
 * picture becomes the reference. */
void aeg_mc_ctx_refresh(aeg_mc_ctx *mc, const uint8_t *ref, int w, int h);
/* Register the reference picture without filling the planes: used when a
 * backend fills them (aeg_be_ops.interp_plane). */
void aeg_mc_ctx_setref(aeg_mc_ctx *mc, const uint8_t *ref);
/* Fill one fractional plane (`which` 1 = horizontal, 2 = vertical, 3 = both) of
 * the padded reference picture into `pl`.  CPU reference implementation of the
 * `interp_plane` backend operation. */
void aeg_hpel_plane_fill(uint8_t *pl, int dstride, const uint8_t *ref,
                         int rstride, int w, int h, int which);
/* Plane-aware motion compensation: `src` points into mc->base[0]. */
void aeg_mc_luma_p(uint8_t *dst, int dstride, const aeg_mc_ctx *mc,
                   const uint8_t *src, int w, int h, int fx, int fy);
/* chroma: 1/8 pel, fx/in 0..7 */
void aeg_mc_chroma(uint8_t *dst, int dstride, const uint8_t *src, int sstride,
                   int w, int h, int fx, int fy);
/* Motion estimation (mc.c).  `ref` points at the block origin for MV(0,0);
 * the result is a quarter-pel motion vector. */
int  aeg_me_block(const uint8_t *ref, int rstride, const uint8_t *cur,
                  int cstride, int bw, int bh, int range, int subme,
                  int pred_x, int pred_y, const int seed[2],
                  const aeg_mc_ctx *mc, int *mvx, int *mvy, int *sad_out);
void aeg_mv_predict(const aeg_mb *mbs, int mb_stride, int mb_w,
                    int mb_x, int mb_y,
                    int *mvx, int *mvy);
int  aeg_ue_bits(uint32_t v);
int  aeg_se_bits(int32_t v);
int  aeg_mvd_bits(int dx, int dy);

/* ------------------------------------------------------------------ */
/* Deblocking filter                                                   */
/* ------------------------------------------------------------------ */
typedef struct aeg_deblock_info {
    const aeg_mb *mb;
    int mb_stride;
    int8_t qp[2];             /* qp of the two MBs adjacent to the edge    */
} aeg_deblock_info;

/* Deblocks a whole frame in place.  Edge type: 0=internal vertical,
 * 1=internal horizontal, 2=MB left, 3=MB top. */
void aeg_deblock_frame(aeg_picture *pic, const aeg_mb *mbs, int mb_stride,
                       int slice_alpha, int slice_beta, int deblock_idc);
void aeg_deblock_mb(aeg_picture *pic, const aeg_mb *mbs, int mb_stride,
                    int mb_x, int mb_y, int slice_alpha, int slice_beta);
void aeg_deblock_mb_dir(aeg_picture *pic, const aeg_mb *mbs, int mb_stride,
                        int mb_x, int mb_y, int salpha, int sbeta, int vertical);

/* Deblocking thresholds (shared with the verification decoder) */
void aeg_deblock_thresholds(int qp, int slice_alpha, int slice_beta,
                            int *alpha, int *beta, const int8_t **tc0);
int  aeg_deblock_bs(const aeg_mb *mbq, const aeg_mb *mbp, int edge_type,
                    int idx, int is_chroma);

/* ------------------------------------------------------------------ */
/* Motion estimation                                                   */
/* ------------------------------------------------------------------ */
typedef struct aeg_me_result {
    int16_t mv[2];            /* best quarter-pel MV                       */
    int     cost;             /* SATD cost (subpel) or SAD (integer)       */
    int     sad;
} aeg_me_result;

typedef struct aeg_me_ctx aeg_me_ctx;

/* ------------------------------------------------------------------ */
/* Miscellaneous                                                       */
/* ------------------------------------------------------------------ */
#define AEG_MIN(a,b) ((a) < (b) ? (a) : (b))
#define AEG_MAX(a,b) ((a) > (b) ? (a) : (b))
#define AEG_CLIP3(lo,hi,v) ((v) < (lo) ? (lo) : ((v) > (hi) ? (hi) : (v)))
#define AEG_ALIGN(x,a) (((x) + ((a)-1)) & ~((size_t)(a)-1))

void aeg_log(int level, const char *fmt, ...);
extern int aeg_log_level;
#define AEG_LOGE(...) aeg_log(0, __VA_ARGS__)
#define AEG_LOGW(...) aeg_log(1, __VA_ARGS__)
#define AEG_LOGI(...) aeg_log(2, __VA_ARGS__)
#define AEG_LOGD(...) aeg_log(3, __VA_ARGS__)

void *aeg_alloc(size_t size);        /* zeroed, 32-byte aligned            */
void *aeg_alloc_aligned(size_t size, size_t align);
void  aeg_free(void *p);
int   aeg_cpu_count(void);

#endif /* AEG_H264_H */
