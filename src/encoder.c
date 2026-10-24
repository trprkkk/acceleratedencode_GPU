/*
 * The encoder: picture management, mode decision, reconstruction, bitstream.
 *
 * One frame is coded macroblock by macroblock in raster order.  For every
 * macroblock a prediction is selected (intra 16x16 / intra 4x4 / inter with
 * quarter-pel motion), the residual is coded for real -- forward transform,
 * quantisation, inverse quantisation, inverse transform, reconstruction --
 * and the reconstructed samples are written straight into the frame buffer.
 * That buffer is both the reference for the following pictures and the source
 * of the PSNR figures, so a mismatch against a decoder would show up as
 * drift; tools/aegverify checks exactly that by comparing the encoder's
 * reconstruction with an independent decode of the bitstream.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include "backend.h"

#define AEG_LAMBDA(qp) (0.85 * pow(2.0, ((qp) - 12) / 3.0))

typedef struct mbenc {
    int16_t  level[AEG_BLK_COUNT][16];
    int8_t   nz[AEG_BLK_COUNT];
    /* Luma coded with the 8x8 transform (High profile): one 64-coefficient
     * block per 8x8 group in raster order, the group's non-zero flag, and the
     * transform_size_8x8_flag itself.  `nz[]` still carries the per-4x4-block
     * numbers the CAVLC contexts need; for these macroblocks they hold each
     * list's coefficient count while the lists are coded and the block's 0/1
     * flag afterwards (see luma8_costs). */
    int16_t  level8[4][64];
    int8_t   nz8[4];
    int      t8;
    int16_t  mv[16][2];             /* per 4x4 block, 1/4 pel               */
    int      type;                  /* AEG_MB_*                             */
    int      i16_mode;              /* 0=V 1=H 2=DC 3=Plane                 */
    uint8_t  intra4[16];            /* Intra_4x4 modes, Figure 6-8 order    */
    uint8_t  intra8[4];             /* Intra_8x8 modes, 8x8 raster order    */
    int      chroma_mode;           /* 0=DC 1=H 2=V 3=Plane                 */
    int      cbp;
    int      qp;
    int      bits;                  /* payload bits (residual + headers)    */
    int      hdr_bits;
    int64_t  sse;
    int64_t  cost;
    /* P_8x8 sub-macroblock partition types: 0 = 8x8, 1 = 8x4, 2 = 4x8,
     * 3 = 4x4 (H.264 table 7-17). */
    uint8_t  sub_type[4];
    /* inter partitioning: one predicted MV per partition (1/4 pel), indexed by
     * the partition's first 4x4 block (the order the bitstream carries them
     * in is the partition order, see mb_part_geom()). */
    int16_t  pmv[16][2];
    uint8_t  nparts;
} mbenc;

/* One inter partitioning candidate gathered by analyse_mb(): the shape, its
 * sub_macroblock types and, per 4x4 block, the motion vector, plus the
 * predictor of every partition (indexed by its first 4x4 block). */
#define AEG_MB_CAND_MAX 4

/* Upper bound on the slices a picture may be cut into.  Slice threads use one
 * thread per slice, so this also bounds the thread count. */
#define AEG_MAX_SLICES 64
typedef struct {
    int     type;
    int     nparts;
    uint8_t sub_type[4];
    int     mv[16][2];
    int     pmv[16][2];
} mbcand;

struct aeg_encoder {
    aeg_params p;
    int w, h, cw, ch;
    int mb_w, mb_h, mb_count;
    int stride[3];
    int frame_num_bits, level_idc;

    uint8_t *src_mem, *rec_mem, *ref_mem;
    uint8_t *srcp[3], *recp[3], *refp[3];

    aeg_mb  *mbs;
    aeg_picture pic;

    /* per-macroblock decision scratch filled by analyse_mb() and consumed by
     * code_mb(): the intra modes ordered by SAD and the inter candidate. */
    int mb_modes[4];
    int mb_nmodes;
    int mb_inter_ok;
    int mb_sad;
    int mb_inter_sad;
    int mb_mv[2];
    int mb_pmv[2];
    /* Fractional part of the macroblock's 16x16 motion vector (0..3 quarter
     * pels).  The 16x16 search always runs the quarter-pel refinement, and
     * its result is a good estimate of the sub-pixel alignment of the whole
     * region, so the smaller partitions start from it instead of each running
     * its own full refinement -- that costs one refinement per macroblock
     * instead of up to eighteen. */
    int mb_frac[2];
    /* Inter partitioning candidates gathered by analyse_mb(): 16x16, 16x8,
     * 8x16 and P_8x8.  Each carries the per-partition MVs and their
     * predictors, which is exactly what the bitstream has to transmit. */
    mbcand mb_cand[4];
    int mb_ncand;

    aeg_dispatcher d;
    int frame_count;         /* total pictures encoded                  */
    int last_idr_frame;      /* frame_count of the last IDR picture     */
    int scene_cuts;          /* IDR pictures inserted by scenecut       */
    int frame_num;           /* frame_num of the current picture (8.2.1) */
    int idr_pic_id;
    int last_qp;

    /* One RBSP buffer per slice: a slice is its own NAL unit, and with slice
     * threads the slices are written by different threads at the same time.
     * They are all allocated before the threads start, because the pointers
     * live in the encoder and a per-thread copy of it must not grow its own
     * (see slice_thread()). */
    uint8_t *slice_bs[AEG_MAX_SLICES];
    size_t slice_bs_cap[AEG_MAX_SLICES];
    int n_threads;         /* threads used for the slice loop (1 = serial) */
    uint8_t *out;   size_t out_cap;

    aeg_stats st;

    /* Every macroblock's decision, kept until the picture's bitstream is
     * written.  The decision pass and the bitstream pass are separate so that
     * the first one can run macroblocks out of order (see the threading note
     * above encode_frame()); the second one needs them in raster order because
     * CAVLC is serial.  About 1 kB per macroblock. */
    mbenc *mb_encs;

    int16_t mv_batch[2 * 256];
    int32_t sad_batch[256];
    /* Batched coefficient pass (AEG_STAGE_TRANSFORM / AEG_STAGE_IQUANT_IDCT).
     * The sixteen luma blocks of an inter candidate are independent -- their
     * prediction comes from the reference picture, not from the block above or
     * to the left -- so they can be gathered into one call of tq_batch() and
     * one of iq_idct_batch() instead of sixteen of each.  On the CPU that is
     * the same arithmetic with a little more copying; on a device it is what
     * makes the stage a kernel launch rather than 1.9 M calls per hundred
     * pictures.  coef_batch says whether the dispatcher wants it. */
    int coef_batch;
    int16_t coef_res[16 * 16];
    int16_t coef_lev[16 * 16];
    int16_t coef_rec[16 * 16];
    int32_t coef_nz[16];
    /* RD batching (provisional, see docs/STATUS.md): every inter candidate of
     * the macroblock is predicted and its residual gathered first, then one
     * tq_batch() and one iq_idct_batch() cover all of the candidates at once,
     * and only then are the costs compared.  Four candidates x sixteen blocks
     * is the largest batch a macroblock can produce. */
    int rd_batch;
    int coef_force, rd_force;   /* -1: ask the dispatcher, 0/1: forced */
    int16_t rd_res[16 * 16 * AEG_MB_CAND_MAX];
    int16_t rd_lev[16 * 16 * AEG_MB_CAND_MAX];
    int16_t rd_rec[16 * 16 * AEG_MB_CAND_MAX];
    int32_t rd_nz[16 * AEG_MB_CAND_MAX];
    uint8_t rd_pred[16 * 16 * AEG_MB_CAND_MAX];
    /* Whole-frame search seed: 2 ints per macroblock, integer pel.  Filled by
     * me_seed_frame() with the best of a fixed candidate grid, evaluated as one
     * batch through the backend (this is the stage the GPU actually runs). */
    aeg_mc_ctx mc;         /* precomputed fractional-pel planes of refp[0] */
    int16_t *seed_mv;      /* integer vectors of the parallel rounds      */
    int16_t *par_pos;      /* 4 int16 per candidate for satd_batch       */
    int16_t *par_mv;       /* quarter-pel result per macroblock          */
    int32_t *par_cost;
    int      par_valid;
    int      par_full;
    int16_t *seed_cand;   /* 2 * ME_ROUND_CAND per macroblock */
    int16_t *prev_seed;   /* the same, as of the previous picture  */
    int32_t *seed_sad;
    int32_t *seed_cost;
    uint8_t pred_l[16 * 16];
    uint8_t pred_c[2][8 * 8];
    int has_ref;
    int cur_is_p;           /* the picture being coded is a P slice */
    /* Slice geometry of the picture being coded.  A slice owns a whole number
     * of macroblock rows, so "the block above is in another slice" is one
     * comparison.  H.264 treats every neighbour outside the current slice as
     * unavailable (7.4.1.2.4 for intra prediction, 8.4.1.3 for MV prediction,
     * the nC contexts of 9.2.1), which is exactly what top_row does: with one
     * slice per picture it is 0 and nothing changes. */
    int top_row;            /* first macroblock row of the current slice */
    int slice_rows;         /* rows per slice, 0 = one slice per picture   */
    int n_slices;
    int cur_qp;             /* QP of the picture being coded         */

    /* Per-macroblock QP offsets of the current picture (adaptive
     * quantisation).  Recomputed at the start of every picture from the source
     * variance and normalised to a zero mean, so AQ redistributes bits inside
     * a picture without changing the picture's average QP. */
    int8_t *mb_dqp;

    /* ABR rate control (`AEG_RC_ABR`): a fractional QP that is nudged after
     * every picture so that the running bit count tracks the target. */
    double rc_qp;
    double rc_bits_per_frame;
    double rc_spent;        /* bits used so far, all pictures               */
    double rc_debt;         /* budget - spent (positive = under budget)     */
    int    rc_frames;       /* pictures coded so far, for the budget        */
    int    rc_active;
    int    rc_qp_offset;    /* user bias on the per-picture QP (two-pass)     */

    /* VBV/HRD (H.264 Annex C): the receiver's buffer is modelled picture by
     * picture.  `vbv_full` is what the model says is in the buffer when the
     * next picture starts.  A picture may not take more bits than that, so the
     * controller raises its QP when the picture it is about to code does not
     * fit -- that is the whole point of a buffer model, and what makes a stream
     * playable at a rate the network actually has to deliver. */
    int    vbv_active;
    double vbv_buf_bits;    /* B: the buffer size                           */
    double vbv_rate_bits;   /* R: bits delivered per picture period         */
    double vbv_full;        /* modelled fullness, bits                      */
    double vbv_min_full;    /* lowest level seen                            */
    int    vbv_underflows;
    int    vbv_clamps;
    int    last_pic_bits;   /* the previous picture, to predict this one    */
    int    last_pic_qp;
};

/* Which of the two batched coefficient code paths to use for this picture.
 * Both need the device to own *both* coefficient stages: an inter candidate
 * cannot transform on one and reconstruct on the other without a round trip in
 * the middle of its RD cost.  AEG_COEF_BATCH / AEG_RD_BATCH override the
 * dispatcher, which is how the conformance cases exercise the paths on a
 * machine with no GPU. */
static void resolve_coef_stages(struct aeg_encoder *e)
{
    int want = e->d.ops.backend != AEG_BACKEND_CPU &&
               e->d.use_gpu[AEG_STAGE_TRANSFORM] &&
               e->d.use_gpu[AEG_STAGE_IQUANT_IDCT] &&
               e->d.ops.tq_batch && e->d.ops.iq_idct_batch;
    e->coef_batch = e->coef_force >= 0 ? e->coef_force : want;
    e->rd_batch = e->rd_force >= 0 ? e->rd_force : want;
    if (e->rd_batch)
        e->coef_batch = 1;
}

static inline int clip255(int v)
{
    return v < 0 ? 0 : (v > 255 ? 255 : v);
}

/* ------------------------------------------------------------------ */
/* Inter partition geometry                                            */
/* ------------------------------------------------------------------ */
static int is_intra_type(int t)
{
    return t == AEG_MB_I4x4 || t == AEG_MB_I8x8 || t == AEG_MB_I16x16 ||
           t == AEG_MB_IPCM;
}

static int is_inter_type(int t)
{
    return !is_intra_type(t);
}

/* Number of partitions of an inter macroblock type. */
static int part_count(int type)
{
    if (type == AEG_MB_P8x8)
        return 4;
    return (type == AEG_MB_P16x8 || type == AEG_MB_P8x16) ? 2 : 1;
}

/* Luma geometry of partition `p`: x, y, w, h in pixels inside the macroblock,
 * plus the index of its first 4x4 block (where its MV is stored). */
static void part_geom(int type, int p, int geo[4], int *blk0)
{
    switch (type) {
    case AEG_MB_P16x8:
        geo[0] = 0; geo[1] = p ? 8 : 0; geo[2] = 16; geo[3] = 8;
        *blk0 = p ? 8 : 0;
        break;
    case AEG_MB_P8x16:
        geo[0] = p ? 8 : 0; geo[1] = 0; geo[2] = 8; geo[3] = 16;
        *blk0 = p ? 2 : 0;
        break;
    case AEG_MB_P8x8:
        /* Four 8x8 sub-macroblock partitions, coded in raster order; their
         * first 4x4 blocks are 0, 2, 8 and 10. */
        geo[0] = (p & 1) ? 8 : 0; geo[1] = (p & 2) ? 8 : 0;
        geo[2] = 8; geo[3] = 8;
        *blk0 = ((p & 2) ? 8 : 0) + ((p & 1) ? 2 : 0);
        break;
    default:
        geo[0] = 0; geo[1] = 0; geo[2] = 16; geo[3] = 16;
        *blk0 = 0;
        break;
    }
}

/* Sub-partitions of an 8x8 sub-macroblock partition type: 8x8 has one, 8x4 and
 * 4x8 have two, 4x4 has four (H.264 table 7-17). */
static const uint8_t sub_cnt[4] = { 1, 2, 2, 4 };

/* Geometry of sub-partition `k` of an 8x8 sub-macroblock partition of type `st`
 * whose top-left 4x4 block is (bx, by): x, y, w, h in pixels inside the
 * macroblock, and the index of its first 4x4 block.  The order matters: it is
 * the order the bitstream carries the mvd values in (top-down, then
 * left-right). */
static void sub_geom(int st, int bx, int by, int k, int geo[4], int *b0)
{
    int x = bx, y = by, w = 2, h = 2;
    switch (st) {
    case 1: y += k; h = 1; break;              /* 8x4: top, bottom          */
    case 2: x += k; w = 1; break;              /* 4x8: left, right          */
    case 3: x += k & 1; y += k >> 1; w = h = 1; break;  /* 4x4: raster      */
    default: break;                            /* 8x8                       */
    }
    geo[0] = x * 4; geo[1] = y * 4; geo[2] = w * 4; geo[3] = h * 4;
    *b0 = y * 4 + x;
}

/* Number of partitions the bitstream carries for this macroblock: one mvd pair
 * each, in this order. */
static int mb_part_count(const mbenc *c)
{
    int i, n = 0;
    if (c->type != AEG_MB_P8x8)
        return part_count(c->type);
    for (i = 0; i < 4; i++)
        n += sub_cnt[c->sub_type[i]];
    return n;
}

static void mb_part_geom(const mbenc *c, int p, int geo[4], int *b0)
{
    int i;
    if (c->type != AEG_MB_P8x8) {
        part_geom(c->type, p, geo, b0);
        return;
    }
    for (i = 0; i < 4 && p >= 0; i++) {
        int n = sub_cnt[c->sub_type[i]];
        if (p < n) {
            sub_geom(c->sub_type[i], (i & 1) * 2, (i >> 1) * 2, p, geo, b0);
            return;
        }
        p -= n;
    }
    geo[0] = 0; geo[1] = 0; geo[2] = 16; geo[3] = 16; *b0 = 0;  /* not reached */
}

/* mb_type code of an inter macroblock (H.264 table 7-13, P slices). */
static int inter_mb_type_code(int type)
{
    if (type == AEG_MB_P8x8)
        return 3;
    return type == AEG_MB_P16x8 ? 1 : (type == AEG_MB_P8x16 ? 2 : 0);
}

/* ------------------------------------------------------------------ */
/* Buffers                                                             */
/* ------------------------------------------------------------------ */
static int plane_stride(int w) { return w + 2 * AEG_PAD; }

static int enc_alloc_picture(struct aeg_encoder *e, uint8_t **mem, uint8_t **plane)
{
    size_t ls = (size_t)plane_stride(e->w), lh = (size_t)(e->h + 2 * AEG_PAD);
    size_t cs = (size_t)plane_stride((e->w + 1) / 2);
    size_t ch = (size_t)((e->h + 1) / 2 + 2 * AEG_PAD);
    size_t total = ls * lh + 2 * cs * ch;

    *mem = (uint8_t *)aeg_alloc(total);
    if (!*mem)
        return AEG_ENOMEM;
    /* Every plane occupies exactly lh*ls (or ch*cs) bytes starting at its own
     * origin: the left/right padding is part of the stride and the top/bottom
     * padding fits inside the row count.  Placing plane 2 at plane 1 plus one
     * chroma plane -- rather than plus the padding of the next origin -- used to
     * write one row band past the end of the allocation. */
    plane[0] = *mem + AEG_PAD * ls + AEG_PAD;
    plane[1] = *mem + ls * lh + AEG_PAD * cs + AEG_PAD;
    plane[2] = *mem + ls * lh + cs * ch + AEG_PAD * cs + AEG_PAD;
    e->stride[0] = (int)ls;
    e->stride[1] = e->stride[2] = (int)cs;
    return AEG_OK;
}

static void copy_plane_pad(uint8_t *dst, int dstride, const uint8_t *src,
                           int sstride, int w, int h)
{
    int x, y;
    for (y = 0; y < h; y++)
        memcpy(dst + y * dstride, src + y * sstride, (size_t)w);
    for (y = 0; y < h; y++) {
        memset(dst + y * dstride - AEG_PAD, dst[y * dstride], AEG_PAD);
        memset(dst + y * dstride + w, dst[y * dstride + w - 1], AEG_PAD);
    }
    /* A padded row is exactly `w + 2 * AEG_PAD` bytes wide and starts AEG_PAD
     * bytes left of the content origin, so the copies must be taken from the
     * row origin -- otherwise every row runs AEG_PAD bytes past the plane. */
    for (y = 1; y <= AEG_PAD; y++) {
        memcpy(dst - y * dstride - AEG_PAD, dst - AEG_PAD,
               (size_t)(w + 2 * AEG_PAD));
        memcpy(dst + (h - 1 + y) * dstride - AEG_PAD,
               dst + (h - 1) * dstride - AEG_PAD, (size_t)(w + 2 * AEG_PAD));
    }
    (void)x;
}

/* Replicate the reconstructed picture's edges into the padding.  The motion
 * compensated prediction of a macroblock on the picture border reads samples
 * outside the visible area, and the decoder pads with the edge sample, so the
 * reference buffer has to hold exactly that. */
static void pad_edges(uint8_t *p, int stride, int w, int h)
{
    int y;
    for (y = 0; y < h; y++) {
        memset(p + y * stride - AEG_PAD, p[y * stride], AEG_PAD);
        memset(p + y * stride + w, p[y * stride + w - 1], AEG_PAD);
    }
    for (y = 1; y <= AEG_PAD; y++) {
        memcpy(p - y * stride - AEG_PAD, p - AEG_PAD,
               (size_t)(w + 2 * AEG_PAD));
        memcpy(p + (h - 1 + y) * stride - AEG_PAD, p + (h - 1) * stride - AEG_PAD,
               (size_t)(w + 2 * AEG_PAD));
    }
}

/* ------------------------------------------------------------------ */
/* Neighbour nC                                                        */
/* ------------------------------------------------------------------ */
static int left_nz(struct aeg_encoder *e, int mbx, int mby, int blk)
{
    return mbx ? e->mbs[mby * e->mb_w + mbx - 1].nz[blk] : 0;
}

static int top_nz(struct aeg_encoder *e, int mbx, int mby, int blk)
{
    return mby > e->top_row ? e->mbs[(mby - 1) * e->mb_w + mbx].nz[blk] : 0;
}

static int luma_nc(struct aeg_encoder *e, int mbx, int mby, const mbenc *c, int blk)
{
    int bx = blk & 3, by = blk >> 2;
    int a = bx ? c->nz[blk - 1] : left_nz(e, mbx, mby, blk + 3);
    int b = by ? c->nz[blk - 4] : top_nz(e, mbx, mby, blk + 12);
    return aeg_cavlc_nc(a, b, mbx > 0 || bx > 0,
                       mby > e->top_row || by > 0);
}

static int chroma_nc(struct aeg_encoder *e, int mbx, int mby, const mbenc *c,
                     int pl, int blk)
{
    int base = AEG_BLK_CHROMA_CB + pl * 4;
    int bx = blk & 1, by = blk >> 1;
    int a = bx ? c->nz[base + blk - 1] : left_nz(e, mbx, mby, base + blk + 1);
    int b = by ? c->nz[base + blk - 2] : top_nz(e, mbx, mby, base + blk + 2);
    return aeg_cavlc_nc(a, b, mbx > 0 || bx > 0,
                       mby > e->top_row || by > 0);
}

/* ------------------------------------------------------------------ */
/* Luma coding: Intra_16x16                                            */
/* ------------------------------------------------------------------ */
/* Transmission order of the sixteen Intra_16x16 AC blocks.  H.264 numbers
 * them i4x4 + 4*i8x8, i.e. quadrant by quadrant, not in 4x4 raster order; the
 * neighbour contexts have to follow that numbering as well. */
static const uint8_t ac_ord[16] = { 0, 1, 4, 5, 2, 3, 6, 7,
                                    8, 9, 12, 13, 10, 11, 14, 15 };

static int luma_ac_nc(struct aeg_encoder *e, int mbx, int mby,
                      const mbenc *c, int i)
{
    int blk = ac_ord[i];
    int bx = blk & 3, by = blk >> 2;
    int a, b;
    if (bx)
        a = c->nz[blk - 1];
    else
        a = left_nz(e, mbx, mby, 4 * by + 3);
    if (by)
        b = c->nz[blk - 4];
    else
        b = top_nz(e, mbx, mby, 12 + bx);
    return aeg_cavlc_nc(a, b, mbx > 0 || bx > 0,
                       mby > e->top_row || by > 0);
}

static void code_luma_i16(struct aeg_encoder *e, int mbx, int mby, mbenc *c)
{
    const uint8_t *src = e->srcp[0] + mby * 16 * e->stride[0] + mbx * 16;
    uint8_t *dst = e->recp[0] + mby * 16 * e->stride[0] + mbx * 16;
    const uint8_t *pred = e->pred_l;
    int16_t dc[16], had[16], dcled[16];
    int32_t dchad[16];
    int blk, i, x, y, nz_dc = 0;
    int qp = c->qp;

    for (blk = 0; blk < 16; blk++) {
        int bx = (blk & 3) * 4, by = (blk >> 2) * 4;
        int16_t res[16], coef[16];
        for (y = 0; y < 4; y++)
            for (x = 0; x < 4; x++)
                res[y * 4 + x] = (int16_t)((int)src[(by + y) * e->stride[0] + bx + x] -
                                           (int)pred[(by + y) * 16 + bx + x]);
        aeg_transform_4x4(res, 4, coef);
        dc[blk] = coef[0];
        coef[0] = 0;
        aeg_quant_4x4(coef, c->level[blk], qp, 1);
    }
    aeg_transform_dc4x4_fwd(dc, had);
    aeg_quant_4x4_dc(had, dcled, qp, 1);
    for (i = 0; i < 16; i++)
        nz_dc += dcled[i] != 0;
    memcpy(c->level[AEG_BLK_LUMA_DC], dcled, sizeof(dcled));
    c->nz[AEG_BLK_LUMA_DC] = (int8_t)nz_dc;
    aeg_dequant_4x4_dc_out(dcled, dchad, qp);
    aeg_dc_block_unpermute(dchad);   /* decoder slot order -> raster order */

    for (blk = 0; blk < 16; blk++) {
        int bx = (blk & 3) * 4, by = (blk >> 2) * 4;
        uint8_t *r = dst + by * e->stride[0] + bx;
        const uint8_t *pr = pred + by * 16 + bx;
        int32_t dcv = dchad[blk];
        int nz = 0;
        for (i = 0; i < 16; i++)
            nz += c->level[blk][i] != 0;
        c->nz[blk] = (int8_t)nz;
        for (y = 0; y < 4; y++)
            for (x = 0; x < 4; x++)
                r[y * e->stride[0] + x] = pr[y * 16 + x];
        aeg_idct_add_block(r, e->stride[0], c->level[blk], qp, r, e->stride[0], &dcv);
        for (y = 0; y < 4; y++)
            for (x = 0; x < 4; x++) {
                int d = (int)r[y * e->stride[0] + x] -
                        (int)src[(by + y) * e->stride[0] + bx + x];
                c->sse += (int64_t)d * d;
            }
    }
    /* CAVLC Intra_16x16 always codes all sixteen AC blocks; the luma part of
     * the coded block pattern comes from the macroblock type. */
    c->cbp = (c->cbp & 0x30) | 0x0f;

    /* The intra 16x16 DC block is context-coded with the nC of luma block
     * (0, 0) -- the reference decoder computes pred_non_zero_count(0) for it
     * (n >= LUMA_DC_BLOCK_INDEX -> (n - 48) * 16). */
    if (nz_dc)
        c->bits += aeg_cavlc_block_cost(c->level[AEG_BLK_LUMA_DC],
                                        luma_ac_nc(e, mbx, mby, c, 0), 1);
    for (blk = 0; blk < 16; blk++)
        if (c->nz[ac_ord[blk]])
            c->bits += aeg_cavlc_block_cost(c->level[ac_ord[blk]],
                                            luma_ac_nc(e, mbx, mby, c, blk), 3);
}

/* 8x8 group index (0..3, raster) of a 4x4 luma block: the coded block pattern
 * has one bit per group, not one per block row. */
static int blk_group(int blk)
{
    int bx = blk & 3, by = blk >> 2;
    return (by >> 1) * 2 + (bx >> 1);
}

/* The four 4x4 blocks of every 8x8 group, in the order the bitstream uses:
 * group by group (TL, TR, BL, BR), each group in raster order. */
static const uint8_t group_blk[4][4] = {
    { 0, 1, 4, 5 }, { 2, 3, 6, 7 }, { 8, 9, 12, 13 }, { 10, 11, 14, 15 }
};

/* ------------------------------------------------------------------ */
/* Luma coding: inter                                                  */
/* ------------------------------------------------------------------ */
/* Residual of an inter prediction: the sixteen 4x4 blocks of the macroblock,
 * contiguous (16 int16 per block, raster order inside the block). */
static void luma_inter_gather(const struct aeg_encoder *e, const uint8_t *src,
                              const uint8_t *pred, int16_t *res)
{
    int blk, x, y;
    for (blk = 0; blk < 16; blk++) {
        int bx = (blk & 3) * 4, by = (blk >> 2) * 4;
        int16_t *r = res + 16 * blk;
        for (y = 0; y < 4; y++)
            for (x = 0; x < 4; x++)
                r[y * 4 + x] = (int16_t)((int)src[(by + y) * e->stride[0] + bx + x] -
                                         (int)pred[(by + y) * 16 + bx + x]);
    }
}

/* Everything an inter candidate needs once its coefficients are known: the
 * non-zero counts, the coded block pattern, the reconstruction written into
 * the picture, the SSE and the CAVLC cost of the residual.
 *
 * `lev` points at the sixteen quantised blocks (contiguous, 16 int16 each) and
 * is copied into the candidate's own array.  `nz` carries the non-zero counts
 * the device kernel produced (NULL: count them here).  `rec` carries the
 * sixteen reconstructed residual blocks when they come from a batch (NULL:
 * reconstruct from the levels here, which is the SSE2 per-block path).  Both
 * forms do the same arithmetic, so a candidate costs the same either way --
 * that is what lets the batched path be bit-exact to the serial one. */
static void luma_inter_finish(struct aeg_encoder *e, int mbx, int mby, mbenc *c,
                              const int16_t *lev, const int32_t *nz,
                              const int16_t *rec, const uint8_t *pred)
{
    const uint8_t *src = e->srcp[0] + mby * 16 * e->stride[0] + mbx * 16;
    uint8_t *dst = e->recp[0] + mby * 16 * e->stride[0] + mbx * 16;
    int blk, i, x, y, luma_cbp = 0, qp = c->qp;

    if (lev != &c->level[0][0])
        memcpy(c->level, lev, 16 * 16 * sizeof(int16_t));
    for (blk = 0; blk < 16; blk++) {
        int bx = (blk & 3) * 4, by = (blk >> 2) * 4;
        uint8_t *r = dst + by * e->stride[0] + bx;
        const uint8_t *pr = pred + by * 16 + bx;
        int n = 0;

        if (nz) {
            n = nz[blk];
        } else {
            for (i = 0; i < 16; i++)
                n += c->level[blk][i] != 0;
        }
        c->nz[blk] = (int8_t)n;
        if (n)
            luma_cbp |= 1 << blk_group(blk);
        if (rec) {
            const int16_t *rr = rec + 16 * blk;
            for (y = 0; y < 4; y++)
                for (x = 0; x < 4; x++) {
                    int p = (int)pr[y * 16 + x] + (int)rr[y * 4 + x];
                    r[y * e->stride[0] + x] =
                        (uint8_t)(p < 0 ? 0 : (p > 255 ? 255 : p));
                }
        } else {
            for (y = 0; y < 4; y++)
                for (x = 0; x < 4; x++)
                    r[y * e->stride[0] + x] = pr[y * 16 + x];
            aeg_idct_add_block(r, e->stride[0], c->level[blk], qp, r,
                               e->stride[0], NULL);
        }
        for (y = 0; y < 4; y++)
            for (x = 0; x < 4; x++) {
                int d = (int)r[y * e->stride[0] + x] -
                        (int)src[(by + y) * e->stride[0] + bx + x];
                c->sse += (int64_t)d * d;
            }
    }
    c->cbp = (c->cbp & 0x30) | luma_cbp;
    {
        int g, k;
        for (g = 0; g < 4; g++) {
            if (!(c->cbp & (1 << g)))
                continue;
            for (k = 0; k < 4; k++) {
                int b = group_blk[g][k];
                c->bits += aeg_cavlc_block_cost(c->level[b],
                                                luma_nc(e, mbx, mby, c, b), 0);
            }
        }
    }
}

static void code_luma_inter(struct aeg_encoder *e, int mbx, int mby, mbenc *c)
{
    const uint8_t *src = e->srcp[0] + mby * 16 * e->stride[0] + mbx * 16;
    const uint8_t *pred = e->pred_l;
    int blk, qp = c->qp;

    luma_inter_gather(e, src, pred, e->coef_res);
    if (e->coef_batch) {
        /* One transform+quantise call and one inverse call for the sixteen
         * blocks instead of sixteen of each. */
        int64_t t0 = aeg_now_ns();
        e->d.ops.tq_batch(e->d.ops.priv, e->coef_res, 16, qp, 0,
                          e->coef_lev, e->coef_nz);
        aeg_dispatch_record(&e->d, AEG_STAGE_TRANSFORM,
                            e->d.ops.backend != AEG_BACKEND_CPU,
                            aeg_now_ns() - t0);
        t0 = aeg_now_ns();
        e->d.ops.iq_idct_batch(e->d.ops.priv, e->coef_lev, 16, qp, e->coef_rec);
        aeg_dispatch_record(&e->d, AEG_STAGE_IQUANT_IDCT,
                            e->d.ops.backend != AEG_BACKEND_CPU,
                            aeg_now_ns() - t0);
        luma_inter_finish(e, mbx, mby, c, e->coef_lev, e->coef_nz, e->coef_rec,
                          pred);
        return;
    }
    for (blk = 0; blk < 16; blk++) {
        int16_t coef[16];
        aeg_transform_4x4(e->coef_res + 16 * blk, 4, coef);
        aeg_quant_4x4(coef, c->level[blk], qp, 0);
    }
    luma_inter_finish(e, mbx, mby, c, &c->level[0][0], NULL, NULL, pred);
}

/* ------------------------------------------------------------------ */
/* Luma coding: the 8x8 transform (High profile)                       */
/* ------------------------------------------------------------------ */
/* Which 8x8 block a 4x4 slot belongs to (the coded block pattern has one bit
 * per 8x8 block) is blk_group() above; the four 4x4 blocks of a group, in the
 * order their coefficient lists are transmitted, are group_blk[][]. */

/* Coefficients, residual and reconstruction of one 8x8 luma block.
 *
 * `pred` holds the predictor of the block (stride pred_stride); the
 * reconstruction is written straight into the picture, because the blocks that
 * follow predict from it -- for Intra_8x8 that includes the next block of the
 * same macroblock. */
static void code_luma8_block(struct aeg_encoder *e, int mbx, int mby, mbenc *c,
                             int g, const uint8_t *pred, int pred_stride,
                             int intra)
{
    const int s = e->stride[0];
    const uint8_t *src = e->srcp[0] + mby * 16 * s + mbx * 16;
    uint8_t *dst = e->recp[0] + mby * 16 * s + mbx * 16;
    const int bx = (g & 1) * 8, by = (g >> 1) * 8;
    int16_t res[64], coef[64];
    int i, x, y;

    for (y = 0; y < 8; y++)
        for (x = 0; x < 8; x++)
            res[y * 8 + x] = (int16_t)((int)src[(by + y) * s + bx + x] -
                                       (int)pred[y * pred_stride + x]);
    aeg_transform_8x8(res, 8, coef);
    aeg_quant_8x8(coef, c->level8[g], c->qp, intra);
    c->nz8[g] = 0;
    for (i = 0; i < 64; i++)
        c->nz8[g] |= c->level8[g][i] != 0;

    aeg_idct_8x8_add(dst + by * s + bx, s, c->level8[g], c->qp, pred,
                     pred_stride);
    for (y = 0; y < 8; y++)
        for (x = 0; x < 8; x++) {
            int d = (int)dst[(by + y) * s + bx + x] -
                    (int)src[(by + y) * s + bx + x];
            c->sse += (int64_t)d * d;
        }
}

/* Coded block pattern, contexts and CAVLC cost of the four 8x8 luma blocks.
 *
 * The order is the transmission order: the groups in raster order, and the
 * four coefficient lists of a group in the order of group_blk[].  The context
 * of a list comes from the neighbouring 4x4 slots, so the slots have to hold
 * what the reference decoder holds when it reaches the same list: a group's
 * four slots start undefined, each list's slot takes that list's TotalCoeff as
 * soon as the list has been read, and when the group is finished the decoder
 * adds the four counts into the group's first slot (decode_luma_residual), so
 * the blocks below and to the right of an 8x8 block see its total.  A group
 * whose coded block pattern bit is clear has all four slots zeroed.  Only
 * slots of blocks that have already been coded are ever read, so coding in
 * this order keeps both sides in step. */
static void luma8_group_done(mbenc *c, int g)
{
    int k, sum = 0;
    for (k = 0; k < 4; k++)
        sum += c->nz[group_blk[g][k]];
    c->nz[group_blk[g][0]] = (int8_t)sum;
}

static void luma8_costs(struct aeg_encoder *e, int mbx, int mby, mbenc *c)
{
    int g, k, cbp = c->cbp & 0x30;

    for (g = 0; g < 4; g++) {
        int flag = c->nz8[g] != 0;
        if (flag)
            cbp |= 1 << g;
        for (k = 0; k < 4; k++) {
            int b = group_blk[g][k];
            if (flag) {
                c->bits += aeg_cavlc_block_cost8(c->level8[g], k,
                                                 luma_nc(e, mbx, mby, c, b));
                c->nz[b] = (int8_t)aeg_cavlc_nz8(c->level8[g], k);
            } else {
                c->nz[b] = 0;
            }
        }
        luma8_group_done(c, g);
    }
    c->cbp = cbp;
}

/* Inter residual with the 8x8 transform.  The prediction of the whole
 * macroblock is already in e->pred_l (every partition is motion compensated
 * before the residual is coded), so the four blocks are independent. */
static void code_luma_inter8(struct aeg_encoder *e, int mbx, int mby, mbenc *c)
{
    int g;
    c->t8 = 1;
    for (g = 0; g < 4; g++)
        code_luma8_block(e, mbx, mby, c, g,
                         e->pred_l + (g >> 1) * 8 * 16 + (g & 1) * 8, 16, 0);
    luma8_costs(e, mbx, mby, c);
}

/* Whether transform_size_8x8_flag may be signalled for this macroblock at all
 * (7.3.5.2): an inter macroblock may only use the 8x8 transform when none of
 * its partitions is smaller than 8x8, and a decoder does not even read the
 * flag otherwise -- writing it there desynchronises the stream. */
static int t8_allowed(const struct aeg_encoder *e, const mbenc *c)
{
    int p;
    if (!e->p.transform_8x8)
        return 0;
    if (!is_inter_type(c->type) || c->type != AEG_MB_P8x8)
        return 1;
    for (p = 0; p < 4; p++)
        if (c->sub_type[p] != 0)        /* 0 = 8x8, 1 = 8x4, 2 = 4x8, 3 = 4x4 */
            return 0;
    return 1;
}

/* Which transform a macroblock's luma residual should use.  The reference
 * encoder's test: compare the summed 4x4 SATD of the residual with its summed
 * 8x8 SA8D, both scaled to the same units.  A residual whose detail lives
 * inside the 4x4 blocks (busy areas) costs more in the 8x8 measure, because
 * the 8x8 transform cannot represent that detail without ringing; a smooth
 * residual costs more in the 4x4 measure, because sixteen small transforms
 * spread it over more coefficients than four large ones. */
static int use_t8(const struct aeg_encoder *e, int mbx, int mby, const mbenc *c)
{
    const int s = e->stride[0];
    const uint8_t *src = e->srcp[0] + mby * 16 * s + mbx * 16;
    int g, x, y, cost4 = 0, cost8 = 0;

    if (!t8_allowed(e, c))
        return 0;
    for (g = 0; g < 4; g++) {
        const int bx = (g & 1) * 8, by = (g >> 1) * 8;
        cost8 += aeg_sa8d_8x8(src + by * s + bx, s, e->pred_l + by * 16 + bx,
                              16);
        for (y = 0; y < 8; y += 4)
            for (x = 0; x < 8; x += 4)
                cost4 += aeg_satd_4x4(src + (by + y) * s + bx + x, s,
                                      e->pred_l + (by + y) * 16 + bx + x, 16);
    }
    return cost8 < cost4;
}

/* ------------------------------------------------------------------ */
/* Chroma coding                                                       */
/* ------------------------------------------------------------------ */
static void code_chroma(struct aeg_encoder *e, int mbx, int mby, mbenc *c,
                        int inter)
{
    static const int bxy[4][2] = { {0, 0}, {4, 0}, {0, 4}, {4, 4} };
    int qpc = aeg_qp_chroma(c->qp, 0);
    int pl, blk, i, x, y;
    int avail_l = mbx > 0, avail_t = mby > e->top_row;

    c->cbp &= 0x0f;
    for (pl = 0; pl < 2; pl++) {
        uint8_t *base = e->recp[1 + pl] + mby * 8 * e->stride[1] + mbx * 8;
        const uint8_t *src = e->srcp[1 + pl] + mby * 8 * e->stride[1] + mbx * 8;
        uint8_t *pred = e->pred_c[pl];
        int16_t dc[4], had[4], dclev[4];
        int32_t dcout[4];
        int nz_dc = 0;

        if (!inter) {
            /* The predictor wants a contiguous left column; the picture is
             * strided, so gather it first (same as the luma path). */
            uint8_t leftc[8];
            for (y = 0; y < 8; y++)
                leftc[y] = avail_l ? base[y * e->stride[1] - 1] : 128;
            aeg_intra_predict_chroma(pred, 8, base - e->stride[1], leftc,
                                     avail_t, avail_l, avail_t && avail_l,
                                     c->chroma_mode);
        }
        for (blk = 0; blk < 4; blk++) {
            int16_t res[16], coef[16];
            for (y = 0; y < 4; y++)
                for (x = 0; x < 4; x++)
                    res[y * 4 + x] =
                        (int16_t)((int)src[(bxy[blk][1] + y) * e->stride[1] +
                                          bxy[blk][0] + x] -
                                  (int)pred[(bxy[blk][1] + y) * 8 + bxy[blk][0] + x]);
            aeg_transform_4x4(res, 4, coef);
            dc[blk] = coef[0];
            coef[0] = 0;
            aeg_quant_4x4(coef, c->level[AEG_BLK_CHROMA_CB + pl * 4 + blk],
                          qpc, 1);
        }
        aeg_transform_dc2x2_fwd(dc, had);
        aeg_quant_2x2_dc(had, dclev, qpc, 1);
        for (i = 0; i < 4; i++)
            nz_dc += dclev[i] != 0;
        memcpy(c->level[AEG_BLK_CHROMA_DC + pl], dclev, sizeof(dclev));
        c->nz[AEG_BLK_CHROMA_DC + pl] = (int8_t)nz_dc;
        aeg_dequant_2x2_dc_out(dclev, dcout, qpc);
        for (blk = 0; blk < 4; blk++) {
            uint8_t tmp[8 * 8];
            const uint8_t *pr = pred + bxy[blk][1] * 8 + bxy[blk][0];
            int32_t dcv = dcout[blk];
            int nz = 0;

            for (i = 0; i < 16; i++)
                nz += c->level[AEG_BLK_CHROMA_CB + pl * 4 + blk][i] != 0;
            c->nz[AEG_BLK_CHROMA_CB + pl * 4 + blk] = (int8_t)nz;
            if (nz)
                c->cbp |= 0x20;

            for (y = 0; y < 4; y++)
                for (x = 0; x < 4; x++)
                    tmp[y * 8 + x] = pr[y * 8 + x];
            aeg_idct_add_block(tmp, 8, c->level[AEG_BLK_CHROMA_CB + pl * 4 + blk],
                               qpc, tmp, 8, &dcv);
            for (y = 0; y < 4; y++)
                for (x = 0; x < 4; x++) {
                    int v = tmp[y * 8 + x];
                    int s = src[(bxy[blk][1] + y) * e->stride[1] + bxy[blk][0] + x];
                    int d = v - s;
                    c->sse += (int64_t)d * d;
                    base[(bxy[blk][1] + y) * e->stride[1] + bxy[blk][0] + x] = (uint8_t)v;
                }
        }
        if (nz_dc) {
            c->cbp |= (pl == 0) ? 0x10 : 0x10;
            c->bits += aeg_cavlc_block_cost(c->level[AEG_BLK_CHROMA_DC + pl], 0, 2);
        }
        for (blk = 0; blk < 4; blk++)
            if (c->nz[AEG_BLK_CHROMA_CB + pl * 4 + blk])
                c->bits += aeg_cavlc_block_cost(
                    c->level[AEG_BLK_CHROMA_CB + pl * 4 + blk],
                    chroma_nc(e, mbx, mby, c, pl, blk), 3);
    }
}

/* ------------------------------------------------------------------ */
/* Prediction candidates                                               */
/* ------------------------------------------------------------------ */
static int intra16_sad(struct aeg_encoder *e, int mbx, int mby, int mode)
{
    const uint8_t *p = e->recp[0] + mby * 16 * e->stride[0] + mbx * 16;
    const uint8_t *src = e->srcp[0] + mby * 16 * e->stride[0] + mbx * 16;
    uint8_t left[16];
    int y;
    /* The predictor expects the left column contiguously. */
    for (y = 0; y < 16; y++)
        left[y] = p[-1 + y * e->stride[0]];
    aeg_intra_predict_16x16(e->pred_l, 16, p - e->stride[0], left,
                            mby > e->top_row, mbx > 0,
                            mby > e->top_row && mbx > 0, mode);
    return aeg_sad(16, 16, e->pred_l, 16, src, e->stride[0]);
}

static int chroma_sad(struct aeg_encoder *e, int mbx, int mby, int pl, int mode)
{
    const uint8_t *p = e->recp[1 + pl] + mby * 8 * e->stride[1] + mbx * 8;
    const uint8_t *src = e->srcp[1 + pl] + mby * 8 * e->stride[1] + mbx * 8;
    uint8_t left[8];
    int y;
    for (y = 0; y < 8; y++)
        left[y] = p[-1 + y * e->stride[1]];
    aeg_intra_predict_chroma(e->pred_c[pl], 8, p - e->stride[1], left,
                             mby > e->top_row, mbx > 0,
                             mby > e->top_row && mbx > 0, mode);
    return aeg_sad(8, 8, e->pred_c[pl], 8, src, e->stride[1]);
}

/* Motion compensation of one rectangle of the macroblock.  The vector is in
 * quarter luma samples: the integer part moves the reference pointer, the
 * remainder selects the interpolation phase.  Chroma runs at eighth-sample
 * precision, so the same vector is already in eighth chroma samples (one luma
 * quarter pel is half a chroma sample) and the chroma rectangle is half the
 * size. */
static void mc_rect_luma(struct aeg_encoder *e, int mbx, int mby,
                         int x, int y, int w, int h, int mvx, int mvy)
{
    const uint8_t *r = e->refp[0] + (mby * 16 + y + (mvy >> 2)) * e->stride[0] +
                       mbx * 16 + x + (mvx >> 2);
    if (e->mc.ready)
        aeg_mc_luma_p(e->pred_l + y * 16 + x, 16, &e->mc, r, w, h,
                      mvx & 3, mvy & 3);
    else
        aeg_mc_luma(e->pred_l + y * 16 + x, 16, r, e->stride[0], w, h,
                    mvx & 3, mvy & 3);
}

static void mc_rect_chroma(struct aeg_encoder *e, int mbx, int mby,
                           int cx, int cy, int cw, int ch, int mvx, int mvy)
{
    int pl;
    for (pl = 0; pl < 2; pl++) {
        const uint8_t *rc = e->refp[1 + pl] + (mby * 8 + cy + (mvy >> 3)) * e->stride[1] +
                            mbx * 8 + cx + (mvx >> 3);
        aeg_mc_chroma(e->pred_c[pl] + cy * 8 + cx, 8, rc, e->stride[1],
                      cw, ch, mvx & 7, mvy & 7);
    }
}

/* Inter prediction of the 16x16 luma block and both chroma planes. */
static void inter_predict(struct aeg_encoder *e, int mbx, int mby, int mvx, int mvy)
{
    mc_rect_luma(e, mbx, mby, 0, 0, 16, 16, mvx, mvy);
    mc_rect_chroma(e, mbx, mby, 0, 0, 8, 8, mvx, mvy);
}

/* Prediction of a whole macroblock coded with `type`, using the per-4x4 MV
 * table of that candidate (every 4x4 block of a partition shares its MV). */
static void inter_predict_typed(struct aeg_encoder *e, int mbx, int mby,
                                const mbenc *c)
{
    int p, n = mb_part_count(c);
    for (p = 0; p < n; p++) {
        int geo[4], b0;
        mb_part_geom(c, p, geo, &b0);
        mc_rect_luma(e, mbx, mby, geo[0], geo[1], geo[2], geo[3],
                     c->mv[b0][0], c->mv[b0][1]);
        mc_rect_chroma(e, mbx, mby, geo[0] / 2, geo[1] / 2,
                       geo[2] / 2, geo[3] / 2, c->mv[b0][0], c->mv[b0][1]);
    }
}

/* MV prediction of one inter partition (H.264 8.4.1.3.1).
 *
 * Partitioned macroblocks do *not* use the plain three-neighbour median: the
 * reference decoder (and x264) short-circuits to a single neighbour as soon as
 * it carries the same reference picture, and its third neighbour differs per
 * partition shape.  The rules below reproduce that derivation exactly --
 * getting them wrong does not desynchronise the stream, but it makes the
 * encoder reconstruct with a different motion vector than the decoder. */

/* Availability of a neighbouring block: -2 not available (outside the
 * picture), -1 available but not inter (intra, so list 0 is not used), 0 the
 * neighbour predicts from reference 0 with a usable vector. */
#define NB_UNAVAIL (-2)
#define NB_NOTUSED (-1)
#define NB_OK        0

typedef struct {
    int ref;
    int mvx, mvy;
} nb_mv;

static void nb_set(nb_mv *n, int ref, int mvx, int mvy)
{
    n->ref = ref; n->mvx = mvx; n->mvy = mvy;
}

/* Fill `n` from macroblock `m` (NULL when the neighbour does not exist). */
static void nb_from_mb(nb_mv *n, const aeg_mb *m, int blk)
{
    if (!m) {
        nb_set(n, NB_UNAVAIL, 0, 0);
        return;
    }
    if (is_intra_type(m->type)) {
        nb_set(n, NB_NOTUSED, 0, 0);
        return;
    }
    nb_set(n, NB_OK, m->mv[blk][0], m->mv[blk][1]);
}

/* Median of three motion vector components (H.264 8.4.1.3.1). */
static int mv_median3(int a, int b, int c)
{
    int mx = a > b ? a : b;
    int mn = a < b ? a : b;
    return c > mx ? mx : (c < mn ? mn : c);
}

/* Motion vector prediction of the partition at 4x4 block (bx, by) that is
 * `pw` blocks wide, in macroblock (mbx, mby).  `cur` holds the vectors already
 * decided for this macroblock so that the following partitions see the earlier
 * ones as neighbours, and `avail` says which of those blocks the decoder has
 * actually filled in already: it starts every macroblock with all sixteen 4x4
 * cache slots PART_NOT_AVAILABLE and fills them partition by partition, so a
 * neighbour that belongs to a *later* sub-partition counts as unavailable and
 * the decoder's fallback applies (getting this wrong makes the encoder predict
 * a different vector than the decoder -- a silent reconstruction mismatch, not
 * a bitstream error).  `type` and `p` only select the shape-specific rule of
 * the reference decoder (16x8 and 8x16 short-circuit to a single neighbour;
 * every other partition, including all the 8x8 sub-partitions, uses the plain
 * three-neighbour process). */
static void predict_mv_at(struct aeg_encoder *e, int mbx, int mby,
                          const int16_t cur[16][2], const uint8_t *avail,
                          int type, int p, int bx, int by, int pw,
                          int *pmvx, int *pmvy)
{
    nb_mv A, B, C;
    int match;
#define AVAIL(x, y) (avail[(y) * 4 + (x)])

    /* A: the block left of the partition (inside this macroblock when bx > 0). */
    if (bx > 0) {
        if (AVAIL(bx - 1, by))
            nb_set(&A, NB_OK, cur[(by << 2) + bx - 1][0],
                   cur[(by << 2) + bx - 1][1]);
        else
            nb_from_mb(&A, NULL, 0);
    } else if (mbx > 0) {
        nb_from_mb(&A, &e->mbs[mby * e->mb_w + mbx - 1], (by << 2) + 3);
    } else {
        nb_from_mb(&A, NULL, 0);
    }

    /* B: the block above the partition (inside this macroblock when by > 0). */
    if (by > 0) {
        if (AVAIL(bx, by - 1))
            nb_set(&B, NB_OK, cur[((by - 1) << 2) + bx][0],
                   cur[((by - 1) << 2) + bx][1]);
        else
            nb_from_mb(&B, NULL, 0);
    } else if (mby > e->top_row) {
        nb_from_mb(&B, &e->mbs[(mby - 1) * e->mb_w + mbx], 12 + bx);
    } else {
        nb_from_mb(&B, NULL, 0);
    }

    /* C: the block above and to the right of the partition.  Which block that
     * is depends on the partition width: a full-width partition reaches into
     * the macroblock above and to the right, a half-width one still sits above
     * the current macroblock.  A block whose right edge is the macroblock's
     * right edge has no above-right neighbour at all (the cache slot it would
     * use belongs to a sub-block coded later, and the decoder leaves it
     * PART_NOT_AVAILABLE), so the decoder's fallback applies: the block to the
     * left of the partition's above neighbour. */
    {
        int cx = bx + pw, cy = by - 1;
        if (cy >= 0 && cx <= 3 && AVAIL(cx, cy)) {
            nb_set(&C, NB_OK, cur[(cy << 2) + cx][0], cur[(cy << 2) + cx][1]);
        } else if (cy >= 0) {
            nb_set(&C, NB_UNAVAIL, 0, 0);
        } else if (cx <= 3) {
            if (mby > e->top_row)
                nb_from_mb(&C, &e->mbs[(mby - 1) * e->mb_w + mbx], 12 + cx);
            else
                nb_from_mb(&C, NULL, 0);
        } else {
            if (mby > e->top_row && mbx + 1 < e->mb_w)
                nb_from_mb(&C, &e->mbs[(mby - 1) * e->mb_w + mbx + 1], 12);
            else
                nb_from_mb(&C, NULL, 0);
        }
        if (C.ref == NB_UNAVAIL) {
            /* The decoder's fallback is the cache slot one before the block
             * directly above the partition, i.e. the block above-left of the
             * partition's top-left corner. */
            if (by > 0 && bx > 0) {
                if (AVAIL(bx - 1, by - 1))
                    nb_set(&C, NB_OK, cur[((by - 1) << 2) + bx - 1][0],
                           cur[((by - 1) << 2) + bx - 1][1]);
                else
                    nb_from_mb(&C, NULL, 0);
            } else if (by > 0) {
                if (mbx > 0)
                    nb_from_mb(&C, &e->mbs[mby * e->mb_w + mbx - 1],
                               ((by - 1) << 2) + 3);
                else
                    nb_from_mb(&C, NULL, 0);
            } else if (bx > 0 && mby > e->top_row) {
                nb_from_mb(&C, &e->mbs[(mby - 1) * e->mb_w + mbx], 12 + bx - 1);
            } else if (mby > e->top_row && mbx > 0) {
                nb_from_mb(&C, &e->mbs[(mby - 1) * e->mb_w + mbx - 1], 15);
            } else {
                nb_from_mb(&C, NULL, 0);
            }
        }
    }

    /* The decoder prefers a single neighbour that shares the reference. */
    if (type == AEG_MB_P16x8) {
        if (p == 0 && B.ref == NB_OK) {
            *pmvx = B.mvx; *pmvy = B.mvy;
            return;
        }
        if (p == 1 && A.ref == NB_OK) {
            *pmvx = A.mvx; *pmvy = A.mvy;
            return;
        }
    } else if (type == AEG_MB_P8x8) {
        /* Sub-macroblock partitions have no short-circuit of their own: the
         * reference decoder runs the ordinary three-neighbour process. */
    } else {
        if (p == 0 && A.ref == NB_OK) {
            *pmvx = A.mvx; *pmvy = A.mvy;
            return;
        }
        if (p == 1 && C.ref == NB_OK) {
            *pmvx = C.mvx; *pmvy = C.mvy;
            return;
        }
    }

    match = (C.ref == NB_OK) + (B.ref == NB_OK) + (A.ref == NB_OK);
    if (match == 1) {
        if (A.ref == NB_OK)      { *pmvx = A.mvx; *pmvy = A.mvy; }
        else if (B.ref == NB_OK) { *pmvx = B.mvx; *pmvy = B.mvy; }
        else                     { *pmvx = C.mvx; *pmvy = C.mvy; }
    } else if (match == 0 && B.ref == NB_UNAVAIL && C.ref == NB_UNAVAIL &&
               A.ref != NB_UNAVAIL) {
        *pmvx = A.mvx; *pmvy = A.mvy;
    } else {
        *pmvx = mv_median3(A.mvx, B.mvx, C.mvx);
        *pmvy = mv_median3(A.mvy, B.mvy, C.mvy);
    }
#undef AVAIL
}

/* Motion vector of a P_Skip macroblock (H.264 8.4.1.1).  It is *not* the
 * median predictor: as soon as the left or the top neighbour is missing, is
 * intra, or sits at a zero vector, the skipped block's vector is (0,0).
 * Decoders reconstruct skipped macroblocks with exactly this rule, so the
 * encoder has to apply it before it may code one as a skip. */
static void pskip_mv(const struct aeg_encoder *e, int mbx, int mby,
                     int *mvx, int *mvy)
{
    const aeg_mb *m;
    int ax = 0, ay = 0, bx = 0, by = 0, cx = 0, cy = 0;
    int refa = -1, refb = -1, refc = -1, match;

    *mvx = *mvy = 0;
    if (mbx <= 0 || mby <= e->top_row)
        return;
    m = &e->mbs[mby * e->mb_w + mbx - 1];          /* A: left */
    if (is_inter_type(m->type)) {
        ax = m->mv[3][0]; ay = m->mv[3][1]; refa = 0;
        if (ax == 0 && ay == 0)
            return;
    }
    m = &e->mbs[(mby - 1) * e->mb_w + mbx];        /* B: above */
    if (is_inter_type(m->type)) {
        bx = m->mv[12][0]; by = m->mv[12][1]; refb = 0;
        if (bx == 0 && by == 0)
            return;
    }
    if (mbx + 1 < e->mb_w) {                       /* C: above-right */
        m = &e->mbs[(mby - 1) * e->mb_w + mbx + 1];
        if (is_inter_type(m->type)) {
            cx = m->mv[12][0]; cy = m->mv[12][1]; refc = 0;
        }
    } else {                                       /* ... else above-left */
        m = &e->mbs[(mby - 1) * e->mb_w + mbx - 1];
        if (is_inter_type(m->type)) {
            cx = m->mv[15][0]; cy = m->mv[15][1]; refc = 0;
        }
    }
    match = (refc == 0) + (refb == 0) + (refa == 0);
    if (match == 1) {
        if (refa == 0)      { *mvx = ax; *mvy = ay; }
        else if (refb == 0) { *mvx = bx; *mvy = by; }
        else                { *mvx = cx; *mvy = cy; }
    } else {
        *mvx = mv_median3(ax, bx, cx);
        *mvy = mv_median3(ay, by, cy);
    }
}

/* ------------------------------------------------------------------ */
/* Frame level helpers                                                 */
/* ------------------------------------------------------------------ */
static void reset_mb_enc(mbenc *c, int qp)
{
    memset(c, 0, sizeof(*c));
    c->qp = qp;
    c->type = AEG_MB_I16x16;
    c->nparts = 1;
}

/* Chroma prediction modes in bitstream order: 0 = DC, 1 = horizontal,
 * 2 = vertical, 3 = plane.  Only the modes whose references exist are legal;
 * "exists" means "inside this slice", so the first row of a slice counts like
 * the first row of the picture (7.4.1.2.4). */
static int chroma_mode_ok(const struct aeg_encoder *e, int cm, int mbx, int mby)
{
    if ((cm == 1 || cm == 3) && mbx == 0)
        return 0;
    if ((cm == 2 || cm == 3) && mby <= e->top_row)
        return 0;
    return 1;
}

static int luma_mode_ok(const struct aeg_encoder *e, int mode, int mbx, int mby)
{
    if ((mode == AEG_MB_I_V || mode == AEG_MB_I_P) && mby <= e->top_row)
        return 0;
    if ((mode == AEG_MB_I_H || mode == AEG_MB_I_P) && mbx == 0)
        return 0;
    return 1;
}

/* ------------------------------------------------------------------ */
/* Decision: intra candidates, chroma candidates, inter candidate      */
/* ------------------------------------------------------------------ */
/* Best chroma prediction mode by SAD (the same mode is used for Cb and Cr). */
static int chroma_mode_by_sad(struct aeg_encoder *e, int mbx, int mby)
{
    int cm, best = 0, best_sad = 1 << 30;
    for (cm = 0; cm < 4; cm++) {
        int s;
        if (!chroma_mode_ok(e, cm, mbx, mby))
            continue;
        s = chroma_sad(e, mbx, mby, 0, cm) + chroma_sad(e, mbx, mby, 1, cm);
        if (s < best_sad) { best_sad = s; best = cm; }
    }
    return best;
}

/* Copy a partition's vector into the whole 4x4 area it covers (the bitstream
 * only carries one mvd per partition, but reconstruction and the predictor of
 * later partitions need it per block). */
static void cand_publish(mbcand *c, const int geo[4], int mvx, int mvy)
{
    int x, y, x0 = geo[0] / 4, y0 = geo[1] / 4;
    int x1 = x0 + geo[2] / 4, y1 = y0 + geo[3] / 4;
    for (y = y0; y < y1; y++)
        for (x = x0; x < x1; x++) {
            c->mv[(y << 2) + x][0] = mvx;
            c->mv[(y << 2) + x][1] = mvy;
        }
}

/* Publish one partition's vector in the 4x4 MV table of the macroblock being
 * searched, so that the following partitions can predict from it. */
static void me_publish(int16_t cur[16][2], const int geo[4], int mvx, int mvy)
{
    int x, y, x0 = geo[0] / 4, y0 = geo[1] / 4;
    int x1 = x0 + geo[2] / 4, y1 = y0 + geo[3] / 4;
    for (y = y0; y < y1; y++)
        for (x = x0; x < x1; x++) {
            cur[(y << 2) + x][0] = (int16_t)mvx;
            cur[(y << 2) + x][1] = (int16_t)mvy;
        }
}

/* Sum of absolute transformed differences of one partition at a given motion
 * vector.  The motion search itself is SAD driven (it has to look at hundreds
 * of candidates), but the *decision* between partition shapes is made on SATD:
 * SAD happily prefers a wrongly aligned block whose DC error happens to be
 * small, which is exactly the mistake a sub-partition decision cannot afford. */
static int part_satd(struct aeg_encoder *e, int mbx, int mby, const int geo[4],
                     int mvx, int mvy)
{
    uint8_t buf[16 * 16];
    int ix = mvx >> 2, iy = mvy >> 2;
    {
        const uint8_t *r = e->refp[0] + (mby * 16 + geo[1] + iy) * e->stride[0] +
                           mbx * 16 + geo[0] + ix;
        if (e->mc.ready)
            aeg_mc_luma_p(buf, geo[2], &e->mc, r, geo[2], geo[3], mvx & 3, mvy & 3);
        else
            aeg_mc_luma(buf, geo[2], r, e->stride[0], geo[2], geo[3],
                        mvx & 3, mvy & 3);
    }
    return aeg_satd(geo[2], geo[3], buf, geo[2],
                    e->srcp[0] + (mby * 16 + geo[1]) * e->stride[0] + mbx * 16 +
                    geo[0], e->stride[0]);
}

/* Motion search for one partition: its own search window, centred on its own
 * MV predictor.  `refine` selects the quarter-pel stage: the P_8x8 shape trials
 * run integer only (they compare shapes, not vectors) and the winning shape is
 * searched again with the refinement, which is where the quality comes from. */
static void me_partition(struct aeg_encoder *e, int mbx, int mby, int type,
                         int p, const int geo[4], int pmvx, int pmvy,
                         int *mvx, int *mvy, int *sad, int refine)
{
    aeg_me_block(e->refp[0] + (mby * 16 + geo[1]) * e->stride[0] +
                     mbx * 16 + geo[0],
                 e->stride[0],
                 e->srcp[0] + (mby * 16 + geo[1]) * e->stride[0] +
                     mbx * 16 + geo[0],
                 e->stride[0], geo[2], geo[3],
                 e->p.search_range ? e->p.search_range : 16,
                 refine ? e->p.subme : 0, pmvx, pmvy, NULL, &e->mc,
                 mvx, mvy, sad);
    /* Sub-pixel alignment: the integer search leaves the vector on an integer
     * pel.  Rather than repeating the whole quarter-pel search for a 4x4
     * partition, apply the fractional offset the macroblock's 16x16 search
     * found -- objects move as a whole, so within one macroblock the offset is
     * usually the same -- and keep it when it improves the match. */
    if (!refine && e->p.subme > 0) {
        int fx = e->mb_frac[0], fy = e->mb_frac[1];
        int qx = (*mvx & ~3) + fx, qy = (*mvy & ~3) + fy;
        if (fx || fy) {
            int c_int = part_satd(e, mbx, mby, geo, *mvx, *mvy);
            int c_frac = part_satd(e, mbx, mby, geo, qx, qy);
            if (c_frac < c_int) {
                *mvx = qx;
                *mvy = qy;
                *sad = c_frac;
            }
        }
    }
    (void)type; (void)p;
}

/* Motion search for one partitioned candidate (16x8 or 8x16): every partition
 * gets its own search centred on its own MV predictor, and the MVs decided so
 * far are visible to the following partitions of the same macroblock. */
/* Mark the 4x4 blocks a partition covers as decoded, so that the following
 * partitions can predict from them (and only from them). */
static void avail_publish(uint8_t avail[16], const int geo[4])
{
    int x, y, x0 = geo[0] / 4, y0 = geo[1] / 4;
    int x1 = x0 + geo[2] / 4, y1 = y0 + geo[3] / 4;
    for (y = y0; y < y1; y++)
        for (x = x0; x < x1; x++)
            avail[(y << 2) + x] = 1;
}

static void me_candidate(struct aeg_encoder *e, int mbx, int mby, int type)
{
    int16_t cur[16][2];
    uint8_t avail[16];
    int p, n = part_count(type);
    int slot = e->mb_ncand;

    if (slot >= (int)(sizeof(e->mb_cand) / sizeof(e->mb_cand[0])))
        return;
    memset(cur, 0, sizeof(cur));
    memset(avail, 0, sizeof(avail));
    memset(e->mb_cand[slot].sub_type, 0, sizeof(e->mb_cand[slot].sub_type));
    e->mb_cand[slot].type = type;
    e->mb_cand[slot].nparts = n;
    for (p = 0; p < n; p++) {
        int geo[4], b0, pmvx, pmvy, mvx, mvy, sad;
        part_geom(type, p, geo, &b0);
        predict_mv_at(e, mbx, mby, (const int16_t (*)[2])cur, avail, type, p,
                      geo[0] / 4, geo[1] / 4, geo[2] / 4, &pmvx, &pmvy);
        me_partition(e, mbx, mby, type, p, geo, pmvx, pmvy, &mvx, &mvy, &sad,
                     0);
        cand_publish(&e->mb_cand[slot], geo, mvx, mvy);
        e->mb_cand[slot].pmv[b0][0] = pmvx;
        e->mb_cand[slot].pmv[b0][1] = pmvy;
        me_publish(cur, geo, mvx, mvy);
        avail_publish(avail, geo);
    }
    e->mb_ncand = slot + 1;
}

/* P_8x8 candidate: four 8x8 sub-macroblock partitions, each of which may itself
 * be split into 8x4, 4x8 or 4x4 (sub_mb_type 1..3).  The shape is chosen per
 * sub-macroblock by SAD plus the rate of its own mvd pairs and sub_mb_type; the
 * macroblock as a whole still goes through the usual RD comparison, so a shape
 * only reaches the bitstream if the finer motion field pays for its header. */
static void me_candidate_p8x8(struct aeg_encoder *e, int mbx, int mby)
{
    int16_t cur[16][2];
    uint8_t avail[16];
    int sub, s, slot = e->mb_ncand;
    mbcand *cand;

    if (slot >= (int)(sizeof(e->mb_cand) / sizeof(e->mb_cand[0])))
        return;
    memset(cur, 0, sizeof(cur));
    memset(avail, 0, sizeof(avail));
    cand = &e->mb_cand[slot];
    cand->type = AEG_MB_P8x8;
    cand->nparts = 0;
    memset(cand->sub_type, 0, sizeof(cand->sub_type));
    memset(cand->mv, 0, sizeof(cand->mv));
    memset(cand->pmv, 0, sizeof(cand->pmv));

    for (sub = 0; sub < 4; sub++) {      /* sub-macroblocks in raster order */
        int bx = (sub & 1) * 2, by = (sub >> 1) * 2;
        int best_st = 0, best_cost = 1 << 30, k;
        int best_mv[4][2] = {{0}}, best_pmv[4][2] = {{0}}, best_geo[4][4];

        for (s = 0; s < 4; s++) {
            int16_t trial[16][2];
            uint8_t tavail[16];
            int geo[4][4], b0, n = sub_cnt[s], satd_sum = 0, bits, mv[4][2],
                pmv[4][2];
            int traded = 0;   /* the trial cannot beat the best shape */
            memcpy(trial, cur, sizeof(trial));
            memcpy(tavail, avail, sizeof(tavail));
            bits = aeg_ue_bits((uint32_t)s);
            for (k = 0; k < n; k++) {
                int sad;
                sub_geom(s, bx, by, k, geo[k], &b0);
                predict_mv_at(e, mbx, mby, (const int16_t (*)[2])trial,
                              tavail, AEG_MB_P8x8, 0, geo[k][0] / 4,
                              geo[k][1] / 4, geo[k][2] / 4, &pmv[k][0],
                              &pmv[k][1]);
                me_partition(e, mbx, mby, AEG_MB_P8x8, k, geo[k], pmv[k][0],
                             pmv[k][1], &mv[k][0], &mv[k][1], &sad, 0);
                satd_sum += part_satd(e, mbx, mby, geo[k], mv[k][0], mv[k][1]);
                me_publish(trial, geo[k], mv[k][0], mv[k][1]);
                avail_publish(tavail, geo[k]);
                bits += aeg_mvd_bits(mv[k][0] - pmv[k][0],
                                     mv[k][1] - pmv[k][1]);
                /* Exact early exit: SATD and bits are both non-negative, so the
                 * cost of the complete shape is at least what the partitions
                 * decided so far cost.  Once that reaches the best shape, no
                 * remaining partition can change the decision -- the reason the
                 * 4x4 shape (four searches per sub-macroblock) is not always
                 * paid for in full. */
                if (satd_sum + 8 * bits >= best_cost) {
                    traded = 1;
                    break;
                }
            }
            if (traded)
                continue;
            /* The shape decision compares SATD (not SAD: on a sub-partition
             * scale a small DC error can make a badly aligned 4x4 look good)
             * against the partition's own header cost.  The weight was swept
             * over 4, 8 and 16 on two clips at three rates; 8 is the best.
             * The macroblock-level RD pass still has the last word. */
            if (satd_sum + 8 * bits < best_cost) {
                best_cost = satd_sum + 8 * bits;
                best_st = s;
                memcpy(best_geo, geo, sizeof(geo));
                memcpy(best_mv, mv, sizeof(mv));
                memcpy(best_pmv, pmv, sizeof(pmv));
            }
        }
        /* Commit the winning shape and search its partitions again, this time
         * with the quarter-pel stage.  The predictors are re-derived as we go
         * from the MVs already committed, which is exactly the state the
         * decoder is in when it reads each partition's mvd. */
        cand->sub_type[sub] = (uint8_t)best_st;
        for (k = 0; k < sub_cnt[best_st]; k++) {
            int b0, pmvx, pmvy, sad;
            sub_geom(best_st, bx, by, k, best_geo[k], &b0);
            predict_mv_at(e, mbx, mby, (const int16_t (*)[2])cur, avail,
                          AEG_MB_P8x8, 0, best_geo[k][0] / 4,
                          best_geo[k][1] / 4, best_geo[k][2] / 4, &pmvx, &pmvy);
            /* An 8x8 partition gets the full quarter-pel search; the smaller
             * ones inherit the macroblock's fractional offset (see
             * me_partition), which is nearly free and within a hundredth of a
             * dB of refining each of them separately. */
            me_partition(e, mbx, mby, AEG_MB_P8x8, k, best_geo[k], pmvx, pmvy,
                         &best_mv[k][0], &best_mv[k][1], &sad,
                         best_geo[k][2] >= 8 && best_geo[k][3] >= 8);
            cand_publish(cand, best_geo[k], best_mv[k][0], best_mv[k][1]);
            cand->pmv[b0][0] = pmvx;
            cand->pmv[b0][1] = pmvy;
            me_publish(cur, best_geo[k], best_mv[k][0], best_mv[k][1]);
            avail_publish(avail, best_geo[k]);
            cand->nparts++;
        }
    }
    e->mb_ncand = slot + 1;
}

/* ------------------------------------------------------------------ */
/* Whole-frame parallel motion search                                  */
/* ------------------------------------------------------------------ */
/* The motion search is the stage that dominates the encode time, and it is
 * also the one stage that is completely data parallel: every macroblock can be
 * searched without looking at any other macroblock of the same picture.  Instead
 * of walking a hexagon around one macroblock at a time -- a serial loop of
 * dependent steps that a GPU cannot help with -- the search is expressed as a
 * fixed number of *rounds*.  In each round every macroblock evaluates the eight
 * neighbours of its current best vector, the whole picture as one batch of SAD
 * evaluations, and the round updates all the best vectors at once.  Four rounds
 * (step 8, 4, 2, 1) therefore do the same work as a hex walk, in four batched
 * calls: on a GPU that is four kernel launches per picture instead of ~10 000
 * dependent 16x16 SADs.
 *
 * The result is not the final vector -- it is the *start* the per-macroblock
 * search uses (aeg_me_block() takes it as an extra candidate along with the MV
 * predictor), so the decision logic and the reconstruction stay exactly where
 * they were.
 */
#define ME_ROUND_CAND 9                     /* eight neighbours + the centre */

/* Candidate grid of one round, laid out per macroblock: (dx,dy) relative to the
 * macroblock origin, so that a single sad_batch() call covers the whole picture
 * (the backend applies dy through the reference stride). */
static void me_fill_round(struct aeg_encoder *e, const int16_t *mv, int step,
                          int range)
{
    static const int8_t d[ME_ROUND_CAND][2] = {
        { 0, 0}, {-1, 0}, { 1, 0}, { 0,-1}, { 0, 1},
        {-1,-1}, { 1,-1}, {-1, 1}, { 1, 1}
    };
    int mb, k;
    for (mb = 0; mb < e->mb_count; mb++) {
        int ox = (mb % e->mb_w) * 16, oy = (mb / e->mb_w) * 16;
        int cx = mv[mb * 2 + 0], cy = mv[mb * 2 + 1];
        for (k = 0; k < ME_ROUND_CAND; k++) {
            int nx = cx + d[k][0] * step, ny = cy + d[k][1] * step;
            if (nx < -range || nx > range || ny < -range || ny > range) {
                nx = cx;    /* outside the window: repeat the centre */
                ny = cy;
            }
            e->seed_cand[(mb * ME_ROUND_CAND + k) * 4 + 0] = (int16_t)ox;
            e->seed_cand[(mb * ME_ROUND_CAND + k) * 4 + 1] = (int16_t)oy;
            /* the op takes the two block positions, not a relative vector */
            e->seed_cand[(mb * ME_ROUND_CAND + k) * 4 + 2] = (int16_t)(ox + nx);
            e->seed_cand[(mb * ME_ROUND_CAND + k) * 4 + 3] = (int16_t)(oy + ny);
        }
    }
}

/* One batched SAD round over the whole picture; returns the number of
 * macroblocks whose winner changed shape. */
static int me_run_round(struct aeg_encoder *e, int16_t *best, int32_t *cost)
{
    int mb, k, improved = 0;
    e->d.ops.sad_batch(e->d.ops.priv, e->srcp[0], e->stride[0],
                       e->refp[0], e->stride[0], 16, 16,
                       e->seed_cand, e->mb_count * ME_ROUND_CAND, e->seed_sad);
    for (mb = 0; mb < e->mb_count; mb++) {
        int bestk = 0, bv = cost[mb];
        for (k = 0; k < ME_ROUND_CAND; k++) {
            int32_t v = e->seed_sad[mb * ME_ROUND_CAND + k];
            if (v < bv) { bv = v; bestk = k; }
        }
        if (bv < cost[mb]) {
            cost[mb] = bv;
            best[mb * 2 + 0] = (int16_t)(e->seed_cand[(mb * ME_ROUND_CAND + bestk) * 4 + 2] -
                                         (mb % e->mb_w) * 16);
            best[mb * 2 + 1] = (int16_t)(e->seed_cand[(mb * ME_ROUND_CAND + bestk) * 4 + 3] -
                                         (mb / e->mb_w) * 16);
            improved++;
        }
    }
    return improved;
}

/* One batched SATD round: every macroblock evaluates the nine candidates of
 * e->par_pos[] and keeps the best one.  `mode` selects what the candidates mean
 * (0 = the centre only, used to establish the cost of the current vector). */
static void me_satd_round(struct aeg_encoder *e, int step, int range4,
                          int centre_only)
{
    static const int8_t d[ME_ROUND_CAND][2] = {
        { 0, 0}, {-1, 0}, { 1, 0}, { 0,-1}, { 0, 1},
        {-1,-1}, { 1,-1}, {-1, 1}, { 1, 1}
    };
    int mb;
    for (mb = 0; mb < e->mb_count; mb++) {
        const int ox = (mb % e->mb_w) * 16, oy = (mb / e->mb_w) * 16;
        const int cx = e->par_mv[mb * 2 + 0], cy = e->par_mv[mb * 2 + 1];
        int k;
        for (k = 0; k < ME_ROUND_CAND; k++) {
            int nx = cx + (centre_only ? 0 : d[k][0] * step);
            int ny = cy + (centre_only ? 0 : d[k][1] * step);
            if (nx < -range4 || nx > range4 || ny < -range4 || ny > range4) {
                nx = cx;
                ny = cy;
            }
            e->par_pos[(mb * ME_ROUND_CAND + k) * 4 + 0] = (int16_t)ox;
            e->par_pos[(mb * ME_ROUND_CAND + k) * 4 + 1] = (int16_t)oy;
            e->par_pos[(mb * ME_ROUND_CAND + k) * 4 + 2] = (int16_t)nx;
            e->par_pos[(mb * ME_ROUND_CAND + k) * 4 + 3] = (int16_t)ny;
        }
    }
    e->d.ops.satd_batch(e->d.ops.priv, e->srcp[0], e->stride[0], 16, 16,
                        e->par_pos, e->mb_count * ME_ROUND_CAND, e->seed_sad);
    for (mb = 0; mb < e->mb_count; mb++) {
        int bestk = -1, bv = e->par_cost[mb], k;
        for (k = 0; k < ME_ROUND_CAND; k++) {
            int32_t v = e->seed_sad[mb * ME_ROUND_CAND + k];
            if (v < bv) { bv = v; bestk = k; }
        }
        if (bestk >= 0) {
            e->par_cost[mb] = bv;
            e->par_mv[mb * 2 + 0] = e->par_pos[(mb * ME_ROUND_CAND + bestk) * 4 + 2];
            e->par_mv[mb * 2 + 1] = e->par_pos[(mb * ME_ROUND_CAND + bestk) * 4 + 3];
        }
    }
}

/* Fractional-pel refinement of the whole picture, as two sets of batched rounds
 * (half-pel steps, then quarter-pel steps) instead of a serial walk per
 * macroblock.  This is the second half of the motion search and, like the
 * integer rounds, it is one launch per round on a GPU. */
static void me_par_subpel(struct aeg_encoder *e, int range4)
{
    int pass, mb;

    for (mb = 0; mb < e->mb_count; mb++) {
        e->par_mv[mb * 2 + 0] = (int16_t)(e->seed_mv[mb * 2 + 0] * 4);
        e->par_mv[mb * 2 + 1] = (int16_t)(e->seed_mv[mb * 2 + 1] * 4);
        e->par_cost[mb] = 1 << 30;
    }
    me_satd_round(e, 1, range4, 1);      /* cost of the integer position */

    for (pass = 0; pass < 2; pass++) {
        const int step = pass ? 1 : 2;
        int iters = 0, improved = 1;
        while (improved && iters++ < AEG_SUBPEL_ITERS) {
            int32_t before = 0;
            for (mb = 0; mb < e->mb_count; mb++)
                before += e->par_cost[mb];
            me_satd_round(e, step, range4, 0);
            {
                int32_t after = 0;
                for (mb = 0; mb < e->mb_count; mb++)
                    after += e->par_cost[mb];
                improved = after < before;
            }
        }
    }
}

/* Search every macroblock of the current P picture in parallel and remember the
 * resulting integer vectors in e->seed_mv[].  Runs on the GPU when the
 * dispatcher puts motion estimation there and on the CPU otherwise -- the same
 * batch interface either way (src/backend.c), so the vectors are identical. */
static void me_seed_frame(struct aeg_encoder *e)
{
    int range = e->p.search_range ? e->p.search_range : 16;
    int mb, n = e->mb_count;
    int step;
    int64_t t0;

    e->par_valid = 0;
    e->par_full = 0;
    if (!e->has_ref || !e->cur_is_p || e->p.subme <= 0) {
        memset(e->seed_mv, 0, (size_t)n * 2 * sizeof(int16_t));
        return;
    }
    /* On a GPU the rounds are a handful of launches, so they shorten the hex
     * walks for free.  On the CPU they cost about what they save (measured:
     * neutral at --subme 1, ~8% of the motion stage better integer-only), so
     * they follow the dispatcher's vote for the motion-estimation stage; a GPU
     * that is faster there for motion estimation gets them, a CPU picture does
     * not.  AEG_SEED=1/0 forces the decision either way, which is what the
     * seed_search conformance case does. */
    {
        const char *ev = getenv("AEG_SEED");
        int on = ev ? atoi(ev) != 0 : e->d.use_gpu[AEG_STAGE_MOTION_EST];
        if (!on) {
            memset(e->seed_mv, 0, (size_t)n * 2 * sizeof(int16_t));
            return;
        }
        /* The fractional half of the search is the same idea -- all
         * macroblocks refine in quarter-pel rounds, one launch per round -- and
         * replaces the per-macroblock walk outright.  Same gate (a GPU that
         * wins the motion stage runs it), AEG_PARFULL=1/0 for the tests. */
        ev = getenv("AEG_PARFULL");
        e->par_full = ev ? atoi(ev) != 0 : e->d.use_gpu[AEG_STAGE_MOTION_EST];
    }
    /* the vectors of the previous picture are the temporal predictor of this
     * one; e->seed_mv still holds them, so copy before overwriting */
    memcpy(e->prev_seed, e->seed_mv, (size_t)n * 2 * sizeof(int16_t));
    t0 = aeg_now_ns();

    /* round 0: the predictor candidates, zero and the previous picture's vector
     * at the same macroblock (rounded to integer pel) */
    for (mb = 0; mb < n; mb++) {
        int ox = (mb % e->mb_w) * 16, oy = (mb / e->mb_w) * 16;
        /* the previous picture's rounded vector, in whole samples */
        int tx = e->prev_seed[mb * 2 + 0];
        int ty = e->prev_seed[mb * 2 + 1];
        int k;
        if (tx < -range || tx > range) tx = 0;
        if (ty < -range || ty > range) ty = 0;
        for (k = 0; k < ME_ROUND_CAND; k++) {
            int dx = (k == 1) ? tx : 0, dy = (k == 1) ? ty : 0;
            e->seed_cand[(mb * ME_ROUND_CAND + k) * 4 + 0] = (int16_t)ox;
            e->seed_cand[(mb * ME_ROUND_CAND + k) * 4 + 1] = (int16_t)oy;
            e->seed_cand[(mb * ME_ROUND_CAND + k) * 4 + 2] = (int16_t)(ox + dx);
            e->seed_cand[(mb * ME_ROUND_CAND + k) * 4 + 3] = (int16_t)(oy + dy);
        }
        e->seed_mv[mb * 2 + 0] = 0;
        e->seed_mv[mb * 2 + 1] = 0;
        e->seed_cost[mb] = 1 << 30;
    }
    me_run_round(e, e->seed_mv, e->seed_cost);

    for (step = range / 2; step >= 1; step /= 2) {
        me_fill_round(e, e->seed_mv, step, range);
        if (!me_run_round(e, e->seed_mv, e->seed_cost))
            break;      /* nothing improved: the walk has converged */
    }

    /* The fractional half of the search, as rounds too -- but only where a
     * satd_batch() backend can work from the registered reference: the CPU
     * operation and both GPU backends provide it, and the vectors it produces
     * are the same samples the serial walk sees. */
    if (e->par_full && e->d.ops.satd_batch && e->d.ops.ref_upload && e->mc.ready) {
        e->d.ops.ref_upload(e->d.ops.priv, e->mc.base, e->mc.stride, e->w, e->h);
        me_par_subpel(e, range * 4);
        e->par_valid = 1;
    }

    t0 = aeg_now_ns() - t0;
    e->st.stage_ns[AEG_STAGE_MOTION_EST] += t0;
    aeg_dispatch_record(&e->d, AEG_STAGE_MOTION_EST,
                        e->d.ops.backend != AEG_BACKEND_CPU, t0);
}

/* Fills e->mb_modes[] with the legal Intra_16x16 modes ordered by SAD, runs
 * the motion search and remembers its result.  Nothing is coded here: code_mb()
 * either uses this SAD ranking directly (preset 0) or refines it with a full
 * rate-distortion search. */
static void analyse_mb(struct aeg_encoder *e, int mbx, int mby, mbenc *c)
{
    int mode, n = 0, sad_of[4];

    for (mode = 0; mode < 4; mode++)
        sad_of[mode] = luma_mode_ok(e, mode, mbx, mby)
                           ? intra16_sad(e, mbx, mby, mode) : (1 << 30);
    /* insertion sort: four entries, cheapest first */
    for (mode = 0; mode < 4; mode++) {
        int i;
        if (sad_of[mode] == (1 << 30))
            continue;
        i = n++;
        while (i > 0 && sad_of[e->mb_modes[i - 1]] > sad_of[mode]) {
            e->mb_modes[i] = e->mb_modes[i - 1];
            i--;
        }
        e->mb_modes[i] = mode;
    }
    e->mb_nmodes = n;
    e->mb_sad = sad_of[e->mb_modes[0]];
    c->i16_mode = e->mb_modes[0];
    c->chroma_mode = chroma_mode_by_sad(e, mbx, mby);

    e->mb_inter_ok = 0;
    e->mb_ncand = 0;
    e->mb_mv[0] = e->mb_mv[1] = 0;
    e->mb_pmv[0] = e->mb_pmv[1] = 0;
    e->mb_frac[0] = e->mb_frac[1] = 0;
    /* The motion search needs a reference picture *and* a P slice: an IDR
     * picture is always an I slice, so inter candidates there could never be
     * coded (writing them as an I macroblock desynchronises the decoder). */
    if (e->p.partitions != 0 && c->qp >= 0 && e->has_ref && e->cur_is_p) {
        int mvx, mvy, sad;
        aeg_mv_predict(e->mbs, e->mb_w, e->mb_w, mbx, mby, e->top_row,
                       &e->mb_pmv[0], &e->mb_pmv[1]);
        /* The block to match is the *source* macroblock.  recp still holds the
         * previous picture at this position (nothing has been reconstructed
         * here yet), so searching against it would always favour mv = 0. */
        if (e->par_valid && e->par_full) {
            /* the whole-frame rounds already produced this macroblock's
             * quarter-pel vector (and its SATD) */
            const int idx = mby * e->mb_w + mbx;
            mvx = e->par_mv[idx * 2 + 0];
            mvy = e->par_mv[idx * 2 + 1];
            sad = e->par_cost[idx];
        } else {
            int seed[2];
            seed[0] = e->seed_mv[(mby * e->mb_w + mbx) * 2 + 0];
            seed[1] = e->seed_mv[(mby * e->mb_w + mbx) * 2 + 1];
            aeg_me_block(e->refp[0] + mby * 16 * e->stride[0] + mbx * 16,
                         e->stride[0],
                         e->srcp[0] + mby * 16 * e->stride[0] + mbx * 16,
                         e->stride[0], 16, 16,
                         e->p.search_range ? e->p.search_range : 16,
                         e->p.subme, e->mb_pmv[0], e->mb_pmv[1], seed,
                         &e->mc, &mvx, &mvy, &sad);
        }
        e->mb_mv[0] = mvx;
        e->mb_mv[1] = mvy;
        e->mb_frac[0] = mvx & 3;
        e->mb_frac[1] = mvy & 3;
        e->mb_inter_ok = 1;
        e->mb_inter_sad = sad;

        {
            int b;
            e->mb_cand[0].type = AEG_MB_P16x16;
            e->mb_cand[0].nparts = 1;
            for (b = 0; b < 16; b++) {
                e->mb_cand[0].mv[b][0] = mvx;
                e->mb_cand[0].mv[b][1] = mvy;
            }
            e->mb_cand[0].pmv[0][0] = e->mb_pmv[0];
            e->mb_cand[0].pmv[0][1] = e->mb_pmv[1];
        }
        e->mb_ncand = 1;
        /* Partitioned candidates: 16x8 and 8x16, bit 1 and 2 of the mask.
         * They only pay off with the RD decision, so ultrafast stays 16x16. */
        if (e->p.preset >= 1) {
            if (e->p.partitions & 2)
                me_candidate(e, mbx, mby, AEG_MB_P16x8);
            if (e->p.partitions & 4)
                me_candidate(e, mbx, mby, AEG_MB_P8x16);
            if (e->p.partitions & 8)
                me_candidate_p8x8(e, mbx, mby);
        }
    }

    /* Preset 0 keeps the plain SAD decision; it leaves both predictions in
     * place so that code_mb() can use them directly. */
    if (e->p.preset <= 0) {
        if (e->mb_inter_ok &&
            e->mb_inter_sad * 8 + aeg_mvd_bits((e->mb_mv[0] - e->mb_pmv[0]) >> 2,
                                               (e->mb_mv[1] - e->mb_pmv[1]) >> 2)
                < e->mb_sad * 8) {
            int i;
            c->type = AEG_MB_P16x16;
            c->nparts = 1;
            c->pmv[0][0] = (int16_t)e->mb_pmv[0];
            c->pmv[0][1] = (int16_t)e->mb_pmv[1];
            for (i = 0; i < 16; i++) {
                c->mv[i][0] = (int16_t)e->mb_mv[0];
                c->mv[i][1] = (int16_t)e->mb_mv[1];
            }
            inter_predict(e, mbx, mby, e->mb_mv[0], e->mb_mv[1]);
        } else {
            c->type = AEG_MB_I16x16;
            intra16_sad(e, mbx, mby, c->i16_mode);   /* leave pred_l */
        }
    }
}

/* CodedBlockPatternChroma as coded in the bitstream: 0 none, 1 DC only,
 * 2 AC (DC is then implied). */
static int chroma_cbp(const mbenc *c)
{
    if (c->cbp & 0x20)
        return 2;
    return (c->cbp & 0x10) ? 1 : 0;
}

/* CodedBlockPatternLuma is binary in Intra_16x16: 0 means "no AC
 * coefficients at all", 15 means "all sixteen AC blocks are coded". */
static int luma_cbp_on(const mbenc *c)
{
    return (c->cbp & 0x0f) ? 15 : 0;
}

/* Bits the macroblock header costs: mb_type (and, for the intra types, the
 * chroma prediction mode and the coded block pattern that mb_type carries). */
static int mb_type_bits(const struct aeg_encoder *e, const mbenc *c, int is_p)
{
    /* An I_NxN macroblock carries transform_size_8x8_flag right after mb_type
     * (7.3.5.2) whenever the picture parameter set enables the 8x8 transform,
     * whether or not the macroblock actually uses it. */
    if (c->type == AEG_MB_I4x4 || c->type == AEG_MB_I8x8)
        return aeg_ue_bits(is_p ? 5u : 0u) + (e->p.transform_8x8 ? 1 : 0);
    if (!is_p)
        return aeg_ue_bits((uint32_t)(1 + c->i16_mode + 4 * chroma_cbp(c) +
                                      12 * (luma_cbp_on(c) ? 1 : 0)));
    if (is_inter_type(c->type))
        return aeg_ue_bits((uint32_t)inter_mb_type_code(c->type));
    return aeg_ue_bits((uint32_t)(6 + c->i16_mode + 4 * chroma_cbp(c) +
                                  12 * (luma_cbp_on(c) ? 1 : 0)));
}

/* Header bits of the inter geometry: mb_type plus one mvd pair per partition
 * (H.264 codes mvd in the same 1/4-pel units as the MV itself). */
static int inter_hdr_bits(const struct aeg_encoder *e, const mbenc *c)
{
    int p, i, bits = mb_type_bits(e, c, 1);
    /* P_8x8 carries a sub_mb_type per 8x8 partition, before any mvd. */
    if (c->type == AEG_MB_P8x8)
        for (i = 0; i < 4; i++)
            bits += aeg_ue_bits((uint32_t)c->sub_type[i]);
    for (p = 0; p < c->nparts; p++) {
        int geo[4], b0;
        mb_part_geom(c, p, geo, &b0);
        bits += aeg_mvd_bits(c->mv[b0][0] - c->pmv[b0][0],
                             c->mv[b0][1] - c->pmv[b0][1]);
    }
    /* transform_size_8x8_flag: for inter macroblocks it comes after the coded
     * block pattern and only when the luma pattern is non-zero. */
    if (t8_allowed(e, c) && (c->cbp & 0x0f) && c->t8)
        bits += 1;
    return bits;
}

/* ------------------------------------------------------------------ */
/* Rate-distortion helpers                                             */
/* ------------------------------------------------------------------ */
/* The distortion is a real SSD (the samples the decoder will reproduce), the
 * rate is the payload the entropy coder will actually spend, so the decision
 * minimises J = SSD + lambda * bits.  lambda = 0.85 * 2^((QP-12)/3) is the
 * usual H.264 choice for an SSD-based cost. */

static void snap_luma(const struct aeg_encoder *e, int mbx, int mby, uint8_t *buf)
{
    const uint8_t *p = e->recp[0] + mby * 16 * e->stride[0] + mbx * 16;
    int y;
    for (y = 0; y < 16; y++)
        memcpy(buf + y * 16, p + y * e->stride[0], 16);
}

static void restore_luma(struct aeg_encoder *e, int mbx, int mby, const uint8_t *buf)
{
    uint8_t *p = e->recp[0] + mby * 16 * e->stride[0] + mbx * 16;
    int y;
    for (y = 0; y < 16; y++)
        memcpy(p + y * e->stride[0], buf + y * 16, 16);
}

static void snap_chroma(const struct aeg_encoder *e, int mbx, int mby, uint8_t *buf)
{
    int pl, y;
    for (pl = 0; pl < 2; pl++) {
        const uint8_t *p = e->recp[1 + pl] + mby * 8 * e->stride[1] + mbx * 8;
        uint8_t *b = buf + pl * 64;
        for (y = 0; y < 8; y++)
            memcpy(b + y * 8, p + y * e->stride[1], 8);
    }
}

static void restore_chroma(struct aeg_encoder *e, int mbx, int mby, const uint8_t *buf)
{
    int pl, y;
    for (pl = 0; pl < 2; pl++) {
        uint8_t *p = e->recp[1 + pl] + mby * 8 * e->stride[1] + mbx * 8;
        const uint8_t *b = buf + pl * 64;
        for (y = 0; y < 8; y++)
            memcpy(p + y * e->stride[1], b + y * 8, 8);
    }
}

/* ------------------------------------------------------------------ */
/* Intra_4x4                                                           */
/* ------------------------------------------------------------------ */
/* The prediction modes a block may use given the available neighbours.  A mode
 * that reads a missing side must not be signalled: the reference decoder
 * remaps the modes of the macroblocks on the picture edges (DC becomes a
 * one-sided DC and the directional modes are rejected), so a mode that needs
 * the missing side would be reconstructed with a different predictor than the
 * encoder assumed.  DC itself is always legal -- the predictor resolves it
 * against whichever sides are available. */
static int intra4_mode_list(int top_avail, int left_avail, int *modes)
{
    int n = 0;
    if (top_avail)
        modes[n++] = 0;                 /* vertical            */
    if (left_avail)
        modes[n++] = 1;                 /* horizontal          */
    modes[n++] = 2;                     /* DC                  */
    if (top_avail) {
        modes[n++] = 3;                 /* diagonal down-left  */
        modes[n++] = 7;                 /* vertical-left       */
    }
    if (top_avail && left_avail) {
        modes[n++] = 4;                 /* diagonal down-right */
        modes[n++] = 5;                 /* vertical-right      */
        modes[n++] = 6;                 /* horizontal-down     */
    }
    if (left_avail)
        modes[n++] = 8;                 /* horizontal-up       */
    return n;
}

/* H.264 codes the sixteen Intra_4x4 blocks in the order of Figure 6-8, which
 * is not raster order: each 8x8 group is coded before the next one.  The index
 * below converts a raster position inside the macroblock to that order (and
 * back, further down).  The order guarantees that both the prediction samples
 * and the mode neighbours of a block are already coded when its turn comes. */
static int intra4_idx(int bx, int by)
{
    return ((by & 2) << 2) | ((bx & 2) << 1) | ((by & 1) << 1) | (bx & 1);
}

/* H.264 8.3.1.1: the Intra_4x4 mode predictor is the smaller of the modes of
 * the left and the above neighbouring blocks, where a neighbour that is not
 * Intra_4x4-coded counts as DC and one outside the picture makes the whole
 * prediction DC.  `cur` holds the modes decided so far, in coding order. */
static int intra4_pred_mode(const struct aeg_encoder *e, int mbx, int mby,
                            const uint8_t *cur, int bx, int by)
{
    int a, b;
    if (bx > 0) {
        a = cur[intra4_idx(bx - 1, by)];
    } else if (mbx > 0) {
        const aeg_mb *m = &e->mbs[mby * e->mb_w + mbx - 1];
        if (m->type == AEG_MB_I4x4)
            a = m->intra4[intra4_idx(3, by)];
        else if (m->type == AEG_MB_I8x8)
            /* An Intra_8x8 macroblock has no 4x4 modes, but a decoder keeps
             * its four prediction modes in the 4x4 mode cache, one per 2x2
             * group of blocks.  The neighbour of this macroblock's block row
             * is therefore the 8x8 group of the same block row in the group
             * column on the right, not the DC fallback. */
            a = m->intra8[(by >> 1) * 2 + 1];
        else
            a = 2;
    } else {
        a = -1;
    }
    if (by > 0) {
        b = cur[intra4_idx(bx, by - 1)];
    } else if (mby > e->top_row) {
        const aeg_mb *m = &e->mbs[(mby - 1) * e->mb_w + mbx];
        if (m->type == AEG_MB_I4x4)
            b = m->intra4[intra4_idx(bx, 3)];
        else if (m->type == AEG_MB_I8x8)
            b = m->intra8[2 + (bx >> 1)];
        else
            b = 2;
    } else {
        b = -1;
    }
    if (a < 0 || b < 0)
        return 2;
    return a < b ? a : b;
}

/* Intra_4x4 luma: one prediction mode per 4x4 block, decided and coded in the
 * block order of Figure 6-8 (see intra4_idx), because every block predicts
 * from the reconstruction of the blocks above and to the left of it --
 * including the ones of the same macroblock. */
static void code_luma_i4x4(struct aeg_encoder *e, int mbx, int mby, mbenc *t,
                           double lambda)
{
    const int s = e->stride[0];
    const uint8_t *src = e->srcp[0] + mby * 16 * s + mbx * 16;
    uint8_t *dst = e->recp[0] + mby * 16 * s + mbx * 16;
    int bits_scale = (int)(lambda * 0.5 + 0.5);
    int blk, k, x, y, g, mode_bits = 0;

    t->type = AEG_MB_I4x4;
    t->cbp &= 0x30;                     /* keep the chroma pattern */
    memset(t->intra4, 0, sizeof(t->intra4));

    for (blk = 0; blk < 16; blk++) {
        int rx = ((blk >> 2) & 1) * 2 + (blk & 1);
        int ry = ((blk >> 3) & 1) * 2 + ((blk >> 1) & 1);
        int bx = rx * 4, by = ry * 4;
        int rb = ry * 4 + rx;           /* raster index of the block */
        int top_avail = mby > e->top_row || by > 0;
        int left_avail = mbx > 0 || bx > 0;
        /* The four samples p[4..7][-1] continue to the right of the block.
         * Inside the macroblock they belong to the block one column right in
         * the block row above, and that block is only decoded in time for the
         * Figure 6-8 order when it comes earlier -- false for the blocks at
         * (rx, ry) = (1, odd).  Blocks in the last block column need the
         * macroblock to the right (not decoded yet), while a block in the
         * first block row takes them from the macroblock above, or from the
         * one above-right when bx == 12. */
        int tr_avail;
        if (by)
            tr_avail = bx < 12 && !((rx & 1) && (ry & 1));
        else
            tr_avail = mby > e->top_row && (bx < 12 || mbx + 1 < e->mb_w);
        uint8_t top[8], left[4], pred[16];
        const uint8_t *sb = src + by * s + bx;
        uint8_t *db = dst + by * s + bx;
        int tl, nmodes, modes[9], best = 2, best_cost = 1 << 30, pmode, m;

        if (top_avail) {
            const uint8_t *tr = by ? dst + (by - 1) * s + bx : dst - s + bx;
            for (k = 0; k < 4; k++)
                top[k] = tr[k];
            for (k = 4; k < 8; k++)
                top[k] = tr_avail ? tr[k] : top[3];
        } else {
            for (k = 0; k < 8; k++)
                top[k] = 128;
        }
        if (left_avail) {
            for (k = 0; k < 4; k++)
                left[k] = db[k * s - 1];
        } else {
            for (k = 0; k < 4; k++)
                left[k] = 128;
        }
        tl = (top_avail && left_avail) ? db[-s - 1] : 128;

        pmode = intra4_pred_mode(e, mbx, mby, t->intra4, rx, ry);
        nmodes = intra4_mode_list(top_avail, left_avail, modes);
        for (k = 0; k < nmodes; k++) {
            int cost;
            m = modes[k];
            aeg_intra_predict_4x4(pred, 4, top, left, tl, m, top_avail,
                                  left_avail, top_avail && left_avail);
            if (e->p.preset >= 3) {
                /* True rate-distortion choice: quantise the residual, add it
                 * back, and weigh the exact SSE against the exact bits the
                 * block will cost.  The SATD approximation below is much
                 * cheaper but it cannot see that a mode which predicts
                 * better can still cost more after quantisation. */
                int16_t res[16], coef[16], lev[16];
                uint8_t rb_rec[16];
                const uint8_t *pr = pred;
                int64_t sse = 0;
                int nzz = 0, i;
                for (y = 0; y < 4; y++)
                    for (x = 0; x < 4; x++)
                        res[y * 4 + x] = (int16_t)((int)sb[y * s + x] -
                                                   (int)pr[y * 4 + x]);
                aeg_transform_4x4(res, 4, coef);
                aeg_quant_4x4(coef, lev, t->qp, 1);
                for (i = 0; i < 16; i++)
                    nzz += lev[i] != 0;
                for (i = 0; i < 16; i++)
                    rb_rec[i] = pr[i];
                aeg_idct_add_block(rb_rec, 4, lev, t->qp, pr, 4, NULL);
                for (y = 0; y < 4; y++)
                    for (x = 0; x < 4; x++) {
                        int d = (int)rb_rec[y * 4 + x] - (int)sb[y * s + x];
                        sse += (int64_t)d * d;
                    }
                cost = (int)(sse + lambda * (double)(nzz ?
                           aeg_cavlc_block_cost(lev, luma_nc(e, mbx, mby, t, rb),
                                                0) : 0));
            } else {
                cost = 2 * aeg_satd_4x4(pred, 4, sb, s);
            }
            /* The mode costs one bit when it matches the predictor and four
             * otherwise, so weigh it into the comparison. */
            cost += bits_scale * (m == pmode ? 1 : 4);
            if (cost < best_cost) {
                best_cost = cost;
                best = m;
            }
        }
        t->intra4[blk] = (uint8_t)best;
        mode_bits += (best == pmode) ? 1 : 4;

        aeg_intra_predict_4x4(pred, 4, top, left, tl, best, top_avail,
                              left_avail, top_avail && left_avail);
        {
            int16_t res[16], coef[16];
            int nz = 0;
            for (y = 0; y < 4; y++)
                for (x = 0; x < 4; x++)
                    res[y * 4 + x] = (int16_t)((int)sb[y * s + x] -
                                               (int)pred[y * 4 + x]);
            aeg_transform_4x4(res, 4, coef);
            aeg_quant_4x4(coef, t->level[rb], t->qp, 1);
            for (k = 0; k < 16; k++)
                nz += t->level[rb][k] != 0;
            t->nz[rb] = (int8_t)nz;
            if (nz)
                t->cbp |= 1 << ((rb >> 3) * 2 + ((rb & 3) >> 1));
        }
        aeg_idct_add_block(db, s, t->level[rb], t->qp, pred, 4, NULL);
        for (y = 0; y < 4; y++)
            for (x = 0; x < 4; x++) {
                int d = (int)db[y * s + x] - (int)sb[y * s + x];
                t->sse += (int64_t)d * d;
            }
    }

    t->bits += mode_bits;
    /* The blocks of every coded 8x8 group are coded, in the order the group
     * table lists them; the context of a block comes from its neighbours. */
    for (g = 0; g < 4; g++) {
        if (!(t->cbp & (1 << g)))
            continue;
        for (k = 0; k < 4; k++) {
            int b = group_blk[g][k];
            if (t->nz[b])
                t->bits += aeg_cavlc_block_cost(t->level[b],
                                                luma_nc(e, mbx, mby, t, b), 0);
        }
    }
}

static int64_t rd_luma_i4x4(struct aeg_encoder *e, int mbx, int mby, mbenc *t,
                            int qp, double lambda)
{
    reset_mb_enc(t, qp);
    code_luma_i4x4(e, mbx, mby, t, lambda);
    t->cost = t->sse + (int64_t)(lambda * t->bits + 0.5);
    return t->cost;
}

/* H.264 8.3.1.2: the Intra_8x8 mode predictor is the smaller of the modes of
 * the 8x8 blocks to the left and above, where a neighbour that is not
 * Intra_8x8-coded counts as DC and one outside the picture makes the whole
 * prediction DC.  `cur` holds the modes decided so far, in raster order. */
static int intra8_pred_mode(const struct aeg_encoder *e, int mbx, int mby,
                            const uint8_t *cur, int gx, int gy)
{
    int a, b;
    if (gx > 0) {
        a = cur[gy * 2 + gx - 1];
    } else if (mbx > 0) {
        const aeg_mb *m = &e->mbs[mby * e->mb_w + mbx - 1];
        if (m->type == AEG_MB_I8x8)
            a = m->intra8[gy * 2 + 1];
        else if (m->type == AEG_MB_I4x4)
            /* A decoder keeps one prediction mode per 4x4 block, so the
             * neighbour of an 8x8 group is the 4x4 mode of the block at the
             * same block row in the column to the left (see the note above
             * intra4_pred_mode). */
            a = m->intra4[intra4_idx(3, gy * 2)];
        else
            a = 2;
    } else {
        a = -1;
    }
    if (gy > 0) {
        b = cur[(gy - 1) * 2 + gx];
    } else if (mby > e->top_row) {
        const aeg_mb *m = &e->mbs[(mby - 1) * e->mb_w + mbx];
        if (m->type == AEG_MB_I8x8)
            b = m->intra8[2 + gx];
        else if (m->type == AEG_MB_I4x4)
            b = m->intra4[intra4_idx(gx * 2, 3)];
        else
            b = 2;
    } else {
        b = -1;
    }
    if (a < 0 || b < 0)
        return 2;
    return a < b ? a : b;
}

/* Intra_8x8 luma: one prediction mode per 8x8 block, decided and coded in
 * raster block order, because every block predicts from the reconstruction of
 * the block to its left and the one above it -- including the ones of the same
 * macroblock.  The mode is chosen by the 8x8 SATD of the prediction plus the
 * bits the mode itself costs; the residual is then coded with the 8x8
 * transform, which is what makes this an Intra_8x8 macroblock. */
static void code_luma_i8x8(struct aeg_encoder *e, int mbx, int mby, mbenc *t,
                           double lambda)
{
    const int s = e->stride[0];
    const uint8_t *src = e->srcp[0] + mby * 16 * s + mbx * 16;
    uint8_t *dst = e->recp[0] + mby * 16 * s + mbx * 16;
    int bits_scale = (int)(lambda * 0.5 + 0.5);
    int g, k, mode_bits = 0;

    t->type = AEG_MB_I8x8;
    t->t8 = 1;
    t->cbp &= 0x30;                     /* keep the chroma pattern */
    memset(t->intra8, 0, sizeof(t->intra8));

    for (g = 0; g < 4; g++) {
        const int gx = g & 1, gy = g >> 1;
        const int bx = gx * 8, by = gy * 8;
        const int top_avail = mby > e->top_row || by > 0;
        const int left_avail = mbx > 0 || bx > 0;
        uint8_t top[16], left[8];
        int tl, tr_avail;
        int nmodes, modes[9], best = 2, best_cost = 1 << 30, pmode, m;
        uint8_t pred[64];

        /* The sixteen top samples.  Inside the macroblock the row above the
         * block has already been reconstructed, so its samples are used as
         * they are; at the top of the macroblock they come from the
         * macroblock above, and the eight on the right only when the
         * macroblock above-right has been decoded -- otherwise p[15][-1] is
         * replicated into them (8.3.3.4.2). */
        if (top_avail) {
            const uint8_t *tr = by ? dst + (by - 1) * s + bx : dst - s + bx;
            /* The eight samples on the right come from the 8x8 block above
             * and to the right (8.3.3.4.2).  For the two blocks in the left
             * column that block has already been reconstructed -- it is the
             * top-right block of this macroblock for the lower one -- so the
             * samples are used as they are.  For the top-right block it is in
             * the macroblock above, which has been decoded when there is one,
             * and for the bottom-right block it is in the macroblock to the
             * right, which comes later in raster order and is therefore not
             * available. */
            tr_avail = (bx == 0) || (by == 0 && mbx + 1 < e->mb_w);
            for (k = 0; k < 8; k++)
                top[k] = tr[k];
            for (k = 8; k < 16; k++)
                top[k] = tr_avail ? tr[k] : top[7];
        } else {
            for (k = 0; k < 16; k++)
                top[k] = 128;
        }
        if (left_avail) {
            for (k = 0; k < 8; k++)
                left[k] = dst[(by + k) * s + bx - 1];
        } else {
            for (k = 0; k < 8; k++)
                left[k] = 128;
        }
        tl = (top_avail && left_avail) ? (by ? dst[(by - 1) * s + bx - 1]
                                             : dst[-s + bx - 1]) : 128;

        pmode = intra8_pred_mode(e, mbx, mby, t->intra8, gx, gy);
        nmodes = intra4_mode_list(top_avail, left_avail, modes);
        for (k = 0; k < nmodes; k++) {
            int cost;
            m = modes[k];
            aeg_intra_predict_8x8(pred, 8, top, left, tl, m, top_avail,
                                  left_avail, top_avail && left_avail);
            cost = 2 * aeg_satd_8x8(pred, 8, src + by * s + bx, s);
            /* The mode costs one bit when it matches the predictor and four
             * otherwise, as for Intra_4x4. */
            cost += bits_scale * (m == pmode ? 1 : 4);
            if (cost < best_cost) {
                best_cost = cost;
                best = m;
            }
        }
        t->intra8[g] = (uint8_t)best;
        mode_bits += (best == pmode) ? 1 : 4;

        aeg_intra_predict_8x8(pred, 8, top, left, tl, best, top_avail,
                              left_avail, top_avail && left_avail);
        code_luma8_block(e, mbx, mby, t, g, pred, 8, 1);
    }

    t->bits += mode_bits;
    luma8_costs(e, mbx, mby, t);
}

static int64_t rd_luma_i8x8(struct aeg_encoder *e, int mbx, int mby, mbenc *t,
                            int qp, double lambda)
{
    reset_mb_enc(t, qp);
    code_luma_i8x8(e, mbx, mby, t, lambda);
    t->cost = t->sse + (int64_t)(lambda * t->bits + 0.5);
    return t->cost;
}

/* Code the luma part of one candidate into `t` and return its RD cost.  The
 * reconstruction is left in the picture buffer, so the caller snapshots it.
 * `cand` selects one of the inter candidates gathered by analyse_mb(); it is
 * ignored for the intra types. */
static int64_t rd_luma(struct aeg_encoder *e, int mbx, int mby, mbenc *t, int qp,
                       int type, int i16_mode, double lambda, int cand)
{
    reset_mb_enc(t, qp);
    t->type = type;
    t->i16_mode = i16_mode;
    if (is_inter_type(type)) {
        int b;
        memcpy(t->sub_type, e->mb_cand[cand].sub_type, sizeof(t->sub_type));
        t->nparts = (uint8_t)e->mb_cand[cand].nparts;
        for (b = 0; b < 16; b++) {
            t->mv[b][0] = (int16_t)e->mb_cand[cand].mv[b][0];
            t->mv[b][1] = (int16_t)e->mb_cand[cand].mv[b][1];
            t->pmv[b][0] = (int16_t)e->mb_cand[cand].pmv[b][0];
            t->pmv[b][1] = (int16_t)e->mb_cand[cand].pmv[b][1];
        }
        inter_predict_typed(e, mbx, mby, t);
        if (use_t8(e, mbx, mby, t))
            code_luma_inter8(e, mbx, mby, t);
        else
            code_luma_inter(e, mbx, mby, t);
        t->bits += inter_hdr_bits(e, t);
    } else {
        t->nparts = 1;
        intra16_sad(e, mbx, mby, i16_mode);   /* leaves the predictor in pred_l */
        code_luma_i16(e, mbx, mby, t);
    }
    t->cost = t->sse + (int64_t)(lambda * t->bits + 0.5);
    return t->cost;
}

/* Fill `t` with one of the inter candidates the motion search produced: the
 * shape, the sub-macroblock types and, per 4x4 block, the vector and its
 * predictor.  This is the part of an inter candidate that decides the
 * bitstream, so it is copied exactly as rd_luma() does it. */
static void cand_enc(const struct aeg_encoder *e, int cand, int qp, mbenc *t)
{
    int b;
    reset_mb_enc(t, qp);
    t->type = e->mb_cand[cand].type;
    memcpy(t->sub_type, e->mb_cand[cand].sub_type, sizeof(t->sub_type));
    t->nparts = (uint8_t)e->mb_cand[cand].nparts;
    for (b = 0; b < 16; b++) {
        t->mv[b][0] = (int16_t)e->mb_cand[cand].mv[b][0];
        t->mv[b][1] = (int16_t)e->mb_cand[cand].mv[b][1];
        t->pmv[b][0] = (int16_t)e->mb_cand[cand].pmv[b][0];
        t->pmv[b][1] = (int16_t)e->mb_cand[cand].pmv[b][1];
    }
}

/* The rate-distortion comparison of every inter candidate of one macroblock,
 * with the coefficient work done as one batch.
 *
 * This is the provisional form of "produce every candidate, then decide": the
 * candidates of a macroblock are independent (their prediction comes from the
 * reference picture), so they are predicted and their residuals gathered
 * first, then one tq_batch() and one iq_idct_batch() cover all of them at
 * once, and only then are the costs compared.  That is two kernel launches per
 * macroblock instead of two per candidate, and the launches grow with the
 * candidate count (64 blocks instead of 16), which is what a device needs.
 *
 * It cannot be lifted to a whole picture without changing the bitstream: the
 * motion search predicts each macroblock's vectors from its neighbours'
 * *decided* vectors, intra prediction needs the reconstructed neighbours, and
 * the CAVLC cost model needs their non-zero counts.  All three are serial in
 * raster order, so the batch has to stop at the macroblock boundary -- see
 * docs/STATUS.md, "Rate-distortion batching (provisional)". */
static int64_t rd_luma_inter_batch(struct aeg_encoder *e, int mbx, int mby,
                                   mbenc *best, uint8_t *snap_l,
                                   int64_t best_cost, double lambda, int qp)
{
    const uint8_t *src = e->srcp[0] + mby * 16 * e->stride[0] + mbx * 16;
    const int n = e->mb_ncand;
    int k;
    int64_t t0;

    /* 1. every candidate's prediction, and its residual in the batch buffer */
    for (k = 0; k < n; k++) {
        mbenc c;
        cand_enc(e, k, qp, &c);
        inter_predict_typed(e, mbx, mby, &c);
        luma_inter_gather(e, src, e->pred_l, e->rd_res + 256 * k);
        memcpy(e->rd_pred + 256 * k, e->pred_l, 256);
    }
    /* 2. one launch each for the whole macroblock's candidates */
    t0 = aeg_now_ns();
    e->d.ops.tq_batch(e->d.ops.priv, e->rd_res, 16 * n, qp, 0, e->rd_lev,
                      e->rd_nz);
    aeg_dispatch_record(&e->d, AEG_STAGE_TRANSFORM,
                        e->d.ops.backend != AEG_BACKEND_CPU,
                        aeg_now_ns() - t0);
    t0 = aeg_now_ns();
    e->d.ops.iq_idct_batch(e->d.ops.priv, e->rd_lev, 16 * n, qp, e->rd_rec);
    aeg_dispatch_record(&e->d, AEG_STAGE_IQUANT_IDCT,
                        e->d.ops.backend != AEG_BACKEND_CPU,
                        aeg_now_ns() - t0);
    /* 3. decide, in the same order and with the same tie-breaking the serial
     *    loop uses (a later candidate has to be strictly cheaper) */
    for (k = 0; k < n; k++) {
        mbenc t;
        int64_t cost;
        cand_enc(e, k, qp, &t);
        /* The batch only produces 4x4 blocks: a candidate whose residual wants
         * the 8x8 transform is coded through the serial path, which is what
         * makes the two decisions agree. */
        memcpy(e->pred_l, e->rd_pred + 256 * k, 256);
        if (use_t8(e, mbx, mby, &t)) {
            code_luma_inter8(e, mbx, mby, &t);
        } else {
            luma_inter_finish(e, mbx, mby, &t, e->rd_lev + 256 * k,
                              e->rd_nz + 16 * k, e->rd_rec + 256 * k,
                              e->rd_pred + 256 * k);
        }
        t.bits += inter_hdr_bits(e, &t);
        cost = t.sse + (int64_t)(lambda * t.bits + 0.5);
        t.cost = cost;
        if (cost < best_cost) {
            best_cost = cost;
            *best = t;
            snap_luma(e, mbx, mby, snap_l);
        }
    }
    return best_cost;
}

/* Code the chroma of one candidate and return the cost of the chroma part. */
static int64_t rd_chroma(struct aeg_encoder *e, int mbx, int mby, mbenc *t,
                         int inter, int cmode, double lambda)
{
    int64_t sse0 = t->sse;
    int bits0 = t->bits;
    t->chroma_mode = cmode;
    code_chroma(e, mbx, mby, t, inter);
    return (t->sse - sse0) +
           (int64_t)(lambda * (t->bits - bits0) + 0.5);
}



static void code_mb(struct aeg_encoder *e, int mbx, int mby, mbenc *c)
{
    /* Both snapshots are written by the first candidate that is evaluated, and
     * at least one always is (a preset above 0 with no legal Intra_16x16 mode
     * returns above), but initialising them keeps the compiler from warning
     * about every memcpy in the snapshot helpers. */
    uint8_t snap_l[16 * 16] = { 0 }, snap_c[2 * 8 * 8] = { 0 };
    mbenc best, t;
    int64_t best_cost = INT64_MAX;
    double lambda = AEG_LAMBDA(c->qp);
    int i, inter;

    if (e->p.preset <= 0 || e->mb_nmodes == 0) {
        /* plain SAD decision (analyse_mb already left the predictor) */
        if (is_inter_type(c->type)) {
            code_luma_inter(e, mbx, mby, c);
            code_chroma(e, mbx, mby, c, 1);
        } else {
            code_luma_i16(e, mbx, mby, c);
            code_chroma(e, mbx, mby, c, 0);
        }
        return;
    }

    /* luma: the SAD-best modes (all of them from preset 2 on) plus every inter
     * partitioning the motion search produced */
    for (i = 0; i < (e->p.preset >= 2 ? e->mb_nmodes : 1); i++) {
        int64_t cost = rd_luma(e, mbx, mby, &t, c->qp, AEG_MB_I16x16,
                               e->mb_modes[i], lambda, -1);
        if (cost < best_cost) {
            best_cost = cost;
            best = t;
            snap_luma(e, mbx, mby, snap_l);
        }
    }
    if (e->rd_batch && e->mb_ncand > 0) {
        best_cost = rd_luma_inter_batch(e, mbx, mby, &best, snap_l, best_cost,
                                        lambda, c->qp);
    } else {
        for (i = 0; i < e->mb_ncand; i++) {
            int64_t cost = rd_luma(e, mbx, mby, &t, c->qp, e->mb_cand[i].type, 0,
                                   lambda, i);
            if (cost < best_cost) {
                best_cost = cost;
                best = t;
                snap_luma(e, mbx, mby, snap_l);
            }
        }
    }
    /* Intra_4x4: its own mode decision per block, so it is a candidate of its
     * own rather than one of the Intra_16x16 prediction modes.  It costs a
     * lot more than Intra_16x16 (sixteen predictions, sixteen transforms and
     * the per-block mode bits), so it is skipped when the motion search found
     * a match that is already twice as good as the best Intra_16x16
     * prediction: sixteen mode decisions cannot turn that around, and this is
     * where most of the encode time of the candidate goes. */
    if (!(e->mb_inter_ok && e->mb_inter_sad * 2 < e->mb_sad)) {
        int64_t cost = rd_luma_i4x4(e, mbx, mby, &t, c->qp, lambda);
        if (cost < best_cost) {
            best_cost = cost;
            best = t;
            snap_luma(e, mbx, mby, snap_l);
        }
        /* Intra_8x8: four mode decisions instead of sixteen, so it is much
         * cheaper to evaluate than Intra_4x4, and it reconstructs smooth
         * regions with fewer coefficients.  It only exists in the High
         * profile, where the picture parameter set enables the 8x8
         * transform. */
        if (e->p.transform_8x8) {
            cost = rd_luma_i8x8(e, mbx, mby, &t, c->qp, lambda);
            if (cost < best_cost) {
                best_cost = cost;
                best = t;
                snap_luma(e, mbx, mby, snap_l);
            }
        }
    }
    restore_luma(e, mbx, mby, snap_l);
    inter = is_inter_type(best.type);

    /* chroma: every legal intra mode, or the motion-compensated prediction */
    if (!inter) {
        int64_t best_chroma = INT64_MAX;
        mbenc cbest;
        int cm;
        cbest = best;
        for (cm = 0; cm < 4; cm++) {
            if (!chroma_mode_ok(e, cm, mbx, mby))
                continue;
            t = best;
            {
                int64_t cost = rd_chroma(e, mbx, mby, &t, 0, cm, lambda);
                if (cost < best_chroma) {
                    best_chroma = cost;
                    t.chroma_mode = cm;
                    cbest = t;
                    snap_chroma(e, mbx, mby, snap_c);
                }
            }
        }
        best = cbest;
    } else {
        inter_predict_typed(e, mbx, mby, &best);
        t = best;
        rd_chroma(e, mbx, mby, &t, 1, t.chroma_mode, lambda);
        best = t;
        snap_chroma(e, mbx, mby, snap_c);
    }
    restore_chroma(e, mbx, mby, snap_c);
    *c = best;
    if (getenv("AEG_MBDBG") && best.type == AEG_MB_P8x8) {
        int q, n = mb_part_count(&best), fi = e->frame_count;
        fprintf(stderr, "F%d MB %d,%d st %d%d%d%d", fi, mbx, mby,
                best.sub_type[0], best.sub_type[1], best.sub_type[2],
                best.sub_type[3]);
        for (q = 0; q < n; q++) {
            int geo[4], b0;
            mb_part_geom(&best, q, geo, &b0);
            fprintf(stderr, " [b%d mv %d,%d pmv %d,%d]", b0, best.mv[b0][0],
                    best.mv[b0][1], best.pmv[b0][0], best.pmv[b0][1]);
        }
        fprintf(stderr, "\n");
    }
}

/* ------------------------------------------------------------------ */
/* Bitstream                                                           */
/* ------------------------------------------------------------------ */
static void write_mb(struct aeg_encoder *e, aeg_bitwriter *bw, int mbx, int mby,
                     mbenc *c, int is_p)
{
    int blk;
    if (!is_p) {
        if (c->type == AEG_MB_I4x4 || c->type == AEG_MB_I8x8)
            aeg_bw_ue(bw, 0);           /* I_NxN */
        else
            aeg_bw_ue(bw, (uint32_t)(1 + c->i16_mode + 4 * chroma_cbp(c) +
                                     12 * (luma_cbp_on(c) ? 1 : 0)));
    } else if (is_inter_type(c->type)) {
        /* P_L0_16x16 / P_L0_16x8 / P_L0_8x16: one mvd pair per partition, all
         * of them before the coded block pattern.  There is a single reference
         * picture, so ref_idx_l0 is omitted. */
        int p;
        aeg_bw_ue(bw, (uint32_t)inter_mb_type_code(c->type));
        if (c->type == AEG_MB_P8x8)
            /* sub_mb_type: four ue(v) symbols, before any mvd (7.3.5.2). */
            for (p = 0; p < 4; p++)
                aeg_bw_ue(bw, (uint32_t)c->sub_type[p]);
        for (p = 0; p < c->nparts; p++) {
            int geo[4], b0;
            mb_part_geom(c, p, geo, &b0);
            aeg_bw_se(bw, c->mv[b0][0] - c->pmv[b0][0]);
            aeg_bw_se(bw, c->mv[b0][1] - c->pmv[b0][1]);
        }
        aeg_bw_ue(bw, (uint32_t)aeg_cbp_codenum[(c->cbp & 0x0f) |
                                                (chroma_cbp(c) << 4)]);
        /* transform_size_8x8_flag of an inter macroblock: after the coded
         * block pattern, and only when the luma pattern is non-zero. */
        if (t8_allowed(e, c) && (c->cbp & 0x0f))
            aeg_bw_put(bw, 1, c->t8 ? 1u : 0u);
    } else if (c->type == AEG_MB_I4x4 || c->type == AEG_MB_I8x8) {
        aeg_bw_ue(bw, 5);               /* P slice: I_NxN */
    } else {
        aeg_bw_ue(bw, (uint32_t)(6 + c->i16_mode + 4 * chroma_cbp(c) +
                                 12 * (luma_cbp_on(c) ? 1 : 0)));
    }
    if (c->type == AEG_MB_I4x4 || c->type == AEG_MB_I8x8) {
        int b;
        /* transform_size_8x8_flag comes before the prediction modes (7.3.5.2)
         * and is present for every I_NxN macroblock when the parameter set
         * enables the 8x8 transform. */
        if (e->p.transform_8x8)
            aeg_bw_put(bw, 1, c->type == AEG_MB_I8x8 ? 1u : 0u);
        if (c->type == AEG_MB_I8x8) {
            /* One prev_intra8x8_pred_mode_flag per 8x8 block (raster order),
             * then rem_intra8x8_pred_mode where the mode is not the predicted
             * one.  The remainder is coded with the predicted value skipped,
             * exactly like Intra_4x4. */
            for (b = 0; b < 4; b++) {
                int pm = intra8_pred_mode(e, mbx, mby, c->intra8, b & 1,
                                          b >> 1);
                int mode = c->intra8[b];
                if (mode == pm) {
                    aeg_bw_put(bw, 1, 1);
                } else {
                    aeg_bw_put(bw, 1, 0);
                    aeg_bw_put(bw, 3, (uint32_t)(mode < pm ? mode : mode - 1));
                }
            }
            aeg_bw_ue(bw, (uint32_t)c->chroma_mode);
            aeg_bw_ue(bw, (uint32_t)aeg_cbp_codenum_i4x4[(c->cbp & 0x0f) |
                                                         (chroma_cbp(c) << 4)]);
            goto after_modes;
        }
        /* One prev_intra4x4_pred_mode_flag per block (raster order), followed
         * by rem_intra4x4_pred_mode when the mode is not the predicted one.
         * The remainder is coded with the predicted value skipped, so it is
         * (mode < pred) ? mode : mode - 1. */
        for (b = 0; b < 16; b++) {
            int rx = ((b >> 2) & 1) * 2 + (b & 1);
            int ry = ((b >> 3) & 1) * 2 + ((b >> 1) & 1);
            int pm = intra4_pred_mode(e, mbx, mby, c->intra4, rx, ry);
            int mode = c->intra4[b];
            if (mode == pm) {
                aeg_bw_put(bw, 1, 1);
            } else {
                aeg_bw_put(bw, 1, 0);
                aeg_bw_put(bw, 3, (uint32_t)(mode < pm ? mode : mode - 1));
            }
        }
        aeg_bw_ue(bw, (uint32_t)c->chroma_mode);
        aeg_bw_ue(bw, (uint32_t)aeg_cbp_codenum_i4x4[(c->cbp & 0x0f) |
                                                     (chroma_cbp(c) << 4)]);
    } else if (is_intra_type(c->type)) {
        aeg_bw_ue(bw, (uint32_t)c->chroma_mode);
    }
after_modes:
    /* mb_qp_delta: only present when the picture actually carries a coded
     * block pattern (H.264 7.3.5.2); emitting it unconditionally desynchronises
     * the decoder on the first all-zero macroblock. */
    if (luma_cbp_on(c) || chroma_cbp(c))
        aeg_bw_se(bw, c->qp - e->last_qp);
    e->last_qp = c->qp;

    if (c->type == AEG_MB_I16x16) {
        /* CAVLC Intra_16x16 always codes the luma DC block and all sixteen AC
         * blocks, even when they are empty (total_coeff == 0). */
        aeg_cavlc_write_block(bw, c->level[AEG_BLK_LUMA_DC],
                              luma_ac_nc(e, mbx, mby, c, 0), 1);
        if (luma_cbp_on(c))
            for (blk = 0; blk < 16; blk++) {
                int rb = ac_ord[blk];
                aeg_cavlc_write_block(bw, c->level[rb],
                                      luma_ac_nc(e, mbx, mby, c, blk), 3);
            }
    } else if (c->t8) {
        /* 8x8 residual: one 64-coefficient block per coded 8x8 group, sent as
         * four interleaved 4x4 lists.  The slots are filled list by list, in
         * the same order and the same state luma8_costs() measured. */
        int g;
        for (g = 0; g < 4; g++) {
            for (blk = 0; blk < 4; blk++) {
                int b = group_blk[g][blk];
                if (c->cbp & (1 << g)) {
                    aeg_cavlc_write_block8(bw, c->level8[g], blk,
                                           luma_nc(e, mbx, mby, c, b));
                    c->nz[b] = (int8_t)aeg_cavlc_nz8(c->level8[g], blk);
                } else {
                    c->nz[b] = 0;
                }
            }
            luma8_group_done(c, g);
        }
    } else {
        /* 4x4 residual: the blocks of every coded 8x8 group are written group
         * by group, and every block of a coded group is present, including the
         * ones whose coeff_token ends up being TotalCoeff 0.  Skipping the
         * empty ones desynchronises the decoder. */
        {
            int g, k;
            for (g = 0; g < 4; g++) {
                if (!(c->cbp & (1 << g)))
                    continue;
                for (k = 0; k < 4; k++) {
                    int b = group_blk[g][k];
                    aeg_cavlc_write_block(bw, c->level[b],
                                          luma_nc(e, mbx, mby, c, b), 0);
                }
            }
        }
    }
    if (chroma_cbp(c)) {
        /* chroma_cbp 1: DC only, 2: DC + AC.  Both planes' DC blocks are always
         * coded when the pattern is non-zero, and with AC_N all eight AC blocks
         * are coded as well -- even the ones that end up empty. */
        int pl, b;
        for (pl = 0; pl < 2; pl++) {
            aeg_cavlc_write_block(bw, c->level[AEG_BLK_CHROMA_DC + pl], 0, 2);
        }
        if (chroma_cbp(c) == 2)
            /* Chroma AC blocks are decoded like Intra_16x16 AC blocks: with
             * the 15-entry scan (zero at DC) and the 15-coefficient
             * total_zeros table. */
            for (pl = 0; pl < 2; pl++)
                for (b = 0; b < 4; b++) {
                    aeg_cavlc_write_block(bw,
                                          c->level[AEG_BLK_CHROMA_CB + pl * 4 + b],
                                          chroma_nc(e, mbx, mby, c, pl, b), 3);
                }
    }
}

/* P_Skip run: macroblocks whose motion equals the prediction and that have no
 * residual are coded with the skip run length instead of a macroblock. */
/* Every coded macroblock of a P slice is preceded by mb_skip_run -- also when
 * no macroblock is skipped (run == 0).  Omitting the zeros desynchronises the
 * decoder at the first macroblock of the frame. */
static void flush_skip_run(aeg_bitwriter *bw, int *run)
{
    aeg_bw_ue(bw, (uint32_t)*run);
    *run = 0;
}

static int ensure_buf(uint8_t **buf, size_t *cap, size_t need)
{
    if (need <= *cap)
        return AEG_OK;
    aeg_free(*buf);
    *cap = need + need / 2 + 4096;
    *buf = (uint8_t *)aeg_alloc(*cap);
    return *buf ? AEG_OK : AEG_ENOMEM;
}

/* Variance of a 16x16 luma block of the current source picture. */
static int mb_variance(const struct aeg_encoder *e, int mbx, int mby)
{
    const uint8_t *p = e->srcp[0] + (mby * 16) * e->stride[0] + mbx * 16;
    int x, y, sum = 0, sum2 = 0;
    for (y = 0; y < 16; y++)
        for (x = 0; x < 16; x++) {
            int v = p[y * e->stride[0] + x];
            sum += v;
            sum2 += v * v;
        }
    /* 256 * var, exact in integers; the caller only needs the logarithm */
    return (sum2 * 256 - sum * sum) / 256;
}

/* Adaptive quantisation (x264's "variance AQ"): flat macroblocks -- sky,
 * gradients, the inside of a moving object -- can take a coarser quantiser
 * without becoming visible, while busy macroblocks are where the eye looks
 * and where the distortion is masked least.  The offset is proportional to
 * `log2(frame variance) - log2(block variance)`, which is the same
 * relationship x264 uses; the result is shifted to a zero mean so the picture
 * QP (and therefore the rate-control target) is unaffected.  `aq_strength`
 * is in tenths, 10 being x264's aq-strength=1.0. */
#define AQ_MAX_DQP 12
static void aq_compute(struct aeg_encoder *e)
{
    double strength = e->p.aq_strength / 10.0;
    int n = e->mb_count, i, x, y, mean = 0, adj;

    if (e->p.aq_mode <= 0 || strength <= 0.0) {
        for (i = 0; i < n; i++)
            e->mb_dqp[i] = 0;
        return;
    }
    {
        double logsum = 0.0;
        for (i = 0; i < n; i++)
            logsum += log2((double)mb_variance(e, i % e->mb_w, i / e->mb_w) + 1.0);
        {
            double avg = logsum / n;
            for (i = 0; i < n; i++) {
                int v = mb_variance(e, i % e->mb_w, i / e->mb_w);
                double d = strength * (avg - log2((double)v + 1.0));
                /* a block that is already flat to within noise gets a little
                 * less than the log suggests: the deadzone of the quantiser
                 * already removes it, and pushing it further only spends
                 * header bits */
                adj = (int)(d + (d < 0 ? -0.5 : 0.5));
                if (adj > AQ_MAX_DQP)  adj = AQ_MAX_DQP;
                if (adj < -AQ_MAX_DQP) adj = -AQ_MAX_DQP;
                e->mb_dqp[i] = (int8_t)adj;
                mean += adj;
            }
        }
        /* zero mean, so the picture's average QP is exactly what the rate
         * controller asked for */
        mean = (int)((double)mean / n + (mean < 0 ? -0.5 : 0.5));
        for (i = 0; i < n; i++) {
            int d = e->mb_dqp[i] - mean;
            if (d > AQ_MAX_DQP)  d = AQ_MAX_DQP;
            if (d < -AQ_MAX_DQP) d = -AQ_MAX_DQP;
            e->mb_dqp[i] = (int8_t)d;
        }
    }
    (void)x; (void)y;
}

/* QP of one macroblock of the current picture. */
static int mb_qp_of(const struct aeg_encoder *e, int mbx, int mby)
{
    int qp = e->cur_qp + e->mb_dqp[mby * e->mb_w + mbx];
    if (qp < 0)  qp = 0;
    if (qp > 51) qp = 51;
    return qp;
}

/* VBV/HRD: how many bits the picture about to be coded may take.  The channel
 * has delivered `vbv_rate_bits` since the last picture; the buffer cannot hold
 * more than B, so a burst is capped at the buffer size (a picture larger than
 * that could never be sent at rate R). */
static double vbv_available(const struct aeg_encoder *e)
{
    double level = e->vbv_full + e->vbv_rate_bits;

    if (level > e->vbv_buf_bits)
        level = e->vbv_buf_bits;
    return level;
}

/* Raise the QP if the picture would not fit in the buffer.
 *
 * The size of a picture is not known before it is coded, so it is predicted
 * from the previous one: rate grows roughly with 2^(-QP/6), so a picture coded
 * at QP q is expected to take last_bits * 2^((last_qp - q)/6) bits.  Solving
 * that for the largest QP whose prediction fits gives the clamp.  A picture
 * that still does not fit at QP 51 (a buffer smaller than a single picture)
 * cannot be helped by the quantiser; it is counted as an underflow when it
 * happens instead of being hidden. */
static void vbv_clamp_qp(struct aeg_encoder *e)
{
    double avail, pred, q_min;

    if (!e->vbv_active || e->last_pic_bits <= 0 || e->last_pic_qp < 0)
        return;
    avail = vbv_available(e);
    if (avail < 1.0)
        avail = 1.0;
    pred = (double)e->last_pic_bits *
           pow(2.0, (double)(e->last_pic_qp - e->cur_qp) / 6.0);
    if (pred <= avail)
        return;
    q_min = (double)e->last_pic_qp -
            6.0 * log2(avail / (double)e->last_pic_bits);
    if ((double)e->cur_qp < q_min) {
        int q = (int)ceil(q_min);

        if (q > 51) q = 51;
        if (q > e->cur_qp) {
            e->cur_qp = q;
            e->vbv_clamps++;
        }
    }
}

/* Take the coded picture out of the modelled buffer. */
static void vbv_update(struct aeg_encoder *e, int bits)
{
    double level;

    e->last_pic_bits = bits;
    e->last_pic_qp = e->cur_qp;
    if (!e->vbv_active)
        return;
    level = vbv_available(e) - (double)bits;
    if (level < 0.0) {
        /* the picture needed more than the buffer held: the model cannot
         * deliver it, and a decoder would starve waiting for the rest */
        e->vbv_underflows++;
        level = 0.0;
    }
    e->vbv_full = level;
    if (level < e->vbv_min_full)
        e->vbv_min_full = level;
}

/* One step of the ABR controller (H.264 Annex; a one-pass, picture-level
 * controller).
 *
 * The target is a *long-term* average, so the error is driven to zero instead
 * of merely being reduced: every picture's bits are accumulated, the difference
 * between the budget for the pictures coded so far and the bits actually spent
 * (`rc_debt`) is spread over the next `RC_WINDOW` pictures, and the QP follows
 * the logarithm of (actual / wanted) -- the log is the right variable because
 * the rate grows roughly exponentially with QP.  A keyframe is many times the
 * size of a P picture; the debt term absorbs it over the following window
 * rather than giving keyframes a target of their own, which is what keeps the
 * average on track for short streams too. */
#define RC_WINDOW 6.0
#define RC_GAIN   1.2
#define RC_DEBT_MAX 3.0
static void rc_update(struct aeg_encoder *e, int bits, int is_idr)
{
    double bpf = e->rc_bits_per_frame;
    double want, ratio, adj;

    if (!e->rc_active || bpf <= 0.0)
        return;
    e->rc_frames++;
    e->rc_spent += (double)bits;
    e->rc_debt = e->rc_frames * bpf - e->rc_spent;
    /* One picture can be many times the target -- a keyframe, or any picture
     * after a scene cut -- and letting that single picture's overshoot follow
     * the controller for the next `RC_WINDOW` pictures is what turns one
     * expensive frame into thirty bad ones.  The debt is therefore clamped to
     * RC_DEBT_MAX pictures' worth: the overshoot is still repaid, just over a
     * window long enough that the quality does not collapse (the alternative,
     * a 20% target floor, lets the QP run away while the debt is large). */
    if (e->rc_debt >  RC_DEBT_MAX * bpf) e->rc_debt =  RC_DEBT_MAX * bpf;
    if (e->rc_debt < -RC_DEBT_MAX * bpf) e->rc_debt = -RC_DEBT_MAX * bpf;
    want = bpf + e->rc_debt / RC_WINDOW;
    if (want < 0.2 * bpf)          /* never ask for the impossible */
        want = 0.2 * bpf;
    if (e->vbv_active) {
        /* ask for a picture the buffer can actually take, with a little room
         * left over so that the clamp in vbv_clamp_qp() is the exception */
        double room = vbv_available(e) * 0.8;

        if (want > room)
            want = room;
        if (want < 1.0)
            want = 1.0;
    }
    ratio = (double)(bits + 1) / want;
    adj = RC_GAIN * (ratio > 0.0 ? log2(ratio) : 0.0);
    if (adj > 1.5)  adj = 1.5;     /* at most 1.5 QP per picture, so a scene */
    if (adj < -1.5) adj = -1.5;    /* cut cannot make the quality collapse   */
    e->rc_qp += adj;
    if (e->rc_qp > 51.0) e->rc_qp = 51.0;
    if (e->rc_qp < 0.0)  e->rc_qp = 0.0;
    (void)is_idr;
}

/* Scene-cut detection.
 *
 * A P picture predicts from the previous reconstructed picture; when the scene
 * changes, that prediction is worthless and the P picture spends a whole frame's
 * worth of bits predicting something that is not there, at poor quality.  The
 * signal is a comparison of the two ways the picture could be coded:
 *
 *   inter: |cur - ref| at zero motion.  Every real motion field is at least as
 *          good as "no motion", so this is an upper bound on the inter cost.
 *   intra: the AC energy of the picture (|cur - block mean| per 16x16
 *          macroblock), which is what Intra_16x16 DC prediction has to code.
 *
 * The ratio is what separates the two cases: on the panning test clip it is
 * 0.7-0.75 for ordinary motion (the frame correlation dominates) and 2.9 at a
 * cut between two unrelated clips, so `p_sad > scenecut/100 * ac` (default 1.2)
 * is a wide margin either way.  Note that the first version of this compared
 * inter against the picture's *vertical gradient* instead, which is not an
 * intra cost at all: fast motion made every ordinary frame look like a cut.
 *
 * A cut also has to respect a minimum distance to the previous IDR (a tenth of
 * the GOP, clamped to [8, 32]): a flash or a hard pan would otherwise turn
 * every picture into a keyframe. */
static int is_scene_cut(struct aeg_encoder *e, int is_idr)
{
    const uint8_t *cur = e->srcp[0], *ref = e->refp[0];
    int stride = e->stride[0];
    int64_t ac = 0, p_sad = 0;
    int bx, by, x, y;

    if (e->p.scenecut <= 0 || is_idr || !e->has_ref || e->frame_count < 2)
        return 0;
    {
        int min_gop = e->p.gop_size > 0 ? e->p.gop_size / 10 : 8;

        if (min_gop < 8) min_gop = 8;
        if (min_gop > 32) min_gop = 32;
        if (e->frame_count - e->last_idr_frame < min_gop)
            return 0;
    }
    /* every other sample in both directions: the ratio of two sums over a
     * quarter of the picture is within a percent or two of the whole one */
    for (by = 0; by + 16 <= e->h; by += 16) {
        for (bx = 0; bx + 16 <= e->w; bx += 16) {
            int64_t sum = 0, n = 0, dev = 0;
            int mean;

            for (y = by; y < by + 16; y += 2)
                for (x = bx; x < bx + 16; x += 2) {
                    sum += cur[(ptrdiff_t)y * stride + x];
                    n++;
                }
            mean = (int)(sum / (n ? n : 1));
            for (y = by; y < by + 16; y += 2)
                for (x = bx; x < bx + 16; x += 2) {
                    int d = (int)cur[(ptrdiff_t)y * stride + x] - mean;

                    dev += d < 0 ? -d : d;
                }
            ac += dev;
        }
    }
    for (y = 0; y < e->h; y += 2)
        for (x = 0; x < e->w; x += 2) {
            int d = (int)cur[(ptrdiff_t)y * stride + x] -
                    (int)ref[(ptrdiff_t)y * stride + x];

            p_sad += d < 0 ? -d : d;
        }
    if (getenv("AEG_SCDBG"))
        fprintf(stderr, "SCD frame %d ac %lld p_sad %lld ratio %.3f\n",
                e->frame_count, (long long)ac, (long long)p_sad,
                ac ? (double)p_sad / (double)ac : 0.0);
    if (ac <= 0)
        return p_sad > 0;
    return p_sad * 100 > ac * (int64_t)e->p.scenecut;
}

/* One slice of the current picture: macroblock rows `first_row` to `last_row`
 * (inclusive), written as a complete NAL payload (RBSP) of its own.
 *
 * A slice is the unit of independence in H.264, which is what makes slice
 * threads possible: every neighbour outside the slice counts as unavailable
 * (see `top_row`), the nC contexts restart, and the QP context restarts with
 * the slice's own slice_qp_delta.  With one slice per picture nothing above
 * changes and the bitstream is exactly what it was before slices existed. */
static int code_slice(struct aeg_encoder *e, uint8_t *bs, size_t cap,
                      int first_row, int last_row, int is_p, int is_idr,
                      size_t *rbsp_len)
{
    int mbx, mby;
    int skip_run = 0;
    aeg_bitwriter bw;

    e->top_row = first_row;
    e->last_qp = e->cur_qp;
    aeg_bw_init(&bw, bs, cap, 0);
    aeg_write_slice_header(&bw, is_p ? 5 : 7,
                           e->frame_num & ((1 << e->frame_num_bits) - 1),
                           e->frame_num_bits, is_idr, e->idr_pic_id,
                           e->cur_qp - 26, 0, 0,
                           e->p.deblock ? 0 : 1, first_row * e->mb_w);

    /* ---- decision pass -------------------------------------------------
     * Everything that decides *what* a macroblock is: the motion search, the
     * mode decision, the transform/quantise/reconstruct of every candidate and
     * the reconstruction written into the picture.  Nothing here writes to the
     * bitstream, so the macroblocks only have to be processed in an order that
     * respects their neighbours (see the note above aeg_encoder_encode). */
    for (mby = first_row; mby <= last_row; mby++) {
        for (mbx = 0; mbx < e->mb_w; mbx++) {
            int idx = mby * e->mb_w + mbx;
            mbenc *tmp = &e->mb_encs[idx];
            aeg_mb *m = &e->mbs[idx];
            int i;
            int64_t t0 = aeg_now_ns();

            reset_mb_enc(tmp, mb_qp_of(e, mbx, mby));
            analyse_mb(e, mbx, mby, tmp);
            e->st.stage_ns[AEG_STAGE_MOTION_EST] += aeg_now_ns() - t0;

            t0 = aeg_now_ns();
            code_mb(e, mbx, mby, tmp);
            e->st.stage_ns[AEG_STAGE_TRANSFORM] += aeg_now_ns() - t0;

            /* store the decisions for the neighbour queries of the following
             * macroblocks (and for the deblocking filter) */
            memset(m, 0, sizeof(*m));
            m->type = (uint8_t)tmp->type;
            m->i16_mode = (uint8_t)tmp->i16_mode;
            m->t8 = (uint8_t)tmp->t8;
            m->chroma_mode = (uint8_t)tmp->chroma_mode;
            m->cbp = (uint8_t)tmp->cbp;
            m->qp = (int8_t)tmp->qp;
            for (i = 0; i < 16; i++)
                m->intra4[i] = tmp->intra4[i];
            for (i = 0; i < 4; i++)
                m->intra8[i] = tmp->intra8[i];
            for (i = 0; i < AEG_BLK_COUNT; i++)
                m->nz[i] = tmp->nz[i];
            for (i = 0; i < 4; i++)
                m->nz8[i] = (int8_t)(tmp->nz8[i] != 0);
            for (i = 0; i < 16; i++) {
                m->mv[i][0] = tmp->mv[i][0];
                m->mv[i][1] = tmp->mv[i][1];
            }
            m->cost = (int32_t)tmp->sse;
        }
    }

    /* ---- bitstream pass ------------------------------------------------
     * CAVLC is serial: the bit cost and the bits themselves depend on the
     * non-zero counts of the neighbours, so the macroblocks go out in raster
     * order, and mb_qp_delta depends on the previous macroblock's QP. */
    for (mby = first_row; mby <= last_row; mby++) {
        for (mbx = 0; mbx < e->mb_w; mbx++) {
            int idx = mby * e->mb_w + mbx;
            mbenc *tmp = &e->mb_encs[idx];
            int64_t t0 = aeg_now_ns();

            /* A macroblock without any coded residual does not carry an
             * mb_qp_delta (H.264 7.3.5), so the decoder keeps the previous QP
             * for it.  Reconstruction is the prediction either way, but the
             * deblocking filter reads the QP, so the encoder has to keep the
             * same value: fall back to the QP the decoder is still using. */
            if (!luma_cbp_on(tmp) && !chroma_cbp(tmp) &&
                tmp->type != AEG_MB_I16x16)
                tmp->qp = e->last_qp;
            e->mbs[idx].qp = (int8_t)tmp->qp;

            /* emit.  A P_L0_16x16 macroblock whose motion vector equals the
             * predictor and that carries no residual at all reconstructs
             * exactly like P_Skip (same prediction, no residual), so it is
             * coded as part of the skip run: same picture, fewer bits. */
            {
                int skipped = 0;
                if (is_p && tmp->type == AEG_MB_P16x16 && tmp->cbp == 0) {
                    int pmvx, pmvy;
                    pskip_mv(e, mbx, mby, &pmvx, &pmvy);
                    if (tmp->mv[0][0] == pmvx && tmp->mv[0][1] == pmvy)
                        skipped = 1;
                }
                if (skipped)
                    skip_run++;
                else {
                    if (is_p)
                        flush_skip_run(&bw, &skip_run);
                    write_mb(e, &bw, mbx, mby, tmp, is_p);
                }
            }
            e->st.stage_ns[AEG_STAGE_TRANSFORM] += aeg_now_ns() - t0;
        }
    }
    /* A skip run that reaches the end of the frame still has to be emitted:
     * the decoder counts macroblocks by consuming mb_skip_run values, so an
     * unwritten tail run makes it stop before the last row. */
    if (is_p && skip_run > 0)
        flush_skip_run(&bw, &skip_run);
    /* aeg_bw_finish() appends rbsp_trailing_bits (the stop bit and the zero
     * padding); aligning here as well used to add a second, all-zero-padded
     * 0x80 byte.  A single slice could get away with it - the decoder stops at
     * the end of the picture and ignores the tail - but a slice that ends in
     * the middle of the picture must end exactly at its last macroblock, or
     * the decoder keeps reading macroblocks out of the next slice's header. */
    *rbsp_len = aeg_bw_finish(&bw);
    return AEG_OK;
}

/* How many threads the slice loop may use.  A device context (one command
 * queue) is not safe to submit to from several threads, so as long as any stage
 * is dispatched to a non-CPU backend the slices stay serial; per-thread device
 * contexts are the follow-up that lifts that. */
static int slice_thread_count(const struct aeg_encoder *e)
{
    int k;

    if (e->n_threads <= 1 || e->n_slices <= 1)
        return 1;
    for (k = 0; k < AEG_STAGE_COUNT; k++)
        if (e->d.use_gpu[k] && e->d.ops.backend != AEG_BACKEND_CPU)
            return 1;
    return e->n_threads;
}

/* ------------------------------------------------------------------ */
/* Slice threads                                                       */
/* ------------------------------------------------------------------ */
/* A slice is coded entirely from the samples and the decisions of its own
 * macroblock rows (that is what code_slice() makes true: every neighbour outside
 * the slice is unavailable), so two slices can be coded at the same time.  What
 * they cannot share is the encoder's *scratch*: `top_row`, the candidate list of
 * the macroblock being analysed, the pending skip run, the bit writer, the stage
 * timers.  Those are plain fields of struct aeg_encoder, so rather than thread a
 * per-thread context through every function of the mode decision, each thread
 * runs against its own shallow copy of the encoder: the copy carries the
 * scratch, and every pointer in it (the source and reference pictures, the
 * coefficient planes, the per-macroblock decisions, the slice buffers) points at
 * the same memory as the original.  The macroblocks a thread touches are exactly
 * the rows of the slices it owns, so its writes into that shared memory are
 * disjoint from every other thread's, and everything else it reads it does not
 * write at all.
 *
 * Two consequences, both handled in encode_frame():
 *   - whatever a thread writes through the struct (the stage timers, the
 *     dispatcher's counters) has to be merged back after the join, and
 *   - a buffer that a thread *grows* would be grown in its private copy, so
 *     every slice buffer is allocated before the threads start.
 * The bitstream does not depend on the thread count or on which thread coded
 * which slice: a slice's bits are a function of its own rows only. */
typedef struct {
    struct aeg_encoder *te;       /* the thread's copy of the encoder     */
    int tid, nthreads;
    int is_p, is_idr;
    size_t lens[AEG_MAX_SLICES];
    int err;
} slice_job;

static void *slice_thread(void *arg)
{
    slice_job *j = (slice_job *)arg;
    struct aeg_encoder *te = j->te;
    int si;

    for (si = j->tid; si < te->n_slices; si += j->nthreads) {
        int first_row = si * te->slice_rows;
        int last_row = first_row + te->slice_rows;

        if (first_row >= te->mb_h)
            break;
        if (last_row > te->mb_h)
            last_row = te->mb_h;
        if (code_slice(te, te->slice_bs[si], te->slice_bs_cap[si], first_row,
                       last_row - 1, j->is_p, j->is_idr, &j->lens[si]) != AEG_OK) {
            j->err = 1;
            break;
        }
    }
    return NULL;
}

static int encode_frame(struct aeg_encoder *e, int is_idr, int64_t pts,
                        aeg_packet *pkt)
{
    int is_p = !is_idr && e->has_ref;
    size_t out_len;
    int64_t t_all = aeg_now_ns();

    if (ensure_buf(&e->out, &e->out_cap,
                   (size_t)e->mb_count * 400 * 3 + 65536) != AEG_OK)
        return AEG_ENOMEM;

    /* the picture's QP: constant in CQP, the controller's output in ABR */
    e->cur_qp = e->p.qp;
    if (e->rc_active) {
        e->cur_qp = (int)(e->rc_qp + 0.5);
        /* a keyframe carries the whole picture on its own, so it gets a
         * slightly finer quantiser than the inter pictures that follow */
        if (is_idr)
            e->cur_qp -= 2;
        if (e->cur_qp < 0)  e->cur_qp = 0;
        if (e->cur_qp > 51) e->cur_qp = 51;
    }
    if (e->rc_qp_offset) {
        e->cur_qp += e->rc_qp_offset;
        if (e->cur_qp < 0)  e->cur_qp = 0;
        if (e->cur_qp > 51) e->cur_qp = 51;
    }
    if (e->vbv_active)
        vbv_clamp_qp(e);

    resolve_coef_stages(e);
    aq_compute(e);
    if (is_p && e->p.subme > 0) {
        /* Build the fractional-pel planes of the reference picture once for the
         * whole picture (AEG_STAGE_INTER_PRED).  Every subpel candidate of every
         * macroblock then reads samples from here instead of re-running the
         * 6-tap chain per block: the same samples, computed ~30x less often. */
        int64_t t0 = aeg_now_ns();
        if (e->d.use_gpu[AEG_STAGE_INTER_PRED] && e->d.ops.interp_plane &&
            e->d.ops.backend != AEG_BACKEND_CPU) {
            /* the planes are what the GPU stage is for: one launch per plane,
             * every output sample independent, and the kernel is checked
             * against aeg_hpel_plane_fill() by tests/kernel_check.c */
            int i;
            aeg_mc_ctx_setref(&e->mc, e->refp[0]);
            for (i = 1; i < 4; i++)
                e->d.ops.interp_plane(e->d.ops.priv, e->refp[0], e->stride[0],
                                      e->w, e->h, i,
                                      (uint8_t *)e->mc.base[i], e->mc.stride);
            e->mc.ready = 1;
        } else {
            aeg_mc_ctx_refresh(&e->mc, e->refp[0], e->w, e->h);
        }
        aeg_dispatch_record(&e->d, AEG_STAGE_INTER_PRED,
                            e->d.use_gpu[AEG_STAGE_INTER_PRED] &&
                            e->d.ops.backend != AEG_BACKEND_CPU,
                            aeg_now_ns() - t0);
        e->st.stage_ns[AEG_STAGE_INTER_PRED] += aeg_now_ns() - t0;
    }
    me_seed_frame(e);

    /* frame_num counts the pictures since the last IDR (H.264 8.2.1): an IDR
     * always carries 0, otherwise the decoder invents "non-existing" frames to
     * fill the gap. */
    if (is_idr)
        e->frame_num = 0;
    e->cur_is_p = is_p;

    /* One slice per picture is the default and keeps the bitstream single-NAL;
     * `--slices N` (or `-t N`, which uses one slice per thread) cuts the picture
     * into N row ranges, each an independently decodable NAL. */
    out_len = 0;
    if (is_idr) {
        uint8_t sps[256], pps[64];
        size_t sl = aeg_write_sps(sps, sizeof(sps), &e->p, e->mb_w, e->mb_h,
                                  e->frame_num_bits, e->level_idc);
        size_t pl = aeg_write_pps(pps, sizeof(pps), e->p.transform_8x8);
        out_len += aeg_nal_write(e->out + out_len, e->out_cap - out_len, 3, 7,
                                 sps, sl, 1);
        out_len += aeg_nal_write(e->out + out_len, e->out_cap - out_len, 3, 8,
                                 pps, pl, 0);
    }
    {
        size_t lens[AEG_MAX_SLICES];
        int si;

        memset(lens, 0, sizeof(lens));
        /* allocate every slice buffer up front: the pointers live in the
         * encoder struct that the threads copy */
        for (si = 0; si < e->n_slices; si++) {
            int first_row = si * e->slice_rows;
            int last_row = first_row + e->slice_rows;

            if (first_row >= e->mb_h)
                break;
            if (last_row > e->mb_h)
                last_row = e->mb_h;
            if (ensure_buf(&e->slice_bs[si], &e->slice_bs_cap[si],
                           (size_t)(last_row - first_row) * e->mb_w * 400 +
                           16384) != AEG_OK)
                return AEG_ENOMEM;
        }

        if (slice_thread_count(e) > 1) {
            int nt = slice_thread_count(e);
            struct aeg_encoder *copies = calloc((size_t)nt, sizeof(*copies));
            slice_job *jobs = calloc((size_t)nt, sizeof(*jobs));
            pthread_t *th = calloc((size_t)nt, sizeof(*th));

            if (!copies || !jobs || !th) {
                free(copies);
                free(jobs);
                free(th);
                return AEG_ENOMEM;
            }
            for (si = 0; si < nt; si++) {
                copies[si] = *e;
                /* the timers are accumulated per thread and summed below, so
                 * the copy must not inherit (and then re-add) the picture's */
                memset(copies[si].st.stage_ns, 0,
                       sizeof(copies[si].st.stage_ns));
                memset(copies[si].st.stage_gpu_ns, 0,
                       sizeof(copies[si].st.stage_gpu_ns));
                memset(copies[si].st.stage_cpu_ns, 0,
                       sizeof(copies[si].st.stage_cpu_ns));
                memset(copies[si].d.cpu_ns, 0, sizeof(copies[si].d.cpu_ns));
                memset(copies[si].d.gpu_ns, 0, sizeof(copies[si].d.gpu_ns));
                jobs[si].te = &copies[si];
                jobs[si].tid = si;
                jobs[si].nthreads = nt;
                jobs[si].is_p = is_p;
                jobs[si].is_idr = is_idr;
                jobs[si].err = 0;
            }
            for (si = 1; si < nt; si++)
                if (pthread_create(&th[si], NULL, slice_thread, &jobs[si]) != 0) {
                    slice_thread(&jobs[si]);  /* no thread: run it here */
                    th[si] = 0;
                }
            slice_thread(&jobs[0]);           /* this thread codes slice 0 */
            for (si = 1; si < nt; si++)
                if (th[si])
                    pthread_join(th[si], NULL);

            {
                int bad = 0;

                for (si = 0; si < nt; si++) {
                    int k;

                    if (jobs[si].err)
                        bad = 1;
                    for (k = 0; k < e->n_slices && k < AEG_MAX_SLICES; k++)
                        if (jobs[si].lens[k])
                            lens[k] = jobs[si].lens[k];
                    for (k = 0; k < AEG_STAGE_COUNT; k++) {
                        e->st.stage_ns[k] += copies[si].st.stage_ns[k];
                        e->st.stage_gpu_ns[k] += copies[si].st.stage_gpu_ns[k];
                        e->st.stage_cpu_ns[k] += copies[si].st.stage_cpu_ns[k];
                        e->d.cpu_ns[k][0] += copies[si].d.cpu_ns[k][0];
                        e->d.gpu_ns[k][0] += copies[si].d.gpu_ns[k][0];
                    }
                }
                free(copies);
                free(jobs);
                free(th);
                if (bad)
                    return AEG_ENOMEM;
            }
        } else {
            for (si = 0; si < e->n_slices; si++) {
                int first_row = si * e->slice_rows;
                int last_row = first_row + e->slice_rows;
                size_t sl_len = 0;

                if (first_row >= e->mb_h)
                    break;
                if (last_row > e->mb_h)
                    last_row = e->mb_h;
                if (code_slice(e, e->slice_bs[si], e->slice_bs_cap[si],
                               first_row, last_row - 1, is_p, is_idr,
                               &sl_len) != AEG_OK)
                    return AEG_ENOMEM;
                lens[si] = sl_len;
            }
        }

        for (si = 0; si < e->n_slices; si++) {
            if (!lens[si])
                break;
            out_len += aeg_nal_write(e->out + out_len, e->out_cap - out_len,
                                     is_idr ? 3 : 2, is_idr ? 5 : 1,
                                     e->slice_bs[si], lens[si], 0);
        }
    }
    e->top_row = 0;

    /* deblock the reconstructed picture */
    {
        int64_t t0 = aeg_now_ns();
        if (e->p.deblock) {
            e->pic.plane[0] = e->recp[0];
            e->pic.plane[1] = e->recp[1];
            e->pic.plane[2] = e->recp[2];
            aeg_deblock_frame(&e->pic, e->mbs, e->mb_w, e->p.deblock_alpha,
                              e->p.deblock_beta, 1);
        }
        e->st.stage_ns[AEG_STAGE_DEBLOCK] += aeg_now_ns() - t0;
    }

    /* the deblocked reconstruction is the next reference: refresh its border */
    {
        int i;
        for (i = 0; i < 3; i++)
            pad_edges(e->recp[i], e->stride[i], i ? e->cw : e->w, i ? e->ch : e->h);
    }

    /* distortion statistics */
    {
        int64_t t0 = aeg_now_ns();
        double sse = 0;
        int i;
        for (i = 0; i < 3; i++) {
            int w = i ? e->cw : e->w, h = i ? e->ch : e->h;
            int x, y;
            int64_t acc = 0;
            for (y = 0; y < h; y++)
                for (x = 0; x < w; x++) {
                    int d = (int)e->recp[i][y * e->stride[i] + x] -
                            (int)e->srcp[i][y * e->stride[i] + x];
                    acc += (int64_t)d * d;
                }
            sse = sse + (double)acc / (double)(w * h);
            e->st.stage_ns[AEG_STAGE_PSNR] += 0;
        }
        e->st.psnr_avg = sse > 0 ? 10.0 * log10(255.0 * 255.0 / (sse / 3.0)) : 99.0;
        (void)t0;
    }

    if (pkt) {
        pkt->data = e->out;
        pkt->size = out_len;
        pkt->pts = pts;
        pkt->keyframe = is_idr;
        pkt->qp = e->cur_qp;
    }

    e->st.bytes += (int64_t)out_len;
    e->st.frames++;
    e->st.enc_ns += aeg_now_ns() - t_all;
    if (!e->has_ref)
        e->has_ref = 1;
    e->frame_count++;
    e->frame_num++;
    /* the reconstruction becomes the reference of the following pictures */
    {
        uint8_t *t;
        int i;
        for (i = 0; i < 3; i++) {
            t = e->refp[i]; e->refp[i] = e->recp[i]; e->recp[i] = t;
        }
        e->pic.plane[0] = e->recp[0];
        e->pic.plane[1] = e->recp[1];
        e->pic.plane[2] = e->recp[2];
    }
    if (is_idr)
        e->idr_pic_id = (e->idr_pic_id + 1) & 0xffff;
    if (getenv("AEG_RCDBG"))
        fprintf(stderr, "RC F%d qp %d bits %d target %.0f debt %.0f vbv %.0f\n",
                e->frame_count, e->cur_qp, (int)(out_len * 8),
                e->rc_bits_per_frame, e->rc_debt, e->vbv_full);
    vbv_update(e, (int)(out_len * 8));
    rc_update(e, (int)(out_len * 8), is_idr);
    aeg_dispatch_update(&e->d);
    return AEG_OK;
}

/* ------------------------------------------------------------------ */
/* Public API                                                          */
/* ------------------------------------------------------------------ */
aeg_encoder *aeg_encoder_open(const aeg_params *params)
{
    struct aeg_encoder *e;
    aeg_params *pp;

    if (!params || params->width < 16 || params->height < 16) {
        aeg_set_last_error("invalid dimensions");
        return NULL;
    }
    e = (struct aeg_encoder *)aeg_alloc(sizeof(*e));
    if (!e) {
        aeg_set_last_error("out of memory");
        return NULL;
    }
    memcpy(&e->p, params, sizeof(e->p));
    pp = &e->p;
    if (pp->fps_num <= 0) pp->fps_num = 30;
    if (pp->fps_den <= 0) pp->fps_den = 1;
    if (pp->qp < 0) pp->qp = 23;
    if (pp->qp > 51) pp->qp = 51;
    /* Presets trade search effort against speed: the integer search range and
     * whether the subpel refinement runs at all are what actually move the
     * quality/bitrate curve, so they scale with the preset unless the caller
     * asked for something specific. */
    if (pp->preset < 0) pp->preset = 0;
    if (pp->preset > 4) pp->preset = 4;
    if (pp->search_range <= 0) {
        static const int rng[5] = { 8, 16, 24, 32, 48 };
        pp->search_range = rng[pp->preset];
    }
    if (pp->subme < 0) pp->subme = pp->preset <= 0 ? 0 : 1;
    if (pp->gop_size < 0) pp->gop_size = 250;
    /* A buffer model without a rate target makes no sense -- a constant-QP
     * stream cannot be squeezed into a channel -- so asking for a VBV turns
     * ABR on, with the maximum rate as the target when no bitrate was given. */
    if (pp->vbv_maxrate_kbps > 0 && pp->vbv_bufsize_kbits > 0) {
        if (pp->bitrate_kbps <= 0)
            pp->bitrate_kbps = pp->vbv_maxrate_kbps;
        if (pp->bitrate_kbps > pp->vbv_maxrate_kbps) {
            aeg_log(1, "aeg: bitrate clamped to the VBV maximum rate (%d kb/s)",
                    pp->vbv_maxrate_kbps);
            pp->bitrate_kbps = pp->vbv_maxrate_kbps;
        }
        pp->rc_mode = AEG_RC_ABR;
    } else
        pp->vbv_maxrate_kbps = pp->vbv_bufsize_kbits = 0;

    if (pp->rc_mode != AEG_RC_ABR || pp->bitrate_kbps <= 0)
        pp->rc_mode = AEG_RC_CQP;
    else {
        double fps = (double)pp->fps_num / (double)pp->fps_den;
        e->rc_active = 1;
        e->rc_qp = (double)pp->qp;
        e->rc_bits_per_frame = (double)pp->bitrate_kbps * 1000.0 / fps;
        e->rc_spent = 0.0;
        e->rc_debt = 0.0;
        e->rc_frames = 0;
        aeg_log(1, "aeg: ABR %d kb/s -> %.0f bits/frame at %g fps, start QP %d",
                pp->bitrate_kbps, e->rc_bits_per_frame, fps, pp->qp);
        if (pp->vbv_maxrate_kbps > 0) {
            e->vbv_active = 1;
            e->vbv_buf_bits = (double)pp->vbv_bufsize_kbits * 1000.0;
            e->vbv_rate_bits = (double)pp->vbv_maxrate_kbps * 1000.0 / fps;
            /* start the model with the buffer nearly full: the first picture is
             * a keyframe and would starve a buffer that starts empty */
            e->vbv_full = 0.9 * e->vbv_buf_bits;
            e->vbv_min_full = e->vbv_full;
            e->last_pic_bits = 0;
            e->last_pic_qp = -1;
            aeg_log(1, "aeg: VBV %d kb/s, buffer %d kbit -> level %.0f bits, "
                        "%.0f bits delivered per picture",
                    pp->vbv_maxrate_kbps, pp->vbv_bufsize_kbits,
                    e->vbv_full, e->vbv_rate_bits);
        }
    }

    e->w = params->width & ~15;
    e->h = params->height & ~15;
    if (e->w != params->width || e->h != params->height)
        aeg_log(1, "aeg: dimensions rounded to %dx%d (multiples of 16)",
                e->w, e->h);
    e->cw = (e->w + 1) / 2;
    e->ch = (e->h + 1) / 2;
    e->mb_w = e->w / 16;
    e->mb_h = e->h / 16;
    e->mb_count = e->mb_w * e->mb_h;
    /* log2_max_frame_num_minus4 has to cover one GOP so that frame_num never
     * wraps back to 0 before the next IDR. */
    {
        int gop = e->p.gop_size > 0 ? e->p.gop_size : 250;
        e->frame_num_bits = 4;
        while ((1 << e->frame_num_bits) <= gop && e->frame_num_bits < 16)
            e->frame_num_bits++;
    }
    e->level_idc = 0;
    e->last_qp = e->cur_qp;

    if (enc_alloc_picture(e, &e->src_mem, e->srcp) != AEG_OK ||
        enc_alloc_picture(e, &e->rec_mem, e->recp) != AEG_OK ||
        enc_alloc_picture(e, &e->ref_mem, e->refp) != AEG_OK) {
        aeg_set_last_error("out of memory");
        aeg_free(e);
        return NULL;
    }
    aeg_mc_ctx_init(&e->mc, e->w, e->h, e->stride[0]);
    e->top_row = 0;
    /* One slice per picture unless the caller asked for more.  Slices own whole
     * macroblock rows, and that is the granularity the picture is cut at. */
    {
        int want = pp->slices > 0 ? pp->slices : 0;
        int th = pp->threads > 0 ? pp->threads : 0;

        /* -t N without --slices means one slice per thread: slices are what
         * makes the rows independent, so a thread needs its own slice. */
        if (want <= 0)
            want = th > 1 ? th : 1;
        if (want > AEG_MAX_SLICES) want = AEG_MAX_SLICES;
        if (want > e->mb_h) want = e->mb_h;
        e->slice_rows = (e->mb_h + want - 1) / want;
        e->n_slices = (e->mb_h + e->slice_rows - 1) / e->slice_rows;

        /* threads: 0 = auto (the CPUs there are, capped at the slice count) */
        if (th <= 0) {
            long nc = sysconf(_SC_NPROCESSORS_ONLN);
            th = (nc > 0 && nc < 64) ? (int)nc : 8;
        }
        if (th > e->n_slices) th = e->n_slices;
        if (th < 1) th = 1;
        e->n_threads = th;
    }
    e->mb_encs = (mbenc *)aeg_alloc((size_t)e->mb_count * sizeof(mbenc));
    e->seed_mv = (int16_t *)aeg_alloc((size_t)e->mb_count * 2 * sizeof(int16_t));
    e->seed_cand = (int16_t *)aeg_alloc((size_t)e->mb_count * ME_ROUND_CAND * 4 *
                                        sizeof(int16_t));
    e->prev_seed = (int16_t *)aeg_alloc((size_t)e->mb_count * 2 * sizeof(int16_t));
    e->seed_sad = (int32_t *)aeg_alloc((size_t)e->mb_count * ME_ROUND_CAND *
                                       sizeof(int32_t));
    e->seed_cost = (int32_t *)aeg_alloc((size_t)e->mb_count * sizeof(int32_t));
    e->par_pos = (int16_t *)aeg_alloc((size_t)e->mb_count * ME_ROUND_CAND * 4 *
                                      sizeof(int16_t));
    e->par_mv = (int16_t *)aeg_alloc((size_t)e->mb_count * 2 * sizeof(int16_t));
    e->par_cost = (int32_t *)aeg_alloc((size_t)e->mb_count * sizeof(int32_t));
    if (e->seed_mv)
        memset(e->seed_mv, 0, (size_t)e->mb_count * 2 * sizeof(int16_t));
    if (e->prev_seed)
        memset(e->prev_seed, 0, (size_t)e->mb_count * 2 * sizeof(int16_t));
    if (e->par_mv)
        memset(e->par_mv, 0, (size_t)e->mb_count * 2 * sizeof(int16_t));
    if (!e->seed_mv || !e->seed_cand || !e->seed_sad || !e->seed_cost ||
        !e->prev_seed || !e->par_pos || !e->par_mv || !e->par_cost) {
        aeg_set_last_error("out of memory");
        aeg_encoder_close(e);
        return NULL;
    }
    e->mb_dqp = (int8_t *)aeg_alloc((size_t)e->mb_count);
    if (!e->mb_dqp) {
        aeg_set_last_error("out of memory");
        aeg_encoder_close(e);
        return NULL;
    }
    e->mbs = (aeg_mb *)aeg_alloc(sizeof(aeg_mb) * (size_t)e->mb_count);
    if (!e->mbs) {
        aeg_set_last_error("out of memory");
        aeg_encoder_close(e);
        return NULL;
    }
    e->pic.w[0] = e->w; e->pic.h[0] = e->h;
    e->pic.w[1] = e->pic.w[2] = e->cw;
    e->pic.h[1] = e->pic.h[2] = e->ch;
    e->pic.mb_w = e->mb_w;
    e->pic.mb_h = e->mb_h;
    e->pic.stride[0] = e->stride[0];
    e->pic.stride[1] = e->pic.stride[2] = e->stride[1];
    e->pic.plane[0] = e->recp[0];
    e->pic.plane[1] = e->recp[1];
    e->pic.plane[2] = e->recp[2];

    aeg_dispatch_init(&e->d, e->p.backend, &e->p);
    /* Both halves of the coefficient pipeline have to be on the device: an
     * inter candidate cannot transform on one and reconstruct on the other
     * without a round trip in the middle of its RD cost.  AEG_COEF_BATCH=1/0
     * forces the choice, which is how the CPU reference runs the batched code
     * path in the conformance tests (it has to produce the same bitstream). */
    {
        /* -1 = ask the dispatcher, 0/1 = forced.  The effective flags are
         * re-resolved at the top of every picture, because the dispatcher only
         * settles a stage after it has measured it (and a -s override only
         * takes effect from the first aeg_dispatch_update() on). */
        const char *ev = getenv("AEG_COEF_BATCH");
        e->coef_force = ev ? atoi(ev) != 0 : -1;
        ev = getenv("AEG_RD_BATCH");
        e->rd_force = ev ? atoi(ev) != 0 : -1;
        resolve_coef_stages(e);
    }
    aeg_log(1, "aeg %s: %dx%d, %dx%d macroblocks, qp %d, gop %d",
            AEG_VERSION_STRING, e->w, e->h, e->mb_w, e->mb_h, e->p.qp, e->p.gop_size);
    return e;
}

int aeg_encoder_encode(aeg_encoder *enc, const uint8_t *yuv420p, size_t size,
                       int64_t pts, aeg_packet *pkt)
{
    struct aeg_encoder *e = (struct aeg_encoder *)enc;
    size_t need;
    int is_idr;

    if (!e || !yuv420p)
        return AEG_EINVAL;
    need = (size_t)e->w * e->h + 2 * (size_t)e->cw * e->ch;
    if (size < need) {
        aeg_set_last_error("input buffer too small");
        return AEG_EINVAL;
    }
    if (!e->frame_count) {
        /* The reference samples of the top/left picture edge do not exist;
         * filling the padding with the mid grey keeps the prediction of the
         * first macroblock defined (and matches the DC-128 fallback). */
        int p, y;
        for (p = 0; p < 3; p++) {
            int pw = p ? e->cw : e->w, ph = p ? e->ch : e->h;
            memset(e->recp[p] - AEG_PAD * e->stride[p] - AEG_PAD, 128,
                   (size_t)e->stride[p] * AEG_PAD);
            for (y = 0; y < ph + AEG_PAD; y++)
                memset(e->recp[p] + y * e->stride[p] - AEG_PAD, 128, AEG_PAD);
            (void)pw;
        }
    }
    copy_plane_pad(e->srcp[0], e->stride[0], yuv420p, e->w, e->w, e->h);
    copy_plane_pad(e->srcp[1], e->stride[1], yuv420p + (size_t)e->w * e->h,
                   e->cw, e->cw, e->ch);
    copy_plane_pad(e->srcp[2], e->stride[2],
                   yuv420p + (size_t)e->w * e->h + (size_t)e->cw * e->ch,
                   e->cw, e->cw, e->ch);

    is_idr = e->frame_count == 0 ||
             (e->p.gop_size > 0 && (e->frame_count % e->p.gop_size) == 0);
    if (is_scene_cut(e, is_idr)) {
        is_idr = 1;
        e->scene_cuts++;
        aeg_log(2, "aeg: scene cut at frame %d -> IDR", e->frame_count);
    }
    if (is_idr)
        e->last_idr_frame = e->frame_count;
    return encode_frame(e, is_idr, pts, pkt);
}

int aeg_encoder_flush(aeg_encoder *enc, aeg_packet *pkt)
{
    (void)enc;
    if (pkt)
        pkt->data = NULL, pkt->size = 0;
    return AEG_EOF;
}

void aeg_packet_free(aeg_packet *pkt)
{
    if (pkt)
        pkt->data = NULL, pkt->size = 0;
}

int aeg_encoder_stats(aeg_encoder *enc, aeg_stats *out)
{
    struct aeg_encoder *e = (struct aeg_encoder *)enc;
    int i;
    if (!e || !out)
        return AEG_EINVAL;
    *out = e->st;
    out->vbv_min_fill_bits = e->vbv_active ? e->vbv_min_full : 0.0;
    out->vbv_underflows = e->vbv_underflows;
    out->vbv_size_clamps = e->vbv_clamps;
    out->scene_cuts = e->scene_cuts;
    for (i = 0; i < AEG_STAGE_COUNT; i++) {
        out->stage_use_gpu[i] = e->d.use_gpu[i];
        out->stage_cpu_ns[i] = e->d.cpu_ns[i][1];
        out->stage_gpu_ns[i] = e->d.gpu_ns[i][1];
    }
    return AEG_OK;
}

int aeg_encoder_recon(aeg_encoder *enc, const uint8_t *plane[3], int stride[3],
                      int mb_qp[])
{
    struct aeg_encoder *e = (struct aeg_encoder *)enc;
    int i;
    if (!e)
        return AEG_EINVAL;
    for (i = 0; i < 3; i++) {
        if (plane)
            plane[i] = e->refp[i];      /* the frame that was just encoded */
        if (stride)
            stride[i] = e->stride[i];
    }
    if (mb_qp) {
        int n = e->mb_count;
        for (i = 0; i < n; i++)
            mb_qp[i] = e->mbs[i].qp;
    }
    return AEG_OK;
}

int aeg_encoder_set_qp_offset(aeg_encoder *enc, int offset)
{
    struct aeg_encoder *e = (struct aeg_encoder *)enc;

    if (!e)
        return AEG_EINVAL;
    if (offset < -12) offset = -12;
    if (offset >  12) offset =  12;
    e->rc_qp_offset = offset;
    return AEG_OK;
}

int aeg_encoder_set_qp(aeg_encoder *enc, int qp)
{
    struct aeg_encoder *e = (struct aeg_encoder *)enc;
    if (!e || qp < 0 || qp > 51)
        return AEG_EINVAL;
    e->p.qp = qp;
    return AEG_OK;
}

void aeg_encoder_close(aeg_encoder *enc)
{
    struct aeg_encoder *e = (struct aeg_encoder *)enc;
    if (!e)
        return;
    aeg_dispatch_destroy(&e->d);
    aeg_mc_ctx_free(&e->mc);
    aeg_free(e->mb_encs);
    aeg_free(e->seed_mv);
    aeg_free(e->seed_cand);
    aeg_free(e->prev_seed);
    aeg_free(e->seed_sad);
    aeg_free(e->seed_cost);
    aeg_free(e->par_pos);
    aeg_free(e->par_mv);
    aeg_free(e->par_cost);
    aeg_free(e->mb_dqp);
    aeg_free(e->mbs);
    aeg_free(e->src_mem);
    aeg_free(e->rec_mem);
    aeg_free(e->ref_mem);
    {
        int i;
        for (i = 0; i < AEG_MAX_SLICES; i++)
            aeg_free(e->slice_bs[i]);
    }
    aeg_free(e->out);
    aeg_free(e);
}
