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
#include "backend.h"

#define AEG_LAMBDA(qp) (0.85 * pow(2.0, ((qp) - 12) / 3.0))

typedef struct mbenc {
    int16_t  level[AEG_BLK_COUNT][16];
    int8_t   nz[AEG_BLK_COUNT];
    int16_t  mv[16][2];             /* per 4x4 block, 1/4 pel               */
    int      type;                  /* AEG_MB_*                             */
    int      i16_mode;              /* 0=V 1=H 2=DC 3=Plane                 */
    int      chroma_mode;           /* 0=DC 1=H 2=V 3=Plane                 */
    int      cbp;
    int      qp;
    int      bits;                  /* payload bits (residual + headers)    */
    int      hdr_bits;
    int64_t  sse;
    int64_t  cost;
} mbenc;

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

    aeg_dispatcher d;
    int frame_count;         /* total pictures encoded                  */
    int frame_num;           /* frame_num of the current picture (8.2.1) */
    int idr_pic_id;
    int last_qp;

    uint8_t *bs;    size_t bs_cap;
    uint8_t *nal;   size_t nal_cap;
    uint8_t *out;   size_t out_cap;

    aeg_stats st;

    int16_t mv_batch[2 * 256];
    int32_t sad_batch[256];
    uint8_t pred_l[16 * 16];
    uint8_t pred_c[2][8 * 8];
    int has_ref;
};

static inline int clip255(int v)
{
    return v < 0 ? 0 : (v > 255 ? 255 : v);
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
    plane[0] = *mem + AEG_PAD * ls + AEG_PAD;
    plane[1] = *mem + ls * lh + AEG_PAD * cs + AEG_PAD;
    plane[2] = plane[1] + cs * ch;
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
    for (y = 1; y <= AEG_PAD; y++) {
        memcpy(dst - y * dstride, dst, (size_t)(w + 2 * AEG_PAD));
        memcpy(dst + (h - 1 + y) * dstride, dst + (h - 1) * dstride,
               (size_t)(w + 2 * AEG_PAD));
    }
    (void)x;
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
    return mby ? e->mbs[(mby - 1) * e->mb_w + mbx].nz[blk] : 0;
}

static int luma_nc(struct aeg_encoder *e, int mbx, int mby, const mbenc *c, int blk)
{
    int bx = blk & 3, by = blk >> 2;
    int a = bx ? c->nz[blk - 1] : left_nz(e, mbx, mby, blk + 3);
    int b = by ? c->nz[blk - 4] : top_nz(e, mbx, mby, blk + 12);
    return aeg_cavlc_nc(a, b, mbx > 0 || bx > 0, mby > 0 || by > 0);
}

static int chroma_nc(struct aeg_encoder *e, int mbx, int mby, const mbenc *c,
                     int pl, int blk)
{
    int base = AEG_BLK_CHROMA_CB + pl * 4;
    int bx = blk & 1, by = blk >> 1;
    int a = bx ? c->nz[base + blk - 1] : left_nz(e, mbx, mby, base + blk + 1);
    int b = by ? c->nz[base + blk - 2] : top_nz(e, mbx, mby, base + blk + 2);
    return aeg_cavlc_nc(a, b, mbx > 0 || bx > 0, mby > 0 || by > 0);
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
    return aeg_cavlc_nc(a, b, mbx > 0 || bx > 0, mby > 0 || by > 0);
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

/* ------------------------------------------------------------------ */
/* Luma coding: inter                                                  */
/* ------------------------------------------------------------------ */
static void code_luma_inter(struct aeg_encoder *e, int mbx, int mby, mbenc *c)
{
    const uint8_t *src = e->srcp[0] + mby * 16 * e->stride[0] + mbx * 16;
    uint8_t *dst = e->recp[0] + mby * 16 * e->stride[0] + mbx * 16;
    const uint8_t *pred = e->pred_l;
    int blk, i, x, y, luma_cbp = 0;
    int qp = c->qp;

    for (blk = 0; blk < 16; blk++) {
        int bx = (blk & 3) * 4, by = (blk >> 2) * 4;
        int16_t res[16], coef[16];
        for (y = 0; y < 4; y++)
            for (x = 0; x < 4; x++)
                res[y * 4 + x] = (int16_t)((int)src[(by + y) * e->stride[0] + bx + x] -
                                           (int)pred[(by + y) * 16 + bx + x]);
        aeg_transform_4x4(res, 4, coef);
        aeg_quant_4x4(coef, c->level[blk], qp, 0);
    }
    for (blk = 0; blk < 16; blk++) {
        int bx = (blk & 3) * 4, by = (blk >> 2) * 4;
        uint8_t *r = dst + by * e->stride[0] + bx;
        const uint8_t *pr = pred + by * 16 + bx;
        int nz = 0;
        for (i = 0; i < 16; i++)
            nz += c->level[blk][i] != 0;
        c->nz[blk] = (int8_t)nz;
        if (nz)
            luma_cbp |= 1 << (blk >> 2);
        for (y = 0; y < 4; y++)
            for (x = 0; x < 4; x++)
                r[y * e->stride[0] + x] = pr[y * 16 + x];
        aeg_idct_add_block(r, e->stride[0], c->level[blk], qp, r, e->stride[0], NULL);
        for (y = 0; y < 4; y++)
            for (x = 0; x < 4; x++) {
                int d = (int)r[y * e->stride[0] + x] -
                        (int)src[(by + y) * e->stride[0] + bx + x];
                c->sse += (int64_t)d * d;
            }
    }
    c->cbp = (c->cbp & 0x30) | luma_cbp;
    for (blk = 0; blk < 16; blk++)
        if (c->nz[blk])
            c->bits += aeg_cavlc_block_cost(c->level[blk],
                                            luma_nc(e, mbx, mby, c, blk), 0);
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
    int avail_l = mbx > 0, avail_t = mby > 0;

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
                            mby > 0, mbx > 0, mby > 0 && mbx > 0, mode);
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
                             mby > 0, mbx > 0, mby > 0 && mbx > 0, mode);
    return aeg_sad(8, 8, e->pred_c[pl], 8, src, e->stride[1]);
}

/* Inter prediction of the 16x16 luma block and both chroma planes. */
static void inter_predict(struct aeg_encoder *e, int mbx, int mby, int mvx, int mvy)
{
    const uint8_t *r = e->refp[0] + mby * 16 * e->stride[0] + mbx * 16;
    int pl;
    aeg_mc_luma(e->pred_l, 16, r, e->stride[0], 16, 16, mvx & 3, mvy & 3);
    for (pl = 0; pl < 2; pl++) {
        const uint8_t *rc = e->refp[1 + pl] + mby * 8 * e->stride[1] + mbx * 8;
        aeg_mc_chroma(e->pred_c[pl], 8, rc, e->stride[1], 8, 8,
                      (mvx >> 2) & 7, (mvy >> 2) & 7);
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
}

static void analyse_mb(struct aeg_encoder *e, int mbx, int mby, mbenc *c)
{
    int mode, best_mode = 2, best_sad = 1 << 30;
    int qp = c->qp;

    /* luma intra 16x16: pick the prediction with the smallest SAD */
    for (mode = 0; mode < 4; mode++) {
        int sad;
        /* Only predict from samples that exist: vertical needs the row above,
         * horizontal the column on the left, plane both plus the corner.  A
         * macroblock on the top/left edge therefore has fewer candidates, and
         * the one on the top-left corner only DC. */
        if ((mode == AEG_MB_I_V || mode == AEG_MB_I_P) && !mby)
            continue;
        if ((mode == AEG_MB_I_H || mode == AEG_MB_I_P) && !mbx)
            continue;
        sad = intra16_sad(e, mbx, mby, mode);
        if (sad < best_sad) { best_sad = sad; best_mode = mode; }
    }
    c->i16_mode = best_mode;

    /* chroma: the same for both planes (one mode for Cb and Cr) */
    {
        int cm, cbest = 0, csad = 1 << 30;
        for (cm = 0; cm < 4; cm++) {
            int s = 0;
            /* same availability rules as luma (1 = HOR, 2 = VERT, 3 = PLANE) */
            if ((cm == 1 || cm == 3) && !mbx)
                continue;
            if ((cm == 2 || cm == 3) && !mby)
                continue;
            s += chroma_sad(e, mbx, mby, 0, cm);
            s += chroma_sad(e, mbx, mby, 1, cm);
            if (s < csad) { csad = s; cbest = cm; }
        }
        c->chroma_mode = cbest;
    }

    /* leave the luma prediction of the chosen 16x16 mode in pred_l */
    intra16_sad(e, mbx, mby, best_mode);
    if (e->p.partitions == 0 || qp < 0)
        return;

    if (e->p.gop_size && e->has_ref) {
        int pmvx, pmvy, mvx, mvy, sad;
        aeg_mv_predict(e->mbs, e->mb_w, mbx, mby, &pmvx, &pmvy);
        aeg_me_block(e->refp[0] + mby * 16 * e->stride[0] + mbx * 16, e->stride[0],
                     e->recp[0] + mby * 16 * e->stride[0] + mbx * 16,
                     e->stride[0], 16, 16,
                     e->p.search_range ? e->p.search_range : 16,
                     e->p.subme, pmvx, pmvy, &mvx, &mvy, &sad);
        (void)pmvx;
        {
            /* compare inter against the best intra candidate */
            int inter_cost = sad * 8 + aeg_mvd_bits((mvx - pmvx) >> 2, (mvy - pmvy) >> 2);
            int intra_cost = best_sad * 8;
            if (inter_cost < intra_cost) {
                int i;
                c->type = AEG_MB_P16x16;
                c->mv[0][0] = c->mv[1][0] = c->mv[2][0] = c->mv[3][0] = (int16_t)mvx;
                c->mv[0][1] = c->mv[1][1] = c->mv[2][1] = c->mv[3][1] = (int16_t)mvy;
                for (i = 4; i < 16; i++) {
                    c->mv[i][0] = (int16_t)mvx;
                    c->mv[i][1] = (int16_t)mvy;
                }
                inter_predict(e, mbx, mby, mvx, mvy);
            }
        }
        /* chroma prediction of the chosen chroma mode is recomputed by
         * code_chroma(); for an inter block the chroma follows the luma
         * motion vector. */
        if (c->type == AEG_MB_P16x16) {
            int cm = c->chroma_mode;
            (void)cm;
        }
    }
}

static void code_mb(struct aeg_encoder *e, int mbx, int mby, mbenc *c)
{
    if (c->type == AEG_MB_P16x16) {
        code_luma_inter(e, mbx, mby, c);
        code_chroma(e, mbx, mby, c, 1);
    } else {
        code_luma_i16(e, mbx, mby, c);
        code_chroma(e, mbx, mby, c, 0);
    }
}

/* ------------------------------------------------------------------ */
/* Bitstream                                                           */
/* ------------------------------------------------------------------ */
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

static int mb_type_bits(const mbenc *c, int is_p) __attribute__((unused));
static int mb_type_bits(const mbenc *c, int is_p)
{
    int t;
    if (!is_p)
        return aeg_ue_bits((uint32_t)(1 + c->i16_mode + 4 * chroma_cbp(c) +
                                      12 * (luma_cbp_on(c) ? 1 : 0)));
    if (c->type == AEG_MB_P16x16)
        t = 0;
    else
        t = 6 + c->i16_mode + 4 * chroma_cbp(c) + 12 * (luma_cbp_on(c) ? 1 : 0);
    return aeg_ue_bits((uint32_t)t);
}

static void write_mb(struct aeg_encoder *e, aeg_bitwriter *bw, int mbx, int mby,
                     const mbenc *c, int is_p)
{
    int blk;
    if (!is_p) {
        aeg_bw_ue(bw, (uint32_t)(1 + c->i16_mode + 4 * chroma_cbp(c) +
                                 12 * (luma_cbp_on(c) ? 1 : 0)));
    } else if (c->type == AEG_MB_P16x16) {
        int pmvx, pmvy;
        aeg_bw_ue(bw, 0);
        aeg_mv_predict(e->mbs, e->mb_w, mbx, mby, &pmvx, &pmvy);
        aeg_bw_se(bw, c->mv[0][0] - pmvx);
        aeg_bw_se(bw, c->mv[0][1] - pmvy);
        aeg_bw_ue(bw, (uint32_t)aeg_cbp_code_num[(c->cbp & 0x0f) |
                                                 (chroma_cbp(c) << 4)]);
    } else {
        aeg_bw_ue(bw, (uint32_t)(6 + c->i16_mode + 4 * chroma_cbp(c) +
                                 12 * (luma_cbp_on(c) ? 1 : 0)));
    }
    if (c->type != AEG_MB_P16x16)
        aeg_bw_ue(bw, (uint32_t)c->chroma_mode);
    /* mb_qp_delta */
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
    } else {
        for (blk = 0; blk < 16; blk++)
            if (c->nz[blk])
                aeg_cavlc_write_block(bw, c->level[blk],
                                      luma_nc(e, mbx, mby, c, blk), 0);
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

static int encode_frame(struct aeg_encoder *e, int is_idr, int64_t pts,
                        aeg_packet *pkt)
{
    int mbx, mby;
    int is_p = !is_idr && e->has_ref;
    size_t rbsp_len, out_len;
    aeg_bitwriter bw;
    int skip_run = 0;
    int64_t t_all = aeg_now_ns();

    if (ensure_buf(&e->bs, &e->bs_cap, (size_t)e->mb_count * 400 + 16384) != AEG_OK)
        return AEG_ENOMEM;

    aeg_bw_init(&bw, e->bs, e->bs_cap, 0);
    /* frame_num counts the pictures since the last IDR (H.264 8.2.1): an IDR
     * always carries 0, otherwise the decoder invents "non-existing" frames to
     * fill the gap. */
    if (is_idr)
        e->frame_num = 0;
    aeg_write_slice_header(&bw, is_p ? 5 : 7,
                           e->frame_num & ((1 << e->frame_num_bits) - 1),
                           e->frame_num_bits, is_idr, e->idr_pic_id,
                           e->p.qp - 26, 0, 0,
                           e->p.deblock ? 0 : 1);
    e->last_qp = e->p.qp;

    for (mby = 0; mby < e->mb_h; mby++) {
        for (mbx = 0; mbx < e->mb_w; mbx++) {
            {
                mbenc tmp;
                int64_t t0 = aeg_now_ns();
                reset_mb_enc(&tmp, e->p.qp);
                analyse_mb(e, mbx, mby, &tmp);
                e->st.stage_ns[AEG_STAGE_MOTION_EST] += aeg_now_ns() - t0;

                if (is_p && tmp.type == AEG_MB_P16x16) {
                    int pmvx, pmvy;
                    aeg_mv_predict(e->mbs, e->mb_w, mbx, mby, &pmvx, &pmvy);
                    (void)pmvx; (void)pmvy;
                }
                t0 = aeg_now_ns();
                code_mb(e, mbx, mby, &tmp);
                e->st.stage_ns[AEG_STAGE_TRANSFORM] += aeg_now_ns() - t0;

                /* store the decisions for the neighbour queries */
                {
                    aeg_mb *m = &e->mbs[mby * e->mb_w + mbx];
                    int i;
                    memset(m, 0, sizeof(*m));
                    m->type = (uint8_t)tmp.type;
                    m->i16_mode = (uint8_t)tmp.i16_mode;
                    m->chroma_mode = (uint8_t)tmp.chroma_mode;
                    m->cbp = (uint8_t)tmp.cbp;
                    m->qp = (int8_t)tmp.qp;
                    for (i = 0; i < AEG_BLK_COUNT; i++)
                        m->nz[i] = tmp.nz[i];
                    for (i = 0; i < 16; i++) {
                        m->mv[i][0] = tmp.mv[i][0];
                        m->mv[i][1] = tmp.mv[i][1];
                    }
                    m->cost = (int32_t)tmp.sse;
                }

                /* emit */
                {
                    if (is_p)
                        flush_skip_run(&bw, &skip_run);
                    write_mb(e, &bw, mbx, mby, &tmp, is_p);
                }
                e->st.frames += 0;
            }
        }
    }
    aeg_bw_align_one(&bw);
    rbsp_len = aeg_bw_finish(&bw);

    if (ensure_buf(&e->nal, &e->nal_cap, rbsp_len * 3 + 64) != AEG_OK ||
        ensure_buf(&e->out, &e->out_cap, rbsp_len * 3 + 512) != AEG_OK)
        return AEG_ENOMEM;

    out_len = 0;
    if (is_idr) {
        uint8_t sps[256], pps[64];
        size_t sl = aeg_write_sps(sps, sizeof(sps), &e->p, e->mb_w, e->mb_h,
                                  e->frame_num_bits, e->level_idc);
        size_t pl = aeg_write_pps(pps, sizeof(pps));
        out_len += aeg_nal_write(e->out + out_len, e->out_cap - out_len, 3, 7,
                                 sps, sl, 1);
        out_len += aeg_nal_write(e->out + out_len, e->out_cap - out_len, 3, 8,
                                 pps, pl, 0);
    }
    out_len += aeg_nal_write(e->out + out_len, e->out_cap - out_len,
                             is_idr ? 3 : 2, is_idr ? 5 : 1, e->bs, rbsp_len, 0);

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
        pkt->qp = e->p.qp;
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
    if (pp->subme < 0) pp->subme = 1;
    if (pp->gop_size < 0) pp->gop_size = 250;

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
    e->last_qp = e->p.qp;

    if (enc_alloc_picture(e, &e->src_mem, e->srcp) != AEG_OK ||
        enc_alloc_picture(e, &e->rec_mem, e->recp) != AEG_OK ||
        enc_alloc_picture(e, &e->ref_mem, e->refp) != AEG_OK) {
        aeg_set_last_error("out of memory");
        aeg_free(e);
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
    aeg_free(e->mbs);
    aeg_free(e->src_mem);
    aeg_free(e->rec_mem);
    aeg_free(e->ref_mem);
    aeg_free(e->bs);
    aeg_free(e->nal);
    aeg_free(e->out);
    aeg_free(e);
}
