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
    uint8_t  intra4[16];            /* Intra_4x4 modes, Figure 6-8 order    */
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
    /* Inter partitioning candidates gathered by analyse_mb(): 16x16, 16x8,
     * 8x16 and P_8x8.  Each carries the per-partition MVs and their
     * predictors, which is exactly what the bitstream has to transmit. */
    mbcand mb_cand[4];
    int mb_ncand;

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
    int cur_is_p;           /* the picture being coded is a P slice */
};

static inline int clip255(int v)
{
    return v < 0 ? 0 : (v > 255 ? 255 : v);
}

/* ------------------------------------------------------------------ */
/* Inter partition geometry                                            */
/* ------------------------------------------------------------------ */
static int is_intra_type(int t)
{
    return t == AEG_MB_I4x4 || t == AEG_MB_I16x16 || t == AEG_MB_IPCM;
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
            luma_cbp |= 1 << blk_group(blk);
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
    } else if (mby > 0) {
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
            if (mby > 0)
                nb_from_mb(&C, &e->mbs[(mby - 1) * e->mb_w + mbx], 12 + cx);
            else
                nb_from_mb(&C, NULL, 0);
        } else {
            if (mby > 0 && mbx + 1 < e->mb_w)
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
            } else if (bx > 0 && mby > 0) {
                nb_from_mb(&C, &e->mbs[(mby - 1) * e->mb_w + mbx], 12 + bx - 1);
            } else if (mby > 0 && mbx > 0) {
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
    if (mbx <= 0 || mby <= 0)
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
 * 2 = vertical, 3 = plane.  Only the modes whose references exist are legal. */
static int chroma_mode_ok(int cm, int mbx, int mby)
{
    if ((cm == 1 || cm == 3) && mbx == 0)
        return 0;
    if ((cm == 2 || cm == 3) && mby == 0)
        return 0;
    return 1;
}

static int luma_mode_ok(int mode, int mbx, int mby)
{
    if ((mode == AEG_MB_I_V || mode == AEG_MB_I_P) && mby == 0)
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
        if (!chroma_mode_ok(cm, mbx, mby))
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

/* Motion search for one partition: its own search window, centred on its own
 * MV predictor. */
static void me_partition(struct aeg_encoder *e, int mbx, int mby, int type,
                         int p, const int geo[4], int pmvx, int pmvy,
                         int *mvx, int *mvy, int *sad)
{
    aeg_me_block(e->refp[0] + (mby * 16 + geo[1]) * e->stride[0] +
                     mbx * 16 + geo[0],
                 e->stride[0],
                 e->srcp[0] + (mby * 16 + geo[1]) * e->stride[0] +
                     mbx * 16 + geo[0],
                 e->stride[0], geo[2], geo[3],
                 e->p.search_range ? e->p.search_range : 16,
                 e->p.subme, pmvx, pmvy, mvx, mvy, sad);
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
        me_partition(e, mbx, mby, type, p, geo, pmvx, pmvy, &mvx, &mvy, &sad);
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
            int geo[4][4], b0, n = sub_cnt[s], sad_sum = 0, bits, mv[4][2],
                pmv[4][2];
            memcpy(trial, cur, sizeof(trial));
            memcpy(tavail, avail, sizeof(tavail));
            for (k = 0; k < n; k++) {
                int sad;
                sub_geom(s, bx, by, k, geo[k], &b0);
                predict_mv_at(e, mbx, mby, (const int16_t (*)[2])trial,
                              tavail, AEG_MB_P8x8, 0, geo[k][0] / 4,
                              geo[k][1] / 4, geo[k][2] / 4, &pmv[k][0],
                              &pmv[k][1]);
                me_partition(e, mbx, mby, AEG_MB_P8x8, k, geo[k], pmv[k][0],
                             pmv[k][1], &mv[k][0], &mv[k][1], &sad);
                sad_sum += sad;
                me_publish(trial, geo[k], mv[k][0], mv[k][1]);
                avail_publish(tavail, geo[k]);
            }
            bits = aeg_ue_bits((uint32_t)s);
            for (k = 0; k < n; k++)
                bits += aeg_mvd_bits(mv[k][0] - pmv[k][0],
                                     mv[k][1] - pmv[k][1]);
            /* 8 SAD units per bit, the same trade the preset-0 decision in
             * analyse_mb() makes; the final RD pass has the last word. */
            if (sad_sum + 8 * bits < best_cost) {
                best_cost = sad_sum + 8 * bits;
                best_st = s;
                memcpy(best_geo, geo, sizeof(geo));
                memcpy(best_mv, mv, sizeof(mv));
                memcpy(best_pmv, pmv, sizeof(pmv));
            }
        }
        /* commit the winning shape; its predictors were derived from the state
         * before this sub-macroblock, which is exactly what the decoder sees */
        cand->sub_type[sub] = (uint8_t)best_st;
        for (k = 0; k < sub_cnt[best_st]; k++) {
            int b0;
            sub_geom(best_st, bx, by, k, best_geo[k], &b0);
            cand_publish(cand, best_geo[k], best_mv[k][0], best_mv[k][1]);
            cand->pmv[b0][0] = best_pmv[k][0];
            cand->pmv[b0][1] = best_pmv[k][1];
            me_publish(cur, best_geo[k], best_mv[k][0], best_mv[k][1]);
            avail_publish(avail, best_geo[k]);
            cand->nparts++;
        }
    }
    e->mb_ncand = slot + 1;
}

/* Fills e->mb_modes[] with the legal Intra_16x16 modes ordered by SAD, runs
 * the motion search and remembers its result.  Nothing is coded here: code_mb()
 * either uses this SAD ranking directly (preset 0) or refines it with a full
 * rate-distortion search. */
static void analyse_mb(struct aeg_encoder *e, int mbx, int mby, mbenc *c)
{
    int mode, n = 0, sad_of[4];

    for (mode = 0; mode < 4; mode++)
        sad_of[mode] = luma_mode_ok(mode, mbx, mby)
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
    /* The motion search needs a reference picture *and* a P slice: an IDR
     * picture is always an I slice, so inter candidates there could never be
     * coded (writing them as an I macroblock desynchronises the decoder). */
    if (e->p.partitions != 0 && c->qp >= 0 && e->has_ref && e->cur_is_p) {
        int mvx, mvy, sad;
        aeg_mv_predict(e->mbs, e->mb_w, e->mb_w, mbx, mby,
                       &e->mb_pmv[0], &e->mb_pmv[1]);
        /* The block to match is the *source* macroblock.  recp still holds the
         * previous picture at this position (nothing has been reconstructed
         * here yet), so searching against it would always favour mv = 0. */
        aeg_me_block(e->refp[0] + mby * 16 * e->stride[0] + mbx * 16,
                     e->stride[0],
                     e->srcp[0] + mby * 16 * e->stride[0] + mbx * 16,
                     e->stride[0], 16, 16,
                     e->p.search_range ? e->p.search_range : 16,
                     e->p.subme, e->mb_pmv[0], e->mb_pmv[1], &mvx, &mvy, &sad);
        e->mb_mv[0] = mvx;
        e->mb_mv[1] = mvy;
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
static int mb_type_bits(const mbenc *c, int is_p)
{
    if (c->type == AEG_MB_I4x4)
        return aeg_ue_bits(is_p ? 5u : 0u);
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
static int inter_hdr_bits(const mbenc *c)
{
    int p, i, bits = mb_type_bits(c, 1);
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
        a = m->type == AEG_MB_I4x4 ? m->intra4[intra4_idx(3, by)] : 2;
    } else {
        a = -1;
    }
    if (by > 0) {
        b = cur[intra4_idx(bx, by - 1)];
    } else if (mby > 0) {
        const aeg_mb *m = &e->mbs[(mby - 1) * e->mb_w + mbx];
        b = m->type == AEG_MB_I4x4 ? m->intra4[intra4_idx(bx, 3)] : 2;
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
        int top_avail = mby > 0 || by > 0;
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
            tr_avail = mby > 0 && (bx < 12 || mbx + 1 < e->mb_w);
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
            /* The mode costs one bit when it matches the predictor and four
             * otherwise, so weigh it into the SAD comparison. */
            cost = 2 * aeg_satd_4x4(pred, 4, sb, s) +
                   bits_scale * (m == pmode ? 1 : 4);
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
        code_luma_inter(e, mbx, mby, t);
        t->bits += inter_hdr_bits(t);
    } else {
        t->nparts = 1;
        intra16_sad(e, mbx, mby, i16_mode);   /* leaves the predictor in pred_l */
        code_luma_i16(e, mbx, mby, t);
    }
    t->cost = t->sse + (int64_t)(lambda * t->bits + 0.5);
    return t->cost;
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
    uint8_t snap_l[16 * 16], snap_c[2 * 8 * 8];
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
    for (i = 0; i < e->mb_ncand; i++) {
        int64_t cost = rd_luma(e, mbx, mby, &t, c->qp, e->mb_cand[i].type, 0,
                               lambda, i);
        if (cost < best_cost) {
            best_cost = cost;
            best = t;
            snap_luma(e, mbx, mby, snap_l);
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
            if (!chroma_mode_ok(cm, mbx, mby))
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
                     const mbenc *c, int is_p)
{
    int blk;
    if (!is_p) {
        if (c->type == AEG_MB_I4x4)
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
    } else if (c->type == AEG_MB_I4x4) {
        aeg_bw_ue(bw, 5);               /* P slice: I_NxN */
    } else {
        aeg_bw_ue(bw, (uint32_t)(6 + c->i16_mode + 4 * chroma_cbp(c) +
                                 12 * (luma_cbp_on(c) ? 1 : 0)));
    }
    if (c->type == AEG_MB_I4x4) {
        int b;
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
    e->cur_is_p = is_p;

    for (mby = 0; mby < e->mb_h; mby++) {
        for (mbx = 0; mbx < e->mb_w; mbx++) {
            {
                mbenc tmp;
                int64_t t0 = aeg_now_ns();
                reset_mb_enc(&tmp, e->p.qp);
                analyse_mb(e, mbx, mby, &tmp);
                e->st.stage_ns[AEG_STAGE_MOTION_EST] += aeg_now_ns() - t0;

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
                    for (i = 0; i < 16; i++)
                        m->intra4[i] = tmp.intra4[i];
                    for (i = 0; i < AEG_BLK_COUNT; i++)
                        m->nz[i] = tmp.nz[i];
                    for (i = 0; i < 16; i++) {
                        m->mv[i][0] = tmp.mv[i][0];
                        m->mv[i][1] = tmp.mv[i][1];
                    }
                    m->cost = (int32_t)tmp.sse;
                }

                /* emit.  A P_L0_16x16 macroblock whose motion vector equals the
                 * predictor and that carries no residual at all reconstructs
                 * exactly like P_Skip (same prediction, no residual), so it is
                 * coded as part of the skip run: same picture, fewer bits. */
                {
                    int skipped = 0;
                    if (is_p && tmp.type == AEG_MB_P16x16 && tmp.cbp == 0) {
                        int pmvx, pmvy;
                        pskip_mv(e, mbx, mby, &pmvx, &pmvy);
                        if (tmp.mv[0][0] == pmvx && tmp.mv[0][1] == pmvy)
                            skipped = 1;
                    }
                    if (skipped)
                        skip_run++;
                    else {
                        if (is_p)
                            flush_skip_run(&bw, &skip_run);
                        write_mb(e, &bw, mbx, mby, &tmp, is_p);
                    }
                }
                e->st.frames += 0;
            }
        }
    }
    /* A skip run that reaches the end of the frame still has to be emitted:
     * the decoder counts macroblocks by consuming mb_skip_run values, so an
     * unwritten tail run makes it stop before the last row. */
    if (is_p && skip_run > 0)
        flush_skip_run(&bw, &skip_run);
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
