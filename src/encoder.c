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
    int16_t  mv[16][2];             /* per 4x4 block, 1/4 pel, list 0        */
    uint8_t  ref[16];              /* P list0 reference index per 4x4 (0/1) */
    /* B pictures: the list 1 vector of each 4x4 block and, per partition, which
     * lists predict it (1 = list 0, 2 = list 1, 3 = both). */
    int16_t  mv1[16][2];
    int16_t  pmv1[16][2];
    uint8_t  blist[4];              /* lists per partition (B_8x8: per 8x8) */
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
/* Candidates per macroblock: a P picture has the searched P16x16, the skip
 * vector and the three partitioned types (16x8, 8x16, P_8x8); a B picture has
 * the three 16x16 list choices, the two partitioned shapes, B_8x8 and direct.
 * The batched RD path stores one residual per candidate, so this is also its
 * array bound -- seven for the B list. */
#define AEG_MB_CAND_MAX 7

/* Upper bound on the slices a picture may be cut into.  Slice threads use one
 * thread per slice, so this also bounds the thread count. */
#define AEG_MAX_SLICES 64

/* Upper bound on the B pictures between two anchors.  Three is what x264 calls
 * "bframes 3" and what the pyramid would need; more than that buys very little
 * and costs lookahead memory. */
#define AEG_MAX_BFRAMES 3
typedef struct {
    int     type;
    int     nparts;
    uint8_t sub_type[4];
    int     mv[16][2];
    int     mv1[16][2];
    int     pmv[16][2];
    int     pmv1[16][2];
    uint8_t blist[4];
    /* P list0 reference index per 4x4 block (0 or 1).  B pictures keep 0. */
    uint8_t ref[16];
} mbcand;

struct aeg_encoder {
    aeg_params p;
    int w, h, cw, ch;
    int mb_w, mb_h, mb_count;
    int stride[3];
    int frame_num_bits, level_idc;

    uint8_t *src_mem;
    uint8_t *srcp[3];
    /* Three picture buffers, because a B picture needs two references alive at
     * once (the anchor before it and the anchor after it) *and* somewhere to
     * reconstruct itself.  The roles are buffer indices rather than pointers so
     * that the rotation below is one assignment per role; bind_pictures() puts
     * the plane pointers where the rest of the encoder expects them. */
    uint8_t *pic_mem[3];
    uint8_t *pic_plane[3][3];
    int role_cur;            /* buffer being reconstructed            */
    int role_last;           /* buffer the last coded picture went to  */
    int role_ref0;           /* oldest reference picture (list 0 of B) */
    int role_ref1;           /* newest reference picture (list 1 of B) */
    int pic_poc[3];          /* picture order count held by each buffer   */
    int nrefs;               /* reference pictures alive, 0..2         */
    uint8_t *recp[3], *refp[3], *ref2p[3];

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
    int mb_frac1[2];        /* the same, for list 1 of a B picture        */
    int me_lambda;          /* motion-search lambda of the current MB
                               (aeg_me_lambda(qp)), 0 = distortion only   */
    /* Inter partitioning candidates gathered by analyse_mb(): 16x16, 16x8,
     * 8x16 and P_8x8.  Each carries the per-partition MVs and their
     * predictors, which is exactly what the bitstream has to transmit. */
    mbcand mb_cand[AEG_MB_CAND_MAX];
    int mb_ncand;

    aeg_dispatcher d;
    int frame_count;         /* total pictures encoded                  */
    int last_idr_frame;      /* frame_count of the last IDR picture     */
    int scene_cuts;          /* IDR pictures inserted by scenecut       */
    /* Coding statistics: how many macroblocks were coded and how many of them
     * as a skip.  A skip is the cheapest macroblock the standard has, so the
     * fraction is the quickest check that the decision pass is doing its job
     * on still content (it is also what the pskip_still case asserts). */
    int64_t stat_mbs;
    int64_t stat_skipped;
    int64_t stat_type[AEG_MB_TYPE_COUNT];
    int frame_num;           /* frame_num of the current picture (8.2.1) */
    int idr_pic_id;
    int last_qp;
    /* CABAC state that outlives a macroblock: whether the previous one carried
     * a non-zero mb_qp_delta (the context of the next one's), and the
     * cabac_init_idc the slices of this picture are coded with. */
    int last_dqp;
    int cabac_init_idc;
    int dbg_cabac;
    /* The decisions are made one pass before the coder runs, so the rate-
     * distortion pass cannot ask the coder what it will charge.  It keeps a
     * shadow of the coder instead: initialized from the slice's QP, advanced
     * once per chosen macroblock (so the context states are the ones the
     * bitstream pass will have), and copied for every candidate to price what
     * that candidate would cost (see rd_cabac_bits()).  `rd_last_qp` and
     * `rd_last_dqp` are the shadow's copy of the two values that cross a
     * macroblock boundary. */
    aeg_cabac cab_shadow;
    int rd_last_qp;
    int rd_last_dqp;

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
    /* The second list's predictor, and the pair averaged in place, for a
     * bi-predictive macroblock of a B picture. */
    uint8_t pred_l2[16 * 16];
    uint8_t pred_c2[2][8 * 8];
    aeg_mc_ctx mc1;         /* fractional planes of ref2p[0] (list 1) */
    int has_ref;
    int has_ref2;
    /* P slice with two active list0 entries: the newest and the oldest
     * reference are both in list0 (H.264 7.4.3: num_ref_idx_active_override).
     * B keeps one per list. */
    int cur_l0_active;
    int cur_is_p;           /* the picture being coded is a P slice */
    int cur_is_b;           /* ... is a B slice                     */
    int cur_is_ref;         /* ... is a reference picture           */
    int cur_poc;            /* its picture order count              */
    int poc_lsb_bits;       /* log2_max_pic_order_cnt_lsb           */
    int prev_poc;           /* POC of the previous picture, decode order */
    int ref_poc[2];         /* POC of the two references, oldest first   */
    int col_ready;          /* the co-located picture's decisions are usable */
    aeg_mb *col_mbs;        /* ... and they live here (see b_direct_mv)  */
    int col_poc;            /* POC of co-located picture */
    int col_ref_poc[2];     /* POC of its two L0 references (if any) */
    /* Lookahead: input pictures are held in display order until the anchors
     * they are coded between have arrived (see aeg_encoder_encode()). */
    uint8_t *in_mem[AEG_MAX_BFRAMES + 3];
    uint8_t *in_plane[AEG_MAX_BFRAMES + 3][3];
    int64_t *in_pts;
    aeg_lowres in_low[AEG_MAX_BFRAMES + 3];   /* b_adapt: lowres of the ring */
    int *la_scratch;
    int in_count;           /* pictures pushed so far        */
    int enc_pos;            /* pictures coded so far         */
    int anchor;             /* display index of the last coded anchor, -1 */
    int last_idr_poc;       /* display index of the last IDR picture      */
    /* The pictures of the next coding step, in the order they go into the
     * bitstream (never more than one group's worth: the next anchor and the B
     * pictures before it). */
    int job_d[AEG_MAX_BFRAMES + 4];
    int job_slice[AEG_MAX_BFRAMES + 4];
    int job_idr[AEG_MAX_BFRAMES + 4];
    int job_n;
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
    double rc_target_sum;   /* the budget so far, all pictures               */
    double rc_b_ratio;      /* measured B-picture cost relative to an anchor */
    int    rc_anchor_bits;  /* the last anchor's size, for the ratio above   */
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
    /* The previous *anchor*, for the VBV model: a B picture is much cheaper
     * than the anchor before it and much cheaper than the one after it, so
     * predicting a B picture from whatever picture happened to come last swings
     * between over- and under-estimating by a factor of several, and the clamp
     * would then fire on a buffer that is nowhere near empty. */
    int    last_anchor_bits;
    int    last_anchor_qp;
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

/* Which kind of slice a macroblock is being written for.  The three differ in
 * the macroblock layer, not only in the header: P and B both have a skip flag
 * (at different context offsets) and a per-list prediction, I has neither. */
enum { AEG_SLICE_I = 0, AEG_SLICE_P = 1, AEG_SLICE_B = 2 };

/* A macroblock of a B slice: its prediction comes from two lists instead of
 * one, and its mb_type carries which list each partition uses. */
static int is_b_type(int t)
{
    return t == AEG_MB_B_DIRECT || t == AEG_MB_B_L0 || t == AEG_MB_B_L1 ||
           t == AEG_MB_B_BI || t == AEG_MB_B16x8 || t == AEG_MB_B8x16 ||
           t == AEG_MB_B_8x8;
}

/* The lists macroblock `m` predicts from, as a mask (1 = list 0, 2 = list 1). */
static int mb_lists(int type, const uint8_t blist[4])
{
    if (is_intra_type(type))
        return 0;
    if (!is_b_type(type))
        return 1;                      /* a P macroblock only has list 0 */
    {
        int m = blist[0] | blist[1] |
                ((type == AEG_MB_B_8x8) ? (blist[2] | blist[3]) : 0);
        return m ? m : 3;
    }
}

/* Number of partitions of an inter macroblock type. */
static int part_count(int type)
{
    if (type == AEG_MB_P8x8 || type == AEG_MB_B_8x8)
        return 4;
    return (type == AEG_MB_P16x8 || type == AEG_MB_P8x16 ||
            type == AEG_MB_B16x8 || type == AEG_MB_B8x16) ? 2 : 1;
}

/* Luma geometry of partition `p`: x, y, w, h in pixels inside the macroblock,
 * plus the index of its first 4x4 block (where its MV is stored). */
static void part_geom(int type, int p, int geo[4], int *blk0)
{
    switch (type) {
    case AEG_MB_P16x8:
    case AEG_MB_B16x8:
        geo[0] = 0; geo[1] = p ? 8 : 0; geo[2] = 16; geo[3] = 8;
        *blk0 = p ? 8 : 0;
        break;
    case AEG_MB_P8x16:
    case AEG_MB_B8x16:
        geo[0] = p ? 8 : 0; geo[1] = 0; geo[2] = 8; geo[3] = 16;
        *blk0 = p ? 2 : 0;
        break;
    case AEG_MB_P8x8:
    case AEG_MB_B_8x8:
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

/* B_8x8 sub_mb_type (table 7-14): one entry per value of the CABAC tree, in
 * the order the tree numbers them.  0 is B_Direct_8x8, which derives both
 * vectors from the co-located picture and sends no mvd; 1/2 are the 8x8 list 0
 * and list 1 predictions, 3 the bi-predictive one; 4..12 are the same three
 * lists crossed with the 8x4, 4x8 and 4x4 splits.  The partition geometry is
 * `sub_geom()` of the shape each entry names. */
static const uint8_t b_sub_shape[13]  = { 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 3, 3, 3 };
static const uint8_t b_sub_lists[13]  = { 3, 1, 2, 3, 1, 1, 2, 2, 3, 3, 1, 2, 3 };
static const uint8_t b_sub_cnt[13]    = { 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 4, 4, 4 };
static const uint8_t b_sub_direct[13] = { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

/* The 8x8 group that holds 4x4 block `b0` (raster order). */
static int b8_group(int b0)
{
    return (((b0 >> 2) >= 2) << 1) + ((b0 & 3) >= 2);
}

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
    if (c->type != AEG_MB_P8x8 && c->type != AEG_MB_B_8x8)
        return part_count(c->type);
    for (i = 0; i < 4; i++)
        n += (c->type == AEG_MB_B_8x8) ? b_sub_cnt[c->sub_type[i]]
                                       : sub_cnt[c->sub_type[i]];
    return n;
}

static void mb_part_geom(const mbenc *c, int p, int geo[4], int *b0)
{
    const int b8 = c->type == AEG_MB_B_8x8;
    int i;
    if (c->type != AEG_MB_P8x8 && !b8) {
        part_geom(c->type, p, geo, b0);
        return;
    }
    for (i = 0; i < 4 && p >= 0; i++) {
        int st = c->sub_type[i];
        int n = b8 ? b_sub_cnt[st] : sub_cnt[st];
        if (p < n) {
            sub_geom(b8 ? b_sub_shape[st] : st, (i & 1) * 2, (i >> 1) * 2, p,
                     geo, b0);
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

/* Point recp/refp/ref2p at the three buffers' current roles.  Every other
 * function in this file reads those names and never the roles, so the only
 * place that has to know which buffer is which is this one. */
static void bind_pictures(struct aeg_encoder *e)
{
    int i;

    for (i = 0; i < 3; i++) {
        e->recp[i] = e->role_cur >= 0 ? e->pic_plane[e->role_cur][i] : NULL;
        e->refp[i] = e->role_ref0 >= 0 ? e->pic_plane[e->role_ref0][i] : NULL;
        e->ref2p[i] = e->role_ref1 >= 0 ? e->pic_plane[e->role_ref1][i] : NULL;
    }
    e->has_ref = e->nrefs > 0;
    e->has_ref2 = e->nrefs > 1;
}

/* A buffer no role holds. */
static int pic_free(const struct aeg_encoder *e)
{
    int i;
    for (i = 0; i < 3; i++)
        if (i != e->role_cur && i != e->role_ref0 && i != e->role_ref1)
            return i;
    return -1;
}

/* The picture just coded becomes a reference, the oldest one is dropped when
 * the buffer would otherwise overflow (the decoder's sliding window does the
 * same, which is why the SPS asks for two reference frames with B pictures),
 * and the buffer it frees is where the next picture is reconstructed.
 *
 * A B picture is not a reference picture (nal_ref_idc = 0), so this does
 * nothing for it: its buffer is simply reused by the next picture. */
static void picture_rotate(struct aeg_encoder *e, int is_ref, int is_idr)
{
    int coded = e->role_cur;

    e->role_last = coded;

    if (!is_ref) {
        bind_pictures(e);
        return;
    }
    /* An IDR picture empties the decoder's buffer before it is stored, so it
     * is the *only* reference afterwards.  With one reference per P picture
     * nothing could tell the difference, but a P picture with two would take
     * the picture before the IDR as list0[1] -- one the decoder no longer has
     * (it conceals it with another one, silently). */
    if (is_idr) {
        e->role_ref0 = coded;
        e->role_ref1 = -1;
        e->nrefs = 1;
        e->role_cur = -1;
        e->role_cur = pic_free(e);
        bind_pictures(e);
        return;
    }
    if (e->nrefs == 0) {
        e->role_ref0 = coded;
        e->nrefs = 1;
        e->role_cur = pic_free(e);
    } else if (e->nrefs == 1) {
        e->role_ref1 = coded;
        e->nrefs = 2;
        e->role_cur = pic_free(e);
    } else {
        /* both references are alive: the oldest one's buffer is reused */
        e->role_cur = e->role_ref0;
        e->role_ref0 = e->role_ref1;
        e->role_ref1 = coded;
    }
    bind_pictures(e);
}

/* Which reference picture each list points at for the picture about to be
 * coded.  list0[0] is the nearest reference picture in display order that comes
 * *before* this one and list1[0] the nearest that comes after it; with one
 * reference picture per anchor direction that is exactly the older and the
 * newer of the two references still alive.  A P picture only uses list 0, and
 * its nearest-past reference is always the newer of the two.  With two active
 * list0 entries (P multi-reference) the older one is list0[1] and lives in
 * ref2p for the motion compensation to find. */
static int p_should_use_2refs(const struct aeg_encoder *e)
{
    /* `refs` is resolved by aeg_encoder_open() (2 from preset 3 up on P-only
     * streams, see there); the first P picture after an IDR only has one
     * reference to offer. */
    return e->cur_is_p && e->nrefs > 1 && e->p.refs >= 2;
}
static void set_current_refs(struct aeg_encoder *e)
{
    int list0 = e->cur_is_b ? e->role_ref0 : (e->nrefs > 1 ? e->role_ref1
                                                          : e->role_ref0);
    int list1 = e->cur_is_b ? e->role_ref1 : -1;
    int i;

    /* P with two list0 entries: list0[0] = newest (role_ref1), list0[1] =
     * oldest (role_ref0).  No reordering syntax is needed: the decoder's
     * default list0 is decreasing frame_num, which is exactly [newest, oldest]
     * for two sliding-window references, so num_ref_idx_active_override is all
     * that has to be sent. */
    if (p_should_use_2refs(e) && e->nrefs > 1) {
        int l0a = e->role_ref1, l0b = e->role_ref0;
        for (i = 0; i < 3; i++) {
            e->refp[i] = e->pic_plane[l0a][i];
            e->ref2p[i] = e->pic_plane[l0b][i];
        }
        e->ref_poc[0] = e->pic_poc[l0a];
        e->ref_poc[1] = e->pic_poc[l0b];
        e->has_ref = 1;
        e->has_ref2 = 1;               /* second list0 entry lives in ref2p */
        e->cur_l0_active = 2;
        return;
    }
    for (i = 0; i < 3; i++) {
        e->refp[i] = list0 >= 0 ? e->pic_plane[list0][i] : NULL;
        e->ref2p[i] = list1 >= 0 ? e->pic_plane[list1][i] : NULL;
    }
    e->ref_poc[0] = list0 >= 0 ? e->pic_poc[list0] : 0;
    e->ref_poc[1] = list1 >= 0 ? e->pic_poc[list1] : 0;
    e->has_ref = list0 >= 0;
    e->has_ref2 = list1 >= 0;
    e->cur_l0_active = 1;
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

static void trellis_i16_ac(struct aeg_encoder *e, int mbx, int mby, mbenc *c,
                           int blk, const uint8_t *src, int stride,
                           const uint8_t *pred, int qp, const int32_t *dcv);

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
        if (e->p.trellis > 0 && e->p.cabac)
            trellis_i16_ac(e, mbx, mby, c, blk, src + by * e->stride[0] + bx,
                           e->stride[0], pr, qp, &dcv);
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

static int cab_cbf_ctx(int cat, int a, int b, int va, int vb, int intra);
static void cab_nz4_luma(struct aeg_encoder *e, int mbx, int mby, const mbenc *c,
                         int blk, int *a, int *b, int *va, int *vb);
static void cab_nz4_chroma(struct aeg_encoder *e, int mbx, int mby,
                           const mbenc *c, int pl, int k,
                           int *a, int *b, int *va, int *vb);

/* ------------------------------------------------------------------ */
/* Rate-distortion level selection                                     */
/* ------------------------------------------------------------------ */
/* The quantiser rounds each coefficient into a deadzone and that is the end of
 * it: nothing asks whether the bits a level costs are worth the distortion it
 * removes, which is why the deadzone is widened on purpose (aeg_quant_bias(),
 * 1.5x the rounding point) -- rounding has no way to answer the question.
 *
 * This pass answers it, block by block, with the price the CABAC coder really
 * charges (aeg_cabac_residual() run in price-only mode against the slice's
 * shadow state) and with the distortion the reconstruction really has (inverse
 * transform, clip, SSE against the source), not a coefficient-domain estimate
 * of it.
 *
 * The search is a coordinate descent over the levels: start from the levels
 * the quantiser chose, consider emptying the block, then sweep the
 * coefficients -- last to first, the order the coder sends them in -- and try
 * each level one step towards zero and one step away from it, keeping a move
 * whenever it lowers SSE + lambda * bits.  A second sweep lets the moves
 * combine, which is the whole point: a level dropped here only pays for itself
 * together with a level raised there, and no per-coefficient rule can see that.
 */

/* Price of one coefficient list in 1/256 bit, from the shadow coder. */
static int64_t cabac_block_price(const aeg_cabac *st, const int16_t *scan,
                                 int n, int cat, int cbf_ctx)
{
    aeg_cabac cb = *st;

    cb.cost = 1;
    cb.cost256 = 0;
    aeg_cabac_residual(&cb, scan, n, cat, cbf_ctx);
    return (int64_t)cb.cost256;
}

/* What a block of levels reconstructs to, and what that costs. */
struct trel {
    int side;                 /* 4 or 8 */
    int n;                    /* coefficients in the block */
    const uint8_t *scan;      /* scan position -> natural (raster) index */
    int cat;                  /* ctxBlockCat for the price */
    int cbf_ctx;              /* coded_block_flag context, -1 when there is none */
    const int32_t *dc;        /* AC blocks: the dequantised DC, else NULL */
};

static int64_t trel_sse(const struct trel *t, const uint8_t *src, int stride,
                        const uint8_t *pred, int pred_stride,
                        const int16_t *lev, int qp)
{
    int64_t sse = 0;
    int x, y;

    if (t->side == 4) {
        uint8_t r[16];
        aeg_idct_add_block(r, 4, lev, qp, pred, pred_stride, t->dc);
        for (y = 0; y < 4; y++)
            for (x = 0; x < 4; x++) {
                int d = (int)r[y * 4 + x] - (int)src[y * stride + x];
                sse += (int64_t)d * d;
            }
    } else {
        uint8_t r[64];
        aeg_idct_8x8_add(r, 8, lev, qp, pred, pred_stride);
        for (y = 0; y < 8; y++)
            for (x = 0; x < 8; x++) {
                int d = (int)r[y * 8 + x] - (int)src[y * stride + x];
                sse += (int64_t)d * d;
            }
    }
    return sse;
}

/* What one coefficient's change does to the price of the block, with the levels
 * the search currently holds in `cur`.  Only the coefficient itself and -- when
 * it is the last significant one and the change empties it -- the new last
 * coefficient cost anything, which is what makes the sweep cheap. */
static int64_t trel_dbits(const aeg_cabac *sh, const struct trel *t,
                          const int16_t *cur, int idx, int newlev, int last)
{
    int pos = t->scan[idx];
    int oldlev = cur[pos];
    int sig_last = idx == last;
    int64_t d = aeg_cabac_coef_price(sh, t->cat, t->n, idx, newlev,
                                     sig_last && newlev != 0, 1) -
                aeg_cabac_coef_price(sh, t->cat, t->n, idx, oldlev,
                                     sig_last, 1);

    if (sig_last && newlev == 0) {
        /* The block's last coefficient moves back: the next one has to say so. */
        int j = idx - 1;
        while (j >= 0 && !cur[t->scan[j]])
            j--;
        if (j >= 0)
            d += aeg_cabac_coef_price(sh, t->cat, t->n, j, cur[t->scan[j]], 1,
                                      0) -
                 aeg_cabac_coef_price(sh, t->cat, t->n, j, cur[t->scan[j]], 0,
                                      0);
    }
    return d;
}

/* Bits to distortion: the price the coder gave in 1/256 bit, converted with
 * the same lambda the mode decisions exchange bits against SSE with. */
static int64_t trel_price(int64_t bits256, double lambda)
{
    return (int64_t)(lambda * (double)bits256 / 256.0 + 0.5);
}

/* The sweep itself: returns the cost of the levels it settles on, which are
 * written back through `lev`.  The moving cost is kept locally -- one inverse
 * transform and one local price per candidate -- and the block that wins is
 * priced exactly before it is accepted, so the local price can only cost
 * choices, never correctness. */
static int64_t trel_descent(const aeg_cabac *sh, const struct trel *t,
                            int16_t *lev, const uint8_t *src, int stride,
                            const uint8_t *pred, int pred_stride, int qp,
                            double lambda, int passes, int allow_empty)
{
    int16_t cur[64], trial[64], best[64], scan[64];
    int i, pass, nnz = 0, last = -1;
    int64_t bits, sse, cost, best_cost, orig_cost;

    for (i = 0; i < t->n; i++) {
        cur[i] = lev[i];
        if (cur[i]) {
            nnz++;
            last = i;
        }
    }
    if (!nnz)
        return 0;
    for (i = 0; i < t->n; i++)
        scan[i] = cur[t->scan[i]];
    bits = cabac_block_price(sh, scan, t->n, t->cat, t->cbf_ctx);
    sse = trel_sse(t, src, stride, pred, pred_stride, cur, qp);
    orig_cost = best_cost = cost = sse + trel_price(bits, lambda);
    memcpy(best, cur, sizeof(int16_t) * t->n);

    /* Emptying a block only pays when there is almost nothing in it: the flag
     * it saves is one bin, the residual it throws away is up to sixteen. */
    if (allow_empty && nnz <= 2) {
        for (i = 0; i < t->n; i++)
            trial[i] = 0;
        for (i = 0; i < t->n; i++)
            scan[i] = 0;
        bits = cabac_block_price(sh, scan, t->n, t->cat, t->cbf_ctx);
        sse = trel_sse(t, src, stride, pred, pred_stride, trial, qp);
        cost = sse + trel_price(bits, lambda);
        if (cost < best_cost) {
            best_cost = cost;
            memcpy(cur, trial, sizeof(int16_t) * t->n);
            memcpy(best, trial, sizeof(int16_t) * t->n);
            last = -1;
        } else {
            /* reprice the original, its bits were overwritten above */
            for (i = 0; i < t->n; i++)
                scan[i] = cur[t->scan[i]];
            bits = cabac_block_price(sh, scan, t->n, t->cat, t->cbf_ctx);
            sse = trel_sse(t, src, stride, pred, pred_stride, cur, qp);
            cost = sse + trel_price(bits, lambda);
        }
    }
    for (pass = 0; pass < passes; pass++) {
        for (i = 0; i < t->n; i++) {
            int pos = t->scan[i], step;
            if (!cur[pos])
                continue;
            for (step = -1; step <= 1; step += 2) {
                int newlev = (int)cur[pos] + step;
                int last2 = last;
                int64_t dbits, sse2, c;

                memcpy(trial, cur, sizeof(int16_t) * t->n);
                trial[pos] = (int16_t)newlev;
                if (i == last && !newlev) {
                    int j = i - 1;
                    while (j >= 0 && !cur[t->scan[j]])
                        j--;
                    last2 = j;
                }
                dbits = trel_dbits(sh, t, cur, i, newlev, last);
                sse2 = trel_sse(t, src, stride, pred, pred_stride, trial, qp);
                c = sse2 + trel_price(bits + dbits, lambda);
                if (c < cost) {
                    cost = c;
                    sse = sse2;
                    bits += dbits;
                    last = last2;
                    cur[pos] = trial[pos];
                    if (c < best_cost) {
                        best_cost = c;
                        memcpy(best, cur, sizeof(int16_t) * t->n);
                    }
                }
            }
        }
    }
    /* The final word is the exact price of the winner: if the local price
     * talked the sweep into something worse than what the quantiser chose,
     * the quantiser's levels stay. */
    if (memcmp(best, lev, sizeof(int16_t) * t->n)) {
        for (i = 0; i < t->n; i++)
            scan[i] = best[t->scan[i]];
        bits = cabac_block_price(sh, scan, t->n, t->cat, t->cbf_ctx);
        sse = trel_sse(t, src, stride, pred, pred_stride, best, qp);
        if (sse + trel_price(bits, lambda) < orig_cost)
            memcpy(lev, best, sizeof(int16_t) * t->n);
    }
    return best_cost;
}

/* Whether the intra blocks (Intra_4x4, Intra_8x8) get the level selection
 * too.  AEG_TINTRA=0/1 overrides, for measurements. */
static int trellis_intra_mode(void)
{
    static int v = -1;
    if (v < 0) {
        const char *ev = getenv("AEG_TINTRA");
        v = ev ? atoi(ev) : 3;
    }
    return v;
}
#define trellis_intra_on() (trellis_intra_mode() == 1 || \
    (trellis_intra_mode() == 2 && (e->cur_is_p || e->cur_is_b)) || \
    trellis_intra_mode() == 3)
#define trellis_intra_lambda(l) (trellis_intra_mode() == 3 ? (l) * 0.5 : (l))

/* Whether the chroma AC blocks get the level selection.  AEG_TCHROMA. */
static int trellis_chroma_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *ev = getenv("AEG_TCHROMA");
        v = ev ? atoi(ev) != 0 : 1;
    }
    return v;
}

/* The AC levels of one Intra_16x16 block (cat 1) against the DC that is
 * already decided; raster order is also the order the flags' neighbours
 * become final in.  AEG_TI16=0 turns it off. */
static void trellis_i16_ac(struct aeg_encoder *e, int mbx, int mby, mbenc *c,
                           int blk, const uint8_t *src, int stride,
                           const uint8_t *pred, int qp, const int32_t *dcv)
{
    static int on = -1;
    struct trel t;
    int a, b, va, vb;

    if (on < 0) {
        const char *ev = getenv("AEG_TI16");
        on = ev ? atoi(ev) != 0 : 1;
    }
    if (!on)
        return;
    cab_nz4_luma(e, mbx, mby, c, blk, &a, &b, &va, &vb);
    t.side = 4;
    t.n = 15;
    t.scan = aeg_scan4x4 + 1;
    t.cat = 1;
    t.cbf_ctx = cab_cbf_ctx(1, a, b, va, vb, 1);
    t.dc = dcv;
    trel_descent(&e->cab_shadow, &t, c->level[blk], src, stride, pred, 16, qp,
                 trellis_intra_lambda(AEG_LAMBDA(qp)), 2, 1);
}

/* One macroblock's luma 4x4 blocks of an inter macroblock, whose prediction is
 * in e->pred_l and whose levels are `lev[block][natural]`. */
static void trellis_luma4(struct aeg_encoder *e, int mbx, int mby, mbenc *c,
                          int qp, int16_t lev[][16], double lambda)
{
    const uint8_t *src = e->srcp[0] + mby * 16 * e->stride[0] + mbx * 16;
    const int s = e->stride[0];
    int blk;

    if (!e->p.cabac || e->p.trellis <= 0)
        return;
    /* The coded_block_flag context is a property of the neighbours, and the
     * neighbours of a block inside this macroblock are its earlier blocks, so
     * the counts are settled before any level moves. */
    for (blk = 0; blk < 16; blk++) {
        int i, n = 0;
        for (i = 0; i < 16; i++)
            n += lev[blk][i] != 0;
        c->nz[blk] = (int8_t)n;
    }
    for (blk = 0; blk < 16; blk++) {
        struct trel t;
        int a, b, va, vb;

        cab_nz4_luma(e, mbx, mby, c, blk, &a, &b, &va, &vb);
        t.side = 4;
        t.n = 16;
        t.scan = aeg_scan4x4;
        t.cat = 2;
        t.cbf_ctx = cab_cbf_ctx(2, a, b, va, vb, 0);
        t.dc = NULL;
        trel_descent(&e->cab_shadow, &t, lev[blk],
                     src + (blk >> 2) * 4 * s + (blk & 3) * 4, s,
                     e->pred_l + (blk >> 2) * 4 * 16 + (blk & 3) * 4, 16,
                     qp, lambda, 2, 1);
    }
}

/* One 8x8 luma block of an inter macroblock.  There is no coded_block_flag
 * here -- the coded block pattern of the macroblock says whether the group is
 * coded -- so the price starts at the significance map. */
static void trellis_luma8(struct aeg_encoder *e, const uint8_t *src, int stride,
                          const uint8_t *pred, int pred_stride, int16_t *lev,
                          int qp, double lambda)
{
    struct trel t;

    if (!e->p.cabac || e->p.trellis <= 0)
        return;
    t.side = 8;
    t.n = 64;
    t.scan = aeg_zigzag8x8;
    t.cat = 5;
    t.cbf_ctx = -1;
    t.dc = NULL;
    trel_descent(&e->cab_shadow, &t, lev, src, stride, pred, pred_stride, qp,
                 lambda, 1, 0);
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
        if (e->p.trellis > 0 && e->p.cabac) {
            /* The levels of the batch are in the same layout as the scalar
             * path's, one block after the other in natural order, so the same
             * selection runs on them -- and the reconstruction has to be
             * rebuilt afterwards, because it was made from the old levels. */
            trellis_luma4(e, mbx, mby, c, qp, (int16_t (*)[16])e->coef_lev,
                          AEG_LAMBDA(qp));
            t0 = aeg_now_ns();
            e->d.ops.iq_idct_batch(e->d.ops.priv, e->coef_lev, 16, qp,
                                   e->coef_rec);
            aeg_dispatch_record(&e->d, AEG_STAGE_IQUANT_IDCT,
                                e->d.ops.backend != AEG_BACKEND_CPU,
                                aeg_now_ns() - t0);
            luma_inter_finish(e, mbx, mby, c, e->coef_lev, NULL,
                              e->coef_rec, pred);
            return;
        }
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
    trellis_luma4(e, mbx, mby, c, qp, c->level, AEG_LAMBDA(qp));
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
    if (!intra || trellis_intra_on())
        trellis_luma8(e, src + by * s + bx, s, pred, pred_stride, c->level8[g],
                      c->qp, intra ? trellis_intra_lambda(AEG_LAMBDA(c->qp))
                                   : AEG_LAMBDA(c->qp));
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

/* The coded value of transform_size_8x8_flag, i.e. the one the neighbours'
 * context (399 + transform_size_8x8_flag of A + of B) sees.  The flag is only
 * present in the syntax for Intra_NxN and for an inter macroblock with a luma
 * pattern, so every other macroblock leaves it zero even if its residual was
 * transformed in 8x8 blocks internally. */
static int mb_t8_coded(const struct aeg_encoder *e, const mbenc *c, int skipped)
{
    if (skipped || !e->p.transform_8x8)
        return 0;
    if (c->type == AEG_MB_I4x4 || c->type == AEG_MB_I8x8)
        return c->type == AEG_MB_I8x8;
    if (!is_inter_type(c->type) || !t8_allowed(e, c) || !(c->cbp & 15))
        return 0;
    return c->t8 ? 1 : 0;
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

            if (e->p.trellis > 0 && e->p.cabac && trellis_chroma_on()) {
                /* the AC levels, against the DC that is already decided; the
                 * earlier blocks of the plane are final, so the flag's
                 * context is the one the coder will use */
                struct trel t;
                int a, b, va, vb;
                double lam = AEG_LAMBDA(c->qp);   /* 0.6x and 1.5x: worse */

                cab_nz4_chroma(e, mbx, mby, c, pl, blk, &a, &b, &va, &vb);
                t.side = 4;
                t.n = 15;
                t.scan = aeg_scan4x4 + 1;
                t.cat = 4;
                t.cbf_ctx = cab_cbf_ctx(4, a, b, va, vb, !inter);
                t.dc = &dcv;
                trel_descent(&e->cab_shadow, &t,
                             c->level[AEG_BLK_CHROMA_CB + pl * 4 + blk],
                             src + bxy[blk][1] * e->stride[1] + bxy[blk][0],
                             e->stride[1], pr, 8, qpc,
                             inter ? lam : trellis_intra_lambda(lam), 2, 1);
            }
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
/* Motion compensation of one rectangle from reference `list` into `dl` (luma,
 * stride 16) or `dc[0..1]` (chroma, stride 8).  A B picture predicts from two
 * references, so the reference picture, its precomputed fractional planes and
 * the destination all have to be selectable: the second list's predictor has to
 * be built somewhere other than on top of the first one, because the two are
 * averaged afterwards. */
static void mc_rect_luma_to(struct aeg_encoder *e, int list, uint8_t *dl,
                            int mbx, int mby, int x, int y, int w, int h,
                            int mvx, int mvy)
{
    const uint8_t *ref = list ? e->ref2p[0] : e->refp[0];
    const aeg_mc_ctx *mc = list ? &e->mc1 : &e->mc;
    const uint8_t *r = ref + (mby * 16 + y + (mvy >> 2)) * e->stride[0] +
                       mbx * 16 + x + (mvx >> 2);
    if (mc->ready)
        aeg_mc_luma_p(dl + y * 16 + x, 16, mc, r, w, h, mvx & 3, mvy & 3);
    else
        aeg_mc_luma(dl + y * 16 + x, 16, r, e->stride[0], w, h,
                    mvx & 3, mvy & 3);
}

static void mc_rect_chroma_to(struct aeg_encoder *e, int list, uint8_t *dc[2],
                              int mbx, int mby, int cx, int cy, int cw, int ch,
                              int mvx, int mvy)
{
    int pl;
    for (pl = 0; pl < 2; pl++) {
        const uint8_t *rc = (list ? e->ref2p[1 + pl] : e->refp[1 + pl]) +
                            (mby * 8 + cy + (mvy >> 3)) * e->stride[1] +
                            mbx * 8 + cx + (mvx >> 3);
        aeg_mc_chroma(dc[pl] + cy * 8 + cx, 8, rc, e->stride[1],
                      cw, ch, mvx & 7, mvy & 7);
    }
}

static void mc_rect_luma(struct aeg_encoder *e, int mbx, int mby,
                         int x, int y, int w, int h, int mvx, int mvy)
{
    mc_rect_luma_to(e, 0, e->pred_l, mbx, mby, x, y, w, h, mvx, mvy);
}

static void mc_rect_chroma(struct aeg_encoder *e, int mbx, int mby,
                           int cx, int cy, int cw, int ch, int mvx, int mvy)
{
    uint8_t *dc[2] = { e->pred_c[0], e->pred_c[1] };
    mc_rect_chroma_to(e, 0, dc, mbx, mby, cx, cy, cw, ch, mvx, mvy);
}

/* Inter prediction of the 16x16 luma block and both chroma planes. */
static void inter_predict(struct aeg_encoder *e, int mbx, int mby, int mvx, int mvy)
{
    mc_rect_luma(e, mbx, mby, 0, 0, 16, 16, mvx, mvy);
    mc_rect_chroma(e, mbx, mby, 0, 0, 8, 8, mvx, mvy);
}

/* Prediction of a whole P macroblock coded with `type`, using the per-4x4 MV
 * table of that candidate (every 4x4 block of a partition shares its MV). */
static void p_predict_typed(struct aeg_encoder *e, int mbx, int mby,
                            const mbenc *c)
{
    int p, n = mb_part_count(c);
    for (p = 0; p < n; p++) {
        int geo[4], b0, ref = 0;
        mb_part_geom(c, p, geo, &b0);
        if (e->cur_l0_active > 1)
            ref = c->ref[b0] & 1;
        mc_rect_luma_to(e, ref, e->pred_l, mbx, mby, geo[0], geo[1], geo[2],
                        geo[3], c->mv[b0][0], c->mv[b0][1]);
        mc_rect_chroma_to(e, ref, (uint8_t *[]){e->pred_c[0], e->pred_c[1]}, mbx,
                          mby, geo[0] / 2, geo[1] / 2, geo[2] / 2, geo[3] / 2,
                          c->mv[b0][0], c->mv[b0][1]);
    }
}

/* Which lists predict each partition of a B macroblock.  The partitioned
 * 16x8/8x16 shapes carry one mask per partition; B_8x8 carries one per 8x8
 * group, and every sub-partition of a group uses the group's lists. */
static int mb_part_lists(const mbenc *c, int p)
{
    int m;

    if (c->type == AEG_MB_B_8x8) {
        int geo[4], b0;

        mb_part_geom(c, p, geo, &b0);
        m = c->blist[b8_group(b0)];
    } else {
        m = c->blist[p < 2 ? p : 1];
    }
    return m ? m : 3;
}

/* Does partition `p` carry an mvd?  A B_8x8 group coded as B_Direct_8x8 does
 * not -- the decoder derives its vectors from the co-located picture -- and
 * neither does a whole B_Direct_16x16 macroblock. */
static int mb_part_sends_mvd(const mbenc *c, int p)
{
    if (c->type == AEG_MB_B_DIRECT)
        return 0;
    if (c->type == AEG_MB_B_8x8) {
        int geo[4], b0;

        mb_part_geom(c, p, geo, &b0);
        return !b_sub_direct[c->sub_type[b8_group(b0)]];
    }
    return 1;
}

/* Prediction of a B macroblock: per partition, the lists in `blist` are
 * predicted and, when both are, averaged with a single rounding -- that is the
 * bi-predictive sample the standard defines (8.4.2.3.1), and rounding the two
 * predictors separately would drift from the decoder on every such block. */
static void b_predict_typed(struct aeg_encoder *e, int mbx, int mby,
                            const mbenc *c)
{
    int p, n = mb_part_count(c);
    for (p = 0; p < n; p++) {
        int geo[4], b0, lists = mb_part_lists(c, p);
        int x, y;
        mb_part_geom(c, p, geo, &b0);
        if (lists & 1)
            mc_rect_luma_to(e, 0, e->pred_l, mbx, mby, geo[0], geo[1],
                            geo[2], geo[3], c->mv[b0][0], c->mv[b0][1]);
        if (lists & 2)
            mc_rect_luma_to(e, 1, (lists & 1) ? e->pred_l2 : e->pred_l,
                            mbx, mby, geo[0], geo[1], geo[2], geo[3],
                            c->mv1[b0][0], c->mv1[b0][1]);
        if ((lists & 3) == 3)
            for (y = 0; y < geo[3]; y++)
                for (x = 0; x < geo[2]; x++) {
                    int o = (geo[1] + y) * 16 + geo[0] + x;
                    e->pred_l[o] = (uint8_t)((e->pred_l[o] + e->pred_l2[o] + 1) >> 1);
                }
        {
            uint8_t *d0[2] = { e->pred_c[0], e->pred_c[1] };
            uint8_t *d1[2] = { (lists & 1) ? e->pred_c2[0] : e->pred_c[0],
                               (lists & 1) ? e->pred_c2[1] : e->pred_c[1] };
            if (lists & 1)
                mc_rect_chroma_to(e, 0, d0, mbx, mby, geo[0] / 2, geo[1] / 2,
                                  geo[2] / 2, geo[3] / 2, c->mv[b0][0],
                                  c->mv[b0][1]);
            if (lists & 2)
                mc_rect_chroma_to(e, 1, d1, mbx, mby, geo[0] / 2, geo[1] / 2,
                                  geo[2] / 2, geo[3] / 2, c->mv1[b0][0],
                                  c->mv1[b0][1]);
        }
        if ((lists & 3) == 3) {
            int pl;
            for (pl = 0; pl < 2; pl++)
                for (y = 0; y < geo[3] / 2; y++)
                    for (x = 0; x < geo[2] / 2; x++) {
                        int o = (geo[1] / 2 + y) * 8 + geo[0] / 2 + x;
                        e->pred_c[pl][o] =
                            (uint8_t)((e->pred_c[pl][o] + e->pred_c2[pl][o] + 1) >> 1);
                    }
        }
    }
}

/* Prediction of a direct macroblock: the vectors are per 8x8 block (see
 * b_direct_mv), and both lists predict every block.  It cannot go through
 * b_predict_typed because that works partition by partition and a direct
 * macroblock's partitions are the co-located picture's, which the bitstream
 * does not carry. */
static void b_direct_predict(struct aeg_encoder *e, int mbx, int mby,
                             const mbenc *c)
{
    int g, x, y, pl;
    const int lists = c->blist[0] ? c->blist[0] : 3;

    if (lists != 3) {
        /* spatial direct on one list: an ordinary single-list prediction
         * per 8x8 group (the groups' vectors differ only through colZero) */
        for (g = 0; g < 4; g++) {
            int x0 = (g & 1) * 8, y0 = (g >> 1) * 8;
            int b0 = (y0 >> 2) * 4 + (x0 >> 2);
            int l = lists == 2;
            int mx = l ? c->mv1[b0][0] : c->mv[b0][0];
            int my = l ? c->mv1[b0][1] : c->mv[b0][1];
            uint8_t *d0[2] = { e->pred_c[0], e->pred_c[1] };

            mc_rect_luma_to(e, l, e->pred_l, mbx, mby, x0, y0, 8, 8, mx, my);
            mc_rect_chroma_to(e, l, d0, mbx, mby, x0 / 2, y0 / 2, 4, 4, mx, my);
        }
        return;
    }
    for (g = 0; g < 4; g++) {
        int x0 = (g & 1) * 8, y0 = (g >> 1) * 8;
        /* Index of the 8x8 group's first 4x4 block: the group is two blocks
         * wide, so the row stride is four blocks, not sixteen. */
        int b0 = (y0 >> 2) * 4 + (x0 >> 2);
        uint8_t *d0[2] = { e->pred_c[0], e->pred_c[1] };
        uint8_t *d1[2] = { e->pred_c2[0], e->pred_c2[1] };

        mc_rect_luma_to(e, 0, e->pred_l, mbx, mby, x0, y0, 8, 8,
                        c->mv[b0][0], c->mv[b0][1]);
        mc_rect_luma_to(e, 1, e->pred_l2, mbx, mby, x0, y0, 8, 8,
                        c->mv1[b0][0], c->mv1[b0][1]);
        for (y = 0; y < 8; y++)
            for (x = 0; x < 8; x++) {
                int o = (y0 + y) * 16 + x0 + x;
                e->pred_l[o] = (uint8_t)((e->pred_l[o] + e->pred_l2[o] + 1) >> 1);
            }
        mc_rect_chroma_to(e, 0, d0, mbx, mby, x0 / 2, y0 / 2, 4, 4,
                          c->mv[b0][0], c->mv[b0][1]);
        mc_rect_chroma_to(e, 1, d1, mbx, mby, x0 / 2, y0 / 2, 4, 4,
                          c->mv1[b0][0], c->mv1[b0][1]);
        for (pl = 0; pl < 2; pl++)
            for (y = 0; y < 4; y++)
                for (x = 0; x < 4; x++) {
                    int o = (y0 / 2 + y) * 8 + x0 / 2 + x;
                    e->pred_c[pl][o] =
                        (uint8_t)((e->pred_c[pl][o] + e->pred_c2[pl][o] + 1) >> 1);
                }
    }
}

/* Prediction of a whole macroblock coded with `type`; which of the two shapes
 * of prediction it is depends on the slice, not on the caller. */
static void inter_predict_typed(struct aeg_encoder *e, int mbx, int mby,
                                const mbenc *c)
{
    if (!is_b_type(c->type))
        p_predict_typed(e, mbx, mby, c);
    else if (c->type == AEG_MB_B_DIRECT)
        b_direct_predict(e, mbx, mby, c);
    else
        b_predict_typed(e, mbx, mby, c);
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
    int ref; /* NB_OK/NOTUSED/UNAVAIL */
    int mvx, mvy;
    int ref_idx; /* -1 for unavailable/notused, 0..1 for available */
} nb_mv;

static void nb_set(nb_mv *n, int ref, int mvx, int mvy)
{
    n->ref = ref; n->mvx = mvx; n->mvy = mvy;
    n->ref_idx = -1;
}
static void nb_set_r(nb_mv *n, int ref, int mvx, int mvy, int ref_idx)
{
    n->ref = ref; n->mvx = mvx; n->mvy = mvy;
    n->ref_idx = ref_idx;
}

/* Fill `n` from macroblock `m` (NULL when the neighbour does not exist) for
 * prediction list `list`.  A neighbour only contributes to a list it actually
 * predicts from: a B macroblock whose partitions are all list 1 is "not used"
 * for list 0 and vice versa, which is exactly how the decoder builds the
 * per-list predictors. */
static void nb_from_mb(nb_mv *n, const aeg_mb *m, int blk, int list)
{
    if (!m) {
        nb_set(n, NB_UNAVAIL, 0, 0);
        return;
    }
    if (is_intra_type(m->type) ||
        !(aeg_mb_block_lists(m, blk) & (1 << list))) {
        nb_set(n, NB_NOTUSED, 0, 0);
        return;
    }
    nb_set_r(n, NB_OK, list ? m->mv1[blk][0] : m->mv[blk][0],
             list ? m->mv1[blk][1] : m->mv[blk][1],
             (int)m->ref_idx[blk]);
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
/* Three states, because the decoder's motion-vector cache has three and they
 * are not interchangeable: `AEG_MVBLK_AVAIL` is a decided vector,
 * `AEG_MVBLK_NOTUSED` is a block that was coded but does not predict from this
 * list (the decoder stores LIST_NOT_USED and a zero vector there), and
 * anything else is PART_NOT_AVAILABLE -- a block the decoder has not reached
 * yet.  The difference matters twice: a LIST_NOT_USED neighbour counts as
 * "available" for the fallback branch of 8.4.1.3.1 and contributes its zero
 * vector to the median, while PART_NOT_AVAILABLE does not; and the match count
 * compares the reference index, so neither matches the current one. */
enum { AEG_MVBLK_UNAVAIL = 0, AEG_MVBLK_AVAIL = 1, AEG_MVBLK_NOTUSED = 2 };

static void nb_from_cur_r(nb_mv *n, const int16_t cur[16][2], const int8_t *cur_ref,
                          int state, int b)
{
    if (state == AEG_MVBLK_NOTUSED)
        nb_set(n, NB_NOTUSED, 0, 0);
    else if (state == AEG_MVBLK_AVAIL)
        nb_set_r(n, NB_OK, cur[b][0], cur[b][1], cur_ref ? (int)cur_ref[b] : 0);
    else
        nb_set(n, NB_UNAVAIL, 0, 0);
}

static void predict_mv_at_ref(struct aeg_encoder *e, int mbx, int mby,
                          const int16_t cur[16][2], const int8_t *cur_ref,
                          const uint8_t *avail,
                          int type, int p, int bx, int by, int pw, int list,
                          int cur_ref_idx,
                          int *pmvx, int *pmvy)
{
    nb_mv A, B, C;
    int match;
#define AVAIL(x, y) (avail[(y) * 4 + (x)])
#define NB_CUR(n, x, y) nb_from_cur_r(&(n), cur, cur_ref, AVAIL(x, y), ((y) << 2) + (x))

    /* A B macroblock predicts each partition from one or both lists, but the
     * shape rules are the P ones: 16x8 and 8x16 short-circuit to a single
     * neighbour, everything else runs the three-neighbour process. */
    if (type == AEG_MB_B16x8)
        type = AEG_MB_P16x8;
    else if (type == AEG_MB_B8x16)
        type = AEG_MB_P8x16;
    else if (type == AEG_MB_B_8x8)
        type = AEG_MB_P8x8;

    /* A: the block left of the partition (inside this macroblock when bx > 0). */
    if (bx > 0) {
        NB_CUR(A, bx - 1, by);
    } else if (mbx > 0) {
        nb_from_mb(&A, &e->mbs[mby * e->mb_w + mbx - 1], (by << 2) + 3, list);
    } else {
        nb_from_mb(&A, NULL, 0, list);
    }

    /* B: the block above the partition (inside this macroblock when by > 0). */
    if (by > 0) {
        NB_CUR(B, bx, by - 1);
    } else if (mby > e->top_row) {
        nb_from_mb(&B, &e->mbs[(mby - 1) * e->mb_w + mbx], 12 + bx, list);
    } else {
        nb_from_mb(&B, NULL, 0, list);
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
            NB_CUR(C, cx, cy);
        } else if (cy >= 0) {
            nb_set(&C, NB_UNAVAIL, 0, 0);
        } else if (cx <= 3) {
            if (mby > e->top_row)
                nb_from_mb(&C, &e->mbs[(mby - 1) * e->mb_w + mbx], 12 + cx, list);
            else
                nb_from_mb(&C, NULL, 0, list);
        } else {
            if (mby > e->top_row && mbx + 1 < e->mb_w)
                nb_from_mb(&C, &e->mbs[(mby - 1) * e->mb_w + mbx + 1], 12, list);
            else
                nb_from_mb(&C, NULL, 0, list);
        }
        if (C.ref == NB_UNAVAIL) {
            /* The decoder's fallback is the cache slot one before the block
             * directly above the partition, i.e. the block above-left of the
             * partition's top-left corner. */
            if (by > 0 && bx > 0) {
                NB_CUR(C, bx - 1, by - 1);
            } else if (by > 0) {
                if (mbx > 0)
                    nb_from_mb(&C, &e->mbs[mby * e->mb_w + mbx - 1],
                               ((by - 1) << 2) + 3, list);
                else
                    nb_from_mb(&C, NULL, 0, list);
            } else if (bx > 0 && mby > e->top_row) {
                nb_from_mb(&C, &e->mbs[(mby - 1) * e->mb_w + mbx], 12 + bx - 1, list);
            } else if (mby > e->top_row && mbx > 0) {
                nb_from_mb(&C, &e->mbs[(mby - 1) * e->mb_w + mbx - 1], 15, list);
            } else {
                nb_from_mb(&C, NULL, 0, list);
            }
        }
    }

    /* The decoder prefers a single neighbour that shares the reference. */
    if (type == AEG_MB_P16x8) {
        if (p == 0 && B.ref_idx == cur_ref_idx) {
            *pmvx = B.mvx; *pmvy = B.mvy;
            return;
        }
        if (p == 1 && A.ref_idx == cur_ref_idx) {
            *pmvx = A.mvx; *pmvy = A.mvy;
            return;
        }
    } else if (type == AEG_MB_P8x8) {
        /* Sub-macroblock partitions have no short-circuit of their own: the
         * reference decoder runs the ordinary three-neighbour process. */
    } else if (type == AEG_MB_P8x16) {
        if (p == 0 && A.ref_idx == cur_ref_idx) {
            *pmvx = A.mvx; *pmvy = A.mvy;
            return;
        }
        if (p == 1 && C.ref_idx == cur_ref_idx) {
            *pmvx = C.mvx; *pmvy = C.mvy;
            return;
        }
    }

    match = (C.ref_idx == cur_ref_idx) + (B.ref_idx == cur_ref_idx) + (A.ref_idx == cur_ref_idx);
    if (match == 1) {
        if (A.ref_idx == cur_ref_idx)      { *pmvx = A.mvx; *pmvy = A.mvy; }
        else if (B.ref_idx == cur_ref_idx) { *pmvx = B.mvx; *pmvy = B.mvy; }
        else                     { *pmvx = C.mvx; *pmvy = C.mvy; }
    } else if (match == 0 && B.ref == NB_UNAVAIL && C.ref == NB_UNAVAIL &&
               A.ref != NB_UNAVAIL) {
        *pmvx = A.mvx; *pmvy = A.mvy;
    } else {
        *pmvx = mv_median3(A.mvx, B.mvx, C.mvx);
        *pmvy = mv_median3(A.mvy, B.mvy, C.mvy);
    }
#undef AVAIL
#undef NB_CUR
}
static void predict_mv_at(struct aeg_encoder *e, int mbx, int mby,
                          const int16_t cur[16][2], const uint8_t *avail,
                          int type, int p, int bx, int by, int pw, int list,
                          int *pmvx, int *pmvy)
{
    predict_mv_at_ref(e, mbx, mby, cur, NULL, avail, type, p, bx, by, pw, list, 0, pmvx, pmvy);
}

/* The 16x16 predictor for list-0 reference `ref` (no neighbour inside the
 * macroblock, no shape short-circuit). */
static void pred16_ref(struct aeg_encoder *e, int mbx, int mby, int ref,
                       int *pmvx, int *pmvy)
{
    int16_t cur[16][2];
    int8_t cur_ref[16];
    uint8_t avail[16];
    memset(cur, 0, sizeof(cur));
    memset(cur_ref, 0, sizeof(cur_ref));
    memset(avail, 0, sizeof(avail));
    predict_mv_at_ref(e, mbx, mby, (const int16_t (*)[2])cur, cur_ref, avail,
                      AEG_MB_P16x16, 0, 0, 0, 4, 0, ref, pmvx, pmvy);
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
        ax = m->mv[3][0]; ay = m->mv[3][1]; refa = m->ref_idx[3];
        /* only a neighbour on reference 0 with a zero vector forces (0,0):
         * one that sits still on the older picture is an ordinary vector */
        if (refa == 0 && ax == 0 && ay == 0)
            return;
    }
    m = &e->mbs[(mby - 1) * e->mb_w + mbx];        /* B: above */
    if (is_inter_type(m->type)) {
        bx = m->mv[12][0]; by = m->mv[12][1]; refb = m->ref_idx[12];
        if (refb == 0 && bx == 0 && by == 0)
            return;
    }
    if (mbx + 1 < e->mb_w) {                       /* C: above-right */
        m = &e->mbs[(mby - 1) * e->mb_w + mbx + 1];
        if (is_inter_type(m->type)) {
            cx = m->mv[12][0]; cy = m->mv[12][1]; refc = m->ref_idx[12];
        }
    } else {                                       /* ... else above-left */
        m = &e->mbs[(mby - 1) * e->mb_w + mbx - 1];
        if (is_inter_type(m->type)) {
            cx = m->mv[15][0]; cy = m->mv[15][1]; refc = m->ref_idx[15];
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
/* ------------------------------------------------------------------ */
/* B pictures                                                          */
/* ------------------------------------------------------------------ */
/* Motion vector prediction of the macroblock for list `list`.  Same rule as the
 * P path (aeg_mv_predict): the median of the neighbours that predict from this
 * list, or the only such neighbour, and (0,0) when there is none. */
static void b_mv_predict(const struct aeg_encoder *e, int mbx, int mby, int list,
                         int *mvx, int *mvy)
{
    const aeg_mb *m;
    int ax = 0, ay = 0, bx = 0, by = 0, cx = 0, cy = 0;
    int a_av = 0, b_av = 0, c_av = 0, n;

    if (mbx > 0) {
        m = &e->mbs[mby * e->mb_w + mbx - 1];
        if (!is_intra_type(m->type) &&
            (aeg_mb_block_lists(m, 3) & (1 << list))) {
            a_av = 1;
            ax = list ? m->mv1[3][0] : m->mv[3][0];
            ay = list ? m->mv1[3][1] : m->mv[3][1];
        }
    }
    if (mby > e->top_row) {
        m = &e->mbs[(mby - 1) * e->mb_w + mbx];
        if (!is_intra_type(m->type) &&
            (aeg_mb_block_lists(m, 12) & (1 << list))) {
            b_av = 1;
            bx = list ? m->mv1[12][0] : m->mv[12][0];
            by = list ? m->mv1[12][1] : m->mv[12][1];
        }
    }
    if (mby > e->top_row && mbx + 1 < e->mb_w) {
        m = &e->mbs[(mby - 1) * e->mb_w + mbx + 1];
        if (!is_intra_type(m->type) &&
            (aeg_mb_block_lists(m, 12) & (1 << list))) {
            c_av = 1;
            cx = list ? m->mv1[12][0] : m->mv[12][0];
            cy = list ? m->mv1[12][1] : m->mv[12][1];
        }
    } else if (mby > e->top_row && mbx > 0) {
        m = &e->mbs[(mby - 1) * e->mb_w + mbx - 1];
        if (!is_intra_type(m->type) &&
            (aeg_mb_block_lists(m, 15) & (1 << list))) {
            c_av = 1;
            cx = list ? m->mv1[15][0] : m->mv[15][0];
            cy = list ? m->mv1[15][1] : m->mv[15][1];
        }
    }
    n = a_av + b_av + c_av;
    if (n == 0) {
        *mvx = *mvy = 0;
    } else if (n == 1) {
        *mvx = a_av ? ax : (b_av ? bx : cx);
        *mvy = a_av ? ay : (b_av ? by : cy);
    } else {
        *mvx = mv_median3(ax, bx, cx);
        *mvy = mv_median3(ay, by, cy);
    }
}

/* May 8x8 group `g` of this B macroblock be coded as direct?  Temporal direct
 * takes the co-located block's reference picture as its list-0 reference
 * (8.4.1.2.3, refIdxL0 = MapColToList0(refIdxCol)).  When the anchor was coded
 * with two references and that block used the older one, the picture it points
 * at is two anchors back: it is not in this B picture's list 0, and with
 * max_num_ref_frames = 2 it is not even in the decoder's buffer any more, so a
 * direct block there would be a reference to a picture that does not exist
 * (FFmpeg conceals it with another one -- a silent mismatch).  Such groups are
 * simply not offered as direct: B_Skip, B_Direct_16x16 and B_Direct_8x8 all
 * stay away from them, everything else is unaffected. */
static int direct_group_ok(const struct aeg_encoder *e, int mbx, int mby, int g)
{
    static const int grp4[4] = { 0, 3, 12, 15 };
    const aeg_mb *col;
    /* spatial direct takes refIdx 0 of each list from the neighbours, never
     * the co-located block's reference, so every group qualifies */
    if (e->p.direct == 1 || !e->col_ready)
        return 1;
    col = &e->col_mbs[mby * e->mb_w + mbx];
    if (is_intra_type(col->type))
        return 1;
    return col->ref_idx[grp4[g]] == 0;
}

/* The vectors of a B_Direct_16x16 macroblock, derived the way the decoder
 * derives them (8.4.1.2.1, temporal direct -- the slice header asks for that).
 *
 * Direct mode is the whole point of B pictures: no motion vector is coded, no
 * residual has to be either, and the vectors come from the co-located
 * macroblock of the *future* picture, scaled by the picture order counts so
 * that they point at the right place in both directions.  Nothing about it
 * depends on the neighbours, so unlike spatial direct it needs no context and
 * cannot drift between encoder and decoder.
 *
 * Which of the co-located macroblock's vectors a direct macroblock takes needs
 * care, because the decoder derives a per-8x8 motion field for it and the
 * derivation does *not* look at the shape it derived from the co-located
 * type the way one would expect.  B_Direct_16x16's own type entry carries no
 * 16x16 shape bit, and the reference decoder's partitioned path reads one
 * vector per 8x8 group from the co-located picture's 4x4 block at offset
 * (3*x8, 3*y8) -- i.e. the blocks numbered 0, 3, 12 and 15 in raster order --
 * for every co-located shape except the single 16x16 case, where all sixteen
 * of its vectors are equal anyway and the two readings coincide.  That makes
 * one uniform rule possible for the encoder: always take one vector per 8x8
 * group from those four blocks.  Reading the co-located macroblock's first
 * block for the whole macroblock instead (which is what a 16x16 derivation
 * would do) leaves every macroblock whose co-located neighbour was partitioned
 * differently reconstructed than the decoder reconstructs it.
 *
 * Mismatches of this kind do not desynchronise the stream, so the syntax checks
 * see nothing; only the frame-by-frame reconstruction comparison catches them.
 */
static void b_direct_mv(const struct aeg_encoder *e, int mbx, int mby,
                        int mv0[16][2], int mv1[16][2])
{
    /* Per 8x8 group: the 4x4 block the co-located vector is read from. */
    static const int grp4[4] = { 0, 3, 12, 15 };
    const aeg_mb *col = &e->col_mbs[mby * e->mb_w + mbx];
    int tb = e->cur_poc - e->ref_poc[0];
    int ds = 256, g, k;

    memset(mv0, 0, 32 * sizeof(mv0[0][0]));
    memset(mv1, 0, 32 * sizeof(mv1[0][0]));
    /* Intra (or unknown) co-located information means "no motion" in both
     * directions, which is exactly what the decoder falls back to. */
    if (!e->col_ready || is_intra_type(col->type))
        return;
    for (g = 0; g < 4; g++) {
        int cx = col->mv[grp4[g]][0], cy = col->mv[grp4[g]][1];
        int td = e->col_poc - e->col_ref_poc[col->ref_idx[grp4[g]] & 1];
        int cur_ds = ds;
        if (td != 0) {
            int atd = td < 0 ? -td : td;
            int tx = (16384 + (atd >> 1)) / td;
            cur_ds = (tb * tx + 32) >> 6;
            if (cur_ds < -1024) cur_ds = -1024;
            if (cur_ds >  1023) cur_ds =  1023;
        } else {
            cur_ds = 256;
        }
        int x0 = (cur_ds * cx + 128) >> 8, y0 = (cur_ds * cy + 128) >> 8;
        int x8 = g & 1, y8 = g >> 1;

        for (k = 0; k < 4; k++) {
            int b = ((y8 << 1) + (k >> 1)) * 4 + (x8 << 1) + (k & 1);
            mv0[b][0] = x0;
            mv0[b][1] = y0;
            mv1[b][0] = x0 - cx;
            mv1[b][1] = y0 - cy;
        }
    }
}

/* Spatial direct (8.4.1.2.2), the derivation FFmpeg's
 * pred_spatial_direct_motion() implements for frame pictures.
 *
 * Per list, the reference index is the smallest non-negative one among the
 * macroblock's neighbours A, B and C (C falling back to D) -- with one
 * reference per list that is simply "0 if any of them predicts from this list,
 * else unused" -- and the vector is the ordinary 16x16 predictor of that list,
 * which b_mv_predict() already computes from the same three neighbours.  When
 * neither list is used by any neighbour both are used with zero vectors.
 * Then, per 8x8 group, colZeroFlag: if the co-located anchor's block at the
 * group's corner (direct_8x8_inference) is inter, on its reference 0, and its
 * vector is within +-1 quarter sample in both components, the vectors of the
 * used lists (both on reference 0) are zero for that group.
 *
 * Returns the list mask, the same for all four groups. */
static int b_direct_spatial(const struct aeg_encoder *e, int mbx, int mby,
                            int mv0[16][2], int mv1[16][2])
{
    static const int grp4[4] = { 0, 3, 12, 15 };
    const aeg_mb *col = e->col_ready ? &e->col_mbs[mby * e->mb_w + mbx] : NULL;
    int used[2], pmv[2][2], list, g, k, lists;

    memset(mv0, 0, 32 * sizeof(mv0[0][0]));
    memset(mv1, 0, 32 * sizeof(mv1[0][0]));
    for (list = 0; list < 2; list++) {
        const aeg_mb *m;
        int u = 0;
        if (mbx > 0) {
            m = &e->mbs[mby * e->mb_w + mbx - 1];
            u |= !is_intra_type(m->type) &&
                 (aeg_mb_block_lists(m, 3) & (1 << list));
        }
        if (mby > e->top_row) {
            m = &e->mbs[(mby - 1) * e->mb_w + mbx];
            u |= !is_intra_type(m->type) &&
                 (aeg_mb_block_lists(m, 12) & (1 << list));
            if (mbx + 1 < e->mb_w) {
                m = &e->mbs[(mby - 1) * e->mb_w + mbx + 1];
                u |= !is_intra_type(m->type) &&
                     (aeg_mb_block_lists(m, 12) & (1 << list));
            } else if (mbx > 0) {
                m = &e->mbs[(mby - 1) * e->mb_w + mbx - 1];
                u |= !is_intra_type(m->type) &&
                     (aeg_mb_block_lists(m, 15) & (1 << list));
            }
        }
        used[list] = u;
        pmv[list][0] = pmv[list][1] = 0;
        if (u)
            b_mv_predict(e, mbx, mby, list, &pmv[list][0], &pmv[list][1]);
    }
    if (!used[0] && !used[1])
        return 3;                       /* both lists, zero vectors */
    lists = used[0] | (used[1] << 1);
    for (g = 0; g < 4; g++) {
        int z = 0, x8 = g & 1, y8 = g >> 1;
        if (col && !is_intra_type(col->type) && col->ref_idx[grp4[g]] == 0) {
            int cx = col->mv[grp4[g]][0], cy = col->mv[grp4[g]][1];
            z = cx >= -1 && cx <= 1 && cy >= -1 && cy <= 1;
        }
        for (k = 0; k < 4; k++) {
            int b = ((y8 << 1) + (k >> 1)) * 4 + (x8 << 1) + (k & 1);
            if (used[0] && !z) {
                mv0[b][0] = pmv[0][0];
                mv0[b][1] = pmv[0][1];
            }
            if (used[1] && !z) {
                mv1[b][0] = pmv[1][0];
                mv1[b][1] = pmv[1][1];
            }
        }
    }
    return lists;
}

/* The direct-mode vectors of this macroblock in the mode the slice header
 * announces, and the lists they predict from (temporal: always both). */
static int b_direct(const struct aeg_encoder *e, int mbx, int mby,
                    int mv0[16][2], int mv1[16][2])
{
    if (e->p.direct == 1)
        return b_direct_spatial(e, mbx, mby, mv0, mv1);
    b_direct_mv(e, mbx, mby, mv0, mv1);
    return 3;
}

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
static void cand_publish_ref(mbcand *c, const int geo[4], int ref)
{
    int x, y, x0 = geo[0] / 4, y0 = geo[1] / 4;
    int x1 = x0 + geo[2] / 4, y1 = y0 + geo[3] / 4;
    for (y = y0; y < y1; y++)
        for (x = x0; x < x1; x++)
            c->ref[(y << 2) + x] = (uint8_t)ref;
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
/* SATD of one partition predicted from prediction list `list`. */
static int b_part_satd(struct aeg_encoder *e, int mbx, int mby,
                       const int geo[4], int mvx, int mvy, int list)
{
    const uint8_t *ref = list ? e->ref2p[0] : e->refp[0];
    const aeg_mc_ctx *mc = list ? &e->mc1 : &e->mc;
    uint8_t buf[16 * 16];
    int ix = mvx >> 2, iy = mvy >> 2;
    const uint8_t *r = ref + (mby * 16 + geo[1] + iy) * e->stride[0] +
                       mbx * 16 + geo[0] + ix;
    if (mc->ready)
        aeg_mc_luma_p(buf, geo[2], mc, r, geo[2], geo[3], mvx & 3, mvy & 3);
    else
        aeg_mc_luma(buf, geo[2], r, e->stride[0], geo[2], geo[3],
                    mvx & 3, mvy & 3);
    return aeg_satd(geo[2], geo[3], buf, geo[2],
                    e->srcp[0] + (mby * 16 + geo[1]) * e->stride[0] +
                        mbx * 16 + geo[0], e->stride[0]);
}

/* SATD of one partition predicted from both lists and averaged -- the
 * bi-predictive prediction, which is what a B picture is for. */
static int b_part_satd_bi(struct aeg_encoder *e, int mbx, int mby,
                          const int geo[4], int mv0x, int mv0y,
                          int mv1x, int mv1y)
{
    uint8_t a[16 * 16], b[16 * 16];
    int x, y;
    {
        const uint8_t *rp = e->refp[0] + (mby * 16 + geo[1] + (mv0y >> 2)) *
                                e->stride[0] + mbx * 16 + geo[0] + (mv0x >> 2);
        const uint8_t *rq = e->ref2p[0] + (mby * 16 + geo[1] + (mv1y >> 2)) *
                                e->stride[0] + mbx * 16 + geo[0] + (mv1x >> 2);
        if (e->mc.ready)
            aeg_mc_luma_p(a, geo[2], &e->mc, rp, geo[2], geo[3],
                          mv0x & 3, mv0y & 3);
        else
            aeg_mc_luma(a, geo[2], rp, e->stride[0], geo[2], geo[3],
                        mv0x & 3, mv0y & 3);
        if (e->mc1.ready)
            aeg_mc_luma_p(b, geo[2], &e->mc1, rq, geo[2], geo[3],
                          mv1x & 3, mv1y & 3);
        else
            aeg_mc_luma(b, geo[2], rq, e->stride[0], geo[2], geo[3],
                        mv1x & 3, mv1y & 3);
    }
    for (y = 0; y < geo[3]; y++)
        for (x = 0; x < geo[2]; x++)
            a[y * geo[2] + x] = (uint8_t)((a[y * geo[2] + x] +
                                           b[y * geo[2] + x] + 1) >> 1);
    return aeg_satd(geo[2], geo[3], a, geo[2],
                    e->srcp[0] + (mby * 16 + geo[1]) * e->stride[0] +
                        mbx * 16 + geo[0], e->stride[0]);
}

static int part_satd_ref(struct aeg_encoder *e, int mbx, int mby, const int geo[4],
                     int mvx, int mvy, int ref)
{
    uint8_t buf[16 * 16];
    int ix = mvx >> 2, iy = mvy >> 2;
    const uint8_t *rp = (ref ? e->ref2p[0] : e->refp[0]) + (mby * 16 + geo[1] + iy) * e->stride[0] +
                        mbx * 16 + geo[0] + ix;
    const aeg_mc_ctx *mc = ref ? &e->mc1 : &e->mc;
    if (mc->ready)
        aeg_mc_luma_p(buf, geo[2], mc, rp, geo[2], geo[3], mvx & 3, mvy & 3);
    else
        aeg_mc_luma(buf, geo[2], rp, e->stride[0], geo[2], geo[3],
                    mvx & 3, mvy & 3);
    return aeg_satd(geo[2], geo[3], buf, geo[2],
                    e->srcp[0] + (mby * 16 + geo[1]) * e->stride[0] + mbx * 16 +
                    geo[0], e->stride[0]);
}
static int part_satd(struct aeg_encoder *e, int mbx, int mby, const int geo[4],
                     int mvx, int mvy)
{
    return part_satd_ref(e, mbx, mby, geo, mvx, mvy, 0);
}

/* Motion search for one partition: its own search window, centred on its own
 * MV predictor.  `refine` selects the quarter-pel stage: the P_8x8 shape trials
 * run integer only (they compare shapes, not vectors) and the winning shape is
 * searched again with the refinement, which is where the quality comes from. */
static void me_partition_ref(struct aeg_encoder *e, int mbx, int mby, int type,
                         int p, const int geo[4], int pmvx, int pmvy, int ref,
                         int *mvx, int *mvy, int *sad, int refine)
{
    const uint8_t *rp = (ref ? e->ref2p[0] : e->refp[0]) + (mby * 16 + geo[1]) * e->stride[0] +
                        mbx * 16 + geo[0];
    const aeg_mc_ctx *mc = ref ? &e->mc1 : &e->mc;
    aeg_me_block_l(rp,
                 e->stride[0],
                 e->srcp[0] + (mby * 16 + geo[1]) * e->stride[0] +
                     mbx * 16 + geo[0],
                 e->stride[0], geo[2], geo[3],
                 e->p.search_range ? e->p.search_range : 16,
                 refine ? e->p.subme : 0, pmvx, pmvy, NULL, NULL, 0,
                 e->me_lambda, mc,
                 mvx, mvy, sad);
    if (!refine && e->p.subme > 0) {
        int fx = e->mb_frac[0], fy = e->mb_frac[1];
        int qx = (*mvx & ~3) + fx, qy = (*mvy & ~3) + fy;
        if (fx || fy) {
            int c_int = part_satd_ref(e, mbx, mby, geo, *mvx, *mvy, ref);
            int c_frac = part_satd_ref(e, mbx, mby, geo, qx, qy, ref);
            if (c_frac < c_int) {
                *mvx = qx;
                *mvy = qy;
                *sad = c_frac;
            }
        }
    }
    (void)type; (void)p;
}
static void me_partition(struct aeg_encoder *e, int mbx, int mby, int type,
                         int p, const int geo[4], int pmvx, int pmvy,
                         int *mvx, int *mvy, int *sad, int refine)
{
    me_partition_ref(e, mbx, mby, type, p, geo, pmvx, pmvy, 0, mvx, mvy, sad, refine);
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
/* Mark the 4x4 blocks a partition covers as decoded from list 0, so that the
 * following partitions can predict from them (and only from them).  This is
 * the P path, where list 0 is the only list there is. */
static void avail_publish(uint8_t avail[16], const int geo[4])
{
    int x, y, x0 = geo[0] / 4, y0 = geo[1] / 4;
    int x1 = x0 + geo[2] / 4, y1 = y0 + geo[3] / 4;
    for (y = y0; y < y1; y++)
        for (x = x0; x < x1; x++)
            avail[(y << 2) + x] = AEG_MVBLK_AVAIL;
}

/* Re-derive the predictor of every partition of an inter macroblock from the
 * vectors the macroblock finally carries.
 *
 * The neighbours inside the same macroblock are part of the predictor: the
 * second half of a 16x8 or 8x16 predicts from the first half (9.3.3.1 rules in
 * 8.4.1.3.1 -- the decoder has just stored those vectors in its block cache),
 * and every 8x8 sub-partition predicts from the ones before it.  The search
 * derives a partition's predictor before the *later* partitions exist, so if a
 * later step changes an earlier vector the stored predictor goes stale by
 * exactly that change, and the decoder -- which derives it from the coded
 * vector -- reconstructs a different vector.  The motion-vector refinement does
 * exactly that, so it has to call this afterwards. */
static void refresh_pmvs(struct aeg_encoder *e, int mbx, int mby, mbenc *c)
{
    int16_t cur[2][16][2];
    uint8_t avail[2][16];
    int8_t cur_ref[2][16];
    int lists = is_b_type(c->type) ? 2 : 1;
    int p, l, n = mb_part_count(c);

    memset(cur, 0, sizeof(cur));
    memset(avail, 0, sizeof(avail));
    memset(cur_ref, 0, sizeof(cur_ref));
    for (p = 0; p < n; p++) {
        int geo[4], b0;
        int masks = lists == 1 ? 1 : mb_part_lists(c, p);
        int x, y, x0, y0, x1, y1;

        mb_part_geom(c, p, geo, &b0);
        x0 = geo[0] / 4; y0 = geo[1] / 4;
        x1 = x0 + geo[2] / 4; y1 = y0 + geo[3] / 4;
        for (l = 0; l < lists; l++) {
            if (p > 0) {
                /* The first partition has no neighbour inside the macroblock,
                 * so its predictor is the same one the search stored and is
                 * left alone. */
                int pmvx, pmvy;
                int cur_r = (l == 0 && e->cur_l0_active > 1) ? (int)c->ref[b0] : 0;
                predict_mv_at_ref(e, mbx, mby, (const int16_t (*)[2])cur[l], cur_ref[l],
                              avail[l], c->type, p, x0, y0, geo[2] / 4, l, cur_r,
                              &pmvx, &pmvy);
                if (l == 0) {
                    c->pmv[b0][0] = (int16_t)pmvx;
                    c->pmv[b0][1] = (int16_t)pmvy;
                } else {
                    c->pmv1[b0][0] = (int16_t)pmvx;
                    c->pmv1[b0][1] = (int16_t)pmvy;
                }
            }
            /* Publish the partition for the ones that follow: a block that does
             * not predict from this list is LIST_NOT_USED to them, not
             * PART_NOT_AVAILABLE. */
            for (y = y0; y < y1; y++)
                for (x = x0; x < x1; x++) {
                    int b = (y << 2) + x;

                    if (masks & (1 << l)) {
                        cur[l][b][0] = l ? c->mv1[b0][0] : c->mv[b0][0];
                        cur[l][b][1] = l ? c->mv1[b0][1] : c->mv[b0][1];
                        avail[l][b] = AEG_MVBLK_AVAIL;
                        cur_ref[l][b] = (l == 0) ? (int8_t)c->ref[b0] : 0;
                    } else {
                        avail[l][b] = AEG_MVBLK_NOTUSED;
                        cur_ref[l][b] = -1;
                    }
                }
        }
    }
}

/* Whether the 16x8 / 8x16 partitions get their own sub-pel search instead of
 * the integer search plus the 16x16 fractional offset: 0 = neither, 1 = the P
 * ones, 2 = the P and the B ones.  With the inherited offset a partition was
 * competing with an integer vector against a quarter-pel 16x16 one, and lost:
 * 16x8/8x16 won 83 of 1500 P macroblocks on cock against x264's ~1 in 4.  On
 * the five real clips (docs/STATUS.md) the P search is worth -2.3% of BD-rate
 * P-only and -1.55% with B pictures, the B one a further -0.65%, for 6% of
 * the time each.  From preset 2 up; AEG_PSUB overrides. */
static int part_subpel(const struct aeg_encoder *e)
{
    static int v = -2;
    if (v == -2) {
        const char *ev = getenv("AEG_PSUB");
        v = ev ? atoi(ev) : -1;
    }
    if (v >= 0)
        return v;
    return e->p.preset >= 2 ? 2 : 0;
}

/* One partition of a partitioned B macroblock (16x8 or 8x16).
 *
 * Both lists are searched independently and then list 0, list 1 and the average
 * of the two compete.  The average is the reason B pictures exist -- it halves
 * the noise of each predictor -- and it is the only one of the three that sends
 * two motion vectors, which is why it is priced with both.
 *
 * `cur`/`avail` are per list: the decoder's cache is, and a partition that only
 * looks forward is *not* available to a following partition that predicts from
 * list 1 (its reference index for list 1 is LIST_NOT_USED, which matches no
 * reference).  Getting that backwards makes the encoder predict a different
 * vector than the decoder and reconstruct a different picture.
 *
 * The search itself is integer-only, with the quarter-pel offset the
 * macroblock's 16x16 search found applied afterwards -- four full quarter-pel
 * searches per macroblock would cost more than the B picture is worth; the
 * macroblock-level RD pass still has the last word on the shape. */
static void b_part_search(struct aeg_encoder *e, int mbx, int mby, int type,
                          int p, int16_t cur[2][16][2],
                          uint8_t avail[2][16], const int mv16[2][2],
                          mbcand *cd)
{
    int geo[4], b0, l, x, y, lists, best;
    int range = e->p.search_range ? e->p.search_range : 16;
    int pmv[2][2], mv[2][2], satd[3], cost[3], bits[2];

    part_geom(type, p, geo, &b0);
    for (l = 0; l < 2; l++) {
        const uint8_t *ref = l ? e->ref2p[0] : e->refp[0];
        const aeg_mc_ctx *mc = l ? &e->mc1 : &e->mc;
        int seed[2], sad;

        predict_mv_at(e, mbx, mby, cur[l], avail[l], type, p, geo[0] / 4,
                      geo[1] / 4, geo[2] / 4, l, &pmv[l][0], &pmv[l][1]);
        seed[0] = mv16[l][0] / 4;
        seed[1] = mv16[l][1] / 4;
        aeg_me_block_l(ref + (mby * 16 + geo[1]) * e->stride[0] + mbx * 16 +
                         geo[0],
                     e->stride[0],
                     e->srcp[0] + (mby * 16 + geo[1]) * e->stride[0] +
                         mbx * 16 + geo[0],
                     e->stride[0], geo[2], geo[3], range,
                     part_subpel(e) >= 2 ? e->p.subme : 0,
                     pmv[l][0], pmv[l][1], seed, NULL, 0, e->me_lambda, mc,
                     &mv[l][0], &mv[l][1],
                     &sad);
        if (e->p.subme > 0 && part_subpel(e) < 2) {
            int fx = l ? e->mb_frac1[0] : e->mb_frac[0];
            int fy = l ? e->mb_frac1[1] : e->mb_frac[1];
            int qx = (mv[l][0] & ~3) + fx, qy = (mv[l][1] & ~3) + fy;

            if (fx || fy) {
                int c_int = b_part_satd(e, mbx, mby, geo, mv[l][0], mv[l][1], l);
                int c_frac = b_part_satd(e, mbx, mby, geo, qx, qy, l);

                if (c_frac < c_int) {
                    mv[l][0] = qx;
                    mv[l][1] = qy;
                }
            }
        }
        satd[l] = b_part_satd(e, mbx, mby, geo, mv[l][0], mv[l][1], l);
        bits[l] = aeg_mvd_bits(mv[l][0] - pmv[l][0], mv[l][1] - pmv[l][1]);
    }
    satd[2] = b_part_satd_bi(e, mbx, mby, geo, mv[0][0], mv[0][1],
                             mv[1][0], mv[1][1]);
    cost[0] = satd[0] + 8 * bits[0];
    cost[1] = satd[1] + 8 * bits[1];
    cost[2] = satd[2] + 8 * (bits[0] + bits[1]);
    best = 0;
    if (cost[1] < cost[best]) best = 1;
    if (cost[2] < cost[best]) best = 2;
    lists = best == 0 ? 1 : (best == 1 ? 2 : 3);
    cd->blist[p] = (uint8_t)lists;
    cd->pmv[b0][0] = (int16_t)pmv[0][0];
    cd->pmv[b0][1] = (int16_t)pmv[0][1];
    cd->pmv1[b0][0] = (int16_t)pmv[1][0];
    cd->pmv1[b0][1] = (int16_t)pmv[1][1];
    for (y = 0; y < geo[3] / 4; y++)
        for (x = 0; x < geo[2] / 4; x++) {
            int b = (geo[1] / 4 + y) * 4 + geo[0] / 4 + x;

            cd->mv[b][0] = (int16_t)mv[0][0];
            cd->mv[b][1] = (int16_t)mv[0][1];
            cd->mv1[b][0] = (int16_t)mv[1][0];
            cd->mv1[b][1] = (int16_t)mv[1][1];
            if (lists & 1) {
                cur[0][b][0] = (int16_t)mv[0][0];
                cur[0][b][1] = (int16_t)mv[0][1];
                avail[0][b] = AEG_MVBLK_AVAIL;
            } else {
                avail[0][b] = AEG_MVBLK_NOTUSED;
            }
            if (lists & 2) {
                cur[1][b][0] = (int16_t)mv[1][0];
                cur[1][b][1] = (int16_t)mv[1][1];
                avail[1][b] = AEG_MVBLK_AVAIL;
            } else {
                avail[1][b] = AEG_MVBLK_NOTUSED;
            }
        }
}

/* A partitioned B candidate (16x8 or 8x16): two partitions, each of which picks
 * its own lists.  The RD pass decides whether the finer motion field pays for
 * the extra mb_type bins and the second mvd pair. */
static void b_candidate_parts(struct aeg_encoder *e, int mbx, int mby, int type,
                              const int mv16[2][2])
{
    int16_t cur[2][16][2];
    uint8_t avail[2][16];
    int p, slot = e->mb_ncand;
    mbcand *cd;

    if (slot >= (int)(sizeof(e->mb_cand) / sizeof(e->mb_cand[0])))
        return;
    memset(cur, 0, sizeof(cur));
    memset(avail, 0, sizeof(avail));
    cd = &e->mb_cand[slot];
    memset(cd, 0, sizeof(*cd));
    cd->type = type;
    cd->nparts = 2;
    for (p = 0; p < 2; p++)
        b_part_search(e, mbx, mby, type, p, cur, avail, mv16, cd);
    e->mb_ncand = slot + 1;
}

/* A B_8x8 candidate: four 8x8 sub-macroblock partitions, each of which picks
 * its own prediction out of direct, list 0, list 1 and both.
 *
 * The vectors come from one integer search per list per group -- seeded with
 * the macroblock's own 16x16 vector of that list, with the macroblock's
 * quarter-pel offset applied afterwards, exactly like the partitioned shapes
 * above -- and direct mode is priced against the vectors the co-located
 * picture hands out.  Direct sends no mvd at all, so it carries no rate term;
 * the other three are priced with their own differences, which is what makes
 * the choice per group rather than per macroblock.
 *
 * Only the four 8x8 sub-types are searched.  The 8x4/4x8/4x4 splits are eight
 * more sub-types and would triple the searches; the shape decision that pays
 * for them in a P picture (SATD against the sub_mb_type's own rate) has no B
 * counterpart yet, and the B value of a split is much smaller than the P one
 * because the two lists already absorb most of the extra freedom. */
static void b_candidate_8x8(struct aeg_encoder *e, int mbx, int mby,
                            const int mv16[2][2])
{
    int16_t cur[2][16][2];
    uint8_t avail[2][16];
    int direct[16][2], direct1[16][2], dlists = 3;
    int g, l, slot = e->mb_ncand;
    int range = e->p.search_range ? e->p.search_range : 16;
    int b8_sub = getenv("AEG_B8SUB") ? atoi(getenv("AEG_B8SUB")) : 0;
    mbcand *cd;

    if (slot >= (int)(sizeof(e->mb_cand) / sizeof(e->mb_cand[0])))
        return;
    memset(cur, 0, sizeof(cur));
    memset(avail, 0, sizeof(avail));
    memset(direct, 0, sizeof(direct));
    memset(direct1, 0, sizeof(direct1));
    if (e->col_ready)
        dlists = b_direct(e, mbx, mby, direct, direct1);
    cd = &e->mb_cand[slot];
    memset(cd, 0, sizeof(*cd));
    cd->type = AEG_MB_B_8x8;
    cd->nparts = 4;

    for (g = 0; g < 4; g++) {
        int geo[4], b0, pmv[2][2], mv[2][2], satd[4], cost[4], bits[2];
        int best, st;
        int x, y;

        geo[0] = (g & 1) * 8; geo[1] = (g >> 1) * 8; geo[2] = geo[3] = 8;
        b0 = (g >> 1) * 8 + (g & 1) * 2;
        for (l = 0; l < 2; l++) {
            const uint8_t *ref = l ? e->ref2p[0] : e->refp[0];
            const aeg_mc_ctx *mc = l ? &e->mc1 : &e->mc;
            int seed[2], sad;

            predict_mv_at(e, mbx, mby, cur[l], avail[l], AEG_MB_B_8x8, 0,
                          geo[0] / 4, geo[1] / 4, geo[2] / 4, l,
                          &pmv[l][0], &pmv[l][1]);
            seed[0] = mv16[l][0] / 4;
            seed[1] = mv16[l][1] / 4;
            aeg_me_block_l(ref + (mby * 16 + geo[1]) * e->stride[0] + mbx * 16 +
                             geo[0],
                         e->stride[0],
                         e->srcp[0] + (mby * 16 + geo[1]) * e->stride[0] +
                             mbx * 16 + geo[0],
                         e->stride[0], 8, 8, range, b8_sub ? e->p.subme : 0,
                         pmv[l][0], pmv[l][1],
                         seed, NULL, 0, e->me_lambda, mc, &mv[l][0],
                         &mv[l][1], &sad);
            if (e->p.subme > 0 && !b8_sub) {
                int fx = l ? e->mb_frac1[0] : e->mb_frac[0];
                int fy = l ? e->mb_frac1[1] : e->mb_frac[1];
                int qx = (mv[l][0] & ~3) + fx, qy = (mv[l][1] & ~3) + fy;

                if ((fx || fy) &&
                    b_part_satd(e, mbx, mby, geo, qx, qy, l) <
                        b_part_satd(e, mbx, mby, geo, mv[l][0], mv[l][1], l)) {
                    mv[l][0] = qx;
                    mv[l][1] = qy;
                }
            }
            satd[l] = b_part_satd(e, mbx, mby, geo, mv[l][0], mv[l][1], l);
            bits[l] = aeg_mvd_bits(mv[l][0] - pmv[l][0],
                                   mv[l][1] - pmv[l][1]);
        }
        satd[2] = b_part_satd_bi(e, mbx, mby, geo, mv[0][0], mv[0][1],
                                 mv[1][0], mv[1][1]);
        /* direct: the co-located vectors, both lists, no difference to send */
        satd[3] = !(e->col_ready && direct_group_ok(e, mbx, mby, g))
                      ? (1 << 30)
                      : dlists == 3
                      ? b_part_satd_bi(e, mbx, mby, geo, direct[b0][0],
                                       direct[b0][1], direct1[b0][0],
                                       direct1[b0][1])
                      : dlists == 1
                      ? b_part_satd(e, mbx, mby, geo, direct[b0][0],
                                    direct[b0][1], 0)
                      : b_part_satd(e, mbx, mby, geo, direct1[b0][0],
                                    direct1[b0][1], 1);
        cost[0] = satd[0] + 8 * bits[0];
        cost[1] = satd[1] + 8 * bits[1];
        cost[2] = satd[2] + 8 * (bits[0] + bits[1]);
        cost[3] = satd[3];
        best = 0;
        for (l = 1; l < 4; l++)
            if (cost[l] < cost[best])
                best = l;
        /* best: 0 = list 0, 1 = list 1, 2 = both, 3 = direct */
        st = best == 0 ? 1 : (best == 1 ? 2 : (best == 2 ? 3 : 0));
        cd->sub_type[g] = (uint8_t)st;
        cd->blist[g] = (uint8_t)(st == 0 ? dlists : b_sub_lists[st]);
        cd->pmv[b0][0] = (int16_t)pmv[0][0];
        cd->pmv[b0][1] = (int16_t)pmv[0][1];
        cd->pmv1[b0][0] = (int16_t)pmv[1][0];
        cd->pmv1[b0][1] = (int16_t)pmv[1][1];
        if (best == 3) {
            mv[0][0] = direct[b0][0];  mv[0][1] = direct[b0][1];
            mv[1][0] = direct1[b0][0]; mv[1][1] = direct1[b0][1];
        }
        for (y = 0; y < 2; y++)
            for (x = 0; x < 2; x++) {
                int b = (geo[1] / 4 + y) * 4 + geo[0] / 4 + x;

                cd->mv[b][0] = (int16_t)mv[0][0];
                cd->mv[b][1] = (int16_t)mv[0][1];
                cd->mv1[b][0] = (int16_t)mv[1][0];
                cd->mv1[b][1] = (int16_t)mv[1][1];
                if (cd->blist[g] & 1) {
                    cur[0][b][0] = (int16_t)mv[0][0];
                    cur[0][b][1] = (int16_t)mv[0][1];
                    avail[0][b] = AEG_MVBLK_AVAIL;
                } else {
                    avail[0][b] = AEG_MVBLK_NOTUSED;
                }
                if (cd->blist[g] & 2) {
                    cur[1][b][0] = (int16_t)mv[1][0];
                    cur[1][b][1] = (int16_t)mv[1][1];
                    avail[1][b] = AEG_MVBLK_AVAIL;
                } else {
                    avail[1][b] = AEG_MVBLK_NOTUSED;
                }
            }
    }
    e->mb_ncand = slot + 1;
}

/* Whether the partitioned shapes pick their reference per partition.  They
 * can (the search, the predictor and the syntax all handle it), but the pick
 * is a SATD + 8 * bits decision that replaces the ref-0 vector instead of
 * adding a candidate, and measured it is worse than leaving the partitions on
 * reference 0 and giving the second reference one whole-macroblock candidate
 * of its own: -0.5 % against -1.2 % mean BD-rate over the four synthetic
 * clips at preset 3, for more time (docs/STATUS.md, *Two P references*). */
static int part_ref_search(const struct aeg_encoder *e)
{
    (void)e;
    return 0;
}

static void me_candidate(struct aeg_encoder *e, int mbx, int mby, int type)
{
    int16_t cur[16][2];
    uint8_t avail[16];
    int8_t cur_ref[16];
    int p, n = part_count(type);
    int slot = e->mb_ncand;

    if (slot >= (int)(sizeof(e->mb_cand) / sizeof(e->mb_cand[0])))
        return;
    memset(cur, 0, sizeof(cur));
    memset(avail, 0, sizeof(avail));
    memset(cur_ref, 0, sizeof(cur_ref));
    memset(e->mb_cand[slot].sub_type, 0, sizeof(e->mb_cand[slot].sub_type));
    memset(e->mb_cand[slot].ref, 0, sizeof(e->mb_cand[slot].ref));
    e->mb_cand[slot].type = type;
    e->mb_cand[slot].nparts = n;
    for (p = 0; p < n; p++) {
        int geo[4], b0;
        part_geom(type, p, geo, &b0);
        if (part_ref_search(e)) {
            int best_ref = 0, best_cost = 1 << 30;
            int best_mvx = 0, best_mvy = 0, best_pmvx = 0, best_pmvy = 0, best_sad = 0;
            int ref;
            for (ref = 0; ref < e->cur_l0_active; ref++) {
                int pmvx, pmvy, mvx, mvy, sad, satd, bits, cost;
                predict_mv_at_ref(e, mbx, mby, (const int16_t (*)[2])cur, cur_ref, avail, type, p,
                                  geo[0] / 4, geo[1] / 4, geo[2] / 4, 0, ref, &pmvx, &pmvy);
                me_partition_ref(e, mbx, mby, type, p, geo, pmvx, pmvy, ref, &mvx, &mvy, &sad, part_subpel(e));
                satd = part_satd_ref(e, mbx, mby, geo, mvx, mvy, ref);
                bits = aeg_mvd_bits(mvx - pmvx, mvy - pmvy) + 1; /* ref_idx bit */
                cost = satd + 8 * bits;
                if (cost < best_cost) {
                    best_cost = cost;
                    best_ref = ref;
                    best_mvx = mvx; best_mvy = mvy;
                    best_pmvx = pmvx; best_pmvy = pmvy;
                    best_sad = sad;
                }
            }
            cand_publish(&e->mb_cand[slot], geo, best_mvx, best_mvy);
            cand_publish_ref(&e->mb_cand[slot], geo, best_ref);
            e->mb_cand[slot].pmv[b0][0] = best_pmvx;
            e->mb_cand[slot].pmv[b0][1] = best_pmvy;
            me_publish(cur, geo, best_mvx, best_mvy);
            avail_publish(avail, geo);
            { int x,y,x0=geo[0]/4,y0=geo[1]/4,x1=x0+geo[2]/4,y1=y0+geo[3]/4; for(y=y0;y<y1;y++) for(x=x0;x<x1;x++) cur_ref[(y<<2)+x]=(int8_t)best_ref; }
            (void)best_sad;
        } else {
            int pmvx, pmvy, mvx, mvy, sad;
            predict_mv_at(e, mbx, mby, (const int16_t (*)[2])cur, avail, type, p,
                          geo[0] / 4, geo[1] / 4, geo[2] / 4, 0, &pmvx, &pmvy);
            me_partition(e, mbx, mby, type, p, geo, pmvx, pmvy, &mvx, &mvy, &sad,
                         part_subpel(e));
            cand_publish(&e->mb_cand[slot], geo, mvx, mvy);
            cand_publish_ref(&e->mb_cand[slot], geo, 0);
            e->mb_cand[slot].pmv[b0][0] = pmvx;
            e->mb_cand[slot].pmv[b0][1] = pmvy;
            me_publish(cur, geo, mvx, mvy);
            avail_publish(avail, geo);
        }
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
    memset(cand->ref, 0, sizeof(cand->ref));

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
                int ref = 0;
                int cand_ref = 0;
                int best_c = 1<<30;
                int cand_mv[2][2] = {{0}}, cand_pmv[2][2] = {{0}}, cand_satd = 0;
                sub_geom(s, bx, by, k, geo[k], &b0);
                if (part_ref_search(e)) {
                    for (ref = 0; ref < e->cur_l0_active; ref++) {
                        int pmvx, pmvy, mvx, mvy, sd, sd_satd, bts, cst;
                        predict_mv_at(e, mbx, mby, (const int16_t (*)[2])trial,
                                      tavail, AEG_MB_P8x8, 0, geo[k][0] / 4,
                                      geo[k][1] / 4, geo[k][2] / 4, 0, &pmvx, &pmvy);
                        /* Note: predictor ref-unaware here keeps 1-ref behaviour for p8x8 trial;
                         * full ref-aware would need cur_ref plumbing. */
                        me_partition_ref(e, mbx, mby, AEG_MB_P8x8, k, geo[k], pmvx, pmvy, ref, &mvx, &mvy, &sd, 0);
                        sd_satd = part_satd_ref(e, mbx, mby, geo[k], mvx, mvy, ref);
                        bts = aeg_mvd_bits(mvx - pmvx, mvy - pmvy) + 1;
                        cst = sd_satd + 8 * bts;
                        if (cst < best_c) { best_c = cst; cand_ref = ref; cand_mv[0][0]=mvx; cand_mv[0][1]=mvy; cand_pmv[0][0]=pmvx; cand_pmv[0][1]=pmvy; cand_satd = sd_satd; (void)sd; }
                    }
                    pmv[k][0]=cand_pmv[0][0]; pmv[k][1]=cand_pmv[0][1];
                    mv[k][0]=cand_mv[0][0]; mv[k][1]=cand_mv[0][1];
                    satd_sum += cand_satd;
                    me_publish(trial, geo[k], mv[k][0], mv[k][1]);
                    avail_publish(tavail, geo[k]);
                    bits += aeg_mvd_bits(mv[k][0] - pmv[k][0], mv[k][1] - pmv[k][1]) + 1;
                    (void)cand_ref;
                } else {
                    predict_mv_at(e, mbx, mby, (const int16_t (*)[2])trial,
                                  tavail, AEG_MB_P8x8, 0, geo[k][0] / 4,
                                  geo[k][1] / 4, geo[k][2] / 4, 0, &pmv[k][0],
                                  &pmv[k][1]);
                    me_partition(e, mbx, mby, AEG_MB_P8x8, k, geo[k], pmv[k][0],
                                 pmv[k][1], &mv[k][0], &mv[k][1], &sad, 0);
                    satd_sum += part_satd(e, mbx, mby, geo[k], mv[k][0], mv[k][1]);
                    me_publish(trial, geo[k], mv[k][0], mv[k][1]);
                    avail_publish(tavail, geo[k]);
                    bits += aeg_mvd_bits(mv[k][0] - pmv[k][0],
                                         mv[k][1] - pmv[k][1]);
                }
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
                          best_geo[k][1] / 4, best_geo[k][2] / 4, 0, &pmvx,
                          &pmvy);
            /* An 8x8 partition gets the full quarter-pel search; the smaller
             * ones inherit the macroblock's fractional offset (see
             * me_partition), which is nearly free and within a hundredth of a
             * dB of refining each of them separately. */
            me_partition(e, mbx, mby, AEG_MB_P8x8, k, best_geo[k], pmvx, pmvy,
                         &best_mv[k][0], &best_mv[k][1], &sad,
                         best_geo[k][2] >= 8 && best_geo[k][3] >= 8);
            cand_publish(cand, best_geo[k], best_mv[k][0], best_mv[k][1]);
            /* P_8x8 sub-macroblocks keep the single reference path for now. */
            cand_publish_ref(cand, best_geo[k], 0);
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
/* Candidates of a macroblock of a B picture.
 *
 * Three of them are searched: list 0, list 1 and the bi-predictive average of
 * the two (which is the reason B pictures exist -- an average of two predictors
 * halves the noise of each, so the residual is smaller than either alone).  The
 * fourth is direct mode, whose vectors are derived rather than searched, and on
 * real content it wins most of the macroblocks: it costs no mvd and, when it
 * also has no residual, a single skip flag.
 *
 * The search runs twice per macroblock (once per reference).  That is the price
 * of a B picture, and it is why they are worth it: each of the two searches is
 * over a reference that may be a whole anchor interval away, and the average of
 * the two is what the residual is computed against.
 */
static int analyse_mb_b(struct aeg_encoder *e, int mbx, int mby)
{
    const uint8_t *src = e->srcp[0] + mby * 16 * e->stride[0] + mbx * 16;
    int pmv[2][2], mv[2][2], sad[2], list, b;
    int range = e->p.search_range ? e->p.search_range : 16;

    b_mv_predict(e, mbx, mby, 0, &pmv[0][0], &pmv[0][1]);
    b_mv_predict(e, mbx, mby, 1, &pmv[1][0], &pmv[1][1]);
    for (list = 0; list < 2; list++) {
        const uint8_t *ref = (list ? e->ref2p[0] : e->refp[0]) +
                             mby * 16 * e->stride[0] + mbx * 16;
        const aeg_mc_ctx *mc = list ? &e->mc1 : &e->mc;
        int seed[2];

        /* The list 0 search starts from the previous picture's vector for this
         * macroblock (always zero here: the whole-frame seed pass only runs for
         * P pictures); the list 1 search starts from the list 0 vector, which
         * is a far better starting point than zero on content that moves. */
        seed[0] = list ? mv[0][0] / 4 : (int)e->seed_mv[(mby * e->mb_w + mbx) * 2];
        seed[1] = list ? mv[0][1] / 4 : (int)e->seed_mv[(mby * e->mb_w + mbx) * 2 + 1];
        aeg_me_block_l(ref, e->stride[0], src, e->stride[0], 16, 16, range,
                     e->p.subme, pmv[list][0], pmv[list][1], seed, NULL, 0,
                     e->me_lambda, mc,
                     &mv[list][0], &mv[list][1], &sad[list]);
        if (!list) {
            /* the smaller partitions would inherit this alignment */
            e->mb_frac[0] = mv[0][0] & 3;
            e->mb_frac[1] = mv[0][1] & 3;
        }
    }
    e->mb_mv[0] = mv[0][0]; e->mb_mv[1] = mv[0][1];
    e->mb_pmv[0] = pmv[0][0]; e->mb_pmv[1] = pmv[0][1];
    e->mb_inter_ok = 1;
    e->mb_inter_sad = sad[0] < sad[1] ? sad[0] : sad[1];
    e->mb_ncand = 0;

    /* list 1's fractional offset, for the partitioned searches below (the
     * 16x16 search of each list leaves its own) */
    e->mb_frac1[0] = mv[1][0] & 3;
    e->mb_frac1[1] = mv[1][1] & 3;

    /* list 0, list 1, and both */
    for (list = 0; list < 3; list++) {
        mbcand *cd = &e->mb_cand[e->mb_ncand];
        memset(cd, 0, sizeof(*cd));
        cd->type = list == 0 ? AEG_MB_B_L0 : (list == 1 ? AEG_MB_B_L1
                                                        : AEG_MB_B_BI);
        cd->nparts = 1;
        cd->blist[0] = (uint8_t)(list == 0 ? 1 : (list == 1 ? 2 : 3));
        for (b = 0; b < 16; b++) {
            cd->mv[b][0] = (int16_t)mv[0][0];
            cd->mv[b][1] = (int16_t)mv[0][1];
            cd->mv1[b][0] = (int16_t)mv[1][0];
            cd->mv1[b][1] = (int16_t)mv[1][1];
        }
        cd->pmv[0][0] = (int16_t)pmv[0][0];
        cd->pmv[0][1] = (int16_t)pmv[0][1];
        cd->pmv1[0][0] = (int16_t)pmv[1][0];
        cd->pmv1[0][1] = (int16_t)pmv[1][1];
        e->mb_ncand++;
    }
    /* The partitioned shapes.  Like the P path they only pay off with the RD
     * decision, so ultrafast stays 16x16; and the candidate list has room for
     * both of them plus the four 16x16 ones. */
    if (e->p.preset >= 1) {
        int mv16[2][2];

        mv16[0][0] = mv[0][0]; mv16[0][1] = mv[0][1];
        mv16[1][0] = mv[1][0]; mv16[1][1] = mv[1][1];
        if (e->p.partitions & 2)
            b_candidate_parts(e, mbx, mby, AEG_MB_B16x8, (const int (*)[2])mv16);
        if (e->p.partitions & 4)
            b_candidate_parts(e, mbx, mby, AEG_MB_B8x16, (const int (*)[2])mv16);
        /* B_8x8 follows the preset rather than the partition mask: the mask's
         * bit 3 is documented as P_8x8, and this shape measured *neutral*
         * (BD-rate within +-0.1 % on five clips at six QPs) for about a tenth
         * of the encode time, so it sits with the slow presets.  Prepending it
         * to the candidate list costs two integer searches per 8x8 group --
         * eight per macroblock -- and the exact RD comparison then rejects it
         * on all but a handful of macroblocks. */
        if (e->p.preset >= 3)
            b_candidate_8x8(e, mbx, mby, (const int (*)[2])mv16);
    }
    /* direct: vectors derived, nothing sent */
    if (e->col_ready && direct_group_ok(e, mbx, mby, 0) &&
        direct_group_ok(e, mbx, mby, 1) && direct_group_ok(e, mbx, mby, 2) &&
        direct_group_ok(e, mbx, mby, 3)) {
        mbcand *cd = &e->mb_cand[e->mb_ncand];
        memset(cd, 0, sizeof(*cd));
        int dl;
        cd->type = AEG_MB_B_DIRECT;
        cd->nparts = 1;
        dl = b_direct(e, mbx, mby, cd->mv, cd->mv1);
        cd->blist[0] = cd->blist[1] = cd->blist[2] = cd->blist[3] = (uint8_t)dl;
        e->mb_ncand++;
    }
    return e->mb_ncand > 0;
}

/* Whether the motion search prices the vector's rate (lambda * mvd bits) as
 * well as its distortion.  AEG_MELAMBDA=0/1 overrides, for measurements. */
static int me_lambda_on(const struct aeg_encoder *e)
{
    const char *ev = getenv("AEG_MELAMBDA");
    (void)e;
    return ev ? atoi(ev) != 0 : 1;
}

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

    e->me_lambda = me_lambda_on(e) ? aeg_me_lambda(c->qp) : 0;
    e->mb_inter_ok = 0;
    e->mb_ncand = 0;
    e->mb_mv[0] = e->mb_mv[1] = 0;
    e->mb_pmv[0] = e->mb_pmv[1] = 0;
    e->mb_frac[0] = e->mb_frac[1] = 0;
    /* A B picture has its own candidate list: two searches, bi-prediction and
     * direct mode (see analyse_mb_b). */
    if (e->p.partitions != 0 && c->qp >= 0 && e->cur_is_b &&
        e->has_ref && e->has_ref2) {
        if (analyse_mb_b(e, mbx, mby))
            return;
    }
    /* The motion search needs a reference picture *and* a P slice: an IDR
     * picture is always an I slice, so inter candidates there could never be
     * coded (writing them as an I macroblock desynchronises the decoder). */
    if (e->p.partitions != 0 && c->qp >= 0 && e->has_ref && e->cur_is_p) {
        int mvx, mvy, sad;
        aeg_mv_predict(e->mbs, e->mb_w, e->mb_w, mbx, mby, e->top_row,
                       &e->mb_pmv[0], &e->mb_pmv[1]);
        /* With two references the neighbours may sit on the older picture,
         * and the predictor then follows the reference-matching rule of
         * 8.4.1.3.1 (a lone neighbour on the same reference wins), which the
         * single-reference shortcut above cannot express. */
        if (e->cur_l0_active > 1)
            pred16_ref(e, mbx, mby, 0, &e->mb_pmv[0], &e->mb_pmv[1]);
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
            aeg_me_block_l(e->refp[0] + mby * 16 * e->stride[0] + mbx * 16,
                         e->stride[0],
                         e->srcp[0] + mby * 16 * e->stride[0] + mbx * 16,
                         e->stride[0], 16, 16,
                         e->p.search_range ? e->p.search_range : 16,
                         e->p.subme, e->mb_pmv[0], e->mb_pmv[1], seed,
                         NULL, 0, e->me_lambda, &e->mc, &mvx, &mvy, &sad);
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
                e->mb_cand[0].ref[b] = 0;
            }
            e->mb_cand[0].pmv[0][0] = e->mb_pmv[0];
            e->mb_cand[0].pmv[0][1] = e->mb_pmv[1];
        }
        e->mb_ncand = 1;
        /* The skip candidate: the vector a P_Skip macroblock reconstructs
         * with (8.4.1.1), which the search itself will rarely land on exactly
         * because it refines away from the seed.  Without this candidate a
         * macroblock whose best prediction is "nothing moved" has nothing that
         * can be coded as a skip, so the cheapest macroblock the standard has
         * is never priced -- and on a still picture that is nearly all of
         * them.  It differs from the candidate above only in the vector, so
         * the two prices are directly comparable. */
        {
            int sx, sy, b;
            pskip_mv(e, mbx, mby, &sx, &sy);
            if ((sx != mvx || sy != mvy) &&
                e->mb_ncand < (int)(sizeof(e->mb_cand) / sizeof(e->mb_cand[0]))) {
                mbcand *sc = &e->mb_cand[e->mb_ncand];
                memset(sc, 0, sizeof(*sc));
                sc->type = AEG_MB_P16x16;
                sc->nparts = 1;
                for (b = 0; b < 16; b++) {
                    sc->mv[b][0] = sx;
                    sc->mv[b][1] = sy;
                    sc->ref[b] = 0;
                }
                sc->pmv[0][0] = e->mb_pmv[0];
                sc->pmv[0][1] = e->mb_pmv[1];
                e->mb_ncand++;
            }
        }
        /* The 16x16 shape on the second reference (list0[1], the anchor
         * before the previous one): its own predictor, its own full search,
         * and the RD pass decides whether the extra ref_idx bin and the
         * older picture pay. */
        if (e->cur_l0_active > 1 && e->p.preset >= 1 &&
            e->mb_ncand < (int)(sizeof(e->mb_cand) / sizeof(e->mb_cand[0]))) {
            static const int geo16[4] = { 0, 0, 16, 16 };
            mbcand *rc = &e->mb_cand[e->mb_ncand];
            int px, py, rx, ry, rsad, b;
            pred16_ref(e, mbx, mby, 1, &px, &py);
            me_partition_ref(e, mbx, mby, AEG_MB_P16x16, 0, geo16, px, py, 1,
                             &rx, &ry, &rsad, 1);
            memset(rc, 0, sizeof(*rc));
            rc->type = AEG_MB_P16x16;
            rc->nparts = 1;
            for (b = 0; b < 16; b++) {
                rc->mv[b][0] = rx;
                rc->mv[b][1] = ry;
                rc->ref[b] = 1;
            }
            rc->pmv[0][0] = px;
            rc->pmv[0][1] = py;
            e->mb_ncand++;
        }
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
/* The cbp syntax element's chroma half is 0, 1 or 2; the store keeps it as two
 * bits (chroma DC in 0x10, chroma AC in 0x20). */
static int packed_chroma_cbp(int cbp)
{
    if (cbp & 0x20)
        return 2;
    return (cbp & 0x10) ? 1 : 0;
}

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
    if (is_b_type(c->type)) {
        /* A B macroblock's mb_type is a CABAC tree (base 27); its exact price
         * comes from the coder, so this is only the stand-in the CAVLC estimate
         * would use -- and CAVLC B pictures are not implemented (the CLI
         * refuses the combination), so it is only ever a heuristic. */
        int list = mb_lists(c->type, c->blist);
        int bits = c->type == AEG_MB_B_DIRECT ? 1 : 4;
        /* The partitioned types (codes 4..21) are the four-bit code plus the
         * fifth bin; a 16x16 one is that code alone. */
        (void)e;
        if (c->type == AEG_MB_B16x8 || c->type == AEG_MB_B8x16)
            return bits + (list == 3 ? 2 : 1);
        if (c->type == AEG_MB_B_8x8)
            return bits;                /* code 22: four bins, no fifth */
        return bits + (list == 3 ? 1 : 0);
    }
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
    /* P_8x8 carries a sub_mb_type per 8x8 partition, before any mvd; B_8x8's
     * is a CABAC tree of one to four bins, priced here by its depth. */
    if (c->type == AEG_MB_P8x8)
        for (i = 0; i < 4; i++)
            bits += aeg_ue_bits((uint32_t)c->sub_type[i]);
    if (c->type == AEG_MB_B_8x8)
        for (i = 0; i < 4; i++) {
            int st = c->sub_type[i];

            bits += (st == 0) ? 1 : (st <= 2 ? 2 : (st >= 11 ? 4 : 3));
        }
    /* ref_idx_l0 for P when two references are active. */
    if (e->cur_l0_active > 1 && is_inter_type(c->type) && !is_b_type(c->type)) {
        if (c->type == AEG_MB_P8x8) {
            for (i = 0; i < 4; i++)
                bits += 1; /* te(v) with cMax=1: 1 bit */
        } else {
            for (p = 0; p < c->nparts; p++)
                bits += 1;
        }
    }
    for (p = 0; p < c->nparts; p++) {
        int geo[4], b0;
        /* A partitioned B macroblock picks its lists per partition, so the
         * mask has to come from the partition rather than from the
         * macroblock's union of them. */
        int lmask = is_b_type(c->type) ? mb_part_lists(c, p)
                                       : mb_lists(c->type, c->blist);
        mb_part_geom(c, p, geo, &b0);
        /* A direct macroblock sends no mvd at all (8.4.1.2), the decoder
         * derives its vectors; every other inter type sends one pair per list
         * it predicts from. */
        if (!mb_part_sends_mvd(c, p))
            continue;
        if (lmask & 1)
            bits += aeg_mvd_bits(c->mv[b0][0] - c->pmv[b0][0],
                                 c->mv[b0][1] - c->pmv[b0][1]);
        if (lmask & 2)
            bits += aeg_mvd_bits(c->mv1[b0][0] - c->pmv1[b0][0],
                                 c->mv1[b0][1] - c->pmv1[b0][1]);
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
            if (e->p.trellis > 0 && e->p.cabac && trellis_intra_on()) {
                /* the same level selection the inter blocks get; the block's
                 * neighbours inside the macroblock are already final */
                struct trel tr;
                int a, b, va, vb;

                cab_nz4_luma(e, mbx, mby, t, rb, &a, &b, &va, &vb);
                tr.side = 4;
                tr.n = 16;
                tr.scan = aeg_scan4x4;
                tr.cat = 2;
                tr.cbf_ctx = cab_cbf_ctx(2, a, b, va, vb, 1);
                tr.dc = NULL;
                trel_descent(&e->cab_shadow, &tr, t->level[rb], sb, s, pred, 4,
                             t->qp, trellis_intra_lambda(lambda), 2, 1);
            }
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
        memcpy(t->blist, e->mb_cand[cand].blist, sizeof(t->blist));
        t->nparts = (uint8_t)e->mb_cand[cand].nparts;
        for (b = 0; b < 16; b++) {
            t->mv[b][0] = (int16_t)e->mb_cand[cand].mv[b][0];
            t->mv[b][1] = (int16_t)e->mb_cand[cand].mv[b][1];
            t->mv1[b][0] = (int16_t)e->mb_cand[cand].mv1[b][0];
            t->mv1[b][1] = (int16_t)e->mb_cand[cand].mv1[b][1];
            t->pmv[b][0] = (int16_t)e->mb_cand[cand].pmv[b][0];
            t->pmv[b][1] = (int16_t)e->mb_cand[cand].pmv[b][1];
            t->pmv1[b][0] = (int16_t)e->mb_cand[cand].pmv1[b][0];
            t->pmv1[b][1] = (int16_t)e->mb_cand[cand].pmv1[b][1];
            t->ref[b] = e->mb_cand[cand].ref[b];
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
    memcpy(t->blist, e->mb_cand[cand].blist, sizeof(t->blist));
    t->nparts = (uint8_t)e->mb_cand[cand].nparts;
    for (b = 0; b < 16; b++) {
        t->mv[b][0] = (int16_t)e->mb_cand[cand].mv[b][0];
        t->mv[b][1] = (int16_t)e->mb_cand[cand].mv[b][1];
        t->mv1[b][0] = (int16_t)e->mb_cand[cand].mv1[b][0];
        t->mv1[b][1] = (int16_t)e->mb_cand[cand].mv1[b][1];
        t->pmv[b][0] = (int16_t)e->mb_cand[cand].pmv[b][0];
        t->pmv[b][1] = (int16_t)e->mb_cand[cand].pmv[b][1];
        t->pmv1[b][0] = (int16_t)e->mb_cand[cand].pmv1[b][0];
        t->pmv1[b][1] = (int16_t)e->mb_cand[cand].pmv1[b][1];
        t->ref[b] = e->mb_cand[cand].ref[b];
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



static void write_mb_cabac(struct aeg_encoder *e, aeg_cabac *cb, int mbx,
                           int mby, mbenc *c, int is_p, int skip);
static int rd_cabac_bits(struct aeg_encoder *e, int mbx, int mby, mbenc *c,
                         int is_p, int commit);

/* The one place a luma candidate is priced: `est` is what the estimate says
 * (rd_luma() and friends write it to t->cost), the return value is what the
 * decision compares -- the coder's own price whenever the coder is CABAC, and
 * the estimate when it is CAVLC, which is what the estimate models.
 *
 * Every candidate is priced.  Pruning the ones the estimate dislikes (scaling
 * its rate term by the 0.65 CABAC actually spends and requiring them to be
 * 25% above the best) was measured and is *not* in: it recovered nothing --
 * the bin walk is a small part of a candidate's cost -- and cost 4% of the
 * bits on the clip where the decisions are closest. */
static int cur_slice_kind(const struct aeg_encoder *e)
{
    return e->cur_is_p ? AEG_SLICE_P : (e->cur_is_b ? AEG_SLICE_B : AEG_SLICE_I);
}

static int64_t rd_price(struct aeg_encoder *e, int mbx, int mby, mbenc *t,
                        int64_t est, double lambda)
{
    if (!e->p.cabac)
        return est;
    return t->sse +
           (int64_t)(lambda * (rd_cabac_bits(e, mbx, mby, t, cur_slice_kind(e),
                                             0) / 256.0) + 0.5);
}

/* Rate-distortion refinement of one *candidate*, before it is compared.
 *
 * The winner refinement (below, in code_mb) re-codes the macroblock type that
 * already won; this does the same for every candidate as it is priced, which is
 * what x264's subpel stage does and the only way a vector that is off by a
 * quarter pel can be rescued for a candidate that would otherwise lose on
 * prediction quality alone.  It costs a whole macroblock evaluation per
 * neighbour per candidate, so it belongs to the slowest preset.
 *
 * Only the 16x16, 16x8 and 8x16 shapes are reached (two partitions at most: the
 * 8x8 shapes are four groups with a list mask each, which is the same budget
 * the winner refinement works within).  That is also what makes the stored
 * predictors reusable after a move: those partitions predict their vectors from
 * *other* macroblocks, so refining one does not invalidate another's mvd.  The
 * 8x8 shapes are the ones whose sub-partitions predict from each other, and
 * they are exactly the ones left out; if they are ever let in, the predictors
 * have to be re-derived between partitions, as refresh_pmvs() does for the
 * winner.
 */
static void refine_cand(struct aeg_encoder *e, int mbx, int mby, int slot,
                        int qp, double lambda)
{
    static const int dx[4] = { 0, -1, 1, 0 };
    static const int dy[4] = { -1, 0, 0, 1 };
    mbcand save = e->mb_cand[slot], win;
    mbenc base;
    int64_t best = INT64_MAX;
    int p, l, j, n, lists, ok = 0;

    cand_enc(e, slot, qp, &base);
    if (!is_inter_type(base.type) || base.type == AEG_MB_B_DIRECT)
        return;
    n = mb_part_count(&base);
    if (n > 2)
        return;                     /* P_8x8 and B_8x8: too many candidates */
    lists = is_b_type(base.type) ? 2 : 1;
    for (p = 0; p < n; p++) {
        int geo[4], b0, masks = lists == 1 ? 1 : mb_part_lists(&base, p);

        mb_part_geom(&base, p, geo, &b0);
        for (l = 0; l < lists; l++) {
            int x, y, x0 = geo[0] / 4, y0 = geo[1] / 4;
            int x1 = x0 + geo[2] / 4, y1 = y0 + geo[3] / 4;

            if (!(masks & (1 << l)))
                continue;
            for (j = 0; j < 4; j++) {
                int64_t cost;
                e->mb_cand[slot] = save;
                for (y = y0; y < y1; y++)
                    for (x = x0; x < x1; x++) {
                        int bb = (y << 2) + x;
                        if (l == 0) {
                            e->mb_cand[slot].mv[bb][0] = base.mv[b0][0] + dx[j];
                            e->mb_cand[slot].mv[bb][1] = base.mv[b0][1] + dy[j];
                        } else {
                            e->mb_cand[slot].mv1[bb][0] = base.mv1[b0][0] + dx[j];
                            e->mb_cand[slot].mv1[bb][1] = base.mv1[b0][1] + dy[j];
                        }
                    }
                cost = rd_luma(e, mbx, mby, &base, qp, save.type, 0, lambda, slot);
                cost = rd_price(e, mbx, mby, &base, cost, lambda);
                if (cost < best) {
                    best = cost;
                    win = e->mb_cand[slot];
                    ok = 1;
                }
            }
        }
    }
    e->mb_cand[slot] = ok ? win : save;
}

/* The price the macroblock decision compares: the luma candidate in `t` with
 * its chroma coded too (the motion-compensated chroma for an inter candidate,
 * mode `cmode` for an intra one, whose own chroma decision comes after).  The
 * candidates used to be compared on luma alone, which cannot see that a
 * vector predicting luma a little better can predict chroma a lot worse, or
 * that a candidate's chroma residual costs bits.  `t` itself is left as it
 * was (luma only): the winner's chroma is coded once more afterwards, and the
 * chroma reconstruction this writes is overwritten then.  Preset 4;
 * AEG_CDECIDE=0/1 overrides. */
static int chroma_decide_on(const struct aeg_encoder *e)
{
    /* -0.34% of BD-rate for 10% of the time on the real clips, so it sits
     * with the slowest preset */
    static int v = -2;
    if (v == -2) {
        const char *ev = getenv("AEG_CDECIDE");
        v = ev ? atoi(ev) != 0 : -1;
    }
    return v >= 0 ? v : e->p.preset >= 4;
}

static int64_t full_price(struct aeg_encoder *e, int mbx, int mby,
                          const mbenc *t, int64_t est, int cmode, double lambda,
                          int64_t bound)
{
    mbenc f;
    int64_t dc;

    if (!chroma_decide_on(e))
        return rd_price(e, mbx, mby, (mbenc *)t, est, lambda);
    /* the luma distortion alone is a lower bound of the price: a candidate
     * it already rules out needs no chroma (and no coder) at all */
    if (e->p.cabac && t->sse >= bound)
        return t->sse;
    f = *t;
    dc = rd_chroma(e, mbx, mby, &f, is_inter_type(t->type),
                   is_inter_type(t->type) ? t->chroma_mode : cmode, lambda);
    return rd_price(e, mbx, mby, &f, est + dc, lambda);
}

/* Whether code_mb() prices the skipped macroblock against the winner.
 * AEG_SKIPCHK=0 turns it off. */
static int skip_check_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *ev = getenv("AEG_SKIPCHK");
        v = ev ? atoi(ev) != 0 : 1;
    }
    return v;
}

static void code_mb(struct aeg_encoder *e, int mbx, int mby, mbenc *c)
{
    /* Both snapshots are written by the first candidate that is evaluated, and
     * at least one always is (a preset above 0 with no legal Intra_16x16 mode
     * returns above), but initialising them keeps the compiler from warning
     * about every memcpy in the snapshot helpers. */
    uint8_t snap_l[16 * 16] = { 0 }, snap_c[2 * 8 * 8] = { 0 };
    mbenc best, t, retry;
    int64_t best_cost = INT64_MAX;
    double lambda = AEG_LAMBDA(c->qp);
    int i, inter;

    if ((e->p.preset <= 0 && !e->cur_is_b) || e->mb_nmodes == 0) {
        /* Plain SAD decision (analyse_mb already left the predictor).  A B
         * picture cannot take it: analyse_mb_b() only fills the candidate list
         * and leaves the macroblock type at its default, so the shortcut would
         * code every B macroblock as Intra_16x16 and throw the motion
         * compensation it had already computed away. */
        if (is_inter_type(c->type)) {
            code_luma_inter(e, mbx, mby, c);
            code_chroma(e, mbx, mby, c, 1);
        } else {
            code_luma_i16(e, mbx, mby, c);
            code_chroma(e, mbx, mby, c, 0);
        }
        if (e->p.cabac)
            rd_cabac_bits(e, mbx, mby, c, cur_slice_kind(e), 1);
        return;
    }

    /* luma: the SAD-best modes (all of them from preset 2 on) plus every inter
     * partitioning the motion search produced */
    for (i = 0; i < (e->p.preset >= 2 ? e->mb_nmodes : 1); i++) {
        int64_t cost = rd_luma(e, mbx, mby, &t, c->qp, AEG_MB_I16x16,
                               e->mb_modes[i], lambda, -1);
        cost = full_price(e, mbx, mby, &t, cost, c->chroma_mode, lambda, best_cost);
        if (cost < best_cost) {
            best_cost = cost;
            best = t;
            snap_luma(e, mbx, mby, snap_l);
        }
    }
    if (e->rd_batch && e->mb_ncand > 0) {
        best_cost = rd_luma_inter_batch(e, mbx, mby, &best, snap_l, best_cost,
                                        lambda, c->qp);
        /* The batch ranks the inter candidates against each other with the
         * estimate; the winner is then priced with the coder's own model, so
         * that comparing it with the intra candidates is like for like. */
        if (e->p.cabac && best_cost != INT64_MAX)
            best_cost = full_price(e, mbx, mby, &best, best_cost, c->chroma_mode, lambda, INT64_MAX);
    } else {
        for (i = 0; i < e->mb_ncand; i++) {
            int64_t cost = rd_luma(e, mbx, mby, &t, c->qp, e->mb_cand[i].type, 0,
                                   lambda, i);
            cost = full_price(e, mbx, mby, &t, cost, c->chroma_mode, lambda, best_cost);
            /* Refining a candidate costs a macroblock evaluation per
             * neighbour, so only the candidates that are still in the running
             * get it: one that is already worse than the best so far cannot be
             * rescued by a quarter pel, and a looser test than this one
             * measured no better (see docs/STATUS.md). */
            if (e->p.cabac && e->p.refine_cand > 0 && e->p.refine_mv > 0 &&
                cost <= best_cost) {
                refine_cand(e, mbx, mby, i, c->qp, lambda);
                cost = rd_luma(e, mbx, mby, &t, c->qp, e->mb_cand[i].type, 0,
                               lambda, i);
                cost = full_price(e, mbx, mby, &t, cost, c->chroma_mode, lambda, best_cost);
            }
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
        cost = full_price(e, mbx, mby, &t, cost, c->chroma_mode, lambda, best_cost);
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
            cost = full_price(e, mbx, mby, &t, cost, c->chroma_mode, lambda, best_cost);
            if (cost < best_cost) {
                best_cost = cost;
                best = t;
                snap_luma(e, mbx, mby, snap_l);
            }
        }
    }
    restore_luma(e, mbx, mby, snap_l);
    inter = is_inter_type(best.type);

    /* Rate-distortion refinement of the winning 16x16 vector.  The search
     * chooses the vector by the SATD of the prediction alone, which cannot see
     * what the residual or the mvd will cost; this re-codes the macroblock
     * with each of the eight quarter-pel neighbours of the winner and keeps
     * whichever the coder's own price prefers.  It runs before the chroma
     * decision because the chroma prediction follows the luma vector, and it
     * restores the luma plane afterwards because every trial wrote its own
     * reconstruction there.  Eight full candidate evaluations per macroblock,
     * so it belongs to the slow presets (AEG_MVREF=1 forces it on for
     * measurement). */
    if (e->p.cabac && e->p.refine_mv > 0 && inter && !e->cur_is_b &&
        mb_part_count(&best) <= 2) {
        static const int dx[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
        static const int dy[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
        static const int cross[4] = { 1, 3, 4, 6 };
        const int cheap = e->p.refine_mv < 2;
        const int nn = cheap ? 4 : 8;
        const int npart = mb_part_count(&best);
        mbcand save = e->mb_cand[0];
        int p, b, j;
        /* One partition at a time: a 16x8 or 8x16 winner has two vectors and
         * they are searched independently, so each is refined with the other
         * held at the value the search returned.  (P_8x8 is left alone: four
         * partitions times eight neighbours is 32 full candidate evaluations,
         * which is more than the rest of the macroblock costs.) */
        for (p = 0; p < npart; p++) {
            int geo[4], b0;
            mb_part_geom(&best, p, geo, &b0);
            for (j = 0; j < nn; j++) {
                int k = cheap ? cross[j] : j;
                int64_t cost;
                e->mb_cand[0] = save;
                e->mb_cand[0].type = best.type;
                e->mb_cand[0].nparts = best.nparts;
                memcpy(e->mb_cand[0].sub_type, best.sub_type,
                       sizeof(e->mb_cand[0].sub_type));
                for (b = 0; b < 16; b++) {
                    e->mb_cand[0].mv[b][0] = best.mv[b][0];
                    e->mb_cand[0].mv[b][1] = best.mv[b][1];
                    e->mb_cand[0].pmv[b][0] = best.pmv[b][0];
                    e->mb_cand[0].pmv[b][1] = best.pmv[b][1];
                    /* the reference is part of what is being refined around:
                     * the trial must predict from the winner's picture */
                    e->mb_cand[0].ref[b] = best.ref[b];
                }
                for (b = 0; b < 16; b++) {
                    int bx = (b & 3) * 4, by = (b >> 2) * 4;
                    if (bx < geo[0] || bx >= geo[0] + geo[2] ||
                        by < geo[1] || by >= geo[1] + geo[3])
                        continue;
                    e->mb_cand[0].mv[b][0] = best.mv[b][0] + dx[k];
                    e->mb_cand[0].mv[b][1] = best.mv[b][1] + dy[k];
                }
                cost = rd_luma(e, mbx, mby, &t, c->qp, best.type, 0, lambda, 0);
                cost = full_price(e, mbx, mby, &t, cost, c->chroma_mode, lambda, best_cost);
                if (cost < best_cost) {
                    best_cost = cost;
                    best = t;
                    snap_luma(e, mbx, mby, snap_l);
                }
            }
        }
        e->mb_cand[0] = save;
        restore_luma(e, mbx, mby, snap_l);
        /* The refinement may have moved the first partition's vector, and the
         * second partition's predictor was derived from the vector the search
         * returned.  Re-derive it from what the macroblock now carries. */
        refresh_pmvs(e, mbx, mby, &best);
    }

    /* The same rate-distortion refinement for a B macroblock's vectors.
     *
     * A B picture's vectors are searched per list by SATD and are never looked
     * at again, which is the one thing the P path does differently: there the
     * winner of the candidate comparison is re-coded with each quarter-pel
     * neighbour of its vector and kept if the coder's price prefers it, and
     * that is worth 0.2-0.5 points of BD-rate.  A B macroblock has up to two
     * vectors per partition, so each list of each partition is refined with
     * the others held where the search left them -- the same one-at-a-time
     * rule the P path uses for its two partitions.  Only the shapes the P path
     * refines are refined (16x16, 16x8, 8x16): four groups times two lists is
     * sixteen more full candidates per macroblock for a shape that the RD
     * comparison rejects almost everywhere.
     *
     * A *direct* macroblock is excluded, and not because of the cost: its
     * vectors are derived from the co-located picture and are not in the
     * bitstream at all, so moving one changes the encoder's reconstruction and
     * nothing else -- a silent desync from the first macroblock of the picture
     * on (the decoder derives 12,8 where the encoder then predicts with 11,7).
     */
    if (e->p.cabac && e->cur_is_b && inter && e->p.refine_mv > 0 &&
        best.type != AEG_MB_B_DIRECT && mb_part_count(&best) <= 2) {
        static const int dx[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
        static const int dy[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
        static const int cross[4] = { 1, 3, 4, 6 };
        const int cheap = e->p.refine_mv < 2;
        const int nn = cheap ? 4 : 8;
        const int npart = mb_part_count(&best);
        mbcand save = e->mb_cand[0];
        const int mb_qp = c->qp;
        int p, l, j, b;

        for (p = 0; p < npart; p++) {
            int geo[4], b0;
            mb_part_geom(&best, p, geo, &b0);
            for (l = 0; l < 2; l++) {
                if (!(mb_part_lists(&best, p) & (1 << l)))
                    continue;
                for (j = 0; j < nn; j++) {
                    int k = cheap ? cross[j] : j;
                    int64_t cost;
                    e->mb_cand[0] = save;
                    e->mb_cand[0].type = best.type;
                    e->mb_cand[0].nparts = best.nparts;
                    memcpy(e->mb_cand[0].sub_type, best.sub_type,
                           sizeof(e->mb_cand[0].sub_type));
                    memcpy(e->mb_cand[0].blist, best.blist,
                           sizeof(e->mb_cand[0].blist));
                    for (b = 0; b < 16; b++) {
                        e->mb_cand[0].mv[b][0] = best.mv[b][0];
                        e->mb_cand[0].mv[b][1] = best.mv[b][1];
                        e->mb_cand[0].mv1[b][0] = best.mv1[b][0];
                        e->mb_cand[0].mv1[b][1] = best.mv1[b][1];
                        e->mb_cand[0].pmv[b][0] = best.pmv[b][0];
                        e->mb_cand[0].pmv[b][1] = best.pmv[b][1];
                        e->mb_cand[0].pmv1[b][0] = best.pmv1[b][0];
                        e->mb_cand[0].pmv1[b][1] = best.pmv1[b][1];
                    }
                    for (b = 0; b < 16; b++) {
                        int bx = (b & 3) * 4, by = (b >> 2) * 4;
                        if (bx < geo[0] || bx >= geo[0] + geo[2] ||
                            by < geo[1] || by >= geo[1] + geo[3])
                            continue;
                        if (l == 0) {
                            e->mb_cand[0].mv[b][0] = best.mv[b][0] + dx[k];
                            e->mb_cand[0].mv[b][1] = best.mv[b][1] + dy[k];
                        } else {
                            e->mb_cand[0].mv1[b][0] = best.mv1[b][0] + dx[k];
                            e->mb_cand[0].mv1[b][1] = best.mv1[b][1] + dy[k];
                        }
                    }
                    cand_enc(e, 0, c->qp, &retry);
                    refresh_pmvs(e, mbx, mby, &retry);
                    for (b = 0; b < 16; b++) {
                        e->mb_cand[0].pmv[b][0] = retry.pmv[b][0];
                        e->mb_cand[0].pmv[b][1] = retry.pmv[b][1];
                        e->mb_cand[0].pmv1[b][0] = retry.pmv1[b][0];
                        e->mb_cand[0].pmv1[b][1] = retry.pmv1[b][1];
                    }
                    cost = rd_luma(e, mbx, mby, &t, mb_qp, best.type, 0, lambda, 0);
                    cost = full_price(e, mbx, mby, &t, cost, c->chroma_mode, lambda, best_cost);
                    if (cost < best_cost) {
                        best_cost = cost;
                        best = t;
                        snap_luma(e, mbx, mby, snap_l);
                    }
                    c->qp = mb_qp;
                }
            }
        }
        e->mb_cand[0] = save;
        restore_luma(e, mbx, mby, snap_l);
        refresh_pmvs(e, mbx, mby, &best);
        c->qp = mb_qp;
    }

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
                t.chroma_mode = cm;
                cost = rd_price(e, mbx, mby, &t, cost, lambda);
                if (cost < best_chroma) {
                    best_chroma = cost;
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
    /* The skipped macroblock, priced as what it is.  The candidates above
     * are compared on luma alone, each with whatever residual its
     * quantiser and level selection leave, so the cheapest macroblock there
     * is -- the skip flag, no header, no residual in either plane -- is only
     * reached when every block of both planes happens to quantise to
     * nothing.  A residual that buys back a little of the quantisation error
     * (a picture that repeats the previous one, re-coded at the same QP)
     * always wins the luma comparison and still costs more than it is
     * worth once the header and the chroma are added; x264 avoids it with
     * its early skip test.  Here the whole macroblock is priced both ways. */
    if (e->p.cabac && (e->cur_is_p || e->cur_is_b) && best.cbp != 0 &&
        skip_check_on()) {
        mbenc s;
        int ok = 0;

        if (e->cur_is_p) {
            int sx, sy, b;

            reset_mb_enc(&s, c->qp);
            s.type = AEG_MB_P16x16;
            s.nparts = 1;
            pskip_mv(e, mbx, mby, &sx, &sy);
            for (b = 0; b < 16; b++) {
                s.mv[b][0] = (int16_t)sx;
                s.mv[b][1] = (int16_t)sy;
                s.pmv[b][0] = (int16_t)sx;
                s.pmv[b][1] = (int16_t)sy;
            }
            ok = 1;
        } else {
            int k;

            for (k = 0; k < e->mb_ncand; k++)
                if (e->mb_cand[k].type == AEG_MB_B_DIRECT) {
                    cand_enc(e, k, c->qp, &s);
                    ok = 1;
                    break;
                }
        }
        if (ok) {
            int64_t full = rd_price(e, mbx, mby, &best, 0, lambda);
            int64_t sk;
            uint8_t cbuf[128];
            int pl, x, y;

            inter_predict_typed(e, mbx, mby, &s);
            s.chroma_mode = 0;
            s.cbp = 0;
            s.sse = 0;
            for (y = 0; y < 16; y++)
                for (x = 0; x < 16; x++) {
                    int d = (int)e->pred_l[y * 16 + x] -
                            (int)e->srcp[0][(mby * 16 + y) * e->stride[0] +
                                            mbx * 16 + x];
                    s.sse += d * d;
                }
            for (pl = 0; pl < 2; pl++)
                for (y = 0; y < 8; y++)
                    for (x = 0; x < 8; x++) {
                        int d = (int)e->pred_c[pl][y * 8 + x] -
                                (int)e->srcp[1 + pl][(mby * 8 + y) *
                                                     e->stride[1] +
                                                     mbx * 8 + x];
                        s.sse += d * d;
                        cbuf[pl * 64 + y * 8 + x] = e->pred_c[pl][y * 8 + x];
                    }
            sk = rd_price(e, mbx, mby, &s, 0, lambda);
            if (sk < full) {
                best = s;
                restore_luma(e, mbx, mby, e->pred_l);
                restore_chroma(e, mbx, mby, cbuf);
            }
        }
    }
    *c = best;
    /* Hand the macroblock to the shadow coder: its context states, its
     * mb_qp_delta history and its per-block motion vectors are what the next
     * macroblock's decisions are priced against. */
    if (e->p.cabac)
        rd_cabac_bits(e, mbx, mby, c, cur_slice_kind(e), 1);
    if (getenv("AEG_MBDBG") && atoi(getenv("AEG_MBDBG")) == 2 &&
        e->cur_is_p) {
        int q, n = mb_part_count(&best);
        fprintf(stderr, "F%d MB %d,%d type %d cbp %d", e->frame_count, mbx,
                mby, best.type, best.cbp);
        if (is_inter_type(best.type))
            for (q = 0; q < n; q++) {
                int geo[4], b0;
                mb_part_geom(&best, q, geo, &b0);
                fprintf(stderr, " [b%d r%d mv %d,%d pmv %d,%d]", b0,
                        best.ref[b0], best.mv[b0][0], best.mv[b0][1],
                        best.pmv[b0][0], best.pmv[b0][1]);
            }
        {
            int b;
            fprintf(stderr, " refs ");
            for (b = 0; b < 16; b++)
                fprintf(stderr, "%d", best.ref[b]);
        }
        fprintf(stderr, "\n");
    }
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
        int p;
        aeg_bw_ue(bw, (uint32_t)inter_mb_type_code(c->type));
        if (c->type == AEG_MB_P8x8)
            /* sub_mb_type: four ue(v) symbols, before any mvd (7.3.5.2). */
            for (p = 0; p < 4; p++)
                aeg_bw_ue(bw, (uint32_t)c->sub_type[p]);
        /* ref_idx_l0: one per partition (P_8x8: one per sub-macroblock) when
         * NumRefIdxL0Active>1.  te(v) with cMax = 1 is a single *inverted*
         * bit (7.4.2.10 / 9.1: value = !read_bits(1)), not ue(v) -- the two
         * only agree for 0, which is why a stream that never picked the
         * second reference decoded fine. */
        if (e->cur_l0_active > 1) {
            if (c->type == AEG_MB_P8x8) {
                for (p = 0; p < 4; p++) {
                    int b0 = (p & 1) * 2 + (p >> 1) * 8;
                    aeg_bw_put(bw, 1, c->ref[b0] ? 0u : 1u);
                }
            } else {
                for (p = 0; p < c->nparts; p++) {
                    int geo[4], b0;
                    mb_part_geom(c, p, geo, &b0);
                    aeg_bw_put(bw, 1, c->ref[b0] ? 0u : 1u);
                }
            }
        }
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

/* ------------------------------------------------------------------ */
/* CABAC macroblock syntax (7.3.5, binarizations of 9.3.2)             */
/* ------------------------------------------------------------------ */
/* Every syntax element of the macroblock layer has a fixed context index
 * offset (Table 9-34) plus a context index increment computed from the
 * neighbours; the residual blocks use the three per-category offsets of
 * Tables 9-39 to 9-41.  The whole table is spelled out here so that a wrong
 * context is one number to check rather than an expression at a call site. */
enum {
    CAB_I16_NXN     = 3,    /* I slice: I_NxN vs Intra_16x16           */
    CAB_I16_CBPL    = 6,    /* Intra_16x16: cbp_luma != 0              */
    CAB_I16_CBPC    = 7,    /* Intra_16x16: chroma cbp, and            */
    CAB_I16_CBPC2   = 8,    /*   whether it is 2                       */
    CAB_I16_PRED    = 9,    /* Intra_16x16 prediction mode, two bits   */
    CAB_SKIP        = 11,   /* P: mb_skip_flag                         */
    CAB_P_TYPE      = 14,   /* P: intra or not                         */
    CAB_P_SHAPE     = 15,   /* P: 16x16/8x8 vs 16x8/8x16               */
    CAB_P_8X8       = 16,   /* P: 8x8                                  */
    CAB_P_16X8      = 17,   /* P: 16x8 vs 8x16                         */
    CAB_P_I16       = 17,   /* P: Intra_16x16 fields, offset by 14     */
    CAB_SUB_TYPE    = 21,   /* P: sub_mb_type                          */
    /* Not an mb_type: the sentinel that makes the B mb_type tree take its
     * intra branch (four-bit code 13).  It has to be distinct from every
     * Table 7-14 index the writer can produce, 13 (B_L0_Bi_8x16) included. */
    CAB_B_TYPE_INTRA = 23,
    CAB_BSUB_DIRECT = 36,   /* B: sub_mb_type, "direct"                */
    CAB_BSUB_L0L1   = 37,   /* B: sub_mb_type, L0/L1 8x8 vs the rest   */
    CAB_BSUB_SHAPE  = 38,   /* B: sub_mb_type, 8x4/4x8/4x4 family      */
    CAB_BSUB_TAIL   = 39,   /* B: sub_mb_type, the shared tail bin     */
    CAB_MVD_X       = 40,   /* P: |mvd| horizontal                     */
    CAB_MVD_Y       = 47,   /* P: |mvd| vertical                       */
    CAB_QP_DELTA    = 60,   /* mb_qp_delta                             */
    CAB_CHROMA_PRED = 64,   /* intra_chroma_pred_mode                  */
    CAB_I4_PREV     = 68,   /* prev_intra_pred_mode_flag               */
    CAB_I4_REM      = 69,   /* rem_intra_pred_mode                     */
    CAB_CBP_LUMA    = 73,   /* coded_block_pattern luma                */
    CAB_CBP_CHROMA  = 77,   /* coded_block_pattern chroma              */
    CAB_T8          = 399   /* transform_size_8x8_flag                 */
};

/* Intra_16x16 / I_NxN, which is also the intra suffix of a P macroblock: the
 * two slices number those contexts differently (they are 14 apart). */
static const uint8_t cab_i16_ctx_i[6] = { 3, 6, 7, 8, 9, 10 };
static const uint8_t cab_i16_ctx_p[6] = { 17, 18, 19, 19, 20, 20 };

/* Context base of the coded_block_flag of each residual category (Table 9-40). */
static const uint16_t cab_cbf[6] = { 85, 89, 93, 97, 101, 460 };

/* coded_block_flag context: which of the two neighbours has coefficients,
 * with the spec's rule for a neighbour that is not available (outside the
 * slice): an intra macroblock assumes it is coded, an inter one assumes not. */
static int cab_cbf_ctx(int cat, int a, int b, int va, int vb, int intra)
{
    int na = va ? (a > 0) : intra;
    int nb = vb ? (b > 0) : intra;
    return cab_cbf[cat] + na + 2 * nb;
}

/* Neighbour non-zero counts of luma 4x4 block `blk` (raster order inside the
 * macroblock), for the coded_block_flag contexts. */
static void cab_nz4_luma(struct aeg_encoder *e, int mbx, int mby, const mbenc *c,
                         int blk, int *a, int *b, int *va, int *vb)
{
    int bx = blk & 3, by = blk >> 2;

    if (bx) { *a = c->nz[blk - 1]; *va = 1; }
    else if (mbx) {
        *a = e->mbs[mby * e->mb_w + mbx - 1].nz[4 * by + 3];
        *va = 1;
    } else { *a = 0; *va = 0; }
    if (by) { *b = c->nz[blk - 4]; *vb = 1; }
    else if (mby > e->top_row) {
        *b = e->mbs[(mby - 1) * e->mb_w + mbx].nz[12 + bx];
        *vb = 1;
    } else { *b = 0; *vb = 0; }
}

/* The same for chroma AC block `k` (0..3, raster order) of plane `pl`. */
static void cab_nz4_chroma(struct aeg_encoder *e, int mbx, int mby,
                           const mbenc *c, int pl, int k,
                           int *a, int *b, int *va, int *vb)
{
    int base = AEG_BLK_CHROMA_CB + pl * 4;
    int bx = k & 1, by = k >> 1;

    if (bx) { *a = c->nz[base + k - 1]; *va = 1; }
    else if (mbx) {
        *a = e->mbs[mby * e->mb_w + mbx - 1].nz[base + k + 1];
        *va = 1;
    } else { *a = 0; *va = 0; }
    if (by) { *b = c->nz[base + k - 2]; *vb = 1; }
    else if (mby > e->top_row) {
        *b = e->mbs[(mby - 1) * e->mb_w + mbx].nz[base + k + 2];
        *vb = 1;
    } else { *b = 0; *vb = 0; }
}

/* The same for one whole-macroblock block (luma or chroma DC). */
static void cab_nz_mb(struct aeg_encoder *e, int mbx, int mby, int blk,
                      int *a, int *b, int *va, int *vb)
{
    if (mbx) {
        *a = e->mbs[mby * e->mb_w + mbx - 1].nz[blk];
        *va = 1;
    } else { *a = 0; *va = 0; }
    if (mby > e->top_row) {
        *b = e->mbs[(mby - 1) * e->mb_w + mbx].nz[blk];
        *vb = 1;
    } else { *b = 0; *vb = 0; }
}


/* Intra_16x16 or I_NxN.  `ctx` is the six offsets of the slice type and
 * `cond` the neighbour condition of the first bin (always 0 in a P slice). */
static void cabac_type_intra(aeg_cabac *cb, const uint8_t *ctx, int cond,
                             const mbenc *c)
{
    int lc = luma_cbp_on(c) ? 1 : 0, cc = chroma_cbp(c);

    if (c->type != AEG_MB_I16x16) {
        aeg_cabac_decision(cb, ctx[0] + cond, 0);
        return;
    }
    aeg_cabac_decision(cb, ctx[0] + cond, 1);
    aeg_cabac_terminate(cb, 0);              /* not I_PCM */
    aeg_cabac_decision(cb, ctx[1], lc);
    aeg_cabac_decision(cb, ctx[2], cc != 0);
    if (cc)
        aeg_cabac_decision(cb, ctx[3], cc == 2);
    aeg_cabac_decision(cb, ctx[4], (c->i16_mode >> 1) & 1);
    aeg_cabac_decision(cb, ctx[5], c->i16_mode & 1);
}

/* prev_intra4x4_pred_mode_flag / rem_intra4x4_pred_mode, which Intra_8x8 uses
 * as well: the mode is replaced by the predicted one, or sent as three bits
 * with the predicted value skipped. */
static void cabac_pred_mode(aeg_cabac *cb, int pred, int mode)
{
    if (mode == pred) {
        aeg_cabac_decision(cb, CAB_I4_PREV, 1);
        return;
    }
    aeg_cabac_decision(cb, CAB_I4_PREV, 0);
    mode = mode < pred ? mode : mode - 1;
    aeg_cabac_decision(cb, CAB_I4_REM, mode & 1);
    aeg_cabac_decision(cb, CAB_I4_REM, (mode >> 1) & 1);
    aeg_cabac_decision(cb, CAB_I4_REM, (mode >> 2) & 1);
}

/* intra_chroma_pred_mode: one bin for "not DC", then a two-bit tree. */
static void cabac_chroma_mode(struct aeg_encoder *e, aeg_cabac *cb, int mbx,
                              int mby, const mbenc *c)
{
    int ctx = 0, m = c->chroma_mode;
    int idx = mby * e->mb_w + mbx;

    if (mbx && is_intra_type(e->mbs[idx - 1].type) &&
        e->mbs[idx - 1].chroma_mode != 0)
        ctx++;
    if (mby > e->top_row && is_intra_type(e->mbs[idx - e->mb_w].type) &&
        e->mbs[idx - e->mb_w].chroma_mode != 0)
        ctx += 1;
    aeg_cabac_decision(cb, CAB_CHROMA_PRED + ctx, m != 0);
    if (m != 0) {
        aeg_cabac_decision(cb, CAB_CHROMA_PRED + 3, m != 1);
        if (m != 1)
            aeg_cabac_decision(cb, CAB_CHROMA_PRED + 3, m != 2);
    }
}

/* transform_size_8x8_flag: the context counts the neighbours that used the 8x8
 * transform, one each (9.3.3.1.1.13). */
static int cab_t8_ctx(struct aeg_encoder *e, int mbx, int mby)
{
    int idx = mby * e->mb_w + mbx, ctx = 0;

    if (mbx && e->mbs[idx - 1].t8)
        ctx++;
    if (mby > e->top_row && e->mbs[idx - e->mb_w].t8)
        ctx++;
    return CAB_T8 + ctx;
}

/* One motion vector difference: the |mvd| > 0 flag with its neighbour-based
 * context, a unary prefix, then the Exp-Golomb escape and the sign. */
static void cabac_mvd(aeg_cabac *cb, int base, int amvd, int mvd)
{
    int abs = mvd < 0 ? -mvd : mvd, i;
    int ctx = base + (amvd > 2) + (amvd > 32);

    if (!abs) {
        aeg_cabac_decision(cb, ctx, 0);
        return;
    }
    aeg_cabac_decision(cb, ctx, 1);
    {
        int off = 3;
        for (i = 1; i < 9; i++) {
            int more = i < abs;
            aeg_cabac_decision(cb, base + off, more);
            if (!more)
                goto sign;
            if (i < 4)
                off++;
        }
    }
    {
        int rest = abs - 9, p = 0;
        while (rest >= (1 << (3 + p))) {
            rest -= 1 << (3 + p);
            p++;
        }
        for (i = 0; i < p; i++)
            aeg_cabac_bypass(cb, 1);
        aeg_cabac_bypass(cb, 0);
        for (i = 3 + p - 1; i >= 0; i--)
            aeg_cabac_bypass(cb, (rest >> i) & 1);
    }
sign:
    aeg_cabac_bypass(cb, mvd < 0);
}

/* The mvd of every partition of an inter macroblock for prediction list
 * `list`, and the per-4x4-block mvd the neighbours' contexts are built from.
 *
 * The context of a difference depends on the absolute differences of the left
 * and top neighbours *of the same list* (9.3.3.1.1.7), which is why the table
 * is per list and why a macroblock that does not predict from a list leaves
 * zeros in it (a direct macroblock, or a B macroblock that only uses list 0). */
static void cabac_mb_mvd(struct aeg_encoder *e, aeg_cabac *cb, int mbx,
                         int mby, mbenc *c, int list)
{
    int idx = mby * e->mb_w + mbx;
    int16_t mvd[16][2];
    const int16_t (*cmv)[2] = list ? c->mv1 : c->mv;
    const int16_t (*cpmv)[2] = list ? c->pmv1 : c->pmv;
    int p;

    memset(mvd, 0, sizeof(mvd));
    for (p = 0; p < c->nparts; p++) {
        int geo[4], b0, x, y, ax, ay;
        int lx = 0, ly = 0, tx = 0, ty = 0;
        int bx, by;

        /* A partition of a B macroblock only sends a difference for the lists
         * it predicts from: the others have no vector at all, and the decoder
         * leaves both the vector and its difference at zero.  A B_8x8 group
         * coded as direct sends none at all, for either list. */
        if (!mb_part_sends_mvd(c, p) ||
            (is_b_type(c->type) && !(mb_part_lists(c, p) & (1 << list))))
            continue;
        mb_part_geom(c, p, geo, &b0);
        bx = geo[0] / 4;
        by = geo[1] / 4;
        if (bx) {
            lx = mvd[by * 4 + bx - 1][0];
            ly = mvd[by * 4 + bx - 1][1];
        } else if (mbx) {
            lx = e->mbs[idx - 1].mvd[list][by * 4 + 3][0];
            ly = e->mbs[idx - 1].mvd[list][by * 4 + 3][1];
        }
        if (by) {
            tx = mvd[(by - 1) * 4 + bx][0];
            ty = mvd[(by - 1) * 4 + bx][1];
        } else if (mby > e->top_row) {
            tx = e->mbs[idx - e->mb_w].mvd[list][12 + bx][0];
            ty = e->mbs[idx - e->mb_w].mvd[list][12 + bx][1];
        }
        ax = (lx < 0 ? -lx : lx) + (tx < 0 ? -tx : tx);
        ay = (ly < 0 ? -ly : ly) + (ty < 0 ? -ty : ty);
        cabac_mvd(cb, CAB_MVD_X, ax, cmv[b0][0] - cpmv[b0][0]);
        cabac_mvd(cb, CAB_MVD_Y, ay, cmv[b0][1] - cpmv[b0][1]);
        for (y = geo[1] / 4; y < (geo[1] + geo[3]) / 4; y++)
            for (x = geo[0] / 4; x < (geo[0] + geo[2]) / 4; x++) {
                mvd[y * 4 + x][0] = (int16_t)(cmv[b0][0] - cpmv[b0][0]);
                mvd[y * 4 + x][1] = (int16_t)(cmv[b0][1] - cpmv[b0][1]);
            }
    }
    memcpy(e->mbs[idx].mvd[list], mvd, sizeof(mvd));
}

/* ref_idx_l0 for P with two active references (CABAC 9.3.2, ctx 54).
 * One bin when NumRefIdxActive==2: 0 or 1 with ctx (refA>0)+(refB>0)*2.
 * The neighbour refs are the 4x4 blocks at (x0-1,y0) and (x0,y0-1): inside the
 * same macroblock they are the previous partitions' refs, otherwise the left/
 * top macroblock's ref at the edge.  Unavailable/intra is 0. */
static void cabac_mb_ref_idx(struct aeg_encoder *e, aeg_cabac *cb, int mbx,
                             int mby, const mbenc *c)
{
    if (e->cur_l0_active <= 1 || is_intra_type(c->type))
        return;
    /* P_8x8 has 4 sub-macroblocks (one ref each), otherwise partitions. */
    int idx = mby * e->mb_w + mbx;
    int n = (c->type == AEG_MB_P8x8) ? 4 : mb_part_count(c);
    uint8_t cur_ref[16] = {0};
    /* Seed with already coded parts of this MB for intra-MB neighbour lookup. */
    for (int i = 0; i < 16; i++)
        cur_ref[i] = 0;
    for (int p = 0; p < n; p++) {
        int geo[4], b0, x0, y0, ctx = 0;
        int refa = 0, refb = 0;
        if (c->type == AEG_MB_P8x8) {
            int bx = (p & 1) * 2, by = (p >> 1) * 2;
            geo[0] = bx * 4; geo[1] = by * 4; geo[2] = 8; geo[3] = 8;
            b0 = by * 4 + bx;
        } else {
            mb_part_geom(c, p, geo, &b0);
        }
        x0 = geo[0] / 4; y0 = geo[1] / 4;
        int ref = c->ref[b0] & 1;
        /* left: inside MB or left MB */
        if (x0 > 0) {
            refa = cur_ref[y0 * 4 + x0 - 1];
        } else if (mbx > 0) {
            const aeg_mb *lm = &e->mbs[idx - 1];
            if (!is_intra_type(lm->type))
                refa = lm->ref_idx[y0 * 4 + 3];
        }
        /* top: inside MB or top MB */
        if (y0 > 0) {
            refb = cur_ref[(y0 - 1) * 4 + x0];
        } else if (mby > e->top_row) {
            const aeg_mb *tm = &e->mbs[idx - e->mb_w];
            if (!is_intra_type(tm->type))
                refb = tm->ref_idx[12 + x0];
        }
        if (refa > 0) ctx++;
        if (refb > 0) ctx += 2;
        /* CABAC binarises ref_idx as plain unary (U, Table 9-34) whatever
         * num_ref_idx_active is -- no truncation at cMax -- so 1 is the two
         * bins 1,0 and the second one uses ctxIdxInc 4 (9.3.3.1.2). */
        aeg_cabac_decision(cb, 54 + ctx, ref);
        if (ref)
            aeg_cabac_decision(cb, 54 + 4, 0);
        /* Publish this partition's ref for following partitions' context. */
        if (c->type == AEG_MB_P8x8) {
            for (int dy = 0; dy < 2; dy++)
                for (int dx = 0; dx < 2; dx++)
                    cur_ref[(y0 + dy) * 4 + x0 + dx] = (uint8_t)ref;
        } else {
            for (int dy = 0; dy < geo[3] / 4; dy++)
                for (int dx = 0; dx < geo[2] / 4; dx++)
                    cur_ref[(y0 + dy) * 4 + x0 + dx] = (uint8_t)ref;
        }
    }
}

/* coded_block_pattern, luma half: four 8x8 groups, each conditioned on the
 * neighbours' patterns (Table 9-39). */
static void cabac_cbp_luma(struct aeg_encoder *e, aeg_cabac *cb, int mbx,
                           int mby, const mbenc *c)
{
    int idx = mby * e->mb_w + mbx;
    int a = mbx ? e->mbs[idx - 1].cbp : 0;
    int b = (mby > e->top_row) ? e->mbs[idx - e->mb_w].cbp : 0;
    int cbp = c->cbp & 15;

    /* A neighbour outside the slice counts as "all four groups coded" (its
     * low nibble is 0x0f) whichever kind of macroblock this is. */
    if (!mbx) a = 0x0f;
    if (mby <= e->top_row) b = 0x0f;
    /* Each bin's context reads the groups decoded *so far* on this side, not
     * the finished pattern (9.3.3.1.1.9): the decoder only knows the lower
     * groups when it has to pick the context. */
    {
        int done = 0, i;
        for (i = 0; i < 4; i++) {
            int bit = (cbp >> i) & 1;
            int ctx;
            switch (i) {
            case 0: ctx = !(a & 0x02) + 2 * !(b & 0x04); break;
            case 1: ctx = !(done & 0x01) + 2 * !(b & 0x08); break;
            case 2: ctx = !(a & 0x08) + 2 * !(done & 0x01); break;
            default: ctx = !(done & 0x04) + 2 * !(done & 0x02); break;
            }
            aeg_cabac_decision(cb, CAB_CBP_LUMA + ctx, bit);
            done |= bit << i;
        }
    }
}

static void cabac_cbp_chroma(struct aeg_encoder *e, aeg_cabac *cb, int mbx,
                             int mby, const mbenc *c)
{
    int idx = mby * e->mb_w + mbx;
    /* The neighbours' cbp is stored as two bits (chroma DC, chroma AC), while
     * the context derivation wants the syntax element's value 0..2. */
    int a = mbx ? packed_chroma_cbp(e->mbs[idx - 1].cbp) : 0;
    int b = (mby > e->top_row) ? packed_chroma_cbp(e->mbs[idx - e->mb_w].cbp) : 0;
    int cc = chroma_cbp(c);

    /* A neighbour outside the slice counts as having cbp_chroma = 0: the
     * decoder's stand-in for an unavailable neighbour is 0x7CF (or 0x00F),
     * whose chroma bits are both clear. */
    if (!mbx) a = 0;
    if (mby <= e->top_row) b = 0;
    aeg_cabac_decision(cb, CAB_CBP_CHROMA + (a > 0) + 2 * (b > 0), cc != 0);
    if (cc)
        aeg_cabac_decision(cb, CAB_CBP_CHROMA + 4 + (a == 2) + 2 * (b == 2),
                           cc == 2);
}

/* The residual of one macroblock, in the order the spec decodes it: luma
 * (DC then AC, or 8x8 groups, or 4x4 groups), then chroma DC, then chroma
 * AC. */
static void cabac_mb_residual(struct aeg_encoder *e, aeg_cabac *cb, int mbx,
                              int mby, mbenc *c)
{
    int intra = is_intra_type(c->type);
    int16_t blk[64];
    int i, g, k;

    if (c->type == AEG_MB_I16x16) {
        int a, b, va, vb;
        for (i = 0; i < 16; i++)
            blk[i] = c->level[AEG_BLK_LUMA_DC][aeg_scan4x4[i]];
        cab_nz_mb(e, mbx, mby, AEG_BLK_LUMA_DC, &a, &b, &va, &vb);
        c->nz[AEG_BLK_LUMA_DC] =
            (int8_t)aeg_cabac_residual(cb, blk, 16, 0, cab_cbf_ctx(0, a, b, va, vb, 1));
        if (luma_cbp_on(c)) {
            for (k = 0; k < 16; k++) {
                int nb = ac_ord[k];
                cab_nz4_luma(e, mbx, mby, c, nb, &a, &b, &va, &vb);
                for (i = 0; i < 15; i++)
                    blk[i] = c->level[nb][aeg_scan4x4[i + 1]];
                c->nz[nb] = (int8_t)aeg_cabac_residual(
                    cb, blk, 15, 1, cab_cbf_ctx(1, a, b, va, vb, intra));
            }
        } else {
            /* The DC block still had its coefficients: only the sixteen AC
             * counts are zero here. */
            memset(c->nz + AEG_BLK_LUMA, 0, 16);
        }
    } else if (c->t8) {
        for (g = 0; g < 4; g++) {
            int cnt = 0;
            if (c->cbp & (1 << g)) {
                for (i = 0; i < 64; i++)
                    blk[i] = c->level8[g][aeg_zigzag8x8[i]];
                /* No coded_block_flag: the coded block pattern already said
                 * this group is coded. */
                cnt = aeg_cabac_residual(cb, blk, 64, 5, -1);
            }
            for (k = 0; k < 4; k++)
                c->nz[group_blk[g][k]] = (int8_t)(cnt ? cnt : 0);
        }
    } else {
        for (g = 0; g < 4; g++) {
            if (!(c->cbp & (1 << g)))
                continue;
            for (k = 0; k < 4; k++) {
                int nb = group_blk[g][k], a, b, va, vb;
                cab_nz4_luma(e, mbx, mby, c, nb, &a, &b, &va, &vb);
                for (i = 0; i < 16; i++)
                    blk[i] = c->level[nb][aeg_scan4x4[i]];
                c->nz[nb] = (int8_t)aeg_cabac_residual(
                    cb, blk, 16, 2, cab_cbf_ctx(2, a, b, va, vb, intra));
            }
        }
    }
    if (chroma_cbp(c)) {
        int pl, cc = chroma_cbp(c), a, b, va, vb;
        for (pl = 0; pl < 2; pl++) {
            for (i = 0; i < 4; i++)
                blk[i] = c->level[AEG_BLK_CHROMA_DC + pl][i];
            cab_nz_mb(e, mbx, mby, AEG_BLK_CHROMA_DC + pl, &a, &b, &va, &vb);
            c->nz[AEG_BLK_CHROMA_DC + pl] = (int8_t)aeg_cabac_residual(
                cb, blk, 4, 3, cab_cbf_ctx(3, a, b, va, vb, intra));
        }
        if (cc == 2)
            for (pl = 0; pl < 2; pl++)
                for (k = 0; k < 4; k++) {
                    int nb = AEG_BLK_CHROMA_CB + pl * 4 + k;
                    cab_nz4_chroma(e, mbx, mby, c, pl, k,
                                   &a, &b, &va, &vb);
                    for (i = 0; i < 15; i++)
                        blk[i] = c->level[nb][aeg_scan4x4[i + 1]];
                    c->nz[nb] = (int8_t)aeg_cabac_residual(
                        cb, blk, 15, 4, cab_cbf_ctx(4, a, b, va, vb, intra));
                }
    }
}

/* The B intra contexts: mb_type of a B slice that turned out to be intra uses
 * the "non-I slice" base 32 (Table 9-34), six contexts wide like the P one. */
static const uint8_t cab_i16_ctx_b[6] = { 32, 33, 34, 34, 35, 35 };

/* mb_type of a macroblock in a B slice, as the index of Table 7-14:
 *   0 B_Direct_16x16, 1 B_L0_16x16, 2 B_L1_16x16, 3 B_L0_L0_16x16 (both lists).
 * Those four are the whole list the analysis produces (analyse_mb_b gathers one
 * 16x16 candidate per list and one bi-predictive one), so a partitioned B
 * macroblock is -1: it would need its own entries in AEG_MB_* first, and the
 * writer refuses rather than emitting a type that does not match the picture. */
static int b_mb_type_code(const mbenc *c)
{
    /* Table 7-14 codes 4..21: the two partitioned shapes share each pair of
     * codes, keyed by the lists the two partitions picked (partition 0 first).
     * The even code of a pair is the 16x8 shape and the odd one the 8x16. */
    static const uint8_t parts[3][3] = {
        /*            part1: L0  L1  BI */
        /* part0 L0 */ {  4,  8, 12 },
        /* part0 L1 */ { 10,  6, 14 },
        /* part0 BI */ { 16, 18, 20 },
    };

    if (c->type == AEG_MB_B_DIRECT)
        return 0;
    if (c->type == AEG_MB_B_8x8)
        return 22;
    if (c->type == AEG_MB_B16x8 || c->type == AEG_MB_B8x16) {
        int l0 = c->blist[0] ? c->blist[0] : 3;
        int l1 = c->blist[1] ? c->blist[1] : 3;

        return parts[l0 - 1][l1 - 1] +
               (c->type == AEG_MB_B8x16 ? 1 : 0);
    }
    if (c->nparts > 1)
        return -1;
    if (c->blist[0] == 1)
        return 1;
    if (c->blist[0] == 2)
        return 2;
    return 3;
}

/* Write that index with the binarization of Table 9-34.  Context 27+ctx
 * separates direct mode -- ctx counts the neighbours that are *not* direct, so
 * a run of direct macroblocks is cheap -- then 27+3 separates the single-list
 * 16x16 types from the rest, and the rest is a four-bit code read off 27+4 and
 * 27+5, with two more bins after the codes 12..14 (of which only 13 is intra,
 * which is why an intra macroblock in a B slice is reached through the mb_type
 * tree and not through a flag of its own). */
/* sub_mb_type of a B_8x8 macroblock (9.3.3.1.1.6).  The tree is: "direct?";
 * "L0/L1 8x8 or something else?"; then either the shape family (8x4/4x8/4x4,
 * with L0_4x4 and the two 4x4 list 1/bi values hanging off the far end) or,
 * for the 8x8 values, which list.  The four 8x8 values (0..3) are the ones the
 * search produces today; the rest of the tree is written out so that adding
 * the splits is a search change, not a syntax change. */
static void code_b_sub_mb_type(aeg_cabac *cb, int st)
{
    if (st == 0) {
        aeg_cabac_decision(cb, CAB_BSUB_DIRECT, 0);
        return;
    }
    aeg_cabac_decision(cb, CAB_BSUB_DIRECT, 1);
    if (st <= 2) {                              /* B_L0_8x8, B_L1_8x8 */
        aeg_cabac_decision(cb, CAB_BSUB_L0L1, 0);
        aeg_cabac_decision(cb, CAB_BSUB_TAIL, st - 1);
        return;
    }
    aeg_cabac_decision(cb, CAB_BSUB_L0L1, 1);
    if (st >= 11) {                             /* L1_4x4, Bi_4x4       */
        aeg_cabac_decision(cb, CAB_BSUB_SHAPE, 1);
        aeg_cabac_decision(cb, CAB_BSUB_TAIL, 1);
        aeg_cabac_decision(cb, CAB_BSUB_TAIL, st - 11);
        return;
    }
    if (st >= 7) {                              /* 4x8 splits, L0_4x4    */
        aeg_cabac_decision(cb, CAB_BSUB_SHAPE, 1);
        aeg_cabac_decision(cb, CAB_BSUB_TAIL, 0);
        aeg_cabac_decision(cb, CAB_BSUB_TAIL, ((st - 7) >> 1) & 1);
        aeg_cabac_decision(cb, CAB_BSUB_TAIL, (st - 7) & 1);
        return;
    }
    /* 3..6: B_Bi_8x8 followed by the three 8x4 values -- the tree puts the
     * bi 8x8 shape in front of the shape question, not behind it. */
    aeg_cabac_decision(cb, CAB_BSUB_SHAPE, 0);
    aeg_cabac_decision(cb, CAB_BSUB_TAIL, ((st - 3) >> 1) & 1);
    aeg_cabac_decision(cb, CAB_BSUB_TAIL, (st - 3) & 1);
}

static void code_b_mb_type(struct aeg_encoder *e, aeg_cabac *cb, int mbx,
                           int mby, int code)
{
    int idx = mby * e->mb_w + mbx;
    /* A neighbour outside the slice counts as direct: the decoder keeps its
     * "left_type" at zero for an unavailable macroblock and tests
     * IS_DIRECT(left_type - 1), which -1 satisfies in every bit pattern.  So
     * only a neighbour that *is* there and is not direct increments the
     * context. */
    int ld = !mbx || e->mbs[idx - 1].type == AEG_MB_B_DIRECT;
    int td = mby <= e->top_row || e->mbs[idx - e->mb_w].type == AEG_MB_B_DIRECT;
    int ctx = !ld + !td;
    int fourbit, fifth = 0, i;

    if (!code) {
        aeg_cabac_decision(cb, 27 + ctx, 0);
        return;
    }
    aeg_cabac_decision(cb, 27 + ctx, 1);
    if (code <= 2) {
        aeg_cabac_decision(cb, 27 + 3, 0);
        aeg_cabac_decision(cb, 27 + 5, code - 1);
        return;
    }
    aeg_cabac_decision(cb, 27 + 3, 1);
    /* The four-bit code the decoder reads next, and whether a fifth bin follows
     * it.  Codes 3..10 use their own four bits (0..7); 11 and B_8x8 (22) are
     * 14 and 15; 12..21 share a code in pairs with the fifth bin picking the
     * odd one; and intra is *not* one of those values at all but the four-bit
     * code 13, which the decoder takes before it would look for a fifth bin.
     *
     * That 13 is a code *value*, not an mb_type index, and mb_type 13 is
     * B_L0_Bi_8x16 -- so it must not be routed to the intra branch.  Telling
     * the two apart is what an "intra" sentinel above 22 is for; using 13 for
     * both is what once made a B_L0_Bi_8x16 macroblock decode as an intra one,
     * and the decoder said so in as many words ("left block unavailable for
     * requested intra4x4 mode") a hundred macroblocks later. */
    if (code == CAB_B_TYPE_INTRA) {
        fourbit = 13;                         /* intra */
    } else if (code >= 12 && code <= 21) {
        fourbit = 8 + (code - 12) / 2;
        fifth = 1;
    } else if (code == 11) {
        fourbit = 14;
    } else if (code == 22) {
        fourbit = 15;
    } else {
        fourbit = code - 3;                   /* 3..10 -> 0..7 */
    }
    aeg_cabac_decision(cb, 27 + 4, (fourbit >> 3) & 1);
    for (i = 2; i >= 0; i--)
        aeg_cabac_decision(cb, 27 + 5, (fourbit >> i) & 1);
    if (fifth)
        aeg_cabac_decision(cb, 27 + 5, (code - 12) & 1);
}

/* One macroblock of a CABAC slice.  `skip` is the P_Skip / B_Direct conversion
 * of the bitstream pass: a P_L0_16x16 macroblock with the predicted motion
 * vector and no residual reconstructs exactly like a skipped one, and a direct
 * macroblock without residual *is* a skipped one in a B slice. */
static void write_mb_cabac(struct aeg_encoder *e, aeg_cabac *cb, int mbx,
                           int mby, mbenc *c, int slice, int skip)
{
    int idx = mby * e->mb_w + mbx;
    int intra = is_intra_type(c->type);

    if (slice != AEG_SLICE_I) {
        /* mb_skip_flag: one context per neighbour that was *not* skipped
         * (9.3.3.1.1.1).  A neighbour outside the slice counts as skipped,
         * which is why the ternaries default to 1.  A B slice has its own
         * thirteen-wide block of skip contexts. */
        int ls, ts, base = CAB_SKIP + (slice == AEG_SLICE_B ? 13 : 0);

        if (slice == AEG_SLICE_B) {
            /* The same derivation as for P -- a neighbour that was skipped does
             * not increment the context, and a neighbour outside the slice
             * counts as skipped -- on top of the B slice's own block of
             * contexts thirteen entries further along (Table 9-34). */
            ls = mbx ? e->mbs[idx - 1].skip : 1;
            ts = (mby > e->top_row) ? e->mbs[idx - e->mb_w].skip : 1;
            base += !ls + !ts;
        } else {
            ls = mbx ? e->mbs[idx - 1].skip : 1;
            ts = (mby > e->top_row) ? e->mbs[idx - e->mb_w].skip : 1;
            base += !ls + !ts;
        }
        aeg_cabac_decision(cb, base, skip);
        if (skip) {
            /* A skipped macroblock has no residual, and its neighbours' coded
             * block flags have to see that. */
            memset(c->nz, 0, sizeof(c->nz));
            memset(c->nz8, 0, sizeof(c->nz8));
            return;
        }
    }
    if (slice == AEG_SLICE_B) {
        if (intra) {
            int cond = 0;
            /* Intra in a B slice is reached through the mb_type tree; the
             * four-bit code 13 is followed by the non-I slice intra
             * binarization (contexts 32..35). */
            code_b_mb_type(e, cb, mbx, mby, CAB_B_TYPE_INTRA);
            cabac_type_intra(cb, cab_i16_ctx_b, cond, c);
        } else {
            int code = b_mb_type_code(c);
            int lmask;
            if (code < 0) {
                aeg_set_last_error("B macroblock type is not implemented");
                return;
            }
            code_b_mb_type(e, cb, mbx, mby, code);
            lmask = mb_lists(c->type, c->blist);
            /* B_8x8 carries one sub_mb_type per 8x8 group, after mb_type and
             * before any mvd (7.3.5). */
            if (c->type == AEG_MB_B_8x8) {
                int g;

                for (g = 0; g < 4; g++)
                    code_b_sub_mb_type(cb, c->sub_type[g]);
            }
            /* A direct macroblock sends no mvd (8.4.1.2); every other one sends
             * a pair per list it predicts from, list 0 first (Table 7-16). */
            if (c->type != AEG_MB_B_DIRECT) {
                if (lmask & 1)
                    cabac_mb_mvd(e, cb, mbx, mby, c, 0);
                if (lmask & 2)
                    cabac_mb_mvd(e, cb, mbx, mby, c, 1);
            }
        }
    } else if (slice == AEG_SLICE_P) {
        if (intra) {
            aeg_cabac_decision(cb, CAB_P_TYPE, 1);
            cabac_type_intra(cb, cab_i16_ctx_p, 0, c);
        } else {
            aeg_cabac_decision(cb, CAB_P_TYPE, 0);
            if (c->type == AEG_MB_P16x16) {
                aeg_cabac_decision(cb, CAB_P_SHAPE, 0);
                aeg_cabac_decision(cb, CAB_P_8X8, 0);
            } else if (c->type == AEG_MB_P8x8) {
                aeg_cabac_decision(cb, CAB_P_SHAPE, 0);
                aeg_cabac_decision(cb, CAB_P_8X8, 1);
            } else {
                aeg_cabac_decision(cb, CAB_P_SHAPE, 1);
                aeg_cabac_decision(cb, CAB_P_16X8,
                                   c->type == AEG_MB_P16x8 ? 1 : 0);
            }
            if (c->type == AEG_MB_P8x8) {
                int p;
                for (p = 0; p < 4; p++) {
                    /* D_L0_8x8 = 0 with a single bin; 8x4, 4x8 and 4x4 use
                     * two more (Table 9-34). */
                    if (c->sub_type[p] == 0) {
                        aeg_cabac_decision(cb, CAB_SUB_TYPE, 1);
                    } else {
                        aeg_cabac_decision(cb, CAB_SUB_TYPE, 0);
                        aeg_cabac_decision(cb, CAB_SUB_TYPE + 1,
                                           c->sub_type[p] == 1 ? 0 : 1);
                        if (c->sub_type[p] != 1)
                            aeg_cabac_decision(cb, CAB_SUB_TYPE + 2,
                                               c->sub_type[p] == 2 ? 1 : 0);
                    }
                }
            }
            if (e->cur_l0_active > 1)
                cabac_mb_ref_idx(e, cb, mbx, mby, c);
            cabac_mb_mvd(e, cb, mbx, mby, c, 0);
        }
    } else {
        int cond = 0;
        if (mbx) {
            int t = e->mbs[idx - 1].type;
            cond += (t == AEG_MB_I16x16 || t == AEG_MB_IPCM);
        }
        if (mby > e->top_row) {
            int t = e->mbs[idx - e->mb_w].type;
            cond += (t == AEG_MB_I16x16 || t == AEG_MB_IPCM);
        }
        cabac_type_intra(cb, cab_i16_ctx_i, cond, c);
    }
    if (intra) {
        if (c->type == AEG_MB_I4x4 || c->type == AEG_MB_I8x8) {
            if (e->p.transform_8x8)
                aeg_cabac_decision(cb, cab_t8_ctx(e, mbx, mby),
                                   c->type == AEG_MB_I8x8);
            if (c->type == AEG_MB_I8x8) {
                int b;
                for (b = 0; b < 4; b++)
                    cabac_pred_mode(cb,
                                    intra8_pred_mode(e, mbx, mby, c->intra8,
                                                     b & 1, b >> 1),
                                    c->intra8[b]);
            } else {
                int b;
                for (b = 0; b < 16; b++) {
                    int rx = ((b >> 2) & 1) * 2 + (b & 1);
                    int ry = ((b >> 3) & 1) * 2 + ((b >> 1) & 1);
                    cabac_pred_mode(cb,
                                    intra4_pred_mode(e, mbx, mby, c->intra4,
                                                     rx, ry),
                                    c->intra4[b]);
                }
            }
        }
        cabac_chroma_mode(e, cb, mbx, mby, c);
    }
    if (c->type != AEG_MB_I16x16) {
        cabac_cbp_luma(e, cb, mbx, mby, c);
        cabac_cbp_chroma(e, cb, mbx, mby, c);
        /* transform_size_8x8_flag of an inter macroblock comes after the coded
         * block pattern, and only when the luma pattern is non-zero. */
        if (is_inter_type(c->type) && t8_allowed(e, c) && (c->cbp & 15))
            aeg_cabac_decision(cb, cab_t8_ctx(e, mbx, mby),
                               c->t8 ? 1 : 0);
    }
    /* mb_qp_delta: present when the macroblock has a pattern or is
     * Intra_16x16, whose pattern is carried by mb_type. */
    if (c->cbp || c->type == AEG_MB_I16x16) {
        int d = c->qp - e->last_qp;
        aeg_cabac_decision(cb, CAB_QP_DELTA + (e->last_dqp ? 1 : 0), d != 0);
        if (d) {
            /* mb_qp_delta is coded as the unsigned value 2|d| - 1 (d > 0) or
             * 2|d| (d < 0) in unary (9.3.2.7): v - 1 one-bins followed by a
             * zero-bin.  Only the first bin uses context 2, every later one
             * context 3 -- which includes the terminating zero-bin when v is
             * larger than one. */
            int v = d > 0 ? 2 * d - 1 : -2 * d, i;
            for (i = 1; i < v; i++)
                aeg_cabac_decision(cb, CAB_QP_DELTA + (i == 1 ? 2 : 3), 1);
            aeg_cabac_decision(cb, CAB_QP_DELTA + (v == 1 ? 2 : 3), 0);
        }
        e->last_dqp = d != 0;
    } else {
        e->last_dqp = 0;
    }
    e->last_qp = c->qp;
    cabac_mb_residual(e, cb, mbx, mby, c);
}

/* What the CABAC coder will charge for one macroblock, in 1/256 bit.
 *
 * The macroblock is priced by running the writer that will emit it, with the
 * arithmetic switched off (`aeg_cabac.cost`): every decision still walks its
 * context, so the price follows the state of the contexts the neighbours left
 * behind, which is the whole point -- a flag is cheap or expensive depending on
 * what came before it, and no static bit count can know that.
 *
 * `commit` = 0 prices a candidate against a copy of the shadow state; the copy
 * is thrown away, and so are the per-block motion vectors the writer produces
 * for the neighbours' contexts.  `commit` = 1 is the chosen macroblock: its
 * state becomes the shadow, so the next macroblock is priced from where the
 * bitstream pass will actually be.  The QP bookkeeping mirrors the bitstream
 * pass exactly (a macroblock with no pattern codes no mb_qp_delta, so the
 * decoder keeps the previous QP). */
static int rd_cabac_bits(struct aeg_encoder *e, int mbx, int mby, mbenc *c,
                         int slice, int commit)
{
    aeg_cabac cb = e->cab_shadow;
    int idx = mby * e->mb_w + mbx;
    int skipped = 0;
    int save_qp = e->last_qp, save_dqp = e->last_dqp;
    int16_t save_mvd[2][16][2];

    memcpy(save_mvd, e->mbs[idx].mvd, sizeof(save_mvd));
    if (c->cbp == 0)
        c->qp = e->rd_last_qp;
    if (slice == AEG_SLICE_P && c->type == AEG_MB_P16x16 && c->cbp == 0 &&
        c->ref[0] == 0) {
        int pmvx, pmvy;
        pskip_mv(e, mbx, mby, &pmvx, &pmvy);
        if (c->mv[0][0] == pmvx && c->mv[0][1] == pmvy)
            skipped = 1;
    }
    /* In a B slice a direct macroblock without residual is a skipped one: the
     * decoder reconstructs it from the co-located picture's vectors either way,
     * so the skip flag is free bits to leave unspent. */
    if (slice == AEG_SLICE_B && c->type == AEG_MB_B_DIRECT && c->cbp == 0)
        skipped = 1;
    e->last_qp = e->rd_last_qp;
    e->last_dqp = e->rd_last_dqp;
    cb.cost = 1;
    cb.cost256 = 0;
    write_mb_cabac(e, &cb, mbx, mby, c, slice, skipped);
    if (commit) {
        e->cab_shadow = cb;
        e->rd_last_qp = e->last_qp;
        e->rd_last_dqp = e->last_dqp;
    } else {
        memcpy(e->mbs[idx].mvd, save_mvd, sizeof(save_mvd));
    }
    e->last_qp = save_qp;
    e->last_dqp = save_dqp;
    return (int)cb.cost256;
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
    int ref_bits = e->last_pic_bits, ref_qp = e->last_pic_qp;

    /* A B picture is predicted from the last anchor rather than from the last
     * picture, for the reason given at last_anchor_bits. */
    if (e->cur_is_b && e->last_anchor_bits > 0 && e->last_anchor_qp >= 0) {
        ref_bits = e->last_anchor_bits;
        ref_qp = e->last_anchor_qp;
    }
    if (!e->vbv_active || ref_bits <= 0 || ref_qp < 0)
        return;
    /* The B picture's QP already carries the B offset, so the prediction has to
     * start from the QP the encoder will actually quantise with. */
    avail = vbv_available(e);
    if (avail < 1.0)
        avail = 1.0;
    pred = (double)ref_bits * pow(2.0, (double)(ref_qp - e->cur_qp) / 6.0);
    if (pred <= avail)
        return;
    q_min = (double)ref_qp - 6.0 * log2(avail / (double)ref_bits);
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
    if (!e->cur_is_b) {
        e->last_anchor_bits = bits;
        e->last_anchor_qp = e->cur_qp;
    }
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

/* The QP below which the B-picture offset is faded out (see the B picture's QP
 * in aeg_encoder_encode): at QP 12 and above it is applied in full. */
#define B_QP_FADE 12

/* How much of a frame's budget each picture is expected to take.
 *
 * Every picture cannot take the same share when they are coded with different
 * quantisers on purpose: a B picture sits b_qp_offset coarser than the anchors
 * around it, so it spends a fraction of what an anchor does, and a controller
 * that asked it for a whole frame's worth would keep lowering the QP trying to
 * reach a target the picture type cannot spend -- the stream then lands well
 * under the requested bitrate (14% at 400 kb/s before this was accounted for)
 * while the anchors get finer than they should.  The weights are normalised so
 * that one group -- one anchor and bframes B pictures -- is worth exactly the
 * pictures it contains.
 *
 * The fraction itself is *measured* rather than derived from the QP offset: the
 * offset only bounds what a B picture may spend, and with three B pictures per
 * group the anchors have to make up a share that no closed form gets right (the
 * offset model leaves the stream 11% short at 400 kb/s where the measured
 * ratio lands on it). */
static double rc_weight(const struct aeg_encoder *e, int is_b)
{
    double wb, wa, g;

    if (e->p.bframes <= 0)
        return 1.0;
    g = (double)(e->p.bframes + 1);
    wb = e->rc_b_ratio;             /* measured: B bits / anchor bits      */
    wa = 1.0 + (g - 1.0) * wb;      /* one group's bits / an anchor's      */
    return (is_b ? wb : 1.0) * g / wa;
}

static void rc_update(struct aeg_encoder *e, int bits, int is_idr)
{
    double bpf = e->rc_bits_per_frame;
    double per, want, ratio, adj;

    if (!e->rc_active || bpf <= 0.0)
        return;
    e->rc_frames++;
    e->rc_spent += (double)bits;
    if (e->p.bframes > 0) {
        /* keep the weight the two paths above agree on up to date */
        if (e->cur_is_b && e->rc_anchor_bits > 0) {
            double r = (double)bits / (double)e->rc_anchor_bits;

            if (r < 0.02) r = 0.02;
            if (r > 1.00) r = 1.00;
            e->rc_b_ratio = 0.75 * e->rc_b_ratio + 0.25 * r;
        } else if (!e->cur_is_b) {
            e->rc_anchor_bits = bits;
        }
    }
    per = rc_weight(e, e->cur_is_b) * bpf;
    e->rc_target_sum += per;
    e->rc_debt = e->rc_target_sum - e->rc_spent;
    /* One picture can be many times the target -- a keyframe, or any picture
     * after a scene cut -- and letting that single picture's overshoot follow
     * the controller for the next `RC_WINDOW` pictures is what turns one
     * expensive frame into thirty bad ones.  The debt is therefore clamped to
     * RC_DEBT_MAX pictures' worth: the overshoot is still repaid, just over a
     * window long enough that the quality does not collapse (the alternative,
     * a 20% target floor, lets the QP run away while the debt is large). */
    if (e->rc_debt >  RC_DEBT_MAX * per) e->rc_debt =  RC_DEBT_MAX * per;
    if (e->rc_debt < -RC_DEBT_MAX * per) e->rc_debt = -RC_DEBT_MAX * per;
    want = per + e->rc_debt / RC_WINDOW;
    if (want < 0.2 * per)          /* never ask for the impossible */
        want = 0.2 * per;
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
    {
        /* Normally at most 1.5 QP per picture, so that a scene cut cannot make
         * the quality collapse.  That cap is also what decides how long the
         * stream takes to reach its target, and the whole group's error is
         * repaid through the anchor alone: with three B pictures per group the
         * equilibrium QP can be ten or more below where the controller starts
         * (a B picture spends a fraction of a frame's budget, so the anchors
         * have to be that much better to fill the rate), and 1.5 QP per picture
         * then needs far more pictures than a short stream has.  A picture's
         * worth of debt therefore buys a bigger step -- still bounded, and the
         * debt clamp keeps it from running away. */
        double cap = 1.5 + fabs(e->rc_debt) / per;

        if (cap > 4.0) cap = 4.0;
        if (adj >  cap) adj =  cap;
        if (adj < -cap) adj = -cap;
    }
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
static int in_slots(const struct aeg_encoder *e);

static int is_scene_cut(struct aeg_encoder *e, int is_idr, int display)
{
    int slots = in_slots(e);
    const uint8_t *cur, *ref;
    int stride = e->stride[0];
    int64_t ac = 0, p_sad = 0;
    int bx, by, x, y;

    /* Both sides of the comparison are pictures *one frame apart in display
     * order*.  Without B pictures that is exactly "this picture against the
     * reference it will predict from", which is what the first version of this
     * test did.  With B pictures it cannot be: the most recent decoded picture
     * when an anchor is decided is the group's own B tail, three or four frames
     * back, and over four frames a fast pan has the same |cur - ref| as a cut
     * between two unrelated clips -- the metric stops being able to tell them
     * apart at all.  The source ring, on the other hand, still holds the
     * picture that precedes this one, so the test keeps measuring a one-frame
     * difference whatever the group structure is. */
    if (e->p.bframes > 0) {
        cur = e->in_plane[display % slots][0];
        ref = e->in_plane[(display - 1) % slots][0];
    } else {
        cur = e->srcp[0];
        ref = e->refp[0];
    }
    if (e->p.scenecut <= 0 || is_idr || display < 2)
        return 0;
    if (e->p.bframes <= 0 && !e->has_ref)
        return 0;
    {
        int min_gop = e->p.gop_size > 0 ? e->p.gop_size / 10 : 8;

        if (min_gop < 8) min_gop = 8;
        if (min_gop > 32) min_gop = 32;
        /* in display order, not in coding order: with B pictures the two
         * differ, and the guard is about what the viewer sees */
        if (display - e->last_idr_poc < min_gop)
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
        fprintf(stderr, "SCD display %d ac %lld p_sad %lld ratio %.3f\n",
                display, (long long)ac, (long long)p_sad,
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
                      int first_row, int last_row, int slice, int is_idr,
                      size_t *rbsp_len)
{
    int mbx, mby;
    int skip_run = 0;
    int l0_saved;
    aeg_bitwriter bw;

    e->top_row = first_row;
    e->last_qp = e->cur_qp;
    e->last_dqp = 0;
    e->dbg_cabac = getenv("AEG_CABAC_TRACE") != NULL;
    /* The decision pass prices candidates with a shadow of this slice's coder,
     * so it starts from the same state the bitstream pass will. */
    if (e->p.cabac) {
        aeg_cabac_set_contexts(&e->cab_shadow, slice == AEG_SLICE_I, e->cur_qp,
                               e->cabac_init_idc);
        e->cab_shadow.cost = 1;
        e->cab_shadow.cost256 = 0;
        e->rd_last_qp = e->cur_qp;
        e->rd_last_dqp = 0;
    }
    aeg_bw_init(&bw, bs, cap, 0);
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
            m->t8 = (uint8_t)mb_t8_coded(e, tmp, 0);
            m->chroma_mode = (uint8_t)tmp->chroma_mode;
            m->cbp = (uint8_t)tmp->cbp;
            m->qp = (int8_t)tmp->qp;
            /* All four masks, not the two the partitioned shapes need: B_8x8
             * carries one per 8x8 group, and the deblocking filter reads them
             * per 4x4 block.  Publishing only the first two left the bottom
             * half of every B_8x8 at "no list" (which the mask fallback then
             * read as *both*), so the filter disagreed with the decoder about
             * the two bottom 8x8 groups of every such macroblock. */
            for (i = 0; i < 4; i++)
                m->blist[i] = tmp->blist[i];
            for (i = 0; i < 4; i++)
                m->sub_type[i] = (int8_t)tmp->sub_type[i];
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
                m->mv1[i][0] = tmp->mv1[i][0];
                m->mv1[i][1] = tmp->mv1[i][1];
                m->ref_idx[i] = tmp->ref[i];
            }
            m->cost = (int32_t)tmp->sse;
        }
    }

    /* The slice header is written after the decision pass, because one of
     * its fields depends on it: num_ref_idx_l0_active.  When no macroblock of
     * this slice ended up on the second reference, the slice is sent with one
     * active reference -- every decision stays valid (all of them predict
     * from list0[0], the predictors and the skip vector are the same with one
     * or two entries), and the ref_idx syntax disappears from every
     * partition.  It is a per-slice choice and is undone at the end of the
     * slice, so the next slice of the picture decides again with both. */
    l0_saved = e->cur_l0_active;
    if (e->cur_l0_active > 1 && slice == AEG_SLICE_P) {
        int used = 0;
        for (mby = first_row; mby <= last_row && !used; mby++)
            for (mbx = 0; mbx < e->mb_w && !used; mbx++) {
                const mbenc *t = &e->mb_encs[mby * e->mb_w + mbx];
                int b;
                if (!is_inter_type(t->type))
                    continue;
                for (b = 0; b < 16; b++)
                    used |= t->ref[b] != 0;
            }
        if (!used)
            e->cur_l0_active = 1;
    }
    /* A stream without B pictures keeps pic_order_cnt_type 2 and spends nothing
     * on the picture order count; with them the POC is what direct mode scales
     * its vectors by, so every header carries it (7.3.3). */
    if (e->p.bframes > 0) {
        int max_lsb = 1 << e->poc_lsb_bits;
        aeg_write_slice_header_poc(&bw,
                                   slice == AEG_SLICE_B ? 6 : (slice ? 5 : 7),
                                   e->frame_num & ((1 << e->frame_num_bits) - 1),
                                   e->frame_num_bits, is_idr, e->idr_pic_id,
                                   e->cur_poc & (max_lsb - 1), e->poc_lsb_bits,
                                   e->cur_is_ref,
                                   e->p.cabac ? e->cabac_init_idc : -1,
                                   e->cur_qp - 26, 0, 0,
                                   e->p.deblock ? 0 : 1, first_row * e->mb_w,
                                   e->cur_l0_active, e->p.direct == 1);
    } else {
        aeg_write_slice_header(&bw, slice ? 5 : 7,
                               e->frame_num & ((1 << e->frame_num_bits) - 1),
                               e->frame_num_bits, is_idr, e->idr_pic_id,
                               e->p.cabac ? e->cabac_init_idc : -1,
                               e->cur_qp - 26, 0, 0,
                               e->p.deblock ? 0 : 1, first_row * e->mb_w,
                               e->cur_l0_active);
    }

    /* The decision pass left `last_qp`/`last_dqp` at the values its shadow
     * ended on; both bitstream passes start from the slice's own state. */
    e->last_qp = e->cur_qp;
    e->last_dqp = 0;

    /* ---- bitstream pass, CABAC -----------------------------------------
     * The same raster order, for the same reason (every context reads the
     * neighbours), but the coder is arithmetic: the macroblocks are fed to it
     * one bin at a time and it writes its bytes itself, so the bit writer is
     * only used for the header and for the trailing bits afterwards. */
    if (e->p.cabac) {
        aeg_cabac cab;
        size_t hdr;

        /* cabac_alignment_one_bit: the arithmetic coder starts on a byte
         * boundary (7.3.4). */
        while (bw.bits)
            aeg_bw_put(&bw, 1, 1);
        hdr = bw.len;
        aeg_cabac_open(&cab, hdr < cap ? bs + hdr : bs, hdr < cap ? cap - hdr : 0);
        cab.trace = e->dbg_cabac;
        aeg_cabac_set_contexts(&cab, slice == AEG_SLICE_I, e->cur_qp,
                               e->cabac_init_idc);
        e->last_dqp = 0;

        for (mby = first_row; mby <= last_row; mby++) {
            for (mbx = 0; mbx < e->mb_w; mbx++) {
                int idx = mby * e->mb_w + mbx;
                mbenc *tmp = &e->mb_encs[idx];
                aeg_mb *m = &e->mbs[idx];
                int skipped = 0, last = 0, i;
                int64_t t0 = aeg_now_ns();

                /* A macroblock without a coded block pattern carries no
                 * mb_qp_delta, so the decoder keeps the QP it has; the
                 * Intra_16x16 one always carries the delta, and it is zero. */
                if (tmp->cbp == 0)
                    tmp->qp = e->last_qp;
                m->qp = (int8_t)tmp->qp;

                if (slice == AEG_SLICE_P && tmp->type == AEG_MB_P16x16 &&
                    tmp->cbp == 0 && tmp->ref[0] == 0) {
                    int pmvx, pmvy;
                    pskip_mv(e, mbx, mby, &pmvx, &pmvy);
                    if (tmp->mv[0][0] == pmvx && tmp->mv[0][1] == pmvy)
                        skipped = 1;
                }
                /* A direct macroblock of a B slice with no residual is exactly
                 * what the skip flag means: same prediction, no residual. */
                if (slice == AEG_SLICE_B && tmp->type == AEG_MB_B_DIRECT &&
                    tmp->cbp == 0)
                    skipped = 1;
                m->skip = (uint8_t)skipped;
                e->stat_mbs++;
                e->stat_skipped += skipped;
                e->stat_type[skipped ? AEG_MB_SKIP : (int)tmp->type]++;
                /* A skipped macroblock carries no mb_qp_delta, so the decoder
                 * forgets the previous delta's value: the next macroblock tests
                 * its first bin against context 60, not 61 (9.3.2.7). */
                if (skipped)
                    e->last_dqp = 0;
                if (skipped) {
                    m->t8 = 0;
                    memset(m->mvd, 0, sizeof(m->mvd));
                    memset(tmp->nz, 0, sizeof(tmp->nz));
                    memset(tmp->nz8, 0, sizeof(tmp->nz8));
                }
                last = (mby == last_row && mbx == e->mb_w - 1);
                if (e->dbg_cabac) {
                    int k;
                    fprintf(stderr, "ENCMB %d,%d type %d cbp %d t8 %d qp %d "
                            "i16 %d ch %d skip %d mv %d,%d i4", mbx, mby,
                            tmp->type, tmp->cbp, tmp->t8, tmp->qp,
                            tmp->i16_mode, tmp->chroma_mode, skipped,
                            tmp->mv[0][0], tmp->mv[0][1]);
                    for (k = 0; k < 16; k++)
                        fprintf(stderr, " %d", tmp->intra4[k]);
                    fprintf(stderr, " i8");
                    for (k = 0; k < 4; k++)
                        fprintf(stderr, " %d", tmp->intra8[k]);
                    fprintf(stderr, " nz");
                    for (k = 0; k < AEG_BLK_COUNT; k++)
                        fprintf(stderr, " %d", tmp->nz[k]);
                    fprintf(stderr, "\n");
                }
                write_mb_cabac(e, &cab, mbx, mby, tmp, slice, skipped);
                aeg_cabac_terminate(&cab, last);

                /* the decisions the following macroblocks' contexts read */
                m->t8 = (uint8_t)mb_t8_coded(e, tmp, skipped);
                for (i = 0; i < AEG_BLK_COUNT; i++)
                    m->nz[i] = tmp->nz[i];
                for (i = 0; i < 4; i++)
                    m->nz8[i] = (int8_t)(tmp->nz8[i] != 0);
                e->st.stage_ns[AEG_STAGE_TRANSFORM] += aeg_now_ns() - t0;
            }
        }
        *rbsp_len = hdr + aeg_cabac_flush(&cab);
        /* rbsp_slice_trailing_bits (7.3.4.4): the stop bit, so that the last
         * byte of the slice is never an all-zero one. */
        if (*rbsp_len + 1 <= cap)
            bs[(*rbsp_len)++] = 0x80;
        e->cur_l0_active = l0_saved;
        return AEG_OK;
    }

    /* ---- bitstream pass ------------------------------------------------
     * CAVLC is serial: the bit cost and the bits depend on the non-zero counts
     * of the neighbours, so the macroblocks go out in raster order, and
     * mb_qp_delta depends on the previous macroblock's QP.
     *
     * There is no CAVLC B macroblock layer here: the B-picture analysis is
     * priced by the CABAC shadow coder and its types only exist for CABAC, so
     * aeg_encoder_open() turns B pictures off when CABAC is off.  Reaching this
     * with a B slice would write a P macroblock layer. */
    if (slice == AEG_SLICE_B) {
        aeg_set_last_error("B pictures need CABAC");
        e->cur_l0_active = l0_saved;
        return AEG_EINVAL;
    }
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
                if (slice == AEG_SLICE_P && tmp->type == AEG_MB_P16x16 &&
                    tmp->cbp == 0 && tmp->ref[0] == 0) {
                    int pmvx, pmvy;
                    pskip_mv(e, mbx, mby, &pmvx, &pmvy);
                    if (tmp->mv[0][0] == pmvx && tmp->mv[0][1] == pmvy)
                        skipped = 1;
                }
                e->stat_mbs++;
                e->stat_skipped += skipped;
                e->stat_type[skipped ? AEG_MB_SKIP : (int)tmp->type]++;
                if (skipped)
                    skip_run++;
                else {
                    if (slice == AEG_SLICE_P)
                        flush_skip_run(&bw, &skip_run);
                    write_mb(e, &bw, mbx, mby, tmp, slice == AEG_SLICE_P);
                }
            }
            e->st.stage_ns[AEG_STAGE_TRANSFORM] += aeg_now_ns() - t0;
        }
    }
    /* A skip run that reaches the end of the frame still has to be emitted:
     * the decoder counts macroblocks by consuming mb_skip_run values, so an
     * unwritten tail run makes it stop before the last row. */
    if (slice == AEG_SLICE_P && skip_run > 0)
        flush_skip_run(&bw, &skip_run);
    /* aeg_bw_finish() appends rbsp_trailing_bits (the stop bit and the zero
     * padding); aligning here as well used to add a second, all-zero-padded
     * 0x80 byte.  A single slice could get away with it - the decoder stops at
     * the end of the picture and ignores the tail - but a slice that ends in
     * the middle of the picture must end exactly at its last macroblock, or
     * the decoder keeps reading macroblocks out of the next slice's header. */
    *rbsp_len = aeg_bw_finish(&bw);
    e->cur_l0_active = l0_saved;
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
    int slice, is_idr;
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
                       last_row - 1, j->slice, j->is_idr, &j->lens[si]) != AEG_OK) {
            j->err = 1;
            break;
        }
    }
    return NULL;
}

/* Code one picture.  `slice` is AEG_SLICE_I/P/B, `poc` its picture order count
 * -- the display index, which is what direct mode scales its vectors by.  The
 * caller has already bound e->srcp to the picture's source samples (see
 * aeg_encoder_encode) and decided that it is this picture's turn. */
static int encode_frame(struct aeg_encoder *e, int slice, int is_idr, int poc,
                        int64_t pts, aeg_packet *pkt)
{
    int is_p;
    size_t out_len;
    int64_t t_all = aeg_now_ns();
    int64_t dbg_type0[AEG_MB_TYPE_COUNT];

    memcpy(dbg_type0, e->stat_type, sizeof(dbg_type0));
    /* Which lists this picture predicts from.  A B picture reads both anchors;
     * a P picture reads the newest one.  This has to happen before the scene
     * cut test, which compares the picture with its list 0 reference. */
    e->cur_poc = poc;
    e->cur_is_b = slice == AEG_SLICE_B;
    e->cur_is_ref = !e->cur_is_b;
    e->pic_poc[e->role_cur] = poc;
    /* set_current_refs() reads cur_is_p (a P picture may take two list-0
     * entries), so it has to describe *this* picture.  It used to be assigned
     * further down, after the call, so every picture saw the previous one's
     * value: a B picture coded after a P anchor was set up as a two-reference
     * P picture (list 0 = the newer anchor, "list 1" = the older one -- both
     * lists swapped), and a P anchor coded after a B picture got one
     * reference.  The P-only streams hid it because there the stale value was
     * the right one. */
    e->cur_is_p = slice == AEG_SLICE_P && !is_idr;
    set_current_refs(e);
    /* A scene cut makes the picture an IDR: everything before it is dropped, so
     * coding it as a P picture would leave the following ones referring to
     * pictures the decoder has just discarded.  With B pictures the decision is
     * made in schedule() instead, before anything is queued: an IDR drops every
     * reference before it, so the B pictures of its group could no longer be
     * coded once it is in the queue. */
    if (e->p.bframes <= 0 && slice == AEG_SLICE_P &&
        is_scene_cut(e, is_idr, e->frame_count)) {
        is_idr = 1;
        slice = AEG_SLICE_I;
        e->scene_cuts++;
        aeg_log(2, "aeg: scene cut at frame %d -> IDR", e->frame_count);
    }
    is_p = slice == AEG_SLICE_P && !is_idr && e->has_ref;
    if (is_p != e->cur_is_p) {
        e->cur_is_p = is_p;
        set_current_refs(e);
    }
    if (is_idr) {
        e->last_idr_frame = e->frame_count;
        e->last_idr_poc = poc;
    }

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
    else if ((is_idr || slice == AEG_SLICE_I) && e->p.i_qp_offset > 0) {
        e->cur_qp -= e->p.i_qp_offset;
        if (e->cur_qp < 0)  e->cur_qp = 0;
    }
    if (e->rc_qp_offset) {
        e->cur_qp += e->rc_qp_offset;
        if (e->cur_qp < 0)  e->cur_qp = 0;
        if (e->cur_qp > 51) e->cur_qp = 51;
    }
    /* A B picture is not a reference, so its quantiser never propagates: the
     * anchors keep the QP the controller gave them and the B pictures in
     * between can be rounded harder without costing the following pictures
     * anything.  This is the cheapest half of what B pictures are for --
     * x264 spends +2 by default for exactly this reason. */
    if (e->cur_is_b && e->p.b_qp_offset) {
        /* The offset fades out towards QP 0.  A QP step is a *relative* change
         * in quality, and it is only worth a fixed number of steps while there
         * is room to pay for it: at QP 4 an offset of twelve puts the B picture
         * at the quality of QP 16, where the error is plainly visible, and the
         * bits it saves cannot be spent on anything better -- which is exactly
         * what happened to the rate controller, whose stream fell 17% short of
         * its target because the B pictures alone were cheap enough that the
         * anchors were already at QP 0.  The fade starts at B_QP_FADE (below),
         * well under the 22..37 the offset was measured in, so no measurement
         * reported anywhere in docs/STATUS.md moves. */
        int off = e->p.b_qp_offset;

        if (e->cur_qp < B_QP_FADE)
            off = off * e->cur_qp / B_QP_FADE;
        e->cur_qp += off;
        if (e->cur_qp < 0)  e->cur_qp = 0;
        if (e->cur_qp > 51) e->cur_qp = 51;
    }
    if (e->vbv_active)
        vbv_clamp_qp(e);

    resolve_coef_stages(e);
    aq_compute(e);
    if ((is_p || e->cur_is_b) && e->p.subme > 0) {
        /* Build the fractional-pel planes of the reference picture once for the
         * whole picture (AEG_STAGE_INTER_PRED).  Every subpel candidate of every
         * macroblock then reads samples from here instead of re-running the
         * 6-tap chain per block: the same samples, computed ~30x less often.
         * A B picture builds them for both lists. */
        int64_t t0 = aeg_now_ns();
        int list;

        if (e->d.use_gpu[AEG_STAGE_INTER_PRED] && e->d.ops.interp_plane &&
            e->d.ops.backend != AEG_BACKEND_CPU) {
            /* the planes are what the GPU stage is for: one launch per plane,
             * every output sample independent, and the kernel is checked
             * against aeg_hpel_plane_fill() by tests/kernel_check.c */
            for (list = 0; list < (e->cur_is_b || e->cur_l0_active > 1 ? 2 : 1); list++) {
                aeg_mc_ctx *ctx = list ? &e->mc1 : &e->mc;
                const uint8_t *ref = list ? e->ref2p[0] : e->refp[0];
                int i;

                aeg_mc_ctx_setref(ctx, ref);
                for (i = 1; i < 4; i++)
                    e->d.ops.interp_plane(e->d.ops.priv, ref, e->stride[0],
                                          e->w, e->h, i,
                                          (uint8_t *)ctx->base[i], ctx->stride);
                ctx->ready = 1;
            }
        } else {
            aeg_mc_ctx_refresh(&e->mc, e->refp[0], e->w, e->h);
            if (e->cur_is_b || e->cur_l0_active > 1)
                aeg_mc_ctx_refresh(&e->mc1, e->ref2p[0], e->w, e->h);
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

    /* One slice per picture is the default and keeps the bitstream single-NAL;
     * `--slices N` (or `-t N`, which uses one slice per thread) cuts the picture
     * into N row ranges, each an independently decodable NAL. */
    out_len = 0;
    if (is_idr) {
        uint8_t sps[256], pps[64];
        size_t sl = aeg_write_sps(sps, sizeof(sps), &e->p, e->mb_w, e->mb_h,
                                  e->frame_num_bits, e->level_idc);
        size_t pl = aeg_write_pps(pps, sizeof(pps), e->p.transform_8x8,
                                  e->p.cabac);
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
                jobs[si].slice = slice;
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
                               first_row, last_row - 1, slice, is_idr,
                               &sl_len) != AEG_OK)
                    return AEG_ENOMEM;
                lens[si] = sl_len;
            }
        }

        for (si = 0; si < e->n_slices; si++) {
            if (!lens[si])
                break;
            /* A B picture is not used as a reference, so its nal_ref_idc is 0
             * and the decoder is free to drop it as soon as it is output. */
            out_len += aeg_nal_write(e->out + out_len, e->out_cap - out_len,
                                     is_idr ? 3 : (slice == AEG_SLICE_B ? 0 : 2),
                                     is_idr ? 5 : 1, e->slice_bs[si], lens[si],
                                     0);
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
    e->frame_count++;
    /* frame_num counts reference pictures only: a non-reference picture carries
     * PrevRefFrameNum + 1 (7.4.3), so all the B pictures of a group repeat the
     * value the anchor after them already had. */
    if (!e->cur_is_b)
        e->frame_num++;
    /* The co-located picture of the following B pictures is this one if it is a
     * reference picture, and it is the most recent reference either way. */
    if (!e->cur_is_b && e->col_mbs) {
        memcpy(e->col_mbs, e->mbs, (size_t)e->mb_count * sizeof(e->mbs[0]));
        e->col_poc = e->cur_poc;
        e->col_ref_poc[0] = e->ref_poc[0];
        e->col_ref_poc[1] = e->ref_poc[1];
        e->col_ready = 1;
    }
    /* The reconstruction becomes a reference picture and the buffer it occupied
     * is handed to the next picture (or, for a B picture, kept: a B is not a
     * reference, so its buffer is simply reused). */
    picture_rotate(e, !e->cur_is_b, is_idr);
    e->pic.plane[0] = e->recp[0];
    e->pic.plane[1] = e->recp[1];
    e->pic.plane[2] = e->recp[2];
    if (is_idr)
        e->idr_pic_id = (e->idr_pic_id + 1) & 0xffff;
    if (getenv("AEG_RCDBG")) {
        /* one line per picture: what it is and where its macroblocks went */
        int64_t sk = e->stat_type[AEG_MB_SKIP] - dbg_type0[AEG_MB_SKIP];
        int64_t in = 0;
        int k;
        for (k = 0; k < AEG_MB_TYPE_COUNT; k++)
            if (k == AEG_MB_I4x4 || k == AEG_MB_I8x8 || k == AEG_MB_I16x16 ||
                k == AEG_MB_IPCM)
                in += e->stat_type[k] - dbg_type0[k];
        fprintf(stderr, "PIC %c poc %d qp %d bits %d skip %lld intra %lld\n",
                is_idr ? 'I' : (e->cur_is_b ? 'B' : (e->cur_is_p ? 'P' : 'i')),
                e->cur_poc, e->cur_qp, (int)(out_len * 8), (long long)sk,
                (long long)in);
    }
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
/* Input lookahead (B pictures)                                        */
/* ------------------------------------------------------------------ */
/* A stream with B pictures is coded out of display order, so the encoder holds
 * the pictures of the group it is working on and codes them in the order the
 * bitstream needs:
 *
 *   I0  P4  B1 B2 B3  P8  B5 B6 B7  P12 ...
 *
 * with bframes = 3 (a group is the anchor P4 followed by B1..B3).  The B
 * pictures come *after* the anchor that follows them because that anchor is
 * their list 1 reference: it has to be decoded before them.
 *
 * The ring holds one group plus the next anchor (bframes + 2 pictures), and the
 * scheduler is one comparison: once display index anchor + bframes + 1 is in the
 * ring, its group can be coded. */

static int in_slots(const struct aeg_encoder *e)
{
    return e->p.bframes + 3;
}

static void copy_yuv(const struct aeg_encoder *e, uint8_t *p[3],
                     const uint8_t *yuv420p)
{
    copy_plane_pad(p[0], e->stride[0], yuv420p, e->w, e->w, e->h);
    copy_plane_pad(p[1], e->stride[1], yuv420p + (size_t)e->w * e->h,
                   e->cw, e->cw, e->ch);
    copy_plane_pad(p[2], e->stride[2],
                   yuv420p + (size_t)e->w * e->h + (size_t)e->cw * e->ch,
                   e->cw, e->cw, e->ch);
}

static void bind_source(struct aeg_encoder *e, int display)
{
    int i, slot = display % in_slots(e);
    for (i = 0; i < 3; i++)
        e->srcp[i] = e->in_plane[slot][i];
}

static void push_input(struct aeg_encoder *e, const uint8_t *yuv420p,
                       int64_t pts)
{
    int slot = e->in_count % in_slots(e);
    copy_yuv(e, e->in_plane[slot], yuv420p);
    if (e->in_low[slot].mem)
        aeg_lowres_build(&e->in_low[slot], e->in_plane[slot][0], e->stride[0],
                         e->w, e->h);
    e->in_pts[slot] = pts;
    e->in_count++;
}

static void add_job(struct aeg_encoder *e, int display, int slice, int idr)
{
    int n = e->job_n;
    if (n >= (int)(sizeof(e->job_d) / sizeof(e->job_d[0])))
        return;
    e->job_d[n] = display;
    e->job_slice[n] = slice;
    e->job_idr[n] = idr;
    e->job_n = n + 1;
}

/* Whether display index `poc` is due to be coded as an IDR.  With B pictures
 * the decision is made on the display index rather than on the count of coded
 * pictures, so that keyframes land at the same place in the stream whatever the
 * group structure is; it can only ever fire on an anchor, which is why the
 * interval is honoured at the first anchor at or after it. */
static int idr_due(const struct aeg_encoder *e, int poc)
{
    if (poc == 0)
        return 1;
    if (e->p.gop_size <= 0)
        return 0;
    if (e->p.bframes > 0)
        return poc - e->last_idr_poc >= e->p.gop_size;
    return (e->frame_count % e->p.gop_size) == 0;
}

/* Code one queued picture.  One picture per call keeps the API's promise that a
 * packet belongs to the picture that was just handed in, even where the coding
 * order needs several pictures per group. */
static int run_job(struct aeg_encoder *e, aeg_packet *pkt)
{
    int d = e->job_d[0], slice = e->job_slice[0], idr = e->job_idr[0];
    int64_t pts = e->in_pts[d % in_slots(e)];
    int i, rc;

    for (i = 1; i < e->job_n; i++) {
        e->job_d[i - 1] = e->job_d[i];
        e->job_slice[i - 1] = e->job_slice[i];
        e->job_idr[i - 1] = e->job_idr[i];
    }
    e->job_n--;
    bind_source(e, d);
    rc = encode_frame(e, slice, idr, d, pts, pkt);
    if (rc == AEG_OK)
        e->enc_pos++;
    return rc;
}

/* Adaptive B pictures: how long the group after anchor `a` should be, given
 * that pictures a+1..last are in the ring (last = a + bframes + 1).  Returns
 * the display index of the group's anchor.  This is x264's "fast" decision
 * (b-adapt 1) on the same kind of estimate (lookahead.c):
 *
 *  - if a+2 predicted from a is mostly intra, a+1 is a P picture;
 *  - if a+1 and a+2 as two P pictures cost less than a+1 as a B picture
 *    between a and a+2 plus a+2 as a P picture from a, a+1 is a P picture;
 *  - otherwise a+1 is a B picture, and so is every following picture j as
 *    long as j+1 predicted from a stays below a cost threshold (which falls
 *    with the distance) and is not one third intra; the first picture that
 *    fails is the anchor. */
static int la_bias(void)
{
    static int v = -1;
    if (v < 0) {
        const char *ev = getenv("AEG_LABIAS");
        v = ev ? atoi(ev) : 120;
    }
    return v;
}

static int64_t la_cost(struct aeg_encoder *e, int p0, int p1, int b,
                       int *intra)
{
    int slots = in_slots(e);
    int64_t c = aeg_lowres_cost(&e->in_low[b % slots], &e->in_low[p0 % slots],
                                p1 != b ? &e->in_low[p1 % slots] : NULL, intra,
                                e->la_scratch);

    /* a B picture is coded at a coarser quantiser and nothing predicts from
     * it; x264 discounts its estimate by the same 1/1.2 */
    if (p1 != b)
        c = c * 100 / la_bias();
    return c;
}

static int b_adapt_decide(struct aeg_encoder *e, int a, int last)
{
    int nmb = e->mb_w > 2 && e->mb_h > 2 ? (e->mb_w - 2) * (e->mb_h - 2)
                                         : e->mb_w * e->mb_h;
    int64_t c2p1, c1b1, c1p0, c2p0;
    int intra, j;

    if (last - a < 2)
        return last;
    c2p1 = la_cost(e, a, a + 2, a + 2, &intra);
    if (intra > nmb / 2)
        return a + 1;
    c1b1 = la_cost(e, a, a + 2, a + 1, NULL);
    c1p0 = la_cost(e, a, a + 1, a + 1, NULL);
    c2p0 = la_cost(e, a + 1, a + 2, a + 2, NULL);
    if (getenv("AEG_LADBG"))
        fprintf(stderr, "LA a %d intra %d/%d  1p0 %lld 2p0 %lld | 1b1 %lld 2p1 %lld\n",
                a, intra, nmb, (long long)c1p0, (long long)c2p0,
                (long long)c1b1, (long long)c2p1);
    if (c1p0 + c2p0 < c1b1 + c2p1)
        return a + 1;
    for (j = a + 2; j < last; j++) {
        int thresh = 300 - 50 * (j - a - 1);
        int64_t pc;

        if (thresh < 30)
            thresh = 30;
        {
            static int ts = -1;
            if (ts < 0) {
                const char *ev = getenv("AEG_LATHRESH");
                ts = ev ? atoi(ev) : 100;
            }
            thresh = thresh * ts / 100;
        }
        pc = la_cost(e, a, j + 1, j + 1, &intra);
        if (getenv("AEG_LADBG"))
            fprintf(stderr, "LA   j %d pcost/mb %lld thresh %d intra %d\n", j,
                    (long long)(pc / nmb), thresh, intra);
        if (pc > (int64_t)thresh * nmb || intra > nmb / 3)
            break;
    }
    return j;
}

/* Fill the job queue with the next coding step, if the input has arrived. */
static void schedule(struct aeg_encoder *e)
{
    int G, nxt, idr, d;

    if (e->job_n)
        return;
    if (e->anchor < 0) {
        /* The first picture stands alone: an IDR needs no reference, so it can
         * be coded as soon as it is in the ring. */
        if (e->in_count < 1)
            return;
        add_job(e, 0, AEG_SLICE_I, 1);
        e->anchor = 0;
        return;
    }
    G = e->p.bframes + 1;
    nxt = e->anchor + G;
    if (e->in_count <= nxt)
        return;                       /* the next anchor has not arrived yet */
    /* The scene cut test has to run before anything is queued: an IDR drops
     * every reference before it, so the B pictures of its group could no longer
     * be coded once it is in the queue.  It also decides *where* the group ends:
     * every picture of the group is in the ring by now, so the cut can be
     * looked for at any of them rather than only at the anchor it happens to
     * fall near.  Testing only the anchor would miss a cut that falls between
     * two anchors entirely -- the anchor after it compares two pictures that are
     * both already past the cut.  The group is then cut short so that the cut
     * picture becomes its IDR. */
    idr = 0;
    if (e->p.scenecut > 0) {
        int c;
        bind_source(e, nxt);
        e->cur_is_b = 0;
        set_current_refs(e);
        for (c = e->anchor + 1; c <= nxt; c++) {
            if (!is_scene_cut(e, 0, c))
                continue;
            nxt = c;
            idr = 1;
            e->scene_cuts++;
            aeg_log(2, "aeg: scene cut at frame %d -> IDR", nxt);
            break;
        }
    }
    if (!idr && e->p.b_adapt)
        nxt = b_adapt_decide(e, e->anchor, nxt);
    if (!idr)
        idr = idr_due(e, nxt);
    if (idr) {
        /* No B pictures can precede an IDR, so the group's B pictures go out as
         * P pictures first, in display order, and the IDR comes last. */
        for (d = e->anchor + 1; d < nxt; d++)
            add_job(e, d, AEG_SLICE_P, 0);
        add_job(e, nxt, AEG_SLICE_I, 1);
    } else {
        add_job(e, nxt, AEG_SLICE_P, 0);
        for (d = e->anchor + 1; d < nxt; d++)
            add_job(e, d, AEG_SLICE_B, 0);
    }
    e->anchor = nxt;
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
    /* The search does not know where the picture ends: it only bounds the
     * vector by +-range around zero.  The reference planes are edge-extended
     * by AEG_PAD samples, which is what a decoder's unbounded edge clamping
     * reduces to as long as every sample read -- the block at +-range, one more
     * pel of sub-pel refinement and the three taps of the 6-tap filter -- stays
     * inside the padding.  A vector that reaches past it reads the neighbouring
     * row's padding instead of the clamped edge and the encoder reconstructs a
     * different block than the decoder.  The padding used to be 32 with a
     * preset-4 range of 48; nothing hit it until the second reference (twice
     * the temporal distance, so twice the vectors) did. */
    if (pp->search_range > AEG_PAD - 8)
        pp->search_range = AEG_PAD - 8;
    if (pp->subme < 0) pp->subme = pp->preset <= 0 ? 0 : 1;
    /* Refining the winning vector by rate-distortion costs one full macroblock
     * candidate per neighbour, so it scales with the preset like the search
     * range does: the two slow presets are where the quality is wanted. */
    if (pp->refine_mv < 0)
        pp->refine_mv = pp->preset >= 4 ? 2 : (pp->preset >= 3 ? 1 : 0);
    /* Refining every candidate is a large amount of work (one macroblock
     * evaluation per neighbour per candidate) for a change that measured
     * between 0.3 and 0.6 points, a worse exchange rate than the level
     * selection's, so it is off unless asked for. */
    if (pp->refine_cand < 0)
        pp->refine_cand = 0;
    /* Level selection is a large amount of work (a price and an inverse
     * transform per candidate level) that measured 0.15-2.4 points of BD-rate
     * for 12-14 % of the encode time, i.e. a better exchange rate than the
     * candidate refinement, so it follows the preset: presets 3 and 4 run it,
     * below them it is off (see "Rate-distortion level selection"). */
    if (pp->trellis < 0)
        pp->trellis = pp->preset >= 3 ? 1 : 0;
    if (pp->gop_size < 0) pp->gop_size = 250;
    /* B pictures: three between two anchors at most (AEG_MAX_BFRAMES), and none
     * at all without CABAC -- the B macroblock layer is only implemented for the
     * arithmetic coder (see code_slice). */
    if (pp->bframes < 0) pp->bframes = 0;
    if (pp->bframes > AEG_MAX_BFRAMES) pp->bframes = AEG_MAX_BFRAMES;
    if (pp->bframes > 0 && !pp->cabac) {
        aeg_log(1, "aeg: B pictures need CABAC; coding without them");
        pp->bframes = 0;
    }
    /* A second P reference (list0[1], the anchor before the previous one).
     * Measured with tools/abbd.py over the four synthetic clips at preset 3:
     * -1.4 % mean BD-rate for +16 % time on P-only streams, but -0.2 % (noise)
     * with three B pictures, where the anchors are four pictures apart and the
     * older one is eight away -- and where a direct block cannot follow a
     * co-located block on it (direct_group_ok).  So it follows the preset only
     * when there are no B pictures; --ref 2 forces it either way. */
    if (pp->refs < 0)
        pp->refs = (pp->preset >= 3 && pp->bframes == 0) ? 2 : 1;
    if (pp->i_qp_offset < 0)
        pp->i_qp_offset = 3;
    if (pp->direct < 0 || pp->direct > 1)
        pp->direct = 1;
    if (pp->b_adapt < 0 || pp->b_adapt > 1)
        pp->b_adapt = 1;
    if (pp->refs < 1)
        pp->refs = 1;
    if (pp->refs > 2)
        pp->refs = 2;
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
        e->rc_target_sum = 0.0;
        e->rc_anchor_bits = 0;
        /* the starting guess is the QP offset's own prediction */
        e->rc_b_ratio = pow(0.5, (double)pp->b_qp_offset / 6.0);
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
            e->last_anchor_bits = 0;
            e->last_anchor_qp = -1;
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
    e->anchor = -1;
    e->job_n = 0;
    e->in_count = 0;
    e->enc_pos = 0;
    e->last_idr_poc = 0;
    /* log2_max_pic_order_cnt_lsb_minus4 = 4 in the SPS of a stream with B
     * pictures (see aeg_write_sps); the header and the direct-mode scaling both
     * read the same field. */
    e->poc_lsb_bits = 8;

    if (enc_alloc_picture(e, &e->src_mem, e->srcp) != AEG_OK) {
        aeg_set_last_error("out of memory");
        aeg_free(e);
        return NULL;
    }
    {
        int i;
        for (i = 0; i < 3; i++)
            if (enc_alloc_picture(e, &e->pic_mem[i], e->pic_plane[i]) != AEG_OK) {
                aeg_set_last_error("out of memory");
                aeg_encoder_close(e);
                return NULL;
            }
        e->role_cur = 0;
        e->role_last = 0;
        e->role_ref0 = e->role_ref1 = -1;
        e->nrefs = 0;
        bind_pictures(e);
    }
    aeg_mc_ctx_init(&e->mc, e->w, e->h, e->stride[0]);
    aeg_mc_ctx_init(&e->mc1, e->w, e->h, e->stride[0]);
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
    /* B pictures: the lookahead ring (raw input, display order) and the second
     * macroblock grid (the co-located picture's decisions, which direct mode
     * derives its vectors from). */
    if (e->p.bframes > 0) {
        int i, slots = e->p.bframes + 3;

        for (i = 0; i < slots; i++) {
            if (enc_alloc_picture(e, &e->in_mem[i], e->in_plane[i]) != AEG_OK) {
                aeg_set_last_error("out of memory");
                aeg_encoder_close(e);
                return NULL;
            }
        }
        e->in_pts = (int64_t *)aeg_alloc((size_t)slots * sizeof(int64_t));
        e->col_mbs = (aeg_mb *)aeg_alloc((size_t)e->mb_count * sizeof(aeg_mb));
        if (!e->in_pts || !e->col_mbs) {
            aeg_set_last_error("out of memory");
            aeg_encoder_close(e);
            return NULL;
        }
        for (i = 0; i < slots; i++)
            e->in_pts[i] = 0;
        if (e->p.b_adapt) {
            e->la_scratch = (int *)aeg_alloc((size_t)e->mb_count * 4 *
                                             sizeof(int));
            for (i = 0; i < slots; i++)
                if (aeg_lowres_alloc(&e->in_low[i], e->mb_w, e->mb_h))
                    break;
            if (i < slots || !e->la_scratch) {
                aeg_set_last_error("out of memory");
                aeg_encoder_close(e);
                return NULL;
            }
        }
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
    if (!e->p.bframes) {
        copy_yuv(e, e->srcp, yuv420p);
        is_idr = idr_due(e, e->frame_count);
        return encode_frame(e, is_idr ? AEG_SLICE_I : AEG_SLICE_P, is_idr,
                            e->frame_count, pts, pkt);
    }
    /* With B pictures the pictures are coded out of display order: the anchor
     * that follows a group of B pictures has to be in the bitstream before
     * them, because it is their list 1 reference.  So the input goes into the
     * ring first and the scheduler decides what can be coded now. */
    push_input(e, yuv420p, pts);
    schedule(e);
    if (!e->job_n)
        return AEG_EAGAIN;
    return run_job(e, pkt);
}

int aeg_encoder_flush(aeg_encoder *enc, aeg_packet *pkt)
{
    struct aeg_encoder *e = (struct aeg_encoder *)enc;

    if (pkt)
        pkt->data = NULL, pkt->size = 0;
    if (!e || !e->p.bframes)
        return AEG_EOF;
    if (e->job_n)
        return run_job(e, pkt);
    /* The B pictures still waiting have no anchor after them any more (the
     * stream ended), and a B picture without a list 1 reference cannot be
     * coded at all.  They go out as P pictures in display order instead: same
     * pictures, each predicting from the one before it.  This is what x264
     * does with the B pictures of the last, incomplete group. */
    if (e->in_count > e->anchor + 1) {
        int d, last = e->in_count - 1;
        for (d = e->anchor + 1; d <= last; d++)
            add_job(e, d, AEG_SLICE_P, 0);
        /* the tail becomes the chain of anchors from here on, so that the next
         * call does not queue it again */
        e->anchor = last;
        return run_job(e, pkt);
    }
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
    out->macroblocks = e->stat_mbs;
    out->skipped = e->stat_skipped;
    for (i = 0; i < AEG_MB_TYPE_COUNT; i++)
        out->mb_types[i] = e->stat_type[i];
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
            /* the frame that was just encoded: the picture it was coded into,
             * which is not necessarily the newest reference any more */
            plane[i] = e->pic_plane[e->role_last][i];
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
    aeg_mc_ctx_free(&e->mc1);
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
    aeg_free(e->col_mbs);
    aeg_free(e->in_pts);
    {
        int i;
        for (i = 0; i < (int)(sizeof(e->in_mem) / sizeof(e->in_mem[0])); i++)
            aeg_free(e->in_mem[i]);
        for (i = 0; i < AEG_MAX_BFRAMES + 3; i++)
            aeg_lowres_free(&e->in_low[i]);
        aeg_free(e->la_scratch);
        for (i = 0; i < 3; i++)
            aeg_free(e->pic_mem[i]);
    }
    aeg_free(e->src_mem);
    {
        int i;
        for (i = 0; i < AEG_MAX_SLICES; i++)
            aeg_free(e->slice_bs[i]);
    }
    aeg_free(e->out);
    aeg_free(e);
}
