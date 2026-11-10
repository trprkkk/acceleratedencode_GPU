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
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include "backend.h"
#define AEG_IMPL_RENAME
#include "api_impl.h"

/* The rate-distortion lambda of a QP, 0.85 * 2^((qp - 12) / 3), from a table
 * built with that same expression (the trellis asks for it per block).  qp
 * is QPY; the lambda is that of QP'Y = QPY + QP_BD_OFFSET, as in x264, which
 * is the 8-bit lambda times 4^(BitDepth - 8) -- the growth of a squared error
 * when every sample is scaled by 2^(BitDepth - 8). */
static double lambda_tab[128];
static pthread_once_t lambda_once = PTHREAD_ONCE_INIT;
static void lambda_init(void)
{
    int i;
    for (i = 0; i < 128; i++)
        lambda_tab[i] = 0.85 * pow(2.0, ((i - 32) - 12) / 3.0);
}
static inline double aeg_lambda_of(int qp)
{
    qp += QP_BD_OFFSET;
    if (qp < -32 || qp >= 96)
        return 0.85 * pow(2.0, (qp - 12) / 3.0);
    pthread_once(&lambda_once, lambda_init);
    return lambda_tab[qp + 32];
}
#define AEG_LAMBDA(qp) aeg_lambda_of(qp)

/* An integer tuning switch from the environment, read once.  The value is
 * cast to plain int: a statement expression ending in an _Atomic object is
 * itself _Atomic(int) to clang <= 18, which then refuses it in arithmetic. */
#define ENV_INT(name, dflt) __extension__ ({ static _Atomic int v_ = -0x7fffffff; \
    if (v_ == -0x7fffffff) { const char *e_ = getenv(name);              \
        v_ = e_ ? atoi(e_) : (dflt); } (int)v_; })

typedef struct mbenc {
    dctcoef  level[AEG_BLK_COUNT][16];
    int8_t   nz[AEG_BLK_COUNT];
    /* Luma coded with the 8x8 transform (High profile): one 64-coefficient
     * block per 8x8 group in raster order, the group's non-zero flag, and the
     * transform_size_8x8_flag itself.  `nz[]` still carries the per-4x4-block
     * numbers the CAVLC contexts need; for these macroblocks they hold each
     * list's coefficient count while the lists are coded and the block's 0/1
     * flag afterwards (see luma8_costs). */
    dctcoef  level8[4][64];
    int8_t   nz8[4];
    int      t8;
    int16_t  mv[16][2];             /* per 4x4 block, 1/4 pel, list 0        */
    uint8_t  ref[16];              /* P list0 reference index per 4x4      */
    uint8_t  ref1[16];             /* list 1's (MBAFF B field macroblocks) */
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

/* Upper bound on the B pictures between two anchors (x264's is 16 too).
 * Presets 3 and 4 use 8, the faster ones 3; the ring, the search-field cache
 * and the per-group arrays are sized from this. */
#define AEG_MAX_BFRAMES 16
/* list0 entries of a P picture at most (--ref); a B picture has one per
 * list.  The reference planes, their fractional planes and the per-reference
 * state are indexed 0..AEG_MAX_REFS-1. */
#define AEG_MAX_REFS 8
/* Entries of the reference buffer (max_num_ref_frames, aeg_sps_max_refs):
 * --ref frames, the pyramid's reference B, and with field pictures the frame
 * whose first field the second one still reads. */
#define AEG_MAX_DPB (AEG_MAX_REFS + 2)
/* Picture buffers: the two anchors around a group, its reference B picture
 * (B pyramid) and the picture being reconstructed. */
#define AEG_NPIC 24           /* picture buffers at most (frame threads) */
#define AEG_NPIC_SERIAL 4    /* ... and without frame threads, at least:
                              * max_refs + 1 with more references */
/* The macroblock tree looks this many pictures past the group being coded at
 * most; the input ring holds them on top of a group and its next anchor. */
#define AEG_MAX_LOOKAHEAD 60
#define AEG_RING (2 * AEG_MAX_BFRAMES + 6 + AEG_MAX_LOOKAHEAD)
/* lookahead costs kept (direct-mapped by a hash of the display numbers):
 * enough for every (p0, p1, b) the macroblock tree's 60-picture window asks
 * for with 16 B pictures, which the window mostly asks for again after it
 * slides by a group */
#define LA_CACHE 16384
/* lookahead search fields kept per ring slot: a B picture in a group of n
 * is searched against up to n + 1 pictures on either side */
#define LA_SF (2 * (AEG_MAX_BFRAMES + 1) + 2)
typedef struct {
    int     type;
    int     nparts;
    uint8_t sub_type[4];
    int     mv[16][2];
    int     mv1[16][2];
    int     pmv[16][2];
    int     pmv1[16][2];
    uint8_t blist[4];
    /* P list0 reference index per 4x4 block (0 or 1).  B pictures keep 0
     * (an MBAFF B field macroblock's are set from b_ref, bfld_stamp). */
    uint8_t ref[16];
    /* the motion search's cost (SATD + lambda * mvd bits) plus one;
     * 0 = not known (see b_rd_gate) */
    int     scost;
} mbcand;

struct aeg_encoder {
    aeg_params p;               /* first: api.c reads p.bit_depth through
                                 * the aeg_encoder pointer */
    int w, h, cw, ch;
    /* the picture as submitted (even, <= w x h): the coded picture is that
     * rounded up to whole macroblocks, the rest is edge samples repeated and
     * cropped away by the SPS (frame_cropping) */
    int vw, vh, vcw, vch;
    int mb_w, mb_h, mb_count;
    /* use_t8()'s answer while code_mb() tries the other transform size of
     * the winner: 0 = decide, 1 = 4x4, 2 = 8x8 (see t8_rd_on()) */
    int t8_force;
    /* The SATD me_partition_ref() measured for the vector it returned, when it
     * measured one (the fractional-offset test), else -1. */
    int me_last_satd;
    int stride[3];
    int frame_num_bits, level_idc;
    /* the parameter sets as every IDR carries them (aeg_encoder_headers) */
    uint8_t hdr[1024];
    size_t  hdr_len;
    int     level_out;        /* the level_idc the SPS signals            */
    /* dts: the submitted pts in submission (= display) order, popped one per
     * packet; packet k decodes at the k-th display time, `bframes` frames
     * early.  dts_dur is one frame in the caller's pts unit. */
    int64_t *dts_q;
    int     dts_cap, dts_head, dts_n;
    double  dts_dur;
    /* aeg_encoder_encode_image(): the converted picture (8-bit 4:2:0) */
    aeg_input_cvt cvt;
    pixel *cvt_pic;
    /* AEG_LATIME: time in the scheduler, waiting for and inside the frame
     * thread workers (per encoder: encoders may run in parallel threads) */
    int64_t sched_ns, ft_wait_ns, ft_worker_ns;

    pixel *src_mem;
    pixel *srcp[3];
    /* Three picture buffers, because a B picture needs two references alive at
     * once (the anchor before it and the anchor after it) *and* somewhere to
     * reconstruct itself.  The roles are buffer indices rather than pointers so
     * that the rotation below is one assignment per role; bind_pictures() puts
     * the plane pointers where the rest of the encoder expects them. */
    pixel *pic_mem[AEG_NPIC];
    pixel *pic_plane[AEG_NPIC][3];
    int role_cur;            /* buffer being reconstructed            */
    int role_last;           /* buffer the last coded picture went to  */
    int pic_poc[AEG_NPIC];   /* picture order count held by each buffer   */
    int nrefs;               /* reference pictures alive (= ndpb)      */
    /* The decoder's reference picture buffer, mirrored (see picture_rotate):
     * which buffer, its POC and frame_num, and whether it is an anchor (a
     * reference B picture of a pyramid is not). */
    /* fm: the fields of the entry that are reference fields (1 top,
     * 2 bottom; 3 for a frame) -- field pictures mark them one at a time */
    struct { int slot, poc, fn, anchor, fm; } dpb[AEG_MAX_DPB];
    int ndpb;
    int max_refs;            /* max_num_ref_frames of the SPS            */
    aeg_refctl refctl;       /* list modification / MMCO of this picture */
    int cur_l1_slot;         /* buffer of list1[0] (co-located picture)  */
    /* Per buffer: the macroblock decisions of the reference picture in it,
     * for direct mode when it is some B picture's list1[0]. */
    aeg_mb *slot_mbs[AEG_NPIC];
    int slot_ref_poc[AEG_NPIC][AEG_MAX_REFS];
    int slot_col_ok[AEG_NPIC];
    /* Per buffer: what the pictures predicting from the reference in it need
     * besides its samples -- the three half-sample planes and the half-
     * resolution copy of the lowres pre-search -- built once, by the picture
     * itself, band by band as its rows become final (rowpost_*).  slot_rows
     * counts the macroblock rows whose samples (padding included), half-pel
     * rows and lowres rows are final; mb_h means the whole picture, borders
     * and all. */
    aeg_mc_ctx slot_mc[AEG_NPIC];
    aeg_lowres slot_low[AEG_NPIC];
    int slot_rows[AEG_NPIC];
    struct ft_sync *fsync;   /* frame threads: slot_rows shared, waitable */
    int npic;                /* picture buffers in use (AEG_NPIC max)    */
    int npic_serial;         /* ... of them without frame threads        */
    int pic_busy[AEG_NPIC];  /* frame threads: jobs still reading/writing */
    /* frame threads, in a worker's copy: the generation of each reference
     * buffer as the picture started reading it (a reference that restarts
     * bumps its generation, and everything that read it restarts too), and
     * whether the picture runs ahead of its weighted-prediction analysis */
    int ft_gslot[AEG_MAX_REFS];
    unsigned ft_gen[AEG_MAX_REFS];
    int ft_reg;              /* counted in fsync->readers                 */
    int ft_spec;             /* coding unweighted until the ref is whole  */
    int ft_abort;            /* a stale reference was seen: start again   */
    int ref_slot[AEG_MAX_REFS]; /* buffer of refpl[k][0], or -1          */
    /* row-by-row post-processing of the reference picture being coded */
    int rp_on;               /* deblocking etc. follow the decision rows  */
    int rp_dbk;              /* rows deblocked                            */
    int rp_band;             /* rows padded (and lowres-built)            */
    int rp_hpel;             /* rows with their half-pel planes           */
    int rp_qp;               /* the bitstream pass's QP, mirrored         */
    aeg_mb *rp_mbs;          /* final decisions the filter reads          */
    /* AEG_POISON=1 (debug, serial): every reference is read through a copy
     * whose rows past ref_rows_needed() are garbage until the macroblock row
     * that may read them, so a stream identical to the normal one proves the
     * bound the frame threads wait on (ref_rows_ensure). */
    int pz_on, pz_nl, pz_rows[AEG_MAX_REFS];
    pixel *pz_mem[AEG_MAX_REFS], *pz_plane[AEG_MAX_REFS][3];
    aeg_mc_ctx pz_mc[AEG_MAX_REFS];
    aeg_lowres pz_low[AEG_MAX_REFS];
    aeg_mb *pz_mbs;
    const pixel *pz_src[AEG_MAX_REFS][3], *pz_hsrc[AEG_MAX_REFS][4];
    const aeg_lowres *pz_lsrc[AEG_MAX_REFS];
    const aeg_mb *pz_msrc;
    /* ... and of the MBAFF field copies the field macroblocks read
     * (slot_fpl, slot_fmc) of buffer pzf_slot[l], list entry l's */
    int pzf_slot[AEG_MAX_REFS];
    pixel *pzf_mem[AEG_MAX_REFS], *pzf_plane[AEG_MAX_REFS][3];
    aeg_mc_ctx pzf_mc[AEG_MAX_REFS];
    /* ... and in a field picture (pzl_setup): copies of the reference
     * frames (frame layout, frame rows) its field views read */
    int pzl_on, pzl_slot[AEG_MAX_REFS], pzl_rows[AEG_MAX_REFS];
    pixel *pzl_mem[AEG_MAX_REFS], *pzl_plane[AEG_MAX_REFS][3];
    aeg_mc_ctx pzl_mc[AEG_MAX_REFS];
    aeg_lowres pzl_low[AEG_MAX_REFS];
    const aeg_lowres *pzl_lsrc[AEG_MAX_REFS];
    aeg_mb *pzl_mbs;
    const aeg_mb *pzl_msrc;
    int job_bref[AEG_MAX_BFRAMES + 4];
    int cur_bref;            /* the picture being coded is a reference B */
    pixel *recp[3];
    /* reference planes: list0[k] of a P picture is refpl[k]; a B picture
     * has list0[0] in refpl[0] and list1[0] in refpl[1] */
    pixel *refpl[AEG_MAX_REFS][3];

    aeg_mb  *mbs;
    aeg_picture pic;

    /* per-macroblock decision scratch filled by analyse_mb() and consumed by
     * code_mb(): the intra modes ordered by SAD and the inter candidate. */
    int mb_modes[4];
    int mb_nmodes;
    int mb_inter_ok;
    /* x264's p_halfpel_thresh over one macroblock's 16x16 reference searches
     * (hpt_on): the best integer-pel cost so far, and whether the last
     * me_partition_ref() stopped before its sub-pel stage */
    int me_hpt, me_hpt_on, me_pruned;
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
    int mb_maxref;           /* partitions search list0 up to this index */

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

    /* code_chroma() results of the current code_mb() call, keyed by what they
     * depend on (see code_chroma()); cmemo_gen moves on per macroblock. */
    struct cmemo {
        unsigned gen;
        int inter, qp;
        pixel pred[2][64];
        dctcoef level[10][16];      /* AEG_BLK_CHROMA_DC .. CHROMA_CR + 3 */
        int8_t nz[10];
        int cbp;                    /* the chroma bits */
        int64_t dsse;
        int dbits;
        pixel rec[2][64];
    } cmemo[8];
    unsigned cmemo_gen;
    int cmemo_next;

    /* The same for the luma of the inter candidates (see rd_luma()): keyed by
     * the prediction and the few settings the residual coding reads, valid for
     * one code_mb() call (lmemo_n is reset there). */
    struct lmemo {
        int qp, t8ok, t8force, trellis;
        pixel pred[256];
        pixel rec[256];
        mbenc t;                    /* before the header bits */
    } lmemo[8];
    int lmemo_n, lmemo_next;

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
    dctcoef coef_lev[16 * 16];
    int16_t coef_rec[16 * 16];
    int32_t coef_nz[16];
    /* RD batching (provisional, see docs/STATUS.md): every inter candidate of
     * the macroblock is predicted and its residual gathered first, then one
     * tq_batch() and one iq_idct_batch() cover all of the candidates at once,
     * and only then are the costs compared.  Four candidates x sixteen blocks
     * is the largest batch a macroblock can produce. */
    int rd_batch;
    int coef_kind, coef_dev;    /* this picture's placement (resolve_coef_stages) */
    int coef_force, rd_force;   /* -1: ask the dispatcher, 0/1: forced */
    int16_t rd_res[16 * 16 * AEG_MB_CAND_MAX];
    dctcoef rd_lev[16 * 16 * AEG_MB_CAND_MAX];
    int16_t rd_rec[16 * 16 * AEG_MB_CAND_MAX];
    int32_t rd_nz[16 * AEG_MB_CAND_MAX];
    pixel rd_pred[16 * 16 * AEG_MB_CAND_MAX];
    pixel rd_predc[2 * 8 * 8 * AEG_MB_CAND_MAX];
    /* Whole-frame search seed: 2 ints per macroblock, integer pel.  Filled by
     * me_seed_frame() with the best of a fixed candidate grid, evaluated as one
     * batch through the backend (this is the stage the GPU actually runs). */
    int16_t *seed_mv;      /* integer vectors of the parallel rounds      */
    /* Half-resolution pre-search (lr_seed_frame): the current source and the
     * reference reconstructions box-filtered by two, one 8x8 block per
     * macroblock searched over twice the full-resolution reach; the vectors
     * (whole samples, per list) start the full search. */
    aeg_lowres lr_pic[3];
    int     *lr_scratch;
    int16_t *lr_mv[2];
    int      lr_valid;
    int      lr_rows;        /* macroblock rows of lr_mv computed so far  */
    int      lr_nl;          /* lists searched                            */
    const aeg_lowres *lr_ref[2];
    int16_t *par_pos;      /* 4 int16 per candidate for satd_batch       */
    int16_t *par_mv;       /* quarter-pel result per macroblock          */
    int32_t *par_cost;
    int      par_valid;
    int      par_full;
    int16_t *seed_cand;   /* 2 * ME_ROUND_CAND per macroblock */
    int16_t *prev_seed;   /* the same, as of the previous picture  */
    int32_t *seed_sad;
    int32_t *seed_cost;
    pixel pred_l[16 * 16];
    pixel pred_c[2][8 * 8];
    /* The second list's predictor, and the pair averaged in place, for a
     * bi-predictive macroblock of a B picture. */
    pixel pred_l2[16 * 16];
    pixel pred_c2[2][8 * 8];
    aeg_mc_ctx mcr[AEG_MAX_REFS]; /* fractional planes of refpl[k][0] */
    int has_ref;
    int has_ref2;
    /* P slice with two active list0 entries: the newest and the oldest
     * reference are both in list0 (H.264 7.4.3: num_ref_idx_active_override).
     * B keeps one per list. */
    int cur_l0_active;
    /* Explicit weighted prediction of the current P picture (list 0 only;
     * see wp_analyse).  When the luma of reference 0 is weighted, refpl[0]
     * points at a weighted copy -- the motion search then compares against
     * what the prediction will roughly be -- and wp_orig[0] at the real
     * reference, which the exact prediction interpolates first and weights
     * afterwards, as the decoder does (8.4.2.3). */
    aeg_pwt wp;
    pixel *wp_mem, *wp_plane[3];
    const pixel *wp_orig0;
    int fwp;                /* field pictures send weights too (p.weightp
                             * with field_coding 1, 2 or 4; AEG_NOFWP) */
    int cur_is_p;           /* the picture being coded is a P slice */
    int cur_is_b;           /* ... is a B slice                     */
    int cur_is_intra;       /* ... is an I slice (IDR or not)       */
    int mode_lambda_off;    /* QP offset of a B picture over its anchors:
                             * the mode decision prices with the lambda of
                             * the QP without it (see code_mb)         */
    int cur_is_ref;         /* ... is a reference picture           */
    int cur_poc;            /* its picture order count              */
    int poc_lsb_bits;       /* log2_max_pic_order_cnt_lsb           */
    int prev_poc;           /* POC of the previous picture, decode order */
    int ref_poc[AEG_MAX_REFS]; /* POC of refpl[k]                         */
    int col_ready;          /* the co-located picture's decisions are usable */
    aeg_mb *col_mbs;        /* ... and they live here (see b_direct_mv)  */
    int col_poc;            /* POC of co-located picture */
    int col_ref_poc[AEG_MAX_REFS]; /* POC of its L0 references (if any) */
    /* Field pictures (p.field_coding 1).  A frame is coded as two pictures,
     * each a view of the frame's buffers: every other row (the plane plus
     * the parity's row, twice the stride), half the height.  field_enter()
     * swaps the geometry below for the field's and field_leave() puts the
     * frame's back, so everything that codes a picture codes a field
     * unchanged; the frame keeps the lookahead, the rate control and the
     * buffers.  fld: a field is being coded; fld_idx: first (0) or second
     * (1) in decoding order; fld_par: its parity (0 top, 1 bottom). */
    int fld, fld_idx, fld_par;
    /* p.field_coding 3: every picture an MBAFF frame -- macroblocks in
     * pairs (top, bottom; then the next pair), each pair a frame or a field
     * pair (aeg_mb.fld) */
    int mbaff;
    int mbaff_ff;     /* mb_field_decoding_flag to write, -1 none (bitstream) */
    /* the macroblock being coded is a field macroblock of an MBAFF frame:
     * mbf_enter() has pointed srcp/recp/stride at its field's lines and the
     * scans and contexts at the field ones (mbf_save: what it replaced) */
    int cfld;
    int pair_fld;     /* the decision for the current pair (decision pass) */
    struct {
        pixel *srcp[3], *recp[3];
        int stride[3];
        /* a P picture's frame references, while a field macroblock has its
         * field references in their place (refs: the swap is in) */
        int refs, l0, has_ref2;
        pixel *refpl[AEG_MAX_REFS][3];
        aeg_mc_ctx mcr[AEG_MAX_REFS];
        int ref_poc[AEG_MAX_REFS], cofs[AEG_MAX_REFS];
    } mbf_save;
    /* MBAFF: per buffer, the reference as field macroblocks read it -- the
     * same samples with each field's own borders, and half-sample planes
     * filtered within each field (mbaff_field_shadow, once the reference
     * picture is complete, or band by band behind rowpost_pad when the
     * picture is post-processed row by row: rowpost_fshadow) */
    pixel *slot_fmem[AEG_NPIC], *slot_fpl[AEG_NPIC][3];
    int rp_fsh;                        /* frame bands copied into it       */
    int rp_fhp;                        /* field bands with half-pel planes */
    int rp_fon;                        /* this picture's built that way    */
    int rp_lfon;                       /* a field's slot_lowf, row by row  */
    aeg_mc_ctx slot_fmc[AEG_NPIC];
    uint8_t slot_fok[AEG_NPIC];        /* ... holds the buffer's picture */
    /* The current P picture's field references for a top (0) and a bottom
     * (1) field macroblock: list0 field k is the field of frame k >> 1 with
     * the macroblock's parity when k is even, the other one when odd
     * (8.4.2.1); fref_n of them (no more than AEG_MAX_REFS), 0: none. */
    int fref_n;
    /* a B picture's: entry 0 is list 0's field of the macroblock's parity,
     * entry 1 list 1's, entries 2 and 3 the other parity's (field reference
     * index 1 of each list, 8.4.2.1) */
    int fref_b;
    /* A field macroblock of an MBAFF B picture predicts each list from one
     * field (b_ref[list]: 0 = its parity, 1 = the other), chosen up front
     * (bfld_prep); entry `list` of refpl / mcr is that field, entry list + 2
     * the other.  bfld_on: this macroblock does; bdr_no: direct mode may not
     * be offered (its derived indices are not the chosen ones). */
    int b_ref[2], bfld_on, bdr_no;
    /* A B field picture (PAFF) whose lists hold two fields each: list0 and
     * list1 index 1 are reference entries 2 and 3, and every macroblock
     * chooses per list as an MBAFF field macroblock does (bfld_prep).  The
     * entries as set_field_refs made them, which bfld_prep picks from: */
    int bfp_on;
    struct bfp_ent {
        pixel *pl[3];
        aeg_mc_ctx mc;
        int poc, cofs, slot, par;
    } bfp_sv[4];
    pixel *fref_pl[2][AEG_MAX_REFS][3];
    aeg_mc_ctx fref_mc[2][AEG_MAX_REFS];
    int fref_poc[2][AEG_MAX_REFS], fref_cofs[2][AEG_MAX_REFS];
    int ref_par[AEG_MAX_REFS];  /* parity of the field refpl[k] views     */
    int fld_cofs[AEG_MAX_REFS]; /* its chroma vertical offset (8.4.1.4)   */
    int vpad;                   /* rows of padding above/below a picture  */
    /* p.field_coding 2 (frame or fields per picture): how each buffer's
     * borders and half-sample planes are laid out (0 a frame's, 1 two
     * fields'), and how the last anchor was coded (its B pictures follow) */
    uint8_t slot_fst[AEG_NPIC];
    int anchor_fld;
    /* field coding (1, 2, 4): per buffer, the half-resolution copy of each
     * of its fields (the lowres pre-search of a field picture), built row by
     * row as the field is post-processed (rowpost_bands); slot_lfok: they
     * hold the buffer's picture.  Frame dimensions, as the copy a field
     * picture would build of its reference itself (lr_seed_frame): its rows
     * past the field's height repeat the field's last row. */
    aeg_lowres slot_lowf[AEG_NPIC][2];
    uint8_t slot_lfok[AEG_NPIC];
    /* the coefficient scans (field scans in a field) and the matching CABAC
     * category offset / CAVLC flags */
    const uint8_t *scan4, *scan8;
    int fcat, fscan, fscan8;
    struct fld_save {
        int h, ch, vh, vch, mb_h, mb_count, stride[3], slice_rows, cur_poc;
        int cur_is_p, cur_is_b, cur_is_intra;
        pixel *srcp[3], *recp[3];
        aeg_picture pic;
        int8_t *mb_dqp;
    } fsv;
    /* Lookahead: input pictures are held in display order until the anchors
     * they are coded between have arrived (see aeg_encoder_encode()). */
    pixel *in_mem[AEG_RING];
    pixel *in_plane[AEG_RING][3];
    int64_t *in_pts;
    aeg_lowres in_low[AEG_RING];   /* b_adapt / mbtree: lowres of the ring */
    int *la_scratch;
    /* the lookahead's per-picture fields (la_field): the intra estimate of
     * the picture in each ring slot, and its searches against up to LA_SF
     * references, tagged with display numbers */
    struct la_sfield { int d, r, age; int *f; } la_sf[AEG_RING][LA_SF];
    int la_icost_d[AEG_RING];
    int *la_icost[AEG_RING];
    int la_age;
    /* lookahead costs already computed, keyed by display indices (the
     * window slides by a group, so most are reused) */
    struct { int p0, p1, b, intra; int64_t c; } la_cache[LA_CACHE];
    int la_sf_n;            /* search fields used per slot (<= LA_SF)        */
    int la_seed;            /* lowres search at distance n seeded from n - 1:
                             * the range grows by this much per step, 0 = off */
    int ba_wide;            /* the B-adapt trellis spans the whole ring       */
    int la_depth;           /* pictures the mbtree looks past the group      */
    int flushing;           /* the input has ended (aeg_encoder_flush)       */
    /* Macroblock tree, per ring slot: the lowres costs of the picture
     * against the references it was last planned with (key d/p0/p1), its
     * propagate_in, and the QP offsets once it is queued (valid for qd). */
    struct mbt_slot {
        int d, p0, p1;
        int *intra, *inter, *mv;
        uint8_t *kind;
        float *prop, *qp;
        int qd;
    } mbt[AEG_RING];
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

    /* a bitrate target (`AEG_RC_ABR`; the controller is abr_on's, below) */
    int    rc_active;
    int    rc_qp_offset;    /* user bias on the per-picture QP (two-pass)     */
    int    in_qpoff[AEG_RING]; /* ... as it was when the picture came in    */

    /* CRF (`AEG_RC_CRF`, x264's constant rate factor): an anchor's QP follows
     * qscale = blurred_complexity^(1 - qcomp) / crf_rf, the complexity being
     * the lookahead's estimate of the picture coded as P.  `crf_cplx` holds
     * that estimate per ring slot, written when the picture is scheduled; the
     * picture QP is fractional, the fraction going to a run of macroblocks
     * coded one step coarser (crf_dither). */
    int     crf_on;
    double  crf_rf;           /* rate factor                               */
    double  crf_qc;           /* 1 - effective qcomp (0 with the tree)     */
    double  crf_tree_off;     /* the QP shift: tree centring + calibration */
    double  mbt_avg;          /* mean tree offset of the current picture */
    int     mbt_on;           /* ... which has tree offsets               */
    double  crf_sum, crf_cnt; /* the short-term complexity blur            */
    double  crf_pqp, crf_pnorm; /* decaying average of the anchors' QP     */
    double  crf_anchor_q;     /* the last anchor's fractional QP            */
    double  crf_frac;         /* this picture's fraction (0..1)             */
    int64_t *crf_cplx;        /* per ring slot (-1 = none)                  */
    aeg_lowres crf_low[2];    /* without B pictures: this and the last input */
    int64_t crf_last;         /* the last estimate (no-B path)              */
    /* ABR, x264's one-pass form, on the model above (crf_on is set as well):
     * the rate factor is no longer a constant but what the stream so far
     * says it should have been -- the bits wanted for the pictures coded
     * over the sum of each picture's bits * qscale / rceq -- and a bounded
     * correction pulls the total back towards the target. */
    int     abr_on;
    double  abr_bpf;          /* bits per picture                          */
    double  abr_fps;
    double  abr_cplxr;        /* sum of bits * qscale / rceq (+ the guess) */
    double  abr_wanted;       /* sum of abr_bpf (+ one for the guess)      */
    double  abr_total;        /* bits spent                                */
    int     abr_frames;
    int     abr_prev_intra;   /* the last picture coded was intra          */       /* pictures coded                            */
    double  abr_rceq;         /* this picture's complexity term            */
    double  abr_last_rceq;    /* the last anchor's, for the B pictures     */
    double  abr_last_pqs;     /* the last P picture's qscale (step limit)  */
    double  abr_qm;           /* this picture's model QP                   */
    int     abr_qpre;         /* cur_qp before the VBV / qp range clamps   */
    /* Two-pass: the QP planned for each picture (display order) from the
     * first pass's statistics, the bits that plan expects, and how far the
     * pictures coded so far are from it. */
    double *plan_q, *plan_bits;
    unsigned char *plan_cls;  /* 0 intra, 1 P, 2 B, as pass 1 coded it    */
    int     plan_n;
    double  plan_total, plan_exp_sum;

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
    /* Field pictures: the second field of a frame leaves the buffer a clock
     * tick (half a picture period) after the first, so the model advances
     * in ticks: `vbv_gap` ticks of delivery before the next access unit (2
     * after a frame, 1 after a field pair), and `vbv_f1` the first field's
     * bits of the frame being accounted (0: a frame picture). */
    int    vbv_gap;
    int    vbv_f1;
    /* the NAL HRD signalled from that model (nal_hrd): rate and CPB size as
     * the SPS states them, and the coded index of the last buffering period */
    int    hrd_on;
    double hrd_rate, hrd_cpb;
    int    hrd_bp_c;
    int    hrd_bp_prev;     /* the buffering period before hrd_bp_c's */
    char   info[768];       /* the info SEI's text (info_sei)                */
    double vbv_min_full;    /* lowest level seen                            */
    int    vbv_underflows;
    int    vbv_clamps;
    int    vbv_recodes;     /* pictures coded again to fit (frame_body)     */
    int    last_pic_bits;   /* the previous picture, to predict this one    */
    int    last_pic_qp;
    /* The previous *anchor*, for the VBV model: a B picture is much cheaper
     * than the anchor before it and much cheaper than the one after it, so
     * predicting a B picture from whatever picture happened to come last swings
     * between over- and under-estimating by a factor of several, and the clamp
     * would then fire on a buffer that is nowhere near empty. */
    int    last_anchor_bits;
    int    last_anchor_qp;
    /* ... and per type: an intra picture is several times a P picture, so
     * neither may be predicted from the other while the type has its own */
    int    last_i_bits, last_i_qp;
    int    last_p_bits, last_p_qp;
    /* x264's size predictor (predict_size / update_predictor):
     * bits = (coef * satd + offset) / (qscale * count) per picture type, satd
     * being the lowres estimate of the picture as it is coded (intra, against
     * its reference, or between its two), the sums decaying by half per
     * picture of the type.  vbv_satd per ring slot is written when the
     * picture is queued (cur_satd: the picture being coded, -1 = none). */
    double vbv_coef[3], vbv_off[3], vbv_cnt[3];
    int64_t *vbv_satd;
    int64_t cur_satd;
    int    job_nb_last, job_nb_prev; /* the last two non-B pictures queued */
    int    mbt_qoff;        /* mean macroblock-tree offset of this picture */
    /* Frame threads (see "Frame threads" near aeg_encoder_encode): a worker
     * that codes non-reference B pictures next to the main thread, and the
     * queue that hands the packets out in coding order. */
    int    ft_on;
    int    ft_worker_copy;  /* a frame-thread worker's copy of the encoder */
    /* Distortion of the last coded picture (pic_quality) and its sums over
     * the encode (qual_account, on the encoder itself only: a frame-thread
     * worker's picture is counted when its packet is handed out). */
    int64_t pic_sse[3];
    double  pic_ssim;
    int64_t qual_sse[3], qual_n;
    double  qual_ssim;
    /* The device lane this copy's thread uses (aeg_be_lane): 0 for the
     * encoder itself, 1 + k * (slice threads) for frame-thread worker k, plus
     * the slice index for the copies of a picture's slice threads -- so that
     * no two threads coding at the same time wait for each other's device
     * calls. */
    int    lane;
    struct ft_worker *ftw;   /* ft_nw workers */
    int    ft_nw;
    int64_t ft_seq;
    struct ft_entry  *ftq;
    int    ftq_head, ftq_n, ft_ret;
};

static int ft_alloc(struct aeg_encoder *e, int nw);
static void ft_free(struct aeg_encoder *e);

/* One measured call of a stage's batch operation: to the dispatcher (its
 * moving averages decide the placement) and to the statistics, whose per-stage
 * device / host totals say where the time actually went -- the copies a
 * picture's slice threads and frame threads run in sum theirs into the
 * encoder's. */
static void stage_rec(struct aeg_encoder *e, int stage, int on_gpu, int64_t ns)
{
    aeg_dispatch_record(&e->d, stage, on_gpu, ns);
    if (on_gpu)
        e->st.stage_gpu_ns[stage] += ns;
    else
        e->st.stage_cpu_ns[stage] += ns;
}

/* x264's SSIM of one 8x8 window from its sums: s1, s2 of the two pictures,
 * ss of both squares, s12 of the products (x264 ssim_end1, the constants
 * scaled to the sample range). */
static double ssim_window(double s1, double s2, double ss, double s12)
{
    const double c1 = .01 * .01 * PIXEL_MAX * PIXEL_MAX * 64;
    const double c2 = .03 * .03 * PIXEL_MAX * PIXEL_MAX * 64 * 63;
    double vars = ss * 64 - s1 * s1 - s2 * s2;
    double covar = s12 * 64 - s1 * s2;
    return (2 * s1 * s2 + c1) * (2 * covar + c2) /
           ((s1 * s1 + s2 * s2 + c1) * (vars + c2));
}

/* The luma SSIM of the visible picture: 8x8 windows on a 4-sample grid, as
 * x264 --ssim measures it. */
static double pic_ssim_y(const struct aeg_encoder *e)
{
    const pixel *a = e->srcp[0], *b = e->recp[0];
    int stride = e->stride[0], w4 = e->vw / 4, h4 = e->vh / 4;
    double sum = 0;
    int x, y, n = 0;
    for (y = 0; y + 1 < h4; y++)
        for (x = 0; x + 1 < w4; x++) {
            int64_t s1 = 0, s2 = 0, ss = 0, s12 = 0;
            int i, j;
            for (j = 0; j < 8; j++)
                for (i = 0; i < 8; i++) {
                    int pa = a[(4 * y + j) * stride + 4 * x + i];
                    int pb = b[(4 * y + j) * stride + 4 * x + i];
                    s1 += pa;
                    s2 += pb;
                    ss += pa * pa + pb * pb;
                    s12 += pa * pb;
                }
            sum += ssim_window((double)s1, (double)s2, (double)ss, (double)s12);
            n++;
        }
    return n ? sum / n : 1.0;
}

/* Distortion of the picture just reconstructed, against its source, over
 * the visible area (the cropped margin is not output): the squared error of
 * each plane, and the luma SSIM when aeg_params.ssim asks for it.  Timed as
 * the PSNR stage, always on the CPU. */
static void pic_quality(struct aeg_encoder *e)
{
    int64_t t0 = aeg_now_ns(), dt;
    int i;
    for (i = 0; i < 3; i++) {
        int w = i ? e->vcw : e->vw, h = i ? e->vch : e->vh;
        int x, y;
        int64_t acc = 0;
        for (y = 0; y < h; y++)
            for (x = 0; x < w; x++) {
                int d = (int)e->recp[i][y * e->stride[i] + x] -
                        (int)e->srcp[i][y * e->stride[i] + x];
                acc += (int64_t)d * d;
            }
        e->pic_sse[i] = acc;
    }
    e->pic_ssim = e->p.ssim ? pic_ssim_y(e) : 0.0;
    dt = aeg_now_ns() - t0;
    e->st.stage_ns[AEG_STAGE_PSNR] += dt;
    e->st.stage_cpu_ns[AEG_STAGE_PSNR] += dt;
}

static double psnr_of(int64_t sse, int64_t samples)
{
    if (sse <= 0)
        return 99.0;
    return 10.0 * log10((double)PIXEL_MAX * PIXEL_MAX * (double)samples /
                        (double)sse);
}

/* Counts one picture's distortion into the encode's aeg_stats: psnr_y/u/v
 * and psnr_avg from the squared error summed over every picture so far
 * (x264's "global" PSNR; psnr_avg weighs the planes by their samples), and
 * ssim_y the mean of the pictures' SSIM. */
static void qual_account(struct aeg_encoder *e, const int64_t sse[3],
                         double ssim)
{
    int64_t ny = (int64_t)e->vw * e->vh, nc = (int64_t)e->vcw * e->vch;
    int i;
    for (i = 0; i < 3; i++)
        e->qual_sse[i] += sse[i];
    e->qual_ssim += ssim;
    e->qual_n++;
    e->st.psnr_y = psnr_of(e->qual_sse[0], e->qual_n * ny);
    e->st.psnr_u = psnr_of(e->qual_sse[1], e->qual_n * nc);
    e->st.psnr_v = psnr_of(e->qual_sse[2], e->qual_n * nc);
    e->st.psnr_avg = psnr_of(e->qual_sse[0] + e->qual_sse[1] + e->qual_sse[2],
                             e->qual_n * (ny + 2 * nc));
    e->st.ssim_y = e->p.ssim ? e->qual_ssim / (double)e->qual_n : 0.0;
}

/* Which of the two batched coefficient code paths to use for this picture.
 * Both need the device to own *both* coefficient stages: an inter candidate
 * cannot transform on one and reconstruct on the other without a round trip in
 * the middle of its RD cost.  AEG_COEF_BATCH / AEG_RD_BATCH override the
 * dispatcher, which is how the conformance cases exercise the paths on a
 * machine with no GPU. */
static void resolve_coef_stages(struct aeg_encoder *e)
{
    const int kind = e->cur_is_b ? 2 : (e->cur_is_p ? 1 : 0);
    int want = e->d.ops.backend != AEG_BACKEND_CPU &&
               e->d.ops.tq_batch && e->d.ops.iq_idct_batch &&
               aeg_dispatch_coef_pick(&e->d, kind);
    /* remembered for aeg_dispatch_coef_done() at the end of the picture, and
     * mirrored into the stage decisions (the statistics report those) */
    e->coef_kind = kind;
    e->coef_dev = want;
    if (e->d.have_gpu && e->d.forced[AEG_STAGE_TRANSFORM] < 0 &&
        e->d.forced[AEG_STAGE_IQUANT_IDCT] < 0)
        e->d.use_gpu[AEG_STAGE_TRANSFORM] =
            e->d.use_gpu[AEG_STAGE_IQUANT_IDCT] = want;
    e->coef_batch = e->coef_force >= 0 ? e->coef_force : want;
    e->rd_batch = e->rd_force >= 0 ? e->rd_force : want;
    if (e->rd_batch)
        e->coef_batch = 1;
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

static int enc_alloc_picture(struct aeg_encoder *e, pixel **mem, pixel **plane)
{
    /* (vpad rows above and below: field pictures read every buffer as two
     * pictures of half the height, which need AEG_PAD rows each) */
    const size_t vp = (size_t)e->vpad;
    size_t ls = (size_t)plane_stride(e->w), lh = (size_t)e->h + 2 * vp;
    size_t cs = (size_t)plane_stride((e->w + 1) / 2);
    size_t ch = (size_t)(e->h + 1) / 2 + 2 * vp;
    size_t total = ls * lh + 2 * cs * ch;

    *mem = (pixel *)aeg_alloc(total * sizeof(pixel));
    if (!*mem)
        return AEG_ENOMEM;
    /* Every plane occupies exactly lh*ls (or ch*cs) bytes starting at its own
     * origin: the left/right padding is part of the stride and the top/bottom
     * padding fits inside the row count.  Placing plane 2 at plane 1 plus one
     * chroma plane -- rather than plus the padding of the next origin -- used to
     * write one row band past the end of the allocation. */
    plane[0] = *mem + vp * ls + AEG_PAD;
    plane[1] = *mem + ls * lh + vp * cs + AEG_PAD;
    plane[2] = *mem + ls * lh + cs * ch + vp * cs + AEG_PAD;
    e->stride[0] = (int)ls;
    e->stride[1] = e->stride[2] = (int)cs;
    return AEG_OK;
}

static size_t pic_bytes(const struct aeg_encoder *e)
{
    const size_t vp = (size_t)e->vpad;
    size_t ls = (size_t)plane_stride(e->w), lh = (size_t)e->h + 2 * vp;
    size_t cs = (size_t)plane_stride((e->w + 1) / 2);
    size_t ch = (size_t)(e->h + 1) / 2 + 2 * vp;
    return (ls * lh + 2 * cs * ch) * sizeof(pixel);
}

/* Picture buffer i with what a reference keeps next to its samples. */
static int slot_alloc(struct aeg_encoder *e, int i)
{
    int st[3] = { e->stride[0], e->stride[1], e->stride[2] }, k;
    if (enc_alloc_picture(e, &e->pic_mem[i], e->pic_plane[i]) != AEG_OK)
        return AEG_ENOMEM;
    if (st[0])
        e->stride[0] = st[0], e->stride[1] = st[1], e->stride[2] = st[2];
    aeg_mc_ctx_init_pad(&e->slot_mc[i], e->w, e->h, e->stride[0], e->vpad);
    if (!e->slot_mc[i].buf)
        return AEG_ENOMEM;  /* references past the second have no others */
    if (aeg_lowres_alloc(&e->slot_low[i], e->mb_w, e->mb_h))
        e->slot_low[i].mem = NULL;
    e->slot_mbs[i] = (aeg_mb *)aeg_alloc((size_t)e->mb_count * sizeof(aeg_mb));
    if (!e->slot_mbs[i])
        return AEG_ENOMEM;
    if (e->p.field_coding && e->p.field_coding != 3 && !e->slot_lowf[i][0].mem)
        for (k = 0; k < 2; k++)
            if (aeg_lowres_alloc(&e->slot_lowf[i][k], e->mb_w, e->mb_h))
                e->slot_lowf[i][k].mem = NULL;
    if (e->p.field_coding >= 3 && !e->slot_fmem[i]) {
        /* MBAFF: the field copy, here rather than on first use (a frame
         * thread's worker builds it in its own copy of the encoder) */
        if (enc_alloc_picture(e, &e->slot_fmem[i], e->slot_fpl[i]) != AEG_OK)
            return AEG_ENOMEM;
        aeg_mc_ctx_init_pad(&e->slot_fmc[i], e->w, e->h, e->stride[0], e->vpad);
    }
    return AEG_OK;
}

/* Point recp at the buffer being reconstructed.  The reference pointers
 * (refpl[]) are set per picture by set_current_refs(). */
static void bind_pictures(struct aeg_encoder *e)
{
    int i;

    for (i = 0; i < 3; i++)
        e->recp[i] = e->role_cur >= 0 ? e->pic_plane[e->role_cur][i] : NULL;
    e->nrefs = e->ndpb;
    e->has_ref = e->ndpb > 0;
    e->has_ref2 = e->ndpb > 1;
}

static int dpb_holds(const struct aeg_encoder *e, int slot)
{
    int i;
    for (i = 0; i < e->ndpb; i++)
        if (e->dpb[i].slot == slot)
            return 1;
    return 0;
}

/* A buffer that is neither being reconstructed nor a reference picture. */
static int pic_free(const struct aeg_encoder *e)
{
    int i;
    for (i = 0; i < e->npic; i++)
        if (i != e->role_cur && !dpb_holds(e, i) && !e->pic_busy[i])
            return i;
    return -1;
}

/* The decoder's reference marking (8.2.5), mirrored.  An IDR picture empties
 * the buffer; a picture whose header carries memory_management_control
 * operations (refctl, planned by set_current_refs) drops exactly those
 * pictures; any other reference picture goes through the sliding window,
 * which drops the one with the smallest frame_num once max_num_ref_frames
 * are stored.  The picture just coded is then stored, and the buffer it
 * frees -- or any buffer no reference holds -- is where the next picture is
 * reconstructed.  A non-reference B picture changes nothing: its buffer is
 * simply reused. */
static void picture_rotate(struct aeg_encoder *e, int is_ref, int is_idr)
{
    int i, k;

    e->role_last = e->role_cur;
    if (is_ref) {
        if (is_idr) {
            e->ndpb = 0;
        } else if (e->refctl.n_mmco) {
            for (k = 0; k < e->refctl.n_mmco; k++)
                for (i = 0; i < e->ndpb; i++)
                    if (e->dpb[i].slot == e->refctl.mmco_slot[k]) {
                        e->dpb[i] = e->dpb[--e->ndpb];
                        break;
                    }
        } else if (e->ndpb >= e->max_refs) {
            int m = 0;
            for (i = 1; i < e->ndpb; i++)
                if (e->dpb[i].fn < e->dpb[m].fn)
                    m = i;
            e->dpb[m] = e->dpb[--e->ndpb];
        }
        e->dpb[e->ndpb].slot = e->role_cur;
        e->dpb[e->ndpb].poc = e->cur_poc;
        e->dpb[e->ndpb].fn = e->frame_num;
        e->dpb[e->ndpb].anchor = !e->cur_is_b;
        e->dpb[e->ndpb].fm = 3;
        e->ndpb++;
        /* frame_num counts reference pictures (7.4.3): the pictures after
         * this one carry PrevRefFrameNum + 1 */
        e->frame_num++;
    }
    e->role_cur = -1;
    e->role_cur = pic_free(e);
    bind_pictures(e);
}

/* Which reference picture each list points at for the picture about to be
 * coded, chosen the way the decoder's default lists (8.2.4.2) order them so
 * that index 0 is what the encoder predicted from:
 *
 *  - B: list0[0] is the nearest reference before it in display order and
 *    list1[0] the nearest after it.  With a B pyramid the reference B of a
 *    group is one of them for the other B pictures of the group.
 *  - P: the newest anchor (the default list is in decreasing frame_num,
 *    which puts a reference B picture of the group just coded first -- that
 *    case gets a ref_pic_list_modification moving the anchor to the front).
 *    With two active list0 entries the older anchor (or, with a pyramid,
 *    the reference B just coded) is list0[1] and lives in refpl[1].  With
 *    three or more (--ref 3..8) see set_refs_multi(): no marking operations,
 *    a sliding window of max_refs pictures, list0 = the newest anchor, then
 *    the rest by decreasing POC.
 *
 * With a pyramid, an anchor's header also removes every reference except the
 * previous anchor once a reference B picture is among them: the group it
 * belonged to is finished, and the sliding window would otherwise drop the
 * previous anchor first (it has the smaller frame_num) while the next
 * group's B pictures still need it. */
static int p_should_use_2refs(const struct aeg_encoder *e)
{
    /* `refs` is resolved by aeg_encoder_open() (2 from preset 3 up on P-only
     * streams, see there); the first P picture after an IDR only has one
     * reference to offer. */
    return e->cur_is_p && e->ndpb > 1 && e->p.refs >= 2;
}
/* ref_idx_l0 of a P macroblock as CAVLC writes it: te(v), which is one
 * inverted bit when list0 has two entries and ue(v) with more (7.4.2.10). */
/* Entries of list0 as the ref_idx syntax of the current macroblock counts
 * them: a field macroblock of an MBAFF frame has two fields per frame
 * (7.4.5.1), even when the slice has one frame (the index is then sent
 * anyway: num_ref_idx_active > 1 or mb_field_decoding_flag !=
 * field_pic_flag, 7.3.5.1).  The decision pass works with the field list in
 * place (mbf_enter), whose frames the swap remembers. */
static int l0_syn(const struct aeg_encoder *e)
{
    if (!e->cfld || !e->mbaff)
        return e->cur_l0_active;
    return 2 * (e->mbf_save.refs ? e->mbf_save.l0 : e->cur_l0_active);
}

static int ref_te_bits(const struct aeg_encoder *e, int ref)
{
    int n = l0_syn(e);
    return n <= 1 ? 0 : n == 2 ? 1 : aeg_ue_bits((uint32_t)ref);
}

static void put_ref_te(const struct aeg_encoder *e, aeg_bitwriter *bw, int ref)
{
    if (l0_syn(e) == 2)
        aeg_bw_put(bw, 1, ref ? 0u : 1u);
    else
        aeg_bw_ue(bw, (uint32_t)ref);
}

/* Reference plane indices in use by the current picture: list0[0..n-1] of a
 * P picture, list0[0] and list1[0] of a B picture. */
static int nrefpl(const struct aeg_encoder *e)
{
    return e->cur_is_b ? (e->bfp_on ? 4 : 2) : e->cur_l0_active;
}

/* Does refpl[k] hold a picture? */
static int ref_valid(const struct aeg_encoder *e, int k)
{
    return k == 0 ? e->has_ref : k == 1 ? e->has_ref2
                  : e->cur_is_b ? e->bfp_on && k < 4
                  : e->cur_is_p && k < e->cur_l0_active;
}

static void refctl_drop(struct aeg_encoder *e, int i)
{
    int k = e->refctl.n_mmco++;
    e->refctl.mmco_diff[k] = e->frame_num - e->dpb[i].fn - 1;
    e->refctl.mmco_slot[k] = e->dpb[i].slot;
}

static int ref2_anchor_only(void)
{
    static _Atomic int v = -1;
    if (v < 0) {
        const char *ev = getenv("AEG_REF2ANCHOR");
        v = ev ? atoi(ev) != 0 : 0;
    }
    return v;
}

/* Three or more P references (--ref 3..AEG_MAX_REFS): list0 is the newest
 * anchor and then every other reference picture, nearest in display order
 * first, as many as --ref allows.  The buffer is a plain sliding window of
 * max_refs pictures (no marking operations): with a pyramid it holds the
 * last two anchors and the reference B between them, which is all a B
 * picture reads, whatever else it keeps for the P pictures.  The default
 * list (decreasing frame_num) gets a ref_pic_list_modification whenever its
 * first entries are not these. */
static void set_refs_multi(struct aeg_encoder *e)
{
    int want[AEG_MAX_REFS], used[AEG_MAX_DPB] = {0};
    int dflt[AEG_MAX_DPB], nd = 0, n = 0, i, j, k, ok = 1;

    for (i = 0; i < e->ndpb; i++)
        if (e->dpb[i].anchor && (n == 0 || e->dpb[i].poc > e->dpb[want[0]].poc))
            want[0] = i, n = 1;
    if (n == 0)
        return;
    used[want[0]] = 1;
    while (n < e->p.refs) {
        int b = -1;
        for (i = 0; i < e->ndpb; i++)
            if (!used[i] && (b < 0 || e->dpb[i].poc > e->dpb[b].poc))
                b = i;
        if (b < 0)
            break;
        used[b] = 1;
        want[n++] = b;
    }
    for (i = 0; i < e->ndpb; i++) {
        for (j = nd; j > 0 && e->dpb[dflt[j - 1]].fn < e->dpb[i].fn; j--)
            dflt[j] = dflt[j - 1];
        dflt[j] = i;
        nd++;
    }
    for (k = 0; k < n; k++)
        ok &= dflt[k] == want[k];
    if (!ok) {
        int pred = e->frame_num;       /* CurrPicNum */
        for (k = 0; k < n; k++) {
            int pn = e->dpb[want[k]].fn;
            if (pn < pred) {
                e->refctl.mod_idc[k] = 0;
                e->refctl.mod_abs[k] = pred - pn - 1;
            } else {
                e->refctl.mod_idc[k] = 1;
                e->refctl.mod_abs[k] = pn - pred - 1;
            }
            pred = pn;
        }
        e->refctl.n_mod = n;
        e->refctl.n_mod_1ref = dflt[0] != want[0];
    }
    for (k = 0; k < n; k++) {
        int sl = e->dpb[want[k]].slot;
        for (i = 0; i < 3; i++)
            e->refpl[k][i] = e->pic_plane[sl][i];
        e->ref_poc[k] = e->dpb[want[k]].poc;
        e->ref_slot[k] = sl;
    }
    e->has_ref = 1;
    e->has_ref2 = n > 1;
    e->cur_l0_active = n;
}

static void set_current_refs(struct aeg_encoder *e)
{
    int l0 = -1, l1 = -1, l0b = -1, i, bref = 0;

    memset(&e->refctl, 0, sizeof(e->refctl));
    for (i = 0; i < AEG_MAX_REFS; i++) {
        e->refpl[i][0] = e->refpl[i][1] = e->refpl[i][2] = NULL;
        e->ref_slot[i] = -1;
        e->ref_poc[i] = 0;
    }
    e->cur_l1_slot = -1;
    if (e->p.refs >= 3 && e->cur_is_p) {
        e->has_ref = e->has_ref2 = 0;
        e->cur_l0_active = 1;
        set_refs_multi(e);
        return;
    }
    for (i = 0; i < e->ndpb; i++)
        bref |= !e->dpb[i].anchor;
    if (e->cur_is_b) {
        for (i = 0; i < e->ndpb; i++) {
            int p = e->dpb[i].poc;
            if (p < e->cur_poc && (l0 < 0 || p > e->dpb[l0].poc)) l0 = i;
            if (p > e->cur_poc && (l1 < 0 || p < e->dpb[l1].poc)) l1 = i;
        }
        /* A reference B picture of a pyramid with two P references: the
         * buffer holds the group's two anchors and the second reference of
         * the later one; that one goes, or the sliding window would drop the
         * earlier anchor (smallest frame_num) that the group still needs. */
        if (e->cur_bref && e->p.b_pyramid && e->p.refs == 2)
            for (i = 0; i < e->ndpb; i++)
                if (i != l0 && i != l1)
                    refctl_drop(e, i);
    } else {
        int two;
        for (i = 0; i < e->ndpb; i++)
            if (e->dpb[i].anchor && (l0 < 0 || e->dpb[i].poc > e->dpb[l0].poc))
                l0 = i;
        /* The second P reference: the older anchor without a pyramid; with
         * one, the newest other picture -- the reference B of the group just
         * coded when there is one (closer than the older anchor). */
        for (i = 0; i < e->ndpb; i++)
            if (i != l0 && (e->dpb[i].anchor || (e->p.b_pyramid &&
                                                 !ref2_anchor_only())) &&
                (l0b < 0 || e->dpb[i].poc > e->dpb[l0b].poc))
                l0b = i;
        two = p_should_use_2refs(e) && l0 >= 0 && l0b >= 0;
        if (l0 >= 0 && e->cur_is_p) {
            /* the default list 0 orders by decreasing frame_num; modify it
             * whenever its first entries are not the wanted ones */
            int want[2], n = two ? 2 : 1, dflt[AEG_MAX_DPB], nd = 0, j, k, ok = 1;
            want[0] = l0;
            want[1] = l0b;
            for (i = 0; i < e->ndpb; i++) {
                for (j = nd; j > 0 && e->dpb[dflt[j - 1]].fn < e->dpb[i].fn; j--)
                    dflt[j] = dflt[j - 1];
                dflt[j] = i;
                nd++;
            }
            for (k = 0; k < n; k++)
                ok &= k < nd && dflt[k] == want[k];
            if (!ok) {
                int pred = e->frame_num;       /* CurrPicNum */
                for (k = 0; k < n; k++) {
                    int pn = e->dpb[want[k]].fn;
                    if (pn < pred) {
                        e->refctl.mod_idc[k] = 0;
                        e->refctl.mod_abs[k] = pred - pn - 1;
                    } else {
                        e->refctl.mod_idc[k] = 1;
                        e->refctl.mod_abs[k] = pn - pred - 1;
                    }
                    pred = pn;
                }
                e->refctl.n_mod = n;
                e->refctl.n_mod_1ref = nd < 1 || dflt[0] != l0;
            }
        }
        if (l0 >= 0 && (bref || (two && e->p.b_pyramid)))
            for (i = 0; i < e->ndpb; i++)
                if (i != l0 && !(two && i == l0b))
                    refctl_drop(e, i);
    }
    e->cur_l1_slot = l1 >= 0 ? e->dpb[l1].slot : -1;
    if (p_should_use_2refs(e) && l0 >= 0 && l0b >= 0) {
        int a = e->dpb[l0].slot, b = e->dpb[l0b].slot;
        for (i = 0; i < 3; i++) {
            e->refpl[0][i] = e->pic_plane[a][i];
            e->refpl[1][i] = e->pic_plane[b][i];
        }
        e->ref_poc[0] = e->pic_poc[a];
        e->ref_poc[1] = e->pic_poc[b];
        e->ref_slot[0] = a;
        e->ref_slot[1] = b;
        e->has_ref = 1;
        e->has_ref2 = 1;               /* second list0 entry is refpl[1] */
        e->cur_l0_active = 2;
        return;
    }
    for (i = 0; i < 3; i++) {
        e->refpl[0][i] = l0 >= 0 ? e->pic_plane[e->dpb[l0].slot][i] : NULL;
        e->refpl[1][i] = l1 >= 0 ? e->pic_plane[e->dpb[l1].slot][i] : NULL;
    }
    e->ref_poc[0] = l0 >= 0 ? e->dpb[l0].poc : 0;
    e->ref_poc[1] = l1 >= 0 ? e->dpb[l1].poc : 0;
    e->ref_slot[0] = l0 >= 0 ? e->dpb[l0].slot : -1;
    e->ref_slot[1] = l1 >= 0 ? e->dpb[l1].slot : -1;
    e->has_ref = l0 >= 0;
    e->has_ref2 = l1 >= 0;
    e->cur_l0_active = 1;
}

static void copy_plane_pad(pixel *dst, int dstride, const pixel *src,
                           int sstride, int w, int h)
{
    int x, y;
    for (y = 0; y < h; y++)
        memcpy(dst + y * dstride, src + y * sstride, (size_t)w * sizeof(pixel));
    for (y = 0; y < h; y++) {
        aeg_pixel_fill(dst + y * dstride - AEG_PAD, dst[y * dstride], AEG_PAD);
        aeg_pixel_fill(dst + y * dstride + w, dst[y * dstride + w - 1], AEG_PAD);
    }
    /* A padded row is exactly `w + 2 * AEG_PAD` bytes wide and starts AEG_PAD
     * bytes left of the content origin, so the copies must be taken from the
     * row origin -- otherwise every row runs AEG_PAD bytes past the plane. */
    for (y = 1; y <= AEG_PAD; y++) {
        memcpy(dst - y * dstride - AEG_PAD, dst - AEG_PAD,
               (size_t)(w + 2 * AEG_PAD) * sizeof(pixel));
        memcpy(dst + (h - 1 + y) * dstride - AEG_PAD,
               dst + (h - 1) * dstride - AEG_PAD, (size_t)(w + 2 * AEG_PAD) * sizeof(pixel));
    }
    (void)x;
}

/* Replicate the reconstructed picture's edges into the padding.  The motion
 * compensated prediction of a macroblock on the picture border reads samples
 * outside the visible area, and the decoder pads with the edge sample, so the
 * reference buffer has to hold exactly that. */
static void pad_edges(pixel *p, int stride, int w, int h)
{
    int y;
    for (y = 0; y < h; y++) {
        aeg_pixel_fill(p + y * stride - AEG_PAD, p[y * stride], AEG_PAD);
        aeg_pixel_fill(p + y * stride + w, p[y * stride + w - 1], AEG_PAD);
    }
    for (y = 1; y <= AEG_PAD; y++) {
        memcpy(p - y * stride - AEG_PAD, p - AEG_PAD,
               (size_t)(w + 2 * AEG_PAD) * sizeof(pixel));
        memcpy(p + (h - 1 + y) * stride - AEG_PAD, p + (h - 1) * stride - AEG_PAD,
               (size_t)(w + 2 * AEG_PAD) * sizeof(pixel));
    }
}

/* ------------------------------------------------------------------ */
/* Neighbouring macroblocks                                            */
/* ------------------------------------------------------------------ */
/* Which macroblock holds a neighbouring sample (6.4.12).  Without MBAFF it is
 * the raster neighbour.  In an MBAFF frame it is Table 6-4: it depends on
 * whether the current macroblock (e->cfld) and the neighbouring pair are
 * frame or field macroblocks, and the row inside the neighbour changes with
 * it.  The left pair of (mbx, mby) is always decoded; the pair above exists
 * from the second pair row of the slice on, and the bottom macroblock of a
 * frame pair has its own top macroblock above it. */
static inline int nb_top_ok(const struct aeg_encoder *e, int mby)
{
    if (!e->mbaff)
        return mby > e->top_row;
    return (!e->cfld && (mby & 1)) || (mby & ~1) > e->top_row;
}

/* The macroblock holding sample row -1 (neighbour B; valid with nb_top_ok):
 * its last block row is the one above. */
static inline const aeg_mb *nb_mbB(const struct aeg_encoder *e, int mbx,
                                   int mby)
{
    const int w = e->mb_w;
    int t;

    if (!e->mbaff || (!e->cfld && (mby & 1)))
        return &e->mbs[(mby - 1) * w + mbx];
    t = mby & ~1;                          /* top macroblock of this pair */
    /* a field top macroblock sees the top field macroblock of a field pair
     * above, every other case the bottom macroblock of the pair above */
    if (e->cfld && !(mby & 1) && e->mbs[(t - 2) * w + mbx].fld)
        return &e->mbs[(t - 2) * w + mbx];
    return &e->mbs[(t - 1) * w + mbx];
}

/* The macroblock holding the sample left of row yN (0..maxH-1; maxH 16 for
 * luma, 8 for chroma) and that sample's row in it (*yM).  mbx > 0. */
static inline const aeg_mb *nb_left(const struct aeg_encoder *e, int mbx,
                                    int mby, int yN, int maxH, int *yM)
{
    const aeg_mb *l0, *l1;

    if (!e->mbaff) {
        *yM = yN;
        return &e->mbs[mby * e->mb_w + mbx - 1];
    }
    l0 = &e->mbs[(mby & ~1) * e->mb_w + mbx - 1];   /* the left pair */
    l1 = l0 + e->mb_w;
    if (!e->cfld) {
        if (!l0->fld) {                 /* frame beside frame */
            *yM = yN;
            return (mby & 1) ? l1 : l0;
        }
        /* frame beside field: the frame row's parity picks the field */
        *yM = ((mby & 1) ? yN + maxH : yN) >> 1;
        return (yN & 1) ? l1 : l0;
    }
    if (l0->fld) {                      /* field beside field */
        *yM = yN;
        return (mby & 1) ? l1 : l0;
    }
    /* field beside frame: field row yN is frame row 2 yN (+1) of the pair */
    *yM = 2 * yN + (mby & 1);
    if (*yM >= maxH) {
        *yM -= maxH;
        return l1;
    }
    return l0;
}

/* The macroblock left of the macroblock as a whole (row 0: the neighbour A
 * of the macroblock-level contexts). */
static inline const aeg_mb *nb_mbA(const struct aeg_encoder *e, int mbx,
                                   int mby)
{
    int yM;
    return nb_left(e, mbx, mby, 0, 16, &yM);
}

/* MBAFF: a neighbour's vertical vector difference as the context of the
 * current macroblock's sees it (9.3.3.1.1.7): halved from a frame neighbour
 * of a field macroblock, doubled from a field neighbour of a frame one. */
static inline int mvd_kind(const struct aeg_encoder *e, const aeg_mb *n,
                           int v)
{
    if (!e->mbaff || n->fld == e->cfld)
        return v;
    v = v < 0 ? -v : v;
    return e->cfld ? v >> 1 : v << 1;
}

/* ... and its reference index, for the zero test of ref_idx's context (a
 * field neighbour of a frame macroblock counts as reference 0 up to 1). */
static inline int ref_kind(const struct aeg_encoder *e, const aeg_mb *n,
                           int r)
{
    if (!e->mbaff || n->fld == e->cfld)
        return r;
    return e->cfld ? r * 2 : r >> 1;
}

/* A field macroblock of an MBAFF frame: the planes as seen through its field
 * (its sixteen lines, every other frame line, at the pair's position), so
 * that everything addressing the picture by (mbx, mby) -- base + 16 mby rows
 * of `stride` -- lands on them; and the field scans and contexts. */
/* Reference entry k of field macroblock row mby: field reference f of the
 * macroblock's parity (fref_*), addressed like the planes mbf_enter sets
 * up: base' + (16 mby + y) 2s is field row 16 py + y, frame line
 * 32 py + 2y + parity.  (mbf_save.stride holds the frame's strides.) */
static void mbf_set_ref(struct aeg_encoder *e, int mby, int k, int f)
{
    const int py = mby >> 1, bot = mby & 1;
    int i;

    for (i = 0; i < 3; i++)
        e->refpl[k][i] = e->fref_pl[bot][f][i] -
                         (i ? (ptrdiff_t)(16 * py + 16 * bot) * e->mbf_save.stride[i]
                            : (ptrdiff_t)(32 * py + 32 * bot) * e->mbf_save.stride[i]);
    e->mcr[k] = e->fref_mc[bot][f];
    for (i = 0; i < 4; i++)
        if (e->mcr[k].base[i])
            e->mcr[k].base[i] -= (ptrdiff_t)(32 * py + 32 * bot) * e->mbf_save.stride[0];
    e->ref_poc[k] = e->fref_poc[bot][f];
    e->fld_cofs[k] = e->fref_cofs[bot][f];
}

static void mbf_enter(struct aeg_encoder *e, int mby)
{
    const int py = mby >> 1, bot = mby & 1;
    int i;

    for (i = 0; i < 3; i++) {
        int s = e->stride[i];
        /* field line 0 of this macroblock is frame line 32 py + bot (luma)
         * of the pair; base' + (16 mby) 2s must be that line */
        ptrdiff_t back = i ? (ptrdiff_t)(16 * py + 15 * bot) * s
                           : (ptrdiff_t)(32 * py + 31 * bot) * s;
        e->mbf_save.srcp[i] = e->srcp[i];
        e->mbf_save.recp[i] = e->recp[i];
        e->mbf_save.stride[i] = s;
        e->srcp[i] -= back;
        e->recp[i] -= back;
        e->stride[i] = 2 * s;
    }
    /* a P picture: the field references in place of the frames */
    e->mbf_save.refs = e->fref_n > 0;
    if (e->mbf_save.refs) {
        const int n = e->fref_n;
        int k;
        e->mbf_save.l0 = e->cur_l0_active;
        e->mbf_save.has_ref2 = e->has_ref2;
        for (k = 0; k < n; k++) {
            memcpy(e->mbf_save.refpl[k], e->refpl[k], sizeof(e->refpl[k]));
            e->mbf_save.mcr[k] = e->mcr[k];
            e->mbf_save.ref_poc[k] = e->ref_poc[k];
            e->mbf_save.cofs[k] = e->fld_cofs[k];
            mbf_set_ref(e, mby, k, k);
        }
        if (!e->fref_b) {
            e->cur_l0_active = n;
            e->has_ref2 = n > 1;
        }
    }
    e->cfld = 1;
    e->scan4 = aeg_field_scan4x4;
    e->scan8 = aeg_field_scan8x8;
    e->fcat = 6;
    e->fscan = AEG_CAVLC_FIELD;
    e->fscan8 = AEG_CAVLC_FIELD8;
}

static void mbf_leave(struct aeg_encoder *e)
{
    int i;

    for (i = 0; i < 3; i++) {
        e->srcp[i] = e->mbf_save.srcp[i];
        e->recp[i] = e->mbf_save.recp[i];
        e->stride[i] = e->mbf_save.stride[i];
    }
    if (e->mbf_save.refs) {
        const int n = e->fref_n;
        int k;
        for (k = 0; k < n; k++) {
            memcpy(e->refpl[k], e->mbf_save.refpl[k], sizeof(e->refpl[k]));
            e->mcr[k] = e->mbf_save.mcr[k];
            e->ref_poc[k] = e->mbf_save.ref_poc[k];
            e->fld_cofs[k] = e->mbf_save.cofs[k];
        }
        e->cur_l0_active = e->mbf_save.l0;
        e->has_ref2 = e->mbf_save.has_ref2;
        e->mbf_save.refs = 0;
    }
    e->b_ref[0] = e->b_ref[1] = e->bfld_on = e->bdr_no = 0;
    e->cfld = 0;
    e->scan4 = aeg_scan4x4;
    e->scan8 = aeg_zigzag8x8;
    e->fcat = e->fscan = e->fscan8 = 0;
}

/* Field macroblock state for the bitstream passes (no planes needed). */
static void mbf_syntax(struct aeg_encoder *e, int fld)
{
    e->cfld = fld;
    e->scan4 = fld ? aeg_field_scan4x4 : aeg_scan4x4;
    e->scan8 = fld ? aeg_field_scan8x8 : aeg_zigzag8x8;
    e->fcat = fld ? 6 : 0;
    e->fscan = fld ? AEG_CAVLC_FIELD : 0;
    e->fscan8 = fld ? AEG_CAVLC_FIELD8 : 0;
}

/* ------------------------------------------------------------------ */
/* Neighbour nC                                                        */
/* ------------------------------------------------------------------ */
/* non-zero count of the block left of luma block row `by` / chroma block
 * row `by` of plane `pl` */
static int left_nz_luma(struct aeg_encoder *e, int mbx, int mby, int by)
{
    int yM;
    const aeg_mb *m;
    if (!mbx)
        return 0;
    m = nb_left(e, mbx, mby, 4 * by, 16, &yM);
    return m->nz[(yM >> 2) * 4 + 3];
}

static int left_nz_chroma(struct aeg_encoder *e, int mbx, int mby, int pl,
                          int by)
{
    int yM;
    const aeg_mb *m;
    if (!mbx)
        return 0;
    m = nb_left(e, mbx, mby, 4 * by, 8, &yM);
    return m->nz[AEG_BLK_CHROMA_CB + pl * 4 + (yM >> 2) * 2 + 1];
}

static int top_nz(struct aeg_encoder *e, int mbx, int mby, int blk)
{
    return nb_top_ok(e, mby) ? nb_mbB(e, mbx, mby)->nz[blk] : 0;
}

static int luma_nc(struct aeg_encoder *e, int mbx, int mby, const mbenc *c, int blk)
{
    int bx = blk & 3, by = blk >> 2;
    int a = bx ? c->nz[blk - 1] : left_nz_luma(e, mbx, mby, by);
    int b = by ? c->nz[blk - 4] : top_nz(e, mbx, mby, blk + 12);
    return aeg_cavlc_nc(a, b, mbx > 0 || bx > 0,
                       nb_top_ok(e, mby) || by > 0);
}

static int chroma_nc(struct aeg_encoder *e, int mbx, int mby, const mbenc *c,
                     int pl, int blk)
{
    int base = AEG_BLK_CHROMA_CB + pl * 4;
    int bx = blk & 1, by = blk >> 1;
    int a = bx ? c->nz[base + blk - 1] : left_nz_chroma(e, mbx, mby, pl, by);
    int b = by ? c->nz[base + blk - 2] : top_nz(e, mbx, mby, base + blk + 2);
    return aeg_cavlc_nc(a, b, mbx > 0 || bx > 0,
                       nb_top_ok(e, mby) || by > 0);
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
        a = left_nz_luma(e, mbx, mby, by);
    if (by)
        b = c->nz[blk - 4];
    else
        b = top_nz(e, mbx, mby, 12 + bx);
    return aeg_cavlc_nc(a, b, mbx > 0 || bx > 0,
                       nb_top_ok(e, mby) || by > 0);
}

static void trellis_i16_ac(struct aeg_encoder *e, int mbx, int mby, mbenc *c,
                           int blk, const pixel *src, int stride,
                           const pixel *pred, int qp, const int32_t *dcv);

static void luma_inter_gather(const struct aeg_encoder *e, const pixel *src,
                              const pixel *pred, int16_t *res);
static void trellis_luma_dc(struct aeg_encoder *e, int mbx, int mby,
                            const mbenc *c, const pixel *src, int stride,
                            const pixel *pred, int qp, const dctcoef *had,
                            dctcoef *dcled);

/* Intra_16x16 and chroma in the coefficient batch: the blocks' forward
 * halves in one tq_dc_batch() call, the DC Hadamard (sixteen or four values)
 * on the host, the inverses in one iq_dc_batch() -- the same arithmetic as the
 * per-block path, so the same stream.  Only where the batch is in use (the
 * dispatcher placed the coefficient stages on the device, or AEG_COEF_BATCH /
 * AEG_RD_BATCH) and no level selection runs between the halves (`sel`):
 * that one is sequential per block.  AEG_DCBATCH=0 keeps the per-block path. */
static int dc_batch_on(const struct aeg_encoder *e, int sel)
{
    static _Atomic int env = -2;
    if (env == -2) {
        const char *ev = getenv("AEG_DCBATCH");
        env = ev ? atoi(ev) != 0 : 1;
    }
    return env && e->coef_batch && e->d.ops.tq_dc_batch &&
           e->d.ops.iq_dc_batch && !sel;
}

/* a 4x4 block's prediction plus residual, clipped: aeg_idct_add_block()'s
 * last step, for a residual that came out of a batch */
static void add_res4(pixel *dst, int stride, const pixel *pred,
                     int pstride, const int16_t *res)
{
    int x, y;
    for (y = 0; y < 4; y++)
        for (x = 0; x < 4; x++)
            dst[y * stride + x] = aeg_clip_pixel(pred[y * pstride + x] +
                                                   res[y * 4 + x]);
}

static void code_luma_i16(struct aeg_encoder *e, int mbx, int mby, mbenc *c)
{
    const pixel *src = e->srcp[0] + mby * 16 * e->stride[0] + mbx * 16;
    pixel *dst = e->recp[0] + mby * 16 * e->stride[0] + mbx * 16;
    const pixel *pred = e->pred_l;
    dctcoef dc[16], had[16], dcled[16];
    int32_t dchad[16];
    int blk, i, x, y, nz_dc = 0;
    int qp = c->qp;
    /* batched (see dc_batch_on): the sixteen blocks' forward halves in one
     * call, the DC Hadamard on the host, the inverses in another */
    const int batch = dc_batch_on(e, e->p.trellis > 0 && e->p.cabac);
    int16_t bres[16 * 16], brec[16 * 16];
    int32_t bdc[16];

    if (batch) {
        int64_t t0 = aeg_now_ns();
        int32_t bnz[16];
        luma_inter_gather(e, src, pred, bres);
        e->d.ops.tq_dc_batch(e->d.ops.priv, bres, 16, qp, 1, &c->level[0][0],
                             bnz, bdc);
        stage_rec(e, AEG_STAGE_TRANSFORM, e->d.ops.backend != AEG_BACKEND_CPU,
                  aeg_now_ns() - t0);
        for (blk = 0; blk < 16; blk++)
            dc[blk] = (dctcoef)bdc[blk];
    }
    for (blk = 0; blk < 16 && !batch; blk++) {
        int bx = (blk & 3) * 4, by = (blk >> 2) * 4;
        int16_t res[16];
        dctcoef coef[16];
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
    if (e->p.trellis > 0 && e->p.cabac)
        trellis_luma_dc(e, mbx, mby, c, src, e->stride[0], pred, qp, had, dcled);
    for (i = 0; i < 16; i++)
        nz_dc += dcled[i] != 0;
    memcpy(c->level[AEG_BLK_LUMA_DC], dcled, sizeof(dcled));
    c->nz[AEG_BLK_LUMA_DC] = (int8_t)nz_dc;
    aeg_dequant_4x4_dc_out(dcled, dchad, qp);
    aeg_dc_block_unpermute(dchad);   /* decoder slot order -> raster order */

    if (batch) {
        int64_t t0 = aeg_now_ns();
        e->d.ops.iq_dc_batch(e->d.ops.priv, &c->level[0][0], dchad, 16, qp,
                             brec);
        stage_rec(e, AEG_STAGE_IQUANT_IDCT, e->d.ops.backend != AEG_BACKEND_CPU,
                  aeg_now_ns() - t0);
        for (blk = 0; blk < 16; blk++) {
            int bx = (blk & 3) * 4, by = (blk >> 2) * 4, dummy;
            c->nz[blk] = (int8_t)aeg_nz_last(c->level[blk], 16, &dummy);
            add_res4(dst + by * e->stride[0] + bx, e->stride[0],
                     pred + by * 16 + bx, 16, brec + 16 * blk);
        }
    }
    for (blk = 0; blk < 16 && !batch; blk++) {
        int bx = (blk & 3) * 4, by = (blk >> 2) * 4;
        pixel *r = dst + by * e->stride[0] + bx;
        const pixel *pr = pred + by * 16 + bx;
        int32_t dcv = dchad[blk];
        int nz = 0;
        if (e->p.trellis > 0 && e->p.cabac)
            trellis_i16_ac(e, mbx, mby, c, blk, src + by * e->stride[0] + bx,
                           e->stride[0], pr, qp, &dcv);
        {
            int dummy;
            nz = aeg_nz_last(c->level[blk], 16, &dummy);
        }
        c->nz[blk] = (int8_t)nz;
        aeg_idct_add_block(r, e->stride[0], c->level[blk], qp, pr, 16, &dcv);
    }
    c->sse += aeg_ssd(16, 16, dst, e->stride[0], src, e->stride[0]);
    /* CAVLC Intra_16x16 always codes all sixteen AC blocks; the luma part of
     * the coded block pattern comes from the macroblock type. */
    c->cbp = (c->cbp & 0x30) | 0x0f;

    /* The intra 16x16 DC block is context-coded with the nC of luma block
     * (0, 0) -- the reference decoder computes pred_non_zero_count(0) for it
     * (n >= LUMA_DC_BLOCK_INDEX -> (n - 48) * 16). */
    /* (The CAVLC estimate of the bits: a CABAC encode prices the macroblock
     * with the coder itself, rd_price(), and never reads it.) */
    if (e->p.cabac)
        return;
    if (nz_dc)
        c->bits += aeg_cavlc_block_cost(c->level[AEG_BLK_LUMA_DC],
                                        luma_ac_nc(e, mbx, mby, c, 0), 1 | e->fscan);
    for (blk = 0; blk < 16; blk++)
        if (c->nz[ac_ord[blk]])
            c->bits += aeg_cavlc_block_cost(c->level[ac_ord[blk]],
                                            luma_ac_nc(e, mbx, mby, c, blk), 3 | e->fscan);
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
static void luma_inter_gather(const struct aeg_encoder *e, const pixel *src,
                              const pixel *pred, int16_t *res)
{
    /* row by row, sixteen differences at a time (this vectorises; the
     * per-block loop did not), each row then dealt out to its four blocks */
    const int stride = e->stride[0];
    int x, y;
    for (y = 0; y < 16; y++) {
        int16_t row[16];
        int16_t *r = res + 64 * (y >> 2) + 4 * (y & 3);
        for (x = 0; x < 16; x++)
            row[x] = (int16_t)((int)src[y * stride + x] - (int)pred[y * 16 + x]);
        memcpy(r, row, 8);
        memcpy(r + 16, row + 4, 8);
        memcpy(r + 32, row + 8, 8);
        memcpy(r + 48, row + 12, 8);
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
                              const dctcoef *lev, const int32_t *nz,
                              const int16_t *rec, const pixel *pred)
{
    const pixel *src = e->srcp[0] + mby * 16 * e->stride[0] + mbx * 16;
    pixel *dst = e->recp[0] + mby * 16 * e->stride[0] + mbx * 16;
    int blk, luma_cbp = 0, qp = c->qp;

    if (lev != &c->level[0][0])
        memcpy(c->level, lev, 16 * 16 * sizeof(dctcoef));
    for (blk = 0; blk < 16; blk++) {
        int n;
        if (nz) {
            n = nz[blk];
        } else {
            int dummy;
            n = aeg_nz_last(c->level[blk], 16, &dummy);
        }
        c->nz[blk] = (int8_t)n;
        if (n)
            luma_cbp |= 1 << blk_group(blk);
    }
    if (rec) {
        /* the batch's residual: add, clip and measure in one pass */
        c->sse += (int64_t)aeg_add_res16_ssd(dst, e->stride[0], pred, rec,
                                             src, e->stride[0]);
    } else {
        for (blk = 0; blk < 16; blk++) {
            int bx = (blk & 3) * 4, by = (blk >> 2) * 4, y;
            if (!c->nz[blk]) {
                /* no levels: the reconstruction is the prediction */
                for (y = 0; y < 4; y++)
                    memcpy(dst + (by + y) * e->stride[0] + bx,
                           pred + (by + y) * 16 + bx, 4 * sizeof(pixel));
                continue;
            }
            aeg_idct_add_block(dst + by * e->stride[0] + bx, e->stride[0],
                               c->level[blk], qp, pred + by * 16 + bx, 16,
                               NULL);
        }
        /* the sixteen blocks' SSE, summed as one */
        c->sse += aeg_ssd(16, 16, dst, e->stride[0], src, e->stride[0]);
    }
    c->cbp = (c->cbp & 0x30) | luma_cbp;
    if (!e->p.cabac) {                  /* CAVLC estimate, see above */
        int g, k;
        for (g = 0; g < 4; g++) {
            if (!(c->cbp & (1 << g)))
                continue;
            for (k = 0; k < 4; k++) {
                int b = group_blk[g][k];
                c->bits += aeg_cavlc_block_cost(c->level[b],
                                                luma_nc(e, mbx, mby, c, b), 0 | e->fscan);
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
static int64_t cabac_block_price(const aeg_cabac *st, const dctcoef *scan,
                                 int n, int cat, int cbf_ctx)
{
    return aeg_cabac_residual_price(st, scan, n, cat, cbf_ctx);
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

static int64_t trel_sse(const struct trel *t, const pixel *src, int stride,
                        const pixel *pred, int pred_stride,
                        const dctcoef *lev, int qp)
{
    if (t->side == 4) {
        return aeg_idct_add_ssd4(lev, qp, pred, pred_stride, t->dc, src, stride);
    } else {
        pixel r[64];
        aeg_idct_8x8_add(r, 8, lev, qp, pred, pred_stride);
        return aeg_ssd(8, 8, r, 8, src, stride);
    }
}

/* What one coefficient's change does to the price of the block, with the levels
 * the search currently holds in `cur`.  Only the coefficient itself and -- when
 * it is the last significant one and the change empties it -- the new last
 * coefficient cost anything, which is what makes the sweep cheap. */
static int64_t trel_dbits(const aeg_cabac *sh, const struct trel *t,
                          const dctcoef *cur, int idx, int newlev, int last)
{
    int pos = t->scan[idx];
    int oldlev = cur[pos];
    int sig_last = idx == last;
    int64_t d = aeg_cabac_coef_price_inl(sh, t->cat, t->n, idx, newlev,
                                     sig_last && newlev != 0, 1) -
                aeg_cabac_coef_price_inl(sh, t->cat, t->n, idx, oldlev,
                                     sig_last, 1);

    if (sig_last && newlev == 0) {
        /* The block's last coefficient moves back: the next one has to say so. */
        int j = idx - 1;
        while (j >= 0 && !cur[t->scan[j]])
            j--;
        if (j >= 0)
            d += aeg_cabac_coef_price_inl(sh, t->cat, t->n, j, cur[t->scan[j]], 1,
                                      0) -
                 aeg_cabac_coef_price_inl(sh, t->cat, t->n, j, cur[t->scan[j]], 0,
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

/* x264's CABAC trellis (quant_trellis_cabac): a dynamic program over the
 * coefficients from the last possible one backwards, the order the coder sends
 * the levels in, whose states are the nodes of the level state machine
 * (9.3.2.2: how many levels of 1 and above 1 are coded already, which is what
 * picks the level contexts).  Node 0 is "nothing coded yet": the next
 * non-zero level is the last one and says so.  Each coefficient may take
 * round(x), round(x) - 1 or 0, x being its unrounded level, so a coefficient
 * the deadzone would remove can survive and one the rounding keeps can go.
 * The bits come from the block's starting context states (the contexts are
 * not advanced inside the block, as in x264), the distortion from the
 * coefficient-domain model of aeg_trel_levels().  The result is a starting
 * point: trel_descent_kn() prices it exactly against the quantiser's levels
 * and polishes the better one. */
static const uint8_t tv_lvl1[8] = { 1, 2, 3, 4, 0, 0, 0, 0 };
static const uint8_t tv_gt1[8]  = { 5, 5, 5, 5, 6, 7, 8, 9 };
static const uint8_t tv_next[2][8] = {
    { 1, 2, 3, 3, 4, 5, 6, 7 },
    { 4, 4, 4, 4, 5, 6, 7, 7 }
};

/* Bits (1/256) of a level of `abs` >= 1 coded from node `node`, sign included. */
static inline uint32_t tv_level_bits(const uint8_t *state, int lvl_ctx, int cat,
                                     int node, int abs)
{
    uint32_t c = aeg_cabac_bin_cost[state[lvl_ctx + tv_lvl1[node]]][abs > 1] + 256;
    if (abs > 1) {
        int g = tv_gt1[node], s, cnt;
        if ((cat % 6) == 3 && g > 8)
            g = 8;
        s = state[lvl_ctx + g];
        for (cnt = 2; cnt < 15 && cnt < abs; cnt++) {
            c += aeg_cabac_bin_cost[s][1];
            s = aeg_cabac_trans[s][1];
        }
        if (cnt < 15) {
            c += aeg_cabac_bin_cost[s][0];
        } else {
            int rest = abs - 14, j = 0;
            while (rest >= (1 << (j + 1)))
                j++;
            c += (uint32_t)(2 * j + 1) * 256;
        }
    }
    return c;
}

/* Fills lev[] (natural order, positions of the block's scan only) with the
 * trellis's levels for the unrounded levels x[] / unit errors e[] / signs
 * sgn[] (natural order).  Returns 0 when there is nothing to choose (every
 * coefficient rounds to 0 or a level is out of the search's range), 1 when
 * lev[] was written. */
static int trel_viterbi(const aeg_cabac *sh, const struct trel *t,
                        const float *x, const float *e, const int8_t *sgn,
                        double lambda, int allow_empty, dctcoef *lev)
{
    const struct aeg_cab_cat_ctx *k = &aeg_cab_cat[t->cat];
    const uint8_t *state = sh->state;
    const int n = t->n, big = n == 64;
    const double lb = lambda / 256.0;
    double score[8], nscore[8];
    int8_t prevn[64][8];
    int16_t lvs[64][8];
    int i, start = -1, nd, best;
    for (i = n - 1; i >= 0; i--) {
        float xi = x[t->scan[i]];
        if (xi >= 0.5f) {
            if (start < 0)
                start = i;
            if (xi > 2000.0f)
                return 0;
        }
    }
    if (start < 0)
        return 0;
    for (nd = 0; nd < 8; nd++)
        score[nd] = 1e300;
    score[0] = 0.0;
    for (i = start; i >= 0; i--) {
        const int pos = t->scan[i];
        const double xi = x[pos], ei = e[pos];
        const int q = (int)(xi + 0.5);
        const int flags = i < n - 1;
        /* the last position of the block has no significance/last flags
         * (and no entry in the 8x8 context maps) */
        const int ci = flags ? i : 0;
        const uint8_t *sig = &state[k->sig + (big ? k->sig8[ci] : ci)];
        const uint8_t *lst = &state[k->last + (big ? aeg_cabac_last8[ci] : ci)];
        const double d0 = ei * xi * xi;
        int L;

        for (nd = 0; nd < 8; nd++)
            nscore[nd] = 1e300;
        for (nd = 0; nd < 8; nd++) {
            double sc = score[nd];
            if (sc >= 1e299)
                continue;
            /* level 0: before the last coefficient nothing is sent */
            {
                double c = sc + d0;
                if (nd && flags)
                    c += lb * aeg_cabac_bin_cost[*sig][0];
                if (c < nscore[nd]) {
                    nscore[nd] = c;
                    prevn[i][nd] = (int8_t)nd;
                    lvs[i][nd] = 0;
                }
            }
            for (L = q; L >= 1 && L >= q - 1; L--) {
                uint32_t bits = tv_level_bits(state, k->level, t->cat, nd, L);
                const int nn = tv_next[L > 1][nd];
                double c;
                if (flags)
                    bits += aeg_cabac_bin_cost[*sig][1] +
                            aeg_cabac_bin_cost[*lst][nd == 0];
                c = sc + ei * (xi - L) * (xi - L) + lb * bits;
                if (c < nscore[nn]) {
                    nscore[nn] = c;
                    prevn[i][nn] = (int8_t)nd;
                    lvs[i][nn] = (int16_t)L;
                }
            }
        }
        memcpy(score, nscore, sizeof(score));
    }
    if (t->cbf_ctx >= 0) {
        score[0] += lb * aeg_cabac_bin_cost[state[t->cbf_ctx]][0];
        for (nd = 1; nd < 8; nd++)
            score[nd] += lb * aeg_cabac_bin_cost[state[t->cbf_ctx]][1];
    }
    if (!allow_empty)
        score[0] = 1e300;
    best = 0;
    for (nd = 1; nd < 8; nd++)
        if (score[nd] < score[best])
            best = nd;
    if (score[best] >= 1e299)
        return 0;
    for (i = 0; i < n; i++)
        lev[t->scan[i]] = 0;
    for (i = 0, nd = best; i <= start; i++) {
        const int pos = t->scan[i], L = lvs[i][nd];
        lev[pos] = (dctcoef)(sgn[pos] < 0 ? -L : L);
        nd = prevn[i][nd];
    }
    return 1;
}

/* Whether the level selection starts from the trellis (AEG_TVIT=0 turns it
 * off, for measurements).  PSNR BD-rate against the descent alone, constant
 * QP, five 30-frame clips: p1 -2.07 %, p2 -2.01 %, p4 -2.42 % (p4: -1.8
 * to -3.5 % per clip); time +10 % (p1, p2), +9 % (p3) and +15 % (p4, whose QP-RD
 * trials run it too) on cockL at CRF 24.  The trellis alone, without the
 * descent after it: +0.30 % for -9 % time. */
static int trel_vit_on(void)
{
    static _Atomic int v = -1;
    if (v < 0) {
        const char *ev = getenv("AEG_TVIT");
        v = ev ? atoi(ev) != 0 : 1;
    }
    return v;
}

/* The sweep itself: returns the cost of the levels it settles on, which are
 * written back through `lev`.  The moving cost is kept locally -- one inverse
 * transform and one local price per candidate -- and the block that wins is
 * priced exactly before it is accepted, so the local price can only cost
 * choices, never correctness. */
static int64_t trel_descent_kn(const aeg_cabac *sh, const struct trel *t,
                               dctcoef *lev, const pixel *src, int stride,
                               const pixel *pred, int pred_stride, int qp,
                               double lambda, int passes, int allow_empty,
                               int64_t *out_sse, int64_t *out_bits,
                               int64_t known_sse, int64_t known_bits);

static int64_t trel_descent_ex(const aeg_cabac *sh, const struct trel *t,
                               dctcoef *lev, const pixel *src, int stride,
                               const pixel *pred, int pred_stride, int qp,
                               double lambda, int passes, int allow_empty,
                               int64_t *out_sse, int64_t *out_bits)
{
    return trel_descent_kn(sh, t, lev, src, stride, pred, pred_stride, qp,
                           lambda, passes, allow_empty, out_sse, out_bits, -1, 0);
}

/* trel_descent_ex(), for a caller that has already measured the levels it
 * passes in: known_sse >= 0 is their trel_sse() and known_bits their
 * cabac_block_price() against `sh`, the two numbers the descent starts from. */
static int64_t trel_descent_kn(const aeg_cabac *sh, const struct trel *t,
                               dctcoef *lev, const pixel *src, int stride,
                               const pixel *pred, int pred_stride, int qp,
                               double lambda, int passes, int allow_empty,
                               int64_t *out_sse, int64_t *out_bits,
                               int64_t known_sse, int64_t known_bits)
{
    dctcoef cur[64], trial[64], best[64], scan[64];
    /* 8x8: the trials reuse the current levels' dequantised block and row
     * pass (aeg_idct8_cache_*, exactly trel_sse()'s numbers) */
    aeg_idct8_cache c8;
    int cache_ready = 0;
    /* The levels are held in natural (raster) order, so every array covers
     * the whole block -- an AC block lists 15 coefficients but its levels
     * still span 16 positions -- while `last` is a position in the scan,
     * which is what the coder's last-coefficient flag refers to. */
    const int nat = t->side * t->side;
    const size_t natsz = sizeof(dctcoef) * (size_t)nat;
    int i, pass, nnz = 0, last = -1;
    int64_t bits, sse, cost, best_cost, orig_cost, best_sse, orig_bits, orig_sse;

    {
        /* most blocks are empty: say so before gathering the scan */
        int dummy;
        if (!aeg_nz_last(lev, nat, &dummy))
            return 0;
    }
    if (trel_vit_on()) {
        /* the trellis's levels against the quantiser's, priced exactly; the
         * cheaper pair is where the descent starts */
        const int sd = t->side;
        int16_t res[64];
        dctcoef co[64], vl[64], sc2[64];
        float xx[64], ee[64];
        int8_t sg[64];
        int y, x, dummy;

        for (y = 0; y < sd; y++)
            for (x = 0; x < sd; x++)
                res[y * sd + x] = (int16_t)((int)src[y * stride + x] -
                                            (int)pred[y * pred_stride + x]);
        if (sd == 4)
            aeg_transform_4x4(res, 4, co);
        else
            aeg_transform_8x8(res, 8, co);
        aeg_trel_levels(co, sd, qp, xx, ee, sg);
        memcpy(vl, lev, natsz);
        if (trel_viterbi(sh, t, xx, ee, sg, lambda, allow_empty, vl) &&
            memcmp(vl, lev, natsz)) {
            int64_t ob, os, vb, vs;
            if (known_sse >= 0) {
                os = known_sse;
                ob = known_bits;
            } else {
                for (i = 0; i < t->n; i++)
                    sc2[i] = lev[t->scan[i]];
                ob = cabac_block_price(sh, sc2, t->n, t->cat, t->cbf_ctx);
                os = trel_sse(t, src, stride, pred, pred_stride, lev, qp);
            }
            for (i = 0; i < t->n; i++)
                sc2[i] = vl[t->scan[i]];
            vb = cabac_block_price(sh, sc2, t->n, t->cat, t->cbf_ctx);
            vs = trel_sse(t, src, stride, pred, pred_stride, vl, qp);
            if (vs + trel_price(vb, lambda) < os + trel_price(ob, lambda)) {
                memcpy(lev, vl, natsz);
                known_sse = vs;
                known_bits = vb;
                if (!aeg_nz_last(lev, nat, &dummy)) {
                    if (out_sse) {
                        *out_sse = vs;
                        *out_bits = vb;
                    }
                    return vs + trel_price(vb, lambda);
                }
            } else {
                known_sse = os;
                known_bits = ob;
            }
        }
    }
    memcpy(cur, lev, natsz);
    for (i = 0; i < t->n; i++)
        scan[i] = cur[t->scan[i]];
    nnz = aeg_nz_last(scan, t->n, &last);
    if (!nnz)
        return 0;
    if (known_sse >= 0) {
        bits = known_bits;
        sse = known_sse;
    } else {
        bits = cabac_block_price(sh, scan, t->n, t->cat, t->cbf_ctx);
        sse = trel_sse(t, src, stride, pred, pred_stride, cur, qp);
    }
    orig_cost = best_cost = cost = sse + trel_price(bits, lambda);
    best_sse = orig_sse = sse;
    orig_bits = bits;
    /* what the levels left in lev[] measure and cost, for callers that need
     * both (the price is aeg_cabac_residual_price() of them in scan order) */
    if (out_sse) {
        *out_sse = orig_sse;
        *out_bits = orig_bits;
    }
    memcpy(best, cur, natsz);

    /* Emptying a block only pays when there is almost nothing in it: the flag
     * it saves is one bin, the residual it throws away is up to sixteen. */
    if (allow_empty && nnz <= 2) {
        memcpy(trial, cur, natsz);
        for (i = 0; i < t->n; i++)
            trial[t->scan[i]] = 0;
        for (i = 0; i < t->n; i++)
            scan[i] = 0;
        bits = cabac_block_price(sh, scan, t->n, t->cat, t->cbf_ctx);
        sse = trel_sse(t, src, stride, pred, pred_stride, trial, qp);
        cost = sse + trel_price(bits, lambda);
        if (cost < best_cost) {
            best_cost = cost;
            best_sse = sse;
            memcpy(cur, trial, natsz);
            memcpy(best, trial, natsz);
            last = -1;
        } else {
            /* back to the original levels and their price (the reconstruction
             * and the pricer are deterministic) */
            for (i = 0; i < t->n; i++)
                scan[i] = cur[t->scan[i]];
            bits = orig_bits;
            sse = best_sse;
            cost = orig_cost;
        }
    }
    /* A trial depends only on the state (levels, cost, bits, last), so once
     * a whole sweep's worth of trials has gone by without a move, the state is
     * the one every following trial already saw and rejected: the descent has
     * converged and the remaining sweeps would change nothing. */
    int idle = 0;
    for (pass = 0; pass < passes && idle < 2 * t->n; pass++) {
        for (i = 0; i < t->n && idle < 2 * t->n; i++) {
            int pos = t->scan[i], step;
            if (!cur[pos]) {
                idle += 2;
                continue;
            }
            for (step = -1; step <= 1; step += 2) {
                const dctcoef oldlev = cur[pos];
                int newlev = (int)oldlev + step;
                int last2 = last;
                int64_t dbits, sse2, c;

                if (i == last && !newlev) {
                    int j = i - 1;
                    while (j >= 0 && !cur[t->scan[j]])
                        j--;
                    last2 = j;
                }
                idle++;
                dbits = trel_dbits(sh, t, cur, i, newlev, last);
                /* the distortion is never negative, so a move whose bits
                 * alone already cost what the current levels do cannot be
                 * taken: no need to reconstruct it */
                if (trel_price(bits + dbits, lambda) >= cost)
                    continue;
                /* the trial is the current levels with this one moved */
                if (!cache_ready) {
                    /* built on the first trial that gets this far */
                    if (t->side == 8)
                        aeg_idct8_cache_init(&c8, cur, qp);
                    cache_ready = 1;
                }
                if (t->side == 8) {
                    sse2 = aeg_idct8_cache_ssd(&c8, pos, newlev, pred,
                                               pred_stride, src, stride);
                    cur[pos] = (dctcoef)newlev;
                } else {
                    cur[pos] = (dctcoef)newlev;
                    sse2 = trel_sse(t, src, stride, pred, pred_stride, cur, qp);
                }
                c = sse2 + trel_price(bits + dbits, lambda);
                if (c >= cost) {
                    cur[pos] = oldlev;
                } else {
                    if (t->side == 8)
                        aeg_idct8_cache_set(&c8, pos, newlev);
                    cost = c;
                    sse = sse2;
                    bits += dbits;
                    last = last2;
                    idle = 0;
                    if (c < best_cost) {
                        best_cost = c;
                        best_sse = sse2;
                        memcpy(best, cur, natsz);
                    }
                }
            }
        }
    }
    /* The final word is the exact price of the winner: if the local price
     * talked the sweep into something worse than what the quantiser chose,
     * the quantiser's levels stay. */
    if (memcmp(best, lev, natsz)) {
        for (i = 0; i < t->n; i++)
            scan[i] = best[t->scan[i]];
        bits = cabac_block_price(sh, scan, t->n, t->cat, t->cbf_ctx);
        /* best_sse is trel_sse() of `best`, measured when it was taken */
        if (best_sse + trel_price(bits, lambda) < orig_cost) {
            memcpy(lev, best, natsz);
            if (out_sse) {
                *out_sse = best_sse;
                *out_bits = bits;
            }
        }
    }
    return best_cost;
}

static int64_t trel_descent(const aeg_cabac *sh, const struct trel *t,
                            dctcoef *lev, const pixel *src, int stride,
                            const pixel *pred, int pred_stride, int qp,
                            double lambda, int passes, int allow_empty)
{
    return trel_descent_ex(sh, t, lev, src, stride, pred, pred_stride, qp,
                           lambda, passes, allow_empty, NULL, NULL);
}

/* Whether the intra blocks (Intra_4x4, Intra_8x8) get the level selection
 * too.  AEG_TINTRA=0/1 overrides, for measurements. */
static int trellis_intra_mode(void)
{
    static _Atomic int v = -1;
    if (v < 0) {
        const char *ev = getenv("AEG_TINTRA");
        v = ev ? atoi(ev) : 3;
    }
    return v;
}
#define trellis_intra_on() (trellis_intra_mode() == 1 || \
    (trellis_intra_mode() == 2 && (e->cur_is_p || e->cur_is_b)) || \
    trellis_intra_mode() == 3)
/* The lambda of the level selection on inter luma blocks: 0.75x the plain
 * one.  The mode decision already prices P pictures at 0.65x (they are the
 * references); the selection used the plain lambda and so dropped levels a
 * reference picture should keep.  Measured, ours against ours: preset 3
 * -0.34 / -0.20 % (5 clips / validation set), preset 4 -0.33 / -0.15 %, and
 * -0.40 % over QP 16-25, where the gap to x264 was largest (0.65x on P with
 * 0.75x on B: -0.34 / -0.24 % at preset 3 but -0.28 / -0.09 % at 4; 0.75x on
 * B alone -0.10 %, on P alone -0.33 %; 0.6x -0.28 %; 1.3x +0.89 % at QP
 * 16-25).  AEG_TLI sets the factor in percent. */
static double inter_trellis_lambda(int qp)
{
    return AEG_LAMBDA(qp) * ENV_INT("AEG_TLI", 75) / 100.0;
}

#define trellis_intra_lambda(l) (trellis_intra_mode() == 3 ? (l) * 0.5 : (l))

/* Whether the chroma AC blocks get the level selection.  AEG_TCHROMA. */
static int trellis_chroma_on(void)
{
    static _Atomic int v = -1;
    if (v < 0) {
        const char *ev = getenv("AEG_TCHROMA");
        v = ev ? atoi(ev) != 0 : 1;
    }
    return v;
}

/* The AC levels of one Intra_16x16 block (cat 1) against the DC that is
 * already decided; raster order is also the order the flags' neighbours
 * become final in.  Off unless AEG_TI16=1: ours against ours it is worth
 * -0.04 / +0.07 % of BD-rate at preset 3 (5 clips / validation set) and
 * -0.04 / +0.05 % at preset 4, for 6-9 % of the encode time -- every
 * macroblock prices two Intra_16x16 candidates and few of them win.  With
 * the CABAC trellis (trel_viterbi) in front of the descent it is still
 * +0.07 % for +9 % at preset 4. */
static void trellis_i16_ac(struct aeg_encoder *e, int mbx, int mby, mbenc *c,
                           int blk, const pixel *src, int stride,
                           const pixel *pred, int qp, const int32_t *dcv)
{
    static _Atomic int on = -1;
    struct trel t;
    int a, b, va, vb;

    if (on < 0) {
        const char *ev = getenv("AEG_TI16");
        on = ev ? atoi(ev) != 0 : 0;
    }
    if (!on)
        return;
    cab_nz4_luma(e, mbx, mby, c, blk, &a, &b, &va, &vb);
    t.side = 4;
    t.n = 15;
    t.scan = e->scan4 + 1;
    t.cat = 1 + e->fcat;
    t.cbf_ctx = cab_cbf_ctx(1, a, b, va, vb, 1);
    t.dc = dcv;
    trel_descent(&e->cab_shadow, &t, c->level[blk], src, stride, pred, 16, qp,
                 trellis_intra_lambda(AEG_LAMBDA(qp)), 2, 1);
}

static void cab_nz_mb(struct aeg_encoder *e, int mbx, int mby, int blk,
                      int *a, int *b, int *va, int *vb);

/* Which DC blocks get the trellis: bit 0 the Intra_16x16 luma DC, bit 1 the
 * chroma DC (AEG_TDC, default both).  Ours against ours, PSNR BD-rate at
 * constant QP on the five 30-frame clips: both -0.52 % at preset 4 (every
 * clip -0.33 to -0.87 %), -0.24 % at preset 2, -0.42 % at preset 1, for no
 * measurable time; the luma DC alone -0.12 %, the chroma DC alone -0.30 %
 * (preset 4). */
static int trellis_dc_mode(void)
{
    static _Atomic int v = -1;
    if (v < 0) {
        const char *ev = getenv("AEG_TDC");
        v = ev ? atoi(ev) : 3;
    }
    return v;
}

/* A DC block through the trellis: `lev` holds the quantiser's levels and
 * `x`/`ue`/`sg` the block's aeg_trel_dc_levels().  The trellis's levels and
 * the quantiser's are priced exactly -- `sse_of` reconstructs the macroblock
 * part the DC spreads over with the AC levels as they stand -- and the
 * cheaper ones are left in lev[]. */
typedef int64_t (*trel_dc_sse_fn)(const void *ctx, const dctcoef *dclev);

static void trel_dc_block(const aeg_cabac *sh, const struct trel *t,
                          const float *x, const float *ue, const int8_t *sg,
                          double lambda, dctcoef *lev, trel_dc_sse_fn sse_of,
                          const void *ctx)
{
    dctcoef vl[16], sc[16];
    int64_t ob, vb;
    int i;

    memcpy(vl, lev, sizeof(dctcoef) * (size_t)t->n);
    if (!trel_viterbi(sh, t, x, ue, sg, lambda, 1, vl) ||
        !memcmp(vl, lev, sizeof(dctcoef) * (size_t)t->n))
        return;
    for (i = 0; i < t->n; i++)
        sc[i] = lev[t->scan[i]];
    ob = cabac_block_price(sh, sc, t->n, t->cat, t->cbf_ctx);
    for (i = 0; i < t->n; i++)
        sc[i] = vl[t->scan[i]];
    vb = cabac_block_price(sh, sc, t->n, t->cat, t->cbf_ctx);
    if (sse_of(ctx, vl) + trel_price(vb, lambda) <
        sse_of(ctx, lev) + trel_price(ob, lambda))
        memcpy(lev, vl, sizeof(dctcoef) * (size_t)t->n);
}

struct luma_dc_ctx {
    const mbenc *c;
    const pixel *src, *pred;
    int stride, qp;
};

static int64_t luma_dc_sse(const void *vctx, const dctcoef *dclev)
{
    const struct luma_dc_ctx *k = vctx;
    int32_t dcv[16];
    int64_t sse = 0;
    int blk;

    aeg_dequant_4x4_dc_out(dclev, dcv, k->qp);
    aeg_dc_block_unpermute(dcv);
    for (blk = 0; blk < 16; blk++) {
        const int bx = (blk & 3) * 4, by = (blk >> 2) * 4;
        sse += aeg_idct_add_ssd4(k->c->level[blk], k->qp,
                                 k->pred + by * 16 + bx, 16, &dcv[blk],
                                 k->src + by * k->stride + bx, k->stride);
    }
    return sse;
}

/* The Intra_16x16 luma DC (cat 0), against the AC levels the quantiser has
 * just produced. */
static void trellis_luma_dc(struct aeg_encoder *e, int mbx, int mby,
                            const mbenc *c, const pixel *src, int stride,
                            const pixel *pred, int qp, const dctcoef *had,
                            dctcoef *dcled)
{
    struct luma_dc_ctx k;
    struct trel t;
    float x[16], ue[16];
    int8_t sg[16];
    int a, b, va, vb;

    if (!(trellis_dc_mode() & 1) || !trel_vit_on())
        return;
    cab_nz_mb(e, mbx, mby, AEG_BLK_LUMA_DC, &a, &b, &va, &vb);
    t.side = 4;
    t.n = 16;
    t.scan = e->scan4;
    t.cat = 0 + e->fcat;
    t.cbf_ctx = cab_cbf_ctx(0, a, b, va, vb, 1);
    t.dc = NULL;
    aeg_trel_dc_levels(had, 16, qp, x, ue, sg);
    k.c = c;
    k.src = src;
    k.pred = pred;
    k.stride = stride;
    k.qp = qp;
    trel_dc_block(&e->cab_shadow, &t, x, ue, sg,
                  trellis_intra_lambda(AEG_LAMBDA(qp)), dcled, luma_dc_sse, &k);
}

struct chroma_dc_ctx {
    const dctcoef (*ac)[16];      /* the plane's four AC blocks */
    const pixel *src, *pred;
    int stride, qp;
};

static int64_t chroma_dc_sse(const void *vctx, const dctcoef *dclev)
{
    static const int bxy[4][2] = { {0, 0}, {4, 0}, {0, 4}, {4, 4} };
    const struct chroma_dc_ctx *k = vctx;
    int32_t dcv[4];
    int64_t sse = 0;
    int blk;

    aeg_dequant_2x2_dc_out(dclev, dcv, k->qp);
    for (blk = 0; blk < 4; blk++)
        sse += aeg_idct_add_ssd4(k->ac[blk], k->qp,
                                 k->pred + bxy[blk][1] * 8 + bxy[blk][0], 8,
                                 &dcv[blk],
                                 k->src + bxy[blk][1] * k->stride + bxy[blk][0],
                                 k->stride);
    return sse;
}

/* One plane's chroma DC (cat 3), against its AC levels as quantised. */
static void trellis_chroma_dc(struct aeg_encoder *e, int mbx, int mby,
                              const mbenc *c, int pl, int inter,
                              const pixel *src, int stride, const pixel *pred,
                              int qpc, const dctcoef *had, dctcoef *dclev)
{
    static const uint8_t ident[4] = { 0, 1, 2, 3 };
    struct chroma_dc_ctx k;
    struct trel t;
    float x[4], ue[4];
    int8_t sg[4];
    int a, b, va, vb;
    double lam = AEG_LAMBDA(c->qp);

    if (!(trellis_dc_mode() & 2) || !trel_vit_on())
        return;
    cab_nz_mb(e, mbx, mby, AEG_BLK_CHROMA_DC + pl, &a, &b, &va, &vb);
    t.side = 4;
    t.n = 4;
    t.scan = ident;
    t.cat = 3 + e->fcat;
    t.cbf_ctx = cab_cbf_ctx(3, a, b, va, vb, !inter);
    t.dc = NULL;
    aeg_trel_dc_levels(had, 4, qpc, x, ue, sg);
    k.ac = (const dctcoef (*)[16])c->level[AEG_BLK_CHROMA_CB + pl * 4];
    k.src = src;
    k.pred = pred;
    k.stride = stride;
    k.qp = qpc;
    trel_dc_block(&e->cab_shadow, &t, x, ue, sg,
                  inter ? lam : trellis_intra_lambda(lam), dclev,
                  chroma_dc_sse, &k);
}

/* One macroblock's luma 4x4 blocks of an inter macroblock, whose prediction is
 * in e->pred_l and whose levels are `lev[block][natural]`. */
static void trellis_luma4(struct aeg_encoder *e, int mbx, int mby, mbenc *c,
                          int qp, dctcoef lev[][16], double lambda)
{
    const pixel *src = e->srcp[0] + mby * 16 * e->stride[0] + mbx * 16;
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
        t.scan = e->scan4;
        t.cat = 2 + e->fcat;
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
static void trellis_luma8(struct aeg_encoder *e, const pixel *src, int stride,
                          const pixel *pred, int pred_stride, dctcoef *lev,
                          int qp, double lambda)
{
    struct trel t;

    if (!e->p.cabac || e->p.trellis <= 0)
        return;
    t.side = 8;
    t.n = 64;
    t.scan = e->scan8;
    t.cat = 5 + e->fcat;
    t.cbf_ctx = -1;
    t.dc = NULL;
    trel_descent(&e->cab_shadow, &t, lev, src, stride, pred, pred_stride, qp,
                 lambda, 1, 0);
}

/* Whether the batched paths reconstruct on the host instead of launching the
 * inverse.  With the level selection on (trellis), the levels the inverse
 * needs are the ones the host has just selected, so a device inverse is an
 * upload of those levels, a launch, a synchronisation and a download of the
 * residual -- a full round trip, for work the host does in one SIMD pass per
 * block (aeg_idct_add_block(), the serial path's own reconstruction, so the
 * result is the same bytes).  Any device's round trip costs more than that,
 * so the batch keeps one launch per macroblock (the forward one, whose
 * output the host needs anyway) instead of two.  An explicitly placed
 * inverse (-s iq=...) still runs where it was put. */
static int batch_host_inverse(const struct aeg_encoder *e)
{
    return e->p.trellis > 0 && e->p.cabac &&
           e->d.forced[AEG_STAGE_IQUANT_IDCT] < 0;
}

/* Without level selection (presets 0-2, CAVLC) nothing runs on the host
 * between the forward and the inverse half of the batch, so a backend with
 * the fused operation does both in one call: on a device one transfer each
 * way and one launch per macroblock instead of two round trips.  The fused
 * kernel is the two kernels' code (kernels.h), so the levels and the residual
 * are the same; AEG_TQIQ=0 makes the two calls instead, which is how the
 * separate kernels stay exercised in an encode.  The time goes to both
 * stages, half each, so -v 2 still shows where each ran. */
static int use_tq_iq(const struct aeg_encoder *e)
{
    static _Atomic int env = -2;
    if (env == -2) {
        const char *ev = getenv("AEG_TQIQ");
        env = ev ? atoi(ev) != 0 : 1;
    }
    return env && e->d.ops.tq_iq_batch && !(e->p.trellis > 0 && e->p.cabac);
}

static void tq_iq_rec(struct aeg_encoder *e, int64_t ns)
{
    int g = e->d.ops.backend != AEG_BACKEND_CPU;
    stage_rec(e, AEG_STAGE_TRANSFORM, g, ns / 2);
    stage_rec(e, AEG_STAGE_IQUANT_IDCT, g, ns - ns / 2);
}

static void code_luma_inter(struct aeg_encoder *e, int mbx, int mby, mbenc *c)
{
    const pixel *src = e->srcp[0] + mby * 16 * e->stride[0] + mbx * 16;
    const pixel *pred = e->pred_l;
    int blk, qp = c->qp;

    luma_inter_gather(e, src, pred, e->coef_res);
    if (e->coef_batch && use_tq_iq(e)) {
        int64_t t0 = aeg_now_ns();
        e->d.ops.tq_iq_batch(e->d.ops.priv, e->coef_res, 16, qp, 0,
                             e->coef_lev, e->coef_nz, e->coef_rec);
        tq_iq_rec(e, aeg_now_ns() - t0);
        luma_inter_finish(e, mbx, mby, c, e->coef_lev, e->coef_nz, e->coef_rec,
                          pred);
        return;
    }
    if (e->coef_batch) {
        /* One transform+quantise call and one inverse call for the sixteen
         * blocks instead of sixteen of each. */
        int64_t t0 = aeg_now_ns();
        e->d.ops.tq_batch(e->d.ops.priv, e->coef_res, 16, qp, 0,
                          e->coef_lev, e->coef_nz);
        stage_rec(e, AEG_STAGE_TRANSFORM,
                            e->d.ops.backend != AEG_BACKEND_CPU,
                            aeg_now_ns() - t0);
        if (e->p.trellis > 0 && e->p.cabac) {
            /* The levels of the batch are in the same layout as the scalar
             * path's, one block after the other in natural order, so the same
             * selection runs on them -- and the reconstruction has to be
             * rebuilt afterwards, because it was made from the old levels. */
            trellis_luma4(e, mbx, mby, c, qp, (dctcoef (*)[16])e->coef_lev,
                          inter_trellis_lambda(qp));
            if (batch_host_inverse(e)) {
                luma_inter_finish(e, mbx, mby, c, e->coef_lev, NULL, NULL,
                                  pred);
                return;
            }
            t0 = aeg_now_ns();
            e->d.ops.iq_idct_batch(e->d.ops.priv, e->coef_lev, 16, qp,
                                   e->coef_rec);
            stage_rec(e, AEG_STAGE_IQUANT_IDCT,
                                e->d.ops.backend != AEG_BACKEND_CPU,
                                aeg_now_ns() - t0);
            luma_inter_finish(e, mbx, mby, c, e->coef_lev, NULL,
                              e->coef_rec, pred);
            return;
        }
        t0 = aeg_now_ns();
        e->d.ops.iq_idct_batch(e->d.ops.priv, e->coef_lev, 16, qp, e->coef_rec);
        stage_rec(e, AEG_STAGE_IQUANT_IDCT,
                            e->d.ops.backend != AEG_BACKEND_CPU,
                            aeg_now_ns() - t0);
        luma_inter_finish(e, mbx, mby, c, e->coef_lev, e->coef_nz, e->coef_rec,
                          pred);
        return;
    }
    for (blk = 0; blk < 16; blk++) {
        dctcoef coef[16];
        aeg_transform_4x4(e->coef_res + 16 * blk, 4, coef);
        aeg_quant_4x4(coef, c->level[blk], qp, 0);
    }
    trellis_luma4(e, mbx, mby, c, qp, c->level, inter_trellis_lambda(qp));
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
static void code_luma8_block_lv(struct aeg_encoder *e, int mbx, int mby, mbenc *c,
                                int g, const pixel *pred, int pred_stride,
                                int intra, int have_levels);

static void code_luma8_block(struct aeg_encoder *e, int mbx, int mby, mbenc *c,
                             int g, const pixel *pred, int pred_stride,
                             int intra)
{
    code_luma8_block_lv(e, mbx, mby, c, g, pred, pred_stride, intra, 0);
}

/* have_levels: c->level8[g] already holds the levels this would select
 * (the Intra_8x8 mode decision measured them) */
static void code_luma8_block_lv(struct aeg_encoder *e, int mbx, int mby, mbenc *c,
                                int g, const pixel *pred, int pred_stride,
                                int intra, int have_levels)
{
    const int s = e->stride[0];
    const pixel *src = e->srcp[0] + mby * 16 * s + mbx * 16;
    pixel *dst = e->recp[0] + mby * 16 * s + mbx * 16;
    const int bx = (g & 1) * 8, by = (g >> 1) * 8;
    int16_t res[64];
    dctcoef coef[64];
    int x, y;

    for (y = 0; y < 8; y++)
        for (x = 0; x < 8; x++)
            res[y * 8 + x] = (int16_t)((int)src[(by + y) * s + bx + x] -
                                       (int)pred[y * pred_stride + x]);
    if (!have_levels) {
        aeg_transform_8x8(res, 8, coef);
        aeg_quant_8x8(coef, c->level8[g], c->qp, intra);
    }
    if (!have_levels && (!intra || trellis_intra_on()))
        trellis_luma8(e, src + by * s + bx, s, pred, pred_stride, c->level8[g],
                      c->qp, intra ? trellis_intra_lambda(AEG_LAMBDA(c->qp))
                                   : inter_trellis_lambda(c->qp));
    {
        int dummy;
        c->nz8[g] = aeg_nz_last(c->level8[g], 64, &dummy) != 0;
    }

    if (c->nz8[g]) {
        aeg_idct_8x8_add(dst + by * s + bx, s, c->level8[g], c->qp, pred,
                         pred_stride);
    } else {
        /* no levels: the reconstruction is the prediction */
        for (y = 0; y < 8; y++)
            memcpy(dst + (by + y) * s + bx, pred + y * pred_stride,
                   8 * sizeof(pixel));
    }
    c->sse += aeg_ssd(8, 8, dst + by * s + bx, s, src + by * s + bx, s);
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
                if (!e->p.cabac)        /* CAVLC estimate */
                    c->bits += aeg_cavlc_block_cost8(c->level8[g], k | e->fscan8,
                                                     luma_nc(e, mbx, mby, c, b));
                c->nz[b] = (int8_t)aeg_cavlc_nz8(c->level8[g], k | e->fscan8);
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
    const pixel *src = e->srcp[0] + mby * 16 * s + mbx * 16;
    int cost4, cost8;

    if (!t8_allowed(e, c))
        return 0;
    if (e->t8_force)
        return e->t8_force == 2;
    /* (aeg_satd is the sum of the sixteen 4x4 SATDs, and aeg_sa8d_16x16
     * the sum of the four 8x8 ones) */
    cost8 = aeg_sa8d_16x16(src, s, e->pred_l, 16);
    cost4 = aeg_satd(16, 16, src, s, e->pred_l, 16);
    return cost8 < cost4;
}

/* ------------------------------------------------------------------ */
/* Chroma coding                                                       */
/* ------------------------------------------------------------------ */
static void code_chroma_run(struct aeg_encoder *e, int mbx, int mby, mbenc *c,
                            int inter);

/* The chroma of a macroblock depends on the prediction (pred_c), the QP, and
 * whether the macroblock is intra -- the neighbours' reconstruction and flags
 * and the shadow coder's state do not change while code_mb() decides.  The
 * decision codes the same chroma many times over (every intra candidate with
 * the same chroma mode, the winner once more at the end), so the result is
 * kept per macroblock and replayed, reconstruction included. */
static void code_chroma(struct aeg_encoder *e, int mbx, int mby, mbenc *c,
                        int inter)
{
    const int cs = e->stride[1];
    struct cmemo *m;
    int pl, y, k;
    int64_t sse0;
    int bits0;

    if (!inter) {
        int avail_l = mbx > 0, avail_t = nb_top_ok(e, mby);
        /* both predictions first: each reads only the neighbours of its own
         * plane, which the other plane's reconstruction does not touch */
        for (pl = 0; pl < 2; pl++) {
            pixel *base = e->recp[1 + pl] + mby * 8 * cs + mbx * 8;
            /* The predictor wants a contiguous left column; the picture is
             * strided, so gather it first (same as the luma path). */
            pixel leftc[8];
            for (y = 0; y < 8; y++)
                leftc[y] = avail_l ? base[y * cs - 1] : PIXEL_MID;
            aeg_intra_predict_chroma(e->pred_c[pl], 8, base - cs, leftc,
                                     avail_t, avail_l, avail_t && avail_l,
                                     c->chroma_mode);
        }
    }
    for (k = 0; k < 8; k++) {
        m = &e->cmemo[k];
        if (m->gen == e->cmemo_gen && m->inter == inter && m->qp == c->qp &&
            !memcmp(m->pred, e->pred_c, sizeof(m->pred))) {
            memcpy(c->level[AEG_BLK_CHROMA_DC], m->level, sizeof(m->level));
            memcpy(&c->nz[AEG_BLK_CHROMA_DC], m->nz, sizeof(m->nz));
            c->cbp = (c->cbp & 0x0f) | m->cbp;
            c->sse += m->dsse;
            c->bits += m->dbits;
            for (pl = 0; pl < 2; pl++) {
                pixel *base = e->recp[1 + pl] + mby * 8 * cs + mbx * 8;
                for (y = 0; y < 8; y++)
                    memcpy(base + y * cs, m->rec[pl] + y * 8, 8 * sizeof(pixel));
            }
            return;
        }
    }
    sse0 = c->sse;
    bits0 = c->bits;
    code_chroma_run(e, mbx, mby, c, inter);
    m = &e->cmemo[e->cmemo_next];
    e->cmemo_next = (e->cmemo_next + 1) & 7;
    m->gen = e->cmemo_gen;
    m->inter = inter;
    m->qp = c->qp;
    memcpy(m->pred, e->pred_c, sizeof(m->pred));
    memcpy(m->level, c->level[AEG_BLK_CHROMA_DC], sizeof(m->level));
    memcpy(m->nz, &c->nz[AEG_BLK_CHROMA_DC], sizeof(m->nz));
    m->cbp = c->cbp & 0x30;
    m->dsse = c->sse - sse0;
    m->dbits = c->bits - bits0;
    for (pl = 0; pl < 2; pl++) {
        const pixel *base = e->recp[1 + pl] + mby * 8 * cs + mbx * 8;
        for (y = 0; y < 8; y++)
            memcpy(m->rec[pl] + y * 8, base + y * cs, 8 * sizeof(pixel));
    }
}

/* code_chroma_run() through the coefficient batch (dc_batch_on): both
 * planes' eight blocks forward in one call, each plane's 2x2 DC on the host,
 * the eight inverses in one call; then per block exactly what the per-block
 * path does after its inverse.  No level selection runs here. */
static void code_chroma_batch(struct aeg_encoder *e, int mbx, int mby,
                              mbenc *c, int qpc)
{
    static const int bxy[4][2] = { {0, 0}, {4, 0}, {0, 4}, {4, 4} };
    const int cs = e->stride[1];
    int16_t res[8 * 16], rec[8 * 16];
    int32_t dc[8], dcout[8], nzb[8];
    dctcoef *lev = &c->level[AEG_BLK_CHROMA_CB][0];
    int pl, blk, i, x, y;
    int64_t t0;

    for (pl = 0; pl < 2; pl++) {
        const pixel *src = e->srcp[1 + pl] + mby * 8 * cs + mbx * 8;
        const pixel *pred = e->pred_c[pl];
        for (blk = 0; blk < 4; blk++) {
            int16_t *r = res + 16 * (pl * 4 + blk);
            for (y = 0; y < 4; y++)
                for (x = 0; x < 4; x++)
                    r[y * 4 + x] =
                        (int16_t)((int)src[(bxy[blk][1] + y) * cs + bxy[blk][0] + x] -
                                  (int)pred[(bxy[blk][1] + y) * 8 + bxy[blk][0] + x]);
        }
    }
    t0 = aeg_now_ns();
    e->d.ops.tq_dc_batch(e->d.ops.priv, res, 8, qpc, 1, lev, nzb, dc);
    stage_rec(e, AEG_STAGE_TRANSFORM, e->d.ops.backend != AEG_BACKEND_CPU,
              aeg_now_ns() - t0);
    for (pl = 0; pl < 2; pl++) {
        dctcoef d4[4], had[4], dclev[4];
        int nz_dc = 0;
        for (blk = 0; blk < 4; blk++)
            d4[blk] = (dctcoef)dc[pl * 4 + blk];
        aeg_transform_dc2x2_fwd(d4, had);
        aeg_quant_2x2_dc(had, dclev, qpc, 1);
        for (i = 0; i < 4; i++)
            nz_dc += dclev[i] != 0;
        memcpy(c->level[AEG_BLK_CHROMA_DC + pl], dclev, sizeof(dclev));
        c->nz[AEG_BLK_CHROMA_DC + pl] = (int8_t)nz_dc;
        aeg_dequant_2x2_dc_out(dclev, dcout + 4 * pl, qpc);
    }
    t0 = aeg_now_ns();
    e->d.ops.iq_dc_batch(e->d.ops.priv, lev, dcout, 8, qpc, rec);
    stage_rec(e, AEG_STAGE_IQUANT_IDCT, e->d.ops.backend != AEG_BACKEND_CPU,
              aeg_now_ns() - t0);
    for (pl = 0; pl < 2; pl++) {
        pixel *base = e->recp[1 + pl] + mby * 8 * cs + mbx * 8;
        const pixel *src = e->srcp[1 + pl] + mby * 8 * cs + mbx * 8;
        for (blk = 0; blk < 4; blk++) {
            const int b = AEG_BLK_CHROMA_CB + pl * 4 + blk;
            pixel *o = base + bxy[blk][1] * cs + bxy[blk][0];
            int dummy, nz = aeg_nz_last(c->level[b], 16, &dummy);
            c->nz[b] = (int8_t)nz;
            if (nz)
                c->cbp |= 0x20;
            add_res4(o, cs, e->pred_c[pl] + bxy[blk][1] * 8 + bxy[blk][0], 8,
                     rec + 16 * (pl * 4 + blk));
            c->sse += aeg_ssd(4, 4, o, cs, src + bxy[blk][1] * cs + bxy[blk][0],
                              cs);
        }
        if (c->nz[AEG_BLK_CHROMA_DC + pl]) {
            c->cbp |= 0x10;
            if (!e->p.cabac)            /* CAVLC estimate */
                c->bits += aeg_cavlc_block_cost(c->level[AEG_BLK_CHROMA_DC + pl],
                                                0, 2);
        }
        for (blk = 0; blk < 4 && !e->p.cabac; blk++)
            if (c->nz[AEG_BLK_CHROMA_CB + pl * 4 + blk])
                c->bits += aeg_cavlc_block_cost(
                    c->level[AEG_BLK_CHROMA_CB + pl * 4 + blk],
                    chroma_nc(e, mbx, mby, c, pl, blk), 3 | e->fscan);
    }
}

/* code_chroma() itself; pred_c already holds the prediction of both planes */
static void code_chroma_run(struct aeg_encoder *e, int mbx, int mby, mbenc *c,
                            int inter)
{
    static const int bxy[4][2] = { {0, 0}, {4, 0}, {0, 4}, {4, 4} };
    int qpc = aeg_qp_chroma(c->qp, e->pic.chroma_qp_offset);
    int pl, blk, i, x, y;

    c->cbp &= 0x0f;
    if (dc_batch_on(e, e->p.trellis > 0 && e->p.cabac && trellis_chroma_on())) {
        code_chroma_batch(e, mbx, mby, c, qpc);
        return;
    }
    for (pl = 0; pl < 2; pl++) {
        pixel *base = e->recp[1 + pl] + mby * 8 * e->stride[1] + mbx * 8;
        const pixel *src = e->srcp[1 + pl] + mby * 8 * e->stride[1] + mbx * 8;
        pixel *pred = e->pred_c[pl];
        dctcoef dc[4], had[4], dclev[4];
        int32_t dcout[4];
        int nz_dc = 0;

        for (blk = 0; blk < 4; blk++) {
            int16_t res[16];
            dctcoef coef[16];
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
        if (e->p.trellis > 0 && e->p.cabac && trellis_chroma_on())
            trellis_chroma_dc(e, mbx, mby, c, pl, inter, src, e->stride[1], pred,
                              qpc, had, dclev);
        for (i = 0; i < 4; i++)
            nz_dc += dclev[i] != 0;
        memcpy(c->level[AEG_BLK_CHROMA_DC + pl], dclev, sizeof(dclev));
        c->nz[AEG_BLK_CHROMA_DC + pl] = (int8_t)nz_dc;
        aeg_dequant_2x2_dc_out(dclev, dcout, qpc);
        for (blk = 0; blk < 4; blk++) {
            const pixel *pr = pred + bxy[blk][1] * 8 + bxy[blk][0];
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
                t.scan = e->scan4 + 1;
                t.cat = 4 + e->fcat;
                t.cbf_ctx = cab_cbf_ctx(4, a, b, va, vb, !inter);
                t.dc = &dcv;
                trel_descent(&e->cab_shadow, &t,
                             c->level[AEG_BLK_CHROMA_CB + pl * 4 + blk],
                             src + bxy[blk][1] * e->stride[1] + bxy[blk][0],
                             e->stride[1], pr, 8, qpc,
                             inter ? lam : trellis_intra_lambda(lam), 2, 1);
            }
            {
                int dummy;
                nz = aeg_nz_last(c->level[AEG_BLK_CHROMA_CB + pl * 4 + blk],
                                 16, &dummy);
            }
            c->nz[AEG_BLK_CHROMA_CB + pl * 4 + blk] = (int8_t)nz;
            if (nz)
                c->cbp |= 0x20;

            {
                pixel *o = base + bxy[blk][1] * e->stride[1] + bxy[blk][0];
                aeg_idct_add_block(o, e->stride[1],
                                   c->level[AEG_BLK_CHROMA_CB + pl * 4 + blk],
                                   qpc, pr, 8, &dcv);
                c->sse += aeg_ssd(4, 4, o, e->stride[1],
                                  src + bxy[blk][1] * e->stride[1] + bxy[blk][0],
                                  e->stride[1]);
            }
        }
        if (nz_dc) {
            c->cbp |= (pl == 0) ? 0x10 : 0x10;
            if (!e->p.cabac)            /* CAVLC estimate */
                c->bits += aeg_cavlc_block_cost(c->level[AEG_BLK_CHROMA_DC + pl], 0, 2);
        }
        for (blk = 0; blk < 4 && !e->p.cabac; blk++)
            if (c->nz[AEG_BLK_CHROMA_CB + pl * 4 + blk])
                c->bits += aeg_cavlc_block_cost(
                    c->level[AEG_BLK_CHROMA_CB + pl * 4 + blk],
                    chroma_nc(e, mbx, mby, c, pl, blk), 3 | e->fscan);
    }
}

/* ------------------------------------------------------------------ */
/* Prediction candidates                                               */
/* ------------------------------------------------------------------ */
static int intra16_sad(struct aeg_encoder *e, int mbx, int mby, int mode)
{
    const pixel *p = e->recp[0] + mby * 16 * e->stride[0] + mbx * 16;
    const pixel *src = e->srcp[0] + mby * 16 * e->stride[0] + mbx * 16;
    pixel left[16];
    int y;
    /* The predictor expects the left column contiguously. */
    for (y = 0; y < 16; y++)
        left[y] = p[-1 + y * e->stride[0]];
    aeg_intra_predict_16x16(e->pred_l, 16, p - e->stride[0], left,
                            nb_top_ok(e, mby), mbx > 0,
                            nb_top_ok(e, mby) && mbx > 0, mode);
    return aeg_sad(16, 16, e->pred_l, 16, src, e->stride[0]);
}

static int chroma_sad(struct aeg_encoder *e, int mbx, int mby, int pl, int mode)
{
    const pixel *p = e->recp[1 + pl] + mby * 8 * e->stride[1] + mbx * 8;
    const pixel *src = e->srcp[1 + pl] + mby * 8 * e->stride[1] + mbx * 8;
    pixel left[8];
    int y;
    for (y = 0; y < 8; y++)
        left[y] = p[-1 + y * e->stride[1]];
    aeg_intra_predict_chroma(e->pred_c[pl], 8, p - e->stride[1], left,
                             nb_top_ok(e, mby), mbx > 0,
                             nb_top_ok(e, mby) && mbx > 0, mode);
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
/* 8.4.2.3.2, one list: weight, round, offset, clip. */
static void wp_apply(pixel *d, int stride, int w, int h, int wt, int off,
                     int denom)
{
    int x, y;
    int rnd = denom ? 1 << (denom - 1) : 0;

    for (y = 0; y < h; y++)
        for (x = 0; x < w; x++) {
            /* 8.4.2.3: the offset is in 8-bit units, o << (BitDepth - 8) */
            int v = ((d[y * stride + x] * wt + rnd) >> denom) +
                    off * (1 << (BIT_DEPTH - 8));
            d[y * stride + x] = aeg_clip_pixel(v);
        }
}

/* Whether the current picture sends a pred_weight_table (and so refers to
 * the weighted twin of the picture parameter set). */
static int wp_used(const struct aeg_encoder *e)
{
    return e->cur_is_p && (e->wp.lflag[0] || e->wp.cflag[0]);
}

static void mc_rect_luma_to(struct aeg_encoder *e, int list, pixel *dl,
                            int mbx, int mby, int x, int y, int w, int h,
                            int mvx, int mvy)
{
    const pixel *ref = e->refpl[list][0];
    const aeg_mc_ctx *mc = &e->mcr[list];
    const pixel *r = ref + (mby * 16 + y + (mvy >> 2)) * e->stride[0] +
                       mbx * 16 + x + (mvx >> 2);
    if (e->cur_is_p && !list && e->wp.lflag[0]) {
        /* the real reference, interpolated, then weighted */
        r = e->wp_orig0 + (r - ref);
        aeg_mc_luma(dl + y * 16 + x, 16, r, e->stride[0], w, h,
                    mvx & 3, mvy & 3);
        wp_apply(dl + y * 16 + x, 16, w, h, e->wp.lw[0], e->wp.lo[0],
                 e->wp.luma_denom);
        return;
    }
    if (mc->ready)
        aeg_mc_luma_p(dl + y * 16 + x, 16, mc, r, w, h, mvx & 3, mvy & 3);
    else
        aeg_mc_luma(dl + y * 16 + x, 16, r, e->stride[0], w, h,
                    mvx & 3, mvy & 3);
}

static void mc_rect_chroma_to(struct aeg_encoder *e, int list, pixel *dc[2],
                              int mbx, int mby, int cx, int cy, int cw, int ch,
                              int mvx, int mvy)
{
    int pl;
    /* a field predicting from the field of the other parity: the chroma
     * rows sit a quarter of a chroma line off (8.4.1.4, Table 8-9/8-10) */
    if (e->fld || e->cfld)
        mvy += e->fld_cofs[list];
    for (pl = 0; pl < 2; pl++) {
        const pixel *rc = e->refpl[list][1 + pl] +
                            (mby * 8 + cy + (mvy >> 3)) * e->stride[1] +
                            mbx * 8 + cx + (mvx >> 3);
        aeg_mc_chroma(dc[pl] + cy * 8 + cx, 8, rc, e->stride[1],
                      cw, ch, mvx & 7, mvy & 7);
        if (e->cur_is_p && e->wp.cflag[list])
            wp_apply(dc[pl] + cy * 8 + cx, 8, cw, ch, e->wp.cw[list][pl],
                     e->wp.co[list][pl], e->wp.chroma_denom);
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
    pixel *dc[2] = { e->pred_c[0], e->pred_c[1] };
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
            ref = c->ref[b0];
        mc_rect_luma_to(e, ref, e->pred_l, mbx, mby, geo[0], geo[1], geo[2],
                        geo[3], c->mv[b0][0], c->mv[b0][1]);
        mc_rect_chroma_to(e, ref, (pixel *[]){e->pred_c[0], e->pred_c[1]}, mbx,
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
                    e->pred_l[o] = (pixel)((e->pred_l[o] + e->pred_l2[o] + 1) >> 1);
                }
        {
            pixel *d0[2] = { e->pred_c[0], e->pred_c[1] };
            pixel *d1[2] = { (lists & 1) ? e->pred_c2[0] : e->pred_c[0],
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
                            (pixel)((e->pred_c[pl][o] + e->pred_c2[pl][o] + 1) >> 1);
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
    /* The groups' vectors usually agree (they differ only through colZero);
     * then one 16x16 motion compensation per list gives the same samples as
     * four 8x8 ones -- the interpolation only depends on the position. */
    const int b0s[4] = { 0, 2, 8, 10 };
    int uni = 1, sz, ng;

    for (g = 1; g < 4 && uni; g++)
        uni = c->mv[b0s[g]][0] == c->mv[0][0] && c->mv[b0s[g]][1] == c->mv[0][1] &&
              c->mv1[b0s[g]][0] == c->mv1[0][0] && c->mv1[b0s[g]][1] == c->mv1[0][1];
    sz = uni ? 16 : 8;
    ng = uni ? 1 : 4;
    if (lists != 3) {
        /* spatial direct on one list: an ordinary single-list prediction
         * per 8x8 group (the groups' vectors differ only through colZero) */
        for (g = 0; g < ng; g++) {
            int x0 = (g & 1) * 8, y0 = (g >> 1) * 8;
            int b0 = (y0 >> 2) * 4 + (x0 >> 2);
            int l = lists == 2;
            int mx = l ? c->mv1[b0][0] : c->mv[b0][0];
            int my = l ? c->mv1[b0][1] : c->mv[b0][1];
            pixel *d0[2] = { e->pred_c[0], e->pred_c[1] };

            mc_rect_luma_to(e, l, e->pred_l, mbx, mby, x0, y0, sz, sz, mx, my);
            mc_rect_chroma_to(e, l, d0, mbx, mby, x0 / 2, y0 / 2, sz / 2, sz / 2, mx, my);
        }
        return;
    }
    for (g = 0; g < ng; g++) {
        int x0 = (g & 1) * 8, y0 = (g >> 1) * 8;
        /* Index of the 8x8 group's first 4x4 block: the group is two blocks
         * wide, so the row stride is four blocks, not sixteen. */
        int b0 = (y0 >> 2) * 4 + (x0 >> 2);
        pixel *d0[2] = { e->pred_c[0], e->pred_c[1] };
        pixel *d1[2] = { e->pred_c2[0], e->pred_c2[1] };

        mc_rect_luma_to(e, 0, e->pred_l, mbx, mby, x0, y0, sz, sz,
                        c->mv[b0][0], c->mv[b0][1]);
        mc_rect_luma_to(e, 1, e->pred_l2, mbx, mby, x0, y0, sz, sz,
                        c->mv1[b0][0], c->mv1[b0][1]);
        for (y = 0; y < sz; y++)
            for (x = 0; x < sz; x++) {
                int o = (y0 + y) * 16 + x0 + x;
                e->pred_l[o] = (pixel)((e->pred_l[o] + e->pred_l2[o] + 1) >> 1);
            }
        mc_rect_chroma_to(e, 0, d0, mbx, mby, x0 / 2, y0 / 2, sz / 2, sz / 2,
                          c->mv[b0][0], c->mv[b0][1]);
        mc_rect_chroma_to(e, 1, d1, mbx, mby, x0 / 2, y0 / 2, sz / 2, sz / 2,
                          c->mv1[b0][0], c->mv1[b0][1]);
        for (pl = 0; pl < 2; pl++)
            for (y = 0; y < sz / 2; y++)
                for (x = 0; x < sz / 2; x++) {
                    int o = (y0 / 2 + y) * 8 + x0 / 2 + x;
                    e->pred_c[pl][o] =
                        (pixel)((e->pred_c[pl][o] + e->pred_c2[pl][o] + 1) >> 1);
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

/* Whether the macroblock above-right of (mbx, mby) -- neighbour C -- can be
 * decoded before it, the row above being in the slice (nb_top_ok): not in
 * the last column, and in an MBAFF frame not for the bottom macroblock of a
 * frame pair (its above-right neighbour is the top macroblock of the next
 * pair; a bottom field macroblock's is the next pair above's bottom one). */
static inline int c_right_ok(const struct aeg_encoder *e, int mbx, int mby)
{
    return mbx + 1 < e->mb_w && !(e->mbaff && !e->cfld && (mby & 1));
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
             (int)(list ? m->ref_idx1[blk] : m->ref_idx[blk]));
}

/* The macroblock holding the sample above-left of macroblock (mbx, mby)
 * (neighbour D, mbx > 0) and the 4x4 block of it, or NULL.  In an MBAFF
 * frame (Table 6-4) a frame bottom macroblock finds it in the left pair: row
 * 15 of its top macroblock, or with a field pair row 7 of the bottom one
 * (frame line 15 of the pair is bottom field line 7); every other kind in
 * the pair above-left, as it finds B in the pair above. */
static const aeg_mb *nb_mbD(const struct aeg_encoder *e, int mbx, int mby,
                            int *blk)
{
    *blk = 15;
    if (e->mbaff && !e->cfld && (mby & 1)) {
        const aeg_mb *l0 = &e->mbs[(mby - 1) * e->mb_w + mbx - 1];
        if (l0->fld) {
            *blk = 7;
            return l0 + e->mb_w;
        }
        return l0;
    }
    if (!nb_top_ok(e, mby))
        return NULL;
    return nb_mbB(e, mbx - 1, mby);
}

/* The vector of the 4x4 block covering luma location (xN, yN) of macroblock
 * (mbx, mby), outside the macroblock: left (xN < 0 <= yN), above (yN < 0 <=
 * xN < 16), above-right (xN > 15) or above-left (both < 0), as 6.4.12
 * locates it (Table 6-4 in an MBAFF frame).  A neighbour of the other kind
 * has its vector and reference index brought to the current macroblock's
 * (8.4.1.3.1): a frame one seen from a field macroblock halves the vertical
 * component and doubles the index, a field one seen from a frame macroblock
 * the other way round. */
static void nb_mv_at(const struct aeg_encoder *e, int mbx, int mby, int xN,
                     int yN, int list, nb_mv *n)
{
    const aeg_mb *m = NULL;
    int blk = 0;

    if (xN < 0 && yN >= 0) {                                    /* A */
        if (mbx > 0) {
            int yM;
            m = nb_left(e, mbx, mby, yN, 16, &yM);
            blk = (yM >> 2) * 4 + 3;
        }
    } else if (xN >= 0 && xN < 16) {                            /* B */
        if (nb_top_ok(e, mby)) {
            m = nb_mbB(e, mbx, mby);
            blk = 12 + (xN >> 2);
        }
    } else if (xN >= 16) {                                      /* C */
        if (nb_top_ok(e, mby) && c_right_ok(e, mbx, mby)) {
            m = nb_mbB(e, mbx + 1, mby);
            blk = 12;
        }
    } else if (mbx > 0) {                                       /* D */
        m = nb_mbD(e, mbx, mby, &blk);
    }
    nb_from_mb(n, m, blk, list);
    if (e->mbaff && m && n->ref == NB_OK && m->fld != e->cfld) {
        if (e->cfld) {
            n->ref_idx *= 2;
            n->mvy /= 2;
        } else {
            n->ref_idx >>= 1;
            n->mvy *= 2;
        }
    }
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
    if (bx > 0)
        NB_CUR(A, bx - 1, by);
    else
        nb_mv_at(e, mbx, mby, -1, 4 * by, list, &A);

    /* B: the block above the partition (inside this macroblock when by > 0). */
    if (by > 0)
        NB_CUR(B, bx, by - 1);
    else
        nb_mv_at(e, mbx, mby, 4 * bx, -1, list, &B);

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
        } else {
            /* above (cx <= 3) or above-right of the macroblock */
            nb_mv_at(e, mbx, mby, 4 * cx, -1, list, &C);
        }
        if (C.ref == NB_UNAVAIL) {
            /* The decoder's fallback is the cache slot one before the block
             * directly above the partition, i.e. the block above-left of the
             * partition's top-left corner (in an MBAFF frame located like
             * any other neighbour sample: FFmpeg's fetch_diagonal_mv). */
            if (by > 0 && bx > 0)
                NB_CUR(C, bx - 1, by - 1);
            else
                nb_mv_at(e, mbx, mby, 4 * bx - 1, 4 * by - 1, list, &C);
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
    /* (a B field macroblock: every partition of a list on its b_ref) */
    int8_t cr[16];
    const int r = e->bfld_on ? e->b_ref[list] : 0;
    if (r)
        memset(cr, r, sizeof(cr));
    predict_mv_at_ref(e, mbx, mby, cur, r ? cr : NULL, avail, type, p, bx, by,
                      pw, list, r, pmvx, pmvy);
}

/* The 16x16 predictor for list-0 reference `ref` (no neighbour inside the
 * macroblock, no shape short-circuit). */
static void pred16_list(const struct aeg_encoder *e, int mbx, int mby,
                        int list, int ref, int *pmvx, int *pmvy)
{
    int16_t cur[16][2];
    int8_t cur_ref[16];
    uint8_t avail[16];
    memset(cur, 0, sizeof(cur));
    memset(cur_ref, 0, sizeof(cur_ref));
    memset(avail, 0, sizeof(avail));
    /* (reads the neighbours only) */
    predict_mv_at_ref((struct aeg_encoder *)e, mbx, mby,
                      (const int16_t (*)[2])cur, cur_ref, avail,
                      AEG_MB_P16x16, 0, 0, 0, 4, list, ref, pmvx, pmvy);
}

static void pred16_ref(struct aeg_encoder *e, int mbx, int mby, int ref,
                       int *pmvx, int *pmvy)
{
    pred16_list(e, mbx, mby, 0, ref, pmvx, pmvy);
}

/* Motion vector of a P_Skip macroblock (H.264 8.4.1.1).  It is *not* the
 * median predictor: as soon as the left or the top neighbour is missing, is
 * intra, or sits at a zero vector, the skipped block's vector is (0,0).
 * Decoders reconstruct skipped macroblocks with exactly this rule, so the
 * encoder has to apply it before it may code one as a skip. */
/* A vector inside the bounds of the motion search (+-range samples), which
 * the reference borders are sized for. */
static int mv_me_ok(const struct aeg_encoder *e, int mvx, int mvy)
{
    int r = 4 * (e->p.search_range ? e->p.search_range : 16);
    return mvx >= -r && mvx <= r && mvy >= -r && mvy <= r;
}

static void pskip_mv(const struct aeg_encoder *e, int mbx, int mby,
                     int *mvx, int *mvy)
{
    nb_mv A, B, C;
    int match;

    *mvx = *mvy = 0;
    if (mbx <= 0 || !nb_top_ok(e, mby))
        return;
    /* only a neighbour on reference 0 with a zero vector forces (0,0): one
     * that sits still on the older picture is an ordinary vector (in an
     * MBAFF frame, after the frame/field scaling) */
    nb_mv_at(e, mbx, mby, -1, 0, 0, &A);                        /* left */
    if (A.ref == NB_OK && A.ref_idx == 0 && !A.mvx && !A.mvy)
        return;
    nb_mv_at(e, mbx, mby, 0, -1, 0, &B);                        /* above */
    if (B.ref == NB_OK && B.ref_idx == 0 && !B.mvx && !B.mvy)
        return;
    nb_mv_at(e, mbx, mby, 16, -1, 0, &C);                       /* above-right */
    if (C.ref == NB_UNAVAIL)
        nb_mv_at(e, mbx, mby, -1, -1, 0, &C);                   /* ... above-left */
    match = (C.ref_idx == 0) + (B.ref_idx == 0) + (A.ref_idx == 0);
    if (match == 1) {
        if (A.ref_idx == 0)      { *mvx = A.mvx; *mvy = A.mvy; }
        else if (B.ref_idx == 0) { *mvx = B.mvx; *mvy = B.mvy; }
        else                     { *mvx = C.mvx; *mvy = C.mvy; }
    } else {
        *mvx = mv_median3(A.mvx, B.mvx, C.mvx);
        *mvy = mv_median3(A.mvy, B.mvy, C.mvy);
    }
}

/* MBAFF: the mb_field_decoding_flag a decoder infers for pair (mbx, mby >>
 * 1) when it has not read one (7.4.4): the left pair's, else the above
 * pair's (in the slice), else frame. */
static int mbaff_infer(const struct aeg_encoder *e, int mbx, int mby)
{
    int top = mby & ~1;

    if (mbx > 0)
        return e->mbs[top * e->mb_w + mbx - 1].fld;
    if (top - 2 >= e->top_row)
        return e->mbs[(top - 2) * e->mb_w + mbx].fld;
    return 0;
}

/* P slice: is decided macroblock `t` a P_Skip -- P_L0_16x16 on reference 0
 * without residual, at the skip vector?  (e->cfld is the macroblock's.)  In
 * an MBAFF frame a pair whose two macroblocks are skipped takes the inferred
 * frame/field kind, so a bottom macroblock under a skipped top one only is
 * one when its pair is of that kind; otherwise it is sent as P_L0_16x16 --
 * same prediction, a few bits more.  `top_skip`: the top one was. */
static int p_skippable(const struct aeg_encoder *e, int mbx, int mby,
                       const mbenc *t, int top_skip)
{
    int pmvx, pmvy;

    if (t->type != AEG_MB_P16x16 || t->cbp != 0 || t->ref[0] != 0)
        return 0;
    pskip_mv(e, mbx, mby, &pmvx, &pmvy);
    if (t->mv[0][0] != pmvx || t->mv[0][1] != pmvy)
        return 0;
    if (e->mbaff && (mby & 1) && top_skip &&
        e->mbs[mby * e->mb_w + mbx].fld != mbaff_infer(e, mbx, mby))
        return 0;
    return 1;
}

/* B slice: is decided macroblock `t` a B_Skip -- B_Direct_16x16 without
 * residual?  With the same MBAFF rule as p_skippable: a bottom macroblock
 * under a skipped top one only is one when its pair is of the inferred kind
 * (the top one's direct vectors were derived with the pair's). */
static int b_skippable(const struct aeg_encoder *e, int mbx, int mby,
                       const mbenc *t, int top_skip)
{
    if (t->type != AEG_MB_B_DIRECT || t->cbp != 0)
        return 0;
    if (e->mbaff && (mby & 1) && top_skip &&
        e->mbs[mby * e->mb_w + mbx].fld != mbaff_infer(e, mbx, mby))
        return 0;
    return 1;
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

    /* In an MBAFF frame the neighbours are located by Table 6-4 and brought
     * to the macroblock's kind, and the index is the macroblock's (b_ref: 0
     * but in a field macroblock that chose the other parity). */
    if (e->mbaff || e->bfld_on) {
        pred16_list(e, mbx, mby, list, e->b_ref[list], mvx, mvy);
        return;
    }
    if (mbx > 0) {
        m = &e->mbs[mby * e->mb_w + mbx - 1];
        if (!is_intra_type(m->type) &&
            (aeg_mb_block_lists(m, 3) & (1 << list))) {
            a_av = 1;
            ax = list ? m->mv1[3][0] : m->mv[3][0];
            ay = list ? m->mv1[3][1] : m->mv[3][1];
        }
    }
    if (nb_top_ok(e, mby)) {
        m = &e->mbs[(mby - 1) * e->mb_w + mbx];
        if (!is_intra_type(m->type) &&
            (aeg_mb_block_lists(m, 12) & (1 << list))) {
            b_av = 1;
            bx = list ? m->mv1[12][0] : m->mv[12][0];
            by = list ? m->mv1[12][1] : m->mv[12][1];
        }
    }
    if (nb_top_ok(e, mby) && c_right_ok(e, mbx, mby)) {
        m = &e->mbs[(mby - 1) * e->mb_w + mbx + 1];
        if (!is_intra_type(m->type) &&
            (aeg_mb_block_lists(m, 12) & (1 << list))) {
            c_av = 1;
            cx = list ? m->mv1[12][0] : m->mv[12][0];
            cy = list ? m->mv1[12][1] : m->mv[12][1];
        }
    } else if (nb_top_ok(e, mby) && mbx > 0) {
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
    /* spatial direct takes each list's index from the neighbours, never the
     * co-located block's reference, so every group qualifies -- unless an
     * MBAFF B field macroblock chose other indices (bfld_prep) */
    if (e->bdr_no)
        return 0;
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
        int td = e->col_poc - e->col_ref_poc[col->ref_idx[grp4[g]] % AEG_MAX_REFS];
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
 * used lists on reference 0 are zero for that group.  (In an MBAFF field
 * macroblock the derived index may be 1: b_direct_refs.)
 *
 * Returns the list mask, the same for all four groups. */
/* The co-located macroblock of 8x8 group g of macroblock (mbx, mby) and the
 * 4x4 block of it that colZeroFlag reads (8.4.1.2.1, direct_8x8_inference:
 * the group's corner).  In an MBAFF frame (Table 8-6) a macroblock whose
 * co-located pair is of the other kind reads:
 *   a frame macroblock, field pair: the field nearer in POC to the current
 *     frame's (the smaller of its fields') -- with field_coding 3 the two
 *     fields of a frame have the same count (no delta_pic_order_cnt_bottom)
 *     and the tie goes to the bottom one; with 4 they are 2n and 2n + 1,
 *     so of the later co-located frame the field coded first, the top one
 *     (bff: the bottom one) -- at 4x4 rows 2 bot + y8;
 *   a field macroblock, frame pair: the top macroblock for group row 0
 *     (4x4 row 0), the bottom one for group row 1 (4x4 row 2).
 * (FFmpeg pred_spatial_direct_motion: col_parity, b8_stride / b4_stride.) */
static const aeg_mb *col_block(const struct aeg_encoder *e, int mbx, int mby,
                               int g, int *blk)
{
    static const int grp4[4] = { 0, 3, 12, 15 };
    const int x8 = g & 1, y8 = g >> 1, top = mby & ~1;

    if (!e->mbaff || e->col_mbs[top * e->mb_w + mbx].fld == e->cfld) {
        *blk = grp4[g];
        return &e->col_mbs[mby * e->mb_w + mbx];
    }
    if (!e->cfld) {
        const int par = e->p.field_coding == 4 ? e->p.interlaced == 2 : 1;
        *blk = (2 * (mby & 1) + y8) * 4 + 3 * x8;
        return &e->col_mbs[(top + par) * e->mb_w + mbx];
    }
    *blk = 8 * y8 + 3 * x8;
    return &e->col_mbs[(top + y8) * e->mb_w + mbx];
}

/* MBAFF: the reference indices spatial direct derives for macroblock (mbx,
 * mby) of the current kind (8.4.1.2.2): per list the smallest of the
 * neighbours' A, B and C (else D), brought to the macroblock's kind; -1
 * when none predicts from the list.  A frame macroblock's are 0 or -1 (a
 * field neighbour's 1 halves to 0). */
static void b_direct_refs(const struct aeg_encoder *e, int mbx, int mby,
                          int ref[2])
{
    int list;
    for (list = 0; list < 2; list++) {
        nb_mv n[3];
        int i;
        nb_mv_at(e, mbx, mby, -1, 0, list, &n[0]);
        nb_mv_at(e, mbx, mby, 0, -1, list, &n[1]);
        nb_mv_at(e, mbx, mby, 16, -1, list, &n[2]);
        if (n[2].ref == NB_UNAVAIL)
            nb_mv_at(e, mbx, mby, -1, -1, list, &n[2]);
        ref[list] = -1;
        for (i = 0; i < 3; i++)
            if (n[i].ref == NB_OK && (ref[list] < 0 || n[i].ref_idx < ref[list]))
                ref[list] = n[i].ref_idx;
    }
}

static int b_direct_spatial(const struct aeg_encoder *e, int mbx, int mby,
                            int mv0[16][2], int mv1[16][2])
{
    int used[2], pmv[2][2], list, g, k, lists, dref[2] = { 0, 0 };

    memset(mv0, 0, 32 * sizeof(mv0[0][0]));
    memset(mv1, 0, 32 * sizeof(mv1[0][0]));
    for (list = 0; list < 2; list++) {
        const aeg_mb *m;
        int u = 0;
        if (e->mbaff || e->bfp_on) {
            /* A, B and C (else D) located by Table 6-4; the index is the
             * smallest of theirs after the frame/field scaling (1 only for
             * a field macroblock whose neighbours all are on the other
             * parity, see bfld_prep) */
            int r[2];
            b_direct_refs(e, mbx, mby, r);
            used[list] = r[list] >= 0;
            dref[list] = used[list] ? r[list] : 0;
            pmv[list][0] = pmv[list][1] = 0;
            if (used[list])
                pred16_list(e, mbx, mby, list, dref[list], &pmv[list][0],
                            &pmv[list][1]);
            continue;
        }
        if (mbx > 0) {
            m = &e->mbs[mby * e->mb_w + mbx - 1];
            u |= !is_intra_type(m->type) &&
                 (aeg_mb_block_lists(m, 3) & (1 << list));
        }
        if (nb_top_ok(e, mby)) {
            m = &e->mbs[(mby - 1) * e->mb_w + mbx];
            u |= !is_intra_type(m->type) &&
                 (aeg_mb_block_lists(m, 12) & (1 << list));
            if (c_right_ok(e, mbx, mby)) {
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
        /* the co-located block's list 0 vector if it has one (and then only
         * on reference 0), else its list 1 vector -- the co-located picture
         * is a reference B picture of a pyramid for some B pictures */
        int cb;
        const aeg_mb *col = e->col_ready ? col_block(e, mbx, mby, g, &cb) : NULL;
        if (col && !is_intra_type(col->type)) {
            int lm = aeg_mb_block_lists(col, cb);
            const int16_t *v = NULL;
            if (lm & 1) {
                if (col->ref_idx[cb] == 0)
                    v = col->mv[cb];
            } else if (lm & 2) {
                if (col->ref_idx1[cb] == 0)
                    v = col->mv1[cb];
            }
            if (v)
                z = v[0] >= -1 && v[0] <= 1 && v[1] >= -1 && v[1] <= 1;
        }
        for (k = 0; k < 4; k++) {
            int b = ((y8 << 1) + (k >> 1)) * 4 + (x8 << 1) + (k & 1);
            /* (colZeroFlag zeroes the lists on index 0 only) */
            if (used[0] && !(z && !dref[0])) {
                mv0[b][0] = pmv[0][0];
                mv0[b][1] = pmv[0][1];
            }
            if (used[1] && !(z && !dref[1])) {
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

/* MBAFF: are the direct vectors inside the reach of the reference borders?
 * One derived from a neighbour of the other kind can be twice the search
 * range vertically (see mv_me_ok); such a macroblock is not offered as
 * direct. */
static int b_direct_reach(const struct aeg_encoder *e, int lists,
                          const int (*mv0)[2], const int (*mv1)[2])
{
    int b;

    if (!e->mbaff)
        return 1;
    for (b = 0; b < 16; b++)
        if (((lists & 1) && !mv_me_ok(e, mv0[b][0], mv0[b][1])) ||
            ((lists & 2) && !mv_me_ok(e, mv1[b][0], mv1[b][1])))
            return 0;
    return 1;
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
    if ((cm == 2 || cm == 3) && !nb_top_ok(e, mby))
        return 0;
    return 1;
}

static int luma_mode_ok(const struct aeg_encoder *e, int mode, int mbx, int mby)
{
    if ((mode == AEG_MB_I_V || mode == AEG_MB_I_P) && !nb_top_ok(e, mby))
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
    const pixel *ref = e->refpl[list][0];
    const aeg_mc_ctx *mc = &e->mcr[list];
    pixel buf[16 * 16];
    int ix = mvx >> 2, iy = mvy >> 2;
    const pixel *r = ref + (mby * 16 + geo[1] + iy) * e->stride[0] +
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
    pixel a[16 * 16], b[16 * 16];
    int x, y;
    {
        const pixel *rp = e->refpl[0][0] + (mby * 16 + geo[1] + (mv0y >> 2)) *
                                e->stride[0] + mbx * 16 + geo[0] + (mv0x >> 2);
        const pixel *rq = e->refpl[1][0] + (mby * 16 + geo[1] + (mv1y >> 2)) *
                                e->stride[0] + mbx * 16 + geo[0] + (mv1x >> 2);
        if (e->mcr[0].ready)
            aeg_mc_luma_p(a, geo[2], &e->mcr[0], rp, geo[2], geo[3],
                          mv0x & 3, mv0y & 3);
        else
            aeg_mc_luma(a, geo[2], rp, e->stride[0], geo[2], geo[3],
                        mv0x & 3, mv0y & 3);
        if (e->mcr[1].ready)
            aeg_mc_luma_p(b, geo[2], &e->mcr[1], rq, geo[2], geo[3],
                          mv1x & 3, mv1y & 3);
        else
            aeg_mc_luma(b, geo[2], rq, e->stride[0], geo[2], geo[3],
                        mv1x & 3, mv1y & 3);
    }
    (void)x; (void)y;
    return aeg_satd_avg(geo[2], geo[3],
                        e->srcp[0] + (mby * 16 + geo[1]) * e->stride[0] +
                            mbx * 16 + geo[0], e->stride[0], a, b, geo[2]);
}

static int part_satd_ref(struct aeg_encoder *e, int mbx, int mby, const int geo[4],
                     int mvx, int mvy, int ref)
{
    pixel buf[16 * 16];
    int ix = mvx >> 2, iy = mvy >> 2;
    const pixel *rp = e->refpl[ref][0] + (mby * 16 + geo[1] + iy) * e->stride[0] +
                        mbx * 16 + geo[0] + ix;
    const aeg_mc_ctx *mc = &e->mcr[ref];
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

/* Extra integer start points for the 16x16 search: the vectors of the left,
 * top and top-right neighbours on list 0 / reference 0 (the median predictor
 * is already a start point, but on a motion boundary the median is none of
 * them).  AEG_MENB=0 turns them off. */
static int me_neighbour_cands(const struct aeg_encoder *e, int mbx, int mby,
                              int list, int nb[8][2])
{
    static _Atomic int on = -1, lrnb = -1;
    const aeg_mb *m;
    int n = 0;

    if (on < 0) {
        const char *ev = getenv("AEG_MENB");
        on = ev ? atoi(ev) != 0 : 1;
    }
    if (!on)
        return 0;
#define NB_ADD(M, B) do {                                                   \
        if (!is_intra_type((M)->type) &&                                   \
            (list ? (M)->ref_idx1[B] : (M)->ref_idx[B]) == e->b_ref[list]) { \
            const int16_t *v_ = list ? (M)->mv1[B] : (M)->mv[B];            \
            nb[n][0] = v_[0]; nb[n][1] = v_[1]; n++;                        \
        }                                                                   \
    } while (0)
    if (mbx > 0) {
        m = &e->mbs[mby * e->mb_w + mbx - 1];
        NB_ADD(m, 3);
    }
    if (nb_top_ok(e, mby)) {
        m = &e->mbs[(mby - 1) * e->mb_w + mbx];
        NB_ADD(m, 12);
        if (c_right_ok(e, mbx, mby)) {
            /* (an MBAFF bottom field macroblock's is the bottom macroblock
             * of the pair above-right: the raster one is the next pair's
             * top macroblock, not decided yet) */
            m = e->mbaff && (mby & 1) ? nb_mbB(e, mbx + 1, mby)
                                      : &e->mbs[(mby - 1) * e->mb_w + mbx + 1];
            NB_ADD(m, 12);
        }
    }
#undef NB_ADD
    /* the half-resolution pre-search's vectors of the macroblocks not coded
     * yet (right, below): the spatial neighbours cannot offer those */
    if (lrnb < 0) {
        const char *ev = getenv("AEG_LRNB");
        lrnb = ev ? atoi(ev) : 1;
    }
    if (lrnb && e->lr_valid) {
        static const int8_t d[3][2] = { { 1, 0 }, { 0, 1 }, { -1, 1 } };
        int k;
        for (k = 0; k < (lrnb > 1 ? 3 : 2); k++) {
            int x = mbx + d[k][0], y = mby + d[k][1];
            if (x < 0 || x >= e->mb_w || y >= e->mb_h)
                continue;
            nb[n][0] = (int16_t)(e->lr_mv[list][(y * e->mb_w + x) * 2] * 4);
            nb[n][1] = (int16_t)(e->lr_mv[list][(y * e->mb_w + x) * 2 + 1] * 4);
            n++;
        }
    }
    return n;
}

static int env_flag(const char *name, int dflt)
{
    const char *ev = getenv(name);
    return ev ? atoi(ev) != 0 : dflt;
}
static int me_bnb_on(void)
{
    static _Atomic int v = -1;
    if (v < 0) v = env_flag("AEG_MEBNB", 1);
    return v;
}
static int env_flag_cached_bscale(void)
{
    static _Atomic int v = -1;
    if (v < 0) v = env_flag("AEG_MEBSCALE", 1);
    return v;
}
static int me_part_on(void)
{
    static _Atomic int v = -1;
    if (v < 0) v = env_flag("AEG_MEPART", 1);
    return v;
}

/* Motion search for one partition: its own search window, centred on its own
 * MV predictor.  `refine` selects the quarter-pel stage: the P_8x8 shape trials
 * run integer only (they compare shapes, not vectors) and the winning shape is
 * searched again with the refinement, which is where the quality comes from. */
static int ref1_cands_on(void)
{
    static _Atomic int v = -1;
    if (v < 0) {
        const char *ev = getenv("AEG_REF1C");
        v = ev ? atoi(ev) != 0 : 1;
    }
    return v;
}

/* x264's 4-point sub-pel diamond (AEG_ME_DIA) per search class, a mask:
 * 1 = B 16x8 / 8x16 / 8x8, 2 = P partitions, 4 = B 16x16, 8 = P 16x16 (every
 * reference).  BD-rate against the 8-point square, main + validation:
 *
 *            1         3         15
 *   p1     +0.04 %   +0.33 %   +0.58 %  (-7 % time)
 *   p2     +0.04 %             +0.60 %
 *   p3     +0.06 %             +0.27 %
 *   p4     +0.04 %
 *
 * The B partitions' searches take it for about -2 % of the time at presets
 * 1 and 2 (vtestL: B analysis 430 -> 389 Mcyc); the rest costs more than the
 * 0.1 % a prune may.  AEG_SPDIA overrides. */
static int spdia_mask(const struct aeg_encoder *e)
{
    static _Atomic int v = -2;
    if (v == -2) {
        const char *ev = getenv("AEG_SPDIA");
        v = ev ? atoi(ev) : -1;
    }
    return v >= 0 ? v : e->p.preset <= 2 ? 1 : 0;
}
#define SUBME_F(e, cls) ((e)->p.subme | (spdia_mask(e) & (cls) ? AEG_ME_DIA : 0))

/* A partition's sub-pel stage seeded with the macroblock's own quarter-pel
 * vector (aeg_me_block_fs()): when the partition's integer search lands on
 * the pel that vector rounds to -- 92 % of the B partitions of static content,
 * and the exact quarter-pel position in 86 % -- the half-pel pass is skipped
 * and the quarter-pel one starts from it.  BD-rate, main + validation:
 *
 *            B (AEG_BPFS)   P (AEG_PPFS)   both
 *   p1       +0.02 %        +0.09 %        +0.12 %
 *   p2       +0.02 %                       +0.11 %
 *   p3                                     +0.04 %
 *   p4                                     +0.01 %
 *
 * The B one is about -2.7 % of the time at preset 1 (vtestL: B analysis
 * 367 -> 321 Mcyc) and on by default at presets 1 and 2; the P one is -1.4 %
 * for +0.09 %, and at presets 3 and 4 neither shows in the time. */
static int bpfs_on(const struct aeg_encoder *e)
{
    static _Atomic int v = -2;
    if (v == -2) {
        const char *ev = getenv("AEG_BPFS");
        v = ev ? atoi(ev) != 0 : -1;
    }
    return v >= 0 ? v : e->p.preset <= 2;
}
static int ppfs_on(void)
{
    static _Atomic int v = -1;
    if (v < 0) {
        const char *ev = getenv("AEG_PPFS");
        v = ev ? atoi(ev) != 0 : 0;
    }
    return v;
}

static void me_partition_ref(struct aeg_encoder *e, int mbx, int mby, int type,
                         int p, const int geo[4], int pmvx, int pmvy, int ref,
                         int *mvx, int *mvy, int *sad, int refine)
{
    const pixel *rp = e->refpl[ref][0] + (mby * 16 + geo[1]) * e->stride[0] +
                        mbx * 16 + geo[0];
    const aeg_mc_ctx *mc = &e->mcr[ref];
    /* the whole-macroblock vector is where a partition usually ends up near */
    int whole[1][2] = { { e->mb_mv[0], e->mb_mv[1] } };
    int nwhole = (!ref && e->mb_inter_ok && me_part_on()) ? 1 : 0;
    int seed[2], *sp = NULL;
    /* The second reference: the first one's whole-macroblock vector carried
     * on to the older picture at constant velocity, and the half-resolution
     * pre-search's vector against it. */
    if (ref && e->mb_inter_ok && ref1_cands_on()) {
        int d0 = e->cur_poc - e->ref_poc[0], d1 = e->cur_poc - e->ref_poc[ref];
        if (d0 > 0) {
            whole[0][0] = e->mb_mv[0] * d1 / d0;
            whole[0][1] = e->mb_mv[1] * d1 / d0;
            nwhole = 1;
        }
    }
    if (ref == 1 && e->lr_valid > 1 && ref1_cands_on()) {
        seed[0] = e->lr_mv[1][(mby * e->mb_w + mbx) * 2];
        seed[1] = e->lr_mv[1][(mby * e->mb_w + mbx) * 2 + 1];
        sp = seed;
    }
    const int fs16[2] = { e->mb_mv[0], e->mb_mv[1] };
    const int use_fs = !ref && e->mb_inter_ok && !(geo[2] == 16 && geo[3] == 16) &&
                       ppfs_on();
    e->me_pruned =
        aeg_me_block_fs(rp,
                 e->stride[0],
                 e->srcp[0] + (mby * 16 + geo[1]) * e->stride[0] +
                     mbx * 16 + geo[0],
                 e->stride[0], geo[2], geo[3],
                 e->p.search_range ? e->p.search_range : 16,
                 refine ? SUBME_F(e, geo[2] == 16 && geo[3] == 16 ? 8 : 2) : 0,
                 pmvx, pmvy, sp,
                 (const int (*)[2])whole, nwhole, e->me_lambda, mc,
                 mvx, mvy, sad, e->me_hpt_on ? &e->me_hpt : NULL,
                 use_fs ? fs16 : NULL) < 0;
    /* Sub-pixel alignment: the integer search leaves the vector on an integer
     * pel.  Rather than repeating the whole quarter-pel search for a small
     * partition, apply the fractional offset the macroblock's 16x16 search
     * found -- objects move as a whole, so within one macroblock the offset is
     * usually the same -- and keep it when it improves the match. */
    e->me_last_satd = -1;
    if (!refine && e->p.subme > 0) {
        int fx = e->mb_frac[0], fy = e->mb_frac[1];
        int qx = (*mvx & ~3) + fx, qy = (*mvy & ~3) + fy;
        if (fx || fy) {
            int c_int = part_satd_ref(e, mbx, mby, geo, *mvx, *mvy, ref);
            int c_frac = part_satd_ref(e, mbx, mby, geo, qx, qy, ref);
            e->me_last_satd = c_int;
            if (c_frac < c_int) {
                *mvx = qx;
                *mvy = qy;
                *sad = c_frac;
                e->me_last_satd = c_frac;
            }
        }
    }
    (void)type; (void)p;
}
static void me_partition(struct aeg_encoder *e, int mbx, int mby, int type,
                         int p, const int geo[4], int pmvx, int pmvy,
                         int *mvx, int *mvy, int *sad, int refine)
{
    /* (me_partition_ref() already applies the 16x16 search's fractional
     * offset where it improves the match: a second pass over the same vector
     * could only compare it with itself) */
    me_partition_ref(e, mbx, mby, type, p, geo, pmvx, pmvy, 0, mvx, mvy, sad, refine);
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
                int cur_r = l ? (int)c->ref1[b0]
                              : (e->cur_l0_active > 1 || e->cur_is_b) ? (int)c->ref[b0] : 0;
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
                        cur_ref[l][b] = (int8_t)(l ? c->ref1[b0] : c->ref[b0]);
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
 * the time each.  From preset 2 up; AEG_PSUB overrides.  Preset 1 too
 * since the search range went to 48: there -1.33 % on the main and
 * validation clips, -2.00 % with T8RD, for +2 % (cockL) / +11 % (vtestL). */
static int part_subpel(const struct aeg_encoder *e)
{
    static _Atomic int v = -2;
    if (v == -2) {
        const char *ev = getenv("AEG_PSUB");
        v = ev ? atoi(ev) : -1;
    }
    if (v >= 0)
        return v;
    return e->p.preset >= 1 ? 2 : 0;
}

static int bidir_refine_geo(struct aeg_encoder *e, int mbx, int mby,
                            const int geo[4], const int pmv[2][2],
                            int mv[2][2]);

static int brdg_g(const struct aeg_encoder *e);

/* The same for the B pictures' 16x8 / 8x16 searches (AEG_PSUBB overrides). */
static int part_subpel_b(const struct aeg_encoder *e)
{
    static _Atomic int v = -2;
    if (v == -2) {
        const char *ev = getenv("AEG_PSUBB");
        v = ev ? atoi(ev) : -1;
    }
    return v >= 0 ? v : part_subpel(e);
}

static void cand_enc(const struct aeg_encoder *e, int cand, int qp, mbenc *t);

/* A field macroblock of an MBAFF B picture: every block of an inter
 * candidate on the indices bfld_prep chose (of both lists: the one a block
 * does not predict from is never read). */
static void bfld_stamp(const struct aeg_encoder *e, mbenc *t)
{
    if (e->bfld_on && is_inter_type(t->type)) {
        memset(t->ref, e->b_ref[0], sizeof(t->ref));
        memset(t->ref1, e->b_ref[1], sizeof(t->ref1));
    }
}

static int bidir_part_on(const struct aeg_encoder *e)
{
    static _Atomic int v = -2;
    if (v == -2) {
        const char *ev = getenv("AEG_BIDIRP");
        v = ev ? atoi(ev) : -1;
    }
    return v >= 0 ? v : e->p.preset >= 3;
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
        const pixel *ref = e->refpl[l][0];
        const aeg_mc_ctx *mc = &e->mcr[l];
        int seed[2], sad;

        predict_mv_at(e, mbx, mby, cur[l], avail[l], type, p, geo[0] / 4,
                      geo[1] / 4, geo[2] / 4, l, &pmv[l][0], &pmv[l][1]);
        seed[0] = mv16[l][0] / 4;
        seed[1] = mv16[l][1] / 4;
        aeg_me_block_fs(ref + (mby * 16 + geo[1]) * e->stride[0] + mbx * 16 +
                          geo[0],
                      e->stride[0],
                      e->srcp[0] + (mby * 16 + geo[1]) * e->stride[0] +
                          mbx * 16 + geo[0],
                      e->stride[0], geo[2], geo[3], range,
                      part_subpel_b(e) >= 2 ? SUBME_F(e, 1) : 0,
                      pmv[l][0], pmv[l][1], seed, NULL, 0, e->me_lambda, mc,
                      &mv[l][0], &mv[l][1], &sad, NULL,
                      bpfs_on(e) ? mv16[l] : NULL);
        /* with the fractional stage the search returns the SATD itself */
        satd[l] = e->p.subme > 0 && part_subpel_b(e) >= 2 ? sad : -1;
        if (e->p.subme > 0 && part_subpel_b(e) < 2) {
            int fx = l ? e->mb_frac1[0] : e->mb_frac[0];
            int fy = l ? e->mb_frac1[1] : e->mb_frac[1];
            int qx = (mv[l][0] & ~3) + fx, qy = (mv[l][1] & ~3) + fy;

            if (fx || fy) {
                int c_int = b_part_satd(e, mbx, mby, geo, mv[l][0], mv[l][1], l);
                int c_frac = b_part_satd(e, mbx, mby, geo, qx, qy, l);

                satd[l] = c_int;
                if (c_frac < c_int) {
                    mv[l][0] = qx;
                    mv[l][1] = qy;
                    satd[l] = c_frac;
                }
            }
        }
        if (satd[l] < 0)
            satd[l] = b_part_satd(e, mbx, mby, geo, mv[l][0], mv[l][1], l);
        bits[l] = aeg_mvd_bits(mv[l][0] - pmv[l][0], mv[l][1] - pmv[l][1]);
    }
    {
        /* the bi-predictive pair, refined jointly (see bidir_refine) */
        int bi[2][2] = { { mv[0][0], mv[0][1] }, { mv[1][0], mv[1][1] } };
        if (bidir_part_on(e))
            bidir_refine_geo(e, mbx, mby, geo, (const int (*)[2])pmv, bi);
        satd[2] = b_part_satd_bi(e, mbx, mby, geo, bi[0][0], bi[0][1],
                                 bi[1][0], bi[1][1]);
        cost[0] = satd[0] + 8 * bits[0];
        cost[1] = satd[1] + 8 * bits[1];
        cost[2] = satd[2] + 8 * (aeg_mvd_bits(bi[0][0] - pmv[0][0], bi[0][1] - pmv[0][1]) +
                                 aeg_mvd_bits(bi[1][0] - pmv[1][0], bi[1][1] - pmv[1][1]));
        best = 0;
        if (cost[1] < cost[best]) best = 1;
        if (cost[2] < cost[best]) best = 2;
        if (best == 2) {
            mv[0][0] = bi[0][0]; mv[0][1] = bi[0][1];
            mv[1][0] = bi[1][0]; mv[1][1] = bi[1][1];
        }
    }
    lists = best == 0 ? 1 : (best == 1 ? 2 : 3);
    if (brdg_g(e) > 0) {
        int pb = best == 2
                 ? aeg_mvd_bits(mv[0][0] - pmv[0][0], mv[0][1] - pmv[0][1]) +
                   aeg_mvd_bits(mv[1][0] - pmv[1][0], mv[1][1] - pmv[1][1])
                 : bits[best];
        cd->scost += (p == 0) + satd[best] + e->me_lambda * pb;
    }
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
    int direct[16][2], direct1[16][2], dlists = 3, dreach;
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
    dreach = b_direct_reach(e, dlists, (const int (*)[2])direct,
                            (const int (*)[2])direct1);
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
            const pixel *ref = e->refpl[l][0];
            const aeg_mc_ctx *mc = &e->mcr[l];
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
                         e->stride[0], 8, 8, range, b8_sub ? SUBME_F(e, 1) : 0,
                         pmv[l][0], pmv[l][1],
                         seed, NULL, 0, e->me_lambda, mc, &mv[l][0],
                         &mv[l][1], &sad);
            satd[l] = e->p.subme > 0 && b8_sub ? sad : -1;
            if (e->p.subme > 0 && !b8_sub) {
                int fx = l ? e->mb_frac1[0] : e->mb_frac[0];
                int fy = l ? e->mb_frac1[1] : e->mb_frac[1];
                int qx = (mv[l][0] & ~3) + fx, qy = (mv[l][1] & ~3) + fy;

                if (fx || fy) {
                    int c_frac = b_part_satd(e, mbx, mby, geo, qx, qy, l);
                    int c_int = b_part_satd(e, mbx, mby, geo, mv[l][0], mv[l][1], l);
                    satd[l] = c_int;
                    if (c_frac < c_int) {
                        mv[l][0] = qx;
                        mv[l][1] = qy;
                        satd[l] = c_frac;
                    }
                }
            }
            if (satd[l] < 0)
                satd[l] = b_part_satd(e, mbx, mby, geo, mv[l][0], mv[l][1], l);
            bits[l] = aeg_mvd_bits(mv[l][0] - pmv[l][0],
                                   mv[l][1] - pmv[l][1]);
        }
        satd[2] = b_part_satd_bi(e, mbx, mby, geo, mv[0][0], mv[0][1],
                                 mv[1][0], mv[1][1]);
        /* direct: the co-located vectors, both lists, no difference to send */
        satd[3] = !(e->col_ready && dreach && direct_group_ok(e, mbx, mby, g))
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
 * clips at preset 3, for more time (docs/STATUS.md, *Two P references*).
 * With three or more references it is the other way round: the 16x8 and
 * 8x16 halves and each 8x8 of P_8x8 (me_candidate_p8x8_mref) search the
 * references themselves, -0.5 % more than the 16x16 candidate alone at
 * --ref 4 (docs/STATUS.md, *More than two P references*). */
static int pref_mode(void);

static int part_ref_search(const struct aeg_encoder *e)
{
    return pref_mode() && e->cur_l0_active > 2;
}

/* The highest list0 index the left, top, top-left and top-right
 * macroblocks use (0 when none of them is an inter macroblock). */
static int nb_maxref(const struct aeg_encoder *e, int mbx, int mby)
{
    int m = 0, d, b;
    static const int dx[4] = { -1, 0, -1, 1 }, dy[4] = { 0, -1, -1, -1 };
    for (d = 0; d < 4; d++) {
        int x = mbx + dx[d], y = mby + dy[d];
        const aeg_mb *n;
        if (x < 0 || x >= e->mb_w || (dy[d] && y < e->top_row))
            continue;
        /* (in an MBAFF frame the bottom macroblock's top-right is the top
         * macroblock of the next pair, not decided yet) */
        if (e->mbaff && (mby & 1) && dx[d] > 0)
            continue;
        n = &e->mbs[y * e->mb_w + x];
        if (!is_inter_type(n->type))
            continue;
        for (b = 0; b < 16; b++)
            if (ref_kind(e, n, n->ref_idx[b]) > m)
                m = ref_kind(e, n, n->ref_idx[b]);
    }
    /* (in an MBAFF frame a neighbour's index, even scaled, can be past the
     * current macroblock's list) */
    return m < e->cur_l0_active ? m : e->cur_l0_active - 1;
}

/* The partitions' own reference search with three or more references:
 * AEG_PREF=0 off (reference 0 only, as with two), 1 every reference, 2 (the
 * default) x264's early termination. */
static int pref_mode(void)
{
    static _Atomic int env = -2;
    if (env == -2) {
        const char *ev = getenv("AEG_PREF");
        env = ev ? atoi(ev) : 2;
    }
    return env;
}

/* How many list0 entries the partitions search: all of them, or (x264's
 * early termination) when the whole macroblock is best on reference 0, only
 * as far as the neighbours reach (mb_maxref, set by analyse_mb). */
static int pref_nref(const struct aeg_encoder *e)
{
    return pref_mode() >= 2 ? e->mb_maxref + 1 : e->cur_l0_active;
}

static void ref_publish(int8_t *cur_ref, const int geo[4], int ref)
{
    int x, y, x0 = geo[0] / 4, y0 = geo[1] / 4;
    for (y = y0; y < y0 + geo[3] / 4; y++)
        for (x = x0; x < x0 + geo[2] / 4; x++)
            cur_ref[(y << 2) + x] = (int8_t)ref;
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
            for (ref = 0; ref < pref_nref(e); ref++) {
                int pmvx, pmvy, mvx, mvy, sad, satd, bits, cost;
                predict_mv_at_ref(e, mbx, mby, (const int16_t (*)[2])cur, cur_ref, avail, type, p,
                                  geo[0] / 4, geo[1] / 4, geo[2] / 4, 0, ref, &pmvx, &pmvy);
                me_partition_ref(e, mbx, mby, type, p, geo, pmvx, pmvy, ref, &mvx, &mvy, &sad, part_subpel(e));
                satd = e->me_last_satd >= 0 ? e->me_last_satd
                                            : part_satd_ref(e, mbx, mby, geo, mvx, mvy, ref);
                bits = aeg_mvd_bits(mvx - pmvx, mvy - pmvy) + ref_te_bits(e, ref);
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

static void me_part_r(struct aeg_encoder *e, int mbx, int mby, int type, int p,
                      const int geo[4], int pmvx, int pmvy, int ref,
                      int *mvx, int *mvy, int *sad, int refine)
{
    if (ref == 0)
        me_partition(e, mbx, mby, type, p, geo, pmvx, pmvy, mvx, mvy, sad, refine);
    else
        me_partition_ref(e, mbx, mby, type, p, geo, pmvx, pmvy, ref, mvx, mvy,
                         sad, refine);
}

/* P_8x8 with three or more references (AEG_PREF): each 8x8 first picks its
 * reference -- the 8x8 searched on every list0 entry the partitions may use,
 * by SATD plus the vector and index bits -- and then its sub-shape on that
 * reference (x264's mixed references).  ref_idx is per 8x8 in the syntax, so
 * the sub-partitions share it. */
static void me_candidate_p8x8_mref(struct aeg_encoder *e, int mbx, int mby)
{
    int16_t cur[16][2];
    uint8_t avail[16];
    int8_t cur_ref[16];
    int sub, s, slot = e->mb_ncand, nref = pref_nref(e);
    mbcand *cand;

    if (slot >= (int)(sizeof(e->mb_cand) / sizeof(e->mb_cand[0])))
        return;
    memset(cur, 0, sizeof(cur));
    memset(avail, 0, sizeof(avail));
    memset(cur_ref, 0, sizeof(cur_ref));
    cand = &e->mb_cand[slot];
    cand->type = AEG_MB_P8x8;
    cand->nparts = 0;
    memset(cand->sub_type, 0, sizeof(cand->sub_type));
    memset(cand->mv, 0, sizeof(cand->mv));
    memset(cand->pmv, 0, sizeof(cand->pmv));
    memset(cand->ref, 0, sizeof(cand->ref));

    for (sub = 0; sub < 4; sub++) {
        int bx = (sub & 1) * 2, by = (sub >> 1) * 2;
        int best_st = 0, best_cost = 1 << 30, k, sref = 0, ref;
        int best_mv[4][2] = {{0}}, best_geo[4][4];

        if (nref > 1) {
            int g8[4], b0, bc = 1 << 30;
            sub_geom(0, bx, by, 0, g8, &b0);
            for (ref = 0; ref < nref; ref++) {
                int px, py, mx, my, sd, c;
                predict_mv_at_ref(e, mbx, mby, (const int16_t (*)[2])cur, cur_ref,
                                  avail, AEG_MB_P8x8, 0, g8[0] / 4, g8[1] / 4,
                                  g8[2] / 4, 0, ref, &px, &py);
                me_part_r(e, mbx, mby, AEG_MB_P8x8, 0, g8, px, py, ref,
                          &mx, &my, &sd, 0);
                c = (e->me_last_satd >= 0 ? e->me_last_satd
                                          : part_satd_ref(e, mbx, mby, g8, mx, my, ref)) +
                    8 * (aeg_mvd_bits(mx - px, my - py) + ref_te_bits(e, ref));
                if (c < bc) {
                    bc = c;
                    sref = ref;
                }
            }
        }
        for (s = 0; s < 4; s++) {
            int16_t trial[16][2];
            uint8_t tavail[16];
            int8_t tref[16];
            int geo[4][4], b0, n = sub_cnt[s], satd_sum = 0, bits, mv[4][2];
            int traded = 0;
            if (s && !(e->p.partitions & 16))
                continue;
            memcpy(trial, cur, sizeof(trial));
            memcpy(tavail, avail, sizeof(tavail));
            memcpy(tref, cur_ref, sizeof(tref));
            bits = aeg_ue_bits((uint32_t)s);
            for (k = 0; k < n; k++) {
                int px, py, sad;
                sub_geom(s, bx, by, k, geo[k], &b0);
                predict_mv_at_ref(e, mbx, mby, (const int16_t (*)[2])trial, tref,
                                  tavail, AEG_MB_P8x8, 0, geo[k][0] / 4,
                                  geo[k][1] / 4, geo[k][2] / 4, 0, sref, &px, &py);
                me_part_r(e, mbx, mby, AEG_MB_P8x8, k, geo[k], px, py, sref,
                          &mv[k][0], &mv[k][1], &sad, 0);
                satd_sum += e->me_last_satd >= 0
                                ? e->me_last_satd
                                : part_satd_ref(e, mbx, mby, geo[k], mv[k][0], mv[k][1], sref);
                me_publish(trial, geo[k], mv[k][0], mv[k][1]);
                avail_publish(tavail, geo[k]);
                ref_publish(tref, geo[k], sref);
                bits += aeg_mvd_bits(mv[k][0] - px, mv[k][1] - py);
                if (satd_sum + 8 * bits >= best_cost) {
                    traded = 1;
                    break;
                }
            }
            if (traded)
                continue;
            if (satd_sum + 8 * bits < best_cost) {
                best_cost = satd_sum + 8 * bits;
                best_st = s;
                memcpy(best_geo, geo, sizeof(geo));
                memcpy(best_mv, mv, sizeof(mv));
            }
        }
        cand->sub_type[sub] = (uint8_t)best_st;
        for (k = 0; k < sub_cnt[best_st]; k++) {
            int b0, pmvx, pmvy, sad;
            sub_geom(best_st, bx, by, k, best_geo[k], &b0);
            predict_mv_at_ref(e, mbx, mby, (const int16_t (*)[2])cur, cur_ref, avail,
                              AEG_MB_P8x8, 0, best_geo[k][0] / 4,
                              best_geo[k][1] / 4, best_geo[k][2] / 4, 0, sref,
                              &pmvx, &pmvy);
            me_part_r(e, mbx, mby, AEG_MB_P8x8, k, best_geo[k], pmvx, pmvy, sref,
                      &best_mv[k][0], &best_mv[k][1], &sad,
                      best_geo[k][2] >= 8 && best_geo[k][3] >= 8);
            cand_publish(cand, best_geo[k], best_mv[k][0], best_mv[k][1]);
            cand_publish_ref(cand, best_geo[k], sref);
            cand->pmv[b0][0] = pmvx;
            cand->pmv[b0][1] = pmvy;
            me_publish(cur, best_geo[k], best_mv[k][0], best_mv[k][1]);
            avail_publish(avail, best_geo[k]);
            ref_publish(cur_ref, best_geo[k], sref);
            cand->nparts++;
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

    if (part_ref_search(e)) {
        me_candidate_p8x8_mref(e, mbx, mby);
        return;
    }
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
            /* The 8x4, 4x8 and 4x4 shapes only with --partitions bit 16:
             * leaving them out is -0.10 / -0.12 % BD-rate at preset 3 (the
             * two clip sets), +0.03 / -0.07 % at preset 4 and -0.3 / -1.0 %
             * at presets 2 / 1, for 2-11 % of the time -- the shape decision
             * by SATD picks them where the RD pass then does not want them,
             * or where it does and is wrong (x264's medium preset leaves
             * them out too). */
            if (s && !(e->p.partitions & 16))
                continue;
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
                        sd_satd = e->me_last_satd >= 0 ? e->me_last_satd
                                  : part_satd_ref(e, mbx, mby, geo[k], mvx, mvy, ref);
                        bts = aeg_mvd_bits(mvx - pmvx, mvy - pmvy) + ref_te_bits(e, ref);
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
                    satd_sum += e->me_last_satd >= 0
                                    ? e->me_last_satd
                                    : part_satd(e, mbx, mby, geo[k], mv[k][0], mv[k][1]);
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
                       e->refpl[0][0], e->stride[0], 16, 16,
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

static int lr_seed_on(const struct aeg_encoder *e)
{
    static _Atomic int v = -2;
    if (v == -2) {
        const char *ev = getenv("AEG_LRSEED");
        v = ev ? atoi(ev) : -1;
    }
    return v >= 0 ? v : e->p.preset >= 1;
}

/* The motion search walks a hexagon from the best of a few starting points
 * (the predictor, zero, the neighbours' vectors).  Motion larger than the
 * neighbourhood suggests -- an object that starts moving, a P picture four
 * frames from its reference -- leaves it in a local minimum, and the
 * macroblock goes intra (measured on cock at P distance 3: an exhaustive
 * +-32 search is -4.6% BD-rate).  x264 seeds its search with the vectors of
 * its half-resolution lookahead; this is the same: the lookahead's block
 * search (lookahead.c) on half-resolution copies of the current source and
 * of each list's reference reconstruction, whose half-sample vectors are
 * whole samples at full resolution. */
/* The vectors of block rows lr_rows..upto (both lists). */
static void lr_seed_rows(struct aeg_encoder *e, int upto)
{
    int l, i, n = e->mb_count;

    if (upto > e->mb_h - 1)
        upto = e->mb_h - 1;
    if (upto < e->lr_rows)
        return;
    for (l = 0; l < e->lr_nl; l++) {
        int *mv = e->lr_scratch + 2 * n * l;
        aeg_lowres_mv_rows(&e->lr_pic[0], e->lr_ref[l], mv, e->lr_rows, upto);
        for (i = 2 * e->lr_rows * e->mb_w; i < 2 * (upto + 1) * e->mb_w; i++)
            e->lr_mv[l][i] = (int16_t)mv[i];
    }
    e->lr_rows = upto + 1;
}

/* Sets the pre-search up; the vectors themselves are computed a row ahead of
 * the macroblock rows that read them (lr_seed_rows from code_slice), so they
 * only need the reference rows those macroblocks reach anyway.  The
 * reference's half-resolution copy is the one its own coding built
 * (slot_low) unless the list predicts from something else (the weighted copy
 * of weighted prediction). */
static void lr_seed_frame(struct aeg_encoder *e)
{
    int l;

    e->lr_valid = 0;
    e->lr_rows = 0;
    if (e->par_valid || !lr_seed_on(e) || e->p.subme <= 0 || !e->has_ref ||
        !(e->cur_is_p || e->cur_is_b) || (e->cur_is_b && !e->has_ref2))
        return;
    aeg_lowres_build(&e->lr_pic[0], e->srcp[0], e->stride[0], e->w, e->h);
    e->lr_nl = e->cur_is_b || e->cur_l0_active > 1 ? 2 : 1;
    for (l = 0; l < e->lr_nl; l++) {
        const pixel *ref = e->refpl[l][0];
        int s = e->ref_slot[l];
        if (s >= 0 && ref == e->pic_plane[s][0] && e->slot_low[s].mem &&
            !e->fld) {
            e->lr_ref[l] = &e->slot_low[s];
        } else if (s >= 0 && e->fld && e->slot_lfok[s] &&
                   ref == e->pic_plane[s][0] + e->ref_par[l] * e->fsv.stride[0]) {
            /* a reference field's own copy (the same rows as built here) */
            e->lr_ref[l] = &e->slot_lowf[s][e->ref_par[l]];
        } else {
            aeg_lowres_build(&e->lr_pic[1 + l], ref, e->stride[0], e->w, e->h);
            e->lr_ref[l] = &e->lr_pic[1 + l];
        }
    }
    /* several slices may be coded at once (slice threads): all rows now */
    if (e->n_slices > 1)
        lr_seed_rows(e, e->mb_h - 1);
    /* 2: list0[1] of a two-reference P picture has its vectors in lr_mv[1] */
    e->lr_valid = e->cur_is_p && e->cur_l0_active > 1 ? 2 : 1;
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
    if (e->par_full && e->d.ops.satd_batch && e->d.ops.ref_upload && e->mcr[0].ready) {
        e->d.ops.ref_upload(e->d.ops.priv, e->mcr[0].base, e->mcr[0].stride, e->w, e->h);
        me_par_subpel(e, range * 4);
        e->par_valid = 1;
    }

    t0 = aeg_now_ns() - t0;
    e->st.stage_ns[AEG_STAGE_MOTION_EST] += t0;
    stage_rec(e, AEG_STAGE_MOTION_EST,
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
static int bidir_on(const struct aeg_encoder *e)
{
    static _Atomic int v = -2;
    if (v == -2) {
        const char *ev = getenv("AEG_BIDIR");
        v = ev ? atoi(ev) : -1;
    }
    return v >= 0 ? v : e->p.preset >= 3;
}

/* The prediction of one partition from list `list` at a quarter-pel
 * vector, stored with stride geo[2] (b_part_satd_bi()'s, exactly). */
static void b_part_pred(struct aeg_encoder *e, int mbx, int mby,
                        const int geo[4], int mvx, int mvy, int list,
                        pixel *buf)
{
    const pixel *ref = e->refpl[list][0];
    const aeg_mc_ctx *mc = &e->mcr[list];
    const pixel *r = ref + (mby * 16 + geo[1] + (mvy >> 2)) * e->stride[0] +
                       mbx * 16 + geo[0] + (mvx >> 2);
    if (mc->ready)
        aeg_mc_luma_p(buf, geo[2], mc, r, geo[2], geo[3], mvx & 3, mvy & 3);
    else
        aeg_mc_luma(buf, geo[2], r, e->stride[0], geo[2], geo[3],
                    mvx & 3, mvy & 3);
}

/* Joint refinement of a bi-predicted pair: each list's vector in turn walks
 * its eight quarter-pel neighbours with the other list held, by the SATD of
 * the averaged prediction plus both mvds.  The held list's prediction and
 * mvd bits do not change while the other walks, so both are kept instead of
 * being recomputed per neighbour (the same costs, in the same order, as
 * b_part_satd_bi() per candidate). */
static int bidir_refine_geo(struct aeg_encoder *e, int mbx, int mby,
                            const int geo[4], const int pmv[2][2],
                            int mv[2][2])
{
    static const int8_t offs[8][2] = {
        {-1,0},{1,0},{0,-1},{0,1},{-1,-1},{1,-1},{-1,1},{1,1}
    };
    const int range4 = (e->p.search_range ? e->p.search_range : 16) * 4;
    const int lam = e->me_lambda;
    const pixel *src = e->srcp[0] + (mby * 16 + geo[1]) * e->stride[0] +
                         mbx * 16 + geo[0];
    pixel store[3][16 * 16];
    pixel *pred[2] = { store[0], store[1] }, *tmp = store[2];
    int bits[2];
    int best, it, moved = 1;

    b_part_pred(e, mbx, mby, geo, mv[0][0], mv[0][1], 0, pred[0]);
    b_part_pred(e, mbx, mby, geo, mv[1][0], mv[1][1], 1, pred[1]);
    bits[0] = aeg_mvd_bits(mv[0][0] - pmv[0][0], mv[0][1] - pmv[0][1]);
    bits[1] = aeg_mvd_bits(mv[1][0] - pmv[1][0], mv[1][1] - pmv[1][1]);
    best = aeg_satd_avg(geo[2], geo[3], src, e->stride[0], pred[0], pred[1],
                        geo[2]) +
           lam * (bits[0] + bits[1]);
    for (it = 0; it < 4 && moved; it++) {
        int l;
        moved = 0;
        for (l = 1; l >= 0; l--) {
            int k, bx = mv[l][0], by = mv[l][1];
            for (k = 0; k < 8; k++) {
                int tx = bx + offs[k][0], ty = by + offs[k][1], c, tb;
                if (tx < -range4 || tx > range4 || ty < -range4 || ty > range4)
                    continue;
                b_part_pred(e, mbx, mby, geo, tx, ty, l, tmp);
                tb = aeg_mvd_bits(tx - pmv[l][0], ty - pmv[l][1]);
                c = aeg_satd_avg(geo[2], geo[3], src, e->stride[0],
                                 l ? pred[0] : tmp, l ? tmp : pred[1],
                                 geo[2]) +
                    lam * (l ? bits[0] + tb : tb + bits[1]);
                if (c < best) {
                    pixel *sw = pred[l];
                    best = c;
                    mv[l][0] = tx;
                    mv[l][1] = ty;
                    bits[l] = tb;
                    pred[l] = tmp;
                    tmp = sw;
                    moved = 1;
                }
            }
        }
    }
    return best;
}

static int bidir_refine(struct aeg_encoder *e, int mbx, int mby,
                        const int pmv[2][2], int mv[2][2])
{
    static const int geo[4] = { 0, 0, 16, 16 };
    return bidir_refine_geo(e, mbx, mby, geo, pmv, mv);
}

/* x264's B early termination (mb_analyse_b_rd): only the candidates whose
 * search cost is within G/16 of the best one -- the direct prediction's SATD
 * included -- go on to the rate-distortion pass; direct itself always does.
 * Against the full pass on main + validation (CQP, BD-rate / encode time):
 *
 *            G = 20                    G = 17 (x264's)
 *   p1      +0.01 %   no change        -
 *   p2      +0.01 %   no change        +0.05 %   no change
 *   p3      -0.00 %   -2.8 %           +0.08 %   -5.6 %
 *   p4      -0.01 %   -5.2 %           +0.08 %   -4.2 %
 *
 * The cheaper presets have fewer candidates left to drop.  On static content
 * the candidates' SATD costs lie close together (vtest: 5.3 candidates, 0.3
 * dropped at G = 17), which is why this does little there.  G = 20 from
 * preset 3 up; AEG_BRDG=G overrides, 0 = off. */
static int brdg_g_env(void)
{
    static _Atomic int v = -2;
    if (v == -2) {
        const char *ev = getenv("AEG_BRDG");
        v = ev ? atoi(ev) : -1;
    }
    return v;
}
static int brdg_g(const struct aeg_encoder *e)
{
    return brdg_g_env() >= 0 ? brdg_g_env() : e->p.preset >= 3 ? 20 : 0;
}

static void b_rd_gate(struct aeg_encoder *e, int mbx, int mby)
{
    const int g = brdg_g(e);
    int k, n = 0, best = 1 << 30;
    int64_t thr;

    if (g <= 0)
        return;
    for (k = 0; k < e->mb_ncand; k++) {
        mbcand *cd = &e->mb_cand[k];
        if (cd->type == AEG_MB_B_DIRECT && !cd->scost) {
            const int s = e->stride[0];
            mbenc t;
            cand_enc(e, k, QP_MIN, &t);
            inter_predict_typed(e, mbx, mby, &t);
            cd->scost = 1 + e->me_lambda +
                        aeg_satd_16x16(e->pred_l, 16,
                                       e->srcp[0] + mby * 16 * s + mbx * 16, s);
        }
        if (cd->scost && cd->scost - 1 < best)
            best = cd->scost - 1;
    }
    thr = (int64_t)best * g / 16 + 1;
    for (k = 0; k < e->mb_ncand; k++) {
        const mbcand *cd = &e->mb_cand[k];
        if (cd->type == AEG_MB_B_DIRECT || !cd->scost || cd->scost - 1 < thr) {
            if (n != k)
                e->mb_cand[n] = *cd;
            n++;
        }
    }
    e->mb_ncand = n;
}

static int analyse_mb_b(struct aeg_encoder *e, int mbx, int mby)
{
    const pixel *src = e->srcp[0] + mby * 16 * e->stride[0] + mbx * 16;
    int pmv[2][2], mv[2][2], mvbi[2][2], sad[2], list, b, bicost;
    int range = e->p.search_range ? e->p.search_range : 16;

    b_mv_predict(e, mbx, mby, 0, &pmv[0][0], &pmv[0][1]);
    b_mv_predict(e, mbx, mby, 1, &pmv[1][0], &pmv[1][1]);
    for (list = 0; list < 2; list++) {
        const pixel *ref = e->refpl[list][0] +
                             mby * 16 * e->stride[0] + mbx * 16;
        const aeg_mc_ctx *mc = &e->mcr[list];
        int seed[2], nb[9][2], nnb = 0;

        if (me_bnb_on())
            nnb = me_neighbour_cands(e, mbx, mby, list, nb);
        /* list 1: the list 0 vector carried on through the current picture
         * at constant velocity (the sign flips: list 1 looks forward) */
        if (list && env_flag_cached_bscale()) {
            int d0 = e->cur_poc - e->ref_poc[0], d1 = e->cur_poc - e->ref_poc[1];
            if (d0) {
                nb[nnb][0] = mv[0][0] * d1 / d0;
                nb[nnb][1] = mv[0][1] * d1 / d0;
                nnb++;
            }
        }
        /* The list 0 search starts from the previous picture's vector for this
         * macroblock (always zero here: the whole-frame seed pass only runs for
         * P pictures); the list 1 search starts from the list 0 vector, which
         * is a far better starting point than zero on content that moves. */
        seed[0] = list ? mv[0][0] / 4 : (int)e->seed_mv[(mby * e->mb_w + mbx) * 2];
        seed[1] = list ? mv[0][1] / 4 : (int)e->seed_mv[(mby * e->mb_w + mbx) * 2 + 1];
        if (e->lr_valid) {
            seed[0] = e->lr_mv[list][(mby * e->mb_w + mbx) * 2];
            seed[1] = e->lr_mv[list][(mby * e->mb_w + mbx) * 2 + 1];
        }
        aeg_me_block_l(ref, e->stride[0], src, e->stride[0], 16, 16, range,
                     SUBME_F(e, 4), pmv[list][0], pmv[list][1], seed,
                     (const int (*)[2])nb, nnb, e->me_lambda, mc,
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

    /* The bi-predictive vectors: each list's own winner is the best
     * predictor *alone*, not the best half of an average.  Refine them
     * jointly (x264's bidir refinement): one list's vector held, the other
     * walks its quarter-pel neighbours on the SATD of the average plus the
     * rate of the vector that moves, alternating until neither moves. */
    {
        int bi[2][2] = { { mv[0][0], mv[0][1] }, { mv[1][0], mv[1][1] } };
        bicost = 0;
        if (bidir_on(e))
            bicost = 1 + bidir_refine(e, mbx, mby, pmv, bi);
        mvbi[0][0] = bi[0][0]; mvbi[0][1] = bi[0][1];
        mvbi[1][0] = bi[1][0]; mvbi[1][1] = bi[1][1];
    }
    /* list 0, list 1, and both */
    for (list = 0; list < 3; list++) {
        mbcand *cd = &e->mb_cand[e->mb_ncand];
        const int (*vv)[2] = list == 2 ? (const int (*)[2])mvbi
                                       : (const int (*)[2])mv;
        memset(cd, 0, sizeof(*cd));
        cd->type = list == 0 ? AEG_MB_B_L0 : (list == 1 ? AEG_MB_B_L1
                                                        : AEG_MB_B_BI);
        cd->nparts = 1;
        cd->blist[0] = (uint8_t)(list == 0 ? 1 : (list == 1 ? 2 : 3));
        for (b = 0; b < 16; b++) {
            cd->mv[b][0] = (int16_t)vv[0][0];
            cd->mv[b][1] = (int16_t)vv[0][1];
            cd->mv1[b][0] = (int16_t)vv[1][0];
            cd->mv1[b][1] = (int16_t)vv[1][1];
        }
        cd->pmv[0][0] = (int16_t)pmv[0][0];
        cd->pmv[0][1] = (int16_t)pmv[0][1];
        cd->pmv1[0][0] = (int16_t)pmv[1][0];
        cd->pmv1[0][1] = (int16_t)pmv[1][1];
        cd->scost = list == 2 ? bicost
                    : 1 + sad[list] + e->me_lambda *
                          aeg_mvd_bits(mv[list][0] - pmv[list][0],
                                       mv[list][1] - pmv[list][1]);
        e->mb_ncand++;
    }
    /* The partitioned shapes.  Like the P path they only pay off with the RD
     * decision, so ultrafast stays 16x16; and the candidate list has room for
     * both of them plus the four 16x16 ones. */
    if (e->p.preset >= 1) {
        int mv16[2][2];

        mv16[0][0] = mv[0][0]; mv16[0][1] = mv[0][1];
        mv16[1][0] = mv[1][0]; mv16[1][1] = mv[1][1];
        /* A macroblock whose better one-list 16x16 prediction already
         * costs less than k motion lambdas (SATD + vector bits) does not try
         * the 16x8 / 8x16 shapes: they rarely win there, and their four
         * searches plus two joint refinements are a third of a B
         * macroblock's analysis.  k = 96 below preset 4 measured +0.07 /
         * +0.04 % (5 clips / validation set) for -7.8 / -7.3 % of the time
         * at preset 3; k = 64 at preset 4 +0.03 / -0.02 % for -6.2 / -4.2 %.
         * A fixed threshold (not scaled by lambda) cost 0.3 % for the same
         * time.  AEG_BPGL overrides, 0 = always try them. */
        const int bpl = ENV_INT("AEG_BPGL", -1) >= 0 ? ENV_INT("AEG_BPGL", -1)
                        : e->p.preset >= 4 ? 64 : 96;
        const int parts_ok = !(bpl > 0 && e->mb_inter_sad < bpl * e->me_lambda);
        if ((e->p.partitions & 2) && parts_ok)
            b_candidate_parts(e, mbx, mby, AEG_MB_B16x8, (const int (*)[2])mv16);
        if ((e->p.partitions & 4) && parts_ok)
            b_candidate_parts(e, mbx, mby, AEG_MB_B8x16, (const int (*)[2])mv16);
        /* B_8x8 follows the preset rather than the partition mask: the mask's
         * bit 3 is documented as P_8x8, and this shape measured *neutral*
         * (BD-rate within +-0.1 % on five clips at six QPs) for about a tenth
         * of the encode time, so it sits with the slow presets.  Prepending it
         * to the candidate list costs two integer searches per 8x8 group --
         * eight per macroblock -- and the exact RD comparison then rejects it
         * on all but a handful of macroblocks.
         *
         * Re-measured once the rest had become cheaper (sub-8x8 P shapes off,
         * UMH early exit), at preset 3 off measured +0.04 / +0.05 % (5 clips /
         * validation set) for -3.7 / -2.5 % of the time, so it is preset 4
         * only (there off is +0.08 / +0.06 % for -4.2 / -2.6 %).  AEG_B8X8
         * overrides.  The 16x8 / 8x16 gate above applies to it as well
         * (+0.01 / +0.01 % at preset 4 for -1.3 / -1.5 %; AEG_BPG8=0 lifts
         * it). */
        if ((ENV_INT("AEG_B8X8", -1) >= 0 ? ENV_INT("AEG_B8X8", -1) > 0
                                          : e->p.preset >= 4) &&
            (parts_ok || !ENV_INT("AEG_BPG8", 1)))
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
        if (b_direct_reach(e, dl, (const int (*)[2])cd->mv,
                           (const int (*)[2])cd->mv1))
            e->mb_ncand++;
    }
    b_rd_gate(e, mbx, mby);
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

static int cand_scost(struct aeg_encoder *e, int mbx, int mby, int cand, int qp);

/* x264's p_halfpel_thresh for the older references' whole-macroblock search
 * (aeg_me_block_t): a reference whose integer-pel cost is more than 8/7 of
 * the best so far (reference 0's first) gets no sub-pel stage and no
 * candidate.  Preset 3 +0.07 %, preset 4 +0.04 % on the main and validation
 * clips for -3 / -4 % of the time.  AEG_HPT=0 turns it off. */
static int hpt_on(const struct aeg_encoder *e)
{
    (void)e;
    return ENV_INT("AEG_HPT", 1) > 0;
}

/* x264's partition order for P macroblocks: P_8x8 is searched first, and
 * 16x8 / 8x16 only when it beats 16x16 by SATD plus vector bits (the value is
 * the percentage of the 16x16 cost P_8x8 has to come in under; 0 = always
 * search them).  Preset 1: +0.10 % on the main and validation clips for
 * -8 % of the time; preset 2 +0.55-0.64 %, so not there.  AEG_PPG
 * overrides. */
static int part_gate_pct(const struct aeg_encoder *e)
{
    const int v = ENV_INT("AEG_PPG", -1);
    return v >= 0 ? v : e->p.preset == 1 ? 100 : 0;
}

static void analyse_mb_ph(struct aeg_encoder *e, int mbx, int mby, mbenc *c);

/* (timed as a phase, see aeg_ph_switch) */
static void analyse_mb(struct aeg_encoder *e, int mbx, int mby, mbenc *c)
{
    const int prev = aeg_ph_push(AEG_PH_ANALYSE);

    analyse_mb_ph(e, mbx, mby, c);
    aeg_ph_pop(prev);
}

static void analyse_mb_ph(struct aeg_encoder *e, int mbx, int mby, mbenc *c)
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
    if (e->p.partitions != 0 && c->qp >= QP_MIN && e->cur_is_b &&
        e->has_ref && e->has_ref2) {
        if (analyse_mb_b(e, mbx, mby))
            return;
    }
    /* The motion search needs a reference picture *and* a P slice: an IDR
     * picture is always an I slice, so inter candidates there could never be
     * coded (writing them as an I macroblock desynchronises the decoder). */
    if (e->p.partitions != 0 && c->qp >= QP_MIN && e->has_ref && e->cur_is_p) {
        int mvx, mvy, sad;
        aeg_mv_predict(e->mbs, e->mb_w, e->mb_w, mbx, mby, e->top_row,
                       &e->mb_pmv[0], &e->mb_pmv[1]);
        /* With two references the neighbours may sit on the older picture,
         * and the predictor then follows the reference-matching rule of
         * 8.4.1.3.1 (a lone neighbour on the same reference wins), which the
         * single-reference shortcut above cannot express (nor the neighbours of
         * an MBAFF frame's macroblock pairs). */
        if (e->cur_l0_active > 1 || e->mbaff)
            pred16_ref(e, mbx, mby, 0, &e->mb_pmv[0], &e->mb_pmv[1]);
        /* The block to match is the *source* macroblock.  recp still holds the
         * previous picture at this position (nothing has been reconstructed
         * here yet), so searching against it would always favour mv = 0. */
        if (e->par_valid && e->par_full && !e->cfld) {
            /* the whole-frame rounds already produced this macroblock's
             * quarter-pel vector (and its SATD) */
            const int idx = mby * e->mb_w + mbx;
            mvx = e->par_mv[idx * 2 + 0];
            mvy = e->par_mv[idx * 2 + 1];
            sad = e->par_cost[idx];
        } else {
            int seed[2], nb[8][2], nnb;
            seed[0] = e->seed_mv[(mby * e->mb_w + mbx) * 2 + 0];
            seed[1] = e->seed_mv[(mby * e->mb_w + mbx) * 2 + 1];
            if (e->lr_valid) {
                seed[0] = e->lr_mv[0][(mby * e->mb_w + mbx) * 2 + 0];
                seed[1] = e->lr_mv[0][(mby * e->mb_w + mbx) * 2 + 1];
            }
            nnb = me_neighbour_cands(e, mbx, mby, 0, nb);
            e->me_hpt = 1 << 30;
            aeg_me_block_t(e->refpl[0][0] + mby * 16 * e->stride[0] + mbx * 16,
                         e->stride[0],
                         e->srcp[0] + mby * 16 * e->stride[0] + mbx * 16,
                         e->stride[0], 16, 16,
                         e->p.search_range ? e->p.search_range : 16,
                         SUBME_F(e, 8), e->mb_pmv[0], e->mb_pmv[1], seed,
                         (const int (*)[2])nb, nnb, e->me_lambda, &e->mcr[0],
                         &mvx, &mvy, &sad, &e->me_hpt);
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
        e->mb_maxref = e->cur_l0_active - 1;
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
            if ((sx != mvx || sy != mvy) && (!e->mbaff || mv_me_ok(e, sx, sy)) &&
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
            int px, py, rx, ry, rsad, b, rr = 1;
            /* (the threshold only exists when reference 0's search ran here,
             * not with the frame-wide batch) */
            e->me_hpt_on = hpt_on(e) && !e->par_valid;
            pred16_ref(e, mbx, mby, 1, &px, &py);
            me_partition_ref(e, mbx, mby, AEG_MB_P16x16, 0, geo16, px, py, 1,
                             &rx, &ry, &rsad, 1);
            if (e->me_pruned)
                rr = -1;
            /* With three or more references every older one gets its own
             * search; the best by SATD plus the vector and index bits becomes
             * the one candidate the RD pass prices. */
            if (e->cur_l0_active > 2) {
                int best = rr < 0 ? 1 << 30 :
                           part_satd_ref(e, mbx, mby, geo16, rx, ry, 1) +
                           8 * (aeg_mvd_bits(rx - px, ry - py) + ref_te_bits(e, 1));
                int k;
                for (k = 2; k < e->cur_l0_active; k++) {
                    int qx, qy, kx, ky, ks, cost;
                    pred16_ref(e, mbx, mby, k, &qx, &qy);
                    me_partition_ref(e, mbx, mby, AEG_MB_P16x16, 0, geo16, qx, qy, k,
                                     &kx, &ky, &ks, 1);
                    if (e->me_pruned)
                        continue;
                    cost = part_satd_ref(e, mbx, mby, geo16, kx, ky, k) +
                           8 * (aeg_mvd_bits(kx - qx, ky - qy) + ref_te_bits(e, k));
                    if (cost < best) {
                        best = cost;
                        rr = k;
                        rx = kx; ry = ky; px = qx; py = qy;
                    }
                }
                if (pref_mode() >= 2) {
                    /* x264: when reference 0 is the best whole-macroblock
                     * match, the partitions look no further back than the
                     * neighbours did */
                    int c0 = part_satd_ref(e, mbx, mby, geo16, mvx, mvy, 0) +
                             8 * (aeg_mvd_bits(mvx - e->mb_pmv[0], mvy - e->mb_pmv[1]) +
                                  ref_te_bits(e, 0));
                    if (c0 <= best)
                        e->mb_maxref = nb_maxref(e, mbx, mby);
                }
            }
            e->me_hpt_on = 0;
            if (rr >= 0) {
            memset(rc, 0, sizeof(*rc));
            rc->type = AEG_MB_P16x16;
            rc->nparts = 1;
            for (b = 0; b < 16; b++) {
                rc->mv[b][0] = rx;
                rc->mv[b][1] = ry;
                rc->ref[b] = (uint8_t)rr;
            }
            rc->pmv[0][0] = px;
            rc->pmv[0][1] = py;
            e->mb_ncand++;
            }
        }
        /* Partitioned candidates: 16x8 and 8x16, bit 1 and 2 of the mask.
         * They only pay off with the RD decision, so ultrafast stays 16x16. */
        if (e->p.preset >= 1 && part_gate_pct(e) > 0) {
            /* x264's order: P_8x8 first, and the two halves only when it
             * beats the whole macroblock (part_gate_pct percent of its cost) */
            int ppg = part_gate_pct(e), n8 = e->mb_ncand, halves = 1;
            if (e->p.partitions & 8) {
                me_candidate_p8x8(e, mbx, mby);
                if (e->mb_ncand > n8)
                    halves = (int64_t)cand_scost(e, mbx, mby, n8, c->qp) * 100 <
                             (int64_t)cand_scost(e, mbx, mby, 0, c->qp) * ppg;
            }
            if ((e->p.partitions & 2) && halves)
                me_candidate(e, mbx, mby, AEG_MB_P16x8);
            if ((e->p.partitions & 4) && halves)
                me_candidate(e, mbx, mby, AEG_MB_P8x16);
        } else if (e->p.preset >= 1) {
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
    /* ref_idx_l0 for P when two or more references are active. */
    if (l0_syn(e) > 1 && is_inter_type(c->type) && !is_b_type(c->type)) {
        if (c->type == AEG_MB_P8x8) {
            for (i = 0; i < 4; i++)  /* te(v): 1 bit with cMax = 1 */
                bits += ref_te_bits(e, c->ref[(i & 1) * 2 + (i >> 1) * 8]);
        } else {
            for (p = 0; p < c->nparts; p++) {
                int geo[4], b0;
                mb_part_geom(c, p, geo, &b0);
                bits += ref_te_bits(e, c->ref[b0]);
            }
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

static void snap_luma(const struct aeg_encoder *e, int mbx, int mby, pixel *buf)
{
    const pixel *p = e->recp[0] + mby * 16 * e->stride[0] + mbx * 16;
    int y;
    for (y = 0; y < 16; y++)
        memcpy(buf + y * 16, p + y * e->stride[0], 16 * sizeof(pixel));
}

static void restore_luma(struct aeg_encoder *e, int mbx, int mby, const pixel *buf)
{
    pixel *p = e->recp[0] + mby * 16 * e->stride[0] + mbx * 16;
    int y;
    for (y = 0; y < 16; y++)
        memcpy(p + y * e->stride[0], buf + y * 16, 16 * sizeof(pixel));
}

static void snap_chroma(const struct aeg_encoder *e, int mbx, int mby, pixel *buf)
{
    int pl, y;
    for (pl = 0; pl < 2; pl++) {
        const pixel *p = e->recp[1 + pl] + mby * 8 * e->stride[1] + mbx * 8;
        pixel *b = buf + pl * 64;
        for (y = 0; y < 8; y++)
            memcpy(b + y * 8, p + y * e->stride[1], 8 * sizeof(pixel));
    }
}

static void restore_chroma(struct aeg_encoder *e, int mbx, int mby, const pixel *buf)
{
    int pl, y;
    for (pl = 0; pl < 2; pl++) {
        pixel *p = e->recp[1 + pl] + mby * 8 * e->stride[1] + mbx * 8;
        const pixel *b = buf + pl * 64;
        for (y = 0; y < 8; y++)
            memcpy(p + y * e->stride[1], b + y * 8, 8 * sizeof(pixel));
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
        int yM, ny;
        const aeg_mb *m = nb_left(e, mbx, mby, 4 * by, 16, &yM);
        ny = yM >> 2;                   /* (the neighbour's block row) */
        if (m->type == AEG_MB_I4x4)
            a = m->intra4[intra4_idx(3, ny)];
        else if (m->type == AEG_MB_I8x8)
            /* An Intra_8x8 macroblock has no 4x4 modes, but a decoder keeps
             * its four prediction modes in the 4x4 mode cache, one per 2x2
             * group of blocks.  The neighbour of this macroblock's block row
             * is therefore the 8x8 group of the same block row in the group
             * column on the right, not the DC fallback. */
            a = m->intra8[(ny >> 1) * 2 + 1];
        else
            a = 2;
    } else {
        a = -1;
    }
    if (by > 0) {
        b = cur[intra4_idx(bx, by - 1)];
    } else if (nb_top_ok(e, mby)) {
        const aeg_mb *m = nb_mbB(e, mbx, mby);
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
/* How many of the best prediction modes of an Intra_4x4 / Intra_8x8 block
 * get a second, exact look with the level selection (see intra_rd_modes()).
 * AEG_IRDK overrides, for measurements. */
static int intra_refine_k(void)
{
    static _Atomic int v = -1;
    if (v < 0) {
        const char *ev = getenv("AEG_IRDK");
        v = ev ? atoi(ev) : 2;
    }
    return v;
}

/* Rate-distortion price of one intra block predicted by `pred` (stride =
 * side): the residual is transformed and quantised as the coder will do it,
 * optionally run through the level selection, reconstructed, and its SSE is
 * weighed against what the CABAC coder charges for the levels in the current
 * context.  The prediction mode's own bits are the caller's business. */
/* What intra_block_rd() without the level selection measured of one
 * prediction, so that pricing the same prediction again with it (the refine
 * pass of intra_rd_modes()) starts where the first evaluation ended instead of
 * transforming, quantising, reconstructing and pricing the block again. */
struct ib_memo {
    struct trel tr;
    dctcoef lev[64];
    int64_t sse, bits;        /* trel_sse() and the price of lev[] */
    int64_t cost;             /* what intra_block_rd(.., 0) returned */
};

static int64_t intra_block_rd_memo(struct aeg_encoder *e, const struct ib_memo *mo,
                                   const pixel *sb, int s, const pixel *pred,
                                   int qp, double lambda, dctcoef *out_lev);

static int64_t intra_block_rd(struct aeg_encoder *e, int mbx, int mby,
                              const mbenc *t, int side, int rb,
                              const pixel *sb, int s, const pixel *pred,
                              double lambda, int trel, struct ib_memo *mo)
{
    int16_t res[64];
    dctcoef coef[64], lev[64], zz[64];
    struct trel tr;
    const int n = side * side;
    int x, y, i, nzz = 0;
    int64_t sse = 0, b256 = 0, cost;

    for (y = 0; y < side; y++)
        for (x = 0; x < side; x++)
            res[y * side + x] = (int16_t)((int)sb[y * s + x] -
                                          (int)pred[y * side + x]);
    tr.side = side;
    tr.n = n;
    tr.dc = NULL;
    if (side == 4) {
        int a, b, va, vb;
        aeg_transform_4x4(res, 4, coef);
        aeg_quant_4x4(coef, lev, t->qp, 1);
        cab_nz4_luma(e, mbx, mby, t, rb, &a, &b, &va, &vb);
        tr.scan = e->scan4;
        tr.cat = 2 + e->fcat;
        tr.cbf_ctx = cab_cbf_ctx(2, a, b, va, vb, 1);
    } else {
        aeg_transform_8x8(res, 8, coef);
        aeg_quant_8x8(coef, lev, t->qp, 1);
        tr.scan = e->scan8;
        tr.cat = 5 + e->fcat;
        tr.cbf_ctx = -1;
    }
    if (trel) {
        /* the same level selection, and the same lambda for it, as the
         * block gets when it is coded (code_luma_i4x4, code_luma8_block) */
        int64_t tsse = -1, tbits = 0;
        trel_descent_ex(&e->cab_shadow, &tr, lev, sb, s, pred, side, t->qp,
                        trellis_intra_lambda(side == 4 ? lambda : AEG_LAMBDA(t->qp)),
                        side == 4 ? 2 : 1, side == 4, &tsse, &tbits);
        if (tsse >= 0) {
            /* the search measured and priced the levels it left */
            for (i = 0; i < n && !nzz; i++)
                nzz |= lev[tr.scan[i]] != 0;
            b256 = (nzz || side == 4) ? tbits : 0;
            return tsse + trel_price(b256, lambda);
        }
    }
    for (i = 0; i < n; i++) {
        zz[i] = lev[tr.scan[i]];
        nzz |= zz[i] != 0;
    }
    sse = trel_sse(&tr, sb, s, pred, side, lev, t->qp);
    /* An 8x8 block without coefficients costs its bit of the coded block
     * pattern, which is the same for every mode; a 4x4 block pays its
     * coded_block_flag, which the price includes. */
    if (nzz || side == 4)
        b256 = cabac_block_price(&e->cab_shadow, zz, n, tr.cat, tr.cbf_ctx);
    cost = sse + trel_price(b256, lambda);
    if (mo) {
        mo->tr = tr;
        memcpy(mo->lev, lev, sizeof(dctcoef) * (size_t)n);
        mo->sse = sse;
        mo->bits = b256;
        mo->cost = cost;
    }
    return cost;
}

/* intra_block_rd(.., trel = 1, ..) of the prediction `mo` was measured
 * with: the same levels, the same descent, the same numbers. */
static int64_t intra_block_rd_memo(struct aeg_encoder *e, const struct ib_memo *mo,
                                   const pixel *sb, int s, const pixel *pred,
                                   int qp, double lambda, dctcoef *lev)
{
    const struct trel *tr = &mo->tr;
    const int side = tr->side, n = tr->n;
    int64_t tsse = -1, tbits = 0, b256;
    int i, nzz = 0;

    /* lev[] receives the levels the selection settles on */
    memcpy(lev, mo->lev, sizeof(dctcoef) * (size_t)n);
    trel_descent_kn(&e->cab_shadow, tr, lev, sb, s, pred, side, qp,
                    trellis_intra_lambda(side == 4 ? lambda : AEG_LAMBDA(qp)),
                    side == 4 ? 2 : 1, side == 4, &tsse, &tbits,
                    mo->sse, mo->bits);
    if (tsse < 0)
        return mo->cost;        /* no levels: nothing to select */
    for (i = 0; i < n && !nzz; i++)
        nzz |= lev[tr->scan[i]] != 0;
    b256 = (nzz || side == 4) ? tbits : 0;
    return tsse + trel_price(b256, lambda);
}

/* The mode decision of one Intra_4x4 / Intra_8x8 block by rate and
 * distortion (preset 3 and up, CABAC): every legal mode is priced exactly
 * with the quantiser's levels, and the best few are priced again with the
 * levels the level selection will settle on -- which is what the block will
 * really cost, and can reorder modes whose plain quantisation is close.  The
 * mode itself costs one bin-ish bit when it is the predicted one and four
 * otherwise.  Returns the chosen mode. */
static int intra_rd_modes(struct aeg_encoder *e, int mbx, int mby,
                          const mbenc *t, int side, int rb, const pixel *sb,
                          int s, const pixel *top, const pixel *left,
                          int tl, int top_avail, int left_avail,
                          const int *modes, int nmodes, int pmode,
                          double lambda, dctcoef *lev_out, int *lev_ok)
{
    pixel pred[64], preds[9][64];
    struct ib_memo memo[9];
    int64_t cost[9], rcost[9];
    dctcoef levs[9][64];
    int same[9], rlev[9], bj = -1;
    int order[9], k, j, best = modes[0], kk = intra_refine_k();
    int64_t best_cost = INT64_MAX;
    int sel[9];
    const int pre = side == 4 ? ENV_INT("AEG_I4K", 0) : ENV_INT("AEG_I8K", 0);
    aeg_iedge ied;

    if (side == 4)
        aeg_intra4x4_edge(&ied, top, left, tl, top_avail, left_avail);
    else
        aeg_intra8x8_edge(&ied, top, left, tl, top_avail, left_avail,
                          top_avail && left_avail);

    /* Pre-selection by SATD: only the `pre` modes whose prediction looks best
     * (SATD + mode bits in the SATD domain) get the exact price. */
    if (pre > 0 && nmodes > pre) {
        int64_t sc[9];
        int ord[9], n2 = 0;
        const double ls = sqrt(lambda);
        for (k = 0; k < nmodes; k++) {
            int m = modes[k];
            if (side == 4)
                aeg_intra4x4_pred(pred, 4, &ied, m);
            else
                aeg_intra8x8_pred(pred, 8, &ied, m);
            sc[k] = (side == 4 ? aeg_satd_4x4(pred, 4, sb, s)
                               : aeg_sa8d_8x8(pred, 8, sb, s)) +
                    (int64_t)(ls * (m == pmode ? 1 : 4) + 0.5);
            for (j = k; j > 0 && sc[ord[j - 1]] > sc[k]; j--)
                ord[j] = ord[j - 1];
            ord[j] = k;
        }
        for (k = 0; k < pre; k++)
            sel[n2++] = modes[ord[k]];
        modes = sel;
        nmodes = n2;
    }

    for (k = 0; k < nmodes; k++) {
        int m = modes[k], d;
        pixel *pk = preds[k];
        if (side == 4)
            aeg_intra4x4_pred(pk, 4, &ied, m);
        else
            aeg_intra8x8_pred(pk, 8, &ied, m);
        /* Two modes that predict the same samples (flat neighbours, missing
         * ones) code the same residual: only the mode's own bits differ. */
        for (d = 0; d < k; d++)
            if (!memcmp(preds[d], pk, (size_t)(side * side) * sizeof(pixel)))
                break;
        same[k] = d < k ? same[d] : k;
        if (same[k] == k)
            intra_block_rd(e, mbx, mby, t, side, rb, sb, s, pk, lambda, 0, &memo[k]);
        cost[k] = memo[same[k]].cost +
                  trel_price(256 * (m == pmode ? 1 : 4), lambda);
        /* insertion into the ranking */
        for (j = k; j > 0 && cost[order[j - 1]] > cost[k]; j--)
            order[j] = order[j - 1];
        order[j] = k;
    }
    if (kk < 2 || e->p.trellis <= 0 || !trellis_intra_on())
        return modes[order[0]];
    if (kk > nmodes)
        kk = nmodes;
    for (j = 0; j < kk; j++) {
        int m = modes[order[j]], g = same[order[j]], d;
        int64_t c;
        for (d = 0; d < j; d++)
            if (same[order[d]] == g)
                break;
        if (d < j) {
            rcost[j] = rcost[d];
            rlev[j] = rlev[d];
        } else {
            rlev[j] = j;
            rcost[j] = intra_block_rd_memo(e, &memo[g], sb, s, preds[g], t->qp,
                                           lambda, levs[j]);
        }
        c = rcost[j] + trel_price(256 * (m == pmode ? 1 : 4), lambda);
        if (c < best_cost) {
            best_cost = c;
            best = m;
            bj = j;
        }
    }
    /* the winner's levels are the ones coding it will select again */
    if (lev_out && bj >= 0) {
        memcpy(lev_out, levs[rlev[bj]], sizeof(dctcoef) * (size_t)(side * side));
        *lev_ok = 1;
    }
    return best;
}

/* Intra_4x4 / Intra_8x8 modes chosen by exact rate-distortion (quantise,
 * reconstruct, count the CABAC bits) rather than SATD + mode bits: presets
 * 3-4.  AEG_IRD overrides. */
static int intra_rd_on(const struct aeg_encoder *e)
{
    const int v = ENV_INT("AEG_IRD", -1);
    return e->p.cabac && (v >= 0 ? v > 0 : e->p.preset >= 3);
}

static void code_luma_i4x4(struct aeg_encoder *e, int mbx, int mby, mbenc *t,
                           double lambda)
{
    const int s = e->stride[0];
    const pixel *src = e->srcp[0] + mby * 16 * s + mbx * 16;
    pixel *dst = e->recp[0] + mby * 16 * s + mbx * 16;
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
        int top_avail = nb_top_ok(e, mby) || by > 0;
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
            tr_avail = nb_top_ok(e, mby) && (bx < 12 || c_right_ok(e, mbx, mby));
        pixel top[8], left[4], pred[16];
        aeg_iedge ied;
        const pixel *sb = src + by * s + bx;
        pixel *db = dst + by * s + bx;
        int tl, nmodes, modes[9], best = 2, best_cost = 1 << 30, pmode, m;
        int lev_ok = 0;

        if (top_avail) {
            const pixel *tr = by ? dst + (by - 1) * s + bx : dst - s + bx;
            for (k = 0; k < 4; k++)
                top[k] = tr[k];
            for (k = 4; k < 8; k++)
                top[k] = tr_avail ? tr[k] : top[3];
        } else {
            for (k = 0; k < 8; k++)
                top[k] = PIXEL_MID;
        }
        if (left_avail) {
            for (k = 0; k < 4; k++)
                left[k] = db[k * s - 1];
        } else {
            for (k = 0; k < 4; k++)
                left[k] = PIXEL_MID;
        }
        tl = (top_avail && left_avail) ? db[-s - 1] : PIXEL_MID;

        aeg_intra4x4_edge(&ied, top, left, tl, top_avail, left_avail);
        pmode = intra4_pred_mode(e, mbx, mby, t->intra4, rx, ry);
        nmodes = intra4_mode_list(top_avail, left_avail, modes);
        if (intra_rd_on(e)) {
            best = intra_rd_modes(e, mbx, mby, t, 4, rb, sb, s, top, left, tl,
                                  top_avail, left_avail, modes, nmodes, pmode,
                                  lambda, t->level[rb], &lev_ok);
        } else for (k = 0; k < nmodes; k++) {
            int cost;
            m = modes[k];
            aeg_intra4x4_pred(pred, 4, &ied, m);
            if (e->p.preset >= 3) {
                /* True rate-distortion choice: quantise the residual, add it
                 * back, and weigh the exact SSE against the exact bits the
                 * block will cost.  The SATD approximation below is much
                 * cheaper but it cannot see that a mode which predicts
                 * better can still cost more after quantisation. */
                int16_t res[16];
                dctcoef coef[16], lev[16];
                pixel rb_rec[16];
                const pixel *pr = pred;
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
                                                0 | e->fscan) : 0));
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

        aeg_intra4x4_pred(pred, 4, &ied, best);
        {
            int16_t res[16];
            dctcoef coef[16];
            int nz = 0;
            for (y = 0; y < 4; y++)
                for (x = 0; x < 4; x++)
                    res[y * 4 + x] = (int16_t)((int)sb[y * s + x] -
                                               (int)pred[y * 4 + x]);
            /* lev_ok: the mode decision quantised and selected these
             * levels already, with the same numbers */
            if (!lev_ok) {
                aeg_transform_4x4(res, 4, coef);
                aeg_quant_4x4(coef, t->level[rb], t->qp, 1);
            }
            if (!lev_ok && e->p.trellis > 0 && e->p.cabac && trellis_intra_on()) {
                /* the same level selection the inter blocks get; the block's
                 * neighbours inside the macroblock are already final */
                struct trel tr;
                int a, b, va, vb;

                cab_nz4_luma(e, mbx, mby, t, rb, &a, &b, &va, &vb);
                tr.side = 4;
                tr.n = 16;
                tr.scan = e->scan4;
                tr.cat = 2 + e->fcat;
                tr.cbf_ctx = cab_cbf_ctx(2, a, b, va, vb, 1);
                tr.dc = NULL;
                trel_descent(&e->cab_shadow, &tr, t->level[rb], sb, s, pred, 4,
                             t->qp, trellis_intra_lambda(lambda), 2, 1);
            }
            {
                int dummy;
                nz = aeg_nz_last(t->level[rb], 16, &dummy);
            }
            t->nz[rb] = (int8_t)nz;
            if (nz)
                t->cbp |= 1 << ((rb >> 3) * 2 + ((rb & 3) >> 1));
        }
        aeg_idct_add_block(db, s, t->level[rb], t->qp, pred, 4, NULL);
        t->sse += aeg_ssd(4, 4, db, s, sb, s);
    }

    t->bits += mode_bits;
    /* The blocks of every coded 8x8 group are coded, in the order the group
     * table lists them; the context of a block comes from its neighbours.
     * (CAVLC estimate only.) */
    for (g = 0; g < 4 && !e->p.cabac; g++) {
        if (!(t->cbp & (1 << g)))
            continue;
        for (k = 0; k < 4; k++) {
            int b = group_blk[g][k];
            if (t->nz[b])
                t->bits += aeg_cavlc_block_cost(t->level[b],
                                                luma_nc(e, mbx, mby, t, b), 0 | e->fscan);
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
        int yM;
        const aeg_mb *m = nb_left(e, mbx, mby, 8 * gy, 16, &yM);
        if (m->type == AEG_MB_I8x8)
            a = m->intra8[(yM >> 3) * 2 + 1];
        else if (m->type == AEG_MB_I4x4)
            /* A decoder keeps one prediction mode per 4x4 block, so the
             * neighbour of an 8x8 group is the 4x4 mode of the block at the
             * same block row in the column to the left (see the note above
             * intra4_pred_mode). */
            a = m->intra4[intra4_idx(3, yM >> 2)];
        else
            a = 2;
    } else {
        a = -1;
    }
    if (gy > 0) {
        b = cur[(gy - 1) * 2 + gx];
    } else if (nb_top_ok(e, mby)) {
        const aeg_mb *m = nb_mbB(e, mbx, mby);
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
    const pixel *src = e->srcp[0] + mby * 16 * s + mbx * 16;
    pixel *dst = e->recp[0] + mby * 16 * s + mbx * 16;
    int bits_scale = (int)(lambda * 0.5 + 0.5);
    int g, k, mode_bits = 0;

    t->type = AEG_MB_I8x8;
    t->t8 = 1;
    t->cbp &= 0x30;                     /* keep the chroma pattern */
    memset(t->intra8, 0, sizeof(t->intra8));

    for (g = 0; g < 4; g++) {
        const int gx = g & 1, gy = g >> 1;
        const int bx = gx * 8, by = gy * 8;
        const int top_avail = nb_top_ok(e, mby) || by > 0;
        const int left_avail = mbx > 0 || bx > 0;
        pixel top[16], left[8];
        int tl, tr_avail;
        aeg_iedge ied;
        int nmodes, modes[9], best = 2, best_cost = 1 << 30, pmode, m;
        int lev_ok = 0;
        pixel pred[64];

        /* The sixteen top samples.  Inside the macroblock the row above the
         * block has already been reconstructed, so its samples are used as
         * they are; at the top of the macroblock they come from the
         * macroblock above, and the eight on the right only when the
         * macroblock above-right has been decoded -- otherwise p[15][-1] is
         * replicated into them (8.3.3.4.2). */
        if (top_avail) {
            const pixel *tr = by ? dst + (by - 1) * s + bx : dst - s + bx;
            /* The eight samples on the right come from the 8x8 block above
             * and to the right (8.3.3.4.2).  For the two blocks in the left
             * column that block has already been reconstructed -- it is the
             * top-right block of this macroblock for the lower one -- so the
             * samples are used as they are.  For the top-right block it is in
             * the macroblock above, which has been decoded when there is one,
             * and for the bottom-right block it is in the macroblock to the
             * right, which comes later in raster order and is therefore not
             * available. */
            tr_avail = (bx == 0) || (by == 0 && c_right_ok(e, mbx, mby));
            for (k = 0; k < 8; k++)
                top[k] = tr[k];
            for (k = 8; k < 16; k++)
                top[k] = tr_avail ? tr[k] : top[7];
        } else {
            for (k = 0; k < 16; k++)
                top[k] = PIXEL_MID;
        }
        if (left_avail) {
            for (k = 0; k < 8; k++)
                left[k] = dst[(by + k) * s + bx - 1];
        } else {
            for (k = 0; k < 8; k++)
                left[k] = PIXEL_MID;
        }
        tl = (top_avail && left_avail) ? (by ? dst[(by - 1) * s + bx - 1]
                                             : dst[-s + bx - 1]) : PIXEL_MID;

        aeg_intra8x8_edge(&ied, top, left, tl, top_avail, left_avail,
                          top_avail && left_avail);
        pmode = intra8_pred_mode(e, mbx, mby, t->intra8, gx, gy);
        nmodes = intra4_mode_list(top_avail, left_avail, modes);
        if (intra_rd_on(e)) {
            best = intra_rd_modes(e, mbx, mby, t, 8, 0, src + by * s + bx, s,
                                  top, left, tl, top_avail, left_avail, modes,
                                  nmodes, pmode, lambda, t->level8[g], &lev_ok);
        } else for (k = 0; k < nmodes; k++) {
            int cost;
            m = modes[k];
            aeg_intra8x8_pred(pred, 8, &ied, m);
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

        aeg_intra8x8_pred(pred, 8, &ied, best);
        code_luma8_block_lv(e, mbx, mby, t, g, pred, 8, 1, lev_ok);
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

static int lmemo_on(void)
{
    static _Atomic int v = -1;
    if (v < 0) {
        const char *ev = getenv("AEG_LMEMO");
        v = ev ? atoi(ev) != 0 : 1;
    }
    return v;
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
        bfld_stamp(e, t);
        inter_predict_typed(e, mbx, mby, t);
        if (lmemo_on() && !e->coef_batch) {
            /* Candidates that predict the same pixels code the same luma: the
             * partitions of static content mostly find the 16x16 vector, and
             * the direct vector is often the searched one.  The residual
             * coding reads the source, the prediction, the QP, whether the
             * 8x8 transform may (or must) be used and the trellis switch --
             * nothing of the candidate's shape or vectors -- so its result is
             * replayed, reconstruction included, and only the header bits are
             * priced anew. */
            const int t8ok = t8_allowed(e, t), tr = e->p.trellis;
            const int s = e->stride[0];
            pixel *dst = e->recp[0] + mby * 16 * s + mbx * 16;
            int k, y;
            for (k = 0; k < e->lmemo_n; k++) {
                const struct lmemo *m = &e->lmemo[k];
                if (m->qp == qp && m->t8ok == t8ok && m->t8force == e->t8_force &&
                    m->trellis == tr &&
                    !memcmp(m->pred, e->pred_l, sizeof(m->pred)))
                    break;
            }
            if (k < e->lmemo_n) {
                const struct lmemo *m = &e->lmemo[k];
                mbenc keep = *t;
                *t = m->t;
                t->type = keep.type;
                t->i16_mode = keep.i16_mode;
                t->nparts = keep.nparts;
                memcpy(t->sub_type, keep.sub_type, sizeof(t->sub_type));
                memcpy(t->blist, keep.blist, sizeof(t->blist));
                memcpy(t->mv, keep.mv, sizeof(t->mv));
                memcpy(t->mv1, keep.mv1, sizeof(t->mv1));
                memcpy(t->pmv, keep.pmv, sizeof(t->pmv));
                memcpy(t->pmv1, keep.pmv1, sizeof(t->pmv1));
                memcpy(t->ref, keep.ref, sizeof(t->ref));
                memcpy(t->ref1, keep.ref1, sizeof(t->ref1));
                for (y = 0; y < 16; y++)
                    memcpy(dst + y * s, m->rec + y * 16, 16 * sizeof(pixel));
            } else {
                struct lmemo *m;
                if (use_t8(e, mbx, mby, t))
                    code_luma_inter8(e, mbx, mby, t);
                else
                    code_luma_inter(e, mbx, mby, t);
                if (e->lmemo_n < 8)
                    k = e->lmemo_n++;
                else
                    k = e->lmemo_next++ & 7;
                m = &e->lmemo[k];
                m->qp = qp;
                m->t8ok = t8ok;
                m->t8force = e->t8_force;
                m->trellis = tr;
                memcpy(m->pred, e->pred_l, sizeof(m->pred));
                for (y = 0; y < 16; y++)
                    memcpy(m->rec + y * 16, dst + y * s, 16 * sizeof(pixel));
                m->t = *t;
            }
        } else if (use_t8(e, mbx, mby, t))
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
    bfld_stamp(e, t);
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
 * With the level selection on it is one: the host reconstructs from the
 * levels it selected (batch_host_inverse()).  Candidates whose residual wants
 * the 8x8 transform stay out of the launches altogether.
 *
 * It cannot be lifted to a whole picture without changing the bitstream: the
 * motion search predicts each macroblock's vectors from its neighbours'
 * *decided* vectors, intra prediction needs the reconstructed neighbours, and
 * the CAVLC cost model needs their non-zero counts.  All three are serial in
 * raster order, so the batch has to stop at the macroblock boundary -- see
 * docs/STATUS.md, "Rate-distortion batching (provisional)". */
static int64_t full_price(struct aeg_encoder *e, int mbx, int mby,
                          const mbenc *t, int64_t est, int cmode, double lambda,
                          int64_t bound);
static int64_t full_price_inner(struct aeg_encoder *e, int mbx, int mby,
                                const mbenc *t, int64_t est, int cmode,
                                double lambda, int64_t bound);
static void refine_cand(struct aeg_encoder *e, int mbx, int mby, int slot,
                        int qp, double lambda);

static int64_t rd_luma_inter_batch(struct aeg_encoder *e, int mbx, int mby,
                                   mbenc *best, pixel *snap_l,
                                   int64_t best_cost, double lambda, int qp,
                                   int cmode, int64_t *jinter)
{
    const pixel *src = e->srcp[0] + mby * 16 * e->stride[0] + mbx * 16;
    const int n = e->mb_ncand;
    const int trel = e->p.trellis > 0 && e->p.cabac;
    const int host_inv = batch_host_inverse(e);
    int k, nb = 0;
    /* per candidate: its slot in the batch, or -1 when its residual wants the
     * 8x8 transform -- the batch only produces 4x4 blocks, so such a candidate
     * is coded by the serial 8x8 path in the decision below and has no
     * business in the launches (decided once, here: use_t8() reads only the
     * source and the candidate's prediction, so asking again later would give
     * the same answer for the price of another SA8D and sixteen SATDs) */
    int slot[AEG_MB_CAND_MAX];
    int64_t t0;

    /* 1. every candidate's prediction (luma and chroma: the chroma one is
     *    what the decision prices the candidate's chroma against), and the
     *    luma residual of each 4x4 candidate in the batch buffer */
    for (k = 0; k < n; k++) {
        mbenc c;
        cand_enc(e, k, qp, &c);
        inter_predict_typed(e, mbx, mby, &c);
        memcpy(e->rd_pred + 256 * k, e->pred_l, 256 * sizeof(pixel));
        memcpy(e->rd_predc + 128 * k, e->pred_c, 128 * sizeof(pixel));
        if (use_t8(e, mbx, mby, &c)) {
            slot[k] = -1;
            continue;
        }
        slot[k] = nb;
        luma_inter_gather(e, src, e->pred_l, e->rd_res + 256 * nb);
        nb++;
    }
    /* 2. one launch each for the whole macroblock's 4x4 candidates (one
     *    for both without level selection, see use_tq_iq()) */
    if (nb && !trel && !host_inv && use_tq_iq(e)) {
        t0 = aeg_now_ns();
        e->d.ops.tq_iq_batch(e->d.ops.priv, e->rd_res, 16 * nb, qp, 0,
                             e->rd_lev, e->rd_nz, e->rd_rec);
        tq_iq_rec(e, aeg_now_ns() - t0);
    } else if (nb) {
        t0 = aeg_now_ns();
        e->d.ops.tq_batch(e->d.ops.priv, e->rd_res, 16 * nb, qp, 0, e->rd_lev,
                          e->rd_nz);
        stage_rec(e, AEG_STAGE_TRANSFORM,
                            e->d.ops.backend != AEG_BACKEND_CPU,
                            aeg_now_ns() - t0);
        /* The level selection runs on the host between the two launches, on
         * every candidate's levels exactly as code_luma_inter() runs it on
         * one candidate's: it reads only the source, the candidate's
         * prediction and its own levels (plus the neighbours, which no
         * candidate changes), so running it for all of them first selects
         * the same levels.  The reconstruction is then made from the selected
         * levels. */
        if (trel) {
            for (k = 0; k < n; k++) {
                mbenc c;
                if (slot[k] < 0)
                    continue;
                cand_enc(e, k, qp, &c);
                memcpy(e->pred_l, e->rd_pred + 256 * k, 256 * sizeof(pixel));
                trellis_luma4(e, mbx, mby, &c, qp,
                              (dctcoef (*)[16])(e->rd_lev + 256 * slot[k]),
                              inter_trellis_lambda(qp));
            }
        }
        if (!host_inv) {
            t0 = aeg_now_ns();
            e->d.ops.iq_idct_batch(e->d.ops.priv, e->rd_lev, 16 * nb, qp,
                                   e->rd_rec);
            stage_rec(e, AEG_STAGE_IQUANT_IDCT,
                                e->d.ops.backend != AEG_BACKEND_CPU,
                                aeg_now_ns() - t0);
        }
    }
    /* 3. decide: the serial candidate loop of code_mb() step for step -- the
     *    same price (full_price() against the running best), the same
     *    refinement of a candidate still in the running, the same strict
     *    tie-breaking -- with only the luma coefficient work taken from the
     *    batch.  Anything else and the two paths choose differently. */
    for (k = 0; k < n; k++) {
        mbenc t;
        int64_t cost;
        cand_enc(e, k, qp, &t);
        memcpy(e->pred_l, e->rd_pred + 256 * k, 256 * sizeof(pixel));
        memcpy(e->pred_c, e->rd_predc + 128 * k, 128 * sizeof(pixel));
        /* The batch only produces 4x4 blocks: a candidate whose residual wants
         * the 8x8 transform is coded through the serial path. */
        if (slot[k] < 0) {
            code_luma_inter8(e, mbx, mby, &t);
        } else {
            const int j = slot[k];
            luma_inter_finish(e, mbx, mby, &t, e->rd_lev + 256 * j,
                              trel ? NULL : e->rd_nz + 16 * j,
                              host_inv ? NULL : e->rd_rec + 256 * j,
                              e->rd_pred + 256 * k);
        }
        t.bits += inter_hdr_bits(e, &t);
        cost = t.sse + (int64_t)(lambda * t.bits + 0.5);
        t.cost = cost;
        cost = full_price(e, mbx, mby, &t, cost, cmode, lambda, best_cost);
        if (e->p.cabac && e->p.refine_cand > 0 && e->p.refine_mv > 0 &&
            cost <= best_cost) {
            /* serial: a whole macroblock evaluation per neighbour vector */
            refine_cand(e, mbx, mby, k, qp, lambda);
            cost = rd_luma(e, mbx, mby, &t, qp, e->mb_cand[k].type, 0, lambda,
                           k);
            cost = full_price(e, mbx, mby, &t, cost, cmode, lambda, best_cost);
        }
        if (cost < *jinter)
            *jinter = cost;
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



/* MBAFF: the mb_field_decoding_flag the bitstream carries for macroblock
 * (mbx, mby), or -1 when it has none (7.3.4): the top macroblock of a pair
 * carries it unless it is skipped, the bottom one only when the top one was
 * skipped and it is not.  A pair with both skipped carries none -- the
 * decoder infers it from the neighbouring pairs, which the decisions have to
 * agree with. */
static int mbaff_flag_coded(const struct aeg_encoder *e, int mbx, int mby,
                            int skipped)
{
    const aeg_mb *top;

    if (!e->mbaff || skipped)
        return -1;
    top = &e->mbs[(mby & ~1) * e->mb_w + mbx];
    if ((mby & 1) && !top->skip)
        return -1;
    return top->fld;
}

/* ctxIdx of mb_field_decoding_flag: 70 + the field pairs among the left and
 * the above pair (9.3.3.1.1.2) */
static int mbaff_ff_ctx(const struct aeg_encoder *e, int mbx, int mby)
{
    int top = mby & ~1, inc = 0;

    if (mbx > 0 && e->mbs[top * e->mb_w + mbx - 1].fld)
        inc++;
    if (top - 2 >= e->top_row && e->mbs[(top - 2) * e->mb_w + mbx].fld)
        inc++;
    return 70 + inc;
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
 * chroma reconstruction this writes is overwritten then.  Preset 2 and up;
 * AEG_CDECIDE=0/1 overrides. */
static int chroma_decide_on(const struct aeg_encoder *e)
{
    /* -0.34% of BD-rate for 10% of the time on the real clips when it was
     * added, so it sat with the slowest preset.  Re-measured after the speed
     * round: preset 3 -0.51 / -0.73 % (5 clips / validation set) for +7 /
     * +10 % of the time, -1.02 % over QP 16-25; preset 2 -0.49 / -0.58 % for
     * +9 %.  On from preset 2. */
    static _Atomic int v = -2;
    if (v == -2) {
        const char *ev = getenv("AEG_CDECIDE");
        v = ev ? atoi(ev) != 0 : -1;
    }
    return v >= 0 ? v : e->p.preset >= 2;
}

/* Whether code_mb() tries the inter winner's other transform size by rate
 * and distortion.  From preset 1 (there -0.70 % on the main and validation
 * clips alone, see part_subpel()).  AEG_T8RD=0/1 overrides. */
static int t8_rd_on(const struct aeg_encoder *e)
{
    static _Atomic int v = -2;
    if (v == -2) {
        const char *ev = getenv("AEG_T8RD");
        v = ev ? atoi(ev) != 0 : -1;
    }
    return v >= 0 ? v : e->p.preset >= 1;
}

/* Deblock RDO: whether the decision measures a candidate's luma distortion
 * after the loop filter has run over the macroblock's internal edges.
 * AEG_DBRDO=0/1 overrides. */
static int dbrdo_on(const struct aeg_encoder *e)
{
    static _Atomic int v = -2;
    if (v == -2) {
        const char *ev = getenv("AEG_DBRDO");
        v = ev ? atoi(ev) != 0 : -1;
    }
    return e->p.deblock && (v >= 0 ? v : e->p.preset >= 3);
}

/* How much the internal-edge filtering changes the luma SSE of the candidate
 * `t`, whose reconstruction is in the picture. */
static int64_t dbrdo_delta(struct aeg_encoder *e, int mbx, int mby,
                           const mbenc *t)
{
    pixel blk[256];
    aeg_mb q;
    const int s = e->stride[0];
    const pixel *rec = e->recp[0] + mby * 16 * s + mbx * 16;
    const pixel *src = e->srcp[0] + mby * 16 * s + mbx * 16;
    int i, y;

    /* only what the filter reads (aeg_mb also carries the levels: no memset) */
    q.type = (uint8_t)t->type;
    q.fld = (uint8_t)(e->fld | e->cfld);
    q.t8 = (uint8_t)(t->type == AEG_MB_I8x8 ||
                     (is_inter_type(t->type) && mb_t8_coded(e, t, 0)));
    q.cbp = (uint8_t)t->cbp;
    q.qp = (int8_t)t->qp;
    memcpy(q.mv, t->mv, sizeof(q.mv));
    if (is_b_type(t->type))
        memcpy(q.mv1, t->mv1, sizeof(q.mv1));
    memcpy(q.blist, t->blist, sizeof(q.blist));
    memcpy(q.ref_idx, t->ref, sizeof(q.ref_idx));
    memcpy(q.ref_idx1, t->ref1, sizeof(q.ref_idx1));
    for (i = 0; i < AEG_BLK_COUNT; i++)
        q.nz[i] = (int8_t)(t->nz[i] != 0);
    for (i = 0; i < 4; i++)
        q.nz8[i] = (int8_t)(t->nz8[i] != 0);
    for (y = 0; y < 16; y++)
        memcpy(blk + y * 16, rec + y * s, 16 * sizeof(pixel));
    if (!aeg_deblock_mb_internal_luma(blk, 16, &q, e->p.deblock_alpha,
                                      e->p.deblock_beta))
        return 0;
    return aeg_ssd(16, 16, blk, 16, src, s) - aeg_ssd(16, 16, rec, s, src, s);
}

static int64_t full_price(struct aeg_encoder *e, int mbx, int mby,
                          const mbenc *t, int64_t est, int cmode, double lambda,
                          int64_t bound)
{
    mbenc f;

    if (dbrdo_on(e)) {
        /* the filtered distortion stands in for the plain one (the price is
         * the distortion plus lambda times the rate, so the change adds) */
        int64_t dd = dbrdo_delta(e, mbx, mby, t);
        f = *t;
        f.sse += dd;
        return full_price_inner(e, mbx, mby, &f, est + dd, cmode, lambda, bound);
    }
    return full_price_inner(e, mbx, mby, t, est, cmode, lambda, bound);
}

static int64_t full_price_inner(struct aeg_encoder *e, int mbx, int mby,
                                const mbenc *t, int64_t est, int cmode,
                                double lambda, int64_t bound)
{
    mbenc f;
    int64_t dc;

    /* The luma distortion alone is a lower bound of the price (the rate term
     * is never negative): a candidate it already rules out needs no coder.
     * (A third of the Intra_4x4 candidates, a fifth of the Intra_8x8 ones.) */
    if (e->p.cabac && t->sse >= bound)
        return t->sse;
    if (!chroma_decide_on(e))
        return rd_price(e, mbx, mby, (mbenc *)t, est, lambda);
    f = *t;
    dc = rd_chroma(e, mbx, mby, &f, is_inter_type(t->type),
                   is_inter_type(t->type) ? t->chroma_mode : cmode, lambda);
    /* the same bound with the chroma distortion added */
    if (e->p.cabac && f.sse >= bound)
        return f.sse;
    return rd_price(e, mbx, mby, &f, est + dc, lambda);
}

/* Whether code_mb() prices the skipped macroblock against the winner.
 * AEG_SKIPCHK=0 turns it off. */
static int skip_check_on(void)
{
    static _Atomic int v = -1;
    if (v < 0) {
        const char *ev = getenv("AEG_SKIPCHK");
        v = ev ? atoi(ev) != 0 : 1;
    }
    return v;
}

#define P_MODE_LAMBDA 0.65

/* AEG_CMTIME=1: cycles per code_mb() section and picture kind (P, B, I),
 * printed at exit.  Section 6 is analyse_mb(). */
#include <x86intrin.h>
static uint64_t cm_cyc[3][8];
static uint64_t cm_cnt[3][8];
static const char *cm_name[8] = { "i16", "inter", "i4", "i8", "refine_mv",
                                  "chroma+skip+final", "analyse", "?" };
static void cm_dump(void)
{
    int k, i;
    for (k = 0; k < 3; k++)
        for (i = 0; i < 8; i++)
            if (cm_cyc[k][i])
                fprintf(stderr, "cm %c %-18s %8.1f Mcyc  n %8llu\n", "PBI"[k],
                        cm_name[i], cm_cyc[k][i] / 1e6,
                        (unsigned long long)cm_cnt[k][i]);
}
static int cm_on(void)
{
    static _Atomic int v = -1;
    if (v < 0) {
        v = ENV_INT("AEG_CMTIME", 0);
        if (v)
            atexit(cm_dump);
    }
    return v;
}
#define CM_KIND(e) ((e)->cur_is_p ? 0 : (e)->cur_is_b ? 1 : 2)
#define CM_T0() uint64_t cm_t_ = cm_on() ? __rdtsc() : 0
#define CM_LAP(e, i) do { if (cm_t_) { uint64_t n_ = __rdtsc();        \
        cm_cyc[CM_KIND(e)][i] += n_ - cm_t_; cm_cnt[CM_KIND(e)][i]++;   \
        cm_t_ = n_; } } while (0)

/* The intra gates of code_mb(), in percent (0 = off; AEG_GIB, AEG_GIP,
 * AEG_GI48B and AEG_GI48P override).  Measured on the real clips at -p 3
 * with the full search as reference (ABBD_GLOBAL=1, tools/real/ab.sh):
 *
 *   gate_b_intra, B 120 + gate_i48, B 50    -0.00 %   -9.4 % time
 *   + gate_i48, P 50                        +0.02 %   -8.5 % more
 *   gate_b_intra, P 60                      +0.24 %   (mega +1.06: no)
 *   gate_i48, P 60 instead of 50            +0.03 %   -1.4 %
 *
 * In a B picture intra wins about one macroblock in sixty, and on cock none
 * of the 65 % that the first gate skips; the second one skips 79 % of the
 * Intra_4x4/8x8 evaluations that are left, again without a winner among
 * them.  P pictures are a different matter -- a quarter of cock's P
 * macroblocks end up intra, a third of them where the inter prediction's
 * SATD is the lower -- so only the Intra_16x16-relative gate is used there,
 * at the point where it still misses almost nothing. */
static int gate_b_intra(const struct aeg_encoder *e)
{
    return e->cur_is_b ? ENV_INT("AEG_GIB", 120) : e->cur_is_p ? ENV_INT("AEG_GIP", 0) : 0;
}
static int gate_i48(const struct aeg_encoder *e)
{
    return e->cur_is_b ? ENV_INT("AEG_GI48B", 50) : ENV_INT("AEG_GI48P", 50);
}
/* SATD of the best (by SAD) Intra_16x16 prediction of the macroblock. */
static int i16_satd(struct aeg_encoder *e, int mbx, int mby)
{
    const pixel *p = e->recp[0] + mby * 16 * e->stride[0] + mbx * 16;
    const pixel *src = e->srcp[0] + mby * 16 * e->stride[0] + mbx * 16;
    pixel left[16], buf[256];
    int y;
    if (e->mb_nmodes <= 0)
        return 0;
    for (y = 0; y < 16; y++)
        left[y] = p[-1 + y * e->stride[0]];
    aeg_intra_predict_16x16(buf, 16, p - e->stride[0], left,
                            nb_top_ok(e, mby), mbx > 0,
                            nb_top_ok(e, mby) && mbx > 0, e->mb_modes[0]);
    return aeg_satd(16, 16, buf, 16, src, e->stride[0]);
}
/* How many of the eight quarter-pel neighbours the rate-distortion vector
 * refinement re-codes the macroblock with (0 = the plain walks: the four of
 * the cross at --refine-mv 1, all eight at 3).  --refine-mv 2 takes the four
 * (three at preset 3) whose prediction SATD plus the mvd's bits is the
 * lowest, in that order,
 * around the vector the search returned: a SATD and an interpolation are a
 * small fraction of one candidate's coding and pricing.  Measured at preset 4
 * on the five real clips plus six other 30-picture segments of the same
 * sources (ours against ours, global PSNR): +0.05 to +0.09 % for -9 % of the
 * time against all eight, where the fixed cross (--refine-mv 1) is +0.17 %;
 * all eight re-ordered and not walked is +0.01 %.  At preset 3 three of
 * them against the cross: -0.11 % / -0.02 % for -2 to -3 % of the time.
 * AEG_MVREFK=K overrides the count (1..8). */
static int mvref_k(const struct aeg_encoder *e)
{
    static _Atomic int v = -2;
    if (v == -2) {
        const char *ev = getenv("AEG_MVREFK");
        v = ev ? atoi(ev) : -1;
    }
    if (v >= 0)
        return v;
    return e->p.refine_mv == 2 ? (e->p.preset >= 4 ? 4 : 3) : 0;
}

static inline int mvref_se_bits(int v)
{
    unsigned m = (unsigned)(v < 0 ? -v : v) * 2u;
    return m ? 2 * (31 - __builtin_clz(m)) + 1 : 1;
}

/* Order the eight neighbours of partition `geo` (vector mv on list/ref
 * `ref`; for a bi-predicted B partition `omv` is the other list's vector,
 * `bi` 1 when the refined list is list 0 and 2 when it is list 1). */
static void mvref_order(struct aeg_encoder *e, int mbx, int mby,
                        const int geo[4], const int16_t mv[2],
                        const int16_t pmv[2], int ref, int bi,
                        const int16_t omv[2], double lambda, int ord[8])
{
    static const int dx[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
    static const int dy[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
    const int lm = (int)(sqrt(lambda) + 0.5);
    int c[8], j, i;

    for (j = 0; j < 8; j++) {
        int nx = mv[0] + dx[j], ny = mv[1] + dy[j], d;
        if (bi == 1)
            d = b_part_satd_bi(e, mbx, mby, geo, nx, ny, omv[0], omv[1]);
        else if (bi == 2)
            d = b_part_satd_bi(e, mbx, mby, geo, omv[0], omv[1], nx, ny);
        else
            d = part_satd_ref(e, mbx, mby, geo, nx, ny, ref);
        c[j] = d + lm * (mvref_se_bits(nx - pmv[0]) + mvref_se_bits(ny - pmv[1]));
        ord[j] = j;
    }
    for (j = 1; j < 8; j++)
        for (i = j; i > 0 && c[ord[i]] < c[ord[i - 1]]; i--) {
            int t = ord[i]; ord[i] = ord[i - 1]; ord[i - 1] = t;
        }
}

/* The SATD of a candidate's luma prediction plus its header bits at the
 * motion-search lambda: the cost the motion search ranks by (AEG_PPG). */
/* --trellis 1 (x264's meaning): level selection for the winner only -- the
 * decision compares candidates quantised plainly, and the winner's luma and
 * the chroma are coded with it.  2 runs it inside every candidate. */
static int trel_final_on(const struct aeg_encoder *e)
{
    return e->p.trellis == 1;
}

static int cand_scost(struct aeg_encoder *e, int mbx, int mby, int cand, int qp)
{
    const int s = e->stride[0];
    mbenc t;
    cand_enc(e, cand, qp, &t);
    inter_predict_typed(e, mbx, mby, &t);
    return aeg_satd_16x16(e->pred_l, 16, e->srcp[0] + mby * 16 * s + mbx * 16, s) +
           e->me_lambda * inter_hdr_bits(e, &t);
}

/* QP-RD (x264's mb_analyse_qp_rd, subme 10): the decided macroblock is
 * coded again at neighbouring QPs and the full price (distortion plus the
 * coder's bits, mb_qp_delta included) picks its QP (--qp-rd).  The QPs are compared with the decision's lambda (0.65x on P,
 * the anchors' on B): with the plain lambda of the macroblock's QP it is
 * -0.12% against -1.94% (cock, constant QP, p4). */
static double mbt_strength(void);
static int in_slots(const struct aeg_encoder *e);
/* The macroblock tree's QP offset of this macroblock (0 without one). */
static double mbt_off(const struct aeg_encoder *e, int mbx, int mby)
{
    const struct mbt_slot *m;
    const int poc = e->fld ? e->fsv.cur_poc : e->cur_poc;

    if (!e->mbt_on)
        return 0.0;
    m = &e->mbt[poc % in_slots(e)];
    if (e->fld)
        return 0.5 * (m->qp[2 * mby * e->mb_w + mbx] +
                      m->qp[(2 * mby + 1) * e->mb_w + mbx]);
    return m->qp[mby * e->mb_w + mbx];
}

static int qprd_on(const struct aeg_encoder *e)
{
    return e->p.cabac && e->p.qp_rd > 0;
}

/* The winner `b` coded once more at QP q, luma and chroma, into *t and the
 * picture; its price. */
static int64_t qprd_try(struct aeg_encoder *e, int mbx, int mby,
                        const mbenc *b, int q, double lambda, mbenc *t)
{
    if (is_inter_type(b->type)) {
        mbcand save = e->mb_cand[0];
        int k;
        e->mb_cand[0].type = b->type;
        e->mb_cand[0].nparts = b->nparts;
        memcpy(e->mb_cand[0].sub_type, b->sub_type, sizeof(b->sub_type));
        memcpy(e->mb_cand[0].blist, b->blist, sizeof(b->blist));
        for (k = 0; k < 16; k++) {
            e->mb_cand[0].mv[k][0] = b->mv[k][0];
            e->mb_cand[0].mv[k][1] = b->mv[k][1];
            e->mb_cand[0].mv1[k][0] = b->mv1[k][0];
            e->mb_cand[0].mv1[k][1] = b->mv1[k][1];
            e->mb_cand[0].pmv[k][0] = b->pmv[k][0];
            e->mb_cand[0].pmv[k][1] = b->pmv[k][1];
            e->mb_cand[0].pmv1[k][0] = b->pmv1[k][0];
            e->mb_cand[0].pmv1[k][1] = b->pmv1[k][1];
            e->mb_cand[0].ref[k] = b->ref[k];
        }
        /* (the transform size stays the winner's) */
        e->t8_force = b->t8 ? 2 : 1;
        rd_luma(e, mbx, mby, t, q, b->type, 0, lambda, 0);
        e->t8_force = 0;
        e->mb_cand[0] = save;
        inter_predict_typed(e, mbx, mby, t);
        rd_chroma(e, mbx, mby, t, 1, b->chroma_mode, lambda);
    } else {
        if (b->type == AEG_MB_I16x16)
            rd_luma(e, mbx, mby, t, q, AEG_MB_I16x16, b->i16_mode, lambda, -1);
        /* (the modes are decided again: with the winner's modes kept,
         * -20% time and +0.62% at p3 constant QP) */
        else if (b->type == AEG_MB_I4x4)
            rd_luma_i4x4(e, mbx, mby, t, q, lambda);
        else
            rd_luma_i8x8(e, mbx, mby, t, q, lambda);
        rd_chroma(e, mbx, mby, t, 0, b->chroma_mode, lambda);
    }
    return rd_price(e, mbx, mby, t, 0, lambda);
}

/* QP-RD's lambda on P pictures, relative to the mode decision's (qprd) */
#define QPRD_P_LAMBDA 1.32

static void qprd(struct aeg_encoder *e, int mbx, int mby, mbenc *best,
                 double lambda)
{
    const int orig = best->qp, last = e->rd_last_qp;
    /* --qp-rd 2: try the QP the weighted lambda points at (below) and walk at
     * most one step either way from the better of the two.  On the long
     * clips at preset 3 (CRF, against x264 veryslow) that keeps -2.25 of the
     * full walk's -2.36 points for 60 % of its time (+9 % of the encode
     * against +15 %); a one-step walk from the decided QP alone gives -1.62
     * (vtestL wants the long moves the weighting asks for, see
     * docs/STATUS.md). */
    const int fast = e->p.qp_rd >= 2;
    pixel sl[16 * 16], sc[2 * 8 * 8];
    mbenc t, bt;
    int64_t bcost, ocost, cctr, cost;   /* best, decided QP, walk start */
    int dir, last_tried = 0, ctr = orig;
    double pshift = 0.0;

    /* What the macroblock tree wants: its offset o = -s*log2(w) for a
     * macroblock whose error is inherited w times over lowers the QP and,
     * with it, the lambda by w^(s/3) -- short of the w a distortion that
     * counts w times should be priced with.  The rest, w^(1 - s/3) =
     * 2^(o (3 - s) / 3s), goes on the lambda the QPs are compared with here
     * (on the mode decision it is neutral: +0.01%).  Within the picture in
     * full; of the picture's mean offset only 0.3, the rest of that is the
     * rate controller's business.  CRF, long clips, p4 --aq 0, BD-rate
     * against QP-RD off (vtestL/cockL/megaL): none of it +4.97/-2.04/~-0.6
     * (the static background of vtestL is coarsened); all of it
     * -6.84/+0.31/+4.96; relative only -1.14/-1.58/-0.16; 0.3 of the mean
     * -4.43/-1.31/-0.53; 0.6 -7.01/-1.00/+0.93.  Not moving up where the
     * tree lowered the QP instead: +0.79/-0.62/-0.53.
     *
     * P pictures decide their modes with 0.65 times the lambda (see
     * P_MODE_LAMBDA), which, used here, buys every coded macroblock about
     * two QPs below what the weighting asks for.  On content where little is
     * inherited (tree2: much motion, 11 % skipped) that overspends: P
     * macroblocks went 1.34 QPs down against a weighting of 0.66 and the
     * clip lost 1.4 %.  Comparing the QPs with 1.32 times that lambda (about
     * 0.86 of the plain one) fixes it and helps nearly everywhere else.
     * BD-rate of p4 against --qp-rd 0, CQP main+val mean / tree2: decision
     * lambda -0.70 / +1.42; x1.25 -1.21 / -0.31; x1.32 -1.25 / -0.66; x1.40
     * -1.28 / -0.89; plain (x1.54) -1.12 / -1.15.  B pictures' finer lambda
     * (mode_lambda_off) is right as it is: undoing half of it -0.51 / +2.59,
     * making it finer still -1.08 / -0.82 (with x1.25 on P).  Scaling the
     * tree weighting down instead: half -1.02 / +0.11, none -0.73 / -0.82
     * (vtest's static background gives its gain back). */
    if (e->cur_is_p)
        lambda *= QPRD_P_LAMBDA;
    if (e->mbt_on) {
        const double s = mbt_strength();
        const double o = mbt_off(e, mbx, mby) - 0.7 * e->mbt_avg;
        /* lambda * 2^(x/3) is the decision lambda of the QP x away, so this
         * weighting points the walk at orig + o (3 - s) / s */
        pshift = o * (3.0 - s) / s;
        lambda *= pow(2.0, o * (3.0 - s) / (3.0 * s));
    }
    snap_luma(e, mbx, mby, sl);
    snap_chroma(e, mbx, mby, sc);
    cctr = ocost = bcost = rd_price(e, mbx, mby, best, 0, lambda);
    bt = *best;
    if (fast) {
        int c = orig + (int)lrint(pshift);
        c = c < e->p.qp_min ? e->p.qp_min : c > e->p.qp_max ? e->p.qp_max : c;
        if (c != orig) {
            if (c == last)
                last_tried = 1;
            cost = qprd_try(e, mbx, mby, best, c, lambda, &t);
            if (cost < bcost) {
                bcost = cost;
                bt = t;
                ctr = c;
                cctr = cost;
                snap_luma(e, mbx, mby, sl);
                snap_chroma(e, mbx, mby, sc);
            }
        }
    }
    for (dir = 1; dir >= -1; dir -= 2) {
        /* monotone moving away from the previous macroblock's QP, one
         * failure allowed moving towards it (x264 without psy-RD) */
        const int thr = (last < ctr && dir < 0) || (last > ctr && dir > 0);
        int64_t prev = cctr;
        int q, fail = 0;
        for (q = ctr + dir; q >= e->p.qp_min && q <= e->p.qp_max; q += dir) {
            if (q == last)
                last_tried = 1;
            if (q == orig) {
                /* (back over the decided QP from a predicted start: its
                 * price is known, it lost, and it is not a step) */
                fail = ocost < prev ? 0 : fail + 1;
                prev = ocost;
                if (fail > thr)
                    break;
                continue;
            }
            cost = qprd_try(e, mbx, mby, best, q, lambda, &t);
            if (cost < bcost) {
                bcost = cost;
                bt = t;
                snap_luma(e, mbx, mby, sl);
                snap_chroma(e, mbx, mby, sc);
            }
            fail = cost < prev ? 0 : fail + 1;
            prev = cost;
            if (fail > thr || fast)
                break;
            /* (coarser and nothing left to code: coarser still is the same) */
            if (dir > 0 && t.cbp == 0 && t.type != AEG_MB_I16x16)
                break;
        }
    }
    if (!last_tried && last != orig && last != ctr && last >= e->p.qp_min &&
        last <= e->p.qp_max) {
        cost = qprd_try(e, mbx, mby, best, last, lambda, &t);
        if (cost < bcost) {
            bcost = cost;
            bt = t;
            snap_luma(e, mbx, mby, sl);
            snap_chroma(e, mbx, mby, sc);
        }
    }
    *best = bt;
    restore_luma(e, mbx, mby, sl);
    restore_chroma(e, mbx, mby, sc);
}

static void code_mb_ph(struct aeg_encoder *e, int mbx, int mby, mbenc *c);

/* (timed as a phase, see aeg_ph_switch) */
static void code_mb(struct aeg_encoder *e, int mbx, int mby, mbenc *c)
{
    const int prev = aeg_ph_push(AEG_PH_CODE);

    code_mb_ph(e, mbx, mby, c);
    aeg_ph_pop(prev);
}

static void code_mb_ph(struct aeg_encoder *e, int mbx, int mby, mbenc *c)
{
    e->cmemo_gen++;             /* code_chroma()'s results are per macroblock */
    e->lmemo_n = 0;             /* and so are rd_luma()'s */
    /* Both snapshots are written by the first candidate that is evaluated, and
     * at least one always is (a preset above 0 with no legal Intra_16x16 mode
     * returns above), but initialising them keeps the compiler from warning
     * about every memcpy in the snapshot helpers. */
    pixel snap_l[16 * 16] = { 0 }, snap_c[2 * 8 * 8] = { 0 };
    mbenc best, t, retry;
    int64_t best_cost = INT64_MAX;
    /* A B picture is quantised `mode_lambda_off` steps coarser than its
     * anchors, but its modes are chosen with the lambda of the anchors' QP:
     * the coarser quantiser already takes the bits the picture can spare, and
     * pricing its decisions at the same rate-distortion slope as the rest of
     * the sequence keeps it from also giving up the cheap gains (fewer skips
     * and merged partitions that cost little).  0.63x at the default offset
     * of 2: -0.34 % mean, -1.4 % on fixed 3-B groups; at --bqp 0 there is
     * nothing to undo and it is neutral; the same shift of the motion-search
     * lambda: +0.2 %. */
    double lambda = AEG_LAMBDA(c->qp - e->mode_lambda_off);
    /* A P picture's modes are priced at 0.65x the lambda of its QP, i.e.
     * 0.55 * 2^((QP - 12) / 3): what is decided in a reference picture is
     * paid for again by every picture predicted from it, and the 0.85 of the
     * formula is the intra/H.264-JM value (HM's inter pictures use 0.57).
     * Measured on the mode decision only (cock + short): 0.8x -0.90%, 0.65x
     * -1.38%, 0.5x -1.00%; P-only -2.56%, without the tree -2.10%; the same
     * on the trellis lambda as well: -0.08%; on I pictures: +0.08%; another
     * 0.8x on B pictures: -0.08% on cock + short, +0.07% on all five; a
     * skip-only version (the skip check's cost x1.15) gets a third of it.
     * Rate-distortion's other users (trellis, the motion search) keep the
     * plain lambda. */
    if (e->cur_is_p)
        lambda *= P_MODE_LAMBDA;
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

    /* luma: the SAD-best Intra_16x16 modes (the best two from preset 2 on)
     * plus every inter partitioning the motion search produced.  The third
     * mode (AEG_I16N=3) measured -0.05 / -0.06 % better without it at preset
     * 3 for -5 % time, +0.01 / +0.07 % at preset 4 for -3 / -4.5 %; ranking
     * the modes by SATD instead of SAD helped neither two nor one. */
    CM_T0();
    /* Level selection on the winner only (trel_final_on): the decision runs
     * with the plain quantiser, the trellis comes back for the re-code of
     * the luma winner below and for the chroma. */
    const int trel_final = e->p.trellis > 0 && e->p.cabac && trel_final_on(e);
    const int trel_saved = e->p.trellis;
    if (trel_final)
        e->p.trellis = 0;
    int n16 = e->p.preset >= 2 ? e->mb_nmodes : 1;
    {
        /* one mode below preset 4 (re-measured once the Intra_16x16 AC
         * level selection was off: +0.08 / +0.02 % at preset 3 for -4.9 /
         * -2.0 %, +0.07 % at preset 2 for -4.4 %) */
        const int nmax = ENV_INT("AEG_I16N", -1) > 0 ? ENV_INT("AEG_I16N", -1)
                         : e->p.preset >= 4 ? 2 : 1;
        if (n16 > nmax)
            n16 = nmax;
    }
    if (e->mb_inter_ok && ENV_INT("AEG_I16G", 0) > 0 &&
        e->mb_inter_sad * ENV_INT("AEG_I16G", 0) < e->mb_sad * 100)
        n16 = 1;
    /* The best price of each kind, for the intra gates below. */
    int64_t j16 = INT64_MAX, jinter = INT64_MAX;
    /* A B picture prices its inter candidates first and skips every intra
     * candidate when the best of them already costs less than gate_b_intra()
     * percent of the SATD of the best Intra_16x16 prediction (see there). */
    const int gb = e->mb_inter_ok ? gate_b_intra(e) : 0;
    int pass, skip_intra = 0;
    for (pass = 0; pass < 2; pass++) {
        if (pass == (gb ? 1 : 0)) {
            if (gb && jinter != INT64_MAX &&
                jinter * 100 < (int64_t)gb * i16_satd(e, mbx, mby)) {
                skip_intra = 1;
                continue;
            }
            for (i = 0; i < n16; i++) {
                int64_t cost = rd_luma(e, mbx, mby, &t, c->qp, AEG_MB_I16x16,
                                       e->mb_modes[i], lambda, -1);
                cost = full_price(e, mbx, mby, &t, cost, c->chroma_mode, lambda, best_cost);
                        if (cost < j16)
                    j16 = cost;
                if (cost < best_cost) {
                    best_cost = cost;
                    best = t;
                    snap_luma(e, mbx, mby, snap_l);
                }
            }
            CM_LAP(e, 0);
            continue;
        }
        if (e->rd_batch && e->mb_ncand > 0) {
            best_cost = rd_luma_inter_batch(e, mbx, mby, &best, snap_l, best_cost,
                                            lambda, c->qp, c->chroma_mode,
                                            &jinter);
        } else {
            for (i = 0; i < e->mb_ncand; i++) {
                int64_t cost = rd_luma(e, mbx, mby, &t, c->qp, e->mb_cand[i].type, 0,
                                       lambda, i);
                cost = full_price(e, mbx, mby, &t, cost, c->chroma_mode, lambda, best_cost);
                /* Refining a candidate costs a macroblock evaluation per
                 * neighbour, so only the candidates that are still in the
                 * running get it: one that is already worse than the best so
                 * far cannot be rescued by a quarter pel, and a looser test
                 * than this one measured no better (see docs/STATUS.md). */
                if (e->p.cabac && e->p.refine_cand > 0 && e->p.refine_mv > 0 &&
                    cost <= best_cost) {
                    refine_cand(e, mbx, mby, i, c->qp, lambda);
                    cost = rd_luma(e, mbx, mby, &t, c->qp, e->mb_cand[i].type, 0,
                                   lambda, i);
                    cost = full_price(e, mbx, mby, &t, cost, c->chroma_mode, lambda, best_cost);
                }
                if (cost < jinter)
                    jinter = cost;
                if (cost < best_cost) {
                    best_cost = cost;
                    best = t;
                    snap_luma(e, mbx, mby, snap_l);
                }
            }
        }
        CM_LAP(e, 1);
    }
    /* Intra_4x4: its own mode decision per block, so it is a candidate of its
     * own rather than one of the Intra_16x16 prediction modes.  It costs a
     * lot more than Intra_16x16 (sixteen predictions, sixteen transforms and
     * the per-block mode bits), so it is skipped when the motion search found
     * a match that is already twice as good as the best Intra_16x16
     * prediction: sixteen mode decisions cannot turn that around, and this is
     * where most of the encode time of the candidate goes. */
    /* ... and both are skipped as well when the best inter candidate costs
     * less than gate_i48() percent of the best Intra_16x16 one: an
     * Intra_16x16 prediction that far behind is not one the finer intra
     * shapes turn around. */
    if (!skip_intra && j16 != INT64_MAX && jinter != INT64_MAX &&
        jinter * 100 < (int64_t)gate_i48(e) * j16)
        skip_intra = 1;
    if (!skip_intra && !(e->mb_inter_ok &&
          e->mb_inter_sad * ENV_INT("AEG_IGATE", 200) < e->mb_sad * 100)) {
        int64_t cost = rd_luma_i4x4(e, mbx, mby, &t, c->qp, lambda);
        cost = full_price(e, mbx, mby, &t, cost, c->chroma_mode, lambda, best_cost);
        if (cost < best_cost) {
            best_cost = cost;
            best = t;
            snap_luma(e, mbx, mby, snap_l);
        }
        CM_LAP(e, 2);
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
    CM_LAP(e, 3);
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
        const int K = cheap ? 0 : mvref_k(e);
        const int nn = K > 0 ? (K < 8 ? K : 8) : cheap ? 4 : 8;
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
            int ord[8];
            int16_t bmv[2];
            mb_part_geom(&best, p, geo, &b0);
            bmv[0] = best.mv[b0][0];
            bmv[1] = best.mv[b0][1];
            if (K > 0)
                mvref_order(e, mbx, mby, geo, bmv, best.pmv[b0],
                            e->cur_l0_active > 1 ? best.ref[b0] : 0, 0,
                            NULL, lambda, ord);
            for (j = 0; j < nn; j++) {
                int k = K > 0 ? ord[j] : cheap ? cross[j] : j;
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
                    e->mb_cand[0].mv[b][0] = (K > 0 ? bmv[0] : best.mv[b][0]) + dx[k];
                    e->mb_cand[0].mv[b][1] = (K > 0 ? bmv[1] : best.mv[b][1]) + dy[k];
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
        const int K = cheap ? 0 : mvref_k(e);
        const int nn = K > 0 ? (K < 8 ? K : 8) : cheap ? 4 : 8;
        const int npart = mb_part_count(&best);
        mbcand save = e->mb_cand[0];
        const int mb_qp = c->qp;
        int p, l, j, b;

        for (p = 0; p < npart; p++) {
            int geo[4], b0;
            mb_part_geom(&best, p, geo, &b0);
            for (l = 0; l < 2; l++) {
                int ord[8], lists = mb_part_lists(&best, p);
                int16_t bmv[2], omv[2];
                if (!(lists & (1 << l)))
                    continue;
                bmv[0] = l ? best.mv1[b0][0] : best.mv[b0][0];
                bmv[1] = l ? best.mv1[b0][1] : best.mv[b0][1];
                omv[0] = l ? best.mv[b0][0] : best.mv1[b0][0];
                omv[1] = l ? best.mv[b0][1] : best.mv1[b0][1];
                if (K > 0)
                    mvref_order(e, mbx, mby, geo, bmv,
                                l ? best.pmv1[b0] : best.pmv[b0], l,
                                lists == 3 ? 1 + l : 0, omv, lambda, ord);
                for (j = 0; j < nn; j++) {
                    int k = K > 0 ? ord[j] : cheap ? cross[j] : j;
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
                            e->mb_cand[0].mv[b][0] = (K > 0 ? bmv[0] : best.mv[b][0]) + dx[k];
                            e->mb_cand[0].mv[b][1] = (K > 0 ? bmv[1] : best.mv[b][1]) + dy[k];
                        } else {
                            e->mb_cand[0].mv1[b][0] = (K > 0 ? bmv[0] : best.mv1[b][0]) + dx[k];
                            e->mb_cand[0].mv1[b][1] = (K > 0 ? bmv[1] : best.mv1[b][1]) + dy[k];
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

    /* The transform size of an inter winner by rate and distortion: use_t8()
     * picks 4x4 or 8x8 from the SATD / SA8D of the residual; this codes the
     * winner once more with the other size and keeps whichever the full price
     * prefers (x264's analyse_transform_rd, subme 8 and up). */
    if (inter && t8_rd_on(e) && t8_allowed(e, &best)) {
        mbcand save = e->mb_cand[0];
        const int mb_qp = c->qp;
        int b;
        int64_t cost;

        e->mb_cand[0].type = best.type;
        e->mb_cand[0].nparts = best.nparts;
        memcpy(e->mb_cand[0].sub_type, best.sub_type, sizeof(best.sub_type));
        memcpy(e->mb_cand[0].blist, best.blist, sizeof(best.blist));
        for (b = 0; b < 16; b++) {
            e->mb_cand[0].mv[b][0] = best.mv[b][0];
            e->mb_cand[0].mv[b][1] = best.mv[b][1];
            e->mb_cand[0].mv1[b][0] = best.mv1[b][0];
            e->mb_cand[0].mv1[b][1] = best.mv1[b][1];
            e->mb_cand[0].pmv[b][0] = best.pmv[b][0];
            e->mb_cand[0].pmv[b][1] = best.pmv[b][1];
            e->mb_cand[0].pmv1[b][0] = best.pmv1[b][0];
            e->mb_cand[0].pmv1[b][1] = best.pmv1[b][1];
            e->mb_cand[0].ref[b] = best.ref[b];
        }
        e->t8_force = best.t8 ? 1 : 2;
        cost = rd_luma(e, mbx, mby, &t, mb_qp, best.type, 0, lambda, 0);
        cost = full_price(e, mbx, mby, &t, cost, c->chroma_mode, lambda, best_cost);
        e->t8_force = 0;
        if (cost < best_cost) {
            best_cost = cost;
            best = t;
            snap_luma(e, mbx, mby, snap_l);
        }
        c->qp = mb_qp;
        e->mb_cand[0] = save;
        restore_luma(e, mbx, mby, snap_l);
    }

    if (trel_final) {
        const int mb_qp = c->qp;
        e->p.trellis = trel_saved;
        e->cmemo_gen++;     /* the decision's chroma was coded without it */
        /* Level selection only shrinks or removes levels the quantiser set
         * (trel_descent_ex() never touches a zero), so a winner whose luma
         * quantised to nothing would come out of it unchanged: the skip
         * vectors of static content, most of what a B picture codes. */
        int lz = inter;
        for (int b = 0; b < 16 && lz; b++)
            lz = !best.nz[b];
        for (int g = 0; g < 4 && lz && best.t8; g++)
            lz = !best.nz8[g];
        if (lz) {
            /* (the picture and snap_l already hold the winner's luma; the
             * coding would have stamped the macroblock's own QP, where the
             * decision may have left a residual-free winner at another) */
            best.qp = mb_qp;
        } else if (inter) {
            mbcand save = e->mb_cand[0];
            int b;
            e->mb_cand[0].type = best.type;
            e->mb_cand[0].nparts = best.nparts;
            memcpy(e->mb_cand[0].sub_type, best.sub_type, sizeof(best.sub_type));
            memcpy(e->mb_cand[0].blist, best.blist, sizeof(best.blist));
            for (b = 0; b < 16; b++) {
                e->mb_cand[0].mv[b][0] = best.mv[b][0];
                e->mb_cand[0].mv[b][1] = best.mv[b][1];
                e->mb_cand[0].mv1[b][0] = best.mv1[b][0];
                e->mb_cand[0].mv1[b][1] = best.mv1[b][1];
                e->mb_cand[0].pmv[b][0] = best.pmv[b][0];
                e->mb_cand[0].pmv[b][1] = best.pmv[b][1];
                e->mb_cand[0].pmv1[b][0] = best.pmv1[b][0];
                e->mb_cand[0].pmv1[b][1] = best.pmv1[b][1];
                e->mb_cand[0].ref[b] = best.ref[b];
            }
            e->t8_force = best.t8 ? 2 : 1;
            rd_luma(e, mbx, mby, &t, mb_qp, best.type, 0, lambda, 0);
            e->t8_force = 0;
            e->mb_cand[0] = save;
        } else if (best.type == AEG_MB_I16x16) {
            rd_luma(e, mbx, mby, &t, mb_qp, AEG_MB_I16x16, best.i16_mode, lambda, -1);
        } else if (best.type == AEG_MB_I4x4) {
            rd_luma_i4x4(e, mbx, mby, &t, mb_qp, lambda);
        } else {
            rd_luma_i8x8(e, mbx, mby, &t, mb_qp, lambda);
        }
        if (!lz) {
            t.chroma_mode = best.chroma_mode;
            best = t;
            snap_luma(e, mbx, mby, snap_l);
        }
        c->qp = mb_qp;
    }

    CM_LAP(e, 4);
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

            int px, py;

            reset_mb_enc(&s, c->qp);
            s.type = AEG_MB_P16x16;
            s.nparts = 1;
            pskip_mv(e, mbx, mby, &sx, &sy);
            px = sx; py = sy;
            /* MBAFF: it may have to be sent as P_L0_16x16 after all (a
             * skipped pair of the wrong kind, p_skippable), with its real
             * predictor; and a field neighbour's vector, doubled for a frame
             * macroblock, may reach past the borders the search keeps to */
            if (e->mbaff)
                pred16_ref(e, mbx, mby, 0, &px, &py);
            for (b = 0; b < 16; b++) {
                s.mv[b][0] = (int16_t)sx;
                s.mv[b][1] = (int16_t)sy;
                s.pmv[b][0] = (int16_t)px;
                s.pmv[b][1] = (int16_t)py;
            }
            ok = !e->mbaff || mv_me_ok(e, sx, sy);
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
            pixel cbuf[128];
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
    if (qprd_on(e) && (best.cbp != 0 || best.type == AEG_MB_I16x16) &&
        e->mb_nmodes > 0) {
        e->cmemo_gen++;     /* (a fresh chroma memo for the trials) */
        qprd(e, mbx, mby, &best, lambda);
    }
    *c = best;
    /* Hand the macroblock to the shadow coder: its context states, its
     * mb_qp_delta history and its per-block motion vectors are what the next
     * macroblock's decisions are priced against. */
    if (e->p.cabac)
        rd_cabac_bits(e, mbx, mby, c, cur_slice_kind(e), 1);
    CM_LAP(e, 5);
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
        if (l0_syn(e) > 1) {
            if (c->type == AEG_MB_P8x8) {
                for (p = 0; p < 4; p++) {
                    int b0 = (p & 1) * 2 + (p >> 1) * 8;
                    put_ref_te(e, bw, c->ref[b0]);
                }
            } else {
                for (p = 0; p < c->nparts; p++) {
                    int geo[4], b0;
                    mb_part_geom(c, p, geo, &b0);
                    put_ref_te(e, bw, c->ref[b0]);
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
                              luma_ac_nc(e, mbx, mby, c, 0), 1 | e->fscan);
        if (luma_cbp_on(c))
            for (blk = 0; blk < 16; blk++) {
                int rb = ac_ord[blk];
                aeg_cavlc_write_block(bw, c->level[rb],
                                      luma_ac_nc(e, mbx, mby, c, blk), 3 | e->fscan);
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
                    aeg_cavlc_write_block8(bw, c->level8[g], blk | e->fscan8,
                                           luma_nc(e, mbx, mby, c, b));
                    c->nz[b] = (int8_t)aeg_cavlc_nz8(c->level8[g], blk | e->fscan8);
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
                                          luma_nc(e, mbx, mby, c, b), 0 | e->fscan);
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
                                          chroma_nc(e, mbx, mby, c, pl, b), 3 | e->fscan);
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
        *a = left_nz_luma(e, mbx, mby, by);
        *va = 1;
    } else { *a = 0; *va = 0; }
    if (by) { *b = c->nz[blk - 4]; *vb = 1; }
    else if (nb_top_ok(e, mby)) {
        *b = nb_mbB(e, mbx, mby)->nz[12 + bx];
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
        *a = left_nz_chroma(e, mbx, mby, pl, by);
        *va = 1;
    } else { *a = 0; *va = 0; }
    if (by) { *b = c->nz[base + k - 2]; *vb = 1; }
    else if (nb_top_ok(e, mby)) {
        *b = nb_mbB(e, mbx, mby)->nz[base + k + 2];
        *vb = 1;
    } else { *b = 0; *vb = 0; }
}

/* The same for one whole-macroblock block (luma or chroma DC). */
static void cab_nz_mb(struct aeg_encoder *e, int mbx, int mby, int blk,
                      int *a, int *b, int *va, int *vb)
{
    if (mbx) {
        *a = nb_mbA(e, mbx, mby)->nz[blk];
        *va = 1;
    } else { *a = 0; *va = 0; }
    if (nb_top_ok(e, mby)) {
        *b = nb_mbB(e, mbx, mby)->nz[blk];
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
    const aeg_mb *na = mbx ? nb_mbA(e, mbx, mby) : NULL;
    const aeg_mb *nb = nb_top_ok(e, mby) ? nb_mbB(e, mbx, mby) : NULL;

    if (na && is_intra_type(na->type) && na->chroma_mode != 0)
        ctx++;
    if (nb && is_intra_type(nb->type) && nb->chroma_mode != 0)
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
    int ctx = 0;

    if (mbx && nb_mbA(e, mbx, mby)->t8)
        ctx++;
    if (nb_top_ok(e, mby) && nb_mbB(e, mbx, mby)->t8)
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
            int yM;
            const aeg_mb *lm = nb_left(e, mbx, mby, 4 * by, 16, &yM);
            lx = lm->mvd[list][(yM >> 2) * 4 + 3][0];
            ly = mvd_kind(e, lm, lm->mvd[list][(yM >> 2) * 4 + 3][1]);
        }
        if (by) {
            tx = mvd[(by - 1) * 4 + bx][0];
            ty = mvd[(by - 1) * 4 + bx][1];
        } else if (nb_top_ok(e, mby)) {
            const aeg_mb *tm = nb_mbB(e, mbx, mby);
            tx = tm->mvd[list][12 + bx][0];
            ty = mvd_kind(e, tm, tm->mvd[list][12 + bx][1]);
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
    if (l0_syn(e) <= 1 || is_intra_type(c->type))
        return;
    /* P_8x8 has 4 sub-macroblocks (one ref each), otherwise partitions. */
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
        int ref = c->ref[b0];
        /* left: inside MB or left MB */
        if (x0 > 0) {
            refa = cur_ref[y0 * 4 + x0 - 1];
        } else if (mbx > 0) {
            int yM;
            const aeg_mb *lm = nb_left(e, mbx, mby, 4 * y0, 16, &yM);
            if (!is_intra_type(lm->type))
                refa = ref_kind(e, lm, lm->ref_idx[(yM >> 2) * 4 + 3]);
        }
        /* top: inside MB or top MB */
        if (y0 > 0) {
            refb = cur_ref[(y0 - 1) * 4 + x0];
        } else if (nb_top_ok(e, mby)) {
            const aeg_mb *tm = nb_mbB(e, mbx, mby);
            if (!is_intra_type(tm->type))
                refb = ref_kind(e, tm, tm->ref_idx[12 + x0]);
        }
        if (refa > 0) ctx++;
        if (refb > 0) ctx += 2;
        /* CABAC binarises ref_idx as plain unary (U, Table 9-34) whatever
         * num_ref_idx_active is -- no truncation at cMax -- so 1 is the two
         * bins 1,0 and the second one uses ctxIdxInc 4 (9.3.3.1.2). */
        aeg_cabac_decision(cb, 54 + ctx, ref > 0);
        if (ref) {
            /* bin 1 ctxIdxInc 4, bins 2.. ctxIdxInc 5 */
            for (int k = 1; k < ref; k++)
                aeg_cabac_decision(cb, 54 + (k == 1 ? 4 : 5), 1);
            aeg_cabac_decision(cb, 54 + (ref == 1 ? 4 : 5), 0);
        }
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
    int a = 0, b = nb_top_ok(e, mby) ? nb_mbB(e, mbx, mby)->cbp : 0;
    int cbp = c->cbp & 15;

    /* the left 8x8 groups beside group rows 0 and 1 (bits 1 and 3 are the
     * ones read): in an MBAFF frame they may sit in either macroblock of the
     * left pair, in either of its group rows */
    if (mbx) {
        int r;
        for (r = 0; r < 2; r++) {
            int yM;
            const aeg_mb *m = nb_left(e, mbx, mby, 8 * r, 16, &yM);
            a |= ((m->cbp >> (1 + 2 * (yM >> 3))) & 1) << (1 + 2 * r);
        }
    }
    /* A neighbour outside the slice counts as "all four groups coded" (its
     * low nibble is 0x0f) whichever kind of macroblock this is. */
    if (!mbx) a = 0x0f;
    if (!nb_top_ok(e, mby)) b = 0x0f;
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
    /* The neighbours' cbp is stored as two bits (chroma DC, chroma AC), while
     * the context derivation wants the syntax element's value 0..2. */
    int a = mbx ? packed_chroma_cbp(nb_mbA(e, mbx, mby)->cbp) : 0;
    int b = nb_top_ok(e, mby) ? packed_chroma_cbp(nb_mbB(e, mbx, mby)->cbp) : 0;
    int cc = chroma_cbp(c);

    /* A neighbour outside the slice counts as having cbp_chroma = 0: the
     * decoder's stand-in for an unavailable neighbour is 0x7CF (or 0x00F),
     * whose chroma bits are both clear. */
    aeg_cabac_decision(cb, CAB_CBP_CHROMA + (a > 0) + 2 * (b > 0), cc != 0);
    if (cc)
        aeg_cabac_decision(cb, CAB_CBP_CHROMA + 4 + (a == 2) + 2 * (b == 2),
                           cc == 2);
}

/* Whether a 4x4 block has a level at raster position `from` or later (0: the
 * whole block, 1: the AC part -- every position but the DC). */
static inline int blk_any(const dctcoef *p, int from)
{
    dctcoef o = 0;
    int i;
    for (i = from; i < 16; i++)
        o |= p[i];
    return o != 0;
}

/* aeg_cabac_residual() of a block without levels: its coded_block_flag. */
static inline int cab_residual_zero(aeg_cabac *cb, int cbf_ctx)
{
    aeg_cabac_decision(cb, cbf_ctx, 0);
    return 0;
}

/* The residual of one macroblock, in the order the spec decodes it: luma
 * (DC then AC, or 8x8 groups, or 4x4 groups), then chroma DC, then chroma
 * AC. */
static void cabac_mb_residual(struct aeg_encoder *e, aeg_cabac *cb, int mbx,
                              int mby, mbenc *c)
{
    int intra = is_intra_type(c->type);
    dctcoef blk[64];
    int i, g, k;

    if (c->type == AEG_MB_I16x16) {
        int a, b, va, vb;
        for (i = 0; i < 16; i++)
            blk[i] = c->level[AEG_BLK_LUMA_DC][e->scan4[i]];
        cab_nz_mb(e, mbx, mby, AEG_BLK_LUMA_DC, &a, &b, &va, &vb);
        c->nz[AEG_BLK_LUMA_DC] =
            (int8_t)aeg_cabac_residual(cb, blk, 16, e->fcat, cab_cbf_ctx(0, a, b, va, vb, 1));
        if (luma_cbp_on(c)) {
            for (k = 0; k < 16; k++) {
                int nb = ac_ord[k];
                cab_nz4_luma(e, mbx, mby, c, nb, &a, &b, &va, &vb);
                if (!blk_any(c->level[nb], 1)) {
                    c->nz[nb] = (int8_t)cab_residual_zero(
                        cb, cab_cbf_ctx(1, a, b, va, vb, intra));
                    continue;
                }
                for (i = 0; i < 15; i++)
                    blk[i] = c->level[nb][e->scan4[i + 1]];
                c->nz[nb] = (int8_t)aeg_cabac_residual(
                    cb, blk, 15, 1 + e->fcat, cab_cbf_ctx(1, a, b, va, vb, intra));
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
                    blk[i] = c->level8[g][e->scan8[i]];
                /* No coded_block_flag: the coded block pattern already said
                 * this group is coded. */
                cnt = aeg_cabac_residual(cb, blk, 64, 5 + e->fcat, -1);
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
                if (!blk_any(c->level[nb], 0)) {
                    c->nz[nb] = (int8_t)cab_residual_zero(
                        cb, cab_cbf_ctx(2, a, b, va, vb, intra));
                    continue;
                }
                for (i = 0; i < 16; i++)
                    blk[i] = c->level[nb][e->scan4[i]];
                c->nz[nb] = (int8_t)aeg_cabac_residual(
                    cb, blk, 16, 2 + e->fcat, cab_cbf_ctx(2, a, b, va, vb, intra));
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
                cb, blk, 4, 3 + e->fcat, cab_cbf_ctx(3, a, b, va, vb, intra));
        }
        if (cc == 2)
            for (pl = 0; pl < 2; pl++)
                for (k = 0; k < 4; k++) {
                    int nb = AEG_BLK_CHROMA_CB + pl * 4 + k;
                    cab_nz4_chroma(e, mbx, mby, c, pl, k,
                                   &a, &b, &va, &vb);
                    if (!blk_any(c->level[nb], 1)) {
                        c->nz[nb] = (int8_t)cab_residual_zero(
                            cb, cab_cbf_ctx(4, a, b, va, vb, intra));
                        continue;
                    }
                    for (i = 0; i < 15; i++)
                        blk[i] = c->level[nb][e->scan4[i + 1]];
                    c->nz[nb] = (int8_t)aeg_cabac_residual(
                        cb, blk, 15, 4 + e->fcat, cab_cbf_ctx(4, a, b, va, vb, intra));
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
/* MBAFF: ref_idx_l0 then ref_idx_l1 of a field macroblock of a B picture,
 * sent although each list has one frame (mb_field_decoding_flag !=
 * field_pic_flag, 7.3.5.1 / 7.3.5.2), per partition (or non-direct 8x8
 * group) predicting from the list: 0 (the macroblock's parity) or 1.
 * ctxIdxInc = condTermA + 2 condTermB (9.3.3.1.1.6): a neighbouring
 * partition counts when it predicts from the list, is not direct (B_Skip,
 * B_Direct_16x16, a direct 8x8 group) and its index, brought to the current
 * macroblock's kind (a field neighbour of a frame macroblock: > 1), is not
 * 0.  Unary, the second bin on ctxIdxInc 4. */
static int bref_cond(const struct aeg_encoder *e, const aeg_mb *m, int blk,
                     int list)
{
    if (!m || is_intra_type(m->type) || m->type == AEG_MB_B_DIRECT)
        return 0;
    if (m->type == AEG_MB_B_8x8 &&
        b_sub_direct[m->sub_type[(((blk >> 2) >= 2) << 1) + ((blk & 3) >= 2)]])
        return 0;
    if (!(aeg_mb_block_lists(m, blk) & (1 << list)))
        return 0;
    return ref_kind(e, m, list ? m->ref_idx1[blk] : m->ref_idx[blk]) > 0;
}

static int bref_cond_cur(const mbenc *c, int blk, int list)
{
    int g = (((blk >> 2) >= 2) << 1) + ((blk & 3) >= 2), lists;
    if (c->type == AEG_MB_B_8x8) {
        if (b_sub_direct[c->sub_type[g]])
            return 0;
        lists = c->blist[g] ? c->blist[g] : 3;
    } else if (c->type == AEG_MB_B16x8) {
        lists = mb_part_lists(c, (blk >> 2) >= 2);
    } else if (c->type == AEG_MB_B8x16) {
        lists = mb_part_lists(c, (blk & 3) >= 2);
    } else {
        lists = mb_part_lists(c, 0);
    }
    return (lists & (1 << list)) && (list ? c->ref1[blk] : c->ref[blk]) > 0;
}

static void cabac_b_ref_one(struct aeg_encoder *e, aeg_cabac *cb, int mbx,
                            int mby, const mbenc *c, int b0, int list)
{
    const int x0 = b0 & 3, y0 = b0 >> 2;
    const int ref = list ? c->ref1[b0] : c->ref[b0];
    int ctx = 0;

    if (x0 > 0) {
        ctx += bref_cond_cur(c, b0 - 1, list);
    } else if (mbx > 0) {
        int yM;
        const aeg_mb *lm = nb_left(e, mbx, mby, 4 * y0, 16, &yM);
        ctx += bref_cond(e, lm, (yM >> 2) * 4 + 3, list);
    }
    if (y0 > 0)
        ctx += 2 * bref_cond_cur(c, b0 - 4, list);
    else if (nb_top_ok(e, mby))
        ctx += 2 * bref_cond(e, nb_mbB(e, mbx, mby), 12 + x0, list);
    aeg_cabac_decision(cb, 54 + ctx, ref > 0);
    if (ref)
        aeg_cabac_decision(cb, 54 + 4, 0);
}

static void cabac_b_ref_idx(struct aeg_encoder *e, aeg_cabac *cb, int mbx,
                            int mby, const mbenc *c)
{
    int list, p;

    for (list = 0; list < 2; list++) {
        if (c->type == AEG_MB_B_8x8) {
            for (p = 0; p < 4; p++)
                if (!b_sub_direct[c->sub_type[p]] &&
                    ((c->blist[p] ? c->blist[p] : 3) & (1 << list)))
                    cabac_b_ref_one(e, cb, mbx, mby, c, (p & 1) * 2 + (p >> 1) * 8,
                                    list);
        } else {
            for (p = 0; p < part_count(c->type); p++)
                if (mb_part_lists(c, p) & (1 << list)) {
                    int geo[4], b0;
                    mb_part_geom(c, p, geo, &b0);
                    cabac_b_ref_one(e, cb, mbx, mby, c, b0, list);
                }
        }
    }
}

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
    /* A neighbour outside the slice counts as direct: the decoder keeps its
     * "left_type" at zero for an unavailable macroblock and tests
     * IS_DIRECT(left_type - 1), which -1 satisfies in every bit pattern.  So
     * only a neighbour that *is* there and is not direct increments the
     * context.  (A and B as an MBAFF frame locates them.) */
    int ld = !mbx || nb_mbA(e, mbx, mby)->type == AEG_MB_B_DIRECT;
    int td = !nb_top_ok(e, mby) || nb_mbB(e, mbx, mby)->type == AEG_MB_B_DIRECT;
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
        int sf = e->cfld;

        /* The same derivation for P and B -- a neighbour that was skipped
         * does not increment the context, and a neighbour outside the slice
         * counts as skipped -- a B slice on its own block of contexts
         * thirteen entries further along (Table 9-34).  In an MBAFF frame
         * the neighbours are located with the pair's kind as the decoder
         * knows it when it reads the flag: inferred for a top macroblock,
         * and for a bottom one whose top one was skipped (its flag, if any,
         * follows) -- 7.3.4, 7.4.4. */
        if (e->mbaff && (!(mby & 1) || e->mbs[idx - e->mb_w].skip))
            e->cfld = mbaff_infer(e, mbx, mby);
        ls = mbx ? nb_mbA(e, mbx, mby)->skip : 1;
        ts = nb_top_ok(e, mby) ? nb_mbB(e, mbx, mby)->skip : 1;
        e->cfld = sf;
        base += !ls + !ts;
        aeg_cabac_decision(cb, base, skip);
        if (skip) {
            /* A skipped macroblock has no residual, and its neighbours' coded
             * block flags have to see that. */
            memset(c->nz, 0, sizeof(c->nz));
            memset(c->nz8, 0, sizeof(c->nz8));
            return;
        }
    }
    /* mb_field_decoding_flag (contexts 70..72: one per neighbouring pair,
     * left and above, that is a field pair -- 9.3.3.1.1.2) */
    if (e->mbaff_ff >= 0)
        aeg_cabac_decision(cb, mbaff_ff_ctx(e, mbx, mby), e->mbaff_ff);
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
                if ((e->mbaff && e->cfld) || (e->fld && e->bfp_on))
                    cabac_b_ref_idx(e, cb, mbx, mby, c);
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
            if (l0_syn(e) > 1)
                cabac_mb_ref_idx(e, cb, mbx, mby, c);
            cabac_mb_mvd(e, cb, mbx, mby, c, 0);
        }
    } else {
        int cond = 0;
        if (mbx) {
            int t = nb_mbA(e, mbx, mby)->type;
            cond += (t == AEG_MB_I16x16 || t == AEG_MB_IPCM);
        }
        if (nb_top_ok(e, mby)) {
            int t = nb_mbB(e, mbx, mby)->type;
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
    if (c->cbp == 0 && c->type != AEG_MB_I16x16)
        c->qp = e->rd_last_qp;
    /* (an estimate: the top macroblock's skip is not final yet) */
    if (slice == AEG_SLICE_P && p_skippable(e, mbx, mby, c, 0))
        skipped = 1;
    /* In a B slice a direct macroblock without residual is a skipped one: the
     * decoder reconstructs it from the co-located picture's vectors either way,
     * so the skip flag is free bits to leave unspent. */
    if (slice == AEG_SLICE_B && b_skippable(e, mbx, mby, c, 0))
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
    const pixel *p = e->srcp[0] + (mby * 16) * e->stride[0] + mbx * 16;
    int x, y, sum = 0, sum2 = 0;
    for (y = 0; y < 16; y++)
        for (x = 0; x < 16; x++) {
            int v = p[y * e->stride[0] + x];
            sum += v;
            sum2 += v * v;
        }
    /* 256 * var, exact in integers; the caller only needs the logarithm
     * (in 64 bits: 256 * sum2 passes 2^31 for a bright block, though the
     * difference always fits) */
    /* on the 8-bit scale at every depth (an exact 8-bit value for a source
     * scaled up from eight bits), so the AQ offsets mean the same */
    return (int)((((int64_t)sum2 * 256 - (int64_t)sum * sum) / 256) >>
                 (2 * (BIT_DEPTH - 8)));
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

/* Macroblock tree (x264's mbtree): how much of each macroblock's information
 * is inherited by the pictures that predict from it.  A block that later
 * pictures copy from (static background) deserves a finer quantiser than one
 * that is overwritten next frame, because its error is paid again in every
 * picture that references it.
 *
 * Per picture the lookahead gives intra(i) and inter(i) (lowres SATD).  The
 * fraction of block i that is predicted rather than coded anew is
 * 1 - inter/intra; that share of (intra + propagate_in) flows to the blocks
 * of the reference it was predicted from, split over the up to four
 * macroblocks the vector lands on.  Processed from the newest picture back,
 * propagate_in of a picture is complete when its turn comes, and the offset
 * is -strength * log2((intra + propagate_in) / intra). */
static void mbt_propagate(int mb_w, int mb_h, const int *intra,
                          const int *inter, const uint8_t *kind, const int *mv,
                          const float *prop_in, float *in0, float *in1, int w0)
{
    int x, y;
    for (y = 0; y < mb_h; y++)
        for (x = 0; x < mb_w; x++) {
            int i = y * mb_w + x, l;
            int ic = intra[i], pc = inter[i] < ic ? inter[i] : ic;
            float amount;
            if (ic <= 0 || pc >= ic)
                continue;
            amount = (float)((prop_in[i] + ic) * (double)(ic - pc) / ic);
            for (l = 0; l < 2; l++) {
                float *dst = l ? in1 : in0, a;
                int px, py, mx, my, fx, fy;
                if (kind[i] != 2 && kind[i] != l)
                    continue;
                if (!dst)
                    continue;
                a = kind[i] == 2 ? amount * (l ? 64 - w0 : w0) / 64.0f : amount;
                /* lowres half-sample units: 16 per macroblock */
                px = 16 * x + mv[4 * i + 2 * l];
                py = 16 * y + mv[4 * i + 2 * l + 1];
                mx = px >> 4; my = py >> 4; fx = px & 15; fy = py & 15;
#define MBT_ADD(X, Y, W) do {                                              \
                if ((X) >= 0 && (X) < mb_w && (Y) >= 0 && (Y) < mb_h && ((W) != 0)) \
                    dst[(Y) * mb_w + (X)] += a * (W) / 256.0f;             \
            } while (0)
                MBT_ADD(mx, my, (16 - fx) * (16 - fy));
                MBT_ADD(mx + 1, my, fx * (16 - fy));
                MBT_ADD(mx, my + 1, (16 - fx) * fy);
                MBT_ADD(mx + 1, my + 1, fx * fy);
#undef MBT_ADD
            }
        }
}

static int in_slots(const struct aeg_encoder *e);

/* Mean offset (rounded) the macroblock tree gives the current picture. */
static int mbt_mean(const struct aeg_encoder *e)
{
    const struct mbt_slot *m;
    double t = 0.0;
    int n = e->mb_count, i;

    if (!e->la_depth || !e->cur_is_ref)
        return 0;
    m = &e->mbt[e->cur_poc % in_slots(e)];
    if (!m->qp || m->qd != e->cur_poc)
        return 0;
    for (i = 0; i < n; i++)
        t += floorf(m->qp[i] + 0.5f);
    return (int)floor(t / n + 0.5);
}

/* CRF's fractional QP: round(frac * n) macroblocks are coded one step
 * coarser.  They form one run (in raster order, wrapping), so the picture
 * pays for two QP changes rather than one per macroblock, and the run starts
 * somewhere else in every picture so that no region is always the coarse
 * one. */
static void crf_dither(struct aeg_encoder *e, int poc)
{
    int n = e->mb_count, k, s, i;

    if (!e->crf_on || e->crf_frac <= 0.0)
        return;
    k = (int)(e->crf_frac * n + 0.5);
    s = (int)(((unsigned)poc * 2654435761u) % (unsigned)n);
    for (i = 0; i < k; i++) {
        int j = (s + i) % n, d = e->mb_dqp[j] + 1;
        e->mb_dqp[j] = (int8_t)(d > AQ_MAX_DQP ? AQ_MAX_DQP : d);
    }
}

/* The offsets the macroblock tree gave this picture when it was queued (a
 * B picture nothing predicts from has none: nothing inherits its error). */
static void mbt_apply(struct aeg_encoder *e)
{
    const struct mbt_slot *m;
    int n = e->mb_count, i;
    /* (a field: the frame's offsets, two macroblock rows to one) */
    const int poc = e->fld ? e->fsv.cur_poc : e->cur_poc;

    e->mbt_avg = 0.0;
    e->mbt_on = 0;
    if (!e->la_depth || !e->cur_is_ref)
        return;
    m = &e->mbt[poc % in_slots(e)];
    if (!m->qp || m->qd != poc)
        return;
    {
        const int fn = e->fld ? 2 * n : n;    /* (the frame's offsets) */
        double sum = 0.0;
        for (i = 0; i < fn; i++)
            sum += m->qp[i];
        e->mbt_avg = sum / fn;
        e->mbt_on = 1;
    }
    for (i = 0; i < n; i++) {
        float q = m->qp[i];
        int d;
        if (e->fld) {
            int x = i % e->mb_w, y = i / e->mb_w;
            q = 0.5f * (m->qp[2 * y * e->mb_w + x] +
                        m->qp[(2 * y + 1) * e->mb_w + x]);
        }
        d = e->mb_dqp[i] + (int)floorf(q + 0.5f);
        if (d < -AQ_MAX_DQP) d = -AQ_MAX_DQP;
        if (d > AQ_MAX_DQP) d = AQ_MAX_DQP;
        e->mb_dqp[i] = (int8_t)d;
    }
}

/* QP of one macroblock of the current picture. */
static int mb_qp_of(const struct aeg_encoder *e, int mbx, int mby)
{
    int qp = e->cur_qp + e->mb_dqp[mby * e->mb_w + mbx];
    if (qp < e->p.qp_min) qp = e->p.qp_min;
    if (qp > e->p.qp_max) qp = e->p.qp_max;
    return qp;
}

/* VBV/HRD: how many bits the picture about to be coded may take.  The channel
 * has delivered `vbv_rate_bits` since the last picture; the buffer cannot hold
 * more than B, so a burst is capped at the buffer size (a picture larger than
 * that could never be sent at rate R). */
static double vbv_available(const struct aeg_encoder *e)
{
    double level = e->vbv_full + e->vbv_rate_bits * 0.5 * e->vbv_gap;

    if (level > e->vbv_buf_bits)
        level = e->vbv_buf_bits;
    return level;
}

static const int *la_intra(struct aeg_encoder *e, int d);

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
     * picture, for the reason given at last_anchor_bits; an intra or a P
     * picture from the last one of its own type. */
    if (e->cur_is_b && e->last_anchor_bits > 0 && e->last_anchor_qp >= QP_MIN) {
        ref_bits = e->last_anchor_bits;
        ref_qp = e->last_anchor_qp;
    } else if (e->cur_is_intra && e->last_i_bits > 0) {
        ref_bits = e->last_i_bits;
        ref_qp = e->last_i_qp;
    } else if (!e->cur_is_intra && !e->cur_is_b) {
        if (e->last_p_bits > 0) {
            ref_bits = e->last_p_bits;
            ref_qp = e->last_p_qp;
        } else if (e->last_i_bits > 0) {
            /* the first P picture: a quarter of the intra picture, which errs
             * high (x264's measure 1/16 on a static scene) */
            ref_bits = e->last_i_bits / 4;
            ref_qp = e->last_i_qp;
        }
    }
    if (!e->vbv_active)
        return;
    /* The B picture's QP already carries the B offset, so the prediction has to
     * start from the QP the encoder will actually quantise with. */
    avail = vbv_available(e);
    if (e->crf_on && !e->abr_on) {
        /* Capped CRF has no rate target pulling the level back, so a run of
         * pictures that each fit could still drain the buffer until one the
         * prediction underestimates no longer does.  A picture may therefore
         * only spend what lies above half the buffer (at least half of one
         * picture's delivery, so that the QP does not jump to the top). */
        double lim = avail - 0.5 * e->vbv_buf_bits;
        double low = 0.5 * e->vbv_rate_bits;

        if (lim < low)
            lim = low < 0.9 * avail ? low : 0.9 * avail;
        avail = lim;
    }
    if (avail < 1.0)
        avail = 1.0;
    /* Two predictions of the QP the picture needs to fit: the size
     * predictor's (the smallest qscale with its estimate <= avail) and the
     * last picture of the type's, scaled by the QP difference.  The smaller
     * one decides: a picture both expect to overflow is raised, and one only
     * a single predictor worries about is left alone -- a stream that fits
     * its buffer must come out byte for byte as it would without one, and
     * the re-coding in frame_body() catches what both miss. */
    q_min = -1e9;
    {
        int have = 0;
        double qs = 1e9, qr = 1e9;

        if (e->cur_satd > 0) {
            int t = e->cur_is_intra ? 0 : e->cur_is_b ? 2 : 1;
            double need = (e->vbv_coef[t] * (double)e->cur_satd +
                           e->vbv_off[t]) / (e->vbv_cnt[t] * avail);
            qs = 12.0 + 6.0 * log2(need / 0.85);
            have = 1;
        }
        if (ref_bits > 0 && ref_qp >= QP_MIN) {
            /* the macroblock tree moves the picture's QP by its mean offset,
             * and the pictures the prediction starts from carry theirs
             * (vbv_update) */
            pred = (double)ref_bits *
                   pow(2.0, (double)(ref_qp - e->cur_qp - e->mbt_qoff) / 6.0);
            qr = pred <= avail ? -1e9
               : (double)ref_qp - 6.0 * log2(avail / (double)ref_bits);
            have = 1;
        }
        if (!have)
            return;
        q_min = qs < qr ? qs : qr;
    }
    if ((double)(e->cur_qp + e->mbt_qoff) < q_min) {
        int q = (int)ceil(q_min) - e->mbt_qoff;

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
    e->last_pic_qp = e->cur_qp + e->mbt_qoff;
    if (e->cur_satd >= 10 && bits > 0) {
        /* x264's update: the new coefficient within 1.5x of the running one,
         * the rest of the picture's bits going to the offset (a picture can
         * cost bits its difference does not explain: noise coded at a low QP) */
        int t = e->cur_is_intra ? 0 : e->cur_is_b ? 2 : 1;
        double bq = (double)bits * 0.85 *
                    pow(2.0, (e->cur_qp + e->mbt_qoff - 12) / 6.0);
        double var = (double)e->cur_satd;
        double oc = e->vbv_coef[t] / e->vbv_cnt[t];
        double oo = e->vbv_off[t] / e->vbv_cnt[t];
        double nc = (bq - oo) / var, ncc, no;

        if (nc < 1.5 / 4.0) nc = 1.5 / 4.0;          /* coeff_min */
        ncc = nc < oc / 1.5 ? oc / 1.5 : nc > oc * 1.5 ? oc * 1.5 : nc;
        no = bq - ncc * var;
        if (no >= 0.0)
            nc = ncc;
        else
            no = 0.0;
        e->vbv_cnt[t] = e->vbv_cnt[t] * 0.5 + 1.0;
        e->vbv_coef[t] = e->vbv_coef[t] * 0.5 + nc;
        e->vbv_off[t] = e->vbv_off[t] * 0.5 + no;
    }
    if (e->cur_is_intra) {
        e->last_i_bits = bits;
        e->last_i_qp = e->cur_qp + e->mbt_qoff;
    } else if (!e->cur_is_b) {
        e->last_p_bits = bits;
        e->last_p_qp = e->cur_qp + e->mbt_qoff;
    }
    if (!e->cur_is_b) {
        e->last_anchor_bits = bits;
        e->last_anchor_qp = e->cur_qp + e->mbt_qoff;
    }
    if (!e->vbv_active)
        return;
    if (e->vbv_f1 > 0 && e->vbv_f1 < bits) {
        /* a field pair: the first field leaves the buffer at the frame's
         * time, the second a tick later */
        level = vbv_available(e) - (double)e->vbv_f1;
        if (level < 0.0) {
            e->vbv_underflows++;
            level = 0.0;
        }
        level += 0.5 * e->vbv_rate_bits;
        if (level > e->vbv_buf_bits)
            level = e->vbv_buf_bits;
        level -= (double)(bits - e->vbv_f1);
        e->vbv_gap = 1;
    } else {
        level = vbv_available(e) - (double)bits;
        e->vbv_gap = 2;
    }
    e->vbv_f1 = 0;
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

/* The QP below which the B-picture offset is faded out (see the B picture's QP
 * in aeg_encoder_encode): at QP 12 and above it is applied in full. */
#define B_QP_FADE 12

static double qp2qs(double q) { return 0.85 * pow(2.0, (q - 12.0) / 6.0); }
static double qs2qp(double s) { return 12.0 + 6.0 * log2(s / 0.85); }

/* x264's qpstep 4: how far an anchor's qscale may move from the last one */
#define ABR_LSTEP 1.5874010519681994
/* the first keyframe's share of the budget, in pictures (see abr_intra_q) */
#define ABR_I_PICS 12.0
/* below this intra estimate per macroblock the first picture is flat */
#define ABR_FLAT 100
/* a P picture under 1/ABR_HELD of the blurred complexity is held */
#define ABR_HELD 16.0

/* Two-pass: the planned qscale of picture poc, pulled towards the plan's
 * total by how far the pictures coded so far are from what it expected
 * (x264's second pass).  The pull is weak at the start and grows towards the
 * end of the stream, where there is no time left to make up a difference
 * gradually. */
/* The plan covers a picture only if pass 2 codes it as the same kind (intra,
 * P, B) as pass 1 did: a plan made for a P picture says nothing useful about
 * an intra one where pass 2 put a GOP boundary or a scene cut (-g 1 in pass
 * 2 overspent 100% following it).  Such pictures fall back to the one-pass
 * controller. */
static int plan_has(const struct aeg_encoder *e, int poc, int cls)
{
    return poc >= 0 && poc < e->plan_n && e->plan_q[poc] >= 0.0 &&
           e->plan_cls[poc] == cls;
}

static int cur_cls(const struct aeg_encoder *e)
{
    return e->cur_is_intra ? 0 : e->cur_is_b ? 2 : 1;
}

static double plan_qs(const struct aeg_encoder *e, int poc)
{
    double qs = qp2qs(e->plan_q[poc]);
    double buf = 2.0 * e->abr_bpf * e->abr_fps, diff, pos, sc, k;

    pos = e->plan_total > 0.0 ? e->plan_exp_sum / e->plan_total : 1.0;
    sc = sqrt((pos < 1.0 ? 1.0 - pos : 0.0) * e->plan_n);
    buf *= 0.5 * (sc > 0.5 ? sc : 0.5);
    diff = e->abr_total - e->plan_exp_sum;
    k = (buf - diff) / buf;
    if (k < 0.5) k = 0.5;
    if (k > 2.0) k = 2.0;
    qs /= k;
    if (e->abr_frames >= e->abr_fps && e->plan_exp_sum >= 1.0) {
        double w = 100.0 * e->abr_frames / e->plan_n;
        if (w > 1.0) w = 1.0;
        qs *= pow(e->abr_total / e->plan_exp_sum, w);
    }
    return qs;
}

/* The QP of a P picture under ABR (x264's rate_estimate_qscale, one pass):
 * the qscale that the bits wanted so far over the complexity spent so far
 * says this picture's complexity deserves, scaled up or down by how far the
 * total is from the target (by at most 2x, the tolerance growing with the
 * square root of the time coded), and kept within qpstep of the last P
 * picture -- one step further in the direction the total needs. */
static double abr_anchor_q(struct aeg_encoder *e, int64_t c, int poc)
{
    double rceq, qs, overflow = 1.0, lmin, lmax;

    if (c >= 0 && e->crf_qc > 0.0 && e->crf_cnt > 0.0 &&
        (double)c * ABR_HELD < e->crf_sum / e->crf_cnt && !plan_has(e, poc, 1)) {
        /* A held picture (a repeat, or next to nothing moving): whatever
         * QP it gets it costs little more than its skip run, so a finer one
         * only re-codes the same picture again -- 20..37 kbit a picture on
         * the tree clip, where the falling blur took the QP from 23 to 17.
         * It keeps the last P picture's qscale, as x264's second pass does
         * for a P picture without texture, and leaves the blur alone. */
        e->abr_rceq = e->abr_last_rceq;
        return qs2qp(e->abr_last_pqs);
    }
    if (e->crf_qc <= 0.0) {
        rceq = 1.0;                       /* the tree owns the distribution */
    } else if (c >= 0) {
        e->crf_sum = e->crf_sum * 0.5 + (double)c;
        e->crf_cnt = e->crf_cnt * 0.5 + 1.0;
        rceq = pow(e->crf_sum / e->crf_cnt, e->crf_qc);
    } else {
        rceq = e->abr_last_rceq;
    }
    if (!(rceq >= 1.0))
        rceq = 1.0;
    e->abr_rceq = e->abr_last_rceq = rceq;
    qs = rceq * e->abr_cplxr / e->abr_wanted;
    if (e->abr_frames > 0) {
        double t = e->abr_frames / e->abr_fps;
        double want = e->abr_frames * e->abr_bpf;
        double buf = 2.0 * e->abr_bpf * e->abr_fps * (t > 1.0 ? sqrt(t) : 1.0);

        overflow = 1.0 + (e->abr_total - want) / buf;
        if (overflow < 0.5) overflow = 0.5;
        if (overflow > 2.0) overflow = 2.0;
        qs *= overflow;
    }
    lmin = e->abr_last_pqs / ABR_LSTEP;
    lmax = e->abr_last_pqs * ABR_LSTEP;
    if (overflow > 1.1 && e->abr_frames > 3)
        lmax *= ABR_LSTEP;
    else if (overflow < 0.9)
        lmin /= ABR_LSTEP;
    /* Far off the target the step limit only delays what the rate factor
     * already knows: under camera noise the estimates miss what the first
     * P pictures cost, and climbing out four steps an anchor cost a 40
     * frame stream 60% of its budget.  It is lifted in the direction the
     * total needs. */
    if (overflow > 1.5)
        lmax = 1e30;
    else if (overflow < 0.67)
        lmin = 0.0;
    if (qs < lmin) qs = lmin;
    if (qs > lmax) qs = lmax;
    if (plan_has(e, poc, 1))
        qs = plan_qs(e, poc);
    e->abr_last_pqs = qs;
    return qs2qp(qs);
}

/* An intra picture under ABR: the anchors' recent average (the caller's q)
 * or the plan's; its own intra estimate is what its bits are measured
 * against afterwards. */
static double abr_intra_q(struct aeg_encoder *e, double q, int poc)
{
    int64_t c = -1;

    if (e->p.bframes > 0) {
        const int *ic = la_intra(e, poc);
        int i;
        if (ic)
            for (c = 0, i = 0; i < e->mb_count; i++)
                c += ic[i];
    } else {
        c = e->cur_satd;
    }
    e->abr_rceq = e->crf_qc <= 0.0 ? 1.0
                : c > 0 ? pow((double)c, e->crf_qc) : e->abr_last_rceq;
    if (!(e->abr_rceq >= 1.0))
        e->abr_rceq = 1.0;
    if (plan_has(e, poc, 0)) {
        q = qs2qp(plan_qs(e, poc));
    } else if (e->abr_frames > 0 && e->abr_prev_intra) {
        /* An intra picture after an intra picture (all-intra, -g 1): there
         * are no anchors whose average to take, and the QP would never move
         * (+39% at -g 1).  As in x264, it is rate-controlled like an anchor,
         * its intra estimate as its complexity. */
        q = abr_anchor_q(e, c, poc);
    } else if (e->abr_frames == 0 && c > ABR_FLAT * e->mb_count) {
        /* The first picture has no history to go by, and a fixed start QP
         * is right for one bitrate only: at a low rate the first keyframe
         * alone can take most of a short stream's budget.  It is given the
         * QP at which its intra estimate predicts ABR_I_PICS pictures'
         * worth of bits instead (bits * qscale / estimate is about 1 at the
         * QPs that matter, docs/STATUS.md), and the P pictures and the rate
         * factor's guess start from there.  A flat picture (a fade from
         * black: 19 per macroblock where real content measured 500..1300)
         * says nothing about what follows and keeps the fixed start. */
        double k = ABR_I_PICS;
        double base = (double)e->mb_count * (e->p.bframes > 0 ? 120.0 : 80.0);
        q = qs2qp((double)c / (k * e->abr_bpf)) + e->p.i_qp_offset;
        if (q < QP_MIN) q = QP_MIN;
        if (q > 51.0) q = 51.0;
        e->crf_pqp = q * e->crf_pnorm;
        e->abr_last_pqs = qp2qs(q);
        e->abr_cplxr = e->abr_bpf * qp2qs(q) /
                       (e->crf_qc > 0.0 ? pow(base, e->crf_qc) : 1.0);
    }
    return q;
}

/* After each picture: what it says about the rate factor.  Its bits are
 * normalised to the model QP it was given -- the B and intra offsets left out,
 * a VBV or qp range clamp kept in -- and to its complexity term. */
static void rc_update(struct aeg_encoder *e, int bits)
{
    double qn;

    if (!e->abr_on)
        return;
    qn = e->abr_qm + (double)(e->cur_qp - e->abr_qpre);
    {
        double t = (double)bits * qp2qs(qn) / e->abr_rceq;
        /* The stream's first picture leaves the rate factor where it was:
         * it is sized from an estimate before there is any evidence, its
         * term is several anchors' worth while it would be wanted only one
         * picture's budget, and counted in it drags the first seconds
         * coarse (tree: -0.75 dB against CRF counted in, -0.42 left out).
         * Its bits still count through the overflow correction.  Later
         * intra pictures (scene cuts, GOP) count like any other: left out,
         * pictures held five frames with an IDR every 25 overspent 53%. */
        if (e->cur_is_intra && e->abr_frames == 0)
            e->abr_wanted += t * e->abr_wanted / e->abr_cplxr;
        else
            e->abr_wanted += e->abr_bpf;
        e->abr_cplxr += t;
    }
    e->abr_total += (double)bits;
    e->abr_frames++;
    e->abr_prev_intra = e->cur_is_intra;
    e->plan_exp_sum += plan_has(e, e->cur_poc, cur_cls(e))
                     ? e->plan_bits[e->cur_poc]
                                               : (double)bits;
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
static const int *la_intra(struct aeg_encoder *e, int d);
static int64_t la_cost_ci(struct aeg_encoder *e, int p0, int p1, int b,
                          int *intra);

/* How badly picture d predicts from d - 1 on the lookahead's lowres
 * estimates: the motion-searched cost (each block at the cheaper of inter
 * and intra) over the all-intra cost, 0 (static) .. 1 (nothing to gain);
 * -1 when the pair is not in the ring. */
static double scene_ratio(struct aeg_encoder *e, int d)
{
    const int slots = in_slots(e), n = e->mb_count;
    const int *ic;
    int64_t icost = 0, pcost;
    int i;

    if (d < 1 || d - 1 < e->in_count - slots || d >= e->in_count ||
        !e->in_low[d % slots].mem || !e->in_low[(d - 1) % slots].mem)
        return -1.0;
    ic = la_intra(e, d);
    if (!ic)
        return -1.0;
    for (i = 0; i < n; i++)
        icost += ic[i];
    if (icost <= 0)
        return 0.0;
    pcost = la_cost_ci(e, d - 1, d, d, NULL);
    return (double)pcost / (double)icost;
}

/* A cut is a spike: the picture predicts badly from the one before (ratio
 * >= 0.5) and much worse (by 0.45) than both its neighbours do from theirs.
 * Fast motion raises the ratio of a run of pictures (cockatoo: 0.4-0.77 for
 * pictures with no cut, at most 0.32 above the picture before; x264's
 * absolute test, ratio >= 1 - bias with the bias 0.1-0.4, cuts it there), a
 * flash raises two in a row; a cut between two shots stands alone
 * (Megamind: 0.82-0.86 against 0.09-0.24 around it; the synthetic cut of
 * the conformance test, smooth patterns that are cheap to code intra: 0.60
 * against 0.03).  The scenecut parameter scales both margins (120 = 1x;
 * larger cuts less often, as with the zero-motion test).  -1: no estimate,
 * use the zero-motion test. */
static int scene_cut_lowres(struct aeg_encoder *e, int display)
{
    const double k = 120.0 / e->p.scenecut;      /* (default 120 -> 1) */
    const double r = scene_ratio(e, display);
    double rp, rn, nb;

    if (r < 0.0)
        return -1;
    rp = scene_ratio(e, display - 1);
    rn = scene_ratio(e, display + 1);
    nb = rp > rn ? rp : rn;
    if (nb < 0.0)
        nb = 0.0;
    return r >= 1.0 - 0.5 * k && r - nb >= 0.45 * k;
}

static int is_scene_cut(struct aeg_encoder *e, int is_idr, int display)
{
    int slots = in_slots(e);
    const pixel *cur, *ref;
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
        ref = e->refpl[0][0];
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
        if (!ENV_INT("AEG_SCOLD", 0)) {
            int r = scene_cut_lowres(e, display);
            if (r >= 0)
                return r;
        }
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

/* Frame threads: how far each picture buffer's reference is (slot_rows),
 * shared by the threads coding pictures from it. */
struct ft_sync {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int rows[AEG_NPIC];
    unsigned gen[AEG_NPIC];  /* bumped when a picture restarts (rows -> 0) */
    int readers[AEG_NPIC];   /* pictures reading the current generation   */
    int stale[AEG_NPIC];     /* ... and an older one, not stopped yet      */
    int wp_hist;             /* the last weighted-prediction analysis used
                              * weights: do not run ahead of the next one */
    int64_t wait_ns;
    int64_t restarts;
};

/* A frame-thread picture starts again from the copy it was launched with. */
#define FT_RESTART (-100)

/* This picture stops reading its references (under fsync->mu): a picture
 * restarting waits until nothing reads what it is about to overwrite. */
static void ft_release(struct aeg_encoder *e)
{
    struct ft_sync *f = e->fsync;
    int l;

    if (!e->ft_reg)
        return;
    for (l = 0; l < AEG_MAX_REFS; l++) {
        int sl = e->ft_gslot[l];
        if (sl < 0)
            continue;
        if (f->gen[sl] == e->ft_gen[l])
            f->readers[sl]--;
        else
            f->stale[sl]--;
    }
    e->ft_reg = 0;
    pthread_cond_broadcast(&f->cv);
}

/* Has a reference this picture reads restarted since it began reading it?
 * (Under fsync->mu, or racy-but-safe outside: generations only grow.) */
static int ft_refs_changed(const struct aeg_encoder *e)
{
    int l;
    for (l = 0; l < AEG_MAX_REFS; l++)
        if (e->ft_gslot[l] >= 0 &&
            __atomic_load_n(&e->fsync->gen[e->ft_gslot[l]], __ATOMIC_ACQUIRE) !=
                e->ft_gen[l])
            return 1;
    return 0;
}

static void slot_publish(struct aeg_encoder *e, int s, int rows)
{
    e->slot_rows[s] = rows;
    if (e->fsync) {
        pthread_mutex_lock(&e->fsync->mu);
        /* rows computed from a reference that has since restarted are not
         * final: this picture restarts as well (at its next row) */
        if (e->ft_abort || ft_refs_changed(e) || (e->ft_spec && rows >= e->mb_h))
            e->ft_abort = 1;
        else
            __atomic_store_n(&e->fsync->rows[s], rows, __ATOMIC_RELEASE);
        pthread_cond_broadcast(&e->fsync->cv);
        pthread_mutex_unlock(&e->fsync->mu);
    }
}

/* Wait until buffer s holds at least `rows` final rows (frame threads; a
 * serial encoder has coded every reference completely already). */
static int ref_wait(struct aeg_encoder *e, int s, int rows)
{
    struct ft_sync *f = e->fsync;
    int64_t t0;
    int rc = 0;

    if (!f || s < 0 || __atomic_load_n(&f->rows[s], __ATOMIC_ACQUIRE) >= rows)
        return 0;
    t0 = aeg_now_ns();
    pthread_mutex_lock(&f->mu);
    while (__atomic_load_n(&f->rows[s], __ATOMIC_ACQUIRE) < rows) {
        if (ft_refs_changed(e)) {
            rc = -1;
            break;
        }
        pthread_cond_wait(&f->cv, &f->mu);
    }
    f->wait_ns += aeg_now_ns() - t0;
    pthread_mutex_unlock(&f->mu);
    return rc;
}

/* ------------------------------------------------------------------ */
/* Row-by-row post-processing of a reference picture                   */
/* ------------------------------------------------------------------ */
/* A reference picture is deblocked, padded, half-resolution filtered and
 * half-pel interpolated a few rows behind the decision pass instead of after
 * it, so that a picture predicting from it can start before it is finished
 * (frame threads; slot_rows says how far it is).  The order of every step is
 * the order of the whole-picture version, so the samples are the same:
 *
 *  - macroblock row k is deblocked once row k+1 is decided, because the
 *    intra prediction of row k+1 reads row k unfiltered;  in raster order
 *    that is exactly aeg_deblock_frame();
 *  - deblocking row k+1 still changes the bottom three lines of row k, so
 *    row k is final (and padded, and filtered down) after that;
 *  - the half-pel rows of row k read three lines into row k+1, so they
 *    follow once row k+1 is final.
 *
 * The filter reads the decisions as the bitstream pass leaves them (QP of a
 * macroblock without residual, skip, transform flag, non-zero counts); the
 * bitstream pass runs after the whole decision pass, so rowpost_finalize
 * applies the same rules to a copy as the rows come (the decision pass of
 * the following rows must keep reading the decision values). */
static void rowpost_finalize_mb(struct aeg_encoder *e, int mbx, int mby,
                                int slice)
{
    int i;
    {
        int idx = mby * e->mb_w + mbx;
        const mbenc *t = &e->mb_encs[idx];
        aeg_mb *d = &e->rp_mbs[idx];
        int qp = t->qp;

        memcpy(d, &e->mbs[idx], sizeof(*d));
        if (e->mbaff)
            e->cfld = d->fld;           /* (pskip_mv reads the neighbours) */
        if (e->p.cabac) {
            int skipped = 0;
            if (t->cbp == 0 && t->type != AEG_MB_I16x16)
                qp = e->rp_qp;
            if (slice == AEG_SLICE_P &&
                p_skippable(e, mbx, mby, t,
                            (mby & 1) && e->rp_mbs[idx - e->mb_w].skip))
                skipped = 1;
            if (slice == AEG_SLICE_B &&
                b_skippable(e, mbx, mby, t,
                            (mby & 1) && e->rp_mbs[idx - e->mb_w].skip))
                skipped = 1;
            d->skip = (uint8_t)skipped;
            d->t8 = (uint8_t)mb_t8_coded(e, t, skipped);
            for (i = 0; i < AEG_BLK_COUNT; i++)
                d->nz[i] = skipped ? 0 : t->nz[i];
            for (i = 0; i < 4; i++)
                d->nz8[i] = (int8_t)(!skipped && t->nz8[i] != 0);
        } else {
            if (!luma_cbp_on(t) && !chroma_cbp(t) && t->type != AEG_MB_I16x16)
                qp = e->rp_qp;
        }
        d->qp = (int8_t)qp;
        e->rp_qp = qp;
        e->cfld = 0;
    }
}

static void rowpost_finalize(struct aeg_encoder *e, int mby, int slice)
{
    int mbx;
    for (mbx = 0; mbx < e->mb_w; mbx++)
        rowpost_finalize_mb(e, mbx, mby, slice);
}

/* Left/right padding of macroblock row b (all three planes), the rows above
 * the picture with row 0 and the rows below it with the last one. */
static void rowpost_pad(struct aeg_encoder *e, int b)
{
    int i, y;

    for (i = 0; i < 3; i++) {
        pixel *p = e->recp[i];
        int st = e->stride[i], w = i ? e->cw : e->w, h = i ? e->ch : e->h;
        int bh = i ? 8 : 16, y0 = b * bh, y1 = y0 + bh;

        if (y1 > h || b == e->mb_h - 1)
            y1 = h;
        for (y = y0; y < y1; y++) {
            aeg_pixel_fill(p + y * st - AEG_PAD, p[y * st], AEG_PAD);
            aeg_pixel_fill(p + y * st + w, p[y * st + w - 1], AEG_PAD);
        }
        if (b == 0)
            for (y = 1; y <= AEG_PAD; y++)
                memcpy(p - y * st - AEG_PAD, p - AEG_PAD,
                       (size_t)(w + 2 * AEG_PAD) * sizeof(pixel));
        if (b == e->mb_h - 1)
            for (y = 1; y <= AEG_PAD; y++)
                memcpy(p + (h - 1 + y) * st - AEG_PAD,
                       p + (h - 1) * st - AEG_PAD, (size_t)(w + 2 * AEG_PAD) * sizeof(pixel));
    }
}

static void rowpost_hpel(struct aeg_encoder *e, int m)
{
    aeg_mc_ctx *mc = &e->slot_mc[e->role_cur];
    int ya = m == 0 ? -AEG_PAD : 16 * m;
    int yb = m == e->mb_h - 1 ? e->h + AEG_PAD - 1 : 16 * m + 15;
    int k;

    int st = mc->stride, off = 0;

    if (!mc->buf)
        return;
    /* a field's planes: every other row of the frame's */
    if (e->fld) {
        off = e->fld_par * st;
        st *= 2;
    }
    for (k = 1; k < 4; k++)
        aeg_hpel_plane_rows((pixel *)mc->base[k] + off, st, e->recp[0],
                            e->stride[0], e->w, e->h, k, ya, yb);
}

/* MBAFF: the field copy of the reference being post-processed (what
 * mbaff_field_shadow makes of the complete picture), band by band behind
 * rowpost_pad: frame band b once it is final and padded (its rows carry
 * their left and right borders already), each field's top border with band
 * 0 and its bottom border with the last band, and the half-sample planes of
 * field band m (16 rows of each field, frame bands 2m and 2m + 1) once frame
 * band 2m + 2 is there -- the 6-tap filter reads three field rows below the
 * band, frame rows up to 32m + 37.  Every sample is computed from the same
 * samples as in the whole-picture version. */
static void rowpost_fshadow(struct aeg_encoder *e)
{
    const int s = e->role_cur, nf = e->mb_h / 2;
    aeg_mc_ctx *mc = &e->slot_fmc[s];
    int i, y, par, k;

    while (e->rp_fsh < e->rp_band) {
        const int b = e->rp_fsh;
        for (i = 0; i < 3; i++) {
            pixel *d = e->slot_fpl[s][i];
            const pixel *src = e->recp[i];
            int st = e->stride[i], w = i ? e->cw : e->w, h = i ? e->ch : e->h;
            int bh = i ? 8 : 16, y0 = b * bh, y1 = y0 + bh;
            size_t row = (size_t)(w + 2 * AEG_PAD) * sizeof(pixel);

            if (y1 > h || b == e->mb_h - 1)
                y1 = h;
            memcpy(d + (ptrdiff_t)y0 * st - AEG_PAD, src + (ptrdiff_t)y0 * st - AEG_PAD,
                   ((size_t)(y1 - y0 - 1) * st + w + 2 * AEG_PAD) * sizeof(pixel));
            for (par = 0; par < 2; par++) {
                pixel *f = d + par * st;
                const int fst = 2 * st, fh = h / 2;
                if (b == 0)
                    for (y = 1; y <= AEG_PAD; y++)
                        memcpy(f - (ptrdiff_t)y * fst - AEG_PAD, f - AEG_PAD, row);
                if (b == e->mb_h - 1)
                    for (y = 1; y <= AEG_PAD; y++)
                        memcpy(f + (ptrdiff_t)(fh - 1 + y) * fst - AEG_PAD,
                               f + (ptrdiff_t)(fh - 1) * fst - AEG_PAD, row);
            }
        }
        e->rp_fsh++;
    }
    while (e->rp_fhp < nf &&
           (e->rp_fsh == e->mb_h || e->rp_fsh >= 2 * e->rp_fhp + 3)) {
        const int m = e->rp_fhp;
        const int ya = m == 0 ? -AEG_PAD : 16 * m;
        const int yb = m == nf - 1 ? e->h / 2 + AEG_PAD - 1 : 16 * m + 15;
        if (mc->buf) {
            const pixel *yp = e->slot_fpl[s][0];
            int st = mc->stride, ys = e->stride[0];
            for (k = 1; k < 4; k++)
                for (par = 0; par < 2; par++)
                    aeg_hpel_plane_rows((pixel *)mc->base[k] + par * st, 2 * st,
                                        yp + par * ys, 2 * ys, e->w, e->h / 2,
                                        k, ya, yb);
        }
        e->rp_fhp++;
    }
}

/* Everything that rows < `final` being final allows. */
static void rowpost_bands(struct aeg_encoder *e, int final)
{
    int pub;
    aeg_lowres *lo = &e->slot_low[e->role_cur];

    while (e->rp_band < final) {
        const int b = e->rp_band;
        rowpost_pad(e, b);
        if (lo->mem && !e->fld)
            aeg_lowres_build_rows(lo, e->recp[0], e->stride[0], e->w, e->h,
                                  8 * b, 8 * b + 7);
        else if (e->rp_lfon) {
            /* a field's own copy (slot_lowf); with the last band the rows
             * past the field's height too */
            aeg_lowres *lf = &e->slot_lowf[e->role_cur][e->fld_par];
            aeg_lowres_build_rows(lf, e->recp[0], e->stride[0], e->w, e->h,
                                  8 * b, b == e->mb_h - 1 ? lf->h - 1 : 8 * b + 7);
        }
        e->rp_band++;
    }
    while (e->rp_hpel + 1 < e->rp_band ||
           (e->rp_band == e->mb_h && e->rp_hpel < e->mb_h)) {
        rowpost_hpel(e, e->rp_hpel);
        e->rp_hpel++;
    }
    pub = e->rp_hpel;
    if (e->fld)
        /* a field picture: published in frame rows, final once both fields
         * are -- nothing during the first field, two frame rows per row of
         * the second */
        pub = e->fld_idx == 0 ? 0 : e->rp_hpel == e->mb_h ? e->fsv.mb_h
                                                          : 2 * e->rp_hpel;
    if (e->rp_fon) {
        /* field macroblocks of the pictures predicting from it read the
         * field copy: published as far as both are there */
        int fr;
        rowpost_fshadow(e);
        fr = e->rp_fhp == e->mb_h / 2 ? e->mb_h : 2 * e->rp_fhp;
        if (fr < pub)
            pub = fr;
    }
    if (e->slot_rows[e->role_cur] != pub)
        slot_publish(e, e->role_cur, pub);
}

static void rowpost_deblock_row_ph(struct aeg_encoder *e, const aeg_mb *mbs, int y);

/* (timed as a phase, see aeg_ph_switch) */
static void rowpost_deblock_row(struct aeg_encoder *e, const aeg_mb *mbs, int y)
{
    const int prev = aeg_ph_push(AEG_PH_DEBLOCK);

    rowpost_deblock_row_ph(e, mbs, y);
    aeg_ph_pop(prev);
}

static void rowpost_deblock_row_ph(struct aeg_encoder *e, const aeg_mb *mbs, int y)
{
    int x;
    for (x = 0; x < e->mb_w; x++)
        aeg_deblock_mb(&e->pic, mbs, e->mb_w, x, y, e->p.deblock_alpha,
                       e->p.deblock_beta);
}

/* Macroblock rows of each reference picture macroblock row `r` may read:
 * the motion search reaches search_range below the block (plus the 6-tap
 * and quarter-sample taps, with margin), the lowres pre-search of the row
 * below (lr_seed_rows) 16 lowres rows plus its half-sample tap below that
 * block row.  mb_h means all of it, borders included. */
static int ref_rows_needed(const struct aeg_encoder *e, int r)
{
    static _Atomic int less = -1;
    int range = e->p.search_range ? e->p.search_range : 16;
    int line = 16 * r + 15 + range + 8, n;

    if (e->mbaff) {
        /* An MBAFF pair is decided at once (both rows).  Its field
         * macroblocks search their field copies, 16 field rows per pair row
         * and the range in field rows: frame row 2 * line + 1 of the copy.
         * A frame macroblock may inherit a vector (skip, direct) from a field
         * neighbour, twice as tall in frame rows. */
        int fl = 16 * (r >> 1) + 15 + range + 8;
        r |= 1;
        line = 16 * r + 15 + 2 * range + 8;
        if (2 * fl + 1 > line)
            line = 2 * fl + 1;
    }

    if (less < 0) {
        /* debug: AEG_RRLESS=n under AEG_POISON shows the margin is real */
        const char *ev = getenv("AEG_RRLESS");
        less = ev ? atoi(ev) : 0;
    }
    if (e->fld) {
        /* A field picture: rows in field lines, the reference's published
         * in frame rows (both its fields final there): field line fl is
         * frame row 2 fl + 1 at most */
        int fl = line;
        if (e->lr_valid && 16 * r + 16 + 63 > fl)
            fl = 16 * r + 16 + 63;
        n = (2 * fl + 1 - less) / 16 + 1;
        return n > e->fsv.mb_h ? e->fsv.mb_h : n;
    }
    if (e->lr_valid && 16 * r + 16 + 63 > line)
        line = 16 * r + 16 + 63;
    line -= less;
    n = line / 16 + 1;
    return n > e->mb_h ? e->mb_h : n;
}

/* es: bytes per sample (the lowres plane is 8-bit at every depth) */
static void pz_copy_rows_es(void *dst_, const void *src_, int stride,
                            int pad, int h, int bh, int b0, int b1, int nb,
                            size_t es)
{
    uint8_t *dst = (uint8_t *)dst_;
    const uint8_t *src = (const uint8_t *)src_;
    int ya = b0 == 0 ? -pad : b0 * bh;
    int yb = b1 == nb ? h + pad : b1 * bh;
    if (yb > h + pad)
        yb = h + pad;
    if (yb > ya)
        memcpy(dst + ((ptrdiff_t)ya * stride - pad) * (ptrdiff_t)es,
               src + ((ptrdiff_t)ya * stride - pad) * (ptrdiff_t)es,
               (size_t)(yb - ya) * stride * es);
}
#define pz_copy_rows(d, s, st, pad, h, bh, b0, b1, nb) \
    pz_copy_rows_es(d, s, st, pad, h, bh, b0, b1, nb, sizeof(pixel))

/* As pz_copy_rows, for a buffer with vpad rows above and below its planes
 * (the MBAFF field copies) */
static void pz_copy_vrows(pixel *dst, const pixel *src, int stride, int vpad,
                          int h, int bh, int b0, int b1, int nb)
{
    int ya = b0 == 0 ? -vpad : b0 * bh;
    int yb = b1 == nb ? h + vpad : b1 * bh;
    if (yb > h + vpad)
        yb = h + vpad;
    if (yb > ya)
        memcpy(dst + (ptrdiff_t)ya * stride - AEG_PAD,
               src + (ptrdiff_t)ya * stride - AEG_PAD,
               (size_t)(yb - ya) * stride * sizeof(pixel));
}

/* A field picture's poisoned reference copies (pzl_setup): the real rows
 * up to frame row `need` of each, the frame's planes, half-sample planes,
 * the field's half-resolution copy and, for direct mode, the co-located
 * field's decisions as far as they go. */
static void pzl_ensure(struct aeg_encoder *e, int need)
{
    const struct fld_save *v = &e->fsv;
    int l, k;

    for (l = 0; l < AEG_MAX_REFS; l++) {
        const int s = e->pzl_slot[l], b0 = e->pzl_rows[l];
        if (s < 0 || need <= b0)
            continue;
        for (k = 0; k < 3; k++)
            pz_copy_vrows(e->pzl_plane[l][k], e->pic_plane[s][k], v->stride[k],
                          e->vpad, k ? v->ch : v->h, k ? 8 : 16, b0, need, v->mb_h);
        if (e->pzl_mc[l].buf && e->slot_mc[s].buf)
            for (k = 1; k < 4; k++)
                pz_copy_vrows((pixel *)e->pzl_mc[l].base[k], e->slot_mc[s].base[k],
                              e->slot_mc[s].stride, e->vpad, v->h, 16, b0, need,
                              v->mb_h);
        if (e->pzl_lsrc[l]) {
            /* frame rows < need: the field's rows < 8 need, its
             * half-resolution rows < 4 need */
            const aeg_lowres *src = e->pzl_lsrc[l];
            int lp = (src->stride - src->w) / 2;
            pz_copy_rows_es(e->pzl_low[l].plane, src->plane, src->stride, lp,
                            src->h, 4, b0, need, v->mb_h, 1);
        }
        if (l == 1 && e->pzl_msrc) {
            const int r0 = b0 / 2, r1 = need >= v->mb_h ? e->mb_h : need / 2;
            const ptrdiff_t o = e->col_mbs - e->pzl_mbs;
            if (r1 > r0)
                memcpy(e->pzl_mbs + o + r0 * e->mb_w, e->pzl_msrc + r0 * e->mb_w,
                       (size_t)(r1 - r0) * e->mb_w * sizeof(aeg_mb));
        }
        e->pzl_rows[l] = need;
    }
}

/* Before the decision pass of row mby: make sure the references are there. */
static int ref_rows_ensure(struct aeg_encoder *e, int mby)
{
    int need, l, k;

    if (e->fsync) {
        int nl = nrefpl(e);
        if (!e->has_ref || !(e->cur_is_p || e->cur_is_b))
            return 0;
        need = ref_rows_needed(e, mby);
        for (l = 0; l < nl; l++)
            /* (the second field's first: this worker coded it already) */
            if (e->ref_slot[l] != e->role_cur &&
                ref_wait(e, e->ref_slot[l], need))
                return -1;
        return 0;
    }
    if (e->pzl_on) {
        pzl_ensure(e, ref_rows_needed(e, mby));
        return 0;
    }
    if (!e->pz_on)
        return 0;
    need = ref_rows_needed(e, mby);
    for (l = 0; l < e->pz_nl; l++) {
        int b0 = e->pz_rows[l];
        if (!e->pz_mem[l] || need <= b0)
            continue;
        for (k = 0; k < 3; k++)
            pz_copy_rows(e->pz_plane[l][k], e->pz_src[l][k], e->stride[k],
                         AEG_PAD, k ? e->ch : e->h, k ? 8 : 16, b0, need,
                         e->mb_h);
        for (k = 1; k < 4; k++)
            if (e->pz_hsrc[l][k])
                pz_copy_rows((pixel *)e->pz_mc[l].base[k], e->pz_hsrc[l][k],
                             e->stride[0], AEG_PAD, e->h, 16, b0, need, e->mb_h);
        if (e->pz_lsrc[l] && e->pz_low[l].mem) {
            int lp = (e->pz_low[l].stride - e->pz_low[l].w) / 2;
            pz_copy_rows_es(e->pz_low[l].plane, e->pz_lsrc[l]->plane,
                            e->pz_low[l].stride, lp, e->pz_low[l].h, 8, b0,
                            need, e->mb_h, 1);
        }
        if (l == 1 && e->pz_msrc)
            memcpy(e->pz_mbs + b0 * e->mb_w, e->pz_msrc + b0 * e->mb_w,
                   (size_t)(need - b0) * e->mb_w * sizeof(aeg_mb));
        if (e->pzf_slot[l] >= 0) {
            const int fs = e->pzf_slot[l];
            for (k = 0; k < 3; k++)
                pz_copy_vrows(e->pzf_plane[l][k], e->slot_fpl[fs][k],
                              e->stride[k], e->vpad, k ? e->ch : e->h,
                              k ? 8 : 16, b0, need, e->mb_h);
            if (e->slot_fmc[fs].buf)
                for (k = 1; k < 4; k++)
                    pz_copy_vrows((pixel *)e->pzf_mc[l].base[k],
                                  e->slot_fmc[fs].base[k], e->slot_fmc[fs].stride,
                                  e->vpad, e->h, 16, b0, need, e->mb_h);
        }
        e->pz_rows[l] = need;
    }
    return 0;
}

/* A field picture under AEG_POISON: every reference frame it reads (but its
 * own, whose first field this worker coded) replaced by a copy that is
 * garbage until pzl_ensure copies the rows the bound allows -- the frame's
 * planes and half-sample planes (the field views keep their offsets), the
 * reference field's half-resolution copy and the co-located field's
 * decisions.  The copies have the frame's dimensions (the field's are in
 * effect, fsv holds the frame's). */
static void pzl_setup(struct aeg_encoder *e)
{
    static unsigned seed = 9876;
    const struct fld_save *v = &e->fsv;
    int l, k, i;

    if (!getenv("AEG_POISON") || e->n_slices > 1 || !e->has_ref ||
        !(e->cur_is_p || e->cur_is_b))
        return;
    e->pzl_on = 1;
    e->pzl_msrc = NULL;
    for (l = 0; l < AEG_MAX_REFS; l++) {
        const int s = e->ref_slot[l];
        aeg_mc_ctx *mc = &e->mcr[l];
        ptrdiff_t off;
        size_t n, nbytes;

        e->pzl_slot[l] = -1;
        e->pzl_lsrc[l] = NULL;
        if (l >= nrefpl(e) || s < 0 || s == e->role_cur || !e->refpl[l][0])
            continue;
        off = e->refpl[l][0] - e->pic_plane[s][0];
        if (off != 0 && off != v->stride[0])
            continue;
        if (!e->pzl_mem[l]) {
            /* (enc_alloc_picture and friends size by the frame's height
             * and set the frame's strides) */
            int h = e->h, st[3] = { e->stride[0], e->stride[1], e->stride[2] };
            e->h = v->h;
            if (enc_alloc_picture(e, &e->pzl_mem[l], e->pzl_plane[l]) != AEG_OK) {
                e->h = h;
                memcpy(e->stride, st, sizeof(st));
                continue;
            }
            aeg_mc_ctx_init_pad(&e->pzl_mc[l], e->w, v->h, v->stride[0], e->vpad);
            e->h = h;
            memcpy(e->stride, st, sizeof(st));
            if (aeg_lowres_alloc(&e->pzl_low[l], e->mb_w, v->mb_h))
                e->pzl_low[l].mem = NULL;
            if (!e->pzl_mbs)
                e->pzl_mbs = (aeg_mb *)aeg_alloc((size_t)v->mb_count * sizeof(aeg_mb));
        }
        {
            int h = e->h;
            e->h = v->h;
            nbytes = pic_bytes(e) / sizeof(pixel);
            e->h = h;
        }
        for (n = 0; n < nbytes; n++) {
            seed = seed * 1103515245u + 12345u;
            e->pzl_mem[l][n] = (pixel)(seed >> 16);
        }
        if (e->pzl_mc[l].buf)
            memset(e->pzl_mc[l].buf, 0x6b,
                   3 * (size_t)v->stride[0] * (v->h + 2 * e->vpad) * sizeof(pixel));
        for (k = 0; k < 3; k++)
            e->refpl[l][k] = e->pzl_plane[l][k] + (e->refpl[l][k] - e->pic_plane[s][k]);
        mc->base[0] = e->refpl[l][0];
        for (i = 1; i < 4; i++)
            if (mc->base[i] && e->pzl_mc[l].buf && e->slot_mc[s].buf)
                mc->base[i] = e->pzl_mc[l].base[i] + (mc->base[i] - e->slot_mc[s].base[i]);
        if (e->lr_valid && l < e->lr_nl && e->pzl_low[l].mem &&
            e->lr_ref[l] == &e->slot_lowf[s][e->ref_par[l]]) {
            e->pzl_lsrc[l] = e->lr_ref[l];
            memset(e->pzl_low[l].mem, 0x3c,
                   (size_t)e->pzl_low[l].stride * (e->pzl_low[l].h + 2 * AEG_LA_PAD));
            e->lr_ref[l] = &e->pzl_low[l];
        }
        if (l == 1 && e->cur_is_b && e->pzl_mbs &&
            e->col_mbs == e->slot_mbs[s] + e->ref_par[1] * e->mb_count) {
            e->pzl_msrc = e->col_mbs;
            memset(e->pzl_mbs, 0x77, (size_t)v->mb_count * sizeof(aeg_mb));
            e->col_mbs = e->pzl_mbs + e->ref_par[1] * e->mb_count;
        }
        e->pzl_slot[l] = s;
        e->pzl_rows[l] = 0;
    }
}

/* MBAFF: a poisoned copy of buffer s's field copy in place of the field
 * references (fref_*) of list entry l that read it */
static void pzf_setup(struct aeg_encoder *e, int l, int s)
{
    static unsigned seed = 54321;
    int cp, k, i, hit = 0;
    size_t n, nbytes;

    if (!e->mbaff || !e->fref_n || !e->slot_fmem[s])
        return;
    for (cp = 0; cp < 2; cp++)
        for (k = 0; k < e->fref_n; k++) {
            ptrdiff_t off = e->fref_pl[cp][k][0] - e->slot_fpl[s][0];
            hit |= off == 0 || off == e->stride[0];
        }
    if (!hit)
        return;
    if (!e->pzf_mem[l]) {
        if (enc_alloc_picture(e, &e->pzf_mem[l], e->pzf_plane[l]) != AEG_OK)
            return;
        aeg_mc_ctx_init_pad(&e->pzf_mc[l], e->w, e->h, e->stride[0], e->vpad);
    }
    nbytes = pic_bytes(e) / sizeof(pixel);
    for (n = 0; n < nbytes; n++) {
        seed = seed * 1103515245u + 12345u;
        e->pzf_mem[l][n] = (pixel)(seed >> 16);
    }
    if (e->pzf_mc[l].buf)
        memset(e->pzf_mc[l].buf, 0x5a,
               3 * (size_t)e->stride[0] * (e->h + 2 * e->vpad) * sizeof(pixel));
    for (cp = 0; cp < 2; cp++)
        for (k = 0; k < e->fref_n; k++) {
            aeg_mc_ctx *c = &e->fref_mc[cp][k];
            ptrdiff_t off = e->fref_pl[cp][k][0] - e->slot_fpl[s][0];
            if (off != 0 && off != e->stride[0])
                continue;
            for (i = 0; i < 3; i++)
                e->fref_pl[cp][k][i] = e->pzf_plane[l][i] +
                                       (e->fref_pl[cp][k][i] - e->slot_fpl[s][i]);
            c->base[0] = e->fref_pl[cp][k][0];
            for (i = 1; i < 4; i++)
                if (c->base[i] && e->pzf_mc[l].buf)
                    c->base[i] = e->pzf_mc[l].base[i] +
                                 (c->base[i] - e->slot_fmc[s].base[i]);
        }
    e->pzf_slot[l] = s;
}

/* Put the poisoned copies in place of this picture's references (the lists
 * that read a reference of the buffers; a weighted list 0 reads a copy that
 * was complete before the picture started). */
static void pz_setup(struct aeg_encoder *e)
{
    int l, k;
    static unsigned seed = 12345;

    e->pz_on = 0;
    e->pzl_on = 0;
    if (e->fld) {
        pzl_setup(e);
        return;
    }
    if (!getenv("AEG_POISON") || e->n_slices > 1 || !e->has_ref ||
        !(e->cur_is_p || e->cur_is_b))
        return;
    e->pz_on = 1;
    e->pz_nl = nrefpl(e);
    for (l = 0; l < e->pz_nl; l++) {
        pixel **rp = e->refpl[l];
        aeg_mc_ctx *mc = &e->mcr[l];
        int s = e->ref_slot[l];
        size_t i, nbytes;
        pixel *base;

        e->pz_rows[l] = 0;
        e->pz_src[l][0] = NULL;
        e->pzf_slot[l] = -1;
        if (s < 0 || rp[0] != e->pic_plane[s][0] ||
            (l == 0 && e->cur_is_p && e->p.weightp)) {
            e->pz_rows[l] = e->mb_h;
            continue;
        }
        if (!e->pz_mem[l]) {
            enc_alloc_picture(e, &e->pz_mem[l], e->pz_plane[l]);
            aeg_mc_ctx_init(&e->pz_mc[l], e->w, e->h, e->stride[0]);
            if (aeg_lowres_alloc(&e->pz_low[l], e->mb_w, e->mb_h))
                e->pz_low[l].mem = NULL;
            if (l == 1)
                e->pz_mbs = (aeg_mb *)aeg_alloc((size_t)e->mb_count * sizeof(aeg_mb));
        }
        /* garbage everywhere */
        base = e->pz_mem[l];
        nbytes = pic_bytes(e) / sizeof(pixel);
        for (i = 0; i < nbytes; i++) {
            seed = seed * 1103515245u + 12345u;
            base[i] = (pixel)(seed >> 16);
        }
        if (e->pz_mc[l].buf)
            memset(e->pz_mc[l].buf, 0xa5,
                   3 * (size_t)e->stride[0] * (e->h + 2 * AEG_PAD) * sizeof(pixel));
        if (e->pz_low[l].mem)
            memset(e->pz_low[l].mem, 0x3c,
                   (size_t)e->pz_low[l].stride * (e->pz_low[l].h + 2 * 32));
        for (k = 0; k < 3; k++) {
            e->pz_src[l][k] = rp[k];
            rp[k] = e->pz_plane[l][k];
        }
        for (k = 1; k < 4; k++) {
            e->pz_hsrc[l][k] = mc->ready ? mc->base[k] : NULL;
            if (mc->ready)
                mc->base[k] = e->pz_mc[l].base[k];
        }
        mc->base[0] = rp[0];
        e->pz_lsrc[l] = NULL;
        if (e->lr_valid && l < e->lr_nl && e->lr_ref[l] == &e->slot_low[s]) {
            e->pz_lsrc[l] = e->lr_ref[l];
            e->lr_ref[l] = &e->pz_low[l];
        }
        pzf_setup(e, l, s);
        if (l == 1 && e->cur_is_b && e->col_mbs == e->slot_mbs[s] && e->pz_mbs) {
            e->pz_msrc = e->col_mbs;
            memset(e->pz_mbs, 0x77, (size_t)e->mb_count * sizeof(aeg_mb));
            e->col_mbs = e->pz_mbs;
        } else if (l == 1) {
            e->pz_msrc = NULL;
        }
    }
}

static void pz_finish(struct aeg_encoder *e)
{
    int l, k;
    e->pzl_on = 0;      /* (a field's lists are set afresh for the next) */
    if (!e->pz_on)
        return;
    for (l = 0; l < e->pz_nl; l++) {
        pixel **rp = e->refpl[l];
        if (!e->pz_src[l][0])
            continue;
        for (k = 0; k < 3; k++)
            rp[k] = (pixel *)e->pz_src[l][k];
    }
    e->pz_on = 0;
}

/* After the decision pass of row mby. */
static void rowpost_row(struct aeg_encoder *e, int mby, int slice)
{
    int64_t t0;

    if (!e->rp_on)
        return;
    t0 = aeg_now_ns();
    rowpost_finalize(e, mby, slice);
    if (mby >= 1) {
        if (e->p.deblock)
            rowpost_deblock_row(e, e->rp_mbs, mby - 1);
        e->rp_dbk = mby;
        rowpost_bands(e, mby - 1);
    }
    e->st.stage_ns[AEG_STAGE_DEBLOCK] += aeg_now_ns() - t0;
}

/* rowpost_row for the MBAFF frame's row of pairs `py`: the decoder's QP and
 * skip state in pair order, and the deblocking one pair row behind (the
 * pair row below predicts from the unfiltered bottom rows of this one). */
static void rowpost_pair(struct aeg_encoder *e, int py, int slice)
{
    int64_t t0;
    int mbx;

    if (!e->rp_on)
        return;
    t0 = aeg_now_ns();
    for (mbx = 0; mbx < e->mb_w; mbx++) {
        rowpost_finalize_mb(e, mbx, 2 * py, slice);
        rowpost_finalize_mb(e, mbx, 2 * py + 1, slice);
    }
    if (py >= 1) {
        if (e->p.deblock)
            aeg_deblock_pair_row(&e->pic, e->rp_mbs, e->mb_w, py - 1,
                                 e->p.deblock_alpha, e->p.deblock_beta);
        e->rp_dbk = 2 * py;
        rowpost_bands(e, 2 * py - 1);
    }
    e->st.stage_ns[AEG_STAGE_DEBLOCK] += aeg_now_ns() - t0;
}

static void wp_analyse(struct aeg_encoder *e);

/* Would the analysis find weights?  The same analysis on the reference's
 * source instead of its reconstruction (which may not exist yet). */
static int wp_predict(struct aeg_encoder *e)
{
    int slots = in_slots(e), d = e->ref_poc[0], r;
    pixel *save[3] = { e->refpl[0][0], e->refpl[0][1], e->refpl[0][2] };

    if (e->in_count - d > slots || d < 0)
        return -1;
    e->refpl[0][0] = e->in_plane[d % slots][0];
    e->refpl[0][1] = e->in_plane[d % slots][1];
    e->refpl[0][2] = e->in_plane[d % slots][2];
    wp_analyse(e);
    r = e->wp.lflag[0] || e->wp.cflag[0];
    memset(&e->wp, 0, sizeof(e->wp));
    e->refpl[0][0] = save[0];
    e->refpl[0][1] = save[1];
    e->refpl[0][2] = save[2];
    return r;
}

/* A weighted P picture running ahead of its analysis (ft_spec) codes as if
 * the analysis had found no weights.  Once its list 0 reference is whole the
 * analysis runs: no weights, and everything coded so far is what the serial
 * encoder codes; weights, and the picture starts again (FT_RESTART).  The
 * last row of the picture waits for the whole reference, so this always
 * happens before the picture's own last rows are published. */
static int ft_wp_check(struct aeg_encoder *e, int wait)
{
    int s = e->ref_slot[0];

    if (!e->ft_spec)
        return 0;
    if (__atomic_load_n(&e->fsync->rows[s], __ATOMIC_ACQUIRE) < e->mb_h) {
        if (!wait)
            return 0;
        if (ref_wait(e, s, e->mb_h))
            return -1;
    }
    if (ft_refs_changed(e))
        return -1;
    e->ft_spec = 0;
    wp_analyse(e);
    if (e->wp.lflag[0] || e->wp.cflag[0]) {
        __atomic_store_n(&e->fsync->wp_hist, 1, __ATOMIC_RELAXED);
        return -1;
    }
    __atomic_store_n(&e->fsync->wp_hist, 0, __ATOMIC_RELAXED);
    return 0;
}

static void mbaff_field_shadow(struct aeg_encoder *e, int s);
static void mbaff_field_refs(struct aeg_encoder *e);

/* MBAFF decision mode: AEG_MBDEC=0 every pair a frame pair (stage 1-4
 * behaviour), 1 x264's vertical-SAD rule alone (mbaff_vsad_field), 2 (the
 * default) the vertical-SAD rule and, where it is unsure (mbaff_trial_gate),
 * both kinds coded and the cheaper kept (mbaff_pair_trial), 3 as 2 but the
 * vertical-SAD choice kept (a check of the trial's state restore: the output
 * has to be mode 1's). */
static int mbdec_mode(void)
{
    static _Atomic int env = -2;
    if (env == -2) {
        const char *ev = getenv("AEG_MBDEC");
        env = ev ? atoi(ev) : 2;
    }
    return env;
}

/* When the trial runs: on the pairs whose vertical SADs (with the neighbour
 * bias) are within this many percent of the frame one, -1 never, >= 1000
 * always.  On the 80-field clips (p2, BD-rate against --tff; VSAD alone
 * -22.05 %): 25 -23.21 %, 50 -23.69 %, 75 -23.98 % (but tree +0.15, VSAD
 * alone -0.10), every pair -23.52 % (tree and vtest lose to VSAD alone, and
 * +171 % time against +53 %).  50 costs +61 % at p2 and +72 % at p3 (VSAD alone +53 /
 * +55 %), where it takes p3 from -19.42 to -20.57 %.  Presets 0/1: the
 * vertical SADs alone.  AEG_MBTG overrides. */
static int mbaff_trial_gate(const struct aeg_encoder *e)
{
    const int g = ENV_INT("AEG_MBTG", -100);
    return g != -100 ? g : e->p.preset >= 2 ? 50 : -1;
}

/* x264's x264_field_vsad: the pair's source luma, vertical differences
 * summed within the frame (32 lines) against within each field (16 + 16),
 * with half a thousand (per 8 bits) in favour of the left and above pairs'
 * kind, so that the choice does not flicker between neighbours. */
static int mbaff_vsad_field(const struct aeg_encoder *e, int mbx, int py,
                            int *unsure)
{
    const int st = e->stride[0];
    const pixel *s0 = e->srcp[0] + (ptrdiff_t)32 * py * st + 16 * mbx;
    const int bias = 512 << (BIT_DEPTH - 8);
    int64_t fr = 0, fl = 0;
    int x, y;

    for (y = 0; y < 31; y++) {
        const pixel *a = s0 + (ptrdiff_t)y * st;
        for (x = 0; x < 16; x++)
            fr += abs((int)a[x + st] - (int)a[x]);
    }
    for (y = 0; y < 30; y++) {
        const pixel *a = s0 + (ptrdiff_t)y * st;
        for (x = 0; x < 16; x++)
            fl += abs((int)a[x + 2 * st] - (int)a[x]);
    }
    if (mbx > 0)
        fl += bias - 2 * bias * e->mbs[2 * py * e->mb_w + mbx - 1].fld;
    if (2 * py - 1 >= e->top_row)
        fl += bias - 2 * bias * e->mbs[(2 * py - 1) * e->mb_w + mbx].fld;
    if (unsure) {
        const int g = mbdec_mode() >= 2 ? mbaff_trial_gate(e) : -1;
        *unsure = g >= 1000 ||
                  (g >= 0 && (fl > fr ? fl - fr : fr - fl) * 100 <= (int64_t)g * fr);
    }
    return fl < fr;
}

/* MBAFF: is the macroblock pair (mbx, py) coded as a field pair?  A forced
 * test pattern (AEG_MBFORCE: "all", or a number seeding a pseudo-random
 * pattern per pair and picture), else the decision (mbdec_mode). */
static int mbaff_pair_field(const struct aeg_encoder *e, int slice, int mbx,
                            int py, int *unsure)
{
    static const char *force;
    static int once;
    unsigned h;

    if (!once) {
        force = getenv("AEG_MBFORCE");
        once = 1;
    }
    (void)slice;
    *unsure = 0;
    if (!force)
        return mbdec_mode() >= 1 ? mbaff_vsad_field(e, mbx, py, unsure) : 0;
    if (!strcmp(force, "all"))
        return 1;
    h = (unsigned)atoi(force) * 2654435761u ^ (unsigned)mbx * 40503u ^
        (unsigned)py * 9973u ^ (unsigned)e->cur_poc * 6151u;
    h ^= h >> 13;
    h *= 0x5bd1e995u;
    h ^= h >> 15;
    return (h >> 7) & 1;
}

/* The decision pass's work on one macroblock: the analysis, the mode
 * decision and reconstruction, and the decisions stored for the neighbour
 * queries of the following macroblocks (and for the deblocking filter). */
/* A field macroblock of an MBAFF B picture: per list, the field of its own
 * parity (field reference index 0) or the other one (1) of the list's frame
 * -- the other parity is the nearer field in time for one of the lists
 * (top-field-first: list 0 for a top macroblock, list 1 for a bottom one).
 * A 16x16 search on both, by distortion plus the vector difference's and
 * the index's bits; every partition of the macroblock then predicts the
 * list from that field (entry `list` of refpl / mcr, the other one moved to
 * list + 2).  Direct mode derives its indices from the neighbours, so it is
 * offered only when those are the chosen ones (bdr_no). */
/* Reference entry k := choice f of bfld_prep: an MBAFF field macroblock's
 * field reference f (mbf_set_ref), or a B field picture's entry f as
 * set_field_refs made it (bfp_sv). */
static void bref_set(struct aeg_encoder *e, int mby, int k, int f)
{
    const struct bfp_ent *s;

    if (e->mbaff) {
        mbf_set_ref(e, mby, k, f);
        return;
    }
    s = &e->bfp_sv[f];
    memcpy(e->refpl[k], s->pl, sizeof(e->refpl[k]));
    e->mcr[k] = s->mc;
    e->ref_poc[k] = s->poc;
    e->fld_cofs[k] = s->cofs;
    e->ref_slot[k] = s->slot;
    e->ref_par[k] = s->par;
}

/* A B field picture with two fields per list: keep its entries as
 * set_field_refs (and the reference views, pz_setup) left them before the
 * macroblocks pick from them (save 1), and put them back after (0). */
static void bfp_keep(struct aeg_encoder *e, int save)
{
    int k;

    if (!e->bfp_on)
        return;
    for (k = 0; k < 4; k++) {
        struct bfp_ent *s = &e->bfp_sv[k];
        if (save) {
            memcpy(s->pl, e->refpl[k], sizeof(s->pl));
            s->mc = e->mcr[k];
            s->poc = e->ref_poc[k];
            s->cofs = e->fld_cofs[k];
            s->slot = e->ref_slot[k];
            s->par = e->ref_par[k];
        } else {
            bref_set(e, 0, k, k);
        }
    }
    if (!save)
        e->b_ref[0] = e->b_ref[1] = e->bfld_on = e->bdr_no = 0;
}

static void bfld_prep(struct aeg_encoder *e, int mbx, int mby, int qp)
{
    const int st = e->stride[0], idx = mby * e->mb_w + mbx;
    const pixel *src = e->srcp[0] + mby * 16 * st + mbx * 16;
    const int range = e->p.search_range ? e->p.search_range : 16;
    int l, r, dref[2];

    e->b_ref[0] = e->b_ref[1] = e->bfld_on = e->bdr_no = 0;
    if (!e->cur_is_b)
        return;
    if (e->mbaff ? !e->cfld || !e->fref_b || e->fref_n < 4
                 : !e->fld || !e->bfp_on)
        return;
    e->bfld_on = 1;
    /* (analyse_mb sets the same) */
    e->me_lambda = me_lambda_on(e) ? aeg_me_lambda(qp) : 0;
    for (l = 0; l < 2; l++) {
        int best = INT_MAX, br = 0;
        for (r = 0; r < 2; r++) {
            int pmv[2], seed[2], mv[2], sad, cost;
            bref_set(e, mby, l, l + 2 * r);
            e->b_ref[l] = r;
            pred16_list(e, mbx, mby, l, r, &pmv[0], &pmv[1]);
            seed[0] = pmv[0] / 4;
            seed[1] = pmv[1] / 4;
            if (e->lr_valid) {
                seed[0] = e->lr_mv[l][idx * 2];
                seed[1] = e->lr_mv[l][idx * 2 + 1];
            }
            aeg_me_block_l(e->refpl[l][0] + mby * 16 * st + mbx * 16, st, src, st,
                           16, 16, range, SUBME_F(e, 4), pmv[0], pmv[1], seed,
                           NULL, 0, e->me_lambda, &e->mcr[l], &mv[0], &mv[1], &sad);
            cost = sad + e->me_lambda * (aeg_mvd_bits(mv[0] - pmv[0], mv[1] - pmv[1]) +
                                         1 + r);
            if (cost < best)
                best = cost, br = r;
        }
        e->b_ref[l] = br;
        bref_set(e, mby, l, l + 2 * br);
        bref_set(e, mby, l + 2, l + 2 * !br);
    }
    /* (no neighbour on either list: both on index 0, zero vectors) */
    b_direct_refs(e, mbx, mby, dref);
    if (dref[0] < 0 && dref[1] < 0)
        dref[0] = dref[1] = 0;
    e->bdr_no = (dref[0] >= 0 && dref[0] != e->b_ref[0]) ||
                (dref[1] >= 0 && dref[1] != e->b_ref[1]);
}

static void decide_mb(struct aeg_encoder *e, int mbx, int mby)
{
    const int idx = mby * e->mb_w + mbx;
    mbenc *tmp = &e->mb_encs[idx];
    aeg_mb *m = &e->mbs[idx];
    int i;
    int64_t t0 = aeg_now_ns();

    reset_mb_enc(tmp, mb_qp_of(e, mbx, mby));
    bfld_prep(e, mbx, mby, tmp->qp);
    {
        CM_T0();
        analyse_mb(e, mbx, mby, tmp);
        CM_LAP(e, 6);
    }
    e->st.stage_ns[AEG_STAGE_MOTION_EST] += aeg_now_ns() - t0;

    bfld_stamp(e, tmp);
    t0 = aeg_now_ns();
    code_mb(e, mbx, mby, tmp);
    e->st.stage_ns[AEG_STAGE_TRANSFORM] += aeg_now_ns() - t0;

    /* store the decisions for the neighbour queries of the following
     * macroblocks (and for the deblocking filter) */
    memset(m, 0, sizeof(*m));
    m->fld = (uint8_t)(e->fld | e->cfld);
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
        m->ref_idx1[i] = tmp->ref1[i];
    }
    m->cost = (int32_t)tmp->sse;
}


#define PAIR_SCR_BEG offsetof(struct aeg_encoder, mb_modes)
#define PAIR_SCR_END (offsetof(struct aeg_encoder, mb_maxref) + sizeof(int))

/* MBAFF trial (mbdec_mode 2, 3): what coding a pair changes -- its two
 * macroblocks' decisions and stored neighbour data, its reconstruction, the
 * shadow coder and its QP history -- so that the pair can be coded both ways
 * from the same start and the cheaper result kept. */
typedef struct {
    aeg_mb mbs[2];
    mbenc enc[2];
    aeg_cabac cab;
    int rlq, rld, lq, ld;
    int lmn;    /* the luma memo's eviction slot (it outlives a macroblock) */
    /* The analysis scratch, mb_modes .. mb_maxref: not every path rewrites
     * all of it, so what a later macroblock may find there is the kept
     * run's. */
    unsigned char scr[PAIR_SCR_END - PAIR_SCR_BEG];
    pixel y[32 * 16], c[2][16 * 8];
} pair_state;

static void pair_state_io(struct aeg_encoder *e, int mbx, int ry,
                          pair_state *ps, int save)
{
    const int idx = ry * e->mb_w + mbx;
    int b, i, y;

    for (b = 0; b < 2; b++) {
        if (save) {
            ps->mbs[b] = e->mbs[idx + b * e->mb_w];
            ps->enc[b] = e->mb_encs[idx + b * e->mb_w];
        } else {
            e->mbs[idx + b * e->mb_w] = ps->mbs[b];
            e->mb_encs[idx + b * e->mb_w] = ps->enc[b];
        }
    }
    if (save) {
        ps->cab = e->cab_shadow;
        ps->rlq = e->rd_last_qp; ps->rld = e->rd_last_dqp;
        ps->lq = e->last_qp; ps->ld = e->last_dqp;
        ps->lmn = e->lmemo_next;
        memcpy(ps->scr, (unsigned char *)e + PAIR_SCR_BEG, sizeof(ps->scr));
    } else {
        e->cab_shadow = ps->cab;
        e->rd_last_qp = ps->rlq; e->rd_last_dqp = ps->rld;
        e->last_qp = ps->lq; e->last_dqp = ps->ld;
        e->lmemo_next = ps->lmn;
        memcpy((unsigned char *)e + PAIR_SCR_BEG, ps->scr, sizeof(ps->scr));
    }
    for (y = 0; y < 32; y++) {
        pixel *r = e->recp[0] + (ptrdiff_t)(16 * ry + y) * e->stride[0] + 16 * mbx;
        if (save)
            memcpy(ps->y + 16 * y, r, 16 * sizeof(pixel));
        else
            memcpy(r, ps->y + 16 * y, 16 * sizeof(pixel));
    }
    for (i = 0; i < 2; i++)
        for (y = 0; y < 16; y++) {
            pixel *r = e->recp[1 + i] + (ptrdiff_t)(8 * ry + y) * e->stride[1] + 8 * mbx;
            if (save)
                memcpy(ps->c[i] + 8 * y, r, 8 * sizeof(pixel));
            else
                memcpy(r, ps->c[i] + 8 * y, 8 * sizeof(pixel));
        }
}

/* Code pair (mbx, ry / 2) as a field (fld) or frame pair; its price: the
 * sum of squared errors of the reconstruction (before the loop filter) plus
 * each macroblock's bits at the mode decision's lambda. */
static int64_t pair_run(struct aeg_encoder *e, int mbx, int ry, int fld)
{
    double bits = 0;
    int64_t sse = 0;
    int b, i, x, y;

    e->pair_fld = fld;
    for (b = 0; b < 2; b++) {
        const int mby = ry + b;
        const mbenc *t = &e->mb_encs[mby * e->mb_w + mbx];
        double lambda;
        if (fld)
            mbf_enter(e, mby);
        decide_mb(e, mbx, mby);
        lambda = AEG_LAMBDA(t->qp - e->mode_lambda_off);
        if (e->cur_is_p)
            lambda *= P_MODE_LAMBDA;
        bits += lambda * (e->p.cabac ? e->cab_shadow.cost256 / 256.0 : (double)t->bits);
        if (e->cfld)
            mbf_leave(e);
    }
    for (y = 0; y < 32; y++) {
        const ptrdiff_t o = (ptrdiff_t)(16 * ry + y) * e->stride[0] + 16 * mbx;
        for (x = 0; x < 16; x++) {
            int d = (int)e->recp[0][o + x] - (int)e->srcp[0][o + x];
            sse += d * d;
        }
    }
    for (i = 1; i < 3; i++)
        for (y = 0; y < 16; y++) {
            const ptrdiff_t o = (ptrdiff_t)(8 * ry + y) * e->stride[1] + 8 * mbx;
            for (x = 0; x < 8; x++) {
                int d = (int)e->recp[i][o + x] - (int)e->srcp[i][o + x];
                sse += d * d;
            }
        }
    return sse + (int64_t)(bits + 0.5);
}

/* Both ways, the other kind than `pref` first, and keeps the cheaper (on a
 * tie `pref`).  Mode 3 runs `pref` first and keeps it: the restore of the
 * first result is what it checks. */
static void mbaff_pair_trial(struct aeg_encoder *e, int mbx, int ry, int pref)
{
    pair_state s0, r1;
    const int chk = mbdec_mode() == 3;
    const int first = chk ? pref : !pref;
    int64_t c1, c2;
    int keep;

    pair_state_io(e, mbx, ry, &s0, 1);
    c1 = pair_run(e, mbx, ry, first);
    pair_state_io(e, mbx, ry, &r1, 1);
    pair_state_io(e, mbx, ry, &s0, 0);
    c2 = pair_run(e, mbx, ry, !first);
    keep = chk ? pref : c1 < c2 ? first : !first;
    if (keep == first)
        pair_state_io(e, mbx, ry, &r1, 0);
    e->pair_fld = keep;
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
    int mbx, mby, ry;
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
    /* (MBAFF: a row of macroblock pairs at a time, in pair order) */
    for (ry = first_row; ry <= last_row; ry += 1 + e->mbaff) {
        int rr, k, nk = e->mb_w << e->mbaff;
        for (rr = ry; rr <= ry + e->mbaff; rr++) {
            if (ref_rows_ensure(e, rr) || (e->fsync && (e->ft_abort ||
                                                         ft_refs_changed(e) ||
                                                         ft_wp_check(e, 0))))
                return FT_RESTART;   /* the worker starts the picture again */
            /* the pre-search vectors this row reads (its own and the row
             * below) */
            if (e->lr_valid && e->lr_rows <= rr + 1)
                lr_seed_rows(e, rr + 1);
        }
        for (k = 0; k < nk; k++) {
            mbx = k >> e->mbaff;
            mby = ry + (k & e->mbaff);
            if (e->mbaff) {
                if (!(mby & 1)) {
                    int unsure;
                    e->pair_fld = mbaff_pair_field(e, slice, mbx, mby >> 1, &unsure);
                    /* both ways (the whole pair at once) */
                    if (unsure) {
                        mbaff_pair_trial(e, mbx, ry, e->pair_fld);
                        k++;
                        continue;
                    }
                }
                if (e->pair_fld)
                    mbf_enter(e, mby);
            }

            decide_mb(e, mbx, mby);
            if (e->cfld)
                mbf_leave(e);
        }
        if (e->mbaff)
            rowpost_pair(e, ry >> 1, slice);
        else
            rowpost_row(e, ry, slice);
    }

    /* The slice header is written after the decision pass, because one of
     * its fields depends on it: num_ref_idx_l0_active.  When no macroblock of
     * this slice ended up on the second reference, the slice is sent with one
     * active reference (with more, as many as the highest index used needs) -- every decision stays valid (all of them predict
     * from list0[0], the predictors and the skip vector are the same with one
     * or two entries), and the ref_idx syntax disappears from every
     * partition.  It is a per-slice choice and is undone at the end of the
     * slice, so the next slice of the picture decides again with both. */
    l0_saved = e->cur_l0_active;
    if (e->cur_l0_active > 1 && slice == AEG_SLICE_P) {
        /* (with three or more: down to the highest index any partition
         * uses -- the list keeps its order, so nothing else changes) */
        int used = 0;
        for (mby = first_row; mby <= last_row && used < l0_saved - 1; mby++)
            for (mbx = 0; mbx < e->mb_w && used < l0_saved - 1; mbx++) {
                const mbenc *t = &e->mb_encs[mby * e->mb_w + mbx];
                /* (an MBAFF field macroblock's index counts fields: frame
                 * entry ref >> 1 of the list, 8.4.2.1) */
                const int sh = e->mbaff && e->mbs[mby * e->mb_w + mbx].fld;
                int b;
                if (!is_inter_type(t->type))
                    continue;
                for (b = 0; b < 16; b++)
                    if ((t->ref[b] >> sh) > used)
                        used = t->ref[b] >> sh;
            }
        e->cur_l0_active = used + 1;
    }
    /* A stream without B pictures keeps pic_order_cnt_type 2 and spends nothing
     * on the picture order count; with them the POC is what direct mode scales
     * its vectors by, so every header carries it (7.3.3). */
    if (e->p.bframes > 0 || e->p.field_coding) {
        int max_lsb = 1 << e->poc_lsb_bits;
        /* counted from the last IDR picture, whose count has to be 0 (8.2.1);
         * a field's is twice its frame's display index, plus one for the
         * second field.  A frame picture among field pictures counts in
         * fields too: its first field 2n, the other 2n + 1 (the PPS sends
         * delta_pic_order_cnt_bottom; the lsb is the top field's). */
        const int fmix = (e->p.field_coding == 2 || e->p.field_coding == 4) && !e->fld;
        const int bff = e->p.interlaced == 2;
        int rel = e->fld ? e->cur_poc - 2 * e->last_idr_poc
                : fmix ? 2 * (e->cur_poc - e->last_idr_poc) + bff
                       : e->cur_poc - e->last_idr_poc;
        aeg_write_slice_header_poc(&bw,
                                   slice == AEG_SLICE_B ? 6 : (slice ? 5 : 7),
                                   e->frame_num & ((1 << e->frame_num_bits) - 1),
                                   e->frame_num_bits |
                                       (e->p.interlaced ? AEG_SH_FIELD : 0) |
                                       (e->fld ? AEG_SH_FPIC : 0) |
                                       (e->fld && e->fld_par ? AEG_SH_BOTTOM : 0) |
                                       (fmix ? AEG_SH_DPOCB : 0) |
                                       (fmix && bff ? AEG_SH_BOTTOM : 0),
                                   is_idr, e->idr_pic_id,
                                   rel & (max_lsb - 1), e->poc_lsb_bits,
                                   e->cur_is_ref,
                                   e->p.cabac ? e->cabac_init_idc : -1,
                                   e->cur_qp - 26, e->p.deblock_alpha,
                                   e->p.deblock_beta,
                                   e->p.deblock ? 0 : 1,
                                   (e->mbaff ? first_row / 2 : first_row) * e->mb_w,
                                   slice == AEG_SLICE_B ? (e->fld && e->bfp_on ? 2 : 1)
                                                        : e->cur_l0_active,
                                   e->p.direct == 1,
                                   wp_used(e) ? &e->wp : NULL, &e->refctl);
    } else {
        aeg_write_slice_header(&bw, slice ? 5 : 7,
                               e->frame_num & ((1 << e->frame_num_bits) - 1),
                               e->frame_num_bits |
                                   (e->p.interlaced ? AEG_SH_FIELD : 0),
                               is_idr, e->idr_pic_id,
                               e->p.cabac ? e->cabac_init_idc : -1,
                               e->cur_qp - 26, e->p.deblock_alpha,
                               e->p.deblock_beta,
                               e->p.deblock ? 0 : 1,
                                   (e->mbaff ? first_row / 2 : first_row) * e->mb_w,
                               e->cur_l0_active,
                               wp_used(e) && slice == AEG_SLICE_P ? &e->wp : NULL);
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

        for (ry = first_row; ry <= last_row; ry += 1 + e->mbaff) {
            int k, nk = e->mb_w << e->mbaff;
            for (k = 0; k < nk; k++) {
                int idx;
                mbenc *tmp;
                aeg_mb *m;
                int skipped = 0, last = 0, i;
                int64_t t0 = aeg_now_ns();

                mbx = k >> e->mbaff;
                mby = ry + (k & e->mbaff);
                idx = mby * e->mb_w + mbx;
                tmp = &e->mb_encs[idx];
                m = &e->mbs[idx];

                /* A macroblock without a coded block pattern carries no
                 * mb_qp_delta, so the decoder keeps the QP it has.  The
                 * Intra_16x16 one always carries the delta: its DC levels
                 * were quantised with its own QP, which is the one it has
                 * to send (with per-macroblock offsets it is not the
                 * previous one). */
                if (tmp->cbp == 0 && tmp->type != AEG_MB_I16x16)
                    tmp->qp = e->last_qp;
                m->qp = (int8_t)tmp->qp;

                if (e->mbaff)
                    mbf_syntax(e, m->fld);
                if (slice == AEG_SLICE_P &&
                    p_skippable(e, mbx, mby, tmp,
                                (mby & 1) && e->mbs[idx - e->mb_w].skip))
                    skipped = 1;
                /* A direct macroblock of a B slice with no residual is exactly
                 * what the skip flag means: same prediction, no residual. */
                if (slice == AEG_SLICE_B &&
                    b_skippable(e, mbx, mby, tmp,
                                (mby & 1) && e->mbs[idx - e->mb_w].skip))
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
                e->mbaff_ff = mbaff_flag_coded(e, mbx, mby, skipped);
                write_mb_cabac(e, &cab, mbx, mby, tmp, slice, skipped);
                if (e->mbaff)
                    mbf_syntax(e, 0);
                e->mbaff_ff = -1;
                /* (no end_of_slice_flag after the top macroblock of a pair) */
                if (!e->mbaff || (mby & 1))
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
    for (ry = first_row; ry <= last_row; ry += 1 + e->mbaff) {
        int k, nk = e->mb_w << e->mbaff;
        for (k = 0; k < nk; k++) {
            int idx;
            mbenc *tmp;
            int64_t t0 = aeg_now_ns();

            mbx = k >> e->mbaff;
            mby = ry + (k & e->mbaff);
            idx = mby * e->mb_w + mbx;
            tmp = &e->mb_encs[idx];

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
                if (e->mbaff)
                    mbf_syntax(e, e->mbs[idx].fld);
                if (slice == AEG_SLICE_P &&
                    p_skippable(e, mbx, mby, tmp,
                                (mby & 1) && e->mbs[idx - e->mb_w].skip))
                    skipped = 1;
                e->stat_mbs++;
                e->stat_skipped += skipped;
                e->stat_type[skipped ? AEG_MB_SKIP : (int)tmp->type]++;
                if (e->mbaff)
                    e->mbs[idx].skip = (uint8_t)skipped;
                if (skipped)
                    skip_run++;
                else {
                    int ff = mbaff_flag_coded(e, mbx, mby, 0);
                    if (slice == AEG_SLICE_P)
                        flush_skip_run(&bw, &skip_run);
                    if (ff >= 0)                /* mb_field_decoding_flag */
                        aeg_bw_put(&bw, 1, (uint32_t)ff);
                    write_mb(e, &bw, mbx, mby, tmp, slice == AEG_SLICE_P);
                }
                if (e->mbaff)
                    mbf_syntax(e, 0);
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

/* How many threads the slice loop may use.  A device stage does not limit it:
 * each slice thread calls the device through a lane of its own (aeg_be_lane:
 * a clone of the backend with its own queue and buffers, behind its own lock
 * in the guard), so the slices' device calls overlap as well as their CPU
 * work.  (This used to drop to one thread whenever the dispatcher had a stage
 * on a device -- and missed the forced batch paths, which then crashed the
 * OpenCL backend from two threads.) */
static int slice_thread_count(const struct aeg_encoder *e)
{
    if (e->n_threads <= 1 || e->n_slices <= 1)
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

/* ------------------------------------------------------------------ */
/* Weighted prediction (fades)                                         */
/* ------------------------------------------------------------------ */
/* A fade changes every sample of the picture by roughly the same gain and
 * offset, which no motion vector can follow: the residual of every block
 * then carries the brightness change.  An explicit weight and offset per
 * reference (pred_weight_table, 8.4.2.3) takes it out of the residual for a
 * few bits per slice.
 *
 * The guess is x264's: the gain that matches the standard deviations, the
 * offset that then matches the means, or the offset alone.  Whether either is
 * worth sending is measured: a small integer search of every 8x8 block of a
 * half-resolution picture against the reference weighted each way, and a
 * weighting is kept only when it saves enough of the summed SAD (motion moves
 * the means of a picture too; on such a picture the weighted reference does
 * not predict better and is not used).  Chroma is decided the same way, on
 * the block vectors the luma test found. */
#define WP_R    4   /* search radius, half-resolution samples */
#define WP_LPAD 8

typedef struct { int w, o, d; } wp_cand;

/* What an 8x8 block would cost coded without the reference: its deviation
 * from its own mean.  A block whose content is new (an object entering, a
 * flash over part of the picture) is coded intra whatever the weights are,
 * so it must not vote for them. */
static int wp_intra_est(const pixel *p, int stride)
{
    int x, y, s = 0, m, d = 0;

    for (y = 0; y < 8; y++)
        for (x = 0; x < 8; x++)
            s += p[y * stride + x];
    m = (s + 32) >> 6;
    for (y = 0; y < 8; y++)
        for (x = 0; x < 8; x++)
            d += abs(p[y * stride + x] - m);
    return d + (64 << (BIT_DEPTH - 8));   /* ~ the mode and DC it still has to send */
}

static void plane_stats(const pixel *p, int stride, int w, int h,
                        double *mean, double *var)
{
    int64_t s = 0, s2 = 0;
    int x, y;

    for (y = 0; y < h; y++)
        for (x = 0; x < w; x++) {
            int v = p[y * stride + x];
            s += v;
            s2 += v * v;
        }
    *mean = (double)s / ((double)w * h);
    *var = (double)s2 / ((double)w * h) - *mean * *mean;
}

static wp_cand wp_guess(double mc, double vc, double mr, double vr, int scaled)
{
    wp_cand c;
    double s = 1.0, o;

    if (scaled && vr > 1.0 && vc > 0.0)
        s = sqrt(vc / vr);
    c.d = 6;
    for (;;) {
        c.w = (int)floor(s * (1 << c.d) + 0.5);
        if (c.w <= 127 || c.d == 0)
            break;
        c.d--;
    }
    if (c.w > 127) c.w = 127;
    if (c.w < 0) c.w = 0;
    o = (mc - (double)c.w * mr / (1 << c.d)) / (1 << (BIT_DEPTH - 8));
    c.o = (int)floor(o + 0.5);
    if (c.o > 127) c.o = 127;
    if (c.o < -128) c.o = -128;
    return c;
}

static int wp_identity(wp_cand c) { return c.w == (1 << c.d) && c.o == 0; }

static pixel wp_px(int v, wp_cand c)
{
    int r = ((v * c.w + (c.d ? 1 << (c.d - 1) : 0)) >> c.d) +
            c.o * (1 << (BIT_DEPTH - 8));
    return aeg_clip_pixel(r);
}

/* Summed best SAD of the 8x8 blocks of `cur` (lw x lh) against `ref`
 * (padded by WP_LPAD, stride ls) weighted by `c`; the block vectors go to
 * `mv` when non-NULL. */
static int64_t wp_cost(const pixel *cur, int lw, int lh, const pixel *ref,
                       int ls, pixel *tmp, wp_cand c, int8_t *mv)
{
    const pixel *r = ref;
    int64_t tot = 0;
    int bx, by, i, n = 0;

    if (!wp_identity(c)) {
        for (i = 0; i < ls * (lh + 2 * WP_LPAD); i++)
            tmp[i] = wp_px(ref[i], c);
        r = tmp;
    }
    for (by = 0; by + 8 <= lh; by += 8)
        for (bx = 0; bx + 8 <= lw; bx += 8, n++) {
            int best = 1 << 30, bdx = 0, bdy = 0, dx, dy;
            for (dy = -WP_R; dy <= WP_R; dy++)
                for (dx = -WP_R; dx <= WP_R; dx++) {
                    int s = aeg_sad(8, 8, cur + by * lw + bx, lw,
                                    r + (by + dy + WP_LPAD) * ls + bx + dx +
                                        WP_LPAD, ls);
                    if (s < best) { best = s; bdx = dx; bdy = dy; }
                }
            if (mv) { mv[n * 2] = (int8_t)bdx; mv[n * 2 + 1] = (int8_t)bdy; }
            {
                int ie = wp_intra_est(cur + by * lw + bx, lw);
                tot += best < ie ? best : ie;
            }
        }
    return tot;
}

/* wp_cost() refined around given block vectors (+-1) instead of a full
 * +-WP_R search: the candidates of one picture differ by a level step. */
static int64_t wp_cost_near(const pixel *cur, int lw, int lh,
                            const pixel *ref, int ls, pixel *tmp,
                            wp_cand c, const int8_t *mv)
{
    int64_t tot = 0;
    int bx, by, i, n = 0;

    for (i = 0; i < ls * (lh + 2 * WP_LPAD); i++)
        tmp[i] = wp_px(ref[i], c);
    for (by = 0; by + 8 <= lh; by += 8)
        for (bx = 0; bx + 8 <= lw; bx += 8, n++) {
            int best = 1 << 30, dx, dy, ie;
            for (dy = mv[n * 2 + 1] - 1; dy <= mv[n * 2 + 1] + 1; dy++)
                for (dx = mv[n * 2] - 1; dx <= mv[n * 2] + 1; dx++) {
                    int sd;
                    if (dx < -WP_R || dx > WP_R || dy < -WP_R || dy > WP_R)
                        continue;
                    sd = aeg_sad(8, 8, cur + by * lw + bx, lw,
                                 tmp + (by + dy + WP_LPAD) * ls + bx + dx +
                                     WP_LPAD, ls);
                    if (sd < best) best = sd;
                }
            ie = wp_intra_est(cur + by * lw + bx, lw);
            tot += best < ie ? best : ie;
        }
    return tot;
}

/* Chroma plane cost of `c` with the block vectors of the luma test. */
static int64_t wp_cost_chroma(const struct aeg_encoder *e, int pl, int lw,
                              int lh, const int8_t *mv, wp_cand c)
{
    const pixel *src = e->srcp[1 + pl], *ref = e->refpl[0][1 + pl];
    int cs = e->stride[1], bx, by, n = 0, x, y;
    int64_t tot = 0;
    pixel blk[64];

    for (by = 0; by + 8 <= lh; by += 8)
        for (bx = 0; bx + 8 <= lw; bx += 8, n++) {
            const pixel *r = ref + (by + mv[n * 2 + 1]) * cs + bx + mv[n * 2];
            for (y = 0; y < 8; y++)
                for (x = 0; x < 8; x++)
                    blk[y * 8 + x] = wp_px(r[y * cs + x], c);
            {
                int s = aeg_sad(8, 8, blk, 8, src + by * cs + bx, cs);
                int ie = wp_intra_est(src + by * cs + bx, cs);
                tot += s < ie ? s : ie;
            }
        }
    return tot;
}

static int wp_threshold(void)
{
    static _Atomic int v = -1;
    if (v < 0) {
        const char *ev = getenv("AEG_WPTH");
        v = ev ? atoi(ev) : 95;
    }
    return v;
}

#define WP_LUMA_PERMILLE 998

static void wp_analyse(struct aeg_encoder *e)
{
    int lw = e->w / 2, lh = e->h / 2, ls = lw + 2 * WP_LPAD;
    int x, y, pl, nblk = (lw / 8) * (lh / 8);
    double mc, vc, mr, vr;
    pixel *cur = NULL, *ref = NULL, *tmp = NULL;
    int8_t *mv = NULL, *mvw = NULL;
    wp_cand none = { 64, 0, 6 }, best = { 64, 0, 6 };
    int64_t c0, bc;
    int th = wp_threshold();
    double sg;

    memset(&e->wp, 0, sizeof(e->wp));
    if (!e->p.weightp || !e->cur_is_p || !e->has_ref || lw < 8 || lh < 8 ||
        e->mbaff || (e->fld && (!e->fwp || !e->wp_mem)))
        return;

    plane_stats(e->srcp[0], e->stride[0], e->w, e->h, &mc, &vc);
    plane_stats(e->refpl[0][0], e->stride[0], e->w, e->h, &mr, &vr);
    /* x264's early exit: no level shift worth half a step and no gain
     * change worth a numerator step at denominator 128 */
    sg = vr > 1.0 && vc > 0.0 ? sqrt(vc / vr) : 1.0;
    if (fabs(mc - mr) < 0.5 * (1 << (BIT_DEPTH - 8)) && fabs(sg - 1.0) < 1.0 / 128)
        goto chroma_only;

    cur = (pixel *)malloc((size_t)lw * lh * sizeof(pixel));
    ref = (pixel *)malloc((size_t)ls * (lh + 2 * WP_LPAD) * sizeof(pixel));
    tmp = (pixel *)malloc((size_t)ls * (lh + 2 * WP_LPAD) * sizeof(pixel));
    mv = (int8_t *)malloc((size_t)nblk * 2 + 2);
    mvw = (int8_t *)malloc((size_t)nblk * 2 + 2);
    if (!cur || !ref || !tmp || !mv || !mvw)
        goto done;
    for (y = 0; y < lh; y++)
        for (x = 0; x < lw; x++) {
            const pixel *s = e->srcp[0] + 2 * y * e->stride[0] + 2 * x;
            cur[y * lw + x] = (pixel)((s[0] + s[1] + s[e->stride[0]] +
                                         s[e->stride[0] + 1] + 2) >> 2);
        }
    for (y = -WP_LPAD; y < lh + WP_LPAD; y++)
        for (x = -WP_LPAD; x < lw + WP_LPAD; x++) {
            const pixel *s = e->refpl[0][0] + 2 * y * e->stride[0] + 2 * x;
            ref[(y + WP_LPAD) * ls + x + WP_LPAD] =
                (pixel)((s[0] + s[1] + s[e->stride[0]] +
                           s[e->stride[0] + 1] + 2) >> 2);
        }
    c0 = bc = wp_cost(cur, lw, lh, ref, ls, tmp, none, mv);
    {
        /* A global level change moves the DC of most well-predicted blocks
         * the same way; content moving in and out of the picture moves the
         * mean without that consistency and must not take an offset (the
         * synthetic pattern of the rate-control tests: -3..+4 mean change,
         * 0.7-2% half-resolution saving for an offset, +36% bits coded). */
        int bx, by, n = 0, npos = 0, nneg = 0, cons;
        for (by = 0; by + 8 <= lh; by += 8)
            for (bx = 0; bx + 8 <= lw; bx += 8, n++) {
                const pixel *r = ref + (by + mv[2 * n + 1] + WP_LPAD) * ls +
                                   bx + mv[2 * n] + WP_LPAD;
                const pixel *c = cur + by * lw + bx;
                int sc = 0, sr = 0, xx, yy;
                if (aeg_sad(8, 8, c, lw, r, ls) >= wp_intra_est(c, lw))
                    continue;
                for (yy = 0; yy < 8; yy++)
                    for (xx = 0; xx < 8; xx++) {
                        sc += c[yy * lw + xx];
                        sr += r[yy * ls + xx];
                    }
                if (sc - sr >= (32 << (BIT_DEPTH - 8))) npos++;
                else if (sc - sr <= -(32 << (BIT_DEPTH - 8))) nneg++;
            }
        /* dominant sign at least twice the other: short 2.5-3.7x, the
         * synthetic pattern 1.1-1.2x; 1.5x is within 0.06 of this on cock
         * and short, 3x loses a third of short's gain */
        cons = getenv("AEG_WPCONS") ? atoi(getenv("AEG_WPCONS")) : 20;
        if (getenv("AEG_WPDBG"))
            fprintf(stderr, "WPS mdiff %.2f pos %d neg %d of %d\n", mc - mr, npos, nneg, n);
        if ((npos > nneg ? npos : nneg) * 10 < (npos > nneg ? nneg : npos) * cons)
            goto chroma_only;
        /* A field, and an eighth of the blocks at least: a cut between a
         * frame's two fields leaves a few well-predicted blocks of the second
         * all one way (megamind: 7 of 176, an offset of 29 that saved
         * nothing); the fields of a fade have 65-90 %, cockatoo's drift
         * 18-39 % */
        if (e->fld && (npos > nneg ? npos : nneg) * 8 < n)
            goto chroma_only;
    }
    {
        /* x264's search (weights_analyse at subme 7): the gain matching the
         * standard deviations and one step either side, each with the offset
         * matching the means (truncated, as x264 does) and one either side,
         * measured around the unweighted vectors.  Kept when it saves 0.2%
         * (x264's own termination); the earlier rule -- 5% of the sum and a
         * quarter of a sample per pixel -- rejected the -1/-2 offsets of a
         * slow exposure drift that x264 uses on every P picture of short. */
        int w0 = (int)floor(sg * 64 + 0.5), ws;
        wp_cand g = { w0 > 127 ? 127 : w0, 0, 6 };

        /* the guess itself with a full search: on a fade the unweighted
         * vectors are wrong, and the other candidates are refined around
         * the guess's */
        g.o = (int)((mc - mr * g.w / 64.0) / (1 << (BIT_DEPTH - 8)));
        g.o = g.o < -128 ? -128 : (g.o > 127 ? 127 : g.o);
        if (!wp_identity(g)) {
            int64_t k = wp_cost(cur, lw, lh, ref, ls, tmp, g, mvw);
            if (k < bc) { bc = k; best = g; }
        } else
            memcpy(mvw, mv, (size_t)nblk * 2 + 2);
        for (ws = w0 - 1; ws <= w0 + 1; ws++) {
            int o0, o;
            if (ws < 0 || ws > 127)
                continue;
            o0 = (int)((mc - mr * ws / 64.0) / (1 << (BIT_DEPTH - 8)));
            for (o = o0 - 1; o <= o0 + 1; o++) {
                wp_cand c = { ws, o < -128 ? -128 : (o > 127 ? 127 : o), 6 };
                int64_t k;
                if (wp_identity(c) || (c.w == g.w && c.o == g.o))
                    continue;
                k = wp_cost_near(cur, lw, lh, ref, ls, tmp, c, mvw);
                if (k < bc) { bc = k; best = c; }
            }
        }
    }
    if (getenv("AEG_WPDBG"))
        fprintf(stderr, "WPL c0 %lld best (%d,%d) %lld\n", (long long)c0,
                best.w, best.o, (long long)bc);
    if (!wp_identity(best) && bc * 1000 <= c0 * WP_LUMA_PERMILLE) {
        wp_cost(cur, lw, lh, ref, ls, tmp, best, mv);   /* chroma's vectors */
        while (best.d > 0 && !(best.w & 1)) { best.w >>= 1; best.d--; }
        e->wp.lflag[0] = 1;
        e->wp.luma_denom = best.d;
        e->wp.lw[0] = best.w;
        e->wp.lo[0] = best.o;
    }

chroma_only:
    if (!mv) {
        /* the luma did not move: vectors of the unweighted test */
        if (!cur) {
            cur = (pixel *)malloc((size_t)lw * lh * sizeof(pixel));
            ref = (pixel *)malloc((size_t)ls * (lh + 2 * WP_LPAD) * sizeof(pixel));
            tmp = (pixel *)malloc((size_t)ls * (lh + 2 * WP_LPAD) * sizeof(pixel));
            mv = (int8_t *)malloc((size_t)nblk * 2 + 2);
        }
        if (!cur || !ref || !tmp || !mv)
            goto done;
        /* identical guesses mean identical statistics: skip the search and
         * test chroma at zero motion */
        memset(mv, 0, (size_t)nblk * 2 + 2);
    }
    {
        wp_cand cc[2][2];
        int64_t cn = 0, cs[2] = { 0, 0 };
        int cbest = -1, k;

        for (pl = 0; pl < 2; pl++) {
            double m1, v1, m2, v2;
            plane_stats(e->srcp[1 + pl], e->stride[1], lw, lh, &m1, &v1);
            plane_stats(e->refpl[0][1 + pl], e->stride[1], lw, lh, &m2, &v2);
            cc[0][pl] = wp_guess(m1, v1, m2, v2, 1);
            cc[1][pl] = wp_guess(m1, v1, m2, v2, 0);
        }
        /* one denominator for both chroma planes */
        for (k = 0; k < 2; k++)
            if (cc[k][0].d != cc[k][1].d) {
                int d = cc[k][0].d < cc[k][1].d ? cc[k][0].d : cc[k][1].d;
                for (pl = 0; pl < 2; pl++)
                    while (cc[k][pl].d > d) {
                        cc[k][pl].w = (cc[k][pl].w + 1) >> 1;
                        cc[k][pl].d--;
                    }
            }
        if ((wp_identity(cc[0][0]) && wp_identity(cc[0][1])) &&
            (wp_identity(cc[1][0]) && wp_identity(cc[1][1])))
            goto done;
        for (pl = 0; pl < 2; pl++) {
            cn += wp_cost_chroma(e, pl, lw, lh, mv, none);
            for (k = 0; k < 2; k++)
                cs[k] += wp_cost_chroma(e, pl, lw, lh, mv, cc[k][pl]);
        }
        for (k = 0; k < 2; k++)
            if (cs[k] * 100 < cn * th &&
                cn - cs[k] >= ((int64_t)lw * lh / 2) << (BIT_DEPTH - 8) &&
                (cbest < 0 || cs[k] < cs[cbest]))
                cbest = k;
        if (cbest >= 0) {
            e->wp.cflag[0] = 1;
            e->wp.chroma_denom = cc[cbest][0].d;
            for (pl = 0; pl < 2; pl++) {
                e->wp.cw[0][pl] = cc[cbest][pl].w;
                e->wp.co[0][pl] = cc[cbest][pl].o;
            }
        }
    }

done:
    free(cur); free(ref); free(tmp); free(mv); free(mvw);
    if (e->wp.lflag[0]) {
        /* the weighted copy the motion search works on */
        size_t ls0 = (size_t)e->stride[0], lh0 = (size_t)(e->h + 2 * AEG_PAD);
        wp_cand c = { e->wp.lw[0], e->wp.lo[0], e->wp.luma_denom };
        const pixel *s;
        pixel *d;
        size_t k;
        /* a field picture: the reference field's rows of the copy (every
         * other row of the frame-sized buffer, allocated before the field
         * views -- enc_alloc_picture sets the frame's strides) */
        ptrdiff_t po = e->fld ? (ptrdiff_t)e->ref_par[0] * (ptrdiff_t)(ls0 / 2) : 0;

        if (!e->wp_mem && enc_alloc_picture(e, &e->wp_mem, e->wp_plane) != AEG_OK) {
            e->wp_mem = NULL;
            memset(&e->wp, 0, sizeof(e->wp));
            return;
        }
        s = e->refpl[0][0] - AEG_PAD * ls0 - AEG_PAD;
        d = e->wp_plane[0] + po - AEG_PAD * ls0 - AEG_PAD;
        if (e->fld) {
            size_t r, rl = ls0 / 2;
            for (r = 0; r < lh0; r++)
                for (k = 0; k < rl; k++)
                    d[r * ls0 + k] = wp_px(s[r * ls0 + k], c);
        } else {
            for (k = 0; k < ls0 * lh0; k++)
                d[k] = wp_px(s[k], c);
        }
        e->wp_orig0 = e->refpl[0][0];
        e->refpl[0][0] = e->wp_plane[0] + po;
    }
    if (getenv("AEG_WPDBG"))
        fprintf(stderr, "WP poc %d luma %d (%d,%d/%d) chroma %d (%d,%d %d,%d /%d)\n",
                e->cur_poc, e->wp.lflag[0], e->wp.lw[0], e->wp.lo[0],
                e->wp.luma_denom, e->wp.cflag[0], e->wp.cw[0][0],
                e->wp.co[0][0], e->wp.cw[0][1], e->wp.co[0][1],
                e->wp.chroma_denom);
}

/* The multi-hexagon stage of the motion search (umh_search, mc.c), per
 * block class: 16x16 / 16x8 and 8x16 / 8x8 / smaller than 8x8.  Measured (ours
 * against ours, global PSNR): 16x16 only -0.14% for +1.6% time; with the
 * 16x8/8x16 searches -0.47% for +8%; every block -0.53% for +26%.  At preset
 * 4, against radius 16 on every block: the 8x8 search at radius 8 and none
 * below 8x8 measured +0.00 % on the five real clips and -0.03 % on six other
 * segments of the same sources, for -8.5 % of the time (none below 8x8 at all
 * +0.05 % for -7 %, radius 8 on both +0.01 % for -4 %).
 * AEG_UMH / AEG_UMHP / AEG_UMH8 / AEG_UMHS override the four radii. */
static void umh_radii(const struct aeg_encoder *e, int r[4], int *et)
{
    static const char *const names[4] = { "AEG_UMH", "AEG_UMHP", "AEG_UMH8",
                                          "AEG_UMHS" };
    int pr = e->p.preset, k;

    r[0] = pr >= 1 ? 16 : 0;
    r[1] = pr >= 3 ? 16 : 0;
    r[2] = pr >= 4 ? 8 : 0;
    r[3] = 0;
    for (k = 0; k < 4; k++) {
        const char *ev = getenv(names[k]);
        if (ev)
            r[k] = atoi(ev);
    }
    /* UMH early exit (mc.c): threshold 1000 per 16x16 below preset 4 measured
     * +0.01 / +0.11 % (two clip sets) for -6.6 / -6.1 % of the time at p3;
     * at p4 500 measured +0.03 / +0.07 % for -6.1 / -6.2 %. */
    *et = pr >= 4 ? 500 : 1000;
    {
        const char *ev = getenv("AEG_UMHET");
        if (ev)
            *et = atoi(ev);
    }
}

static int chroma_me_on(const struct aeg_encoder *e)
{
    static _Atomic int v = -2;
    if (v == -2) {
        const char *ev = getenv("AEG_CHROMAME");
        v = ev ? atoi(ev) : -1;
    }
    /* measured -0.02 % for +18 % time (docs/STATUS.md): off by default;
     * not in a field (the cost would need the chroma field offset) */
    return v > 0 && !e->fld;
}

static int bref_qp_num(void)
{
    static _Atomic int v = -1;
    if (v < 0) {
        const char *ev = getenv("AEG_BREFQP");   /* offset in halves */
        v = ev ? atoi(ev) : 2;
    }
    return v;
}

/* Undo wp_analyse's substitution once the picture is coded. */
static void wp_finish(struct aeg_encoder *e)
{
    if (e->wp.lflag[0])
        e->refpl[0][0] = (pixel *)e->wp_orig0;
    memset(&e->wp, 0, sizeof(e->wp));
}

/* Code one picture.  `slice` is AEG_SLICE_I/P/B, `poc` its picture order count
 * -- the display index, which is what direct mode scales its vectors by.  The
 * caller has already bound e->srcp to the picture's source samples (see
 * aeg_encoder_encode) and decided that it is this picture's turn. */
/* encode_frame() is the prologue -- everything that reads the encoder's
 * picture-to-picture state (reference lists, quantiser, lookahead offsets)
 * -- followed by the body, which codes the picture from what the prologue
 * left.  The frame threads run the prologue on the main thread and the body
 * on a worker. */
static int frame_prologue(struct aeg_encoder *e, int *slice_io, int *idr_io,
                          int poc)
{
    int is_p, slice = *slice_io, is_idr = *idr_io;

    /* Which lists this picture predicts from.  A B picture reads both anchors;
     * a P picture reads the newest one.  This has to happen before the scene
     * cut test, which compares the picture with its list 0 reference. */
    e->cur_poc = poc;
    e->cur_is_b = slice == AEG_SLICE_B;
    e->cur_is_ref = !e->cur_is_b || e->cur_bref;
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
        aeg_log(AEG_LOG_DEBUG, "aeg: scene cut at frame %d -> IDR", e->frame_count);
    }
    is_p = slice == AEG_SLICE_P && !is_idr && e->has_ref;
    if (is_p != e->cur_is_p) {
        e->cur_is_p = is_p;
        set_current_refs(e);
    }
    /* direct mode reads the decisions of list1[0], whichever buffer that is
     * (the next anchor, or the group's reference B picture) */
    if (e->cur_is_b && e->cur_l1_slot >= 0 && e->slot_mbs[e->cur_l1_slot]) {
        int s = e->cur_l1_slot;
        e->col_mbs = e->slot_mbs[s];
        e->col_poc = e->pic_poc[s];
        memcpy(e->col_ref_poc, e->slot_ref_poc[s], sizeof(e->col_ref_poc));
        e->col_ready = e->slot_col_ok[s];
    } else if (e->cur_is_b) {
        e->col_ready = 0;
    }
    if (is_idr) {
        e->last_idr_frame = e->frame_count;
        e->last_idr_poc = poc;
    }

    /* the picture's QP: constant in CQP, the controller's output in ABR */
    e->cur_qp = e->p.qp;
    e->crf_frac = 0.0;
    if (e->crf_on) {
        /* An anchor follows its complexity (x264's get_qscale: blurred over
         * the last few anchors, so that one odd picture does not swing the
         * quality); an intra picture takes the anchors' recent average and a
         * B picture the anchor coded just before it -- the offsets the
         * picture type gets in CQP are applied below, as they would be to
         * -q. */
        double q;
        if (is_idr || slice == AEG_SLICE_I) {
            q = e->crf_pqp / e->crf_pnorm;
            if (e->abr_on)
                q = abr_intra_q(e, q, poc);
        } else if (e->cur_is_b) {
            q = e->crf_anchor_q;
            e->abr_rceq = e->abr_last_rceq;
        } else {
            int64_t c = e->crf_qc <= 0.0 ? -1
                      : e->p.bframes > 0 ? e->crf_cplx[poc % in_slots(e)]
                      : e->crf_last;
            if (e->abr_on) {
                q = abr_anchor_q(e, c, poc);
            } else if (c >= 0) {
                double qs;
                e->crf_sum = e->crf_sum * 0.5 + (double)c;
                e->crf_cnt = e->crf_cnt * 0.5 + 1.0;
                qs = pow(e->crf_sum / e->crf_cnt, e->crf_qc) / e->crf_rf;
                q = 12.0 + 6.0 * log2(qs / 0.85);
            } else {
                q = e->p.crf + e->crf_tree_off;
            }
            if (q < QP_MIN) q = QP_MIN;
            if (q > 51.0) q = 51.0;
            e->crf_pqp = e->crf_pqp * 0.95 + q;
            e->crf_pnorm = e->crf_pnorm * 0.95 + 1.0;
        }
        if (!e->cur_is_b)
            e->crf_anchor_q = q;
        e->abr_qm = q;
        e->cur_qp = (int)floor(q);
        e->crf_frac = q - e->cur_qp;
        if (e->cur_qp >= 51) e->cur_qp = 51, e->crf_frac = 0.0;
    }
    if ((is_idr || slice == AEG_SLICE_I) && e->p.i_qp_offset > 0) {
        e->cur_qp -= e->p.i_qp_offset;
        if (e->cur_qp < QP_MIN) e->cur_qp = QP_MIN;
    }
    {
        int qoff = e->p.bframes > 0 ? e->in_qpoff[poc % in_slots(e)]
                                    : e->rc_qp_offset;
        e->cur_qp += qoff;
        if (e->cur_qp < QP_MIN) e->cur_qp = QP_MIN;
        if (e->cur_qp > 51) e->cur_qp = 51;
    }
    /* A B picture is not a reference, so its quantiser never propagates: the
     * anchors keep the QP the controller gave them and the B pictures in
     * between can be rounded harder without costing the following pictures
     * anything.  This is the cheapest half of what B pictures are for --
     * x264 spends +2 by default for exactly this reason. */
    e->mode_lambda_off = 0;
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

        /* a reference B picture is predicted from by its neighbours, so it
         * sits half way between the anchors and the other B pictures */
        if (e->cur_bref) {
            const char *ev = getenv("AEG_BREFOFF");
            off = ev ? atoi(ev) : off * bref_qp_num() / 2;
        }
        if (e->cur_qp < B_QP_FADE)
            off = off * e->cur_qp / B_QP_FADE;
        e->cur_qp += off;
        e->mode_lambda_off = off;
        if (e->cur_qp < QP_MIN) e->cur_qp = QP_MIN;
        if (e->cur_qp > 51) e->cur_qp = 51;
    }
    e->mbt_qoff = mbt_mean(e);
    e->cur_is_intra = is_idr || slice == AEG_SLICE_I;
    if (e->p.bframes > 0)
        e->cur_satd = e->vbv_satd ? e->vbv_satd[poc % in_slots(e)] : -1;
    e->abr_qpre = e->cur_qp;
    if (e->vbv_active)
        vbv_clamp_qp(e);
    if (e->cur_qp < e->p.qp_min) e->cur_qp = e->p.qp_min;
    if (e->cur_qp > e->p.qp_max) e->cur_qp = e->p.qp_max;

    resolve_coef_stages(e);
    aq_compute(e);
    mbt_apply(e);
    crf_dither(e, poc);
    if (aeg_log_level >= AEG_LOG_DEBUG) {
        long t = 0;
        int i;
        for (i = 0; i < e->mb_count; i++)
            t += mb_qp_of(e, i % e->mb_w, i / e->mb_w);
        aeg_log(AEG_LOG_DEBUG, "aeg: picture %d %c qp %d, macroblock average %.2f", poc,
                is_idr || slice == AEG_SLICE_I ? 'I' : e->cur_is_b ? 'B' : 'P',
                e->cur_qp, (double)t / e->mb_count);
    }
    *slice_io = slice;
    *idr_io = is_idr;
    return AEG_OK;
}

/* The slices of the picture (or field) being coded, threaded or not, as NAL
 * units appended to e->out at *out_len. */
static int code_slices(struct aeg_encoder *e, int slice, int is_idr,
                       size_t *out_len)
{
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
                /* the dispatcher measurements go straight to the root's
                 * (aeg_dispatcher.root), the device calls to a lane of the
                 * slice's own */
                copies[si].lane = e->lane + si;
                aeg_be_lane(&e->d.ops, copies[si].lane, &copies[si].d.ops);
                copies[si].d.root = e->d.root ? e->d.root : &e->d;
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
                int rc = code_slice(e, e->slice_bs[si], e->slice_bs_cap[si],
                                    first_row, last_row - 1, slice, is_idr,
                                    &sl_len);
                if (rc == FT_RESTART)
                    return FT_RESTART;
                if (rc != AEG_OK)
                    return AEG_ENOMEM;
                lens[si] = sl_len;
            }
        }

        for (si = 0; si < e->n_slices; si++) {
            if (!lens[si])
                break;
            /* A B picture is not used as a reference, so its nal_ref_idc is 0
             * and the decoder is free to drop it as soon as it is output. */
            *out_len += aeg_nal_write(e->out + *out_len, e->out_cap - *out_len,
                                     is_idr ? 3 : (e->cur_is_ref ? 2 : 0),
                                     is_idr ? 5 : 1, e->slice_bs[si], lens[si],
                                     0);
        }
    }
    return AEG_OK;
}

/* ------------------------------------------------------------------ */
/* Field pictures (PAFF)                                               */
/* ------------------------------------------------------------------ */
/* With p.field_coding a frame is coded as two field pictures, the first
 * field in display order first.  Each is an access unit of its own
 * (delimiter, picture timing SEI with pic_struct 1 or 2, slices with
 * field_pic_flag 1), both carry the frame's frame_num and nal_ref_idc, and
 * the second may predict from the first.  The frame stays the unit of
 * everything around the picture coder: the lookahead, the rate control and
 * the macroblock tree decide per frame, the buffers hold frames (a field is
 * every other row of one, see field_enter), and one packet carries both
 * fields. */
static int field_first_par(const struct aeg_encoder *e)
{
    return e->p.interlaced == 2;    /* bottom field first */
}

/* The picture order count of field `par` of the frame at display index
 * `poc`: the first field 2 * poc, the second one more. */
static int field_poc(const struct aeg_encoder *e, int poc, int par)
{
    return 2 * poc + (par != field_first_par(e));
}

/* Make field `f` (0 = first in decoding order) the picture being coded. */
static void field_enter(struct aeg_encoder *e, int f)
{
    struct fld_save *v = &e->fsv;
    int i, par = f ^ field_first_par(e);

    v->h = e->h; v->ch = e->ch; v->vh = e->vh; v->vch = e->vch;
    v->mb_h = e->mb_h; v->mb_count = e->mb_count;
    v->slice_rows = e->slice_rows;
    v->cur_poc = e->cur_poc;
    v->cur_is_p = e->cur_is_p; v->cur_is_b = e->cur_is_b;
    v->cur_is_intra = e->cur_is_intra;
    v->pic = e->pic;
    v->mb_dqp = e->mb_dqp;
    for (i = 0; i < 3; i++) {
        v->stride[i] = e->stride[i];
        v->srcp[i] = e->srcp[i];
        v->recp[i] = e->recp[i];
        e->srcp[i] += par * e->stride[i];
        e->recp[i] += par * e->stride[i];
        e->stride[i] *= 2;
        e->pic.plane[i] = e->recp[i];
        e->pic.stride[i] = e->stride[i];
    }
    e->h /= 2; e->ch /= 2; e->vh /= 2; e->vch /= 2;
    e->mb_h /= 2; e->mb_count /= 2;
    e->pic.h[0] = e->h;
    e->pic.h[1] = e->pic.h[2] = e->ch;
    e->pic.mb_h = e->mb_h;
    e->slice_rows = (e->mb_h + e->n_slices - 1) / e->n_slices;
    /* the frame's per-macroblock QP offsets: the field's half of the array */
    e->mb_dqp = v->mb_dqp + f * e->mb_count;
    e->cur_poc = field_poc(e, v->cur_poc, par);
    e->fld = 1;
    e->fld_idx = f;
    e->fld_par = par;
    e->scan4 = aeg_field_scan4x4;
    e->scan8 = aeg_field_scan8x8;
    e->fcat = 6;
    e->fscan = AEG_CAVLC_FIELD;
    e->fscan8 = AEG_CAVLC_FIELD8;
}

static void field_leave(struct aeg_encoder *e)
{
    struct fld_save *v = &e->fsv;
    int i;

    e->h = v->h; e->ch = v->ch; e->vh = v->vh; e->vch = v->vch;
    e->mb_h = v->mb_h; e->mb_count = v->mb_count;
    e->slice_rows = v->slice_rows;
    e->cur_poc = v->cur_poc;
    e->cur_is_p = v->cur_is_p; e->cur_is_b = v->cur_is_b;
    e->cur_is_intra = v->cur_is_intra;
    e->pic = v->pic;
    e->mb_dqp = v->mb_dqp;
    for (i = 0; i < 3; i++) {
        e->stride[i] = v->stride[i];
        e->srcp[i] = v->srcp[i];
        e->recp[i] = v->recp[i];
    }
    e->fld = 0;
    e->scan4 = aeg_scan4x4;
    e->scan8 = aeg_zigzag8x8;
    e->fcat = e->fscan = e->fscan8 = 0;
}

/* The default reference field lists of 8.2.4.2.2/8.2.4.2.4 with 8.2.4.2.5:
 * the reference frames (and the first field of the current frame, once it
 * is a reference) ordered as for a frame -- by decreasing frame_num for a
 * P field; for a B field by picture order count, the ones before it
 * nearest first then the ones after (list 0) or the other way round
 * (list 1) -- then their fields taken alternately, starting with the
 * current field's parity.  Nothing is reordered or marked explicitly: the
 * sliding window does the marking (field_mark). */
/* P field, `pw` 1: the order the encoder wants instead (with a B pyramid
 * the default puts the reference B just coded first): the current frame (its
 * first field), the newest anchor, then the rest by decreasing POC -- which
 * without a reference B is the default order (anchors are coded in display
 * order).  set_field_refs modifies the list to it. */
static int pfield_rank(const struct aeg_encoder *e, int i, int newest)
{
    return e->dpb[i].slot == e->role_cur ? 0 : i == newest ? 1 : 2;
}

static int field_list(const struct aeg_encoder *e, int l, int out[][2], int pw)
{
    int ord[AEG_MAX_DPB], n = 0, i, j, cnt = 0;
    int ptr[2] = { 0, 0 }, want = e->fld_par, newest = -1;
    const int cur = e->cur_poc;

    if (pw)
        for (i = 0; i < e->ndpb; i++)
            if (e->dpb[i].fm && e->dpb[i].anchor && e->dpb[i].slot != e->role_cur &&
                (newest < 0 || e->dpb[i].poc > e->dpb[newest].poc))
                newest = i;

    for (i = 0; i < e->ndpb; i++) {
        int ki, kj;
        if (!e->dpb[i].fm)
            continue;
        for (j = n; j > 0; j--) {
            const int a = ord[j - 1];
            int before;
            if (e->cur_is_b) {
                /* frame order count: its first field's */
                int pa = 2 * e->dpb[a].poc, pi = 2 * e->dpb[i].poc;
                int ga = pa > cur, gi = pi > cur;   /* 0: before, 1: after */
                if (l)
                    ga = !ga, gi = !gi;
                ki = gi; kj = ga;
                if (ki != kj)
                    before = ki < kj;
                else
                    /* (both lists: the ones after the current field
                     * nearest first ascending, the ones before
                     * descending) */
                    before = pi > cur ? pi < pa : pi > pa;
            } else if (pw) {
                int ri = pfield_rank(e, i, newest), ra = pfield_rank(e, a, newest);
                before = ri != ra ? ri < ra : e->dpb[i].poc > e->dpb[a].poc;
            } else {
                before = e->dpb[i].fn > e->dpb[a].fn;
            }
            if (!before)
                break;
            ord[j] = ord[j - 1];
        }
        ord[j] = i;
        n++;
    }
    for (;;) {
        int *p = &ptr[want != e->fld_par];
        while (*p < n && !(e->dpb[ord[*p]].fm & (1 << want)))
            (*p)++;
        if (*p < n) {
            out[cnt][0] = ord[*p];
            out[cnt][1] = want;
            cnt++;
            (*p)++;
            want ^= 1;
            continue;
        }
        /* none of this parity left: the rest of the other one */
        want ^= 1;
        p = &ptr[want != e->fld_par];
        for (; *p < n; (*p)++)
            if (e->dpb[ord[*p]].fm & (1 << want)) {
                out[cnt][0] = ord[*p];
                out[cnt][1] = want;
                cnt++;
            }
        break;
    }
    return cnt;
}

static void field_set_ref(struct aeg_encoder *e, int k, int di, int par)
{
    int s = e->dpb[di].slot, i;

    for (i = 0; i < 3; i++)
        e->refpl[k][i] = e->pic_plane[s][i] + par * e->fsv.stride[i];
    e->ref_slot[k] = s;
    e->ref_par[k] = par;
    e->ref_poc[k] = field_poc(e, e->dpb[di].poc, par);
    e->fld_cofs[k] = 2 * (e->fld_par - par);
}

static void set_field_refs(struct aeg_encoder *e)
{
    int l0[2 * AEG_MAX_DPB][2], l1[2 * AEG_MAX_DPB][2];
    int n0, n1 = 0, k;

    memset(&e->refctl, 0, sizeof(e->refctl));
    for (k = 0; k < AEG_MAX_REFS; k++) {
        e->refpl[k][0] = e->refpl[k][1] = e->refpl[k][2] = NULL;
        e->ref_slot[k] = -1;
        e->ref_poc[k] = 0;
        e->ref_par[k] = 0;
        e->fld_cofs[k] = 0;
    }
    e->cur_l1_slot = -1;
    e->has_ref = e->has_ref2 = 0;
    e->cur_l0_active = 1;
    e->bfp_on = 0;
    if (!e->cur_is_p && !e->cur_is_b)
        return;
    n0 = field_list(e, 0, l0, 0);
    if (e->cur_is_b) {
        n1 = field_list(e, 1, l1, 0);
        if (n1 > 1 && n1 == n0) {
            /* list 1 identical to list 0: its first two swap (8.2.4.2.4) */
            int same = 1;
            for (k = 0; k < n0; k++)
                same &= l0[k][0] == l1[k][0] && l0[k][1] == l1[k][1];
            if (same) {
                int t0 = l1[0][0], t1 = l1[0][1];
                l1[0][0] = l1[1][0]; l1[0][1] = l1[1][1];
                l1[1][0] = t0; l1[1][1] = t1;
            }
        }
        if (n0) {
            field_set_ref(e, 0, l0[0][0], l0[0][1]);
            e->has_ref = 1;
        }
        if (n1) {
            field_set_ref(e, 1, l1[0][0], l1[0][1]);
            e->has_ref2 = 1;
            e->cur_l1_slot = e->ref_slot[1];
        }
        /* Two fields per list (index 1: the other parity of the nearest
         * frame, as a rule) when both lists have them and share none -- the
         * deblocking filter tells pictures apart by list and index
         * (AEG_BSAMEPAR: one each, as before) */
        if (n0 >= 2 && n1 >= 2 && !getenv("AEG_BSAMEPAR")) {
            int i, j, dis = 1;
            for (i = 0; i < 2; i++)
                for (j = 0; j < 2; j++)
                    dis &= l0[i][0] != l1[j][0] || l0[i][1] != l1[j][1];
            if (dis) {
                field_set_ref(e, 2, l0[1][0], l0[1][1]);
                field_set_ref(e, 3, l1[1][0], l1[1][1]);
                e->bfp_on = 1;
            }
        }
        return;
    }
    /* P: both fields of each of --ref frames (the second field of a frame
     * has its first field as the nearest picture of the other parity) */
    {
        int na = 2 * e->p.refs, lw[2 * AEG_MAX_DPB][2], ok = 1;
        if (na > n0) na = n0;
        if (na > AEG_MAX_REFS) na = AEG_MAX_REFS;
        /* the wanted order (field_list pw), by ref_pic_list_modification
         * when the default's first entries differ: field picture numbers,
         * 2 FrameNumWrap + 1 for the current parity and 2 FrameNumWrap for
         * the other, from CurrPicNum = 2 frame_num + 1 (8.2.4.1, 8.2.4.3.1) */
        field_list(e, 0, lw, 1);
        for (k = 0; k < na; k++)
            ok &= lw[k][0] == l0[k][0] && lw[k][1] == l0[k][1];
        if (!ok) {
            int pred = 2 * e->frame_num + 1;
            for (k = 0; k < na; k++) {
                int pn = 2 * e->dpb[lw[k][0]].fn + (lw[k][1] == e->fld_par);
                if (pn < pred) {
                    e->refctl.mod_idc[k] = 0;
                    e->refctl.mod_abs[k] = pred - pn - 1;
                } else {
                    e->refctl.mod_idc[k] = 1;
                    e->refctl.mod_abs[k] = pn - pred - 1;
                }
                pred = pn;
            }
            e->refctl.n_mod = na;
            e->refctl.n_mod_1ref = lw[0][0] != l0[0][0] || lw[0][1] != l0[0][1];
            memcpy(l0, lw, sizeof(lw));
        }
        for (k = 0; k < na; k++)
            field_set_ref(e, k, l0[k][0], l0[k][1]);
        e->has_ref = na > 0;
        e->has_ref2 = na > 1;
        e->cur_l0_active = na > 0 ? na : 1;
    }
}

/* The reference marking after a field of a reference frame (8.2.5.3): the
 * first field goes through the sliding window -- which drops the frame with
 * the smallest frame_num once max_num_ref_frames are held -- and is stored
 * as a frame with one reference field; the second only marks its field. */
static void field_mark(struct aeg_encoder *e, int fidr)
{
    int i, bit = 1 << e->fld_par;

    if (e->fld_idx == 0) {
        if (fidr) {
            e->ndpb = 0;
        } else if (e->ndpb >= e->max_refs) {
            int m = 0;
            for (i = 1; i < e->ndpb; i++)
                if (e->dpb[i].fn < e->dpb[m].fn)
                    m = i;
            e->dpb[m] = e->dpb[--e->ndpb];
        }
        e->dpb[e->ndpb].slot = e->role_cur;
        e->dpb[e->ndpb].poc = e->fsv.cur_poc;
        e->dpb[e->ndpb].fn = e->frame_num;
        e->dpb[e->ndpb].anchor = !e->fsv.cur_is_b;
        e->dpb[e->ndpb].fm = bit;
        e->ndpb++;
    } else {
        for (i = 0; i < e->ndpb; i++)
            if (e->dpb[i].slot == e->role_cur)
                e->dpb[i].fm |= bit;
        e->frame_num++;
    }
}

static void hrd_timing(struct aeg_encoder *e, aeg_sei *s, int is_idr, int poc,
                       int tick);

static int field_frame_body(struct aeg_encoder *e, int slice, int is_idr,
                            int poc, int64_t pts, aeg_packet *pkt,
                            int64_t t_all)
{
    const int is_ref = e->cur_is_ref;
    const int fp = e->cur_is_p, fb = e->cur_is_b;
    size_t out_len = 0;
    int f, mstride[AEG_MAX_REFS];

    /* (the reference views below double the strides; a frame picture
     * after this one expects the frame's) */
    for (f = 0; f < AEG_MAX_REFS; f++)
        mstride[f] = e->mcr[f].stride;

    if (ensure_buf(&e->out, &e->out_cap,
                   (size_t)e->mb_count * 400 * 3 + 65536) != AEG_OK)
        return AEG_ENOMEM;
    e->ft_spec = 0;
    e->ft_abort = 0;
    /* frame threads: field pictures never restart, and nothing they read
     * does (a weighted field waits for its whole reference field instead of
     * speculating), so they register no generations */
    for (f = 0; f < AEG_MAX_REFS; f++)
        e->ft_gslot[f] = -1;
    e->ft_reg = 0;
    memset(&e->wp, 0, sizeof(e->wp));
    if (e->fwp && !e->wp_mem &&
        enc_alloc_picture(e, &e->wp_mem, e->wp_plane) != AEG_OK)
        e->wp_mem = NULL;      /* (the fields go without weights) */
    if (is_idr)
        e->frame_num = 0;
    for (f = 0; f < 2; f++) {
        /* the second field of an intra frame predicts from the first */
        int fslice = f && slice == AEG_SLICE_I ? AEG_SLICE_P : slice;
        int fidr = is_idr && f == 0;
        int rc, fretry = 0, bfp_on0, srows0;
        size_t fstart;
        int64_t stat0[AEG_MB_TYPE_COUNT];

        field_enter(e, f);
        e->cur_is_p = fslice == AEG_SLICE_P;
        e->cur_is_b = fslice == AEG_SLICE_B;
        e->cur_is_intra = fslice == AEG_SLICE_I;
        set_field_refs(e);
        if ((e->cur_is_p || e->cur_is_b) && !e->has_ref) {
            fslice = AEG_SLICE_I;      /* (nothing to predict from) */
            e->cur_is_p = e->cur_is_b = 0;
            e->cur_is_intra = 1;
            set_field_refs(e);
        }
        e->col_ready = 0;
        if (e->cur_is_b && e->has_ref2 && e->slot_mbs[e->cur_l1_slot]) {
            /* direct mode's co-located picture: that field of list1[0] */
            int s = e->cur_l1_slot;
            e->col_mbs = e->slot_mbs[s] + e->ref_par[1] * e->mb_count;
            e->col_poc = e->ref_poc[1];
            memset(e->col_ref_poc, 0, sizeof(e->col_ref_poc));
            e->col_ready = e->slot_col_ok[s];
        }
        aq_compute(e);
        mbt_apply(e);
        crf_dither(e, e->cur_poc);
        if (e->fwp && e->cur_is_p && e->has_ref) {
            /* explicit weighted prediction of list 0's first field, from
             * its reconstruction, as a frame's: the whole field is read
             * (frame threads wait for it; the current frame's first field
             * is this worker's own) */
            if (e->ref_slot[0] != e->role_cur)
                (void)ref_wait(e, e->ref_slot[0], e->fsv.mb_h);
            wp_analyse(e);
        }
        if ((e->cur_is_p || e->cur_is_b) && e->p.subme > 0) {
            /* the reference fields' planes: every other row of their
             * frames' (built as each field was coded, rowpost_hpel) */
            int64_t t0 = aeg_now_ns();
            int l, i;
            for (l = 0; l < nrefpl(e); l++) {
                aeg_mc_ctx *ctx = &e->mcr[l];
                const aeg_mc_ctx *sm;
                int s = e->ref_slot[l];
                if (s < 0)
                    continue;
                sm = &e->slot_mc[s];
                ctx->base[0] = e->refpl[l][0];
                if (l == 0 && e->wp.lflag[0] && ctx->buf) {
                    /* a weighted field: planes of the weighted copy, in
                     * the field's rows of this list's own buffer */
                    const int fs = e->fsv.stride[0];
                    const int vp = ctx->vpad ? ctx->vpad : AEG_PAD;
                    const size_t rows = (size_t)e->fsv.h + 2 * vp;
                    for (i = 1; i < 4; i++) {
                        ctx->base[i] = ctx->buf + (size_t)(i - 1) * fs * rows +
                                       (size_t)vp * fs + AEG_PAD +
                                       (size_t)e->ref_par[0] * fs;
                        aeg_hpel_plane_fill((pixel *)ctx->base[i], 2 * fs,
                                            e->refpl[0][0], 2 * fs, e->w,
                                            e->h, i);
                    }
                    ctx->stride = 2 * fs;
                    ctx->ready = 1;
                    continue;
                }
                for (i = 1; i < 4; i++)
                    ctx->base[i] = sm->base[i] + e->ref_par[l] * sm->stride;
                ctx->stride = 2 * sm->stride;
                ctx->ready = 1;
            }
            for (l = 0; l < AEG_MAX_REFS; l++) {
                aeg_mc_ctx *ctx = &e->mcr[l];
                ctx->chroma = 0;
                ctx->cref[0] = e->refpl[l][1];
                ctx->cref[1] = e->refpl[l][2];
                ctx->cur_y = e->srcp[0];
                ctx->cur_c[0] = e->srcp[1];
                ctx->cur_c[1] = e->srcp[2];
                ctx->cstride = e->stride[1];
                umh_radii(e, ctx->umh, &ctx->umh_et);
            }
            e->st.stage_ns[AEG_STAGE_INTER_PRED] += aeg_now_ns() - t0;
        }
        me_seed_frame(e);
        lr_seed_frame(e);
        pz_setup(e);
        bfp_keep(e, 1);
        /* (what a VBV re-code of this field starts from again) */
        fstart = out_len;
        bfp_on0 = e->bfp_on;
        srows0 = e->slot_rows[e->role_cur];
        memcpy(stat0, e->stat_type, sizeof(stat0));
frecode:
        if (e->p.aud) {
            uint8_t aud[2];
            aud[0] = (uint8_t)(((e->cur_is_b ? 2 : e->cur_is_p ? 1 : 0) << 5) | 0x10);
            out_len += aeg_nal_write(e->out + out_len, e->out_cap - out_len, 0, 9,
                                     aud, 1, 1);
        }
        if (fidr) {
            memcpy(e->out + out_len, e->hdr, e->hdr_len);
            out_len += e->hdr_len;
        }
        {
            /* picture timing: pic_struct 1 (top field) or 2 (bottom) */
            aeg_sei s;
            uint8_t sei[1200];
            size_t sn;

            memset(&s, 0, sizeof(s));
            if (e->hrd_on)
                hrd_timing(e, &s, fidr, poc, f);
            s.pt = 1;
            s.pt_struct = 1 + (e->fld_par ? 2 : 1);
            if (e->p.info_sei && e->frame_count == 0 && f == 0)
                s.info = e->info;
            sn = aeg_write_sei(sei, sizeof(sei), &s);
            out_len += aeg_nal_write(e->out + out_len, e->out_cap - out_len, 0, 6,
                                     sei, sn, 0);
        }
        /* a reference field coded as one slice is post-processed row by row
         * behind its decision pass, as a frame is (rowpost_row) */
        e->rp_on = is_ref && e->n_slices == 1 && e->slot_mbs[e->role_cur] &&
                   slice_thread_count(e) == 1 && !getenv("AEG_NORP");
        e->rp_fon = 0;
        e->rp_lfon = is_ref && e->slot_lowf[e->role_cur][e->fld_par].mem != NULL;
        e->rp_dbk = e->rp_band = e->rp_hpel = 0;
        e->rp_qp = e->cur_qp;
        e->rp_mbs = e->slot_mbs[e->role_cur] ?
                    e->slot_mbs[e->role_cur] + e->fld_par * e->mb_count : NULL;
        if (f == 0)
            e->slot_rows[e->role_cur] = 0;
        rc = code_slices(e, fslice, fidr, &out_len);
        bfp_keep(e, 0);
        e->bfp_on = 0;
        if (rc != AEG_OK) {
            wp_finish(e);
            field_leave(e);
            for (f = 0; f < AEG_MAX_REFS; f++)
                e->mcr[f].stride = mstride[f];
            return rc;
        }
        /* VBV emergency, as for a frame (frame_body): a field the buffer
         * cannot take is coded again, coarser by the QP its overshoot asks
         * for.  The first field leaves the buffer at the frame's time, the
         * second a tick later with what the first left plus half a picture
         * period's delivery (vbv_update).  Up to three attempts each. */
        if (e->vbv_active && fretry < 3 && e->cur_qp < e->p.qp_max) {
            double avail = vbv_available(e);
            double bits = (double)(out_len - fstart) * 8.0;
            if (f == 1) {
                avail -= (double)e->vbv_f1;
                if (avail < 0.0)
                    avail = 0.0;
                avail += 0.5 * e->vbv_rate_bits;
                if (avail > e->vbv_buf_bits)
                    avail = e->vbv_buf_bits;
            }
            if (bits > avail) {
                int dq = (int)ceil(6.0 * log2(bits / (avail > 1.0 ? avail : 1.0))) + 1;
                e->cur_qp += dq;
                if (e->cur_qp > e->p.qp_max)
                    e->cur_qp = e->p.qp_max;
                fretry++;
                e->vbv_recodes++;
                out_len = fstart;
                e->bfp_on = bfp_on0;
                e->slot_rows[e->role_cur] = srows0;
                memcpy(e->stat_type, stat0, sizeof(stat0));
                goto frecode;
            }
        }
        wp_finish(e);
        e->top_row = 0;
        {
            int64_t t0 = aeg_now_ns();
            if (e->rp_on) {
                int y;
                if (e->p.deblock)
                    for (y = e->rp_dbk; y < e->mb_h; y++)
                        rowpost_deblock_row(e, e->rp_mbs, y);
            } else if (e->p.deblock) {
            aeg_deblock_frame(&e->pic, e->mbs, e->mb_w, e->p.deblock_alpha,
                                  e->p.deblock_beta, 1);
            }
            if (is_ref) {
                if (!e->rp_on)
                    e->rp_band = e->rp_hpel = 0;
                rowpost_bands(e, e->mb_h);
            } else {
                int i;
                for (i = 0; i < 3; i++)
                    pad_edges(e->recp[i], e->stride[i], i ? e->cw : e->w,
                              i ? e->ch : e->h);
            }
            e->st.stage_ns[AEG_STAGE_DEBLOCK] += aeg_now_ns() - t0;
        }
        if (is_ref && e->slot_mbs[e->role_cur] && !e->rp_on)
            /* (by parity: direct mode finds a co-located field by its;
             * rowpost_finalize has stored them already when rp_on) */
            memcpy(e->slot_mbs[e->role_cur] + e->fld_par * e->mb_count, e->mbs,
                   (size_t)e->mb_count * sizeof(e->mbs[0]));
        e->rp_on = 0;
        e->rp_lfon = 0;
        pz_finish(e);
        if (is_ref)
            field_mark(e, fidr);
        field_leave(e);
        if (f == 0 && e->vbv_active)
            e->vbv_f1 = (int)(out_len * 8);   /* (vbv_update) */
    }
    for (f = 0; f < AEG_MAX_REFS; f++)
        e->mcr[f].stride = mstride[f];

    /* distortion statistics, of the frame */
    pic_quality(e);
    if (!e->ft_worker_copy)
        qual_account(e, e->pic_sse, e->pic_ssim);
    if (pkt) {
        int64_t t = 0;
        int i;
        pkt->data = e->out;
        pkt->size = out_len;
        pkt->pts = pts;
        pkt->keyframe = is_idr;
        pkt->qp = e->cur_qp;
        pkt->type = is_idr || slice == AEG_SLICE_I ? 0
                  : fb ? (e->cur_bref ? 3 : 2) : 1;
        /* (the offsets of both fields: the whole array, field by field) */
        for (i = 0; i < e->mb_count; i++)
            t += mb_qp_of(e, i % e->mb_w, i / e->mb_w);
        pkt->qp_avg = (double)t / e->mb_count;
        pkt->qp_base = e->cur_qp + e->crf_frac;
    }
    e->st.bytes += (int64_t)out_len;
    e->st.frames++;
    e->st.enc_ns += aeg_now_ns() - t_all;
    e->frame_count++;
    if (is_ref && e->slot_mbs[e->role_cur]) {
        memset(e->slot_ref_poc[e->role_cur], 0, sizeof(e->slot_ref_poc[0]));
        e->slot_col_ok[e->role_cur] = 1;
    }
    e->slot_lfok[e->role_cur] = is_ref && e->slot_lowf[e->role_cur][0].mem;
    if (is_ref && (e->p.field_coding == 2 || e->p.field_coding == 4) &&
        e->slot_low[e->role_cur].mem)
        /* (the frame pictures' lowres pre-search reads it) */
        aeg_lowres_build_rows(&e->slot_low[e->role_cur], e->recp[0],
                              e->stride[0], e->w, e->h, 0, 8 * e->mb_h - 1);
    /* field_coding 4: the field macroblocks of a later MBAFF frame read it
     * through its field copy */
    if (e->p.field_coding == 4) {
        e->slot_fok[e->role_cur] = 0;
        if (is_ref)
            mbaff_field_shadow(e, e->role_cur);
    }
    /* the buffer becomes a reference frame (field_mark stored it), or is
     * reused by the next picture */
    e->role_last = e->role_cur;
    e->role_cur = -1;
    e->role_cur = pic_free(e);
    bind_pictures(e);
    e->pic.plane[0] = e->recp[0];
    e->pic.plane[1] = e->recp[1];
    e->pic.plane[2] = e->recp[2];
    if (is_idr)
        e->idr_pic_id = (e->idr_pic_id + 1) & 0xffff;
    (void)poc; (void)fp;
    if (getenv("AEG_RCDBG"))
        fprintf(stderr, "PIC %c poc %d qp %d bits %d (fields)\n",
                is_idr ? 'I' : (fb ? 'B' : (fp ? 'P' : 'i')), e->cur_poc,
                e->cur_qp, (int)(out_len * 8));
    vbv_update(e, (int)(out_len * 8));
    rc_update(e, (int)(out_len * 8));
    aeg_dispatch_update(&e->d);
    return AEG_OK;
}

/* The NAL HRD's timing SEI fields of the access unit about to be written
 * (picture timing on every one, a buffering period on an IDR picture's
 * first): a frame lasts two clock ticks (time_scale = 2 * fps_num), a field
 * picture one -- `tick` is 1 for the second field of a frame, which leaves
 * the CPB and is output a tick after the first.  The removal delays count
 * from the last buffering period's picture, the output delay is the same
 * for both fields of a frame. */
static void hrd_timing(struct aeg_encoder *e, aeg_sei *s, int is_idr, int poc,
                       int tick)
{
    const int c = e->frame_count;
    const int reorder = e->p.bframes > 0 ? e->p.bframes : 0;
    int dpb = 2 * (poc - c + reorder);

    if (is_idr && !tick) {
        /* the picture leaves the CPB when the model's level before it has
         * arrived; delay + offset spans the whole buffer */
        double whole = e->hrd_cpb / e->hrd_rate * 90000.0;
        double d0 = vbv_available(e) / e->hrd_rate * 90000.0;
        if (d0 > whole) d0 = whole;
        if (d0 < 1.0) d0 = 1.0;
        s->bp = 1;
        s->bp_delay = (uint32_t)d0;
        s->bp_offset = (uint32_t)(whole - (double)s->bp_delay);
        /* (a picture coded again -- the VBV emergency -- comes here twice:
         * its delay still counts from the buffering period before it) */
        if (e->hrd_bp_c != c) {
            e->hrd_bp_prev = e->hrd_bp_c;
            e->hrd_bp_c = c;
        }
        s->cpb_delay = c ? (uint32_t)(2 * (c - e->hrd_bp_prev)) : 0;
    } else {
        s->cpb_delay = (uint32_t)(2 * (c - e->hrd_bp_c) + tick);
    }
    s->pt = s->pt_delays = 1;
    s->dpb_delay = (uint32_t)(dpb > 0 ? dpb : 0);
}

static int frame_body(struct aeg_encoder *e, int slice, int is_idr, int poc,
                      int64_t pts, aeg_packet *pkt, int64_t t_all)
{
    int is_p = e->cur_is_p;
    size_t out_len;
    int64_t dbg_type0[AEG_MB_TYPE_COUNT];
    int vbv_retry = 0;

    (void)poc;
    memcpy(dbg_type0, e->stat_type, sizeof(dbg_type0));
recode:
    if (ensure_buf(&e->out, &e->out_cap,
                   (size_t)e->mb_count * 400 * 3 + 65536) != AEG_OK)
        return AEG_ENOMEM;
    /* weighted prediction compares the picture with the whole of its list 0
     * reference.  A frame-thread picture whose reference is still being
     * coded starts unweighted instead, and checks once it is whole
     * (ft_wp_check) -- unless the last analysis did use weights. */
    e->ft_spec = 0;
    e->ft_abort = 0;
    if (e->fsync) {
        int l;
        pthread_mutex_lock(&e->fsync->mu);
        for (l = 0; l < AEG_MAX_REFS; l++) {
            int sl = e->ref_slot[l];
            e->ft_gslot[l] = ref_valid(e, l) && sl >= 0 ? sl : -1;
            e->ft_gen[l] = sl >= 0 ? e->fsync->gen[sl] : 0;
            if (e->ft_gslot[l] >= 0)
                e->fsync->readers[sl]++;
        }
        e->ft_reg = 1;
        pthread_mutex_unlock(&e->fsync->mu);
    }
    if (e->p.weightp && e->cur_is_p && e->has_ref && !e->mbaff) {
        int pred = -1;
        if (e->fsync && !getenv("AEG_NOWPSPEC") &&
            __atomic_load_n(&e->fsync->rows[e->ref_slot[0]], __ATOMIC_ACQUIRE) <
                e->mb_h) {
            /* the analysis on the reference's source predicts the one on
             * its reconstruction well (cock 27 of 28 P pictures, short 25
             * of 32); without the source, what the last analysis found */
            pred = wp_predict(e);
            if (pred < 0)
                pred = __atomic_load_n(&e->fsync->wp_hist, __ATOMIC_RELAXED);
        }
        if (pred == 0) {
            e->ft_spec = 1;
            memset(&e->wp, 0, sizeof(e->wp));
        } else {
            if (ref_wait(e, e->ref_slot[0], e->mb_h))
                return FT_RESTART;
            wp_analyse(e);
            if (e->fsync)
                __atomic_store_n(&e->fsync->wp_hist,
                                 e->wp.lflag[0] || e->wp.cflag[0], __ATOMIC_RELAXED);
        }
    } else {
        wp_analyse(e);
    }
    if ((is_p || e->cur_is_b) && e->p.subme > 0) {
        /* Build the fractional-pel planes of the reference picture once for the
         * whole picture (AEG_STAGE_INTER_PRED).  Every subpel candidate of every
         * macroblock then reads samples from here instead of re-running the
         * 6-tap chain per block: the same samples, computed ~30x less often.
         * A B picture builds them for both lists. */
        int64_t t0 = aeg_now_ns();
        int list;

        /* a frame-thread worker's reference may still be being coded: its
         * rows are only there up to fsync->rows, and filtering the whole
         * plane would wait for all of them.  The planes built as the
         * reference was coded are the same samples, read row by row. */
        int dev_ip = e->d.use_gpu[AEG_STAGE_INTER_PRED] &&
                     e->d.ops.interp_plane &&
                     e->d.ops.backend != AEG_BACKEND_CPU && !e->fsync;

        if (dev_ip) {
            /* the planes are what the GPU stage is for: one launch per plane,
             * every output sample independent, and the kernel is checked
             * against aeg_hpel_plane_fill() by tests/kernel_check.c */
            for (list = 0; list < nrefpl(e); list++) {
                aeg_mc_ctx *ctx = &e->mcr[list];
                const pixel *ref = e->refpl[list][0];
                int i, sl = e->ref_slot[list];

                if (!ctx->buf) {
                    /* references past the second have no planes of their
                     * own: their buffer's, the same samples */
                    ctx->base[0] = ref;
                    for (i = 1; i < 4; i++)
                        ctx->base[i] = e->slot_mc[sl].base[i];
                    ctx->ready = 1;
                    continue;
                }
                aeg_mc_ctx_own(ctx, e->h);
                aeg_mc_ctx_setref(ctx, ref);
                for (i = 1; i < 4; i++)
                    e->d.ops.interp_plane(e->d.ops.priv, ref, e->stride[0],
                                          e->w, e->h, i,
                                          (pixel *)ctx->base[i], ctx->stride);
                ctx->ready = 1;
            }
        } else {
            /* the reference's own planes (built as it was coded), unless the
             * list predicts from a weighted copy */
            for (list = 0; list < nrefpl(e); list++) {
                aeg_mc_ctx *ctx = &e->mcr[list];
                const pixel *ref = e->refpl[list][0];
                int s = e->ref_slot[list], i;

                if ((s >= 0 && ref == e->pic_plane[s][0] && e->slot_mc[s].buf) ||
                    !ctx->buf) {
                    ctx->base[0] = ref;
                    for (i = 1; i < 4; i++)
                        ctx->base[i] = e->slot_mc[s].base[i];
                    ctx->ready = 1;
                } else {
                    aeg_mc_ctx_own(ctx, e->h);
                    aeg_mc_ctx_refresh(ctx, ref, e->w, e->h);
                }
            }
        }
        {
            /* chroma in the quarter-pel cost (x264's chroma-me) */
            int on = chroma_me_on(e), l;
            for (l = 0; l < AEG_MAX_REFS; l++) {
                aeg_mc_ctx *ctx = &e->mcr[l];
                pixel **rp = e->refpl[l];
                ctx->chroma = on && rp[1] != NULL;
                ctx->cref[0] = rp[1];
                ctx->cref[1] = rp[2];
                ctx->cur_y = e->srcp[0];
                ctx->cur_c[0] = e->srcp[1];
                ctx->cur_c[1] = e->srcp[2];
                ctx->cstride = e->stride[1];
                umh_radii(e, ctx->umh, &ctx->umh_et);
            }
        }
        stage_rec(e, AEG_STAGE_INTER_PRED, dev_ip, aeg_now_ns() - t0);
        e->st.stage_ns[AEG_STAGE_INTER_PRED] += aeg_now_ns() - t0;
    }
    mbaff_field_refs(e);
    me_seed_frame(e);
    lr_seed_frame(e);
    pz_setup(e);
    if (getenv("AEG_MBSTALE"))
        /* debug: the decisions left by an earlier picture must not matter
         * (frame threads hand each picture a worker's own array, holding
         * whatever that worker coded last); the output has to stay the same */
        memset(e->mbs, atoi(getenv("AEG_MBSTALE")), (size_t)e->mb_count * sizeof(aeg_mb));

    /* frame_num counts the pictures since the last IDR (H.264 8.2.1): an IDR
     * always carries 0, otherwise the decoder invents "non-existing" frames to
     * fill the gap. */
    if (is_idr)
        e->frame_num = 0;

    /* One slice per picture is the default and keeps the bitstream single-NAL;
     * `--slices N` (or `-t N`, which uses one slice per thread) cuts the picture
     * into N row ranges, each an independently decodable NAL. */
    out_len = 0;
    if (e->p.aud) {
        /* access_unit_delimiter: primary_pic_type 0 = I, 1 = I/P, 2 = I/P/B */
        uint8_t aud[2];
        aud[0] = (uint8_t)(((e->cur_is_b ? 2 : is_p ? 1 : 0) << 5) | 0x10);
        out_len += aeg_nal_write(e->out + out_len, e->out_cap - out_len, 0, 9,
                                 aud, 1, 1);
    }
    if (is_idr) {
        memcpy(e->out + out_len, e->hdr, e->hdr_len);
        out_len += e->hdr_len;
    }
    /* SEI, before the slices: the HRD's buffering period (IDR) and picture
     * timing (every picture; also for the field order of an interlaced
     * stream), and the info text on the first picture.  A frame lasts two
     * clock ticks (time_scale = 2 * fps_num). */
    if (e->hrd_on || e->p.interlaced ||
        (e->p.info_sei && e->frame_count == 0)) {
        aeg_sei s;
        uint8_t sei[1200];
        size_t sn;

        memset(&s, 0, sizeof(s));
        if (e->hrd_on)
            hrd_timing(e, &s, is_idr, poc, 0);
        if (e->p.interlaced) {
            /* pic_struct 3 = top field, bottom field; 4 = the reverse */
            s.pt = 1;
            s.pt_struct = 1 + (e->p.interlaced == 1 ? 3 : 4);
        }
        if (e->p.info_sei && e->frame_count == 0)
            s.info = e->info;
        sn = aeg_write_sei(sei, sizeof(sei), &s);
        out_len += aeg_nal_write(e->out + out_len, e->out_cap - out_len, 0, 6,
                                 sei, sn, 0);
    }
    /* a reference picture coded as one slice is post-processed row by row
     * behind its decision pass (rowpost_row) */
    e->rp_on = e->cur_is_ref && e->n_slices == 1 && e->slot_mbs[e->role_cur] &&
               slice_thread_count(e) == 1 && !getenv("AEG_NORP");
    e->rp_dbk = e->rp_band = e->rp_hpel = 0;
    e->rp_fon = e->mbaff && e->cur_is_ref && e->slot_fmem[e->role_cur];
    e->rp_fsh = e->rp_fhp = 0;
    e->rp_qp = e->cur_qp;
    e->rp_mbs = e->slot_mbs[e->role_cur];
    e->slot_rows[e->role_cur] = 0;
    e->pic.plane[0] = e->recp[0];
    e->pic.plane[1] = e->recp[1];
    e->pic.plane[2] = e->recp[2];
    {
        int rc = code_slices(e, slice, is_idr, &out_len);
        if (rc != AEG_OK)
            return rc;
    }
    e->top_row = 0;
    if (e->fsync && (ft_wp_check(e, 1) || e->ft_abort))
        return FT_RESTART;

    /* VBV emergency: a picture the buffer cannot take (the size predictor
     * cannot foresee everything -- a static picture coded much finer than its
     * reference costs far more than its difference from it suggests) is coded
     * again, coarser by the QP its overshoot asks for (bits ~ 2^(-QP/6)),
     * before anything else sees it.  Nothing the first attempt did outlives
     * it: the macroblocks, the reconstruction and its row post-processing are
     * written afresh, and the weighted reference is rebuilt.  Up to three
     * attempts; at the QP ceiling the model reports the underflow. */
    if (e->vbv_active && !e->fsync && vbv_retry < 3 &&
        e->cur_qp < e->p.qp_max) {
        double avail = vbv_available(e), bits = (double)out_len * 8.0;

        if (bits > avail) {
            int dq = (int)ceil(6.0 * log2(bits / (avail > 1.0 ? avail : 1.0))) + 1;

            e->cur_qp += dq;
            if (e->cur_qp > e->p.qp_max) e->cur_qp = e->p.qp_max;
            vbv_retry++;
            e->vbv_recodes++;
            memcpy(e->stat_type, dbg_type0, sizeof(dbg_type0));
            wp_finish(e);
            goto recode;
        }
    }

    /* deblock the reconstructed picture (what rowpost_row has not done yet)
     * and, for a reference, finish its border, half-resolution copy and
     * half-pel planes */
    {
        int64_t t0 = aeg_now_ns();
        e->pic.plane[0] = e->recp[0];
        e->pic.plane[1] = e->recp[1];
        e->pic.plane[2] = e->recp[2];
        if (e->rp_on) {
            int y;
            if (e->p.deblock && e->mbaff)
                for (y = e->rp_dbk; y < e->mb_h; y += 2)
                    aeg_deblock_pair_row(&e->pic, e->rp_mbs, e->mb_w, y >> 1,
                                         e->p.deblock_alpha,
                                         e->p.deblock_beta);
            else if (e->p.deblock)
                for (y = e->rp_dbk; y < e->mb_h; y++)
                    rowpost_deblock_row(e, e->rp_mbs, y);
        } else if (e->p.deblock) {
            aeg_deblock_frame(&e->pic, e->mbs, e->mb_w, e->p.deblock_alpha,
                              e->p.deblock_beta, 1);
        }
        if (e->cur_is_ref) {
            if (!e->rp_on)
                e->rp_band = e->rp_hpel = e->rp_fsh = e->rp_fhp = 0;
            rowpost_bands(e, e->mb_h);
        } else {
            int i;
            for (i = 0; i < 3; i++)
                pad_edges(e->recp[i], e->stride[i], i ? e->cw : e->w,
                          i ? e->ch : e->h);
        }
        e->st.stage_ns[AEG_STAGE_DEBLOCK] += aeg_now_ns() - t0;
    }

    /* distortion statistics */
    pic_quality(e);
    if (!e->ft_worker_copy)
        qual_account(e, e->pic_sse, e->pic_ssim);

    if (pkt) {
        pkt->data = e->out;
        pkt->size = out_len;
        pkt->pts = pts;
        pkt->keyframe = is_idr;
        pkt->qp = e->cur_qp;
        pkt->type = is_idr || slice == AEG_SLICE_I ? 0
                  : e->cur_is_b ? (e->cur_bref ? 3 : 2) : 1;
        {
            int64_t t = 0;
            int i;
            for (i = 0; i < e->mb_count; i++)
                t += mb_qp_of(e, i % e->mb_w, i / e->mb_w);
            pkt->qp_avg = (double)t / e->mb_count;
            pkt->qp_base = e->cur_qp + e->crf_frac;
        }
    }

    e->st.bytes += (int64_t)out_len;
    e->st.frames++;
    e->st.enc_ns += aeg_now_ns() - t_all;
    e->frame_count++;
    /* MBAFF: the field macroblocks' copy of a reference */
    e->slot_lfok[e->role_cur] = 0;
    if (e->mbaff) {
        e->slot_fok[e->role_cur] = 0;
        if (e->rp_fon && e->rp_fhp == e->mb_h / 2)
            e->slot_fok[e->role_cur] = 1;    /* (rowpost_fshadow built it) */
        else if (e->cur_is_ref)
            mbaff_field_shadow(e, e->role_cur);
    }
    /* frame_num counts reference pictures only: a non-reference picture carries
     * PrevRefFrameNum + 1 (7.4.3), so all the B pictures of a group repeat the
     * value the anchor after them already had. */
    /* (frame_num is advanced by picture_rotate, for reference pictures.)
     * A reference picture may be some later B picture's list1[0], whose
     * direct mode reads this picture's decisions: keep them with its buffer. */
    if (e->cur_is_ref && e->slot_mbs[e->role_cur]) {
        /* rowpost_finalize has already stored them (as the filter reads
         * them); AEG_RPCHK=1 checks that against the bitstream pass */
        if (!e->rp_on)
            memcpy(e->slot_mbs[e->role_cur], e->mbs,
                   (size_t)e->mb_count * sizeof(e->mbs[0]));
        else if (getenv("AEG_RPCHK")) {
            int i, k;
            for (i = 0; i < e->mb_count; i++) {
                const aeg_mb *a = &e->mbs[i], *b = &e->rp_mbs[i];
                int bad = a->type != b->type || a->qp != b->qp ||
                          a->t8 != b->t8 || a->cbp != b->cbp ||
                          a->skip != b->skip ||
                          memcmp(a->mv, b->mv, sizeof(a->mv)) ||
                          memcmp(a->mv1, b->mv1, sizeof(a->mv1)) ||
                          memcmp(a->ref_idx, b->ref_idx, sizeof(a->ref_idx)) ||
                          memcmp(a->blist, b->blist, sizeof(a->blist));
                for (k = 0; k < AEG_BLK_COUNT; k++)
                    bad |= a->nz[k] != b->nz[k];
                for (k = 0; k < 4; k++)
                    bad |= a->nz8[k] != b->nz8[k];
                if (bad) {
                    fprintf(stderr, "RPCHK poc %d mb %d: type %d/%d qp %d/%d "
                            "t8 %d/%d skip %d/%d\n", e->cur_poc, i, a->type,
                            b->type, a->qp, b->qp, a->t8, b->t8, a->skip,
                            b->skip);
                    abort();
                }
            }
        }
        memcpy(e->slot_ref_poc[e->role_cur], e->ref_poc, sizeof(e->ref_poc));
        e->slot_col_ok[e->role_cur] = 1;
    }
    /* The reconstruction becomes a reference picture and the buffer it occupied
     * is handed to the next picture (or, for a B picture, kept: a B is not a
     * reference, so its buffer is simply reused). */
    e->rp_on = 0;
    pz_finish(e);
    wp_finish(e);
    picture_rotate(e, e->cur_is_ref, is_idr);
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
        fprintf(stderr, "PIC %c poc %d qp %d bits %d skip %lld intra %lld ms %.1f%s\n",
                is_idr ? 'I' : (e->cur_is_b ? (e->cur_bref ? 'b' : 'B') : (e->cur_is_p ? 'P' : 'i')),
                e->cur_poc, e->cur_qp, (int)(out_len * 8), (long long)sk,
                (long long)in, (aeg_now_ns() - t_all) / 1e6,
                e->coef_dev ? " coef-on-device" : "");
    }
    if (getenv("AEG_RCDBG"))
        fprintf(stderr, "RC F%d poc %d qp %d (model %.2f) bits %d total %.0f "
                "wanted %.0f rf %.4g vbv %.0f exp %.0f\n",
                e->frame_count, e->cur_poc, e->cur_qp, e->abr_qm,
                (int)(out_len * 8), e->abr_total,
                e->abr_frames * e->abr_bpf, e->abr_wanted / e->abr_cplxr,
                e->vbv_full, plan_has(e, e->cur_poc, cur_cls(e)) ? e->plan_bits[e->cur_poc] : -1.0);
    vbv_update(e, (int)(out_len * 8));
    rc_update(e, (int)(out_len * 8));
    /* the placement trial of the coefficient stages measures whole pictures
     * (see aeg_dispatch_coef_pick); a forced batch path is not a trial */
    if (e->coef_force < 0 && e->rd_force < 0)
        aeg_dispatch_coef_done(&e->d, e->coef_kind, e->coef_dev,
                               aeg_now_ns() - t_all, e->mb_count);
    aeg_dispatch_update(&e->d);
    return AEG_OK;
}

/* MBAFF: the field macroblocks' copy of reference buffer s (slot_fpl,
 * slot_fmc), once the picture in it is complete -- its samples with every
 * field's rows replicated above and below the field, and the half-sample
 * planes filtered within each field (frame macroblocks read the buffer
 * itself: frame borders, filtered across the frame's rows). */
static void mbaff_field_shadow(struct aeg_encoder *e, int s)
{
    aeg_mc_ctx *mc = &e->slot_fmc[s];
    int i, par, k;

    if (!e->slot_fmem[s]) {
        int st[3] = { e->stride[0], e->stride[1], e->stride[2] };
        if (enc_alloc_picture(e, &e->slot_fmem[s], e->slot_fpl[s]) != AEG_OK)
            return;
        e->stride[0] = st[0]; e->stride[1] = st[1]; e->stride[2] = st[2];
        aeg_mc_ctx_init_pad(mc, e->w, e->h, e->stride[0], e->vpad);
    }
    for (i = 0; i < 3; i++) {
        int w = i ? e->cw : e->w, h = i ? e->ch : e->h, st = e->stride[i];
        memcpy(e->slot_fpl[s][i] - AEG_PAD, e->pic_plane[s][i] - AEG_PAD,
               ((size_t)(h - 1) * st + w + 2 * AEG_PAD) * sizeof(pixel));
        for (par = 0; par < 2; par++)
            pad_edges(e->slot_fpl[s][i] + par * st, 2 * st, w, h / 2);
    }
    if (mc->buf) {
        const pixel *y = e->slot_fpl[s][0];
        int st = mc->stride, ys = e->stride[0];
        for (k = 1; k < 4; k++)
            for (par = 0; par < 2; par++)
                aeg_hpel_plane_rows((pixel *)mc->base[k] + par * st, 2 * st,
                                    y + par * ys, 2 * ys, e->w, e->h / 2, k,
                                    -AEG_PAD, e->h / 2 + AEG_PAD - 1);
    }
    e->slot_fok[s] = 1;
}

/* Field reference entry k for field macroblocks of parity cp: field `par`
 * of frame list entry f (whose slot has its field copy). */
static void fref_fill(struct aeg_encoder *e, int cp, int k, int f, int par)
{
    const int sl = e->ref_slot[f];
    const aeg_mc_ctx *fm = &e->slot_fmc[sl];
    aeg_mc_ctx *c = &e->fref_mc[cp][k];
    int i;

    for (i = 0; i < 3; i++)
        e->fref_pl[cp][k][i] = e->slot_fpl[sl][i] + par * e->stride[i];
    *c = e->mcr[f];
    c->base[0] = e->fref_pl[cp][k][0];
    for (i = 1; i < 4; i++)
        c->base[i] = fm->buf ? fm->base[i] + par * fm->stride : NULL;
    c->stride = 2 * e->stride[0];
    c->ready = e->mcr[f].ready && fm->buf != NULL;
    c->chroma = 0;
    c->cstride = 2 * e->stride[1];
    e->fref_poc[cp][k] = e->ref_poc[f];
    /* the other parity's chroma rows sit a quarter line off */
    e->fref_cofs[cp][k] = 2 * (cp - par);
}

static int fref_frame_ok(const struct aeg_encoder *e, int f)
{
    const int sl = e->ref_slot[f];
    return sl >= 0 && e->slot_fok[sl] && e->refpl[f][0] == e->pic_plane[sl][0];
}

/* MBAFF: the field references of the current picture (fref_*), for field
 * macroblocks of both parities.  A P picture's from its list0 frames (2n
 * entries, 8.4.2.1; a frame without its field copy ends the list there); a
 * B picture's are list 0's and list 1's fields of the macroblock's own
 * parity (fref_b), or none. */
static void mbaff_field_refs(struct aeg_encoder *e)
{
    int n = 2 * e->cur_l0_active, cp, k;

    e->fref_n = 0;
    e->fref_b = 0;
    if (!e->mbaff || !e->has_ref)
        return;
    if (e->cur_is_b) {
        if (!e->has_ref2 || !fref_frame_ok(e, 0) || !fref_frame_ok(e, 1))
            return;
        /* (AEG_BSAMEPAR: the macroblock's parity only, as before) */
        const int nk = getenv("AEG_BSAMEPAR") ? 2 : 4;
        for (cp = 0; cp < 2; cp++)
            for (k = 0; k < nk; k++)
                fref_fill(e, cp, k, k & 1, k >= 2 ? !cp : cp);
        e->fref_n = nk;
        e->fref_b = 1;
        return;
    }
    if (!e->cur_is_p)
        return;
    if (n > AEG_MAX_REFS)
        n = AEG_MAX_REFS;
    for (k = 0; k < n; k++)
        if (!fref_frame_ok(e, k >> 1))
            break;
    n = k & ~1;
    for (cp = 0; cp < 2; cp++)
        for (k = 0; k < n; k++)
            fref_fill(e, cp, k, k >> 1, (k & 1) ? !cp : cp);
    e->fref_n = n;
}

/* p.field_coding 2: a reference frame coded one way is read the other way
 * by the next picture.  The samples are the same; the borders (a frame's
 * edge rows replicated, or each field's) and the half-sample planes (6-tap
 * across the frame's rows or within each field) are not, so they are
 * redone for the structure about to read them. */
static void slot_restructure(struct aeg_encoder *e, int s, int fld)
{
    aeg_mc_ctx *mc = &e->slot_mc[s];
    int i, par, k;

    if (e->slot_fst[s] == fld)
        return;
    for (i = 0; i < 3; i++) {
        int w = i ? e->cw : e->w, h = i ? e->ch : e->h, st = e->stride[i];
        if (!fld)
            pad_edges(e->pic_plane[s][i], st, w, h);
        else
            for (par = 0; par < 2; par++)
                pad_edges(e->pic_plane[s][i] + par * st, 2 * st, w, h / 2);
    }
    if (mc->buf) {
        const pixel *y = e->pic_plane[s][0];
        int st = mc->stride, ys = e->stride[0];
        for (k = 1; k < 4; k++) {
            if (!fld)
                aeg_hpel_plane_rows((pixel *)mc->base[k], st, y, ys, e->w,
                                    e->h, k, -AEG_PAD, e->h + AEG_PAD - 1);
            else
                for (par = 0; par < 2; par++)
                    aeg_hpel_plane_rows((pixel *)mc->base[k] + par * st,
                                        2 * st, y + par * ys, 2 * ys, e->w,
                                        e->h / 2, k, -AEG_PAD,
                                        e->h / 2 + AEG_PAD - 1);
        }
    }
    e->slot_fst[s] = (uint8_t)fld;
}

/* The estimate behind the frame / field choice (fld_decide): a 16x16 block
 * of `cur` matched in `ref` by a small integer diamond search kept inside
 * the picture (w x h, so the borders, which differ between the structures,
 * are never read), started from the vector of the block to its left.  The
 * cost is the SAD at the vector found less the block's mean difference --
 * a brightness change `m` (the frame's weighted prediction's business; the
 * field pictures do without it) is taken out first. */
static int fld_block_cost(const pixel *cur, int cs, const pixel *ref, int rs,
                          int x, int y, int w, int h, int mv[2],
                          const int up[2], int wg, int wo)
{
    static const int dia[4][2] = { { 0, -1 }, { -1, 0 }, { 1, 0 }, { 0, 1 } };
    int bx = 0, by = 0, best = INT_MAX, it, i, j, k, sad = 0, step;
    const pixel *c = cur + y * cs + x;
    const int cand[3][2] = { { 0, 0 }, { mv[0], mv[1] }, { up[0], up[1] } };

#define FLD_IN(cx, cy) (x + (cx) >= 0 && x + (cx) <= w - 16 && \
                        y + (cy) >= 0 && y + (cy) <= h - 16)
    for (k = 0; k < 3; k++) {
        int v;
        if (!FLD_IN(cand[k][0], cand[k][1]))
            continue;
        v = aeg_sad(16, 16, c, cs, ref + (y + cand[k][1]) * rs + x + cand[k][0], rs);
        if (v < best)
            best = v, bx = cand[k][0], by = cand[k][1];
    }
    /* diamonds of steps 8, 4, 2, 1 (motion over several frames) */
    for (step = 8; step >= 1; step >>= 1)
        for (it = 0; it < 8; it++) {
            int nx = bx, ny = by;
            for (i = 0; i < 4; i++) {
                int cx = bx + step * dia[i][0], cy = by + step * dia[i][1], v;
                if (!FLD_IN(cx, cy))
                    continue;
                v = aeg_sad(16, 16, c, cs, ref + (y + cy) * rs + x + cx, rs);
                if (v < best)
                    best = v, nx = cx, ny = cy;
            }
            if (nx == bx && ny == by)
                break;
            bx = nx; by = ny;
        }
#undef FLD_IN
    mv[0] = bx; mv[1] = by;
    {
        const pixel *r = ref + (y + by) * rs + x + bx;
        /* SATD: what the transform makes of the residual (a field's rows
         * are further apart, its residual less correlated); no more than
         * the block costs against its own mean (intra, roughly) */
        pixel t[256];
        int sum = 0, intra;
        for (j = 0; j < 16; j++)
            for (i = 0; i < 16; i++) {
                int v = (((int)r[j * rs + i] * wg + 32) >> 6) + wo;
                t[16 * j + i] = (pixel)(v < 0 ? 0 : v > PIXEL_MAX ? PIXEL_MAX : v);
                sum += c[j * cs + i];
            }
        sad = aeg_satd_16x16(c, cs, t, 16);
        for (i = 0; i < 256; i++)
            t[i] = (pixel)((sum + 128) >> 8);
        intra = aeg_satd_16x16(c, cs, t, 16);
        if (intra < sad)
            sad = intra;
    }
    return sad;
}

/* The two structures' prediction cost from the newest anchor `ref` (the
 * picture a P frame predicts from): the frame's blocks against the frame;
 * each field's against both fields of the reference, the second field's
 * also against the first (half a frame period away -- what field pictures
 * gain on motion). */
static void fld_costs(const struct aeg_encoder *e, const pixel *ref,
                      int64_t *frame_cost, int64_t *field_cost)
{
    const pixel *src = e->srcp[0];
    const int st = e->stride[0], w = e->w, h = e->h, hf = e->h / 2;
    const int first = field_first_par(e);
    int x, y, f, mv[2], wg = 64, wo = 0;
    int64_t fc = 0, dc = 0;
    /* the vectors of the row above, per reference: upv[k * nb + x / 16] */
    const int nb = w / 16 + 1;
    int (*upv)[2] = (int (*)[2])calloc((size_t)3 * nb, sizeof(*upv));
    const char *fwev = getenv("AEG_FWPC");
    const int fwc = e->fwp && (fwev ? atoi(fwev) : e->p.field_coding == 4);

    if (!upv) {
        *frame_cost = *field_cost = 0;
        return;
    }

    if ((e->p.weightp && e->p.field_coding == 2) || e->fwp) {
        /* the frame picture's weighted prediction takes out a brightness
         * change -- x264's estimate: the gain the ratio of the standard
         * deviations, the offset what is left of the means.  The field
         * pictures' (fwp) take the same out of their prediction from the
         * reference's fields; an MBAFF frame (field_coding 4) has none */
        double sc = 0, sr = 0, qc = 0, qr = 0, n = (double)w * h, g;
        for (y = 0; y < h; y++)
            for (x = 0; x < w; x++) {
                double a = src[y * st + x], b = ref[y * st + x];
                sc += a; sr += b; qc += a * a; qr += b * b;
            }
        sc /= n; sr /= n;
        qc = qc / n - sc * sc;
        qr = qr / n - sr * sr;
        g = qr > 1.0 ? sqrt(qc / qr) : 1.0;
        if (g < 0.5) g = 0.5;
        if (g > 2.0) g = 2.0;
        wg = (int)(g * 64 + 0.5);
        wo = (int)lrint(sc - sr * wg / 64.0);
        if (wg == 64 && wo >= -1 && wo <= 1)
            wo = 0;
    }
    for (y = 0; y + 16 <= h; y += 16) {
        mv[0] = mv[1] = 0;
        for (x = 0; x + 16 <= w; x += 16) {
            int *u = upv[x / 16];
            const int fw = e->p.field_coding == 2;
            fc += fld_block_cost(src, st, ref, st, x, y, w, h, mv, u,
                                 fw ? wg : 64, fw ? wo : 0);
            u[0] = mv[0]; u[1] = mv[1];
        }
    }
    for (f = 0; f < 2; f++) {
        const int par = f ^ first;
        const pixel *cur = src + par * st;
        int mvs[3][2], k;
        memset(upv, 0, (size_t)3 * nb * sizeof(*upv));
        for (y = 0; y + 16 <= hf; y += 16) {
            memset(mvs, 0, sizeof(mvs));
            for (x = 0; x + 16 <= w; x += 16) {
                /* same parity, other parity, (second field) the first */
                const pixel *rf[3] = { ref + par * st, ref + (par ^ 1) * st,
                                       src + first * st };
                int b = INT_MAX;
                for (k = 0; k < 2 + f; k++) {
                    int *u = upv[k * nb + x / 16];
                    /* Against an MBAFF frame (4), which has no weights,
                     * the fields' estimate takes them out too: on two woven
                     * fades (BD-rate against --tff, p2) -17.29 -> -18.36 %
                     * moving, +44 -> +3 % still (where the frame pictures of
                     * 2 are best).  Against a weighted frame picture (2) it
                     * flips anchors the fields then code worse: -17.75 ->
                     * -16.78 % moving, +0 -> +1 / +8 % (p2 / p3) still.
                     * (Not the first field: the gain between the fields is
                     * half the frames'.)  AEG_FWPC=0/1 overrides. */
                    const int kw = fwc && k < 2;
                    int o = fld_block_cost(cur, 2 * st, rf[k], 2 * st, x, y,
                                           w, hf, mvs[k], u, kw ? wg : 64,
                                           kw ? wo : 0);
                    u[0] = mvs[k][0]; u[1] = mvs[k][1];
                    if (o < b)
                        b = o;
                }
                dc += b;
            }
        }
    }
    free(upv);
    *frame_cost = fc;
    *field_cost = dc;
}

/* Frame or fields for an anchor (p.field_coding 2), from its source: the
 * rows of a frame of moving interlaced material alternate between two
 * instants, so a row resembles the rows two above and below (its field)
 * more than its neighbours (the other field) -- the comb.  Per macroblock,
 * the vertical differences between neighbouring rows against those between
 * rows of the same field; the picture goes to fields when enough of its
 * macroblocks comb. */
static int fld_decide(const struct aeg_encoder *e)
{
    const pixel *p = e->srcp[0];
    const int st = e->stride[0], w = e->vw, h = e->vh & ~3;
    int mbx, mby, comb = 0, n = 0;
    static int thr = -1, frac = -1, mabs = -1;

    if (thr < 0) {
        const char *ev = getenv("AEG_FLDTHR");
        thr = ev ? atoi(ev) : 110;          /* % of the same-field difference */
        ev = getenv("AEG_FLDFRAC");
        frac = ev ? atoi(ev) : 5;           /* % of macroblocks */
        ev = getenv("AEG_FLDABS");
        mabs = ev ? atoi(ev) : 1;           /* and this much per sample */
    }
    if (getenv("AEG_FLDFORCE")) {
        /* debug: a pattern of 0 / 1 per display index, repeated */
        const char *f = getenv("AEG_FLDFORCE");
        size_t L = strlen(f);
        return L ? f[e->cur_poc % L] == '1' : 0;
    }
    if (!getenv("AEG_FLDCOMB")) {
        /* a P frame: by the prediction cost estimates, the fields to win
         * by `bias` percent (two pictures' headers, more vectors).  Against
         * an MBAFF frame (field_coding 4), which codes its moving pairs as
         * fields itself, by 16.  On the 80-field clips (BD-rate against
         * --tff, means of four clips) 16 gives -25.05 % at p2 and -22.70 %
         * at p3, 20 -25.23 / -22.27 %; at p2 12 gives -24.66 % and 30
         * -25.17 %.  Only megamind moves (p2 at 12/16/20/30: -25.63, -27.07,
         * -27.81, -27.55 %; p3 at 12/16/20: -24.31, -25.37, -23.63 %), a few
         * anchors flipping either way. */
        const int bias = ENV_INT("AEG_FLDBIAS", -1) >= 0 ? ENV_INT("AEG_FLDBIAS", -1)
                       : e->p.field_coding == 4 ? 16 : 12;
        int i, a = -1;
        for (i = 0; i < e->ndpb; i++)
            if (e->dpb[i].anchor && e->dpb[i].fm == 3 &&
                (a < 0 || e->dpb[i].poc > e->dpb[a].poc))
                a = i;
        if (a >= 0) {
            int64_t fc, dc;
            fld_costs(e, e->pic_plane[e->dpb[a].slot][0], &fc, &dc);
            if (getenv("AEG_FLDDBG"))
                fprintf(stderr, "FLD poc %d frame %lld field %lld (%.1f%%)\n",
                        e->cur_poc, (long long)fc, (long long)dc,
                        fc ? 100.0 * dc / fc : 0.0);
            return 100 * dc < (int64_t)(100 - bias) * fc;
        }
    }
    for (mby = 0; mby + 16 <= h; mby += 16)
        for (mbx = 0; mbx + 16 <= w; mbx += 16) {
            int64_t fr = 0, fi = 0;
            int x, y, sm[2] = { 0, 0 }, dc;
            /* (less the difference between the fields' means: a fade or a
             * flash changes the brightness between the fields, which field
             * pictures, without weighted prediction, code worse) */
            for (y = mby; y < mby + 16; y++)
                for (x = 0; x < 16; x++)
                    sm[y & 1] += p[y * st + mbx + x];
            dc = (sm[0] - sm[1]) / 128;          /* top minus bottom */
            for (y = mby; y < mby + 16 && y + 2 < h; y++) {
                const pixel *a = p + y * st + mbx;
                const int o = (y & 1) ? -dc : dc;
                for (x = 0; x < 16; x++) {
                    fr += abs((int)a[x] - (int)a[x + st] - o);
                    fi += abs((int)a[x] - (int)a[x + 2 * st]);
                }
            }
            n++;
            comb += 100 * fr > (int64_t)thr * fi + (int64_t)mabs * 16 * 16 * 100;
        }
    if (getenv("AEG_FLDDBG"))
        fprintf(stderr, "FLD poc %d comb %d/%d\n", e->cur_poc, comb, n);
    return 100 * comb > frac * n;
}

static int encode_frame(struct aeg_encoder *e, int slice, int is_idr, int poc,
                        int64_t pts, aeg_packet *pkt)
{
    int64_t t_all = aeg_now_ns();
    int rc, fld = e->p.field_coding == 1;

    if (e->p.field_coding == 2 || e->p.field_coding == 4) {
        int i;
        e->cur_poc = poc;
        fld = slice == AEG_SLICE_B ? e->anchor_fld : fld_decide(e);
        for (i = 0; i < e->ndpb; i++)
            slot_restructure(e, e->dpb[i].slot, fld);
        e->slot_fst[e->role_cur] = (uint8_t)fld;
        /* 4: a frame picture is an MBAFF frame */
        if (e->p.field_coding == 4)
            e->pic.mbaff = e->mbaff = !fld;
    }
    rc = frame_prologue(e, &slice, &is_idr, poc);
    if (rc != AEG_OK)
        return rc;
    if (slice != AEG_SLICE_B)
        e->anchor_fld = fld;
    if (fld)
        return field_frame_body(e, slice, is_idr, poc, pts, pkt, t_all);
    return frame_body(e, slice, is_idr, poc, pts, pkt, t_all);
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
    /* With the tree's window a group is queued la_depth pictures later than
     * it could be, and the scheduler can run up to a group behind the input
     * (the first picture alone, or a long group followed by a short one), so
     * the B pictures still waiting must survive another group of input. */
    /* (+2: a non-reference B picture coded by the frame thread is still being
     * read while the next two pictures come in) */
    return e->p.bframes + 5 + (e->la_depth ? e->la_depth + e->p.bframes + 1 : 0);
}

/* A submitted plane of vw x vh samples into a coded one of w x h: the last
 * column and row are repeated out to the macroblock edge (what the SPS crops
 * away; repeating them keeps the edge macroblocks as cheap as the content
 * next to them), then the usual padding. */
static void copy_plane_crop(pixel *dst, int dstride, const pixel *src,
                            int sstride, int vw, int vh, int w, int h)
{
    int y;
    if (vw == w && vh == h) {
        copy_plane_pad(dst, dstride, src, sstride, w, h);
        return;
    }
    for (y = 0; y < vh; y++) {
        memcpy(dst + y * dstride, src + (size_t)y * sstride, (size_t)vw * sizeof(pixel));
        if (w > vw)
            aeg_pixel_fill(dst + y * dstride + vw, dst[y * dstride + vw - 1],
                           (size_t)(w - vw));
    }
    for (y = vh; y < h; y++)
        memcpy(dst + y * dstride, dst + (vh - 1) * dstride, (size_t)w * sizeof(pixel));
    pad_edges(dst, dstride, w, h);
}

/* The caller's picture: three planes, each with its own row stride. */
typedef struct aeg_src {
    const pixel *plane[3];
    int stride[3];
} aeg_src;

static void copy_yuv(const struct aeg_encoder *e, pixel *p[3],
                     const aeg_src *in)
{
    copy_plane_crop(p[0], e->stride[0], in->plane[0], in->stride[0], e->vw,
                    e->vh, e->w, e->h);
    copy_plane_crop(p[1], e->stride[1], in->plane[1], in->stride[1], e->vcw,
                    e->vch, e->cw, e->ch);
    copy_plane_crop(p[2], e->stride[2], in->plane[2], in->stride[2], e->vcw,
                    e->vch, e->cw, e->ch);
}

/* SPS, PPS 0 and (with weighted prediction) PPS 1, as NAL units with start
 * codes: what every IDR picture begins with. */
static int write_param_sets(struct aeg_encoder *e)
{
    uint8_t sps[256], pps[64];
    size_t n = 0, cap = sizeof(e->hdr);
    size_t sl = aeg_write_sps(sps, sizeof(sps), &e->p, e->mb_w, e->mb_h,
                              e->frame_num_bits, e->level_idc);
    size_t pl = aeg_write_pps(pps, sizeof(pps), e->p.chroma_qp_offset,
                              e->p.transform_8x8, e->p.cabac, 0,
                              e->p.field_coding == 2 || e->p.field_coding == 4);
    /* worst case of a NAL: start code, header, and an emulation prevention
     * byte every two payload bytes */
    if (!sl || !pl || 3 * (sl + 2 * pl) / 2 + 3 * 6 > cap)
        return AEG_ERROR;
    e->level_out = sps[2];   /* profile_idc, constraint flags, level_idc */
    if (e->p.interlaced && (e->level_out < 21 || e->level_out > 41))
        aeg_log(AEG_LOG_WARNING, "aeg: level %d.%d does not allow an "
                "interlaced stream (only 2.1 to 4.1 do)", e->level_out / 10,
                e->level_out % 10);
    n += aeg_nal_write(e->hdr + n, cap - n, 3, 7, sps, sl, 1);
    n += aeg_nal_write(e->hdr + n, cap - n, 3, 8, pps, pl, 0);
    if (e->p.weightp && (e->p.field_coding == 0 || e->p.field_coding == 2 ||
                         e->fwp)) {
        /* PPS 1: the same with weighted_pred_flag, for the P pictures that
         * send weights (the others keep referring to PPS 0) */
        pl = aeg_write_pps(pps, sizeof(pps), e->p.chroma_qp_offset,
                           e->p.transform_8x8, e->p.cabac, 1,
                           e->p.field_coding == 2 || e->p.field_coding == 4);
        n += aeg_nal_write(e->hdr + n, cap - n, 3, 8, pps, pl, 0);
    }
    e->hdr_len = n;
    return AEG_OK;
}

static void bind_source(struct aeg_encoder *e, int display)
{
    int i, slot = display % in_slots(e);
    for (i = 0; i < 3; i++)
        e->srcp[i] = e->in_plane[slot][i];
}

static void push_input(struct aeg_encoder *e, const aeg_src *in,
                       int64_t pts)
{
    int slot = e->in_count % in_slots(e);
    copy_yuv(e, e->in_plane[slot], in);
    if (e->in_low[slot].mem)
        aeg_lowres_build(&e->in_low[slot], e->in_plane[slot][0], e->stride[0],
                         e->w, e->h);
    e->in_pts[slot] = pts;
    e->in_qpoff[slot] = e->rc_qp_offset;
    e->in_count++;
}

static int64_t la_cost(struct aeg_encoder *e, int p0, int p1, int b,
                       int *intra);

static void add_job_r(struct aeg_encoder *e, int display, int slice, int idr,
                      int bref)
{
    int n = e->job_n;
    if (n >= (int)(sizeof(e->job_d) / sizeof(e->job_d[0])))
        return;
    e->job_d[n] = display;
    e->job_slice[n] = slice;
    e->job_idr[n] = idr;
    e->job_bref[n] = bref;
    e->job_n = n + 1;
    if (e->vbv_satd) {
        int s = display % in_slots(e);
        int64_t c = -1;
        if (slice == AEG_SLICE_I) {
            const int *ic = la_intra(e, display);
            int i;
            if (ic)
                for (c = 0, i = 0; i < e->mb_count; i++)
                    c += ic[i];
        } else if (slice == AEG_SLICE_P && e->job_nb_last >= 0) {
            c = la_cost(e, e->job_nb_last, display, display, NULL);
        } else if (slice == AEG_SLICE_B && e->job_nb_prev >= 0) {
            c = la_cost(e, e->job_nb_prev, e->job_nb_last, display, NULL);
        }
        e->vbv_satd[s] = c;
    }
    if (slice != AEG_SLICE_B) {
        e->job_nb_prev = e->job_nb_last;
        e->job_nb_last = display;
    }
}
static void add_job(struct aeg_encoder *e, int display, int slice, int idr)
{
    add_job_r(e, display, slice, idr, 0);
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

    e->cur_bref = e->job_bref[0];
    for (i = 1; i < e->job_n; i++) {
        e->job_d[i - 1] = e->job_d[i];
        e->job_slice[i - 1] = e->job_slice[i];
        e->job_idr[i - 1] = e->job_idr[i];
        e->job_bref[i - 1] = e->job_bref[i];
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
    static _Atomic int v = -1;
    if (v < 0) {
        const char *ev = getenv("AEG_LABIAS");
        v = ev ? atoi(ev) : 120;
    }
    return v;
}

/* The intra estimate of display picture d, computed once per picture. */
static const int *la_intra(struct aeg_encoder *e, int d)
{
    int s = d % in_slots(e);

    if (!e->la_icost[s]) {
        e->la_icost[s] = (int *)aeg_alloc(sizeof(int) * (size_t)e->mb_count);
        e->la_icost_d[s] = -1;
        if (!e->la_icost[s])
            return NULL;
    }
    if (e->la_icost_d[s] != d) {
        aeg_lowres_intra(&e->in_low[s], e->la_icost[s]);
        e->la_icost_d[s] = d;
    }
    return e->la_icost[s];
}

/* One field's lowres search into f (vectors, then the costs at
 * 2 * mb_count), on the device when the lookahead stage is placed there:
 * the kernel is the same search, so the field is the same either way
 * (tests/backend_check.c), and the placement is left to timing
 * (aeg_dispatch_la_pick).  Without a device this is the plain call. */
static void la_search(struct aeg_encoder *e, const aeg_lowres *c,
                      const aeg_lowres *r, int *f, const int *seed, int num,
                      int den, int range)
{
    const int nb = c->mb_w * c->mb_h;
    int64_t t0, dt;
    int dev;

    if (!e->d.have_gpu || !e->d.ops.lowres_search || nb != e->mb_count) {
        aeg_lowres_search_seeded(c, r, f, f + 2 * e->mb_count, seed, num, den,
                                 range);
        return;
    }
    dev = aeg_dispatch_la_pick(&e->d);
    t0 = aeg_now_ns();
    if (dev) {
        int rc;
        AEG_PH(AEG_PH_LOWRES,
               rc = e->d.ops.lowres_search(e->d.ops.priv, c->plane, r->plane,
                                           c->stride, c->w, c->h, seed, num,
                                           den, range, f));
        if (rc != AEG_OK) {
            dev = 0;
            aeg_lowres_search_seeded(c, r, f, f + 2 * nb, seed, num, den,
                                     range);
        }
    } else {
        aeg_lowres_search_seeded(c, r, f, f + 2 * nb, seed, num, den, range);
    }
    dt = aeg_now_ns() - t0;
    aeg_dispatch_la_done(&e->d, dev, dt, nb);
    e->st.stage_ns[AEG_STAGE_LOOKAHEAD] += dt;
    if (dev)
        e->st.stage_gpu_ns[AEG_STAGE_LOOKAHEAD] += dt;
    else
        e->st.stage_cpu_ns[AEG_STAGE_LOOKAHEAD] += dt;
}

/* The search field of display picture d against display picture r (vectors
 * at f, costs at f + 2 * mb_count), computed once per pair while d is in the
 * ring: the group decision prices the same pair inside several (p0, p1, b)
 * estimates and the macroblock tree again, 36 % repeats on cock.  An entry
 * of a picture that left the ring is reused first, then the oldest. */
static const int *la_field(struct aeg_encoder *e, int d, int r)
{
    int slots = in_slots(e), s = d % slots, i, v = 0;
    struct la_sfield *t = e->la_sf[s];
    const int dist = d > r ? d - r : r - d;
    const int *seed = NULL;
    int srange = 16;

    for (i = 0; i < e->la_sf_n; i++)
        if (t[i].f && t[i].d == d && t[i].r == r) {
            t[i].age = ++e->la_age;
            return t[i].f;
        }
    /* at distance n > 1 the field at n - 1, scaled up, is a candidate of every
     * block and the range grows with n: a fixed +-16 lowres pels is the same
     * bound at distance 1 and 8, so a long group's far pictures lost their
     * motion and looked dearer than they are (x264 seeds the same way).  The
     * recursion fills the shorter distances, which the group decision prices
     * anyway, and leaves this field the newest of picture d, so it cannot be
     * the victim chosen below. */
    if (dist > 1 && e->la_seed > 0) {
        seed = la_field(e, d, r < d ? r + 1 : r - 1);
        if (!seed)
            return NULL;
        srange = 16 + e->la_seed * (dist - 1);
        if (srange > 64)
            srange = 64;
    }
    for (i = 1; i < e->la_sf_n; i++) {
        int sv = t[v].f && t[v].d == d, si = t[i].f && t[i].d == d;
        if (!t[i].f) { v = i; break; }
        if (sv > si || (sv == si && t[i].age < t[v].age))
            v = i;
    }
    if (!t[v].f) {
        t[v].f = (int *)aeg_alloc(sizeof(int) * 3 * (size_t)e->mb_count);
        if (!t[v].f)
            return NULL;
    }
    la_search(e, &e->in_low[s], &e->in_low[r % slots], t[v].f, seed,
              seed ? dist : 1, seed ? dist - 1 : 1, srange);
    t[v].d = d;
    t[v].r = r;
    t[v].age = ++e->la_age;
    return t[v].f;
}

static int64_t la_cost(struct aeg_encoder *e, int p0, int p1, int b,
                       int *intra)
{
    int slots = in_slots(e), n = e->mb_count;
    const int *ic = la_intra(e, b), *f0 = la_field(e, b, p0);
    const int *f1 = p1 != b ? la_field(e, b, p1) : NULL;
    int64_t c;

    if (!ic || !f0 || (p1 != b && !f1))
        c = aeg_lowres_cost(&e->in_low[b % slots], &e->in_low[p0 % slots],
                            p1 != b ? &e->in_low[p1 % slots] : NULL, intra,
                            e->la_scratch);
    else
        c = aeg_lowres_cost_fields(&e->in_low[b % slots],
                                   &e->in_low[p0 % slots],
                                   f1 ? &e->in_low[p1 % slots] : NULL, ic,
                                   f0, f0 + 2 * n, f1, f1 ? f1 + 2 * n : NULL,
                                   intra);

    /* a B picture is coded at a coarser quantiser and nothing predicts from
     * it; x264 discounts its estimate by the same 1/1.2 */
    if (p1 != b)
        c = c * 100 / la_bias();
    return c;
}

static int64_t la_cost_ci(struct aeg_encoder *e, int p0, int p1, int b,
                          int *intra);

static int b_adapt_decide(struct aeg_encoder *e, int a, int last)
{
    int nmb = e->mb_w > 2 && e->mb_h > 2 ? (e->mb_w - 2) * (e->mb_h - 2)
                                         : e->mb_w * e->mb_h;
    int64_t c2p1, c1b1, c1p0, c2p0;
    int intra, j;

    if (last - a < 2)
        return last;
    c2p1 = la_cost_ci(e, a, a + 2, a + 2, &intra);
    if (intra > nmb / 2)
        return a + 1;
    c1b1 = la_cost_ci(e, a, a + 2, a + 1, NULL);
    c1p0 = la_cost_ci(e, a, a + 1, a + 1, NULL);
    c2p0 = la_cost_ci(e, a + 1, a + 2, a + 2, NULL);
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
            static _Atomic int ts = -1;
            if (ts < 0) {
                const char *ev = getenv("AEG_LATHRESH");
                ts = ev ? atoi(ev) : 100;
            }
            thresh = thresh * ts / 100;
        }
        pc = la_cost_ci(e, a, j + 1, j + 1, &intra);
        if (getenv("AEG_LADBG"))
            fprintf(stderr, "LA   j %d pcost/mb %lld thresh %d intra %d\n", j,
                    (long long)(pc / nmb), thresh, intra);
        if (pc > (int64_t)thresh * nmb || intra > nmb / 3)
            break;
    }
    return j;
}

/* b-adapt 2, x264's "trellis" decision on the same window: every way of
 * splitting a+1..last into groups (k B pictures and their anchor, k <=
 * bframes) is priced as the sum of its pictures' lookahead costs -- the B
 * pictures against their actual references, the pyramid's middle one
 * included (x264's slicetype_path_cost) -- and the first group of the
 * cheapest split is the one coded.  Unlike the fast rule it sees what each
 * B picture would cost, not only how far the anchor can reach. */
static int64_t la_cost_ci(struct aeg_encoder *e, int p0, int p1, int b,
                          int *intra)
{
    /* (b + 1 is stored, so the zeroed table holds no entry; display numbers
     * are never negative) */
    const unsigned hsh = ((unsigned)b * 2654435761u) ^
                         ((unsigned)(b - p0) * 40503u) ^
                         ((unsigned)(p1 - b) * 97u);
    const int i = (int)((hsh ^ (hsh >> 15)) & (LA_CACHE - 1));
    int in = 0;
    int64_t c;

    if (e->la_cache[i].b == b + 1 && e->la_cache[i].p0 == p0 &&
        e->la_cache[i].p1 == p1) {
        if (intra)
            *intra = e->la_cache[i].intra;
        return e->la_cache[i].c;
    }
    c = la_cost(e, p0, p1, b, &in);
    e->la_cache[i].p0 = p0;
    e->la_cache[i].p1 = p1;
    e->la_cache[i].b = b + 1;
    e->la_cache[i].c = c;
    e->la_cache[i].intra = in;
    if (intra)
        *intra = in;
    return c;
}

static int64_t la_cost_cached(struct aeg_encoder *e, int p0, int p1, int b)
{
    return la_cost_ci(e, p0, p1, b, NULL);
}

static int64_t la_group_cost(struct aeg_encoder *e, int s, int t)
{
    int64_t c = la_cost_cached(e, s, t, t);
    int b;

    if (e->p.b_pyramid && t - s > 2) {
        int mid = s + (t - s) / 2;
        c += la_cost_cached(e, s, t, mid);
        for (b = s + 1; b < t; b++)
            if (b != mid)
                c += la_cost_cached(e, b < mid ? s : mid, b < mid ? mid : t, b);
    } else {
        for (b = s + 1; b < t; b++)
            c += la_cost_cached(e, s, t, b);
    }
    return c;
}

/* where the B-adapt trellis from anchor a ends: last (= a + bframes + 1),
 * or with ba_wide every picture already in the ring, as x264's path search
 * runs over its whole lookahead: the first group is then chosen knowing
 * what the ones after it will cost, not just whether it ends well */
static int ba_end(struct aeg_encoder *e, int a, int last)
{
    int l = e->in_count - 1;

    if (!e->ba_wide || e->p.b_adapt != 2)
        return last;
    if (l > a + AEG_MAX_LOOKAHEAD + AEG_MAX_BFRAMES + 1)
        l = a + AEG_MAX_LOOKAHEAD + AEG_MAX_BFRAMES + 1;
    return l > last ? l : last;
}

static int b_adapt_trellis(struct aeg_encoder *e, int a, int last)
{
    int64_t best[AEG_MAX_LOOKAHEAD + AEG_MAX_BFRAMES + 3];
    int from[AEG_MAX_LOOKAHEAD + AEG_MAX_BFRAMES + 3];
    int n = last - a, i, k;

    if (n < 2)
        return last;
    best[0] = 0;
    for (i = 1; i <= n; i++) {
        best[i] = INT64_MAX;
        from[i] = i - 1;
        for (k = 0; k <= e->p.bframes && k <= i - 1; k++) {
            int s = i - k - 1;
            int64_t c = best[s] + la_group_cost(e, a + s, a + i);
            if (c < best[i]) {
                best[i] = c;
                from[i] = s;
            }
        }
    }
    for (i = n; from[i] > 0; i = from[i])
        ;
    if (getenv("AEG_LADBG"))
        fprintf(stderr, "LA2 a %d -> group to %d\n", a, a + i);
    return a + i;
}

/* Macroblock tree over the lookahead window.
 *
 * Every time a coding step is queued, the pictures in the ring past it are
 * planned the way schedule() will code them -- the same group decisions on
 * the same (cached) lowres costs, the keyframe interval honoured, scene cuts
 * not anticipated -- and the tree is propagated from the newest picture of
 * the plan back to the anchor the step predicts from.  Only the offsets of
 * the pictures just queued are kept; the next step plans again with one
 * more group of input in the window.  This is x264's macroblock_tree() on
 * our group structure. */
struct mbt_ent { int d, r0, r1, ref; };

static int mbt_add(struct mbt_ent *pl, int n, int d, int r0, int r1, int ref)
{
    if (n >= AEG_RING)
        return n;
    pl[n].d = d;
    pl[n].r0 = r0;
    pl[n].r1 = r1;
    pl[n].ref = ref;
    return n + 1;
}

/* the pictures of the group a -> nxt in the order schedule() queues them */
static int mbt_group(const struct aeg_encoder *e, struct mbt_ent *pl, int n,
                     int a, int nxt, int idr)
{
    int d;

    if (idr) {
        for (d = a + 1; d < nxt; d++)
            n = mbt_add(pl, n, d, d - 1, -1, 1);
        return mbt_add(pl, n, nxt, -1, -1, 1);
    }
    n = mbt_add(pl, n, nxt, a, -1, 1);
    if (e->p.b_pyramid && nxt - a - 1 >= 2) {
        int mid = a + (nxt - a) / 2;
        n = mbt_add(pl, n, mid, a, nxt, 1);
        for (d = a + 1; d < nxt; d++)
            if (d != mid)
                n = mbt_add(pl, n, d, d < mid ? a : mid, d < mid ? mid : nxt, 0);
    } else {
        for (d = a + 1; d < nxt; d++)
            n = mbt_add(pl, n, d, a, nxt, 0);
    }
    return n;
}

static double mbt_strength(void)
{
    static _Atomic double v = -1.0;
    if (v < 0.0) {
        const char *ev = getenv("AEG_MBTS");
        /* 1.6 since the lowres intra estimate has x264's directional 8x8
         * modes (lookahead.c): 1.3 was the best at constant QP on the short
         * clips before, 1.6 is now level with it there (-0.37 against
         * -0.40% vs x264 veryslow) and 1.5 points ahead at CRF on the long
         * clips, where x264's own 2.0 would be 1.3 more but costs cock and
         * tree 1-3 points at constant QP (docs/STATUS.md). */
        v = ev ? atoi(ev) / 10.0 : 1.6;
    }
    return v;
}

/* pl[0..nq-1] are the pictures just queued; the plan of the rest of the
 * window is appended here.  `root` is the anchor they predict from (-1 for
 * the first picture of the stream). */
static void mbt_run(struct aeg_encoder *e, struct mbt_ent *pl, int nq,
                    int root, int a, int lidr)
{
    int slots = in_slots(e), n = nq, nmb = e->mb_count, k, i;
    int G = e->p.bframes + 1, end = e->in_count - 1;
    double strength = mbt_strength();

    /* the rest of the window, as schedule() will see it */
    while (a < end) {
        int last = a + G, nx, idr;

        if (last > end) {
            if (e->flushing)
                for (i = a + 1; i <= end; i++)
                    n = mbt_add(pl, n, i, i - 1, -1, 1);
            break;
        }
        nx = e->p.b_adapt == 2 ? b_adapt_trellis(e, a, ba_end(e, a, last))
           : e->p.b_adapt ? b_adapt_decide(e, a, last) : last;
        idr = e->p.gop_size > 0 && nx - lidr >= e->p.gop_size;
        n = mbt_group(e, pl, n, a, nx, idr);
        if (idr)
            lidr = nx;
        a = nx;
    }
    if (root >= 0)
        memset(e->mbt[root % slots].prop, 0, sizeof(float) * (size_t)nmb);
    for (k = 0; k < n; k++) {
        struct mbt_slot *m = &e->mbt[pl[k].d % slots];

        memset(m->prop, 0, sizeof(float) * (size_t)nmb);
        if (m->d != pl[k].d || m->p0 != pl[k].r0 || m->p1 != pl[k].r1) {
            int d = pl[k].d, r0 = pl[k].r0, r1 = pl[k].r1;
            const aeg_lowres *l0 = r0 >= 0 ? &e->in_low[r0 % slots] : NULL;
            const aeg_lowres *l1 = r1 >= 0 ? &e->in_low[r1 % slots] : NULL;
            const int *ic = la_intra(e, d);
            const int *f0 = r0 >= 0 ? la_field(e, d, r0) : NULL;
            const int *f1 = r1 >= 0 ? la_field(e, d, r1) : NULL;

            if (!ic || (r0 >= 0 && !f0) || (r1 >= 0 && !f1))
                aeg_lowres_mbcosts(&e->in_low[d % slots], l0, l1, m->intra,
                                   m->inter, m->kind, m->mv, e->la_scratch);
            else
                aeg_lowres_mbcosts_fields(&e->in_low[d % slots], l0, l1, ic,
                                          f0, f0 ? f0 + 2 * nmb : NULL,
                                          f1, f1 ? f1 + 2 * nmb : NULL,
                                          m->intra, m->inter, m->kind, m->mv);
            m->d = pl[k].d;
            m->p0 = pl[k].r0;
            m->p1 = pl[k].r1;
        }
    }
    for (k = n - 1; k >= 0; k--) {
        const struct mbt_slot *m = &e->mbt[pl[k].d % slots];

        if (pl[k].r0 < 0)
            continue;
        mbt_propagate(e->mb_w, e->mb_h, m->intra, m->inter, m->kind, m->mv,
                      m->prop, e->mbt[pl[k].r0 % slots].prop,
                      pl[k].r1 >= 0 ? e->mbt[pl[k].r1 % slots].prop : NULL, 32);
    }
    for (k = 0; k < nq; k++) {
        struct mbt_slot *m = &e->mbt[pl[k].d % slots];

        if (!pl[k].ref)
            continue;
        for (i = 0; i < nmb; i++)
            m->qp[i] = (float)(-strength *
                log2((m->intra[i] + m->prop[i]) / (double)m->intra[i]));
        m->qd = pl[k].d;
    }
}

/* Fill the job queue with the next coding step, if the input has arrived. */
static void schedule_(struct aeg_encoder *e);
static void schedule(struct aeg_encoder *e)
{
    int64_t t0 = aeg_now_ns();
    schedule_(e);
    e->sched_ns += aeg_now_ns() - t0;
}
static void schedule__ph(struct aeg_encoder *e);

/* (timed as a phase, see aeg_ph_switch) */
static void schedule_(struct aeg_encoder *e)
{
    const int prev = aeg_ph_push(AEG_PH_LADEC);

    schedule__ph(e);
    aeg_ph_pop(prev);
}

static void schedule__ph(struct aeg_encoder *e)
{
    struct mbt_ent pl[AEG_RING];

    int G, nxt, idr, d;

    if (e->job_n)
        return;
    if (e->anchor < 0) {
        /* The first picture stands alone: an IDR needs no reference, so it can
         * be coded as soon as it is in the ring. */
        if (e->in_count < 1)
            return;
        if (e->la_depth && !e->flushing &&
            e->in_count <= e->p.bframes + 1 + e->la_depth)
            return;                   /* the tree's window is not full yet */
        add_job(e, 0, AEG_SLICE_I, 1);
        e->anchor = 0;
        if (e->la_depth) {
            mbt_add(pl, 0, 0, -1, -1, 1);
            mbt_run(e, pl, 1, -1, 0, 0);
        }
        return;
    }
    G = e->p.bframes + 1;
    nxt = e->anchor + G;
    if (e->in_count <= nxt) {
        /* the next anchor has not arrived yet -- or never will: at the end of
         * the input the B pictures still waiting have no anchor after them,
         * and a B picture without a list 1 reference cannot be coded at all.
         * They go out as P pictures in display order instead: same pictures,
         * each predicting from the one before it.  This is what x264 does
         * with the B pictures of the last, incomplete group. */
        if (e->flushing && e->in_count > e->anchor + 1) {
            int last = e->in_count - 1, nq = 0;
            for (d = e->anchor + 1; d <= last; d++) {
                add_job(e, d, AEG_SLICE_P, 0);
                nq = mbt_add(pl, nq, d, d - 1, -1, 1);
                if (e->crf_on && e->crf_qc > 0.0)
                    e->crf_cplx[d % in_slots(e)] = la_cost(e, d - 1, d, d, NULL);
            }
            if (e->la_depth)
                mbt_run(e, pl, nq, e->anchor, last, e->last_idr_poc);
            /* the tail becomes the chain of anchors from here on */
            e->anchor = last;
        }
        return;
    }
    if (e->la_depth && !e->flushing && e->in_count <= nxt + e->la_depth)
        return;                       /* the tree's window is not full yet */
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
            aeg_log(AEG_LOG_DEBUG, "aeg: scene cut at frame %d -> IDR", nxt);
            break;
        }
    }
    if (!idr && e->p.b_adapt == 2)
        nxt = b_adapt_trellis(e, e->anchor, ba_end(e, e->anchor, nxt));
    else if (!idr && e->p.b_adapt)
        nxt = b_adapt_decide(e, e->anchor, nxt);
    if (!idr)
        idr = idr_due(e, nxt);
    if (idr) {
        /* No B pictures can precede an IDR, so the group's B pictures go out as
         * P pictures first, in display order, and the IDR comes last. */
        for (d = e->anchor + 1; d < nxt; d++) {
            add_job(e, d, AEG_SLICE_P, 0);
            if (e->crf_on && e->crf_qc > 0.0)
                e->crf_cplx[d % in_slots(e)] = la_cost(e, d - 1, d, d, NULL);
        }
        add_job(e, nxt, AEG_SLICE_I, 1);
    } else {
        add_job(e, nxt, AEG_SLICE_P, 0);
        if (e->p.b_pyramid && nxt - e->anchor - 1 >= 2) {
            /* the middle B picture first, as a reference for the others:
             *   P4 B2(ref) B1 B3 */
            int mid = e->anchor + (nxt - e->anchor) / 2;
            add_job_r(e, mid, AEG_SLICE_B, 0, 1);
            for (d = e->anchor + 1; d < nxt; d++)
                if (d != mid)
                    add_job(e, d, AEG_SLICE_B, 0);
        } else {
            for (d = e->anchor + 1; d < nxt; d++)
                add_job(e, d, AEG_SLICE_B, 0);
        }
    }
    if (e->crf_on && e->crf_qc > 0.0 && !idr)
        e->crf_cplx[nxt % in_slots(e)] = la_cost(e, e->anchor, nxt, nxt, NULL);
    if (e->la_depth) {
        int nq = mbt_group(e, pl, 0, e->anchor, nxt, idr);
        mbt_run(e, pl, nq, e->anchor, nxt, idr ? nxt : e->last_idr_poc);
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

    if (!params || params->width < 2 || params->height < 2 ||
        params->width > 16384 || params->height > 16384) {
        aeg_set_last_error("invalid dimensions");
        return NULL;
    }
    e = (struct aeg_encoder *)aeg_alloc(sizeof(*e));
    if (!e) {
        aeg_set_last_error("out of memory");
        return NULL;
    }
    memcpy(&e->p, params, sizeof(e->p));
    /* the phase profiler (a measurement aid; see h264.h) */
    if (getenv("AEG_PHASES") && atoi(getenv("AEG_PHASES")) > 0) {
        aeg_ph_on = 1;
        aeg_ph_reset();
    }
    pp = &e->p;
    if (pp->bit_depth != 0 && pp->bit_depth != BIT_DEPTH) {
        aeg_set_last_error("bit_depth not supported by this build");
        aeg_free(e);
        return NULL;
    }
    pp->bit_depth = BIT_DEPTH;
    if (pp->fps_num <= 0) pp->fps_num = 30;
    if (pp->fps_den <= 0) pp->fps_den = 1;
    if (pp->qp < 0) pp->qp = 23;
    if (pp->qp > 51) pp->qp = 51;
    if (pp->qp_max <= 0 || pp->qp_max > 51) pp->qp_max = 51;
    if (pp->qp_min <= 0) pp->qp_min = QP_MIN;   /* 0 = unset */
    if (pp->qp_min > pp->qp_max) pp->qp_min = pp->qp_max;
    if (pp->chroma_qp_offset < -12) pp->chroma_qp_offset = -12;
    if (pp->chroma_qp_offset > 12) pp->chroma_qp_offset = 12;
    if (pp->level_idc < 0 || pp->level_idc > 62) pp->level_idc = 0;
    if (pp->interlaced < 0 || pp->interlaced > 2) pp->interlaced = 0;
    if (pp->field_coding < 0 || pp->field_coding > 4) pp->field_coding = 0;
    if (pp->field_coding && !pp->interlaced) pp->interlaced = 1;
    /* frame_mbs_only_flag 0 crops in units of 4 rows (as x264 refuses) */
    if (pp->interlaced && (pp->height & 3)) {
        aeg_set_last_error("interlaced: the height must be a multiple of 4");
        aeg_free(e);
        return NULL;
    }
    if (pp->deblock_alpha < -6) pp->deblock_alpha = -6;
    if (pp->deblock_alpha > 6) pp->deblock_alpha = 6;
    if (pp->deblock_beta < -6) pp->deblock_beta = -6;
    if (pp->deblock_beta > 6) pp->deblock_beta = 6;
    /* Presets trade search effort against speed: the integer search range and
     * whether the subpel refinement runs at all are what actually move the
     * quality/bitrate curve, so they scale with the preset unless the caller
     * asked for something specific. */
    if (pp->preset < 0) pp->preset = 0;
    if (pp->preset > 4) pp->preset = 4;
    if (pp->search_range <= 0) {
        /* The bound is a box of +-range around the zero vector (not around
         * the prediction, as x264's merange is), and the search walks a
         * pattern rather than the box, so a wide box costs nothing measurable;
         * what a narrow one does is cut off real motion: at preset 1, 16 -> 48
         * was -8.95 % on cock and -4.00 % on cock2 (P vectors span four
         * pictures with three B pictures), -1.20 % on the main and validation
         * clips; preset 2 (24) -0.47 %, preset 3 (32) -0.11 %, preset 0 (8)
         * -3.05 % (cock -22 %) and no slower, also with frame threads (the
         * wavefront waits range + 24 rows below the block; from preset 1 the
         * lowres pre-search already waits 63). */
        static const int rng[5] = { 48, 48, 48, 48, 48 };
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
        pp->refine_mv = pp->preset >= 3 ? 2 : 0;
    /* Refining every candidate is a large amount of work (one macroblock
     * evaluation per neighbour per candidate) for a change that measured
     * between 0.3 and 0.6 points, a worse exchange rate than the level
     * selection's, so it is off unless asked for. */
    if (pp->refine_cand < 0)
        pp->refine_cand = 0;
    /* QP-RD: the full walk at preset 4; at preset 3 the fast one (CRF, long
     * clips against x264 veryslow: +3.21 -> +0.64 on long, +1.59 -> +0.81 on
     * long2, -1.19 at constant QP, for about +9 % time; the full walk would
     * be +15 % for 0.01-0.05 more). */
    if (pp->qp_rd < 0)
        pp->qp_rd = pp->preset >= 4 ? 1 : pp->preset == 3 ? 2 : 0;
    /* Level selection is a large amount of work (a price and an inverse
     * transform per candidate level), so presets 3 and 4 run it inside every
     * candidate (2: 0.15-2.4 points of BD-rate for 12-14 % of the encode time
     * when it was added) and presets 1 and 2 on the winner only (1, x264's
     * --trellis 1: -4.1 / -3.7 % on the main and validation clips for 6-8 % of
     * the time, against -4.4 / -4.1 % for 26 % in every candidate; see
     * "Presets 1 and 2"). */
    if (pp->trellis < 0)
        pp->trellis = pp->preset >= 3 ? 2 : pp->preset >= 1 ? 1 : 0;
    if (pp->trellis > 2)
        pp->trellis = 2;
    if (pp->gop_size < 0) pp->gop_size = 250;
    /* B pictures: -1 = the preset's, up to AEG_MAX_BFRAMES, and none at all
     * without CABAC -- the B macroblock layer is only implemented for the
     * arithmetic coder (see code_slice).  Presets 3 and 4 allow eight in a
     * group, as x264's slower presets do: the adaptive decision still picks
     * the length, and on still or slowly moving content a group of 6-8 is
     * what x264 veryslow chooses (on the tree clip: 4.99 % behind x264 against
     * 7.49 % with three; long and long2 sets -0.48 / -0.60 points at preset 4
     * with the two lookahead changes below; see docs/STATUS.md, "Long B
     * groups").  The faster presets keep three, which the speed of the
     * smaller decision pays for. */
    if (pp->bframes == -1) pp->bframes = pp->preset >= 3 ? 8 : 3;
    if (pp->bframes < 0) pp->bframes = 0;
    if (pp->bframes > AEG_MAX_BFRAMES) pp->bframes = AEG_MAX_BFRAMES;
    if (pp->bframes > 0 && !pp->cabac) {
        aeg_log(AEG_LOG_WARNING, "aeg: B pictures need CABAC; coding without them");
        pp->bframes = 0;
    }
    /* A second P reference (list0[1], the anchor before the previous one).
     * Measured with tools/abbd.py over the four synthetic clips at preset 3:
     * -1.4 % mean BD-rate for +16 % time on P-only streams, but -0.2 % (noise)
     * with three B pictures, where the anchors are four pictures apart and the
     * older one is eight away -- and where a direct block cannot follow a
     * co-located block on it (direct_group_ok).  On the five real clips, with
     * the adaptive B decision, it is -0.18 % (four of five better) for +3 %
     * time, so preset 3 and up use it with B pictures too.  Beyond two (see
     * set_refs_multi, docs/STATUS.md *More than two P references*): -0.97 %
     * at preset 3 with four and -1.45 % at preset 4 with eight, on the long,
     * main and val clips at constant QP, for a few per cent of time. */
    if (pp->refs < 0)
        pp->refs = pp->preset >= 4 ? 8 : pp->preset >= 3 ? 4 : 1;
    if (pp->i_qp_offset < 0)
        pp->i_qp_offset = 3;
    if (pp->direct < 0 || pp->direct > 1)
        pp->direct = 1;
    /* The group decision.  The trellis (b-adapt 2) prices every split of the
     * window; the fast rule stops adding B pictures once a far P picture's
     * lowres cost per macroblock passes an absolute threshold, which busy
     * texture (tree2: ~650 against 250) always does even where B pictures
     * pay -- one B per group there, against three from the trellis and from
     * x264.  Global-PSNR BD-rate of the trellis against the fast rule, vs
     * x264 medium: preset 3 -5.37 -> -5.51 % on the five clips and -4.37 ->
     * -6.07 % on the six validation clips (preset 4: -5.84 -> -5.91, -4.66
     * -> -6.47); short loses (-9.39 -> -6.22), mega, mega2 and tree2 gain
     * 4-5 points each.  +3 % time at preset 3, none measurable at 4.  x264
     * also switches to the trellis at its slow preset. */
    if (pp->b_adapt < 0 || pp->b_adapt > 2)
        pp->b_adapt = pp->preset >= 3 ? 2 : 1;
    /* (preset 1 too since the p1/p2 retune: -0.16 % on the main and
     * validation clips -- short, which fades, -1.8 -- at no measurable time) */
    if (pp->weightp < 0)
        pp->weightp = pp->preset >= 1;
    if (pp->refs < 1)
        pp->refs = 1;
    if (pp->refs > AEG_MAX_REFS)
        pp->refs = AEG_MAX_REFS;
    /* The pyramid needs two B pictures to have a middle one, spatial direct
     * (temporal direct from a reference B picture would need the co-located
     * picture's list 1 mapped into this picture's lists, which is not
     * implemented) and a single P reference (the anchors drop every older
     * picture, see set_current_refs). */
    if (pp->b_pyramid < 0)
        pp->b_pyramid = 1;
    if (pp->bframes < 2 || pp->direct != 1)
        pp->b_pyramid = 0;
    pp->b_pyramid = pp->b_pyramid ? 1 : 0;
    /* Field pictures: not (yet) with temporal direct (the co-located
     * field's vectors), and chroma stays out of the motion search (its field
     * offset is not in the cost).  (The pyramid, the NAL HRD and weighted
     * prediction run in field pictures; MBAFF frames go without weights.) */
    if (pp->field_coding) {
        /* (AEG_NOMBPYR / AEG_NOFPYR: none in MBAFF frames / field pictures) */
        if (pp->field_coding == 3 ? getenv("AEG_NOMBPYR") != NULL
                                  : getenv("AEG_NOFPYR") != NULL)
            pp->b_pyramid = 0;
        pp->direct = 1;
        /* (field pictures do without; frame pictures among them keep them) */
        if (pp->field_coding == 3)
            pp->weightp = 0;
        if (pp->field_coding != 2)
            pp->chroma_me = 0;
    }
    /* The macroblock tree needs the lookahead ring, which only exists with B
     * pictures (without them every picture is coded as it arrives). */
    /* On by default except with a buffer model, which predicts a picture's
     * size from its QP and knows nothing of the offsets beyond their mean
     * (asked for explicitly, it gets that).  ABR keeps it, as x264 does: on
     * six-second clips the tree took the one-pass stream from -0.13..-2.70
     * dB against CRF to -0.02..-0.81, and the two-pass one to within 0.15
     * dB of CRF (docs/STATUS.md). */
    if (pp->mbtree < 0)
        pp->mbtree = pp->preset >= 1 && pp->vbv_maxrate_kbps <= 0;
    if (pp->bframes <= 0)
        pp->mbtree = 0;
    pp->mbtree = pp->mbtree ? 1 : 0;
    if (pp->rc_lookahead < 0) {
        /* preset 1 was 10; 20 measured -0.45 % at no measurable time (on
         * 30-picture clips, too short to show a longer window).  On the long
         * clips (CRF, BD-rate against x264 veryslow, tools/x264cmp.py): 20 ->
         * 40 at presets 1 / 2 -1.40 / -1.53 % (long2: -0.40 / -0.40 %), and
         * 30 -> 60 at preset 3 -1.10 % (long2 -0.18 %), 40 -> 60 at preset 4
         * -0.53 % (long2 0.00 %); almost all of it on the static surveillance
         * clips (vtestL), the rest within +-0.13 % per clip.  No measurable
         * time; the ring costs about 2.5 MB per 720p picture (+76 MB from 30
         * to 60) and the window adds as many pictures of latency.  x264 uses
         * 40 at medium and 60 at slower / veryslow. */
        static const int la[5] = { 0, 40, 40, 60, 60 };
        pp->rc_lookahead = la[pp->preset];
    }
    if (pp->rc_lookahead > AEG_MAX_LOOKAHEAD)
        pp->rc_lookahead = AEG_MAX_LOOKAHEAD;
    if (pp->rc_lookahead <= 0)
        pp->mbtree = 0;
    e->la_depth = pp->mbtree ? pp->rc_lookahead : 0;
    e->la_sf_n = 2 * (pp->bframes + 1) + 2 > 8 ? 2 * (pp->bframes + 1) + 2 : 8;
    /* presets 3 and 4: the seeded lowres search and the B-adapt trellis over
     * the whole ring (see la_field and ba_end) */
    e->la_seed = pp->preset >= 3 ? 8 : 0;
    e->ba_wide = pp->preset >= 3;
    if (pp->b_qp_offset < 0)
        pp->b_qp_offset = pp->mbtree ? 2 : 4;
    /* (an offset past the QP scale only ever means "as coarse as it goes";
     * clamped here, it cannot overflow the QP arithmetic) */
    if (pp->b_qp_offset > 51)
        pp->b_qp_offset = 51;
    if (pp->i_qp_offset > 51)
        pp->i_qp_offset = 51;
    /* A buffer model without a rate target makes no sense -- a constant-QP
     * stream cannot be squeezed into a channel -- so asking for a VBV turns
     * ABR on, with the maximum rate as the target when no bitrate was given.
     * CRF is the exception: there the buffer is a cap on the constant-quality
     * stream ("capped CRF"), the QP rising only where the buffer runs low. */
    if (pp->qcomp < 0) pp->qcomp = 0;
    if (pp->qcomp > 100) pp->qcomp = 100;
    if (pp->rc_mode == AEG_RC_CRF) {
        if (!(pp->crf >= QP_MIN)) pp->crf = QP_MIN; /* also catches NaN */
        if (pp->crf > 51.0) pp->crf = 51.0;
        pp->bitrate_kbps = 0;
        if (pp->vbv_maxrate_kbps <= 0 || pp->vbv_bufsize_kbits <= 0)
            pp->vbv_maxrate_kbps = pp->vbv_bufsize_kbits = 0;
        e->crf_on = 1;
    } else if (pp->vbv_maxrate_kbps > 0 && pp->vbv_bufsize_kbits > 0) {
        if (pp->bitrate_kbps <= 0)
            pp->bitrate_kbps = pp->vbv_maxrate_kbps;
        if (pp->bitrate_kbps > pp->vbv_maxrate_kbps) {
            aeg_log(AEG_LOG_WARNING, "aeg: bitrate clamped to the VBV maximum rate (%d kb/s)",
                    pp->vbv_maxrate_kbps);
            pp->bitrate_kbps = pp->vbv_maxrate_kbps;
        }
        pp->rc_mode = AEG_RC_ABR;
    } else
        pp->vbv_maxrate_kbps = pp->vbv_bufsize_kbits = 0;

    if (e->crf_on)
        ;
    else if (pp->rc_mode != AEG_RC_ABR || pp->bitrate_kbps <= 0)
        pp->rc_mode = AEG_RC_CQP;
    else {
        double fps = (double)pp->fps_num / (double)pp->fps_den;
        e->rc_active = 1;
        e->crf_on = 1;              /* the model; see abr_on */
        e->abr_on = 1;
        e->abr_fps = fps;
        e->abr_bpf = (double)pp->bitrate_kbps * 1000.0 / fps;
        aeg_log(AEG_LOG_INFO, "aeg: ABR %d kb/s -> %.0f bits/frame at %g fps, start QP %d",
                pp->bitrate_kbps, e->abr_bpf, fps, pp->qp);
    }
    e->vbv_gap = 2;
    if (pp->vbv_maxrate_kbps > 0 && (e->rc_active || e->crf_on)) {
        double fps = (double)pp->fps_num / (double)pp->fps_den;
        {
            e->vbv_active = 1;
            e->vbv_buf_bits = (double)pp->vbv_bufsize_kbits * 1000.0;
            e->vbv_rate_bits = (double)pp->vbv_maxrate_kbps * 1000.0 / fps;
            /* start the model with the buffer nearly full: the first picture is
             * a keyframe and would starve a buffer that starts empty */
            e->vbv_full = 0.9 * e->vbv_buf_bits;
            e->vbv_min_full = e->vbv_full;
            e->last_pic_bits = 0;
            e->last_pic_qp = -1000;     /* none (below any QP) */
            e->last_anchor_bits = 0;
            e->last_anchor_qp = -1000;
            aeg_log(AEG_LOG_INFO, "aeg: VBV %d kb/s, buffer %d kbit -> level %.0f bits, "
                        "%.0f bits delivered per picture",
                    pp->vbv_maxrate_kbps, pp->vbv_bufsize_kbits,
                    e->vbv_full, e->vbv_rate_bits);
        }
    }

    /* 4:2:0 crops in units of two samples, so an odd size cannot be
     * signalled: it is rounded down to even (as x264 refuses it) */
    e->vw = params->width & ~1;
    e->vh = params->height & ~1;
    if (e->vw != params->width || e->vh != params->height)
        aeg_log(AEG_LOG_WARNING, "aeg: odd dimensions rounded down to %dx%d", e->vw, e->vh);
    e->vcw = e->vw / 2;
    e->vch = e->vh / 2;
    pp->width = e->vw;
    pp->height = e->vh;
    e->w = (e->vw + 15) & ~15;
    /* interlaced: the SPS counts the height in pairs of macroblock rows */
    e->h = pp->interlaced ? (e->vh + 31) & ~31 : (e->vh + 15) & ~15;
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
    e->level_idc = pp->level_idc > 0 ? pp->level_idc : 0;
    /* An explicit level bounds the decoded picture buffer: fewer P
     * references until it fits (x264 does the same); the automatic level is
     * picked to fit instead (see pick_level). */
    if (e->level_idc > 0 && aeg_level_dpb_mbs(e->level_idc) > 0) {
        int64_t lim = aeg_level_dpb_mbs(e->level_idc);
        int r0 = pp->refs;
        while (pp->refs > 1 && (int64_t)aeg_dpb_frames(pp) * e->mb_count > lim)
            pp->refs--;
        if (pp->refs != r0)
            aeg_log(AEG_LOG_WARNING, "aeg: level %d.%d: %d reference pictures "
                    "(not %d) fit its decoded picture buffer",
                    e->level_idc / 10, e->level_idc % 10, pp->refs, r0);
    }
    e->pic.chroma_qp_offset = pp->chroma_qp_offset;
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
    e->vpad = pp->field_coding ? 2 * AEG_PAD : AEG_PAD;
    e->mbaff = pp->field_coding >= 3;   /* (4: per picture, encode_frame) */
    /* (AEG_NOFWP: the field pictures without weights, as before) */
    e->fwp = pp->weightp && pp->field_coding && pp->field_coding != 3 &&
             !getenv("AEG_NOFWP");
    e->mbaff_ff = -1;
    e->scan4 = aeg_scan4x4;
    e->scan8 = aeg_zigzag8x8;

    if (enc_alloc_picture(e, &e->src_mem, e->srcp) != AEG_OK) {
        aeg_set_last_error("out of memory");
        aeg_free(e);
        return NULL;
    }
    {
        int i;
        e->max_refs = aeg_sps_max_refs(pp);
        e->npic_serial = e->max_refs + 1 > AEG_NPIC_SERIAL ? e->max_refs + 1
                                                           : AEG_NPIC_SERIAL;
        for (i = 0; i < e->npic_serial; i++)
            if (slot_alloc(e, i) != AEG_OK) {
                aeg_set_last_error("out of memory");
                aeg_encoder_close(e);
                return NULL;
            }
        e->npic = e->npic_serial;
        e->role_cur = 0;
        e->role_last = 0;
        e->ndpb = 0;
        bind_pictures(e);
    }
    /* own planes for references 0 and 1 (a weighted list 0, the GPU's
     * planes); the others always read their buffer's (slot_mc) */
    aeg_mc_ctx_init_pad(&e->mcr[0], e->w, e->h, e->stride[0], e->vpad);
    aeg_mc_ctx_init_pad(&e->mcr[1], e->w, e->h, e->stride[0], e->vpad);
    /* references past the second borrow their buffer's planes (no buffer
     * of their own), at the same stride */
    for (int k = 2; k < AEG_MAX_REFS; k++)
        e->mcr[k].stride = e->stride[0];
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
        /* MBAFF: a slice holds whole macroblock pairs */
        if (pp->field_coding >= 3)
            e->slice_rows += e->slice_rows & 1;
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
    {
        int k;
        e->lr_scratch = (int *)aeg_alloc((size_t)e->mb_count * 4 * sizeof(int));
        e->lr_mv[0] = (int16_t *)aeg_alloc((size_t)e->mb_count * 2 * sizeof(int16_t));
        e->lr_mv[1] = (int16_t *)aeg_alloc((size_t)e->mb_count * 2 * sizeof(int16_t));
        for (k = 0; k < 3; k++)
            if (aeg_lowres_alloc(&e->lr_pic[k], e->mb_w, e->mb_h))
                e->lr_pic[k].mem = NULL;
    }
    if (!e->mb_encs || !e->seed_mv || !e->seed_cand || !e->seed_sad ||
        !e->seed_cost || !e->prev_seed || !e->par_pos || !e->par_mv || !e->par_cost ||
        !e->lr_scratch || !e->lr_mv[0] || !e->lr_mv[1] ||
        !e->lr_pic[0].mem || !e->lr_pic[1].mem || !e->lr_pic[2].mem) {
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
        int i, slots = in_slots(e);

        for (i = 0; i < slots; i++) {
            if (enc_alloc_picture(e, &e->in_mem[i], e->in_plane[i]) != AEG_OK) {
                aeg_set_last_error("out of memory");
                aeg_encoder_close(e);
                return NULL;
            }
        }
        e->in_pts = (int64_t *)aeg_alloc((size_t)slots * sizeof(int64_t));
        e->col_mbs = e->slot_mbs[0];
        if (!e->in_pts || !e->slot_mbs[0] || !e->slot_mbs[1] ||
            !e->slot_mbs[2] || !e->slot_mbs[3]) {
            aeg_set_last_error("out of memory");
            aeg_encoder_close(e);
            return NULL;
        }
        for (i = 0; i < slots; i++)
            e->in_pts[i] = 0;
        if (e->p.b_adapt || e->la_depth || e->crf_on || e->vbv_active) {
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
        if (e->vbv_active) {
            e->vbv_satd = (int64_t *)aeg_alloc((size_t)slots * sizeof(int64_t));
            if (!e->vbv_satd) {
                aeg_set_last_error("out of memory");
                aeg_encoder_close(e);
                return NULL;
            }
            for (i = 0; i < slots; i++)
                e->vbv_satd[i] = -1;
        }
        if (e->crf_on) {
            e->crf_cplx = (int64_t *)aeg_alloc((size_t)slots * sizeof(int64_t));
            if (!e->crf_cplx) {
                aeg_set_last_error("out of memory");
                aeg_encoder_close(e);
                return NULL;
            }
            for (i = 0; i < slots; i++)
                e->crf_cplx[i] = -1;
        }
        for (i = 0; e->la_depth && i < slots; i++) {
            struct mbt_slot *m = &e->mbt[i];
            size_t nm = (size_t)e->mb_count;
            m->intra = (int *)aeg_alloc(nm * sizeof(int));
            m->inter = (int *)aeg_alloc(nm * sizeof(int));
            m->mv = (int *)aeg_alloc(nm * 4 * sizeof(int));
            m->kind = (uint8_t *)aeg_alloc(nm);
            m->prop = (float *)aeg_alloc(nm * sizeof(float));
            m->qp = (float *)aeg_alloc(nm * sizeof(float));
            m->d = m->qd = -1;
            if (!m->intra || !m->inter || !m->mv || !m->kind || !m->prop ||
                !m->qp) {
                aeg_set_last_error("out of memory");
                aeg_encoder_close(e);
                return NULL;
            }
        }
    }
    if (e->crf_on) {
        /* x264's form: a picture whose lowres estimate is base_cplx (80 per
         * macroblock, 120 with B pictures) is coded at qscale(crf + shift);
         * with the tree, qcomp is the tree's business and the shift also
         * centres the tree's (all negative) offsets -- x264's
         * (1 - qcomp) * 13.5 for its strength 5 * (1 - qcomp), scaled to
         * ours.  On top of that the scale is calibrated so that a CRF gives
         * the PSNR x264 --preset medium --tune psnr gives at the same CRF
         * (preset 3, four clips, CRF 16..38; docs/STATUS.md): +2.25 with the
         * tree, +1 without (B pictures or not). */
        double base = (double)e->mb_count * (e->p.bframes > 0 ? 120.0 : 80.0);
        double q0;

        e->crf_qc = e->la_depth ? 0.0 : 1.0 - e->p.qcomp / 100.0;
        e->crf_tree_off = e->la_depth ? 2.7 * mbt_strength() + 2.25 : 1.0;
        q0 = e->abr_on ? (double)e->p.qp : e->p.crf + e->crf_tree_off;
        e->crf_rf = pow(base, e->crf_qc) / (0.85 * pow(2.0, (q0 - 12.0) / 6.0));
        if (e->abr_on) {
            /* the start: the rate factor that would code a picture of
             * base_cplx at the start QP, worth one picture of evidence */
            e->abr_wanted = e->abr_bpf;
            e->abr_cplxr = e->abr_bpf / e->crf_rf;
            e->abr_last_pqs = qp2qs(q0);
            e->abr_last_rceq = e->crf_qc > 0.0 ? pow(base, e->crf_qc) : 1.0;
            e->abr_rceq = e->abr_last_rceq;
        }
        e->crf_pnorm = 0.01;
        e->crf_pqp = q0 * e->crf_pnorm;
        e->crf_anchor_q = q0;
        e->crf_last = -1;
        if (e->p.bframes <= 0 && (e->crf_qc > 0.0 || e->vbv_active)) {
            e->la_scratch = (int *)aeg_alloc((size_t)e->mb_count * 4 *
                                             sizeof(int));
            if (!e->la_scratch ||
                aeg_lowres_alloc(&e->crf_low[0], e->mb_w, e->mb_h) ||
                aeg_lowres_alloc(&e->crf_low[1], e->mb_w, e->mb_h)) {
                aeg_set_last_error("out of memory");
                aeg_encoder_close(e);
                return NULL;
            }
        }
        if (!e->abr_on)
            aeg_log(AEG_LOG_INFO, "aeg: CRF %.2f, qcomp %.2f%s", e->p.crf,
                    e->p.qcomp / 100.0,
                    e->la_depth ? " (with the macroblock tree)" : "");
    }
    e->job_nb_last = e->job_nb_prev = -1;
    e->cur_satd = -1;
    e->crf_last = -1;
    if (e->vbv_active && e->p.bframes <= 0 && !e->crf_low[0].mem) {
        if (!e->la_scratch)
            e->la_scratch = (int *)aeg_alloc((size_t)e->mb_count * 4 *
                                             sizeof(int));
        if (!e->la_scratch ||
            aeg_lowres_alloc(&e->crf_low[0], e->mb_w, e->mb_h) ||
            aeg_lowres_alloc(&e->crf_low[1], e->mb_w, e->mb_h)) {
            aeg_set_last_error("out of memory");
            aeg_encoder_close(e);
            return NULL;
        }
    }
    {
        /* bits * qscale / satd measured 0.44..1.76 for intra pictures over
         * five clips (docs/STATUS.md); the start value errs high */
        int t;
        for (t = 0; t < 3; t++) {
            e->vbv_coef[t] = 1.5;
            e->vbv_off[t] = 0.0;
            e->vbv_cnt[t] = 1.0;
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
    e->pic.mbaff = e->mbaff;
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
    /* frame threads: 0 = auto (on with more than one CPU, unless -t 1) */
    {
        /* the workers wait for each other's rows, so one more than there
         * are CPUs keeps them busy */
        int ft = pp->frame_threads;
        if (ft <= 0) {
            long nc = sysconf(_SC_NPROCESSORS_ONLN);
            ft = (nc > 1 && pp->threads != 1) ? (int)nc + 1 : 1;
        }
        if (getenv("AEG_FTW"))
            ft = atoi(getenv("AEG_FTW"));
        if (ft >= 2 && e->p.bframes > 0 && !e->rc_active && !e->vbv_active &&
            e->n_slices == 1 && !getenv("AEG_SEED") && !getenv("AEG_POISON") &&
            !(e->d.have_gpu &&
              (e->p.stage_backend[AEG_STAGE_MOTION_EST] == AEG_BACKEND_OPENCL ||
               e->p.stage_backend[AEG_STAGE_MOTION_EST] == AEG_BACKEND_CUDA))) {
            if (ft_alloc(e, ft) != AEG_OK) {
                aeg_set_last_error("out of memory");
                aeg_encoder_close(e);
                return NULL;
            }
            e->ft_on = 1;
        }
    }
    if (aeg_hrd_values(&e->p, &e->hrd_rate, &e->hrd_cpb) && e->vbv_active) {
        e->hrd_on = 1;
    } else if (e->p.nal_hrd) {
        aeg_log(AEG_LOG_WARNING, "aeg: nal_hrd needs the VBV model (vbv maxrate and buffer "
                   "size); no HRD is signalled");
        e->p.nal_hrd = 0;
    }
    if (e->p.info_sei) {
        /* what shaped the stream -- not how it was computed (threads,
         * backend), so a threaded encode writes the same bytes */
        char crf_txt[48] = "";
        if (e->abr_on)
            snprintf(crf_txt, sizeof(crf_txt), " qcomp=%.2f",
                     e->p.qcomp / 100.0);
        else if (e->crf_on)
            snprintf(crf_txt, sizeof(crf_txt), " crf=%.2f qcomp=%.2f",
                     e->p.crf, e->p.qcomp / 100.0);
        snprintf(e->info, sizeof(e->info),
                 "aeg %s - H.264/AVC encoder (CPU, CUDA/OpenCL stages) - "
                 "options: preset=%d rc=%s qp=%d bitrate=%d vbv=%d:%d "
                 "keyint=%d scenecut=%d bframes=%d b-adapt=%d b-pyramid=%d "
                 "bqp=%d iqp=%d direct=%s ref=%d weightp=%d mbtree=%d "
                 "rc-lookahead=%d aq=%d:%d trellis=%d subme=%d range=%d "
                 "refine-mv=%d refine-cand=%d qp-rd=%d partitions=%d cabac=%d "
                 "8x8dct=%d deblock=%d:%d:%d slices=%d%s",
                 AEG_VERSION_STRING, e->p.preset,
                 e->rc_active ? (e->vbv_active ? "abr+vbv" : "abr")
                 : e->crf_on ? (e->vbv_active ? "crf+vbv" : "crf") : "cqp",
                 e->p.qp, e->p.bitrate_kbps, e->p.vbv_maxrate_kbps,
                 e->p.vbv_bufsize_kbits, e->p.gop_size, e->p.scenecut,
                 e->p.bframes, e->p.b_adapt, e->p.b_pyramid, e->p.b_qp_offset,
                 e->p.i_qp_offset, e->p.direct ? "spatial" : "temporal",
                 e->p.refs, e->p.weightp, e->p.mbtree,
                 e->p.mbtree ? e->p.rc_lookahead : 0, e->p.aq_mode,
                 e->p.aq_strength, e->p.trellis, e->p.subme,
                 e->p.search_range, e->p.refine_mv, e->p.refine_cand, e->p.qp_rd,
                 e->p.partitions, e->p.cabac, e->p.transform_8x8,
                 e->p.deblock, e->p.deblock_alpha, e->p.deblock_beta,
                 e->n_slices, crf_txt);
        if (e->p.interlaced) {
            size_t n = strlen(e->info);
            snprintf(e->info + n, sizeof(e->info) - n, " interlaced=%s",
                     e->p.interlaced == 1 ? "tff" : "bff");
        }
    }
    if (write_param_sets(e) != AEG_OK) {
        aeg_set_last_error("parameter sets do not fit");
        aeg_encoder_close(e);
        return NULL;
    }
    /* one frame in the caller's pts unit: (fps_den / fps_num) seconds over
     * (tb_num / tb_den) seconds per tick */
    if (pp->timebase_num > 0 && pp->timebase_den > 0)
        e->dts_dur = (double)pp->fps_den * pp->timebase_den /
                     ((double)pp->fps_num * pp->timebase_num);
    else {
        pp->timebase_num = pp->fps_den;
        pp->timebase_den = pp->fps_num;
        e->dts_dur = 1.0;
    }
    aeg_log(AEG_LOG_INFO, "aeg %s: %dx%d, %dx%d macroblocks, qp %d, gop %d",
            AEG_VERSION_STRING, e->w, e->h, e->mb_w, e->mb_h, e->p.qp, e->p.gop_size);
    return e;
}

/* ------------------------------------------------------------------ */
/* Frame threads                                                        */
/* ------------------------------------------------------------------ */
/* Pictures are coded by a pool of worker threads, several at a time, and the
 * stream is the same byte for byte as the serial one.  What makes that true:
 *
 *  - Each call starts exactly the picture the serial encoder would code in
 *    it (while input still arrives): its prologue -- slice type, references,
 *    macroblock-tree offsets, adaptive quantisation -- runs on the encoder
 *    itself, in the serial order, and reads only what the lookahead knows at
 *    that moment.  The picture is then coded (frame_body) by a worker on its
 *    own copy of the encoder struct whose every *written* buffer is private:
 *    macroblock decisions, coefficient and seed scratch, the half-resolution
 *    pre-search, the fractional and weighted planes, the slice and packet
 *    buffers.  What coding the picture would have done to the encoder (the
 *    frame counters, the zeroed seed vectors, reference marking and buffer
 *    rotation) the main thread does at launch; statistics merge at the join.
 *  - Row wavefront: a reference picture is reconstructed into the buffer the
 *    pictures predicting from it read, and post-processed (deblocked, padded,
 *    filtered down, interpolated) row by row behind the decision pass; each
 *    finished row is published (slot_publish).  A picture predicting from it
 *    waits per macroblock row (ref_rows_ensure -> ref_wait) only for the
 *    rows its search can reach: the vector is bounded by the search range,
 *    so the samples it reads are final and the decisions are the serial
 *    ones.  AEG_POISON=1 (serial) fills the rows that would not be final yet
 *    with noise to prove the bounds.
 *  - Weighted prediction analyses the whole reference.  Rather than wait for
 *    it, a P picture starts as if the analysis had found no weights
 *    (ft_spec) and runs the analysis once the reference is whole -- always
 *    before its own last rows, which need the whole reference anyway.  No
 *    weights: everything coded is what the serial encoder codes.  Weights:
 *    the picture restarts (FT_RESTART) from the copy it was launched with.
 *    A restarting reference bumps its generation; every picture that read
 *    it (counted in readers) notices at its next row, stops reading, and
 *    restarts in turn, and the reference is rewritten only after that
 *    (stale).  Rows are published only while every reference read is still
 *    the current generation, and a picture's last rows only after its
 *    analysis, so a picture whose last rows are out never restarts and a
 *    finished picture read nothing stale.  A picture the same analysis on
 *    the reference's source (wp_predict) expects to be weighted waits for
 *    the reference instead.
 *  - A non-reference picture reconstructs into its worker's own buffer.  A
 *    buffer that reference marking drops stays in use while a picture in
 *    flight still reads it (pic_busy), so there are NW + 2 more buffers.
 *  - An input picture's ring slot is not overwritten while a picture coded
 *    from it is in flight (ft_ring_guard).
 *  - Packets leave in coding order through a small queue; the reconstruction
 *    of each is kept with it for aeg_encoder_recon().
 *
 * Gate (aeg_encoder_open): B pictures, constant QP (rate control and the VBV
 * model feed every picture's size into the next picture's QP), one slice per
 * picture, the CPU backend, and more than one CPU.  NW defaults to the CPU
 * count + 1 (the workers wait for each other's rows), up to FT_MAXW. */
#ifndef FT_Q
#define FT_Q 16
#endif
#define FT_MAXW 8

struct ft_entry {
    uint8_t *buf;
    size_t   cap, size;
    int64_t  pts;
    int      keyframe, qp, type;
    double   qp_avg, qp_base;
    int64_t  sse[3];            /* the picture's distortion (pic_quality) */
    double   ssim;
    int      pending;           /* in a worker, not joined yet */
    pixel *rec_mem;
    int     *mbqp;
};

struct ft_worker {
    pthread_t th;
    int busy;                   /* 1 = thread running, 2 = done inline */
    int done;                   /* the body has returned (atomic) */
    int64_t seq;                /* launch order */
    struct aeg_encoder *c;
    struct aeg_encoder *c0;     /* c as launched, for a restart */
    int8_t *mb_dqp0;
    aeg_mb *mbs;
    mbenc *mb_encs;
    int8_t *mb_dqp;
    int16_t *seed_mv, *prev_seed, *seed_cand, *par_pos, *par_mv;
    int32_t *seed_sad, *seed_cost, *par_cost;
    int *lr_scratch;
    int16_t *lr_mv[2];
    aeg_lowres lr_pic[3];
    aeg_mc_ctx mcr[2];          /* own planes of references 0 and 1 */
    pixel *pic_mem, *plane[3];
    pixel *wp_mem, *wp_plane[3];
    uint8_t *slice_bs[AEG_MAX_SLICES];
    size_t slice_bs_cap[AEG_MAX_SLICES];
    uint8_t *out;
    size_t out_cap;
    int slice, idr, d, rc;
    int dref;                   /* input picture of the list 0 reference
                                 * (wp_predict reads its source), or -1 */
    int rec_slot;               /* buffer reconstructed into, -1 = private */
    int uses[AEG_NPIC + 1], nuses; /* buffers counted in pic_busy */
    int fld;                    /* field pictures (field_frame_body) */
    int64_t pts, t_all, ns;
    aeg_packet pkt;
    struct ft_entry *ent;
};

static void ft_worker_free(struct ft_worker *w)
{
    int k;

    if (w->busy == 1)
        pthread_join(w->th, NULL);
    free(w->c);
    free(w->c0);
    aeg_free(w->mb_dqp0);
    aeg_free(w->mbs);
    aeg_free(w->mb_encs);
    aeg_free(w->mb_dqp);
    aeg_free(w->seed_mv);
    aeg_free(w->prev_seed);
    aeg_free(w->seed_cand);
    aeg_free(w->par_pos);
    aeg_free(w->par_mv);
    aeg_free(w->seed_sad);
    aeg_free(w->seed_cost);
    aeg_free(w->par_cost);
    aeg_free(w->lr_scratch);
    aeg_free(w->lr_mv[0]);
    aeg_free(w->lr_mv[1]);
    for (k = 0; k < 3; k++)
        if (w->lr_pic[k].mem)
            aeg_lowres_free(&w->lr_pic[k]);
    aeg_mc_ctx_free(&w->mcr[0]);
    aeg_mc_ctx_free(&w->mcr[1]);
    aeg_free(w->pic_mem);
    aeg_free(w->wp_mem);
    for (k = 0; k < AEG_MAX_SLICES; k++)
        aeg_free(w->slice_bs[k]);
    aeg_free(w->out);
}

static void ft_free(struct aeg_encoder *e)
{
    int k;

    if (e->ftw) {
        for (k = 0; k < e->ft_nw; k++)
            ft_worker_free(&e->ftw[k]);
        free(e->ftw);
        e->ftw = NULL;
    }
    if (e->ftq) {
        for (k = 0; k < FT_Q; k++) {
            aeg_free(e->ftq[k].buf);
            aeg_free(e->ftq[k].rec_mem);
            free(e->ftq[k].mbqp);
        }
        free(e->ftq);
        e->ftq = NULL;
    }
    if (e->fsync) {
        pthread_mutex_destroy(&e->fsync->mu);
        pthread_cond_destroy(&e->fsync->cv);
        free(e->fsync);
        e->fsync = NULL;
    }
    e->ft_on = 0;
}

static int ft_worker_alloc(struct aeg_encoder *e, struct ft_worker *w)
{
    size_t n = (size_t)e->mb_count;
    int k;

    w->c = (struct aeg_encoder *)calloc(1, sizeof(*w->c));
    w->c0 = (struct aeg_encoder *)calloc(1, sizeof(*w->c0));
    w->mb_dqp0 = (int8_t *)aeg_alloc(n);
    w->mbs = (aeg_mb *)aeg_alloc(n * sizeof(aeg_mb));
    w->mb_encs = (mbenc *)aeg_alloc(n * sizeof(mbenc));
    w->mb_dqp = (int8_t *)aeg_alloc(n);
    w->seed_mv = (int16_t *)aeg_alloc(n * 2 * sizeof(int16_t));
    w->prev_seed = (int16_t *)aeg_alloc(n * 2 * sizeof(int16_t));
    w->seed_cand = (int16_t *)aeg_alloc(n * ME_ROUND_CAND * 4 * sizeof(int16_t));
    w->par_pos = (int16_t *)aeg_alloc(n * ME_ROUND_CAND * 4 * sizeof(int16_t));
    w->par_mv = (int16_t *)aeg_alloc(n * 2 * sizeof(int16_t));
    w->seed_sad = (int32_t *)aeg_alloc(n * ME_ROUND_CAND * sizeof(int32_t));
    w->seed_cost = (int32_t *)aeg_alloc(n * sizeof(int32_t));
    w->par_cost = (int32_t *)aeg_alloc(n * sizeof(int32_t));
    w->lr_scratch = (int *)aeg_alloc(n * 4 * sizeof(int));
    w->lr_mv[0] = (int16_t *)aeg_alloc(n * 2 * sizeof(int16_t));
    w->lr_mv[1] = (int16_t *)aeg_alloc(n * 2 * sizeof(int16_t));
    for (k = 0; k < 3; k++)
        if (aeg_lowres_alloc(&w->lr_pic[k], e->mb_w, e->mb_h))
            w->lr_pic[k].mem = NULL;
    /* (field coding pads every plane vpad rows, twice AEG_PAD) */
    aeg_mc_ctx_init_pad(&w->mcr[0], e->w, e->h, e->stride[0], e->vpad);
    aeg_mc_ctx_init_pad(&w->mcr[1], e->w, e->h, e->stride[0], e->vpad);
    if (!w->c || !w->c0 || !w->mb_dqp0 ||
        !w->mbs || !w->mb_encs || !w->mb_dqp || !w->seed_mv ||
        !w->prev_seed || !w->seed_cand || !w->par_pos || !w->par_mv ||
        !w->seed_sad || !w->seed_cost || !w->par_cost || !w->lr_scratch ||
        !w->lr_mv[0] || !w->lr_mv[1] || !w->lr_pic[0].mem ||
        !w->lr_pic[1].mem || !w->lr_pic[2].mem || !w->mcr[0].buf || !w->mcr[1].buf)
        return AEG_ENOMEM;
    {
        /* enc_alloc_picture() also sets the strides; they are the same */
        int st[3] = { e->stride[0], e->stride[1], e->stride[2] };
        if (enc_alloc_picture(e, &w->pic_mem, w->plane) != AEG_OK)
            return AEG_ENOMEM;
        e->stride[0] = st[0], e->stride[1] = st[1], e->stride[2] = st[2];
    }
    memset(w->par_mv, 0, n * 2 * sizeof(int16_t));
    memset(w->prev_seed, 0, n * 2 * sizeof(int16_t));
    return AEG_OK;
}

static int ft_alloc(struct aeg_encoder *e, int nw)
{
    size_t n = (size_t)e->mb_count;
    int k;

    if (nw > FT_MAXW)
        nw = FT_MAXW;
    e->ft_nw = nw;
    e->ftw = (struct ft_worker *)calloc((size_t)nw, sizeof(*e->ftw));
    e->ftq = (struct ft_entry *)calloc(FT_Q, sizeof(*e->ftq));
    e->fsync = (struct ft_sync *)calloc(1, sizeof(*e->fsync));
    if (!e->ftw || !e->ftq || !e->fsync)
        return AEG_ENOMEM;
    pthread_mutex_init(&e->fsync->mu, NULL);
    pthread_cond_init(&e->fsync->cv, NULL);
    for (k = 0; k < AEG_NPIC; k++)
        e->fsync->rows[k] = e->mb_h;
    for (k = 0; k < nw; k++)
        if (ft_worker_alloc(e, &e->ftw[k]) != AEG_OK)
            return AEG_ENOMEM;
    /* more picture buffers: a reference dropped from the buffer may still be
     * read by a picture in flight (pic_busy) */
    for (k = e->npic; k < AEG_NPIC && k < e->npic_serial + nw + 2; k++) {
        if (slot_alloc(e, k) != AEG_OK)
            return AEG_ENOMEM;
        e->npic = k + 1;
    }
    for (k = 0; k < FT_Q; k++) {
        e->ftq[k].rec_mem = (pixel *)aeg_alloc(pic_bytes(e));
        e->ftq[k].mbqp = (int *)malloc(n * sizeof(int));
        if (!e->ftq[k].rec_mem || !e->ftq[k].mbqp)
            return AEG_ENOMEM;
    }
    e->ft_ret = -1;
    return AEG_OK;
}

/* Start the worker's picture again: a reference it read has restarted, or
 * its weighted-prediction analysis found weights after it ran ahead
 * unweighted.  If it is a reference itself, whatever it published goes, and
 * the pictures that read it restart in turn (generation). */
static void ft_restart(struct ft_worker *w)
{
    struct aeg_encoder *c = w->c;
    struct ft_sync *f = c->fsync;
    uint8_t *out = c->out;
    pixel *wp_mem = c->wp_mem, *wp_plane[3];
    uint8_t *slice_bs[AEG_MAX_SLICES];
    size_t out_cap = c->out_cap, slice_bs_cap[AEG_MAX_SLICES];
    int k;

    pthread_mutex_lock(&f->mu);
    ft_release(c);
    if (w->rec_slot >= 0) {
        /* what it published goes; it is rewritten once every picture that
         * read it has noticed (they stop at their next row) */
        int s = w->rec_slot;
        __atomic_add_fetch(&f->gen[s], 1, __ATOMIC_RELEASE);
        f->stale[s] += f->readers[s];
        f->readers[s] = 0;
        __atomic_store_n(&f->rows[s], 0, __ATOMIC_RELEASE);
        pthread_cond_broadcast(&f->cv);
        while (f->stale[s] > 0)
            pthread_cond_wait(&f->cv, &f->mu);
    }
    f->restarts++;
    pthread_mutex_unlock(&f->mu);
    /* buffers the aborted run may have grown or allocated stay */
    for (k = 0; k < 3; k++)
        wp_plane[k] = c->wp_plane[k];
    for (k = 0; k < AEG_MAX_SLICES; k++) {
        slice_bs[k] = c->slice_bs[k];
        slice_bs_cap[k] = c->slice_bs_cap[k];
    }
    *c = *w->c0;
    c->out = out;
    c->out_cap = out_cap;
    c->wp_mem = wp_mem;
    for (k = 0; k < 3; k++)
        c->wp_plane[k] = wp_plane[k];
    for (k = 0; k < AEG_MAX_SLICES; k++) {
        c->slice_bs[k] = slice_bs[k];
        c->slice_bs_cap[k] = slice_bs_cap[k];
    }
    memcpy(c->mb_dqp, w->mb_dqp0, (size_t)c->mb_count);
}

static void *ft_thread(void *arg)
{
    struct ft_worker *w = (struct ft_worker *)arg;
    int64_t t0 = aeg_now_ns();
    for (;;) {
        w->rc = w->fld ? field_frame_body(w->c, w->slice, w->idr, w->d, w->pts,
                                          &w->pkt, w->t_all)
                       : frame_body(w->c, w->slice, w->idr, w->d, w->pts, &w->pkt,
                                    w->t_all);
        if (w->rc == AEG_OK) {
            /* the last check: every row it read was final */
            int stale;
            pthread_mutex_lock(&w->c->fsync->mu);
            stale = ft_refs_changed(w->c);
            if (!stale)
                ft_release(w->c);
            pthread_mutex_unlock(&w->c->fsync->mu);
            if (!stale)
                break;
        } else if (w->rc != FT_RESTART) {
            break;
        }
        ft_restart(w);
    }
    pthread_mutex_lock(&w->c->fsync->mu);
    ft_release(w->c);
    pthread_mutex_unlock(&w->c->fsync->mu);
    w->ns = aeg_now_ns() - t0;
    __atomic_store_n(&w->done, 1, __ATOMIC_RELEASE);
    return NULL;
}

static struct ft_entry *ft_push(struct aeg_encoder *e)
{
    struct ft_entry *q = &e->ftq[(e->ftq_head + e->ftq_n) % FT_Q];
    e->ftq_n++;
    q->pending = 0;
    q->size = 0;
    return q;
}

static int ft_fill(struct aeg_encoder *e, struct ft_entry *q,
                   const aeg_packet *pkt, const pixel *rec_mem,
                   const aeg_mb *mbs, const struct aeg_encoder *c)
{
    int i;

    if (ensure_buf(&q->buf, &q->cap, pkt->size + 1) != AEG_OK)
        return AEG_ENOMEM;
    memcpy(q->buf, pkt->data, pkt->size);
    q->size = pkt->size;
    q->pts = pkt->pts;
    q->keyframe = pkt->keyframe;
    q->qp = pkt->qp;
    q->type = pkt->type;
    q->qp_avg = pkt->qp_avg;
    q->qp_base = pkt->qp_base;
    memcpy(q->sse, c->pic_sse, sizeof(q->sse));
    q->ssim = c->pic_ssim;
    memcpy(q->rec_mem, rec_mem, pic_bytes(e));
    for (i = 0; i < e->mb_count; i++)
        q->mbqp[i] = mbs[i].qp;
    return AEG_OK;
}

static int ft_join(struct aeg_encoder *e, struct ft_worker *w)
{
    struct aeg_encoder *c = w->c;
    int k, rc;

    if (!w->busy)
        return AEG_OK;
    if (w->busy == 1) {
        int64_t t0 = aeg_now_ns();
        pthread_join(w->th, NULL);
        e->ft_wait_ns += aeg_now_ns() - t0;
        e->ft_worker_ns += w->ns;
    }
    w->busy = 0;
    /* buffers the coder may have grown or allocated */
    w->out = c->out;
    w->out_cap = c->out_cap;
    w->wp_mem = c->wp_mem;
    for (k = 0; k < 3; k++)
        w->wp_plane[k] = c->wp_plane[k];
    for (k = 0; k < AEG_MAX_SLICES; k++) {
        w->slice_bs[k] = c->slice_bs[k];
        w->slice_bs_cap[k] = c->slice_bs_cap[k];
    }
    e->st.bytes += c->st.bytes;
    e->st.frames += c->st.frames;
    e->st.enc_ns += c->st.enc_ns;
    for (k = 0; k < AEG_STAGE_COUNT; k++) {
        e->st.stage_ns[k] += c->st.stage_ns[k];
        e->st.stage_gpu_ns[k] += c->st.stage_gpu_ns[k];
        e->st.stage_cpu_ns[k] += c->st.stage_cpu_ns[k];
    }
    e->stat_mbs += c->stat_mbs;
    e->stat_skipped += c->stat_skipped;
    for (k = 0; k < AEG_MB_TYPE_COUNT; k++)
        e->stat_type[k] += c->stat_type[k];
    w->ent->pending = 0;
    rc = w->rc;
    if (rc == AEG_OK)
        rc = ft_fill(e, w->ent, &w->pkt,
                     w->rec_slot >= 0 ? e->pic_mem[w->rec_slot] : w->pic_mem,
                     w->mbs, c);
    for (k = 0; k < w->nuses; k++)
        e->pic_busy[w->uses[k]]--;
    w->nuses = 0;
    return rc;
}

/* The busy worker launched first (whose packet is the oldest pending). */
static struct ft_worker *ft_oldest(struct aeg_encoder *e)
{
    struct ft_worker *o = NULL;
    int k;
    for (k = 0; k < e->ft_nw; k++)
        if (e->ftw[k].busy && (!o || e->ftw[k].seq < o->seq))
            o = &e->ftw[k];
    return o;
}

static int ft_join_oldest(struct aeg_encoder *e)
{
    struct ft_worker *o = ft_oldest(e);
    if (!o) {
        aeg_set_last_error("frame threads: no picture in flight to wait for");
        return AEG_EINVAL;
    }
    return ft_join(e, o);
}

/* Join the workers that have finished, without waiting. */
static int ft_reap(struct aeg_encoder *e)
{
    int k, rc;
    for (k = 0; k < e->ft_nw; k++) {
        struct ft_worker *w = &e->ftw[k];
        if (w->busy && __atomic_load_n(&w->done, __ATOMIC_ACQUIRE) &&
            (rc = ft_join(e, w)) != AEG_OK)
            return rc;
    }
    return AEG_OK;
}

/* Before input picture in_count overwrites its ring slot: nothing coding
 * from that slot may still be running. */
static int ft_ring_guard(struct aeg_encoder *e)
{
    int slots = in_slots(e), slot = e->in_count % slots, k, rc;
    for (k = 0; k < e->ft_nw; k++) {
        struct ft_worker *w = &e->ftw[k];
        if (w->busy && (w->d % slots == slot ||
                        (w->dref >= 0 && w->dref % slots == slot)) &&
            (rc = ft_join(e, w)) != AEG_OK)
            return rc;
    }
    return AEG_OK;
}

/* Start the job at the head of the queue on a worker.  The prologue runs
 * here, on the encoder itself, exactly as the serial encoder would run it in
 * this call; then the worker gets a copy of the encoder with private scratch
 * and codes the picture (frame_body), while this thread does to the encoder
 * what coding the picture does to it (frame counters, reference marking,
 * the buffer rotation). */
/* field_coding 2 / 4 under frame threads: encode_frame's frame / field
 * choice, made where the pictures are launched.  An anchor's estimate reads
 * the newest anchor's reconstruction (fld_decide), so that anchor's worker
 * is waited for; and a change of structure rewrites the reference buffers'
 * borders and half-sample planes in place (slot_restructure), which the
 * pictures in flight read, so they all are.  B pictures take their anchor's
 * structure and overlap as before. */
static int ft_fld_choose(struct aeg_encoder *e, int slice, int poc, int *fldp)
{
    int fld, i, k, rc;

    e->cur_poc = poc;
    if (slice == AEG_SLICE_B) {
        fld = e->anchor_fld;
    } else {
        int a = -1;
        for (i = 0; i < e->ndpb; i++)
            if (e->dpb[i].anchor && e->dpb[i].fm == 3 &&
                (a < 0 || e->dpb[i].poc > e->dpb[a].poc))
                a = i;
        for (;;) {
            int busy = 0;
            for (k = 0; k < e->ft_nw; k++)
                busy |= a >= 0 && e->ftw[k].busy &&
                        e->ftw[k].rec_slot == e->dpb[a].slot;
            if (!busy)
                break;
            if ((rc = ft_join_oldest(e)) != AEG_OK)
                return rc;
        }
        fld = fld_decide(e);
    }
    for (i = 0; i < e->ndpb; i++)
        if (e->slot_fst[e->dpb[i].slot] != fld)
            break;
    if (i < e->ndpb)
        for (;;) {
            int busy = 0;
            for (k = 0; k < e->ft_nw; k++)
                busy |= e->ftw[k].busy != 0;
            if (!busy)
                break;
            if ((rc = ft_join_oldest(e)) != AEG_OK)
                return rc;
        }
    for (i = 0; i < e->ndpb; i++)
        slot_restructure(e, e->dpb[i].slot, fld);
    e->slot_fst[e->role_cur] = (uint8_t)fld;
    if (e->p.field_coding == 4)
        e->pic.mbaff = e->mbaff = !fld;
    *fldp = fld;
    return AEG_OK;
}

static int ft_launch(struct aeg_encoder *e)
{
    struct ft_worker *w = NULL;
    struct aeg_encoder *c;
    int d, slice, idr, i, k, rc, fld;
    int64_t t_all;

    /* A call pushes at most one packet and hands out the oldest one if it is
     * done, so with at most FT_Q - 1 queued when a call ends -- the oldest
     * finished first whenever the push fills the queue -- the call can always
     * return a packet and the queue never overflows.  (Waiting for a free
     * queue entry instead would deadlock on a queue of finished packets.) */
    for (;;) {
        for (k = 0; k < e->ft_nw && !w; k++)
            if (!e->ftw[k].busy)
                w = &e->ftw[k];
        if (w && (e->ftq_n < FT_Q - 1 || !e->ftq[e->ftq_head].pending))
            break;
        w = NULL;
        if ((rc = ft_join_oldest(e)) != AEG_OK)
            return rc;
    }
    d = e->job_d[0];
    slice = e->job_slice[0];
    idr = e->job_idr[0];
    w->pts = e->in_pts[d % in_slots(e)];
    e->cur_bref = e->job_bref[0];
    for (i = 1; i < e->job_n; i++) {
        e->job_d[i - 1] = e->job_d[i];
        e->job_slice[i - 1] = e->job_slice[i];
        e->job_idr[i - 1] = e->job_idr[i];
        e->job_bref[i - 1] = e->job_bref[i];
    }
    e->job_n--;
    bind_source(e, d);
    t_all = aeg_now_ns();
    fld = e->p.field_coding == 1;
    if ((e->p.field_coding == 2 || e->p.field_coding == 4) &&
        (rc = ft_fld_choose(e, slice, d, &fld)) != AEG_OK)
        return rc;
    if ((rc = frame_prologue(e, &slice, &idr, d)) != AEG_OK)
        return rc;
    if ((e->p.field_coding == 2 || e->p.field_coding == 4) && slice != AEG_SLICE_B)
        e->anchor_fld = fld;
    w->d = d;
    w->dref = e->p.weightp && e->cur_is_p && e->has_ref ? e->ref_poc[0] : -1;
    w->slice = slice;
    w->idr = idr;
    w->t_all = t_all;

    c = w->c;
    /* the workers still coding report their measurements to e->d */
    aeg_dispatch_lock(&e->d);
    *c = *e;
    aeg_dispatch_unlock(&e->d);
    c->ft_on = 0;
    c->ft_worker_copy = 1;
    {
        int nt = slice_thread_count(e);
        c->lane = 1 + (int)(w - e->ftw) * (nt > 1 ? nt : 1);
        aeg_be_lane(&e->d.ops, c->lane, &c->d.ops);
        c->d.root = &e->d;
    }
    c->ftw = NULL;
    c->ftq = NULL;
    c->mbs = w->mbs;
    c->mb_encs = w->mb_encs;
    memcpy(w->mb_dqp, e->mb_dqp, (size_t)e->mb_count);
    c->mb_dqp = w->mb_dqp;
    c->seed_mv = w->seed_mv;
    c->prev_seed = w->prev_seed;
    c->seed_cand = w->seed_cand;
    c->par_pos = w->par_pos;
    c->par_mv = w->par_mv;
    c->seed_sad = w->seed_sad;
    c->seed_cost = w->seed_cost;
    c->par_cost = w->par_cost;
    c->lr_scratch = w->lr_scratch;
    c->lr_mv[0] = w->lr_mv[0];
    c->lr_mv[1] = w->lr_mv[1];
    for (k = 0; k < 3; k++)
        c->lr_pic[k] = w->lr_pic[k];
    c->mcr[0].buf = w->mcr[0].buf;
    c->mcr[1].buf = w->mcr[1].buf;
    for (k = 1; k < 4; k++) {
        c->mcr[0].base[k] = w->mcr[0].base[k];
        c->mcr[1].base[k] = w->mcr[1].base[k];
    }
    c->wp_mem = w->wp_mem;
    for (k = 0; k < 3; k++)
        c->wp_plane[k] = w->wp_plane[k];
    w->nuses = 0;
    /* field pictures: each field has its own lists, from any frame in the
     * DPB */
    w->fld = fld;
    if (w->fld)
        for (k = 0; k < e->ndpb; k++)
            w->uses[w->nuses++] = e->dpb[k].slot;
    else
        for (k = 0; k < AEG_MAX_REFS; k++)
            if (ref_valid(e, k) && e->ref_slot[k] >= 0)
                w->uses[w->nuses++] = e->ref_slot[k];
    if (e->cur_is_ref) {
        /* a reference is reconstructed where the pictures predicting from
         * it read it, and published row by row (rowpost_*) */
        w->rec_slot = e->role_cur;
        w->uses[w->nuses++] = e->role_cur;
        e->slot_rows[e->role_cur] = 0;
        pthread_mutex_lock(&e->fsync->mu);
        e->fsync->rows[e->role_cur] = 0;
        pthread_mutex_unlock(&e->fsync->mu);
        c->slot_rows[e->role_cur] = 0;
    } else {
        /* the next picture is reconstructed into the same buffer */
        w->rec_slot = -1;
        for (k = 0; k < 3; k++) {
            c->pic_plane[c->role_cur][k] = w->plane[k];
            c->recp[k] = w->plane[k];
            c->pic.plane[k] = w->plane[k];
        }
    }
    for (k = 0; k < w->nuses; k++)
        e->pic_busy[w->uses[k]]++;
    for (k = 0; k < AEG_MAX_SLICES; k++) {
        c->slice_bs[k] = w->slice_bs[k];
        c->slice_bs_cap[k] = w->slice_bs_cap[k];
    }
    c->out = w->out;
    c->out_cap = w->out_cap;
    c->n_threads = 1;
    /* statistics are merged as deltas after the join */
    c->st.bytes = 0;
    c->st.frames = 0;
    c->st.enc_ns = 0;
    memset(c->st.stage_ns, 0, sizeof(c->st.stage_ns));
    memset(c->st.stage_gpu_ns, 0, sizeof(c->st.stage_gpu_ns));
    memset(c->st.stage_cpu_ns, 0, sizeof(c->st.stage_cpu_ns));
    c->stat_mbs = 0;
    c->stat_skipped = 0;
    memset(c->stat_type, 0, sizeof(c->stat_type));

    /* what frame_body() does to the encoder, done here */
    if (idr)
        e->frame_num = 0;
    memset(e->seed_mv, 0, (size_t)e->mb_count * 2 * sizeof(int16_t));
    e->par_valid = 0;
    e->par_full = 0;
    e->frame_count++;
    e->enc_pos++;
    if (e->cur_is_ref) {
        if (w->fld)
            memset(e->slot_ref_poc[e->role_cur], 0, sizeof(e->slot_ref_poc[0]));
        else
            memcpy(e->slot_ref_poc[e->role_cur], e->ref_poc, sizeof(e->ref_poc));
        e->slot_col_ok[e->role_cur] = 1;
    }
    /* its fields' half-resolution copies: built behind the rows it
     * publishes, like the rest */
    e->slot_lfok[e->role_cur] = w->fld && e->cur_is_ref &&
                                e->slot_lowf[e->role_cur][0].mem;
    if (e->mbaff || e->p.field_coding == 4)
        /* its field copy: built behind the rows it publishes (rowpost_fshadow
         * -- the pictures reading it wait for those rows); field pictures of
         * field_coding 4 build theirs at the end, and only pictures that
         * restructure (wait for them) read it */
        e->slot_fok[e->role_cur] = e->cur_is_ref && e->slot_fmem[e->role_cur];
    {
        /* (field pictures mark their reference frame as field_mark does,
         * which is this without the frame's marking operations) */
        int mmco = e->refctl.n_mmco;
        if (w->fld)
            e->refctl.n_mmco = 0;
        picture_rotate(e, e->cur_is_ref, idr);
        e->refctl.n_mmco = mmco;
    }
    while (e->role_cur < 0) {
        /* every buffer is a reference or still read by a picture in flight */
        if ((rc = ft_join_oldest(e)) != AEG_OK)
            return rc;
        e->role_cur = pic_free(e);
        bind_pictures(e);
    }
    e->pic.plane[0] = e->recp[0];
    e->pic.plane[1] = e->recp[1];
    e->pic.plane[2] = e->recp[2];
    if (idr)
        e->idr_pic_id = (e->idr_pic_id + 1) & 0xffff;
    aeg_dispatch_update(&e->d);

    *w->c0 = *c;
    memcpy(w->mb_dqp0, w->mb_dqp, (size_t)e->mb_count);
    w->ent = ft_push(e);
    w->ent->pending = 1;
    w->seq = e->ft_seq++;
    w->done = 0;
    w->busy = 1;
    if (pthread_create(&w->th, NULL, ft_thread, w) != 0) {
        ft_thread(w);
        w->busy = 2;
    }
    return AEG_OK;
}

/* One call's worth of frame-thread coding, then the oldest packet if it is
 * done.
 *
 * While input still arrives, a call starts exactly the job the serial encoder
 * would code in it: what the lookahead knows when a picture's prologue runs
 * (its macroblock-tree offsets are refined as more input comes in) is part of
 * the result, so no picture may start in a different call than it would
 * serially.  Its packet comes out in a later call once it is done.  When
 * flushing nothing changes between calls any more, and every queued picture
 * can start as soon as a worker is free. */
static int ft_run(struct aeg_encoder *e, aeg_packet *pkt, int flushing)
{
    struct ft_entry *q;
    int rc;

    e->ft_ret = -1;             /* the previous call's packet is released */
    if ((rc = ft_reap(e)) != AEG_OK)
        return rc;
    if (!flushing) {
        if (e->job_n && (rc = ft_launch(e)) != AEG_OK)
            return rc;
        if (!e->ftq_n || e->ftq[e->ftq_head].pending)
            return AEG_EAGAIN;
    } else {
        for (;;) {
            if (e->ftq_n && !e->ftq[e->ftq_head].pending)
                break;
            if (!e->job_n)
                schedule(e);
            if (e->job_n) {
                if ((rc = ft_launch(e)) != AEG_OK)
                    return rc;
                continue;
            }
            if (!e->ftq_n)
                return AEG_EAGAIN;
            if ((rc = ft_join_oldest(e)) != AEG_OK)
                return rc;
        }
    }
    q = &e->ftq[e->ftq_head];
    if (pkt) {
        pkt->data = q->buf;
        pkt->size = q->size;
        pkt->pts = q->pts;
        pkt->keyframe = q->keyframe;
        pkt->qp = q->qp;
        pkt->type = q->type;
        pkt->qp_avg = q->qp_avg;
        pkt->qp_base = q->qp_base;
    }
    qual_account(e, q->sse, q->ssim);
    e->ft_ret = e->ftq_head;
    e->ftq_head = (e->ftq_head + 1) % FT_Q;
    e->ftq_n--;
    return AEG_OK;
}

static int encode_core(struct aeg_encoder *e, const aeg_src *in, int64_t pts,
                       aeg_packet *pkt)
{
    int is_idr;

    if (!e->frame_count) {
        /* The reference samples of the top/left picture edge do not exist;
         * filling the padding with the mid grey keeps the prediction of the
         * first macroblock defined (and matches the DC-128 fallback). */
        int p, y;
        for (p = 0; p < 3; p++) {
            int pw = p ? e->cw : e->w, ph = p ? e->ch : e->h;
            aeg_pixel_fill(e->recp[p] - AEG_PAD * e->stride[p] - AEG_PAD, PIXEL_MID,
                           (size_t)e->stride[p] * AEG_PAD);
            for (y = 0; y < ph + AEG_PAD; y++)
                aeg_pixel_fill(e->recp[p] + y * e->stride[p] - AEG_PAD, PIXEL_MID,
                               AEG_PAD);
            (void)pw;
        }
    }
    if (!e->p.bframes) {
        copy_yuv(e, e->srcp, in);
        is_idr = idr_due(e, e->frame_count);
        if (e->crf_low[0].mem) {
            /* CRF's complexity and the VBV's size estimate: this picture
             * against the last one (or alone, intra), from the lowres
             * pictures the B-picture lookahead would have used */
            aeg_lowres *cur = &e->crf_low[e->frame_count & 1];
            aeg_lowres_build(cur, e->srcp[0], e->stride[0], e->w, e->h);
            e->crf_last = e->frame_count && !is_idr
                ? aeg_lowres_cost(cur, &e->crf_low[(e->frame_count - 1) & 1],
                                  NULL, NULL, e->la_scratch)
                : -1;
            e->cur_satd = e->crf_last;
            if ((e->vbv_active || e->abr_on) && e->crf_last < 0) {
                int i;
                aeg_lowres_intra(cur, e->la_scratch);
                for (e->cur_satd = 0, i = 0; i < e->mb_count; i++)
                    e->cur_satd += e->la_scratch[i];
            }
        }
        return encode_frame(e, is_idr ? AEG_SLICE_I : AEG_SLICE_P, is_idr,
                            e->frame_count, pts, pkt);
    }
    /* With B pictures the pictures are coded out of display order: the anchor
     * that follows a group of B pictures has to be in the bitstream before
     * them, because it is their list 1 reference.  So the input goes into the
     * ring first and the scheduler decides what can be coded now. */
    if (e->ft_on) {
        int rc = ft_ring_guard(e);
        if (rc != AEG_OK)
            return rc;
    }
    push_input(e, in, pts);
    schedule(e);
    if (e->ft_on)
        return ft_run(e, pkt, 0);
    if (!e->job_n)
        return AEG_EAGAIN;
    return run_job(e, pkt);
}

/* A packet left the encoder: it decodes at the display time of the picture
 * with its index in coding order, held back by the reorder depth. */
static void set_dts(struct aeg_encoder *e, aeg_packet *pkt)
{
    int64_t t = pkt->pts;
    if (e->dts_n) {
        t = e->dts_q[e->dts_head];
        e->dts_head = (e->dts_head + 1) % e->dts_cap;
        e->dts_n--;
    }
    pkt->dts = t - (int64_t)llround(e->p.bframes * e->dts_dur);
}

static int dts_push(struct aeg_encoder *e, int64_t pts)
{
    if (e->dts_n == e->dts_cap) {
        /* never more pictures in flight than the input ring holds; grow
         * anyway rather than lose a timestamp */
        int cap = e->dts_cap ? 2 * e->dts_cap : 64, i;
        int64_t *q = (int64_t *)malloc(sizeof(*q) * (size_t)cap);
        if (!q)
            return AEG_ENOMEM;
        for (i = 0; i < e->dts_n; i++)
            q[i] = e->dts_q[(e->dts_head + i) % e->dts_cap];
        free(e->dts_q);
        e->dts_q = q;
        e->dts_cap = cap;
        e->dts_head = 0;
    }
    e->dts_q[(e->dts_head + e->dts_n) % e->dts_cap] = pts;
    e->dts_n++;
    return AEG_OK;
}

static int encode_src(struct aeg_encoder *e, const aeg_src *in, int64_t pts,
                      aeg_packet *pkt)
{
    int rc;
    if (dts_push(e, pts) != AEG_OK) {
        aeg_set_last_error("out of memory");
        return AEG_ENOMEM;
    }
    rc = encode_core(e, in, pts, pkt);
    if (rc == AEG_OK && pkt && pkt->data)
        set_dts(e, pkt);
    else if (rc != AEG_OK && rc != AEG_EAGAIN && e->dts_n) {
        /* the picture was refused: forget its timestamp again */
        e->dts_n--;
    }
    return rc;
}

int aeg_encoder_encode(aeg_encoder *enc, const uint8_t *yuv420p, size_t size,
                       int64_t pts, aeg_packet *pkt)
{
    struct aeg_encoder *e = (struct aeg_encoder *)enc;
    size_t ls, cs;
    aeg_src in;

    if (!e || !yuv420p || !pkt) {
        aeg_set_last_error("aeg_encoder_encode: NULL argument");
        return AEG_EINVAL;
    }
    ls = (size_t)e->vw * e->vh;
    cs = (size_t)e->vcw * e->vch;
    if (size < ls + 2 * cs) {
        aeg_set_last_error("input buffer too small");
        return AEG_EINVAL;
    }
#if BIT_DEPTH == 8
    in.plane[0] = yuv420p;
    in.plane[1] = yuv420p + ls;
    in.plane[2] = yuv420p + ls + cs;
    in.stride[0] = e->vw;
    in.stride[1] = in.stride[2] = e->vcw;
    return encode_src(e, &in, pts, pkt);
#else
    {
        /* 8-bit samples into a high-depth encoder: scaled up exactly */
        aeg_image img;
        (void)in;
        img.csp = AEG_CSP_I420;
        img.depth = 8;
        img.plane[0] = yuv420p;
        img.plane[1] = yuv420p + ls;
        img.plane[2] = yuv420p + ls + cs;
        img.stride[0] = e->vw;
        img.stride[1] = img.stride[2] = e->vcw;
        return aeg_encoder_encode_image(enc, &img, pts, pkt);
    }
#endif
}

int aeg_encoder_encode_planes(aeg_encoder *enc, const uint8_t *const plane[3],
                              const int stride[3], int64_t pts,
                              aeg_packet *pkt)
{
    struct aeg_encoder *e = (struct aeg_encoder *)enc;
    aeg_src in;
    int i;

    if (!e || !plane || !stride || !pkt) {
        aeg_set_last_error("aeg_encoder_encode_planes: NULL argument");
        return AEG_EINVAL;
    }
    for (i = 0; i < 3; i++) {
        if (!plane[i]) {
            aeg_set_last_error("aeg_encoder_encode_planes: NULL plane");
            return AEG_EINVAL;
        }
        if (stride[i] < (i ? e->vcw : e->vw)) {
            aeg_set_last_error("aeg_encoder_encode_planes: stride smaller "
                               "than the plane's width");
            return AEG_EINVAL;
        }
#if BIT_DEPTH == 8
        in.plane[i] = plane[i];
        in.stride[i] = stride[i];
#endif
    }
#if BIT_DEPTH == 8
    return encode_src(e, &in, pts, pkt);
#else
    {
        aeg_image img;
        (void)in;
        img.csp = AEG_CSP_I420;
        img.depth = 8;
        for (i = 0; i < 3; i++) {
            img.plane[i] = plane[i];
            img.stride[i] = stride[i];
        }
        return aeg_encoder_encode_image(enc, &img, pts, pkt);
    }
#endif
}

#if BIT_DEPTH > 8
/* Whether a w x h plane of 16-bit samples (stride in bytes) holds a sample
 * above PIXEL_MAX: the OR of all of them, PIXEL_MAX being 2^n - 1. */
static int plane_over_max(const void *plane, int stride, int w, int h)
{
    unsigned m = 0;
    int x, y;
    for (y = 0; y < h; y++) {
        const uint16_t *s = (const uint16_t *)((const uint8_t *)plane +
                                               (size_t)y * stride);
        for (x = 0; x < w; x++)
            m |= s[x];
    }
    return m > PIXEL_MAX;
}
#endif

int aeg_encoder_encode_image(aeg_encoder *enc, const aeg_image *img,
                             int64_t pts, aeg_packet *pkt)
{
    struct aeg_encoder *e = (struct aeg_encoder *)enc;
    aeg_src in;
    int i, rc;

    if (!e || !img || !pkt) {
        aeg_set_last_error("aeg_encoder_encode_image: NULL argument");
        return AEG_EINVAL;
    }
    if ((rc = aeg_image_check(img, e->vw, e->vh)) != AEG_OK)
        return rc;
    if (img->depth == BIT_DEPTH &&
        (img->csp == AEG_CSP_I420 || img->csp == AEG_CSP_YV12) &&
        (img->stride[0] | img->stride[1] | img->stride[2]) % sizeof(pixel) == 0 &&
        ((uintptr_t)img->plane[0] | (uintptr_t)img->plane[1] |
         (uintptr_t)img->plane[2]) % sizeof(pixel) == 0
#if BIT_DEPTH > 8
        /* a sample above PIXEL_MAX (the caller's error) takes the
         * conversion, which clips it: coded as it is it would make the
         * residual, and the levels, overflow */
        && !plane_over_max(img->plane[0], img->stride[0], e->vw, e->vh)
        && !plane_over_max(img->plane[1], img->stride[1], e->vcw, e->vch)
        && !plane_over_max(img->plane[2], img->stride[2], e->vcw, e->vch)
#endif
        ) {
        /* the encoder's own format: no conversion */
        int u = img->csp == AEG_CSP_YV12 ? 2 : 1;
        in.plane[0] = (const pixel *)img->plane[0];
        in.plane[1] = (const pixel *)img->plane[u];
        in.plane[2] = (const pixel *)img->plane[3 - u];
        in.stride[0] = img->stride[0] / (int)sizeof(pixel);
        in.stride[1] = img->stride[u] / (int)sizeof(pixel);
        in.stride[2] = img->stride[3 - u] / (int)sizeof(pixel);
        return encode_src(e, &in, pts, pkt);
    }
    if (!e->cvt_pic) {
        e->cvt_pic = (pixel *)malloc(((size_t)e->vw * e->vh +
                                      2 * (size_t)e->vcw * e->vch) * sizeof(pixel));
        if (!e->cvt_pic) {
            aeg_set_last_error("out of memory");
            return AEG_ENOMEM;
        }
    }
    {
        pixel *dst[3];
        int st[3];
        dst[0] = e->cvt_pic;
        dst[1] = dst[0] + (size_t)e->vw * e->vh;
        dst[2] = dst[1] + (size_t)e->vcw * e->vch;
        st[0] = e->vw;
        st[1] = st[2] = e->vcw;
        if (aeg_input_convert(&e->cvt, img, e->vw, e->vh, e->p.interlaced,
                              dst, st) != AEG_OK) {
            aeg_set_last_error("out of memory");
            return AEG_ENOMEM;
        }
        for (i = 0; i < 3; i++) {
            in.plane[i] = dst[i];
            in.stride[i] = st[i];
        }
    }
    return encode_src(e, &in, pts, pkt);
}

int aeg_encoder_headers(aeg_encoder *enc, aeg_packet *pkt)
{
    struct aeg_encoder *e = (struct aeg_encoder *)enc;
    if (!e || !pkt) {
        aeg_set_last_error("aeg_encoder_headers: NULL argument");
        return AEG_EINVAL;
    }
    memset(pkt, 0, sizeof(*pkt));
    pkt->data = e->hdr;
    pkt->size = e->hdr_len;
    pkt->keyframe = 1;
    return AEG_OK;
}

int aeg_encoder_get_params(aeg_encoder *enc, aeg_params *out)
{
    struct aeg_encoder *e = (struct aeg_encoder *)enc;
    if (!e || !out) {
        aeg_set_last_error("aeg_encoder_get_params: NULL argument");
        return AEG_EINVAL;
    }
    *out = e->p;
    out->level_idc = e->level_out;
    out->slices = e->n_slices;
    out->frame_threads = e->ft_on ? e->ft_nw : 1;   /* the workers in use */
    return AEG_OK;
}

static int flush_core(struct aeg_encoder *e, aeg_packet *pkt);

int aeg_encoder_flush(aeg_encoder *enc, aeg_packet *pkt)
{
    struct aeg_encoder *e = (struct aeg_encoder *)enc;
    int rc;
    if (!e) {
        if (pkt)
            pkt->data = NULL, pkt->size = 0;
        return AEG_EOF;
    }
    rc = flush_core(e, pkt);
    if (rc == AEG_OK && pkt && pkt->data)
        set_dts(e, pkt);
    return rc;
}

static int flush_core(struct aeg_encoder *e, aeg_packet *pkt)
{
    if (pkt)
        pkt->data = NULL, pkt->size = 0;
    if (!e->p.bframes)
        return AEG_EOF;
    if (e->ft_on) {
        int rc;
        if (!e->job_n) {
            e->flushing = 1;
            schedule(e);
        }
        rc = ft_run(e, pkt, 1);
        return rc == AEG_EAGAIN ? AEG_EOF : rc;
    }
    if (e->job_n)
        return run_job(e, pkt);
    /* No more input: the groups the lookahead was waiting on can go now, and
     * the incomplete one at the end goes out as P pictures (schedule()). */
    e->flushing = 1;
    schedule(e);
    if (e->job_n)
        return run_job(e, pkt);
    return AEG_EOF;
}

#if BIT_DEPTH == 8                  /* public, depth-independent */
void aeg_packet_free(aeg_packet *pkt)
{
    if (pkt)
        pkt->data = NULL, pkt->size = 0;
}
#endif

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
    out->vbv_recodes = e->vbv_recodes;
    out->bframes = e->p.bframes;
    out->scene_cuts = e->scene_cuts;
    out->macroblocks = e->stat_mbs;
    out->skipped = e->stat_skipped;
    for (i = 0; i < AEG_MB_TYPE_COUNT; i++)
        out->mb_types[i] = e->stat_type[i];
    for (i = 0; i < AEG_STAGE_COUNT; i++) {
        /* on the device: the dispatcher has it there now, or some of its
         * work ran there (a forced batch path, or an earlier placement) */
        out->stage_use_gpu[i] = e->d.use_gpu[i] || e->st.stage_gpu_ns[i] > 0;
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
    if (e->ft_on && e->ft_ret >= 0) {
        /* frame threads: the copy kept with the packet just handed out */
        const struct ft_entry *q = &e->ftq[e->ft_ret];
        /* (laid out as enc_alloc_picture does: vpad rows above and below) */
        const size_t vp = (size_t)e->vpad;
        size_t ls = (size_t)e->stride[0];
        size_t lh = (size_t)e->h + 2 * vp;
        size_t cs = (size_t)e->stride[1];
        size_t ch = (size_t)(e->h + 1) / 2 + 2 * vp;
        for (i = 0; i < 3; i++) {
            if (plane)
                plane[i] = (const uint8_t *)(q->rec_mem +
                                             (i ? ls * lh + (size_t)(i - 1) * cs * ch +
                                                  vp * cs + AEG_PAD
                                                : vp * ls + AEG_PAD));
            if (stride)
                stride[i] = e->stride[i] * (int)sizeof(pixel);
        }
        if (mb_qp)
            for (i = 0; i < e->mb_count; i++)
                mb_qp[i] = q->mbqp[i];
        return AEG_OK;
    }
    for (i = 0; i < 3; i++) {
        if (plane)
            /* the frame that was just encoded: the picture it was coded into,
             * which is not necessarily the newest reference any more */
            plane[i] = (const uint8_t *)e->pic_plane[e->role_last][i];
        if (stride)
            stride[i] = e->stride[i] * (int)sizeof(pixel);
    }
    if (mb_qp) {
        int n = e->mb_count;
        for (i = 0; i < n; i++)
            mb_qp[i] = e->mbs[i].qp;
    }
    return AEG_OK;
}

/* The QP a picture of the given type is coded at when its model QP is base:
 * the offsets the prologue of aeg_encoder_encode applies, without the VBV. */
static double plan_type_qp(const struct aeg_encoder *e, double base, int type)
{
    double q = base;

    if (type == 0 && e->p.i_qp_offset > 0)
        q -= e->p.i_qp_offset;
    if ((type == 2 || type == 3) && e->p.b_qp_offset) {
        int off = e->p.b_qp_offset, cur = (int)floor(q);
        if (type == 3)
            off = off * bref_qp_num() / 2;
        if (cur < 0) cur = 0;
        if (cur < B_QP_FADE)
            off = off * cur / B_QP_FADE;
        q += off;
    }
    if (q < e->p.qp_min) q = e->p.qp_min;
    if (q > e->p.qp_max) q = e->p.qp_max;
    return q;
}

/* The base (model) QP of every picture for rate factor rf, and the bits the
 * plan expects in total.  P pictures follow their blurred complexity; an
 * intra picture takes the level of the P pictures after it (the ones that
 * predict from it -- x264's ip factor is then the I offset) and a B picture
 * the level of the anchor after it, which is the one coded just before it. */
static double plan_eval(const struct aeg_encoder *e, const aeg_pass1_frame *f,
                        int n, const double *blur, const unsigned char *held,
                        double rf, double *base, double *bits)
{
    double qc = e->crf_qc, total = 0.0;
    int i, j;

    for (i = 0; i < n; i++) {
        base[i] = -1.0;
        if (f[i].bits >= 0 && f[i].type == 1 && !held[i]) {
            double qs = (qc > 0.0 ? pow(blur[i], qc) : 1.0) / rf;
            base[i] = qs2qp(qs);
            if (base[i] < QP_MIN) base[i] = QP_MIN;
            if (base[i] > 51.0) base[i] = 51.0;
        }
    }
    /* a held P picture keeps the level of the P picture before it */
    for (i = 0; i < n; i++) {
        if (f[i].bits < 0 || f[i].type != 1 || !held[i])
            continue;
        for (j = i - 1; j >= 0 && !(f[j].type == 1 && base[j] >= 0.0); j--)
            ;
        if (j < 0)
            for (j = i + 1; j < n && !(f[j].type == 1 && base[j] >= 0.0 &&
                                       !held[j]); j++)
                ;
        base[i] = j >= 0 && j < n ? base[j] : qs2qp(1.0 / rf);
    }
    for (i = 0; i < n; i++) {
        double sum = 0.0;
        int cnt = 0;
        if (f[i].bits < 0 || f[i].type != 0)
            continue;
        for (j = i + 1; j < n && cnt < 3 && f[j].type != 0; j++)
            if (base[j] >= 0.0 && f[j].type == 1)
                sum += base[j], cnt++;
        for (j = i - 1; j >= 0 && cnt == 0; j--)
            if (base[j] >= 0.0 && f[j].type == 1)
                sum += base[j], cnt++;
        if (cnt) {
            base[i] = sum / cnt;
        } else {                       /* nothing but intra pictures */
            double qs = (qc > 0.0 ? pow(blur[i], qc) : 1.0) / rf;
            base[i] = qs2qp(qs);
            if (base[i] < QP_MIN) base[i] = QP_MIN;
            if (base[i] > 51.0) base[i] = 51.0;
        }
    }
    for (i = 0; i < n; i++) {
        if (f[i].bits < 0 || (f[i].type != 2 && f[i].type != 3))
            continue;
        for (j = i + 1; j < n && !(f[j].bits >= 0 && f[j].type <= 1); j++)
            ;
        if (j >= n)
            for (j = i - 1; j >= 0 && !(f[j].bits >= 0 && f[j].type <= 1); j--)
                ;
        base[i] = j >= 0 && j < n ? base[j] : qs2qp(1.0 / rf);
    }
    for (i = 0; i < n; i++) {
        double q1, q2;
        if (f[i].bits < 0) {
            bits[i] = -1.0;
            continue;
        }
        q1 = f[i].qp < QP_MIN ? QP_MIN : f[i].qp > 51.0 ? 51.0 : f[i].qp;
        q2 = plan_type_qp(e, base[i], f[i].type);
        /* with the tree the picture is coded at its QP plus the tree's
         * mean offset, which the first pass's tree measured */
        if (e->la_depth)
            q2 += f[i].qp - f[i].qp_base;
        bits[i] = f[i].type == 1 && held[i] && q2 >= q1
                ? (double)f[i].bits     /* a skip run does not scale */
                : (double)f[i].bits * qp2qs(q1) / qp2qs(q2);
        total += bits[i];
    }
    return total;
}

int aeg_encoder_set_pass1(aeg_encoder *enc, const aeg_pass1_frame *f, int n)
{
    struct aeg_encoder *e = (struct aeg_encoder *)enc;
    double *cplx, *blur, *base, *bits, target, lo, hi, total = 0.0, qsum = 0.0;
    unsigned char *held;
    int i, j, it, pass, cnt = 0;

    if (!e || !f || n <= 0 || !e->abr_on || e->frame_count > 0)
        return AEG_EINVAL;
    {
        /* what a first pass of this encoder can have written: anything
         * else is a damaged or foreign file, and planning from it codes a
         * stream at whatever rate the garbage implies (QP 999: +85%) */
        double sum = 0.0;
        int nb = 0;
        for (i = 0; i < n; i++) {
            if (f[i].bits < 0)
                continue;
            if (f[i].type < 0 || f[i].type > 3 ||
                !(f[i].qp >= QP_MIN && f[i].qp <= 51.0) ||
                !(f[i].qp_base >= QP_MIN && f[i].qp_base <= 51.0)) {
                aeg_set_last_error("pass 1 statistics: a picture's type or QP "
                                   "is out of range");
                return AEG_EINVAL;
            }
            sum += (double)f[i].bits;
            nb += f[i].type >= 2;
        }
        if (!(sum > 0.0)) {
            aeg_set_last_error("pass 1 statistics: no bits measured");
            return AEG_EINVAL;
        }
        /* x264 refuses a second pass with other B-picture settings; one
         * without B pictures cannot code the B pictures planned for */
        if (nb > 0 && e->p.bframes <= 0) {
            aeg_set_last_error("pass 1 statistics have B pictures and this "
                               "pass has none: use the same --bframes");
            return AEG_EINVAL;
        }
    }
    cplx = (double *)aeg_alloc((size_t)n * 4 * sizeof(double));
    if (!cplx)
        return AEG_ENOMEM;
    blur = cplx + n;
    base = blur + n;
    bits = base + n;
    for (i = 0; i < n; i++) {
        double q = f[i].qp < QP_MIN ? QP_MIN : f[i].qp > 51.0 ? 51.0 : f[i].qp;
        cplx[i] = f[i].bits >= 0
                ? (double)(f[i].bits > 0 ? f[i].bits : 1) * qp2qs(q) : -1.0;
        if (f[i].bits >= 0)
            cnt++;
    }
    /* Complexity blurred over the neighbouring P pictures (a Gaussian of
     * x264's width, in pictures), stopping at an intra picture: what comes
     * after a cut has nothing to do with what came before it.  Twice: the
     * first blur finds the held pictures (see ABR_HELD), which the second
     * leaves out -- their bits say how little changed, not how hard the
     * pictures around them are. */
    held = (unsigned char *)aeg_alloc((size_t)n);
    if (!held) {
        aeg_free(cplx);
        return AEG_ENOMEM;
    }
    memset(held, 0, (size_t)n);
    for (pass = 0; pass < 2; pass++) {
        for (i = 0; i < n; i++) {
            double ws = 1.0, cs = cplx[i];
            int dir;
            if (f[i].bits < 0 || f[i].type != 1) {
                if (!pass)
                    blur[i] = cplx[i];
                continue;
            }
            if (pass && held[i])
                ws = cs = 0.0;
            for (dir = -1; dir <= 1; dir += 2)
                for (j = i + dir; j >= 0 && j < n && abs(j - i) <= 40;
                     j += dir) {
                    double w;
                    if (f[j].bits < 0 || f[j].type == 2 || f[j].type == 3 ||
                        (pass && held[j]))
                        continue;
                    if (f[j].type == 0)
                        break;
                    w = exp(-(double)(j - i) * (j - i) / 200.0);
                    ws += w;
                    cs += w * cplx[j];
                }
            if (!pass)
                held[i] = cplx[i] * ABR_HELD < cs / ws;
            else
                blur[i] = ws > 0.0 ? cs / ws : cplx[i];
        }
    }
    /* the rate factor whose plan meets the target, by bisection on its log
     * (the expected total grows with it) */
    target = e->abr_bpf * cnt;
    lo = log(1e-9);
    hi = log(1e9);
    for (it = 0; it < 80; it++) {
        double mid = 0.5 * (lo + hi);
        if (plan_eval(e, f, n, blur, held, exp(mid), base, bits) > target)
            hi = mid;
        else
            lo = mid;
    }
    plan_eval(e, f, n, blur, held, exp(0.5 * (lo + hi)), base, bits);
    aeg_free(held);
    aeg_free(e->plan_q);
    aeg_free(e->plan_bits);
    aeg_free(e->plan_cls);
    e->plan_q = (double *)aeg_alloc((size_t)n * sizeof(double));
    e->plan_bits = (double *)aeg_alloc((size_t)n * sizeof(double));
    e->plan_cls = (unsigned char *)aeg_alloc((size_t)n);
    if (!e->plan_q || !e->plan_bits || !e->plan_cls) {
        aeg_free(e->plan_q);
        aeg_free(e->plan_bits);
        aeg_free(e->plan_cls);
        e->plan_q = e->plan_bits = NULL;
        e->plan_cls = NULL;
        e->plan_n = 0;
        aeg_free(cplx);
        return AEG_ENOMEM;   /* held is freed above */
    }
    for (i = 0; i < n; i++) {
        e->plan_q[i] = f[i].bits >= 0 ? base[i] : -1.0;
        e->plan_bits[i] = bits[i];
        e->plan_cls[i] = f[i].type == 0 ? 0 : f[i].type == 1 ? 1 : 2;
        if (f[i].bits >= 0) {
            total += bits[i];
            qsum += plan_type_qp(e, base[i], f[i].type);
        }
    }
    e->plan_n = n;
    e->plan_total = total;
    e->plan_exp_sum = 0.0;
    aeg_log(AEG_LOG_INFO, "aeg: two-pass plan: %d pictures, %.1f kb/s expected "
               "(target %d), average QP %.2f", cnt,
            cnt ? total * e->abr_fps / cnt / 1000.0 : 0.0,
            e->p.bitrate_kbps, cnt ? qsum / cnt : 0.0);
    aeg_free(cplx);
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
    if (!e || qp < QP_MIN || qp > 51)
        return AEG_EINVAL;
    e->p.qp = qp;
    return AEG_OK;
}

void aeg_encoder_close(aeg_encoder *enc)
{
    struct aeg_encoder *e = (struct aeg_encoder *)enc;
    if (!e)
        return;
    if (aeg_ph_on)
        aeg_ph_report();
    if (getenv("AEG_LATIME"))
        fprintf(stderr, "schedule total %.0f ms, main waited %.0f ms, worker busy %.0f ms, "
                "workers waited for rows %.0f ms, restarts %lld\n", e->sched_ns / 1e6,
                e->ft_wait_ns / 1e6, e->ft_worker_ns / 1e6,
                e->fsync ? e->fsync->wait_ns / 1e6 : 0.0,
                e->fsync ? (long long)e->fsync->restarts : 0LL);
    ft_free(e);
    free(e->dts_q);
    aeg_input_cvt_free(&e->cvt);
    free(e->cvt_pic);
    aeg_dispatch_destroy(&e->d);
    for (int k = 0; k < AEG_MAX_REFS; k++)
        aeg_mc_ctx_free(&e->mcr[k]);
    aeg_free(e->mb_encs);
    aeg_free(e->seed_mv);
    aeg_free(e->lr_scratch);
    aeg_free(e->lr_mv[0]);
    aeg_free(e->lr_mv[1]);
    {
        int k;
        for (k = 0; k < 3; k++)
            if (e->lr_pic[k].mem)
                aeg_lowres_free(&e->lr_pic[k]);
    }
    aeg_free(e->seed_cand);
    aeg_free(e->prev_seed);
    aeg_free(e->seed_sad);
    aeg_free(e->seed_cost);
    aeg_free(e->par_pos);
    aeg_free(e->par_mv);
    aeg_free(e->par_cost);
    aeg_free(e->mb_dqp);
    aeg_free(e->crf_cplx);
    aeg_free(e->plan_q);
    aeg_free(e->plan_bits);
    aeg_free(e->plan_cls);
    aeg_free(e->vbv_satd);
    aeg_lowres_free(&e->crf_low[0]);
    aeg_lowres_free(&e->crf_low[1]);
    {
        int i;
        for (i = 0; i < AEG_RING; i++) {
            aeg_free(e->mbt[i].intra);
            aeg_free(e->mbt[i].inter);
            aeg_free(e->mbt[i].mv);
            aeg_free(e->mbt[i].kind);
            aeg_free(e->mbt[i].prop);
            aeg_free(e->mbt[i].qp);
        }
    }
    aeg_free(e->mbs);
    aeg_free(e->in_pts);
    {
        int i;
        for (i = 0; i < (int)(sizeof(e->in_mem) / sizeof(e->in_mem[0])); i++)
            aeg_free(e->in_mem[i]);
        for (i = 0; i < AEG_RING; i++) {
            int j;
            aeg_lowres_free(&e->in_low[i]);
            aeg_free(e->la_icost[i]);
            for (j = 0; j < LA_SF; j++)
                aeg_free(e->la_sf[i][j].f);
        }
        aeg_free(e->la_scratch);
        for (i = 0; i < AEG_NPIC; i++) {
            aeg_free(e->pic_mem[i]);
            aeg_mc_ctx_free(&e->slot_mc[i]);
            aeg_free(e->slot_fmem[i]);
            aeg_mc_ctx_free(&e->slot_fmc[i]);
            aeg_lowres_free(&e->slot_low[i]);
            aeg_lowres_free(&e->slot_lowf[i][0]);
            aeg_lowres_free(&e->slot_lowf[i][1]);
            aeg_free(e->slot_mbs[i]);
        }
        for (i = 0; i < AEG_MAX_REFS; i++) {
            /* (the AEG_POISON debug copies) */
            aeg_free(e->pz_mem[i]);
            aeg_mc_ctx_free(&e->pz_mc[i]);
            aeg_lowres_free(&e->pz_low[i]);
            aeg_free(e->pzf_mem[i]);
            aeg_mc_ctx_free(&e->pzf_mc[i]);
            aeg_free(e->pzl_mem[i]);
            aeg_mc_ctx_free(&e->pzl_mc[i]);
            aeg_lowres_free(&e->pzl_low[i]);
        }
        aeg_free(e->pz_mbs);
        aeg_free(e->pzl_mbs);
        aeg_free(e->wp_mem);
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
