/*
 * H.264 in-loop deblocking filter (clause 8.7).
 *
 * The filter is fully specified by data: for every 4x4-block edge the
 * boundary strength bS is derived from the coding modes, the coded block
 * pattern and the motion vectors of the two adjacent blocks, and then the
 * edge is filtered with the normative thresholds alpha/beta/tc.
 *
 * Because the result only depends on the reconstructed picture and the
 * per-macroblock decisions, the whole pass is embarrassingly parallel over
 * edges: vertical edges never write outside their own macroblock row, so a
 * "all vertical edges, then all horizontal edges" schedule produces exactly
 * the same output as the sequential raster order while allowing the GPU (or
 * several CPU threads) to process independent edges concurrently.
 */
#include "h264.h"

static inline int clip3(int lo, int hi, int v)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

/* slice_alpha / slice_beta are the slice header's *_div2 values: 8.7.2.2
 * adds FilterOffsetA = slice_alpha_c0_offset_div2 << 1 (and B likewise) */
void aeg_deblock_thresholds(int qp, int slice_alpha, int slice_beta,
                            int *alpha, int *beta, const int8_t **tc0)
{
    int ia = AEG_CLIP3(0, 51, qp + 2 * slice_alpha);
    int ib = AEG_CLIP3(0, 51, qp + 2 * slice_beta);
    /* 8.7.2.2: alpha, beta and tc0 scale with the bit depth */
    *alpha = aeg_alpha_tab[ia] << (BIT_DEPTH - 8);
    *beta = aeg_beta_tab[ib] << (BIT_DEPTH - 8);
    *tc0 = aeg_tc0_tab[ia];
}

/* ------------------------------------------------------------------ */
/* Edge filters                                                        */
/* ------------------------------------------------------------------ */
static void filter_luma_edge(pixel *pix, int xs, int alpha, int beta, int tc0)
{
    int p2 = pix[-3 * xs], p1 = pix[-2 * xs], p0 = pix[-1 * xs];
    int q0 = pix[0], q1 = pix[1 * xs], q2 = pix[2 * xs];

    if (abs(p0 - q0) < alpha && abs(p1 - p0) < beta && abs(q1 - q0) < beta) {
        int tc = tc0, delta;
        if (abs(p2 - p0) < beta) {
            if (tc0)
                pix[-2 * xs] = aeg_clip_pixel(p1 + clip3(-tc0, tc0,
                                   ((p2 + ((p0 + q0 + 1) >> 1)) >> 1) - p1));
            tc++;
        }
        if (abs(q2 - q0) < beta) {
            if (tc0)
                pix[1 * xs] = aeg_clip_pixel(q1 + clip3(-tc0, tc0,
                                  ((q2 + ((p0 + q0 + 1) >> 1)) >> 1) - q1));
            tc++;
        }
        delta = clip3(-tc, tc, (((q0 - p0) * 4) + (p1 - q1) + 4) >> 3);
        pix[-1 * xs] = aeg_clip_pixel(p0 + delta);
        pix[0] = aeg_clip_pixel(q0 - delta);
    }
}

static void filter_luma_intra_edge(pixel *pix, int xs, int alpha, int beta)
{
    int p2 = pix[-3 * xs], p1 = pix[-2 * xs], p0 = pix[-1 * xs];
    int q0 = pix[0], q1 = pix[1 * xs], q2 = pix[2 * xs];

    if (abs(p0 - q0) < alpha && abs(p1 - p0) < beta && abs(q1 - q0) < beta) {
        if (abs(p0 - q0) < ((alpha >> 2) + 2)) {
            if (abs(p2 - p0) < beta) {
                int p3 = pix[-4 * xs];
                pix[-1 * xs] = (pixel)((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3);
                pix[-2 * xs] = (pixel)((p2 + p1 + p0 + q0 + 2) >> 2);
                pix[-3 * xs] = (pixel)((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3);
            } else {
                pix[-1 * xs] = (pixel)((2 * p1 + p0 + q1 + 2) >> 2);
            }
            if (abs(q2 - q0) < beta) {
                int q3 = pix[3 * xs];
                pix[0] = (pixel)((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3);
                pix[1 * xs] = (pixel)((p0 + q0 + q1 + q2 + 2) >> 2);
                pix[2 * xs] = (pixel)((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3);
            } else {
                pix[0] = (pixel)((2 * q1 + q0 + p1 + 2) >> 2);
            }
        } else {
            pix[-1 * xs] = (pixel)((2 * p1 + p0 + q1 + 2) >> 2);
            pix[0] = (pixel)((2 * q1 + q0 + p1 + 2) >> 2);
        }
    }
}

static void filter_chroma_edge(pixel *pix, int xs, int alpha, int beta, int tc)
{
    int p1 = pix[-2 * xs], p0 = pix[-1 * xs], q0 = pix[0], q1 = pix[1 * xs];
    if (abs(p0 - q0) < alpha && abs(p1 - p0) < beta && abs(q1 - q0) < beta) {
        int delta = clip3(-tc, tc, (((q0 - p0) * 4) + (p1 - q1) + 4) >> 3);
        pix[-1 * xs] = aeg_clip_pixel(p0 + delta);
        pix[0] = aeg_clip_pixel(q0 - delta);
    }
}

static void filter_chroma_intra_edge(pixel *pix, int xs, int alpha, int beta)
{
    int p1 = pix[-2 * xs], p0 = pix[-1 * xs], q0 = pix[0], q1 = pix[1 * xs];
    if (abs(p0 - q0) < alpha && abs(p1 - p0) < beta && abs(q1 - q0) < beta) {
        pix[-1 * xs] = (pixel)((2 * p1 + p0 + q1 + 2) >> 2);
        pix[0] = (pixel)((2 * q1 + q0 + p1 + 2) >> 2);
    }
}

/* ------------------------------------------------------------------ */
/* Boundary strength                                                  */
/* ------------------------------------------------------------------ */
static inline int is_intra(const aeg_mb *m)
{
    return m->type == AEG_MB_I4x4 || m->type == AEG_MB_I8x8 ||
           m->type == AEG_MB_I16x16 || m->type == AEG_MB_IPCM;
}

static inline int is_b(const aeg_mb *m)
{
    return m->type == AEG_MB_B_DIRECT || m->type == AEG_MB_B_L0 ||
           m->type == AEG_MB_B_L1 || m->type == AEG_MB_B_BI ||
           m->type == AEG_MB_B16x8 || m->type == AEG_MB_B8x16 ||
           m->type == AEG_MB_B_8x8;
}

/* Index of the 8x8 group that holds 4x4 block `blk` (raster order). */
static inline int nz8_group(int blk)
{
    int bx = blk & 3, by = blk >> 2;
    return (by >> 1) * 2 + (bx >> 1);
}

/* bS for the luma edge between block `pa` of MB p and block `qa` of MB q,
 * where both blocks are 4x4 luma blocks addressed in raster order. */
static int boundary_strength(const aeg_mb *p, int pa, const aeg_mb *q, int qa,
                             int mb_edge, int is_chroma, int vertical)
{
    int pi = is_intra(p), qi = is_intra(q);
    int nzp, nzq, dx, dy;
    /* a field macroblock's vertical vectors are in field lines: a step of 4
     * quarter frame samples is 2 of them (8.7.2.1) */
    const int my = q->fld ? 2 : 4;

    /* An intra macroblock edge is filtered at full strength between two frame
     * macroblocks; a horizontal one with a field macroblock on either side
     * (every macroblock of a field picture) only at 3, its rows being two
     * frame lines apart. */
    if (pi || qi)
        return mb_edge && (vertical || (!p->fld && !q->fld)) ? 4 : 3;

    if (is_chroma) {
        nzp = (p->cbp >> 4) & 3;
        nzq = (q->cbp >> 4) & 3;
    } else {
        /* With the 8x8 transform a decoder keeps one non-zero count per 8x8
         * group and compares groups, not 4x4 blocks. */
        nzp = p->t8 ? p->nz8[nz8_group(pa)] != 0 : p->nz[pa] != 0;
        nzq = q->t8 ? q->nz8[nz8_group(qa)] != 0 : q->nz[qa] != 0;
    }
    if (nzp || nzq)
        return 2;
    /* a horizontal macroblock edge between a frame and a field macroblock of
     * an MBAFF frame: at least 1, whatever the vectors (8.7.2.1,
     * mixedModeEdgeFlag) */
    if (mb_edge && !vertical && p->fld != q->fld)
        return 1;

    /* A B picture compares the vectors of *both* lists, and the filter is on as
     * soon as the two blocks do not predict from the same lists: a macroblock
     * that only looks forward is a different prediction from its neighbour that
     * only looks back, even when both vectors are zero (8.7.2.1). */
    if (is_b(p) || is_b(q)) {
        /* per block, not per macroblock: a partitioned B macroblock may
         * predict one partition from list 0 and the other from list 1 */
        int lp = aeg_mb_block_lists(p, pa), lq = aeg_mb_block_lists(q, qa);
        if ((lp & 1) != (lq & 1) || (lp & 2) != (lq & 2))
            return 1;
        if ((lp & 1) && (abs(p->mv[pa][0] - q->mv[qa][0]) >= 4 ||
                         abs(p->mv[pa][1] - q->mv[qa][1]) >= my))
            return 1;
        if ((lp & 2) && (abs(p->mv1[pa][0] - q->mv1[qa][0]) >= 4 ||
                         abs(p->mv1[pa][1] - q->mv1[qa][1]) >= my))
            return 1;
        /* (an MBAFF field macroblock's lists predict from the field of
         * either parity: different fields are different pictures) */
        if (((lp & 1) && p->ref_idx[pa] != q->ref_idx[qa]) ||
            ((lp & 2) && p->ref_idx1[pa] != q->ref_idx1[qa]))
            return 1;
        return 0;
    }
    /* P multi-reference: different reference indices also cause filtering,
     * even when vectors are equal (8.7.2.1). */
    if (p->ref_idx[pa] != q->ref_idx[qa])
        return 1;
    dx = abs(p->mv[pa][0] - q->mv[qa][0]);
    dy = abs(p->mv[pa][1] - q->mv[qa][1]);
    if (dx >= 4 || dy >= my)
        return 1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Macroblock deblocking                                              */
/* ------------------------------------------------------------------ */
/* bS of the four 4x4 blocks that meet at one edge of macroblock `q`.
 * `k` is the edge index (0 = macroblock boundary, 1..3 internal) and
 * `vertical` selects vertical (columns x = 4k) or horizontal edges. */
static void edge_bs(const aeg_mb *p, const aeg_mb *q, int vertical, int k,
                    int mb_edge, int bs[4])
{
    /* The p side of an internal edge belongs to `q` itself (the macroblock
     * being filtered); only the macroblock edge at k == 0 reaches into the
     * neighbouring macroblock `p`. */
    const aeg_mb *pp = mb_edge ? p : q;
    int j;
    for (j = 0; j < 4; j++) {
        int pa, qa;
        if (vertical) {
            pa = j * 4 + (mb_edge ? 3 : k - 1);
            qa = j * 4 + k;
        } else {
            pa = (mb_edge ? 3 : k - 1) * 4 + j;
            qa = k * 4 + j;
        }
        bs[j] = boundary_strength(pp, pa, q, qa, mb_edge, 0, vertical);
    }
}

/* Luma edge: 16 samples, four per bS entry (one 4x4 block).  As in the
 * reference decoder the strong intra filter is chosen once per edge from
 * bS[0]; otherwise every sample line with a non-negative tc0 is filtered
 * (tc0 == -1 means "do not filter").  `step` is the distance between two
 * samples along the edge. */
static void luma_edge_filter(pixel *pix, int xs, int step, int alpha,
                             int beta, const int8_t *tc0, const int bs[4],
                             int intra)
{
    int i, s;
    if (bs[0] == 4 && intra) {
        for (i = 0; i < 16; i++)
            filter_luma_intra_edge(pix + i * step, xs, alpha, beta);
        return;
    }
    for (i = 0; i < 4; i++) {
        int tc = tc0[bs[i]];
        if (tc < 0)
            continue;
        tc <<= BIT_DEPTH - 8;
        for (s = 0; s < 4; s++)
            filter_luma_edge(pix + (i * 4 + s) * step, xs, alpha, beta, tc);
    }
}

/* Chroma edge: 8 samples, the first pair uses bS[0], the second bS[1], and so
 * on (the reference decoder's 4:2:0 loop runs two samples per bS entry).  The
 * intra filter is again selected from bS[0] for the whole edge. */
static void chroma_edge_filter(pixel *pix, int xs, int step, int alpha,
                               int beta, const int8_t *tc0, const int bs[4],
                               int intra)
{
    int i, s;
    if (bs[0] == 4 && intra) {
        for (s = 0; s < 8; s++)
            filter_chroma_intra_edge(pix + s * step, xs, alpha, beta);
        return;
    }
    for (i = 0; i < 4; i++) {
        int tc = tc0[bs[i]];
        if (tc < 0)
            continue;
        tc = (tc << (BIT_DEPTH - 8)) + 1;    /* chroma: tC0 + 1 */
        for (s = 0; s < 2; s++)
            filter_chroma_edge(pix + (i * 2 + s) * step, xs, alpha, beta, tc);
    }
}

/* The four vertical (or horizontal) edges of macroblock `q`, whose samples
 * are seen through a view: luma at Y with line stride ys, chroma at C[0..1]
 * with stride cs (a field macroblock of an MBAFF frame sees every other
 * line).  `p_mb` is the macroblock across edge 0, NULL when that edge is not
 * filtered (picture or pair edge without a neighbour, or one filtered
 * separately).  `ki` is the first edge to filter (1: the macroblock edge is
 * done elsewhere). */
static void dbk_dir(const aeg_picture *pic, pixel *Y, pixel *C[2], int ys,
                    int cs, const aeg_mb *q, const aeg_mb *p_mb, int salpha,
                    int sbeta, int vertical)
{
    int k;

    for (k = 0; k < 4; k++) {
        int pos = k * 4;              /* edge coordinate inside the MB       */
        int mb_edge = (k == 0);
        int bs[4], alpha, beta;
        const int8_t *tc0;
        int qp;

        if (mb_edge && !p_mb)
            continue;
        /* A macroblock whose luma was coded with the 8x8 transform has no
         * transform boundary at 4 or 12, so only the internal edge at 8 is a
         * filtering candidate (7.3.5, 8.7: the reference decoder filters the
         * internal edges 0 and 2 only when IS_8x8DCT is set).  Chroma keeps
         * its 4x4 transforms, whose single internal edge is the one at luma
         * position 8, so this skip never removes a chroma edge. */
        if (q->t8 && (k & 1))
            continue;

        edge_bs(mb_edge ? p_mb : q, q, vertical, k, mb_edge, bs);
        if (!bs[0] && !bs[1] && !bs[2] && !bs[3])
            continue;

        /* The macroblock edge averages the two quantisers; the internal
         * edges use the quantiser of the macroblock being filtered. */
        qp = mb_edge ? (q->qp + p_mb->qp + 1) >> 1 : q->qp;
        aeg_deblock_thresholds(qp, salpha, sbeta, &alpha, &beta, &tc0);
        /* luma off (alpha or beta 0) does not mean chroma off: with a
         * positive chroma_qp_index_offset the chroma QP is the higher one */
        if (alpha && beta)
            luma_edge_filter(Y + (vertical ? pos : pos * ys),
                             vertical ? 1 : ys, vertical ? ys : 1,
                             alpha, beta, tc0, bs, mb_edge);

        /* Chroma has half the resolution: edges at chroma coordinates 0 and
         * 4, i.e. only luma edges k = 0 and k = 2. */
        if ((k & 1) == 0) {
            int cpos = pos >> 1;
            int plane;
            for (plane = 1; plane <= 2; plane++) {
                /* 8.7.2.2: each side's chroma QP from its own luma QP,
                 * then the average (not the chroma QP of the average luma
                 * QP, which differs once neighbouring QPs differ) */
                const int co = pic->chroma_qp_offset;
                int cqp = mb_edge ? (aeg_qp_chroma(q->qp, co) +
                                     aeg_qp_chroma(p_mb->qp, co) + 1) >> 1
                                  : aeg_qp_chroma(q->qp, co);
                int calpha, cbeta;
                const int8_t *ctc0;

                aeg_deblock_thresholds(cqp, salpha, sbeta, &calpha, &cbeta,
                                       &ctc0);
                if (!calpha || !cbeta)
                    continue;
                chroma_edge_filter(C[plane - 1] + (vertical ? cpos : cpos * cs),
                                   vertical ? 1 : cs, vertical ? cs : 1,
                                   calpha, cbeta, ctc0, bs, mb_edge);
            }
        }
    }
}

/* Applies the four vertical (or horizontal) edges of one macroblock. */
void aeg_deblock_mb_dir(aeg_picture *pic, const aeg_mb *mbs, int mb_stride,
                        int mb_x, int mb_y, int salpha, int sbeta, int vertical)
{
    const aeg_mb *q = &mbs[mb_y * mb_stride + mb_x];
    const aeg_mb *p_mb = vertical ? (mb_x > 0 ? q - 1 : NULL)
                                  : (mb_y > 0 ? q - mb_stride : NULL);
    pixel *C[2];

    C[0] = pic->plane[1] + mb_y * 8 * pic->stride[1] + mb_x * 8;
    C[1] = pic->plane[2] + mb_y * 8 * pic->stride[2] + mb_x * 8;
    dbk_dir(pic, pic->plane[0] + mb_y * 16 * pic->stride[0] + mb_x * 16, C,
            pic->stride[0], pic->stride[1], q, p_mb, salpha, sbeta, vertical);
}

/* non-zero coefficients in luma 4x4 block `blk` of m, as the filter sees them
 * (with the 8x8 transform, per 8x8 group) */
static inline int nz_blk(const aeg_mb *m, int blk)
{
    return m->t8 ? m->nz8[nz8_group(blk)] != 0 : m->nz[blk] != 0;
}

/* MBAFF: the left macroblock edge of q where the left pair is the other kind
 * (frame beside field or field beside frame).  Each side's lines meet lines
 * of both left macroblocks, so there are eight bS values and two QPs, one per
 * left macroblock (8.7.2, mixedModeEdgeFlag; FFmpeg's filter_mb_mbaff_edgev).
 * The view is q's own. */
static void dbk_left_mixed(const aeg_picture *pic, pixel *Y, pixel *C[2],
                           int ys, int cs, const aeg_mb *q, const aeg_mb *l0,
                           const aeg_mb *l1, int bot, int salpha, int sbeta)
{
    const int fld = q->fld, co = pic->chroma_qp_offset;
    int bs[8], i, half;

    for (i = 0; i < 8; i++) {
        /* field q: lines 0-7 meet the top left macroblock, 8-15 the bottom;
         * frame q: its even lines meet the top (field) one, odd the bottom */
        const aeg_mb *n = (fld ? i >> 2 : i & 1) ? l1 : l0;
        int nrow = fld ? (i & 3) : (bot ? 2 : 0) + (i >> 2);
        if (is_intra(q) || is_intra(n))
            bs[i] = 4;
        else
            bs[i] = 1 + (nz_blk(q, (i >> 1) * 4) || nz_blk(n, nrow * 4 + 3));
    }
    /* half 0 against l0, half 1 against l1 */
    for (half = 0; half < 2; half++) {
        const aeg_mb *n = half ? l1 : l0;
        int alpha, beta, pl, r;
        const int8_t *tc0;
        /* the eight luma lines of this half and their bS: field q -- lines
         * 8 half + r (stride ys), bS[4 half + r / 2]; frame q -- lines
         * 2 r + half (stride 2 ys), bS[2 (r / 2) + half] */
        pixel *y0 = fld ? Y + 8 * half * ys : Y + half * ys;
        int ls = fld ? ys : 2 * ys;
        const int *b = fld ? bs + 4 * half : bs + half;
        int bstep = fld ? 1 : 2;

        aeg_deblock_thresholds((q->qp + n->qp + 1) >> 1, salpha, sbeta,
                               &alpha, &beta, &tc0);
        if (alpha && beta) {
            if (b[0] == 4) {
                for (r = 0; r < 8; r++)
                    filter_luma_intra_edge(y0 + r * ls, 1, alpha, beta);
            } else {
                for (r = 0; r < 8; r++) {
                    int tc = tc0[b[(r >> 1) * bstep]];
                    if (tc >= 0)
                        filter_luma_edge(y0 + r * ls, 1, alpha, beta,
                                         tc << (BIT_DEPTH - 8));
                }
            }
        }
        for (pl = 0; pl < 2; pl++) {
            int cqp = (aeg_qp_chroma(q->qp, co) + aeg_qp_chroma(n->qp, co) +
                       1) >> 1;
            int calpha, cbeta;
            const int8_t *ctc0;
            /* four chroma lines, one bS each */
            pixel *c0 = fld ? C[pl] + 4 * half * cs : C[pl] + half * cs;
            int cls = fld ? cs : 2 * cs;

            aeg_deblock_thresholds(cqp, salpha, sbeta, &calpha, &cbeta, &ctc0);
            if (!calpha || !cbeta)
                continue;
            if (b[0] == 4) {
                for (r = 0; r < 4; r++)
                    filter_chroma_intra_edge(c0 + r * cls, 1, calpha, cbeta);
            } else {
                for (r = 0; r < 4; r++) {
                    int tc = ctc0[b[r * bstep]];
                    if (tc >= 0)
                        filter_chroma_edge(c0 + r * cls, 1, calpha, cbeta,
                                           (tc << (BIT_DEPTH - 8)) + 1);
                }
            }
        }
    }
}

/* MBAFF: the top edge of a frame top macroblock under a field pair is
 * filtered twice, once per field: q's even lines against the top field
 * macroblock above, its odd lines against the bottom one, each in field
 * lines -- and at bS 3 at most (FFmpeg's "special case in the norm"). */
static void dbk_top_fields(const aeg_picture *pic, pixel *Y, pixel *C[2],
                           int ys, int cs, const aeg_mb *q, const aeg_mb *a0,
                           int mb_stride, int salpha, int sbeta)
{
    const int co = pic->chroma_qp_offset;
    int j;

    for (j = 0; j < 2; j++) {
        const aeg_mb *n = a0 + j * mb_stride;
        int bs[4], i, alpha, beta, pl;
        const int8_t *tc0;

        for (i = 0; i < 4; i++)
            bs[i] = (is_intra(q) || is_intra(n)) ? 3
                    : 1 + (nz_blk(q, i) || nz_blk(n, 12 + i));
        aeg_deblock_thresholds((q->qp + n->qp + 1) >> 1, salpha, sbeta,
                               &alpha, &beta, &tc0);
        if (alpha && beta)
            luma_edge_filter(Y + j * ys, 2 * ys, 1, alpha, beta, tc0, bs, 0);
        for (pl = 0; pl < 2; pl++) {
            int cqp = (aeg_qp_chroma(q->qp, co) + aeg_qp_chroma(n->qp, co) +
                       1) >> 1;
            int calpha, cbeta;
            const int8_t *ctc0;
            aeg_deblock_thresholds(cqp, salpha, sbeta, &calpha, &cbeta, &ctc0);
            if (calpha && cbeta)
                chroma_edge_filter(C[pl] + j * cs, 2 * cs, 1, calpha, cbeta,
                                   ctc0, bs, 0);
        }
    }
}

/* One macroblock of an MBAFF frame (8.7 with MbaffFrameFlag): the view of a
 * field macroblock is its field's lines; the left edge beside the other kind
 * of pair and the top edge of a frame macroblock under a field pair have
 * their own rules; the rest is the frame filter in the view. */
void aeg_deblock_mb_mbaff(aeg_picture *pic, const aeg_mb *mbs, int mb_stride,
                          int mb_x, int mb_y, int salpha, int sbeta)
{
    const aeg_mb *q = &mbs[mb_y * mb_stride + mb_x];
    const int fld = q->fld, bot = mb_y & 1, py = mb_y >> 1;
    const int s0 = pic->stride[0], s1 = pic->stride[1];
    const int ys = fld ? 2 * s0 : s0, cs = fld ? 2 * s1 : s1;
    const int yl = fld ? 32 * py + bot : 16 * mb_y;       /* first line */
    const int cl = fld ? 16 * py + bot : 8 * mb_y;
    pixel *Y = pic->plane[0] + yl * s0 + 16 * mb_x;
    pixel *C[2];
    const aeg_mb *l0 = mb_x > 0 ? &mbs[2 * py * mb_stride + mb_x - 1] : NULL;
    const aeg_mb *top = NULL;

    C[0] = pic->plane[1] + cl * s1 + 8 * mb_x;
    C[1] = pic->plane[2] + cl * s1 + 8 * mb_x;

    /* vertical edges */
    if (l0 && l0->fld != fld) {
        dbk_left_mixed(pic, Y, C, ys, cs, q, l0, l0 + mb_stride, bot, salpha,
                       sbeta);
        dbk_dir(pic, Y, C, ys, cs, q, NULL, salpha, sbeta, 1);
    } else {
        dbk_dir(pic, Y, C, ys, cs, q, l0 ? q - 1 : NULL, salpha, sbeta, 1);
    }

    /* horizontal edges */
    if (!fld && bot) {
        top = q - mb_stride;                    /* the pair's top macroblock */
    } else if (py > 0) {
        const aeg_mb *a0 = &mbs[(2 * py - 2) * mb_stride + mb_x];
        if (!fld && a0->fld) {
            dbk_top_fields(pic, Y, C, ys, cs, q, a0, mb_stride, salpha, sbeta);
        } else if (fld) {
            /* a field macroblock: the same field of a field pair above, or
             * the bottom macroblock of a frame pair */
            top = (!bot && a0->fld) ? a0 : a0 + mb_stride;
        } else {
            top = a0 + mb_stride;
        }
    }
    dbk_dir(pic, Y, C, ys, cs, q, top, salpha, sbeta, 0);
}

/* The internal luma edges of one macroblock, `q`, whose reconstruction is
 * the 16x16 block at `pix` (stride `s`): the vertical edges at 4, 8 and 12,
 * then the horizontal ones, with the macroblock's own QP -- what the loop
 * filter will do inside the macroblock whatever its neighbours turn out to
 * be.  For the rate-distortion decision (x264's deblock RDO, subme 9 and up):
 * the distortion the decoder sees is that of the filtered picture.  Returns
 * whether any edge was filtered. */
int aeg_deblock_mb_internal_luma(pixel *pix, int s, const aeg_mb *q,
                                 int salpha, int sbeta)
{
    int alpha, beta, dir, k, any = 0;
    const int8_t *tc0;

    aeg_deblock_thresholds(q->qp, salpha, sbeta, &alpha, &beta, &tc0);
    if (!alpha || !beta)
        return 0;
    for (dir = 1; dir >= 0; dir--)
        for (k = 1; k < 4; k++) {
            int bs[4];
            if (q->t8 && (k & 1))
                continue;
            edge_bs(q, q, dir, k, 0, bs);
            if (!bs[0] && !bs[1] && !bs[2] && !bs[3])
                continue;
            luma_edge_filter(pix + (dir ? k * 4 : k * 4 * s), dir ? 1 : s,
                             dir ? s : 1, alpha, beta, tc0, bs, 0);
            any = 1;
        }
    return any;
}

void aeg_deblock_mb(aeg_picture *pic, const aeg_mb *mbs, int mb_stride,
                    int mb_x, int mb_y, int slice_alpha, int slice_beta)
{
    aeg_deblock_mb_dir(pic, mbs, mb_stride, mb_x, mb_y, slice_alpha, slice_beta, 1);
    aeg_deblock_mb_dir(pic, mbs, mb_stride, mb_x, mb_y, slice_alpha, slice_beta, 0);
}

/* One row of macroblock pairs of an MBAFF frame, in decoding order. */
void aeg_deblock_pair_row(aeg_picture *pic, const aeg_mb *mbs, int mb_stride,
                          int pair_y, int slice_alpha, int slice_beta)
{
    int x;
    for (x = 0; x < pic->mb_w; x++) {
        aeg_deblock_mb_mbaff(pic, mbs, mb_stride, x, 2 * pair_y, slice_alpha,
                             slice_beta);
        aeg_deblock_mb_mbaff(pic, mbs, mb_stride, x, 2 * pair_y + 1,
                             slice_alpha, slice_beta);
    }
}

void aeg_deblock_frame(aeg_picture *pic, const aeg_mb *mbs, int mb_stride,
                       int slice_alpha, int slice_beta, int deblock_idc)
{
    int x, y;
    if (!deblock_idc)
        return;

    /* H.264 8.7 defines the filter as a raster-order walk in which each
     * macroblock filters all of its vertical edges and then all of its
     * horizontal edges before the next macroblock is started.  The order
     * matters: a horizontal edge reads samples that the vertical edge of the
     * macroblock to the upper right has already modified, so macroblock
     * diagonals (or two frame-wide passes) are not equivalent.  Keeping the
     * normative order here is what makes the reconstructed picture match the
     * decoder bit-exactly.  In an MBAFF frame that order is the macroblock
     * address order: pair by pair, top then bottom. */
    if (pic->mbaff) {
        for (y = 0; y < pic->mb_h; y += 2)
            aeg_deblock_pair_row(pic, mbs, mb_stride, y >> 1, slice_alpha,
                                 slice_beta);
        return;
    }
    for (y = 0; y < pic->mb_h; y++)
        for (x = 0; x < pic->mb_w; x++)
            aeg_deblock_mb(pic, mbs, mb_stride, x, y, slice_alpha, slice_beta);
}
