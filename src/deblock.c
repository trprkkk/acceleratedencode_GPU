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

static inline uint8_t clip8(int v)
{
    return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
}
static inline int clip3(int lo, int hi, int v)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

void aeg_deblock_thresholds(int qp, int slice_alpha, int slice_beta,
                            int *alpha, int *beta, const int8_t **tc0)
{
    int ia = AEG_CLIP3(0, 51, qp + slice_alpha);
    int ib = AEG_CLIP3(0, 51, qp + slice_beta);
    *alpha = aeg_alpha_tab[ia];
    *beta = aeg_beta_tab[ib];
    *tc0 = aeg_tc0_tab[ia];
}

/* ------------------------------------------------------------------ */
/* Edge filters                                                        */
/* ------------------------------------------------------------------ */
static void filter_luma_edge(uint8_t *pix, int xs, int alpha, int beta, int tc0)
{
    int p2 = pix[-3 * xs], p1 = pix[-2 * xs], p0 = pix[-1 * xs];
    int q0 = pix[0], q1 = pix[1 * xs], q2 = pix[2 * xs];

    if (abs(p0 - q0) < alpha && abs(p1 - p0) < beta && abs(q1 - q0) < beta) {
        int tc = tc0, delta;
        if (abs(p2 - p0) < beta) {
            if (tc0)
                pix[-2 * xs] = clip8(p1 + clip3(-tc0, tc0,
                                   ((p2 + ((p0 + q0 + 1) >> 1)) >> 1) - p1));
            tc++;
        }
        if (abs(q2 - q0) < beta) {
            if (tc0)
                pix[1 * xs] = clip8(q1 + clip3(-tc0, tc0,
                                  ((q2 + ((p0 + q0 + 1) >> 1)) >> 1) - q1));
            tc++;
        }
        delta = clip3(-tc, tc, (((q0 - p0) * 4) + (p1 - q1) + 4) >> 3);
        pix[-1 * xs] = clip8(p0 + delta);
        pix[0] = clip8(q0 - delta);
    }
}

static void filter_luma_intra_edge(uint8_t *pix, int xs, int alpha, int beta)
{
    int p2 = pix[-3 * xs], p1 = pix[-2 * xs], p0 = pix[-1 * xs];
    int q0 = pix[0], q1 = pix[1 * xs], q2 = pix[2 * xs];

    if (abs(p0 - q0) < alpha && abs(p1 - p0) < beta && abs(q1 - q0) < beta) {
        if (abs(p0 - q0) < ((alpha >> 2) + 2)) {
            if (abs(p2 - p0) < beta) {
                int p3 = pix[-4 * xs];
                pix[-1 * xs] = (uint8_t)((p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3);
                pix[-2 * xs] = (uint8_t)((p2 + p1 + p0 + q0 + 2) >> 2);
                pix[-3 * xs] = (uint8_t)((2 * p3 + 3 * p2 + p1 + p0 + q0 + 4) >> 3);
            } else {
                pix[-1 * xs] = (uint8_t)((2 * p1 + p0 + q1 + 2) >> 2);
            }
            if (abs(q2 - q0) < beta) {
                int q3 = pix[3 * xs];
                pix[0] = (uint8_t)((p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3);
                pix[1 * xs] = (uint8_t)((p0 + q0 + q1 + q2 + 2) >> 2);
                pix[2 * xs] = (uint8_t)((2 * q3 + 3 * q2 + q1 + q0 + p0 + 4) >> 3);
            } else {
                pix[0] = (uint8_t)((2 * q1 + q0 + p1 + 2) >> 2);
            }
        } else {
            pix[-1 * xs] = (uint8_t)((2 * p1 + p0 + q1 + 2) >> 2);
            pix[0] = (uint8_t)((2 * q1 + q0 + p1 + 2) >> 2);
        }
    }
}

static void filter_chroma_edge(uint8_t *pix, int xs, int alpha, int beta, int tc)
{
    int p1 = pix[-2 * xs], p0 = pix[-1 * xs], q0 = pix[0], q1 = pix[1 * xs];
    if (abs(p0 - q0) < alpha && abs(p1 - p0) < beta && abs(q1 - q0) < beta) {
        int delta = clip3(-tc, tc, (((q0 - p0) * 4) + (p1 - q1) + 4) >> 3);
        pix[-1 * xs] = clip8(p0 + delta);
        pix[0] = clip8(q0 - delta);
    }
}

static void filter_chroma_intra_edge(uint8_t *pix, int xs, int alpha, int beta)
{
    int p1 = pix[-2 * xs], p0 = pix[-1 * xs], q0 = pix[0], q1 = pix[1 * xs];
    if (abs(p0 - q0) < alpha && abs(p1 - p0) < beta && abs(q1 - q0) < beta) {
        pix[-1 * xs] = (uint8_t)((2 * p1 + p0 + q1 + 2) >> 2);
        pix[0] = (uint8_t)((2 * q1 + q0 + p1 + 2) >> 2);
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

/* bS for the luma edge between block `pa` of MB p and block `qa` of MB q,
 * where both blocks are 4x4 luma blocks addressed in raster order. */
static int boundary_strength(const aeg_mb *p, int pa, const aeg_mb *q, int qa,
                             int mb_edge, int is_chroma)
{
    int pi = is_intra(p), qi = is_intra(q);
    int nzp, nzq, dx, dy;

    if (pi || qi)
        return mb_edge ? 4 : 3;

    if (is_chroma) {
        nzp = (p->cbp >> 4) & 3;
        nzq = (q->cbp >> 4) & 3;
    } else {
        nzp = p->nz[pa] != 0;
        nzq = q->nz[qa] != 0;
    }
    if (nzp || nzq)
        return 2;

    dx = abs(p->mv[pa][0] - q->mv[qa][0]);
    dy = abs(p->mv[pa][1] - q->mv[qa][1]);
    if (dx >= 4 || dy >= 4)
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
        bs[j] = boundary_strength(pp, pa, q, qa, mb_edge, 0);
    }
}

/* Luma edge: 16 samples, four per bS entry (one 4x4 block).  As in the
 * reference decoder the strong intra filter is chosen once per edge from
 * bS[0]; otherwise every sample line with a non-negative tc0 is filtered
 * (tc0 == -1 means "do not filter").  `step` is the distance between two
 * samples along the edge. */
static void luma_edge_filter(uint8_t *pix, int xs, int step, int alpha,
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
        for (s = 0; s < 4; s++)
            filter_luma_edge(pix + (i * 4 + s) * step, xs, alpha, beta, tc);
    }
}

/* Chroma edge: 8 samples, the first pair uses bS[0], the second bS[1], and so
 * on (the reference decoder's 4:2:0 loop runs two samples per bS entry).  The
 * intra filter is again selected from bS[0] for the whole edge. */
static void chroma_edge_filter(uint8_t *pix, int xs, int step, int alpha,
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
        int tc = tc0[bs[i]] + 1;
        if (tc <= 0)
            continue;
        for (s = 0; s < 2; s++)
            filter_chroma_edge(pix + (i * 2 + s) * step, xs, alpha, beta, tc);
    }
}

/* Applies the four vertical (or horizontal) edges of one macroblock. */
void aeg_deblock_mb_dir(aeg_picture *pic, const aeg_mb *mbs, int mb_stride,
                        int mb_x, int mb_y, int salpha, int sbeta, int vertical)
{
    const aeg_mb *q = &mbs[mb_y * mb_stride + mb_x];
    const aeg_mb *left = mb_x > 0 ? q - 1 : q;
    const aeg_mb *above = mb_y > 0 ? q - mb_stride : q;
    const aeg_mb *p_mb = vertical ? left : above;
    int k;

    for (k = 0; k < 4; k++) {
        int pos = k * 4;              /* edge coordinate inside the MB       */
        int mb_edge = (k == 0);
        int bs[4], alpha, beta;
        const int8_t *tc0;
        int qp, xs, step, lx, ly;

        if (mb_edge && ((vertical && mb_x == 0) || (!vertical && mb_y == 0)))
            continue;
        /* A macroblock whose luma was coded with the 8x8 transform has no
         * transform boundary at 4 or 12, so only the internal edge at 8 is a
         * filtering candidate (7.3.5, 8.7: the reference decoder filters the
         * internal edges 0 and 2 only when IS_8x8DCT is set).  Chroma keeps
         * its 4x4 transforms, whose single internal edge is the one at luma
         * position 8, so this skip never removes a chroma edge. */
        if (q->t8 && (k & 1))
            continue;

        edge_bs(p_mb, q, vertical, k, mb_edge, bs);
        if (!bs[0] && !bs[1] && !bs[2] && !bs[3])
            continue;

        /* The macroblock edge averages the two quantisers; the internal
         * edges use the quantiser of the macroblock being filtered. */
        qp = mb_edge ? (q->qp + p_mb->qp + 1) >> 1 : q->qp;
        aeg_deblock_thresholds(qp, salpha, sbeta, &alpha, &beta, &tc0);
        if (!alpha || !beta)
            continue;

        xs = vertical ? 1 : pic->stride[0];
        step = vertical ? pic->stride[0] : 1;
        lx = mb_x * 16 + (vertical ? pos : 0);
        ly = mb_y * 16 + (vertical ? 0 : pos);
        luma_edge_filter(pic->plane[0] + ly * pic->stride[0] + lx, xs, step,
                         alpha, beta, tc0, bs, mb_edge);

        /* Chroma has half the resolution: edges at chroma coordinates 0 and
         * 4, i.e. only luma edges k = 0 and k = 2. */
        if ((k & 1) == 0) {
            int cpos = pos >> 1;
            int plane;
            for (plane = 1; plane <= 2; plane++) {
                int cqp = mb_edge ? aeg_qp_chroma(qp, 0)
                                  : aeg_qp_chroma(q->qp, 0);
                int calpha, cbeta;
                const int8_t *ctc0;
                int cxs = vertical ? 1 : pic->stride[plane];
                int cstep = vertical ? pic->stride[plane] : 1;
                int cx = mb_x * 8 + (vertical ? cpos : 0);
                int cy = mb_y * 8 + (vertical ? 0 : cpos);

                aeg_deblock_thresholds(cqp, salpha, sbeta, &calpha, &cbeta,
                                       &ctc0);
                if (!calpha || !cbeta)
                    continue;
                chroma_edge_filter(pic->plane[plane] + cy * pic->stride[plane] + cx,
                                   cxs, cstep, calpha, cbeta, ctc0, bs, mb_edge);
            }
        }
    }
}

void aeg_deblock_mb(aeg_picture *pic, const aeg_mb *mbs, int mb_stride,
                    int mb_x, int mb_y, int slice_alpha, int slice_beta)
{
    aeg_deblock_mb_dir(pic, mbs, mb_stride, mb_x, mb_y, slice_alpha, slice_beta, 1);
    aeg_deblock_mb_dir(pic, mbs, mb_stride, mb_x, mb_y, slice_alpha, slice_beta, 0);
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
     * decoder bit-exactly. */
    for (y = 0; y < pic->mb_h; y++)
        for (x = 0; x < pic->mb_w; x++)
            aeg_deblock_mb(pic, mbs, mb_stride, x, y, slice_alpha, slice_beta);
}
