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
    return m->type == AEG_MB_I4x4 || m->type == AEG_MB_I16x16 ||
           m->type == AEG_MB_IPCM;
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
/* Applies the edges of one macroblock.  `vertical` selects the four
 * vertical (x = 0,4,8,12) or horizontal (y = 0,4,8,12) 4x4 block edges. */
void aeg_deblock_mb_dir(aeg_picture *pic, const aeg_mb *mbs, int mb_stride,
                        int mb_x, int mb_y, int salpha, int sbeta, int vertical)
{
    const aeg_mb *q = &mbs[mb_y * mb_stride + mb_x];
    const aeg_mb *left = mb_x > 0 ? q - 1 : q;
    const aeg_mb *above = mb_y > 0 ? q - mb_stride : q;
    int k;

    for (k = 0; k < 4; k++) {
        int pos = k * 4;              /* luma edge coordinate inside MB      */
        int mb_edge = (k == 0);
        const aeg_mb *p_mb = vertical ? left : above;
        int px = (k == 0) ? 3 : k - 1;      /* block column left of edge   */
        int py = (k == 0) ? 3 : k - 1;
        int qx = k, qy = k;
        int j;

        if (mb_edge && ((vertical && mb_x == 0) || (!vertical && mb_y == 0)))
            continue;

        for (j = 0; j < 4; j++) {
            int pa, qa, bs, alpha, beta;
            const int8_t *tc0;
            int qp = (q->qp + p_mb->qp + 1) >> 1;
            uint8_t *pix;
            int xs;
            int lx, ly;

            if (vertical) {
                pa = j * 4 + px; qa = j * 4 + qx;
                lx = mb_x * 16 + pos; ly = mb_y * 16 + j * 4;
            } else {
                pa = py * 4 + j; qa = qy * 4 + j;
                lx = mb_x * 16 + j * 4; ly = mb_y * 16 + pos;
            }
            bs = boundary_strength(p_mb, pa, q, qa, mb_edge, 0);
            if (bs < 1)
                continue;
            aeg_deblock_thresholds(qp, salpha, sbeta, &alpha, &beta, &tc0);
            if (!alpha || !beta)
                continue;
            xs = vertical ? 1 : pic->stride[0];
            pix = pic->plane[0] + ly * pic->stride[0] + lx;
            if (bs == 4)
                filter_luma_intra_edge(pix, xs, alpha, beta);
            else if (tc0[bs] > 0)
                filter_luma_edge(pix, xs, alpha, beta, tc0[bs]);
        }

        /* chroma: only two edges per macroblock (chroma 4x4 blocks) */
        if ((pos & 7) == 0) {
            int cpos = pos >> 1;
            int j;
            for (j = 0; j < 2; j++) {
                int pa, qa, bs, plane;
                for (plane = 1; plane <= 2; plane++) {
                    int cqp = aeg_qp_chroma((q->qp + p_mb->qp + 1) >> 1, 0);
                    int alpha, beta;
                    const int8_t *tc0;
                    uint8_t *pix;
                    int xs, cx, cy;

                    if (vertical) {
                        pa = (j * 2) * 4 + px / 2; qa = (j * 2) * 4 + qx / 2;
                        cx = mb_x * 8 + cpos; cy = mb_y * 8 + j * 2;
                    } else {
                        pa = (py / 2) * 4 + j * 2; qa = (qy / 2) * 4 + j * 2;
                        cx = mb_x * 8 + j * 2; cy = mb_y * 8 + cpos;
                    }
                    bs = boundary_strength(p_mb, pa, q, qa, mb_edge, 1);
                    if (bs < 1)
                        continue;
                    aeg_deblock_thresholds(cqp, salpha, sbeta, &alpha, &beta, &tc0);
                    if (!alpha || !beta)
                        continue;
                    xs = vertical ? 1 : pic->stride[plane];
                    pix = pic->plane[plane] + cy * pic->stride[plane] + cx;
                    if (bs == 4)
                        filter_chroma_intra_edge(pix, xs, alpha, beta);
                    else if (tc0[bs] >= 0)
                        filter_chroma_edge(pix, xs, alpha, beta, tc0[bs] + 1);
                }
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

    /* vertical edges for the whole frame, then horizontal edges: each pass
     * only reads/writes pixels of a single macroblock row, so the two
     * passes are equivalent to the sequential order and are safe to
     * parallelise (see backend_aeg_deblock_* ). */
    for (y = 0; y < pic->mb_h; y++)
        for (x = 0; x < pic->mb_w; x++)
            aeg_deblock_mb_dir(pic, mbs, mb_stride, x, y, slice_alpha, slice_beta, 1);
    for (y = 0; y < pic->mb_h; y++)
        for (x = 0; x < pic->mb_w; x++)
            aeg_deblock_mb_dir(pic, mbs, mb_stride, x, y, slice_alpha, slice_beta, 0);
}
