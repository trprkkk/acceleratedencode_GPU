/*
 * Motion compensation and motion estimation.
 *
 * The fractional-pel rules follow H.264 8.4.2.2 exactly:
 *   luma   : 6-tap filter (1,-5,20,20,-5,1) for the half-pel positions,
 *            quarter-pel positions are averages of the two nearest
 *            half-pel/integer samples.
 *   chroma : bilinear interpolation with 1/8 pel precision.
 *
 * Motion estimation is a hierarchical hexagonal search (step 8/4/2/1)
 * followed by quarter-pel refinement on SATD, i.e. the same search family
 * x264 uses for its "hex" mode.  Both the integer search and the subpel
 * refinement are pure data-parallel kernels; backend.c can run them on a
 * GPU and compare against these CPU versions.
 */
#include "h264.h"

static inline uint8_t clip8(int v)
{
    return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

/* ------------------------------------------------------------------ */
/* Half-pel interpolation                                              */
/* ------------------------------------------------------------------ */
/* 6-tap horizontal half-pel at p[0] (needs p[-2..3]) */
static inline int hpel_h(const uint8_t *p)
{
    return (p[-2] + p[3] - 5 * (p[-1] + p[2]) + 20 * (p[0] + p[1]) + 16) >> 5;
}
static inline int hpel_v(const uint8_t *p, int s)
{
    return ((int)p[-2 * s] + p[3 * s] - 5 * (p[-s] + p[2 * s]) +
            20 * (p[0] + p[s]) + 16) >> 5;
}
/* both axes: vertical 6-tap applied to clipped horizontal half-pel values */
static inline int hpel_hv(const uint8_t *p, int s)
{
    int a = clip8(hpel_h(p - 2 * s));
    int b = clip8(hpel_h(p - s));
    int c = clip8(hpel_h(p));
    int d = clip8(hpel_h(p + s));
    int e = clip8(hpel_h(p + 2 * s));
    int f = clip8(hpel_h(p + 3 * s));
    return (a - 5 * b + 20 * c + 20 * d - 5 * e + f + 512) >> 10;
}

/* Which of the 4 interpolated planes to combine for each 1/4-pel position
 * (same decomposition the spec's averaging rules describe). */
static const uint8_t hpel_ref0[16] = { 0, 1, 1, 1, 0, 1, 1, 1, 2, 3, 3, 3, 0, 1, 1, 1 };
static const uint8_t hpel_ref1[16] = { 0, 0, 1, 0, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 2 };

static inline int plane_get(int plane, const uint8_t *p, int s)
{
    switch (plane) {
    case 0:  return p[0];
    case 1:  return clip8(hpel_h(p));
    case 2:  return clip8(hpel_v(p, s));
    default: return clip8(hpel_hv(p, s));
    }
}

void aeg_mc_luma(uint8_t *dst, int dstride, const uint8_t *src, int sstride,
                 int w, int h, int fx, int fy)
{
    int x, y;
    const int idx = (fy << 2) | fx;

    if (!fx && !fy) {
        for (y = 0; y < h; y++)
            memcpy(dst + y * dstride, src + y * sstride, (size_t)w);
        return;
    }
    if (!(idx & 5)) {
        /* exactly an integer or half-pel position: single plane */
        int plane = hpel_ref0[idx];
        for (y = 0; y < h; y++)
            for (x = 0; x < w; x++)
                dst[y * dstride + x] = (uint8_t)plane_get(plane, src + y * sstride + x, sstride);
        return;
    }
    {
        int p1 = hpel_ref0[idx], p2 = hpel_ref1[idx];
        int o1y = (fy == 3) ? 1 : 0;   /* src1 is vertically offset by one */
        int o2x = (fx == 3) ? 1 : 0;   /* src2 is horizontally offset by one */
        for (y = 0; y < h; y++) {
            const uint8_t *r1 = src + (y + o1y) * sstride;
            const uint8_t *r2 = src + y * sstride;
            for (x = 0; x < w; x++) {
                int a = plane_get(p1, r1 + x, sstride);
                int b = plane_get(p2, r2 + x + o2x, sstride);
                dst[y * dstride + x] = (uint8_t)((a + b + 1) >> 1);
            }
        }
    }
}

void aeg_mc_chroma(uint8_t *dst, int dstride, const uint8_t *src, int sstride,
                   int w, int h, int fx, int fy)
{
    int x, y;
    if (!fx && !fy) {
        for (y = 0; y < h; y++)
            memcpy(dst + y * dstride, src + y * sstride, (size_t)w);
        return;
    }
    for (y = 0; y < h; y++)
        for (x = 0; x < w; x++) {
            const uint8_t *p = src + y * sstride + x;
            int a = p[0], b = p[1], c = p[sstride], d = p[sstride + 1];
            int v = (8 - fx) * (8 - fy) * a + fx * (8 - fy) * b +
                    (8 - fx) * fy * c + fx * fy * d;
            dst[y * dstride + x] = (uint8_t)((v + 32) >> 6);
        }
}

/* ------------------------------------------------------------------ */
/* Motion estimation                                                   */
/* ------------------------------------------------------------------ */
/* Six-point hexagonal search pattern around the centre. */
static const int8_t hex_dx[7] = { 0, -2, -1,  1,  2,  1, -1 };
static const int8_t hex_dy[7] = { 0,  0, -1, -1,  0,  1,  1 };
static const int8_t sq_dx[9]  = { 0, -1,  0,  1, -1,  1, -1,  0,  1 };
static const int8_t sq_dy[9]  = { 0,  0, -1,  0, -1,  1,  1,  1, -1 };

/* integer-pel SAD at displacement (dx, dy) */
static inline int me_sad(const uint8_t *ref, int rstride, const uint8_t *cur,
                         int cstride, int bw, int bh, int dx, int dy)
{
    return aeg_sad(bw, bh, cur, cstride, ref + dy * rstride + dx, rstride);
}

static int hex_search(const uint8_t *ref, int rstride, const uint8_t *cur,
                      int cstride, int bw, int bh, int range,
                      int *mvx, int *mvy)
{
    int cx = *mvx, cy = *mvy;
    int best = me_sad(ref, rstride, cur, cstride, bw, bh, cx, cy);
    int step, i, improved = 1;

    for (step = 0; step < 2 && improved; step++) {
        const int8_t *dx = step ? sq_dx : hex_dx;
        const int8_t *dy = step ? sq_dy : hex_dy;
        int n = step ? 9 : 7;
        improved = 1;
        while (improved) {
            int bx = cx, by = cy, bc = best;
            improved = 0;
            for (i = 1; i < n; i++) {
                int nx = cx + dx[i], ny = cy + dy[i], c;
                if (nx < -range || nx > range || ny < -range || ny > range)
                    continue;
                c = me_sad(ref, rstride, cur, cstride, bw, bh, nx, ny);
                if (c < bc) { bc = c; bx = nx; by = ny; improved = 1; }
            }
            cx = bx; cy = by; best = bc;
        }
    }
    *mvx = cx; *mvy = cy;
    return best;
}

/* subpel refinement: quarter-pel, SATD driven */
static int subpel_refine(const uint8_t *ref, int rstride, const uint8_t *cur,
                         int cstride, int bw, int bh, int range,
                         int *mvx, int *mvy)
{
    static const int8_t offs[9][2] = {
        {0,0},{-1,0},{1,0},{0,-1},{0,1},{-1,-1},{1,-1},{-1,1},{1,1}
    };
    uint8_t buf[16 * 16];
    int best = 1 << 30, bx = *mvx, by = *mvy;
    int i, pass;
    int cx = *mvx, cy = *mvy;

    /* two passes: half-pel then quarter-pel steps */
    for (pass = 0; pass < 2; pass++) {
        int step = pass ? 1 : 2;
        int improved = 1;
        while (improved) {
            improved = 0;
            for (i = 0; i < 9; i++) {
                int nx = cx + offs[i][0] * step, ny = cy + offs[i][1] * step;
                int ix = nx >> 2, iy = ny >> 2, cost;
                if (ix < -range || ix > range || iy < -range || iy > range)
                    continue;
                if (nx < -range * 4 || nx > range * 4 || ny < -range * 4 || ny > range * 4)
                    continue;
                aeg_mc_luma(buf, bw, ref + iy * rstride + ix, rstride, bw, bh,
                            nx & 3, ny & 3);
                cost = aeg_satd(bw, bh, buf, bw, cur, cstride);
                if (cost < best) { best = cost; bx = nx; by = ny; improved = 1; }
            }
            cx = bx; cy = by;
        }
    }
    *mvx = bx; *mvy = by;
    return best;
}

int aeg_me_block(const uint8_t *ref, int rstride, const uint8_t *cur,
                 int cstride, int bw, int bh, int range, int subme,
                 int pred_x, int pred_y, int *mvx, int *mvy, int *sad_out)
{
    int best_sad = 1 << 30, bx = 0, by = 0;

    /* candidate starts: predicted MV, zero, and the integer-rounded
     * predicted MV.  Evaluating all of them keeps the search robust for
     * fast motion without the cost of a full search. */
    {
        int cand[3][2];
        int i;
        cand[0][0] = pred_x >> 2; cand[0][1] = pred_y >> 2;
        cand[1][0] = 0;           cand[1][1] = 0;
        cand[2][0] = (pred_x + 2) >> 2; cand[2][1] = (pred_y + 2) >> 2;
        for (i = 0; i < 3; i++) {
            int c;
            if (cand[i][0] < -range || cand[i][0] > range ||
                cand[i][1] < -range || cand[i][1] > range)
                continue;
            c = me_sad(ref, rstride, cur, cstride, bw, bh, cand[i][0], cand[i][1]);
            if (c < best_sad) { best_sad = c; bx = cand[i][0]; by = cand[i][1]; }
        }
    }
    best_sad = hex_search(ref, rstride, cur, cstride, bw, bh, range, &bx, &by);

    if (subme <= 0) {
        *mvx = bx * 4; *mvy = by * 4; *sad_out = best_sad;
        return best_sad;
    }

    {
        int qx = bx * 4, qy = by * 4;
        best_sad = subpel_refine(ref, rstride, cur, cstride, bw, bh, range, &qx, &qy);
        *mvx = qx; *mvy = qy;
    }
    *sad_out = best_sad;
    return best_sad;
}

/* ------------------------------------------------------------------ */
/* MV prediction                                                       */
/* ------------------------------------------------------------------ */
static inline int is_intra_mb(const aeg_mb *m)
{
    return m->type == AEG_MB_I4x4 || m->type == AEG_MB_I16x16 ||
           m->type == AEG_MB_IPCM;
}

static inline int mv_median(int a, int b, int c)
{
    int mx = a > b ? a : b;
    int mn = a < b ? a : b;
    if (c > mx) return mx;
    if (c < mn) return mn;
    return c;
}

/* Motion vector prediction (H.264 8.4.1.3.1).
 *
 * The three candidates are the blocks to the left (A), above (B) and above to
 * the right (C).  C falls back to the block above and to the left when the
 * macroblock above-right does not exist, which is what makes a macroblock on
 * the right border predict differently from one in the middle of the line. */
void aeg_mv_predict(const aeg_mb *mbs, int mb_stride, int mb_w, int mb_x, int mb_y,
                    int *mvx, int *mvy)
{
    const aeg_mb *m;
    int ax = 0, ay = 0, bx = 0, by = 0, cx = 0, cy = 0;
    int a_av = 0, b_av = 0, c_av = 0, n;

    if (mb_x > 0) {
        m = &mbs[mb_y * mb_stride + mb_x - 1];
        a_av = !is_intra_mb(m);
        ax = m->mv[3][0];  ay = m->mv[3][1];
    }
    if (mb_y > 0) {
        m = &mbs[(mb_y - 1) * mb_stride + mb_x];
        b_av = !is_intra_mb(m);
        bx = m->mv[12][0]; by = m->mv[12][1];
    }
    if (mb_y > 0 && mb_x + 1 < mb_w) {
        m = &mbs[(mb_y - 1) * mb_stride + mb_x + 1];
        c_av = !is_intra_mb(m);
        cx = m->mv[12][0]; cy = m->mv[12][1];
    } else if (mb_y > 0 && mb_x > 0) {
        m = &mbs[(mb_y - 1) * mb_stride + mb_x - 1];
        c_av = !is_intra_mb(m);
        cx = m->mv[15][0]; cy = m->mv[15][1];
    }

    n = a_av + b_av + c_av;
    if (n == 0) {
        *mvx = *mvy = 0;
    } else if (n == 1) {
        *mvx = a_av ? ax : (b_av ? bx : cx);
        *mvy = a_av ? ay : (b_av ? by : cy);
    } else {
        *mvx = mv_median(ax, bx, cx);
        *mvy = mv_median(ay, by, cy);
    }
}

/* ------------------------------------------------------------------ */
/* Rate cost helpers                                                   */
/* ------------------------------------------------------------------ */
int aeg_ue_bits(uint32_t v)
{
    int n = 0, t = (int)v;
    while (t) { n++; t >>= 1; }
    return 2 * n + 1;
}

int aeg_se_bits(int32_t v)
{
    uint32_t m = (uint32_t)(v <= 0 ? -(int64_t)v : v);
    return aeg_ue_bits(2 * m - (v > 0 ? 1 : 0));
}

int aeg_mvd_bits(int dx, int dy)
{
    return aeg_se_bits(dx) + aeg_se_bits(dy);
}
