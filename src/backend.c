/*
 * CPU implementation of the accelerator operations, the backend factory and
 * the per-stage dispatcher.
 *
 * The dispatcher exists because a GPU is not automatically faster for every
 * stage of an encoder.  A 16x16 SAD costs a few hundred nanoseconds on a CPU
 * core; the same work on a GPU costs a kernel launch plus, for the search
 * results, a transfer back.  The dispatcher therefore:
 *
 *   1. measures every stage on whichever backend actually ran it,
 *   2. keeps an exponential moving average of the CPU and GPU times,
 *   3. flips a stage to the GPU only when the GPU time is clearly lower
 *      (hysteresis avoids oscillating between the two), and
 *   4. always keeps the stages whose cost is dominated by serial
 *      dependencies (deblocking, CAVLC bit emission, intra prediction of the
 *      current block) on the CPU.
 *
 * Because both implementations are bit-identical, the decisions can change
 * from frame to frame without affecting the produced bitstream.
 */
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include "backend.h"

/* ------------------------------------------------------------------ */
/* CPU reference implementations                                       */
/* ------------------------------------------------------------------ */
static int cpu_sad_batch(void *priv, const pixel *cur, int cstride,
                         const pixel *ref, int rstride, int bw, int bh,
                         const int16_t *mv, int count, int32_t *out)
{
    int i;
    (void)priv;
    for (i = 0; i < count; i++) {
        const pixel *c0 = cur + (ptrdiff_t)mv[4 * i + 1] * cstride + mv[4 * i + 0];
        const pixel *r = ref + (ptrdiff_t)mv[4 * i + 3] * rstride + mv[4 * i + 2];
        int x, y, sad = 0;
        for (y = 0; y < bh; y++) {
            const pixel *c = c0 + y * cstride;
            const pixel *rr = r + y * rstride;
            for (x = 0; x < bw; x++) {
                int d = (int)c[x] - (int)rr[x];
                sad += d < 0 ? -d : d;
            }
        }
        out[i] = sad;
    }
    return AEG_OK;
}

static int cpu_tq_batch(void *priv, const int16_t *res, int nblk, int qp,
                        int intra, dctcoef *level, int32_t *nz)
{
    int b;
    (void)priv;
    for (b = 0; b < nblk; b++) {
        dctcoef coef[16];
        int i, n = 0;
        aeg_transform_4x4(res + 16 * b, 4, coef);
        aeg_quant_4x4(coef, level + 16 * b, qp, intra);
        for (i = 0; i < 16; i++)
            n += level[16 * b + i] != 0;
        if (nz)
            nz[b] = n;
    }
    return AEG_OK;
}

static int cpu_iq_idct_batch(void *priv, const dctcoef *level, int nblk,
                             int qp, int16_t *res)
{
    int b;
    (void)priv;
    /* This used to carry its own copy of the inverse transform, and that copy
     * was *not* the one the encoder uses: it skipped the int16 truncation of
     * the dequantised coefficients (bug 8), skipped the DC rounding before the
     * first pass, and rounded the second pass instead of truncating it.  The
     * device kernel was checked against this copy, so both agreed and both
     * disagreed with the reference decoder -- exactly the trap that made the
     * first SIMD inverse transform look right.  It now calls the same function
     * the scalar path is built from, so there is one arithmetic to be wrong. */
    /* Most blocks of a batch are empty or DC-only (the quantiser and the
     * level selection see to that), and for those the two passes reduce to a
     * constant: d[0] = (int16)(level[0] * scale << qp/6) + 32 goes through
     * the first pass truncated to int16 and through the second unchanged, so
     * every sample is ((int16)d[0]) >> 6 -- the same value aeg_idct_residual()
     * computes, including at the int16 extremes (coef_check covers both). */
    for (b = 0; b < nblk; b++) {
        const dctcoef *l = level + 16 * b;
        int16_t *r = res + 16 * b;
        int i, ac = 0;
        for (i = 1; i < 16; i++)
            ac |= l[i];
        if (!ac) {
            const int qpi = qp + QP_BD_OFFSET;          /* QP'Y */
            dctcoef d0 = (dctcoef)((dctcoef)((int32_t)l[0] *
                                             aeg_dequant_ls[qpi % 6][0] *
                                             (1 << (qpi / 6))) + 32);
            int16_t v = (int16_t)(d0 >> 6);
            for (i = 0; i < 16; i++)
                r[i] = v;
        } else {
            aeg_idct_residual(l, qp, r);
        }
    }
    return AEG_OK;
}

/* the reference the fused device kernel is checked against: literally the
 * two operations one after the other */
static int cpu_tq_iq_batch(void *priv, const int16_t *res, int nblk, int qp,
                           int intra, dctcoef *level, int32_t *nz,
                           int16_t *rec)
{
    cpu_tq_batch(priv, res, nblk, qp, intra, level, nz);
    return cpu_iq_idct_batch(priv, level, nblk, qp, rec);
}

/* the DC-split pair: exactly the per-block steps of code_luma_i16() and
 * code_chroma_run() */
static int cpu_tq_dc_batch(void *priv, const int16_t *res, int nblk, int qp,
                           int intra, dctcoef *level, int32_t *nz, int32_t *dc)
{
    int b;
    (void)priv;
    for (b = 0; b < nblk; b++) {
        dctcoef coef[16];
        int i, n = 0;
        aeg_transform_4x4(res + 16 * b, 4, coef);
        dc[b] = coef[0];
        coef[0] = 0;
        aeg_quant_4x4(coef, level + 16 * b, qp, intra);
        for (i = 0; i < 16; i++)
            n += level[16 * b + i] != 0;
        if (nz)
            nz[b] = n;
    }
    return AEG_OK;
}

static int cpu_iq_dc_batch(void *priv, const dctcoef *level, const int32_t *dc,
                           int nblk, int qp, int16_t *res)
{
    int b;
    (void)priv;
    for (b = 0; b < nblk; b++)
        aeg_idct_residual_dc(level + 16 * b, qp, dc[b], res + 16 * b);
    return AEG_OK;
}

static int cpu_sse_batch(void *priv, const pixel *a, int sa,
                         const pixel *b, int sb, int w, int h,
                         uint64_t *out)
{
    int tw = w / 8, th = h / 8, tx, ty, x, y;
    uint64_t sse = 0;
    (void)priv;
    for (ty = 0; ty < th; ty++)
        for (tx = 0; tx < tw; tx++)
            for (y = 0; y < 8; y++)
                for (x = 0; x < 8; x++) {
                    int d = (int)a[(ty * 8 + y) * sa + tx * 8 + x] -
                            (int)b[(ty * 8 + y) * sb + tx * 8 + x];
                    sse += (uint64_t)(d * d);
                }
    *out = sse;
    return AEG_OK;
}

/* One fractional-pel plane of the reference picture: the same function the
 * encoder's plane cache uses when the stage stays on the CPU, so the two
 * implementations of the operation cannot drift apart. */
static int cpu_interp_plane(void *priv, const pixel *ref, int rstride,
                            int w, int h, int which, pixel *dst, int dstride)
{
    (void)priv;
    aeg_hpel_plane_fill(dst, dstride, ref, rstride, w, h, which);
    return AEG_OK;
}

/* ------------------------------------------------------------------ */
/* Fractional-pel SATD batch (the subpel stage of the motion search)   */
/* ------------------------------------------------------------------ */
/* The reference picture and its planes are registered once per picture, like a
 * GPU would keep them in device memory; the CPU "registration" is the pointer
 * set itself.  It is per thread: with frame threads two pictures run their
 * searches at the same time, each against its own reference, and the
 * registration and the batches of one picture always run on one thread. */
static __thread struct {
    const pixel *base[4];
    int            stride;
} g_cpu_ref;

static int cpu_ref_upload(void *priv, const pixel *const planes[4],
                          int pstride, int w, int h)
{
    (void)priv; (void)w; (void)h;
    g_cpu_ref.base[0] = planes[0];
    g_cpu_ref.base[1] = planes[1];
    g_cpu_ref.base[2] = planes[2];
    g_cpu_ref.base[3] = planes[3];
    g_cpu_ref.stride = pstride;
    return AEG_OK;
}

/* Lowres ring (M-3.3a): the CPU reference keeps the host plane pointers the
 * encoder registered, exactly as the device keeps its buffers. */
static __thread struct {
    const uint8_t *plane;
    int stride, w, h;
    uint64_t id;
} g_cpu_lo[AEG_LOWRES_SLOTS];

static int cpu_lowres_upload(void *priv, int slot, const uint8_t *plane,
                             int stride, int w, int h, uint64_t id)
{
    (void)priv;
    if (slot < 0 || slot >= AEG_LOWRES_SLOTS)
        return AEG_EINVAL;
    g_cpu_lo[slot].plane = plane;
    g_cpu_lo[slot].stride = stride;
    g_cpu_lo[slot].w = w;
    g_cpu_lo[slot].h = h;
    g_cpu_lo[slot].id = id;
    return AEG_OK;
}

static int cpu_lowres_sad_batch(void *priv, int cur_slot, int ref_slot, int bw,
                                int bh, const int16_t *mv, int count,
                                int32_t *out)
{
    int i;
    (void)priv;
    if (cur_slot < 0 || cur_slot >= AEG_LOWRES_SLOTS ||
        ref_slot < 0 || ref_slot >= AEG_LOWRES_SLOTS ||
        !g_cpu_lo[cur_slot].plane || !g_cpu_lo[ref_slot].plane)
        return AEG_EINVAL;
    for (i = 0; i < count; i++) {
        const uint8_t *c = g_cpu_lo[cur_slot].plane +
                           (ptrdiff_t)mv[4 * i + 1] * g_cpu_lo[cur_slot].stride +
                           mv[4 * i + 0];
        const uint8_t *r = g_cpu_lo[ref_slot].plane +
                           (ptrdiff_t)mv[4 * i + 3] * g_cpu_lo[ref_slot].stride +
                           mv[4 * i + 2];
        int x, y, sad = 0;
        for (y = 0; y < bh; y++) {
            const uint8_t *cc = c + y * g_cpu_lo[cur_slot].stride;
            const uint8_t *rr = r + y * g_cpu_lo[ref_slot].stride;
            for (x = 0; x < bw; x++) {
                int d = (int)cc[x] - (int)rr[x];
                sad += d < 0 ? -d : d;
            }
        }
        out[i] = sad;
    }
    return AEG_OK;
}

/* M-3.3b CPU reference: fixed batched descent.  The same candidate order,
 * tie breaks and integer arithmetic are in the kernel. */
static int lb_se(int v)
{
    unsigned u = (v > 0 ? 2u * (unsigned)v - 1u : 2u * (unsigned)(-v)) + 1u;
    int n = 0;
    while (u) { n++; u >>= 1; }
    return 2 * n - 1;
}

static int lb_mv_cost(int mx, int my, int px, int py)
{
    return lb_se(2 * (mx - px)) + lb_se(2 * (my - py));
}

static int lb_median(int a, int b, int c)
{
    int mx = a > b ? a : b, mn = a < b ? a : b;
    return c > mx ? mx : c < mn ? mn : c;
}

static int lb_satd8(const uint8_t *c, int cs, const uint8_t *r, int rs)
{
    int sx, sy, x, y, sum = 0;
    for (sy = 0; sy < 8; sy += 4)
        for (sx = 0; sx < 8; sx += 4) {
            int t[16], d[16], acc = 0;
            for (y = 0; y < 4; y++) {
                int q[4];
                for (x = 0; x < 4; x++)
                    q[x] = (int)r[(sy + y) * rs + sx + x] -
                           (int)c[(sy + y) * cs + sx + x];
                {
                    int a0 = q[0] + q[1], a1 = q[0] - q[1];
                    int a2 = q[2] + q[3], a3 = q[2] - q[3];
                    t[y * 4 + 0] = a0 + a2; t[y * 4 + 1] = a1 + a3;
                    t[y * 4 + 2] = a0 - a2; t[y * 4 + 3] = a1 - a3;
                }
            }
            for (x = 0; x < 4; x++) {
                int a0 = t[0 * 4 + x] + t[1 * 4 + x];
                int a1 = t[0 * 4 + x] - t[1 * 4 + x];
                int a2 = t[2 * 4 + x] + t[3 * 4 + x];
                int a3 = t[2 * 4 + x] - t[3 * 4 + x];
                d[0 * 4 + x] = a0 + a2; d[1 * 4 + x] = a1 + a3;
                d[2 * 4 + x] = a0 - a2; d[3 * 4 + x] = a1 - a3;
            }
            for (y = 0; y < 16; y++)
                acc += d[y] < 0 ? -d[y] : d[y];
            sum += (acc + 1) >> 1;
        }
    return sum;
}

/* bilinear lowres prediction at a half-sample vector, then 8x8 SATD */
static int lb_cost_at(const uint8_t *cur, int stride, const uint8_t *ref,
                      int rstride, int bx, int by, int hx, int hy)
{
    uint8_t pred[64];
    int ix = bx + (hx >> 1), iy = by + (hy >> 1);
    int fx = hx & 1, fy = hy & 1, x, y;
    for (y = 0; y < 8; y++)
        for (x = 0; x < 8; x++) {
            int p = ref[(iy + y) * rstride + ix + x];
            if (fx && fy)
                p = (p + ref[(iy + y) * rstride + ix + x + 1] +
                     ref[(iy + y + 1) * rstride + ix + x] +
                     ref[(iy + y + 1) * rstride + ix + x + 1] + 2) >> 2;
            else if (fx)
                p = (p + ref[(iy + y) * rstride + ix + x + 1] + 1) >> 1;
            else if (fy)
                p = (p + ref[(iy + y + 1) * rstride + ix + x] + 1) >> 1;
            pred[y * 8 + x] = (uint8_t)p;
        }
    return lb_satd8(cur + by * stride + bx, stride, pred, 8);
}

static int cpu_lowres_batch_search(void *priv, const int32_t *pairs, int npairs,
                                   int mb_w, int mb_h, int iters,
                                   int32_t *mv, int32_t *cost)
{
    static const int steps[] = { 16, 16, 8, 8, 4, 4, 2, 2, 1, 1 };
    int nblk = mb_w * mb_h, n = npairs * nblk, p, i, it;
    int32_t *prev, *next;
    (void)priv;
    if (iters <= 0 || iters > (int)(sizeof(steps) / sizeof(steps[0])))
        iters = 5;
    if (npairs <= 0 || nblk <= 0)
        return AEG_OK;
    prev = (int32_t *)calloc((size_t)n * 2, sizeof(int32_t));
    next = (int32_t *)malloc((size_t)n * 2 * sizeof(int32_t));
    if (!prev || !next) {
        free(prev); free(next);
        return AEG_ENOMEM;
    }
    for (it = 0; it < iters; it++) {
        const int step = steps[it];
        for (p = 0; p < npairs; p++) {
            int cs = pairs[2 * p], rs = pairs[2 * p + 1];
            const uint8_t *cbase, *rbase;
            int cstride, rstride, bw, bh;
            if (cs < 0 || cs >= AEG_LOWRES_SLOTS || rs < 0 ||
                rs >= AEG_LOWRES_SLOTS || !g_cpu_lo[cs].plane ||
                !g_cpu_lo[rs].plane)
                continue;
            cbase = g_cpu_lo[cs].plane;
            rbase = g_cpu_lo[rs].plane;
            cstride = g_cpu_lo[cs].stride;
            rstride = g_cpu_lo[rs].stride;
            bw = g_cpu_lo[cs].w;
            bh = g_cpu_lo[cs].h;
            for (i = 0; i < nblk; i++) {
                int x = i % mb_w, y = i / mb_w, bx = 8 * x, by = 8 * y;
                int idx = p * nblk + i;
                int pxm = 0, pym = 0, nx = 0, ny = 0, k, best = 0x7fffffff;
                int lx = 0, ly = 0, tx = 0, ty = 0, rx = 0, ry = 0;
                int curx = prev[2 * idx], cury = prev[2 * idx + 1];
                int lo_x = -bx - AEG_LA_PAD + 1;
                int hi_x = bw - bx + AEG_LA_PAD - 9;
                int lo_y = -by - AEG_LA_PAD + 1;
                int hi_y = bh - by + AEG_LA_PAD - 9;
                if (x > 0) {
                    lx = prev[2 * (idx - 1)]; ly = prev[2 * (idx - 1) + 1];
                }
                if (y > 0) {
                    tx = prev[2 * (idx - mb_w)];
                    ty = prev[2 * (idx - mb_w) + 1];
                    if (x < mb_w - 1) {
                        rx = prev[2 * (idx - mb_w + 1)];
                        ry = prev[2 * (idx - mb_w + 1) + 1];
                    }
                }
                pxm = lb_median(lx, tx, rx);
                pym = lb_median(ly, ty, ry);
                {
                    int cand[14][2];
                    int nc = 0;
                    cand[nc][0] = curx; cand[nc][1] = cury; nc++;
                    cand[nc][0] = 0; cand[nc][1] = 0; nc++;
                    cand[nc][0] = lx; cand[nc][1] = ly; nc++;
                    cand[nc][0] = tx; cand[nc][1] = ty; nc++;
                    cand[nc][0] = rx; cand[nc][1] = ry; nc++;
                    cand[nc][0] = pxm; cand[nc][1] = pym; nc++;
                    cand[nc][0] = curx - step; cand[nc][1] = cury; nc++;
                    cand[nc][0] = curx + step; cand[nc][1] = cury; nc++;
                    cand[nc][0] = curx; cand[nc][1] = cury - step; nc++;
                    cand[nc][0] = curx; cand[nc][1] = cury + step; nc++;
                    cand[nc][0] = curx - step; cand[nc][1] = cury - step; nc++;
                    cand[nc][0] = curx + step; cand[nc][1] = cury - step; nc++;
                    cand[nc][0] = curx - step; cand[nc][1] = cury + step; nc++;
                    for (k = 0; k < nc; k++) {
                        int hx = cand[k][0], hy = cand[k][1];
                        int cst;
                        if (hx < 2 * lo_x || hx > 2 * hi_x ||
                            hy < 2 * lo_y || hy > 2 * hi_y)
                            continue;
                        cst = lb_cost_at(cbase, cstride, rbase, rstride, bx, by,
                                         hx, hy) + lb_mv_cost(hx, hy, pxm, pym);
                        if (cst < best) {
                            best = cst;
                            nx = hx; ny = hy;
                        }
                    }
                }
                next[2 * idx] = nx;
                next[2 * idx + 1] = ny;
                cost[idx] = best;
            }
        }
        memcpy(prev, next, (size_t)n * 2 * sizeof(int32_t));
    }
    memcpy(mv, prev, (size_t)n * 2 * sizeof(int32_t));
    free(prev);
    free(next);
    return AEG_OK;
}

/* M-3.2 CPU reference: the same fixed-point contributions as the encoder's
 * gather.  Integer addition is order-independent, so the scatter order here
 * does not change the value. */
static int cpu_mbt_batch(void *priv, int npics, int mb_w, int mb_h,
                         const int *intra, const int *inter,
                         const uint8_t *kind, const int *mv, const int *r0,
                         const int *r1, int w0, int32_t *prop)
{
    int nmb = mb_w * mb_h, p, i, l;
    (void)priv;
    for (p = npics - 1; p >= 0; p--)
        for (i = 0; i < nmb; i++) {
            int base = p * nmb + i;
            int ic = intra[base];
            int pc = inter[base] < ic ? inter[base] : ic;
            int64_t amount;
            if (ic <= 0 || pc >= ic)
                continue;
            amount = ((int64_t)prop[base] + (int64_t)ic * 64) * (ic - pc) / ic;
            for (l = 0; l < 2; l++) {
                int ref = l ? r1[p] : r0[p];
                int64_t a;
                int px, py, mx, my, fx, fy;
                if (ref < 0 || (kind[base] != 2 && kind[base] != l))
                    continue;
                a = kind[base] == 2 ? amount * (l ? 64 - w0 : w0) / 64 : amount;
                px = 16 * (i % mb_w) + mv[4 * base + 2 * l];
                py = 16 * (i / mb_w) + mv[4 * base + 2 * l + 1];
                mx = px >> 4; my = py >> 4; fx = px & 15; fy = py & 15;
#define MBT_C(X, Y, W) do {                                            \
            if ((X) >= 0 && (X) < mb_w && (Y) >= 0 && (Y) < mb_h && ((W) != 0)) \
                prop[ref * nmb + (Y) * mb_w + (X)] += (int32_t)(a * (W) / 256); \
        } while (0)
                MBT_C(mx, my, (16 - fx) * (16 - fy));
                MBT_C(mx + 1, my, fx * (16 - fy));
                MBT_C(mx, my + 1, (16 - fx) * fy);
                MBT_C(mx + 1, my + 1, fx * fy);
#undef MBT_C
            }
        }
    return AEG_OK;
}

/* Whole-picture integer motion search (see aeg_be_ops.me_batch).
 *
 * The walk is deliberately simple and fully specified, because the device
 * kernel has to reproduce it bit for bit: from the better of the start vector
 * and zero, a nine-point descent at step range/2, range/4 ... 1, repeating at
 * the same step while a neighbour improves, then a final one-pel descent.
 * Every comparison is strict (`<`) and the eight neighbours are visited in a
 * fixed order, so there is exactly one answer for a given picture. */
static const int8_t me_walk_d[8][2] = {
    {-1, 0}, { 1, 0}, { 0,-1}, { 0, 1}, {-1,-1}, { 1,-1}, {-1, 1}, { 1, 1}
};

static inline int me_sad_at(const pixel *cur, int cstride, const pixel *ref,
                            int rstride, int bw, int bh, int x, int y)
{
    int i, j, sad = 0;
    const pixel *r = ref + (ptrdiff_t)y * rstride + x;
    for (j = 0; j < bh; j++) {
        const pixel *c = cur + (ptrdiff_t)j * cstride;
        const pixel *rr = r + (ptrdiff_t)j * rstride;
        for (i = 0; i < bw; i++) {
            int d = (int)c[i] - (int)rr[i];
            sad += d < 0 ? -d : d;
        }
    }
    return sad;
}

static int cpu_rec_upload(void *priv, int slot, const pixel *y,
                          const pixel *u, const pixel *v, int ys, int cs,
                          int w, int h, uint64_t id)
{
    (void)priv; (void)slot; (void)y; (void)u; (void)v;
    (void)ys; (void)cs; (void)w; (void)h; (void)id;
    return AEG_OK;          /* the host reconstruction already exists */
}

static int cpu_me_batch(void *priv, const pixel *cur, int cstride,
                        const pixel *ref, int rstride, int bw, int bh,
                        int range, const int16_t *job, int count,
                        int avail_rows, int16_t *out_mv, int32_t *out_sad,
                        uint64_t cur_id, uint64_t ref_id)
{
    int b;
    (void)priv; (void)avail_rows; (void)cur_id; (void)ref_id;
    for (b = 0; b < count; b++) {
        const int ox = job[4 * b + 0], oy = job[4 * b + 1];
        int sx = job[4 * b + 2], sy = job[4 * b + 3];
        const pixel *c = cur + (ptrdiff_t)oy * cstride + ox;
        /* the reference block's base: the block's own position plus the
         * displacement, so `ref` here is the picture origin and the search
         * offsets it per candidate (it used to compare against the origin,
         * which is only right for the macroblock at 0,0) */
        const pixel *r = ref + (ptrdiff_t)oy * rstride + ox;
        int bx = 0, by = 0, best, step;

        if (sx < -range) sx = -range;
        if (sx > range) sx = range;
        if (sy < -range) sy = -range;
        if (sy > range) sy = range;
        best = me_sad_at(c, cstride, r, rstride, bw, bh, 0, 0);
        {
            int d = me_sad_at(c, cstride, r, rstride, bw, bh, sx, sy);
            if (d < best) { best = d; bx = sx; by = sy; }
        }
        for (step = range / 2; step >= 1; step /= 2) {
            int improved = 1;
            while (improved) {
                int k;
                improved = 0;
                for (k = 0; k < 8; k++) {
                    int nx = bx + me_walk_d[k][0] * step;
                    int ny = by + me_walk_d[k][1] * step;
                    int d;
                    if (nx < -range || nx > range || ny < -range || ny > range)
                        continue;
                    d = me_sad_at(c, cstride, r, rstride, bw, bh, nx, ny);
                    if (d < best) { best = d; bx = nx; by = ny; improved = 1; }
                }
            }
        }
        out_mv[2 * b + 0] = (int16_t)bx;
        out_mv[2 * b + 1] = (int16_t)by;
        out_sad[b] = best;
    }
    return AEG_OK;
}

static int cpu_satd_batch(void *priv, const pixel *cur, int cstride, int bw,
                          int bh, const int16_t *pos, int count, int32_t *out)
{
    aeg_mc_ctx mc;
    pixel buf[16 * 16];
    int i;
    (void)priv;
    if (bw > 16 || bh > 16)
        return AEG_EINVAL;
    mc.base[0] = g_cpu_ref.base[0];
    mc.base[1] = g_cpu_ref.base[1];
    mc.base[2] = g_cpu_ref.base[2];
    mc.base[3] = g_cpu_ref.base[3];
    mc.buf = NULL;
    mc.stride = g_cpu_ref.stride;
    mc.ready = 1;
    for (i = 0; i < count; i++) {
        const int bx = pos[4 * i + 0], by = pos[4 * i + 1];
        const int mvx = pos[4 * i + 2], mvy = pos[4 * i + 3];
        aeg_mc_luma_p(buf, bw, &mc,
                      g_cpu_ref.base[0] + (ptrdiff_t)(by + (mvy >> 2)) * mc.stride +
                      bx + (mvx >> 2),
                      bw, bh, mvx & 3, mvy & 3);
        out[i] = aeg_satd(bw, bh, buf, bw, cur + (ptrdiff_t)by * cstride + bx,
                          cstride);
    }
    return AEG_OK;
}

void aeg_be_cpu_ops(aeg_be_ops *ops)
{
    memset(ops, 0, sizeof(*ops));
    ops->backend = AEG_BACKEND_CPU;
    ops->is_gpu = 0;
    ops->sad_batch = cpu_sad_batch;
    ops->tq_batch = cpu_tq_batch;
    ops->iq_idct_batch = cpu_iq_idct_batch;
    ops->tq_iq_batch = cpu_tq_iq_batch;
    ops->tq_dc_batch = cpu_tq_dc_batch;
    ops->iq_dc_batch = cpu_iq_dc_batch;
    ops->sse_batch = cpu_sse_batch;
    ops->interp_plane = cpu_interp_plane;
    ops->ref_upload = cpu_ref_upload;
    ops->satd_batch = cpu_satd_batch;
    ops->me_batch = cpu_me_batch;
    ops->rec_upload = cpu_rec_upload;
    ops->lowres_upload = cpu_lowres_upload;
    ops->lowres_sad_batch = cpu_lowres_sad_batch;
    ops->lowres_batch_search = cpu_lowres_batch_search;
    ops->mbt_batch = cpu_mbt_batch;
    ops->me_pic_upload = NULL;   /* no device: the host arrays are the picture */
    snprintf(ops->desc, sizeof(ops->desc), "cpu");
}

/* Implemented in backend_opencl.c / backend_cuda.c */
int aeg_be_opencl_create(aeg_be_ops *ops);
int aeg_be_cuda_create(aeg_be_ops *ops);

/* ------------------------------------------------------------------ */
/* Device guard                                                        */
/* ------------------------------------------------------------------ */
/* Every device backend is wrapped in this before the encoder sees it, for
 * three reasons the backends themselves do not handle:
 *
 *  1. Threads.  Frame threads and slice threads are copies of the encoder,
 *     and a backend keeps its staging buffers and kernel arguments in its
 *     instance, so two calls at once into one instance corrupt each other (on
 *     a real OpenCL device: a crash on the first picture with -t 2).  The
 *     guard has *lanes*: lane 0 is the backend as created, lane k a clone of
 *     it (aeg_be_ops.clone: same device, shared context and program, own
 *     queue, kernel objects and buffers), made when a thread first asks for
 *     it (aeg_be_lane).  Each lane has its own lock, so threads on different
 *     lanes run their device calls concurrently and threads that share a lane
 *     are serialised.  A backend without clone() maps every lane to lane 0.
 *  2. The registered reference (ref_upload, then satd_batch) is device state
 *     that outlives a call.  Two threads on one lane would read each other's
 *     reference, so each lane remembers, per thread, what that thread
 *     registered and re-uploads it before a batch whenever another thread's
 *     registration replaced it.
 *  3. A failing device call must not change the stream.  The encoder ignores
 *     the return codes of the batch operations (they cannot fail on the CPU),
 *     so the guard runs the CPU reference instead -- the same arithmetic, so
 *     the same bytes -- and keeps the device off from then on, on every lane. */
#define BE_LANES 16

typedef struct be_guard be_guard;

typedef struct be_lane {
    be_guard       *g;
    aeg_be_ops      dev;        /* this lane's device instance             */
    int             owned;      /* a clone made for this lane              */
    pthread_mutex_t mu;
    uint64_t        seq;        /* registrations so far                     */
    uint64_t        dev_seq;    /* the one resident on the device           */
} be_lane;

struct be_guard {
    aeg_be_ops      cpu;
    pthread_mutex_t mk;         /* lane creation                            */
    be_lane        *lane[BE_LANES];
    int             failed;     /* a device call failed: CPU from now on    */
    int             fail_streak; /* consecutive failures (transient ones retry) */
};

static __thread struct {
    const be_lane  *l;
    uint64_t        seq;
    const pixel  *planes[4];
    int             stride, w, h;
} g_tls_ref[BE_LANES];

static int guard_failed(be_guard *g)
{
    return __atomic_load_n(&g->failed, __ATOMIC_ACQUIRE);
}

/* lock, or return 0 when the device is off and the CPU op has to run */
static int guard_enter(be_lane *l)
{
    if (guard_failed(l->g))
        return 0;
    pthread_mutex_lock(&l->mu);
    if (guard_failed(l->g)) {
        pthread_mutex_unlock(&l->mu);
        return 0;
    }
    return 1;
}

/* unlock.  A single failed call may be transient (a device reset, a
 * temporarily exhausted allocation), so the call is retried on the next use;
 * three consecutive failures disable the device for good.  The fallback ran
 * the CPU reference either way, so the stream is unaffected. */
static int guard_leave(be_lane *l, int rc, const char *op)
{
    if (rc != AEG_OK) {
        int n = __atomic_add_fetch(&l->g->fail_streak, 1, __ATOMIC_ACQ_REL);
        if (n >= 3 && !__atomic_exchange_n(&l->g->failed, 1, __ATOMIC_ACQ_REL))
            aeg_log(AEG_LOG_WARNING, "aeg: device %s keeps failing (%d), "
                    "continuing on the CPU", op, rc);
        else if (n == 1)
            aeg_log(AEG_LOG_WARNING, "aeg: device %s failed (%d), retrying; "
                    "this call ran on the CPU", op, rc);
    } else {
        __atomic_store_n(&l->g->fail_streak, 0, __ATOMIC_RELEASE);
    }
    pthread_mutex_unlock(&l->mu);
    return rc == AEG_OK;
}

/* the slot of g_tls_ref for lane l.  A thread normally uses one lane; two
 * lanes on one slot (another encoder in the same thread, a hash collision)
 * evict each other, and an evicted registration only sends satd_batch to the
 * CPU reference -- the same bytes. */
static int tls_slot(const be_lane *l)
{
    return (int)(((uintptr_t)l >> 6) % BE_LANES);
}

static int g_sad_batch(void *priv, const pixel *cur, int cstride,
                       const pixel *ref, int rstride, int bw, int bh,
                       const int16_t *mv, int count, int32_t *out)
{
    be_lane *l = (be_lane *)priv;
    if (guard_enter(l) &&
        guard_leave(l, l->dev.sad_batch(l->dev.priv, cur, cstride, ref, rstride,
                                        bw, bh, mv, count, out), "sad_batch"))
        return AEG_OK;
    return l->g->cpu.sad_batch(l->g->cpu.priv, cur, cstride, ref, rstride, bw,
                               bh, mv, count, out);
}

static int g_me_batch(void *priv, const pixel *cur, int cstride,
                      const pixel *ref, int rstride, int bw, int bh, int range,
                      const int16_t *job, int count, int avail_rows,
                      int16_t *out_mv, int32_t *out_sad,
                      uint64_t cur_id, uint64_t ref_id)
{
    be_lane *l = (be_lane *)priv;
    if (guard_enter(l) &&
        guard_leave(l, l->dev.me_batch(l->dev.priv, cur, cstride, ref, rstride,
                                       bw, bh, range, job, count, avail_rows,
                                       out_mv, out_sad, cur_id, ref_id),
                    "me_batch"))
        return AEG_OK;
    return l->g->cpu.me_batch(l->g->cpu.priv, cur, cstride, ref, rstride, bw, bh,
                              range, job, count, avail_rows, out_mv, out_sad,
                              cur_id, ref_id);
}

static int g_me_pic_upload(void *priv, int slot, const pixel *plane, int stride,
                           int w, int h, int rows, uint64_t id)
{
    be_lane *l = (be_lane *)priv;
    if (guard_enter(l))
        guard_leave(l, l->dev.me_pic_upload(l->dev.priv, slot, plane, stride,
                                            w, h, rows, id), "me_pic_upload");
    return AEG_OK;                /* the host picture is still valid */
}

static int g_rec_upload(void *priv, int slot, const pixel *y, const pixel *u,
                        const pixel *v, int ys, int cs, int w, int h,
                        uint64_t id)
{
    be_lane *l = (be_lane *)priv;
    l->g->cpu.rec_upload(l->g->cpu.priv, slot, y, u, v, ys, cs, w, h, id);
    if (guard_enter(l))
        guard_leave(l, l->dev.rec_upload(l->dev.priv, slot, y, u, v, ys, cs,
                                         w, h, id), "rec_upload");
    return AEG_OK;
}

static int g_lowres_upload(void *priv, int slot, const uint8_t *plane,
                           int stride, int w, int h, uint64_t id)
{
    be_lane *l = (be_lane *)priv;
    /* the CPU reference keeps the host plane even when the device is off */
    l->g->cpu.lowres_upload(l->g->cpu.priv, slot, plane, stride, w, h, id);
    if (guard_enter(l))
        guard_leave(l, l->dev.lowres_upload(l->dev.priv, slot, plane, stride,
                                            w, h, id), "lowres_upload");
    return AEG_OK;
}

static int g_lowres_sad_batch(void *priv, int cur_slot, int ref_slot, int bw,
                              int bh, const int16_t *mv, int count,
                              int32_t *out)
{
    be_lane *l = (be_lane *)priv;
    if (guard_enter(l) &&
        guard_leave(l, l->dev.lowres_sad_batch(l->dev.priv, cur_slot, ref_slot,
                                               bw, bh, mv, count, out),
                    "lowres_sad_batch"))
        return AEG_OK;
    return l->g->cpu.lowres_sad_batch(l->g->cpu.priv, cur_slot, ref_slot, bw, bh,
                                      mv, count, out);
}

static int g_lowres_batch_search(void *priv, const int32_t *pairs, int npairs,
                                 int mb_w, int mb_h, int iters, int32_t *mv,
                                 int32_t *cost)
{
    be_lane *l = (be_lane *)priv;
    if (guard_enter(l) &&
        guard_leave(l, l->dev.lowres_batch_search(l->dev.priv, pairs, npairs,
                                                  mb_w, mb_h, iters, mv, cost),
                    "lowres_batch_search"))
        return AEG_OK;
    return l->g->cpu.lowres_batch_search(l->g->cpu.priv, pairs, npairs, mb_w,
                                         mb_h, iters, mv, cost);
}

static int g_mbt_batch(void *priv, int npics, int mb_w, int mb_h,
                       const int *intra, const int *inter,
                       const uint8_t *kind, const int *mv, const int *r0,
                       const int *r1, int w0, int32_t *prop)
{
    be_lane *l = (be_lane *)priv;
    if (guard_enter(l) &&
        guard_leave(l, l->dev.mbt_batch(l->dev.priv, npics, mb_w, mb_h, intra,
                                        inter, kind, mv, r0, r1, w0, prop),
                    "mbt_batch"))
        return AEG_OK;
    return l->g->cpu.mbt_batch(l->g->cpu.priv, npics, mb_w, mb_h, intra, inter,
                               kind, mv, r0, r1, w0, prop);
}

static int g_tq_batch(void *priv, const int16_t *res, int nblk, int qp,
                      int intra, dctcoef *level, int32_t *nz)
{
    be_lane *l = (be_lane *)priv;
    if (guard_enter(l) &&
        guard_leave(l, l->dev.tq_batch(l->dev.priv, res, nblk, qp, intra, level,
                                       nz), "tq_batch"))
        return AEG_OK;
    return l->g->cpu.tq_batch(l->g->cpu.priv, res, nblk, qp, intra, level, nz);
}

static int g_iq_idct_batch(void *priv, const dctcoef *level, int nblk, int qp,
                           int16_t *res)
{
    be_lane *l = (be_lane *)priv;
    if (guard_enter(l) &&
        guard_leave(l, l->dev.iq_idct_batch(l->dev.priv, level, nblk, qp, res),
                    "iq_idct_batch"))
        return AEG_OK;
    return l->g->cpu.iq_idct_batch(l->g->cpu.priv, level, nblk, qp, res);
}

static int g_tq_iq_batch(void *priv, const int16_t *res, int nblk, int qp,
                         int intra, dctcoef *level, int32_t *nz, int16_t *rec)
{
    be_lane *l = (be_lane *)priv;
    if (guard_enter(l) &&
        guard_leave(l, l->dev.tq_iq_batch(l->dev.priv, res, nblk, qp, intra,
                                          level, nz, rec), "tq_iq_batch"))
        return AEG_OK;
    return l->g->cpu.tq_iq_batch(l->g->cpu.priv, res, nblk, qp, intra, level,
                                 nz, rec);
}

static int g_tq_dc_batch(void *priv, const int16_t *res, int nblk, int qp,
                         int intra, dctcoef *level, int32_t *nz, int32_t *dc)
{
    be_lane *l = (be_lane *)priv;
    if (guard_enter(l) &&
        guard_leave(l, l->dev.tq_dc_batch(l->dev.priv, res, nblk, qp, intra,
                                          level, nz, dc), "tq_dc_batch"))
        return AEG_OK;
    return l->g->cpu.tq_dc_batch(l->g->cpu.priv, res, nblk, qp, intra, level,
                                 nz, dc);
}

static int g_iq_dc_batch(void *priv, const dctcoef *level, const int32_t *dc,
                         int nblk, int qp, int16_t *res)
{
    be_lane *l = (be_lane *)priv;
    if (guard_enter(l) &&
        guard_leave(l, l->dev.iq_dc_batch(l->dev.priv, level, dc, nblk, qp, res),
                    "iq_dc_batch"))
        return AEG_OK;
    return l->g->cpu.iq_dc_batch(l->g->cpu.priv, level, dc, nblk, qp, res);
}

static int g_interp_plane(void *priv, const pixel *ref, int rstride, int w,
                          int h, int which, pixel *dst, int dstride)
{
    be_lane *l = (be_lane *)priv;
    if (guard_enter(l) &&
        guard_leave(l, l->dev.interp_plane(l->dev.priv, ref, rstride, w, h,
                                           which, dst, dstride),
                    "interp_plane"))
        return AEG_OK;
    return l->g->cpu.interp_plane(l->g->cpu.priv, ref, rstride, w, h, which,
                                  dst, dstride);
}

static int g_sse_batch(void *priv, const pixel *a, int sa, const pixel *b,
                       int sb, int w, int h, uint64_t *out)
{
    be_lane *l = (be_lane *)priv;
    if (guard_enter(l) &&
        guard_leave(l, l->dev.sse_batch(l->dev.priv, a, sa, b, sb, w, h, out),
                    "sse_batch"))
        return AEG_OK;
    return l->g->cpu.sse_batch(l->g->cpu.priv, a, sa, b, sb, w, h, out);
}

static int g_ref_upload(void *priv, const pixel *const planes[4], int pstride,
                        int w, int h)
{
    be_lane *l = (be_lane *)priv;
    int k, t = tls_slot(l);
    /* the CPU registration too, so that a fallback has a reference */
    l->g->cpu.ref_upload(l->g->cpu.priv, planes, pstride, w, h);
    for (k = 0; k < 4; k++)
        g_tls_ref[t].planes[k] = planes[k];
    g_tls_ref[t].stride = pstride;
    g_tls_ref[t].w = w;
    g_tls_ref[t].h = h;
    g_tls_ref[t].l = l;
    g_tls_ref[t].seq = 0;
    if (!guard_enter(l))
        return AEG_OK;
    k = l->dev.ref_upload(l->dev.priv, planes, pstride, w, h);
    if (k == AEG_OK)
        l->dev_seq = g_tls_ref[t].seq = ++l->seq;
    guard_leave(l, k, "ref_upload");
    return AEG_OK;
}

static int g_satd_batch(void *priv, const pixel *cur, int cstride, int bw,
                        int bh, const int16_t *pos, int count, int32_t *out)
{
    be_lane *l = (be_lane *)priv;
    int t = tls_slot(l);
    if (g_tls_ref[t].l == l && guard_enter(l)) {
        int rc = AEG_OK;
        if (g_tls_ref[t].seq != l->dev_seq) {
            /* another thread on this lane registered its reference since */
            rc = l->dev.ref_upload(l->dev.priv, g_tls_ref[t].planes,
                                   g_tls_ref[t].stride, g_tls_ref[t].w,
                                   g_tls_ref[t].h);
            if (rc == AEG_OK)
                l->dev_seq = g_tls_ref[t].seq = ++l->seq;
        }
        if (rc == AEG_OK)
            rc = l->dev.satd_batch(l->dev.priv, cur, cstride, bw, bh, pos, count,
                                   out);
        if (guard_leave(l, rc, "satd_batch"))
            return AEG_OK;
    }
    return l->g->cpu.satd_batch(l->g->cpu.priv, cur, cstride, bw, bh, pos, count,
                                out);
}

static void g_sync(void *priv)
{
    be_lane *l = (be_lane *)priv;
    if (!guard_enter(l))
        return;
    if (l->dev.sync)
        l->dev.sync(l->dev.priv);
    pthread_mutex_unlock(&l->mu);
}

static be_lane *lane_new(be_guard *g, const aeg_be_ops *dev, int owned)
{
    be_lane *l = (be_lane *)calloc(1, sizeof(*l));
    if (!l)
        return NULL;
    l->g = g;
    l->dev = *dev;
    l->owned = owned;
    pthread_mutex_init(&l->mu, NULL);
    return l;
}

static void g_destroy(void *priv)
{
    be_lane *l0 = (be_lane *)priv;
    be_guard *g;
    int k;
    if (!l0)
        return;
    g = l0->g;
    /* the clones first: they share the first instance's context */
    for (k = BE_LANES - 1; k >= 0; k--) {
        be_lane *l = g->lane[k];
        if (!l || (k > 0 && l == l0))
            continue;
        if ((k == 0 || l->owned) && l->dev.destroy)
            l->dev.destroy(l->dev.priv);
        pthread_mutex_destroy(&l->mu);
        free(l);
    }
    pthread_mutex_destroy(&g->mk);
    free(g);
}

static void g_lane_destroy(void *priv)
{
    (void)priv;                 /* lanes go with lane 0 (g_destroy) */
}

void aeg_be_lane(const aeg_be_ops *ops, int k, aeg_be_ops *out)
{
    be_lane *l0, *l;
    be_guard *g;

    *out = *ops;
    if (!ops->priv || (ops->destroy != g_destroy &&
                       ops->destroy != g_lane_destroy))
        return;                 /* the CPU backend: nothing to split */
    g = ((be_lane *)ops->priv)->g;
    l0 = g->lane[0];
    k = k < 0 ? 0 : k % BE_LANES;
    {
        /* AEG_LANES=0: every thread on lane 0, serialised (the measurement
         * of what the lanes buy, and a way out should a driver mishandle
         * several queues) */
        const char *ev = getenv("AEG_LANES");
        if (ev && atoi(ev) == 0)
            k = 0;
    }
    pthread_mutex_lock(&g->mk);
    if (!g->lane[k]) {
        aeg_be_ops c;
        l = NULL;
        if (!guard_failed(g) && l0->dev.clone &&
            l0->dev.clone(l0->dev.priv, &c) == AEG_OK) {
            l = lane_new(g, &c, 1);
            if (!l && c.destroy)
                c.destroy(c.priv);
        }
        if (l)
            aeg_log(AEG_LOG_DEBUG, "aeg: device lane %d: %s", k, c.desc);
        else if (l0->dev.clone)
            aeg_log(AEG_LOG_DEBUG, "aeg: device lane %d is lane 0 (%s)", k, aeg_last_error());
        g->lane[k] = l ? l : l0;
    }
    l = g->lane[k];
    pthread_mutex_unlock(&g->mk);
    /* every table has the same wrappers; only priv and destroy differ */
    if (l != ops->priv) {
        out->priv = l;
        out->destroy = g_lane_destroy;
    }
}

/* Wrap a freshly created device backend; an operation the device does not
 * provide stays absent, exactly as before. */
static int be_guard_wrap(aeg_be_ops *ops)
{
    be_guard *g = (be_guard *)calloc(1, sizeof(*g));
    be_lane *l0;
    if (!g)
        return AEG_ENOMEM;
    aeg_be_cpu_ops(&g->cpu);
    pthread_mutex_init(&g->mk, NULL);
    l0 = lane_new(g, ops, 0);
    if (!l0) {
        pthread_mutex_destroy(&g->mk);
        free(g);
        return AEG_ENOMEM;
    }
    g->lane[0] = l0;
    ops->priv = l0;
    if (ops->sad_batch) ops->sad_batch = g_sad_batch;
    if (ops->me_batch) ops->me_batch = g_me_batch;
    if (ops->me_pic_upload) ops->me_pic_upload = g_me_pic_upload;
    if (ops->rec_upload) ops->rec_upload = g_rec_upload;
    if (ops->lowres_upload) ops->lowres_upload = g_lowres_upload;
    if (ops->lowres_sad_batch) ops->lowres_sad_batch = g_lowres_sad_batch;
    if (ops->lowres_batch_search) ops->lowres_batch_search = g_lowres_batch_search;
    if (ops->mbt_batch) ops->mbt_batch = g_mbt_batch;
    if (ops->tq_batch) ops->tq_batch = g_tq_batch;
    if (ops->iq_idct_batch) ops->iq_idct_batch = g_iq_idct_batch;
    if (ops->tq_iq_batch) ops->tq_iq_batch = g_tq_iq_batch;
    if (ops->tq_dc_batch && ops->iq_dc_batch) {
        ops->tq_dc_batch = g_tq_dc_batch;
        ops->iq_dc_batch = g_iq_dc_batch;
    } else {
        ops->tq_dc_batch = NULL;
        ops->iq_dc_batch = NULL;
    }
    if (ops->interp_plane) ops->interp_plane = g_interp_plane;
    if (ops->sse_batch) ops->sse_batch = g_sse_batch;
    if (ops->ref_upload && ops->satd_batch) {
        ops->ref_upload = g_ref_upload;
        ops->satd_batch = g_satd_batch;
    } else {
        ops->ref_upload = NULL;
        ops->satd_batch = NULL;
    }
    ops->sync = g_sync;
    ops->destroy = g_destroy;
    ops->clone = NULL;
    return AEG_OK;
}

static int be_create_raw(int which, aeg_be_ops *ops);

int aeg_be_create(int which, aeg_be_ops *ops)
{
    int rc = be_create_raw(which, ops);
    if (ops->backend != AEG_BACKEND_CPU && be_guard_wrap(ops) != AEG_OK) {
        if (ops->destroy)
            ops->destroy(ops->priv);
        aeg_be_cpu_ops(ops);
        return AEG_EBACKEND;
    }
    return rc;
}

static int be_create_raw(int which, aeg_be_ops *ops)
{
    aeg_be_cpu_ops(ops);
#if BIT_DEPTH != 8
    /* the device kernels are 8-bit: a high-depth encoder runs on the CPU */
    if (which == AEG_BACKEND_OPENCL || which == AEG_BACKEND_CUDA) {
        aeg_log(AEG_LOG_WARNING, "aeg: %s requested, but the device kernels "
                "are 8-bit: this 10-bit encoder runs on the CPU",
                aeg_backend_name(which));
        aeg_set_last_error("GPU backends support 8-bit only, using CPU");
        return AEG_EBACKEND;
    }
    return AEG_OK;
#endif
    switch (which) {
    case AEG_BACKEND_CPU:
        return AEG_OK;
    case AEG_BACKEND_OPENCL:
        if (aeg_be_opencl_create(ops) == AEG_OK)
            return AEG_OK;
        aeg_be_cpu_ops(ops);
        aeg_set_last_error("OpenCL backend unavailable, using CPU");
        return AEG_EBACKEND;
    case AEG_BACKEND_CUDA:
        if (aeg_be_cuda_create(ops) == AEG_OK)
            return AEG_OK;
        aeg_be_cpu_ops(ops);
        aeg_set_last_error("CUDA backend unavailable, using CPU");
        return AEG_EBACKEND;
    case AEG_BACKEND_AUTO:
    default:
        if (aeg_be_cuda_create(ops) == AEG_OK)
            return AEG_OK;
        aeg_be_cpu_ops(ops);
        if (aeg_be_opencl_create(ops) == AEG_OK)
            return AEG_OK;
        aeg_be_cpu_ops(ops);
        return AEG_OK;
    }
}

/* ------------------------------------------------------------------ */
/* Dispatcher                                                          */
/* ------------------------------------------------------------------ */
void aeg_dispatch_init(aeg_dispatcher *d, int backend, const aeg_params *p)
{
    int i;
    memset(d, 0, sizeof(*d));
    aeg_be_create(backend, &d->ops);
    d->have_gpu = d->ops.backend != AEG_BACKEND_CPU &&
                  d->ops.tq_batch != NULL;
    /* Automatic placement needs a real GPU.  A CPU *device* (PoCL, an
     * OpenCL implementation that runs on the CPU) executes the same work on
     * the same cores through a runtime and an extra copy, so it can never
     * win a measurement -- and letting the dispatcher measure it anyway
     * costs a slow picture or two before it settles.  Such a device is still
     * reachable with -s, which is how the device paths are tested without
     * hardware. */
    d->auto_ok = d->have_gpu && d->ops.is_gpu;
    for (i = 0; i < AEG_STAGE_COUNT; i++) {
        d->forced[i] = p ? p->stage_backend[i] : -1;
        /* -1 is "auto"; AEG_BACKEND_AUTO (0) written explicitly in a stage
         * slot also means auto -- it used to mean "force the CPU", because
         * the test below was `forced >= 0`. */
        if (d->forced[i] == AEG_BACKEND_AUTO)
            d->forced[i] = -1;
        d->use_gpu[i] = 0;
    }
    /* Stages that are inherently serial (they consume the result of the
     * previous macroblock as they go) are always kept on the CPU: the
     * latency of a device round trip would be paid per macroblock. */
    d->use_gpu[AEG_STAGE_DEBLOCK] = 0;
    d->mu = malloc(sizeof(pthread_mutex_t));
    if (d->mu)
        pthread_mutex_init((pthread_mutex_t *)d->mu, NULL);
    d->launch_ns = 5 * 1000;      /* refined by aeg_backend_calibrate() */
    d->min_items_for_gpu = 256;
    aeg_log(AEG_LOG_INFO, "aeg: compute backend = %s (%s)", aeg_backend_name(d->ops.backend),
            d->ops.desc);
    if (d->have_gpu && !d->auto_ok)
        aeg_log(AEG_LOG_INFO, "aeg: %s is a CPU device: the stages run on the "
                "host unless one is forced with -s", d->ops.desc);
}

void aeg_dispatch_destroy(aeg_dispatcher *d)
{
    if (d->ops.destroy)
        d->ops.destroy(d->ops.priv);
    d->ops.destroy = NULL;
    d->ops.priv = NULL;
    if (d->mu && !d->root) {
        pthread_mutex_destroy((pthread_mutex_t *)d->mu);
        free(d->mu);
    }
    d->mu = NULL;
}

/* the dispatcher measurements go to, locked */
static aeg_dispatcher *disp_lock(aeg_dispatcher *d)
{
    aeg_dispatcher *r = d->root ? d->root : d;
    if (r->mu)
        pthread_mutex_lock((pthread_mutex_t *)r->mu);
    return r;
}

static void disp_unlock(aeg_dispatcher *r)
{
    if (r->mu)
        pthread_mutex_unlock((pthread_mutex_t *)r->mu);
}

void aeg_dispatch_lock(aeg_dispatcher *d)
{
    disp_lock(d);
}

void aeg_dispatch_unlock(aeg_dispatcher *d)
{
    disp_unlock(d->root ? d->root : d);
}

static void dispatch_record(aeg_dispatcher *d, int stage, int on_gpu,
                            int64_t ns)
{
    int64_t *slot;
    slot = on_gpu ? &d->gpu_ns[stage][1] : &d->cpu_ns[stage][1];
    if (*slot == 0)
        *slot = ns;
    else
        *slot = (*slot * 7 + ns) / 8;      /* EMA */
    d->cpu_ns[stage][0] += on_gpu ? 0 : ns;
    d->gpu_ns[stage][0] += on_gpu ? ns : 0;
}

void aeg_dispatch_record(aeg_dispatcher *d, int stage, int on_gpu, int64_t ns)
{
    if (stage < 0 || stage >= AEG_STAGE_COUNT)
        return;
    d = disp_lock(d);
    dispatch_record(d, stage, on_gpu, ns);
    disp_unlock(d);
}

static void dispatch_update(aeg_dispatcher *d)
{
    int i;
    d->frames_measured++;
    for (i = 0; i < AEG_STAGE_COUNT; i++) {
        if (d->forced[i] >= 0) {
            d->use_gpu[i] = d->forced[i] == AEG_BACKEND_OPENCL ||
                            d->forced[i] == AEG_BACKEND_CUDA;
            if (d->use_gpu[i] && !d->have_gpu)
                d->use_gpu[i] = 0;
            continue;
        }
        if (!d->auto_ok || i == AEG_STAGE_DEBLOCK) {
            d->use_gpu[i] = 0;
            continue;
        }
        /* The device motion search (aeg_be_ops.me_batch) is the same walk as
         * its CPU reference, so placing it is a pure speed decision and the
         * stream does not depend on where it ran.  A real GPU is worth using
         * for it unconditionally: it is the stage that dominates the encode
         * (31-80 % of CPU time depending on the preset), and the whole-picture
         * batch is what the device is good at.  PoCL/CPU devices are still
         * excluded by auto_ok. */
        if (i == AEG_STAGE_MOTION_EST) {
            d->use_gpu[i] = 1;
            continue;
        }
        /* the coefficient stages are placed by aeg_dispatch_coef_pick() */
        if (i == AEG_STAGE_TRANSFORM || i == AEG_STAGE_IQUANT_IDCT)
            continue;
        /* Only switch to the GPU when the measurement is unambiguous
         * (>= 10% faster) and the frame had enough work to amortise a
         * launch; this includes the launch cost in the comparison. */
        if (d->gpu_ns[i][1] > 0 && d->cpu_ns[i][1] > 0) {
            int64_t gpu = d->gpu_ns[i][1] + d->launch_ns;
            if (gpu * 100 < d->cpu_ns[i][1] * 90)
                d->use_gpu[i] = 1;
            else if (d->cpu_ns[i][1] * 100 < gpu * 90)
                d->use_gpu[i] = 0;
        }
        d->cpu_ns[i][0] = 0;
        d->gpu_ns[i][0] = 0;
    }
}

void aeg_dispatch_update(aeg_dispatcher *d)
{
    if (d->root)
        return;
    disp_lock(d);
    dispatch_update(d);
    disp_unlock(d);
}

/* Placing the coefficient stages.
 *
 * These two stages are placed on the CPU unless a device placement is asked
 * for explicitly (`-s tq=<be> -s iq=<be>`, or AEG_COEF_BATCH=1).  That is not
 * a heuristic any more, it is a measurement: the batched form is one round
 * trip per macroblock (sixteen 4x4 blocks in, sixteen levels and sixteen
 * residuals out), so 1920x1080 pays ~8000 blocking copies, and on a GeForce
 * RTX 2060 SUPER the whole encode runs **2.9x slower** with it than without
 * (docs/DESIGN_ja.md, "Measured: what the device does today").  The CPU form
 * of the same work is sixteen SSE2 transforms inline in the candidate's
 * coding, ~0.4 us per macroblock.
 *
 * The trial below is kept because it is the right mechanism for a device
 * placement that *is* worth comparing -- one that batches a whole picture
 * instead of a macroblock (DESIGN_ja.md, M-3).  AEG_COEF_PROBE=1 turns it
 * back on, which is also how the numbers above were taken. */
#define AEG_COEF_PROBE 3        /* counted pictures per placement and kind */
#define AEG_COEF_RETRY 64

static int coef_probe_on(void)
{
    static _Atomic int v = -2;
    if (v == -2) {
        const char *ev = getenv("AEG_COEF_PROBE");
        v = ev ? atoi(ev) != 0 : 0;
    }
    return v;
}

static int dispatch_coef_pick(aeg_dispatcher *d, int kind)
{
    if (kind < 0 || kind > 2)
        return 0;
    if (d->forced[AEG_STAGE_TRANSFORM] >= 0 ||
        d->forced[AEG_STAGE_IQUANT_IDCT] >= 0)
        return d->use_gpu[AEG_STAGE_TRANSFORM] &&
               d->use_gpu[AEG_STAGE_IQUANT_IDCT];
    if (!d->auto_ok || kind == 0 || !coef_probe_on())
        return 0;               /* an I picture has no inter candidates */
    if (!d->coef_decided[kind])
        return d->coef_n[kind][1] < d->coef_n[kind][0];
    if (++d->coef_since[kind] >= AEG_COEF_RETRY) {
        d->coef_since[kind] = 0;
        return !d->coef_dev[kind];
    }
    return d->coef_dev[kind];
}

int aeg_dispatch_coef_pick(aeg_dispatcher *d, int kind)
{
    aeg_dispatcher *r = disp_lock(d);
    int dev = dispatch_coef_pick(r, kind);
    disp_unlock(r);
    return dev;
}

static void dispatch_coef_done(aeg_dispatcher *d, int kind, int dev,
                               int64_t ns, int mbs)
{
    int64_t per, *slot;
    if (kind <= 0 || kind > 2 || mbs <= 0 || !d->auto_ok || !coef_probe_on() ||
        d->forced[AEG_STAGE_TRANSFORM] >= 0 ||
        d->forced[AEG_STAGE_IQUANT_IDCT] >= 0)
        return;
    dev = dev != 0;
    per = ns / mbs;
    slot = &d->coef_ns[kind][dev];
    if (d->coef_n[kind][dev]++ == 0)
        return;                 /* the warm-up picture, see above */
    *slot = *slot ? (*slot * 3 + per) / 4 : per;
    if (d->coef_n[kind][0] > 1 && d->coef_n[kind][1] > 1) {
        int64_t c = d->coef_ns[kind][0], g = d->coef_ns[kind][1];
        int was = d->coef_dev[kind], first = !d->coef_decided[kind];
        /* settle after AEG_COEF_PROBE counted pictures a side -- or after the
         * first one when the two are a factor of two apart, which no amount
         * of content variation between neighbouring pictures explains */
        if (first && g < 2 * c && c < 2 * g &&
            (d->coef_n[kind][0] <= AEG_COEF_PROBE ||
             d->coef_n[kind][1] <= AEG_COEF_PROBE))
            return;
        d->coef_decided[kind] = 1;
        if (g * 100 < c * 90)
            d->coef_dev[kind] = 1;
        else if (c * 100 < g * 90)
            d->coef_dev[kind] = 0;
        if (d->coef_dev[kind] != was || first)
            aeg_log(AEG_LOG_DEBUG, "aeg: coefficient stages of %c pictures on the %s "
                    "(%.2f us/MB on the CPU, %.2f us/MB on the device)",
                    kind == 1 ? 'P' : 'B', d->coef_dev[kind] ? "device" : "CPU",
                    c / 1000.0, g / 1000.0);
    }
}

void aeg_dispatch_coef_done(aeg_dispatcher *d, int kind, int dev, int64_t ns,
                            int mbs)
{
    aeg_dispatcher *r = disp_lock(d);
    dispatch_coef_done(r, kind, dev, ns, mbs);
    disp_unlock(r);
}

#if BIT_DEPTH == 8   /* public, and the device stubs: the 8-bit build's */
/* ------------------------------------------------------------------ */
/* Device enumeration                                                  */
/* ------------------------------------------------------------------ */
int aeg_query_devices(aeg_backend backend, aeg_device_info *out, int max);
int aeg_calibrate(aeg_backend backend);

int aeg_query_devices(aeg_backend backend, aeg_device_info *out, int max)
{
    return aeg_backend_query(backend, out, max);
}

int aeg_calibrate(aeg_backend backend)
{
    return aeg_backend_calibrate(backend);
}

int aeg_backend_query(aeg_backend b, aeg_device_info *out, int max)
{
    int n = 0;
    if (b == AEG_BACKEND_CPU || b == AEG_BACKEND_AUTO) {
        if (n < max) {
            memset(&out[n], 0, sizeof(out[n]));
            out[n].backend = AEG_BACKEND_CPU;
            snprintf(out[n].name, sizeof(out[n].name), "CPU (%d cores)",
                     aeg_cpu_count());
            snprintf(out[n].vendor, sizeof(out[n].vendor), "host");
            out[n].is_gpu = 0;
            out[n].max_compute_units = aeg_cpu_count();
        }
        n++;
        if (b == AEG_BACKEND_CPU)
            return n;
    }
    n += aeg_backend_query_opencl(out + n, max - n > 0 ? max - n : 0);
    n += aeg_backend_query_cuda(out + n, max - n > 0 ? max - n : 0);
    return n;
}

/* Launch + transfer overhead measured by the last calibration, in ns. */
static int64_t g_launch_ns;

int64_t aeg_backend_launch_ns(void)
{
    return g_launch_ns;
}

int aeg_backend_calibrate(aeg_backend b)
{
    aeg_be_ops ops;
    int i;
    int64_t best = 0;
    int16_t res[64], level[64];
    int32_t nz[4];

    aeg_be_create(b, &ops);
    if (ops.backend == AEG_BACKEND_CPU)
        return AEG_EBACKEND;
    for (i = 0; i < 64; i++)
        res[i] = (int16_t)(i * 3 - 40);
    /* A batch far smaller than the device's comfortable size measures the
     * fixed cost of a call: kernel launch plus the two transfers. */
    for (i = 0; i < 8; i++) {
        int64_t t0 = aeg_now_ns();
        ops.tq_batch(ops.priv, res, 4, 26, 1, level, nz);
        if (ops.sync)
            ops.sync(ops.priv);
        else
            (void)0;
        {
            int64_t dt = aeg_now_ns() - t0;
            if (i && (!best || dt < best))
                best = dt;
        }
    }
    if (ops.destroy)
        ops.destroy(ops.priv);
    g_launch_ns = best;
    aeg_log(AEG_LOG_INFO, "aeg: %s launch+transfer overhead ~%lld us",
            aeg_backend_name(ops.backend), (long long)(best / 1000));
    return AEG_OK;
}

/* Weak stubs so the library links without OpenCL/CUDA. */
#if !defined(AEG_HAVE_OPENCL)
int aeg_be_opencl_create(aeg_be_ops *ops)
{
    (void)ops;
    return AEG_EBACKEND;
}
int aeg_backend_query_opencl(aeg_device_info *out, int max)
{
    (void)out; (void)max;
    return 0;
}
#endif
#if !defined(AEG_HAVE_CUDA)
int aeg_be_cuda_create(aeg_be_ops *ops)
{
    (void)ops;
    return AEG_EBACKEND;
}
int aeg_backend_query_cuda(aeg_device_info *out, int max)
{
    (void)out; (void)max;
    return 0;
}
int aeg_cuda_compile_check(int cc, char *log, size_t logsz, size_t *ptx_size)
{
    (void)cc; (void)log; (void)logsz; (void)ptx_size;
    return AEG_EBACKEND;
}
#endif
#endif /* BIT_DEPTH == 8 */
