/*
 * simd_check - the vector kernels against scalar references, at the bit depth
 * this file is compiled for.
 *
 * The SSE2/AVX2 kernels exist in two forms, for 8-bit and for 16-bit samples
 * (the 10-bit build), and must return exactly what the scalar code returns:
 * a distortion feeds a decision, a coefficient goes into the bitstream, a
 * reconstructed sample has to match the decoder's.  The Makefile builds this
 * file twice: build/simd_check (8 bits) and build/simd_check10, compiled with
 * -DBIT_DEPTH=10 and its references to the library renamed to the _10 build
 * the way the library's own 10-bit objects are.  Run each also with
 * AEG_NOAVX2=1 to cover the SSE2 forms.
 *
 * The references are written here (the metrics) or are the library's scalar
 * functions (_c, the transforms); inputs are random samples, the extremes
 * (0 / PIXEL_MAX only), and nearly flat blocks, at random strides.
 *
 * usage: simd_check [iterations]
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "h264.h"

static int failures;
static unsigned rng = 12345;

static unsigned rnd(void)
{
    rng = rng * 1103515245u + 12345u;
    return rng >> 8;
}

#define CHECK(cond, ...)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            if (failures < 20) {                                            \
                printf("FAIL %s:%d: ", __FILE__, __LINE__);                 \
                printf(__VA_ARGS__);                                        \
                printf("\n");                                               \
            }                                                               \
            failures++;                                                     \
        }                                                                   \
    } while (0)

enum { PW = 96, PH = 48, PS = 128 };      /* scratch planes, stride PS */

/* mode 0: uniform, 1: only 0 and PIXEL_MAX, 2: nearly flat, 3: a ramp */
static void fill(pixel *p, int n, int mode)
{
    int i, base = (int)(rnd() % (PIXEL_MAX + 1));
    for (i = 0; i < n; i++) {
        int v;
        switch (mode) {
        case 0: v = (int)(rnd() % (PIXEL_MAX + 1)); break;
        case 1: v = (rnd() & 1) ? PIXEL_MAX : 0; break;
        case 2: v = base + (int)(rnd() % 5) - 2; break;
        default: v = base + i % 37 * 7; break;
        }
        p[i] = (pixel)(v < 0 ? 0 : v > PIXEL_MAX ? PIXEL_MAX : v);
    }
}

/* ---- scalar references -------------------------------------------- */
static int sad_ref(int w, int h, const pixel *a, int sa, const pixel *b, int sb)
{
    int x, y, s = 0;
    for (y = 0; y < h; y++)
        for (x = 0; x < w; x++)
            s += abs((int)a[y * sa + x] - (int)b[y * sb + x]);
    return s;
}

static int64_t ssd_ref(int w, int h, const pixel *a, int sa, const pixel *b, int sb)
{
    int x, y;
    int64_t s = 0;
    for (y = 0; y < h; y++)
        for (x = 0; x < w; x++) {
            int d = (int)a[y * sa + x] - (int)b[y * sb + x];
            s += d * d;
        }
    return s;
}

static int satd4_ref(const pixel *a, int sa, const pixel *b, int sb)
{
    int d[16], t[16], i, k, s = 0;
    for (i = 0; i < 16; i++)
        d[i] = (int)a[(i >> 2) * sa + (i & 3)] - (int)b[(i >> 2) * sb + (i & 3)];
    for (k = 0; k < 4; k++) {          /* rows */
        int *r = d + 4 * k;
        int s0 = r[0] + r[1], d0 = r[0] - r[1], s1 = r[2] + r[3], d1 = r[2] - r[3];
        t[4 * k + 0] = s0 + s1; t[4 * k + 1] = s0 - s1;
        t[4 * k + 2] = d0 - d1; t[4 * k + 3] = d0 + d1;
    }
    for (k = 0; k < 4; k++) {          /* columns */
        int s0 = t[k] + t[4 + k], d0 = t[k] - t[4 + k];
        int s1 = t[8 + k] + t[12 + k], d1 = t[8 + k] - t[12 + k];
        s += abs(s0 + s1) + abs(s0 - s1) + abs(d0 - d1) + abs(d0 + d1);
    }
    return (s + 1) >> 1;
}

static int satd_ref(int w, int h, const pixel *a, int sa, const pixel *b, int sb)
{
    int x, y, s = 0;
    for (y = 0; y < h; y += 4)
        for (x = 0; x < w; x += 4)
            s += satd4_ref(a + y * sa + x, sa, b + y * sb + x, sb) * 2;
    return s >> 1;
}

/* ---- metrics -------------------------------------------------------- */
static void check_metrics(int iters)
{
    static pixel A[PS * PH], B[PS * PH], C[PS * PH], AV[16 * 16];
    /* every shape the encoder measures, and the odd ones the kernels'
     * dispatch has to send elsewhere */
    static const int ws[] = { 2, 4, 8, 12, 16, 24, 32, 48 };
    static const int hs[] = { 1, 2, 4, 6, 8, 12, 16, 20, 24, 32 };
    int sizes[sizeof(ws) / sizeof(ws[0]) * (sizeof(hs) / sizeof(hs[0]))][2];
    int it, k, n = 0, i, j0;
    for (i = 0; i < (int)(sizeof(ws) / sizeof(ws[0])); i++)
        for (j0 = 0; j0 < (int)(sizeof(hs) / sizeof(hs[0])); j0++) {
            sizes[n][0] = ws[i];
            sizes[n++][1] = hs[j0];
        }
    for (it = 0; it < iters; it++) {
        int mode = it & 3;
        fill(A, PS * PH, mode);
        fill(B, PS * PH, mode == 2 ? 2 : (int)(rnd() & 3));
        fill(C, PS * PH, (int)(rnd() & 3));
        for (k = 0; k < n; k++) {
            int w = sizes[k][0], h = sizes[k][1];
            int sa = w + (int)(rnd() % (PS - w)), sb = w + (int)(rnd() % (PS - w));
            int maxh = PS * PH / (sa > sb ? sa : sb);
            const pixel *b4[4];
            int out[4], j, x, y, r, e;
            if (h > maxh || h > PH)
                continue;
            r = aeg_sad(w, h, A, sa, B, sb);
            e = sad_ref(w, h, A, sa, B, sb);
            CHECK(r == e, "sad %dx%d mode %d: %d, expected %d", w, h, mode, r, e);
            for (j = 0; j < 4; j++)
                b4[j] = B + (rnd() % 3) * sb + j;
            aeg_sad_x4(w, h, A, sa, b4, sb, out);
            for (j = 0; j < 4; j++) {
                e = sad_ref(w, h, A, sa, b4[j], sb);
                CHECK(out[j] == e, "sad_x4 %dx%d [%d]: %d, expected %d", w, h, j,
                      out[j], e);
            }
            {
                int64_t s1 = aeg_ssd(w, h, A, sa, B, sb), s2 = ssd_ref(w, h, A, sa, B, sb);
                CHECK(s1 == s2, "ssd %dx%d mode %d: %lld, expected %lld", w, h,
                      mode, (long long)s1, (long long)s2);
            }
            if ((w & 3) || (h & 3))
                continue;
            r = aeg_satd(w, h, A, sa, B, sb);
            e = satd_ref(w, h, A, sa, B, sb);
            CHECK(r == e, "satd %dx%d mode %d: %d, expected %d", w, h, mode, r, e);
            CHECK(aeg_satd_c(w, h, A, sa, B, sb) == e, "satd_c %dx%d", w, h);
            if (w <= 16 && h <= 16) {
                const pixel *r1 = B + (rnd() % 2), *r2 = C + (rnd() % 2);
                int sr = sb;
                for (y = 0; y < h; y++)
                    for (x = 0; x < w; x++)
                        AV[y * w + x] = (pixel)((r1[y * sr + x] + r2[y * sr + x] + 1) >> 1);
                r = aeg_satd_avg(w, h, A, sa, r1, r2, sr);
                e = satd_ref(w, h, A, sa, AV, w);
                CHECK(r == e, "satd_avg %dx%d mode %d: %d, expected %d", w, h,
                      mode, r, e);
            }
            if (w == 4 && h == 4) {
                r = aeg_satd_4x4(A, sa, B, sb);
                e = satd4_ref(A, sa, B, sb);
                CHECK(r == e, "satd_4x4 mode %d: %d, expected %d", mode, r, e);
            }
            if (w == 8 && h == 8) {
                r = aeg_sa8d_8x8(A, sa, B, sb);
                e = aeg_sa8d_8x8_c(A, sa, B, sb);
                CHECK(r == e, "sa8d mode %d: %d, expected %d", mode, r, e);
            }
        }
    }
}

/* ---- transforms ------------------------------------------------------ */
static int rnd_qp(void)
{
    return QP_MIN + (int)(rnd() % (52 - QP_MIN));
}

/* a residual of the depth's range: random, extreme, small or flat */
static void fill_res(int16_t *r, int n, int mode)
{
    int i, base = (int)(rnd() % (2 * PIXEL_MAX + 1)) - PIXEL_MAX;
    for (i = 0; i < n; i++) {
        int v;
        switch (mode) {
        case 0: v = (int)(rnd() % (2 * PIXEL_MAX + 1)) - PIXEL_MAX; break;
        case 1: v = (rnd() & 1) ? PIXEL_MAX : -PIXEL_MAX; break;
        case 2: v = (int)(rnd() % 9) - 4; break;
        default: v = base; break;
        }
        r[i] = (int16_t)v;
    }
}

/* realistic levels: a residual through the scalar transform and quantiser,
 * then thinned out (or reduced to the DC) now and then */
static void make_levels4(dctcoef *lv, int qp)
{
    int16_t r[16];
    dctcoef c[16];
    int i, mode = (int)(rnd() % 4);
    fill_res(r, 16, mode);
    aeg_transform_4x4_c(r, 4, c);
    aeg_quant_4x4_c(c, lv, qp, (int)(rnd() & 1));
    if ((rnd() & 3) == 0)
        for (i = 1; i < 16; i++)
            lv[i] = 0;
    else if ((rnd() & 3) == 0)
        for (i = 0; i < 16; i++)
            if (rnd() & 1)
                lv[i] = 0;
}

static void make_levels8(dctcoef *lv, int qp)
{
    int16_t r[64];
    dctcoef c[64];
    int i, mode = (int)(rnd() % 4);
    fill_res(r, 64, mode);
    aeg_transform_8x8_c(r, 8, c);
    aeg_quant_8x8_c(c, lv, qp, (int)(rnd() & 1));
    if ((rnd() & 3) == 0)
        for (i = 1; i < 64; i++)
            lv[i] = 0;
    else if ((rnd() & 3) == 0)
        for (i = 0; i < 64; i++)
            if (rnd() & 1)
                lv[i] = 0;
}

static void check_transforms(int iters)
{
    static pixel P[PS * 16], S[PS * 16], D1[PS * 16], D2[PS * 16];
    int it, i, x, y;
    for (it = 0; it < iters; it++) {
        const int qp = rnd_qp(), intra = (int)(rnd() & 1), mode = it & 3;
        const int st = 16 + (int)(rnd() % (PS - 16));
        int16_t r[64], o1[64], o2[64];
        dctcoef c1[64], c2[64], l1[64], l2[64], lv[64];
        int32_t dc = (int32_t)(rnd() % 4001) - 2000;
        fill(P, PS * 16, (int)(rnd() & 3));
        fill(S, PS * 16, (int)(rnd() & 3));

        /* 4x4 forward transform and quantiser */
        fill_res(r, 64, mode);
        aeg_transform_4x4(r, 8, c1);
        aeg_transform_4x4_c(r, 8, c2);
        CHECK(!memcmp(c1, c2, 16 * sizeof(dctcoef)), "transform_4x4 mode %d", mode);
        aeg_quant_4x4(c2, l1, qp, intra);
        aeg_quant_4x4_c(c2, l2, qp, intra);
        CHECK(!memcmp(l1, l2, 16 * sizeof(dctcoef)), "quant_4x4 qp %d", qp);

        /* ... and on coefficients over the whole legal range (and, at 10
         * bits, a little past it, where the vector form must hand over) */
        {
            const int range = BIT_DEPTH == 8 ? 32767 : (1 << 17) + 64;
            for (i = 0; i < 64; i++)
                c2[i] = (dctcoef)((int)(rnd() % (2 * range + 1)) - range);
            if (mode == 1)
                for (i = 0; i < 64; i++)
                    c2[i] = (dctcoef)((rnd() & 1) ? -(BIT_DEPTH == 8 ? 32767 : 1 << 17)
                                                  : (BIT_DEPTH == 8 ? 32767 : 1 << 17));
            aeg_quant_4x4(c2, l1, qp, intra);
            aeg_quant_4x4_c(c2, l2, qp, intra);
            CHECK(!memcmp(l1, l2, 16 * sizeof(dctcoef)), "quant_4x4 wide qp %d", qp);
            aeg_quant_8x8(c2, l1, qp, intra);
            aeg_quant_8x8_c(c2, l2, qp, intra);
            CHECK(!memcmp(l1, l2, sizeof(l1)), "quant_8x8 wide qp %d", qp);
        }

        /* 4x4 inverse: residual, with and without the DC override */
        make_levels4(lv, qp);
        aeg_idct_residual(lv, qp, o1);
        aeg_idct_residual_c(lv, qp, NULL, o2);
        CHECK(!memcmp(o1, o2, 16 * sizeof(int16_t)), "idct_residual qp %d", qp);
        aeg_idct_residual_dc(lv, qp, dc, o1);
        aeg_idct_residual_c(lv, qp, &dc, o2);
        CHECK(!memcmp(o1, o2, 16 * sizeof(int16_t)), "idct_residual_dc qp %d", qp);

        /* ... added to a prediction (or none), clipped */
        for (i = 0; i < 4; i++) {
            const pixel *pr = (i & 1) ? NULL : P;
            const int32_t *dp = (i & 2) ? &dc : NULL;
            memset(D1, 0, sizeof(D1));
            memset(D2, 0, sizeof(D2));
            aeg_idct_add_block(D1, st, lv, qp, pr, st, dp);
            aeg_idct_add_block_c(D2, st, lv, qp, pr, st, dp);
            CHECK(!memcmp(D1, D2, sizeof(D1)), "idct_add_block qp %d pred %d dc %d",
                  qp, !(i & 1), !!(i & 2));
            if (pr) {
                int64_t e1 = aeg_idct_add_ssd4(lv, qp, P, st, dp, S, st);
                int64_t e2 = ssd_ref(4, 4, D2, st, S, st);
                CHECK(e1 == e2, "idct_add_ssd4 qp %d: %lld, expected %lld", qp,
                      (long long)e1, (long long)e2);
            }
        }

        /* the batched macroblock reconstruction */
        {
            static int16_t rec[256];
            static pixel pred[256];
            uint64_t e1, e2;
            for (i = 0; i < 256; i++) {
                rec[i] = (int16_t)((mode == 1) ? ((rnd() & 1) ? 32767 : -32768)
                                   : (int)(rnd() % (4 * PIXEL_MAX + 1)) - 2 * PIXEL_MAX);
                pred[i] = (pixel)(rnd() % (PIXEL_MAX + 1));
            }
            e1 = aeg_add_res16_ssd(D1, st, pred, rec, S, st);
            e2 = aeg_add_res16_ssd_c(D2, st, pred, rec, S, st);
            CHECK(e1 == e2, "add_res16_ssd: %llu, expected %llu",
                  (unsigned long long)e1, (unsigned long long)e2);
            for (y = 0; y < 16; y++)
                CHECK(!memcmp(D1 + y * st, D2 + y * st, 16 * sizeof(pixel)),
                      "add_res16_ssd row %d", y);
        }

        /* 8x8: forward, quantiser, dequantiser, inverse */
        fill_res(r, 64, mode);
        aeg_transform_8x8(r, 8, c1);
        aeg_transform_8x8_c(r, 8, c2);
        CHECK(!memcmp(c1, c2, sizeof(c1)), "transform_8x8 mode %d", mode);
        aeg_quant_8x8(c2, l1, qp, intra);
        aeg_quant_8x8_c(c2, l2, qp, intra);
        CHECK(!memcmp(l1, l2, sizeof(l1)), "quant_8x8 qp %d mode %d", qp, mode);
        make_levels8(lv, qp);
        aeg_dequant_8x8(lv, c1, qp);
        aeg_dequant_8x8_c(lv, c2, qp);
        CHECK(!memcmp(c1, c2, sizeof(c1)), "dequant_8x8 qp %d", qp);
        /* levels of any size: the vector dequantiser has to hand over where
         * its 32-bit product would not hold them */
        if ((rnd() & 7) == 0) {
            dctcoef w[64];
            for (i = 0; i < 64; i++)
                w[i] = (dctcoef)((int)(rnd() % 140001) - 70000);
            aeg_dequant_8x8(w, c1, qp);
            aeg_dequant_8x8_c(w, c2, qp);
            CHECK(!memcmp(c1, c2, sizeof(c1)), "dequant_8x8 wide qp %d", qp);
        }
        aeg_idct_8x8_residual(lv, qp, o1);
        aeg_idct_8x8_residual_c(lv, qp, o2);
        CHECK(!memcmp(o1, o2, sizeof(o1)), "idct_8x8_residual qp %d", qp);
        aeg_idct_8x8_add(D1, st, lv, qp, P, st);
        for (y = 0; y < 8; y++)
            for (x = 0; x < 8; x++) {
                int v = (int)P[y * st + x] + o2[y * 8 + x];
                D2[y * st + x] = (pixel)(v < 0 ? 0 : v > PIXEL_MAX ? PIXEL_MAX : v);
            }
        for (y = 0; y < 8; y++)
            CHECK(!memcmp(D1 + y * st, D2 + y * st, 8 * sizeof(pixel)),
                  "idct_8x8_add qp %d row %d", qp, y);
        /* the level search's incremental reconstruction */
        {
            aeg_idct8_cache cache;
            int pos = (int)(rnd() % 64), lev = (int)(rnd() % 41) - 20;
            int64_t e1, e2;
            aeg_idct8_cache_init(&cache, lv, qp);
            e1 = aeg_idct8_cache_ssd(&cache, -1, 0, P, st, S, st);
            e2 = ssd_ref(8, 8, D2, st, S, st);
            CHECK(e1 == e2, "idct8_cache_ssd qp %d: %lld, expected %lld", qp,
                  (long long)e1, (long long)e2);
            e1 = aeg_idct8_cache_ssd(&cache, pos, lev, P, st, S, st);
            memcpy(l1, lv, sizeof(l1));
            l1[pos] = (dctcoef)lev;
            aeg_idct_8x8_residual_c(l1, qp, o2);
            for (y = 0; y < 8; y++)
                for (x = 0; x < 8; x++) {
                    int v = (int)P[y * st + x] + o2[y * 8 + x];
                    D2[y * st + x] = (pixel)(v < 0 ? 0 : v > PIXEL_MAX ? PIXEL_MAX : v);
                }
            e2 = ssd_ref(8, 8, D2, st, S, st);
            CHECK(e1 == e2, "idct8_cache_ssd qp %d pos %d lev %d: %lld, expected %lld",
                  qp, pos, lev, (long long)e1, (long long)e2);
        }
    }
}

/* ---- motion compensation --------------------------------------------- */
/* The half-pel planes and the plane-based prediction against aeg_mc_luma(),
 * the scalar per-block filter; the chroma filter against its formula. */
static void check_mc(int iters)
{
    enum { W = 48, H = 32, PAD = AEG_PAD, ST = W + 2 * PAD };
    static pixel buf[4][ST * (H + 2 * PAD)], o1[16 * 16], o2[16 * 16];
    static const int bs[][2] = { { 16, 16 }, { 16, 8 }, { 8, 16 }, { 8, 8 },
                                 { 8, 4 }, { 4, 8 }, { 4, 4 }, { 2, 2 }, { 2, 4 } };
    aeg_mc_ctx mc;
    pixel *org[4];
    int it, k, i, x, y;
    for (k = 0; k < 4; k++)
        org[k] = buf[k] + PAD * ST + PAD;
    for (it = 0; it < iters; it++) {
        fill(buf[0], ST * (H + 2 * PAD), it & 3);
        /* the planes over the whole picture, in two bands */
        for (k = 1; k < 4; k++) {
            int split = (int)(rnd() % H);
            aeg_hpel_plane_rows(org[k], ST, org[0], ST, W, H, k, -PAD, split);
            aeg_hpel_plane_rows(org[k], ST, org[0], ST, W, H, k, split + 1, H + PAD - 1);
            /* every sample whose taps stay inside the padded picture */
            for (y = -PAD + 2; y <= H + PAD - 4; y++) {
                pixel row[ST];
                const int x0 = -PAD + 2, n = W + 2 * PAD - 6;
                aeg_mc_luma(row, ST, org[0] + y * ST + x0, ST, n, 1,
                            k == 2 ? 0 : 2, k == 1 ? 0 : 2);
                CHECK(!memcmp(row, org[k] + y * ST + x0, n * sizeof(pixel)),
                      "hpel plane %d row %d mode %d", k, y, it & 3);
            }
        }
        memset(&mc, 0, sizeof(mc));
        for (k = 0; k < 4; k++)
            mc.base[k] = org[k];
        mc.stride = ST;
        mc.ready = 1;
        for (i = 0; i < (int)(sizeof(bs) / sizeof(bs[0])); i++) {
            const int w = bs[i][0], h = bs[i][1];
            const int bx = (int)(rnd() % (W + 32)) - 16, by = (int)(rnd() % (H + 32)) - 16;
            const pixel *src = org[0] + by * ST + bx;
            int fx, fy;
            for (fy = 0; fy < 4; fy++)
                for (fx = 0; fx < 4; fx++) {
                    if (w < 4)
                        break;
                    memset(o1, 0, sizeof(o1));
                    memset(o2, 0, sizeof(o2));
                    aeg_mc_luma_p(o1, 16, &mc, src, w, h, fx, fy);
                    aeg_mc_luma(o2, 16, src, ST, w, h, fx, fy);
                    CHECK(!memcmp(o1, o2, sizeof(o1)), "mc_luma_p %dx%d (%d,%d)", w, h,
                          fx, fy);
                }
            for (fy = 0; fy < 8; fy++)
                for (fx = 0; fx < 8; fx++) {
                    memset(o1, 0, sizeof(o1));
                    aeg_mc_chroma(o1, 16, src, ST, w, h, fx, fy);
                    for (y = 0; y < h; y++)
                        for (x = 0; x < w; x++) {
                            const pixel *q = src + y * ST + x;
                            int v = (8 - fx) * (8 - fy) * q[0] + fx * (8 - fy) * q[1] +
                                    (8 - fx) * fy * q[ST] + fx * fy * q[ST + 1];
                            CHECK(o1[y * 16 + x] == (pixel)((v + 32) >> 6),
                                  "mc_chroma %dx%d (%d,%d) at %d,%d", w, h, fx, fy, x, y);
                        }
                }
        }
    }
}

/* ---- input conversion ------------------------------------------------ */
/* aeg_input_convert() against the definition: every 4:2:0 picture widened to
 * 16 bits and dithered to BIT_DEPTH (the conversion takes a direct path for
 * depths up to BIT_DEPTH, which has to agree), for I420 and NV12 at input
 * depths 8 .. BIT_DEPTH and one deeper. */
static void check_input(int iters)
{
    enum { W = 46, H = 18, CW = 23, CH = 9 };
    static uint16_t s16[3][W * H], nv16[2 * CW * CH], w16[W * H];
    static uint8_t s8[3][W * H], nv8[2 * CW * CH];
    static pixel out[W * H + 2 * CW * CH], ref[W * H + 2 * CW * CH];
    static int16_t err[W + 1];
    aeg_input_cvt cvt;
    int it, depth, p, i, nv;
    memset(&cvt, 0, sizeof(cvt));
    for (it = 0; it < iters; it++)
        for (depth = 8; depth <= BIT_DEPTH + 2; depth++) {
            const int max = (1 << depth) - 1;
            if (depth > BIT_DEPTH && depth != BIT_DEPTH + 2)
                continue;
            for (p = 0; p < 3; p++)
                for (i = 0; i < W * H; i++) {
                    int v = (it & 1) ? ((rnd() & 1) ? max : 0) : (int)(rnd() % (max + 1));
                    s16[p][i] = (uint16_t)v;
                    s8[p][i] = (uint8_t)v;
                }
            for (i = 0; i < CW * CH; i++) {
                nv16[2 * i] = s16[1][i], nv16[2 * i + 1] = s16[2][i];
                nv8[2 * i] = s8[1][i], nv8[2 * i + 1] = s8[2][i];
            }
            /* the definition */
            for (p = 0; p < 3; p++) {
                const int n = p ? CW * CH : W * H, w = p ? CW : W, h = p ? CH : H;
                pixel *d = ref + (p ? W * H + (p - 1) * CW * CH : 0);
                for (i = 0; i < n; i++)
                    w16[i] = (uint16_t)(s16[p][i] << (16 - depth));
                aeg_dither_plane(d, w, w16, w, w, h, BIT_DEPTH, err);
            }
            for (nv = 0; nv <= 1; nv++) {
                aeg_image im;
                pixel *dst[3] = { out, out + W * H, out + W * H + CW * CH };
                int st[3] = { W, CW, CW };
                memset(&im, 0, sizeof(im));
                im.csp = nv ? AEG_CSP_NV12 : AEG_CSP_I420;
                im.depth = depth;
                im.plane[0] = depth > 8 ? (void *)s16[0] : (void *)s8[0];
                im.stride[0] = W * (depth > 8 ? 2 : 1);
                if (nv) {
                    im.plane[1] = depth > 8 ? (void *)nv16 : (void *)nv8;
                    im.stride[1] = 2 * CW * (depth > 8 ? 2 : 1);
                } else {
                    im.plane[1] = depth > 8 ? (void *)s16[1] : (void *)s8[1];
                    im.plane[2] = depth > 8 ? (void *)s16[2] : (void *)s8[2];
                    im.stride[1] = im.stride[2] = CW * (depth > 8 ? 2 : 1);
                }
                memset(out, 0x55, sizeof(out));
                CHECK(aeg_input_convert(&cvt, &im, W, H, 0, dst, st) == 0,
                      "input_convert depth %d", depth);
                CHECK(!memcmp(out, ref, sizeof(out)), "input %s depth %d differs "
                      "from the dither", nv ? "NV12" : "I420", depth);
            }
        }
    aeg_input_cvt_free(&cvt);
}

int main(int argc, char **argv)
{
    int iters = argc > 1 ? atoi(argv[1]) : 2000;
    const char *na = getenv("AEG_NOAVX2");
    check_metrics(iters);
    check_transforms(iters * 20);
    check_input(iters / 20 + 1);
    check_mc(iters / 10 + 1);
    if (failures) {
        printf("simd_check (%d-bit%s): %d failure(s)\n", BIT_DEPTH,
               na && atoi(na) ? ", no AVX2" : "", failures);
        return 1;
    }
    printf("simd_check (%d-bit%s): %d iterations, vector kernels = scalar\n",
           BIT_DEPTH, na && atoi(na) ? ", no AVX2" : "", iters);
    return 0;
}
