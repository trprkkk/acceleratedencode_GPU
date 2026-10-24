/*
 * Integer transforms, quantisation and distortion metrics.
 *
 * All arithmetic matches the normative H.264 integer transform pair:
 *
 *   forward :  W = C_f X C_f^T          (DC gain 16 for a 4x4 block)
 *   quant   :  Z = (|W| * MF[qp%6][c] + f) >> (15 + qp/6)
 *   dequant :  W' = Z * LS[qp%6][c] << (qp/6)
 *   inverse :  xt = C_i W' C_i^T,  r = (xt + 32) >> 6
 *
 * with c = (x&1) + (y&1) the position class.  This pairing reconstructs the
 * residual with the same rounding as the reference decoders, which is why a
 * stream produced here decodes bit-identically to the encoder's own
 * reconstruction (see tools/aegverify).
 */
#include "h264.h"

/* The distortion metrics are the innermost loop of the motion search -- a
 * 320x192 picture at preset 2 runs ~480 block comparisons per macroblock, so a
 * 16x16 SAD is evaluated millions of times per second and its cost is the
 * search's cost.  The sums below are exact integers, so the vectorised path
 * returns bit-identical values to the scalar one and cannot change a decision:
 * the CPU fallback, the batch operations and the device kernels all keep
 * agreeing (tests/kernel_check.c compares them sample by sample). */
#if defined(__SSE2__)
#include <emmintrin.h>
#endif

/* ------------------------------------------------------------------ */
/* 4x4 forward transform                                               */
/* ------------------------------------------------------------------ */
#define DCT4_1D(d0,d1,d2,d3,s0,s1,s2,s3) do {                        \
    int t0_ = (d0) + (d3);                                           \
    int t1_ = (d1) + (d2);                                           \
    int t2_ = (d1) - (d2);                                           \
    int t3_ = (d0) - (d3);                                           \
    (s0) = t0_ + t1_;                                                \
    (s2) = t0_ - t1_;                                                \
    (s1) = (t3_ << 1) + t2_;                                         \
    (s3) = t3_ - (t2_ << 1);                                         \
} while (0)

void aeg_transform_4x4(const int16_t *res, int stride, int16_t *coef)
{
    int d[16];
    int i, t[16];

    for (i = 0; i < 4; i++) {
        DCT4_1D(res[i * stride + 0], res[i * stride + 1], res[i * stride + 2],
                res[i * stride + 3], t[i * 4 + 0], t[i * 4 + 1],
                t[i * 4 + 2], t[i * 4 + 3]);
    }
    for (i = 0; i < 4; i++) {
        DCT4_1D(t[0 * 4 + i], t[1 * 4 + i], t[2 * 4 + i], t[3 * 4 + i],
                d[0 * 4 + i], d[1 * 4 + i], d[2 * 4 + i], d[3 * 4 + i]);
    }
    for (i = 0; i < 16; i++)
        coef[i] = (int16_t)d[i];
}

/* 4x4 Hadamard used by the luma DC transform of Intra_16x16 */
void aeg_transform_dc4x4(const int16_t dc[16], int16_t out[16])
{
    int t[16], d[16], i;
    for (i = 0; i < 4; i++) {
        int s0 = dc[i * 4 + 0] + dc[i * 4 + 3];
        int s1 = dc[i * 4 + 1] + dc[i * 4 + 2];
        int s2 = dc[i * 4 + 1] - dc[i * 4 + 2];
        int s3 = dc[i * 4 + 0] - dc[i * 4 + 3];
        t[i * 4 + 0] = s0 + s1; t[i * 4 + 1] = s3 + s2;
        t[i * 4 + 2] = s0 - s1; t[i * 4 + 3] = s3 - s2;
    }
    for (i = 0; i < 4; i++) {
        int s0 = t[0 * 4 + i] + t[3 * 4 + i];
        int s1 = t[1 * 4 + i] + t[2 * 4 + i];
        int s2 = t[1 * 4 + i] - t[2 * 4 + i];
        int s3 = t[0 * 4 + i] - t[3 * 4 + i];
        d[0 * 4 + i] = s0 + s1; d[1 * 4 + i] = s3 + s2;
        d[2 * 4 + i] = s0 - s1; d[3 * 4 + i] = s3 - s2;
    }
    for (i = 0; i < 16; i++)
        out[i] = (int16_t)d[i];
}

/* inverse luma DC Hadamard; total gain 1/4 (matches the spec's rounding) */
void aeg_itransform_dc4x4(const int16_t in[16], int16_t out[16])
{
    int t[16], d[16], i;
    for (i = 0; i < 4; i++) {
        int s0 = in[i * 4 + 0] + in[i * 4 + 2];
        int s1 = in[i * 4 + 0] - in[i * 4 + 2];
        int s2 = in[i * 4 + 1] - in[i * 4 + 3];
        int s3 = in[i * 4 + 1] + in[i * 4 + 3];
        t[i * 4 + 0] = (s0 + s3 + 1) >> 1;
        t[i * 4 + 1] = (s1 + s2 + 1) >> 1;
        t[i * 4 + 2] = (s1 - s2 + 1) >> 1;
        t[i * 4 + 3] = (s0 - s3 + 1) >> 1;
    }
    for (i = 0; i < 4; i++) {
        int s0 = t[0 * 4 + i] + t[2 * 4 + i];
        int s1 = t[0 * 4 + i] - t[2 * 4 + i];
        int s2 = t[1 * 4 + i] - t[3 * 4 + i];
        int s3 = t[1 * 4 + i] + t[3 * 4 + i];
        d[0 * 4 + i] = (s0 + s3 + 1) >> 1;
        d[1 * 4 + i] = (s1 + s2 + 1) >> 1;
        d[2 * 4 + i] = (s1 - s2 + 1) >> 1;
        d[3 * 4 + i] = (s0 - s3 + 1) >> 1;
    }
    for (i = 0; i < 16; i++)
        out[i] = (int16_t)d[i];
}

/* 2x2 Hadamard for chroma DC (4:2:0) */
void aeg_transform_dc2x2(const int16_t dc[4], int16_t out[4])
{
    int s0 = dc[0] + dc[3];
    int s1 = dc[1] + dc[2];
    int s2 = dc[1] - dc[2];
    int s3 = dc[0] - dc[3];
    out[0] = (int16_t)(s0 + s1);
    out[1] = (int16_t)(s3 + s2);
    out[2] = (int16_t)(s0 - s1);
    out[3] = (int16_t)(s3 - s2);
}

void aeg_itransform_dc2x2(const int16_t in[4], int16_t out[4])
{
    int s0 = in[0] + in[2];
    int s1 = in[0] - in[2];
    int s2 = in[1] - in[3];
    int s3 = in[1] + in[3];
    out[0] = (int16_t)(s0 + s3);
    out[1] = (int16_t)(s1 + s2);
    out[2] = (int16_t)(s1 - s2);
    out[3] = (int16_t)(s0 - s3);
}

/* ------------------------------------------------------------------ */
/* Quantisation                                                        */
/* ------------------------------------------------------------------ */
/* Rounding offset of the forward quantiser: `f = (1 << qbits) / den` is what
 * the reference implementation uses (1/3 for intra, 1/6 for inter blocks),
 * i.e. a deadzone that rounds small coefficients away.  With no trellis -- and
 * without any other rate-distortion optimal choice of the levels -- that
 * deadzone is too wide for us: measured over the IDR frame and a 60-frame P
 * encode, retaining coefficients 1.5x more eagerly moves the rate-distortion
 * curve up by 0.2-0.3 dB per percent of rate, well above the ~0.07 dB/% slope
 * of the QP axis itself.  f is therefore widened by 3/2 (exactly, in
 * integers) instead of being rounded. */
#define QUANT_BIAS_NUM 3
#define QUANT_BIAS_DEN 2

int aeg_quant_bias(int qbits, int den)
{
    return (int)(((int64_t)(1 << qbits) * QUANT_BIAS_NUM) /
                 (den * QUANT_BIAS_DEN));
}

static inline int quant_coef(int32_t coef, int qp, int cls, int bias_num,
                             int bias_den)
{
    const int qbits = 15 + (qp / 6);
    const int mf = aeg_quant_mf[qp % 6][cls];
    const int f = aeg_quant_bias(qbits, bias_den) * bias_num;
    int abs = coef < 0 ? -coef : coef;
    int level = (int)(((int64_t)abs * mf + f) >> qbits);
    return coef < 0 ? -level : level;
}

void aeg_quant_4x4(const int16_t *coef, int16_t *level, int qp, int intra)
{
    int x, y;
    /* The decoder's 4x4 inverse transform is the transpose of ours: a
     * coefficient in slot (row 1, column 0) comes out as a horizontal ramp
     * there and as a vertical one here.  Storing the quantised block
     * transposed therefore makes the transmitted coefficients reconstruct the
     * residual we measured, and the per-position scale factors are symmetric
     * in x and y, so the quantiser itself is unaffected. */
    for (y = 0; y < 4; y++)
        for (x = 0; x < 4; x++) {
            int cls = (x & 1) + (y & 1);
            level[y * 4 + x] = (int16_t)quant_coef(coef[x * 4 + y], qp, cls,
                                                   intra ? 1 : 1,
                                                   intra ? 3 : 6);
        }
}

void aeg_quant_4x4_dc(const int16_t *coef, int16_t *level, int qp, int intra)
{
    /* The 4x4 Hadamard multiplies the DC by another 16 on top of the 16 the
     * per-block transform already put there; the reference decoder in turn
     * normalises with a 1/4 in the dequantiser and a 1/64 in its DC inverse,
     * which leaves two more bits of shift than for an AC block. */
    const int qbits = 17 + (qp / 6);
    const int mf = aeg_quant_mf[qp % 6][0];
    const int f = aeg_quant_bias(qbits, intra ? 3 : 6);
    int i;
    for (i = 0; i < 16; i++) {
        int abs = coef[i] < 0 ? -coef[i] : coef[i];
        int lv = (int)(((int64_t)abs * mf + f) >> qbits);
        level[i] = (int16_t)(coef[i] < 0 ? -lv : lv);
    }
}

void aeg_quant_2x2_dc(const int16_t *coef, int16_t *level, int qp, int intra)
{
    /* 2x2 Hadamard gain 4 on top of the per-block gain of 16; the chroma DC
     * inverse normalises by one bit less than the luma one. */
    const int qbits = 16 + (qp / 6);
    const int mf = aeg_quant_mf[qp % 6][0];
    const int f = aeg_quant_bias(qbits, intra ? 3 : 6);
    int i;
    for (i = 0; i < 4; i++) {
        int abs = coef[i] < 0 ? -coef[i] : coef[i];
        int lv = (int)(((int64_t)abs * mf + f) >> qbits);
        level[i] = (int16_t)(coef[i] < 0 ? -lv : lv);
    }
}

void aeg_dequant_4x4(const int16_t *level, int32_t *coef, int qp)
{
    int x, y, sh = qp / 6;
    for (y = 0; y < 4; y++)
        for (x = 0; x < 4; x++) {
            int cls = (x & 1) + (y & 1);
            coef[y * 4 + x] = (int32_t)level[x * 4 + y] *
                              aeg_dequant_ls[qp % 6][cls] << sh;
        }
}

/* Intra_16x16 luma DC: this is a *bit-exact* counterpart of the reference
 * decoder's ff_h264_luma_dc_dequant_idct(), including the coefficient order it
 * expects (the levels arrive scanned, the outputs land in the 16 luma blocks)
 * and its rounding.  `dc[b]` is the DC coefficient of block b expressed in the
 * same units as the dequantised AC coefficients (the inverse transform below
 * divides by 64). */
void aeg_dequant_4x4_dc_out(const int16_t *level, int32_t *dc, int qp)
{
    static const int base[4] = { 0, 2, 8, 10 };
    int in[16], temp[16];
    int i, qmul = aeg_dequant_ls[qp % 6][0] << (qp / 6 + 6);

    for (i = 0; i < 16; i++)
        in[i] = level[i];
    for (i = 0; i < 4; i++) {
        int z0 = in[4 * i + 0] + in[4 * i + 1];
        int z1 = in[4 * i + 0] - in[4 * i + 1];
        int z2 = in[4 * i + 2] - in[4 * i + 3];
        int z3 = in[4 * i + 2] + in[4 * i + 3];
        temp[4 * i + 0] = z0 + z3;
        temp[4 * i + 1] = z0 - z3;
        temp[4 * i + 2] = z1 - z2;
        temp[4 * i + 3] = z1 + z2;
    }
    for (i = 0; i < 4; i++) {
        int z0 = temp[i] + temp[8 + i];
        int z1 = temp[i] - temp[8 + i];
        int z2 = temp[4 + i] - temp[12 + i];
        int z3 = temp[4 + i] + temp[12 + i];
        dc[base[i] + 0] = (int16_t)(((z0 + z3) * qmul + 128) >> 8);
        dc[base[i] + 1] = (int16_t)(((z1 + z2) * qmul + 128) >> 8);
        dc[base[i] + 4] = (int16_t)(((z1 - z2) * qmul + 128) >> 8);
        dc[base[i] + 5] = (int16_t)(((z0 - z3) * qmul + 128) >> 8);
    }
}

/* Forward counterpart: the DC levels an Intra_16x16 macroblock has to transmit
 * for `dc[]` to come back out of the function above.  The Hadamard is its own
 * inverse up to a factor of 16, which is folded in here. */
/* The decoder keeps the sixteen luma-DC results in its own block slots and
 * renders slot i at raster position p2[i] (h->block_offset[] is built from
 * scan8), so the reconstructed picture only lines up after this reorder. */
void aeg_dc_block_unpermute(int32_t *dc)
{
    static const int p2[16] = { 0, 1, 4, 5, 2, 3, 6, 7,
                                8, 9, 12, 13, 10, 11, 14, 15 };
    int32_t t[16];
    int i;

    for (i = 0; i < 16; i++)
        t[i] = dc[i];
    for (i = 0; i < 16; i++)
        dc[i] = t[p2[i]];
}

void aeg_transform_dc4x4_fwd(const int16_t *dc, int16_t *level)
{
    /* Exact inverse of aeg_dequant_4x4_dc_out().  The decoder runs a
     * 4x4 Hadamard as two passes, temp = M * in (rows) followed by
     * out = temp * M (columns), and scatters the sixteen results to the 4x4
     * blocks as abase[i] + rsub[r].  M * M = 4 * I, so applying the very same
     * two passes to the gathered per-block DCs yields 16 * dc and the
     * quantiser makes up for it with four extra shift bits (see
     * aeg_quant_4x4_dc()). */
    static const int abase[4] = { 0, 2, 8, 10 };
    static const int rsub[4]  = { 0, 1, 4, 5 };
    /* The decoder's finished block array is not in plain raster order: the
     * sixteen Hadamard results land on blocks (2,4), (3,5), (10,12) and
     * (11,13) swapped.  Gathering through p2 undoes that as well. */
    static const int p2[16] = { 0, 1, 4, 5, 2, 3, 6, 7,
                                8, 9, 12, 13, 10, 11, 14, 15 };
    int t[16], u[16], i, r;

    for (i = 0; i < 4; i++)
        for (r = 0; r < 4; r++)
            t[4 * r + i] = dc[p2[abase[i] + rsub[r]]];

    for (i = 0; i < 4; i++) {
        int x0 = t[4 * i + 0], x1 = t[4 * i + 1];
        int x2 = t[4 * i + 2], x3 = t[4 * i + 3];
        u[4 * i + 0] = x0 + x1 + x2 + x3;
        u[4 * i + 1] = x0 + x1 - x2 - x3;
        u[4 * i + 2] = x0 - x1 - x2 + x3;
        u[4 * i + 3] = x0 - x1 + x2 - x3;
    }
    for (i = 0; i < 4; i++) {
        int x0 = u[0 + i], x1 = u[4 + i], x2 = u[8 + i], x3 = u[12 + i];
        t[0 + i]  = x0 + x1 + x2 + x3;
        t[4 + i]  = x0 + x1 - x2 - x3;
        t[8 + i]  = x0 - x1 - x2 + x3;
        t[12 + i] = x0 - x1 + x2 - x3;
    }
    for (i = 0; i < 16; i++)
        level[i] = (int16_t)t[i];
}

/* 4:2:0 chroma DC, again bit-exact with the reference decoder. */
void aeg_dequant_2x2_dc_out(const int16_t *level, int32_t *dc, int qp)
{
    int qmul = aeg_dequant_ls[qp % 6][0] << (qp / 6 + 6);
    int a = level[0], b = level[1], c = level[2], d = level[3];
    int e = a - b;
    a = a + b;
    b = c - d;
    c = c + d;
    /* The reference decoder keeps the dequantised block in int16_t, so the
     * values wrap at high quantisers; reproducing that keeps the recon
     * bit-identical to a real decoder. */
    dc[0] = (int16_t)((a + c) * qmul >> 7);
    dc[1] = (int16_t)((e + b) * qmul >> 7);
    dc[2] = (int16_t)((a - c) * qmul >> 7);
    dc[3] = (int16_t)((e - b) * qmul >> 7);
}

void aeg_transform_dc2x2_fwd(const int16_t *dc, int16_t *level)
{
    int a = dc[0], b = dc[1], c = dc[2], d = dc[3];
    level[0] = (int16_t)(a + b + c + d);
    level[1] = (int16_t)(a - b + c - d);
    level[2] = (int16_t)(a + b - c - d);
    level[3] = (int16_t)(a - b - c + d);
}

/* ------------------------------------------------------------------ */
/* Inverse transform + reconstruction                                  */
/* ------------------------------------------------------------------ */
/* Applies the inverse transform to the dequantised coefficients and adds
 * the result to the prediction.  `dc`, when non-NULL, replaces coefficient
 * 0 (Intra_16x16 luma DC path). */
void aeg_idct_add_block(uint8_t *dst, int stride, const int16_t *level,
                        int qp, const uint8_t *pred, int pred_stride,
                        const int32_t *dc_in)
{
    int32_t d[16];
    int t[16];
    int i, x, y, sh = qp / 6;
    int ac_nonzero = 0;

    for (y = 0; y < 4; y++)
        for (x = 0; x < 4; x++) {
            int cls = (x & 1) + (y & 1);
            int v = level[x * 4 + y];   /* level[] is in decoder order */
            if (v && (x | y))
                ac_nonzero = 1;
            d[y * 4 + x] = (int16_t)((int32_t)v * aeg_dequant_ls[qp % 6][cls] << sh);
        }
    if (pred && !ac_nonzero) {
        /* DC only: every pixel gets (dc + 32) >> 6 */
        int dc = dc_in ? (int16_t)*dc_in : d[0];
        int v = (dc + 32) >> 6;
        for (y = 0; y < 4; y++)
            for (x = 0; x < 4; x++) {
                int p = pred[y * pred_stride + x] + v;
                dst[y * stride + x] = (uint8_t)(p < 0 ? 0 : (p > 255 ? 255 : p));
            }
        return;
    }
    if (dc_in)
        d[0] = (int16_t)*dc_in;
    /* The reference decoder rounds the DC coefficient into the first pass and
     * then truncates in the second one (block[0] += 1 << 5 ... >> 6). */
    d[0] += 1 << 5;

    /* rows */
    for (i = 0; i < 4; i++) {
        int32_t s0 = d[i * 4 + 0] + d[i * 4 + 2];
        int32_t s1 = d[i * 4 + 0] - d[i * 4 + 2];
        int32_t s2 = (d[i * 4 + 1] >> 1) - d[i * 4 + 3];
        int32_t s3 = d[i * 4 + 1] + (d[i * 4 + 3] >> 1);
        t[i * 4 + 0] = (int16_t)(s0 + s3);
        t[i * 4 + 1] = (int16_t)(s1 + s2);
        t[i * 4 + 2] = (int16_t)(s1 - s2);
        t[i * 4 + 3] = (int16_t)(s0 - s3);
    }
    /* columns */
    for (i = 0; i < 4; i++) {
        int32_t s0 = t[0 * 4 + i] + t[2 * 4 + i];
        int32_t s1 = t[0 * 4 + i] - t[2 * 4 + i];
        int32_t s2 = (t[1 * 4 + i] >> 1) - t[3 * 4 + i];
        int32_t s3 = t[1 * 4 + i] + (t[3 * 4 + i] >> 1);
        d[0 * 4 + i] = (s0 + s3) >> 6;
        d[1 * 4 + i] = (s1 + s2) >> 6;
        d[2 * 4 + i] = (s1 - s2) >> 6;
        d[3 * 4 + i] = (s0 - s3) >> 6;
    }

    for (y = 0; y < 4; y++)
        for (x = 0; x < 4; x++) {
            int p = (pred ? pred[y * pred_stride + x] : 0) + (int)d[y * 4 + x];
            dst[y * stride + x] = (uint8_t)(p < 0 ? 0 : (p > 255 ? 255 : p));
        }
}

void aeg_recon_add(uint8_t *dst, int stride, const uint8_t *pred,
                   int pred_stride, const int16_t *res, int w, int h)
{
    int x, y;
    for (y = 0; y < h; y++)
        for (x = 0; x < w; x++) {
            int p = pred[y * pred_stride + x] + res[y * w + x];
            dst[y * stride + x] = (uint8_t)(p < 0 ? 0 : (p > 255 ? 255 : p));
        }
}

/* ------------------------------------------------------------------ */
/* Distortion                                                          */
/* ------------------------------------------------------------------ */
#if defined(__SSE2__)
/* One 16-byte row pair (w == 16) or a zero-extended 8/4-byte load. */
static inline int sad_row(__m128i *acc, const uint8_t *a, const uint8_t *b, int w)
{
    __m128i va, vb;
    if (w >= 16) {
        va = _mm_loadu_si128((const __m128i *)a);
        vb = _mm_loadu_si128((const __m128i *)b);
    } else if (w == 8) {
        va = _mm_loadl_epi64((const __m128i *)a);
        vb = _mm_loadl_epi64((const __m128i *)b);
    } else if (w == 4) {
        va = _mm_cvtsi32_si128(*(const int *)a);
        vb = _mm_cvtsi32_si128(*(const int *)b);
    } else {
        return 0;
    }
    *acc = _mm_add_epi64(*acc, _mm_sad_epu8(va, vb));
    return 1;
}

static inline int sad_reduce(__m128i acc)
{
    int64_t v[2];
    _mm_storeu_si128((__m128i *)v, acc);
    return (int)(v[0] + v[1]);
}
#endif

int aeg_sad(int w, int h, const uint8_t *a, int sa, const uint8_t *b, int sb)
{
    int x, y, sum = 0;
#if defined(__SSE2__)
    if (w == 16 || w == 8 || w == 4) {
        __m128i acc = _mm_setzero_si128();
        for (y = 0; y < h; y++)
            sad_row(&acc, a + (size_t)y * sa, b + (size_t)y * sb, w);
        return sad_reduce(acc);
    }
    if (w >= 32 && (w & 15) == 0) {
        __m128i acc = _mm_setzero_si128();
        int k, chunks = w >> 4;
        for (y = 0; y < h; y++)
            for (k = 0; k < chunks; k++)
                sad_row(&acc, a + (size_t)y * sa + k * 16,
                        b + (size_t)y * sb + k * 16, 16);
        return sad_reduce(acc);
    }
#endif
    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++)
            sum += abs((int)a[y * sa + x] - (int)b[y * sb + x]);
    }
    return sum;
}

#define HADAMARD4(d0,d1,d2,d3,s0,s1,s2,s3) do {                      \
    int t0_ = (s0) + (s1);                                           \
    int t1_ = (s0) - (s1);                                           \
    int t2_ = (s2) + (s3);                                           \
    int t3_ = (s2) - (s3);                                           \
    (d0) = t0_ + t2_;                                                \
    (d1) = t1_ + t3_;                                                \
    (d2) = t0_ - t2_;                                                \
    (d3) = t1_ - t3_;                                                \
} while (0)

#if defined(__SSE2__)
/* Hadamard transform of one 4x4 block, four int16 lanes per register: the
 * butterfly runs across the four rows first, then the block is transposed and
 * the same butterfly runs across the four columns.  The sum of absolute values
 * does not care about the order of the sixteen outputs, so this is the scalar
 * transform's result exactly (16-bit is enough: |coef| <= 255*16). */
static inline __m128i satd_row(const uint8_t *a, const uint8_t *b)
{
    const __m128i zero = _mm_setzero_si128();
    __m128i va = _mm_cvtsi32_si128(*(const int *)a);
    __m128i vb = _mm_cvtsi32_si128(*(const int *)b);
    return _mm_sub_epi16(_mm_unpacklo_epi8(va, zero), _mm_unpacklo_epi8(vb, zero));
}

static inline void hadamard4x4(__m128i *r0, __m128i *r1, __m128i *r2, __m128i *r3)
{
    const __m128i s0 = _mm_add_epi16(*r0, *r1), d0 = _mm_sub_epi16(*r0, *r1);
    const __m128i s1 = _mm_add_epi16(*r2, *r3), d1 = _mm_sub_epi16(*r2, *r3);
    *r0 = _mm_add_epi16(s0, s1);
    *r1 = _mm_sub_epi16(s0, s1);
    *r2 = _mm_sub_epi16(d0, d1);
    *r3 = _mm_add_epi16(d0, d1);
}

/* 4x4 transpose of int16 held four lanes per register (the upper four lanes of
 * each register are zero): the two column pairs the unpacks produce are split
 * apart again so that every register holds one column. */
static inline void transpose4x4(__m128i *r0, __m128i *r1, __m128i *r2, __m128i *r3)
{
    const __m128i z = _mm_setzero_si128();
    const __m128i t0 = _mm_unpacklo_epi16(*r0, *r1);
    const __m128i t1 = _mm_unpacklo_epi16(*r2, *r3);
    const __m128i lo = _mm_unpacklo_epi32(t0, t1);   /* column 0 | column 1 */
    const __m128i hi = _mm_unpackhi_epi32(t0, t1);   /* column 2 | column 3 */
    *r0 = _mm_unpacklo_epi64(lo, z);
    *r1 = _mm_unpackhi_epi64(lo, z);
    *r2 = _mm_unpacklo_epi64(hi, z);
    *r3 = _mm_unpackhi_epi64(hi, z);
}

static int satd_4x4_sse2(const uint8_t *a, int sa, const uint8_t *b, int sb)
{
    const __m128i zero = _mm_setzero_si128();
    const __m128i ones = _mm_set1_epi16(1);
    __m128i r0 = satd_row(a, b);
    __m128i r1 = satd_row(a + sa, b + sb);
    __m128i r2 = satd_row(a + 2 * sa, b + 2 * sb);
    __m128i r3 = satd_row(a + 3 * sa, b + 3 * sb);
    __m128i acc;

    hadamard4x4(&r0, &r1, &r2, &r3);
    transpose4x4(&r0, &r1, &r2, &r3);
    hadamard4x4(&r0, &r1, &r2, &r3);
    r0 = _mm_max_epi16(r0, _mm_sub_epi16(zero, r0));
    r1 = _mm_max_epi16(r1, _mm_sub_epi16(zero, r1));
    r2 = _mm_max_epi16(r2, _mm_sub_epi16(zero, r2));
    r3 = _mm_max_epi16(r3, _mm_sub_epi16(zero, r3));
    acc = _mm_add_epi32(_mm_madd_epi16(r0, ones), _mm_madd_epi16(r1, ones));
    acc = _mm_add_epi32(acc, _mm_madd_epi16(r2, ones));
    acc = _mm_add_epi32(acc, _mm_madd_epi16(r3, ones));
    acc = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, _MM_SHUFFLE(1, 0, 3, 2)));
    acc = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, _MM_SHUFFLE(2, 3, 0, 1)));
    return (_mm_cvtsi128_si32(acc) + 1) >> 1;
}
#endif

int aeg_satd_4x4(const uint8_t *a, int sa, const uint8_t *b, int sb)
{
    int t[16], d[16], i, y, sum = 0;
#if defined(__SSE2__)
    return satd_4x4_sse2(a, sa, b, sb);
#endif
    for (y = 0; y < 4; y++) {
        HADAMARD4(t[y * 4 + 0], t[y * 4 + 1], t[y * 4 + 2], t[y * 4 + 3],
                  (int)a[y * sa + 0] - (int)b[y * sb + 0],
                  (int)a[y * sa + 1] - (int)b[y * sb + 1],
                  (int)a[y * sa + 2] - (int)b[y * sb + 2],
                  (int)a[y * sa + 3] - (int)b[y * sb + 3]);
    }
    for (i = 0; i < 4; i++) {
        HADAMARD4(d[0 * 4 + i], d[1 * 4 + i], d[2 * 4 + i], d[3 * 4 + i],
                  t[0 * 4 + i], t[1 * 4 + i], t[2 * 4 + i], t[3 * 4 + i]);
    }
    for (i = 0; i < 16; i++)
        sum += abs(d[i]);
    return (sum + 1) >> 1;
}

int aeg_satd_8x8(const uint8_t *a, int sa, const uint8_t *b, int sb)
{
    int sum = 0, x, y;
    for (y = 0; y < 8; y += 4)
        for (x = 0; x < 8; x += 4)
            sum += aeg_satd_4x4(a + y * sa + x, sa, b + y * sb + x, sb);
    return sum;
}

int aeg_satd(int w, int h, const uint8_t *a, int sa, const uint8_t *b, int sb)
{
    int sum = 0, x, y;
    for (y = 0; y < h; y += 4)
        for (x = 0; x < w; x += 4)
            sum += aeg_satd_4x4(a + y * sa + x, sa, b + y * sb + x, sb);
    return sum;
}

int aeg_satd_16x16(const uint8_t *a, int sa, const uint8_t *b, int sb)
{
    return aeg_satd(16, 16, a, sa, b, sb);
}
