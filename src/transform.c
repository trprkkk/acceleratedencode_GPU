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
static inline int quant_coef(int32_t coef, int qp, int cls, int bias_num,
                             int bias_den)
{
    const int qbits = 15 + (qp / 6);
    const int mf = aeg_quant_mf[qp % 6][cls];
    const int f = (1 << qbits) / bias_den * bias_num;
    int abs = coef < 0 ? -coef : coef;
    int level = (int)(((int64_t)abs * mf + f) >> qbits);
    return coef < 0 ? -level : level;
}

void aeg_quant_4x4(const int16_t *coef, int16_t *level, int qp, int intra)
{
    int x, y;
    for (y = 0; y < 4; y++)
        for (x = 0; x < 4; x++) {
            int cls = (x & 1) + (y & 1);
            level[y * 4 + x] = (int16_t)quant_coef(coef[y * 4 + x], qp, cls,
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
    const int f = intra ? (1 << qbits) / 3 : (1 << qbits) / 6;
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
    const int f = intra ? (1 << qbits) / 3 : (1 << qbits) / 6;
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
            coef[y * 4 + x] = (int32_t)level[y * 4 + x] *
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
        dc[base[i] + 0] = ((z0 + z3) * qmul + 128) >> 8;
        dc[base[i] + 1] = ((z1 + z2) * qmul + 128) >> 8;
        dc[base[i] + 4] = ((z1 - z2) * qmul + 128) >> 8;
        dc[base[i] + 5] = ((z0 - z3) * qmul + 128) >> 8;
    }
}

/* Forward counterpart: the DC levels an Intra_16x16 macroblock has to transmit
 * for `dc[]` to come back out of the function above.  The Hadamard is its own
 * inverse up to a factor of 16, which is folded in here. */
void aeg_transform_dc4x4_fwd(const int16_t *dc, int16_t *level)
{
    static const int base[4] = { 0, 2, 8, 10 };
    int y[16], temp[16];
    int i;

    /* undo the output permutation of aeg_dequant_4x4_dc_out() */
    for (i = 0; i < 4; i++) {
        int s0 = dc[base[i] + 0] + dc[base[i] + 5];
        int s1 = dc[base[i] + 1] + dc[base[i] + 4];
        int s2 = dc[base[i] + 1] - dc[base[i] + 4];
        int s3 = dc[base[i] + 0] - dc[base[i] + 5];
        temp[4 * i + 0] = s0 + s1;
        temp[4 * i + 1] = s0 - s1;
        temp[4 * i + 2] = s3 + s2;
        temp[4 * i + 3] = s3 - s2;
    }
    for (i = 0; i < 4; i++) {
        int a = temp[0 + i], b = temp[4 + i], c = temp[8 + i], d = temp[12 + i];
        int s0 = a + d, s1 = b + c, s2 = b - c, s3 = a - d;
        y[4 * i + 0] = s0 + s1;
        y[4 * i + 1] = s0 - s1;
        y[4 * i + 2] = s3 + s2;
        y[4 * i + 3] = s3 - s2;
    }
    /* The Hadamard gain (16) is compensated by the quantiser shift below, so
     * the raw sum is what the quantiser expects -- exactly like the reference
     * encoder's dct4x4dc(). */
    for (i = 0; i < 16; i++)
        level[i] = (int16_t)y[i];
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
    dc[0] = (a + c) * qmul >> 7;
    dc[1] = (e + b) * qmul >> 7;
    dc[2] = (a - c) * qmul >> 7;
    dc[3] = (e - b) * qmul >> 7;
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
            int v = level[y * 4 + x];
            if (v && (x | y))
                ac_nonzero = 1;
            d[y * 4 + x] = (int32_t)v * aeg_dequant_ls[qp % 6][cls] << sh;
        }
    if (pred && !ac_nonzero) {
        /* DC only: every pixel gets (dc + 32) >> 6 */
        int v = ((dc_in ? *dc_in : d[0]) + 32) >> 6;
        for (y = 0; y < 4; y++)
            for (x = 0; x < 4; x++) {
                int p = pred[y * pred_stride + x] + v;
                dst[y * stride + x] = (uint8_t)(p < 0 ? 0 : (p > 255 ? 255 : p));
            }
        return;
    }
    if (dc_in)
        d[0] = *dc_in;

    /* rows */
    for (i = 0; i < 4; i++) {
        int32_t s0 = d[i * 4 + 0] + d[i * 4 + 2];
        int32_t s1 = d[i * 4 + 0] - d[i * 4 + 2];
        int32_t s2 = (d[i * 4 + 1] >> 1) - d[i * 4 + 3];
        int32_t s3 = d[i * 4 + 1] + (d[i * 4 + 3] >> 1);
        t[i * 4 + 0] = (int)(s0 + s3);
        t[i * 4 + 1] = (int)(s1 + s2);
        t[i * 4 + 2] = (int)(s1 - s2);
        t[i * 4 + 3] = (int)(s0 - s3);
    }
    /* columns */
    for (i = 0; i < 4; i++) {
        int32_t s0 = t[0 * 4 + i] + t[2 * 4 + i];
        int32_t s1 = t[0 * 4 + i] - t[2 * 4 + i];
        int32_t s2 = (t[1 * 4 + i] >> 1) - t[3 * 4 + i];
        int32_t s3 = t[1 * 4 + i] + (t[3 * 4 + i] >> 1);
        d[0 * 4 + i] = (s0 + s3 + 32) >> 6;
        d[1 * 4 + i] = (s1 + s2 + 32) >> 6;
        d[2 * 4 + i] = (s1 - s2 + 32) >> 6;
        d[3 * 4 + i] = (s0 - s3 + 32) >> 6;
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
int aeg_sad(int w, int h, const uint8_t *a, int sa, const uint8_t *b, int sb)
{
    int x, y, sum = 0;
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

int aeg_satd_4x4(const uint8_t *a, int sa, const uint8_t *b, int sb)
{
    int t[16], d[16], i, y, sum = 0;
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

int aeg_satd_16x16(const uint8_t *a, int sa, const uint8_t *b, int sb)
{
    int sum = 0, x, y;
    for (y = 0; y < 16; y += 4)
        for (x = 0; x < 16; x += 4)
            sum += aeg_satd_4x4(a + y * sa + x, sa, b + y * sb + x, sb);
    return sum;
}
