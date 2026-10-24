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

#if defined(__SSE2__)
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

#if defined(__SSE2__)
/* 4x4 forward transform, four lanes at a time.
 *
 * Every intermediate of the transform is an exact integer: a residual sample is
 * in [-255, 255], the first pass multiplies that by at most 4, the second by at
 * most 4 again, so |value| <= 4080 and every int16 lane holds it exactly.  The
 * final store truncates to int16 just like the scalar version, and the SSE2
 * path therefore produces the *same* coefficients, not merely close ones -- a
 * decision that reads them cannot change (tests/coef_check.c compares this
 * against the independent transcription of the transform that the tq_batch
 * device kernel carries). */
static inline void fdct4_1d_sse2(__m128i *r0, __m128i *r1, __m128i *r2,
                                __m128i *r3)
{
    /* d0..d3 -> rows (t0 + t1), (2 * t3 + t2), (t0 - t1), (t3 - 2 * t2) */
    const __m128i t0 = _mm_add_epi16(*r0, *r3);
    const __m128i t1 = _mm_add_epi16(*r1, *r2);
    const __m128i t2 = _mm_sub_epi16(*r1, *r2);
    const __m128i t3 = _mm_sub_epi16(*r0, *r3);

    *r0 = _mm_add_epi16(t0, t1);
    *r1 = _mm_add_epi16(_mm_slli_epi16(t3, 1), t2);
    *r2 = _mm_sub_epi16(t0, t1);
    *r3 = _mm_sub_epi16(t3, _mm_slli_epi16(t2, 1));
}

static void transform_4x4_sse2(const int16_t *res, int stride, int16_t *coef)
{
    __m128i r0 = _mm_loadl_epi64((const __m128i *)(res + 0 * stride));
    __m128i r1 = _mm_loadl_epi64((const __m128i *)(res + 1 * stride));
    __m128i r2 = _mm_loadl_epi64((const __m128i *)(res + 2 * stride));
    __m128i r3 = _mm_loadl_epi64((const __m128i *)(res + 3 * stride));

    /* C_f X C_f^T is separable, so the order of the two passes does not matter
     * -- but the result has to come out in the index order the quantiser and
     * the scan expect.  A register holds one *row* of the block, so the first
     * butterfly transforms the columns (register index), the transpose turns
     * that into a row pass, and the second transpose puts the result back into
     * (row, column) order.  Two transposes are ~16 instructions; the arithmetic
     * they save is an int->int16 shuffle per sample. */
    fdct4_1d_sse2(&r0, &r1, &r2, &r3);      /* columns                          */
    transpose4x4(&r0, &r1, &r2, &r3);
    fdct4_1d_sse2(&r0, &r1, &r2, &r3);      /* rows                             */
    transpose4x4(&r0, &r1, &r2, &r3);       /* back to (row, column)            */
    _mm_storel_epi64((__m128i *)(coef + 0), r0);
    _mm_storel_epi64((__m128i *)(coef + 4), r1);
    _mm_storel_epi64((__m128i *)(coef + 8), r2);
    _mm_storel_epi64((__m128i *)(coef + 12), r3);
}
#endif

void aeg_transform_4x4(const int16_t *res, int stride, int16_t *coef)
{
    int d[16];
    int i, t[16];

#if defined(__SSE2__)
    transform_4x4_sse2(res, stride, coef);
    return;
#endif

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

#if defined(__SSE2__)
/* Quantiser for the 4x4 block, sixteen coefficients at a time.  The multiply
 * goes through pmaddwd (|coef| <= 4080 for any residual, so the 16x16 product
 * is exact) and the result is stored transposed, which is the layout the
 * decoder expects -- see the scalar loop below. */
static int quant_4x4_sse2(const int16_t *coef, int16_t *level, int qp,
                          int intra)
{
    const int qbits = 15 + (qp / 6);
    const int f = aeg_quant_bias(qbits, intra ? 3 : 6);
    const int mf0 = aeg_quant_mf[qp % 6][0];
    const int mf1 = aeg_quant_mf[qp % 6][1];
    const int mf2 = aeg_quant_mf[qp % 6][2];
    const __m128i zero = _mm_setzero_si128();
    const __m128i qb = _mm_cvtsi32_si128(qbits);
    const __m128i fv = _mm_set1_epi32(f);
    const __m128i me = _mm_setr_epi16(mf0, 0, mf1, 0, mf0, 0, mf1, 0);
    const __m128i mo = _mm_setr_epi16(mf1, 0, mf2, 0, mf1, 0, mf2, 0);
    __m128i r0 = _mm_loadu_si128((const __m128i *)(coef + 0));
    __m128i r1 = _mm_loadu_si128((const __m128i *)(coef + 8));
    __m128i o0, o1, o2, o3;

    /* |coef| must fit 16 bits for the pmaddwd product below: the forward
     * transform of a residual never gets near this, but a caller passing
     * INT16_MIN would, and the scalar loop handles that case exactly. */
    if (_mm_movemask_epi8(_mm_or_si128(
            _mm_cmpeq_epi16(r0, _mm_set1_epi16(-32768)),
            _mm_cmpeq_epi16(r1, _mm_set1_epi16(-32768)))))
        return 0;

#define QUANT_ROW(r, m, sgn)                                                   \
    do {                                                                      \
        (sgn) = _mm_srai_epi16((r), 15);                                       \
        {                                                                     \
            const __m128i ab = _mm_sub_epi16(_mm_xor_si128((r), (sgn)), (sgn));\
            __m128i lv = _mm_madd_epi16(                                       \
                _mm_unpacklo_epi16(_mm_srli_si128(ab, 0), zero), (m));         \
            lv = _mm_sra_epi32(_mm_add_epi32(lv, fv), qb);                      \
            (r) = _mm_srai_epi16((r), 15);                                      \
            (r) = _mm_packs_epi32(                                             \
                _mm_sub_epi32(_mm_xor_si128(lv, _mm_unpacklo_epi16(sgn, sgn)),  \
                              _mm_unpacklo_epi16(sgn, sgn)), lv);              \
        }                                                                     \
    } while (0)

    /* r0/r1 hold coef rows 0-3 and 4-7; the odd rows use the [mf1 mf2] pattern */
    {
        __m128i s0, s1, s2, s3;
        __m128i c0 = _mm_unpacklo_epi64(r0, zero);      /* coef row 0 */
        __m128i c1 = _mm_unpackhi_epi64(r0, zero);      /* coef row 1 */
        __m128i c2 = _mm_unpacklo_epi64(r1, zero);      /* coef row 2 */
        __m128i c3 = _mm_unpackhi_epi64(r1, zero);      /* coef row 3 */
        QUANT_ROW(c0, me, s0);
        QUANT_ROW(c1, mo, s1);
        QUANT_ROW(c2, me, s2);
        QUANT_ROW(c3, mo, s3);
        o0 = c0; o1 = c1; o2 = c2; o3 = c3;
    }
#undef QUANT_ROW
    /* level[y * 4 + x] = quant(coef[x * 4 + y]): one transpose at the end */
    transpose4x4(&o0, &o1, &o2, &o3);
    _mm_storel_epi64((__m128i *)(level + 0), o0);
    _mm_storel_epi64((__m128i *)(level + 4), o1);
    _mm_storel_epi64((__m128i *)(level + 8), o2);
    _mm_storel_epi64((__m128i *)(level + 12), o3);
    return 1;
}
#endif

void aeg_quant_4x4(const int16_t *coef, int16_t *level, int qp, int intra)
{
#if defined(__SSE2__)
    if (quant_4x4_sse2(coef, level, qp, intra))
        return;
#endif
    /* The rounding offset and the scale factors depend only on qp (and on
     * whether this is an intra block), so they are computed once for the
     * sixteen coefficients: aeg_quant_bias() is a 64-bit division and this
     * function runs a million times per thirty frames. */
    const int qbits = 15 + (qp / 6);
    const int f = aeg_quant_bias(qbits, intra ? 3 : 6);
    const int mf0 = aeg_quant_mf[qp % 6][0];
    const int mf1 = aeg_quant_mf[qp % 6][1];
    const int mf2 = aeg_quant_mf[qp % 6][2];
    int x, y;
    /* The decoder's 4x4 inverse transform is the transpose of ours: a
     * coefficient in slot (row 1, column 0) comes out as a horizontal ramp
     * there and as a vertical one here.  Storing the quantised block
     * transposed therefore makes the transmitted coefficients reconstruct the
     * residual we measured, and the per-position scale factors are symmetric
     * in x and y, so the quantiser itself is unaffected. */
    for (y = 0; y < 4; y++)
        for (x = 0; x < 4; x++) {
            const int c = coef[x * 4 + y];
            const int cls = (x & 1) + (y & 1);
            const int mf = cls == 0 ? mf0 : (cls == 1 ? mf1 : mf2);
            const int abs = c < 0 ? -c : c;
            const int lv = (int)(((int64_t)abs * mf + f) >> qbits);
            level[y * 4 + x] = (int16_t)(c < 0 ? -lv : lv);
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
#if defined(__SSE2__)
/* Four int16 in the low lanes of a register, and the reverse. */
static inline __m128i load4_i16(const int16_t *p)
{
    return _mm_loadl_epi64((const __m128i *)p);
}

/* 4x4 transpose of int32 held one value per lane. */
static inline void transpose4x4_epi32(__m128i *r0, __m128i *r1, __m128i *r2,
                                      __m128i *r3)
{
    const __m128i t0 = _mm_unpacklo_epi32(*r0, *r1);
    const __m128i t1 = _mm_unpackhi_epi32(*r0, *r1);
    const __m128i t2 = _mm_unpacklo_epi32(*r2, *r3);
    const __m128i t3 = _mm_unpackhi_epi32(*r2, *r3);
    *r0 = _mm_unpacklo_epi64(t0, t2);
    *r1 = _mm_unpackhi_epi64(t0, t2);
    *r2 = _mm_unpacklo_epi64(t1, t3);
    *r3 = _mm_unpackhi_epi64(t1, t3);
}

/* The reference decoder's inverse transform, one pass: the four inputs are the
 * four values along one axis, held in the same lane of four registers. */
static inline void idct_pass(__m128i *a0, __m128i *a1, __m128i *a2, __m128i *a3)
{
    const __m128i s0 = _mm_add_epi32(*a0, *a2);
    const __m128i s1 = _mm_sub_epi32(*a0, *a2);
    const __m128i s2 = _mm_sub_epi32(_mm_srai_epi32(*a1, 1), *a3);
    const __m128i s3 = _mm_add_epi32(*a1, _mm_srai_epi32(*a3, 1));
    *a0 = _mm_add_epi32(s0, s3);
    *a1 = _mm_add_epi32(s1, s2);
    *a2 = _mm_sub_epi32(s1, s2);
    *a3 = _mm_sub_epi32(s0, s3);
}

/* Dequantisation + both inverse-transform passes + prediction + clip, in
 * int32 lanes: 3x the scalar block.  The scalar buffer truncates to int16
 * between the two passes; a quarter-pel stream that is conformant never leaves
 * the int16 range there (which is why the reference decoder can use int16
 * buffers at all), and the differential test in tests/run_tests.sh compares
 * this against the scalar loop on random levels at every QP to keep it that
 * way. */
static int idct_add_sse2(uint8_t *dst, int stride, const int16_t *level,
                         int qp, const uint8_t *pred, int pred_stride)
{
    const int sh = qp / 6;
    const int qm = qp % 6;
    const __m128i zero = _mm_setzero_si128();
    const __m128i shv = _mm_cvtsi32_si128(sh);
    const int16_t ls0 = aeg_dequant_ls[qm][0], ls1 = aeg_dequant_ls[qm][1],
                  ls2 = aeg_dequant_ls[qm][2];
    const __m128i e = _mm_setr_epi16(ls0, 0, ls1, 0, ls0, 0, ls1, 0);
    const __m128i o = _mm_setr_epi16(ls1, 0, ls2, 0, ls1, 0, ls2, 0);
    __m128i a0 = load4_i16(level + 0), a1 = load4_i16(level + 4);
    __m128i a2 = load4_i16(level + 8), a3 = load4_i16(level + 12);
    __m128i v;

    /* level[] is in decoder order, so register i lane j holds d[j][i]: the
     * register index is the block's column and the lane its row.  Two
     * consequences, both good: the dequantiser's (x&1)+(y&1) class pattern is
     * the same per register as it is per lane, and the *first* inverse pass --
     * the one along x -- is the one that runs across registers, so this needs
     * a single transpose and applies the passes in the same order (and at the
     * same int16 truncation points) as the scalar loop, which is the order the
     * reference decoder sees. */
#define IDCT_ROW8(p, off) \
    _mm_unpacklo_epi16( \
        _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int *)((p) + (off))), zero), zero)
    /* DC-only blocks (very common: a macroblock that predicts well leaves one
     * coefficient per block) skip the transform entirely, as in the scalar
     * loop -- there the fast path is worth more than the transform is. */
    {
        const __m128i dc_mask = _mm_setr_epi16(-1, 0, 0, 0, 0, 0, 0, 0);
        /* the DC coefficient is the first one; every other one must be zero
         * for the single-value shortcut to be exact */
        const __m128i ac = _mm_or_si128(_mm_andnot_si128(dc_mask, a0),
                                        _mm_or_si128(_mm_or_si128(a1, a2), a3));
        /* Only the prediction variant has this shortcut in the scalar loop;
         * without a prediction the scalar runs the transform, whose int16
         * intermediate truncation a very large DC coefficient does depend on.
         */
        if (pred && _mm_movemask_epi8(_mm_cmpeq_epi16(ac, zero)) == 0xFFFF) {
            const int dc = (int16_t)(((int32_t)(int16_t)_mm_extract_epi16(a0, 0) *
                                      ls0) << sh);
            const int add = (dc + (1 << 5)) >> 6;
            const __m128i bias = _mm_set1_epi32(add);
            __m128i p0 = bias, p1 = bias, p2 = bias, p3 = bias;
            if (pred) {
                p0 = _mm_add_epi32(bias, IDCT_ROW8(pred, 0));
                p1 = _mm_add_epi32(bias, IDCT_ROW8(pred, pred_stride));
                p2 = _mm_add_epi32(bias, IDCT_ROW8(pred, 2 * pred_stride));
                p3 = _mm_add_epi32(bias, IDCT_ROW8(pred, 3 * pred_stride));
            }
            v = _mm_packus_epi16(_mm_packs_epi32(p0, p1), _mm_packs_epi32(p2, p3));
            *(int *)(dst) = _mm_cvtsi128_si32(v);
            *(int *)(dst + stride) = _mm_cvtsi128_si32(_mm_srli_si128(v, 4));
            *(int *)(dst + 2 * stride) = _mm_cvtsi128_si32(_mm_srli_si128(v, 8));
            *(int *)(dst + 3 * stride) = _mm_cvtsi128_si32(_mm_srli_si128(v, 12));
            return 1;
        }
    }
    a0 = _mm_madd_epi16(_mm_unpacklo_epi16(a0, zero), e);
    a1 = _mm_madd_epi16(_mm_unpacklo_epi16(a1, zero), o);
    a2 = _mm_madd_epi16(_mm_unpacklo_epi16(a2, zero), e);
    a3 = _mm_madd_epi16(_mm_unpacklo_epi16(a3, zero), o);
    a0 = _mm_sll_epi32(a0, shv);
    a1 = _mm_sll_epi32(a1, shv);
    a2 = _mm_sll_epi32(a2, shv);
    a3 = _mm_sll_epi32(a3, shv);
    /* the scalar's (int16_t) cast after the dequantiser ... */
    a0 = _mm_srai_epi32(_mm_slli_epi32(a0, 16), 16);
    a1 = _mm_srai_epi32(_mm_slli_epi32(a1, 16), 16);
    a2 = _mm_srai_epi32(_mm_slli_epi32(a2, 16), 16);
    a3 = _mm_srai_epi32(_mm_slli_epi32(a3, 16), 16);
    /* ... and only then the reference decoder's DC rounding, which the scalar
     * loop also adds after that cast */
    a0 = _mm_add_epi32(a0, _mm_setr_epi32(1 << 5, 0, 0, 0));
    idct_pass(&a0, &a1, &a2, &a3);
    /* and the one between the two passes */
    a0 = _mm_srai_epi32(_mm_slli_epi32(a0, 16), 16);
    a1 = _mm_srai_epi32(_mm_slli_epi32(a1, 16), 16);
    a2 = _mm_srai_epi32(_mm_slli_epi32(a2, 16), 16);
    a3 = _mm_srai_epi32(_mm_slli_epi32(a3, 16), 16);
    transpose4x4_epi32(&a0, &a1, &a2, &a3);
    idct_pass(&a0, &a1, &a2, &a3);
    a0 = _mm_srai_epi32(a0, 6);
    a1 = _mm_srai_epi32(a1, 6);
    a2 = _mm_srai_epi32(a2, 6);
    a3 = _mm_srai_epi32(a3, 6);
    if (pred) {
        a0 = _mm_add_epi32(a0, IDCT_ROW8(pred, 0));
        a1 = _mm_add_epi32(a1, IDCT_ROW8(pred, pred_stride));
        a2 = _mm_add_epi32(a2, IDCT_ROW8(pred, 2 * pred_stride));
        a3 = _mm_add_epi32(a3, IDCT_ROW8(pred, 3 * pred_stride));
    }
#undef IDCT_ROW8
    /* saturating packs do exactly the scalar's clip to [0, 255] */
    v = _mm_packus_epi16(_mm_packs_epi32(a0, a1), _mm_packs_epi32(a2, a3));
    *(int *)(dst) = _mm_cvtsi128_si32(v);
    *(int *)(dst + stride) = _mm_cvtsi128_si32(_mm_srli_si128(v, 4));
    *(int *)(dst + 2 * stride) = _mm_cvtsi128_si32(_mm_srli_si128(v, 8));
    *(int *)(dst + 3 * stride) = _mm_cvtsi128_si32(_mm_srli_si128(v, 12));
    return 1;
}
#endif

/* The scalar inverse transform: dequantise (truncating every coefficient to
 * int16, as the reference decoder effectively does -- bug 8), round the DC
 * into the first pass, truncate through the second one.  `res` receives the
 * signed residual; without a prediction to add it is also what the batched
 * device path returns, so this is the single definition of the arithmetic. */
#if defined(__SSE2__)
/* One pass of the inverse transform, four values at a time.  The registers are
 * the four *input* values and the four outputs replace them, so a pass along
 * one axis is four register operations whatever the axis is -- which is why the
 * block has to be transposed between the two passes. */
static inline void idct4_1d_epi32(__m128i *r0, __m128i *r1, __m128i *r2,
                                  __m128i *r3)
{
    const __m128i s0 = _mm_add_epi32(*r0, *r2);
    const __m128i s1 = _mm_sub_epi32(*r0, *r2);
    const __m128i s2 = _mm_sub_epi32(_mm_srai_epi32(*r1, 1), *r3);
    const __m128i s3 = _mm_add_epi32(*r1, _mm_srai_epi32(*r3, 1));

    *r0 = _mm_add_epi32(s0, s3);
    *r1 = _mm_add_epi32(s1, s2);
    *r2 = _mm_sub_epi32(s1, s2);
    *r3 = _mm_sub_epi32(s0, s3);
}

/* The scalar code truncates every intermediate to int16; a 32-bit lane does the
 * same with two shifts (the reference decoder's dequantised coefficients are
 * int16, so this is not an optimisation of its arithmetic but part of it -- see
 * bug 8). */
static inline __m128i trunc16_epi32(__m128i v)
{
    return _mm_srai_epi32(_mm_slli_epi32(v, 16), 16);
}
#endif

static void idct_core(const int16_t *level, int qp, const int32_t *dc_in,
                      int16_t *res)
{
    int32_t d[16];
    int t[16];
    int i, x, y, sh = qp / 6;
#if defined(__SSE2__)
    __m128i r0, r1, r2, r3;
#endif

    for (y = 0; y < 4; y++)
        for (x = 0; x < 4; x++) {
            int cls = (x & 1) + (y & 1);
            int v = level[x * 4 + y];   /* level[] is in decoder order */
            d[y * 4 + x] = (int16_t)((int32_t)v * aeg_dequant_ls[qp % 6][cls] << sh);
        }
    if (dc_in)
        d[0] = (int16_t)*dc_in;
    /* The reference decoder rounds the DC coefficient into the first pass and
     * then truncates in the second one (block[0] += 1 << 5 ... >> 6). */
    d[0] += 1 << 5;

#if defined(__SSE2__)
    /* Same two passes, same two int16 truncation points, four lanes at a time.
     * The first pass runs along the *rows* of the block, so it has to run along
     * the register index: load (and transpose) so that a register holds a
     * column, transform across the registers, truncate, transpose back, and the
     * second pass -- which runs along the columns -- becomes a transform across
     * the registers again. */
    r0 = _mm_loadu_si128((const __m128i *)(d + 0));
    r1 = _mm_loadu_si128((const __m128i *)(d + 4));
    r2 = _mm_loadu_si128((const __m128i *)(d + 8));
    r3 = _mm_loadu_si128((const __m128i *)(d + 12));
    transpose4x4_epi32(&r0, &r1, &r2, &r3);              /* registers = columns */
    idct4_1d_epi32(&r0, &r1, &r2, &r3);                  /* the row pass        */
    r0 = trunc16_epi32(r0); r1 = trunc16_epi32(r1);
    r2 = trunc16_epi32(r2); r3 = trunc16_epi32(r3);
    transpose4x4_epi32(&r0, &r1, &r2, &r3);
    idct4_1d_epi32(&r0, &r1, &r2, &r3);                  /* the column pass     */
    r0 = _mm_srai_epi32(r0, 6); r1 = _mm_srai_epi32(r1, 6);
    r2 = _mm_srai_epi32(r2, 6); r3 = _mm_srai_epi32(r3, 6);
    _mm_storel_epi64((__m128i *)(res + 0), _mm_packs_epi32(r0, r0));
    _mm_storel_epi64((__m128i *)(res + 4), _mm_packs_epi32(r1, r1));
    _mm_storel_epi64((__m128i *)(res + 8), _mm_packs_epi32(r2, r2));
    _mm_storel_epi64((__m128i *)(res + 12), _mm_packs_epi32(r3, r3));
    return;
#endif

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
        /* |t| <= 32767, so the result always fits int16 */
        res[0 * 4 + i] = (int16_t)((s0 + s3) >> 6);
        res[1 * 4 + i] = (int16_t)((s1 + s2) >> 6);
        res[2 * 4 + i] = (int16_t)((s1 - s2) >> 6);
        res[3 * 4 + i] = (int16_t)((s0 - s3) >> 6);
    }
}

void aeg_idct_residual(const int16_t *level, int qp, int16_t *res)
{
    idct_core(level, qp, NULL, res);
}

static void idct_add_scalar(uint8_t *dst, int stride, const int16_t *level,
                            int qp, const uint8_t *pred, int pred_stride,
                            const int32_t *dc_in)
{
    int16_t res[16];
    int x, y, sh = qp / 6;
    int ac_nonzero = 0;

    if (pred) {
        for (y = 0; y < 4 && !ac_nonzero; y++)
            for (x = 0; x < 4; x++)
                if (level[x * 4 + y] && (x | y)) {
                    ac_nonzero = 1;
                    break;
                }
    }
    if (pred && !ac_nonzero) {
        /* DC only: every pixel gets (dc + 32) >> 6 */
        int dc = dc_in ? (int16_t)*dc_in
                       : (int16_t)((int32_t)level[0] *
                                   aeg_dequant_ls[qp % 6][0] << sh);
        int v = (dc + 32) >> 6;
        for (y = 0; y < 4; y++)
            for (x = 0; x < 4; x++) {
                int p = pred[y * pred_stride + x] + v;
                dst[y * stride + x] = (uint8_t)(p < 0 ? 0 : (p > 255 ? 255 : p));
            }
        return;
    }
    idct_core(level, qp, dc_in, res);
    for (y = 0; y < 4; y++)
        for (x = 0; x < 4; x++) {
            int p = (pred ? pred[y * pred_stride + x] : 0) + (int)res[y * 4 + x];
            dst[y * stride + x] = (uint8_t)(p < 0 ? 0 : (p > 255 ? 255 : p));
        }
}

void aeg_idct_add_block(uint8_t *dst, int stride, const int16_t *level,
                        int qp, const uint8_t *pred, int pred_stride,
                        const int32_t *dc_in)
{
#if defined(__SSE2__)
    /* The Intra_16x16 DC path (dc_in) keeps the scalar loop: it is one block per
     * macroblock and it overrides coefficient 0 from the DC Hadamard. */
    if (!dc_in && idct_add_sse2(dst, stride, level, qp, pred, pred_stride))
        return;
#endif
    idct_add_scalar(dst, stride, level, qp, pred, pred_stride, dc_in);
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
