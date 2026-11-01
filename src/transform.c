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
#include <pthread.h>
#include <string.h>

/* 32-bit loads and stores at any alignment (one movd; a cast to int * is
 * undefined behaviour on an unaligned pixel address) */
static inline int ld32(const void *p) { int v; memcpy(&v, p, 4); return v; }
static inline void st32(void *p, int v) { memcpy(p, &v, 4); }

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

/* AVX2 kernels are compiled per function (target attribute) and chosen at run
 * time, so the build stays baseline x86-64; AEG_NOAVX2=1 forces the SSE2 ones.
 * Every AVX2 kernel computes exactly what its SSE2/C counterpart does. */
#if defined(__SSE2__) && defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define AEG_X86_AVX2 1
static int cpu_avx2(void)
{
    static _Atomic int v = -1;
    if (v < 0) {
        const char *ev = getenv("AEG_NOAVX2");
        __builtin_cpu_init();
        v = (ev && atoi(ev)) ? 0 : __builtin_cpu_supports("avx2") != 0;
    }
    return v;
}
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
    /* the two denominators in use are exact shifts: 2^q * 3 / 6 and / 12 */
    if (QUANT_BIAS_NUM == 3 && QUANT_BIAS_DEN == 2 && qbits >= 2) {
        if (den == 3)
            return 1 << (qbits - 1);
        if (den == 6)
            return 1 << (qbits - 2);
    }
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
    /* All sixteen at once: |coef| <= 32768 read as unsigned and mf < 2^14,
     * so the unsigned 16x16 product (mullo / mulhi_epu16) plus the rounding
     * offset fits a 32-bit lane, and the level (< 2^15) packs back exactly. */
    const int qbits = 15 + (qp / 6);
    const int f = aeg_quant_bias(qbits, intra ? 3 : 6);
    const int mf0 = aeg_quant_mf[qp % 6][0];
    const int mf1 = aeg_quant_mf[qp % 6][1];
    const int mf2 = aeg_quant_mf[qp % 6][2];
    const __m128i qb = _mm_cvtsi32_si128(qbits);
    const __m128i fv = _mm_set1_epi32(f);
    /* coef rows 0/2 use [mf0 mf1 mf0 mf1], rows 1/3 [mf1 mf2 mf1 mf2] */
    const __m128i mf = _mm_setr_epi16(mf0, mf1, mf0, mf1, mf1, mf2, mf1, mf2);
    __m128i a = _mm_loadu_si128((const __m128i *)(coef + 0));
    __m128i b = _mm_loadu_si128((const __m128i *)(coef + 8));
    __m128i sa = _mm_srai_epi16(a, 15), sb = _mm_srai_epi16(b, 15);
    __m128i aa = _mm_sub_epi16(_mm_xor_si128(a, sa), sa);
    __m128i ab = _mm_sub_epi16(_mm_xor_si128(b, sb), sb);
    __m128i la = _mm_mullo_epi16(aa, mf), ha = _mm_mulhi_epu16(aa, mf);
    __m128i lb = _mm_mullo_epi16(ab, mf), hb = _mm_mulhi_epu16(ab, mf);
    __m128i p0 = _mm_srl_epi32(_mm_add_epi32(_mm_unpacklo_epi16(la, ha), fv), qb);
    __m128i p1 = _mm_srl_epi32(_mm_add_epi32(_mm_unpackhi_epi16(la, ha), fv), qb);
    __m128i p2 = _mm_srl_epi32(_mm_add_epi32(_mm_unpacklo_epi16(lb, hb), fv), qb);
    __m128i p3 = _mm_srl_epi32(_mm_add_epi32(_mm_unpackhi_epi16(lb, hb), fv), qb);
    __m128i t0, t1;
    a = _mm_packs_epi32(p0, p1);
    b = _mm_packs_epi32(p2, p3);
    a = _mm_sub_epi16(_mm_xor_si128(a, sa), sa);
    b = _mm_sub_epi16(_mm_xor_si128(b, sb), sb);
    /* level[y * 4 + x] = quant(coef[x * 4 + y]): transpose the 4x4 */
    t0 = _mm_unpacklo_epi16(a, b);
    t1 = _mm_unpackhi_epi16(a, b);
    _mm_storeu_si128((__m128i *)(level + 0), _mm_unpacklo_epi16(t0, t1));
    _mm_storeu_si128((__m128i *)(level + 8), _mm_unpackhi_epi16(t0, t1));
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
                              aeg_dequant_ls[qp % 6][cls] * (1 << sh);
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
                         int qp, const uint8_t *pred, int pred_stride,
                         const int32_t *dc_in)
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
        _mm_unpacklo_epi8(_mm_cvtsi32_si128(ld32((p) + (off))), zero), zero)
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
            const int dc = dc_in ? (int16_t)*dc_in :
                           (int16_t)((int32_t)(int16_t)_mm_extract_epi16(a0, 0) *
                                     ls0 * (1 << sh));
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
            st32(dst, _mm_cvtsi128_si32(v));
            st32(dst + stride, _mm_cvtsi128_si32(_mm_srli_si128(v, 4)));
            st32(dst + 2 * stride, _mm_cvtsi128_si32(_mm_srli_si128(v, 8)));
            st32(dst + 3 * stride, _mm_cvtsi128_si32(_mm_srli_si128(v, 12)));
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
    /* the Intra_16x16 path replaces coefficient 0 with the DC Hadamard's
     * value, truncated the same way */
    if (dc_in)
        a0 = _mm_or_si128(_mm_andnot_si128(_mm_setr_epi32(-1, 0, 0, 0), a0),
                          _mm_cvtsi32_si128((int16_t)*dc_in));
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
    st32(dst, _mm_cvtsi128_si32(v));
    st32(dst + stride, _mm_cvtsi128_si32(_mm_srli_si128(v, 4)));
    st32(dst + 2 * stride, _mm_cvtsi128_si32(_mm_srli_si128(v, 8)));
    st32(dst + 3 * stride, _mm_cvtsi128_si32(_mm_srli_si128(v, 12)));
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
            d[y * 4 + x] = (int16_t)((int32_t)v * aeg_dequant_ls[qp % 6][cls] * (1 << sh));
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

void aeg_idct_residual_dc(const int16_t *level, int qp, int32_t dc,
                          int16_t *res)
{
    idct_core(level, qp, &dc, res);
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
                                   aeg_dequant_ls[qp % 6][0] * (1 << sh));
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
    /* The Intra_16x16 AC blocks (dc_in) too: coefficient 0 is overridden from
     * the DC Hadamard at the scalar loop's truncation point. */
    if (idct_add_sse2(dst, stride, level, qp, pred, pred_stride, dc_in))
        return;
#endif
    idct_add_scalar(dst, stride, level, qp, pred, pred_stride, dc_in);
}

/* aeg_idct_add_block() into a scratch block, and its SSE against src. */
int64_t aeg_idct_add_ssd4(const int16_t *level, int qp, const uint8_t *pred,
                          int pred_stride, const int32_t *dc_in,
                          const uint8_t *src, int stride)
{
    uint8_t r[16];
    aeg_idct_add_block(r, 4, level, qp, pred, pred_stride, dc_in);
#if defined(__SSE2__)
    {
        const __m128i z = _mm_setzero_si128();
        __m128i va = _mm_loadu_si128((const __m128i *)r);
        __m128i vb = _mm_setr_epi32(ld32(src), ld32(src + stride),
                                    ld32(src + 2 * stride),
                                    ld32(src + 3 * stride));
        __m128i d0 = _mm_sub_epi16(_mm_unpacklo_epi8(va, z), _mm_unpacklo_epi8(vb, z));
        __m128i d1 = _mm_sub_epi16(_mm_unpackhi_epi8(va, z), _mm_unpackhi_epi8(vb, z));
        __m128i acc = _mm_add_epi32(_mm_madd_epi16(d0, d0), _mm_madd_epi16(d1, d1));
        acc = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, _MM_SHUFFLE(1, 0, 3, 2)));
        acc = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, _MM_SHUFFLE(2, 3, 0, 1)));
        return (uint32_t)_mm_cvtsi128_si32(acc);
    }
#else
    return aeg_ssd(4, 4, r, 4, src, stride);
#endif
}

/* The scalar reference, for the differential test. */
void aeg_idct_add_block_c(uint8_t *dst, int stride, const int16_t *level,
                          int qp, const uint8_t *pred, int pred_stride,
                          const int32_t *dc_in)
{
    idct_add_scalar(dst, stride, level, qp, pred, pred_stride, dc_in);
}

/* Sum of squared differences of two blocks of `w` x `h` (w = 4, 8 or 16). */
int64_t aeg_ssd(int w, int h, const uint8_t *a, int sa, const uint8_t *b,
                int sb)
{
#if defined(__SSE2__)
    const __m128i z = _mm_setzero_si128();
    __m128i acc = z;
    int y;
    if ((w == 4 && (h & 3) == 0) || (w == 8 && (h & 1) == 0)) {
        /* four 4-byte rows or two 8-byte rows per register */
        for (y = 0; y < h; y += 16 / w) {
            const uint8_t *pa = a + (size_t)y * sa, *pb = b + (size_t)y * sb;
            __m128i va, vb, d0, d1;
            if (w == 4) {
                va = _mm_setr_epi32(ld32(pa), ld32(pa + sa),
                                    ld32(pa + 2 * sa),
                                    ld32(pa + 3 * sa));
                vb = _mm_setr_epi32(ld32(pb), ld32(pb + sb),
                                    ld32(pb + 2 * sb),
                                    ld32(pb + 3 * sb));
            } else {
                va = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *)pa),
                                        _mm_loadl_epi64((const __m128i *)(pa + sa)));
                vb = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *)pb),
                                        _mm_loadl_epi64((const __m128i *)(pb + sb)));
            }
            d0 = _mm_sub_epi16(_mm_unpacklo_epi8(va, z), _mm_unpacklo_epi8(vb, z));
            d1 = _mm_sub_epi16(_mm_unpackhi_epi8(va, z), _mm_unpackhi_epi8(vb, z));
            acc = _mm_add_epi32(acc, _mm_add_epi32(_mm_madd_epi16(d0, d0),
                                                   _mm_madd_epi16(d1, d1)));
        }
        h = 0;
    }
    for (y = 0; y < h; y++) {
        __m128i va, vb, d;
        if (w == 16) {
            va = _mm_loadu_si128((const __m128i *)(a + (size_t)y * sa));
            vb = _mm_loadu_si128((const __m128i *)(b + (size_t)y * sb));
            d = _mm_sub_epi16(_mm_unpackhi_epi8(va, z), _mm_unpackhi_epi8(vb, z));
            acc = _mm_add_epi32(acc, _mm_madd_epi16(d, d));
        } else if (w == 8) {
            va = _mm_loadl_epi64((const __m128i *)(a + (size_t)y * sa));
            vb = _mm_loadl_epi64((const __m128i *)(b + (size_t)y * sb));
        } else {
            va = _mm_cvtsi32_si128(ld32(a + (size_t)y * sa));
            vb = _mm_cvtsi32_si128(ld32(b + (size_t)y * sb));
        }
        d = _mm_sub_epi16(_mm_unpacklo_epi8(va, z), _mm_unpacklo_epi8(vb, z));
        acc = _mm_add_epi32(acc, _mm_madd_epi16(d, d));
    }
    acc = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, _MM_SHUFFLE(1, 0, 3, 2)));
    acc = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, _MM_SHUFFLE(2, 3, 0, 1)));
    return (uint32_t)_mm_cvtsi128_si32(acc);
#else
    int64_t s = 0;
    int x, y;
    for (y = 0; y < h; y++)
        for (x = 0; x < w; x++) {
            int d = (int)a[y * sa + x] - (int)b[y * sb + x];
            s += d * d;
        }
    return s;
#endif
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

/* A macroblock's luma reconstruction from batched residuals: `rec` holds the
 * sixteen 4x4 blocks (contiguous, raster order inside each block and between
 * them, as iq_idct_batch() writes them), `pred` is 16x16 with stride 16.
 * dst = clip(pred + rec), and the return value is the SSE of dst against
 * src -- one pass instead of a per-pixel clip loop followed by aeg_ssd().
 * Exact: pred + res saturated to int16 and then packed with unsigned
 * saturation clips to 0..255 exactly as the scalar form does (a sum outside
 * int16 is outside 0..255 either way), and the SSE is an integer sum
 * (at most 256 * 255^2, so int32 lanes cannot overflow). */
static uint64_t add_res16_ssd_c(uint8_t *dst, int stride, const uint8_t *pred,
                                const int16_t *rec, const uint8_t *src,
                                int sstride)
{
    uint64_t sse = 0;
    int x, y;
    for (y = 0; y < 16; y++)
        for (x = 0; x < 16; x++) {
            int r = rec[16 * ((y >> 2) * 4 + (x >> 2)) + (y & 3) * 4 + (x & 3)];
            int p = (int)pred[y * 16 + x] + r, d;
            p = p < 0 ? 0 : (p > 255 ? 255 : p);
            dst[y * stride + x] = (uint8_t)p;
            d = p - (int)src[y * sstride + x];
            sse += (uint64_t)(d * d);
        }
    return sse;
}

uint64_t aeg_add_res16_ssd(uint8_t *dst, int stride, const uint8_t *pred,
                           const int16_t *rec, const uint8_t *src, int sstride)
{
#if defined(__SSE2__)
    const __m128i z = _mm_setzero_si128();
    __m128i acc = z;
    int y;
    for (y = 0; y < 16; y++) {
        const int16_t *r = rec + 64 * (y >> 2) + (y & 3) * 4;
        __m128i lo = _mm_unpacklo_epi64(
            _mm_loadl_epi64((const __m128i *)(r + 0)),
            _mm_loadl_epi64((const __m128i *)(r + 16)));
        __m128i hi = _mm_unpacklo_epi64(
            _mm_loadl_epi64((const __m128i *)(r + 32)),
            _mm_loadl_epi64((const __m128i *)(r + 48)));
        __m128i p = _mm_loadu_si128((const __m128i *)(pred + y * 16));
        __m128i o = _mm_packus_epi16(
            _mm_adds_epi16(_mm_unpacklo_epi8(p, z), lo),
            _mm_adds_epi16(_mm_unpackhi_epi8(p, z), hi));
        __m128i sv = _mm_loadu_si128((const __m128i *)(src + y * sstride));
        __m128i d0 = _mm_sub_epi16(_mm_unpacklo_epi8(o, z),
                                   _mm_unpacklo_epi8(sv, z));
        __m128i d1 = _mm_sub_epi16(_mm_unpackhi_epi8(o, z),
                                   _mm_unpackhi_epi8(sv, z));
        _mm_storeu_si128((__m128i *)(dst + y * stride), o);
        acc = _mm_add_epi32(acc, _mm_madd_epi16(d0, d0));
        acc = _mm_add_epi32(acc, _mm_madd_epi16(d1, d1));
    }
    acc = _mm_add_epi32(acc, _mm_srli_si128(acc, 8));
    acc = _mm_add_epi32(acc, _mm_srli_si128(acc, 4));
    return (uint64_t)(uint32_t)_mm_cvtsi128_si32(acc);
#else
    return add_res16_ssd_c(dst, stride, pred, rec, src, sstride);
#endif
}

uint64_t aeg_add_res16_ssd_c(uint8_t *dst, int stride, const uint8_t *pred,
                             const int16_t *rec, const uint8_t *src,
                             int sstride)
{
    return add_res16_ssd_c(dst, stride, pred, rec, src, sstride);
}

/* ------------------------------------------------------------------ */
/* 8x8 transform (High profile)                                        */
/* ------------------------------------------------------------------ */
/* Two passes of the eight-point integer butterfly, rows then columns on the
 * way in, columns then rows on the way out.  The forward pair has a DC gain of
 * 64 and the inverse one brings it back with a >>6 in its second pass.
 *
 * The forward and inverse butterflies are the ones the H.264 8x8 transform is
 * built from, and the inverse is written to the *decoder's* arithmetic: the
 * dequantiser applies the reference scaling with its rounding and its int16
 * clamp, the first pass truncates into int16 (8.5.10.2 keeps 16-bit
 * intermediates) and the second one shifts by six.  That is not an
 * optimisation of the arithmetic, it *is* the arithmetic: it is what makes the
 * encoder's reconstruction bit-identical to a decoder's.
 *
 * Scaling.  The dequantiser is the reference decoder's: d = level * V8(qp%6,
 * class) * 16 * 2^(qp/6), clipped to int16 and rounded by >>6.  The quantiser
 * multiplier of each coefficient position is the one that inverts exactly that
 * path (aeg_quant8_mf, see tables.c), applied with a 16 + qp/6 shift, so
 * quantise-then-dequantise reproduces the coefficient at any QP.
 *
 * The 8x8 transform is not in the batched interface yet (tq_batch and
 * iq_idct_batch take 4x4 blocks), so it always runs on the CPU. */
#if defined(__SSE2__)
/* 8x8 transpose of int16 rows, in place. */
static inline void tr8_epi16(__m128i r[8])
{
    __m128i a0 = _mm_unpacklo_epi16(r[0], r[1]), a1 = _mm_unpackhi_epi16(r[0], r[1]);
    __m128i a2 = _mm_unpacklo_epi16(r[2], r[3]), a3 = _mm_unpackhi_epi16(r[2], r[3]);
    __m128i a4 = _mm_unpacklo_epi16(r[4], r[5]), a5 = _mm_unpackhi_epi16(r[4], r[5]);
    __m128i a6 = _mm_unpacklo_epi16(r[6], r[7]), a7 = _mm_unpackhi_epi16(r[6], r[7]);
    __m128i b0 = _mm_unpacklo_epi32(a0, a2), b1 = _mm_unpackhi_epi32(a0, a2);
    __m128i b2 = _mm_unpacklo_epi32(a1, a3), b3 = _mm_unpackhi_epi32(a1, a3);
    __m128i b4 = _mm_unpacklo_epi32(a4, a6), b5 = _mm_unpackhi_epi32(a4, a6);
    __m128i b6 = _mm_unpacklo_epi32(a5, a7), b7 = _mm_unpackhi_epi32(a5, a7);
    r[0] = _mm_unpacklo_epi64(b0, b4); r[1] = _mm_unpackhi_epi64(b0, b4);
    r[2] = _mm_unpacklo_epi64(b1, b5); r[3] = _mm_unpackhi_epi64(b1, b5);
    r[4] = _mm_unpacklo_epi64(b2, b6); r[5] = _mm_unpackhi_epi64(b2, b6);
    r[6] = _mm_unpacklo_epi64(b3, b7); r[7] = _mm_unpackhi_epi64(b3, b7);
}

#endif

#define DCT8_QUANT_SHIFT 16

static inline void fdct8_1d(const int *s, int *d)
{
    const int s07 = s[0] + s[7], s16 = s[1] + s[6];
    const int s25 = s[2] + s[5], s34 = s[3] + s[4];
    const int d07 = s[0] - s[7], d16 = s[1] - s[6];
    const int d25 = s[2] - s[5], d34 = s[3] - s[4];
    const int a0 = s07 + s34, a1 = s16 + s25;
    const int a2 = s07 - s34, a3 = s16 - s25;
    const int a4 = d16 + d25 + (d07 + (d07 >> 1));
    const int a5 = d07 - d34 - (d25 + (d25 >> 1));
    const int a6 = d07 + d34 - (d16 + (d16 >> 1));
    const int a7 = d16 - d25 + (d34 + (d34 >> 1));
    d[0] = a0 + a1;
    d[1] = a4 + (a7 >> 2);
    d[2] = a2 + (a3 >> 1);
    d[3] = a5 + (a6 >> 2);
    d[4] = a0 - a1;
    d[5] = a6 - (a5 >> 2);
    d[6] = (a2 >> 1) - a3;
    d[7] = (a4 >> 2) - a7;
}

static inline void idct8_1d(const int *s, int *d)
{
    const int a0 = s[0] + s[4], a2 = s[0] - s[4];
    const int a4 = (s[2] >> 1) - s[6], a6 = (s[6] >> 1) + s[2];
    const int b0 = a0 + a6, b2 = a2 + a4, b4 = a2 - a4, b6 = a0 - a6;
    const int a1 = -s[3] + s[5] - s[7] - (s[7] >> 1);
    const int a3 = s[1] + s[7] - s[3] - (s[3] >> 1);
    const int a5 = -s[1] + s[7] + s[5] + (s[5] >> 1);
    const int a7 = s[3] + s[5] + s[1] + (s[1] >> 1);
    const int b1 = (a7 >> 2) + a1, b3 = a3 + (a5 >> 2);
    const int b5 = (a3 >> 2) - a5, b7 = a7 - (a1 >> 2);
    d[0] = b0 + b7;
    d[1] = b2 + b5;
    d[2] = b4 + b3;
    d[3] = b6 + b1;
    d[4] = b6 - b1;
    d[5] = b4 - b3;
    d[6] = b2 - b5;
    d[7] = b0 - b7;
}

#if defined(__SSE2__)
/* fdct8_1d() across eight vectors of four int32 lanes. */
static inline void fdct8_v(const __m128i *s, __m128i *d)
{
#define ADD _mm_add_epi32
#define SUB _mm_sub_epi32
#define SR(x, n) _mm_srai_epi32(x, n)
    const __m128i s07 = ADD(s[0], s[7]), s16 = ADD(s[1], s[6]);
    const __m128i s25 = ADD(s[2], s[5]), s34 = ADD(s[3], s[4]);
    const __m128i d07 = SUB(s[0], s[7]), d16 = SUB(s[1], s[6]);
    const __m128i d25 = SUB(s[2], s[5]), d34 = SUB(s[3], s[4]);
    const __m128i a0 = ADD(s07, s34), a1 = ADD(s16, s25);
    const __m128i a2 = SUB(s07, s34), a3 = SUB(s16, s25);
    const __m128i a4 = ADD(ADD(d16, d25), ADD(d07, SR(d07, 1)));
    const __m128i a5 = SUB(SUB(d07, d34), ADD(d25, SR(d25, 1)));
    const __m128i a6 = SUB(ADD(d07, d34), ADD(d16, SR(d16, 1)));
    const __m128i a7 = ADD(SUB(d16, d25), ADD(d34, SR(d34, 1)));
    d[0] = ADD(a0, a1);
    d[1] = ADD(a4, SR(a7, 2));
    d[2] = ADD(a2, SR(a3, 1));
    d[3] = ADD(a5, SR(a6, 2));
    d[4] = SUB(a0, a1);
    d[5] = SUB(a6, SR(a5, 2));
    d[6] = SUB(SR(a2, 1), a3);
    d[7] = SUB(SR(a4, 2), a7);
#undef ADD
#undef SUB
#undef SR
}

static inline void fdct8_pass(__m128i r[8])
{
    __m128i lo[8], hi[8], dl[8], dh[8];
    int k;
    for (k = 0; k < 8; k++) {
        lo[k] = _mm_srai_epi32(_mm_unpacklo_epi16(r[k], r[k]), 16);
        hi[k] = _mm_srai_epi32(_mm_unpackhi_epi16(r[k], r[k]), 16);
    }
    fdct8_v(lo, dl);
    fdct8_v(hi, dh);
    for (k = 0; k < 8; k++)
        r[k] = _mm_packs_epi32(dl[k], dh[k]);
}
#endif

/* The SIMD path keeps the row pass in sixteen bits, which is exact for any
 * residual of eight-bit samples (|res| <= 255: the largest row output is
 * 8 * 255); anything wider takes the C path. */
void aeg_transform_8x8(const int16_t *res, int stride, int16_t *coef)
{
#if defined(__SSE2__)
    {
        __m128i r[8], mx = _mm_setzero_si128(), mn = _mm_setzero_si128();
        int k;
        for (k = 0; k < 8; k++) {
            r[k] = _mm_loadu_si128((const __m128i *)(res + k * stride));
            mx = _mm_max_epi16(mx, r[k]);
            mn = _mm_min_epi16(mn, r[k]);
        }
        mx = _mm_max_epi16(mx, _mm_sub_epi16(_mm_setzero_si128(), mn));
        mx = _mm_cmpgt_epi16(mx, _mm_set1_epi16(255));
        if (!_mm_movemask_epi8(mx)) {
            tr8_epi16(r);
            fdct8_pass(r);
            tr8_epi16(r);
            fdct8_pass(r);
            for (k = 0; k < 8; k++)
                _mm_storeu_si128((__m128i *)(coef + 8 * k), r[k]);
            return;
        }
    }
#endif
    aeg_transform_8x8_c(res, stride, coef);
}

void aeg_transform_8x8_c(const int16_t *res, int stride, int16_t *coef)
{
    int tmp[64], out[64], s[8], d[8];
    int x, y;

    for (y = 0; y < 8; y++)
        for (x = 0; x < 8; x++)
            tmp[y * 8 + x] = res[y * stride + x];
    for (y = 0; y < 8; y++) {                   /* rows */
        fdct8_1d(tmp + y * 8, out + y * 8);
    }
    for (x = 0; x < 8; x++) {                   /* columns */
        for (y = 0; y < 8; y++)
            s[y] = out[y * 8 + x];
        fdct8_1d(s, d);
        for (y = 0; y < 8; y++) {
            int v = d[y];
            /* The coefficient buffer is 16 bits wide, as in the reference
             * encoder; an eight-bit residual cannot reach the clamp (the
             * largest single coefficient of a +-255 residual is about 7700). */
            coef[y * 8 + x] = (int16_t)(v < -32768 ? -32768 :
                                        (v > 32767 ? 32767 : v));
        }
    }
}

#if defined(__SSE2__)
/* The level clamp below for every QP and position. */
static int32_t q8_lim[52][64];
static void q8_init(void)
{
    int q, i;
    for (q = 0; q < 52; q++)
        for (i = 0; i < 64; i++)
            q8_lim[q][i] = (int32_t)(((int64_t)32767 << 6) /
                ((int64_t)aeg_dequant8_ls[q % 6][aeg_cls8[i >> 3][i & 7]] << (q / 6)));
}
static pthread_once_t q8_once = PTHREAD_ONCE_INIT;
#endif

void aeg_quant_8x8(const int16_t *coef, int16_t *level, int qp, int intra)
{
#if defined(__SSE2__)
    /* The scalar loop eight at a time: |coef| <= 32768 and mf < 65536, so the
     * product and the rounding offset (< 2^24) fit an unsigned 32-bit lane;
     * the level is clamped as below, then wrapped to 16 bits and signed the
     * way the scalar int16 store does. */
    if (qp >= 0 && qp < 52) {
        const int qbits = DCT8_QUANT_SHIFT + qp / 6, m = qp % 6;
        const int f = aeg_quant_bias(qbits, intra ? 3 : 6);
        const __m128i fv = _mm_set1_epi32(f), qb = _mm_cvtsi32_si128(qbits);
        int k;
        pthread_once(&q8_once, q8_init);
        for (k = 0; k < 64; k += 8) {
            __m128i c = _mm_loadu_si128((const __m128i *)(coef + k));
            __m128i mf = _mm_loadu_si128((const __m128i *)(aeg_quant8_mf[m] + k));
            __m128i sg = _mm_srai_epi16(c, 15);
            __m128i ab = _mm_sub_epi16(_mm_xor_si128(c, sg), sg);
            __m128i lo = _mm_mullo_epi16(ab, mf), hi = _mm_mulhi_epu16(ab, mf);
            __m128i p0 = _mm_unpacklo_epi16(lo, hi), p1 = _mm_unpackhi_epi16(lo, hi);
            __m128i l0 = _mm_loadu_si128((const __m128i *)(q8_lim[qp] + k));
            __m128i l1 = _mm_loadu_si128((const __m128i *)(q8_lim[qp] + k + 4));
            __m128i m0, m1;
            p0 = _mm_srl_epi32(_mm_add_epi32(p0, fv), qb);
            p1 = _mm_srl_epi32(_mm_add_epi32(p1, fv), qb);
            m0 = _mm_cmpgt_epi32(p0, l0);
            m1 = _mm_cmpgt_epi32(p1, l1);
            p0 = _mm_or_si128(_mm_andnot_si128(m0, p0), _mm_and_si128(m0, l0));
            p1 = _mm_or_si128(_mm_andnot_si128(m1, p1), _mm_and_si128(m1, l1));
            p0 = _mm_srai_epi32(_mm_slli_epi32(p0, 16), 16);
            p1 = _mm_srai_epi32(_mm_slli_epi32(p1, 16), 16);
            c = _mm_packs_epi32(p0, p1);
            _mm_storeu_si128((__m128i *)(level + k),
                             _mm_sub_epi16(_mm_xor_si128(c, sg), sg));
        }
        return;
    }
#endif
    aeg_quant_8x8_c(coef, level, qp, intra);
}

/* The scalar reference, for the differential test. */
void aeg_quant_8x8_c(const int16_t *coef, int16_t *level, int qp, int intra)
{
    const int qbits = DCT8_QUANT_SHIFT + qp / 6;
    const int f = aeg_quant_bias(qbits, intra ? 3 : 6);
    const int m = qp % 6, sh = qp / 6;
    int x, y;

    for (y = 0; y < 8; y++)
        for (x = 0; x < 8; x++) {
            const int c = coef[y * 8 + x];
            const int abs = c < 0 ? -c : c;
            int lv = (int)(((int64_t)abs * aeg_quant8_mf[m][y * 8 + x] + f)
                           >> qbits);
            /* Keep the dequantised level inside the decoder's int16 range:
             * |level * V8 * 16 * 2^(qp/6) >> 6| <= 32767. */
            if (lv) {
                const int64_t lim = ((int64_t)32767 << 6) /
                                    ((int64_t)aeg_dequant8_ls[m][aeg_cls8[y][x]]
                                     << sh);
                if (lv > lim)
                    lv = (int)lim;
            }
            level[y * 8 + x] = (int16_t)(c < 0 ? -lv : lv);
        }
}

#if defined(__SSE2__)
/* level * (V8 << qp/6) for every position and QP: at most 58 << 8, so it fits
 * in int16 and the product in int32 (16 * x + 32 >> 6 == x + 2 >> 2). */
static int16_t dq8_mul[52][64];
static void dq8_init(void)
{
    int q, i;
    for (q = 0; q < 52; q++)
        for (i = 0; i < 64; i++)
            dq8_mul[q][i] = (int16_t)(aeg_dequant8_ls[q % 6][aeg_cls8[i >> 3][i & 7]] << (q / 6));
}
static pthread_once_t dq8_once = PTHREAD_ONCE_INIT;
#endif

void aeg_dequant_8x8(const int16_t *level, int16_t *coef, int qp)
{
#if defined(__SSE2__)
    if (qp >= 0 && qp < 52) {
        const __m128i two = _mm_set1_epi32(2);
        int k;
        pthread_once(&dq8_once, dq8_init);
        for (k = 0; k < 64; k += 8) {
            __m128i l = _mm_loadu_si128((const __m128i *)(level + k));
            __m128i m = _mm_loadu_si128((const __m128i *)(dq8_mul[qp] + k));
            __m128i lo16 = _mm_mullo_epi16(l, m), hi16 = _mm_mulhi_epi16(l, m);
            __m128i a = _mm_unpacklo_epi16(lo16, hi16), b = _mm_unpackhi_epi16(lo16, hi16);
            a = _mm_srai_epi32(_mm_add_epi32(a, two), 2);
            b = _mm_srai_epi32(_mm_add_epi32(b, two), 2);
            _mm_storeu_si128((__m128i *)(coef + k), _mm_packs_epi32(a, b));
        }
        return;
    }
#endif
    aeg_dequant_8x8_c(level, coef, qp);
}

void aeg_dequant_8x8_c(const int16_t *level, int16_t *coef, int qp)
{
    const int m = qp % 6, sh = qp / 6;
    int x, y;
    for (y = 0; y < 8; y++)
        for (x = 0; x < 8; x++) {
            const int64_t v = ((int64_t)level[y * 8 + x] *
                               (aeg_dequant8_ls[m][aeg_cls8[y][x]] << sh) * 16
                               + 32) >> 6;
            coef[y * 8 + x] = (int16_t)(v < -32768 ? -32768 :
                                        (v > 32767 ? 32767 : v));
        }
}

#if defined(__SSE2__)
/* idct8_1d() across eight vectors of four int32 lanes. */
static inline void idct8_v(const __m128i *s, __m128i *d)
{
#define ADD _mm_add_epi32
#define SUB _mm_sub_epi32
#define SR(x, n) _mm_srai_epi32(x, n)
    const __m128i a0 = ADD(s[0], s[4]), a2 = SUB(s[0], s[4]);
    const __m128i a4 = SUB(SR(s[2], 1), s[6]), a6 = ADD(SR(s[6], 1), s[2]);
    const __m128i b0 = ADD(a0, a6), b2 = ADD(a2, a4), b4 = SUB(a2, a4), b6 = SUB(a0, a6);
    const __m128i a1 = SUB(SUB(SUB(s[5], s[3]), s[7]), SR(s[7], 1));
    const __m128i a3 = SUB(SUB(ADD(s[1], s[7]), s[3]), SR(s[3], 1));
    const __m128i a5 = ADD(ADD(SUB(s[7], s[1]), s[5]), SR(s[5], 1));
    const __m128i a7 = ADD(ADD(ADD(s[3], s[5]), s[1]), SR(s[1], 1));
    const __m128i b1 = ADD(SR(a7, 2), a1), b3 = ADD(a3, SR(a5, 2));
    const __m128i b5 = SUB(SR(a3, 2), a5), b7 = SUB(a7, SR(a1, 2));
    d[0] = ADD(b0, b7); d[1] = ADD(b2, b5); d[2] = ADD(b4, b3); d[3] = ADD(b6, b1);
    d[4] = SUB(b6, b1); d[5] = SUB(b4, b3); d[6] = SUB(b2, b5); d[7] = SUB(b0, b7);
#undef ADD
#undef SUB
#undef SR
}

/* One 1-D pass down the columns of eight int16 rows: lanes 0-3 and 4-7 in
 * int32, each result truncated to sixteen bits (wrapped, as the scalar
 * path's int16 store does) after `shift`. */
static inline void idct8_pass(__m128i r[8], int shift)
{
    __m128i lo[8], hi[8], dl[8], dh[8];
    int k;
    for (k = 0; k < 8; k++) {
        lo[k] = _mm_srai_epi32(_mm_unpacklo_epi16(r[k], r[k]), 16);
        hi[k] = _mm_srai_epi32(_mm_unpackhi_epi16(r[k], r[k]), 16);
    }
    idct8_v(lo, dl);
    idct8_v(hi, dh);
    for (k = 0; k < 8; k++) {
        __m128i l = _mm_srai_epi32(dl[k], shift), h = _mm_srai_epi32(dh[k], shift);
        l = _mm_srai_epi32(_mm_slli_epi32(l, 16), 16);
        h = _mm_srai_epi32(_mm_slli_epi32(h, 16), 16);
        r[k] = _mm_packs_epi32(l, h);
    }
}
#endif

#if defined(AEG_X86_AVX2)
/* idct8_pass() with all eight lanes of a row in one int32 register. */
__attribute__((target("avx2")))
static inline void idct8_pass_avx2(__m128i r[8], int shift)
{
#define ADD _mm256_add_epi32
#define SUB _mm256_sub_epi32
#define SR(x, n) _mm256_srai_epi32(x, n)
    __m256i s[8], d[8];
    int k;
    for (k = 0; k < 8; k++)
        s[k] = _mm256_cvtepi16_epi32(r[k]);
    {
        const __m256i a0 = ADD(s[0], s[4]), a2 = SUB(s[0], s[4]);
        const __m256i a4 = SUB(SR(s[2], 1), s[6]), a6 = ADD(SR(s[6], 1), s[2]);
        const __m256i b0 = ADD(a0, a6), b2 = ADD(a2, a4), b4 = SUB(a2, a4), b6 = SUB(a0, a6);
        const __m256i a1 = SUB(SUB(SUB(s[5], s[3]), s[7]), SR(s[7], 1));
        const __m256i a3 = SUB(SUB(ADD(s[1], s[7]), s[3]), SR(s[3], 1));
        const __m256i a5 = ADD(ADD(SUB(s[7], s[1]), s[5]), SR(s[5], 1));
        const __m256i a7 = ADD(ADD(ADD(s[3], s[5]), s[1]), SR(s[1], 1));
        const __m256i b1 = ADD(SR(a7, 2), a1), b3 = ADD(a3, SR(a5, 2));
        const __m256i b5 = SUB(SR(a3, 2), a5), b7 = SUB(a7, SR(a1, 2));
        d[0] = ADD(b0, b7); d[1] = ADD(b2, b5); d[2] = ADD(b4, b3); d[3] = ADD(b6, b1);
        d[4] = SUB(b6, b1); d[5] = SUB(b4, b3); d[6] = SUB(b2, b5); d[7] = SUB(b0, b7);
    }
#undef ADD
#undef SUB
#undef SR
    for (k = 0; k < 8; k++) {
        __m256i v = _mm256_srai_epi32(d[k], shift);
        v = _mm256_srai_epi32(_mm256_slli_epi32(v, 16), 16);
        r[k] = _mm_packs_epi32(_mm256_castsi256_si128(v),
                               _mm256_extracti128_si256(v, 1));
    }
}

__attribute__((target("avx2")))
static void idct_8x8_residual_avx2(const int16_t *level, int qp, int16_t *res)
{
    int16_t d[64];
    __m128i r[8];
    int k;
    aeg_dequant_8x8(level, d, qp);
    d[0] = (int16_t)(d[0] + 32);
    for (k = 0; k < 8; k++)
        r[k] = _mm_loadu_si128((const __m128i *)(d + 8 * k));
    tr8_epi16(r);
    idct8_pass_avx2(r, 0);
    tr8_epi16(r);
    idct8_pass_avx2(r, 6);
    for (k = 0; k < 8; k++)
        _mm_storeu_si128((__m128i *)(res + 8 * k), r[k]);
}
#endif

void aeg_idct_8x8_residual(const int16_t *level, int qp, int16_t *res)
{
#if defined(AEG_X86_AVX2)
    if (cpu_avx2()) {
        idct_8x8_residual_avx2(level, qp, res);
        return;
    }
#endif
#if defined(__SSE2__)
    {
        int16_t d[64];
        __m128i r[8];
        int k;
        aeg_dequant_8x8(level, d, qp);
        d[0] = (int16_t)(d[0] + 32);
        for (k = 0; k < 8; k++)
            r[k] = _mm_loadu_si128((const __m128i *)(d + 8 * k));
        /* rows: transpose so that a pass down the columns runs along them */
        tr8_epi16(r);
        idct8_pass(r, 0);
        tr8_epi16(r);
        idct8_pass(r, 6);
        for (k = 0; k < 8; k++)
            _mm_storeu_si128((__m128i *)(res + 8 * k), r[k]);
        return;
    }
#endif
    aeg_idct_8x8_residual_c(level, qp, res);
}

void aeg_idct_8x8_residual_c(const int16_t *level, int qp, int16_t *res)
{
    int16_t d[64];
    int16_t t[64];
    int s[8], o[8];
    int x, y;

    aeg_dequant_8x8(level, d, qp);
    d[0] = (int16_t)(d[0] + 32);        /* the decoder's rounding offset */
    /* Rows first, then columns.  The forward transform post-multiplies by the
     * 1-D transform along the rows and pre-multiplies by it along the columns,
     * so its inverse has to run in the same order to undo both; running the
     * columns first would reconstruct the transposed residual.  The reference
     * decoder truncates to sixteen bits after the first pass and shifts only
     * at the end of the second one, which is what the two int16 arrays here
     * do. */
    for (y = 0; y < 8; y++) {                   /* rows */
        for (x = 0; x < 8; x++)
            s[x] = d[y * 8 + x];
        idct8_1d(s, o);
        for (x = 0; x < 8; x++)
            t[y * 8 + x] = (int16_t)o[x];
    }
    for (x = 0; x < 8; x++) {                   /* columns */
        for (y = 0; y < 8; y++)
            s[y] = t[y * 8 + x];
        idct8_1d(s, o);
        for (y = 0; y < 8; y++)
            res[y * 8 + x] = (int16_t)(o[y] >> 6);
    }
}

/* Incremental 8x8 reconstruction for the level search: the dequantised
 * block and its first (row) pass are kept for the current levels, so a trial
 * that moves one level redoes one row of the first pass and the column pass
 * only.  Every step is the one aeg_idct_8x8_add() takes -- same dequantiser,
 * same int16 truncation after the rows, same shift, same clip -- so the SSE
 * of a trial is exactly aeg_ssd() of what aeg_idct_8x8_add() would make. */
static int16_t dq8_one(int lev, int qp, int pos)
{
    const int64_t v = ((int64_t)(int16_t)lev *
                       (aeg_dequant8_ls[qp % 6][aeg_cls8[pos >> 3][pos & 7]] << (qp / 6)) * 16
                       + 32) >> 6;
    int16_t d = (int16_t)(v < -32768 ? -32768 : (v > 32767 ? 32767 : v));
    return pos ? d : (int16_t)(d + 32);
}

static void idct8_row(const int16_t *d, int16_t *t)
{
    int s[8], o[8], x;
    for (x = 0; x < 8; x++)
        s[x] = d[x];
    idct8_1d(s, o);
    for (x = 0; x < 8; x++)
        t[x] = (int16_t)o[x];
}

void aeg_idct8_cache_init(aeg_idct8_cache *c, const int16_t *level, int qp)
{
    int y;
    aeg_dequant_8x8(level, c->d, qp);
    c->d[0] = (int16_t)(c->d[0] + 32);
    c->qp = qp;
    for (y = 0; y < 8; y++)
        idct8_row(c->d + 8 * y, c->t + 8 * y);
}

void aeg_idct8_cache_set(aeg_idct8_cache *c, int pos, int lev)
{
    c->d[pos] = dq8_one(lev, c->qp, pos);
    idct8_row(c->d + (pos & ~7), c->t + (pos & ~7));
}

static int64_t col_ssd_c(const int16_t *t, const int16_t *trow, int ty,
                         const uint8_t *pred, int ps, const uint8_t *src, int ss)
{
    int s[8], o[8], x, y;
    int64_t sum = 0;
    for (x = 0; x < 8; x++) {
        for (y = 0; y < 8; y++)
            s[y] = y == ty ? trow[x] : t[y * 8 + x];
        idct8_1d(s, o);
        for (y = 0; y < 8; y++) {
            int v = (int)pred[y * ps + x] + (int16_t)(o[y] >> 6), dd;
            v = v < 0 ? 0 : (v > 255 ? 255 : v);
            dd = v - (int)src[y * ss + x];
            sum += dd * dd;
        }
    }
    return sum;
}

#if defined(AEG_X86_AVX2)
__attribute__((target("avx2")))
static int64_t col_ssd_avx2(const int16_t *t, const int16_t *trow, int ty,
                            const uint8_t *pred, int ps, const uint8_t *src, int ss)
{
#define ADD _mm256_add_epi32
#define SUB _mm256_sub_epi32
#define SR(x, n) _mm256_srai_epi32(x, n)
    __m256i s[8], d[8], acc = _mm256_setzero_si256();
    const __m256i zero = _mm256_setzero_si256(), mx = _mm256_set1_epi32(255);
    __m128i lo;
    int k;
    for (k = 0; k < 8; k++)
        s[k] = _mm256_cvtepi16_epi32(_mm_loadu_si128(
            (const __m128i *)(k == ty ? trow : t + 8 * k)));
    {
        const __m256i a0 = ADD(s[0], s[4]), a2 = SUB(s[0], s[4]);
        const __m256i a4 = SUB(SR(s[2], 1), s[6]), a6 = ADD(SR(s[6], 1), s[2]);
        const __m256i b0 = ADD(a0, a6), b2 = ADD(a2, a4), b4 = SUB(a2, a4), b6 = SUB(a0, a6);
        const __m256i a1 = SUB(SUB(SUB(s[5], s[3]), s[7]), SR(s[7], 1));
        const __m256i a3 = SUB(SUB(ADD(s[1], s[7]), s[3]), SR(s[3], 1));
        const __m256i a5 = ADD(ADD(SUB(s[7], s[1]), s[5]), SR(s[5], 1));
        const __m256i a7 = ADD(ADD(ADD(s[3], s[5]), s[1]), SR(s[1], 1));
        const __m256i b1 = ADD(SR(a7, 2), a1), b3 = ADD(a3, SR(a5, 2));
        const __m256i b5 = SUB(SR(a3, 2), a5), b7 = SUB(a7, SR(a1, 2));
        d[0] = ADD(b0, b7); d[1] = ADD(b2, b5); d[2] = ADD(b4, b3); d[3] = ADD(b6, b1);
        d[4] = SUB(b6, b1); d[5] = SUB(b4, b3); d[6] = SUB(b2, b5); d[7] = SUB(b0, b7);
    }
#undef ADD
#undef SUB
#undef SR
    for (k = 0; k < 8; k++) {
        __m256i r = _mm256_srai_epi32(d[k], 6);
        __m256i p = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(pred + k * ps)));
        __m256i o = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(src + k * ss)));
        r = _mm256_srai_epi32(_mm256_slli_epi32(r, 16), 16);
        r = _mm256_min_epi32(_mm256_max_epi32(_mm256_add_epi32(p, r), zero), mx);
        r = _mm256_sub_epi32(r, o);
        acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(r, r));
    }
    lo = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, _MM_SHUFFLE(1, 0, 3, 2)));
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, _MM_SHUFFLE(2, 3, 0, 1)));
    return (uint32_t)_mm_cvtsi128_si32(lo);
}
#endif

int64_t aeg_idct8_cache_ssd(const aeg_idct8_cache *c, int pos, int lev,
                            const uint8_t *pred, int ps, const uint8_t *src, int ss)
{
    int16_t drow[8], trow[8];
    const int y = pos >> 3;
    if (pos >= 0) {
        memcpy(drow, c->d + 8 * y, sizeof(drow));
        drow[pos & 7] = dq8_one(lev, c->qp, pos);
        idct8_row(drow, trow);
    }
#if defined(AEG_X86_AVX2)
    if (cpu_avx2())
        return col_ssd_avx2(c->t, trow, pos >= 0 ? y : -1, pred, ps, src, ss);
#endif
    return col_ssd_c(c->t, trow, pos >= 0 ? y : -1, pred, ps, src, ss);
}

void aeg_idct_8x8_add(uint8_t *dst, int stride, const int16_t *level, int qp,
                      const uint8_t *pred, int pred_stride)
{
    int16_t res[64];
    int x, y;
    aeg_idct_8x8_residual(level, qp, res);
#if defined(__SSE2__)
    {
        const __m128i z = _mm_setzero_si128();
        for (y = 0; y < 8; y++) {
            __m128i p = _mm_unpacklo_epi8(
                _mm_loadl_epi64((const __m128i *)(pred + y * pred_stride)), z);
            __m128i r = _mm_loadu_si128((const __m128i *)(res + 8 * y));
            _mm_storel_epi64((__m128i *)(dst + y * stride),
                             _mm_packus_epi16(_mm_adds_epi16(p, r), z));
        }
        (void)x;
        return;
    }
#endif
    for (y = 0; y < 8; y++)
        for (x = 0; x < 8; x++) {
            int p = (int)pred[y * pred_stride + x] + (int)res[y * 8 + x];
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
        va = _mm_cvtsi32_si128(ld32(a));
        vb = _mm_cvtsi32_si128(ld32(b));
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
    if (w == 16) {
        /* the motion searches' common case: two rows per step, two sums */
        __m128i acc0 = _mm_setzero_si128(), acc1 = _mm_setzero_si128();
        for (y = 0; y + 1 < h; y += 2) {
            const uint8_t *pa = a + (size_t)y * sa, *pb = b + (size_t)y * sb;
            acc0 = _mm_add_epi64(acc0, _mm_sad_epu8(
                _mm_loadu_si128((const __m128i *)pa),
                _mm_loadu_si128((const __m128i *)pb)));
            acc1 = _mm_add_epi64(acc1, _mm_sad_epu8(
                _mm_loadu_si128((const __m128i *)(pa + sa)),
                _mm_loadu_si128((const __m128i *)(pb + sb))));
        }
        if (y < h)
            sad_row(&acc0, a + (size_t)y * sa, b + (size_t)y * sb, 16);
        return sad_reduce(_mm_add_epi64(acc0, acc1));
    }
    if (w == 8) {
        /* two 8-byte rows fill one register */
        __m128i acc = _mm_setzero_si128();
        for (y = 0; y + 1 < h; y += 2) {
            const uint8_t *pa = a + (size_t)y * sa, *pb = b + (size_t)y * sb;
            __m128i va = _mm_unpacklo_epi64(
                _mm_loadl_epi64((const __m128i *)pa),
                _mm_loadl_epi64((const __m128i *)(pa + sa)));
            __m128i vb = _mm_unpacklo_epi64(
                _mm_loadl_epi64((const __m128i *)pb),
                _mm_loadl_epi64((const __m128i *)(pb + sb)));
            acc = _mm_add_epi64(acc, _mm_sad_epu8(va, vb));
        }
        if (y < h)
            sad_row(&acc, a + (size_t)y * sa, b + (size_t)y * sb, 8);
        return sad_reduce(acc);
    }
    if (w == 4) {
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

/* aeg_sad() of one block against four candidate positions, each row of
 * the block loaded once (the motion searches' patterns). */
void aeg_sad_x4(int w, int h, const uint8_t *a, int sa, const uint8_t *const b[4],
                int sb, int out[4])
{
#if defined(__SSE2__)
    if (w == 16) {
        __m128i s0 = _mm_setzero_si128(), s1 = s0, s2 = s0, s3 = s0;
        const uint8_t *b0 = b[0], *b1 = b[1], *b2 = b[2], *b3 = b[3];
        int y;
        for (y = 0; y < h; y++) {
            const __m128i va = _mm_loadu_si128((const __m128i *)(a + (size_t)y * sa));
            const size_t o = (size_t)y * sb;
            s0 = _mm_add_epi64(s0, _mm_sad_epu8(va, _mm_loadu_si128((const __m128i *)(b0 + o))));
            s1 = _mm_add_epi64(s1, _mm_sad_epu8(va, _mm_loadu_si128((const __m128i *)(b1 + o))));
            s2 = _mm_add_epi64(s2, _mm_sad_epu8(va, _mm_loadu_si128((const __m128i *)(b2 + o))));
            s3 = _mm_add_epi64(s3, _mm_sad_epu8(va, _mm_loadu_si128((const __m128i *)(b3 + o))));
        }
        out[0] = sad_reduce(s0);
        out[1] = sad_reduce(s1);
        out[2] = sad_reduce(s2);
        out[3] = sad_reduce(s3);
        return;
    }
    if (w == 8 && (h & 1) == 0) {
        __m128i s0 = _mm_setzero_si128(), s1 = s0, s2 = s0, s3 = s0;
        int y, k;
        for (y = 0; y < h; y += 2) {
            const uint8_t *pa = a + (size_t)y * sa;
            const size_t o = (size_t)y * sb;
            const __m128i va = _mm_unpacklo_epi64(
                _mm_loadl_epi64((const __m128i *)pa),
                _mm_loadl_epi64((const __m128i *)(pa + sa)));
            __m128i vb[4];
            for (k = 0; k < 4; k++)
                vb[k] = _mm_unpacklo_epi64(
                    _mm_loadl_epi64((const __m128i *)(b[k] + o)),
                    _mm_loadl_epi64((const __m128i *)(b[k] + o + sb)));
            s0 = _mm_add_epi64(s0, _mm_sad_epu8(va, vb[0]));
            s1 = _mm_add_epi64(s1, _mm_sad_epu8(va, vb[1]));
            s2 = _mm_add_epi64(s2, _mm_sad_epu8(va, vb[2]));
            s3 = _mm_add_epi64(s3, _mm_sad_epu8(va, vb[3]));
        }
        out[0] = sad_reduce(s0);
        out[1] = sad_reduce(s1);
        out[2] = sad_reduce(s2);
        out[3] = sad_reduce(s3);
        return;
    }
#endif
    out[0] = aeg_sad(w, h, a, sa, b[0], sb);
    out[1] = aeg_sad(w, h, a, sa, b[1], sb);
    out[2] = aeg_sad(w, h, a, sa, b[2], sb);
    out[3] = aeg_sad(w, h, a, sa, b[3], sb);
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
    __m128i va = _mm_cvtsi32_si128(ld32(a));
    __m128i vb = _mm_cvtsi32_si128(ld32(b));
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

/* 8x8 Walsh-Hadamard transform of `v`, in place. */
static void hadamard8(int *v)
{
    int step, i, j;
    for (step = 1; step < 8; step <<= 1)
        for (i = 0; i < 8; i += 2 * step)
            for (j = i; j < i + step; j++) {
                int a = v[j], b = v[j + step];
                v[j] = a + b;
                v[j + step] = a - b;
            }
}

#if defined(__SSE2__)
/* One row of eight differences, as int16. */
static inline __m128i diff8(const uint8_t *a, const uint8_t *b)
{
    const __m128i z = _mm_setzero_si128();
    return _mm_sub_epi16(_mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)a), z),
                         _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)b), z));
}

/* Sum of |4x4 Hadamard| of the two 4x4 blocks side by side in four rows of
 * eight differences: the vertical butterfly runs across the registers, the
 * horizontal one inside each group of four lanes (two shuffle-and-add stages
 * with alternating signs).  The coefficients come out permuted and with some
 * signs flipped, which a sum of absolute values does not see. */
static inline __m128i satd8x4_abs(__m128i r0, __m128i r1, __m128i r2, __m128i r3)
{
    const __m128i m1 = _mm_setr_epi16(0, -1, 0, -1, 0, -1, 0, -1);
    const __m128i m2 = _mm_setr_epi16(0, 0, -1, -1, 0, 0, -1, -1);
    const __m128i ones = _mm_set1_epi16(1);
    __m128i v[4], acc;
    int k;
    hadamard4x4(&r0, &r1, &r2, &r3);
    v[0] = r0; v[1] = r1; v[2] = r2; v[3] = r3;
    for (k = 0; k < 4; k++) {
        __m128i x = v[k], sw;
        sw = _mm_shufflehi_epi16(_mm_shufflelo_epi16(x, _MM_SHUFFLE(2, 3, 0, 1)),
                                 _MM_SHUFFLE(2, 3, 0, 1));
        x = _mm_add_epi16(sw, _mm_sub_epi16(_mm_xor_si128(x, m1), m1));
        sw = _mm_shufflehi_epi16(_mm_shufflelo_epi16(x, _MM_SHUFFLE(1, 0, 3, 2)),
                                 _MM_SHUFFLE(1, 0, 3, 2));
        x = _mm_add_epi16(sw, _mm_sub_epi16(_mm_xor_si128(x, m2), m2));
        v[k] = _mm_max_epi16(x, _mm_sub_epi16(_mm_setzero_si128(), x));
    }
    acc = _mm_add_epi16(_mm_add_epi16(v[0], v[1]), _mm_add_epi16(v[2], v[3]));
    return _mm_madd_epi16(acc, ones);
}

static inline int hsum_epi32(__m128i acc)
{
    acc = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, _MM_SHUFFLE(1, 0, 3, 2)));
    acc = _mm_add_epi32(acc, _mm_shuffle_epi32(acc, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(acc);
}

/* Eight-point butterfly across eight registers (three stages). */
static inline void hadamard8_v(__m128i r[8])
{
    int step, i, j;
    for (step = 1; step < 8; step <<= 1)
        for (i = 0; i < 8; i += 2 * step)
            for (j = i; j < i + step; j++) {
                __m128i a = r[j], b = r[j + step];
                r[j] = _mm_add_epi16(a, b);
                r[j + step] = _mm_sub_epi16(a, b);
            }
}
#endif

/* Sum of absolute 8x8 Hadamard coefficients of the difference, scaled like
 * aeg_satd_4x4 so that the two numbers are directly comparable: a flat
 * residual of v costs 8v per 4x4 SATD block and 16v per 8x8 block, while a
 * residual with detail at the 4x4 level costs more in the 8x8 measure.  That
 * is exactly the comparison the transform-size decision needs -- the 8x8
 * transform concentrates smooth energy into fewer coefficients, but it cannot
 * represent the high frequencies of a busy block without ringing.
 *
 * The vector path keeps the transform in int16 (|coefficient| <= 255 * 64)
 * and is therefore the scalar result exactly. */
int aeg_sa8d_8x8(const uint8_t *a, int sa, const uint8_t *b, int sb)
{
#if defined(__SSE2__)
    __m128i r[8], acc = _mm_setzero_si128();
    const __m128i ones = _mm_set1_epi16(1);
    int k;
    for (k = 0; k < 8; k++)
        r[k] = diff8(a + (size_t)k * sa, b + (size_t)k * sb);
    hadamard8_v(r);
    tr8_epi16(r);
    hadamard8_v(r);
    for (k = 0; k < 8; k++) {
        __m128i x = _mm_max_epi16(r[k], _mm_sub_epi16(_mm_setzero_si128(), r[k]));
        acc = _mm_add_epi32(acc, _mm_madd_epi16(x, ones));
    }
    return (hsum_epi32(acc) + 2) >> 2;
#else
    return aeg_sa8d_8x8_c(a, sa, b, sb);
#endif
}

int aeg_sa8d_8x8_c(const uint8_t *a, int sa, const uint8_t *b, int sb)
{
    int t[64], d[64], x, y, sum = 0;
    for (y = 0; y < 8; y++) {
        for (x = 0; x < 8; x++)
            d[x] = (int)a[y * sa + x] - (int)b[y * sb + x];
        hadamard8(d);
        for (x = 0; x < 8; x++)
            t[y * 8 + x] = d[x];
    }
    for (x = 0; x < 8; x++) {
        for (y = 0; y < 8; y++)
            d[y] = t[y * 8 + x];
        hadamard8(d);
        for (y = 0; y < 8; y++)
            t[y * 8 + x] = d[y];
    }
    for (x = 0; x < 64; x++)
        sum += abs(t[x]);
    return (sum + 2) >> 2;
}

int aeg_satd_8x8(const uint8_t *a, int sa, const uint8_t *b, int sb)
{
    return aeg_satd(8, 8, a, sa, b, sb);
}

/* The sum of a 4x4 block's absolute Hadamard coefficients is always even
 * (their plain sum is 16 times the first sample), so aeg_satd_4x4()'s
 * rounding is an exact halving and a larger block is the halved total. */
#if defined(__SSE2__) && defined(__GNUC__)
/* Half the summed |4x4 Hadamard| of the two 4x4 blocks in four rows of eight
 * differences, x264's way: the vertical butterflies across the registers, a
 * transpose of each 4x4 (within each 128-bit lane, so the AVX2 version below
 * is the same code on two lanes), the first horizontal stage, and in place of
 * the last one the identity |a + b| + |a - b| = 2 max(|a|, |b|) -- which is
 * also exactly the halving aeg_satd() applies, so the result needs no shift
 * and is the same integer as satd8x4_abs() >> 1.  36 instructions instead of
 * 56 per 8x4. */
#define SATD8X4_HALF(V, add16, sub16, ulo16, uhi16, ulo32, uhi32, ulo64, uhi64, \
                     max16, abs16, madd, ones, r0, r1, r2, r3, out)            \
    do {                                                                       \
        V s0_ = add16(r0, r1), d0_ = sub16(r0, r1);                            \
        V s1_ = add16(r2, r3), d1_ = sub16(r2, r3);                            \
        V h0_ = add16(s0_, s1_), h1_ = sub16(s0_, s1_);                        \
        V h2_ = sub16(d0_, d1_), h3_ = add16(d0_, d1_);                        \
        V t0_ = ulo16(h0_, h1_), t1_ = ulo16(h2_, h3_);                        \
        V t2_ = uhi16(h0_, h1_), t3_ = uhi16(h2_, h3_);                        \
        /* block A: u0 = columns 0|1, u1 = columns 2|3; block B: u2, u3 */      \
        V u0_ = ulo32(t0_, t1_), u1_ = uhi32(t0_, t1_);                        \
        V u2_ = ulo32(t2_, t3_), u3_ = uhi32(t2_, t3_);                        \
        V a0_ = abs16(add16(u0_, u1_)), a1_ = abs16(sub16(u0_, u1_));          \
        V b0_ = abs16(add16(u2_, u3_)), b1_ = abs16(sub16(u2_, u3_));          \
        V ma_ = max16(ulo64(a0_, a1_), uhi64(a0_, a1_));                       \
        V mb_ = max16(ulo64(b0_, b1_), uhi64(b0_, b1_));                       \
        out = madd(add16(ma_, mb_), ones);                                     \
    } while (0)

static inline __m128i abs16_sse2(__m128i x)
{
    return _mm_max_epi16(x, _mm_sub_epi16(_mm_setzero_si128(), x));
}

static int satd_w8_sse2(int w, int h, const uint8_t *a, int sa,
                        const uint8_t *b, int sb)
{
    const __m128i ones = _mm_set1_epi16(1);
    __m128i acc = _mm_setzero_si128();
    int x, y;
    for (y = 0; y < h; y += 4) {
        const uint8_t *pa = a + (size_t)y * sa, *pb = b + (size_t)y * sb;
        for (x = 0; x < w; x += 8) {
            __m128i r0 = diff8(pa + x, pb + x);
            __m128i r1 = diff8(pa + sa + x, pb + sb + x);
            __m128i r2 = diff8(pa + 2 * sa + x, pb + 2 * sb + x);
            __m128i r3 = diff8(pa + 3 * sa + x, pb + 3 * sb + x), o;
            SATD8X4_HALF(__m128i, _mm_add_epi16, _mm_sub_epi16,
                         _mm_unpacklo_epi16, _mm_unpackhi_epi16,
                         _mm_unpacklo_epi32, _mm_unpackhi_epi32,
                         _mm_unpacklo_epi64, _mm_unpackhi_epi64,
                         _mm_max_epi16, abs16_sse2, _mm_madd_epi16, ones,
                         r0, r1, r2, r3, o);
            acc = _mm_add_epi32(acc, o);
        }
    }
    return hsum_epi32(acc);
}

#if defined(AEG_X86_AVX2)
#define AEG_HAVE_AVX2_SATD 1
/* Sixteen columns at a time: rows of sixteen differences widened to int16
 * fill a 256-bit register whose two lanes are two 8x4 halves. */
__attribute__((target("avx2")))
static int satd_w16_avx2(int w, int h, const uint8_t *a, int sa,
                         const uint8_t *b, int sb)
{
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i acc = _mm256_setzero_si256();
    __m128i lo;
    int x, y;
#define LD16(p) _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(p)))
    for (y = 0; y < h; y += 4) {
        const uint8_t *pa = a + (size_t)y * sa, *pb = b + (size_t)y * sb;
        for (x = 0; x < w; x += 16) {
            __m256i r0 = _mm256_sub_epi16(LD16(pa + x), LD16(pb + x));
            __m256i r1 = _mm256_sub_epi16(LD16(pa + sa + x), LD16(pb + sb + x));
            __m256i r2 = _mm256_sub_epi16(LD16(pa + 2 * sa + x), LD16(pb + 2 * sb + x));
            __m256i r3 = _mm256_sub_epi16(LD16(pa + 3 * sa + x), LD16(pb + 3 * sb + x)), o;
            SATD8X4_HALF(__m256i, _mm256_add_epi16, _mm256_sub_epi16,
                         _mm256_unpacklo_epi16, _mm256_unpackhi_epi16,
                         _mm256_unpacklo_epi32, _mm256_unpackhi_epi32,
                         _mm256_unpacklo_epi64, _mm256_unpackhi_epi64,
                         _mm256_max_epi16, _mm256_abs_epi16, _mm256_madd_epi16,
                         ones, r0, r1, r2, r3, o);
            acc = _mm256_add_epi32(acc, o);
        }
    }
#undef LD16
    lo = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, _MM_SHUFFLE(1, 0, 3, 2)));
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(lo);
}

/* Eight columns, eight rows at a time: rows y..y+3 in the low lane and
 * y+4..y+7 in the high lane. */
__attribute__((target("avx2")))
static int satd_w8_avx2(int h, const uint8_t *a, int sa, const uint8_t *b, int sb)
{
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i acc = _mm256_setzero_si256();
    __m128i lo;
    int y;
#define LD8X2(p, s) _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(                 \
        _mm_loadl_epi64((const __m128i *)(p)),                               \
        _mm_loadl_epi64((const __m128i *)((p) + 4 * (size_t)(s)))))
    for (y = 0; y < h; y += 8) {
        const uint8_t *pa = a + (size_t)y * sa, *pb = b + (size_t)y * sb;
        __m256i r0 = _mm256_sub_epi16(LD8X2(pa, sa), LD8X2(pb, sb));
        __m256i r1 = _mm256_sub_epi16(LD8X2(pa + sa, sa), LD8X2(pb + sb, sb));
        __m256i r2 = _mm256_sub_epi16(LD8X2(pa + 2 * sa, sa), LD8X2(pb + 2 * sb, sb));
        __m256i r3 = _mm256_sub_epi16(LD8X2(pa + 3 * sa, sa), LD8X2(pb + 3 * sb, sb)), o;
        SATD8X4_HALF(__m256i, _mm256_add_epi16, _mm256_sub_epi16,
                     _mm256_unpacklo_epi16, _mm256_unpackhi_epi16,
                     _mm256_unpacklo_epi32, _mm256_unpackhi_epi32,
                     _mm256_unpacklo_epi64, _mm256_unpackhi_epi64,
                     _mm256_max_epi16, _mm256_abs_epi16, _mm256_madd_epi16,
                     ones, r0, r1, r2, r3, o);
        acc = _mm256_add_epi32(acc, o);
    }
#undef LD8X2
    lo = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, _MM_SHUFFLE(1, 0, 3, 2)));
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(lo);
}

#endif
#endif

#if defined(AEG_HAVE_AVX2_SATD)
/* satd_w16_avx2()/satd_w8_avx2() of the rounded average of two blocks
 * (both with stride sr) against `a`: the average is pavgb, exactly. */
__attribute__((target("avx2")))
static int satd_avg_avx2(int w, int h, const uint8_t *a, int sa,
                         const uint8_t *r1, const uint8_t *r2, int sr)
{
    const __m256i ones = _mm256_set1_epi16(1);
    __m256i acc = _mm256_setzero_si256();
    __m128i lo;
    int x, y;
    if (w == 8) {
#define LDA8(p, s) _mm256_cvtepu8_epi16(_mm_unpacklo_epi64(                  \
        _mm_loadl_epi64((const __m128i *)(p)),                               \
        _mm_loadl_epi64((const __m128i *)((p) + 4 * (size_t)(s)))))
#define LDB8(o) _mm256_cvtepu8_epi16(_mm_avg_epu8(                           \
        _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *)(r1 + (o))),     \
                           _mm_loadl_epi64((const __m128i *)(r1 + (o) + 4 * (size_t)sr))), \
        _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i *)(r2 + (o))),     \
                           _mm_loadl_epi64((const __m128i *)(r2 + (o) + 4 * (size_t)sr)))))
        for (y = 0; y < h; y += 8) {
            const uint8_t *pa = a + (size_t)y * sa;
            const size_t o = (size_t)y * sr;
            __m256i q0 = _mm256_sub_epi16(LDA8(pa, sa), LDB8(o));
            __m256i q1 = _mm256_sub_epi16(LDA8(pa + sa, sa), LDB8(o + sr));
            __m256i q2 = _mm256_sub_epi16(LDA8(pa + 2 * sa, sa), LDB8(o + 2 * sr));
            __m256i q3 = _mm256_sub_epi16(LDA8(pa + 3 * sa, sa), LDB8(o + 3 * sr)), v;
            SATD8X4_HALF(__m256i, _mm256_add_epi16, _mm256_sub_epi16,
                         _mm256_unpacklo_epi16, _mm256_unpackhi_epi16,
                         _mm256_unpacklo_epi32, _mm256_unpackhi_epi32,
                         _mm256_unpacklo_epi64, _mm256_unpackhi_epi64,
                         _mm256_max_epi16, _mm256_abs_epi16, _mm256_madd_epi16,
                         ones, q0, q1, q2, q3, v);
            acc = _mm256_add_epi32(acc, v);
        }
#undef LDA8
#undef LDB8
    } else {
#define LDA16(p) _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(p)))
#define LDB16(o) _mm256_cvtepu8_epi16(_mm_avg_epu8(                          \
        _mm_loadu_si128((const __m128i *)(r1 + (o))),                        \
        _mm_loadu_si128((const __m128i *)(r2 + (o)))))
        for (y = 0; y < h; y += 4) {
            const uint8_t *pa = a + (size_t)y * sa;
            const size_t o = (size_t)y * sr;
            for (x = 0; x < w; x += 16) {
                __m256i q0 = _mm256_sub_epi16(LDA16(pa + x), LDB16(o + x));
                __m256i q1 = _mm256_sub_epi16(LDA16(pa + sa + x), LDB16(o + sr + x));
                __m256i q2 = _mm256_sub_epi16(LDA16(pa + 2 * sa + x), LDB16(o + 2 * sr + x));
                __m256i q3 = _mm256_sub_epi16(LDA16(pa + 3 * sa + x), LDB16(o + 3 * sr + x)), v;
                SATD8X4_HALF(__m256i, _mm256_add_epi16, _mm256_sub_epi16,
                             _mm256_unpacklo_epi16, _mm256_unpackhi_epi16,
                             _mm256_unpacklo_epi32, _mm256_unpackhi_epi32,
                             _mm256_unpacklo_epi64, _mm256_unpackhi_epi64,
                             _mm256_max_epi16, _mm256_abs_epi16, _mm256_madd_epi16,
                             ones, q0, q1, q2, q3, v);
                acc = _mm256_add_epi32(acc, v);
            }
        }
#undef LDA16
#undef LDB16
    }
    lo = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, _MM_SHUFFLE(1, 0, 3, 2)));
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(lo);
}
#endif

/* aeg_satd(w, h, a, sa, avg, w) where avg is the rounded average of the two
 * blocks r1 and r2 (stride sr) -- a quarter-pel prediction measured without
 * being stored first. */
int aeg_satd_avg(int w, int h, const uint8_t *a, int sa, const uint8_t *r1,
                 const uint8_t *r2, int sr)
{
    uint8_t buf[16 * 16];
    int x, y;
#if defined(AEG_HAVE_AVX2_SATD)
    if ((((w & 15) == 0 && (h & 3) == 0) || (w == 8 && (h & 7) == 0)) && cpu_avx2())
        return satd_avg_avx2(w, h, a, sa, r1, r2, sr);
#endif
    if (w > 16 || h > 16) {
        int s = 0;
        for (y = 0; y < h; y += 16)
            for (x = 0; x < w; x += 16)
                s += aeg_satd_avg(w - x < 16 ? w - x : 16, h - y < 16 ? h - y : 16,
                                  a + (size_t)y * sa + x, sa,
                                  r1 + (size_t)y * sr + x, r2 + (size_t)y * sr + x, sr);
        return s;
    }
    for (y = 0; y < h; y++)
        for (x = 0; x < w; x++)
            buf[y * w + x] = (uint8_t)((r1[(size_t)y * sr + x] + r2[(size_t)y * sr + x] + 1) >> 1);
    return aeg_satd(w, h, buf, w, a, sa);
}

int aeg_satd(int w, int h, const uint8_t *a, int sa, const uint8_t *b, int sb)
{
#if defined(__SSE2__) && defined(__GNUC__)
    if ((h & 3) == 0) {
#if defined(AEG_HAVE_AVX2_SATD)
        if ((w & 15) == 0 && cpu_avx2())
            return satd_w16_avx2(w, h, a, sa, b, sb);
        if (w == 8 && (h & 7) == 0 && cpu_avx2())
            return satd_w8_avx2(h, a, sa, b, sb);
#endif
        if ((w & 7) == 0)
            return satd_w8_sse2(w, h, a, sa, b, sb);
    }
#endif
    return aeg_satd_c(w, h, a, sa, b, sb);
}

/* The previous SSE2 form, kept as a second reference for the test. */
int aeg_satd_sse2_ref(int w, int h, const uint8_t *a, int sa, const uint8_t *b, int sb)
{
#if defined(__SSE2__)
    if ((w & 7) == 0 && (h & 3) == 0) {
        __m128i acc = _mm_setzero_si128();
        int x, y;
        for (y = 0; y < h; y += 4) {
            const uint8_t *pa = a + (size_t)y * sa, *pb = b + (size_t)y * sb;
            for (x = 0; x < w; x += 8)
                acc = _mm_add_epi32(acc, satd8x4_abs(
                    diff8(pa + x, pb + x), diff8(pa + sa + x, pb + sb + x),
                    diff8(pa + 2 * sa + x, pb + 2 * sb + x),
                    diff8(pa + 3 * sa + x, pb + 3 * sb + x)));
        }
        return hsum_epi32(acc) >> 1;
    }
#endif
    return aeg_satd_c(w, h, a, sa, b, sb);
}

int aeg_satd_c(int w, int h, const uint8_t *a, int sa, const uint8_t *b, int sb)
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
