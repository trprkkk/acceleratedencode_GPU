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
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#if AEG_SIMD
#include <emmintrin.h>
#endif
#if AEG_SSE2
/* 32-bit loads and stores at any alignment (one movd; a cast to int * is
 * undefined behaviour on an unaligned pixel address); only the 8-bit SIMD
 * paths use them */
static inline int ld32(const void *p) { int v; memcpy(&v, p, 4); return v; }
static inline void st32(void *p, int v) { memcpy(p, &v, 4); }
#endif



/* ------------------------------------------------------------------ */
/* Half-pel interpolation                                              */
/* ------------------------------------------------------------------ */
/* 6-tap horizontal half-pel at p[0] (needs p[-2..3]) */
static inline int hpel_h(const pixel *p)
{
    return (p[-2] + p[3] - 5 * (p[-1] + p[2]) + 20 * (p[0] + p[1]) + 16) >> 5;
}
static inline int hpel_v(const pixel *p, int s)
{
    return ((int)p[-2 * s] + p[3 * s] - 5 * (p[-s] + p[2 * s]) +
            20 * (p[0] + p[s]) + 16) >> 5;
}
/* Horizontal 6-tap *without* the shift and clip: the intermediate the
 * two-dimensional half-pel position is built from (H.264 8.4.2.2.1).  The
 * reference decoder keeps these at 16-bit precision and shifts once at the
 * end, so rounding the horizontal stage first would give different samples. */
static inline int hsum_h(const pixel *p)
{
    return (p[0] + p[1]) * 20 - (p[-1] + p[2]) * 5 + (p[-2] + p[3]);
}

/* both axes: vertical 6-tap over the unshifted horizontal intermediates, with
 * both gains (32 * 32) removed at once */
static inline int hpel_hv(const pixel *p, int s)
{
    return (hsum_h(p - 2 * s) - 5 * hsum_h(p - s) + 20 * hsum_h(p) +
            20 * hsum_h(p + s) - 5 * hsum_h(p + 2 * s) + hsum_h(p + 3 * s) +
            512) >> 10;
}

/* Which of the 4 interpolated planes to combine for each 1/4-pel position
 * (same decomposition the spec's averaging rules describe). */
static const uint8_t hpel_ref0[16] = { 0, 1, 1, 1, 0, 1, 1, 1, 2, 3, 3, 3, 0, 1, 1, 1 };
static const uint8_t hpel_ref1[16] = { 0, 0, 1, 0, 2, 2, 3, 2, 2, 2, 3, 2, 2, 2, 3, 2 };

static inline int plane_get(int plane, const pixel *p, int s)
{
    switch (plane) {
    case 0:  return p[0];
    case 1:  return aeg_clip_pixel(hpel_h(p));
    case 2:  return aeg_clip_pixel(hpel_v(p, s));
    default: return aeg_clip_pixel(hpel_hv(p, s));
    }
}

void aeg_mc_luma(pixel *dst, int dstride, const pixel *src, int sstride,
                 int w, int h, int fx, int fy)
{
    int x, y;
    const int idx = (fy << 2) | fx;

    if (!fx && !fy) {
        for (y = 0; y < h; y++)
            memcpy(dst + y * dstride, src + y * sstride, (size_t)w * sizeof(pixel));
        return;
    }
    if (!(idx & 5)) {
        /* exactly an integer or half-pel position: single plane */
        int plane = hpel_ref0[idx];
        for (y = 0; y < h; y++)
            for (x = 0; x < w; x++)
                dst[y * dstride + x] = (pixel)plane_get(plane, src + y * sstride + x, sstride);
        return;
    }
    {
        int p1 = hpel_ref0[idx], p2 = hpel_ref1[idx];
        int o1y = (fy == 3) ? 1 : 0;   /* src1 is vertically offset by one */
        int o2x = (fx == 3) ? 1 : 0;   /* src2 is horizontally offset by one */
        for (y = 0; y < h; y++) {
            const pixel *r1 = src + (y + o1y) * sstride;
            const pixel *r2 = src + y * sstride;
            for (x = 0; x < w; x++) {
                int a = plane_get(p1, r1 + x, sstride);
                int b = plane_get(p2, r2 + x + o2x, sstride);
                dst[y * dstride + x] = (pixel)((a + b + 1) >> 1);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Half-pel planes                                                     */
/* ------------------------------------------------------------------ */
/* The three fractional planes are stored with the same geometry as the integer
 * picture -- origin at picture (0,0), `stride` bytes per row, AEG_PAD rows of
 * padding above and below -- so a pointer into the integer picture can be
 * reused on a plane by offsetting from base[0]. */

void aeg_mc_ctx_init(aeg_mc_ctx *mc, int w, int h, int stride)
{
    aeg_mc_ctx_init_pad(mc, w, h, stride, AEG_PAD);
}

void aeg_mc_ctx_init_pad(aeg_mc_ctx *mc, int w, int h, int stride, int vpad)
{
    size_t rows = (size_t)h + 2 * vpad;
    int i;
    (void)w;
    mc->stride = stride;
    mc->vpad = vpad;
    mc->buf = aeg_alloc(3 * (size_t)stride * rows * sizeof(pixel));
    mc->base[0] = NULL;
    mc->ready = 0;
    mc->chroma = 0;
    mc->umh[0] = mc->umh[1] = mc->umh[2] = mc->umh[3] = 0;
    mc->umh_et = 0;
    for (i = 1; i < 4; i++)
        mc->base[i] = mc->buf ? mc->buf + (size_t)(i - 1) * stride * rows +
                                (size_t)vpad * stride + AEG_PAD
                              : NULL;
}

void aeg_mc_ctx_own(aeg_mc_ctx *mc, int h)
{
    int vpad = mc->vpad ? mc->vpad : AEG_PAD;
    size_t rows = (size_t)h + 2 * vpad;
    int i;
    for (i = 1; i < 4; i++)
        mc->base[i] = mc->buf ? mc->buf + (size_t)(i - 1) * mc->stride * rows +
                                (size_t)vpad * mc->stride + AEG_PAD
                              : NULL;
}

void aeg_mc_ctx_free(aeg_mc_ctx *mc)
{
    aeg_free(mc->buf);
    mc->buf = NULL;
    mc->base[0] = NULL;
    mc->ready = 0;
}

/* Fill one plane over the region whose filter taps stay inside the padded
 * integer picture, then replicate the computed border outwards.  Nothing reads
 * the replicated margin: a motion vector plus a block stays inside the padding
 * the encoder allocates, and the region filled here covers it.
 *
 * This is also the CPU reference implementation of the `interp_plane` backend
 * operation: the GPU kernel computes the same samples for the same region and
 * fills the margin the same way, so the buffer it produces is byte-identical
 * and the two paths can be cross-checked (tests/kernel_check.c). */
#if AEG_SSE2
/* Six taps of eight consecutive output samples, as int16 lanes: byte loads are
 * widened to int16 and combined with the filter of hpel_h().  Every
 * intermediate is inside int16 (20 * (255 + 255) + 2 * 255 = 10710), so the
 * lanes hold the scalar value exactly, and the trailing `packus` is the same
 * clip to [0, 255] that aeg_clip_pixel() does -- `packus` saturates unsigned. */
static inline __m128i taps8(const pixel *p, int off)
{
    return _mm_unpacklo_epi8(
        _mm_loadl_epi64((const __m128i *)(p + off)), _mm_setzero_si128());
}

static inline __m128i hpel_h8(const pixel *p)
{
    const __m128i c20 = _mm_set1_epi16(20), c5 = _mm_set1_epi16(5),
                  c16 = _mm_set1_epi16(16);
    const __m128i s0 = _mm_add_epi16(taps8(p, 0), taps8(p, 1));
    const __m128i s1 = _mm_add_epi16(taps8(p, -1), taps8(p, 2));
    const __m128i s2 = _mm_add_epi16(taps8(p, -2), taps8(p, 3));
    __m128i v = _mm_sub_epi16(_mm_mullo_epi16(s0, c20), _mm_mullo_epi16(s1, c5));

    v = _mm_add_epi16(v, s2);
    return _mm_srai_epi16(_mm_add_epi16(v, c16), 5);
}

static inline __m128i hpel_v8(const pixel *p, int s)
{
    const __m128i c20 = _mm_set1_epi16(20), c5 = _mm_set1_epi16(5),
                  c16 = _mm_set1_epi16(16);
    const __m128i s0 = _mm_add_epi16(taps8(p, 0), taps8(p, s));
    const __m128i s1 = _mm_add_epi16(taps8(p, -s), taps8(p, 2 * s));
    const __m128i s2 = _mm_add_epi16(taps8(p, -2 * s), taps8(p, 3 * s));
    __m128i v = _mm_sub_epi16(_mm_mullo_epi16(s0, c20), _mm_mullo_epi16(s1, c5));

    v = _mm_add_epi16(v, s2);
    return _mm_srai_epi16(_mm_add_epi16(v, c16), 5);
}

/* The unshifted horizontal intermediate of hsum_h(), eight lanes at a time. */
static inline __m128i hsum_h8(const pixel *p)
{
    const __m128i c20 = _mm_set1_epi16(20), c5 = _mm_set1_epi16(5);
    const __m128i s0 = _mm_add_epi16(taps8(p, 0), taps8(p, 1));
    const __m128i s1 = _mm_add_epi16(taps8(p, -1), taps8(p, 2));
    const __m128i s2 = _mm_add_epi16(taps8(p, -2), taps8(p, 3));

    return _mm_add_epi16(_mm_sub_epi16(_mm_mullo_epi16(s0, c20),
                                      _mm_mullo_epi16(s1, c5)), s2);
}

/* Wide a 4-lane group of int16 to int32 (sign extended), for the vertical pass
 * of the two-dimensional position: its intermediates reach +-690 000. */
static inline __m128i widen_lo(__m128i v)
{
    return _mm_unpacklo_epi16(v, _mm_srai_epi16(v, 15));
}
static inline __m128i widen_hi(__m128i v)
{
    return _mm_unpackhi_epi16(v, _mm_srai_epi16(v, 15));
}
static inline __m128i times20_epi32(__m128i v)
{
    return _mm_add_epi32(_mm_slli_epi32(v, 4), _mm_slli_epi32(v, 2));
}
static inline __m128i times5_epi32(__m128i v)
{
    return _mm_add_epi32(_mm_slli_epi32(v, 2), v);
}

/* hpel_hv() for eight samples: six horizontal intermediates, then the vertical
 * 6-tap over them in int32 lanes, one shift at the end (H.264 8.4.2.2.1 keeps
 * the horizontal stage unshifted, which is why this cannot be built from the
 * clipped plane-1 samples). */
static inline __m128i hpel_hv8(const pixel *p, int s)
{
    const __m128i a0 = hsum_h8(p - 2 * s), a1 = hsum_h8(p - s);
    const __m128i a2 = hsum_h8(p), a3 = hsum_h8(p + s);
    const __m128i a4 = hsum_h8(p + 2 * s), a5 = hsum_h8(p + 3 * s);
    const __m128i c512 = _mm_set1_epi32(512);
    __m128i lo, hi;

    lo = _mm_add_epi32(widen_lo(a0), times20_epi32(_mm_add_epi32(widen_lo(a2),
                                                                widen_lo(a3))));
    lo = _mm_sub_epi32(lo, times5_epi32(_mm_add_epi32(widen_lo(a1), widen_lo(a4))));
    lo = _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(lo, widen_lo(a5)), c512), 10);
    hi = _mm_add_epi32(widen_hi(a0), times20_epi32(_mm_add_epi32(widen_hi(a2),
                                                                widen_hi(a3))));
    hi = _mm_sub_epi32(hi, times5_epi32(_mm_add_epi32(widen_hi(a1), widen_hi(a4))));
    hi = _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(hi, widen_hi(a5)), c512), 10);
    return _mm_packs_epi32(lo, hi);
}

/* eight filtered samples (int16 lanes) clipped to [0, 255] and stored */
static inline void st_hpel8(pixel *d, __m128i v)
{
    _mm_storel_epi64((__m128i *)d, _mm_packus_epi16(v, _mm_setzero_si128()));
}
#elif AEG_SIMD
/* The same filters for 16-bit samples.  20 (a + b) - 5 (c + d) + (e + f) of
 * 10-bit samples lies in [-10230, 42966]: past int16, but the span is below
 * 2^16, so wrapping 16-bit arithmetic plus a bias of at least 10230 holds the
 * biased value exactly as an unsigned lane. */
static inline __m128i tap6_px16(const pixel *p, ptrdiff_t s, int bias)
{
    const __m128i c20 = _mm_set1_epi16(20), c5 = _mm_set1_epi16(5);
#define T(k) _mm_loadu_si128((const __m128i *)(p + (k) * s))
    const __m128i s0 = _mm_add_epi16(T(0), T(1));
    const __m128i s1 = _mm_add_epi16(T(-1), T(2));
    const __m128i s2 = _mm_add_epi16(T(-2), T(3));
#undef T
    __m128i v = _mm_sub_epi16(_mm_mullo_epi16(s0, c20), _mm_mullo_epi16(s1, c5));
    return _mm_add_epi16(_mm_add_epi16(v, s2), _mm_set1_epi16((short)bias));
}

/* (v + 16) >> 5 == ((v + 16 + 320 * 32) >>> 5) - 320 */
static inline __m128i hpel_h8(const pixel *p)
{
    return _mm_sub_epi16(_mm_srli_epi16(tap6_px16(p, 1, 16 + 320 * 32), 5),
                         _mm_set1_epi16(320));
}

static inline __m128i hpel_v8(const pixel *p, int s)
{
    return _mm_sub_epi16(_mm_srli_epi16(tap6_px16(p, s, 16 + 320 * 32), 5),
                         _mm_set1_epi16(320));
}

/* hpel_hv(): the six horizontal sums biased by K = 10230, widened unsigned;
 * the vertical taps add up to 32, so the bias leaves as 32 K at the end */
static inline __m128i hpel_hv8(const pixel *p, int s)
{
    enum { K = 10230 };
    const __m128i z = _mm_setzero_si128();
    const __m128i a0 = tap6_px16(p - 2 * s, 1, K), a1 = tap6_px16(p - s, 1, K);
    const __m128i a2 = tap6_px16(p, 1, K), a3 = tap6_px16(p + s, 1, K);
    const __m128i a4 = tap6_px16(p + 2 * s, 1, K), a5 = tap6_px16(p + 3 * s, 1, K);
    const __m128i rnd = _mm_set1_epi32(512 - 32 * K);
    __m128i lo, hi;
#define LO(v) _mm_unpacklo_epi16(v, z)
#define HI(v) _mm_unpackhi_epi16(v, z)
#define M20(v) _mm_add_epi32(_mm_slli_epi32(v, 4), _mm_slli_epi32(v, 2))
#define M5(v) _mm_add_epi32(_mm_slli_epi32(v, 2), v)
    lo = _mm_add_epi32(LO(a0), M20(_mm_add_epi32(LO(a2), LO(a3))));
    lo = _mm_sub_epi32(lo, M5(_mm_add_epi32(LO(a1), LO(a4))));
    lo = _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(lo, LO(a5)), rnd), 10);
    hi = _mm_add_epi32(HI(a0), M20(_mm_add_epi32(HI(a2), HI(a3))));
    hi = _mm_sub_epi32(hi, M5(_mm_add_epi32(HI(a1), HI(a4))));
    hi = _mm_srai_epi32(_mm_add_epi32(_mm_add_epi32(hi, HI(a5)), rnd), 10);
#undef LO
#undef HI
#undef M20
#undef M5
    return _mm_packs_epi32(lo, hi);
}

static inline void st_hpel8(pixel *d, __m128i v)
{
    _mm_storeu_si128((__m128i *)d, _mm_min_epi16(_mm_max_epi16(v, _mm_setzero_si128()),
                                                 _mm_set1_epi16(PIXEL_MAX)));
}
#endif

/* Rows ya..yb (picture coordinates, -AEG_PAD..h+AEG_PAD-1) of one plane.
 * Output row y reads reference rows y-2..y+3; the replicated rows above y0 and
 * below y1 are written by the call whose range reaches them (they copy y0 and
 * y1, so those must be part of the same call or an earlier one).  Covering
 * the rows in several calls gives the same plane as one call over all of
 * them, which is what lets a reference be filtered band by band as it is
 * reconstructed (frame threads). */
void aeg_hpel_plane_rows(pixel *pl, int dstride, const pixel *ref,
                         int rstride, int w, int h, int which, int ya, int yb)
{
    int x0 = -AEG_PAD + 2, x1 = w + AEG_PAD - 4;
    int y0 = -AEG_PAD + 2, y1 = h + AEG_PAD - 4;
    int x, y, ys = ya < y0 ? y0 : ya, ye = yb > y1 ? y1 : yb;

    for (y = ys; y <= ye; y++) {
        const pixel *rrow = ref + (ptrdiff_t)y * rstride;
        pixel *orow = pl + (ptrdiff_t)y * dstride;
#if AEG_SIMD
        /* eight samples per iteration; the tail stays scalar, and every load
         * above reaches at most three samples past the last output of the
         * row, which is inside the picture's padding */
        for (x = x0; x + 7 <= x1; x += 8) {
            __m128i v;
            if (which == 1)
                v = hpel_h8(rrow + x);
            else if (which == 2)
                v = hpel_v8(rrow + x, rstride);
            else
                v = hpel_hv8(rrow + x, rstride);
            st_hpel8(orow + x, v);
        }
        for (; x <= x1; x++) {
            const pixel *p = rrow + x;
            int v;
            if (which == 1)
                v = hpel_h(p);
            else if (which == 2)
                v = hpel_v(p, rstride);
            else
                v = hpel_hv(p, rstride);
            orow[x] = aeg_clip_pixel(v);
        }
        x = x1 + 1;
#else
        for (x = x0; x <= x1; x++) {
            const pixel *p = rrow + x;
            int v;
            if (which == 1)
                v = hpel_h(p);
            else if (which == 2)
                v = hpel_v(p, rstride);
            else
                v = hpel_hv(p, rstride);
            orow[x] = aeg_clip_pixel(v);
        }
        x = x1 + 1;
#endif
        for (x = -AEG_PAD; x < x0; x++)
            orow[x] = orow[x0];
        for (x = x1 + 1; x < w + AEG_PAD; x++)
            orow[x] = orow[x1];
    }
    for (y = ya < -AEG_PAD ? -AEG_PAD : ya; y < y0 && y <= yb; y++)
        memcpy(pl + (ptrdiff_t)y * dstride - AEG_PAD,
               pl + (ptrdiff_t)y0 * dstride - AEG_PAD,
               (size_t)(w + 2 * AEG_PAD) * sizeof(pixel));
    for (y = ya > y1 + 1 ? ya : y1 + 1; y < h + AEG_PAD && y <= yb; y++)
        memcpy(pl + (ptrdiff_t)y * dstride - AEG_PAD,
               pl + (ptrdiff_t)y1 * dstride - AEG_PAD,
               (size_t)(w + 2 * AEG_PAD) * sizeof(pixel));
}

void aeg_hpel_plane_fill(pixel *pl, int dstride, const pixel *ref,
                         int rstride, int w, int h, int which)
{
    aeg_hpel_plane_rows(pl, dstride, ref, rstride, w, h, which, -AEG_PAD,
                        h + AEG_PAD - 1);
}

void aeg_mc_ctx_setref(aeg_mc_ctx *mc, const pixel *ref)
{
    mc->base[0] = ref;
}

void aeg_mc_ctx_refresh(aeg_mc_ctx *mc, const pixel *ref, int w, int h)
{
    int i;
    mc->base[0] = ref;
    if (!mc->buf) {
        mc->ready = 0;
        return;
    }
    for (i = 1; i < 4; i++)
        aeg_hpel_plane_fill((pixel *)mc->base[i], mc->stride, ref, mc->stride,
                            w, h, i);
    mc->ready = 1;
}

/* Plane-based motion compensation: same samples as aeg_mc_luma(), read from the
 * precomputed planes instead of being filtered per block. */
void aeg_mc_luma_p(pixel *dst, int dstride, const aeg_mc_ctx *mc,
                   const pixel *src, int w, int h, int fx, int fy)
{
    const ptrdiff_t off = src - mc->base[0];
    const int idx = (fy << 2) | fx;
    int x, y;

    if (!fx && !fy) {
        aeg_mc_luma(dst, dstride, src, mc->stride, w, h, 0, 0);
        return;
    }
    if (!(idx & 5)) {
        /* exactly a half-pel position: a single plane (the integer plane is
         * one of the four a quarter-pel position may combine) */
        const pixel *p = mc->base[hpel_ref0[idx]] + off;
        for (y = 0; y < h; y++)
            memcpy(dst + (size_t)y * dstride, p + (ptrdiff_t)y * mc->stride,
                   (size_t)w * sizeof(pixel));
        return;
    }
    {
        const int pl1 = hpel_ref0[idx], pl2 = hpel_ref1[idx];
        const int o1y = (fy == 3) ? 1 : 0;
        const int o2x = (fx == 3) ? 1 : 0;
        for (y = 0; y < h; y++) {
            const pixel *r1 = mc->base[pl1] + off + (ptrdiff_t)(y + o1y) * mc->stride;
            const pixel *r2 = mc->base[pl2] + off + (ptrdiff_t)y * mc->stride + o2x;
            pixel *d = dst + (size_t)y * dstride;
            x = 0;
#if AEG_SSE2
            /* pavgb is exactly (r1 + r2 + 1) >> 1, the average H.264 8.4.2.2.2
             * specifies, sixteen samples at a time.  The search averages a
             * block per candidate (a million times per 30 frames), so this is
             * one of the hottest loops in the encoder. */
            for (x = 0; x + 16 <= w; x += 16)
                _mm_storeu_si128((__m128i *)(d + x),
                                 _mm_avg_epu8(_mm_loadu_si128((const __m128i *)(r1 + x)),
                                              _mm_loadu_si128((const __m128i *)(r2 + x))));
            if (w - x >= 8) {
                _mm_storel_epi64((__m128i *)(d + x),
                                 _mm_avg_epu8(_mm_loadl_epi64((const __m128i *)(r1 + x)),
                                              _mm_loadl_epi64((const __m128i *)(r2 + x))));
                x += 8;
            }
            if (w - x >= 4) {
                /* (not _mm_storeu_si32: GCC < 9 lacks it) */
                st32(d + x, _mm_cvtsi128_si32(
                                _mm_avg_epu8(_mm_cvtsi32_si128(ld32(r1 + x)),
                                             _mm_cvtsi32_si128(ld32(r2 + x)))));
                x += 4;
            }
#elif AEG_SIMD
            /* pavgw: the same average of 16-bit samples */
            for (x = 0; x + 8 <= w; x += 8)
                _mm_storeu_si128((__m128i *)(d + x),
                                 _mm_avg_epu16(_mm_loadu_si128((const __m128i *)(r1 + x)),
                                               _mm_loadu_si128((const __m128i *)(r2 + x))));
            if (w - x >= 4) {
                _mm_storel_epi64((__m128i *)(d + x),
                                 _mm_avg_epu16(_mm_loadl_epi64((const __m128i *)(r1 + x)),
                                               _mm_loadl_epi64((const __m128i *)(r2 + x))));
                x += 4;
            }
#endif
            for (; x < w; x++)
                d[x] = (pixel)((r1[x] + r2[x] + 1) >> 1);
        }
    }
}

/* aeg_satd() of aeg_mc_luma_p()'s prediction at a quarter-pel position
 * (one that averages two planes) against `cur`, without storing it. */
static int mc_satd_qpel(const aeg_mc_ctx *mc, const pixel *src, int w, int h,
                        int fx, int fy, const pixel *cur, int cstride)
{
    const ptrdiff_t off = src - mc->base[0];
    const int idx = (fy << 2) | fx;
    const pixel *r1 = mc->base[hpel_ref0[idx]] + off + (fy == 3 ? mc->stride : 0);
    const pixel *r2 = mc->base[hpel_ref1[idx]] + off + (fx == 3 ? 1 : 0);
    return aeg_satd_avg(w, h, cur, cstride, r1, r2, mc->stride);
}

void aeg_mc_chroma(pixel *dst, int dstride, const pixel *src, int sstride,
                   int w, int h, int fx, int fy)
{
    int x, y;
    if (!fx && !fy) {
        for (y = 0; y < h; y++)
            memcpy(dst + y * dstride, src + y * sstride, (size_t)w * sizeof(pixel));
        return;
    }
#if AEG_SSE2
    if (w == 8 || w == 4) {
        /* the same four products and rounding, eight (or four) pixels at a
         * time in int16: 64 * 255 fits */
        const __m128i z = _mm_setzero_si128();
        const __m128i wa = _mm_set1_epi16((short)((8 - fx) * (8 - fy)));
        const __m128i wb = _mm_set1_epi16((short)(fx * (8 - fy)));
        const __m128i wc = _mm_set1_epi16((short)((8 - fx) * fy));
        const __m128i wd = _mm_set1_epi16((short)(fx * fy));
        const __m128i r32 = _mm_set1_epi16(32);
#define LDC(q) (w == 8 ? _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(q)), z) \
                       : _mm_unpacklo_epi8(_mm_cvtsi32_si128(ld32(q)), z))
        __m128i a = LDC(src), b = LDC(src + 1);
        for (y = 0; y < h; y++) {
            const pixel *q = src + (size_t)(y + 1) * sstride;
            __m128i c = LDC(q), d = LDC(q + 1), v;
            v = _mm_add_epi16(_mm_add_epi16(_mm_mullo_epi16(a, wa), _mm_mullo_epi16(b, wb)),
                              _mm_add_epi16(_mm_mullo_epi16(c, wc), _mm_mullo_epi16(d, wd)));
            v = _mm_srli_epi16(_mm_add_epi16(v, r32), 6);
            v = _mm_packus_epi16(v, v);
            if (w == 8)
                _mm_storel_epi64((__m128i *)(dst + (size_t)y * dstride), v);
            else
                st32(dst + (size_t)y * dstride, _mm_cvtsi128_si32(v));
            a = c;
            b = d;
        }
#undef LDC
        (void)x;
        return;
    }
#elif AEG_SIMD
    if (w == 8 || w == 4) {
        /* 16-bit samples: the weights add up to 64 and 64 * 1023 + 32 < 2^16,
         * so the four products and the rounding fit unsigned 16-bit lanes */
        const __m128i wa = _mm_set1_epi16((short)((8 - fx) * (8 - fy)));
        const __m128i wb = _mm_set1_epi16((short)(fx * (8 - fy)));
        const __m128i wc = _mm_set1_epi16((short)((8 - fx) * fy));
        const __m128i wd = _mm_set1_epi16((short)(fx * fy));
        const __m128i r32 = _mm_set1_epi16(32);
#define LDC(q) (w == 8 ? _mm_loadu_si128((const __m128i *)(q)) \
                       : _mm_loadl_epi64((const __m128i *)(q)))
        __m128i a = LDC(src), b = LDC(src + 1);
        for (y = 0; y < h; y++) {
            const pixel *q = src + (size_t)(y + 1) * sstride;
            __m128i c = LDC(q), d = LDC(q + 1), v;
            v = _mm_add_epi16(_mm_add_epi16(_mm_mullo_epi16(a, wa), _mm_mullo_epi16(b, wb)),
                              _mm_add_epi16(_mm_mullo_epi16(c, wc), _mm_mullo_epi16(d, wd)));
            v = _mm_srli_epi16(_mm_add_epi16(v, r32), 6);
            if (w == 8)
                _mm_storeu_si128((__m128i *)(dst + (size_t)y * dstride), v);
            else
                _mm_storel_epi64((__m128i *)(dst + (size_t)y * dstride), v);
            a = c;
            b = d;
        }
#undef LDC
        (void)x;
        return;
    }
#endif
    for (y = 0; y < h; y++)
        for (x = 0; x < w; x++) {
            const pixel *p = src + y * sstride + x;
            int a = p[0], b = p[1], c = p[sstride], d = p[sstride + 1];
            int v = (8 - fx) * (8 - fy) * a + fx * (8 - fy) * b +
                    (8 - fx) * fy * c + fx * fy * d;
            dst[y * dstride + x] = (pixel)((v + 32) >> 6);
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
static inline int me_sad(const pixel *ref, int rstride, const pixel *cur,
                         int cstride, int bw, int bh, int dx, int dy)
{
    return aeg_sad(bw, bh, cur, cstride, ref + dy * rstride + dx, rstride);
}

/* me_sad() of n integer positions pts[i] (all inside the range), four at a
 * time through aeg_sad_x4(). */
static void me_sad_list(const pixel *ref, int rstride, const pixel *cur,
                        int cstride, int bw, int bh, const int (*pts)[2], int n,
                        int *out)
{
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        const pixel *b[4];
        int k;
        for (k = 0; k < 4; k++)
            b[k] = ref + pts[i + k][1] * rstride + pts[i + k][0];
        aeg_sad_x4(bw, bh, cur, cstride, b, rstride, out + i);
    }
    if (n - i >= 2) {
        /* two or three left: one more four-way call (the last position
         * repeated) is cheaper than two or three single ones */
        const pixel *b[4];
        int k, o[4];
        for (k = 0; k < 4; k++) {
            int q = i + k < n ? i + k : n - 1;
            b[k] = ref + pts[q][1] * rstride + pts[q][0];
        }
        aeg_sad_x4(bw, bh, cur, cstride, b, rstride, o);
        for (k = 0; i + k < n; k++)
            out[i + k] = o[k];
        return;
    }
    for (; i < n; i++)
        out[i] = me_sad(ref, rstride, cur, cstride, bw, bh, pts[i][0], pts[i][1]);
}

/* Bits of one signed Exp-Golomb mvd component (quarter-pel units). */
static inline int me_se_bits(int v)
{
    unsigned m = (unsigned)(v < 0 ? -v : v) * 2u;
    return m ? 2 * (31 - __builtin_clz(m)) + 1 : 1;   /* 1 for 0, 3 for +-1, 5 for +-2..3, ... = 2*floor(log2(2|v|))+1 */
}

/* What the search minimises: distortion plus lambda times the bits of the
 * vector's difference from its predictor.  Without the rate term a search on
 * real (noisy) content wanders to whichever vector the noise favours, and that
 * vector then costs an mvd and breaks the skip/direct runs the predictor would
 * have given for free. */
typedef struct me_rate {
    int lambda;          /* 0 = distortion only                         */
    int px, py;          /* predictor, quarter-pel                      */
    int dia;             /* sub-pel steps on the 4-point diamond        */
    int qonly;           /* start at a fractional vector, quarter-pel
                          * steps only (aeg_me_block_fs)                 */
} me_rate;

static inline int me_mv_cost(const me_rate *r, int qx, int qy)
{
    if (!r->lambda)
        return 0;
    return r->lambda * (me_se_bits(qx - r->px) + me_se_bits(qy - r->py));
}

/* `cost` is the cost (SAD + rate) of the start, already measured */
static int hex_search(const pixel *ref, int rstride, const pixel *cur,
                      int cstride, int bw, int bh, int range,
                      const me_rate *rt, int cost, int *mvx, int *mvy, int *dist)
{
    int cx = *mvx, cy = *mvy;
    int best = cost;
    int bd = best - me_mv_cost(rt, cx * 4, cy * 4);
    int step, i, improved = 1;

    /* The positions the previous round measured (and its centre): one of
     * them cannot beat the best it already lost to, and a hexagon or square
     * that moved by one step overlaps the last one by three to five points
     * (the first square round by four of the last hexagon's six), so they
     * are not measured again -- the same walk, fewer SADs. */
    int vis[9][2], nvis = 1;
    vis[0][0] = cx;
    vis[0][1] = cy;
    for (step = 0; step < 2 && improved; step++) {
        const int8_t *dx = step ? sq_dx : hex_dx;
        const int8_t *dy = step ? sq_dy : hex_dy;
        int n = step ? 9 : 7;
        improved = 1;
        while (improved) {
            int bx = cx, by = cy, bc = best, bdd = bd;
            int pts[8][2], sads[8], np = 0, k;
            improved = 0;
            for (i = 1; i < n; i++) {
                int nx = cx + dx[i], ny = cy + dy[i];
                if (nx < -range || nx > range || ny < -range || ny > range)
                    continue;
                for (k = 0; k < nvis; k++)
                    if (vis[k][0] == nx && vis[k][1] == ny)
                        break;
                if (k < nvis)
                    continue;
                pts[np][0] = nx;
                pts[np][1] = ny;
                np++;
            }
            /* this round's positions, for the next */
            for (k = 0; k < np; k++) {
                vis[k][0] = pts[k][0];
                vis[k][1] = pts[k][1];
            }
            vis[np][0] = cx;
            vis[np][1] = cy;
            nvis = np + 1;
            me_sad_list(ref, rstride, cur, cstride, bw, bh,
                        (const int (*)[2])pts, np, sads);
            for (i = 0; i < np; i++) {
                int nx = pts[i][0], ny = pts[i][1], d = sads[i];
                int c = d + me_mv_cost(rt, nx * 4, ny * 4);
                if (c < bc) { bc = c; bdd = d; bx = nx; by = ny; improved = 1; }
            }
            cx = bx; cy = by; best = bc; bd = bdd;
        }
    }
    *mvx = cx; *mvy = cy;
    *dist = bd;
    return best;
}

/* best/bx/by of the n positions pts[] with SADs sads[]: each position's cost
 * is its SAD plus me_mv_cost(), and a position replaces the best only when it
 * is strictly cheaper, in the listed order.  The rate term four positions at a
 * time: the bits of an Exp-Golomb component are 2*floor(log2(2|d| | 1)) + 1,
 * and floor(log2) of a small integer is the exponent of its float. */
static void me_pick(const me_rate *rt, const int (*pts)[2], const int *sads,
                    int n, int *best, int *bx, int *by)
{
    int k = 0, b = *best, x = *bx, y = *by;
#if AEG_SIMD     /* rate lanes only: the same at both depths */
    if (rt->lambda > 0 && rt->lambda < 32768) {
        const __m128i px = _mm_set1_epi32(rt->px), py = _mm_set1_epi32(rt->py);
        const __m128i one = _mm_set1_epi32(1), lam = _mm_set1_epi32(rt->lambda);
        const __m128i e127 = _mm_set1_epi32(127 * 2 - 1);
        for (; k + 4 <= n; k += 4) {
            __m128 lo = _mm_castsi128_ps(_mm_loadu_si128((const __m128i *)pts[k]));
            __m128 hi = _mm_castsi128_ps(_mm_loadu_si128((const __m128i *)pts[k + 2]));
            __m128i vx = _mm_castps_si128(_mm_shuffle_ps(lo, hi, _MM_SHUFFLE(2, 0, 2, 0)));
            __m128i vy = _mm_castps_si128(_mm_shuffle_ps(lo, hi, _MM_SHUFFLE(3, 1, 3, 1)));
            __m128i dx = _mm_sub_epi32(_mm_slli_epi32(vx, 2), px);
            __m128i dy = _mm_sub_epi32(_mm_slli_epi32(vy, 2), py);
            __m128i sx = _mm_srai_epi32(dx, 31), sy = _mm_srai_epi32(dy, 31);
            __m128i ex, ey, bits, c;
            int cc[4], j;
            dx = _mm_sub_epi32(_mm_xor_si128(dx, sx), sx);
            dy = _mm_sub_epi32(_mm_xor_si128(dy, sy), sy);
            dx = _mm_or_si128(_mm_slli_epi32(dx, 1), one);
            dy = _mm_or_si128(_mm_slli_epi32(dy, 1), one);
            ex = _mm_srli_epi32(_mm_castps_si128(_mm_cvtepi32_ps(dx)), 23);
            ey = _mm_srli_epi32(_mm_castps_si128(_mm_cvtepi32_ps(dy)), 23);
            /* (2(ex-127)+1) + (2(ey-127)+1) */
            bits = _mm_sub_epi32(_mm_slli_epi32(_mm_add_epi32(ex, ey), 1),
                                 _mm_slli_epi32(e127, 1));
            c = _mm_add_epi32(_mm_loadu_si128((const __m128i *)(sads + k)),
                              _mm_madd_epi16(bits, lam));
            _mm_storeu_si128((__m128i *)cc, c);
            for (j = 0; j < 4; j++)
                if (cc[j] < b) { b = cc[j]; x = pts[k + j][0]; y = pts[k + j][1]; }
        }
    }
#endif
    for (; k < n; k++) {
        int c = sads[k] + me_mv_cost(rt, pts[k][0] * 4, pts[k][1] * 4);
        if (c < b) { b = c; x = pts[k][0]; y = pts[k][1]; }
    }
    *best = b; *bx = x; *by = y;
}

/* x264's uneven multi-hexagon search, simplified: from the best start, a
 * cross of every other sample out to +-r horizontally and +-r/2 vertically,
 * a 5x5 square, and 16-point hexagons at radii 4, 8, .. r; each stage keeps
 * the best position.  The caller's hexagon walk refines the result. */
static void umh_search(const pixel *ref, int rstride, const pixel *cur,
                       int cstride, int bw, int bh, int range, int r,
                       const me_rate *rt, int *mvx, int *mvy, int *best_cost)
{
    static const int8_t hex16[16][2] = {
        {-4, 2}, {-4, 1}, {-4, 0}, {-4, -1}, {-4, -2},
        { 4, 2}, { 4, 1}, { 4, 0}, { 4, -1}, { 4, -2},
        { 2, 3}, { 0, 4}, {-2, 3}, { 2, -3}, { 0, -4}, {-2, -3}
    };
    int cx = *mvx, cy = *mvy, best = *best_cost, bx = cx, by = cy, i, j;
    /* Each stage's positions depend only on where the stage starts, so a
     * stage is listed first, its SADs measured four at a time, and then
     * compared in the listed order -- the same winner as one at a time. */
    enum { UMH_MAX = 512 };
    int pts[UMH_MAX][2], sads[UMH_MAX], np = 0;

#define UMH_TRY(X, Y) do {                                                  \
        int x_ = (X), y_ = (Y);                                              \
        if (x_ >= -range && x_ <= range && y_ >= -range && y_ <= range &&    \
            np < UMH_MAX) {                                                  \
            pts[np][0] = x_;                                                 \
            pts[np][1] = y_;                                                 \
            np++;                                                            \
        }                                                                    \
    } while (0)
#define UMH_FLUSH() do {                                                    \
        int k_;                                                              \
        me_sad_list(ref, rstride, cur, cstride, bw, bh,                      \
                    (const int (*)[2])pts, np, sads);                        \
        me_pick(rt, (const int (*)[2])pts, sads, np, &best, &bx, &by);       \
        (void)k_;                                                            \
        np = 0;                                                              \
    } while (0)
    for (i = 1; i <= r; i += 2) {
        UMH_TRY(cx + i, cy);
        UMH_TRY(cx - i, cy);
    }
    for (i = 1; i <= r / 2; i += 2) {
        UMH_TRY(cx, cy + i);
        UMH_TRY(cx, cy - i);
    }
    UMH_FLUSH();
    cx = bx; cy = by;
    for (j = -2; j <= 2; j++)
        for (i = -2; i <= 2; i++)
            if (i || j)
                UMH_TRY(cx + i, cy + j);
    UMH_FLUSH();
    cx = bx; cy = by;
    for (j = 1; j <= r / 4; j++) {
        for (i = 0; i < 16; i++)
            UMH_TRY(cx + hex16[i][0] * j, cy + hex16[i][1] * j);
        if (np > UMH_MAX - 16)
            UMH_FLUSH();
    }
    UMH_FLUSH();
#undef UMH_TRY
#undef UMH_FLUSH
    *mvx = bx; *mvy = by; *best_cost = best;
}

static int me_umh_radius(const aeg_mc_ctx *mc, int bw, int bh)
{
    if (!mc)
        return 0;
    if (bw == 16 && bh == 16)
        return mc->umh[0];
    if (bw >= 16 || bh >= 16)
        return mc->umh[1];
    return bw == 8 && bh == 8 ? mc->umh[2] : mc->umh[3];
}

static int me_try_pmv(void)
{
    static _Atomic int v = -1;
    if (v < 0) {
        const char *ev = getenv("AEG_MEPMV");
        v = ev ? atoi(ev) != 0 : 0;
    }
    return v;
}

/* SATD of both chroma planes of the block at `cur` (luma) predicted with the
 * quarter-pel luma vector (nx, ny), i.e. an eighth-pel chroma vector. */
static int me_chroma_satd(const aeg_mc_ctx *mc, const pixel *cur,
                          int rstride, int bw, int bh, int nx, int ny)
{
    ptrdiff_t off = cur - mc->cur_y;
    int y = (int)(off / rstride), x = (int)(off - (ptrdiff_t)y * rstride);
    int cw = bw >> 1, ch = bh >> 1, pl, d = 0;
    pixel buf[8 * 8];

    for (pl = 0; pl < 2; pl++) {
        const pixel *src = mc->cref[pl] +
                             ((y >> 1) + (ny >> 3)) * mc->cstride +
                             (x >> 1) + (nx >> 3);
        aeg_mc_chroma(buf, 8, src, mc->cstride, cw, ch, nx & 7, ny & 7);
        d += aeg_satd(cw, ch, buf, 8,
                      mc->cur_c[pl] + (y >> 1) * mc->cstride + (x >> 1),
                      mc->cstride);
    }
    return d;
}

/* subpel refinement: quarter-pel, SATD (+ lambda * mvd bits) driven */
static int subpel_refine(const pixel *ref, int rstride, const pixel *cur,
                         int cstride, int bw, int bh, int range,
                         const aeg_mc_ctx *mc, const me_rate *rt,
                         int *mvx, int *mvy, int *dist)
{
    static const int8_t offs[9][2] = {
        {0,0},{-1,0},{1,0},{0,-1},{0,1},{-1,-1},{1,-1},{-1,1},{1,1}
    };
    pixel buf[16 * 16];
    int best, bd, bx = *mvx, by = *mvy;
    int i, pass;
    int cx = *mvx, cy = *mvy;

    /* The cost of the position the walk starts from -- the centre candidate of
     * every iteration -- is the block at the integer position, which needs no
     * interpolation, so evaluate it once here and skip the centre from then on:
     * it can never improve on itself, and it is 1 of the 9 candidates of every
     * step of every pass. */
    const int cme = mc && mc->chroma && bw >= 8 && bh >= 8 && cstride == rstride;

    if (rt->qonly && ((cx | cy) & 3)) {
        const pixel *r0 = ref + (cy >> 2) * rstride + (cx >> 2);
        if (mc && mc->ready)
            aeg_mc_luma_p(buf, bw, mc, r0, bw, bh, cx & 3, cy & 3);
        else
            aeg_mc_luma(buf, bw, r0, rstride, bw, bh, cx & 3, cy & 3);
        bd = aeg_satd(bw, bh, buf, bw, cur, cstride);
    } else {
        bd = aeg_satd(bw, bh, cur, cstride,
                      ref + (cy >> 2) * rstride + (cx >> 2), rstride);
    }
    if (cme)
        bd += me_chroma_satd(mc, cur, rstride, bw, bh, cx, cy);
    best = bd + me_mv_cost(rt, cx, cy);
    /* The predictor itself, at its own fractional position: it costs no mvd
     * bits at all, and the integer search can only reach its rounded
     * neighbourhood (x264 tries it the same way). */
    if (me_try_pmv() && (rt->px != cx || rt->py != cy) &&
        ((rt->px | rt->py) & 3) &&
        abs(rt->px - cx) <= 8 && abs(rt->py - cy) <= 8) {
        int nx = rt->px, ny = rt->py, ix = nx >> 2, iy = ny >> 2;
        if (ix >= -range && ix <= range && iy >= -range && iy <= range) {
            int d, cost;
            if (mc && mc->ready)
                aeg_mc_luma_p(buf, bw, mc, ref + iy * rstride + ix, bw, bh,
                              nx & 3, ny & 3);
            else
                aeg_mc_luma(buf, bw, ref + iy * rstride + ix, rstride, bw,
                            bh, nx & 3, ny & 3);
            d = aeg_satd(bw, bh, buf, bw, cur, cstride);
            if (cme)
                d += me_chroma_satd(mc, cur, rstride, bw, bh, nx, ny);
            cost = d + me_mv_cost(rt, nx, ny);
            if (cost < best) {
                best = cost; bd = d; bx = cx = nx; by = cy = ny;
            }
        }
    }

    /* Two passes: half-pel then quarter-pel steps.  Each pass walks until it
     * stops improving, but at most `iters` steps: past the second step the
     * search has almost always converged (the vectors it would still find move
     * by a quarter pel more and change the SATD by less than the mvd they
     * cost), and every step is a full motion compensation. */
    for (pass = rt->qonly ? 1 : 0; pass < 2; pass++) {
        int step = pass ? 1 : 2;
        int improved = 1, iters = 0;
        int pcx = 0, pcy = 0, pstep = 0;
        while (improved && iters++ < AEG_SUBPEL_ITERS) {
            int pvx = pcx, pvy = pcy, pst = pstep;
            improved = 0;
            /* the next round skips what this one measures (hex_search()) */
            pcx = cx; pcy = cy; pstep = step;
            for (i = 1; i < (rt->dia ? 5 : 9); i++) {
                int nx = cx + offs[i][0] * step, ny = cy + offs[i][1] * step;
                int ix = nx >> 2, iy = ny >> 2, cost, d;
                if (ix < -range || ix > range || iy < -range || iy > range)
                    continue;
                if (nx < -range * 4 || nx > range * 4 || ny < -range * 4 || ny > range * 4)
                    continue;
                /* measured by the previous round of this pass: its centre
                 * and the eight around it */
                if (pst == step && (rt->dia ? abs(nx - pvx) + abs(ny - pvy) <= step
                                            : abs(nx - pvx) <= step && abs(ny - pvy) <= step) &&
                    (nx - pvx) % step == 0 && (ny - pvy) % step == 0)
                    continue;
                if (mc && mc->ready && mc->stride == rstride &&
                    !(((ny & 3) << 2 | (nx & 3)) & 5)) {
                    /* integer or half-pel: the samples are a plane's as they
                     * are, so measure them in place instead of copying */
                    const ptrdiff_t off = ref + iy * rstride + ix - mc->base[0];
                    d = aeg_satd(bw, bh, mc->base[hpel_ref0[((ny & 3) << 2) | (nx & 3)]] + off,
                                 rstride, cur, cstride);
                } else if (mc && mc->ready) {
                    d = mc_satd_qpel(mc, ref + iy * rstride + ix, bw, bh,
                                     nx & 3, ny & 3, cur, cstride);
                } else {
                    aeg_mc_luma(buf, bw, ref + iy * rstride + ix, rstride, bw,
                                bh, nx & 3, ny & 3);
                    d = aeg_satd(bw, bh, buf, bw, cur, cstride);
                }
                if (cme)
                    d += me_chroma_satd(mc, cur, rstride, bw, bh, nx, ny);
                cost = d + me_mv_cost(rt, nx, ny);
                if (cost < best) { best = cost; bd = d; bx = nx; by = ny; improved = 1; }
            }
            cx = bx; cy = by;
        }
    }
    *mvx = bx; *mvy = by;
    *dist = bd;
    return best;
}

/* The motion-search lambda (SAD/SATD scale): 2^((qp - 12) / 6) rounded to
 * the nearest integer, at least 1 -- a factor of 2 per 6 QP, the step size's
 * own growth, with 1 at QP 12.  (This is the lambda x264 uses; the values
 * are computed from that definition here.  Every one of them is at least
 * 0.008 away from a rounding boundary, so pow() cannot tip any.)  Exposed so
 * that every caller prices vectors the same way. */
static int me_lambda_tab[52];
static pthread_once_t me_lambda_once = PTHREAD_ONCE_INIT;

static void me_lambda_init(void)
{
    int qp;
    for (qp = 0; qp < 52; qp++) {
        const int v = (int)floor(pow(2.0, (qp - 12) / 6.0) + 0.5);
        me_lambda_tab[qp] = v < 1 ? 1 : v;
    }
}

int aeg_me_lambda(int qp)
{
    pthread_once(&me_lambda_once, me_lambda_init);
    if (qp < 0) qp = 0;
    if (qp > 51) qp = 51;
    /* a SAD at high bit depth is 2^(BIT_DEPTH-8) times the 8-bit one */
    return me_lambda_tab[qp] << (BIT_DEPTH - 8);
}

int aeg_me_block(const pixel *ref, int rstride, const pixel *cur,
                 int cstride, int bw, int bh, int range, int subme,
                 int pred_x, int pred_y, const int seed[2],
                 const aeg_mc_ctx *mc, int *mvx, int *mvy, int *sad_out)
{
    return aeg_me_block_l(ref, rstride, cur, cstride, bw, bh, range, subme,
                          pred_x, pred_y, seed, NULL, 0, 0, mc, mvx, mvy,
                          sad_out);
}

/* The motion search proper.  `extra` holds `nextra` more integer-pel start
 * candidates (neighbours' vectors and the like) besides the predictor, zero
 * and `seed`; `lambda` > 0 adds lambda * mvd bits to every cost.  *sad_out is
 * the distortion (SAD, or SATD after the fractional stage) of the vector
 * returned, without the rate term. */
int aeg_me_block_l(const pixel *ref, int rstride, const pixel *cur,
                   int cstride, int bw, int bh, int range, int subme,
                   int pred_x, int pred_y, const int seed[2],
                   const int (*extra)[2], int nextra, int lambda,
                   const aeg_mc_ctx *mc, int *mvx, int *mvy, int *sad_out)
{
    return aeg_me_block_t(ref, rstride, cur, cstride, bw, bh, range, subme,
                          pred_x, pred_y, seed, extra, nextra, lambda, mc,
                          mvx, mvy, sad_out, NULL);
}

/* aeg_me_block_l() with x264's p_halfpel_thresh: when `hpt` is given, the
 * integer-pel cost (SAD + vector bits) of the searches of one macroblock's
 * references, made in order, is compared with the best of them so far; one
 * more than 8/7 of it returns -1 before the sub-pel stage (the vector is the
 * integer one), otherwise *hpt keeps the minimum. */
static int me_block_inner(const pixel *ref, int rstride, const pixel *cur,
                          int cstride, int bw, int bh, int range, int subme,
                          int pred_x, int pred_y, const int seed[2],
                          const int (*extra)[2], int nextra, int lambda,
                          const aeg_mc_ctx *mc, int *mvx, int *mvy,
                          int *sad_out, int *hpt, const int fseed[2]);

int aeg_me_block_t(const pixel *ref, int rstride, const pixel *cur,
                   int cstride, int bw, int bh, int range, int subme,
                   int pred_x, int pred_y, const int seed[2],
                   const int (*extra)[2], int nextra, int lambda,
                   const aeg_mc_ctx *mc, int *mvx, int *mvy, int *sad_out,
                   int *hpt)
{
    return me_block_inner(ref, rstride, cur, cstride, bw, bh, range, subme,
                          pred_x, pred_y, seed, extra, nextra, lambda, mc,
                          mvx, mvy, sad_out, hpt, NULL);
}

/* aeg_me_block_t() with a quarter-pel vector known to be good (a partition's
 * own macroblock vector): when the integer search ends on its integer
 * position, the fractional stage starts from it and takes quarter-pel steps
 * only. */
int aeg_me_block_fs(const pixel *ref, int rstride, const pixel *cur,
                    int cstride, int bw, int bh, int range, int subme,
                    int pred_x, int pred_y, const int seed[2],
                    const int (*extra)[2], int nextra, int lambda,
                    const aeg_mc_ctx *mc, int *mvx, int *mvy, int *sad_out,
                    int *hpt, const int fseed[2])
{
    return me_block_inner(ref, rstride, cur, cstride, bw, bh, range, subme,
                          pred_x, pred_y, seed, extra, nextra, lambda, mc,
                          mvx, mvy, sad_out, hpt, fseed);
}

static int me_block_inner(const pixel *ref, int rstride, const pixel *cur,
                          int cstride, int bw, int bh, int range, int subme,
                          int pred_x, int pred_y, const int seed[2],
                          const int (*extra)[2], int nextra, int lambda,
                          const aeg_mc_ctx *mc, int *mvx, int *mvy,
                          int *sad_out, int *hpt, const int fseed[2])
{
    int best_cost = 1 << 30, best_sad = 1 << 30, bx = 0, by = 0;
    me_rate rt;

    rt.lambda = lambda;
    rt.px = pred_x;
    rt.py = pred_y;
    rt.dia = !!(subme & AEG_ME_DIA);
    rt.qonly = 0;
    subme &= ~AEG_ME_DIA;

    /* Candidate starts: predicted MV, zero, and the integer-rounded
     * predicted MV -- plus `seed`, the best vector a batched search over the
     * whole frame found for this block (see me_seed_frame() in the encoder;
     * it is the one part of the motion search that runs as a single GPU
     * batch).  Evaluating all of them keeps the search robust for fast
     * motion without the cost of a full search. */
    {
        int cand[4 + 16][2];
        int i, n = 3;
        cand[0][0] = pred_x >> 2; cand[0][1] = pred_y >> 2;
        cand[1][0] = 0;           cand[1][1] = 0;
        cand[2][0] = (pred_x + 2) >> 2; cand[2][1] = (pred_y + 2) >> 2;
        if (seed) {
            cand[3][0] = seed[0];
            cand[3][1] = seed[1];
            n = 4;
        }
        for (i = 0; i < nextra && i < 16; i++) {
            cand[n][0] = (extra[i][0] + 2) >> 2;
            cand[n][1] = (extra[i][1] + 2) >> 2;
            n++;
        }
        int pts[4 + 16][2], sads[4 + 16], np = 0, k;
        for (i = 0; i < n; i++) {
            if (cand[i][0] < -range || cand[i][0] > range ||
                cand[i][1] < -range || cand[i][1] > range)
                continue;
            for (k = 0; k < np; k++)
                if (pts[k][0] == cand[i][0] && pts[k][1] == cand[i][1])
                    break;
            if (k < np)
                continue;
            pts[np][0] = cand[i][0];
            pts[np][1] = cand[i][1];
            np++;
        }
        /* measured four at a time, compared in the listed order */
        me_sad_list(ref, rstride, cur, cstride, bw, bh,
                    (const int (*)[2])pts, np, sads);
        me_pick(&rt, (const int (*)[2])pts, sads, np, &best_cost, &bx, &by);
    }
    {
        static _Atomic int fr = -1;
        if (fr < 0) { const char *ev = getenv("AEG_MEFULL"); fr = ev ? atoi(ev) : 0; }
        if (fr > 0) {
            int x, y, cx = 0, cy = 0;
            for (y = -fr; y <= fr; y++)
                for (x = -fr; x <= fr; x++) {
                    int c;
                    if (x < -range || x > range || y < -range || y > range)
                        continue;
                    c = me_sad(ref, rstride, cur, cstride, bw, bh, x, y) +
                        me_mv_cost(&rt, x * 4, y * 4);
                    if (c < best_cost) { best_cost = c; cx = x; cy = y; bx = x; by = y; }
                }
            (void)cx; (void)cy;
        }
    }
    {
        int ur = me_umh_radius(mc, bw, bh);
        /* x264-style early exit: when the best of the predictors is the
         * (rounded) predicted MV itself and already cheap, the
         * multi-hexagon stage is skipped and only the hexagon walk runs. */
        /* (the threshold is in 8-bit cost units: x4 at 10 bits, as the SAD
         * and the motion lambda) */
        if (ur > 0 && mc->umh_et > 0 &&
            best_cost * 256 < (mc->umh_et << (BIT_DEPTH - 8)) * bw * bh &&
            ((bx == pred_x >> 2 && by == pred_y >> 2) ||
             (bx == (pred_x + 2) >> 2 && by == (pred_y + 2) >> 2)))
            ur = 0;
        if (ur > 0)
            umh_search(ref, rstride, cur, cstride, bw, bh, range, ur, &rt,
                       &bx, &by, &best_cost);
    }
    hex_search(ref, rstride, cur, cstride, bw, bh, range, &rt, best_cost,
               &bx, &by, &best_sad);
    if (hpt) {
        const int ic = best_sad + me_mv_cost(&rt, bx * 4, by * 4);
        if (((int64_t)ic * 7 >> 3) > *hpt) {
            *mvx = bx * 4; *mvy = by * 4; *sad_out = best_sad;
            return -1;
        }
        if (ic < *hpt)
            *hpt = ic;
    }

    if (subme <= 0) {
        *mvx = bx * 4; *mvy = by * 4; *sad_out = best_sad;
        return best_sad;
    }

    {
        int qx = bx * 4, qy = by * 4;
        if (fseed && fseed[0] >> 2 == bx && fseed[1] >> 2 == by) {
            qx = fseed[0];
            qy = fseed[1];
            rt.qonly = 1;
        }
        subpel_refine(ref, rstride, cur, cstride, bw, bh, range, mc, &rt,
                      &qx, &qy, &best_sad);
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
    return m->type == AEG_MB_I4x4 || m->type == AEG_MB_I8x8 ||
           m->type == AEG_MB_I16x16 || m->type == AEG_MB_IPCM;
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
                    int top_row, int *mvx, int *mvy)
{
    const aeg_mb *m;
    int ax = 0, ay = 0, bx = 0, by = 0, cx = 0, cy = 0;
    int a_av = 0, b_av = 0, c_av = 0, n;

    if (mb_x > 0) {
        m = &mbs[mb_y * mb_stride + mb_x - 1];
        a_av = !is_intra_mb(m);
        ax = m->mv[3][0];  ay = m->mv[3][1];
    }
    if (mb_y > top_row) {
        m = &mbs[(mb_y - 1) * mb_stride + mb_x];
        b_av = !is_intra_mb(m);
        bx = m->mv[12][0]; by = m->mv[12][1];
    }
    if (mb_y > top_row && mb_x + 1 < mb_w) {
        m = &mbs[(mb_y - 1) * mb_stride + mb_x + 1];
        c_av = !is_intra_mb(m);
        cx = m->mv[12][0]; cy = m->mv[12][1];
    } else if (mb_y > top_row && mb_x > 0) {
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
    /* ue(v) is 2 * bitlen(v + 1) - 1 bits long */
    uint32_t t = v + 1;
    int n = 0;

    while (t) { n++; t >>= 1; }
    return 2 * n - 1;
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
