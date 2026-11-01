/*
 * The sample and coefficient types of one build (see h264.h).
 */
#ifndef AEG_BITDEPTH_H
#define AEG_BITDEPTH_H

#include <stdint.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* Bit depth                                                           */
/* ------------------------------------------------------------------ */
/* The depth-dependent sources are compiled twice, as x264 does: once with
 * BIT_DEPTH 8 (pixel = uint8_t) and once with BIT_DEPTH 10 (pixel = uint16_t,
 * the symbols renamed with a _10 suffix by the Makefile); api.c picks the
 * build an encoder runs on from aeg_params.bit_depth.  dctcoef holds the
 * transform's coefficients and quantised levels, which outgrow 16 bits at
 * high depth.  QP_BD_OFFSET is QpBdOffsetY.  Every QP the encoder holds
 * (macroblock QPs, the rate control, the slice header, mb_qp_delta, the
 * deblocking filter) is QPY, -QP_BD_OFFSET..51; the offset is added only
 * where a quantiser table is indexed (QP'Y = QPY + QP_BD_OFFSET). */
#ifndef BIT_DEPTH
#define BIT_DEPTH 8
#endif
#if BIT_DEPTH == 8
typedef uint8_t pixel;
typedef int16_t dctcoef;
#elif BIT_DEPTH == 10
typedef uint16_t pixel;
typedef int32_t dctcoef;
#else
#error "BIT_DEPTH must be 8 or 10"
#endif
#define PIXEL_MAX     ((1 << BIT_DEPTH) - 1)
#define PIXEL_MID     (1 << (BIT_DEPTH - 1))
#define QP_BD_OFFSET  (6 * (BIT_DEPTH - 8))
#define QP_MIN        (-QP_BD_OFFSET)   /* the lowest QPY */
/* AEG_SSE2: the SIMD paths written for 8-bit samples and 16-bit
 * coefficients.  AEG_SIMD: SSE2 at either depth, for the kernels that have a
 * 16-bit-sample form too (pixel loads through the helpers in transform.c). */
#if defined(__SSE2__) && BIT_DEPTH == 8
#define AEG_SSE2 1
#else
#define AEG_SSE2 0
#endif
#if defined(__SSE2__)
#define AEG_SIMD 1
#else
#define AEG_SIMD 0
#endif
/* n samples of value v (memset at 8 bits) */
static inline void aeg_pixel_fill(pixel *d, int v, size_t n)
{
#if BIT_DEPTH == 8
    memset(d, v, n);
#else
    size_t i;
    for (i = 0; i < n; i++)
        d[i] = (pixel)v;
#endif
}
static inline pixel aeg_clip_pixel(int v)
{
    return (pixel)(v < 0 ? 0 : v > PIXEL_MAX ? PIXEL_MAX : v);
}

#endif /* AEG_BITDEPTH_H */
