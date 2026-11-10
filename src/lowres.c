/*
 * The lookahead's half-resolution luma (see lookahead.c), built from the
 * source picture.  Depth-dependent: compiled once per bit depth.  The lowres
 * plane itself is always 8-bit -- a high-depth source is scaled down to
 * eight bits on the way (exactly the 8-bit result for a source that was
 * scaled up from eight bits) -- so the lookahead, its decisions and the cost
 * estimates the rate control runs on are the same code at every depth.
 */
#include "h264.h"

#define LA_PAD AEG_LA_PAD

/* Box-filter the luma plane down by two in each direction; the part of the
 * macroblock grid outside the picture and the padding repeat the edge. */
static void aeg_lowres_build_ph(aeg_lowres *l, const pixel *src, int sstride,
                      int w, int h);

/* (timed as a phase, see aeg_ph_switch) */
void aeg_lowres_build(aeg_lowres *l, const pixel *src, int sstride,
                      int w, int h)
{
    const int prev = aeg_ph_push(AEG_PH_LOWRES);

    aeg_lowres_build_ph(l, src, sstride, w, h);
    aeg_ph_pop(prev);
}

static void aeg_lowres_build_ph(aeg_lowres *l, const pixel *src, int sstride,
                      int w, int h)
{
    aeg_lowres_build_rows(l, src, sstride, w, h, 0, l->h - 1);
}

/* Rows y0..y1 of aeg_lowres_build(); the padding above follows row 0 and the
 * padding below the last row, in the call that builds them. */
static void aeg_lowres_build_rows_ph(aeg_lowres *l, const pixel *src, int sstride,
                           int w, int h, int y0, int y1);

/* (timed as a phase, see aeg_ph_switch) */
void aeg_lowres_build_rows(aeg_lowres *l, const pixel *src, int sstride,
                           int w, int h, int y0, int y1)
{
    const int prev = aeg_ph_push(AEG_PH_LOWRES);

    aeg_lowres_build_rows_ph(l, src, sstride, w, h, y0, y1);
    aeg_ph_pop(prev);
}

static void aeg_lowres_build_rows_ph(aeg_lowres *l, const pixel *src, int sstride,
                           int w, int h, int y0, int y1)
{
    int x, y;

    if (y1 > l->h - 1)
        y1 = l->h - 1;
    for (y = y0; y <= y1; y++) {
        int y0 = 2 * y, y1 = 2 * y + 1;
        uint8_t *d = l->plane + (size_t)y * l->stride;

        if (y0 > h - 1) y0 = h - 1;
        if (y1 > h - 1) y1 = h - 1;
        for (x = 0; x < l->w; x++) {
            int x0 = 2 * x, x1 = 2 * x + 1;
            const pixel *r0 = src + (size_t)y0 * sstride;
            const pixel *r1 = src + (size_t)y1 * sstride;

            if (x0 > w - 1) x0 = w - 1;
            if (x1 > w - 1) x1 = w - 1;
            d[x] = (uint8_t)((r0[x0] + r0[x1] + r1[x0] + r1[x1] +
                              (2 << (BIT_DEPTH - 8))) >> (BIT_DEPTH - 6));
        }
        memset(d - LA_PAD, d[0], LA_PAD);
        memset(d + l->w, d[l->w - 1], LA_PAD);
    }
    for (y = 1; y <= LA_PAD; y++) {
        if (y0 == 0)
            memcpy(l->plane - LA_PAD - (size_t)y * l->stride, l->plane - LA_PAD,
                   (size_t)l->stride);
        if (y1 == l->h - 1)
            memcpy(l->plane - LA_PAD + (size_t)(l->h - 1 + y) * l->stride,
                   l->plane - LA_PAD + (size_t)(l->h - 1) * l->stride,
                   (size_t)l->stride);
    }
}
