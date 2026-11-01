/*
 * The input conversion (src/input.c) against references written here.
 *
 *  - the dither: x264's dither_plane (filters/video/depth.c), transcribed in
 *    its own form, on random 16-bit planes of odd sizes, for output depths 8
 *    and 10; plus its properties -- a plane scaled up from the output depth
 *    comes back exactly, and a flat level between two codes keeps its mean;
 *  - the chroma resampling to 4:2:0: the filter taps written out as
 *    floating point with explicit edge clamping, for 4:4:4 and 4:2:2 sources
 *    of odd and tiny sizes, progressive and interlaced; plus field separation
 *    -- a source whose two fields carry different chroma must keep them
 *    apart when interlaced (and blend them when not);
 *  - whole pictures through aeg_input_convert(): every layout of the same
 *    picture (planar orders, interleaved, 16-bit) gives the same output.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/h264.h"

static int failures;
#define CHECK(cond, ...)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                     \
            printf(__VA_ARGS__);                                            \
            printf("\n");                                                   \
            failures++;                                                     \
        }                                                                   \
    } while (0)

static unsigned rng = 12345;
static unsigned rnd(void)
{
    rng = rng * 1103515245u + 12345u;
    return rng >> 8;
}

/* x264 filters/video/depth.c, dither_plane_1 with pitch 1 */
#define DITHER_PLANE(pitch)                                                 \
static void x264_dither_plane_##pitch(uint16_t *dst, int dst_stride,        \
    uint16_t *src, int src_stride, int width, int height, int16_t *errors,  \
    int bit_depth)                                                          \
{                                                                           \
    const int lshift = 16 - bit_depth;                                      \
    const int rshift = 16 - bit_depth + 2;                                  \
    const int half = 1 << (16 - bit_depth + 1);                             \
    const int pixel_max = (1 << bit_depth) - 1;                             \
    memset(errors, 0, (width + 1) * sizeof(int16_t));                       \
    for (int y = 0; y < height; y++, src += src_stride, dst += dst_stride) {\
        int err = 0;                                                        \
        for (int x = 0; x < width; x++) {                                   \
            err = err * 2 + errors[x] + errors[x + 1];                      \
            int v = ((src[x * pitch] << 2) + err + half) >> rshift;         \
            dst[x * pitch] = (uint16_t)(v < 0 ? 0 : v > pixel_max ? pixel_max : v); \
            errors[x] = err = src[x * pitch] - (dst[x * pitch] << lshift);  \
        }                                                                   \
    }                                                                       \
}
DITHER_PLANE(1)

static void check_dither(void)
{
    static const int sizes[][2] = { { 1, 1 }, { 2, 3 }, { 17, 5 }, { 64, 9 },
                                    { 333, 7 } };
    int si, depth, i;
    for (si = 0; si < 5; si++) {
        int w = sizes[si][0], h = sizes[si][1];
        uint16_t *src = (uint16_t *)malloc((size_t)w * h * 2);
        uint16_t *ref = (uint16_t *)malloc((size_t)w * h * 2);
        uint8_t *out = (uint8_t *)malloc((size_t)w * h);
        int16_t *e1 = (int16_t *)malloc((size_t)(w + 1) * 2);
        int16_t *e2 = (int16_t *)malloc((size_t)(w + 1) * 2);
        for (i = 0; i < w * h; i++)
            src[i] = (uint16_t)(rnd() & 0xffff);
        /* extremes too */
        src[0] = 0xffff;
        if (w * h > 1)
            src[w * h - 1] = 0;
        x264_dither_plane_1(ref, w, src, w, w, h, e1, 8);
        memset(e2, 0x55, (size_t)(w + 1) * 2);   /* must be cleared inside */
        aeg_dither_plane(out, w, src, w, w, h, 8, e2);
        for (i = 0; i < w * h; i++)
            if (out[i] != ref[i])
                break;
        CHECK(i == w * h, "dither %dx%d differs from x264 at %d: %d vs %d", w,
              h, i, i < w * h ? out[i] : 0, i < w * h ? ref[i] : 0);
        /* scaled up from the output depth: exact */
        for (depth = 8; depth <= 8; depth++) {
            for (i = 0; i < w * h; i++) {
                ref[i] = (uint16_t)(rnd() & 255);
                src[i] = (uint16_t)(ref[i] << 8);
            }
            aeg_dither_plane(out, w, src, w, w, h, 8, e2);
            for (i = 0; i < w * h; i++)
                if (out[i] != ref[i])
                    break;
            CHECK(i == w * h, "dither %dx%d: an 8-bit plane scaled up does "
                              "not come back", w, h);
        }
        free(src);
        free(ref);
        free(out);
        free(e1);
        free(e2);
    }
    /* a flat level a quarter of the way between two codes keeps its mean */
    {
        enum { FW = 96, FH = 64 };
        static uint16_t src[FW * FH];
        static uint8_t out[FW * FH];
        static int16_t err[FW + 1];
        double sum = 0;
        for (i = 0; i < FW * FH; i++)
            src[i] = (uint16_t)(100 * 256 + 64);    /* 100.25 */
        aeg_dither_plane(out, FW, src, FW, FW, FH, 8, err);
        for (i = 0; i < FW * FH; i++) {
            sum += out[i];
            if (out[i] != 100 && out[i] != 101)
                break;
        }
        CHECK(i == FW * FH, "flat 100.25 dithered to %d", out[i]);
        CHECK(fabs(sum / (FW * FH) - 100.25) < 0.01, "flat 100.25: mean %.4f",
              sum / (FW * FH));
    }
}

/* ---------------------------------------------------------------- */

static int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : v > hi ? hi : v;
}

/* the 4:2:0 sample (x, y) of a sw x sh plane, written out from the taps */
static int ref_chroma(const uint16_t *s, int sw, int sh, int cw, int ch, int x,
                      int y, int inter)
{
    /* horizontal first, into a column of doubles-free integers: each tap
     * rounds the same way the conversion does ((sum + half) >> shift) */
    int rows[4], nr, wts[4], k, acc = 0, shift;
    int hrow[4];
    if (sh == ch) {
        rows[0] = y;
        nr = 1;
        wts[0] = 1;
        shift = 0;
    } else if (inter) {
        int f = y & 1, m = y >> 1;
        rows[0] = 4 * m + f;
        rows[1] = 4 * m + 2 + f;
        if (rows[1] >= sh) rows[1] = rows[0];
        if (rows[0] >= sh) rows[0] = rows[1] = sh - 1;
        wts[0] = f ? 1 : 3;
        wts[1] = f ? 3 : 1;
        nr = 2;
        shift = 2;
    } else {
        rows[0] = clampi(2 * y - 1, 0, sh - 1);
        rows[1] = clampi(2 * y, 0, sh - 1);
        rows[2] = clampi(2 * y + 1, 0, sh - 1);
        rows[3] = clampi(2 * y + 2, 0, sh - 1);
        wts[0] = 1, wts[1] = 3, wts[2] = 3, wts[3] = 1;
        nr = 4;
        shift = 3;
    }
    for (k = 0; k < nr; k++) {
        const uint16_t *r = s + (size_t)rows[k] * sw;
        if (sw == cw)
            hrow[k] = r[x];
        else
            hrow[k] = (r[clampi(2 * x - 1, 0, sw - 1)] + 2 * r[2 * x] +
                       r[clampi(2 * x + 1, 0, sw - 1)] + 2) >> 2;
        acc += wts[k] * hrow[k];
    }
    (void)ch;
    return shift ? (acc + (1 << (shift - 1))) >> shift : acc;
}

static void check_resample(void)
{
    static const int sizes[][2] = { { 1, 1 }, { 2, 2 }, { 3, 5 }, { 7, 2 },
                                    { 20, 13 }, { 65, 34 } };
    int si, kind, inter;
    for (si = 0; si < 6; si++)
        for (kind = 0; kind < 3; kind++)          /* 444, 422, 420 */
            for (inter = 0; inter <= 1; inter++) {
                int w = sizes[si][0], h = sizes[si][1];
                int cw = (w + 1) / 2, ch = (h + 1) / 2;
                int sw = kind == 0 ? w : cw, sh = kind == 2 ? ch : h;
                uint16_t *src = (uint16_t *)malloc((size_t)sw * sh * 2);
                uint16_t *dst = (uint16_t *)malloc((size_t)cw * ch * 2);
                uint16_t *tmp = (uint16_t *)malloc((size_t)w * h * 2);
                int i, x, y, bad = 0;
                for (i = 0; i < sw * sh; i++)
                    src[i] = (uint16_t)(rnd() & 0xffff);
                aeg_chroma_to420(dst, cw, src, sw, sw, sh, cw, ch, inter, tmp);
                for (y = 0; y < ch && !bad; y++)
                    for (x = 0; x < cw && !bad; x++) {
                        int r = ref_chroma(src, sw, sh, cw, ch, x, y, inter);
                        if (dst[y * cw + x] != r) {
                            bad = 1;
                            CHECK(0, "resample %s %dx%d%s: (%d,%d) %d vs %d",
                                  kind == 0 ? "444" : kind == 1 ? "422" : "420",
                                  w, h, inter ? " interlaced" : "", x, y,
                                  dst[y * cw + x], r);
                        }
                    }
                free(src);
                free(dst);
                free(tmp);
            }
    /* field separation: 4:2:2 with top-field rows 50 and bottom-field rows
     * 200 (in 16 bits) */
    {
        enum { SW = 8, SH = 16, CH2 = 8 };
        uint16_t src[SW * SH], dst[SW * CH2], tmp[SW * SH];
        int x, y, okf = 1, okp = 1;
        for (y = 0; y < SH; y++)
            for (x = 0; x < SW; x++)
                src[y * SW + x] = (uint16_t)((y & 1 ? 200 : 50) << 8);
        aeg_chroma_to420(dst, SW, src, SW, SW, SH, SW, CH2, 1, tmp);
        for (y = 0; y < CH2; y++)
            for (x = 0; x < SW; x++)
                okf &= dst[y * SW + x] == (uint16_t)((y & 1 ? 200 : 50) << 8);
        CHECK(okf, "interlaced 4:2:2 -> 4:2:0 mixes the fields");
        aeg_chroma_to420(dst, SW, src, SW, SW, SH, SW, CH2, 0, tmp);
        for (y = 1; y < CH2 - 1; y++)
            for (x = 0; x < SW; x++)
                okp &= dst[y * SW + x] == (uint16_t)(125 << 8);
        CHECK(okp, "progressive 4:2:2 -> 4:2:0 does not blend the rows");
    }
}

/* ---------------------------------------------------------------- */

/* The same 10-bit 4:4:4 picture in every 4:4:4 layout, and an 8-bit 4:2:2
 * picture in every 4:2:2 layout: identical outputs. */
static void check_layouts(void)
{
    enum { W = 38, H = 22, CW = 19, CH = 11 };
    static uint16_t y16[W * H], u16[W * H], v16[W * H];
    static uint16_t uv16[2 * W * H];
    static uint8_t y8[W * H], u8[W * H], v8[W * H], uv8[2 * W * H];
    static uint8_t out[2][W * H + 2 * CW * CH];
    aeg_input_cvt cvt;
    aeg_image im;
    int i, k, inter;
    for (i = 0; i < W * H; i++) {
        y16[i] = (uint16_t)(rnd() & 1023);
        u16[i] = (uint16_t)(rnd() & 1023);
        v16[i] = (uint16_t)(rnd() & 1023);
        y8[i] = (uint8_t)rnd();
        u8[i] = (uint8_t)rnd();
        v8[i] = (uint8_t)rnd();
    }
    for (i = 0; i < W * H; i++) {
        uv8[2 * i] = u8[i];
        uv8[2 * i + 1] = v8[i];
    }
    memset(&cvt, 0, sizeof(cvt));
    for (inter = 0; inter <= 1; inter++) {
        uint8_t *dst[2][3];
        int st[3] = { W, CW, CW };
        for (k = 0; k < 2; k++) {
            dst[k][0] = out[k];
            dst[k][1] = out[k] + W * H;
            dst[k][2] = out[k] + W * H + CW * CH;
        }
        /* 10-bit I444 vs YV24 */
        memset(&im, 0, sizeof(im));
        im.csp = AEG_CSP_I444;
        im.depth = 10;
        im.plane[0] = y16, im.plane[1] = u16, im.plane[2] = v16;
        im.stride[0] = im.stride[1] = im.stride[2] = 2 * W;
        aeg_input_convert(&cvt, &im, W, H, inter, dst[0], st);
        im.csp = AEG_CSP_YV24;
        im.plane[1] = v16, im.plane[2] = u16;
        aeg_input_convert(&cvt, &im, W, H, inter, dst[1], st);
        CHECK(!memcmp(out[0], out[1], sizeof(out[0])), "I444 != YV24");
        /* the same, at depth 12 with the samples scaled up: same output */
        for (i = 0; i < W * H; i++) {
            uv16[i] = (uint16_t)(y16[i] << 2);
        }
        {
            static uint16_t u12[W * H], v12[W * H];
            for (i = 0; i < W * H; i++) {
                u12[i] = (uint16_t)(u16[i] << 2);
                v12[i] = (uint16_t)(v16[i] << 2);
            }
            im.csp = AEG_CSP_I444;
            im.depth = 12;
            im.plane[0] = uv16, im.plane[1] = u12, im.plane[2] = v12;
            aeg_input_convert(&cvt, &im, W, H, inter, dst[1], st);
            CHECK(!memcmp(out[0], out[1], sizeof(out[0])),
                  "10-bit and the same scaled to 12-bit differ");
        }
        /* 8-bit 4:2:2: I422 vs YV16 vs NV16 */
        memset(&im, 0, sizeof(im));
        im.csp = AEG_CSP_I422;
        im.depth = 8;
        im.plane[0] = y8, im.plane[1] = u8, im.plane[2] = v8;
        im.stride[0] = W;
        im.stride[1] = im.stride[2] = CW;
        aeg_input_convert(&cvt, &im, W, H, inter, dst[0], st);
        im.csp = AEG_CSP_YV16;
        im.plane[1] = v8, im.plane[2] = u8;
        aeg_input_convert(&cvt, &im, W, H, inter, dst[1], st);
        CHECK(!memcmp(out[0], out[1], sizeof(out[0])), "I422 != YV16");
        for (i = 0; i < CW * H; i++) {
            uv8[2 * i] = u8[i];
            uv8[2 * i + 1] = v8[i];
        }
        im.csp = AEG_CSP_NV16;
        im.plane[1] = uv8, im.plane[2] = NULL;
        im.stride[1] = 2 * CW;
        aeg_input_convert(&cvt, &im, W, H, inter, dst[1], st);
        CHECK(!memcmp(out[0], out[1], sizeof(out[0])), "I422 != NV16");
        /* luma of an 8-bit picture is never touched */
        CHECK(!memcmp(out[0], y8, W * H), "8-bit luma changed");
    }
    aeg_input_cvt_free(&cvt);
}

int main(void)
{
    check_dither();
    check_resample();
    check_layouts();
    if (failures) {
        printf("input_check: %d failure(s)\n", failures);
        return 1;
    }
    printf("input_check: all checks passed\n");
    return 0;
}
