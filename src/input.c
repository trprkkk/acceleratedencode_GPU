/*
 * Input conversion: whatever the caller has (aeg_image: planar, semi-planar
 * or grey, 4:2:0 / 4:2:2 / 4:4:4, 8 to 16 bits per sample) to the 8-bit
 * planar 4:2:0 picture the encoder codes.
 *
 * Every plane goes through a 16-bit stage (a sample of depth d is scaled by
 * 2^(16 - d), as x264's input filters do), the chroma planes are resampled
 * to 4:2:0 in that precision, and the result is dithered down to the output
 * depth with x264's Sierra-2-4A error diffusion (filters/video/depth.c),
 * reproduced exactly: the error buffer is cleared at the start of every
 * plane, and a source that was itself scaled up from the output depth comes
 * back unchanged.  8-bit 4:2:0 input does not come here at all.
 *
 * Chroma resampling, for chroma_loc_type 0 (MPEG-2/H.264's default: chroma
 * co-sited with the even luma columns, vertically half-way between rows):
 *   horizontal 2:1 (4:4:4)   (1 2 1)/4 centred on the even column;
 *   vertical 2:1, progressive (1 3 3 1)/8 centred between rows 2j, 2j+1;
 *   vertical 2:1, interlaced  within each field, (3 1)/4 for the top field
 *                             and (1 3)/4 for the bottom field -- where the
 *                             4:2:0 chroma lines of the two fields sit.
 * Edges repeat the outermost sample.
 */
#include "h264.h"

int aeg_csp_info(int csp, aeg_csp_info_t *ci)
{
    memset(ci, 0, sizeof(*ci));
    ci->planes = 3;
    ci->u = 1;
    ci->v = 2;
    ci->interleave = 1;
    switch (csp) {
    case AEG_CSP_I420: ci->xs = ci->ys = 1; break;
    case AEG_CSP_YV12: ci->xs = ci->ys = 1; ci->u = 2; ci->v = 1; break;
    case AEG_CSP_NV12: ci->xs = ci->ys = 1; ci->planes = 2; ci->interleave = 2; break;
    case AEG_CSP_NV21: ci->xs = ci->ys = 1; ci->planes = 2; ci->interleave = 2;
                       ci->swap_uv = 1; break;
    case AEG_CSP_I422: ci->xs = 1; break;
    case AEG_CSP_YV16: ci->xs = 1; ci->u = 2; ci->v = 1; break;
    case AEG_CSP_NV16: ci->xs = 1; ci->planes = 2; ci->interleave = 2; break;
    case AEG_CSP_I444: break;
    case AEG_CSP_YV24: ci->u = 2; ci->v = 1; break;
    case AEG_CSP_I400: ci->planes = 1; ci->grey = 1; break;
    default:
        return AEG_EINVAL;
    }
    return AEG_OK;
}

/* x264's dither_plane_1/_2 (filters/video/depth.c) with BIT_DEPTH = depth:
 * src holds 16-bit samples, dst gets depth-bit ones (8 here; 10 later). */
void aeg_dither_plane(uint8_t *dst, int dst_stride, const uint16_t *src,
                      int src_stride, int width, int height, int depth,
                      int16_t *errors)
{
    const int lshift = 16 - depth;
    const int rshift = 16 - depth + 2;
    const int half = 1 << (16 - depth + 1);
    const int pixel_max = (1 << depth) - 1;
    int x, y;
    memset(errors, 0, (size_t)(width + 1) * sizeof(int16_t));
    for (y = 0; y < height; y++, src += src_stride, dst += dst_stride) {
        int err = 0;
        for (x = 0; x < width; x++) {
            int v;
            err = err * 2 + errors[x] + errors[x + 1];
            v = ((src[x] << 2) + err + half) >> rshift;
            v = v < 0 ? 0 : v > pixel_max ? pixel_max : v;
            dst[x] = (uint8_t)v;
            errors[x] = (int16_t)(err = src[x] - (v << lshift));
        }
    }
}

/* One source plane (or one component of an interleaved one) to 16 bits. */
static void plane_to16(uint16_t *dst, int dst_stride, const void *src,
                       int src_stride, int pitch, int w, int h, int depth)
{
    int x, y;
    if (depth <= 8) {
        for (y = 0; y < h; y++) {
            const uint8_t *s = (const uint8_t *)src + (size_t)y * src_stride;
            uint16_t *d = dst + (size_t)y * dst_stride;
            for (x = 0; x < w; x++)
                d[x] = (uint16_t)(s[x * pitch] << 8);
        }
    } else {
        const int sh = 16 - depth, max = (1 << depth) - 1;
        for (y = 0; y < h; y++) {
            const uint16_t *s = (const uint16_t *)((const uint8_t *)src +
                                                   (size_t)y * src_stride);
            uint16_t *d = dst + (size_t)y * dst_stride;
            for (x = 0; x < w; x++) {
                int v = s[x * pitch];
                d[x] = (uint16_t)((v > max ? max : v) << sh);
            }
        }
    }
}

/* 16-bit chroma plane of sw x sh to the 4:2:0 size cw x ch. */
void aeg_chroma_to420(uint16_t *dst, int dst_stride, const uint16_t *src,
                      int src_stride, int sw, int sh, int cw, int ch,
                      int interlaced, uint16_t *tmp)
{
    int x, y;
    const uint16_t *h_out = src;
    int h_stride = src_stride;

    /* horizontal: (1 2 1)/4 on the even columns */
    if (sw != cw) {
        for (y = 0; y < sh; y++) {
            const uint16_t *s = src + (size_t)y * src_stride;
            uint16_t *d = tmp + (size_t)y * cw;
            for (x = 0; x < cw; x++) {
                int c = 2 * x, l = c > 0 ? c - 1 : 0, r = c + 1 < sw ? c + 1 : sw - 1;
                d[x] = (uint16_t)((s[l] + 2 * s[c] + s[r] + 2) >> 2);
            }
        }
        h_out = tmp;
        h_stride = cw;
    }
    if (sh == ch) {
        for (y = 0; y < ch; y++)
            memcpy(dst + (size_t)y * dst_stride, h_out + (size_t)y * h_stride,
                   (size_t)cw * sizeof(uint16_t));
        return;
    }
    for (y = 0; y < ch; y++) {
        uint16_t *d = dst + (size_t)y * dst_stride;
        if (interlaced) {
            /* output line y belongs to field f = y & 1, line m of it; the
             * field's source lines 2m and 2m+1 are frame lines
             * 4m + f and 4m + 2 + f */
            int f = y & 1, m = y >> 1;
            int r0 = 4 * m + f, r1 = 4 * m + 2 + f;
            int w0 = f ? 1 : 3, w1 = f ? 3 : 1;
            const uint16_t *a, *b;
            if (r1 >= sh) r1 = r0;
            if (r0 >= sh) r0 = r1 = sh - 1;
            a = h_out + (size_t)r0 * h_stride;
            b = h_out + (size_t)r1 * h_stride;
            for (x = 0; x < cw; x++)
                d[x] = (uint16_t)((w0 * a[x] + w1 * b[x] + 2) >> 2);
        } else {
            int r0 = 2 * y - 1, r1 = 2 * y, r2 = 2 * y + 1, r3 = 2 * y + 2;
            const uint16_t *a, *b, *c, *e;
            if (r0 < 0) r0 = 0;
            if (r2 >= sh) r2 = sh - 1;
            if (r3 >= sh) r3 = sh - 1;
            if (r1 >= sh) r1 = sh - 1;
            a = h_out + (size_t)r0 * h_stride;
            b = h_out + (size_t)r1 * h_stride;
            c = h_out + (size_t)r2 * h_stride;
            e = h_out + (size_t)r3 * h_stride;
            for (x = 0; x < cw; x++)
                d[x] = (uint16_t)((a[x] + 3 * b[x] + 3 * c[x] + e[x] + 4) >> 3);
        }
    }
}

int aeg_image_check(const aeg_image *img, int w, int h)
{
    aeg_csp_info_t ci;
    int i, bps;
    if (!img || aeg_csp_info(img->csp, &ci) != AEG_OK) {
        aeg_set_last_error("aeg_image: unknown colour space");
        return AEG_EINVAL;
    }
    if (img->depth < 8 || img->depth > 16) {
        aeg_set_last_error("aeg_image: depth must be 8..16");
        return AEG_EINVAL;
    }
    bps = img->depth > 8 ? 2 : 1;
    for (i = 0; i < ci.planes; i++) {
        int pw = i ? (w + ci.xs) >> ci.xs : w;
        if (!img->plane[i]) {
            aeg_set_last_error("aeg_image: NULL plane");
            return AEG_EINVAL;
        }
        if (img->stride[i] < pw * bps * (i ? ci.interleave : 1)) {
            aeg_set_last_error("aeg_image: stride smaller than the plane's "
                               "width");
            return AEG_EINVAL;
        }
        if (bps == 2 && ((uintptr_t)img->plane[i] & 1 || img->stride[i] & 1)) {
            aeg_set_last_error("aeg_image: 16-bit samples must be aligned");
            return AEG_EINVAL;
        }
    }
    (void)h;
    return AEG_OK;
}

int aeg_input_convert(aeg_input_cvt *c, const aeg_image *img, int w, int h,
                      int interlaced, uint8_t *dst[3], const int dst_stride[3])
{
    aeg_csp_info_t ci;
    int cw = (w + 1) / 2, ch = (h + 1) / 2, p, y;
    size_t need;

    aeg_csp_info(img->csp, &ci);
    need = (size_t)w * h;
    if (c->cap < need) {
        free(c->a);
        free(c->b);
        free(c->t);
        free(c->err);
        c->a = (uint16_t *)malloc(need * sizeof(uint16_t));
        c->b = (uint16_t *)malloc(need * sizeof(uint16_t));
        c->t = (uint16_t *)malloc(need * sizeof(uint16_t));
        c->err = (int16_t *)malloc((size_t)(w + 1) * sizeof(int16_t));
        if (!c->a || !c->b || !c->t || !c->err) {
            c->cap = 0;
            return AEG_ENOMEM;
        }
        c->cap = need;
    }
    /* luma */
    if (img->depth == 8) {
        for (y = 0; y < h; y++)
            memcpy(dst[0] + (size_t)y * dst_stride[0],
                   (const uint8_t *)img->plane[0] + (size_t)y * img->stride[0],
                   (size_t)w);
    } else {
        plane_to16(c->a, w, img->plane[0], img->stride[0], 1, w, h, img->depth);
        aeg_dither_plane(dst[0], dst_stride[0], c->a, w, w, h, 8, c->err);
    }
    /* chroma */
    for (p = 1; p <= 2; p++) {
        int sw = ci.xs ? cw : w, sh = ci.ys ? ch : h;
        const void *src;
        int sstride, pitch = 1;
        if (ci.grey) {
            for (y = 0; y < ch; y++)
                memset(dst[p] + (size_t)y * dst_stride[p], 128, (size_t)cw);
            continue;
        }
        if (ci.interleave == 2) {
            int comp = (p == 2) ^ ci.swap_uv;
            src = (const uint8_t *)img->plane[1] + comp * (img->depth > 8 ? 2 : 1);
            sstride = img->stride[1];
            pitch = 2;
        } else {
            int idx = p == 1 ? ci.u : ci.v;
            src = img->plane[idx];
            sstride = img->stride[idx];
        }
        if (img->depth == 8 && sw == cw && sh == ch) {
            /* 8-bit 4:2:0, only rearranged (NV12/NV21): exact */
            for (y = 0; y < ch; y++) {
                const uint8_t *s = (const uint8_t *)src + (size_t)y * sstride;
                uint8_t *d = dst[p] + (size_t)y * dst_stride[p];
                int x;
                for (x = 0; x < cw; x++)
                    d[x] = s[x * pitch];
            }
            continue;
        }
        plane_to16(c->a, sw, src, sstride, pitch, sw, sh, img->depth);
        aeg_chroma_to420(c->b, cw, c->a, sw, sw, sh, cw, ch, interlaced, c->t);
        aeg_dither_plane(dst[p], dst_stride[p], c->b, cw, cw, ch, 8, c->err);
    }
    return AEG_OK;
}

void aeg_input_cvt_free(aeg_input_cvt *c)
{
    free(c->a);
    free(c->b);
    free(c->t);
    free(c->err);
    memset(c, 0, sizeof(*c));
}
