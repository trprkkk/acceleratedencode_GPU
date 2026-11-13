/*
 * gpu_hints - generate x264 motion-vector hints with the aeg backend.
 *
 * The hint is a two-stage integer search:
 *   1. a 1/2-resolution batched search on the GPU (or the CPU reference),
 *   2. a +-2 full-resolution SAD refinement around 2 * the lowres vector.
 *
 * Output is the same text format x264's X264_MV_DUMP writes:
 *   frame mbx mby mvx mvy cost ref i_pixel
 * with mvx/mvy in quarter-pel units, ref = 0 and i_pixel = 1.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <time.h>
#include <stdarg.h>

#include "backend.h"
#include "aeg/aeg.h"

#define HP 32
#define REFINE 8

static void fill_pad(uint8_t *p, int stride, int w, int h, int pad)
{
    int x, y;
    for (y = 0; y < h; y++)
        for (x = 0; x < pad; x++) {
            p[(ptrdiff_t)y * stride + x] = p[(ptrdiff_t)y * stride + pad];
            p[(ptrdiff_t)y * stride + pad + w + x] =
                p[(ptrdiff_t)y * stride + pad + w - 1];
        }
    for (y = 0; y < pad; y++)
        for (x = 0; x < stride; x++) {
            p[(ptrdiff_t)y * stride + x] = p[(ptrdiff_t)pad * stride + x];
            p[(ptrdiff_t)(pad + h + y) * stride + x] =
                p[(ptrdiff_t)(pad + h - 1) * stride + x];
        }
}

static uint8_t *new_plane(int w, int h, int pad)
{
    int stride = w + 2 * pad;
    uint8_t *p = (uint8_t *)calloc((size_t)stride * (h + 2 * pad), 1);
    return p;
}

static void brec(unsigned char *b, int *n, int f, int x, int y, int mvx, int mvy, int ref)
{
    uint16_t a = (uint16_t)f, bb = (uint16_t)x, c = (uint16_t)y;
    int16_t d = (int16_t)mvx, e = (int16_t)mvy;
    unsigned char *p = b + *n;
    memcpy(p + 0, &a, 2); memcpy(p + 2, &bb, 2); memcpy(p + 4, &c, 2);
    memcpy(p + 6, &d, 2); memcpy(p + 8, &e, 2);
    p[10] = (unsigned char)ref; p[11] = 3; *n += 12;
}

static void wline(FILE *f, char *buf, size_t *len, const char *fmt, ...)
{
    va_list ap;
    int n;
    if (*len > (1u << 20) - 128) {
        fwrite(buf, 1, *len, f);
        *len = 0;
    }
    va_start(ap, fmt);
    n = vsnprintf(buf + *len, (1u << 20) - *len, fmt, ap);
    va_end(ap);
    if (n > 0) *len += (size_t)n;
}

static int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : v > hi ? hi : v;
}

int main(int argc, char **argv)
{
    const char *in = NULL, *out = NULL;
    int w = 0, h = 0, n = 0, backend = AEG_BACKEND_AUTO;
    int mbw, mbh, lw, lh, y, i, binary = 0, blen = 0;
    char *obuf = (char *)malloc(1u << 20);
    unsigned char *bbuf = NULL;
    unsigned char *allbuf = NULL;
    size_t alloff = 0;
    size_t olen = 0;
    FILE *fi, *fo;
    aeg_be_ops be;
    uint8_t *src = NULL, *ref = NULL, *lo = NULL, *loref = NULL;
    int32_t *pairs = NULL;
    int32_t *lmv = NULL, *lcost = NULL;
    int16_t *job = NULL;
    int16_t *omv = NULL;
    int32_t *osad = NULL;
    int stride, lostride;

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-i") && i + 1 < argc) in = argv[++i];
        else if (!strcmp(argv[i], "-o") && i + 1 < argc) out = argv[++i];
        else if (!strcmp(argv[i], "-w") && i + 1 < argc) w = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-h") && i + 1 < argc) h = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-b") && i + 1 < argc) {
            const char *b = argv[++i];
            backend = !strcmp(b, "cpu") ? AEG_BACKEND_CPU :
                      !strcmp(b, "cuda") ? AEG_BACKEND_CUDA :
                      !strcmp(b, "opencl") ? AEG_BACKEND_OPENCL :
                      AEG_BACKEND_AUTO;
        } else {
            fprintf(stderr, "usage: %s -i in.yuv -w W -h H -n N -o hints.txt "
                    "[-b auto|cpu|cuda|opencl]\n", argv[0]);
            return 2;
        }
    }
    if (out && strlen(out) > 4 && !strcmp(out + strlen(out) - 4, ".bin"))
        binary = 1;
    if (!in || !out || w <= 0 || h <= 0 || n <= 0) {
        fprintf(stderr, "missing arguments\n");
        return 2;
    }
    fi = fopen(in, "rb");
    if (!fi) { perror(in); return 1; }
    fo = fopen(out, "w");
    if (!fo) { perror(out); fclose(fi); return 1; }

    mbw = (w + 15) / 16;
    mbh = (h + 15) / 16;
    lw = mbw * 8;
    lh = mbh * 8;
    stride = w + 2 * HP;
    lostride = lw + 2 * AEG_LA_PAD;
    bbuf = (unsigned char *)malloc((size_t)mbw * mbh * 12);
    allbuf = (unsigned char *)malloc((size_t)n * mbw * mbh * 24);
    if (!bbuf || !allbuf) { fprintf(stderr, "oom\n"); return 1; }

    if (aeg_be_create(backend, &be) != AEG_OK)
        fprintf(stderr, "backend unavailable, using CPU\n");
    fprintf(stderr, "gpu_hints: backend=%s %s\n",
            aeg_backend_name(be.backend), be.desc);

    src = new_plane(w, mbh * 16, HP);
    ref = new_plane(w, mbh * 16, HP);
    lo = new_plane(lw, lh, AEG_LA_PAD);
    loref = new_plane(lw, lh, AEG_LA_PAD);
    if (!src || !ref || !lo || !loref) { fprintf(stderr, "oom\n"); return 1; }
    pairs = (int32_t *)malloc(2 * sizeof(int32_t));
    lmv = (int32_t *)malloc((size_t)mbw * mbh * 2 * sizeof(int32_t));
    lcost = (int32_t *)malloc((size_t)mbw * mbh * sizeof(int32_t));
    job = (int16_t *)malloc((size_t)mbw * mbh * 4 * sizeof(int16_t));
    omv = (int16_t *)malloc((size_t)mbw * mbh * 2 * sizeof(int16_t));
    osad = (int32_t *)malloc((size_t)mbw * mbh * sizeof(int32_t));
    if (!pairs || !lmv || !lcost || !job || !omv || !osad) { fprintf(stderr, "oom\n"); return 1; }

    double t_read = 0, t_lo = 0, t_me = 0;
    for (int f = 0; f < n; f++) {
        int ret;
        struct timespec ts0, ts1;
        clock_gettime(CLOCK_MONOTONIC, &ts0);
        size_t got = fread(src + (ptrdiff_t)HP * stride + HP, 1,
                           (size_t)w * h, fi);
        if (got != (size_t)w * h) break;
        /* skip chroma */
        if (fseek(fi, (long)w * h / 2, SEEK_CUR) != 0) break;
        for (y = h; y < mbh * 16; y++)
            memcpy(src + (ptrdiff_t)(HP + y) * stride + HP,
                   src + (ptrdiff_t)(HP + h - 1) * stride + HP, w);
        fill_pad(src, stride, w, mbh * 16, HP);
        clock_gettime(CLOCK_MONOTONIC, &ts1);
        t_read += ts1.tv_sec - ts0.tv_sec + 1e-9 * (ts1.tv_nsec - ts0.tv_nsec);
        clock_gettime(CLOCK_MONOTONIC, &ts0);

        /* half-resolution luma */
        for (y = 0; y < lh; y++)
            for (i = 0; i < lw; i++) {
                int x0 = clampi(2 * i, 0, w - 1), y0 = clampi(2 * y, 0, h - 1);
                int x1 = clampi(2 * i + 1, 0, w - 1), y1 = clampi(2 * y + 1, 0, h - 1);
                const uint8_t *a = src + (ptrdiff_t)(HP + y0) * stride + HP;
                const uint8_t *b = src + (ptrdiff_t)(HP + y1) * stride + HP;
                lo[(ptrdiff_t)(AEG_LA_PAD + y) * lostride + AEG_LA_PAD + i] =
                    (uint8_t)((a[x0] + a[x1] + b[x0] + b[x1] + 2) >> 2);
            }
        fill_pad(lo, lostride, lw, lh, AEG_LA_PAD);

        if (be.lowres_upload)
            be.lowres_upload(be.priv, 0, lo + (ptrdiff_t)AEG_LA_PAD * lostride + AEG_LA_PAD,
                             lostride, lw, lh, (uint64_t)f);
        if (f > 0 && be.lowres_upload)
            be.lowres_upload(be.priv, 1,
                             loref + (ptrdiff_t)AEG_LA_PAD * lostride + AEG_LA_PAD,
                             lostride, lw, lh, (uint64_t)f - 1);

        for (i = 0; i < mbw * mbh * 2; i++) lmv[i] = 0;
        if (f > 0 && be.lowres_batch_search) {
            pairs[0] = 0; pairs[1] = 1;
            ret = be.lowres_batch_search(be.priv, pairs, 1,
                                         mbw, mbh, 10, lmv, lcost);
            if (ret != AEG_OK) fprintf(stderr, "lowres_search failed %d\n", ret);
        }

        clock_gettime(CLOCK_MONOTONIC, &ts1);
        t_lo += ts1.tv_sec - ts0.tv_sec + 1e-9 * (ts1.tv_nsec - ts0.tv_nsec);
        clock_gettime(CLOCK_MONOTONIC, &ts0);

        /* full-resolution refine in bands: one launch per band */
        if (f > 0 && be.me_batch) {
            int y0;
            for (y0 = 0; y0 < mbh; y0 += 4) {
                int y1 = y0 + 4 < mbh ? y0 + 4 : mbh;
                int nx = 0, yy;
                for (yy = y0; yy < y1; yy++)
                    for (i = 0; i < mbw; i++) {
                        int sx = i * 16, sy = (yy - y0) * 16;
                        int mx = lmv[2 * (yy * mbw + i)] * 2;
                        int my = lmv[2 * (yy * mbw + i) + 1] * 2;
                        job[4 * nx + 0] = (int16_t)sx;
                        job[4 * nx + 1] = (int16_t)sy;
                        job[4 * nx + 2] = (int16_t)mx;
                        job[4 * nx + 3] = (int16_t)my;
                        nx++;
                    }
                be.me_batch(be.priv,
                            src + (ptrdiff_t)(HP + y0 * 16) * stride + HP, stride,
                            ref + (ptrdiff_t)(HP + y0 * 16) * stride + HP, stride,
                            16, 16, REFINE, job, nx,
                            (y1 - y0) * 16 + 2 * HP, omv, osad,
                            ~(uint64_t)0, ~(uint64_t)0);
                for (yy = y0; yy < y1; yy++)
                    for (i = 0; i < mbw; i++) {
                        int k = (yy - y0) * mbw + i;
                        int mvx = (job[4 * k + 2] + omv[2 * k]) * 4;
                        int mvy = (job[4 * k + 3] + omv[2 * k + 1]) * 4;
                        if (binary) { brec(allbuf, (int *)&alloff, f, i, yy, mvx, mvy, 0); brec(allbuf, (int *)&alloff, f, i, yy, mvx, mvy, 1); }
                        else wline(fo, obuf, &olen, "%d %d %d %d %d %d 0 3\n",
                                   f, i, yy, mvx, mvy, osad[k]);
                    }
            }
        } else {
            for (y = 0; y < mbh; y++)
                for (i = 0; i < mbw; i++)
                    if (binary) { brec(allbuf, (int *)&alloff, f, i, y, 0, 0, 0); brec(allbuf, (int *)&alloff, f, i, y, 0, 0, 1); }
                    else wline(fo, obuf, &olen, "%d %d %d 0 0 0 0 3\n", f, i, y);
        }

        clock_gettime(CLOCK_MONOTONIC, &ts1);
        t_me += ts1.tv_sec - ts0.tv_sec + 1e-9 * (ts1.tv_nsec - ts0.tv_nsec);
        clock_gettime(CLOCK_MONOTONIC, &ts0);

        /* current becomes the reference for the next frame */
        memcpy(ref, src, (size_t)stride * (mbh * 16 + 2 * HP));
        memcpy(loref, lo, (size_t)lostride * (lh + 2 * AEG_LA_PAD));
    }

    if (getenv("GPU_HINTS_DEBUG"))
        fprintf(stderr, "gpu_hints: read=%.3f lowres=%.3f refine+out=%.3f\n",
                t_read, t_lo, t_me);
    if (olen) fwrite(obuf, 1, olen, fo);
    if (binary && alloff) fwrite(allbuf, 1, alloff, fo);
    free(obuf); free(bbuf); free(allbuf);
    free(src); free(ref); free(lo); free(loref);
    free(pairs); free(lmv); free(lcost); free(job); free(omv); free(osad);
    fclose(fi); fclose(fo);
    return 0;
}
