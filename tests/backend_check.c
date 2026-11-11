/*
 * The device backends, end to end, against the CPU reference.
 *
 * tests/kernel_check.c runs the kernel *source* as C, so it proves the kernels'
 * arithmetic on a machine with no device.  What it cannot see is the host side
 * of every operation: which bytes are transferred, from where, how the
 * positions are laid out, what comes back and where it is written.  Those
 * wrappers are where the device paths were actually broken -- sad_batch()
 * transferred one packed block while the encoder sends a whole picture's
 * candidates, interp_plane() transferred from the picture origin while its
 * kernel reads and writes the padding -- and no test noticed, because none ran
 * on a device.
 *
 * This one creates a real backend through aeg_be_create() -- the guarded
 * operations the encoder calls -- and compares every operation with the CPU
 * implementation on encoder-shaped data: padded pictures allocated exactly as
 * the encoder allocates them, whole-picture candidate lists, vectors reaching
 * into the padding.  Then it does the same from two threads at once, which is
 * how slice threads use the one device context, including the case where the
 * threads register different reference pictures.
 *
 * Any OpenCL implementation runs it, a CPU one included (PoCL: see
 * tools/pocl_env.sh).  Without a device it says so and passes.
 *
 *   backend_check [opencl|cuda|auto]      (default auto)
 */
#define _POSIX_C_SOURCE 200809L
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "backend.h"

static uint32_t rng_state = 12345u;

static int rnd(int n)
{
    rng_state = rng_state * 1103515245u + 12345u;
    return (int)((rng_state >> 16) % (uint32_t)n);
}

static aeg_be_ops dev, cpu;
static int fails, checks;

static void check(int ok, const char *what)
{
    checks++;
    if (!ok) {
        printf("   backend_check: %s differs from the CPU reference\n", what);
        fails++;
    }
}

/* A picture as the encoder keeps it: stride w + 2 * AEG_PAD, h + 2 * AEG_PAD
 * rows, the origin AEG_PAD rows and columns in, the padding replicated from
 * the edges -- and the allocation exactly that large, so a transfer that reads
 * past it is caught by a sanitizer build. */
typedef struct {
    uint8_t *mem, *org;
    int w, h, stride;
    size_t bytes;
} pic;

static void pic_alloc(pic *p, int w, int h)
{
    p->w = w;
    p->h = h;
    p->stride = w + 2 * AEG_PAD;
    p->bytes = (size_t)p->stride * (size_t)(h + 2 * AEG_PAD);
    p->mem = (uint8_t *)malloc(p->bytes);
    memset(p->mem, 0, p->bytes);
    p->org = p->mem + (size_t)AEG_PAD * p->stride + AEG_PAD;
}

static void pic_fill(pic *p, int smooth)
{
    int x, y;
    for (y = 0; y < p->h; y++)
        for (x = 0; x < p->w; x++) {
            int v = smooth ? (x * 3 + y * 2 + rnd(24)) & 255 : rnd(256);
            p->org[(size_t)y * p->stride + x] = (uint8_t)v;
        }
    for (y = -AEG_PAD; y < p->h + AEG_PAD; y++)
        for (x = -AEG_PAD; x < p->w + AEG_PAD; x++) {
            int sy = y < 0 ? 0 : (y >= p->h ? p->h - 1 : y);
            int sx = x < 0 ? 0 : (x >= p->w ? p->w - 1 : x);
            p->org[(ptrdiff_t)y * p->stride + x] =
                p->org[(ptrdiff_t)sy * p->stride + sx];
        }
}

static void pic_free(pic *p)
{
    free(p->mem);
    p->mem = p->org = NULL;
}

/* ---------------------------------------------------------------- */
static void check_coef(void)
{
    static const int qps[] = { 0, 12, 28, 40, 51 };
    static const int nblks[] = { 1, 16, 64, 16 * 9 };
    int qi, ni, intra;
    for (qi = 0; qi < 5; qi++)
        for (ni = 0; ni < 4; ni++)
            for (intra = 0; intra < 2; intra++) {
                int n = nblks[ni], i;
                int16_t *res = (int16_t *)malloc((size_t)n * 32);
                int16_t *l0 = (int16_t *)malloc((size_t)n * 32);
                int16_t *l1 = (int16_t *)malloc((size_t)n * 32);
                int16_t *r0 = (int16_t *)malloc((size_t)n * 32);
                int16_t *r1 = (int16_t *)malloc((size_t)n * 32);
                int32_t *z0 = (int32_t *)malloc((size_t)n * 4);
                int32_t *z1 = (int32_t *)malloc((size_t)n * 4);
                for (i = 0; i < n * 16; i++)
                    res[i] = (int16_t)(rnd(4) ? rnd(511) - 255 : rnd(17) - 8);
                cpu.tq_batch(cpu.priv, res, n, qps[qi], intra, l0, z0);
                dev.tq_batch(dev.priv, res, n, qps[qi], intra, l1, z1);
                check(!memcmp(l0, l1, (size_t)n * 32) &&
                          !memcmp(z0, z1, (size_t)n * 4),
                      "tq_batch");
                cpu.iq_idct_batch(cpu.priv, l0, n, qps[qi], r0);
                dev.iq_idct_batch(dev.priv, l0, n, qps[qi], r1);
                check(!memcmp(r0, r1, (size_t)n * 32), "iq_idct_batch");
                /* the fused operation: tq_batch's levels and counts and
                 * iq_idct_batch's residual of them, from one call */
                check(dev.tq_iq_batch != NULL, "tq_iq_batch (present)");
                if (dev.tq_iq_batch) {
                    memset(l1, 0x55, (size_t)n * 32);
                    memset(r1, 0x55, (size_t)n * 32);
                    memset(z1, 0x55, (size_t)n * 4);
                    dev.tq_iq_batch(dev.priv, res, n, qps[qi], intra, l1, z1,
                                    r1);
                    check(!memcmp(l0, l1, (size_t)n * 32) &&
                              !memcmp(z0, z1, (size_t)n * 4) &&
                              !memcmp(r0, r1, (size_t)n * 32),
                          "tq_iq_batch");
                }
                /* the DC-split pair (Intra_16x16, chroma) */
                check(dev.tq_dc_batch && dev.iq_dc_batch,
                      "tq_dc_batch / iq_dc_batch (present)");
                if (dev.tq_dc_batch && dev.iq_dc_batch) {
                    int32_t *d0 = (int32_t *)malloc((size_t)n * 4);
                    int32_t *d1 = (int32_t *)malloc((size_t)n * 4);
                    memset(l1, 0x55, (size_t)n * 32);
                    memset(z1, 0x55, (size_t)n * 4);
                    memset(d1, 0x55, (size_t)n * 4);
                    cpu.tq_dc_batch(cpu.priv, res, n, qps[qi], intra, l0, z0, d0);
                    dev.tq_dc_batch(dev.priv, res, n, qps[qi], intra, l1, z1, d1);
                    check(!memcmp(l0, l1, (size_t)n * 32) &&
                              !memcmp(z0, z1, (size_t)n * 4) &&
                              !memcmp(d0, d1, (size_t)n * 4),
                          "tq_dc_batch");
                    for (i = 0; i < n; i++)
                        d0[i] = rnd(4) ? rnd(8001) - 4000 : rnd(80001) - 40000;
                    memset(r1, 0x55, (size_t)n * 32);
                    cpu.iq_dc_batch(cpu.priv, l0, d0, n, qps[qi], r0);
                    dev.iq_dc_batch(dev.priv, l0, d0, n, qps[qi], r1);
                    check(!memcmp(r0, r1, (size_t)n * 32), "iq_dc_batch");
                    free(d0); free(d1);
                }
                free(res); free(l0); free(l1); free(r0); free(r1);
                free(z0); free(z1);
            }
}

/* the motion search's integer rounds: every macroblock of the picture, nine
 * candidates each, displacements reaching into the padding on all sides */
static void check_sad(int w, int h)
{
    pic cur, ref;
    int mbw = w / 16, mbh = h / 16, n = mbw * mbh * 9, i;
    int16_t *mv = (int16_t *)malloc((size_t)n * 8);
    int32_t *o0 = (int32_t *)malloc((size_t)n * 4);
    int32_t *o1 = (int32_t *)malloc((size_t)n * 4);
    const int range = AEG_PAD - 16;

    pic_alloc(&cur, w, h);
    pic_alloc(&ref, w, h);
    pic_fill(&cur, 1);
    pic_fill(&ref, 1);
    for (i = 0; i < n; i++) {
        int mb = i / 9, ox = (mb % mbw) * 16, oy = (mb / mbw) * 16;
        mv[4 * i + 0] = (int16_t)ox;
        mv[4 * i + 1] = (int16_t)oy;
        mv[4 * i + 2] = (int16_t)(ox + rnd(2 * range + 1) - range);
        mv[4 * i + 3] = (int16_t)(oy + rnd(2 * range + 1) - range);
    }
    cpu.sad_batch(cpu.priv, cur.org, cur.stride, ref.org, ref.stride, 16, 16,
                  mv, n, o0);
    dev.sad_batch(dev.priv, cur.org, cur.stride, ref.org, ref.stride, 16, 16,
                  mv, n, o1);
    check(!memcmp(o0, o1, (size_t)n * 4), "sad_batch (whole picture)");
    /* one candidate in the bottom-right corner, the smallest span there is */
    mv[0] = (int16_t)(w - 16);
    mv[1] = (int16_t)(h - 16);
    mv[2] = (int16_t)(w - 16 + range);
    mv[3] = (int16_t)(h - 16 + range);
    cpu.sad_batch(cpu.priv, cur.org, cur.stride, ref.org, ref.stride, 16, 16,
                  mv, 1, o0);
    dev.sad_batch(dev.priv, cur.org, cur.stride, ref.org, ref.stride, 16, 16,
                  mv, 1, o1);
    check(o0[0] == o1[0], "sad_batch (one candidate)");
    pic_free(&cur);
    pic_free(&ref);
    free(mv); free(o0); free(o1);
}

/* the three fractional planes, compared over the whole allocation: the
 * padding is part of what motion compensation reads */
static void check_interp(int w, int h, pic planes[4])
{
    int which;
    pic_alloc(&planes[0], w, h);
    pic_fill(&planes[0], 0);
    for (which = 1; which <= 3; which++) {
        pic d0, d1;
        pic_alloc(&d0, w, h);
        pic_alloc(&d1, w, h);
        memset(d1.mem, 0xa5, d1.bytes);
        cpu.interp_plane(cpu.priv, planes[0].org, planes[0].stride, w, h, which,
                         d0.org, d0.stride);
        dev.interp_plane(dev.priv, planes[0].org, planes[0].stride, w, h, which,
                         d1.org, d1.stride);
        check(!memcmp(d0.mem, d1.mem, d0.bytes), "interp_plane");
        planes[which] = d0;
        pic_free(&d1);
    }
}

static void check_sse(int w, int h)
{
    pic a, b;
    uint64_t s0 = 0, s1 = 1;
    pic_alloc(&a, w, h);
    pic_alloc(&b, w, h);
    pic_fill(&a, 0);
    pic_fill(&b, 0);
    cpu.sse_batch(cpu.priv, a.org, a.stride, b.org, b.stride, w, h, &s0);
    dev.sse_batch(dev.priv, a.org, a.stride, b.org, b.stride, w, h, &s1);
    check(s0 == s1, "sse_batch");
    pic_free(&a);
    pic_free(&b);
}

/* the fractional rounds: quarter-pel candidates of every macroblock against
 * the registered reference */
static int16_t *satd_positions(int w, int h, int bs, int *count)
{
    int nbw = w / bs, nbh = h / bs, n = nbw * nbh * 8, i;
    const int range4 = (AEG_PAD - 24) * 4;
    int16_t *pos = (int16_t *)malloc((size_t)n * 8);
    for (i = 0; i < n; i++) {
        int b = i / 8;
        pos[4 * i + 0] = (int16_t)((b % nbw) * bs);
        pos[4 * i + 1] = (int16_t)((b / nbw) * bs);
        pos[4 * i + 2] = (int16_t)(rnd(2 * range4 + 1) - range4);
        pos[4 * i + 3] = (int16_t)(rnd(2 * range4 + 1) - range4);
    }
    *count = n;
    return pos;
}

static int satd_agree_on(const aeg_be_ops *d, const pic *cur, pic planes[4],
                         int bs, int upload)
{
    const uint8_t *pl[4] = { planes[0].org, planes[1].org, planes[2].org,
                             planes[3].org };
    int n, ok;
    int16_t *pos = satd_positions(cur->w, cur->h, bs, &n);
    int32_t *o0 = (int32_t *)malloc((size_t)n * 4);
    int32_t *o1 = (int32_t *)malloc((size_t)n * 4);
    if (upload)
        d->ref_upload(d->priv, pl, planes[0].stride, cur->w, cur->h);
    cpu.ref_upload(cpu.priv, pl, planes[0].stride, cur->w, cur->h);
    cpu.satd_batch(cpu.priv, cur->org, cur->stride, bs, bs, pos, n, o0);
    d->satd_batch(d->priv, cur->org, cur->stride, bs, bs, pos, n, o1);
    ok = !memcmp(o0, o1, (size_t)n * 4);
    free(pos); free(o0); free(o1);
    return ok;
}

static int satd_agree(const pic *cur, pic planes[4], int bs, int upload)
{
    return satd_agree_on(&dev, cur, planes, bs, upload);
}

/* ---------------------------------------------------------------- */
/* Two threads, one device                                           */
/* ---------------------------------------------------------------- */
typedef struct {
    pic cur, planes[4];
    int id, bad;
    aeg_be_ops ops;             /* the device as this thread sees it */
} worker;

static pthread_barrier_t g_bar;

/* lock-step: A registers, B registers, A searches, B searches -- so A's batch
 * runs while B's reference is the resident one */
static void *lockstep(void *arg)
{
    worker *wk = (worker *)arg;
    const uint8_t *pl[4] = { wk->planes[0].org, wk->planes[1].org,
                             wk->planes[2].org, wk->planes[3].org };
    int round;
    for (round = 0; round < 4; round++) {
        int step;
        for (step = 0; step < 4; step++) {
            if (step == wk->id)
                wk->ops.ref_upload(wk->ops.priv, pl, wk->planes[0].stride,
                                   wk->cur.w, wk->cur.h);
            if (step == 2 + wk->id &&
                !satd_agree_on(&wk->ops, &wk->cur, wk->planes, 16, 0))
                wk->bad = 1;
            pthread_barrier_wait(&g_bar);
        }
    }
    return NULL;
}

/* free-running: both threads hammer the coefficient batches */
static void *hammer(void *arg)
{
    worker *wk = (worker *)arg;
    uint32_t s = 777u + (uint32_t)wk->id;
    int it, i;
    int16_t res[64 * 16], l0[64 * 16], l1[64 * 16], r0[64 * 16], r1[64 * 16];
    int32_t z0[64], z1[64];
    for (it = 0; it < 200; it++) {
        int qp = it % 52;
        for (i = 0; i < 64 * 16; i++) {
            s = s * 1103515245u + 12345u;
            res[i] = (int16_t)((int)((s >> 16) % 511u) - 255);
        }
        cpu.tq_batch(cpu.priv, res, 64, qp, 0, l0, z0);
        wk->ops.tq_batch(wk->ops.priv, res, 64, qp, 0, l1, z1);
        cpu.iq_idct_batch(cpu.priv, l0, 64, qp, r0);
        wk->ops.iq_idct_batch(wk->ops.priv, l0, 64, qp, r1);
        if (memcmp(l0, l1, sizeof(l0)) || memcmp(z0, z1, sizeof(z0)) ||
            memcmp(r0, r1, sizeof(r0)))
            wk->bad = 1;
    }
    return NULL;
}

/* free-running on a lane of its own: every operation, the reference
 * registered once and searched against repeatedly (no other thread can
 * replace it on this lane) */
static void *lane_run(void *arg)
{
    worker *wk = (worker *)arg;
    const aeg_be_ops *d = &wk->ops;
    const pic *c = &wk->cur;
    int it, i, n = 64;
    int16_t res[64 * 16], l0[64 * 16], l1[64 * 16], r0[64 * 16], r1[64 * 16];
    int16_t c0[64 * 16], c1[64 * 16], mv[4 * 64];
    int32_t z0[64], z1[64], s0[64], s1[64];
    uint64_t e0, e1;
    uint32_t s = 4242u + (uint32_t)wk->id;
    const uint8_t *pl[4] = { wk->planes[0].org, wk->planes[1].org,
                             wk->planes[2].org, wk->planes[3].org };
    pic ip;

    pic_alloc(&ip, c->w, c->h);
    d->ref_upload(d->priv, pl, wk->planes[0].stride, c->w, c->h);
    for (it = 0; it < 40 && !wk->bad; it++) {
        int qp = (it * 7 + wk->id * 13) % 52;
        for (i = 0; i < 64 * 16; i++) {
            s = s * 1103515245u + 12345u;
            res[i] = (int16_t)((int)((s >> 16) % 511u) - 255);
        }
        cpu.tq_iq_batch(cpu.priv, res, n, qp, it & 1, l0, z0, r0);
        d->tq_iq_batch(d->priv, res, n, qp, it & 1, l1, z1, r1);
        if (memcmp(l0, l1, sizeof(l0)) || memcmp(z0, z1, sizeof(z0)) ||
            memcmp(r0, r1, sizeof(r0)))
            wk->bad = 1;
        cpu.iq_idct_batch(cpu.priv, l0, n, qp, c0);
        d->iq_idct_batch(d->priv, l0, n, qp, c1);
        if (memcmp(c0, c1, sizeof(c0)))
            wk->bad = 2;
        for (i = 0; i < n; i++) {
            s = s * 1103515245u + 12345u;
            mv[4 * i + 0] = (int16_t)((i % (c->w / 16)) * 16);
            mv[4 * i + 1] = (int16_t)(((i / (c->w / 16)) % (c->h / 16)) * 16);
            mv[4 * i + 2] = (int16_t)(mv[4 * i + 0] + (int)((s >> 16) % 33u) - 16);
            mv[4 * i + 3] = (int16_t)(mv[4 * i + 1] + (int)((s >> 8) % 33u) - 16);
        }
        cpu.sad_batch(cpu.priv, c->org, c->stride, wk->planes[0].org,
                      wk->planes[0].stride, 16, 16, mv, n, s0);
        d->sad_batch(d->priv, c->org, c->stride, wk->planes[0].org,
                     wk->planes[0].stride, 16, 16, mv, n, s1);
        if (memcmp(s0, s1, sizeof(s0)))
            wk->bad = 3;
        cpu.sse_batch(cpu.priv, c->org, c->stride, wk->planes[1].org,
                      wk->planes[1].stride, c->w, c->h, &e0);
        d->sse_batch(d->priv, c->org, c->stride, wk->planes[1].org,
                     wk->planes[1].stride, c->w, c->h, &e1);
        if (e0 != e1)
            wk->bad = 4;
        d->interp_plane(d->priv, wk->planes[0].org, wk->planes[0].stride, c->w,
                        c->h, 1 + it % 3, ip.org, ip.stride);
        for (i = 0; i < c->h; i++)
            if (memcmp(ip.org + (ptrdiff_t)i * ip.stride,
                       wk->planes[1 + it % 3].org +
                           (ptrdiff_t)i * wk->planes[0].stride,
                       (size_t)c->w))
                wk->bad = 5;
        if (!satd_agree_on(d, c, wk->planes, it & 1 ? 8 : 16, 0))
            wk->bad = 6;
    }
    pic_free(&ip);
    return NULL;
}

static void check_threads(int w, int h, int lanes)
{
    worker wk[2];
    pthread_t th[2];
    int i, k;
    for (i = 0; i < 2; i++) {
        memset(&wk[i], 0, sizeof(wk[i]));
        wk[i].id = i;
        if (lanes)
            aeg_be_lane(&dev, 1 + i, &wk[i].ops);
        else
            wk[i].ops = dev;
        pic_alloc(&wk[i].cur, w, h);
        pic_fill(&wk[i].cur, 1);
        pic_alloc(&wk[i].planes[0], w, h);
        pic_fill(&wk[i].planes[0], 1);
        for (k = 1; k < 4; k++) {
            pic_alloc(&wk[i].planes[k], w, h);
            cpu.interp_plane(cpu.priv, wk[i].planes[0].org,
                             wk[i].planes[0].stride, w, h, k,
                             wk[i].planes[k].org, wk[i].planes[k].stride);
        }
    }
    pthread_barrier_init(&g_bar, NULL, 2);
    for (i = 0; i < 2; i++)
        pthread_create(&th[i], NULL, lockstep, &wk[i]);
    for (i = 0; i < 2; i++)
        pthread_join(th[i], NULL);
    pthread_barrier_destroy(&g_bar);
    check(!wk[0].bad && !wk[1].bad,
          "satd_batch with two threads' references interleaved");
    wk[0].bad = wk[1].bad = 0;
    for (i = 0; i < 2; i++)
        pthread_create(&th[i], NULL, hammer, &wk[i]);
    for (i = 0; i < 2; i++)
        pthread_join(th[i], NULL);
    check(!wk[0].bad && !wk[1].bad, "tq/iq batches from two threads at once");
    if (lanes) {
        aeg_be_ops again;
        wk[0].bad = wk[1].bad = 0;
        for (i = 0; i < 2; i++)
            pthread_create(&th[i], NULL, lane_run, &wk[i]);
        for (i = 0; i < 2; i++)
            pthread_join(th[i], NULL);
        if (wk[0].bad || wk[1].bad)
            printf("   lane_run: %d %d\n", wk[0].bad, wk[1].bad);
        check(!wk[0].bad && !wk[1].bad,
              "every operation on two lanes at once, own references");
        /* both device backends can clone: the lanes are instances of their
         * own (a clone that fails maps the lane to lane 0, correct but
         * serialised -- this would catch it) */
        if (wk[0].ops.priv == dev.priv && wk[1].ops.priv == dev.priv &&
            strstr(aeg_last_error(), "PoCL")) {
            /* a runtime known to break with several queues: the backend
             * declined, both threads above ran on lane 0, serialised */
            printf("   backend_check: lanes share lane 0 here (%s)\n",
                   aeg_last_error());
        } else {
            check(wk[0].ops.priv != dev.priv && wk[1].ops.priv != dev.priv &&
                      wk[0].ops.priv != wk[1].ops.priv,
                  "lanes are separate instances");
        }
        aeg_be_lane(&wk[1].ops, 1, &again);
        check(again.priv == wk[0].ops.priv, "a lane asked for twice is one lane");
        aeg_be_lane(&wk[1].ops, 0, &again);
        check(again.priv == dev.priv, "lane 0 is the backend itself");
    }
    for (i = 0; i < 2; i++)
        for (k = 0; k < 4; k++)
            pic_free(i ? &wk[1].planes[k] : &wk[0].planes[k]);
    pic_free(&wk[0].cur);
    pic_free(&wk[1].cur);
}

/* lowres_search on lowres planes allocated as the lookahead allocates them
 * (aeg_lowres_alloc), the reference a shifted noisy copy, with and without
 * a seed field */
static void check_lowres(int mb_w, int mb_h, int seeded)
{
    aeg_lowres c, r;
    const int nb = mb_w * mb_h, rows = mb_h * 8 + 2 * AEG_LA_PAD;
    const int range = seeded ? 48 : AEG_LA_RANGE;
    int *seed = (int *)malloc(sizeof(int) * 2 * (size_t)nb);
    int *a = (int *)malloc(sizeof(int) * 3 * (size_t)nb);
    int *b = (int *)malloc(sizeof(int) * 3 * (size_t)nb);
    int x, y, i, dx = rnd(15) - 7, dy = rnd(11) - 5;

    if (aeg_lowres_alloc(&c, mb_w, mb_h) || aeg_lowres_alloc(&r, mb_w, mb_h) ||
        !seed || !a || !b) {
        check(0, "lowres_search (allocation)");
        return;
    }
    for (y = 0; y < rows; y++)
        for (x = 0; x < c.stride; x++)
            c.mem[(size_t)y * c.stride + x] =
                (uint8_t)(100 + 50 * ((x / 6 + y / 4) & 1) + rnd(30) +
                          (x * 3 + y * 5) % 41);
    for (y = 0; y < rows; y++)
        for (x = 0; x < r.stride; x++) {
            int sx = x + dx, sy = y + dy, v;
            sx = sx < 0 ? 0 : sx >= c.stride ? c.stride - 1 : sx;
            sy = sy < 0 ? 0 : sy >= rows ? rows - 1 : sy;
            v = c.mem[(size_t)sy * c.stride + sx] + rnd(7) - 3;
            r.mem[(size_t)y * r.stride + x] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
        }
    for (i = 0; i < 2 * nb; i++)
        seed[i] = rnd(61) - 30;
    memset(a, 0x55, sizeof(int) * 3 * (size_t)nb);
    memset(b, 0xaa, sizeof(int) * 3 * (size_t)nb);
    cpu.lowres_search(cpu.priv, c.plane, r.plane, c.stride, c.w, c.h,
                      seeded ? seed : NULL, 5, 4, range, a);
    dev.lowres_search(dev.priv, c.plane, r.plane, c.stride, c.w, c.h,
                      seeded ? seed : NULL, 5, 4, range, b);
    check(!memcmp(a, b, sizeof(int) * 3 * (size_t)nb),
          seeded ? "lowres_search (seeded)" : "lowres_search");
    aeg_lowres_free(&c);
    aeg_lowres_free(&r);
    free(seed);
    free(a);
    free(b);
}

/* lowres_chains: three calls on one ring -- the first uploads every slot,
 * the second runs other chains on the resident pictures, the third after
 * one slot received a new picture (new tag), which must be uploaded again
 * rather than served stale */
static void lr_fill(aeg_lowres *l, int k)
{
    int x, y;
    for (y = 0; y < l->mb_h * 8 + 2 * AEG_LA_PAD; y++)
        for (x = 0; x < l->stride; x++) {
            int u = x + 3 * k, v = y - 2 * k;
            l->mem[(size_t)y * l->stride + x] =
                (uint8_t)(110 + 50 * ((u / 5 + v / 4) & 1) +
                          (u * 7 + v * 3) % 29 + rnd(9));
        }
}

static void check_chains(int mb_w, int mb_h)
{
    enum { NS = 6, NJ = 9 };
    aeg_lowres lr[NS];
    aeg_la_pic pics[NS];
    aeg_la_chain chs[5];
    aeg_la_job ja[NJ], jb[NJ];
    const int nb = mb_w * mb_h;
    int *a = (int *)malloc(sizeof(int) * 3 * (size_t)nb * NJ);
    int *b = (int *)malloc(sizeof(int) * 3 * (size_t)nb * NJ);
    int *seed = (int *)malloc(sizeof(int) * 2 * (size_t)nb);
    int k, i, call;

    if (!a || !b || !seed) {
        check(0, "lowres_chains (allocation)");
        return;
    }
    for (k = 0; k < NS; k++) {
        if (aeg_lowres_alloc(&lr[k], mb_w, mb_h)) {
            check(0, "lowres_chains (allocation)");
            return;
        }
        lr_fill(&lr[k], k);
        pics[k].plane = lr[k].plane;
        pics[k].slot = (k * 5 + 1) % NS;
        pics[k].tag = k;
    }
    for (i = 0; i < 2 * nb; i++)
        seed[i] = rnd(41) - 20;
    for (call = 0; call < 3; call++) {
        /* chains: (cur, first, count, seeded from `seed`) */
        static const int cc[3][5][4] = {
            { { 5, 0, 4, 0 }, { 0, 4, 3, 0 }, { 2, 7, 2, 1 }, { 0, 0, 0, 0 } },
            { { 3, 0, 2, 0 }, { 1, 2, 3, 1 }, { 4, 5, 1, 0 }, { 2, 6, 3, 0 } },
            { { 2, 0, 3, 0 }, { 4, 3, 2, 0 }, { 0, 5, 2, 1 }, { 5, 7, 2, 0 } } };
        int nc = call == 0 ? 3 : 4, cur, cnt;
        if (call == 2) {
            /* slot of picture 2 gets new samples under a new tag */
            lr_fill(&lr[2], 17);
            pics[2].tag = 100;
        }
        for (i = 0; i < nc; i++) {
            chs[i].cur = cc[call][i][0];
            chs[i].first = cc[call][i][1];
            chs[i].count = cc[call][i][2];
            chs[i].seed = cc[call][i][3] ? seed : NULL;
        }
        for (i = 0; i < nc; i++) {
            cur = chs[i].cur;
            for (cnt = 0; cnt < chs[i].count; cnt++) {
                aeg_la_job *j = &ja[chs[i].first + cnt];
                int dist = cnt + 1;
                j->ref = (cur + (i & 1 ? dist : NS - dist)) % NS;
                j->seeded = cnt > 0 || chs[i].seed;
                j->num = j->seeded ? dist + 1 : 1;
                j->den = j->seeded ? dist : 1;
                j->range = 16 + 8 * cnt;
                j->out = a + (size_t)(chs[i].first + cnt) * 3 * nb;
                jb[chs[i].first + cnt] = *j;
                jb[chs[i].first + cnt].out =
                    b + (size_t)(chs[i].first + cnt) * 3 * nb;
            }
        }
        cnt = chs[nc - 1].first + chs[nc - 1].count;
        memset(a, 0x55, sizeof(int) * 3 * (size_t)nb * NJ);
        memset(b, 0xaa, sizeof(int) * 3 * (size_t)nb * NJ);
        cpu.lowres_chains(cpu.priv, pics, NS, NS, chs, nc, ja, lr[0].stride,
                          lr[0].w, lr[0].h);
        dev.lowres_chains(dev.priv, pics, NS, NS, chs, nc, jb, lr[0].stride,
                          lr[0].w, lr[0].h);
        check(!memcmp(a, b, sizeof(int) * 3 * (size_t)nb * (size_t)cnt),
              call == 0 ? "lowres_chains (upload)"
              : call == 1 ? "lowres_chains (resident)"
                          : "lowres_chains (slot reused)");
    }
    for (k = 0; k < NS; k++)
        aeg_lowres_free(&lr[k]);
    free(a);
    free(b);
    free(seed);
}

/* lowres_build, lowres_intra, lowres_costs on one ring: the device builds
 * every slot (and must hand back the host's bytes); the intra estimates,
 * a chain and the cost estimates then run on those resident slots, with
 * nothing uploaded; finally one slot holds a new picture built on the host
 * (new tag), which the next calls must upload rather than serve stale. */
static void check_rest(int mb_w, int mb_h)
{
    enum { NS = 5, NJ = 6 };
    aeg_lowres lc[NS], ld[NS];
    aeg_la_pic pc[NS], pd[NS];
    aeg_la_cost ka[NJ], kb[NJ];
    aeg_la_chain ch;
    aeg_la_job ja[2], jb[2];
    const int nb = mb_w * mb_h;
    const int sw = 16 * mb_w - (mb_w > 1 ? 3 : 0), sh = 16 * mb_h - (mb_h > 1 ? 7 : 0);
    const int sstride = sw + 24;
    int *ica[NS], *icb[NS], *fld[NS], *fa, *fb, intra_a[NJ], intra_b[NJ];
    int64_t ta[NJ], tb[NJ];
    uint8_t *src = (uint8_t *)malloc((size_t)sstride * (size_t)sh);
    size_t sz;
    int k, i, x, y, ok, round;

    for (k = 0; k < NS; k++)
        if (aeg_lowres_alloc(&lc[k], mb_w, mb_h) ||
            aeg_lowres_alloc(&ld[k], mb_w, mb_h)) {
            check(0, "lowres_build (allocation)");
            return;
        }
    sz = (size_t)lc[0].stride * (size_t)(lc[0].h + 2 * AEG_LA_PAD);
    fa = (int *)malloc(sizeof(int) * 3 * (size_t)nb * 2);
    fb = (int *)malloc(sizeof(int) * 3 * (size_t)nb * 2);
    for (k = 0; k < NS; k++) {
        ica[k] = (int *)malloc(sizeof(int) * (size_t)nb);
        icb[k] = (int *)malloc(sizeof(int) * (size_t)nb);
        fld[k] = (int *)malloc(sizeof(int) * 3 * (size_t)nb);
        if (!ica[k] || !icb[k] || !fld[k])
            return;
    }
    if (!src || !fa || !fb)
        return;
    for (round = 0; round < 2; round++) {
        /* round 0: every picture built by both; round 1: slot of picture 2
         * gets a new picture (tag 7) built on the host only */
        ok = 1;
        for (k = 0; k < NS; k++) {
            if (round && k != 2)
                continue;
            for (y = 0; y < sh; y++)
                for (x = 0; x < sstride; x++) {
                    int u = x + 4 * k + 9 * round, v = y + 2 * k;
                    src[(size_t)y * sstride + x] =
                        (uint8_t)(80 + 80 * (((u + 3 * v) / 9) & 1) +
                                  (u * v) % 31 + rnd(11));
                }
            pc[k].plane = lc[k].plane;
            pd[k].plane = ld[k].plane;
            pc[k].slot = pd[k].slot = (k + 3) % NS;
            pc[k].tag = pd[k].tag = round ? 7 : k;
            memset(lc[k].mem, 0x11, sz);
            memset(ld[k].mem, 0xee, sz);
            cpu.lowres_build(cpu.priv, src, sstride, sw, sh, &pc[k], NS,
                             lc[k].stride, lc[k].w, lc[k].h);
            if (round)
                memcpy(ld[k].mem, lc[k].mem, sz);   /* the host's picture */
            else
                dev.lowres_build(dev.priv, src, sstride, sw, sh, &pd[k], NS,
                                 ld[k].stride, ld[k].w, ld[k].h);
            ok &= !memcmp(lc[k].mem, ld[k].mem, sz);
        }
        if (!round)
            check(ok, "lowres_build");

        /* the intra estimates of every picture, resident */
        for (k = 0; k < NS; k++) {
            memset(ica[k], 0x55, sizeof(int) * (size_t)nb);
            memset(icb[k], 0xaa, sizeof(int) * (size_t)nb);
        }
        cpu.lowres_intra(cpu.priv, pc, NS, NS, ica, aeg_lowres_i8(),
                         lc[0].stride, lc[0].w, lc[0].h);
        dev.lowres_intra(dev.priv, pd, NS, NS, icb, aeg_lowres_i8(),
                         ld[0].stride, ld[0].w, ld[0].h);
        ok = 1;
        for (k = 0; k < NS; k++)
            ok &= !memcmp(ica[k], icb[k], sizeof(int) * (size_t)nb);
        check(ok, round ? "lowres_intra (slot reused)" : "lowres_intra");

        /* a chain on the built slots: picture 2 against 1, then 0 */
        ch.cur = 2;
        ch.first = 0;
        ch.count = 2;
        ch.seed = NULL;
        for (i = 0; i < 2; i++) {
            ja[i].ref = 1 - i;
            ja[i].seeded = i > 0;
            ja[i].num = i ? 2 : 1;
            ja[i].den = 1;
            ja[i].range = 16 + 8 * i;
            ja[i].out = fa + (size_t)i * 3 * nb;
            jb[i] = ja[i];
            jb[i].out = fb + (size_t)i * 3 * nb;
        }
        cpu.lowres_chains(cpu.priv, pc, NS, NS, &ch, 1, ja, lc[0].stride,
                          lc[0].w, lc[0].h);
        dev.lowres_chains(dev.priv, pd, NS, NS, &ch, 1, jb, ld[0].stride,
                          ld[0].w, ld[0].h);
        check(!memcmp(fa, fb, sizeof(int) * 6 * (size_t)nb),
              round ? "lowres_chains after lowres_build (slot reused)"
                    : "lowres_chains after lowres_build");

        /* the estimates: fields searched on the CPU, two made up with
         * independent zero vectors; B and P, and one with its own picture
         * as both references */
        for (k = 0; k < NS; k++) {
            if (k < 3) {
                cpu.lowres_search(cpu.priv, lc[k].plane, lc[k + 1].plane,
                                  lc[k].stride, lc[k].w, lc[k].h, NULL, 1, 1,
                                  16, fld[k]);
            } else {
                for (i = 0; i < nb; i++) {
                    const int z = rnd(2);
                    fld[k][2 * i] = z ? 0 : rnd(81) - 40;
                    fld[k][2 * i + 1] = z ? 0 : rnd(81) - 40;
                    fld[k][2 * nb + i] = ica[0][i] + rnd(301) + 100;
                }
            }
        }
        {
            static const int jj[NJ][5] = {
                { 1, 0, 2, 0, 1 }, { 2, 1, 3, 1, 2 }, { 0, 1, -1, 0, 0 },
                { 4, 3, -1, 3, 0 }, { 2, 0, 4, 3, 4 }, { 1, 1, 1, 3, 4 }
            };
            for (i = 0; i < NJ; i++) {
                ka[i].cur = jj[i][0];
                ka[i].r0 = jj[i][1];
                ka[i].r1 = jj[i][2];
                ka[i].f0 = fld[jj[i][3]];
                ka[i].f1 = fld[jj[i][4]];
                ka[i].icost = ica[jj[i][0]];
                ka[i].total = &ta[i];
                ka[i].intra = &intra_a[i];
                kb[i] = ka[i];
                kb[i].total = &tb[i];
                kb[i].intra = &intra_b[i];
                ta[i] = tb[i] = -1;
                intra_a[i] = intra_b[i] = -1;
            }
        }
        cpu.lowres_costs(cpu.priv, pc, NS, NS, ka, NJ, lc[0].stride, lc[0].w,
                         lc[0].h);
        dev.lowres_costs(dev.priv, pd, NS, NS, kb, NJ, ld[0].stride, ld[0].w,
                         ld[0].h);
        ok = 1;
        for (i = 0; i < NJ; i++)
            ok &= ta[i] == tb[i] && intra_a[i] == intra_b[i];
        check(ok, round ? "lowres_costs (slot reused)" : "lowres_costs");
    }
    for (k = 0; k < NS; k++) {
        aeg_lowres_free(&lc[k]);
        aeg_lowres_free(&ld[k]);
        free(ica[k]);
        free(icb[k]);
        free(fld[k]);
    }
    free(src);
    free(fa);
    free(fb);
}

int main(int argc, char **argv)
{
    int which = AEG_BACKEND_AUTO, k;
    static const int sizes[][2] = { { 176, 144 }, { 640, 352 }, { 48, 32 } };

    if (argc > 1 && !strcmp(argv[1], "opencl"))
        which = AEG_BACKEND_OPENCL;
    else if (argc > 1 && !strcmp(argv[1], "cuda"))
        which = AEG_BACKEND_CUDA;
    aeg_be_cpu_ops(&cpu);
    aeg_be_create(which, &dev);
    if (dev.backend == AEG_BACKEND_CPU) {
        printf("   backend_check: no device backend in this build or on this "
               "machine, skipped\n");
        return 0;
    }
    printf("   backend_check: %s\n", dev.desc);
    check_coef();
    for (k = 0; k < 3; k++) {
        int w = sizes[k][0], h = sizes[k][1], i;
        pic planes[4], cur;
        check_sad(w, h);
        check_sse(w, h);
        check_interp(w, h, planes);
        pic_alloc(&cur, w, h);
        pic_fill(&cur, 1);
        check(satd_agree(&cur, planes, 16, 1), "satd_batch 16x16");
        check(satd_agree(&cur, planes, 8, 1), "satd_batch 8x8");
        pic_free(&cur);
        for (i = 0; i < 4; i++)
            pic_free(&planes[i]);
    }
    if (dev.lowres_search) {
        /* QCIF, the long clips' cockL, 1080p, and the degenerate shapes */
        static const int lsz[][2] = { { 11, 9 }, { 40, 22 }, { 120, 68 },
                                      { 1, 1 }, { 1, 7 }, { 9, 1 } };
        for (k = 0; k < 6; k++) {
            check_lowres(lsz[k][0], lsz[k][1], 0);
            check_lowres(lsz[k][0], lsz[k][1], 1);
        }
    }
    if (dev.lowres_chains) {
        static const int csz[][2] = { { 11, 9 }, { 40, 22 }, { 120, 68 },
                                      { 1, 1 }, { 3, 1 } };
        for (k = 0; k < 5; k++)
            check_chains(csz[k][0], csz[k][1]);
    }
    if (dev.lowres_build && dev.lowres_intra && dev.lowres_costs) {
        static const int rsz[][2] = { { 11, 9 }, { 40, 22 }, { 120, 68 },
                                      { 1, 1 }, { 3, 1 }, { 1, 4 } };
        for (k = 0; k < 6; k++)
            check_rest(rsz[k][0], rsz[k][1]);
    }
    check_threads(176, 144, 0);
    check_threads(176, 144, 1);
    if (dev.destroy)
        dev.destroy(dev.priv);
    if (fails) {
        printf("   backend_check: %d of %d checks FAILED\n", fails, checks);
        return 1;
    }
    printf("   backend_check: %d checks agree with the CPU reference\n", checks);
    return 0;
}
