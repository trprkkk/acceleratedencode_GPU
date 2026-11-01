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

static int satd_agree(const pic *cur, pic planes[4], int bs, int upload)
{
    const uint8_t *pl[4] = { planes[0].org, planes[1].org, planes[2].org,
                             planes[3].org };
    int n, ok;
    int16_t *pos = satd_positions(cur->w, cur->h, bs, &n);
    int32_t *o0 = (int32_t *)malloc((size_t)n * 4);
    int32_t *o1 = (int32_t *)malloc((size_t)n * 4);
    if (upload)
        dev.ref_upload(dev.priv, pl, planes[0].stride, cur->w, cur->h);
    cpu.ref_upload(cpu.priv, pl, planes[0].stride, cur->w, cur->h);
    cpu.satd_batch(cpu.priv, cur->org, cur->stride, bs, bs, pos, n, o0);
    dev.satd_batch(dev.priv, cur->org, cur->stride, bs, bs, pos, n, o1);
    ok = !memcmp(o0, o1, (size_t)n * 4);
    free(pos); free(o0); free(o1);
    return ok;
}

/* ---------------------------------------------------------------- */
/* Two threads, one device                                           */
/* ---------------------------------------------------------------- */
typedef struct {
    pic cur, planes[4];
    int id, bad;
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
                dev.ref_upload(dev.priv, pl, wk->planes[0].stride, wk->cur.w,
                               wk->cur.h);
            if (step == 2 + wk->id && !satd_agree(&wk->cur, wk->planes, 16, 0))
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
        dev.tq_batch(dev.priv, res, 64, qp, 0, l1, z1);
        cpu.iq_idct_batch(cpu.priv, l0, 64, qp, r0);
        dev.iq_idct_batch(dev.priv, l0, 64, qp, r1);
        if (memcmp(l0, l1, sizeof(l0)) || memcmp(z0, z1, sizeof(z0)) ||
            memcmp(r0, r1, sizeof(r0)))
            wk->bad = 1;
    }
    return NULL;
}

static void check_threads(int w, int h)
{
    worker wk[2];
    pthread_t th[2];
    int i, k;
    for (i = 0; i < 2; i++) {
        memset(&wk[i], 0, sizeof(wk[i]));
        wk[i].id = i;
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
    for (i = 0; i < 2; i++)
        for (k = 0; k < 4; k++)
            pic_free(i ? &wk[1].planes[k] : &wk[0].planes[k]);
    pic_free(&wk[0].cur);
    pic_free(&wk[1].cur);
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
    check_threads(176, 144);
    if (dev.destroy)
        dev.destroy(dev.priv);
    if (fails) {
        printf("   backend_check: %d of %d checks FAILED\n", fails, checks);
        return 1;
    }
    printf("   backend_check: %d checks agree with the CPU reference\n", checks);
    return 0;
}
