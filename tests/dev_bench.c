/*
 * Device micro-benchmark: what each batched operation costs on the device
 * next to the CPU reference, at picture sizes that matter.
 *
 *   make dev_bench
 *   ./build/dev_bench [W H]
 *
 * Why this exists: tools/gpu_validate.sh proves the device paths are *correct*,
 * and --bench measures whole encodes, but neither says whether an individual
 * kernel is worth launching.  Every number in docs/DESIGN_ja.md's "Measured"
 * table comes from here, and the milestone gates in that document are written
 * against this output -- so a change that makes the device slower shows up as a
 * diff of this output rather than as a feeling.
 *
 * The picture geometry is the encoder's own (AEG_PAD of border, one padded
 * stride per plane) and the candidate counts are the ones me_seed_frame() and
 * me_satd_round() actually issue, so the numbers transfer to an encode.
 *
 * Exit status: 0 if a device answered, 1 if only the CPU is available (the CPU
 * column is printed either way).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"

#define ME_CAND 9               /* ME_ROUND_CAND in encoder.c */

/* one timed call plus the device synchronisation that ends it */
typedef enum { OP_SAD, OP_SATD, OP_UPLOAD, OP_TQ } op_kind;

typedef struct {
    const int16_t *mv;          /* sad: candidate list (candidate-major)     */
    const int16_t *pos;         /* satd: block position + quarter-pel vector */
    int            count;
    int32_t       *out;
    const uint8_t *plane[4];
    int16_t       *res;         /* tq: residuals                             */
    int16_t       *level;       /* tq: levels out                            */
    int32_t       *nz;          /* tq: non-zero counts out                   */
    int            nblk;
    int            w, h;
} op_arg;

static int64_t time_op(aeg_be_ops *o, op_kind k, op_arg *a, int reps)
{
    int64_t best = -1;
    int i;
    for (i = 0; i < reps; i++) {
        int64_t t0 = aeg_now_ns();
        switch (k) {
        case OP_SAD:
            o->sad_batch(o->priv, a->plane[0], a->w, a->plane[1], a->w, 16, 16,
                         a->mv, a->count, a->out);
            break;
        case OP_SATD:
            o->satd_batch(o->priv, a->plane[0], a->w, 16, 16, a->pos, a->count,
                          a->out);
            break;
        case OP_UPLOAD:
            o->ref_upload(o->priv, a->plane, a->w, a->w - 2 * AEG_PAD,
                          a->h - 2 * AEG_PAD);
            break;
        case OP_TQ:
            o->tq_batch(o->priv, a->res, a->nblk, 26, 1, a->level, a->nz);
            break;        }
        if (o->sync)
            o->sync(o->priv);
        {
            int64_t dt = aeg_now_ns() - t0;
            if (best < 0 || dt < best)
                best = dt;
        }
    }
    return best;
}

static void row(const char *name, int64_t cpu_ns, int64_t dev_ns, int have_dev)
{
    if (have_dev)
        printf("  %-12s %9.3f ms %9.3f ms %8.2fx\n", name, cpu_ns / 1e6,
               dev_ns / 1e6, (double)cpu_ns / (double)(dev_ns > 0 ? dev_ns : 1));
    else
        printf("  %-12s %9.3f ms %9s %9s\n", name, cpu_ns / 1e6, "-", "-");
}

#define LO_SLOTS 64           /* representative lookahead ring */
#define LO_PAIRS 100

/* M-3.3a: the lowres ring resident on the device.  Compares 100 pair
 * searches against a resident 64-slot ring with the same searches when both
 * planes are transferred every call. */
static void bench_lowres(aeg_be_ops *cpu, aeg_be_ops *dev, int have_dev,
                         int w, int h)
{
    aeg_lowres lo[LO_SLOTS];
    int mbw = (w + 15) / 16, mbh = (h + 15) / 16;
    int nblk = mbw * mbh, count = nblk * 9;
    int stride, low_w, low_h, i, k, r;
    size_t plane_bytes;
    int16_t *mv;
    int32_t *out, *ref_out;
    int64_t cr = 0, dr = 0, cp = 0, dp = 0;
    int ok = 1;

    memset(lo, 0, sizeof(lo));
    for (i = 0; i < LO_SLOTS; i++) {
        if (aeg_lowres_alloc(&lo[i], mbw, mbh) != 0) {
            while (i-- > 0)
                aeg_lowres_free(&lo[i]);
            return;
        }
        for (k = 0; k < (int)((size_t)lo[i].stride * (lo[i].h + 2 * AEG_LA_PAD)); k++)
            lo[i].mem[k] = (uint8_t)((i * 37 + k * 13) ^ (k >> 9));
    }
    stride = lo[0].stride;
    low_w = lo[0].w;
    low_h = lo[0].h;
    plane_bytes = (size_t)stride * (size_t)(low_h + 2 * AEG_LA_PAD);
    mv = (int16_t *)malloc((size_t)count * 8);
    out = (int32_t *)malloc((size_t)count * 4);
    ref_out = (int32_t *)malloc((size_t)count * 4);
    if (!mv || !out || !ref_out) {
        free(mv); free(out); free(ref_out);
        for (i = 0; i < LO_SLOTS; i++)
            aeg_lowres_free(&lo[i]);
        return;
    }
    for (i = 0; i < nblk; i++)
        for (k = 0; k < 9; k++) {
            int idx = i * 9 + k;
            int bx = (i % mbw) * 8, by = (i / mbw) * 8;
            mv[4 * idx + 0] = (int16_t)bx;
            mv[4 * idx + 1] = (int16_t)by;
            mv[4 * idx + 2] = (int16_t)(bx + (k % 3 - 1) * 8);
            mv[4 * idx + 3] = (int16_t)(by + (k / 3 - 1) * 8);
        }
    for (i = 0; i < LO_SLOTS; i++) {
        cpu->lowres_upload(cpu->priv, i, lo[i].plane, stride, low_w, low_h,
                           (uint64_t)i + 1);
        if (have_dev)
            dev->lowres_upload(dev->priv, i, lo[i].plane, stride, low_w, low_h,
                               (uint64_t)i + 1);
    }
    /* resident ring: one upload per slot, no transfer inside the loop */
    for (r = 0; r < LO_PAIRS; r++) {
        int cs = r % LO_SLOTS, rs = (r * 7 + 3) % LO_SLOTS;
        int64_t t0 = aeg_now_ns();
        cpu->lowres_sad_batch(cpu->priv, cs, rs, 8, 8, mv, count, out);
        cr += aeg_now_ns() - t0;
        if (have_dev) {
            t0 = aeg_now_ns();
            dev->lowres_sad_batch(dev->priv, cs, rs, 8, 8, mv, count, ref_out);
            dr += aeg_now_ns() - t0;
            if (r == 0 && memcmp(out, ref_out, (size_t)count * 4))
                ok = 0;
        }
    }
    /* per-call: both planes go through sad_batch every time */
    for (r = 0; r < LO_PAIRS; r++) {
        int cs = r % LO_SLOTS, rs = (r * 7 + 3) % LO_SLOTS;
        int64_t t0 = aeg_now_ns();
        cpu->sad_batch(cpu->priv, lo[cs].plane, stride, lo[rs].plane, stride,
                       8, 8, mv, count, out);
        cp += aeg_now_ns() - t0;
        if (have_dev) {
            t0 = aeg_now_ns();
            dev->sad_batch(dev->priv, lo[cs].plane, stride, lo[rs].plane,
                           stride, 8, 8, mv, count, ref_out);
            dp += aeg_now_ns() - t0;
            if (r == 0 && memcmp(out, ref_out, (size_t)count * 4))
                ok = 0;
        }
    }
    printf("\n  lowres ring (M-3.3a): %d slots, %.2f MB each, %d pair searches\n",
           LO_SLOTS, plane_bytes / 1048576.0, LO_PAIRS);
    printf("    resident   cpu %7.2f ms   dev %7.2f ms   uploads %6.2f MB\n",
           cr / 1e6, dr / 1e6, LO_SLOTS * plane_bytes / 1048576.0);
    printf("    per-call   cpu %7.2f ms   dev %7.2f ms   uploads %6.2f MB\n",
           cp / 1e6, dp / 1e6, 2.0 * LO_PAIRS * plane_bytes / 1048576.0);
    if (have_dev && dp > 0)
        printf("    device resident/per-call: %.2fx, transfer %.2fx less\n",
               (double)dp / (dr > 0 ? dr : 1),
               (2.0 * LO_PAIRS) / LO_SLOTS);
    if (cpu->lowres_batch_search && (!have_dev || dev->lowres_batch_search)) {
        const int NP = 8;
        int32_t *pairs = (int32_t *)malloc((size_t)NP * 2 * sizeof(int32_t));
        int32_t *cmv = (int32_t *)malloc((size_t)NP * nblk * 8);
        int32_t *dmv = (int32_t *)malloc((size_t)NP * nblk * 8);
        int32_t *ccost = (int32_t *)malloc((size_t)NP * nblk * 4);
        int32_t *dcost = (int32_t *)malloc((size_t)NP * nblk * 4);
        if (pairs && cmv && dmv && ccost && dcost) {
            int64_t cb = 0, db = 0;
            for (i = 0; i < NP; i++) {
                pairs[2 * i] = i * 3 % LO_SLOTS;
                pairs[2 * i + 1] = (i * 7 + 5) % LO_SLOTS;
            }
            {
                int64_t t0 = aeg_now_ns();
                cpu->lowres_batch_search(cpu->priv, pairs, NP, mbw, mbh, 10, cmv,
                                         ccost);
                cb = aeg_now_ns() - t0;
            }
            if (have_dev) {
                int64_t t0 = aeg_now_ns();
                dev->lowres_batch_search(dev->priv, pairs, NP, mbw, mbh, 10, dmv,
                                         dcost);
                db = aeg_now_ns() - t0;
                ok = ok && !memcmp(cmv, dmv, (size_t)NP * nblk * 8) &&
                     !memcmp(ccost, dcost, (size_t)NP * nblk * 4);
            }
            printf("    batch search %d pairs x %d blocks: cpu %7.2f ms   dev %7.2f ms%s\n",
                   NP, nblk, cb / 1e6, db / 1e6,
                   have_dev ? (ok ? "  (equal)" : "  (DIFFER)") : "");
        }
        free(pairs); free(cmv); free(dmv); free(ccost); free(dcost);
    }
    if (!ok)
        printf("    WARNING: resident and per-call results differ\n");
    for (i = 0; i < LO_SLOTS; i++)
        aeg_lowres_free(&lo[i]);
    free(mv); free(out); free(ref_out);
}

int main(int argc, char **argv)
{
    int w = argc > 1 ? atoi(argv[1]) : 1920;
    int h = argc > 2 ? atoi(argv[2]) : 1080;
    int which = AEG_BACKEND_AUTO;
    int mbw, mbh, nmb, count, i, k;
    int stride, rows;
    uint8_t *pic;
    int16_t *mv, *pos;
    int32_t *out;
    int16_t *res, *level;
    int32_t *nz_out;
    op_arg a;
    int64_t cpu_ns, dev_ns = 0;
    aeg_be_ops cpu, dev;
    int have_dev;

    if (argc > 3 && !strcmp(argv[3], "opencl"))
        which = AEG_BACKEND_OPENCL;
    else if (argc > 3 && !strcmp(argv[3], "cuda"))
        which = AEG_BACKEND_CUDA;
    else if (argc > 3 && !strcmp(argv[3], "cpu"))
        which = AEG_BACKEND_CPU;
    if (w < 64 || h < 64 || (w & 15)) {
        fprintf(stderr, "usage: dev_bench [W H [auto|opencl|cuda|cpu]]\n");
        return 2;
    }
    /* the encoder codes whole macroblocks: a 1080-line picture is 68 rows of
     * macroblocks, i.e. 1088 lines of allocated picture */
    h = (h + 15) & ~15;
    stride = w + 2 * AEG_PAD;
    rows = h + 2 * AEG_PAD;
    mbw = w / 16;
    mbh = h / 16;
    nmb = mbw * mbh;
    count = nmb * ME_CAND;

    pic = (uint8_t *)malloc((size_t)stride * rows);
    mv = (int16_t *)malloc((size_t)count * 8);
    pos = (int16_t *)malloc((size_t)count * 8);
    out = (int32_t *)malloc((size_t)count * 4);
    /* the whole-picture coefficient batch is nmb*16 blocks of 16 samples */
    res = (int16_t *)malloc((size_t)nmb * 256 * sizeof(int16_t));
    level = (int16_t *)malloc((size_t)nmb * 256 * sizeof(int16_t));
    nz_out = (int32_t *)malloc((size_t)nmb * 16 * sizeof(int32_t));
    if (!pic || !mv || !pos || !out || !res || !level || !nz_out) {
        fprintf(stderr, "out of memory\n");
        return 2;
    }
    /* deterministic, with enough structure that the caches behave as they do
     * on real content (a constant picture would be a pure cache hit) */
    for (i = 0; i < stride * rows; i++)
        pic[i] = (uint8_t)(((i * 37) >> 3) ^ (i >> 7));
    for (i = 0; i < nmb * 256; i++)
        res[i] = (int16_t)((i * 29) % 511 - 255);

    /* the candidate layout the encoder issues: every macroblock with its nine
     * neighbours, candidate-major */
    for (i = 0; i < nmb; i++)
        for (k = 0; k < ME_CAND; k++) {
            int idx = k * nmb + i;
            int ox = (i % mbw) * 16, oy = (i / mbw) * 16;
            mv[4 * idx + 0] = (int16_t)ox;
            mv[4 * idx + 1] = (int16_t)oy;
            mv[4 * idx + 2] = (int16_t)(ox + (k % 3) - 1);
            mv[4 * idx + 3] = (int16_t)(oy + (k / 3) - 1);
            pos[4 * idx + 0] = (int16_t)ox;
            pos[4 * idx + 1] = (int16_t)oy;
            pos[4 * idx + 2] = (int16_t)(k % 4);
            pos[4 * idx + 3] = (int16_t)(k / 4);
        }

    aeg_be_cpu_ops(&cpu);
    have_dev = aeg_be_create(which, &dev) == AEG_OK &&
               dev.backend != AEG_BACKEND_CPU;
    memset(&a, 0, sizeof(a));
    a.w = stride;
    a.h = rows;
    for (k = 0; k < 4; k++)
        a.plane[k] = pic + AEG_PAD * stride + AEG_PAD;
    a.mv = mv;
    a.pos = pos;
    a.count = count;
    a.out = out;
    a.res = res;
    a.level = level;
    a.nz = nz_out;
    a.nblk = 4096;

    printf("dev_bench %dx%d: %d macroblocks, %d candidates per round\n", w, h,
           nmb, count);
    printf("  device: %s\n\n", have_dev ? dev.desc : "none");
    printf("  %-12s %12s %12s %9s\n", "operation", "cpu", "device", "cpu/device");

    /* the fractional-pel batches read the registered reference: register it
     * first, exactly as me_seed_frame() does */
    if (have_dev && dev.ref_upload) {
        cpu_ns = time_op(&cpu, OP_UPLOAD, &a, 3);
        dev_ns = time_op(&dev, OP_UPLOAD, &a, 5);
        row("ref_upload", cpu_ns, dev_ns, have_dev);
    } else {
        cpu.satd_batch = NULL;  /* nothing to compare the SATD batch against */
    }

    cpu_ns = time_op(&cpu, OP_SAD, &a, 3);
    if (have_dev)
        dev_ns = time_op(&dev, OP_SAD, &a, 5);
    row("sad_batch", cpu_ns, dev_ns, have_dev);

    if (cpu.satd_batch) {
        cpu_ns = time_op(&cpu, OP_SATD, &a, 3);
        if (have_dev)
            dev_ns = time_op(&dev, OP_SATD, &a, 5);
        row("satd_batch", cpu_ns, dev_ns, have_dev);
    }

    /* The size the encoder issues per call: one macroblock's sixteen blocks.
     * This is the number that decides whether per-macroblock batching can pay
     * at all. */
    a.nblk = 16;
    cpu_ns = time_op(&cpu, OP_TQ, &a, 20);
    if (have_dev)
        dev_ns = time_op(&dev, OP_TQ, &a, 20);
    row("tq 16blk", cpu_ns, dev_ns, have_dev);
    if (have_dev)
        printf("\n  one macroblock's coefficient batch (16 blocks):\n"
               "    device %6.2f us per call, CPU %6.2f us for the same work\n",
               dev_ns / 1e3, cpu_ns / 1e3);

    /* ... and a whole picture's worth, which is the shape a device placement
     * would have to use. */
    a.nblk = nmb * 16;
    cpu_ns = time_op(&cpu, OP_TQ, &a, 5);
    if (have_dev)
        dev_ns = time_op(&dev, OP_TQ, &a, 5);
    row("tq whole pic", cpu_ns, dev_ns, have_dev);

    if (cpu.lowres_upload && cpu.lowres_sad_batch)
        bench_lowres(&cpu, &dev, have_dev, w, h);

    if (cpu.mbt_batch && (!have_dev || dev.mbt_batch)) {
        int np = 4, nmb2 = mbw * mbh, n2 = np * nmb2, q;
        int *bi = (int *)malloc((size_t)n2 * 4);
        int *bn = (int *)malloc((size_t)n2 * 4);
        int *bmv = (int *)malloc((size_t)n2 * 16);
        uint8_t *bk = (uint8_t *)malloc((size_t)n2);
        int *br0 = (int *)malloc((size_t)np * 4);
        int *br1 = (int *)malloc((size_t)np * 4);
        int32_t *cp = (int32_t *)calloc((size_t)n2, 4);
        int32_t *dp = (int32_t *)calloc((size_t)n2, 4);
        if (bi && bn && bmv && bk && br0 && br1 && cp && dp) {
            for (q = 0; q < n2; q++) {
                bi[q] = 50 + (q * 17) % 900;
                bn[q] = bi[q] - (q * 7) % 40;
                bmv[4 * q + 0] = (q * 3) % 17 - 8;
                bmv[4 * q + 1] = (q * 5) % 17 - 8;
                bmv[4 * q + 2] = -bmv[4 * q + 0];
                bmv[4 * q + 3] = -bmv[4 * q + 1];
                bk[q] = (uint8_t)(q % 3);
            }
            br0[0] = br1[0] = -1;
            br0[1] = 0; br1[1] = -1;
            br0[2] = 1; br1[2] = 0;
            br0[3] = 2; br1[3] = 1;
            cpu.mbt_batch(cpu.priv, np, mbw, mbh, bi, bn, bk, bmv, br0, br1,
                          32, cp);
            if (have_dev) {
                dev.mbt_batch(dev.priv, np, mbw, mbh, bi, bn, bk, bmv, br0,
                              br1, 32, dp);
                printf("  mbtree propagation: %d pictures x %d blocks: %s\n",
                       np, nmb2, memcmp(cp, dp, (size_t)n2 * 4)
                           ? "DIFFER" : "equal");
            }
        }
        free(bi); free(bn); free(bmv); free(bk); free(br0); free(br1);
        free(cp); free(dp);
    }

    if (have_dev && dev.destroy)
        dev.destroy(dev.priv);
    free(pic);
    free(mv);
    free(pos);
    free(out);
    free(res);
    free(level);
    free(nz_out);
    return have_dev ? 0 : 1;
}
