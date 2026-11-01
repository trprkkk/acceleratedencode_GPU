/*
 * CPU implementation of the accelerator operations, the backend factory and
 * the per-stage dispatcher.
 *
 * The dispatcher exists because a GPU is not automatically faster for every
 * stage of an encoder.  A 16x16 SAD costs a few hundred nanoseconds on a CPU
 * core; the same work on a GPU costs a kernel launch plus, for the search
 * results, a transfer back.  The dispatcher therefore:
 *
 *   1. measures every stage on whichever backend actually ran it,
 *   2. keeps an exponential moving average of the CPU and GPU times,
 *   3. flips a stage to the GPU only when the GPU time is clearly lower
 *      (hysteresis avoids oscillating between the two), and
 *   4. always keeps the stages whose cost is dominated by serial
 *      dependencies (deblocking, CAVLC bit emission, intra prediction of the
 *      current block) on the CPU.
 *
 * Because both implementations are bit-identical, the decisions can change
 * from frame to frame without affecting the produced bitstream.
 */
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include "backend.h"

/* ------------------------------------------------------------------ */
/* CPU reference implementations                                       */
/* ------------------------------------------------------------------ */
static int cpu_sad_batch(void *priv, const uint8_t *cur, int cstride,
                         const uint8_t *ref, int rstride, int bw, int bh,
                         const int16_t *mv, int count, int32_t *out)
{
    int i;
    (void)priv;
    for (i = 0; i < count; i++) {
        const uint8_t *c0 = cur + (ptrdiff_t)mv[4 * i + 1] * cstride + mv[4 * i + 0];
        const uint8_t *r = ref + (ptrdiff_t)mv[4 * i + 3] * rstride + mv[4 * i + 2];
        int x, y, sad = 0;
        for (y = 0; y < bh; y++) {
            const uint8_t *c = c0 + y * cstride;
            const uint8_t *rr = r + y * rstride;
            for (x = 0; x < bw; x++) {
                int d = (int)c[x] - (int)rr[x];
                sad += d < 0 ? -d : d;
            }
        }
        out[i] = sad;
    }
    return AEG_OK;
}

static int cpu_tq_batch(void *priv, const int16_t *res, int nblk, int qp,
                        int intra, int16_t *level, int32_t *nz)
{
    int b;
    (void)priv;
    for (b = 0; b < nblk; b++) {
        int16_t coef[16];
        int i, n = 0;
        aeg_transform_4x4(res + 16 * b, 4, coef);
        aeg_quant_4x4(coef, level + 16 * b, qp, intra);
        for (i = 0; i < 16; i++)
            n += level[16 * b + i] != 0;
        if (nz)
            nz[b] = n;
    }
    return AEG_OK;
}

static int cpu_iq_idct_batch(void *priv, const int16_t *level, int nblk,
                             int qp, int16_t *res)
{
    int b;
    (void)priv;
    /* This used to carry its own copy of the inverse transform, and that copy
     * was *not* the one the encoder uses: it skipped the int16 truncation of
     * the dequantised coefficients (bug 8), skipped the DC rounding before the
     * first pass, and rounded the second pass instead of truncating it.  The
     * device kernel was checked against this copy, so both agreed and both
     * disagreed with the reference decoder -- exactly the trap that made the
     * first SIMD inverse transform look right.  It now calls the same function
     * the scalar path is built from, so there is one arithmetic to be wrong. */
    /* Most blocks of a batch are empty or DC-only (the quantiser and the
     * level selection see to that), and for those the two passes reduce to a
     * constant: d[0] = (int16)(level[0] * scale << qp/6) + 32 goes through
     * the first pass truncated to int16 and through the second unchanged, so
     * every sample is ((int16)d[0]) >> 6 -- the same value aeg_idct_residual()
     * computes, including at the int16 extremes (coef_check covers both). */
    for (b = 0; b < nblk; b++) {
        const int16_t *l = level + 16 * b;
        int16_t *r = res + 16 * b;
        int i, ac = 0;
        for (i = 1; i < 16; i++)
            ac |= l[i];
        if (!ac) {
            int16_t d0 = (int16_t)((int16_t)((int32_t)l[0] *
                                             aeg_dequant_ls[qp % 6][0] *
                                             (1 << (qp / 6))) + 32);
            int16_t v = (int16_t)(d0 >> 6);
            for (i = 0; i < 16; i++)
                r[i] = v;
        } else {
            aeg_idct_residual(l, qp, r);
        }
    }
    return AEG_OK;
}

/* the reference the fused device kernel is checked against: literally the
 * two operations one after the other */
static int cpu_tq_iq_batch(void *priv, const int16_t *res, int nblk, int qp,
                           int intra, int16_t *level, int32_t *nz,
                           int16_t *rec)
{
    cpu_tq_batch(priv, res, nblk, qp, intra, level, nz);
    return cpu_iq_idct_batch(priv, level, nblk, qp, rec);
}

/* the DC-split pair: exactly the per-block steps of code_luma_i16() and
 * code_chroma_run() */
static int cpu_tq_dc_batch(void *priv, const int16_t *res, int nblk, int qp,
                           int intra, int16_t *level, int32_t *nz, int32_t *dc)
{
    int b;
    (void)priv;
    for (b = 0; b < nblk; b++) {
        int16_t coef[16];
        int i, n = 0;
        aeg_transform_4x4(res + 16 * b, 4, coef);
        dc[b] = coef[0];
        coef[0] = 0;
        aeg_quant_4x4(coef, level + 16 * b, qp, intra);
        for (i = 0; i < 16; i++)
            n += level[16 * b + i] != 0;
        if (nz)
            nz[b] = n;
    }
    return AEG_OK;
}

static int cpu_iq_dc_batch(void *priv, const int16_t *level, const int32_t *dc,
                           int nblk, int qp, int16_t *res)
{
    int b;
    (void)priv;
    for (b = 0; b < nblk; b++)
        aeg_idct_residual_dc(level + 16 * b, qp, dc[b], res + 16 * b);
    return AEG_OK;
}

static int cpu_sse_batch(void *priv, const uint8_t *a, int sa,
                         const uint8_t *b, int sb, int w, int h,
                         uint64_t *out)
{
    int tw = w / 8, th = h / 8, tx, ty, x, y;
    uint64_t sse = 0;
    (void)priv;
    for (ty = 0; ty < th; ty++)
        for (tx = 0; tx < tw; tx++)
            for (y = 0; y < 8; y++)
                for (x = 0; x < 8; x++) {
                    int d = (int)a[(ty * 8 + y) * sa + tx * 8 + x] -
                            (int)b[(ty * 8 + y) * sb + tx * 8 + x];
                    sse += (uint64_t)(d * d);
                }
    *out = sse;
    return AEG_OK;
}

/* One fractional-pel plane of the reference picture: the same function the
 * encoder's plane cache uses when the stage stays on the CPU, so the two
 * implementations of the operation cannot drift apart. */
static int cpu_interp_plane(void *priv, const uint8_t *ref, int rstride,
                            int w, int h, int which, uint8_t *dst, int dstride)
{
    (void)priv;
    aeg_hpel_plane_fill(dst, dstride, ref, rstride, w, h, which);
    return AEG_OK;
}

/* ------------------------------------------------------------------ */
/* Fractional-pel SATD batch (the subpel stage of the motion search)   */
/* ------------------------------------------------------------------ */
/* The reference picture and its planes are registered once per picture, like a
 * GPU would keep them in device memory; the CPU "registration" is the pointer
 * set itself.  It is per thread: with frame threads two pictures run their
 * searches at the same time, each against its own reference, and the
 * registration and the batches of one picture always run on one thread. */
static __thread struct {
    const uint8_t *base[4];
    int            stride;
} g_cpu_ref;

static int cpu_ref_upload(void *priv, const uint8_t *const planes[4],
                          int pstride, int w, int h)
{
    (void)priv; (void)w; (void)h;
    g_cpu_ref.base[0] = planes[0];
    g_cpu_ref.base[1] = planes[1];
    g_cpu_ref.base[2] = planes[2];
    g_cpu_ref.base[3] = planes[3];
    g_cpu_ref.stride = pstride;
    return AEG_OK;
}

static int cpu_satd_batch(void *priv, const uint8_t *cur, int cstride, int bw,
                          int bh, const int16_t *pos, int count, int32_t *out)
{
    aeg_mc_ctx mc;
    uint8_t buf[16 * 16];
    int i;
    (void)priv;
    if (bw > 16 || bh > 16)
        return AEG_EINVAL;
    mc.base[0] = g_cpu_ref.base[0];
    mc.base[1] = g_cpu_ref.base[1];
    mc.base[2] = g_cpu_ref.base[2];
    mc.base[3] = g_cpu_ref.base[3];
    mc.buf = NULL;
    mc.stride = g_cpu_ref.stride;
    mc.ready = 1;
    for (i = 0; i < count; i++) {
        const int bx = pos[4 * i + 0], by = pos[4 * i + 1];
        const int mvx = pos[4 * i + 2], mvy = pos[4 * i + 3];
        aeg_mc_luma_p(buf, bw, &mc,
                      g_cpu_ref.base[0] + (ptrdiff_t)(by + (mvy >> 2)) * mc.stride +
                      bx + (mvx >> 2),
                      bw, bh, mvx & 3, mvy & 3);
        out[i] = aeg_satd(bw, bh, buf, bw, cur + (ptrdiff_t)by * cstride + bx,
                          cstride);
    }
    return AEG_OK;
}

void aeg_be_cpu_ops(aeg_be_ops *ops)
{
    memset(ops, 0, sizeof(*ops));
    ops->backend = AEG_BACKEND_CPU;
    ops->sad_batch = cpu_sad_batch;
    ops->tq_batch = cpu_tq_batch;
    ops->iq_idct_batch = cpu_iq_idct_batch;
    ops->tq_iq_batch = cpu_tq_iq_batch;
    ops->tq_dc_batch = cpu_tq_dc_batch;
    ops->iq_dc_batch = cpu_iq_dc_batch;
    ops->sse_batch = cpu_sse_batch;
    ops->interp_plane = cpu_interp_plane;
    ops->ref_upload = cpu_ref_upload;
    ops->satd_batch = cpu_satd_batch;
    snprintf(ops->desc, sizeof(ops->desc), "cpu");
}

/* Implemented in backend_opencl.c / backend_cuda.c */
int aeg_be_opencl_create(aeg_be_ops *ops);
int aeg_be_cuda_create(aeg_be_ops *ops);

/* ------------------------------------------------------------------ */
/* Device guard                                                        */
/* ------------------------------------------------------------------ */
/* Every device backend is wrapped in this before the encoder sees it, for
 * three reasons the backends themselves do not handle:
 *
 *  1. Threads.  Frame threads and slice threads are copies of the encoder,
 *     and a backend keeps its staging buffers and kernel arguments in its
 *     instance, so two calls at once into one instance corrupt each other (on
 *     a real OpenCL device: a crash on the first picture with -t 2).  The
 *     guard has *lanes*: lane 0 is the backend as created, lane k a clone of
 *     it (aeg_be_ops.clone: same device, shared context and program, own
 *     queue, kernel objects and buffers), made when a thread first asks for
 *     it (aeg_be_lane).  Each lane has its own lock, so threads on different
 *     lanes run their device calls concurrently and threads that share a lane
 *     are serialised.  A backend without clone() maps every lane to lane 0.
 *  2. The registered reference (ref_upload, then satd_batch) is device state
 *     that outlives a call.  Two threads on one lane would read each other's
 *     reference, so each lane remembers, per thread, what that thread
 *     registered and re-uploads it before a batch whenever another thread's
 *     registration replaced it.
 *  3. A failing device call must not change the stream.  The encoder ignores
 *     the return codes of the batch operations (they cannot fail on the CPU),
 *     so the guard runs the CPU reference instead -- the same arithmetic, so
 *     the same bytes -- and keeps the device off from then on, on every lane. */
#define BE_LANES 16

typedef struct be_guard be_guard;

typedef struct be_lane {
    be_guard       *g;
    aeg_be_ops      dev;        /* this lane's device instance             */
    int             owned;      /* a clone made for this lane              */
    pthread_mutex_t mu;
    uint64_t        seq;        /* registrations so far                     */
    uint64_t        dev_seq;    /* the one resident on the device           */
} be_lane;

struct be_guard {
    aeg_be_ops      cpu;
    pthread_mutex_t mk;         /* lane creation                            */
    be_lane        *lane[BE_LANES];
    int             failed;     /* a device call failed: CPU from now on    */
};

static __thread struct {
    const be_lane  *l;
    uint64_t        seq;
    const uint8_t  *planes[4];
    int             stride, w, h;
} g_tls_ref[BE_LANES];

static int guard_failed(be_guard *g)
{
    return __atomic_load_n(&g->failed, __ATOMIC_ACQUIRE);
}

/* lock, or return 0 when the device is off and the CPU op has to run */
static int guard_enter(be_lane *l)
{
    if (guard_failed(l->g))
        return 0;
    pthread_mutex_lock(&l->mu);
    if (guard_failed(l->g)) {
        pthread_mutex_unlock(&l->mu);
        return 0;
    }
    return 1;
}

/* unlock; a failure switches the device off for good */
static int guard_leave(be_lane *l, int rc, const char *op)
{
    if (rc != AEG_OK && !__atomic_exchange_n(&l->g->failed, 1, __ATOMIC_ACQ_REL))
        aeg_log(AEG_LOG_WARNING, "aeg: device %s failed (%d), continuing on the CPU", op, rc);
    pthread_mutex_unlock(&l->mu);
    return rc == AEG_OK;
}

/* the slot of g_tls_ref for lane l.  A thread normally uses one lane; two
 * lanes on one slot (another encoder in the same thread, a hash collision)
 * evict each other, and an evicted registration only sends satd_batch to the
 * CPU reference -- the same bytes. */
static int tls_slot(const be_lane *l)
{
    return (int)(((uintptr_t)l >> 6) % BE_LANES);
}

static int g_sad_batch(void *priv, const uint8_t *cur, int cstride,
                       const uint8_t *ref, int rstride, int bw, int bh,
                       const int16_t *mv, int count, int32_t *out)
{
    be_lane *l = (be_lane *)priv;
    if (guard_enter(l) &&
        guard_leave(l, l->dev.sad_batch(l->dev.priv, cur, cstride, ref, rstride,
                                        bw, bh, mv, count, out), "sad_batch"))
        return AEG_OK;
    return l->g->cpu.sad_batch(l->g->cpu.priv, cur, cstride, ref, rstride, bw,
                               bh, mv, count, out);
}

static int g_tq_batch(void *priv, const int16_t *res, int nblk, int qp,
                      int intra, int16_t *level, int32_t *nz)
{
    be_lane *l = (be_lane *)priv;
    if (guard_enter(l) &&
        guard_leave(l, l->dev.tq_batch(l->dev.priv, res, nblk, qp, intra, level,
                                       nz), "tq_batch"))
        return AEG_OK;
    return l->g->cpu.tq_batch(l->g->cpu.priv, res, nblk, qp, intra, level, nz);
}

static int g_iq_idct_batch(void *priv, const int16_t *level, int nblk, int qp,
                           int16_t *res)
{
    be_lane *l = (be_lane *)priv;
    if (guard_enter(l) &&
        guard_leave(l, l->dev.iq_idct_batch(l->dev.priv, level, nblk, qp, res),
                    "iq_idct_batch"))
        return AEG_OK;
    return l->g->cpu.iq_idct_batch(l->g->cpu.priv, level, nblk, qp, res);
}

static int g_tq_iq_batch(void *priv, const int16_t *res, int nblk, int qp,
                         int intra, int16_t *level, int32_t *nz, int16_t *rec)
{
    be_lane *l = (be_lane *)priv;
    if (guard_enter(l) &&
        guard_leave(l, l->dev.tq_iq_batch(l->dev.priv, res, nblk, qp, intra,
                                          level, nz, rec), "tq_iq_batch"))
        return AEG_OK;
    return l->g->cpu.tq_iq_batch(l->g->cpu.priv, res, nblk, qp, intra, level,
                                 nz, rec);
}

static int g_tq_dc_batch(void *priv, const int16_t *res, int nblk, int qp,
                         int intra, int16_t *level, int32_t *nz, int32_t *dc)
{
    be_lane *l = (be_lane *)priv;
    if (guard_enter(l) &&
        guard_leave(l, l->dev.tq_dc_batch(l->dev.priv, res, nblk, qp, intra,
                                          level, nz, dc), "tq_dc_batch"))
        return AEG_OK;
    return l->g->cpu.tq_dc_batch(l->g->cpu.priv, res, nblk, qp, intra, level,
                                 nz, dc);
}

static int g_iq_dc_batch(void *priv, const int16_t *level, const int32_t *dc,
                         int nblk, int qp, int16_t *res)
{
    be_lane *l = (be_lane *)priv;
    if (guard_enter(l) &&
        guard_leave(l, l->dev.iq_dc_batch(l->dev.priv, level, dc, nblk, qp, res),
                    "iq_dc_batch"))
        return AEG_OK;
    return l->g->cpu.iq_dc_batch(l->g->cpu.priv, level, dc, nblk, qp, res);
}

static int g_interp_plane(void *priv, const uint8_t *ref, int rstride, int w,
                          int h, int which, uint8_t *dst, int dstride)
{
    be_lane *l = (be_lane *)priv;
    if (guard_enter(l) &&
        guard_leave(l, l->dev.interp_plane(l->dev.priv, ref, rstride, w, h,
                                           which, dst, dstride),
                    "interp_plane"))
        return AEG_OK;
    return l->g->cpu.interp_plane(l->g->cpu.priv, ref, rstride, w, h, which,
                                  dst, dstride);
}

static int g_sse_batch(void *priv, const uint8_t *a, int sa, const uint8_t *b,
                       int sb, int w, int h, uint64_t *out)
{
    be_lane *l = (be_lane *)priv;
    if (guard_enter(l) &&
        guard_leave(l, l->dev.sse_batch(l->dev.priv, a, sa, b, sb, w, h, out),
                    "sse_batch"))
        return AEG_OK;
    return l->g->cpu.sse_batch(l->g->cpu.priv, a, sa, b, sb, w, h, out);
}

static int g_ref_upload(void *priv, const uint8_t *const planes[4], int pstride,
                        int w, int h)
{
    be_lane *l = (be_lane *)priv;
    int k, t = tls_slot(l);
    /* the CPU registration too, so that a fallback has a reference */
    l->g->cpu.ref_upload(l->g->cpu.priv, planes, pstride, w, h);
    for (k = 0; k < 4; k++)
        g_tls_ref[t].planes[k] = planes[k];
    g_tls_ref[t].stride = pstride;
    g_tls_ref[t].w = w;
    g_tls_ref[t].h = h;
    g_tls_ref[t].l = l;
    g_tls_ref[t].seq = 0;
    if (!guard_enter(l))
        return AEG_OK;
    k = l->dev.ref_upload(l->dev.priv, planes, pstride, w, h);
    if (k == AEG_OK)
        l->dev_seq = g_tls_ref[t].seq = ++l->seq;
    guard_leave(l, k, "ref_upload");
    return AEG_OK;
}

static int g_satd_batch(void *priv, const uint8_t *cur, int cstride, int bw,
                        int bh, const int16_t *pos, int count, int32_t *out)
{
    be_lane *l = (be_lane *)priv;
    int t = tls_slot(l);
    if (g_tls_ref[t].l == l && guard_enter(l)) {
        int rc = AEG_OK;
        if (g_tls_ref[t].seq != l->dev_seq) {
            /* another thread on this lane registered its reference since */
            rc = l->dev.ref_upload(l->dev.priv, g_tls_ref[t].planes,
                                   g_tls_ref[t].stride, g_tls_ref[t].w,
                                   g_tls_ref[t].h);
            if (rc == AEG_OK)
                l->dev_seq = g_tls_ref[t].seq = ++l->seq;
        }
        if (rc == AEG_OK)
            rc = l->dev.satd_batch(l->dev.priv, cur, cstride, bw, bh, pos, count,
                                   out);
        if (guard_leave(l, rc, "satd_batch"))
            return AEG_OK;
    }
    return l->g->cpu.satd_batch(l->g->cpu.priv, cur, cstride, bw, bh, pos, count,
                                out);
}

static void g_sync(void *priv)
{
    be_lane *l = (be_lane *)priv;
    if (!guard_enter(l))
        return;
    if (l->dev.sync)
        l->dev.sync(l->dev.priv);
    pthread_mutex_unlock(&l->mu);
}

static be_lane *lane_new(be_guard *g, const aeg_be_ops *dev, int owned)
{
    be_lane *l = (be_lane *)calloc(1, sizeof(*l));
    if (!l)
        return NULL;
    l->g = g;
    l->dev = *dev;
    l->owned = owned;
    pthread_mutex_init(&l->mu, NULL);
    return l;
}

static void g_destroy(void *priv)
{
    be_lane *l0 = (be_lane *)priv;
    be_guard *g;
    int k;
    if (!l0)
        return;
    g = l0->g;
    /* the clones first: they share the first instance's context */
    for (k = BE_LANES - 1; k >= 0; k--) {
        be_lane *l = g->lane[k];
        if (!l || (k > 0 && l == l0))
            continue;
        if ((k == 0 || l->owned) && l->dev.destroy)
            l->dev.destroy(l->dev.priv);
        pthread_mutex_destroy(&l->mu);
        free(l);
    }
    pthread_mutex_destroy(&g->mk);
    free(g);
}

static void g_lane_destroy(void *priv)
{
    (void)priv;                 /* lanes go with lane 0 (g_destroy) */
}

void aeg_be_lane(const aeg_be_ops *ops, int k, aeg_be_ops *out)
{
    be_lane *l0, *l;
    be_guard *g;

    *out = *ops;
    if (!ops->priv || (ops->destroy != g_destroy &&
                       ops->destroy != g_lane_destroy))
        return;                 /* the CPU backend: nothing to split */
    g = ((be_lane *)ops->priv)->g;
    l0 = g->lane[0];
    k = k < 0 ? 0 : k % BE_LANES;
    {
        /* AEG_LANES=0: every thread on lane 0, serialised (the measurement
         * of what the lanes buy, and a way out should a driver mishandle
         * several queues) */
        const char *ev = getenv("AEG_LANES");
        if (ev && atoi(ev) == 0)
            k = 0;
    }
    pthread_mutex_lock(&g->mk);
    if (!g->lane[k]) {
        aeg_be_ops c;
        l = NULL;
        if (!guard_failed(g) && l0->dev.clone &&
            l0->dev.clone(l0->dev.priv, &c) == AEG_OK) {
            l = lane_new(g, &c, 1);
            if (!l && c.destroy)
                c.destroy(c.priv);
        }
        if (l)
            aeg_log(AEG_LOG_DEBUG, "aeg: device lane %d: %s", k, c.desc);
        else if (l0->dev.clone)
            aeg_log(AEG_LOG_DEBUG, "aeg: device lane %d is lane 0 (%s)", k, aeg_last_error());
        g->lane[k] = l ? l : l0;
    }
    l = g->lane[k];
    pthread_mutex_unlock(&g->mk);
    /* every table has the same wrappers; only priv and destroy differ */
    if (l != ops->priv) {
        out->priv = l;
        out->destroy = g_lane_destroy;
    }
}

/* Wrap a freshly created device backend; an operation the device does not
 * provide stays absent, exactly as before. */
static int be_guard_wrap(aeg_be_ops *ops)
{
    be_guard *g = (be_guard *)calloc(1, sizeof(*g));
    be_lane *l0;
    if (!g)
        return AEG_ENOMEM;
    aeg_be_cpu_ops(&g->cpu);
    pthread_mutex_init(&g->mk, NULL);
    l0 = lane_new(g, ops, 0);
    if (!l0) {
        pthread_mutex_destroy(&g->mk);
        free(g);
        return AEG_ENOMEM;
    }
    g->lane[0] = l0;
    ops->priv = l0;
    if (ops->sad_batch) ops->sad_batch = g_sad_batch;
    if (ops->tq_batch) ops->tq_batch = g_tq_batch;
    if (ops->iq_idct_batch) ops->iq_idct_batch = g_iq_idct_batch;
    if (ops->tq_iq_batch) ops->tq_iq_batch = g_tq_iq_batch;
    if (ops->tq_dc_batch && ops->iq_dc_batch) {
        ops->tq_dc_batch = g_tq_dc_batch;
        ops->iq_dc_batch = g_iq_dc_batch;
    } else {
        ops->tq_dc_batch = NULL;
        ops->iq_dc_batch = NULL;
    }
    if (ops->interp_plane) ops->interp_plane = g_interp_plane;
    if (ops->sse_batch) ops->sse_batch = g_sse_batch;
    if (ops->ref_upload && ops->satd_batch) {
        ops->ref_upload = g_ref_upload;
        ops->satd_batch = g_satd_batch;
    } else {
        ops->ref_upload = NULL;
        ops->satd_batch = NULL;
    }
    ops->sync = g_sync;
    ops->destroy = g_destroy;
    ops->clone = NULL;
    return AEG_OK;
}

static int be_create_raw(int which, aeg_be_ops *ops);

int aeg_be_create(int which, aeg_be_ops *ops)
{
    int rc = be_create_raw(which, ops);
    if (ops->backend != AEG_BACKEND_CPU && be_guard_wrap(ops) != AEG_OK) {
        if (ops->destroy)
            ops->destroy(ops->priv);
        aeg_be_cpu_ops(ops);
        return AEG_EBACKEND;
    }
    return rc;
}

static int be_create_raw(int which, aeg_be_ops *ops)
{
    aeg_be_cpu_ops(ops);
    switch (which) {
    case AEG_BACKEND_CPU:
        return AEG_OK;
    case AEG_BACKEND_OPENCL:
        if (aeg_be_opencl_create(ops) == AEG_OK)
            return AEG_OK;
        aeg_be_cpu_ops(ops);
        aeg_set_last_error("OpenCL backend unavailable, using CPU");
        return AEG_EBACKEND;
    case AEG_BACKEND_CUDA:
        if (aeg_be_cuda_create(ops) == AEG_OK)
            return AEG_OK;
        aeg_be_cpu_ops(ops);
        aeg_set_last_error("CUDA backend unavailable, using CPU");
        return AEG_EBACKEND;
    case AEG_BACKEND_AUTO:
    default:
        if (aeg_be_cuda_create(ops) == AEG_OK)
            return AEG_OK;
        aeg_be_cpu_ops(ops);
        if (aeg_be_opencl_create(ops) == AEG_OK)
            return AEG_OK;
        aeg_be_cpu_ops(ops);
        return AEG_OK;
    }
}

/* ------------------------------------------------------------------ */
/* Dispatcher                                                          */
/* ------------------------------------------------------------------ */
void aeg_dispatch_init(aeg_dispatcher *d, int backend, const aeg_params *p)
{
    int i;
    memset(d, 0, sizeof(*d));
    aeg_be_create(backend, &d->ops);
    d->have_gpu = d->ops.backend != AEG_BACKEND_CPU &&
                  d->ops.tq_batch != NULL;
    for (i = 0; i < AEG_STAGE_COUNT; i++) {
        d->forced[i] = p ? p->stage_backend[i] : -1;
        d->use_gpu[i] = 0;
    }
    /* Stages that are inherently serial (they consume the result of the
     * previous macroblock as they go) are always kept on the CPU: the
     * latency of a device round trip would be paid per macroblock. */
    d->use_gpu[AEG_STAGE_DEBLOCK] = 0;
    d->mu = malloc(sizeof(pthread_mutex_t));
    if (d->mu)
        pthread_mutex_init((pthread_mutex_t *)d->mu, NULL);
    d->launch_ns = 5 * 1000;      /* refined by aeg_backend_calibrate() */
    d->min_items_for_gpu = 256;
    aeg_log(AEG_LOG_INFO, "aeg: compute backend = %s (%s)", aeg_backend_name(d->ops.backend),
            d->ops.desc);
}

void aeg_dispatch_destroy(aeg_dispatcher *d)
{
    if (d->ops.destroy)
        d->ops.destroy(d->ops.priv);
    d->ops.destroy = NULL;
    d->ops.priv = NULL;
    if (d->mu && !d->root) {
        pthread_mutex_destroy((pthread_mutex_t *)d->mu);
        free(d->mu);
    }
    d->mu = NULL;
}

/* the dispatcher measurements go to, locked */
static aeg_dispatcher *disp_lock(aeg_dispatcher *d)
{
    aeg_dispatcher *r = d->root ? d->root : d;
    if (r->mu)
        pthread_mutex_lock((pthread_mutex_t *)r->mu);
    return r;
}

static void disp_unlock(aeg_dispatcher *r)
{
    if (r->mu)
        pthread_mutex_unlock((pthread_mutex_t *)r->mu);
}

void aeg_dispatch_lock(aeg_dispatcher *d)
{
    disp_lock(d);
}

void aeg_dispatch_unlock(aeg_dispatcher *d)
{
    disp_unlock(d->root ? d->root : d);
}

static void dispatch_record(aeg_dispatcher *d, int stage, int on_gpu,
                            int64_t ns)
{
    int64_t *slot;
    slot = on_gpu ? &d->gpu_ns[stage][1] : &d->cpu_ns[stage][1];
    if (*slot == 0)
        *slot = ns;
    else
        *slot = (*slot * 7 + ns) / 8;      /* EMA */
    d->cpu_ns[stage][0] += on_gpu ? 0 : ns;
    d->gpu_ns[stage][0] += on_gpu ? ns : 0;
}

void aeg_dispatch_record(aeg_dispatcher *d, int stage, int on_gpu, int64_t ns)
{
    if (stage < 0 || stage >= AEG_STAGE_COUNT)
        return;
    d = disp_lock(d);
    dispatch_record(d, stage, on_gpu, ns);
    disp_unlock(d);
}

static void dispatch_update(aeg_dispatcher *d)
{
    int i;
    d->frames_measured++;
    for (i = 0; i < AEG_STAGE_COUNT; i++) {
        if (d->forced[i] >= 0) {
            d->use_gpu[i] = d->forced[i] == AEG_BACKEND_OPENCL ||
                            d->forced[i] == AEG_BACKEND_CUDA;
            if (d->use_gpu[i] && !d->have_gpu)
                d->use_gpu[i] = 0;
            continue;
        }
        if (!d->have_gpu || i == AEG_STAGE_DEBLOCK) {
            d->use_gpu[i] = 0;
            continue;
        }
        /* The device form of the motion search is a different search (the
         * whole-frame rounds), so placing it by timing would make the stream
         * depend on how fast the machine happened to be.  Only a -s override
         * (or AEG_SEED / AEG_PARFULL) puts it on a device. */
        if (i == AEG_STAGE_MOTION_EST) {
            d->use_gpu[i] = 0;
            continue;
        }
        /* the coefficient stages are placed by aeg_dispatch_coef_pick() */
        if (i == AEG_STAGE_TRANSFORM || i == AEG_STAGE_IQUANT_IDCT)
            continue;
        /* Only switch to the GPU when the measurement is unambiguous
         * (>= 10% faster) and the frame had enough work to amortise a
         * launch; this includes the launch cost in the comparison. */
        if (d->gpu_ns[i][1] > 0 && d->cpu_ns[i][1] > 0) {
            int64_t gpu = d->gpu_ns[i][1] + d->launch_ns;
            if (gpu * 100 < d->cpu_ns[i][1] * 90)
                d->use_gpu[i] = 1;
            else if (d->cpu_ns[i][1] * 100 < gpu * 90)
                d->use_gpu[i] = 0;
        }
        d->cpu_ns[i][0] = 0;
        d->gpu_ns[i][0] = 0;
    }
}

void aeg_dispatch_update(aeg_dispatcher *d)
{
    if (d->root)
        return;
    disp_lock(d);
    dispatch_update(d);
    disp_unlock(d);
}

/* Placing the coefficient stages.
 *
 * Timing the batch calls themselves cannot decide this one: the CPU form of
 * the stages is not a batch at all but sixteen SSE2 transforms inline in each
 * candidate's coding, with nothing to time per call, and the batched form adds
 * host work of its own around the launches (the gather, the scalar
 * reconstruction add).  What can be compared is the whole picture: both
 * placements code exactly the same stream -- the batched paths are
 * byte-identical to the serial one -- so every other part of the picture's
 * work is the same, and the difference in its time per macroblock *is* the
 * cost of the placement.  So the first pictures of each kind alternate between
 * the two, and once both have been seen enough the faster one stays, with
 * the slower one re-tried every AEG_COEF_RETRY pictures in case the balance
 * moved (another process on the GPU, a resolution change).
 *
 * The first picture on each side is not counted: it pays one-off costs (the
 * OpenCL runtime compiles a kernel at its first launch, the buffers are
 * allocated) that would otherwise decide the comparison. */
#define AEG_COEF_PROBE 3        /* counted pictures per placement and kind */
#define AEG_COEF_RETRY 64

static int dispatch_coef_pick(aeg_dispatcher *d, int kind)
{
    if (kind < 0 || kind > 2)
        return 0;
    if (d->forced[AEG_STAGE_TRANSFORM] >= 0 ||
        d->forced[AEG_STAGE_IQUANT_IDCT] >= 0)
        return d->use_gpu[AEG_STAGE_TRANSFORM] &&
               d->use_gpu[AEG_STAGE_IQUANT_IDCT];
    if (!d->have_gpu || kind == 0)
        return 0;               /* an I picture has no inter candidates */
    if (!d->coef_decided[kind])
        return d->coef_n[kind][1] < d->coef_n[kind][0];
    if (++d->coef_since[kind] >= AEG_COEF_RETRY) {
        d->coef_since[kind] = 0;
        return !d->coef_dev[kind];
    }
    return d->coef_dev[kind];
}

int aeg_dispatch_coef_pick(aeg_dispatcher *d, int kind)
{
    aeg_dispatcher *r = disp_lock(d);
    int dev = dispatch_coef_pick(r, kind);
    disp_unlock(r);
    return dev;
}

static void dispatch_coef_done(aeg_dispatcher *d, int kind, int dev,
                               int64_t ns, int mbs)
{
    int64_t per, *slot;
    if (kind <= 0 || kind > 2 || mbs <= 0 || !d->have_gpu ||
        d->forced[AEG_STAGE_TRANSFORM] >= 0 ||
        d->forced[AEG_STAGE_IQUANT_IDCT] >= 0)
        return;
    dev = dev != 0;
    per = ns / mbs;
    slot = &d->coef_ns[kind][dev];
    if (d->coef_n[kind][dev]++ == 0)
        return;                 /* the warm-up picture, see above */
    *slot = *slot ? (*slot * 3 + per) / 4 : per;
    if (d->coef_n[kind][0] > 1 && d->coef_n[kind][1] > 1) {
        int64_t c = d->coef_ns[kind][0], g = d->coef_ns[kind][1];
        int was = d->coef_dev[kind], first = !d->coef_decided[kind];
        /* settle after AEG_COEF_PROBE counted pictures a side -- or after the
         * first one when the two are a factor of two apart, which no amount
         * of content variation between neighbouring pictures explains */
        if (first && g < 2 * c && c < 2 * g &&
            (d->coef_n[kind][0] <= AEG_COEF_PROBE ||
             d->coef_n[kind][1] <= AEG_COEF_PROBE))
            return;
        d->coef_decided[kind] = 1;
        if (g * 100 < c * 90)
            d->coef_dev[kind] = 1;
        else if (c * 100 < g * 90)
            d->coef_dev[kind] = 0;
        if (d->coef_dev[kind] != was || first)
            aeg_log(AEG_LOG_DEBUG, "aeg: coefficient stages of %c pictures on the %s "
                    "(%.2f us/MB on the CPU, %.2f us/MB on the device)",
                    kind == 1 ? 'P' : 'B', d->coef_dev[kind] ? "device" : "CPU",
                    c / 1000.0, g / 1000.0);
    }
}

void aeg_dispatch_coef_done(aeg_dispatcher *d, int kind, int dev, int64_t ns,
                            int mbs)
{
    aeg_dispatcher *r = disp_lock(d);
    dispatch_coef_done(r, kind, dev, ns, mbs);
    disp_unlock(r);
}

/* ------------------------------------------------------------------ */
/* Device enumeration                                                  */
/* ------------------------------------------------------------------ */
int aeg_query_devices(aeg_backend backend, aeg_device_info *out, int max);
int aeg_calibrate(aeg_backend backend);

int aeg_query_devices(aeg_backend backend, aeg_device_info *out, int max)
{
    return aeg_backend_query(backend, out, max);
}

int aeg_calibrate(aeg_backend backend)
{
    return aeg_backend_calibrate(backend);
}

int aeg_backend_query(aeg_backend b, aeg_device_info *out, int max)
{
    int n = 0;
    if (b == AEG_BACKEND_CPU || b == AEG_BACKEND_AUTO) {
        if (n < max) {
            memset(&out[n], 0, sizeof(out[n]));
            out[n].backend = AEG_BACKEND_CPU;
            snprintf(out[n].name, sizeof(out[n].name), "CPU (%d cores)",
                     aeg_cpu_count());
            snprintf(out[n].vendor, sizeof(out[n].vendor), "host");
            out[n].is_gpu = 0;
            out[n].max_compute_units = aeg_cpu_count();
        }
        n++;
        if (b == AEG_BACKEND_CPU)
            return n;
    }
    n += aeg_backend_query_opencl(out + n, max - n > 0 ? max - n : 0);
    n += aeg_backend_query_cuda(out + n, max - n > 0 ? max - n : 0);
    return n;
}

/* Launch + transfer overhead measured by the last calibration, in ns. */
static int64_t g_launch_ns;

int64_t aeg_backend_launch_ns(void)
{
    return g_launch_ns;
}

int aeg_backend_calibrate(aeg_backend b)
{
    aeg_be_ops ops;
    int i;
    int64_t best = 0;
    int16_t res[64], level[64];
    int32_t nz[4];

    aeg_be_create(b, &ops);
    if (ops.backend == AEG_BACKEND_CPU)
        return AEG_EBACKEND;
    for (i = 0; i < 64; i++)
        res[i] = (int16_t)(i * 3 - 40);
    /* A batch far smaller than the device's comfortable size measures the
     * fixed cost of a call: kernel launch plus the two transfers. */
    for (i = 0; i < 8; i++) {
        int64_t t0 = aeg_now_ns();
        ops.tq_batch(ops.priv, res, 4, 26, 1, level, nz);
        if (ops.sync)
            ops.sync(ops.priv);
        else
            (void)0;
        {
            int64_t dt = aeg_now_ns() - t0;
            if (i && (!best || dt < best))
                best = dt;
        }
    }
    if (ops.destroy)
        ops.destroy(ops.priv);
    g_launch_ns = best;
    aeg_log(AEG_LOG_INFO, "aeg: %s launch+transfer overhead ~%lld us",
            aeg_backend_name(ops.backend), (long long)(best / 1000));
    return AEG_OK;
}

/* Weak stubs so the library links without OpenCL/CUDA. */
#if !defined(AEG_HAVE_OPENCL)
int aeg_be_opencl_create(aeg_be_ops *ops)
{
    (void)ops;
    return AEG_EBACKEND;
}
int aeg_backend_query_opencl(aeg_device_info *out, int max)
{
    (void)out; (void)max;
    return 0;
}
#endif
#if !defined(AEG_HAVE_CUDA)
int aeg_be_cuda_create(aeg_be_ops *ops)
{
    (void)ops;
    return AEG_EBACKEND;
}
int aeg_backend_query_cuda(aeg_device_info *out, int max)
{
    (void)out; (void)max;
    return 0;
}
int aeg_cuda_compile_check(int cc, char *log, size_t logsz, size_t *ptx_size)
{
    (void)cc; (void)log; (void)logsz; (void)ptx_size;
    return AEG_EBACKEND;
}
#endif
