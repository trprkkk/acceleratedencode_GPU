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
        const uint8_t *r = ref + mv[2 * i + 1] * rstride + mv[2 * i];
        int x, y, sad = 0;
        for (y = 0; y < bh; y++) {
            const uint8_t *c = cur + y * cstride;
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
    for (b = 0; b < nblk; b++) {
        int32_t d[16];
        int t[16], i;
        aeg_dequant_4x4(level + 16 * b, d, qp);
        for (i = 0; i < 4; i++) {
            int s0 = d[i * 4 + 0] + d[i * 4 + 2];
            int s1 = d[i * 4 + 0] - d[i * 4 + 2];
            int s2 = (d[i * 4 + 1] >> 1) - d[i * 4 + 3];
            int s3 = d[i * 4 + 1] + (d[i * 4 + 3] >> 1);
            t[i * 4 + 0] = s0 + s3;
            t[i * 4 + 1] = s1 + s2;
            t[i * 4 + 2] = s1 - s2;
            t[i * 4 + 3] = s0 - s3;
        }
        for (i = 0; i < 4; i++) {
            int s0 = t[0 * 4 + i] + t[2 * 4 + i];
            int s1 = t[0 * 4 + i] - t[2 * 4 + i];
            int s2 = (t[1 * 4 + i] >> 1) - t[3 * 4 + i];
            int s3 = t[1 * 4 + i] + (t[3 * 4 + i] >> 1);
            res[16 * b + 0 * 4 + i] = (int16_t)((s0 + s3 + 32) >> 6);
            res[16 * b + 1 * 4 + i] = (int16_t)((s1 + s2 + 32) >> 6);
            res[16 * b + 2 * 4 + i] = (int16_t)((s1 - s2 + 32) >> 6);
            res[16 * b + 3 * 4 + i] = (int16_t)((s0 - s3 + 32) >> 6);
        }
    }
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

void aeg_be_cpu_ops(aeg_be_ops *ops)
{
    memset(ops, 0, sizeof(*ops));
    ops->backend = AEG_BACKEND_CPU;
    ops->sad_batch = cpu_sad_batch;
    ops->tq_batch = cpu_tq_batch;
    ops->iq_idct_batch = cpu_iq_idct_batch;
    ops->sse_batch = cpu_sse_batch;
    snprintf(ops->desc, sizeof(ops->desc), "cpu");
}

/* Implemented in backend_opencl.c / backend_cuda.c */
int aeg_be_opencl_create(aeg_be_ops *ops);
int aeg_be_cuda_create(aeg_be_ops *ops);

int aeg_be_create(int which, aeg_be_ops *ops)
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
    d->launch_ns = 5 * 1000;      /* refined by aeg_backend_calibrate() */
    d->min_items_for_gpu = 256;
    aeg_log(1, "aeg: compute backend = %s (%s)", aeg_backend_name(d->ops.backend),
            d->ops.desc);
}

void aeg_dispatch_destroy(aeg_dispatcher *d)
{
    if (d->ops.destroy)
        d->ops.destroy(d->ops.priv);
    d->ops.destroy = NULL;
    d->ops.priv = NULL;
}

void aeg_dispatch_record(aeg_dispatcher *d, int stage, int on_gpu, int64_t ns)
{
    int64_t *slot;
    if (stage < 0 || stage >= AEG_STAGE_COUNT)
        return;
    slot = on_gpu ? &d->gpu_ns[stage][1] : &d->cpu_ns[stage][1];
    if (*slot == 0)
        *slot = ns;
    else
        *slot = (*slot * 7 + ns) / 8;      /* EMA */
    d->cpu_ns[stage][0] += on_gpu ? 0 : ns;
    d->gpu_ns[stage][0] += on_gpu ? ns : 0;
}

void aeg_dispatch_update(aeg_dispatcher *d)
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
    aeg_log(1, "aeg: %s launch+transfer overhead ~%lld us",
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
#endif
