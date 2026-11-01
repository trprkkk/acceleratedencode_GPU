/*
 * Compute backend interface and per-stage dispatcher.
 *
 * A backend exposes a small number of *batch* operations.  Batching matters:
 * the thin operations of an encoder (a 4x4 transform, a 16x16 SAD) are far
 * cheaper than a kernel launch or a PCIe transfer, so each operation is
 * called with many work items at once, and the dispatcher decides per stage
 * and per frame whether the GPU version is actually faster than the CPU one.
 */
#ifndef AEG_BACKEND_H
#define AEG_BACKEND_H

#include "h264.h"

typedef struct aeg_be_ops {
    void *priv;
    int   backend;          /* AEG_BACKEND_* actually in use             */
    char  desc[192];

    /* SAD of `count` block pairs.
     * mv[] holds four int16 per candidate: the source block position (cx, cy)
     * in `cur` and the displacement of the reference block (dx, dy) in `ref` --
     * both relative to the passed pointers.  out[] receives the SADs.  The
     * source position has to be part of the candidate because a batch covers
     * many macroblocks: a whole picture's search is one call, so `cur` is the
     * picture and not one block. */
    int (*sad_batch)(void *priv, const uint8_t *cur, int cstride,
                     const uint8_t *ref, int rstride, int bw, int bh,
                     const int16_t *mv, int count, int32_t *out);

    /* Forward transform + quantisation of `nblk` 4x4 blocks.
     * res is 16 int16 per block (raster), level receives 16 int16 per block. */
    int (*tq_batch)(void *priv, const int16_t *res, int nblk, int qp,
                    int intra, int16_t *level, int32_t *nz);

    /* Inverse quantisation + inverse transform of `nblk` 4x4 blocks.
     * Writes the reconstructed residual (signed) into `res`. */
    int (*iq_idct_batch)(void *priv, const int16_t *level, int nblk, int qp,
                         int16_t *res);

    /* tq_batch() and then iq_idct_batch() on its levels, in one call: on a
     * device one transfer each way and one launch instead of two round
     * trips.  `level`/`nz` are tq_batch()'s outputs, `rec` iq_idct_batch()'s
     * of the same levels.  Optional (NULL: the caller makes the two calls). */
    int (*tq_iq_batch)(void *priv, const int16_t *res, int nblk, int qp,
                       int intra, int16_t *level, int32_t *nz, int16_t *rec);

    /* One fractional-pel plane of a reference picture (H.264 8.4.2.2.1),
     * `which` 1 = horizontal, 2 = vertical, 3 = both.  `ref` points at picture
     * (0,0) in a padded buffer (AEG_PAD samples of edge replication on every
     * side); `dst` receives the same geometry, filled exactly as
     * aeg_hpel_plane_fill() does.  This is the per-picture interpolation pass
     * the motion search reads all its subpel candidates from. */
    int (*interp_plane)(void *priv, const uint8_t *ref, int rstride,
                        int w, int h, int which, uint8_t *dst, int dstride);
    int (*sse_batch)(void *priv, const uint8_t *a, int sa, const uint8_t *b,
                     int sb, int w, int h, uint64_t *out);

    /* Register the reference picture for the calls below: planes[0] is the
     * integer picture, planes[1..3] the horizontal / vertical / both-axes
     * fractional planes (aeg_mc_ctx).  A GPU keeps them resident for the whole
     * picture, which is why this is a separate call and not an argument of
     * every batch. */
    int (*ref_upload)(void *priv, const uint8_t *const planes[4], int pstride,
                      int w, int h);

    /* SATD of a batch of fractional-pel candidates against `cur`.
     * pos holds four int16 per candidate: block x, block y (picture
     * coordinates of the block origin) and the quarter-pel motion vector
     * (mvx, mvy).  The samples come from the registered reference and its
     * planes, i.e. this is exactly aeg_mc_luma_p() followed by aeg_satd(). */
    int (*satd_batch)(void *priv, const uint8_t *cur, int cstride, int bw,
                      int bh, const int16_t *pos, int count, int32_t *out);

    void (*sync)(void *priv);
    void (*destroy)(void *priv);
} aeg_be_ops;

/* sad_batch() on a device: the candidates of one call cover a whole picture
 * (source positions anywhere in `cur`, reference positions anywhere in `ref`
 * including its padding, so negative), and the call does not say how large
 * either picture is.  A device backend therefore transfers exactly the span the
 * positions touch -- from the top-left sample any candidate reads to the
 * bottom-right one, whole rows in between -- and hands the kernel positions
 * relative to that span.  `out` receives the rebased positions (4 int16 per
 * candidate, same layout); the offsets are relative to the passed pointers. */
typedef struct aeg_sad_span {
    ptrdiff_t coff, roff;       /* first byte transferred, from cur / ref  */
    size_t    cbytes, rbytes;   /* bytes transferred                       */
} aeg_sad_span;

static inline void aeg_sad_span_rebase(const int16_t *mv, int count, int bw,
                                       int bh, int cstride, int rstride,
                                       int16_t *out, aeg_sad_span *sp)
{
    int i, cx0 = 0, cx1 = 0, cy0 = 0, cy1 = 0, rx0 = 0, rx1 = 0, ry0 = 0,
           ry1 = 0;
    for (i = 0; i < count; i++) {
        int cx = mv[4 * i + 0], cy = mv[4 * i + 1];
        int rx = mv[4 * i + 2], ry = mv[4 * i + 3];
        if (!i || cx < cx0) cx0 = cx;
        if (!i || cx > cx1) cx1 = cx;
        if (!i || cy < cy0) cy0 = cy;
        if (!i || cy > cy1) cy1 = cy;
        if (!i || rx < rx0) rx0 = rx;
        if (!i || rx > rx1) rx1 = rx;
        if (!i || ry < ry0) ry0 = ry;
        if (!i || ry > ry1) ry1 = ry;
    }
    for (i = 0; i < count; i++) {
        out[4 * i + 0] = (int16_t)(mv[4 * i + 0] - cx0);
        out[4 * i + 1] = (int16_t)(mv[4 * i + 1] - cy0);
        out[4 * i + 2] = (int16_t)(mv[4 * i + 2] - rx0);
        out[4 * i + 3] = (int16_t)(mv[4 * i + 3] - ry0);
    }
    sp->coff = (ptrdiff_t)cy0 * cstride + cx0;
    sp->roff = (ptrdiff_t)ry0 * rstride + rx0;
    sp->cbytes = (size_t)(cy1 - cy0 + bh - 1) * (size_t)cstride +
                 (size_t)(cx1 - cx0 + bw);
    sp->rbytes = (size_t)(ry1 - ry0 + bh - 1) * (size_t)rstride +
                 (size_t)(rx1 - rx0 + bw);
}

/* CPU implementations of every operation (always available). */
void aeg_be_cpu_ops(aeg_be_ops *ops);

/* Create a backend.  `which` may be AEG_BACKEND_CPU/OPENCL/CUDA or AUTO.
 * Returns 0 on success and fills ops; on failure returns AEG_EBACKEND and
 * fills ops with the CPU implementation. */
int aeg_be_create(int which, aeg_be_ops *ops);

/* Per-stage dispatch ------------------------------------------------- */
typedef struct aeg_dispatcher {
    aeg_be_ops ops;                  /* accelerator (or CPU)            */
    int        have_gpu;
    int        use_gpu[AEG_STAGE_COUNT];   /* current decision           */
    int        forced[AEG_STAGE_COUNT];    /* -1 auto, else forced       */
    int64_t    cpu_ns[AEG_STAGE_COUNT][2]; /* [stage][last/ema]          */
    int64_t    gpu_ns[AEG_STAGE_COUNT][2];
    int64_t    bytes_moved[AEG_STAGE_COUNT];
    int64_t    launch_ns;                  /* measured kernel launch cost */
    int64_t    min_items_for_gpu;
    int        frames_measured;
    /* The coefficient stages (transform + inverse) are placed by trial, per
     * picture kind (aeg_dispatch_coef_pick / _done): [kind][0 cpu, 1 device]
     * picture time per macroblock (moving average), pictures measured,
     * pictures since the losing placement was last re-tried, and the
     * current choice. */
    int64_t    coef_ns[3][2];
    int        coef_n[3][2];
    int        coef_since[3];
    int        coef_dev[3];
    int        coef_decided[3];
} aeg_dispatcher;

void aeg_dispatch_init(aeg_dispatcher *d, int backend, const aeg_params *p);
void aeg_dispatch_destroy(aeg_dispatcher *d);
/* Record the wall time spent in a stage on the CPU or the GPU this frame. */
void aeg_dispatch_record(aeg_dispatcher *d, int stage, int on_gpu, int64_t ns);
/* Recompute the per-stage decisions from the measurements of the frame. */
void aeg_dispatch_update(aeg_dispatcher *d);
/* The coefficient stages of one picture of `kind` (0 I, 1 P, 2 B): 1 = run
 * them on the device.  While a kind is being measured this alternates, later it
 * returns the faster placement and re-tries the other one now and then. */
int  aeg_dispatch_coef_pick(aeg_dispatcher *d, int kind);
/* ... and how long that picture took, `mbs` macroblocks, placed as `dev`. */
void aeg_dispatch_coef_done(aeg_dispatcher *d, int kind, int dev, int64_t ns,
                            int mbs);

/* Device enumeration (aeg_query_devices in the public API). */
int  aeg_backend_query(aeg_backend b, aeg_device_info *out, int max);
int  aeg_backend_query_opencl(aeg_device_info *out, int max);
int  aeg_backend_query_cuda(aeg_device_info *out, int max);
int  aeg_backend_calibrate(aeg_backend b);
/* Compile the CUDA kernels with NVRTC only (no GPU, no driver): see
 * backend_cuda.c.  AEG_EBACKEND without NVRTC or without a CUDA build. */
int  aeg_cuda_compile_check(int cc, char *log, size_t logsz, size_t *ptx_size);

/* Small timing helper used by the dispatcher. */
#define AEG_TIME_BEGIN(var) int64_t var = aeg_now_ns()
#define AEG_TIME_END(d, stage, on_gpu, var) \
    aeg_dispatch_record((d), (stage), (on_gpu), aeg_now_ns() - (var))

#endif /* AEG_BACKEND_H */
