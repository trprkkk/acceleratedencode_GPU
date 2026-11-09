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

#define AEG_ME_PIC_CUR 0
#define AEG_ME_PIC_REF 1
#define AEG_ME_PIC_SLOTS 16

/* Device-resident lookahead pictures (M-3.3a): the lowres ring is at most
 * AEG_RING entries; the backend keeps one luma plane per slot. */
#define AEG_LOWRES_SLOTS 96

typedef struct aeg_be_ops {
    void *priv;
    int   backend;          /* AEG_BACKEND_* actually in use             */
    int   is_gpu;           /* a real GPU (0: a CPU device such as PoCL)  */
    char  desc[192];

    /* SAD of `count` block pairs.
     * mv[] holds four int16 per candidate: the source block position (cx, cy)
     * in `cur` and the displacement of the reference block (dx, dy) in `ref` --
     * both relative to the passed pointers.  out[] receives the SADs.  The
     * source position has to be part of the candidate because a batch covers
     * many macroblocks: a whole picture's search is one call, so `cur` is the
     * picture and not one block. */
    int (*sad_batch)(void *priv, const pixel *cur, int cstride,
                     const pixel *ref, int rstride, int bw, int bh,
                     const int16_t *mv, int count, int32_t *out);

    /* Forward transform + quantisation of `nblk` 4x4 blocks.
     * res is 16 int16 per block (raster), level receives 16 int16 per block. */
    int (*tq_batch)(void *priv, const int16_t *res, int nblk, int qp,
                    int intra, dctcoef *level, int32_t *nz);

    /* Inverse quantisation + inverse transform of `nblk` 4x4 blocks.
     * Writes the reconstructed residual (signed) into `res`. */
    int (*iq_idct_batch)(void *priv, const dctcoef *level, int nblk, int qp,
                         int16_t *res);

    /* tq_batch() and then iq_idct_batch() on its levels, in one call: on a
     * device one transfer each way and one launch instead of two round
     * trips.  `level`/`nz` are tq_batch()'s outputs, `rec` iq_idct_batch()'s
     * of the same levels.  Optional (NULL: the caller makes the two calls). */
    int (*tq_iq_batch)(void *priv, const int16_t *res, int nblk, int qp,
                       int intra, dctcoef *level, int32_t *nz, int16_t *rec);

    /* The DC-split pair for Intra_16x16 and chroma, whose block DCs go
     * through a Hadamard of their own (on the host: four or sixteen values).
     * tq_dc_batch(): tq_batch() with each block's transform DC written to
     * `dc` (the coefficient before quantisation) and its level forced to 0,
     * `nz` counting the AC levels.  iq_dc_batch(): iq_idct_batch() with the
     * dequantised DC of block b replaced by dc[b] (as aeg_idct_add_block()'s
     * dc argument).  Optional, both or neither. */
    int (*tq_dc_batch)(void *priv, const int16_t *res, int nblk, int qp,
                       int intra, dctcoef *level, int32_t *nz, int32_t *dc);
    int (*iq_dc_batch)(void *priv, const dctcoef *level, const int32_t *dc,
                       int nblk, int qp, int16_t *res);

    /* One fractional-pel plane of a reference picture (H.264 8.4.2.2.1),
     * `which` 1 = horizontal, 2 = vertical, 3 = both.  `ref` points at picture
     * (0,0) in a padded buffer (AEG_PAD samples of edge replication on every
     * side); `dst` receives the same geometry, filled exactly as
     * aeg_hpel_plane_fill() does.  This is the per-picture interpolation pass
     * the motion search reads all its subpel candidates from. */
    int (*interp_plane)(void *priv, const pixel *ref, int rstride,
                        int w, int h, int which, pixel *dst, int dstride);
    int (*sse_batch)(void *priv, const pixel *a, int sa, const pixel *b,
                     int sb, int w, int h, uint64_t *out);

    /* Register the reference picture for the calls below: planes[0] is the
     * integer picture, planes[1..3] the horizontal / vertical / both-axes
     * fractional planes (aeg_mc_ctx).  A GPU keeps them resident for the whole
     * picture, which is why this is a separate call and not an argument of
     * every batch. */
    int (*ref_upload)(void *priv, const pixel *const planes[4], int pstride,
                      int w, int h);

    /* SATD of a batch of fractional-pel candidates against `cur`.
     * pos holds four int16 per candidate: block x, block y (picture
     * coordinates of the block origin) and the quarter-pel motion vector
     * (mvx, mvy).  The samples come from the registered reference and its
     * planes, i.e. this is exactly aeg_mc_luma_p() followed by aeg_satd(). */
    int (*satd_batch)(void *priv, const pixel *cur, int cstride, int bw,
                      int bh, const int16_t *pos, int count, int32_t *out);

    /* Whole-picture integer motion search: `count` blocks of bw x bh, each
     * searched on its own (no block depends on another's result), so one call
     * covers a picture instead of one call per block.  job holds four int16
     * per block: block x, block y (picture coordinates of the block origin),
     * and the start vector in whole samples.  out_mv receives two int16 (the
     * vector, whole samples) and out_sad one int32 (its SAD) per block.
     *
     * The walk is defined by the CPU reference in backend.c and reproduced
     * exactly by the device kernel (tests/backend_check.c compares them on
     * random pictures): a nine-point descent at steps range/2 .. 1 followed by
     * a one-pel descent, strict comparison in a fixed order.  Because the two
     * are the same function, the encoder can place this stage on either side
     * without changing the stream. */
    int (*me_batch)(void *priv, const pixel *cur, int cstride,
                    const pixel *ref, int rstride, int bw, int bh, int range,
                    const int16_t *job, int count, int avail_rows,
                    int16_t *out_mv, int32_t *out_sad,
                    uint64_t cur_id, uint64_t ref_id);

    /* Device-resident luma pictures for me_batch (slot 0 = current source,
     * 1 = reference).  `id` identifies the content: the same id means the
     * bytes already on the device can be reused, a new one re-uploads.  A
     * backend without this op simply transfers in me_batch as before. */
    int (*me_pic_upload)(void *priv, int slot, const pixel *plane, int stride,
                         int w, int h, int rows, uint64_t id);

    /* Lowres ring (M-3.3a): upload one 8-bit lowres plane to a device slot,
     * then run SAD batches between two resident slots without re-uploading.
     * `id` identifies the content; the same id + size is a cache hit. */
    int (*lowres_upload)(void *priv, int slot, const uint8_t *plane, int stride,
                         int w, int h, uint64_t id);
    int (*lowres_sad_batch)(void *priv, int cur_slot, int ref_slot, int bw,
                            int bh, const int16_t *mv, int count, int32_t *out);
    /* M-3.3b: every (pair, block) in one grid.  pairs holds two slot numbers
     * per entry; one host call launches `iters` kernels (one per step) on
     * the lane stream and waits once.  mv/cost are npairs * mb_w * mb_h. */
    int (*lowres_batch_search)(void *priv, const int32_t *pairs, int npairs,
                               int mb_w, int mb_h, int iters,
                               int32_t *mv, int32_t *cost);
    /* M-3.2: fixed-point mbtree propagation for a whole planned list, one
     * host call and one sync.  prop is npics*nmb int32 (in/out). */
    int (*mbt_batch)(void *priv, int npics, int mb_w, int mb_h,
                     const int *intra, const int *inter, const uint8_t *kind,
                     const int *mv, const int *r0, const int *r1, int w0,
                     int32_t *prop);

    void (*sync)(void *priv);
    void (*destroy)(void *priv);

    /* A second instance on the same device for another thread: shares what
     * is expensive and read-only (OpenCL context and built program, CUDA
     * context and module) and owns everything a call writes (queue, kernel
     * objects -- clSetKernelArg is not thread-safe --, buffers, the
     * registered reference).  Raw backends only; the guard uses it for its
     * lanes (aeg_be_lane) and does not pass it on.  Optional. */
    int (*clone)(void *priv, struct aeg_be_ops *out);
} aeg_be_ops;

/* The operations of lane `k` of a backend, for a thread that should not wait
 * for the others' device calls (frame-thread worker k, slice thread k).  Lane
 * 0 is `ops` itself; lane k >= 1 is a clone made on first use, with its own
 * lock and its own registered reference, so calls of different lanes run
 * concurrently.  Without clone() (or if it fails) a lane is lane 0, which is
 * correct and serialised.  The CPU backend has no state to split: its ops are
 * returned as they are.  Lanes live as long as `ops`; do not destroy them. */
void aeg_be_lane(const aeg_be_ops *ops, int k, aeg_be_ops *out);

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
    int        have_gpu;             /* a device is present             */
    int        auto_ok;              /* ... and it may be chosen by timing
                                      *     (a real GPU: a CPU device is
                                      *     only ever used when forced)  */
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
    /* Copies of the encoder made for other threads (frame threads, slice
     * threads) carry a copy of the dispatcher.  Their measurements
     * (aeg_dispatch_record, aeg_dispatch_coef_done) go to `root`, the
     * encoder's own, under `mu`, which aeg_dispatch_init creates and every
     * copy shares; their aeg_dispatch_update is a no-op -- the decisions are
     * the root's, made by the thread that owns it, and a copy keeps the ones
     * it was made with.  NULL root: this is the root. */
    struct aeg_dispatcher *root;
    void      *mu;
} aeg_dispatcher;

void aeg_dispatch_init(aeg_dispatcher *d, int backend, const aeg_params *p);
/* Hold the root's lock, e.g. while copying the dispatcher for another thread
 * when copies made earlier may be reporting to it. */
void aeg_dispatch_lock(aeg_dispatcher *d);
void aeg_dispatch_unlock(aeg_dispatcher *d);
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
