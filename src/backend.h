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

/* lowres_chains(): the pictures, the chains and the fields of a batch */
typedef struct aeg_la_pic {
    const uint8_t *plane;   /* origin of a padded lowres plane            */
    int slot, tag;          /* residency key: ring slot, display number   */
} aeg_la_pic;

typedef struct aeg_la_chain {
    int cur;                /* index into pics                            */
    int first, count;       /* jobs[first .. first + count), in order     */
    const int *seed;        /* the first job's seed if it is seeded       */
} aeg_la_chain;

typedef struct aeg_la_job {
    int ref;                /* index into pics                            */
    int seeded;             /* seed = the previous field of the chain     */
    int num, den, range;    /* as lowres_search()                         */
    int *out;               /* 3 ints per block, as lowres_search()       */
} aeg_la_job;

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

    /* The lowres motion search of a whole field, aeg_lowres_search_seeded()
     * exactly (AEG_STAGE_LOOKAHEAD).  `cur` and `ref` point at the origins
     * of lowres planes allocated with AEG_LA_PAD samples of border on every
     * side (aeg_lowres_alloc), `stride` apart, w x h samples = (w/8) x (h/8)
     * blocks.  `seed` (NULL = none) is the field of the same picture at
     * distance n-1, scaled by num/den, `range` the vector bound.  out[]
     * receives two ints per block (the vector, half samples) and then one
     * per block (the cost), as the lookahead's field cache stores them.
     * Optional. */
    int (*lowres_search)(void *priv, const uint8_t *cur, const uint8_t *ref,
                         int stride, int w, int h, const int *seed, int num,
                         int den, int range, int *out);

    /* Many fields in one call: each chain is one picture's fields against a
     * run of references, in order, and a job marked `seeded` takes the
     * previous job's output of its chain as its seed (the first job of a
     * chain takes chain.seed).  Chains are independent of each other, so a
     * device runs them concurrently (one work-group each) while the fields
     * of a chain follow one another.  Every field is lowres_search() with
     * the same arguments; the output is the same ints, at jobs[i].out.
     * pics[] are the lowres pictures the chains refer to: plane origin as
     * for lowres_search(), and a ring `slot` (0..nslots-1) with a `tag`
     * (the picture's display number) under which a device may keep the
     * plane resident between calls -- a slot is uploaded again only when its
     * tag changes, so the caller must not reuse a (slot, tag) pair for
     * different samples.  Optional. */
    int (*lowres_chains)(void *priv, const aeg_la_pic *pics, int npics,
                         int nslots, const aeg_la_chain *chains, int nchains,
                         const aeg_la_job *jobs, int stride, int w, int h);

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
int aeg_be_create_dev(int which, int dev, aeg_be_ops *ops);

/* Per-stage dispatch ------------------------------------------------- */
typedef struct aeg_dispatcher {
    aeg_be_ops ops;                  /* accelerator (or CPU)            */
    int        have_gpu;
    int        use_gpu[AEG_STAGE_COUNT];   /* current decision           */
    int        forced[AEG_STAGE_COUNT];    /* -1 auto, else forced       */
    int64_t    cpu_ns[AEG_STAGE_COUNT][2]; /* [stage][last/ema]          */
    int64_t    gpu_ns[AEG_STAGE_COUNT][2];
    int64_t    bytes_moved[AEG_STAGE_COUNT];
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
    /* The lookahead's lowres search (aeg_dispatch_la_pick / _done): time
     * per 1024 blocks (moving average) on [0] the CPU and [1] the device,
     * calls measured, calls since the slower side was last re-tried. */
    int64_t    la_ns[2];
    int        la_n[2];
    int        la_since;
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

/* The lookahead's lowres search of one field (batch = 0), or of one
 * scheduling step's batch of fields (batch = 1): 1 = on the device.  Both
 * forms give the same fields, so this one is placed by timing, in time per
 * block searched: each side is probed a few times, then the faster stays
 * and the other is re-tried now and then. */
int  aeg_dispatch_la_pick(aeg_dispatcher *d, int batch);
/* ... and how long that field took, `blocks` lowres blocks, placed as `dev`. */
void aeg_dispatch_la_done(aeg_dispatcher *d, int dev, int64_t ns, int blocks);

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
