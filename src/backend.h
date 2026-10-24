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

    /* SAD of `count` candidate blocks.
     * mv[] holds (x,y) integer-pel offsets, out[] receives the SADs. */
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

    void (*sync)(void *priv);
    void (*destroy)(void *priv);
} aeg_be_ops;

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
} aeg_dispatcher;

void aeg_dispatch_init(aeg_dispatcher *d, int backend, const aeg_params *p);
void aeg_dispatch_destroy(aeg_dispatcher *d);
/* Record the wall time spent in a stage on the CPU or the GPU this frame. */
void aeg_dispatch_record(aeg_dispatcher *d, int stage, int on_gpu, int64_t ns);
/* Recompute the per-stage decisions from the measurements of the frame. */
void aeg_dispatch_update(aeg_dispatcher *d);

/* Device enumeration (aeg_query_devices in the public API). */
int  aeg_backend_query(aeg_backend b, aeg_device_info *out, int max);
int  aeg_backend_query_opencl(aeg_device_info *out, int max);
int  aeg_backend_query_cuda(aeg_device_info *out, int max);
int  aeg_backend_calibrate(aeg_backend b);

/* Small timing helper used by the dispatcher. */
#define AEG_TIME_BEGIN(var) int64_t var = aeg_now_ns()
#define AEG_TIME_END(d, stage, on_gpu, var) \
    aeg_dispatch_record((d), (stage), (on_gpu), aeg_now_ns() - (var))

#endif /* AEG_BACKEND_H */
