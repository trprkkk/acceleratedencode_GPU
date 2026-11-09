/*
 * CUDA backend (general-purpose compute only).
 *
 * Like the OpenCL backend, nothing is required at build or link time: the CUDA
 * driver API (libcuda.so.1) and the runtime compiler (libnvrtc.so) are
 * resolved with dlopen(), the shared kernel source in src/kernels.h is
 * compiled to PTX at run time and loaded as a module.  A machine without a
 * driver therefore still builds and runs the encoder -- the backend reports
 * itself unavailable and the dispatcher keeps the CPU path.
 *
 * No NVENC, no NPP, no vendor video pipeline: the nine kernels are the same
 * integer arithmetic as the CPU reference, and tests/kernel_check.c runs the
 * very same source against that reference on a machine with no GPU at all.
 *
 * The data flow (this file is why the backend is worth having at all):
 *
 *  - One *stream* per backend instance, made non-blocking, so the lanes of
 *    the guard (src/backend.c) run their work on the device concurrently.
 *    The legacy default stream serialises every lane behind every other one
 *    and makes a synchronous copy wait for the whole device to drain -- on a
 *    720p encode that alone cost a factor of ten.
 *
 *  - One transfer per direction per operation.  What the operation uploads
 *    (pixels, positions, residuals) is packed into a *pinned* staging arena
 *    and sent with a single cuMemcpyHtoDAsync(); what it downloads is written
 *    by the kernel into a device mirror of that arena and brought back with a
 *    single cuMemcpyDtoHAsync().  A MB at a time that turns three pageable
 *    round trips (each of them a device-wide synchronisation) into one
 *    asynchronous pair, and it keeps the number of driver calls independent
 *    of how many buffers the operation has.
 *
 *  - One synchronisation per operation, on the instance's own stream, so a
 *    lane never waits for another lane's pictures.
 *
 *  Fallbacks keep it portable: without pinned memory the arena is pageable
 * (the same calls, just not overlapping), without any of the new entry points
 * the backend still works through the synchronous calls.
 */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "backend.h"
#include "kernels.h"

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUstream;
typedef unsigned long long CUdeviceptr;
typedef void *nvrtcProgram;
typedef int nvrtcResult;

#define CUDA_SUCCESS 0
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT 16
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76
/* CU_STREAM_NON_BLOCKING: the stream must not synchronise against the legacy
 * default stream, which every other lane may be using. */
#define CU_STREAM_NON_BLOCKING 1
/* CU_MEMHOSTALLOC_PORTABLE: the pinned staging arena is valid on every lane's
 * thread (the guard clones this instance for each of them). */
#define CU_MEMHOSTALLOC_PORTABLE 1

/* The kernels are written in the common subset of OpenCL C and CUDA C; this
 * header turns the OpenCL spelling into the CUDA one.  The kernels are
 * declared `extern "C"` so that cuModuleGetFunction() finds them by name. */
static const char aeg_cuda_preamble[] =
    "typedef unsigned char uchar;\n"
    "typedef unsigned long long ulong;\n"
    "#define __kernel extern \"C\" __global__\n"
    "#define __global\n"
    "#define AEG_DEVFN static __device__\n"
    "#define get_global_id(d) ((int)(blockIdx.x * blockDim.x + threadIdx.x))\n";

#define CUDA_SYMS(X)                                                          \
    X(CUresult, cuInit, (unsigned int))                                       \
    X(CUresult, cuDeviceGetCount, (int *))                                    \
    X(CUresult, cuDeviceGet, (CUdevice *, int))                               \
    X(CUresult, cuDeviceGetName, (char *, int, CUdevice))                     \
    X(CUresult, cuDeviceTotalMem_v2, (size_t *, CUdevice))                    \
    X(CUresult, cuDeviceGetAttribute, (int *, int, CUdevice))                 \
    X(CUresult, cuCtxCreate_v2, (CUcontext *, unsigned int, CUdevice))        \
    X(CUresult, cuCtxDestroy_v2, (CUcontext))                                 \
    X(CUresult, cuModuleLoadDataEx, (CUmodule *, const void *, unsigned int,  \
                                     int *, void **))                         \
    X(CUresult, cuModuleUnload, (CUmodule))                                   \
    X(CUresult, cuModuleGetFunction, (CUfunction *, CUmodule, const char *))  \
    X(CUresult, cuMemAlloc_v2, (CUdeviceptr *, size_t))                       \
    X(CUresult, cuMemFree_v2, (CUdeviceptr))                                  \
    X(CUresult, cuMemHostAlloc, (void **, size_t, unsigned int))              \
    X(CUresult, cuMemFreeHost, (void *))                                      \
    X(CUresult, cuMemcpyHtoD_v2, (CUdeviceptr, const void *, size_t))         \
    X(CUresult, cuMemcpyHtoDAsync_v2, (CUdeviceptr, const void *, size_t,     \
                                       CUstream))                              \
    X(CUresult, cuMemcpyDtoHAsync_v2, (void *, CUdeviceptr, size_t,           \
                                       CUstream))                              \
    X(CUresult, cuLaunchKernel, (CUfunction, unsigned int, unsigned int,      \
                                 unsigned int, unsigned int, unsigned int,    \
                                 unsigned int, unsigned int, CUstream,        \
                                 void **, void **))                           \
    X(CUresult, cuStreamCreate, (CUstream *, unsigned int))                   \
    X(CUresult, cuStreamDestroy, (CUstream))                                  \
    X(CUresult, cuStreamSynchronize, (CUstream))                              \
    X(CUresult, cuCtxSetCurrent, (CUcontext))                                 \
    X(CUresult, cuCtxGetCurrent, (CUcontext *))                               \
    X(CUresult, cuCtxSynchronize, (void))

#define NVRTC_SYMS(X)                                                         \
    X(nvrtcResult, nvrtcCreateProgram, (nvrtcProgram *, const char *,         \
                                        const char *, int, const char **,     \
                                        const char **))                       \
    X(nvrtcResult, nvrtcCompileProgram, (nvrtcProgram, int,                   \
                                         const char **))                      \
    X(nvrtcResult, nvrtcGetProgramLogSize, (nvrtcProgram, size_t *))          \
    X(nvrtcResult, nvrtcGetProgramLog, (nvrtcProgram, char *))                \
    X(nvrtcResult, nvrtcGetPTXSize, (nvrtcProgram, size_t *))                 \
    X(nvrtcResult, nvrtcGetPTX, (nvrtcProgram, char *))                       \
    X(nvrtcResult, nvrtcDestroyProgram, (nvrtcProgram *))

#define DECL(ret, name, args) typedef ret (*fn_##name) args; static fn_##name p_##name;
CUDA_SYMS(DECL)
NVRTC_SYMS(DECL)
#undef DECL

#define AEG_BLOCK 64

/* Staging arena: a pinned host buffer and its device mirror.  One operation
 * packs everything it uploads into `host` (at `used`-growing offsets) and
 * everything it downloads is addressed as an offset of `dev`; the flush is
 * then one async copy in each direction and one stream synchronisation. */
#define ARENA_ALIGN 256u
/* Pinning is only useful for the small transfers: the coefficient batches,
 * where the per-call overhead is what matters and 512 KB is a lifetime's worth
 * of packets.  A picture-sized arena is left pageable: the driver stages it
 * through its own pinned buffer at the same bandwidth, and several large
 * pinned allocations per instance provoke a driver that aborts on the second
 * one (observed: a growing arena on a 720p reference upload). */
#define AEG_PIN_MAX ((size_t)512 << 10)

typedef struct cuda_arena {
    void       *host;      /* staging mirror (pinned, or pageable)          */
    CUdeviceptr dev;       /* the same bytes on the device                  */
    size_t      cap;       /* bytes allocated                              */
    size_t      used;      /* bytes packed by the operation in flight       */
    int         pinned;
} cuda_arena;

typedef struct cuda_ctx {
    void *lib_cuda;
    void *lib_nvrtc;
    CUdevice device;
    CUcontext ctx;
    CUstream st;               /* this instance's stream (its own work)      */
    cuda_arena up, dn;         /* upload staging, download mirror           */
    CUmodule mod;
    CUfunction k_sad, k_tq, k_iq, k_sse, k_intp, k_satd, k_tqiq, k_tqdc, k_iqdc;
    int cus;
    uint64_t mem;
    /* the registered reference picture (see ref_upload): one device
     * allocation, four planes at ref_off[] */
    int ref_stride, ref_w, ref_h, ref_valid;
    CUdeviceptr ref_dev;
    size_t ref_sz, ref_off[4];
    char name[96];
    CUcontext bind_ctx;        /* the context current where the arenas are  */
    int16_t *pos;               /* sad_batch: the rebased positions */
    size_t pos_cap;
    int is_clone;               /* shares ctx/mod/libraries with the first */
} cuda_ctx;

/* ------------------------------------------------------------------ */
/* Staging arena                                                      */
/* ------------------------------------------------------------------ */
/* Release the arena's buffers.  `used` is *kept*: the caller that packs a
 * batch of uploads may have to grow the arena in the middle of it, and the
 * packed prefix survives the move (arena_reserve copies it), so the offsets
 * already handed out stay valid and the offset of the next slot continues
 * where the last one ended. */
static void arena_clear(cuda_arena *a)
{
    if (a->dev) {
        p_cuMemFree_v2(a->dev);
        a->dev = 0;
    }
    if (a->host) {
        if (a->pinned)
            p_cuMemFreeHost(a->host);
        else
            free(a->host);
    }
    a->host = NULL;
    a->cap = 0;
    a->pinned = 0;
}

/* Room for `need` bytes (the packed size of the operation in flight, counted
 * from 0: every operation resets `used` first).  Growing keeps what is already
 * packed -- the device side is re-filled from the host side, which holds the
 * same bytes. */
static int arena_reserve(cuda_arena *a, size_t need)
{
    size_t cap;
    void *host = NULL;
    CUdeviceptr dev = 0;
    CUcontext cur = NULL;
    int pin;

    if (a->cap >= need)
        return AEG_OK;
    cap = a->cap ? a->cap : 65536;
    while (cap < need)
        cap *= 2;
    /* an arena belongs to the thread that made it: its pinned buffer is
     * registered against the context current on that thread, and an async
     * transfer from another thread's driver state would be rejected. */
    if (p_cuCtxGetCurrent(&cur) != CUDA_SUCCESS || !cur)
        pin = 0;
    else
        pin = cap <= AEG_PIN_MAX;
    if (pin && p_cuMemHostAlloc(&host, cap, CU_MEMHOSTALLOC_PORTABLE) !=
                  CUDA_SUCCESS)
        host = NULL;
    if (!host) {
        host = malloc(cap);
        if (!host)
            return AEG_ENOMEM;
        pin = 0;
    }
    if (p_cuMemAlloc_v2(&dev, cap) != CUDA_SUCCESS) {
        if (pin)
            p_cuMemFreeHost(host);
        else
            free(host);
        return AEG_EBACKEND;
    }
    if (a->used) {
        memcpy(host, a->host, a->used);
        if (p_cuMemcpyHtoD_v2(dev, host, a->used) != CUDA_SUCCESS) {
            if (pin)
                p_cuMemFreeHost(host);
            else
                free(host);
            p_cuMemFree_v2(dev);
            return AEG_EBACKEND;
        }
    }
    arena_clear(a);
    a->host = host;
    a->dev = dev;
    a->cap = cap;
    a->pinned = pin;
    return AEG_OK;
}

/* Pack an upload slot and report its offset (the device address of the same
 * bytes is a->dev + *off).  0 on success. */
static int arena_up(cuda_ctx *o, const void *src, size_t bytes, size_t *off)
{
    cuda_arena *a = &o->up;
    size_t al = (bytes + ARENA_ALIGN - 1) & ~(size_t)(ARENA_ALIGN - 1);
    char *dst;
    if (arena_reserve(a, a->used + al) != AEG_OK)
        return AEG_EBACKEND;
    dst = (char *)a->host + a->used;
    if (bytes)
        memcpy(dst, src, bytes);
    *off = a->used;
    a->used += al;
    return AEG_OK;
}

/* Reserve a download slot: the kernel writes into a->dev + *off, the host
 * receives the same bytes at a->host + *off. */
static int arena_dn(cuda_ctx *o, size_t bytes, size_t *off)
{
    cuda_arena *a = &o->dn;
    size_t al = (bytes + ARENA_ALIGN - 1) & ~(size_t)(ARENA_ALIGN - 1);
    if (arena_reserve(a, a->used + al) != AEG_OK)
        return AEG_EBACKEND;
    *off = a->used;
    a->used += al;
    return AEG_OK;
}

static CUdeviceptr arena_dev(const cuda_arena *a, size_t off)
{
    return a->dev + (CUdeviceptr)off;
}

/* The `_v2` spelling of the async copy entry points: the plain spelling is
 * deprecated by the driver and answers CUDA_ERROR_INVALID_CONTEXT on a driver
 * that keeps no compatibility path for it (observed on the WSL 590.52
 * driver), which would take the whole backend down with it. */
static int cuda_flush_up(cuda_ctx *o)
{
    if (!o->up.used)
        return AEG_OK;
    return p_cuMemcpyHtoDAsync_v2(o->up.dev, o->up.host, o->up.used, o->st) ==
                   CUDA_SUCCESS
               ? AEG_OK
               : AEG_EBACKEND;
}

static int cuda_flush_dn(cuda_ctx *o)
{
    if (!o->dn.used)
        return AEG_OK;
    return p_cuMemcpyDtoHAsync_v2(o->dn.host, o->dn.dev, o->dn.used, o->st) ==
                   CUDA_SUCCESS
               ? AEG_OK
               : AEG_EBACKEND;
}

/* The only synchronisation: this instance's stream, so a lane waits for its own
 * work and never for another lane's pictures. */
static int cuda_wait(cuda_ctx *o)
{
    return p_cuStreamSynchronize(o->st) == CUDA_SUCCESS ? AEG_OK : AEG_EBACKEND;
}

/* Scatter a download slot back to its caller's buffer. */
static void cuda_take(cuda_ctx *o, size_t off, void *dst, size_t bytes)
{
    if (bytes)
        memcpy(dst, (char *)o->dn.host + off, bytes);
}

/* A staging arena could not be reserved (out of memory on the device, or the
 * driver refused the allocation).  The stream carries nothing pending, so the
 * numbers the operation has produced so far are simply lost -- and the caller
 * has to fall back to the CPU reference for this operation. */
static int arena_failed(cuda_ctx *o)
{
    (void)o;
    return AEG_ENOMEM;
}

static void cuda_sync(void *priv);

/* ------------------------------------------------------------------ */
/* Kernel launch                                                       */
/* ------------------------------------------------------------------ */
static int cuda_launch(cuda_ctx *o, CUfunction f, int count, void **args)
{
    unsigned int blocks = (unsigned int)((count + AEG_BLOCK - 1) / AEG_BLOCK);
    if (!blocks)
        return AEG_OK;
    if (p_cuLaunchKernel(f, blocks, 1, 1, AEG_BLOCK, 1, 1, 0, o->st, args,
                         NULL) != CUDA_SUCCESS)
        return AEG_EBACKEND;
    return AEG_OK;
}

/* Both arenas are made on the thread that creates them, with the context of
 * this instance current on that thread -- the guard clones the backend for a
 * lane on the thread that first uses it, and a driver-API context is current
 * on the thread that created it only. */
static int cuda_ctx_bind(cuda_ctx *o)
{
    if (o->ctx && p_cuCtxGetCurrent(&o->bind_ctx) == CUDA_SUCCESS &&
        o->bind_ctx == o->ctx)
        return CUDA_SUCCESS;
    o->bind_ctx = NULL;
    return p_cuCtxSetCurrent(o->ctx);
}

/* A driver-API context is current on the thread that created it only, and the
 * operations are called from slice threads too (one at a time: the backend
 * guard in backend.c serialises them), so every entry point makes it current
 * on the calling thread first. */
static int cuda_bind(cuda_ctx *o)
{
    return cuda_ctx_bind(o) == CUDA_SUCCESS ? AEG_OK : AEG_EBACKEND;
}

/* ------------------------------------------------------------------ */
/* Operations                                                         */
/* ------------------------------------------------------------------ */
static int cuda_sad_batch(void *priv, const uint8_t *cur, int cstride,
                          const uint8_t *ref, int rstride, int bw, int bh,
                          const int16_t *mv, int count, int32_t *out)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    CUdeviceptr dcur, dref, dmv, dout;
    int icur = cstride, iref = rstride, ibw = bw, ibh = bh, icount = count;
    aeg_sad_span sp;
    void *args[9];
    size_t coff, roff, moff, ooff;
    int st;

    if (cuda_bind(o) != AEG_OK)
        return AEG_EBACKEND;
    if (count <= 0)
        return AEG_OK;
    if (o->pos_cap < (size_t)count) {
        free(o->pos);
        o->pos_cap = (size_t)count;
        o->pos = (int16_t *)malloc(o->pos_cap * 8);
        if (!o->pos) {
            o->pos_cap = 0;
            return AEG_ENOMEM;
        }
    }
    /* only the span the candidates read, see aeg_sad_span_rebase() */
    aeg_sad_span_rebase(mv, count, bw, bh, cstride, rstride, o->pos, &sp);

    o->up.used = 0;
    o->dn.used = 0;
    if (arena_up(o, cur + sp.coff, sp.cbytes, &coff) != AEG_OK ||
        arena_up(o, ref + sp.roff, sp.rbytes, &roff) != AEG_OK ||
        arena_up(o, o->pos, (size_t)count * 8, &moff) != AEG_OK ||
        arena_dn(o, (size_t)count * 4, &ooff) != AEG_OK)
        return arena_failed(o);
    dcur = arena_dev(&o->up, coff);
    dref = arena_dev(&o->up, roff);
    dmv = arena_dev(&o->up, moff);
    dout = arena_dev(&o->dn, ooff);
    if ((st = cuda_flush_up(o)) != AEG_OK)
        return st;

    args[0] = &dcur; args[1] = &icur; args[2] = &dref; args[3] = &iref;
    args[4] = &ibw;  args[5] = &ibh;  args[6] = &icount; args[7] = &dmv;
    args[8] = &dout;
    if ((st = cuda_launch(o, o->k_sad, count, args)) != AEG_OK)
        return st;
    if ((st = cuda_flush_dn(o)) != AEG_OK)
        return st;
    if ((st = cuda_wait(o)) != AEG_OK)
        return st;
    cuda_take(o, ooff, out, (size_t)count * 4);
    return AEG_OK;
}

static int cuda_tq_batch(void *priv, const int16_t *res, int nblk, int qp,
                         int intra, int16_t *level, int32_t *nz)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    CUdeviceptr dres, dlev, dnz;
    size_t bytes = (size_t)nblk * 16 * 2, nzb = (size_t)nblk * 4;
    size_t roff, loff, zoff;
    int qbits = 15 + qp / 6;
    int f = aeg_quant_bias(qbits, intra ? 3 : 6);
    int mf0 = aeg_quant_mf[qp % 6][0], mf1 = aeg_quant_mf[qp % 6][1],
        mf2 = aeg_quant_mf[qp % 6][2];
    void *args[10];
    int st;

    if (nblk <= 0)
        return AEG_OK;
    if (cuda_bind(o) != AEG_OK)
        return AEG_EBACKEND;

    o->up.used = 0;
    o->dn.used = 0;
    /* the kernel always writes the non-zero count, so a caller that does not
     * want it (nz == NULL) still gets the slot to write it into */
    if (arena_up(o, res, bytes, &roff) != AEG_OK ||
        arena_dn(o, bytes, &loff) != AEG_OK ||
        arena_dn(o, nzb, &zoff) != AEG_OK)
        return arena_failed(o);
    dres = arena_dev(&o->up, roff);
    dlev = arena_dev(&o->dn, loff);
    dnz = arena_dev(&o->dn, zoff);
    if ((st = cuda_flush_up(o)) != AEG_OK)
        return st;
    args[0] = &dres; args[1] = &nblk; args[2] = &qbits; args[3] = &f;
    args[4] = &mf0;  args[5] = &mf1;  args[6] = &mf2;
    args[7] = &dlev; args[8] = &dnz;
    if ((st = cuda_launch(o, o->k_tq, nblk, args)) != AEG_OK)
        return st;
    if ((st = cuda_flush_dn(o)) != AEG_OK)
        return st;
    if ((st = cuda_wait(o)) != AEG_OK)
        return st;
    cuda_take(o, loff, level, bytes);
    if (nz)
        cuda_take(o, zoff, nz, nzb);
    return AEG_OK;
}

static int cuda_iq_idct_batch(void *priv, const int16_t *level, int nblk,
                               int qp, int16_t *res)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    CUdeviceptr dlev, dres;
    size_t bytes = (size_t)nblk * 16 * 2;
    size_t loff, roff;
    int sh = qp / 6;
    int ls0 = aeg_dequant_ls[qp % 6][0], ls1 = aeg_dequant_ls[qp % 6][1],
        ls2 = aeg_dequant_ls[qp % 6][2];
    void *args[7];
    int st;

    if (nblk <= 0)
        return AEG_OK;
    if (cuda_bind(o) != AEG_OK)
        return AEG_EBACKEND;

    o->up.used = 0;
    o->dn.used = 0;
    if (arena_up(o, level, bytes, &loff) != AEG_OK ||
        arena_dn(o, bytes, &roff) != AEG_OK)
        return arena_failed(o);
    dlev = arena_dev(&o->up, loff);
    dres = arena_dev(&o->dn, roff);
    if ((st = cuda_flush_up(o)) != AEG_OK)
        return st;
    args[0] = &dlev; args[1] = &nblk; args[2] = &sh;
    args[3] = &ls0;  args[4] = &ls1;  args[5] = &ls2; args[6] = &dres;
    if ((st = cuda_launch(o, o->k_iq, nblk, args)) != AEG_OK)
        return st;
    if ((st = cuda_flush_dn(o)) != AEG_OK)
        return st;
    if ((st = cuda_wait(o)) != AEG_OK)
        return st;
    cuda_take(o, roff, res, bytes);
    return AEG_OK;
}

static int cuda_tq_iq_batch(void *priv, const int16_t *res, int nblk, int qp,
                             int intra, int16_t *level, int32_t *nz,
                             int16_t *rec)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    CUdeviceptr dres, dlev, dnz, drec;
    size_t bytes = (size_t)nblk * 16 * 2, nzb = (size_t)nblk * 4;
    size_t roff, loff, zoff, koff;
    int qbits = 15 + qp / 6;
    int f = aeg_quant_bias(qbits, intra ? 3 : 6);
    int mf0 = aeg_quant_mf[qp % 6][0], mf1 = aeg_quant_mf[qp % 6][1],
        mf2 = aeg_quant_mf[qp % 6][2];
    int sh = qp / 6;
    int ls0 = aeg_dequant_ls[qp % 6][0], ls1 = aeg_dequant_ls[qp % 6][1],
        ls2 = aeg_dequant_ls[qp % 6][2];
    void *args[14];
    int st;

    if (nblk <= 0)
        return AEG_OK;
    if (cuda_bind(o) != AEG_OK)
        return AEG_EBACKEND;

    o->up.used = 0;
    o->dn.used = 0;
    /* the kernel always writes the non-zero count, so a caller that does not
     * want it (nz == NULL) still gets the slot to write it into */
    if (arena_up(o, res, bytes, &roff) != AEG_OK ||
        arena_dn(o, bytes, &loff) != AEG_OK ||
        arena_dn(o, nzb, &zoff) != AEG_OK ||
        arena_dn(o, bytes, &koff) != AEG_OK)
        return arena_failed(o);
    dres = arena_dev(&o->up, roff);
    dlev = arena_dev(&o->dn, loff);
    dnz = arena_dev(&o->dn, zoff);
    drec = arena_dev(&o->dn, koff);
    if ((st = cuda_flush_up(o)) != AEG_OK)
        return st;
    args[0] = &dres; args[1] = &nblk; args[2] = &qbits; args[3] = &f;
    args[4] = &mf0;  args[5] = &mf1;  args[6] = &mf2;  args[7] = &sh;
    args[8] = &ls0;  args[9] = &ls1; args[10] = &ls2;
    args[11] = &dlev; args[12] = &dnz; args[13] = &drec;
    if ((st = cuda_launch(o, o->k_tqiq, nblk, args)) != AEG_OK)
        return st;
    if ((st = cuda_flush_dn(o)) != AEG_OK)
        return st;
    if ((st = cuda_wait(o)) != AEG_OK)
        return st;
    cuda_take(o, loff, level, bytes);
    if (nz)
        cuda_take(o, zoff, nz, nzb);
    cuda_take(o, koff, rec, bytes);
    return AEG_OK;
}

static int cuda_tq_dc_batch(void *priv, const int16_t *res, int nblk, int qp,
                             int intra, int16_t *level, int32_t *nz,
                             int32_t *dc)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    CUdeviceptr dres, dlev, dnz, ddc;
    size_t bytes = (size_t)nblk * 16 * 2, nzb = (size_t)nblk * 4;
    size_t roff, loff, zoff, coff;
    int qbits = 15 + qp / 6;
    int f = aeg_quant_bias(qbits, intra ? 3 : 6);
    int mf0 = aeg_quant_mf[qp % 6][0], mf1 = aeg_quant_mf[qp % 6][1],
        mf2 = aeg_quant_mf[qp % 6][2];
    void *args[10];
    int st;

    if (nblk <= 0)
        return AEG_OK;
    if (cuda_bind(o) != AEG_OK)
        return AEG_EBACKEND;

    o->up.used = 0;
    o->dn.used = 0;
    if (arena_up(o, res, bytes, &roff) != AEG_OK ||
        arena_dn(o, bytes, &loff) != AEG_OK ||
        arena_dn(o, nzb, &zoff) != AEG_OK ||
        arena_dn(o, nzb, &coff) != AEG_OK)
        return arena_failed(o);
    dres = arena_dev(&o->up, roff);
    dlev = arena_dev(&o->dn, loff);
    dnz = arena_dev(&o->dn, zoff);
    ddc = arena_dev(&o->dn, coff);
    if ((st = cuda_flush_up(o)) != AEG_OK)
        return st;
    args[0] = &dres; args[1] = &nblk; args[2] = &qbits; args[3] = &f;
    args[4] = &mf0;  args[5] = &mf1;  args[6] = &mf2;
    args[7] = &dlev; args[8] = &dnz;  args[9] = &ddc;
    if ((st = cuda_launch(o, o->k_tqdc, nblk, args)) != AEG_OK)
        return st;
    if ((st = cuda_flush_dn(o)) != AEG_OK)
        return st;
    if ((st = cuda_wait(o)) != AEG_OK)
        return st;
    cuda_take(o, loff, level, bytes);
    if (nz)
        cuda_take(o, zoff, nz, nzb);
    if (dc)
        cuda_take(o, coff, dc, nzb);
    return AEG_OK;
}

static int cuda_iq_dc_batch(void *priv, const int16_t *level,
                             const int32_t *dc, int nblk, int qp, int16_t *res)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    CUdeviceptr dlev, ddc, dres;
    size_t bytes = (size_t)nblk * 16 * 2, nzb = (size_t)nblk * 4;
    size_t loff, coff, roff;
    int sh = qp / 6;
    int ls0 = aeg_dequant_ls[qp % 6][0], ls1 = aeg_dequant_ls[qp % 6][1],
        ls2 = aeg_dequant_ls[qp % 6][2];
    void *args[8];
    int st;

    if (nblk <= 0)
        return AEG_OK;
    if (cuda_bind(o) != AEG_OK)
        return AEG_EBACKEND;

    o->up.used = 0;
    o->dn.used = 0;
    if (arena_up(o, level, bytes, &loff) != AEG_OK ||
        arena_up(o, dc, nzb, &coff) != AEG_OK ||
        arena_dn(o, bytes, &roff) != AEG_OK)
        return arena_failed(o);
    dlev = arena_dev(&o->up, loff);
    ddc = arena_dev(&o->up, coff);
    dres = arena_dev(&o->dn, roff);
    if ((st = cuda_flush_up(o)) != AEG_OK)
        return st;
    args[0] = &dlev; args[1] = &ddc; args[2] = &nblk; args[3] = &sh;
    args[4] = &ls0;  args[5] = &ls1; args[6] = &ls2;  args[7] = &dres;
    if ((st = cuda_launch(o, o->k_iqdc, nblk, args)) != AEG_OK)
        return st;
    if ((st = cuda_flush_dn(o)) != AEG_OK)
        return st;
    if ((st = cuda_wait(o)) != AEG_OK)
        return st;
    cuda_take(o, roff, res, bytes);
    return AEG_OK;
}

static int cuda_sse_batch(void *priv, const uint8_t *a, int sa, const uint8_t *b,
                           int sb, int w, int h, uint64_t *out)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    CUdeviceptr da, db, dout;
    size_t tw = (size_t)w / 8, th = (size_t)h / 8, count = tw * th;
    size_t abytes = (size_t)sa * h, bbytes = (size_t)sb * h;
    size_t aoff, boff, ooff;
    uint64_t *tmp;
    size_t i, acc = 0;
    int itw = (int)tw, ith = (int)th, icount = (int)count;
    void *args[8];
    int st;

    if (cuda_bind(o) != AEG_OK)
        return AEG_EBACKEND;

    if (!count) {
        *out = 0;
        return AEG_OK;
    }
    o->up.used = 0;
    o->dn.used = 0;
    if (arena_up(o, a, abytes, &aoff) != AEG_OK ||
        arena_up(o, b, bbytes, &boff) != AEG_OK ||
        arena_dn(o, count * 8, &ooff) != AEG_OK)
        return arena_failed(o);
    da = arena_dev(&o->up, aoff);
    db = arena_dev(&o->up, boff);
    dout = arena_dev(&o->dn, ooff);
    if ((st = cuda_flush_up(o)) != AEG_OK)
        return st;
    args[0] = &da; args[1] = &sa; args[2] = &db; args[3] = &sb;
    args[4] = &itw; args[5] = &ith; args[6] = &icount; args[7] = &dout;
    if ((st = cuda_launch(o, o->k_sse, (int)count, args)) != AEG_OK)
        return st;
    if ((st = cuda_flush_dn(o)) != AEG_OK)
        return st;
    if ((st = cuda_wait(o)) != AEG_OK)
        return st;
    /* the sum over the tiles, summed on the host (it is a reduction over a few
     * thousand 64-bit values, and a per-frame number the encoder needs at
     * once) */
    tmp = (uint64_t *)((char *)o->dn.host + ooff);
    for (i = 0; i < count; i++)
        acc += tmp[i];
    *out = (uint64_t)acc;
    return AEG_OK;
}

/* One fractional-pel plane of the reference picture (see kernels.h). */
static int cuda_interp_plane(void *priv, const uint8_t *ref, int rstride,
                              int w, int h, int which, uint8_t *dst, int dstride)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    CUdeviceptr dref, ddst;
    size_t rows = (size_t)h + 2 * AEG_PAD;
    size_t count = (size_t)(w + 2 * AEG_PAD) * rows;
    size_t rbytes = (size_t)rstride * rows, dbytes = (size_t)dstride * rows;
    size_t roff, doff;
    int pad = AEG_PAD;
    void *args[8];
    int st;

    if (cuda_bind(o) != AEG_OK)
        return AEG_EBACKEND;

    o->up.used = 0;
    o->dn.used = 0;
    /* the kernel takes the whole padded allocations, see backend_opencl.c */
    if (arena_up(o, ref - (ptrdiff_t)AEG_PAD * rstride - AEG_PAD, rbytes,
                 &roff) != AEG_OK ||
        arena_dn(o, dbytes, &doff) != AEG_OK)
        return arena_failed(o);
    dref = arena_dev(&o->up, roff);
    ddst = arena_dev(&o->dn, doff);
    if ((st = cuda_flush_up(o)) != AEG_OK)
        return st;
    args[0] = &dref; args[1] = &rstride; args[2] = &w; args[3] = &h;
    args[4] = &pad;  args[5] = &which;   args[6] = &ddst; args[7] = &dstride;
    if ((st = cuda_launch(o, o->k_intp, (int)count, args)) != AEG_OK)
        return st;
    if ((st = cuda_flush_dn(o)) != AEG_OK)
        return st;
    if ((st = cuda_wait(o)) != AEG_OK)
        return st;
    cuda_take(o, doff, dst - (ptrdiff_t)AEG_PAD * dstride - AEG_PAD, dbytes);
    return AEG_OK;
}

/* Register the reference picture (four planes, resident in device memory).
 * The four planes are staged into the arena together and sent with one
 * transfer: four copies that read the same staging buffer would be four
 * pending reads of a buffer the next plane overwrites.  satd_batch() finds
 * each plane at refdev + ref_off[k]. */
static int cuda_ref_upload(void *priv, const uint8_t *const planes[4],
                            int pstride, int w, int h)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    size_t rows = (size_t)h + 2 * AEG_PAD;
    size_t bytes = (size_t)pstride * rows;
    size_t off[4], total;
    int k;

    if (cuda_bind(o) != AEG_OK)
        return AEG_EBACKEND;

    o->up.used = 0;
    for (k = 0; k < 4; k++)
        if (arena_up(o, planes[k] - (ptrdiff_t)AEG_PAD * pstride - AEG_PAD,
                     bytes, &off[k]) != AEG_OK)
            return arena_failed(o);
    total = o->up.used;
    /* one allocation for all four planes: the whole picture stays resident
     * between the upload and the searches that read it */
    if (!o->ref_dev || o->ref_sz < total) {
        if (o->ref_dev)
            p_cuMemFree_v2(o->ref_dev);
        o->ref_sz = total;
        if (p_cuMemAlloc_v2(&o->ref_dev, o->ref_sz) != CUDA_SUCCESS) {
            o->ref_dev = 0;
            o->ref_sz = 0;
            return AEG_EBACKEND;
        }
    }
    if (p_cuMemcpyHtoDAsync_v2(o->ref_dev, o->up.host, total, o->st) !=
        CUDA_SUCCESS)
        return AEG_EBACKEND;
    for (k = 0; k < 4; k++)
        o->ref_off[k] = off[k];
    o->ref_stride = pstride;
    o->ref_w = w;
    o->ref_h = h;
    o->ref_valid = 1;
    /* the upload is pending on this stream; nothing may re-pack the staging
     * arena before it has left it, so this waits here */
    return cuda_wait(o);
}

static int cuda_satd_batch(void *priv, const uint8_t *cur, int cstride, int bw,
                            int bh, const int16_t *pos, int count, int32_t *out)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    CUdeviceptr dcur, dpos, dout, dref[4];
    size_t rows = (size_t)(o->ref_valid ? o->ref_h : 0) + 2 * AEG_PAD;
    size_t cbytes = (size_t)cstride * rows;
    size_t coff, poff, ooff;
    int pad = AEG_PAD, k, st;
    void *args[13];

    if (cuda_bind(o) != AEG_OK)
        return AEG_EBACKEND;

    if (!o->ref_valid || !count)
        return AEG_EBACKEND;
    o->up.used = 0;
    o->dn.used = 0;
    if (arena_up(o, cur - (ptrdiff_t)AEG_PAD * cstride - AEG_PAD, cbytes,
                 &coff) != AEG_OK ||
        arena_up(o, pos, (size_t)count * 8, &poff) != AEG_OK ||
        arena_dn(o, (size_t)count * 4, &ooff) != AEG_OK)
        return arena_failed(o);
    dcur = arena_dev(&o->up, coff);
    dpos = arena_dev(&o->up, poff);
    dout = arena_dev(&o->dn, ooff);
    for (k = 0; k < 4; k++)
        dref[k] = o->ref_dev + (CUdeviceptr)o->ref_off[k];
    if ((st = cuda_flush_up(o)) != AEG_OK)
        return st;
    args[0] = &dcur;  args[1] = &cstride;
    for (k = 0; k < 4; k++)
        args[2 + k] = &dref[k];
    args[6] = &o->ref_stride; args[7] = &pad;
    args[8] = &bw;   args[9] = &bh;
    args[10] = &dpos; args[11] = &count; args[12] = &dout;
    if ((st = cuda_launch(o, o->k_satd, count, args)) != AEG_OK)
        return st;
    if ((st = cuda_flush_dn(o)) != AEG_OK)
        return st;
    if ((st = cuda_wait(o)) != AEG_OK)
        return st;
    cuda_take(o, ooff, out, (size_t)count * 4);
    return AEG_OK;
}

static void cuda_sync(void *priv)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    if (cuda_bind(o) == AEG_OK)
        p_cuStreamSynchronize(o->st);
}

static int cuda_ops_fill(cuda_ctx *o, aeg_be_ops *ops);

static void cuda_destroy(void *priv)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    if (!o)
        return;
    if (o->ctx)
        cuda_bind(o);
    arena_clear(&o->up);
    arena_clear(&o->dn);
    if (o->ref_dev) {
        p_cuMemFree_v2(o->ref_dev);
        o->ref_dev = 0;
    }
    if (o->st && p_cuStreamDestroy(o->st) != CUDA_SUCCESS) {
        /* nothing to do: the context teardown below reclaims it anyway */
    }
    free(o->pos);
    if (o->is_clone) {
        free(o);
        return;
    }
    if (o->ctx) {
        if (o->mod)
            p_cuModuleUnload(o->mod);
        p_cuCtxDestroy_v2(o->ctx);
    }
    if (o->lib_nvrtc)
        dlclose(o->lib_nvrtc);
    if (o->lib_cuda)
        dlclose(o->lib_cuda);
    free(o);
}

/* ------------------------------------------------------------------ */
/* Setup                                                               */
/* ------------------------------------------------------------------ */
static int cuda_load(void **lib)
{
    static const char *names[] = {
        "libcuda.so.1", "libcuda.so", "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        NULL
    };
    int i;
    /* AEG_CUDA_LIB: a library outside the loader's search path (a pip-installed
     * runtime, a second driver) -- tried before the usual names */
    const char *ev = getenv("AEG_CUDA_LIB");
    if (ev && *ev) {
        *lib = dlopen(ev, RTLD_NOW | RTLD_LOCAL);
        if (*lib)
            return AEG_OK;
        aeg_log(AEG_LOG_WARNING, "aeg: AEG_CUDA_LIB=%s: %s", ev, dlerror());
    }
    for (i = 0; names[i]; i++) {
        *lib = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
        if (*lib)
            return AEG_OK;
    }
    return AEG_EBACKEND;
}

static int nvrtc_load(void **lib)
{
    static const char *names[] = {
        "libnvrtc.so", "libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so.11",
        "/usr/local/cuda/lib64/libnvrtc.so", NULL
    };
    int i;
    /* AEG_NVRTC_LIB: a library outside the loader's search path (a pip-installed
     * runtime, a second driver) -- tried before the usual names */
    const char *ev = getenv("AEG_NVRTC_LIB");
    if (ev && *ev) {
        *lib = dlopen(ev, RTLD_NOW | RTLD_LOCAL);
        if (*lib)
            return AEG_OK;
        aeg_log(AEG_LOG_WARNING, "aeg: AEG_NVRTC_LIB=%s: %s", ev, dlerror());
    }
    for (i = 0; names[i]; i++) {
        *lib = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
        if (*lib)
            return AEG_OK;
    }
    return AEG_EBACKEND;
}

/* The kernels the backend looks up in the module. */
static const char *const aeg_cuda_kernels[] = {
    "sad_batch", "tq_batch", "iq_idct_batch", "sse_batch", "interp_plane",
    "satd_batch", "tq_iq_batch", "tq_dc_batch", "iq_dc_batch", NULL
};

/* NVRTC 11.2+ can list the architectures it compiles for; optional, so an
 * older NVRTC still loads. */
typedef nvrtcResult (*fn_nvrtcGetNumSupportedArchs)(int *);
typedef nvrtcResult (*fn_nvrtcGetSupportedArchs)(int *);
static fn_nvrtcGetNumSupportedArchs p_nvrtcGetNumSupportedArchs;
static fn_nvrtcGetSupportedArchs p_nvrtcGetSupportedArchs;

/* The shared kernel source (behind the CUDA preamble) to PTX for virtual
 * architecture compute_<cc>.  The PTX is malloc'ed into *ptx_out; on failure
 * the compiler log goes to `log`. */
static int cuda_compile_ptx(int cc, char **ptx_out, size_t *ptx_size,
                             char *log, size_t logsz)
{
    nvrtcProgram prog = NULL;
    size_t src_len = strlen(aeg_cuda_preamble) + strlen(aeg_kernel_src) + 1;
    char arch[64], *src, *ptx;
    const char *opts[1];
    size_t ptx_sz = 0;
    nvrtcResult nr;

    if (log && logsz)
        log[0] = '\0';
    snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d", cc);
    opts[0] = arch;
    src = (char *)malloc(src_len);
    if (!src)
        return AEG_ENOMEM;
    snprintf(src, src_len, "%s%s", aeg_cuda_preamble, aeg_kernel_src);
    nr = p_nvrtcCreateProgram(&prog, src, "aeg_kernels.cu", 0, NULL, NULL);
    if (nr == 0)
        nr = p_nvrtcCompileProgram(prog, 1, opts);
    if (nr != 0) {
        size_t log_sz = 0;
        if (log && logsz && prog &&
            p_nvrtcGetProgramLogSize(prog, &log_sz) == 0 && log_sz > 1 &&
            log_sz <= logsz)
            p_nvrtcGetProgramLog(prog, log);
        if (prog)
            p_nvrtcDestroyProgram(&prog);
        free(src);
        return AEG_EBACKEND;
    }
    p_nvrtcGetPTXSize(prog, &ptx_sz);
    ptx = (char *)malloc(ptx_sz + 1);
    if (!ptx) {
        p_nvrtcDestroyProgram(&prog);
        free(src);
        return AEG_ENOMEM;
    }
    p_nvrtcGetPTX(prog, ptx);
    ptx[ptx_sz] = '\0';
    p_nvrtcDestroyProgram(&prog);
    free(src);
    *ptx_out = ptx;
    if (ptx_size)
        *ptx_size = ptx_sz;
    return AEG_OK;
}

/* The newest architecture NVRTC supports that is not newer than `cc` (0: none
 * known).  A device newer than the NVRTC at hand still runs PTX built for an
 * older virtual architecture -- the driver JIT-compiles it forward. */
static int cuda_fallback_arch(int cc)
{
    int n = 0, i, best = 0, *list;
    if (!p_nvrtcGetNumSupportedArchs || !p_nvrtcGetSupportedArchs ||
        p_nvrtcGetNumSupportedArchs(&n) != 0 || n <= 0)
        return 0;
    list = (int *)malloc((size_t)n * sizeof(int));
    if (!list)
        return 0;
    if (p_nvrtcGetSupportedArchs(list) == 0)
        for (i = 0; i < n; i++)
            if (list[i] <= cc && list[i] > best)
                best = list[i];
    free(list);
    return best;
}

/* ------------------------------------------------------------------ */
/* PTX cache                                                          */
/* ------------------------------------------------------------------ */
/* NVRTC spends about a second compiling nine kernels, which on a short clip
 * is most of the startup.  The PTX is a pure function of the source and the
 * virtual architecture, so it is kept on disk under the user's cache
 * directory and reused.  Every failure to read or write it is silent: a
 * missing cache costs a second, a broken one would cost the backend. */
/* how many PTX files this run took from the cache (0: it compiled) */
static int ptx_cache_kept;

/* how many PTX files this run took from the cache (0: it compiled) -- used by
 * the traffic the encoder prints, and by the tests */
int aeg_cuda_ptx_cached(void)
{
    return ptx_cache_kept;
}

static uint64_t ptx_hash(const char *s)
{
    uint64_t h = 1469598103934665603ULL;
    for (; *s; s++) {
        h ^= (unsigned char)*s++;
        h *= 1099511628211ULL;
    }
    return h;
}

static int ptx_cache_dir(char *out, size_t outsz)
{
    const char *ev = getenv("AEG_PTX_CACHE");
    const char *home;
    if (ev && *ev)
        snprintf(out, outsz, "%s", ev);
    else if ((home = getenv("HOME")) || (home = getenv("USERPROFILE")) ||
             (home = getenv("LOCALAPPDATA")))
        snprintf(out, outsz, "%s/.cache/aeg/ptx", home);
    else
        return 0;
    return 1;
}

/* Read the cached PTX for (cc) into *ptx_out, or return AEG_EBACKEND. */
static int ptx_cache_load(int cc, char **ptx_out, size_t *ptx_size)
{
    char dir[512], path[640];
    uint64_t h;
    FILE *fp;
    char *buf;
    size_t n;

    if (getenv("AEG_PTX_NOCACHE"))
        return AEG_EBACKEND;
    if (!ptx_cache_dir(dir, sizeof(dir)))
        return AEG_EBACKEND;
    h = ptx_hash(aeg_cuda_preamble) ^ (ptx_hash(aeg_kernel_src) * 31);
    snprintf(path, sizeof(path), "%s/compute_%d_%016llx.ptx", dir, cc,
             (unsigned long long)h);
    fp = fopen(path, "rb");
    if (!fp)
        return AEG_EBACKEND;
    buf = (char *)malloc(1 << 20);
    if (!buf) {
        fclose(fp);
        return AEG_ENOMEM;
    }
    n = fread(buf, 1, (1 << 20) - 1, fp);
    fclose(fp);
    /* a partial write would be a module the driver rejects */
    if (!n || buf[n - 1] != '\n') {
        free(buf);
        return AEG_EBACKEND;
    }
    buf[n] = '\0';
    *ptx_out = buf;
    ptx_cache_kept = 1;
    if (ptx_size)
        *ptx_size = n;
    return AEG_OK;
}

/* Keep the PTX for (cc) on disk.  Directory creation is best effort: the
 * process may not have the rights, and the next run can compile again. */
static void ptx_cache_store(int cc, const char *ptx, size_t n)
{
    char dir[512], path[640];
    uint64_t h;
    FILE *fp;

    if (getenv("AEG_PTX_NOCACHE"))
        return;
    if (!ptx_cache_dir(dir, sizeof(dir)))
        return;
    h = ptx_hash(aeg_cuda_preamble) ^ (ptx_hash(aeg_kernel_src) * 31);
    snprintf(path, sizeof(path), "%s/compute_%d_%016llx.ptx", dir, cc,
             (unsigned long long)h);
    fp = fopen(path, "rb");
    if (fp) {            /* already there: nothing to write */
        fclose(fp);
        return;
    }
    fp = fopen(path, "ab");
    if (!fp) {
        /* make the directory chain, then try again (a run as root, a first
         * use of a fresh cache location) */
        char mk[600];
        int i;
        snprintf(mk, sizeof(mk), "%s", dir);
        for (i = 1; mk[i]; i++)
            if (mk[i] == '/') {
                mk[i] = '\0';
                mkdir(mk, 0777);
                mk[i] = '/';
            }
        mkdir(mk, 0777);
        fp = fopen(path, "ab");
        if (!fp)
            return;
    }
    fwrite(ptx, 1, n, fp);
    fclose(fp);
}

/* Compile the shared kernel source to PTX and load it as a module.  The PTX
 * comes from the cache when one holds it (see ptx_cache_load), so a second run
 * of the encoder on the same machine starts as fast as a CPU-only one. */
static int cuda_build_module(cuda_ctx *o)
{
    char *ptx = NULL, log[2048];
    int major = 5, minor = 0, cc, k, rc;
    CUresult cr;
    CUfunction *fn[10];
    size_t ptx_n = 0;

    p_cuDeviceGetAttribute(&major,
                           CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                           o->device);
    p_cuDeviceGetAttribute(&minor,
                           CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                           o->device);
    cc = major * 10 + minor;
    rc = ptx_cache_load(cc, &ptx, &ptx_n);
    if (rc != AEG_OK) {
        rc = cuda_compile_ptx(cc, &ptx, &ptx_n, log, sizeof(log));
        if (rc == AEG_EBACKEND) {
            int fb = cuda_fallback_arch(cc);
            if (fb && fb != cc) {
                aeg_log(AEG_LOG_WARNING, "aeg: NVRTC cannot target compute_%d, using compute_%d",
                        cc, fb);
                rc = cuda_compile_ptx(fb, &ptx, &ptx_n, log, sizeof(log));
                cc = fb;
            }
        }
        if (rc != AEG_OK) {
            aeg_log(AEG_LOG_WARNING, "aeg: NVRTC build failed (compute_%d): %s", cc, log);
            return rc;
        }
        ptx_cache_store(cc, ptx, ptx_n);
    }
    cr = p_cuModuleLoadDataEx(&o->mod, ptx, 0, NULL, NULL);
    free(ptx);
    if (cr != CUDA_SUCCESS)
        return AEG_EBACKEND;
    fn[0] = &o->k_sad; fn[1] = &o->k_tq; fn[2] = &o->k_iq;
    fn[3] = &o->k_sse; fn[4] = &o->k_intp; fn[5] = &o->k_satd;
    fn[6] = &o->k_tqiq; fn[7] = &o->k_tqdc; fn[8] = &o->k_iqdc;
    for (k = 0; aeg_cuda_kernels[k]; k++)
        if (p_cuModuleGetFunction(fn[k], o->mod, aeg_cuda_kernels[k]) !=
            CUDA_SUCCESS)
            return AEG_EBACKEND;
    return AEG_OK;
}

static int nvrtc_bind(void *lib_nvrtc)
{
#define BIND_NVRTC(ret, name, args) \
    if (!(*(void **)(&p_##name) = dlsym(lib_nvrtc, #name))) { \
        aeg_log(AEG_LOG_DEBUG, "aeg: NVRTC is missing %s", #name); \
        return AEG_EBACKEND; \
    }
    NVRTC_SYMS(BIND_NVRTC)
#undef BIND_NVRTC
    *(void **)(&p_nvrtcGetNumSupportedArchs) =
        dlsym(lib_nvrtc, "nvrtcGetNumSupportedArchs");
    *(void **)(&p_nvrtcGetSupportedArchs) =
        dlsym(lib_nvrtc, "nvrtcGetSupportedArchs");
    return AEG_OK;
}

/* Compile-only check of the device code, no GPU or driver needed: the exact
 * source, preamble and options the backend builds with, for compute_<cc>
 * (0: every architecture this NVRTC supports, newest first in `log` on
 * failure).  Checks that every kernel the backend looks up is an entry point
 * of the PTX.  AEG_EBACKEND when NVRTC cannot be loaded at all. */
int aeg_cuda_compile_check(int cc, char *log, size_t logsz, size_t *ptx_size)
{
    static void *lib_nvrtc;
    char *ptx = NULL;
    int rc, k;

    if (!lib_nvrtc) {
        if (nvrtc_load(&lib_nvrtc) != AEG_OK)
            return AEG_EBACKEND;
        if (nvrtc_bind(lib_nvrtc) != AEG_OK) {
            dlclose(lib_nvrtc);
            lib_nvrtc = NULL;
            return AEG_EBACKEND;
        }
    }
    if (cc <= 0) {
        int n = 0, *list, i;
        if (!p_nvrtcGetNumSupportedArchs || p_nvrtcGetNumSupportedArchs(&n) ||
            n <= 0)
            return aeg_cuda_compile_check(52, log, logsz, ptx_size);
        list = (int *)malloc((size_t)n * sizeof(int));
        if (!list)
            return AEG_ENOMEM;
        p_nvrtcGetSupportedArchs(list);
        rc = AEG_OK;
        for (i = n - 1; i >= 0 && rc == AEG_OK; i--)
            rc = aeg_cuda_compile_check(list[i], log, logsz, ptx_size);
        free(list);
        return rc;
    }
    rc = cuda_compile_ptx(cc, &ptx, ptx_size, log, logsz);
    if (rc != AEG_OK)
        return rc == AEG_EBACKEND ? AEG_EINVAL : rc;
    for (k = 0; aeg_cuda_kernels[k]; k++) {
        char want[64];
        snprintf(want, sizeof(want), ".entry %s(", aeg_cuda_kernels[k]);
        if (!strstr(ptx, want)) {
            snprintf(log, logsz, "compute_%d: no entry point %s in the PTX",
                     cc, aeg_cuda_kernels[k]);
            free(ptx);
            return AEG_EINVAL;
        }
    }
    free(ptx);
    return AEG_OK;
}

/* Which device to use.  AEG_CUDA_DEVICE (an ordinal, or a name substring to
 * match case-insensitively) picks one; the ordinal has to be in range.  Falls
 * back to device 0, which is what "the first NVIDIA GPU" has always meant. */
static int name_has(const char *name, const char *want)
{
    size_t i;
    for (i = 0; name[i] && want[i]; i++) {
        int a = (unsigned char)name[i], b = (unsigned char)want[i];
        if (a >= 'A' && a <= 'Z') a += 32;
        if (b >= 'A' && b <= 'Z') b += 32;
        if (a != b)
            return 0;
    }
    return want[i] == '\0';
}

static int cuda_pick_device(int count)
{
    const char *ev = getenv("AEG_CUDA_DEVICE");
    char want[96], name[96];
    int i;

    if (!ev || !*ev || count <= 0)
        return 0;
    if (*ev >= '0' && *ev <= '9') {
        char *end = NULL;
        long v = strtol(ev, &end, 10);
        if (end && *end == '\0') {
            if (v < 0 || v >= count) {
                aeg_log(AEG_LOG_WARNING,
                        "aeg: AEG_CUDA_DEVICE=%ld is out of range (0..%d), using 0",
                        v, count - 1);
                return 0;
            }
            return (int)v;
        }
    }
    snprintf(want, sizeof(want), "%s", ev);
    for (i = 0; i < count; i++) {
        CUdevice d;
        if (p_cuDeviceGet(&d, i) != CUDA_SUCCESS)
            continue;
        p_cuDeviceGetName(name, sizeof(name), d);
        if (name_has(name, want))
            return i;
    }
    return 0;
}

/* A driver handle on this context is per-thread, so a lane's stream is made
 * when the instance is created, on the thread that creates it. */
static int cuda_clone(void *priv, aeg_be_ops *out);

static int cuda_stream_make(cuda_ctx *o)
{
    if (p_cuStreamCreate(&o->st, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
        o->st = NULL;
        return AEG_EBACKEND;
    }
    return AEG_OK;
}

static int cuda_ops_fill(cuda_ctx *o, aeg_be_ops *ops)
{
    memset(ops, 0, sizeof(*ops));
    ops->priv = o;
    ops->backend = AEG_BACKEND_CUDA;
    ops->sad_batch = cuda_sad_batch;
    ops->tq_batch = cuda_tq_batch;
    ops->iq_idct_batch = cuda_iq_idct_batch;
    ops->tq_iq_batch = cuda_tq_iq_batch;
    ops->tq_dc_batch = cuda_tq_dc_batch;
    ops->iq_dc_batch = cuda_iq_dc_batch;
    ops->sse_batch = cuda_sse_batch;
    ops->interp_plane = cuda_interp_plane;
    ops->ref_upload = cuda_ref_upload;
    ops->satd_batch = cuda_satd_batch;
    ops->sync = cuda_sync;
    ops->destroy = cuda_destroy;
    ops->clone = cuda_clone;
    snprintf(ops->desc, sizeof(ops->desc), "CUDA %.48s (%d SMs)", o->name,
             o->cus);
    return AEG_OK;
}

/* Another instance for another thread (aeg_be_ops.clone): the context, the
 * module and its functions are shared (a context may be current on several
 * threads, and launches take their arguments per call); the buffers, the
 * registered reference, the staging arenas and the *stream* are its own, so
 * the lanes of the guard keep their work concurrent instead of queueing it
 * all behind one another on the default stream. */
static int cuda_clone(void *priv, aeg_be_ops *out)
{
    const cuda_ctx *src = (const cuda_ctx *)priv;
    cuda_ctx *o = (cuda_ctx *)calloc(1, sizeof(*o));

    if (!o)
        return AEG_ENOMEM;
    o->is_clone = 1;
    o->device = src->device;
    o->ctx = src->ctx;
    o->mod = src->mod;
    o->k_sad = src->k_sad;
    o->k_tq = src->k_tq;
    o->k_iq = src->k_iq;
    o->k_sse = src->k_sse;
    o->k_intp = src->k_intp;
    o->k_satd = src->k_satd;
    o->k_tqiq = src->k_tqiq;
    o->k_tqdc = src->k_tqdc;
    o->k_iqdc = src->k_iqdc;
    o->cus = src->cus;
    o->mem = src->mem;
    memcpy(o->name, src->name, sizeof(o->name));
    /* a stream belongs to the context, and the context is already current on
     * the thread that created it -- but this call comes from the thread that
     * is about to use the lane, so make the stream here and make the context
     * current on that thread first (cuda_bind below. */
    if (cuda_stream_make(o) != AEG_OK) {
        free(o);
        return AEG_EBACKEND;
    }
    cuda_ops_fill(o, out);
    return AEG_OK;
}

int aeg_be_cuda_create(aeg_be_ops *ops)
{
    cuda_ctx *o;
    void *lib_cuda = NULL, *lib_nvrtc = NULL;
    int count = 0, pick;

#define LOAD_CUDA(ret, name, args) do { \
        *(void **)(&p_##name) = dlsym(lib_cuda, #name); \
        if (!p_##name) { \
            aeg_log(AEG_LOG_DEBUG, "aeg: CUDA driver is missing %s", #name); \
            dlclose(lib_cuda); \
            return AEG_EBACKEND; \
        } \
    } while (0);
#define LOAD_NVRTC(ret, name, args) do { \
        *(void **)(&p_##name) = dlsym(lib_nvrtc, #name); \
        if (!p_##name) { \
            aeg_log(AEG_LOG_DEBUG, "aeg: NVRTC is missing %s", #name); \
            dlclose(lib_nvrtc); \
            dlclose(lib_cuda); \
            return AEG_EBACKEND; \
        } \
    } while (0);

    if (cuda_load(&lib_cuda) != AEG_OK)
        return AEG_EBACKEND;
    CUDA_SYMS(LOAD_CUDA)
    if (nvrtc_load(&lib_nvrtc) != AEG_OK) {
        dlclose(lib_cuda);
        return AEG_EBACKEND;
    }
    NVRTC_SYMS(LOAD_NVRTC)
    nvrtc_bind(lib_nvrtc);      /* the optional symbols */

    o = (cuda_ctx *)calloc(1, sizeof(*o));
    if (!o) {
        dlclose(lib_nvrtc);
        dlclose(lib_cuda);
        return AEG_ENOMEM;
    }
    o->lib_cuda = lib_cuda;
    o->lib_nvrtc = lib_nvrtc;

    if (p_cuInit(0) != CUDA_SUCCESS || p_cuDeviceGetCount(&count) != CUDA_SUCCESS
        || count < 1) {
        cuda_destroy(o);
        return AEG_EBACKEND;
    }
    pick = cuda_pick_device(count);
    if (p_cuDeviceGet(&o->device, pick) != CUDA_SUCCESS ||
        p_cuCtxCreate_v2(&o->ctx, 0, o->device) != CUDA_SUCCESS) {
        o->ctx = NULL;
        cuda_destroy(o);
        return AEG_EBACKEND;
    }
    p_cuDeviceGetName(o->name, sizeof(o->name), o->device);
    p_cuDeviceTotalMem_v2((size_t *)&o->mem, o->device);
    p_cuDeviceGetAttribute(&o->cus, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                           o->device);

    if (cuda_build_module(o) != AEG_OK) {
        cuda_destroy(o);
        return AEG_EBACKEND;
    }
    if (cuda_stream_make(o) != AEG_OK) {
        cuda_destroy(o);
        return AEG_EBACKEND;
    }

    cuda_ops_fill(o, ops);
    return AEG_OK;
}

/* Enumerate the CUDA devices.  This needs no context and no module, so it is
 * also the cheap answer to "is there a CUDA driver at all", which is what
 * --list-devices asks.  A device that cannot be queried is skipped, not fatal:
 * the first one that can be is enough to answer the question. */
int aeg_backend_query_cuda(aeg_device_info *out, int max)
{
    static void *lib_cuda;
    int count = 0, i, n = 0;

    if (max <= 0)
        return 0;
    if (!lib_cuda) {
        if (cuda_load(&lib_cuda) != AEG_OK)
            return 0;
    }
    /* the driver entry points the query uses, bound once (a later
     * aeg_be_cuda_create() binds the same set again -- the pointers are
     * process-wide, so this is safe either way) */
    {
#define BIND_Q(ret, name, args) do { \
            *(void **)(&p_##name) = dlsym(lib_cuda, #name); \
            if (!p_##name) \
                return 0; \
        } while (0);
        BIND_Q(CUresult, cuInit, (unsigned int))
        BIND_Q(CUresult, cuDeviceGetCount, (int *))
        BIND_Q(CUresult, cuDeviceGet, (CUdevice *, int))
        BIND_Q(CUresult, cuDeviceGetName, (char *, int, CUdevice))
        BIND_Q(CUresult, cuDeviceTotalMem_v2, (size_t *, CUdevice))
        BIND_Q(CUresult, cuDeviceGetAttribute, (int *, int, CUdevice))
#undef BIND_Q
    }
    if (p_cuInit(0) != CUDA_SUCCESS || p_cuDeviceGetCount(&count) != CUDA_SUCCESS)
        return 0;
    for (i = 0; i < count && n < max; i++) {
        CUdevice d;
        int cus = 0;
        uint64_t mem = 0;
        char name[192];
        if (p_cuDeviceGet(&d, i) != CUDA_SUCCESS)
            continue;
        memset(&out[n], 0, sizeof(out[n]));
        out[n].backend = AEG_BACKEND_CUDA;
        p_cuDeviceGetName(name, sizeof(name), d);
        snprintf(out[n].name, sizeof(out[n].name), "%s", name);
        snprintf(out[n].vendor, sizeof(out[n].vendor), "NVIDIA");
        out[n].is_gpu = 1;
        p_cuDeviceTotalMem_v2((size_t *)&mem, d);
        out[n].global_mem = mem;
        p_cuDeviceGetAttribute(&cus, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                               d);
        out[n].max_compute_units = cus;
        out[n].ordinal = i;
        n++;
    }
    return n;
}
