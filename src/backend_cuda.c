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
 * No NVENC, no NPP, no vendor video pipeline: the four kernels are the same
 * integer arithmetic as the CPU reference, and tests/kernel_check.c runs the
 * very same source against that reference on a machine with no GPU at all.
 */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "backend.h"
#include "kernels.h"

typedef int CUresult;
typedef int CUdevice;
typedef void *CUcontext;
typedef void *CUmodule;
typedef void *CUfunction;
typedef unsigned long long CUdeviceptr;
typedef void *CUstream;
typedef void *nvrtcProgram;
typedef int nvrtcResult;

#define CUDA_SUCCESS 0
#define CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT 16
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR 75
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR 76

/* The kernels are written in the common subset of OpenCL C and CUDA C; this
 * header turns the OpenCL spelling into the CUDA one.  The kernels are
 * declared `extern "C"` so that cuModuleGetFunction() finds them by name. */
static const char aeg_cuda_preamble[] =
    "typedef unsigned char uchar;\n"
    "typedef unsigned long long ulong;\n"
    "#define __kernel extern \"C\" __global__\n"
    "#define __global\n"
    "#define AEG_DEVFN static __device__\n"
    "#define get_global_id(d) ((int)(blockIdx.x * blockDim.x + threadIdx.x))\n"
    "#define AEG_ATOMIC_ADD(p, v) atomicAdd((p), (v))\n";

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
    X(CUresult, cuMemAllocHost_v2, (void **, size_t))                         \
    X(CUresult, cuMemFreeHost, (void *))                                      \
    X(CUresult, cuMemcpyHtoD_v2, (CUdeviceptr, const void *, size_t))         \
    X(CUresult, cuMemcpyDtoH_v2, (void *, CUdeviceptr, size_t))               \
    X(CUresult, cuMemcpyHtoDAsync_v2, (CUdeviceptr, const void *, size_t,      \
                                      CUstream))                             \
    X(CUresult, cuMemcpyDtoHAsync_v2, (void *, CUdeviceptr, size_t, CUstream))\
    X(CUresult, cuStreamCreate, (CUstream *, unsigned int))                   \
    X(CUresult, cuStreamDestroy_v2, (CUstream))                               \
    X(CUresult, cuStreamSynchronize, (CUstream))                              \
    X(CUresult, cuLaunchKernel, (CUfunction, unsigned int, unsigned int,      \
                                 unsigned int, unsigned int, unsigned int,    \
                                 unsigned int, unsigned int, CUstream,        \
                                 void **, void **))                           \
    X(CUresult, cuCtxSetCurrent, (CUcontext))                                 \
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

/* Optional since CUDA 4.0: retaining the primary context lets the encoder
 * share a host application's context instead of replacing it. */
typedef CUresult (*fn_primary_retain)(CUcontext *, CUdevice);
typedef CUresult (*fn_primary_release)(CUcontext);
static fn_primary_retain p_cuDevicePrimaryCtxRetain;
static fn_primary_release p_cuDevicePrimaryCtxRelease;

#define NBUF 32
#define AEG_BLOCK 256

typedef struct cuda_ctx {
    void *lib_cuda;
    void *lib_nvrtc;
    CUdevice device;
    CUcontext ctx;
    CUstream stream;            /* this lane's ordered work (not legacy NULL) */
    CUmodule mod;
    CUfunction k_sad, k_tq, k_iq, k_sse, k_intp, k_satd, k_tqiq, k_tqdc, k_iqdc;
    CUfunction k_me, k_lowres, k_mbt;
    CUdeviceptr lo_work[4];
    size_t lo_work_sz[4];
    CUdeviceptr mbt_work[7];
    size_t mbt_work_sz[7];
    CUdeviceptr buf[NBUF];
    size_t buf_sz[NBUF];
    int cus;
    uint64_t mem;
    int primary;            /* ctx is a retained primary context (do not destroy) */
    /* the registered reference picture (see ref_upload) */
    int ref_stride, ref_w, ref_h, ref_valid;
    /* device-resident luma pictures for me_batch (see aeg_be_ops) */
    CUdeviceptr mepic[AEG_ME_PIC_SLOTS];
    size_t mepic_sz[AEG_ME_PIC_SLOTS];
    uint64_t mepic_id[AEG_ME_PIC_SLOTS];
    int mepic_ok[AEG_ME_PIC_SLOTS];
    /* lowres ring (M-3.3a) */
    CUdeviceptr lo_ring;        /* AEG_LOWRES_SLOTS contiguous planes */
    size_t lo_ring_sz, lo_slot_bytes;
    int lo_stride, lo_w, lo_h;
    uint64_t lo_id[AEG_LOWRES_SLOTS];
    int lo_ok[AEG_LOWRES_SLOTS];
    char name[96];
    int16_t *pos;               /* sad_batch: the rebased positions */
    size_t pos_cap;
    void *pin;                  /* pinned staging for me_batch job/results */
    size_t pin_cap;
    int is_clone;               /* shares ctx/mod/libraries with the first */
} cuda_ctx;

/* ------------------------------------------------------------------ */
/* Buffer cache                                                        */
/* ------------------------------------------------------------------ */
static CUdeviceptr cuda_buf(cuda_ctx *o, int slot, size_t bytes)
{
    if (o->buf[slot] && o->buf_sz[slot] >= bytes)
        return o->buf[slot];
    if (o->buf[slot]) {
        p_cuMemFree_v2(o->buf[slot]);
        o->buf[slot] = 0;
    }
    o->buf_sz[slot] = bytes;
    if (p_cuMemAlloc_v2(&o->buf[slot], bytes) != CUDA_SUCCESS) {
        o->buf[slot] = 0;
        o->buf_sz[slot] = 0;
    }
    return o->buf[slot];
}

static int cuda_launch(CUfunction f, int count, void **args)
{
    unsigned int blocks = (unsigned int)((count + AEG_BLOCK - 1) / AEG_BLOCK);
    if (!blocks)
        return AEG_OK;
    if (p_cuLaunchKernel(f, blocks, 1, 1, AEG_BLOCK, 1, 1, 0, NULL, args,
                         NULL) != CUDA_SUCCESS)
        return AEG_EBACKEND;
    return AEG_OK;
}

static int cuda_launch_on(CUfunction f, int count, void **args, CUstream s)
{
    unsigned int blocks = (unsigned int)((count + AEG_BLOCK - 1) / AEG_BLOCK);
    if (!blocks)
        return AEG_OK;
    if (p_cuLaunchKernel(f, blocks, 1, 1, AEG_BLOCK, 1, 1, 0, s, args,
                         NULL) != CUDA_SUCCESS)
        return AEG_EBACKEND;
    return AEG_OK;
}

/* A driver-API context is current on the thread that created it only, and the
 * operations are called from slice threads too (one at a time: the backend
 * guard in backend.c serialises them), so every entry point makes it current
 * on the calling thread first. */
static int cuda_bind(cuda_ctx *o)
{
    return p_cuCtxSetCurrent(o->ctx) == CUDA_SUCCESS ? AEG_OK : AEG_EBACKEND;
}

/* ------------------------------------------------------------------ */
/* Operations                                                          */
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
    dcur = cuda_buf(o, 0, sp.cbytes);
    dref = cuda_buf(o, 1, sp.rbytes);
    dmv = cuda_buf(o, 2, (size_t)count * 8);
    dout = cuda_buf(o, 3, (size_t)count * 4);
    if (!dcur || !dref || !dmv || !dout)
        return AEG_EBACKEND;
    if (p_cuMemcpyHtoD_v2(dcur, cur + sp.coff, sp.cbytes) != CUDA_SUCCESS ||
        p_cuMemcpyHtoD_v2(dref, ref + sp.roff, sp.rbytes) != CUDA_SUCCESS ||
        p_cuMemcpyHtoD_v2(dmv, o->pos, (size_t)count * 8) != CUDA_SUCCESS)
        return AEG_EBACKEND;

    args[0] = &dcur; args[1] = &icur; args[2] = &dref; args[3] = &iref;
    args[4] = &ibw;  args[5] = &ibh;  args[6] = &icount; args[7] = &dmv;
    args[8] = &dout;
    st = cuda_launch(o->k_sad, count, args);
    if (st != AEG_OK)
        return st;
    if (p_cuMemcpyDtoH_v2(out, dout, (size_t)count * 4) != CUDA_SUCCESS)
        return AEG_EBACKEND;
    return AEG_OK;
}

/* Whole-picture integer motion search (see aeg_be_ops.me_batch). */
static int cuda_me_pic_upload(void *priv, int slot, const uint8_t *plane,
                              int stride, int w, int h, int rows, uint64_t id)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    CUdeviceptr d;
    size_t bytes;
    (void)w; (void)h;
    if (slot < 0 || slot >= AEG_ME_PIC_SLOTS)
        return AEG_EINVAL;
    if (cuda_bind(o) != AEG_OK)
        return AEG_EBACKEND;
    if (rows <= 0)
        return AEG_EINVAL;
    bytes = (size_t)stride * (size_t)rows;
    {
        int k;
        for (k = 0; k < AEG_ME_PIC_SLOTS; k++)
            if (o->mepic_ok[k] && o->mepic_id[k] == id &&
                o->mepic_sz[k] >= bytes)
                return AEG_OK;
    }
    d = cuda_buf(o, 8 + slot, bytes);
    if (!d)
        return AEG_EBACKEND;
    if (p_cuMemcpyHtoDAsync_v2(d, plane - (ptrdiff_t)AEG_PAD * stride - AEG_PAD,
                               bytes, o->stream) != CUDA_SUCCESS)
        return AEG_EBACKEND;
    o->mepic[slot] = d;
    o->mepic_sz[slot] = bytes;
    o->mepic_id[slot] = id;
    o->mepic_ok[slot] = 1;
    return AEG_OK;
}

/* Lowres ring (M-3.3a): one 8-bit luma plane per slot, uploaded once and
 * reused by every later SAD batch that names the slot. */
static int cuda_lowres_upload(void *priv, int slot, const uint8_t *plane,
                              int stride, int w, int h, uint64_t id)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    size_t bytes;
    int k;
    if (slot < 0 || slot >= AEG_LOWRES_SLOTS)
        return AEG_EINVAL;
    if (cuda_bind(o) != AEG_OK)
        return AEG_EBACKEND;
    bytes = (size_t)stride * (size_t)(h + 2 * AEG_LA_PAD);
    if (bytes != o->lo_slot_bytes || stride != o->lo_stride) {
        if (o->lo_ring)
            p_cuMemFree_v2(o->lo_ring);
        o->lo_ring = 0;
        o->lo_ring_sz = 0;
        o->lo_slot_bytes = 0;
        for (k = 0; k < AEG_LOWRES_SLOTS; k++)
            o->lo_ok[k] = 0;
        if (p_cuMemAlloc_v2(&o->lo_ring, bytes * AEG_LOWRES_SLOTS) !=
            CUDA_SUCCESS) {
            o->lo_ring = 0;
            return AEG_EBACKEND;
        }
        o->lo_ring_sz = bytes * AEG_LOWRES_SLOTS;
        o->lo_slot_bytes = bytes;
        o->lo_stride = stride;
        o->lo_w = w;
        o->lo_h = h;
    }
    if (o->lo_ok[slot] && o->lo_id[slot] == id)
        return AEG_OK;
    if (p_cuMemcpyHtoD_v2(o->lo_ring + (size_t)slot * o->lo_slot_bytes,
                          plane - (ptrdiff_t)AEG_LA_PAD * stride - AEG_LA_PAD,
                          bytes) != CUDA_SUCCESS)
        return AEG_EBACKEND;
    o->lo_id[slot] = id;
    o->lo_ok[slot] = 1;
    return AEG_OK;
}

static int cuda_lowres_sad_batch(void *priv, int cur_slot, int ref_slot, int bw,
                                 int bh, const int16_t *mv, int count,
                                 int32_t *out)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    CUdeviceptr dcur, dref, dmv, dout;
    int icur, iref, ibw = bw, ibh = bh, icount = count;
    void *args[9];
    int st;
    if (cur_slot < 0 || cur_slot >= AEG_LOWRES_SLOTS ||
        ref_slot < 0 || ref_slot >= AEG_LOWRES_SLOTS ||
        !o->lo_ok[cur_slot] || !o->lo_ok[ref_slot])
        return AEG_EINVAL;
    if (count <= 0)
        return AEG_OK;
    if (cuda_bind(o) != AEG_OK)
        return AEG_EBACKEND;
    icur = o->lo_stride;
    iref = o->lo_stride;
    /* the resident buffer starts at the padding; the SAD kernel takes the
     * origin (the host wrapper rebases to the same place) */
    dcur = o->lo_ring + (size_t)cur_slot * o->lo_slot_bytes +
           (ptrdiff_t)AEG_LA_PAD * icur + AEG_LA_PAD;
    dref = o->lo_ring + (size_t)ref_slot * o->lo_slot_bytes +
           (ptrdiff_t)AEG_LA_PAD * iref + AEG_LA_PAD;
    dmv = cuda_buf(o, 2, (size_t)count * 8);
    dout = cuda_buf(o, 3, (size_t)count * 4);
    if (!dmv || !dout)
        return AEG_EBACKEND;
    if (p_cuMemcpyHtoDAsync_v2(dmv, mv, (size_t)count * 8, o->stream) !=
        CUDA_SUCCESS)
        return AEG_EBACKEND;
    args[0] = &dcur; args[1] = &icur; args[2] = &dref; args[3] = &iref;
    args[4] = &ibw;  args[5] = &ibh;  args[6] = &icount;
    args[7] = &dmv;  args[8] = &dout;
    st = cuda_launch_on(o->k_sad, count, args, o->stream);
    if (st != AEG_OK)
        return st;
    if (p_cuMemcpyDtoHAsync_v2(out, dout, (size_t)count * 4, o->stream) !=
        CUDA_SUCCESS)
        return AEG_EBACKEND;
    if (p_cuStreamSynchronize(o->stream) != CUDA_SUCCESS)
        return AEG_EBACKEND;
    return AEG_OK;
}

static CUdeviceptr lo_work_buf(cuda_ctx *o, int slot, size_t bytes)
{
    if (o->lo_work[slot] && o->lo_work_sz[slot] >= bytes)
        return o->lo_work[slot];
    if (o->lo_work[slot])
        p_cuMemFree_v2(o->lo_work[slot]);
    o->lo_work[slot] = 0;
    o->lo_work_sz[slot] = 0;
    if (p_cuMemAlloc_v2(&o->lo_work[slot], bytes) != CUDA_SUCCESS)
        return 0;
    o->lo_work_sz[slot] = bytes;
    return o->lo_work[slot];
}

static int cuda_lowres_batch_search(void *priv, const int32_t *pairs, int npairs,
                                    int mb_w, int mb_h, int iters,
                                    int32_t *mv, int32_t *cost)
{
    static const int steps[] = { 16, 16, 8, 8, 4, 4, 2, 2, 1, 1 };
    cuda_ctx *o = (cuda_ctx *)priv;
    CUdeviceptr dpairs, dprev, dnext, dcost;
    int32_t *zero = NULL;
    int nblk = mb_w * mb_h, n = npairs * nblk, it;
    int istep = 0, ilast = 0;
    void *args[12];
    int st;
    if (!o->lo_ring || npairs <= 0 || nblk <= 0)
        return AEG_EINVAL;
    if (iters <= 0 || iters > (int)(sizeof(steps) / sizeof(steps[0])))
        iters = 5;
    if (cuda_bind(o) != AEG_OK)
        return AEG_EBACKEND;
    dpairs = lo_work_buf(o, 0, (size_t)npairs * 8);
    dprev = lo_work_buf(o, 1, (size_t)n * 8);
    dnext = lo_work_buf(o, 2, (size_t)n * 8);
    dcost = lo_work_buf(o, 3, (size_t)n * 4);
    if (!dpairs || !dprev || !dnext || !dcost)
        return AEG_EBACKEND;
    zero = (int32_t *)calloc((size_t)n * 2, sizeof(int32_t));
    if (!zero)
        return AEG_ENOMEM;
    if (p_cuMemcpyHtoDAsync_v2(dpairs, pairs, (size_t)npairs * 8, o->stream) !=
            CUDA_SUCCESS ||
        p_cuMemcpyHtoDAsync_v2(dprev, zero, (size_t)n * 8, o->stream) !=
            CUDA_SUCCESS) {
        free(zero);
        return AEG_EBACKEND;
    }
    free(zero);
    args[0] = &o->lo_ring; args[1] = &o->lo_stride;
    args[2] = &o->lo_slot_bytes; args[3] = &mb_w; args[4] = &mb_h;
    args[5] = &istep; args[6] = &ilast; args[7] = &dpairs; args[8] = &npairs;
    args[9] = &dprev; args[10] = &dnext; args[11] = &dcost;
    for (it = 0; it < iters; it++) {
        CUdeviceptr tmp;
        istep = steps[it];
        ilast = it == iters - 1;
        st = cuda_launch_on(o->k_lowres, n, args, o->stream);
        if (st != AEG_OK)
            return st;
        tmp = dprev; dprev = dnext; dnext = tmp;
        args[9] = &dprev; args[10] = &dnext;
    }
    if (p_cuMemcpyDtoHAsync_v2(mv, dprev, (size_t)n * 8, o->stream) !=
            CUDA_SUCCESS ||
        p_cuMemcpyDtoHAsync_v2(cost, dcost, (size_t)n * 4, o->stream) !=
            CUDA_SUCCESS)
        return AEG_EBACKEND;
    if (p_cuStreamSynchronize(o->stream) != CUDA_SUCCESS)
        return AEG_EBACKEND;
    return AEG_OK;
}

static CUdeviceptr mbt_work_buf(cuda_ctx *o, int slot, size_t bytes)
{
    if (o->mbt_work[slot] && o->mbt_work_sz[slot] >= bytes)
        return o->mbt_work[slot];
    if (o->mbt_work[slot])
        p_cuMemFree_v2(o->mbt_work[slot]);
    o->mbt_work[slot] = 0;
    o->mbt_work_sz[slot] = 0;
    if (p_cuMemAlloc_v2(&o->mbt_work[slot], bytes) != CUDA_SUCCESS)
        return 0;
    o->mbt_work_sz[slot] = bytes;
    return o->mbt_work[slot];
}

static int cuda_mbt_batch(void *priv, int npics, int mb_w, int mb_h,
                          const int *intra, const int *inter,
                          const uint8_t *kind, const int *mv, const int *r0,
                          const int *r1, int w0, int32_t *prop)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    CUdeviceptr di, dinter, dkind, dmv, dr0, dr1, dprop;
    int nmb = mb_w * mb_h, n = npics * nmb, p, st;
    size_t ib = (size_t)n * 4, kb = (size_t)n;
    void *args[12];
    if (!npics || nmb <= 0)
        return AEG_OK;
    if (cuda_bind(o) != AEG_OK)
        return AEG_EBACKEND;
    di = mbt_work_buf(o, 0, ib);
    dinter = mbt_work_buf(o, 1, ib);
    dkind = mbt_work_buf(o, 2, kb);
    dmv = mbt_work_buf(o, 3, (size_t)n * 16);
    dr0 = mbt_work_buf(o, 4, (size_t)npics * 4);
    dr1 = mbt_work_buf(o, 5, (size_t)npics * 4);
    dprop = mbt_work_buf(o, 6, ib);
    if (!di || !dinter || !dkind || !dmv || !dr0 || !dr1 || !dprop)
        return AEG_EBACKEND;
    if (p_cuMemcpyHtoDAsync_v2(di, intra, ib, o->stream) != CUDA_SUCCESS ||
        p_cuMemcpyHtoDAsync_v2(dinter, inter, ib, o->stream) != CUDA_SUCCESS ||
        p_cuMemcpyHtoDAsync_v2(dkind, kind, kb, o->stream) != CUDA_SUCCESS ||
        p_cuMemcpyHtoDAsync_v2(dmv, mv, (size_t)n * 16, o->stream) != CUDA_SUCCESS ||
        p_cuMemcpyHtoDAsync_v2(dr0, r0, (size_t)npics * 4, o->stream) != CUDA_SUCCESS ||
        p_cuMemcpyHtoDAsync_v2(dr1, r1, (size_t)npics * 4, o->stream) != CUDA_SUCCESS ||
        p_cuMemcpyHtoDAsync_v2(dprop, prop, ib, o->stream) != CUDA_SUCCESS)
        return AEG_EBACKEND;
    args[0] = &di; args[1] = &dinter; args[2] = &dkind; args[3] = &dmv;
    args[4] = &dr0; args[5] = &dr1; args[6] = &w0;
    args[7] = &mb_w; args[8] = &mb_h; args[10] = &nmb; args[11] = &dprop;
    for (p = npics - 1; p >= 0; p--) {
        args[9] = &p;
        st = cuda_launch_on(o->k_mbt, nmb, args, o->stream);
        if (st != AEG_OK)
            return st;
    }
    if (p_cuMemcpyDtoHAsync_v2(prop, dprop, ib, o->stream) != CUDA_SUCCESS)
        return AEG_EBACKEND;
    if (p_cuStreamSynchronize(o->stream) != CUDA_SUCCESS)
        return AEG_EBACKEND;
    return AEG_OK;
}

static int cuda_mepic_find(cuda_ctx *o, uint64_t id, size_t bytes)
{
    int i;
    for (i = 0; i < AEG_ME_PIC_SLOTS; i++)
        if (o->mepic_ok[i] && o->mepic_id[i] == id && o->mepic_sz[i] >= bytes)
            return i;
    return -1;
}

static int cuda_rec_upload(void *priv, int slot, const uint8_t *y,
                           const uint8_t *u, const uint8_t *v, int ys,
                           int cs, int w, int h, uint64_t id)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    size_t cbytes;
    CUdeviceptr du, dv;
    int rows = (h + 1) / 2 + 2 * AEG_PAD;
    int st;
    if (slot < 2 || slot >= AEG_ME_PIC_SLOTS)
        return AEG_EINVAL;
    if (cuda_bind(o) != AEG_OK)
        return AEG_EBACKEND;
    st = cuda_me_pic_upload(o, slot, y, ys, w, h,
                            h + 2 * AEG_PAD, id);
    if (st != AEG_OK)
        return st;
    cbytes = (size_t)cs * (size_t)rows;
    du = cuda_buf(o, 24 + slot * 2, cbytes);
    dv = cuda_buf(o, 25 + slot * 2, cbytes);
    if (!du || !dv)
        return AEG_EBACKEND;
    if (p_cuMemcpyHtoD_v2(du, u - (ptrdiff_t)AEG_PAD * cs - AEG_PAD,
                          cbytes) != CUDA_SUCCESS ||
        p_cuMemcpyHtoD_v2(dv, v - (ptrdiff_t)AEG_PAD * cs - AEG_PAD,
                          cbytes) != CUDA_SUCCESS)
        return AEG_EBACKEND;
    return AEG_OK;
}

static int cuda_me_batch(void *priv, const uint8_t *cur, int cstride,
                         const uint8_t *ref, int rstride, int bw, int bh,
                         int range, const int16_t *job, int count,
                         int avail_rows, int16_t *out_mv, int32_t *out_sad,
                         uint64_t cur_id, uint64_t ref_id)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    CUdeviceptr dcur, dref, djob, dmv, dsad;
    size_t cbytes;
    int hi, icur = cstride, iref = rstride, ipad = AEG_PAD, ibw = bw, ibh = bh;
    int irange = range;
    int icount = count;
    void *args[12];
    int st;
    if (cuda_bind(o) != AEG_OK)
        return AEG_EBACKEND;
    if (count <= 0)
        return AEG_OK;
    hi = 0;
    {
        int i;
        for (i = 0; i < count; i++)
            if (job[4 * i + 1] > hi)
                hi = job[4 * i + 1];
    }
    /* never read past the caller's allocation: the bottom macroblock row of a
     * picture whose height is not a multiple of 16 reaches into the padding,
     * and the padding itself is  rows deep at most */
    hi = hi + bh + 2 * AEG_PAD;
    if (avail_rows > 0 && hi > avail_rows)
        hi = avail_rows;
    cbytes = (size_t)cstride * (size_t)hi;
    {
        int ci = cuda_mepic_find(o, cur_id, cbytes);
        int ri = cuda_mepic_find(o, ref_id, cbytes);
        dcur = ci >= 0 ? o->mepic[ci] : cuda_buf(o, 0, cbytes);
        dref = ri >= 0 ? o->mepic[ri] : cuda_buf(o, 1, cbytes);
        if (!dcur || !dref)
            return AEG_EBACKEND;
        if ((ci < 0 &&
             p_cuMemcpyHtoDAsync_v2(dcur,
                                    cur - (ptrdiff_t)AEG_PAD * cstride - AEG_PAD,
                                    cbytes, o->stream) != CUDA_SUCCESS) ||
            (ri < 0 &&
             p_cuMemcpyHtoDAsync_v2(dref,
                                    ref - (ptrdiff_t)AEG_PAD * rstride - AEG_PAD,
                                    cbytes, o->stream) != CUDA_SUCCESS))
            return AEG_EBACKEND;
    }
    djob = cuda_buf(o, 2, (size_t)count * 8);
    dmv = cuda_buf(o, 3, (size_t)count * 4);
    dsad = cuda_buf(o, 4, (size_t)count * 4);
    if (!djob || !dmv || !dsad)
        return AEG_EBACKEND;
    {
        /* Pinned staging: the job list and both results are page-locked, so
         * the asynchronous copies do not bounce through a driver buffer. */
        const size_t jb = (size_t)count * 8, mb = (size_t)count * 4;
        const size_t need = jb + 2 * mb;
        void *pjob, *pmv, *psad;
        if (o->pin_cap < need) {
            if (o->pin)
                p_cuMemFreeHost(o->pin);
            o->pin = NULL;
            o->pin_cap = 0;
            if (p_cuMemAllocHost_v2(&o->pin, need) != CUDA_SUCCESS) {
                o->pin = NULL;
                return AEG_EBACKEND;
            }
            o->pin_cap = need;
        }
        pjob = o->pin;
        pmv = (char *)pjob + jb;
        psad = (char *)pmv + mb;
        memcpy(pjob, job, jb);
        if (p_cuMemcpyHtoDAsync_v2(djob, pjob, jb, o->stream) != CUDA_SUCCESS)
            return AEG_EBACKEND;
        args[0] = &dcur;  args[1] = &icur;  args[2] = &dref;   args[3] = &iref;
        args[4] = &ipad;  args[5] = &ibw;   args[6] = &ibh;    args[7] = &irange;
        args[8] = &djob;  args[9] = &icount; args[10] = &dmv;  args[11] = &dsad;
        st = cuda_launch_on(o->k_me, count, args, o->stream);
        if (st != AEG_OK)
            return st;
        if (p_cuMemcpyDtoHAsync_v2(pmv, dmv, mb, o->stream) != CUDA_SUCCESS ||
            p_cuMemcpyDtoHAsync_v2(psad, dsad, mb, o->stream) != CUDA_SUCCESS)
            return AEG_EBACKEND;
        if (p_cuStreamSynchronize(o->stream) != CUDA_SUCCESS)
            return AEG_EBACKEND;
        memcpy(out_mv, pmv, mb);
        memcpy(out_sad, psad, mb);
    }
    return AEG_OK;
}

static int cuda_tq_batch(void *priv, const int16_t *res, int nblk, int qp,
                         int intra, int16_t *level, int32_t *nz)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    CUdeviceptr dres, dlev, dnz;
    size_t bytes = (size_t)nblk * 16 * 2;
    int qbits = 15 + qp / 6;
    int f = aeg_quant_bias(qbits, intra ? 3 : 6);
    int mf0 = aeg_quant_mf[qp % 6][0], mf1 = aeg_quant_mf[qp % 6][1],
        mf2 = aeg_quant_mf[qp % 6][2];
    void *args[9];
    int st;
    if (cuda_bind(o) != AEG_OK)
        return AEG_EBACKEND;
    if (nblk <= 0)
        return AEG_OK;

    dres = cuda_buf(o, 0, bytes);
    dlev = cuda_buf(o, 1, bytes);
    dnz = cuda_buf(o, 2, (size_t)nblk * 4);
    if (!dres || !dlev || !dnz)
        return AEG_EBACKEND;
    if (p_cuMemcpyHtoD_v2(dres, res, bytes) != CUDA_SUCCESS)
        return AEG_EBACKEND;
    args[0] = &dres; args[1] = &nblk; args[2] = &qbits; args[3] = &f;
    args[4] = &mf0;  args[5] = &mf1;  args[6] = &mf2;
    args[7] = &dlev; args[8] = &dnz;
    st = cuda_launch(o->k_tq, nblk, args);
    if (st != AEG_OK)
        return st;
    if (p_cuMemcpyDtoH_v2(level, dlev, bytes) != CUDA_SUCCESS ||
        (nz && p_cuMemcpyDtoH_v2(nz, dnz, (size_t)nblk * 4) != CUDA_SUCCESS))
        return AEG_EBACKEND;
    return AEG_OK;
}

static int cuda_iq_idct_batch(void *priv, const int16_t *level, int nblk,
                              int qp, int16_t *res)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    CUdeviceptr dlev, dres;
    size_t bytes = (size_t)nblk * 16 * 2;
    int sh = qp / 6;
    int ls0 = aeg_dequant_ls[qp % 6][0], ls1 = aeg_dequant_ls[qp % 6][1],
        ls2 = aeg_dequant_ls[qp % 6][2];
    void *args[7];
    int st;
    if (cuda_bind(o) != AEG_OK)
        return AEG_EBACKEND;
    if (nblk <= 0)
        return AEG_OK;

    dlev = cuda_buf(o, 0, bytes);
    dres = cuda_buf(o, 1, bytes);
    if (!dlev || !dres)
        return AEG_EBACKEND;
    if (p_cuMemcpyHtoD_v2(dlev, level, bytes) != CUDA_SUCCESS)
        return AEG_EBACKEND;
    args[0] = &dlev; args[1] = &nblk; args[2] = &sh;
    args[3] = &ls0;  args[4] = &ls1;  args[5] = &ls2; args[6] = &dres;
    st = cuda_launch(o->k_iq, nblk, args);
    if (st != AEG_OK)
        return st;
    if (p_cuMemcpyDtoH_v2(res, dres, bytes) != CUDA_SUCCESS)
        return AEG_EBACKEND;
    return AEG_OK;
}

static int cuda_tq_iq_batch(void *priv, const int16_t *res, int nblk, int qp,
                            int intra, int16_t *level, int32_t *nz,
                            int16_t *rec)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    CUdeviceptr dres, dlev, dnz, drec;
    size_t bytes = (size_t)nblk * 16 * 2;
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

    dres = cuda_buf(o, 0, bytes);
    dlev = cuda_buf(o, 1, bytes);
    dnz = cuda_buf(o, 2, (size_t)nblk * 4);
    drec = cuda_buf(o, 3, bytes);
    if (!dres || !dlev || !dnz || !drec)
        return AEG_EBACKEND;
    if (p_cuMemcpyHtoD_v2(dres, res, bytes) != CUDA_SUCCESS)
        return AEG_EBACKEND;
    args[0] = &dres; args[1] = &nblk; args[2] = &qbits; args[3] = &f;
    args[4] = &mf0;  args[5] = &mf1;  args[6] = &mf2;  args[7] = &sh;
    args[8] = &ls0;  args[9] = &ls1;  args[10] = &ls2;
    args[11] = &dlev; args[12] = &dnz; args[13] = &drec;
    st = cuda_launch(o->k_tqiq, nblk, args);
    if (st != AEG_OK)
        return st;
    if (p_cuMemcpyDtoH_v2(level, dlev, bytes) != CUDA_SUCCESS ||
        (nz && p_cuMemcpyDtoH_v2(nz, dnz, (size_t)nblk * 4) != CUDA_SUCCESS) ||
        p_cuMemcpyDtoH_v2(rec, drec, bytes) != CUDA_SUCCESS)
        return AEG_EBACKEND;
    return AEG_OK;
}

static int cuda_tq_dc_batch(void *priv, const int16_t *res, int nblk, int qp,
                            int intra, int16_t *level, int32_t *nz,
                            int32_t *dc)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    CUdeviceptr dres, dlev, dnz, ddc;
    size_t bytes = (size_t)nblk * 16 * 2;
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

    dres = cuda_buf(o, 0, bytes);
    dlev = cuda_buf(o, 1, bytes);
    dnz = cuda_buf(o, 2, (size_t)nblk * 4);
    ddc = cuda_buf(o, 3, (size_t)nblk * 4);
    if (!dres || !dlev || !dnz || !ddc)
        return AEG_EBACKEND;
    if (p_cuMemcpyHtoD_v2(dres, res, bytes) != CUDA_SUCCESS)
        return AEG_EBACKEND;
    args[0] = &dres; args[1] = &nblk; args[2] = &qbits; args[3] = &f;
    args[4] = &mf0;  args[5] = &mf1;  args[6] = &mf2;
    args[7] = &dlev; args[8] = &dnz;  args[9] = &ddc;
    st = cuda_launch(o->k_tqdc, nblk, args);
    if (st != AEG_OK)
        return st;
    if (p_cuMemcpyDtoH_v2(level, dlev, bytes) != CUDA_SUCCESS ||
        (nz && p_cuMemcpyDtoH_v2(nz, dnz, (size_t)nblk * 4) != CUDA_SUCCESS) ||
        p_cuMemcpyDtoH_v2(dc, ddc, (size_t)nblk * 4) != CUDA_SUCCESS)
        return AEG_EBACKEND;
    return AEG_OK;
}

static int cuda_iq_dc_batch(void *priv, const int16_t *level,
                            const int32_t *dc, int nblk, int qp, int16_t *res)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    CUdeviceptr dlev, ddc, dres;
    size_t bytes = (size_t)nblk * 16 * 2;
    int sh = qp / 6;
    int ls0 = aeg_dequant_ls[qp % 6][0], ls1 = aeg_dequant_ls[qp % 6][1],
        ls2 = aeg_dequant_ls[qp % 6][2];
    void *args[8];
    int st;
    if (nblk <= 0)
        return AEG_OK;
    if (cuda_bind(o) != AEG_OK)
        return AEG_EBACKEND;

    dlev = cuda_buf(o, 0, bytes);
    ddc = cuda_buf(o, 1, (size_t)nblk * 4);
    dres = cuda_buf(o, 2, bytes);
    if (!dlev || !ddc || !dres)
        return AEG_EBACKEND;
    if (p_cuMemcpyHtoD_v2(dlev, level, bytes) != CUDA_SUCCESS ||
        p_cuMemcpyHtoD_v2(ddc, dc, (size_t)nblk * 4) != CUDA_SUCCESS)
        return AEG_EBACKEND;
    args[0] = &dlev; args[1] = &ddc; args[2] = &nblk; args[3] = &sh;
    args[4] = &ls0;  args[5] = &ls1; args[6] = &ls2;  args[7] = &dres;
    st = cuda_launch(o->k_iqdc, nblk, args);
    if (st != AEG_OK)
        return st;
    if (p_cuMemcpyDtoH_v2(res, dres, bytes) != CUDA_SUCCESS)
        return AEG_EBACKEND;
    return AEG_OK;
}

static int cuda_sse_batch(void *priv, const uint8_t *a, int sa, const uint8_t *b,
                          int sb, int w, int h, uint64_t *out)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    CUdeviceptr da, db, dout;
    size_t tw = (size_t)w / 8, th = (size_t)h / 8, count = tw * th;
    size_t abytes = (size_t)sa * h, bbytes = (size_t)sb * h;
    if (cuda_bind(o) != AEG_OK)
        return AEG_EBACKEND;
    uint64_t *tmp;
    size_t i, acc = 0;
    int itw = (int)tw, ith = (int)th, icount = (int)count;
    void *args[8];
    int st;

    if (!count) {
        *out = 0;
        return AEG_OK;
    }
    da = cuda_buf(o, 0, abytes);
    db = cuda_buf(o, 1, bbytes);
    dout = cuda_buf(o, 2, count * 8);
    if (!da || !db || !dout)
        return AEG_EBACKEND;
    if (p_cuMemcpyHtoD_v2(da, a, abytes) != CUDA_SUCCESS ||
        p_cuMemcpyHtoD_v2(db, b, bbytes) != CUDA_SUCCESS)
        return AEG_EBACKEND;
    args[0] = &da; args[1] = &sa; args[2] = &db; args[3] = &sb;
    args[4] = &itw; args[5] = &ith; args[6] = &icount; args[7] = &dout;
    st = cuda_launch(o->k_sse, (int)count, args);
    if (st != AEG_OK)
        return st;
    tmp = (uint64_t *)malloc(count * 8);
    if (!tmp)
        return AEG_ENOMEM;
    if (p_cuMemcpyDtoH_v2(tmp, dout, count * 8) != CUDA_SUCCESS) {
        free(tmp);
        return AEG_EBACKEND;
    }
    for (i = 0; i < count; i++)
        acc += (size_t)tmp[i];
    free(tmp);
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
    int pad = AEG_PAD;
    void *args[8];
    int st;
    if (cuda_bind(o) != AEG_OK)
        return AEG_EBACKEND;

    dref = cuda_buf(o, 0, rbytes);
    ddst = cuda_buf(o, 1, dbytes);
    if (!dref || !ddst)
        return AEG_EBACKEND;
    /* the kernel takes the whole padded allocations, see backend_opencl.c */
    if (p_cuMemcpyHtoD_v2(dref, ref - (ptrdiff_t)AEG_PAD * rstride - AEG_PAD,
                          rbytes) != CUDA_SUCCESS)
        return AEG_EBACKEND;
    args[0] = &dref; args[1] = &rstride; args[2] = &w; args[3] = &h;
    args[4] = &pad;  args[5] = &which;   args[6] = &ddst; args[7] = &dstride;
    st = cuda_launch(o->k_intp, (int)count, args);
    if (st != AEG_OK)
        return st;
    if (p_cuMemcpyDtoH_v2(dst - (ptrdiff_t)AEG_PAD * dstride - AEG_PAD, ddst,
                          dbytes) != CUDA_SUCCESS)
        return AEG_EBACKEND;
    return AEG_OK;
}

/* Register the reference picture (four planes, resident in device memory). */
static int cuda_ref_upload(void *priv, const uint8_t *const planes[4],
                           int pstride, int w, int h)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    size_t rows = (size_t)h + 2 * AEG_PAD;
    size_t bytes = (size_t)pstride * rows;
    int k;
    if (cuda_bind(o) != AEG_OK)
        return AEG_EBACKEND;

    for (k = 0; k < 4; k++) {
        CUdeviceptr d = cuda_buf(o, 4 + k, bytes);
        if (!d)
            return AEG_EBACKEND;
        if (p_cuMemcpyHtoD_v2(d, planes[k] - (ptrdiff_t)AEG_PAD * pstride - AEG_PAD,
                              bytes) != CUDA_SUCCESS)
            return AEG_EBACKEND;
    }
    o->ref_stride = pstride;
    o->ref_w = w;
    o->ref_h = h;
    o->ref_valid = 1;
    return AEG_OK;
}

static int cuda_satd_batch(void *priv, const uint8_t *cur, int cstride, int bw,
                           int bh, const int16_t *pos, int count, int32_t *out)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    CUdeviceptr dcur, dpos, dout, dref[4];
    size_t rows = (size_t)(o->ref_valid ? o->ref_h : 0) + 2 * AEG_PAD;
    size_t cbytes = (size_t)cstride * rows;
    int pad = AEG_PAD, k, st;
    void *args[13];
    if (cuda_bind(o) != AEG_OK)
        return AEG_EBACKEND;

    if (!o->ref_valid || !count)
        return AEG_EBACKEND;
    dcur = cuda_buf(o, 0, cbytes);
    dpos = cuda_buf(o, 1, (size_t)count * 8);
    dout = cuda_buf(o, 2, (size_t)count * 4);
    if (!dcur || !dpos || !dout)
        return AEG_EBACKEND;
    for (k = 0; k < 4; k++)
        dref[k] = o->buf[4 + k];
    if (p_cuMemcpyHtoD_v2(dcur, cur - (ptrdiff_t)AEG_PAD * cstride - AEG_PAD, cbytes) != CUDA_SUCCESS ||
        p_cuMemcpyHtoD_v2(dpos, pos, (size_t)count * 8) != CUDA_SUCCESS)
        return AEG_EBACKEND;
    args[0] = &dcur;  args[1] = &cstride;
    for (k = 0; k < 4; k++)
        args[2 + k] = &dref[k];
    args[6] = &o->ref_stride; args[7] = &pad;
    args[8] = &bw;   args[9] = &bh;
    args[10] = &dpos; args[11] = &count; args[12] = &dout;
    st = cuda_launch(o->k_satd, count, args);
    if (st != AEG_OK)
        return st;
    if (p_cuMemcpyDtoH_v2(out, dout, (size_t)count * 4) != CUDA_SUCCESS)
        return AEG_EBACKEND;
    return AEG_OK;
}

static void cuda_sync(void *priv)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    if (cuda_bind(o) == AEG_OK)
        p_cuCtxSynchronize();
}

static void cuda_destroy(void *priv)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    int i;
    if (!o)
        return;
    if (o->ctx)
        cuda_bind(o);
    if (o->stream) {
        p_cuStreamSynchronize(o->stream);
        p_cuStreamDestroy_v2(o->stream);
        o->stream = NULL;
    }
    if (o->pin) {
        p_cuMemFreeHost(o->pin);
        o->pin = NULL;
        o->pin_cap = 0;
    }
    for (i = 0; i < NBUF; i++)
        if (o->buf[i])
            p_cuMemFree_v2(o->buf[i]);
    if (o->lo_ring)
        p_cuMemFree_v2(o->lo_ring);
    for (i = 0; i < 4; i++)
        if (o->lo_work[i])
            p_cuMemFree_v2(o->lo_work[i]);
    for (i = 0; i < 7; i++)
        if (o->mbt_work[i])
            p_cuMemFree_v2(o->mbt_work[i]);
    free(o->pos);
    if (o->is_clone) {
        free(o);
        return;
    }
    if (o->ctx) {
        if (o->mod)
            p_cuModuleUnload(o->mod);
        if (o->primary && p_cuDevicePrimaryCtxRelease)
            p_cuDevicePrimaryCtxRelease(o->ctx);
        else
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
    "satd_batch", "tq_iq_batch", "tq_dc_batch", "iq_dc_batch",
    "me_search", "lowres_batch", "mbt_propagate", NULL
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

/* Compile the shared kernel source to PTX and load it as a module. */
static int cuda_build_module(cuda_ctx *o)
{
    char *ptx = NULL, log[2048];
    int major = 5, minor = 0, cc, k, rc;
    CUresult cr;
    CUfunction *fn[13];

    p_cuDeviceGetAttribute(&major,
                           CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                           o->device);
    p_cuDeviceGetAttribute(&minor,
                           CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                           o->device);
    cc = major * 10 + minor;
    rc = cuda_compile_ptx(cc, &ptx, NULL, log, sizeof(log));
    if (rc == AEG_EBACKEND) {
        int fb = cuda_fallback_arch(cc);
        if (fb && fb != cc) {
            aeg_log(AEG_LOG_WARNING, "aeg: NVRTC cannot target compute_%d, using compute_%d",
                    cc, fb);
            rc = cuda_compile_ptx(fb, &ptx, NULL, log, sizeof(log));
        }
    }
    if (rc != AEG_OK) {
        aeg_log(AEG_LOG_WARNING, "aeg: NVRTC build failed (compute_%d): %s", cc, log);
        return rc;
    }
    cr = p_cuModuleLoadDataEx(&o->mod, ptx, 0, NULL, NULL);
    free(ptx);
    if (cr != CUDA_SUCCESS)
        return AEG_EBACKEND;
    fn[0] = &o->k_sad; fn[1] = &o->k_tq; fn[2] = &o->k_iq;
    fn[3] = &o->k_sse; fn[4] = &o->k_intp; fn[5] = &o->k_satd;
    fn[6] = &o->k_tqiq; fn[7] = &o->k_tqdc; fn[8] = &o->k_iqdc;
    fn[9] = &o->k_me; fn[10] = &o->k_lowres; fn[11] = &o->k_mbt;
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

/* Another instance for another thread (aeg_be_ops.clone): the context, the
 * module and its functions are shared (a context may be current on several
 * threads, and launches take their arguments per call); the buffers, the
 * registered reference and the sad_batch scratch are its own.  It uses the
 * legacy default stream like the first instance, so its work is ordered with
 * theirs on the device -- correct, and still free of the host-side lock. */
static int cuda_clone(void *priv, aeg_be_ops *out)
{
    const cuda_ctx *src = (const cuda_ctx *)priv;
    cuda_ctx *o = (cuda_ctx *)calloc(1, sizeof(*o));

    if (!o)
        return AEG_ENOMEM;
    o->is_clone = 1;
    o->device = src->device;
    o->ctx = src->ctx;
    p_cuCtxSetCurrent(o->ctx);
    if (p_cuStreamCreate(&o->stream, 0) != CUDA_SUCCESS) {
        free(o);
        return AEG_EBACKEND;
    }
    o->mod = src->mod;
    o->k_sad = src->k_sad;
    o->k_me = src->k_me;
    o->k_lowres = src->k_lowres;
    o->k_mbt = src->k_mbt;
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
    memset(out, 0, sizeof(*out));
    out->priv = o;
    out->backend = AEG_BACKEND_CUDA;
    out->is_gpu = 1;
    out->sad_batch = cuda_sad_batch;
    out->me_batch = cuda_me_batch;
    out->me_pic_upload = cuda_me_pic_upload;
    out->rec_upload = cuda_rec_upload;
    out->lowres_upload = cuda_lowres_upload;
    out->lowres_sad_batch = cuda_lowres_sad_batch;
    out->lowres_batch_search = cuda_lowres_batch_search;
    out->mbt_batch = cuda_mbt_batch;
    out->tq_batch = cuda_tq_batch;
    out->iq_idct_batch = cuda_iq_idct_batch;
    out->tq_iq_batch = cuda_tq_iq_batch;
    out->tq_dc_batch = cuda_tq_dc_batch;
    out->iq_dc_batch = cuda_iq_dc_batch;
    out->sse_batch = cuda_sse_batch;
    out->interp_plane = cuda_interp_plane;
    out->ref_upload = cuda_ref_upload;
    out->satd_batch = cuda_satd_batch;
    out->sync = cuda_sync;
    out->destroy = cuda_destroy;
    out->clone = cuda_clone;
    snprintf(out->desc, sizeof(out->desc), "CUDA %.48s (%d SMs)", o->name,
             o->cus);
    return AEG_OK;
}

int aeg_be_cuda_create(aeg_be_ops *ops)
{
    cuda_ctx *o;
    void *lib_cuda = NULL, *lib_nvrtc = NULL;
    int count = 0;

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
    /* Preferred: the primary context, so a host application's current
     * context is not replaced (and no push/pop is needed).  Older drivers
     * without the entry point keep the created context. */
    p_cuDevicePrimaryCtxRetain =
        (fn_primary_retain)dlsym(lib_cuda, "cuDevicePrimaryCtxRetain");
    p_cuDevicePrimaryCtxRelease =
        (fn_primary_release)dlsym(lib_cuda, "cuDevicePrimaryCtxRelease_v2");
    if (!p_cuDevicePrimaryCtxRelease)
        p_cuDevicePrimaryCtxRelease =
            (fn_primary_release)dlsym(lib_cuda, "cuDevicePrimaryCtxRelease");
    {
        int want = 0;
        const char *ev = getenv("AEG_DEVICE");
        if (ev && *ev)
            want = atoi(ev);
        if (want < 0 || want >= count) {
            if (want != 0)
                aeg_log(AEG_LOG_WARNING, "aeg: CUDA device %d not found (%d "
                        "devices), using device 0", want, count);
            want = 0;
        }
        o->device = want;
        if (p_cuDeviceGet(&o->device, want) != CUDA_SUCCESS) {
            cuda_destroy(o);
            return AEG_EBACKEND;
        }
        o->primary = 0;
        if (p_cuDevicePrimaryCtxRetain &&
            p_cuDevicePrimaryCtxRetain(&o->ctx, o->device) == CUDA_SUCCESS &&
            o->ctx) {
            o->primary = 1;
            p_cuCtxSetCurrent(o->ctx);
        } else if (p_cuCtxCreate_v2(&o->ctx, 0, o->device) != CUDA_SUCCESS) {
            o->ctx = NULL;
            cuda_destroy(o);
            return AEG_EBACKEND;
        }
    }
    if (p_cuStreamCreate(&o->stream, 0) != CUDA_SUCCESS) {
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

    memset(ops, 0, sizeof(*ops));
    ops->priv = o;
    ops->backend = AEG_BACKEND_CUDA;
    ops->is_gpu = 1;
    ops->sad_batch = cuda_sad_batch;
    ops->me_batch = cuda_me_batch;
    ops->me_pic_upload = cuda_me_pic_upload;
    ops->rec_upload = cuda_rec_upload;
    ops->lowres_upload = cuda_lowres_upload;
    ops->lowres_sad_batch = cuda_lowres_sad_batch;
    ops->lowres_batch_search = cuda_lowres_batch_search;
    ops->mbt_batch = cuda_mbt_batch;
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

int aeg_backend_query_cuda(aeg_device_info *out, int max)
{
    aeg_be_ops ops;
    int n = 0;
    if (max <= 0)
        return 0;
    if (aeg_be_cuda_create(&ops) == AEG_OK) {
        cuda_ctx *o = (cuda_ctx *)ops.priv;
        memset(&out[0], 0, sizeof(out[0]));
        out[0].backend = AEG_BACKEND_CUDA;
        snprintf(out[0].name, sizeof(out[0].name), "%s", o->name);
        snprintf(out[0].vendor, sizeof(out[0].vendor), "NVIDIA");
        out[0].is_gpu = 1;
        out[0].global_mem = o->mem;
        out[0].max_compute_units = o->cus;
        n = 1;
        if (ops.destroy)
            ops.destroy(ops.priv);
    }
    return n;
}
