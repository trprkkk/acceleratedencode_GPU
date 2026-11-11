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
/* a transfer that fails sends the call back to the guard (backend.c),
 * which runs the CPU reference instead -- never a silently wrong result */
#define CU_CK(x) do { if ((x) != CUDA_SUCCESS) return AEG_EBACKEND; } while (0)

static const char aeg_cuda_preamble[] =
    "typedef unsigned char uchar;\n"
    "typedef unsigned short ushort;\n"
    "typedef unsigned long long ulong;\n"
    "#define __kernel extern \"C\" __global__\n"
    "#define __global\n"
    "#define AEG_DEVFN static __device__\n"
    "#define get_global_id(d) ((int)(blockIdx.x * blockDim.x + threadIdx.x))\n"
    "#define get_local_id(d) ((int)threadIdx.x)\n"
    "#define get_local_size(d) ((int)blockDim.x)\n"
    "#define get_group_id(d) ((int)blockIdx.x)\n"
    "#define CLK_GLOBAL_MEM_FENCE 0\n"
    "#define barrier(f) __syncthreads()\n";

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
    X(CUresult, cuMemcpyHtoD_v2, (CUdeviceptr, const void *, size_t))         \
    X(CUresult, cuMemcpyDtoH_v2, (void *, CUdeviceptr, size_t))               \
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

#define NBUF 20
#define AEG_BLOCK 256

typedef struct cuda_ctx {
    void *lib_cuda;
    void *lib_nvrtc;
    CUdevice device;
    CUcontext ctx;
    CUmodule mod;
    CUfunction k_sad, k_tq, k_iq, k_sse, k_intp, k_satd, k_tqiq, k_tqdc, k_iqdc;
    CUfunction k_lrs, k_lrc, k_lrb, k_lri, k_lrk, k_lrb16;
    CUdeviceptr buf[NBUF];
    size_t buf_sz[NBUF];
    int cus;
    uint64_t mem;
    /* the registered reference picture (see ref_upload) */
    int ref_stride, ref_w, ref_h, ref_valid;
    char name[96];
    int16_t *pos;               /* sad_batch: the rebased positions */
    size_t pos_cap;
    int is_clone;               /* shares ctx/mod/libraries with the first */
    /* lowres_chains: the ring's lowres planes, resident (slot k at k *
     * lr_slot bytes), each tagged with the picture it holds (-1 = none) */
    CUdeviceptr lr_mem;
    size_t lr_slot;
    int lr_nslots;
    int *lr_tag;
    int *lr_host;               /* the packed chain and job tables */
    size_t lr_host_cap;
    int *lk_blk;                /* lowres_costs: the per-block results */
    size_t lk_blk_cap;
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
    if (count <= 0)
        return AEG_OK;

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

static int cuda_tq_batch(void *priv, const int16_t *res, int nblk, int qp,
                         int intra, int16_t *level, int32_t *nz)
{
    if (nblk <= 0)
        return AEG_OK;

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

    dres = cuda_buf(o, 0, bytes);
    dlev = cuda_buf(o, 1, bytes);
    dnz = cuda_buf(o, 2, (size_t)nblk * 4);
    if (!dres || !dlev || !dnz)
        return AEG_EBACKEND;
    CU_CK(p_cuMemcpyHtoD_v2(dres, res, bytes));
    args[0] = &dres; args[1] = &nblk; args[2] = &qbits; args[3] = &f;
    args[4] = &mf0;  args[5] = &mf1;  args[6] = &mf2;
    args[7] = &dlev; args[8] = &dnz;
    st = cuda_launch(o->k_tq, nblk, args);
    if (st != AEG_OK)
        return st;
    CU_CK(p_cuMemcpyDtoH_v2(level, dlev, bytes));
    if (nz)
        CU_CK(p_cuMemcpyDtoH_v2(nz, dnz, (size_t)nblk * 4));
    return AEG_OK;
}

static int cuda_iq_idct_batch(void *priv, const int16_t *level, int nblk,
                              int qp, int16_t *res)
{
    if (nblk <= 0)
        return AEG_OK;

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

    dlev = cuda_buf(o, 0, bytes);
    dres = cuda_buf(o, 1, bytes);
    if (!dlev || !dres)
        return AEG_EBACKEND;
    CU_CK(p_cuMemcpyHtoD_v2(dlev, level, bytes));
    args[0] = &dlev; args[1] = &nblk; args[2] = &sh;
    args[3] = &ls0;  args[4] = &ls1;  args[5] = &ls2; args[6] = &dres;
    st = cuda_launch(o->k_iq, nblk, args);
    if (st != AEG_OK)
        return st;
    CU_CK(p_cuMemcpyDtoH_v2(res, dres, bytes));
    return AEG_OK;
}

static int cuda_tq_iq_batch(void *priv, const int16_t *res, int nblk, int qp,
                            int intra, int16_t *level, int32_t *nz,
                            int16_t *rec)
{
    if (nblk <= 0)
        return AEG_OK;

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
    if (nblk <= 0)
        return AEG_OK;

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
    if (nblk <= 0)
        return AEG_OK;

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
    if (w <= 0 || h <= 0) {
        *out = 0;
        return AEG_OK;
    }

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
    CU_CK(p_cuMemcpyHtoD_v2(da, a, abytes));
    CU_CK(p_cuMemcpyHtoD_v2(db, b, bbytes));
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
    if (w <= 0 || h <= 0)
        return AEG_OK;

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
        CU_CK(p_cuMemcpyHtoD_v2(d, planes[k] - (ptrdiff_t)AEG_PAD * pstride - AEG_PAD,
                          bytes));
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
    if (count <= 0)
        return AEG_OK;

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
    CU_CK(p_cuMemcpyHtoD_v2(dcur, cur - (ptrdiff_t)AEG_PAD * cstride - AEG_PAD, cbytes));
    CU_CK(p_cuMemcpyHtoD_v2(dpos, pos, (size_t)count * 8));
    args[0] = &dcur;  args[1] = &cstride;
    for (k = 0; k < 4; k++)
        args[2 + k] = &dref[k];
    args[6] = &o->ref_stride; args[7] = &pad;
    args[8] = &bw;   args[9] = &bh;
    args[10] = &dpos; args[11] = &count; args[12] = &dout;
    st = cuda_launch(o->k_satd, count, args);
    if (st != AEG_OK)
        return st;
    CU_CK(p_cuMemcpyDtoH_v2(out, dout, (size_t)count * 4));
    return AEG_OK;
}

/* The lowres search of one field (aeg_be_ops.lowres_search): one block of
 * 64 threads walks the field's diagonals (see the kernel), slots 8-11. */
static int cuda_lowres_search(void *priv, const uint8_t *cur,
                              const uint8_t *ref, int stride, int w, int h,
                              const int *seed, int num, int den, int range,
                              int *out)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    int pad = AEG_LA_PAD, mb_w = w / 8, mb_h = h / 8;
    int has_seed = seed != NULL, hexit = AEG_LA_RANGE;
    const size_t pbytes = (size_t)stride * (size_t)(h + 2 * pad);
    const size_t nb = (size_t)mb_w * (size_t)mb_h;
    const ptrdiff_t org = (ptrdiff_t)pad * stride + pad;
    CUdeviceptr dc, dr, ds, dout;
    void *args[15];

    if (mb_w <= 0 || mb_h <= 0)
        return AEG_OK;
    if (cuda_bind(o) != AEG_OK)
        return AEG_EBACKEND;
    dc = cuda_buf(o, 8, pbytes);
    dr = cuda_buf(o, 9, pbytes);
    dout = cuda_buf(o, 11, nb * 3 * sizeof(int));
    ds = has_seed ? cuda_buf(o, 10, nb * 2 * sizeof(int)) : dout;
    if (!dc || !dr || !dout || !ds)
        return AEG_EBACKEND;
    CU_CK(p_cuMemcpyHtoD_v2(dc, cur - org, pbytes));
    CU_CK(p_cuMemcpyHtoD_v2(dr, ref - org, pbytes));
    if (has_seed)
        CU_CK(p_cuMemcpyHtoD_v2(ds, seed, nb * 2 * sizeof(int)));
    args[0] = &dc;     args[1] = &dr;      args[2] = &stride;
    args[3] = &pad;    args[4] = &w;       args[5] = &h;
    args[6] = &mb_w;   args[7] = &mb_h;    args[8] = &ds;
    args[9] = &has_seed; args[10] = &num;  args[11] = &den;
    args[12] = &range; args[13] = &hexit;  args[14] = &dout;
    if (p_cuLaunchKernel(o->k_lrs, 1, 1, 1, 64, 1, 1, 0, NULL, args,
                         NULL) != CUDA_SUCCESS)
        return AEG_EBACKEND;
    CU_CK(p_cuMemcpyDtoH_v2(out, dout, nb * 3 * sizeof(int)));
    return AEG_OK;
}

/* The ring of resident lowres planes (see the OpenCL version). */
static int cuda_lr_ring(cuda_ctx *o, size_t slot, int nslots)
{
    int i;

    if (nslots <= 0 || slot * (size_t)nslots > 0x7fffffffu)
        return AEG_EBACKEND;
    if (o->lr_mem && o->lr_slot == slot && o->lr_nslots >= nslots)
        return AEG_OK;
    {
        int *tag = (int *)realloc(o->lr_tag, sizeof(int) * (size_t)nslots);
        if (!tag)
            return AEG_ENOMEM;
        o->lr_tag = tag;
    }
    if (o->lr_mem)
        p_cuMemFree_v2(o->lr_mem);
    o->lr_mem = 0;
    o->lr_nslots = 0;
    o->lr_slot = slot;
    if (p_cuMemAlloc_v2(&o->lr_mem, slot * (size_t)nslots) != CUDA_SUCCESS) {
        o->lr_mem = 0;
        return AEG_EBACKEND;
    }
    o->lr_nslots = nslots;
    for (i = 0; i < nslots; i++)
        o->lr_tag[i] = -1;
    return AEG_OK;
}

static int cuda_lr_pics(cuda_ctx *o, const aeg_la_pic *pics, int npics,
                        int nslots, size_t slot, ptrdiff_t org)
{
    int i, k;

    for (i = 0; i < npics; i++) {
        k = pics[i].slot;
        if (k < 0 || k >= nslots)
            return AEG_EINVAL;
        if (o->lr_tag[k] == pics[i].tag)
            continue;
        o->lr_tag[k] = -1;
        if (p_cuMemcpyHtoD_v2(o->lr_mem + slot * (size_t)k,
                              pics[i].plane - org, slot) != CUDA_SUCCESS)
            return AEG_EBACKEND;
        o->lr_tag[k] = pics[i].tag;
    }
    return AEG_OK;
}

static int cuda_lr_fail(cuda_ctx *o, int rc)
{
    int i;
    for (i = 0; i < o->lr_nslots; i++)
        o->lr_tag[i] = -1;
    return rc;
}

/* buffers 12-15: chain table, job table, seeds, fields (see the OpenCL
 * version, which this follows with synchronous copies) */
static int cuda_lowres_chains(void *priv, const aeg_la_pic *pics, int npics,
                              int nslots, const aeg_la_chain *chains,
                              int nchains, const aeg_la_job *jobs, int stride,
                              int w, int h)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    int pad = AEG_LA_PAD, mb_w = w / 8, mb_h = h / 8, hexit = AEG_LA_RANGE;
    const size_t slot = (size_t)stride * (size_t)(h + 2 * pad);
    const ptrdiff_t org = (ptrdiff_t)pad * stride + pad;
    const int fsz = 3 * mb_w * mb_h;
    const size_t sb = sizeof(int) * (size_t)(2 * fsz / 3);
    int njobs = 0, nseed = 0, c, i, ss;
    size_t need;
    CUdeviceptr dc, dj, ds, dout;
    void *args[13];

    if (mb_w <= 0 || mb_h <= 0 || nchains <= 0)
        return AEG_OK;
    if (cuda_bind(o) != AEG_OK)
        return AEG_EBACKEND;
    for (c = 0; c < nchains; c++) {
        if (chains[c].first + chains[c].count > njobs)
            njobs = chains[c].first + chains[c].count;
        nseed += chains[c].seed != NULL;
    }
    if ((size_t)fsz * (size_t)njobs > 0x1fffffffu)
        return AEG_EBACKEND;
    if ((i = cuda_lr_ring(o, slot, nslots)) != AEG_OK)
        return i;
    need = 4 * (size_t)nchains + 5 * (size_t)njobs;
    if (need > o->lr_host_cap) {
        int *t = (int *)realloc(o->lr_host, sizeof(int) * need);
        if (!t)
            return AEG_ENOMEM;
        o->lr_host = t;
        o->lr_host_cap = need;
    }
    dc = cuda_buf(o, 12, sizeof(int) * 4 * (size_t)nchains);
    dj = cuda_buf(o, 13, sizeof(int) * 5 * (size_t)njobs);
    dout = cuda_buf(o, 15, sizeof(int) * (size_t)fsz * (size_t)njobs);
    ds = nseed ? cuda_buf(o, 14, sb * (size_t)nseed) : dout;
    if (!dc || !dj || !ds || !dout)
        return AEG_EBACKEND;
    if (cuda_lr_pics(o, pics, npics, nslots, slot, org) != AEG_OK)
        goto fail;
    nseed = 0;
    for (c = 0; c < nchains; c++) {
        int *ci = o->lr_host + 4 * c;
        ci[0] = pics[chains[c].cur].slot;
        ci[1] = chains[c].first;
        ci[2] = chains[c].count;
        ci[3] = -1;
        if (chains[c].seed) {
            ci[3] = nseed * (2 * fsz / 3);
            if (p_cuMemcpyHtoD_v2(ds + sb * (size_t)nseed, chains[c].seed,
                                  sb) != CUDA_SUCCESS)
                goto fail;
            nseed++;
        }
    }
    for (i = 0; i < njobs; i++) {
        int *ji = o->lr_host + 4 * nchains + 5 * i;
        ji[0] = pics[jobs[i].ref].slot;
        ji[1] = jobs[i].seeded;
        ji[2] = jobs[i].num;
        ji[3] = jobs[i].den;
        ji[4] = jobs[i].range;
    }
    if (p_cuMemcpyHtoD_v2(dc, o->lr_host, sizeof(int) * 4 * (size_t)nchains) !=
            CUDA_SUCCESS ||
        p_cuMemcpyHtoD_v2(dj, o->lr_host + 4 * nchains,
                          sizeof(int) * 5 * (size_t)njobs) != CUDA_SUCCESS)
        goto fail;
    ss = (int)slot;
    args[0] = &o->lr_mem; args[1] = &ss;     args[2] = &stride;
    args[3] = &pad;       args[4] = &w;      args[5] = &h;
    args[6] = &mb_w;      args[7] = &mb_h;   args[8] = &dc;
    args[9] = &dj;        args[10] = &ds;    args[11] = &hexit;
    args[12] = &dout;
    if (p_cuLaunchKernel(o->k_lrc, (unsigned)nchains, 1, 1, 64, 1, 1, 0, NULL,
                         args, NULL) != CUDA_SUCCESS)
        goto fail;
    for (i = 0; i < njobs; i++)
        if (p_cuMemcpyDtoH_v2(jobs[i].out,
                              dout + sizeof(int) * (size_t)fsz * (size_t)i,
                              sizeof(int) * (size_t)fsz) != CUDA_SUCCESS)
            goto fail;
    return AEG_OK;
fail:
    for (i = 0; i < o->lr_nslots; i++)
        o->lr_tag[i] = -1;
    return AEG_EBACKEND;
}

/* The rest of the lookahead on the ring (see the OpenCL versions);
 * buffer 16: the source luma */
static int cuda_lowres_build(void *priv, const void *src, int sstride,
                             int depth, int sw, int sh, const aeg_la_pic *pic,
                             int nslots, int stride, int w, int h)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    int pad = AEG_LA_PAD, rc, dst, a = 0;
    const int hi = depth > 8;
    const size_t slot = (size_t)stride * (size_t)(h + 2 * pad);
    const ptrdiff_t org = (ptrdiff_t)pad * stride + pad;
    const size_t sbytes = ((size_t)sstride * (size_t)(sh - 1) + (size_t)sw)
                          << hi;
    CUdeviceptr ds;
    void *args[11];

    if (w <= 0 || h <= 0 || sw <= 0 || sh <= 0 || depth < 8 || depth > 14)
        return AEG_EINVAL;
    if (cuda_bind(o) != AEG_OK)
        return AEG_EBACKEND;
    if ((rc = cuda_lr_ring(o, slot, nslots)) != AEG_OK)
        return rc;
    if (pic->slot < 0 || pic->slot >= nslots)
        return AEG_EINVAL;
    if (!(ds = cuda_buf(o, 16, sbytes)))
        return AEG_EBACKEND;
    dst = (int)(slot * (size_t)pic->slot);
    o->lr_tag[pic->slot] = -1;
    args[a++] = &ds;
    args[a++] = &sstride;
    args[a++] = &sw;
    args[a++] = &sh;
    if (hi)
        args[a++] = &depth;
    args[a++] = &o->lr_mem;
    args[a++] = &dst;
    args[a++] = &stride;
    args[a++] = &pad;
    args[a++] = &w;
    args[a++] = &h;
    if (p_cuMemcpyHtoD_v2(ds, src, sbytes) != CUDA_SUCCESS ||
        cuda_launch(hi ? o->k_lrb16 : o->k_lrb, (int)slot, args) != AEG_OK ||
        p_cuMemcpyDtoH_v2((uint8_t *)pic->plane - org,
                          o->lr_mem + slot * (size_t)pic->slot,
                          slot) != CUDA_SUCCESS)
        return cuda_lr_fail(o, AEG_EBACKEND);
    o->lr_tag[pic->slot] = pic->tag;
    return AEG_OK;
}

/* buffers 17, 18: the slot table, the estimates */
static int cuda_lowres_intra(void *priv, const aeg_la_pic *pics, int npics,
                             int nslots, int *const *out, int i8, int stride,
                             int w, int h)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    int pad = AEG_LA_PAD, mb_w = w / 8, mb_h = h / 8, rc, i, ss;
    const size_t slot = (size_t)stride * (size_t)(h + 2 * pad);
    const ptrdiff_t org = (ptrdiff_t)pad * stride + pad;
    const size_t nb = (size_t)mb_w * (size_t)mb_h;
    const size_t global = nb * (size_t)(npics > 0 ? npics : 0);
    CUdeviceptr dt, dout;
    void *args[10];

    if (npics <= 0 || nb == 0)
        return AEG_OK;
    if (global > 0x7fffffffu)
        return AEG_EBACKEND;
    if (cuda_bind(o) != AEG_OK)
        return AEG_EBACKEND;
    if ((rc = cuda_lr_ring(o, slot, nslots)) != AEG_OK)
        return rc;
    if ((size_t)npics > o->lr_host_cap) {
        int *t = (int *)realloc(o->lr_host, sizeof(int) * (size_t)npics);
        if (!t)
            return AEG_ENOMEM;
        o->lr_host = t;
        o->lr_host_cap = (size_t)npics;
    }
    dt = cuda_buf(o, 17, sizeof(int) * (size_t)npics);
    dout = cuda_buf(o, 18, sizeof(int) * global);
    if (!dt || !dout)
        return AEG_EBACKEND;
    if ((rc = cuda_lr_pics(o, pics, npics, nslots, slot, org)) != AEG_OK)
        return cuda_lr_fail(o, rc);
    for (i = 0; i < npics; i++)
        o->lr_host[i] = pics[i].slot;
    ss = (int)slot;
    args[0] = &o->lr_mem; args[1] = &ss;    args[2] = &stride;
    args[3] = &pad;       args[4] = &mb_w;  args[5] = &mb_h;
    args[6] = &dt;        args[7] = &npics; args[8] = &i8;
    args[9] = &dout;
    if (p_cuMemcpyHtoD_v2(dt, o->lr_host, sizeof(int) * (size_t)npics) !=
            CUDA_SUCCESS ||
        cuda_launch(o->k_lri, (int)global, args) != AEG_OK)
        return cuda_lr_fail(o, AEG_EBACKEND);
    for (i = 0; i < npics; i++)
        if (p_cuMemcpyDtoH_v2(out[i], dout + sizeof(int) * nb * (size_t)i,
                              sizeof(int) * nb) != CUDA_SUCCESS)
            return cuda_lr_fail(o, AEG_EBACKEND);
    return AEG_OK;
}

/* buffers 17-19: the job table, the per-block results, the fields and
 * intra estimates (each distinct array once) */
static int cuda_lowres_costs(void *priv, const aeg_la_pic *pics, int npics,
                             int nslots, const aeg_la_cost *costs, int ncosts,
                             int stride, int w, int h)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    int pad = AEG_LA_PAD, mb_w = w / 8, mb_h = h / 8, rc, i, k, ss;
    int narr = 0, *off;
    const int **arr;
    const size_t slot = (size_t)stride * (size_t)(h + 2 * pad);
    const ptrdiff_t org = (ptrdiff_t)pad * stride + pad;
    const size_t nb = (size_t)mb_w * (size_t)mb_h;
    const size_t global = nb * (size_t)(ncosts > 0 ? ncosts : 0);
    size_t need, ndata = 0;
    CUdeviceptr dj, dout, dd;
    void *args[10];

    if (ncosts <= 0 || nb == 0)
        return AEG_OK;
    if (global > 0x1fffffffu)
        return AEG_EBACKEND;
    if (cuda_bind(o) != AEG_OK)
        return AEG_EBACKEND;
    if ((rc = cuda_lr_ring(o, slot, nslots)) != AEG_OK)
        return rc;
    need = 6 * (size_t)ncosts +
           3 * (size_t)ncosts * (1 + sizeof(void *) / sizeof(int) + 1);
    if (need > o->lr_host_cap) {
        int *t = (int *)realloc(o->lr_host, sizeof(int) * need);
        if (!t)
            return AEG_ENOMEM;
        o->lr_host = t;
        o->lr_host_cap = need;
    }
    if (global > o->lk_blk_cap) {
        int *t = (int *)realloc(o->lk_blk, sizeof(int) * global);
        if (!t)
            return AEG_ENOMEM;
        o->lk_blk = t;
        o->lk_blk_cap = global;
    }
    off = o->lr_host + 6 * (size_t)ncosts;
    arr = (const int **)(void *)(off + 3 * (size_t)ncosts +
                                 (3 * (size_t)ncosts & 1));
    for (i = 0; i < ncosts; i++) {
        const int *a[3];
        const size_t len[3] = { 3 * nb, 3 * nb, nb };
        int *ji = o->lr_host + 6 * i, j;
        if (costs[i].cur < 0 || costs[i].cur >= npics || costs[i].r0 < 0 ||
            costs[i].r0 >= npics || costs[i].r1 >= npics)
            return AEG_EINVAL;
        a[0] = costs[i].f0;
        a[1] = costs[i].r1 >= 0 ? costs[i].f1 : costs[i].f0;
        a[2] = costs[i].icost;
        ji[0] = pics[costs[i].cur].slot;
        ji[1] = pics[costs[i].r0].slot;
        ji[2] = costs[i].r1 >= 0 ? pics[costs[i].r1].slot : -1;
        for (j = 0; j < 3; j++) {
            for (k = 0; k < narr && arr[k] != a[j]; k++)
                ;
            if (k == narr) {
                arr[narr] = a[j];
                off[narr++] = (int)ndata;
                ndata += len[j];
                if (ndata > 0x1fffffffu)
                    return AEG_EBACKEND;
            }
            ji[3 + j] = off[k];
        }
    }
    dj = cuda_buf(o, 17, sizeof(int) * 6 * (size_t)ncosts);
    dout = cuda_buf(o, 18, sizeof(int) * global);
    dd = cuda_buf(o, 19, sizeof(int) * ndata);
    if (!dj || !dout || !dd)
        return AEG_EBACKEND;
    if ((rc = cuda_lr_pics(o, pics, npics, nslots, slot, org)) != AEG_OK)
        return cuda_lr_fail(o, rc);
    for (k = 0; k < narr; k++) {
        const size_t len = (k + 1 < narr ? (size_t)off[k + 1] : ndata) -
                           (size_t)off[k];
        if (p_cuMemcpyHtoD_v2(dd + sizeof(int) * (size_t)off[k], arr[k],
                              sizeof(int) * len) != CUDA_SUCCESS)
            return cuda_lr_fail(o, AEG_EBACKEND);
    }
    ss = (int)slot;
    args[0] = &o->lr_mem; args[1] = &ss;     args[2] = &stride;
    args[3] = &pad;       args[4] = &mb_w;   args[5] = &mb_h;
    args[6] = &dj;        args[7] = &ncosts; args[8] = &dd;
    args[9] = &dout;
    if (p_cuMemcpyHtoD_v2(dj, o->lr_host, sizeof(int) * 6 * (size_t)ncosts) !=
            CUDA_SUCCESS ||
        cuda_launch(o->k_lrk, (int)global, args) != AEG_OK ||
        p_cuMemcpyDtoH_v2(o->lk_blk, dout, sizeof(int) * global) !=
            CUDA_SUCCESS)
        return cuda_lr_fail(o, AEG_EBACKEND);
    for (i = 0; i < ncosts; i++)
        aeg_la_cost_sum(o->lk_blk + nb * (size_t)i, mb_w, mb_h,
                        costs[i].total, costs[i].intra);
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
    for (i = 0; i < NBUF; i++)
        if (o->buf[i])
            p_cuMemFree_v2(o->buf[i]);
    free(o->pos);
    if (o->lr_mem)
        p_cuMemFree_v2(o->lr_mem);
    free(o->lr_tag);
    free(o->lr_host);
    free(o->lk_blk);
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
    "satd_batch", "tq_iq_batch", "tq_dc_batch", "iq_dc_batch",
    "lowres_search", "lowres_chains", "lowres_build", "lowres_intra",
    "lowres_costs", "lowres_build16", NULL
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
    CUfunction *fn[15];

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
    fn[9] = &o->k_lrs; fn[10] = &o->k_lrc; fn[11] = &o->k_lrb;
    fn[12] = &o->k_lri; fn[13] = &o->k_lrk; fn[14] = &o->k_lrb16;
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
    o->k_lrs = src->k_lrs;
    o->k_lrc = src->k_lrc;
    o->k_lrb = src->k_lrb;
    o->k_lri = src->k_lri;
    o->k_lrk = src->k_lrk;
    o->k_lrb16 = src->k_lrb16;
    o->cus = src->cus;
    o->mem = src->mem;
    memcpy(o->name, src->name, sizeof(o->name));
    memset(out, 0, sizeof(*out));
    out->priv = o;
    out->backend = AEG_BACKEND_CUDA;
    out->sad_batch = cuda_sad_batch;
    out->tq_batch = cuda_tq_batch;
    out->iq_idct_batch = cuda_iq_idct_batch;
    out->tq_iq_batch = cuda_tq_iq_batch;
    out->tq_dc_batch = cuda_tq_dc_batch;
    out->iq_dc_batch = cuda_iq_dc_batch;
    out->sse_batch = cuda_sse_batch;
    out->interp_plane = cuda_interp_plane;
    out->ref_upload = cuda_ref_upload;
    out->satd_batch = cuda_satd_batch;
    out->lowres_search = cuda_lowres_search;
    out->lowres_chains = cuda_lowres_chains;
    out->lowres_build = cuda_lowres_build;
    out->lowres_intra = cuda_lowres_intra;
    out->lowres_costs = cuda_lowres_costs;
    out->sync = cuda_sync;
    out->destroy = cuda_destroy;
    out->clone = cuda_clone;
    snprintf(out->desc, sizeof(out->desc), "CUDA %.48s (%d SMs)", o->name,
             o->cus);
    return AEG_OK;
}

/* `want` >= 0: that CUDA device (cuDeviceGet's ordinal, the --device
 * number); -1: device 0. */
int aeg_be_cuda_create(aeg_be_ops *ops, int want)
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
        || count < 1 || want >= count) {
        cuda_destroy(o);
        return AEG_EBACKEND;
    }
    if (p_cuDeviceGet(&o->device, want > 0 ? want : 0) != CUDA_SUCCESS ||
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
    ops->lowres_search = cuda_lowres_search;
    ops->lowres_chains = cuda_lowres_chains;
    ops->lowres_build = cuda_lowres_build;
    ops->lowres_intra = cuda_lowres_intra;
    ops->lowres_costs = cuda_lowres_costs;
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
    int n = 0, k;
    /* every device by ordinal (the list ends at the first that fails) */
    for (k = 0; k < 16 && n < max; k++) {
        cuda_ctx *o;
        if (aeg_be_cuda_create(&ops, k) != AEG_OK)
            break;
        o = (cuda_ctx *)ops.priv;
        memset(&out[n], 0, sizeof(out[n]));
        out[n].backend = AEG_BACKEND_CUDA;
        snprintf(out[n].name, sizeof(out[n].name), "%s", o->name);
        snprintf(out[n].vendor, sizeof(out[n].vendor), "NVIDIA");
        out[n].is_gpu = 1;
        out[n].global_mem = o->mem;
        out[n].max_compute_units = o->cus;
        out[n].device_index = k;
        n++;
        if (ops.destroy)
            ops.destroy(ops.priv);
    }
    return n;
}
