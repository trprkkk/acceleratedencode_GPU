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
    X(CUresult, cuMemcpyHtoD_v2, (CUdeviceptr, const void *, size_t))         \
    X(CUresult, cuMemcpyDtoH_v2, (void *, CUdeviceptr, size_t))               \
    X(CUresult, cuLaunchKernel, (CUfunction, unsigned int, unsigned int,      \
                                 unsigned int, unsigned int, unsigned int,    \
                                 unsigned int, unsigned int, CUstream,        \
                                 void **, void **))                           \
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

#define NBUF 12
#define AEG_BLOCK 256

typedef struct cuda_ctx {
    void *lib_cuda;
    void *lib_nvrtc;
    CUdevice device;
    CUcontext ctx;
    CUmodule mod;
    CUfunction k_sad, k_tq, k_iq, k_sse, k_intp, k_satd;
    CUdeviceptr buf[NBUF];
    size_t buf_sz[NBUF];
    int cus;
    uint64_t mem;
    /* the registered reference picture (see ref_upload) */
    int ref_stride, ref_w, ref_h, ref_valid;
    char name[96];
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

/* ------------------------------------------------------------------ */
/* Operations                                                          */
/* ------------------------------------------------------------------ */
static int cuda_sad_batch(void *priv, const uint8_t *cur, int cstride,
                          const uint8_t *ref, int rstride, int bw, int bh,
                          const int16_t *mv, int count, int32_t *out)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    CUdeviceptr dcur, dref, dmv, dout;
    size_t cur_sz = (size_t)bw * bh;
    size_t ref_sz = (size_t)rstride * (bh + 32);
    static uint8_t *tmp;
    static size_t tmp_cap;
    int y, icur = bw, iref = rstride, ibw = bw, ibh = bh, icount = count;
    void *args[9];
    int st;

    dcur = cuda_buf(o, 0, cur_sz);
    dref = cuda_buf(o, 1, ref_sz);
    dmv = cuda_buf(o, 2, (size_t)count * 8);
    dout = cuda_buf(o, 3, (size_t)count * 4);
    if (!dcur || !dref || !dmv || !dout)
        return AEG_EBACKEND;

    /* the kernel reads the current block contiguously */
    if (tmp_cap < cur_sz) {
        free(tmp);
        tmp_cap = cur_sz;
        tmp = (uint8_t *)malloc(tmp_cap);
        if (!tmp)
            return AEG_ENOMEM;
    }
    for (y = 0; y < bh; y++)
        memcpy(tmp + (size_t)y * bw, cur + (size_t)y * cstride, (size_t)bw);

    p_cuMemcpyHtoD_v2(dcur, tmp, cur_sz);
    p_cuMemcpyHtoD_v2(dref, ref, ref_sz);
    p_cuMemcpyHtoD_v2(dmv, mv, (size_t)count * 8);

    args[0] = &dcur; args[1] = &icur; args[2] = &dref; args[3] = &iref;
    args[4] = &ibw;  args[5] = &ibh;  args[6] = &icount; args[7] = &dmv;
    args[8] = &dout;
    st = cuda_launch(o->k_sad, count, args);
    if (st != AEG_OK)
        return st;
    p_cuMemcpyDtoH_v2(out, dout, (size_t)count * 4);
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

    dres = cuda_buf(o, 0, bytes);
    dlev = cuda_buf(o, 1, bytes);
    dnz = cuda_buf(o, 2, (size_t)nblk * 4);
    if (!dres || !dlev || !dnz)
        return AEG_EBACKEND;
    p_cuMemcpyHtoD_v2(dres, res, bytes);
    args[0] = &dres; args[1] = &nblk; args[2] = &qbits; args[3] = &f;
    args[4] = &mf0;  args[5] = &mf1;  args[6] = &mf2;
    args[7] = &dlev; args[8] = &dnz;
    st = cuda_launch(o->k_tq, nblk, args);
    if (st != AEG_OK)
        return st;
    p_cuMemcpyDtoH_v2(level, dlev, bytes);
    if (nz)
        p_cuMemcpyDtoH_v2(nz, dnz, (size_t)nblk * 4);
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

    dlev = cuda_buf(o, 0, bytes);
    dres = cuda_buf(o, 1, bytes);
    if (!dlev || !dres)
        return AEG_EBACKEND;
    p_cuMemcpyHtoD_v2(dlev, level, bytes);
    args[0] = &dlev; args[1] = &nblk; args[2] = &sh;
    args[3] = &ls0;  args[4] = &ls1;  args[5] = &ls2; args[6] = &dres;
    st = cuda_launch(o->k_iq, nblk, args);
    if (st != AEG_OK)
        return st;
    p_cuMemcpyDtoH_v2(res, dres, bytes);
    return AEG_OK;
}

static int cuda_sse_batch(void *priv, const uint8_t *a, int sa, const uint8_t *b,
                          int sb, int w, int h, uint64_t *out)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    CUdeviceptr da, db, dout;
    size_t tw = (size_t)w / 8, th = (size_t)h / 8, count = tw * th;
    size_t abytes = (size_t)sa * h, bbytes = (size_t)sb * h;
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
    p_cuMemcpyHtoD_v2(da, a, abytes);
    p_cuMemcpyHtoD_v2(db, b, bbytes);
    args[0] = &da; args[1] = &sa; args[2] = &db; args[3] = &sb;
    args[4] = &itw; args[5] = &ith; args[6] = &icount; args[7] = &dout;
    st = cuda_launch(o->k_sse, (int)count, args);
    if (st != AEG_OK)
        return st;
    tmp = (uint64_t *)malloc(count * 8);
    if (!tmp)
        return AEG_ENOMEM;
    p_cuMemcpyDtoH_v2(tmp, dout, count * 8);
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

    dref = cuda_buf(o, 0, rbytes);
    ddst = cuda_buf(o, 1, dbytes);
    if (!dref || !ddst)
        return AEG_EBACKEND;
    p_cuMemcpyHtoD_v2(dref, ref, rbytes);
    args[0] = &dref; args[1] = &rstride; args[2] = &w; args[3] = &h;
    args[4] = &pad;  args[5] = &which;   args[6] = &ddst; args[7] = &dstride;
    st = cuda_launch(o->k_intp, (int)count, args);
    if (st != AEG_OK)
        return st;
    p_cuMemcpyDtoH_v2(dst, ddst, dbytes);
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

    for (k = 0; k < 4; k++) {
        CUdeviceptr d = cuda_buf(o, 4 + k, bytes);
        if (!d)
            return AEG_EBACKEND;
        p_cuMemcpyHtoD_v2(d, planes[k] - (ptrdiff_t)AEG_PAD * pstride - AEG_PAD,
                          bytes);
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

    if (!o->ref_valid || !count)
        return AEG_EBACKEND;
    dcur = cuda_buf(o, 0, cbytes);
    dpos = cuda_buf(o, 1, (size_t)count * 8);
    dout = cuda_buf(o, 2, (size_t)count * 4);
    if (!dcur || !dpos || !dout)
        return AEG_EBACKEND;
    for (k = 0; k < 4; k++)
        dref[k] = o->buf[4 + k];
    p_cuMemcpyHtoD_v2(dcur, cur - (ptrdiff_t)AEG_PAD * cstride - AEG_PAD, cbytes);
    p_cuMemcpyHtoD_v2(dpos, pos, (size_t)count * 8);
    args[0] = &dcur;  args[1] = &cstride;
    for (k = 0; k < 4; k++)
        args[2 + k] = &dref[k];
    args[6] = &o->ref_stride; args[7] = &pad;
    args[8] = &bw;   args[9] = &bh;
    args[10] = &dpos; args[11] = &count; args[12] = &dout;
    st = cuda_launch(o->k_satd, count, args);
    if (st != AEG_OK)
        return st;
    p_cuMemcpyDtoH_v2(out, dout, (size_t)count * 4);
    return AEG_OK;
}

static void cuda_sync(void *priv)
{
    (void)priv;
    p_cuCtxSynchronize();
}

static void cuda_destroy(void *priv)
{
    cuda_ctx *o = (cuda_ctx *)priv;
    int i;
    if (!o)
        return;
    for (i = 0; i < NBUF; i++)
        if (o->buf[i])
            p_cuMemFree_v2(o->buf[i]);
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
        "libnvrtc.so", "libnvrtc.so.12", "libnvrtc.so.11",
        "/usr/local/cuda/lib64/libnvrtc.so", NULL
    };
    int i;
    for (i = 0; names[i]; i++) {
        *lib = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
        if (*lib)
            return AEG_OK;
    }
    return AEG_EBACKEND;
}

/* Compile the shared kernel source to PTX and load it as a module. */
static int cuda_build_module(cuda_ctx *o)
{
    nvrtcProgram prog = NULL;
    char *src = NULL;
    size_t src_len = strlen(aeg_cuda_preamble) + strlen(aeg_kernel_src) + 1;
    char arch[64];
    const char *opts[1];
    char *ptx = NULL;
    size_t ptx_sz = 0;
    int major = 5, minor = 0;
    nvrtcResult nr;
    CUresult cr;

    p_cuDeviceGetAttribute(&major,
                           CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                           o->device);
    p_cuDeviceGetAttribute(&minor,
                           CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                           o->device);
    snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", major,
             minor);
    opts[0] = arch;

    src = (char *)malloc(src_len);
    if (!src)
        return AEG_ENOMEM;
    snprintf(src, src_len, "%s%s", aeg_cuda_preamble, aeg_kernel_src);

    nr = p_nvrtcCreateProgram(&prog, src, "aeg_kernels.cu", 0, NULL, NULL);
    if (nr == 0)
        nr = p_nvrtcCompileProgram(prog, 1, opts);
    if (nr != 0) {
        char log[2048];
        size_t log_sz = 0;
        log[0] = '\0';
        if (prog && p_nvrtcGetProgramLogSize(prog, &log_sz) == 0 &&
            log_sz > 1 && log_sz < sizeof(log))
            p_nvrtcGetProgramLog(prog, log);
        aeg_log(1, "aeg: NVRTC build failed (%s): %s", arch, log);
        if (prog)
            p_nvrtcDestroyProgram(&prog);
        free(src);
        return AEG_EBACKEND;
    }
    p_nvrtcGetPTXSize(prog, &ptx_sz);
    ptx = (char *)malloc(ptx_sz);
    if (!ptx) {
        p_nvrtcDestroyProgram(&prog);
        free(src);
        return AEG_ENOMEM;
    }
    p_nvrtcGetPTX(prog, ptx);
    p_nvrtcDestroyProgram(&prog);
    free(src);

    cr = p_cuModuleLoadDataEx(&o->mod, ptx, 0, NULL, NULL);
    free(ptx);
    if (cr != CUDA_SUCCESS)
        return AEG_EBACKEND;
    if (p_cuModuleGetFunction(&o->k_sad, o->mod, "sad_batch") != CUDA_SUCCESS ||
        p_cuModuleGetFunction(&o->k_tq, o->mod, "tq_batch") != CUDA_SUCCESS ||
        p_cuModuleGetFunction(&o->k_iq, o->mod, "iq_idct_batch") != CUDA_SUCCESS ||
        p_cuModuleGetFunction(&o->k_sse, o->mod, "sse_batch") != CUDA_SUCCESS ||
        p_cuModuleGetFunction(&o->k_intp, o->mod, "interp_plane") != CUDA_SUCCESS ||
        p_cuModuleGetFunction(&o->k_satd, o->mod, "satd_batch") != CUDA_SUCCESS)
        return AEG_EBACKEND;
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
            aeg_log(2, "aeg: CUDA driver is missing %s", #name); \
            dlclose(lib_cuda); \
            return AEG_EBACKEND; \
        } \
    } while (0);
#define LOAD_NVRTC(ret, name, args) do { \
        *(void **)(&p_##name) = dlsym(lib_nvrtc, #name); \
        if (!p_##name) { \
            aeg_log(2, "aeg: NVRTC is missing %s", #name); \
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
    if (p_cuDeviceGet(&o->device, 0) != CUDA_SUCCESS ||
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
    ops->sse_batch = cuda_sse_batch;
    ops->interp_plane = cuda_interp_plane;
    ops->ref_upload = cuda_ref_upload;
    ops->satd_batch = cuda_satd_batch;
    ops->sync = cuda_sync;
    ops->destroy = cuda_destroy;
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
