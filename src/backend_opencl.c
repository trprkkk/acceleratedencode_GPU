/*
 * OpenCL backend.
 *
 * The OpenCL runtime is loaded with dlopen(), so the encoder still builds and
 * runs on machines without any OpenCL installation -- the backend simply
 * reports itself unavailable and the dispatcher keeps the CPU path.  Nothing
 * in this file is required at compile time, which also means the same binary
 * carries the CPU, OpenCL and (if built with CUDA=1) CUDA paths.
 *
 * Only general-purpose compute is used: plain OpenCL C kernels doing integer
 * arithmetic.  No vendor video extension, no fixed-function encoder block.
 */
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "kernels.h"

typedef void *cl_platform_id;
typedef void *cl_device_id;
typedef void *cl_context;
typedef void *cl_command_queue;
typedef void *cl_program;
typedef void *cl_kernel;
typedef void *cl_mem;
typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef intptr_t cl_context_properties;

#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_ALL 0xFFFFFFFFull   /* cl_device_type is 64 bits */
#define CL_MEM_READ_ONLY (1 << 2)
#define CL_MEM_WRITE_ONLY (1 << 1)
#define CL_MEM_READ_WRITE (1 << 0)
#define CL_TRUE 1
#define CL_FALSE 0
#define CL_PROGRAM_BUILD_LOG 0x1183
#define CL_DEVICE_TYPE 0x1000
#define CL_DEVICE_TYPE_GPU (1 << 2)
#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_VENDOR 0x102C
#define CL_DRIVER_VERSION 0x102D
#define CL_DEVICE_VERSION 0x102F
#define CL_DEVICE_GLOBAL_MEM_SIZE 0x101F
#define CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE 0x1010
#define CL_KERNEL_WORK_GROUP_SIZE 0x11B0

/* The OpenCL entry points are resolved at run time so that the encoder keeps
 * building (and running on the CPU) on machines without an OpenCL loader. */
/* a transfer that fails sends the call back to the guard (backend.c),
 * which runs the CPU reference instead -- never a silently wrong result */
#define OCL_CK(x) do { if ((x) != CL_SUCCESS) return AEG_EBACKEND; } while (0)

static void ocl_load_error(const char *name)
{
    aeg_log(AEG_LOG_DEBUG, "aeg: OpenCL loader is missing %s", name);
}

#define OCL_SYMS(X) \
    X(cl_int, clGetPlatformIDs, (cl_uint, cl_platform_id *, cl_uint *)) \
    X(cl_int, clGetDeviceIDs, (cl_platform_id, cl_ulong, cl_uint, cl_device_id *, cl_uint *)) \
    X(cl_int, clGetDeviceInfo, (cl_device_id, cl_uint, size_t, void *, size_t *)) \
    X(cl_context, clCreateContext, (cl_context_properties *, cl_uint, cl_device_id *, void *, void *, cl_int *)) \
    X(cl_command_queue, clCreateCommandQueue, (cl_context, cl_device_id, cl_ulong, cl_int *)) \
    X(cl_program, clCreateProgramWithSource, (cl_context, cl_uint, const char **, const size_t *, cl_int *)) \
    X(cl_int, clBuildProgram, (cl_program, cl_uint, const cl_device_id *, const char *, void *, void *)) \
    X(cl_int, clGetProgramBuildInfo, (cl_program, cl_device_id, cl_uint, size_t, void *, size_t *)) \
    X(cl_kernel, clCreateKernel, (cl_program, const char *, cl_int *)) \
    X(cl_int, clGetKernelWorkGroupInfo, (cl_kernel, cl_device_id, cl_uint, size_t, void *, size_t *)) \
    X(cl_int, clSetKernelArg, (cl_kernel, cl_uint, size_t, const void *)) \
    X(cl_int, clEnqueueNDRangeKernel, (cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const void *, void *)) \
    X(cl_int, clEnqueueWriteBuffer, (cl_command_queue, cl_mem, cl_uint, size_t, size_t, const void *, cl_uint, const void *, void *)) \
    X(cl_int, clEnqueueReadBuffer, (cl_command_queue, cl_mem, cl_uint, size_t, size_t, void *, cl_uint, const void *, void *)) \
    X(cl_mem, clCreateBuffer, (cl_context, cl_ulong, size_t, void *, cl_int *)) \
    X(cl_int, clReleaseMemObject, (cl_mem)) \
    X(cl_int, clReleaseKernel, (cl_kernel)) \
    X(cl_int, clReleaseProgram, (cl_program)) \
    X(cl_int, clReleaseCommandQueue, (cl_command_queue)) \
    X(cl_int, clReleaseContext, (cl_context)) \
    X(cl_int, clRetainContext, (cl_context)) \
    X(cl_int, clRetainProgram, (cl_program)) \
    X(cl_int, clFinish, (cl_command_queue))

#define DECL(ret, name, args) typedef ret (*fn_##name) args; static fn_##name p_##name;
OCL_SYMS(DECL)
#undef DECL

#define NBUF 20

typedef struct ocl_ctx {
    cl_platform_id platform;
    cl_device_id device;
    cl_context ctx;
    cl_command_queue queue;
    cl_program prog;
    cl_kernel k_sad, k_tq, k_iq, k_tqiq, k_sse, k_intp, k_satd, k_tqdc, k_iqdc;
    cl_kernel k_lrs;
    size_t lrs_wg;              /* lowres_search's work-group size */
    cl_mem buf[NBUF];
    size_t buf_sz[NBUF];
    int is_gpu;
    int index;                  /* in ocl_devices(), the --device number */
    /* the registered reference picture (see ref_upload) */
    int ref_stride, ref_w, ref_h, ref_valid;
    char name[96];
    char vendor[48];
    uint64_t mem;
    int cus;
    int16_t *pos;               /* sad_batch: the rebased positions */
    size_t pos_cap;
    int no_clone;               /* the runtime cannot take several queues */
    /* lowres_chains: the ring's lowres planes, resident (slot k at k *
     * lr_slot bytes), each tagged with the picture it holds (-1 = none) */
    cl_kernel k_lrc;
    cl_mem lr_mem;
    size_t lr_slot;
    int lr_nslots;
    int *lr_tag;
    int *lr_host;               /* the packed chain and job tables */
    size_t lr_host_cap;
    /* the rest of the lookahead on the same ring (lowres_build / _intra /
     * _costs): their kernels and the host side of the costs' blocks */
    cl_kernel k_lrb, k_lri, k_lrk;
    int *lk_blk;
    size_t lk_blk_cap;
} ocl_ctx;

/* PoCL 4.x-6.x abort when two command queues launch the same kernel at the
 * same time: pocl_release_dlhandle_cache() asserts on its reference count
 * (the cache of compiled work-group functions was not safe across threads;
 * fixed by PoCL c9ac4c0c3 / 69d0ae9ac, both first released in 7.0).  Seen
 * on Ubuntu 24.04's PoCL 5.0 with two slice threads.  PoCL 3.0 does not
 * have it (the lane tests run on it).  On those versions clone() declines,
 * so every device lane is lane 0 -- serialised, as before the lanes, and the
 * same stream.  AEG_POCL_LANES=1 clones on any PoCL (to test a fixed
 * build), AEG_POCL_LANES=0 declines on any PoCL (to test the fallback). */
static int ocl_pocl_unsafe_queues(cl_device_id dev)
{
    char ver[128] = "", drv[64] = "";
    const char *ev = getenv("AEG_POCL_LANES");
    int major = 0;
    if (ev && *ev && atoi(ev) != 0)
        return 0;
    p_clGetDeviceInfo(dev, CL_DEVICE_VERSION, sizeof(ver) - 1, ver, NULL);
    if (!strstr(ver, "PoCL"))
        return 0;
    if (ev && *ev)
        return 1;
    p_clGetDeviceInfo(dev, CL_DRIVER_VERSION, sizeof(drv) - 1, drv, NULL);
    major = atoi(drv);
    return major >= 4 && major < 7;
}

/* ------------------------------------------------------------------ */
/* Buffer cache                                                        */
/* ------------------------------------------------------------------ */
static cl_mem ocl_buf(ocl_ctx *o, int slot, size_t bytes, int flags)
{
    if (o->buf[slot] && o->buf_sz[slot] >= bytes)
        return o->buf[slot];
    if (o->buf[slot]) {
        p_clReleaseMemObject(o->buf[slot]);
        o->buf[slot] = NULL;
    }
    o->buf_sz[slot] = bytes;
    o->buf[slot] = p_clCreateBuffer(o->ctx, flags, bytes, NULL, NULL);
    return o->buf[slot];
}

/* ------------------------------------------------------------------ */
/* Operations                                                          */
/* ------------------------------------------------------------------ */
static int ocl_sad_batch(void *priv, const uint8_t *cur, int cstride,
                         const uint8_t *ref, int rstride, int bw, int bh,
                         const int16_t *mv, int count, int32_t *out)
{
    if (count <= 0)
        return AEG_OK;

    ocl_ctx *o = (ocl_ctx *)priv;
    cl_mem mcur, mref, mmv, mout;
    size_t nthreads = (size_t)count;
    aeg_sad_span sp;
    cl_int st = CL_SUCCESS;

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
    mcur = ocl_buf(o, 0, sp.cbytes, CL_MEM_READ_ONLY);
    mref = ocl_buf(o, 1, sp.rbytes, CL_MEM_READ_ONLY);
    mmv = ocl_buf(o, 2, (size_t)count * 8, CL_MEM_READ_ONLY);
    mout = ocl_buf(o, 3, (size_t)count * 4, CL_MEM_WRITE_ONLY);
    if (!mcur || !mref || !mmv || !mout)
        return AEG_EBACKEND;
    st |= p_clEnqueueWriteBuffer(o->queue, mcur, CL_TRUE, 0, sp.cbytes,
                                 cur + sp.coff, 0, NULL, NULL);
    st |= p_clEnqueueWriteBuffer(o->queue, mref, CL_TRUE, 0, sp.rbytes,
                                 ref + sp.roff, 0, NULL, NULL);
    st |= p_clEnqueueWriteBuffer(o->queue, mmv, CL_TRUE, 0, (size_t)count * 8,
                                 o->pos, 0, NULL, NULL);
    if (st != CL_SUCCESS)
        return AEG_EBACKEND;

    p_clSetKernelArg(o->k_sad, 0, sizeof(cl_mem), &mcur);
    p_clSetKernelArg(o->k_sad, 1, sizeof(int), &cstride);
    p_clSetKernelArg(o->k_sad, 2, sizeof(cl_mem), &mref);
    p_clSetKernelArg(o->k_sad, 3, sizeof(int), &rstride);
    p_clSetKernelArg(o->k_sad, 4, sizeof(int), &bw);
    p_clSetKernelArg(o->k_sad, 5, sizeof(int), &bh);
    p_clSetKernelArg(o->k_sad, 6, sizeof(int), &count);
    p_clSetKernelArg(o->k_sad, 7, sizeof(cl_mem), &mmv);
    p_clSetKernelArg(o->k_sad, 8, sizeof(cl_mem), &mout);
    st = p_clEnqueueNDRangeKernel(o->queue, o->k_sad, 1, NULL, &nthreads, NULL,
                                  0, NULL, NULL);
    if (st != CL_SUCCESS)
        return AEG_EBACKEND;
    if (p_clEnqueueReadBuffer(o->queue, mout, CL_TRUE, 0, (size_t)count * 4,
                              out, 0, NULL, NULL) != CL_SUCCESS)
        return AEG_EBACKEND;
    return AEG_OK;
}

static int ocl_tq_batch(void *priv, const int16_t *res, int nblk, int qp,
                        int intra, int16_t *level, int32_t *nz)
{
    if (nblk <= 0)
        return AEG_OK;

    ocl_ctx *o = (ocl_ctx *)priv;
    cl_mem mres, mlev, mnz;
    size_t bytes = (size_t)nblk * 16 * 2;
    size_t nthreads = (size_t)nblk;
    int qbits = 15 + qp / 6;
    int f = aeg_quant_bias(qbits, intra ? 3 : 6);
    int mf0 = aeg_quant_mf[qp % 6][0], mf1 = aeg_quant_mf[qp % 6][1],
        mf2 = aeg_quant_mf[qp % 6][2];
    cl_int st;

    mres = ocl_buf(o, 0, bytes, CL_MEM_READ_ONLY);
    mlev = ocl_buf(o, 1, bytes, CL_MEM_WRITE_ONLY);
    mnz = ocl_buf(o, 2, (size_t)nblk * 4, CL_MEM_WRITE_ONLY);
    if (!mres || !mlev || !mnz)
        return AEG_EBACKEND;
    OCL_CK(p_clEnqueueWriteBuffer(o->queue, mres, CL_TRUE, 0, bytes, res, 0, NULL, NULL));
    p_clSetKernelArg(o->k_tq, 0, sizeof(cl_mem), &mres);
    p_clSetKernelArg(o->k_tq, 1, sizeof(int), &nblk);
    p_clSetKernelArg(o->k_tq, 2, sizeof(int), &qbits);
    p_clSetKernelArg(o->k_tq, 3, sizeof(int), &f);
    p_clSetKernelArg(o->k_tq, 4, sizeof(int), &mf0);
    p_clSetKernelArg(o->k_tq, 5, sizeof(int), &mf1);
    p_clSetKernelArg(o->k_tq, 6, sizeof(int), &mf2);
    p_clSetKernelArg(o->k_tq, 7, sizeof(cl_mem), &mlev);
    p_clSetKernelArg(o->k_tq, 8, sizeof(cl_mem), &mnz);
    st = p_clEnqueueNDRangeKernel(o->queue, o->k_tq, 1, NULL, &nthreads, NULL,
                                  0, NULL, NULL);
    if (st != CL_SUCCESS)
        return AEG_EBACKEND;
    OCL_CK(p_clEnqueueReadBuffer(o->queue, mlev, CL_TRUE, 0, bytes, level, 0, NULL, NULL));
    if (nz)
        OCL_CK(p_clEnqueueReadBuffer(o->queue, mnz, CL_TRUE, 0, (size_t)nblk * 4, nz,
                              0, NULL, NULL));
    return AEG_OK;
}

static int ocl_iq_idct_batch(void *priv, const int16_t *level, int nblk, int qp,
                             int16_t *res)
{
    if (nblk <= 0)
        return AEG_OK;

    ocl_ctx *o = (ocl_ctx *)priv;
    cl_mem mlev, mres;
    size_t bytes = (size_t)nblk * 16 * 2;
    size_t nthreads = (size_t)nblk;
    int sh = qp / 6;
    int ls0 = aeg_dequant_ls[qp % 6][0], ls1 = aeg_dequant_ls[qp % 6][1],
        ls2 = aeg_dequant_ls[qp % 6][2];
    cl_int st;

    mlev = ocl_buf(o, 0, bytes, CL_MEM_READ_ONLY);
    mres = ocl_buf(o, 1, bytes, CL_MEM_WRITE_ONLY);
    if (!mlev || !mres)
        return AEG_EBACKEND;
    OCL_CK(p_clEnqueueWriteBuffer(o->queue, mlev, CL_TRUE, 0, bytes, level, 0, NULL, NULL));
    p_clSetKernelArg(o->k_iq, 0, sizeof(cl_mem), &mlev);
    p_clSetKernelArg(o->k_iq, 1, sizeof(int), &nblk);
    p_clSetKernelArg(o->k_iq, 2, sizeof(int), &sh);
    p_clSetKernelArg(o->k_iq, 3, sizeof(int), &ls0);
    p_clSetKernelArg(o->k_iq, 4, sizeof(int), &ls1);
    p_clSetKernelArg(o->k_iq, 5, sizeof(int), &ls2);
    p_clSetKernelArg(o->k_iq, 6, sizeof(cl_mem), &mres);
    st = p_clEnqueueNDRangeKernel(o->queue, o->k_iq, 1, NULL, &nthreads, NULL,
                                  0, NULL, NULL);
    if (st != CL_SUCCESS)
        return AEG_EBACKEND;
    OCL_CK(p_clEnqueueReadBuffer(o->queue, mres, CL_TRUE, 0, bytes, res, 0, NULL, NULL));
    return AEG_OK;
}

static int ocl_tq_iq_batch(void *priv, const int16_t *res, int nblk, int qp,
                           int intra, int16_t *level, int32_t *nz,
                           int16_t *rec)
{
    if (nblk <= 0)
        return AEG_OK;

    ocl_ctx *o = (ocl_ctx *)priv;
    cl_mem mres, mlev, mnz, mrec;
    size_t bytes = (size_t)nblk * 16 * 2;
    size_t nthreads = (size_t)nblk;
    int qbits = 15 + qp / 6;
    int f = aeg_quant_bias(qbits, intra ? 3 : 6);
    int mf0 = aeg_quant_mf[qp % 6][0], mf1 = aeg_quant_mf[qp % 6][1],
        mf2 = aeg_quant_mf[qp % 6][2];
    int sh = qp / 6;
    int ls0 = aeg_dequant_ls[qp % 6][0], ls1 = aeg_dequant_ls[qp % 6][1],
        ls2 = aeg_dequant_ls[qp % 6][2];
    cl_int st = CL_SUCCESS;

    if (nblk <= 0)
        return AEG_OK;
    mres = ocl_buf(o, 0, bytes, CL_MEM_READ_ONLY);
    mlev = ocl_buf(o, 1, bytes, CL_MEM_WRITE_ONLY);
    mnz = ocl_buf(o, 2, (size_t)nblk * 4, CL_MEM_WRITE_ONLY);
    mrec = ocl_buf(o, 3, bytes, CL_MEM_WRITE_ONLY);
    if (!mres || !mlev || !mnz || !mrec)
        return AEG_EBACKEND;
    st |= p_clEnqueueWriteBuffer(o->queue, mres, CL_TRUE, 0, bytes, res, 0,
                                 NULL, NULL);
    st |= p_clSetKernelArg(o->k_tqiq, 0, sizeof(cl_mem), &mres);
    st |= p_clSetKernelArg(o->k_tqiq, 1, sizeof(int), &nblk);
    st |= p_clSetKernelArg(o->k_tqiq, 2, sizeof(int), &qbits);
    st |= p_clSetKernelArg(o->k_tqiq, 3, sizeof(int), &f);
    st |= p_clSetKernelArg(o->k_tqiq, 4, sizeof(int), &mf0);
    st |= p_clSetKernelArg(o->k_tqiq, 5, sizeof(int), &mf1);
    st |= p_clSetKernelArg(o->k_tqiq, 6, sizeof(int), &mf2);
    st |= p_clSetKernelArg(o->k_tqiq, 7, sizeof(int), &sh);
    st |= p_clSetKernelArg(o->k_tqiq, 8, sizeof(int), &ls0);
    st |= p_clSetKernelArg(o->k_tqiq, 9, sizeof(int), &ls1);
    st |= p_clSetKernelArg(o->k_tqiq, 10, sizeof(int), &ls2);
    st |= p_clSetKernelArg(o->k_tqiq, 11, sizeof(cl_mem), &mlev);
    st |= p_clSetKernelArg(o->k_tqiq, 12, sizeof(cl_mem), &mnz);
    st |= p_clSetKernelArg(o->k_tqiq, 13, sizeof(cl_mem), &mrec);
    if (st != CL_SUCCESS)
        return AEG_EBACKEND;
    st = p_clEnqueueNDRangeKernel(o->queue, o->k_tqiq, 1, NULL, &nthreads,
                                  NULL, 0, NULL, NULL);
    if (st != CL_SUCCESS)
        return AEG_EBACKEND;
    /* blocking reads in order: the last one returns when all are done */
    st |= p_clEnqueueReadBuffer(o->queue, mlev, CL_TRUE, 0, bytes, level, 0,
                                NULL, NULL);
    if (nz)
        st |= p_clEnqueueReadBuffer(o->queue, mnz, CL_TRUE, 0,
                                    (size_t)nblk * 4, nz, 0, NULL, NULL);
    st |= p_clEnqueueReadBuffer(o->queue, mrec, CL_TRUE, 0, bytes, rec, 0,
                                NULL, NULL);
    return st == CL_SUCCESS ? AEG_OK : AEG_EBACKEND;
}

static int ocl_tq_dc_batch(void *priv, const int16_t *res, int nblk, int qp,
                           int intra, int16_t *level, int32_t *nz, int32_t *dc)
{
    if (nblk <= 0)
        return AEG_OK;

    ocl_ctx *o = (ocl_ctx *)priv;
    cl_mem mres, mlev, mnz, mdc;
    size_t bytes = (size_t)nblk * 16 * 2;
    size_t nthreads = (size_t)nblk;
    int qbits = 15 + qp / 6;
    int f = aeg_quant_bias(qbits, intra ? 3 : 6);
    int mf0 = aeg_quant_mf[qp % 6][0], mf1 = aeg_quant_mf[qp % 6][1],
        mf2 = aeg_quant_mf[qp % 6][2];
    cl_int st = CL_SUCCESS;

    if (nblk <= 0)
        return AEG_OK;
    mres = ocl_buf(o, 0, bytes, CL_MEM_READ_ONLY);
    mlev = ocl_buf(o, 1, bytes, CL_MEM_WRITE_ONLY);
    mnz = ocl_buf(o, 2, (size_t)nblk * 4, CL_MEM_WRITE_ONLY);
    mdc = ocl_buf(o, 3, (size_t)nblk * 4, CL_MEM_WRITE_ONLY);
    if (!mres || !mlev || !mnz || !mdc)
        return AEG_EBACKEND;
    st |= p_clEnqueueWriteBuffer(o->queue, mres, CL_TRUE, 0, bytes, res, 0,
                                 NULL, NULL);
    st |= p_clSetKernelArg(o->k_tqdc, 0, sizeof(cl_mem), &mres);
    st |= p_clSetKernelArg(o->k_tqdc, 1, sizeof(int), &nblk);
    st |= p_clSetKernelArg(o->k_tqdc, 2, sizeof(int), &qbits);
    st |= p_clSetKernelArg(o->k_tqdc, 3, sizeof(int), &f);
    st |= p_clSetKernelArg(o->k_tqdc, 4, sizeof(int), &mf0);
    st |= p_clSetKernelArg(o->k_tqdc, 5, sizeof(int), &mf1);
    st |= p_clSetKernelArg(o->k_tqdc, 6, sizeof(int), &mf2);
    st |= p_clSetKernelArg(o->k_tqdc, 7, sizeof(cl_mem), &mlev);
    st |= p_clSetKernelArg(o->k_tqdc, 8, sizeof(cl_mem), &mnz);
    st |= p_clSetKernelArg(o->k_tqdc, 9, sizeof(cl_mem), &mdc);
    if (st != CL_SUCCESS)
        return AEG_EBACKEND;
    st = p_clEnqueueNDRangeKernel(o->queue, o->k_tqdc, 1, NULL, &nthreads,
                                  NULL, 0, NULL, NULL);
    if (st != CL_SUCCESS)
        return AEG_EBACKEND;
    st |= p_clEnqueueReadBuffer(o->queue, mlev, CL_TRUE, 0, bytes, level, 0,
                                NULL, NULL);
    if (nz)
        st |= p_clEnqueueReadBuffer(o->queue, mnz, CL_TRUE, 0,
                                    (size_t)nblk * 4, nz, 0, NULL, NULL);
    st |= p_clEnqueueReadBuffer(o->queue, mdc, CL_TRUE, 0, (size_t)nblk * 4,
                                dc, 0, NULL, NULL);
    return st == CL_SUCCESS ? AEG_OK : AEG_EBACKEND;
}

static int ocl_iq_dc_batch(void *priv, const int16_t *level, const int32_t *dc,
                           int nblk, int qp, int16_t *res)
{
    if (nblk <= 0)
        return AEG_OK;

    ocl_ctx *o = (ocl_ctx *)priv;
    cl_mem mlev, mdc, mres;
    size_t bytes = (size_t)nblk * 16 * 2;
    size_t nthreads = (size_t)nblk;
    int sh = qp / 6;
    int ls0 = aeg_dequant_ls[qp % 6][0], ls1 = aeg_dequant_ls[qp % 6][1],
        ls2 = aeg_dequant_ls[qp % 6][2];
    cl_int st = CL_SUCCESS;

    if (nblk <= 0)
        return AEG_OK;
    mlev = ocl_buf(o, 0, bytes, CL_MEM_READ_ONLY);
    mdc = ocl_buf(o, 1, (size_t)nblk * 4, CL_MEM_READ_ONLY);
    mres = ocl_buf(o, 2, bytes, CL_MEM_WRITE_ONLY);
    if (!mlev || !mdc || !mres)
        return AEG_EBACKEND;
    st |= p_clEnqueueWriteBuffer(o->queue, mlev, CL_TRUE, 0, bytes, level, 0,
                                 NULL, NULL);
    st |= p_clEnqueueWriteBuffer(o->queue, mdc, CL_TRUE, 0, (size_t)nblk * 4,
                                 dc, 0, NULL, NULL);
    st |= p_clSetKernelArg(o->k_iqdc, 0, sizeof(cl_mem), &mlev);
    st |= p_clSetKernelArg(o->k_iqdc, 1, sizeof(cl_mem), &mdc);
    st |= p_clSetKernelArg(o->k_iqdc, 2, sizeof(int), &nblk);
    st |= p_clSetKernelArg(o->k_iqdc, 3, sizeof(int), &sh);
    st |= p_clSetKernelArg(o->k_iqdc, 4, sizeof(int), &ls0);
    st |= p_clSetKernelArg(o->k_iqdc, 5, sizeof(int), &ls1);
    st |= p_clSetKernelArg(o->k_iqdc, 6, sizeof(int), &ls2);
    st |= p_clSetKernelArg(o->k_iqdc, 7, sizeof(cl_mem), &mres);
    if (st != CL_SUCCESS)
        return AEG_EBACKEND;
    st = p_clEnqueueNDRangeKernel(o->queue, o->k_iqdc, 1, NULL, &nthreads,
                                  NULL, 0, NULL, NULL);
    if (st != CL_SUCCESS)
        return AEG_EBACKEND;
    st = p_clEnqueueReadBuffer(o->queue, mres, CL_TRUE, 0, bytes, res, 0, NULL,
                               NULL);
    return st == CL_SUCCESS ? AEG_OK : AEG_EBACKEND;
}

static int ocl_sse_batch(void *priv, const uint8_t *a, int sa, const uint8_t *b,
                         int sb, int w, int h, uint64_t *out)
{
    if (w <= 0 || h <= 0) {
        *out = 0;
        return AEG_OK;
    }

    ocl_ctx *o = (ocl_ctx *)priv;
    cl_mem ma, mb, mout;
    size_t tw = (size_t)w / 8, th = (size_t)h / 8, count = tw * th;
    size_t abytes = (size_t)sa * h;
    uint64_t *tmp;
    size_t i, acc = 0;
    cl_int st;

    ma = ocl_buf(o, 0, abytes, CL_MEM_READ_ONLY);
    mb = ocl_buf(o, 1, (size_t)sb * h, CL_MEM_READ_ONLY);
    mout = ocl_buf(o, 2, count * 8, CL_MEM_WRITE_ONLY);
    if (!ma || !mb || !mout)
        return AEG_EBACKEND;
    OCL_CK(p_clEnqueueWriteBuffer(o->queue, ma, CL_TRUE, 0, abytes, a, 0, NULL, NULL));
    OCL_CK(p_clEnqueueWriteBuffer(o->queue, mb, CL_TRUE, 0, (size_t)sb * h, b, 0, NULL, NULL));
    if (!count) {
        *out = 0;              /* no whole 8x8 tile, as the CPU reference */
        return AEG_OK;
    }
    p_clSetKernelArg(o->k_sse, 0, sizeof(cl_mem), &ma);
    p_clSetKernelArg(o->k_sse, 1, sizeof(int), &sa);
    p_clSetKernelArg(o->k_sse, 2, sizeof(cl_mem), &mb);
    p_clSetKernelArg(o->k_sse, 3, sizeof(int), &sb);
    {
        int itw = (int)tw, ith = (int)th, icount = (int)count;
        p_clSetKernelArg(o->k_sse, 4, sizeof(int), &itw);
        p_clSetKernelArg(o->k_sse, 5, sizeof(int), &ith);
        p_clSetKernelArg(o->k_sse, 6, sizeof(int), &icount);
    }
    p_clSetKernelArg(o->k_sse, 7, sizeof(cl_mem), &mout);
    st = p_clEnqueueNDRangeKernel(o->queue, o->k_sse, 1, NULL, &count, NULL,
                                  0, NULL, NULL);
    if (st != CL_SUCCESS)
        return AEG_EBACKEND;
    tmp = (uint64_t *)malloc(count * 8);
    if (!tmp)
        return AEG_ENOMEM;
    if (p_clEnqueueReadBuffer(o->queue, mout, CL_TRUE, 0, count * 8, tmp, 0,
                              NULL, NULL) != CL_SUCCESS) {
        free(tmp);
        return AEG_EBACKEND;
    }
    for (i = 0; i < count; i++)
        acc += tmp[i];
    free(tmp);
    *out = (uint64_t)acc;
    return AEG_OK;
}

/* One fractional-pel plane of the reference picture.  The padded picture is
 * transferred once and every output sample is computed by an independent work
 * item: no reduction, no local memory, one launch per plane. */
static int ocl_interp_plane(void *priv, const uint8_t *ref, int rstride,
                            int w, int h, int which, uint8_t *dst, int dstride)
{
    if (w <= 0 || h <= 0)
        return AEG_OK;

    ocl_ctx *o = (ocl_ctx *)priv;
    cl_mem mref, mdst;
    size_t rows = (size_t)h + 2 * AEG_PAD;
    size_t count = (size_t)(w + 2 * AEG_PAD) * rows;
    size_t rbytes = (size_t)rstride * rows, dbytes = (size_t)dstride * rows;
    int pad = AEG_PAD;
    cl_int st;

    /* `ref` and `dst` point at the picture origins; the kernel takes the
     * whole padded allocations (rows * stride bytes from pad rows and pad
     * columns before the origin), because its output covers the padding */
    mref = ocl_buf(o, 0, rbytes, CL_MEM_READ_ONLY);
    mdst = ocl_buf(o, 1, dbytes, CL_MEM_WRITE_ONLY);
    if (!mref || !mdst)
        return AEG_EBACKEND;
    if (p_clEnqueueWriteBuffer(o->queue, mref, CL_TRUE, 0, rbytes,
                               ref - (ptrdiff_t)AEG_PAD * rstride - AEG_PAD, 0,
                               NULL, NULL) != CL_SUCCESS)
        return AEG_EBACKEND;
    p_clSetKernelArg(o->k_intp, 0, sizeof(cl_mem), &mref);
    p_clSetKernelArg(o->k_intp, 1, sizeof(int), &rstride);
    p_clSetKernelArg(o->k_intp, 2, sizeof(int), &w);
    p_clSetKernelArg(o->k_intp, 3, sizeof(int), &h);
    p_clSetKernelArg(o->k_intp, 4, sizeof(int), &pad);
    p_clSetKernelArg(o->k_intp, 5, sizeof(int), &which);
    p_clSetKernelArg(o->k_intp, 6, sizeof(cl_mem), &mdst);
    p_clSetKernelArg(o->k_intp, 7, sizeof(int), &dstride);
    st = p_clEnqueueNDRangeKernel(o->queue, o->k_intp, 1, NULL, &count, NULL,
                                  0, NULL, NULL);
    if (st != CL_SUCCESS)
        return AEG_EBACKEND;
    if (p_clEnqueueReadBuffer(o->queue, mdst, CL_TRUE, 0, dbytes,
                              dst - (ptrdiff_t)AEG_PAD * dstride - AEG_PAD, 0,
                              NULL, NULL) != CL_SUCCESS)
        return AEG_EBACKEND;
    return AEG_OK;
}

/* Register the reference picture: its four planes go to device memory once and
 * stay there for every satd_batch() of the picture (slots 4..7, which no other
 * operation touches). */
static int ocl_ref_upload(void *priv, const uint8_t *const planes[4],
                          int pstride, int w, int h)
{
    ocl_ctx *o = (ocl_ctx *)priv;
    size_t rows = (size_t)h + 2 * AEG_PAD;
    size_t bytes = (size_t)pstride * rows;
    int k;

    for (k = 0; k < 4; k++) {
        cl_mem m = ocl_buf(o, 4 + k, bytes, CL_MEM_READ_ONLY);
        if (!m)
            return AEG_EBACKEND;
        OCL_CK(p_clEnqueueWriteBuffer(o->queue, m, CL_TRUE, 0, bytes,
                               planes[k] - (ptrdiff_t)AEG_PAD * pstride - AEG_PAD,
                               0, NULL, NULL));
    }
    o->ref_stride = pstride;
    o->ref_w = w;
    o->ref_h = h;
    o->ref_valid = 1;
    return AEG_OK;
}

static int ocl_satd_batch(void *priv, const uint8_t *cur, int cstride, int bw,
                          int bh, const int16_t *pos, int count, int32_t *out)
{
    if (count <= 0)
        return AEG_OK;

    ocl_ctx *o = (ocl_ctx *)priv;
    cl_mem mcur, mpos, mout, mref[4];
    size_t rows = (size_t)(o->ref_valid ? o->ref_h : 0) + 2 * AEG_PAD;
    size_t cbytes = (size_t)cstride * rows;
    int pad = AEG_PAD, k;
    cl_int st;

    if (!o->ref_valid || !count)
        return AEG_EBACKEND;
    mcur = ocl_buf(o, 0, cbytes, CL_MEM_READ_ONLY);
    mpos = ocl_buf(o, 1, (size_t)count * 8, CL_MEM_READ_ONLY);
    mout = ocl_buf(o, 2, (size_t)count * 4, CL_MEM_WRITE_ONLY);
    if (!mcur || !mpos || !mout)
        return AEG_EBACKEND;
    for (k = 0; k < 4; k++)
        mref[k] = o->buf[4 + k];
    OCL_CK(p_clEnqueueWriteBuffer(o->queue, mcur, CL_TRUE, 0, cbytes,
                           cur - (ptrdiff_t)AEG_PAD * cstride - AEG_PAD,
                           0, NULL, NULL));
    OCL_CK(p_clEnqueueWriteBuffer(o->queue, mpos, CL_TRUE, 0, (size_t)count * 8, pos,
                           0, NULL, NULL));
    p_clSetKernelArg(o->k_satd, 0, sizeof(cl_mem), &mcur);
    p_clSetKernelArg(o->k_satd, 1, sizeof(int), &cstride);
    for (k = 0; k < 4; k++)
        p_clSetKernelArg(o->k_satd, 2 + k, sizeof(cl_mem), &mref[k]);
    p_clSetKernelArg(o->k_satd, 6, sizeof(int), &o->ref_stride);
    p_clSetKernelArg(o->k_satd, 7, sizeof(int), &pad);
    p_clSetKernelArg(o->k_satd, 8, sizeof(int), &bw);
    p_clSetKernelArg(o->k_satd, 9, sizeof(int), &bh);
    p_clSetKernelArg(o->k_satd, 10, sizeof(cl_mem), &mpos);
    p_clSetKernelArg(o->k_satd, 11, sizeof(int), &count);
    p_clSetKernelArg(o->k_satd, 12, sizeof(cl_mem), &mout);
    {
        size_t n = (size_t)count;
        st = p_clEnqueueNDRangeKernel(o->queue, o->k_satd, 1, NULL, &n, NULL,
                                      0, NULL, NULL);
    }
    if (st != CL_SUCCESS)
        return AEG_EBACKEND;
    OCL_CK(p_clEnqueueReadBuffer(o->queue, mout, CL_TRUE, 0, (size_t)count * 4, out,
                          0, NULL, NULL));
    return AEG_OK;
}

/* The lowres search of one field (aeg_be_ops.lowres_search): both padded
 * planes and the seed go over, one work-group walks the field's
 * diagonals, and the vectors and costs come back in one read (slots 8-11,
 * which nothing else uses). */
static int ocl_lowres_search(void *priv, const uint8_t *cur,
                             const uint8_t *ref, int stride, int w, int h,
                             const int *seed, int num, int den, int range,
                             int *out)
{
    ocl_ctx *o = (ocl_ctx *)priv;
    const int pad = AEG_LA_PAD, mb_w = w / 8, mb_h = h / 8;
    const int has_seed = seed != NULL, hexit = AEG_LA_RANGE;
    const size_t pbytes = (size_t)stride * (size_t)(h + 2 * pad);
    const size_t nb = (size_t)mb_w * (size_t)mb_h;
    const ptrdiff_t org = (ptrdiff_t)pad * stride + pad;
    cl_mem mc, mr, ms, mo;
    size_t local = o->lrs_wg, global = o->lrs_wg;
    cl_int st = 0;

    if (mb_w <= 0 || mb_h <= 0)
        return AEG_OK;
    mc = ocl_buf(o, 8, pbytes, CL_MEM_READ_ONLY);
    mr = ocl_buf(o, 9, pbytes, CL_MEM_READ_ONLY);
    mo = ocl_buf(o, 11, nb * 3 * sizeof(int), CL_MEM_READ_WRITE);
    ms = has_seed ? ocl_buf(o, 10, nb * 2 * sizeof(int), CL_MEM_READ_ONLY)
                  : mo;   /* not read: any valid buffer */
    if (!mc || !mr || !mo || !ms)
        return AEG_EBACKEND;
    OCL_CK(p_clEnqueueWriteBuffer(o->queue, mc, CL_TRUE, 0, pbytes, cur - org,
                                  0, NULL, NULL));
    OCL_CK(p_clEnqueueWriteBuffer(o->queue, mr, CL_TRUE, 0, pbytes, ref - org,
                                  0, NULL, NULL));
    if (has_seed)
        OCL_CK(p_clEnqueueWriteBuffer(o->queue, ms, CL_TRUE, 0,
                                      nb * 2 * sizeof(int), seed, 0, NULL,
                                      NULL));
    st |= p_clSetKernelArg(o->k_lrs, 0, sizeof(cl_mem), &mc);
    st |= p_clSetKernelArg(o->k_lrs, 1, sizeof(cl_mem), &mr);
    st |= p_clSetKernelArg(o->k_lrs, 2, sizeof(int), &stride);
    st |= p_clSetKernelArg(o->k_lrs, 3, sizeof(int), &pad);
    st |= p_clSetKernelArg(o->k_lrs, 4, sizeof(int), &w);
    st |= p_clSetKernelArg(o->k_lrs, 5, sizeof(int), &h);
    st |= p_clSetKernelArg(o->k_lrs, 6, sizeof(int), &mb_w);
    st |= p_clSetKernelArg(o->k_lrs, 7, sizeof(int), &mb_h);
    st |= p_clSetKernelArg(o->k_lrs, 8, sizeof(cl_mem), &ms);
    st |= p_clSetKernelArg(o->k_lrs, 9, sizeof(int), &has_seed);
    st |= p_clSetKernelArg(o->k_lrs, 10, sizeof(int), &num);
    st |= p_clSetKernelArg(o->k_lrs, 11, sizeof(int), &den);
    st |= p_clSetKernelArg(o->k_lrs, 12, sizeof(int), &range);
    st |= p_clSetKernelArg(o->k_lrs, 13, sizeof(int), &hexit);
    st |= p_clSetKernelArg(o->k_lrs, 14, sizeof(cl_mem), &mo);
    if (st != CL_SUCCESS)
        return AEG_EBACKEND;
    OCL_CK(p_clEnqueueNDRangeKernel(o->queue, o->k_lrs, 1, NULL, &global,
                                    &local, 0, NULL, NULL));
    OCL_CK(p_clEnqueueReadBuffer(o->queue, mo, CL_TRUE, 0,
                                 nb * 3 * sizeof(int), out, 0, NULL, NULL));
    return AEG_OK;
}

/* The ring of resident lowres planes, slot k at k * slot bytes, each tagged
 * with the picture it holds (-1 = none); (re)allocated, all slots empty,
 * when the geometry changes or more slots are asked for. */
static int ocl_lr_ring(ocl_ctx *o, size_t slot, int nslots)
{
    cl_int st = 0;
    int i;

    /* the kernels' offsets are ints */
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
        p_clReleaseMemObject(o->lr_mem);
    /* read-write: lowres_build writes a slot */
    o->lr_mem = p_clCreateBuffer(o->ctx, CL_MEM_READ_WRITE,
                                 slot * (size_t)nslots, NULL, &st);
    o->lr_slot = slot;
    o->lr_nslots = o->lr_mem && st == CL_SUCCESS ? nslots : 0;
    for (i = 0; i < nslots; i++)
        o->lr_tag[i] = -1;
    return o->lr_nslots ? AEG_OK : AEG_EBACKEND;
}

/* the pictures whose slot holds something else go over (queued) */
static int ocl_lr_pics(ocl_ctx *o, const aeg_la_pic *pics, int npics,
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
        if (p_clEnqueueWriteBuffer(o->queue, o->lr_mem, CL_FALSE,
                                   slot * (size_t)k, slot, pics[i].plane - org,
                                   0, NULL, NULL) != CL_SUCCESS)
            return AEG_EBACKEND;
        o->lr_tag[k] = pics[i].tag;
    }
    return AEG_OK;
}

/* buffers 12-15: chain table, job table, seeds, fields */
static int ocl_lowres_chains(void *priv, const aeg_la_pic *pics, int npics,
                             int nslots, const aeg_la_chain *chains,
                             int nchains, const aeg_la_job *jobs, int stride,
                             int w, int h)
{
    ocl_ctx *o = (ocl_ctx *)priv;
    const int pad = AEG_LA_PAD, mb_w = w / 8, mb_h = h / 8, hexit = AEG_LA_RANGE;
    const size_t slot = (size_t)stride * (size_t)(h + 2 * pad);
    const ptrdiff_t org = (ptrdiff_t)pad * stride + pad;
    const int fsz = 3 * mb_w * mb_h;
    int njobs = 0, nseed = 0, c, i, rc = AEG_EBACKEND;
    size_t need, local = o->lrs_wg, global;
    cl_mem mc, mj, ms, mo;
    cl_int st = 0;

    if (mb_w <= 0 || mb_h <= 0 || nchains <= 0)
        return AEG_OK;
    for (c = 0; c < nchains; c++) {
        if (chains[c].first + chains[c].count > njobs)
            njobs = chains[c].first + chains[c].count;
        nseed += chains[c].seed != NULL;
    }
    /* the kernel's offsets are ints */
    if ((size_t)fsz * (size_t)njobs > 0x1fffffffu)
        return AEG_EBACKEND;
    if ((rc = ocl_lr_ring(o, slot, nslots)) != AEG_OK)
        return rc;
    rc = AEG_EBACKEND;
    need = 4 * (size_t)nchains + 5 * (size_t)njobs;
    if (need > o->lr_host_cap) {
        int *t = (int *)realloc(o->lr_host, sizeof(int) * need);
        if (!t)
            return AEG_ENOMEM;
        o->lr_host = t;
        o->lr_host_cap = need;
    }
    mc = ocl_buf(o, 12, sizeof(int) * 4 * (size_t)nchains, CL_MEM_READ_ONLY);
    mj = ocl_buf(o, 13, sizeof(int) * 5 * (size_t)njobs, CL_MEM_READ_ONLY);
    mo = ocl_buf(o, 15, sizeof(int) * (size_t)fsz * (size_t)njobs,
                 CL_MEM_READ_WRITE);
    ms = nseed ? ocl_buf(o, 14, sizeof(int) * (size_t)(2 * fsz / 3) * nseed,
                         CL_MEM_READ_ONLY)
               : mo;            /* not read: any valid buffer */
    if (!mc || !mj || !ms || !mo)
        return AEG_EBACKEND;

    if (ocl_lr_pics(o, pics, npics, nslots, slot, org) != AEG_OK)
        goto fail;
    nseed = 0;
    for (c = 0; c < nchains; c++) {
        int *ci = o->lr_host + 4 * c;
        ci[0] = pics[chains[c].cur].slot;
        ci[1] = chains[c].first;
        ci[2] = chains[c].count;
        ci[3] = -1;
        if (chains[c].seed) {
            const size_t sb = sizeof(int) * (size_t)(2 * fsz / 3);
            ci[3] = nseed * (2 * fsz / 3);
            if (p_clEnqueueWriteBuffer(o->queue, ms, CL_FALSE, sb * nseed, sb,
                                       chains[c].seed, 0, NULL,
                                       NULL) != CL_SUCCESS)
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
    if (p_clEnqueueWriteBuffer(o->queue, mc, CL_FALSE, 0,
                               sizeof(int) * 4 * (size_t)nchains, o->lr_host,
                               0, NULL, NULL) != CL_SUCCESS ||
        p_clEnqueueWriteBuffer(o->queue, mj, CL_FALSE, 0,
                               sizeof(int) * 5 * (size_t)njobs,
                               o->lr_host + 4 * nchains, 0, NULL,
                               NULL) != CL_SUCCESS)
        goto fail;
    {
        int ss = (int)slot;
        st |= p_clSetKernelArg(o->k_lrc, 0, sizeof(cl_mem), &o->lr_mem);
        st |= p_clSetKernelArg(o->k_lrc, 1, sizeof(int), &ss);
        st |= p_clSetKernelArg(o->k_lrc, 2, sizeof(int), &stride);
        st |= p_clSetKernelArg(o->k_lrc, 3, sizeof(int), &pad);
        st |= p_clSetKernelArg(o->k_lrc, 4, sizeof(int), &w);
        st |= p_clSetKernelArg(o->k_lrc, 5, sizeof(int), &h);
        st |= p_clSetKernelArg(o->k_lrc, 6, sizeof(int), &mb_w);
        st |= p_clSetKernelArg(o->k_lrc, 7, sizeof(int), &mb_h);
        st |= p_clSetKernelArg(o->k_lrc, 8, sizeof(cl_mem), &mc);
        st |= p_clSetKernelArg(o->k_lrc, 9, sizeof(cl_mem), &mj);
        st |= p_clSetKernelArg(o->k_lrc, 10, sizeof(cl_mem), &ms);
        st |= p_clSetKernelArg(o->k_lrc, 11, sizeof(int), &hexit);
        st |= p_clSetKernelArg(o->k_lrc, 12, sizeof(cl_mem), &mo);
    }
    global = local * (size_t)nchains;
    if (st != CL_SUCCESS ||
        p_clEnqueueNDRangeKernel(o->queue, o->k_lrc, 1, NULL, &global, &local,
                                 0, NULL, NULL) != CL_SUCCESS)
        goto fail;
    for (i = 0; i < njobs; i++)
        if (p_clEnqueueReadBuffer(o->queue, mo, CL_FALSE,
                                  sizeof(int) * (size_t)fsz * (size_t)i,
                                  sizeof(int) * (size_t)fsz, jobs[i].out, 0,
                                  NULL, NULL) != CL_SUCCESS)
            goto fail;
    if (p_clFinish(o->queue) == CL_SUCCESS)
        return AEG_OK;
fail:
    /* whatever was enqueued is drained before the host buffers go away,
     * and no slot is trusted after a failure */
    p_clFinish(o->queue);
    for (i = 0; i < o->lr_nslots; i++)
        o->lr_tag[i] = -1;
    return rc;
}

/* The rest of the lookahead on the ring.  A failure leaves no slot
 * trusted, as in lowres_chains. */
static int ocl_lr_fail(ocl_ctx *o, int rc)
{
    int i;
    p_clFinish(o->queue);
    for (i = 0; i < o->lr_nslots; i++)
        o->lr_tag[i] = -1;
    return rc;
}

/* buffer 16: the source luma */
static int ocl_lowres_build(void *priv, const uint8_t *src, int sstride,
                            int sw, int sh, const aeg_la_pic *pic, int nslots,
                            int stride, int w, int h)
{
    ocl_ctx *o = (ocl_ctx *)priv;
    const int pad = AEG_LA_PAD;
    const size_t slot = (size_t)stride * (size_t)(h + 2 * pad);
    const ptrdiff_t org = (ptrdiff_t)pad * stride + pad;
    const size_t sbytes = (size_t)sstride * (size_t)(sh - 1) + (size_t)sw;
    size_t global = slot;
    cl_mem ms;
    cl_int st = 0;
    int rc, dst;

    if (w <= 0 || h <= 0 || sw <= 0 || sh <= 0)
        return AEG_EINVAL;
    if ((rc = ocl_lr_ring(o, slot, nslots)) != AEG_OK)
        return rc;
    if (pic->slot < 0 || pic->slot >= nslots)
        return AEG_EINVAL;
    ms = ocl_buf(o, 16, sbytes, CL_MEM_READ_ONLY);
    if (!ms)
        return AEG_EBACKEND;
    dst = (int)(slot * (size_t)pic->slot);
    o->lr_tag[pic->slot] = -1;
    if (p_clEnqueueWriteBuffer(o->queue, ms, CL_FALSE, 0, sbytes, src, 0, NULL,
                               NULL) != CL_SUCCESS)
        return ocl_lr_fail(o, AEG_EBACKEND);
    st |= p_clSetKernelArg(o->k_lrb, 0, sizeof(cl_mem), &ms);
    st |= p_clSetKernelArg(o->k_lrb, 1, sizeof(int), &sstride);
    st |= p_clSetKernelArg(o->k_lrb, 2, sizeof(int), &sw);
    st |= p_clSetKernelArg(o->k_lrb, 3, sizeof(int), &sh);
    st |= p_clSetKernelArg(o->k_lrb, 4, sizeof(cl_mem), &o->lr_mem);
    st |= p_clSetKernelArg(o->k_lrb, 5, sizeof(int), &dst);
    st |= p_clSetKernelArg(o->k_lrb, 6, sizeof(int), &stride);
    st |= p_clSetKernelArg(o->k_lrb, 7, sizeof(int), &pad);
    st |= p_clSetKernelArg(o->k_lrb, 8, sizeof(int), &w);
    st |= p_clSetKernelArg(o->k_lrb, 9, sizeof(int), &h);
    if (st != CL_SUCCESS ||
        p_clEnqueueNDRangeKernel(o->queue, o->k_lrb, 1, NULL, &global, NULL, 0,
                                 NULL, NULL) != CL_SUCCESS ||
        p_clEnqueueReadBuffer(o->queue, o->lr_mem, CL_TRUE,
                              slot * (size_t)pic->slot, slot,
                              (uint8_t *)pic->plane - org, 0, NULL,
                              NULL) != CL_SUCCESS)
        return ocl_lr_fail(o, AEG_EBACKEND);
    o->lr_tag[pic->slot] = pic->tag;
    return AEG_OK;
}

/* buffers 17, 18: the slot table, the estimates */
static int ocl_lowres_intra(void *priv, const aeg_la_pic *pics, int npics,
                            int nslots, int *const *out, int i8, int stride,
                            int w, int h)
{
    ocl_ctx *o = (ocl_ctx *)priv;
    const int pad = AEG_LA_PAD, mb_w = w / 8, mb_h = h / 8;
    const size_t slot = (size_t)stride * (size_t)(h + 2 * pad);
    const ptrdiff_t org = (ptrdiff_t)pad * stride + pad;
    const size_t nb = (size_t)mb_w * (size_t)mb_h;
    size_t global = nb * (size_t)npics;
    cl_mem mt, mo;
    cl_int st = 0;
    int rc, i, ss = (int)slot;

    if (npics <= 0 || nb == 0)
        return AEG_OK;
    if (global > 0x7fffffffu)
        return AEG_EBACKEND;
    if ((rc = ocl_lr_ring(o, slot, nslots)) != AEG_OK)
        return rc;
    if ((size_t)npics > o->lr_host_cap) {
        int *t = (int *)realloc(o->lr_host, sizeof(int) * (size_t)npics);
        if (!t)
            return AEG_ENOMEM;
        o->lr_host = t;
        o->lr_host_cap = (size_t)npics;
    }
    mt = ocl_buf(o, 17, sizeof(int) * (size_t)npics, CL_MEM_READ_ONLY);
    mo = ocl_buf(o, 18, sizeof(int) * global, CL_MEM_READ_WRITE);
    if (!mt || !mo)
        return AEG_EBACKEND;
    if ((rc = ocl_lr_pics(o, pics, npics, nslots, slot, org)) != AEG_OK)
        return ocl_lr_fail(o, rc);
    for (i = 0; i < npics; i++)
        o->lr_host[i] = pics[i].slot;
    if (p_clEnqueueWriteBuffer(o->queue, mt, CL_FALSE, 0,
                               sizeof(int) * (size_t)npics, o->lr_host, 0,
                               NULL, NULL) != CL_SUCCESS)
        return ocl_lr_fail(o, AEG_EBACKEND);
    st |= p_clSetKernelArg(o->k_lri, 0, sizeof(cl_mem), &o->lr_mem);
    st |= p_clSetKernelArg(o->k_lri, 1, sizeof(int), &ss);
    st |= p_clSetKernelArg(o->k_lri, 2, sizeof(int), &stride);
    st |= p_clSetKernelArg(o->k_lri, 3, sizeof(int), &pad);
    st |= p_clSetKernelArg(o->k_lri, 4, sizeof(int), &mb_w);
    st |= p_clSetKernelArg(o->k_lri, 5, sizeof(int), &mb_h);
    st |= p_clSetKernelArg(o->k_lri, 6, sizeof(cl_mem), &mt);
    st |= p_clSetKernelArg(o->k_lri, 7, sizeof(int), &npics);
    st |= p_clSetKernelArg(o->k_lri, 8, sizeof(int), &i8);
    st |= p_clSetKernelArg(o->k_lri, 9, sizeof(cl_mem), &mo);
    if (st != CL_SUCCESS ||
        p_clEnqueueNDRangeKernel(o->queue, o->k_lri, 1, NULL, &global, NULL, 0,
                                 NULL, NULL) != CL_SUCCESS)
        return ocl_lr_fail(o, AEG_EBACKEND);
    for (i = 0; i < npics; i++)
        if (p_clEnqueueReadBuffer(o->queue, mo, CL_FALSE,
                                  sizeof(int) * nb * (size_t)i,
                                  sizeof(int) * nb, out[i], 0, NULL,
                                  NULL) != CL_SUCCESS)
            return ocl_lr_fail(o, AEG_EBACKEND);
    if (p_clFinish(o->queue) != CL_SUCCESS)
        return ocl_lr_fail(o, AEG_EBACKEND);
    return AEG_OK;
}

/* buffers 17-19: the job table, the per-block results, the fields and
 * intra estimates (each distinct array once) */
static int ocl_lowres_costs(void *priv, const aeg_la_pic *pics, int npics,
                            int nslots, const aeg_la_cost *costs, int ncosts,
                            int stride, int w, int h)
{
    ocl_ctx *o = (ocl_ctx *)priv;
    const int pad = AEG_LA_PAD, mb_w = w / 8, mb_h = h / 8;
    const size_t slot = (size_t)stride * (size_t)(h + 2 * pad);
    const ptrdiff_t org = (ptrdiff_t)pad * stride + pad;
    const size_t nb = (size_t)mb_w * (size_t)mb_h;
    size_t global = nb * (size_t)ncosts, need, ndata = 0;
    const int **arr;
    int *off, narr = 0, rc, i, k, ss = (int)slot;
    cl_mem mj, mo, md;
    cl_int st = 0;

    if (ncosts <= 0 || nb == 0)
        return AEG_OK;
    if (global > 0x1fffffffu)
        return AEG_EBACKEND;
    if ((rc = ocl_lr_ring(o, slot, nslots)) != AEG_OK)
        return rc;
    /* host tables: the jobs (6 ints each), then the distinct arrays'
     * pointers and their offsets in the data buffer */
    need = 6 * (size_t)ncosts + 3 * (size_t)ncosts * (1 + sizeof(void *) / sizeof(int) + 1);
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
    mj = ocl_buf(o, 17, sizeof(int) * 6 * (size_t)ncosts, CL_MEM_READ_ONLY);
    mo = ocl_buf(o, 18, sizeof(int) * global, CL_MEM_READ_WRITE);
    md = ocl_buf(o, 19, sizeof(int) * ndata, CL_MEM_READ_ONLY);
    if (!mj || !mo || !md)
        return AEG_EBACKEND;
    if ((rc = ocl_lr_pics(o, pics, npics, nslots, slot, org)) != AEG_OK)
        return ocl_lr_fail(o, rc);
    for (k = 0; k < narr; k++) {
        const size_t len = (k + 1 < narr ? (size_t)off[k + 1] : ndata) -
                           (size_t)off[k];
        if (p_clEnqueueWriteBuffer(o->queue, md, CL_FALSE,
                                   sizeof(int) * (size_t)off[k],
                                   sizeof(int) * len, arr[k], 0, NULL,
                                   NULL) != CL_SUCCESS)
            return ocl_lr_fail(o, AEG_EBACKEND);
    }
    if (p_clEnqueueWriteBuffer(o->queue, mj, CL_FALSE, 0,
                               sizeof(int) * 6 * (size_t)ncosts, o->lr_host, 0,
                               NULL, NULL) != CL_SUCCESS)
        return ocl_lr_fail(o, AEG_EBACKEND);
    st |= p_clSetKernelArg(o->k_lrk, 0, sizeof(cl_mem), &o->lr_mem);
    st |= p_clSetKernelArg(o->k_lrk, 1, sizeof(int), &ss);
    st |= p_clSetKernelArg(o->k_lrk, 2, sizeof(int), &stride);
    st |= p_clSetKernelArg(o->k_lrk, 3, sizeof(int), &pad);
    st |= p_clSetKernelArg(o->k_lrk, 4, sizeof(int), &mb_w);
    st |= p_clSetKernelArg(o->k_lrk, 5, sizeof(int), &mb_h);
    st |= p_clSetKernelArg(o->k_lrk, 6, sizeof(cl_mem), &mj);
    st |= p_clSetKernelArg(o->k_lrk, 7, sizeof(int), &ncosts);
    st |= p_clSetKernelArg(o->k_lrk, 8, sizeof(cl_mem), &md);
    st |= p_clSetKernelArg(o->k_lrk, 9, sizeof(cl_mem), &mo);
    if (st != CL_SUCCESS ||
        p_clEnqueueNDRangeKernel(o->queue, o->k_lrk, 1, NULL, &global, NULL, 0,
                                 NULL, NULL) != CL_SUCCESS ||
        p_clEnqueueReadBuffer(o->queue, mo, CL_TRUE, 0, sizeof(int) * global,
                              o->lk_blk, 0, NULL, NULL) != CL_SUCCESS)
        return ocl_lr_fail(o, AEG_EBACKEND);
    for (i = 0; i < ncosts; i++)
        aeg_la_cost_sum(o->lk_blk + nb * (size_t)i, mb_w, mb_h,
                        costs[i].total, costs[i].intra);
    return AEG_OK;
}

static void ocl_sync(void *priv)
{
    ocl_ctx *o = (ocl_ctx *)priv;
    p_clFinish(o->queue);
}

static void ocl_destroy(void *priv)
{
    ocl_ctx *o = (ocl_ctx *)priv;
    int i;
    if (!o)
        return;
    for (i = 0; i < NBUF; i++)
        if (o->buf[i])
            p_clReleaseMemObject(o->buf[i]);
    free(o->pos);
    if (o->lr_mem)
        p_clReleaseMemObject(o->lr_mem);
    free(o->lr_tag);
    free(o->lr_host);
    free(o->lk_blk);
    if (o->k_lrb) p_clReleaseKernel(o->k_lrb);
    if (o->k_lri) p_clReleaseKernel(o->k_lri);
    if (o->k_lrk) p_clReleaseKernel(o->k_lrk);
    if (o->k_lrc) p_clReleaseKernel(o->k_lrc);
    if (o->k_sad) p_clReleaseKernel(o->k_sad);
    if (o->k_tq) p_clReleaseKernel(o->k_tq);
    if (o->k_iq) p_clReleaseKernel(o->k_iq);
    if (o->k_tqiq) p_clReleaseKernel(o->k_tqiq);
    if (o->k_sse) p_clReleaseKernel(o->k_sse);
    if (o->k_intp) p_clReleaseKernel(o->k_intp);
    if (o->k_satd) p_clReleaseKernel(o->k_satd);
    if (o->k_tqdc) p_clReleaseKernel(o->k_tqdc);
    if (o->k_iqdc) p_clReleaseKernel(o->k_iqdc);
    if (o->k_lrs) p_clReleaseKernel(o->k_lrs);
    if (o->prog) p_clReleaseProgram(o->prog);
    if (o->queue) p_clReleaseCommandQueue(o->queue);
    if (o->ctx) p_clReleaseContext(o->ctx);
    free(o);
}

/* ------------------------------------------------------------------ */
/* Setup                                                               */
/* ------------------------------------------------------------------ */
static int ocl_load(void **lib)
{
    static const char *names[] = {
        "libOpenCL.so.1", "libOpenCL.so", "libOpenCL.dylib",
        "/usr/lib/x86_64-linux-gnu/libOpenCL.so.1", NULL
    };
    int i;
    /* AEG_OPENCL_LIB: a library outside the loader's search path (a pip-installed
     * runtime, a second driver) -- tried before the usual names */
    const char *ev = getenv("AEG_OPENCL_LIB");
    if (ev && *ev) {
        *lib = dlopen(ev, RTLD_NOW | RTLD_LOCAL);
        if (*lib)
            return AEG_OK;
        aeg_log(AEG_LOG_WARNING, "aeg: AEG_OPENCL_LIB=%s: %s", ev, dlerror());
    }
    for (i = 0; names[i]; i++) {
        *lib = dlopen(names[i], RTLD_NOW | RTLD_LOCAL);
        if (*lib)
            return AEG_OK;
    }
    return AEG_EBACKEND;
}

/* The kernel objects of one instance (each clone has its own: setting a
 * kernel's arguments and launching it is not thread-safe). */
static int ocl_kernels(ocl_ctx *o)
{
    cl_int st;
    o->k_sad = p_clCreateKernel(o->prog, "sad_batch", &st);
    o->k_tq = p_clCreateKernel(o->prog, "tq_batch", &st);
    o->k_iq = p_clCreateKernel(o->prog, "iq_idct_batch", &st);
    o->k_tqiq = p_clCreateKernel(o->prog, "tq_iq_batch", &st);
    o->k_sse = p_clCreateKernel(o->prog, "sse_batch", &st);
    o->k_intp = p_clCreateKernel(o->prog, "interp_plane", &st);
    o->k_satd = p_clCreateKernel(o->prog, "satd_batch", &st);
    o->k_tqdc = p_clCreateKernel(o->prog, "tq_dc_batch", &st);
    o->k_iqdc = p_clCreateKernel(o->prog, "iq_dc_batch", &st);
    o->k_lrs = p_clCreateKernel(o->prog, "lowres_search", &st);
    o->k_lrc = p_clCreateKernel(o->prog, "lowres_chains", &st);
    o->k_lrb = p_clCreateKernel(o->prog, "lowres_build", &st);
    o->k_lri = p_clCreateKernel(o->prog, "lowres_intra", &st);
    o->k_lrk = p_clCreateKernel(o->prog, "lowres_costs", &st);
    /* one work-group walks a whole field; 64 items cover the longest
     * diagonal of a 1080p lowres picture (68 block rows) in about one
     * pass, and the kernel's own limit (its private arrays) may be lower */
    o->lrs_wg = 64;
    if (o->k_lrs) {
        size_t wg = 0;
        if (p_clGetKernelWorkGroupInfo(o->k_lrs, o->device,
                                       CL_KERNEL_WORK_GROUP_SIZE, sizeof(wg),
                                       &wg, NULL) == CL_SUCCESS &&
            wg > 0 && wg < o->lrs_wg)
            o->lrs_wg = wg;
    }
    if (o->k_lrc) {
        size_t wg = 0;
        if (p_clGetKernelWorkGroupInfo(o->k_lrc, o->device,
                                       CL_KERNEL_WORK_GROUP_SIZE, sizeof(wg),
                                       &wg, NULL) == CL_SUCCESS &&
            wg > 0 && wg < o->lrs_wg)
            o->lrs_wg = wg;
    }
    return o->k_sad && o->k_tq && o->k_iq && o->k_tqiq && o->k_sse &&
           o->k_intp && o->k_satd && o->k_tqdc && o->k_iqdc && o->k_lrs &&
           o->k_lrc && o->k_lrb && o->k_lri && o->k_lrk
               ? AEG_OK : AEG_EBACKEND;
}

static int ocl_clone(void *priv, aeg_be_ops *out);

static void ocl_fill_ops(ocl_ctx *o, aeg_be_ops *ops)
{
    memset(ops, 0, sizeof(*ops));
    ops->priv = o;
    ops->backend = AEG_BACKEND_OPENCL;
    ops->sad_batch = ocl_sad_batch;
    ops->tq_batch = ocl_tq_batch;
    ops->iq_idct_batch = ocl_iq_idct_batch;
    ops->tq_iq_batch = ocl_tq_iq_batch;
    ops->tq_dc_batch = ocl_tq_dc_batch;
    ops->iq_dc_batch = ocl_iq_dc_batch;
    ops->sse_batch = ocl_sse_batch;
    ops->interp_plane = ocl_interp_plane;
    ops->ref_upload = ocl_ref_upload;
    ops->satd_batch = ocl_satd_batch;
    ops->lowres_search = ocl_lowres_search;
    ops->lowres_chains = ocl_lowres_chains;
    ops->lowres_build = ocl_lowres_build;
    ops->lowres_intra = ocl_lowres_intra;
    ops->lowres_costs = ocl_lowres_costs;
    ops->sync = ocl_sync;
    ops->destroy = ocl_destroy;
    ops->clone = ocl_clone;
    snprintf(ops->desc, sizeof(ops->desc), "%.48s %.96s (%d CUs, %s)",
             o->vendor, o->name, o->cus, o->is_gpu ? "GPU" : "CPU device");
}

/* Another instance for another thread (aeg_be_ops.clone): the context and
 * the built program are retained, not rebuilt; the queue, the kernel objects
 * and the buffers are its own.  The library stays loaded by the first
 * instance, which the guard destroys last. */
static int ocl_clone(void *priv, aeg_be_ops *out)
{
    const ocl_ctx *src = (const ocl_ctx *)priv;
    ocl_ctx *o = (ocl_ctx *)calloc(1, sizeof(*o));
    cl_int st;

    if (src->no_clone) {
        free(o);
        aeg_set_last_error("PoCL 4.x-6.x cannot launch from several queues "
                           "at once (fixed in PoCL 7.0): device lanes share "
                           "one instance");
        return AEG_EBACKEND;
    }
    if (!o)
        return AEG_ENOMEM;
    o->no_clone = 0;
    o->platform = src->platform;
    o->device = src->device;
    o->is_gpu = src->is_gpu;
    o->index = src->index;
    o->mem = src->mem;
    o->cus = src->cus;
    memcpy(o->name, src->name, sizeof(o->name));
    memcpy(o->vendor, src->vendor, sizeof(o->vendor));
    if (p_clRetainContext(src->ctx) != CL_SUCCESS) {
        free(o);
        return AEG_EBACKEND;
    }
    o->ctx = src->ctx;
    if (p_clRetainProgram(src->prog) != CL_SUCCESS) {
        ocl_destroy(o);
        return AEG_EBACKEND;
    }
    o->prog = src->prog;
    o->queue = p_clCreateCommandQueue(o->ctx, o->device, 0, &st);
    if (!o->queue || st != CL_SUCCESS || ocl_kernels(o) != AEG_OK) {
        ocl_destroy(o);
        return AEG_EBACKEND;
    }
    ocl_fill_ops(o, out);
    return AEG_OK;
}

/* Every device of every platform, in platform order: the list `--device`
 * indexes.  (The first platform's first device used to be the only one
 * looked at, so a CPU runtime listed before a GPU's took its place.) */
#define OCL_MAXPLAT 16
#define OCL_MAXDEV  32
static int ocl_devices(cl_platform_id *plat, cl_device_id *dev)
{
    cl_platform_id pl[OCL_MAXPLAT];
    cl_uint np = 0, i;
    int n = 0;

    if (p_clGetPlatformIDs(OCL_MAXPLAT, pl, &np) != CL_SUCCESS)
        return 0;
    if (np > OCL_MAXPLAT)
        np = OCL_MAXPLAT;
    for (i = 0; i < np && n < OCL_MAXDEV; i++) {
        cl_device_id d[OCL_MAXDEV];
        cl_uint nd = 0, j;
        if (p_clGetDeviceIDs(pl[i], CL_DEVICE_TYPE_ALL, OCL_MAXDEV, d, &nd) !=
            CL_SUCCESS)
            continue;
        if (nd > OCL_MAXDEV)
            nd = OCL_MAXDEV;
        for (j = 0; j < nd && n < OCL_MAXDEV; j++) {
            plat[n] = pl[i];
            dev[n] = d[j];
            n++;
        }
    }
    return n;
}

static int ocl_dev_is_gpu(cl_device_id d)
{
    /* cl_device_type is a 64-bit cl_bitfield: a smaller buffer makes the
     * query fail (CL_INVALID_VALUE) and every device read as a non-GPU */
    cl_ulong t = 0;
    if (p_clGetDeviceInfo(d, CL_DEVICE_TYPE, sizeof(t), &t, NULL) != CL_SUCCESS)
        return 0;
    return (t & CL_DEVICE_TYPE_GPU) != 0;
}

/* The loader is opened once per process and kept: the entry points are
 * process-wide anyway, and an ICD loader sets up its platform list on first
 * use and never frees it -- closing and reopening the library with every
 * encoder would set it up (and lose it) again each time. */
static pthread_once_t g_ocl_once = PTHREAD_ONCE_INIT;
static int g_ocl_ok;

static void ocl_load_once(void)
{
    void *lib = NULL;
#define LOAD(ret, name, args) do { \
        *(void **)(&p_##name) = dlsym(lib, #name); \
        if (!p_##name) { ocl_load_error(#name); dlclose(lib); return; } \
    } while (0);
    if (ocl_load(&lib) != AEG_OK)
        return;
    OCL_SYMS(LOAD)
#undef LOAD
    g_ocl_ok = 1;
}

/* `want` >= 0: that entry of ocl_devices(); -1: the first GPU, else (unless
 * gpu_only) the first device of any kind.  backend.c asks for gpu_only when
 * the backend is chosen automatically: an OpenCL runtime on the host CPU
 * (PoCL) is a test vehicle, never faster than the native code. */
int aeg_be_opencl_create(aeg_be_ops *ops, int want, int gpu_only)
{
    ocl_ctx *o;
    cl_int st;
    size_t got;
    cl_platform_id plats[OCL_MAXDEV];
    cl_device_id devs[OCL_MAXDEV];
    int ndev, pick = -1, k;

    pthread_once(&g_ocl_once, ocl_load_once);
    if (!g_ocl_ok)
        return AEG_EBACKEND;

    o = (ocl_ctx *)calloc(1, sizeof(*o));
    if (!o)
        return AEG_ENOMEM;
    ndev = ocl_devices(plats, devs);
    if (want >= 0) {
        pick = want < ndev ? want : -1;
    } else {
        for (k = 0; k < ndev && pick < 0; k++)
            if (ocl_dev_is_gpu(devs[k]))
                pick = k;
        if (pick < 0 && !gpu_only && ndev > 0)
            pick = 0;
    }
    if (pick < 0) {
        ocl_destroy(o);
        return AEG_EBACKEND;
    }
    o->platform = plats[pick];
    o->device = devs[pick];
    o->index = pick;
    p_clGetDeviceInfo(o->device, CL_DEVICE_NAME, sizeof(o->name), o->name, &got);
    p_clGetDeviceInfo(o->device, CL_DEVICE_VENDOR, sizeof(o->vendor), o->vendor, &got);
    p_clGetDeviceInfo(o->device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(o->mem), &o->mem, NULL);
    p_clGetDeviceInfo(o->device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(o->cus), &o->cus, NULL);
    o->is_gpu = ocl_dev_is_gpu(o->device);
    o->no_clone = ocl_pocl_unsafe_queues(o->device);

    o->ctx = p_clCreateContext(NULL, 1, &o->device, NULL, NULL, &st);
    if (!o->ctx || st != CL_SUCCESS) {
        ocl_destroy(o);
        return AEG_EBACKEND;
    }
    o->queue = p_clCreateCommandQueue(o->ctx, o->device, 0, &st);
    if (!o->queue) {
        ocl_destroy(o);
        return AEG_EBACKEND;
    }
    {
        const char *src = aeg_kernel_src;
        size_t len = strlen(src);
        o->prog = p_clCreateProgramWithSource(o->ctx, 1, &src, &len, &st);
    }
    if (!o->prog || st != CL_SUCCESS) {
        ocl_destroy(o);
        return AEG_EBACKEND;
    }
    st = p_clBuildProgram(o->prog, 1, &o->device, "", NULL, NULL);
    if (st != CL_SUCCESS) {
        char log[2048];
        p_clGetProgramBuildInfo(o->prog, o->device, CL_PROGRAM_BUILD_LOG,
                                sizeof(log), log, NULL);
        aeg_log(AEG_LOG_WARNING, "aeg: OpenCL build failed: %s", log);
        ocl_destroy(o);
        return AEG_EBACKEND;
    }
    if (ocl_kernels(o) != AEG_OK) {
        ocl_destroy(o);
        return AEG_EBACKEND;
    }
    ocl_fill_ops(o, ops);
    return AEG_OK;
}

/* The devices by --device index (the list ends at the first one the kernels
 * do not build on). */
int aeg_backend_query_opencl(aeg_device_info *out, int max)
{
    aeg_be_ops ops;
    int n = 0, k;
    for (k = 0; k < OCL_MAXDEV && n < max; k++) {
        ocl_ctx *o;
        if (aeg_be_opencl_create(&ops, k, 0) != AEG_OK)
            break;
        o = (ocl_ctx *)ops.priv;
        memset(&out[n], 0, sizeof(out[n]));
        out[n].backend = AEG_BACKEND_OPENCL;
        snprintf(out[n].name, sizeof(out[n].name), "%s", o->name);
        snprintf(out[n].vendor, sizeof(out[n].vendor), "%s", o->vendor);
        out[n].is_gpu = o->is_gpu;
        out[n].global_mem = o->mem;
        out[n].max_compute_units = o->cus;
        out[n].device_index = k;
        n++;
        if (ops.destroy)
            ops.destroy(ops.priv);
    }
    return n;
}
