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
typedef struct { size_t origin; size_t size; } cl_buffer_region;

#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_ALL 0xFFFFFFFF
#define CL_MEM_READ_ONLY (1 << 2)
#define CL_MEM_WRITE_ONLY (1 << 1)
#define CL_MEM_READ_WRITE (1 << 0)
#define CL_TRUE 1
#define CL_FALSE 0
#define CL_PROGRAM_BUILD_LOG 0x1183
#define CL_BUFFER_CREATE_TYPE_REGION 0x1220
#define CL_DEVICE_TYPE 0x1000
#define CL_DEVICE_TYPE_GPU (1 << 2)
#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_VENDOR 0x102C
#define CL_DRIVER_VERSION 0x102D
#define CL_DEVICE_VERSION 0x102F
#define CL_DEVICE_GLOBAL_MEM_SIZE 0x101F
#define CL_DEVICE_MAX_COMPUTE_UNITS 0x1002
#define CL_DEVICE_MAX_MEM_ALLOC_SIZE 0x1010

/* The OpenCL entry points are resolved at run time so that the encoder keeps
 * building (and running on the CPU) on machines without an OpenCL loader. */
static void ocl_load_error(const char *name)
{
    aeg_log(AEG_LOG_DEBUG, "aeg: OpenCL loader is missing %s", name);
}

#define OCL_SYMS(X) \
    X(cl_int, clGetPlatformIDs, (cl_uint, cl_platform_id *, cl_uint *)) \
    X(cl_int, clGetDeviceIDs, (cl_platform_id, cl_uint, cl_uint, cl_device_id *, cl_uint *)) \
    X(cl_int, clGetDeviceInfo, (cl_device_id, cl_uint, size_t, void *, size_t *)) \
    X(cl_context, clCreateContext, (cl_context_properties *, cl_uint, cl_device_id *, void *, void *, cl_int *)) \
    X(cl_command_queue, clCreateCommandQueue, (cl_context, cl_device_id, cl_ulong, cl_int *)) \
    X(cl_program, clCreateProgramWithSource, (cl_context, cl_uint, const char **, const size_t *, cl_int *)) \
    X(cl_int, clBuildProgram, (cl_program, cl_uint, const cl_device_id *, const char *, void *, void *)) \
    X(cl_int, clGetProgramBuildInfo, (cl_program, cl_device_id, cl_uint, size_t, void *, size_t *)) \
    X(cl_kernel, clCreateKernel, (cl_program, const char *, cl_int *)) \
    X(cl_int, clSetKernelArg, (cl_kernel, cl_uint, size_t, const void *)) \
    X(cl_int, clEnqueueNDRangeKernel, (cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const void *, void *)) \
    X(cl_int, clEnqueueWriteBuffer, (cl_command_queue, cl_mem, cl_uint, size_t, size_t, const void *, cl_uint, const void *, void *)) \
    X(cl_int, clEnqueueReadBuffer, (cl_command_queue, cl_mem, cl_uint, size_t, size_t, void *, cl_uint, const void *, void *)) \
    X(cl_mem, clCreateBuffer, (cl_context, cl_ulong, size_t, void *, cl_int *)) \
    X(cl_mem, clCreateSubBuffer, (cl_mem, cl_ulong, cl_uint, const void *, cl_int *)) \
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

#define NBUF 32

typedef struct ocl_ctx {
    void *lib;
    cl_platform_id platform;
    cl_device_id device;
    cl_context ctx;
    cl_command_queue queue;
    cl_program prog;
    cl_kernel k_sad, k_tq, k_iq, k_tqiq, k_sse, k_intp, k_satd, k_tqdc, k_iqdc;
    cl_kernel k_me;
    cl_mem buf[NBUF];
    size_t buf_sz[NBUF];
    int is_gpu;
    /* the registered reference picture (see ref_upload) */
    int ref_stride, ref_w, ref_h, ref_valid;
    /* device-resident luma pictures for me_batch (see aeg_be_ops) */
    cl_mem mepic[AEG_ME_PIC_SLOTS];
    size_t mepic_sz[AEG_ME_PIC_SLOTS];
    uint64_t mepic_id[AEG_ME_PIC_SLOTS];
    int mepic_ok[AEG_ME_PIC_SLOTS];
    /* lowres ring (M-3.3a) */
    cl_mem lo_ring;
    size_t lo_ring_sz, lo_slot_bytes;
    cl_mem lo_work[4];
    size_t lo_work_sz[4];
    cl_kernel k_lowres;
    int lo_stride, lo_w, lo_h;
    uint64_t lo_id[AEG_LOWRES_SLOTS];
    int lo_ok[AEG_LOWRES_SLOTS];
    char name[96];
    char vendor[48];
    uint64_t mem;
    int cus;
    int16_t *pos;               /* sad_batch: the rebased positions */
    size_t pos_cap;
    int no_clone;               /* the runtime cannot take several queues */
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

/* Whole-picture integer motion search (see aeg_be_ops.me_batch): one work item
 * per block, the reference and the source transferred whole. */
static int ocl_me_pic_upload(void *priv, int slot, const uint8_t *plane,
                             int stride, int w, int h, int rows, uint64_t id)
{
    ocl_ctx *o = (ocl_ctx *)priv;
    cl_mem m;
    size_t bytes;
    (void)w; (void)h;
    if (slot < 0 || slot >= AEG_ME_PIC_SLOTS)
        return AEG_EINVAL;
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
    m = ocl_buf(o, 8 + slot, bytes, CL_MEM_READ_ONLY);
    if (!m)
        return AEG_EBACKEND;
    if (p_clEnqueueWriteBuffer(o->queue, m, CL_TRUE, 0, bytes,
                               plane - (ptrdiff_t)AEG_PAD * stride - AEG_PAD,
                               0, NULL, NULL) != CL_SUCCESS)
        return AEG_EBACKEND;
    o->mepic[slot] = m;
    o->mepic_sz[slot] = bytes;
    o->mepic_id[slot] = id;
    o->mepic_ok[slot] = 1;
    return AEG_OK;
}

static int ocl_lowres_upload(void *priv, int slot, const uint8_t *plane,
                             int stride, int w, int h, uint64_t id)
{
    ocl_ctx *o = (ocl_ctx *)priv;
    size_t bytes;
    cl_int st;
    int k;
    if (slot < 0 || slot >= AEG_LOWRES_SLOTS)
        return AEG_EINVAL;
    bytes = (size_t)stride * (size_t)(h + 2 * AEG_LA_PAD);
    if (bytes != o->lo_slot_bytes || stride != o->lo_stride) {
        if (o->lo_ring)
            p_clReleaseMemObject(o->lo_ring);
        o->lo_ring = p_clCreateBuffer(o->ctx, CL_MEM_READ_ONLY,
                                      bytes * AEG_LOWRES_SLOTS, NULL, &st);
        o->lo_ring_sz = o->lo_ring ? bytes * AEG_LOWRES_SLOTS : 0;
        o->lo_slot_bytes = o->lo_ring ? bytes : 0;
        o->lo_stride = stride;
        o->lo_w = w;
        o->lo_h = h;
        for (k = 0; k < AEG_LOWRES_SLOTS; k++)
            o->lo_ok[k] = 0;
        if (!o->lo_ring)
            return AEG_EBACKEND;
    }
    if (o->lo_ok[slot] && o->lo_id[slot] == id)
        return AEG_OK;
    st = p_clEnqueueWriteBuffer(o->queue, o->lo_ring, CL_TRUE,
                                (size_t)slot * o->lo_slot_bytes, bytes,
                                plane - (ptrdiff_t)AEG_LA_PAD * stride - AEG_LA_PAD,
                                0, NULL, NULL);
    if (st != CL_SUCCESS)
        return AEG_EBACKEND;
    o->lo_id[slot] = id;
    o->lo_ok[slot] = 1;
    return AEG_OK;
}

static int ocl_lowres_sad_batch(void *priv, int cur_slot, int ref_slot, int bw,
                                int bh, const int16_t *mv, int count,
                                int32_t *out)
{
    ocl_ctx *o = (ocl_ctx *)priv;
    cl_mem mmv, mout, mcur, mref;
    cl_buffer_region reg;
    size_t nthreads = (size_t)count;
    cl_int st = CL_SUCCESS;
    int16_t *pos = NULL;
    int i;
    if (cur_slot < 0 || cur_slot >= AEG_LOWRES_SLOTS ||
        ref_slot < 0 || ref_slot >= AEG_LOWRES_SLOTS ||
        !o->lo_ok[cur_slot] || !o->lo_ok[ref_slot])
        return AEG_EINVAL;
    if (count <= 0)
        return AEG_OK;
    mmv = ocl_buf(o, 2, (size_t)count * 8, CL_MEM_READ_ONLY);
    mout = ocl_buf(o, 3, (size_t)count * 4, CL_MEM_WRITE_ONLY);
    if (!mmv || !mout)
        return AEG_EBACKEND;
    reg.origin = (size_t)cur_slot * o->lo_slot_bytes;
    reg.size = o->lo_slot_bytes;
    mcur = p_clCreateSubBuffer(o->lo_ring, CL_MEM_READ_ONLY,
                               CL_BUFFER_CREATE_TYPE_REGION, &reg, &st);
    reg.origin = (size_t)ref_slot * o->lo_slot_bytes;
    mref = p_clCreateSubBuffer(o->lo_ring, CL_MEM_READ_ONLY,
                               CL_BUFFER_CREATE_TYPE_REGION, &reg, &st);
    if (!mcur || !mref) {
        if (mcur) p_clReleaseMemObject(mcur);
        if (mref) p_clReleaseMemObject(mref);
        return AEG_EBACKEND;
    }
    /* the resident buffer starts at the padding: shift the positions by it
     * (CUDA offsets the pointer instead) */
    pos = (int16_t *)malloc((size_t)count * 8);
    if (!pos)
        return AEG_ENOMEM;
    for (i = 0; i < count; i++) {
        pos[4 * i + 0] = (int16_t)(mv[4 * i + 0] + AEG_LA_PAD);
        pos[4 * i + 1] = (int16_t)(mv[4 * i + 1] + AEG_LA_PAD);
        pos[4 * i + 2] = (int16_t)(mv[4 * i + 2] + AEG_LA_PAD);
        pos[4 * i + 3] = (int16_t)(mv[4 * i + 3] + AEG_LA_PAD);
    }
    st |= p_clEnqueueWriteBuffer(o->queue, mmv, CL_FALSE, 0, (size_t)count * 8,
                                 pos, 0, NULL, NULL);
    p_clSetKernelArg(o->k_sad, 0, sizeof(cl_mem), &mcur);
    p_clSetKernelArg(o->k_sad, 1, sizeof(int), &o->lo_stride);
    p_clSetKernelArg(o->k_sad, 2, sizeof(cl_mem), &mref);
    p_clSetKernelArg(o->k_sad, 3, sizeof(int), &o->lo_stride);
    p_clSetKernelArg(o->k_sad, 4, sizeof(int), &bw);
    p_clSetKernelArg(o->k_sad, 5, sizeof(int), &bh);
    p_clSetKernelArg(o->k_sad, 6, sizeof(int), &count);
    p_clSetKernelArg(o->k_sad, 7, sizeof(cl_mem), &mmv);
    p_clSetKernelArg(o->k_sad, 8, sizeof(cl_mem), &mout);
    st = p_clEnqueueNDRangeKernel(o->queue, o->k_sad, 1, NULL, &nthreads, NULL,
                                  0, NULL, NULL);
    if (st != CL_SUCCESS) {
        free(pos);
        p_clReleaseMemObject(mcur);
        p_clReleaseMemObject(mref);
        return AEG_EBACKEND;
    }
    st |= p_clEnqueueReadBuffer(o->queue, mout, CL_FALSE, 0, (size_t)count * 4,
                                out, 0, NULL, NULL);
    if (st == CL_SUCCESS)
        st = p_clFinish(o->queue);
    free(pos);
    p_clReleaseMemObject(mcur);
    p_clReleaseMemObject(mref);
    return st == CL_SUCCESS ? AEG_OK : AEG_EBACKEND;
}

static cl_mem ocl_lo_work(ocl_ctx *o, int slot, size_t bytes)
{
    cl_int st;
    if (o->lo_work[slot] && o->lo_work_sz[slot] >= bytes)
        return o->lo_work[slot];
    if (o->lo_work[slot])
        p_clReleaseMemObject(o->lo_work[slot]);
    o->lo_work[slot] = p_clCreateBuffer(o->ctx, CL_MEM_READ_WRITE, bytes, NULL, &st);
    o->lo_work_sz[slot] = o->lo_work[slot] ? bytes : 0;
    return o->lo_work[slot];
}

static int ocl_lowres_batch_search(void *priv, const int32_t *pairs, int npairs,
                                   int mb_w, int mb_h, int iters,
                                   int32_t *mv, int32_t *cost)
{
    static const int steps[] = { 16, 16, 8, 8, 4, 4, 2, 2, 1, 1 };
    ocl_ctx *o = (ocl_ctx *)priv;
    cl_mem mpairs, mprev, mnext, mcost;
    int32_t *zero = NULL;
    int nblk = mb_w * mb_h, n = npairs * nblk, it, istep = 0, ilast = 0;
    int slot_bytes = (int)o->lo_slot_bytes;
    size_t nthreads = (size_t)n;
    cl_int st = CL_SUCCESS;
    if (!o->lo_ring || npairs <= 0 || nblk <= 0)
        return AEG_EINVAL;
    if (iters <= 0 || iters > (int)(sizeof(steps) / sizeof(steps[0])))
        iters = 5;
    mpairs = ocl_lo_work(o, 0, (size_t)npairs * 8);
    mprev = ocl_lo_work(o, 1, (size_t)n * 8);
    mnext = ocl_lo_work(o, 2, (size_t)n * 8);
    mcost = ocl_lo_work(o, 3, (size_t)n * 4);
    if (!mpairs || !mprev || !mnext || !mcost)
        return AEG_EBACKEND;
    zero = (int32_t *)calloc((size_t)n * 2, sizeof(int32_t));
    if (!zero)
        return AEG_ENOMEM;
    st |= p_clEnqueueWriteBuffer(o->queue, mpairs, CL_FALSE, 0,
                                 (size_t)npairs * 8, pairs, 0, NULL, NULL);
    st |= p_clEnqueueWriteBuffer(o->queue, mprev, CL_FALSE, 0, (size_t)n * 8,
                                 zero, 0, NULL, NULL);
    free(zero);
    if (st != CL_SUCCESS)
        return AEG_EBACKEND;
    p_clSetKernelArg(o->k_lowres, 0, sizeof(cl_mem), &o->lo_ring);
    p_clSetKernelArg(o->k_lowres, 1, sizeof(int), &o->lo_stride);
    p_clSetKernelArg(o->k_lowres, 2, sizeof(int), &slot_bytes);
    p_clSetKernelArg(o->k_lowres, 3, sizeof(int), &mb_w);
    p_clSetKernelArg(o->k_lowres, 4, sizeof(int), &mb_h);
    p_clSetKernelArg(o->k_lowres, 6, sizeof(int), &ilast);
    p_clSetKernelArg(o->k_lowres, 7, sizeof(cl_mem), &mpairs);
    p_clSetKernelArg(o->k_lowres, 8, sizeof(int), &npairs);
    p_clSetKernelArg(o->k_lowres, 11, sizeof(cl_mem), &mcost);
    for (it = 0; it < iters; it++) {
        cl_mem tmp;
        istep = steps[it];
        ilast = it == iters - 1;
        p_clSetKernelArg(o->k_lowres, 5, sizeof(int), &istep);
        p_clSetKernelArg(o->k_lowres, 6, sizeof(int), &ilast);
        p_clSetKernelArg(o->k_lowres, 9, sizeof(cl_mem), &mprev);
        p_clSetKernelArg(o->k_lowres, 10, sizeof(cl_mem), &mnext);
        st = p_clEnqueueNDRangeKernel(o->queue, o->k_lowres, 1, NULL, &nthreads,
                                      NULL, 0, NULL, NULL);
        if (st != CL_SUCCESS)
            return AEG_EBACKEND;
        tmp = mprev; mprev = mnext; mnext = tmp;
    }
    st |= p_clEnqueueReadBuffer(o->queue, mprev, CL_FALSE, 0, (size_t)n * 8,
                                mv, 0, NULL, NULL);
    st |= p_clEnqueueReadBuffer(o->queue, mcost, CL_FALSE, 0, (size_t)n * 4,
                                cost, 0, NULL, NULL);
    if (st == CL_SUCCESS)
        st = p_clFinish(o->queue);
    return st == CL_SUCCESS ? AEG_OK : AEG_EBACKEND;
}

static int ocl_mepic_find(ocl_ctx *o, uint64_t id, size_t bytes)
{
    int i;
    for (i = 0; i < AEG_ME_PIC_SLOTS; i++)
        if (o->mepic_ok[i] && o->mepic_id[i] == id && o->mepic_sz[i] >= bytes)
            return i;
    return -1;
}

static int ocl_rec_upload(void *priv, int slot, const uint8_t *y,
                          const uint8_t *u, const uint8_t *v, int ys,
                          int cs, int w, int h, uint64_t id)
{
    ocl_ctx *o = (ocl_ctx *)priv;
    cl_mem mu, mv;
    size_t cbytes;
    int rows = (h + 1) / 2 + 2 * AEG_PAD;
    int st;
    if (slot < 2 || slot >= AEG_ME_PIC_SLOTS)
        return AEG_EINVAL;
    st = ocl_me_pic_upload(o, slot, y, ys, w, h, h + 2 * AEG_PAD, id);
    if (st != AEG_OK)
        return st;
    cbytes = (size_t)cs * (size_t)rows;
    mu = ocl_buf(o, 24 + slot * 2, cbytes, CL_MEM_READ_ONLY);
    mv = ocl_buf(o, 25 + slot * 2, cbytes, CL_MEM_READ_ONLY);
    if (!mu || !mv)
        return AEG_EBACKEND;
    if (p_clEnqueueWriteBuffer(o->queue, mu, CL_TRUE, 0, cbytes,
                               u - (ptrdiff_t)AEG_PAD * cs - AEG_PAD,
                               0, NULL, NULL) != CL_SUCCESS ||
        p_clEnqueueWriteBuffer(o->queue, mv, CL_TRUE, 0, cbytes,
                               v - (ptrdiff_t)AEG_PAD * cs - AEG_PAD,
                               0, NULL, NULL) != CL_SUCCESS)
        return AEG_EBACKEND;
    return AEG_OK;
}

static int ocl_me_batch(void *priv, const uint8_t *cur, int cstride,
                        const uint8_t *ref, int rstride, int bw, int bh,
                        int range, const int16_t *job, int count,
                        int avail_rows, int16_t *out_mv, int32_t *out_sad,
                        uint64_t cur_id, uint64_t ref_id)
{
    ocl_ctx *o = (ocl_ctx *)priv;
    cl_mem mcur, mref, mjob, mmv, msad;
    size_t nthreads, cbytes, rbytes;
    int hi, pad = AEG_PAD;
    cl_int st = CL_SUCCESS;

    if (count <= 0)
        return AEG_OK;
    /* the job list gives the picture geometry its blocks span */
    hi = 0;
    {
        int i;
        for (i = 0; i < count; i++)
            if (job[4 * i + 1] > hi)
                hi = job[4 * i + 1];
    }
    hi = hi + bh + 2 * AEG_PAD;
    if (avail_rows > 0 && hi > avail_rows)
        hi = avail_rows;
    cbytes = (size_t)cstride * (size_t)hi;
    rbytes = cbytes;
    nthreads = (size_t)count;
    {
        int ci = ocl_mepic_find(o, cur_id, cbytes);
        int ri = ocl_mepic_find(o, ref_id, rbytes);
        mcur = ci >= 0 ? o->mepic[ci]
                       : ocl_buf(o, 0, cbytes, CL_MEM_READ_ONLY);
        mref = ri >= 0 ? o->mepic[ri]
                       : ocl_buf(o, 1, rbytes, CL_MEM_READ_ONLY);
        if (!mcur || !mref)
            return AEG_EBACKEND;
        if (ci < 0)
            st |= p_clEnqueueWriteBuffer(o->queue, mcur, CL_FALSE, 0, cbytes,
                                         cur - (ptrdiff_t)AEG_PAD * cstride - AEG_PAD,
                                         0, NULL, NULL);
        if (ri < 0)
            st |= p_clEnqueueWriteBuffer(o->queue, mref, CL_FALSE, 0, rbytes,
                                         ref - (ptrdiff_t)AEG_PAD * rstride - AEG_PAD,
                                         0, NULL, NULL);
    }
    mjob = ocl_buf(o, 2, (size_t)count * 8, CL_MEM_READ_ONLY);
    mmv = ocl_buf(o, 3, (size_t)count * 4, CL_MEM_WRITE_ONLY);
    msad = ocl_buf(o, 4, (size_t)count * 4, CL_MEM_WRITE_ONLY);
    if (!mjob || !mmv || !msad)
        return AEG_EBACKEND;
    st |= p_clEnqueueWriteBuffer(o->queue, mjob, CL_FALSE, 0, (size_t)count * 8,
                                 job, 0, NULL, NULL);
    if (st != CL_SUCCESS)
        return AEG_EBACKEND;
    p_clSetKernelArg(o->k_me, 0, sizeof(cl_mem), &mcur);
    p_clSetKernelArg(o->k_me, 1, sizeof(int), &cstride);
    p_clSetKernelArg(o->k_me, 2, sizeof(cl_mem), &mref);
    p_clSetKernelArg(o->k_me, 3, sizeof(int), &rstride);
    p_clSetKernelArg(o->k_me, 4, sizeof(int), &pad);
    p_clSetKernelArg(o->k_me, 5, sizeof(int), &bw);
    p_clSetKernelArg(o->k_me, 6, sizeof(int), &bh);
    p_clSetKernelArg(o->k_me, 7, sizeof(int), &range);
    p_clSetKernelArg(o->k_me, 8, sizeof(cl_mem), &mjob);
    p_clSetKernelArg(o->k_me, 9, sizeof(int), &count);
    p_clSetKernelArg(o->k_me, 10, sizeof(cl_mem), &mmv);
    p_clSetKernelArg(o->k_me, 11, sizeof(cl_mem), &msad);
    st = p_clEnqueueNDRangeKernel(o->queue, o->k_me, 1, NULL, &nthreads, NULL,
                                  0, NULL, NULL);
    if (st != CL_SUCCESS)
        return AEG_EBACKEND;
    st |= p_clEnqueueReadBuffer(o->queue, mmv, CL_FALSE, 0, (size_t)count * 4,
                                out_mv, 0, NULL, NULL);
    st |= p_clEnqueueReadBuffer(o->queue, msad, CL_FALSE, 0, (size_t)count * 4,
                                out_sad, 0, NULL, NULL);
    if (st == CL_SUCCESS)
        st = p_clFinish(o->queue);
    return st == CL_SUCCESS ? AEG_OK : AEG_EBACKEND;
}

static int ocl_tq_batch(void *priv, const int16_t *res, int nblk, int qp,
                        int intra, int16_t *level, int32_t *nz)
{
    ocl_ctx *o = (ocl_ctx *)priv;
    cl_mem mres, mlev, mnz;
    size_t bytes = (size_t)nblk * 16 * 2;
    size_t nthreads = (size_t)nblk;
    int qbits = 15 + qp / 6;
    int f = aeg_quant_bias(qbits, intra ? 3 : 6);
    int mf0 = aeg_quant_mf[qp % 6][0], mf1 = aeg_quant_mf[qp % 6][1],
        mf2 = aeg_quant_mf[qp % 6][2];
    cl_int st;

    if (nblk <= 0)
        return AEG_OK;
    mres = ocl_buf(o, 0, bytes, CL_MEM_READ_ONLY);
    mlev = ocl_buf(o, 1, bytes, CL_MEM_WRITE_ONLY);
    mnz = ocl_buf(o, 2, (size_t)nblk * 4, CL_MEM_WRITE_ONLY);
    if (!mres || !mlev || !mnz)
        return AEG_EBACKEND;
    p_clEnqueueWriteBuffer(o->queue, mres, CL_TRUE, 0, bytes, res, 0, NULL, NULL);
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
    p_clEnqueueReadBuffer(o->queue, mlev, CL_TRUE, 0, bytes, level, 0, NULL, NULL);
    if (nz)
        p_clEnqueueReadBuffer(o->queue, mnz, CL_TRUE, 0, (size_t)nblk * 4, nz,
                              0, NULL, NULL);
    return AEG_OK;
}

static int ocl_iq_idct_batch(void *priv, const int16_t *level, int nblk, int qp,
                             int16_t *res)
{
    ocl_ctx *o = (ocl_ctx *)priv;
    cl_mem mlev, mres;
    size_t bytes = (size_t)nblk * 16 * 2;
    size_t nthreads = (size_t)nblk;
    int sh = qp / 6;
    int ls0 = aeg_dequant_ls[qp % 6][0], ls1 = aeg_dequant_ls[qp % 6][1],
        ls2 = aeg_dequant_ls[qp % 6][2];
    cl_int st;

    if (nblk <= 0)
        return AEG_OK;
    mlev = ocl_buf(o, 0, bytes, CL_MEM_READ_ONLY);
    mres = ocl_buf(o, 1, bytes, CL_MEM_WRITE_ONLY);
    if (!mlev || !mres)
        return AEG_EBACKEND;
    p_clEnqueueWriteBuffer(o->queue, mlev, CL_TRUE, 0, bytes, level, 0, NULL, NULL);
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
    p_clEnqueueReadBuffer(o->queue, mres, CL_TRUE, 0, bytes, res, 0, NULL, NULL);
    return AEG_OK;
}

static int ocl_tq_iq_batch(void *priv, const int16_t *res, int nblk, int qp,
                           int intra, int16_t *level, int32_t *nz,
                           int16_t *rec)
{
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
    ocl_ctx *o = (ocl_ctx *)priv;
    cl_mem ma, mb, mout;
    size_t tw = (size_t)w / 8, th = (size_t)h / 8, count = tw * th;
    size_t abytes = (size_t)sa * h;
    uint64_t *tmp;
    size_t i, acc = 0;
    cl_int st;

    *out = 0;
    if (count == 0 || !w || !h)
        return AEG_OK;
    ma = ocl_buf(o, 0, abytes, CL_MEM_READ_ONLY);
    mb = ocl_buf(o, 1, (size_t)sb * h, CL_MEM_READ_ONLY);
    mout = ocl_buf(o, 2, count * 8, CL_MEM_WRITE_ONLY);
    if (!ma || !mb || !mout)
        return AEG_EBACKEND;
    p_clEnqueueWriteBuffer(o->queue, ma, CL_TRUE, 0, abytes, a, 0, NULL, NULL);
    p_clEnqueueWriteBuffer(o->queue, mb, CL_TRUE, 0, (size_t)sb * h, b, 0, NULL, NULL);
    if (!count)
        return AEG_OK;
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
    p_clEnqueueReadBuffer(o->queue, mout, CL_TRUE, 0, count * 8, tmp, 0, NULL, NULL);
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
        p_clEnqueueWriteBuffer(o->queue, m, CL_TRUE, 0, bytes,
                               planes[k] - (ptrdiff_t)AEG_PAD * pstride - AEG_PAD,
                               0, NULL, NULL);
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
    p_clEnqueueWriteBuffer(o->queue, mcur, CL_TRUE, 0, cbytes,
                           cur - (ptrdiff_t)AEG_PAD * cstride - AEG_PAD,
                           0, NULL, NULL);
    p_clEnqueueWriteBuffer(o->queue, mpos, CL_TRUE, 0, (size_t)count * 8, pos,
                           0, NULL, NULL);
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
    p_clEnqueueReadBuffer(o->queue, mout, CL_TRUE, 0, (size_t)count * 4, out,
                          0, NULL, NULL);
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
    if (o->lo_ring)
        p_clReleaseMemObject(o->lo_ring);
    for (i = 0; i < 4; i++)
        if (o->lo_work[i])
            p_clReleaseMemObject(o->lo_work[i]);
    free(o->pos);
    if (o->k_sad) p_clReleaseKernel(o->k_sad);
    if (o->k_me) p_clReleaseKernel(o->k_me);
    if (o->k_lowres) p_clReleaseKernel(o->k_lowres);
    if (o->k_tq) p_clReleaseKernel(o->k_tq);
    if (o->k_iq) p_clReleaseKernel(o->k_iq);
    if (o->k_tqiq) p_clReleaseKernel(o->k_tqiq);
    if (o->k_sse) p_clReleaseKernel(o->k_sse);
    if (o->k_intp) p_clReleaseKernel(o->k_intp);
    if (o->k_satd) p_clReleaseKernel(o->k_satd);
    if (o->k_tqdc) p_clReleaseKernel(o->k_tqdc);
    if (o->k_iqdc) p_clReleaseKernel(o->k_iqdc);
    if (o->prog) p_clReleaseProgram(o->prog);
    if (o->queue) p_clReleaseCommandQueue(o->queue);
    if (o->ctx) p_clReleaseContext(o->ctx);
    if (o->lib) dlclose(o->lib);
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
    o->k_me = p_clCreateKernel(o->prog, "me_search", &st);
    o->k_lowres = p_clCreateKernel(o->prog, "lowres_batch", &st);
    o->k_tq = p_clCreateKernel(o->prog, "tq_batch", &st);
    o->k_iq = p_clCreateKernel(o->prog, "iq_idct_batch", &st);
    o->k_tqiq = p_clCreateKernel(o->prog, "tq_iq_batch", &st);
    o->k_sse = p_clCreateKernel(o->prog, "sse_batch", &st);
    o->k_intp = p_clCreateKernel(o->prog, "interp_plane", &st);
    o->k_satd = p_clCreateKernel(o->prog, "satd_batch", &st);
    o->k_tqdc = p_clCreateKernel(o->prog, "tq_dc_batch", &st);
    o->k_iqdc = p_clCreateKernel(o->prog, "iq_dc_batch", &st);
    return o->k_sad && o->k_tq && o->k_iq && o->k_tqiq && o->k_sse &&
           o->k_intp && o->k_satd && o->k_tqdc && o->k_iqdc && o->k_me &&
           o->k_lowres ? AEG_OK : AEG_EBACKEND;
}

static int ocl_clone(void *priv, aeg_be_ops *out);

static void ocl_fill_ops(ocl_ctx *o, aeg_be_ops *ops)
{
    memset(ops, 0, sizeof(*ops));
    ops->priv = o;
    ops->backend = AEG_BACKEND_OPENCL;
    ops->is_gpu = o->is_gpu;
    ops->sad_batch = ocl_sad_batch;
    ops->me_batch = ocl_me_batch;
    ops->me_pic_upload = ocl_me_pic_upload;
    ops->rec_upload = ocl_rec_upload;
    ops->lowres_upload = ocl_lowres_upload;
    ops->lowres_sad_batch = ocl_lowres_sad_batch;
    ops->lowres_batch_search = ocl_lowres_batch_search;
    ops->tq_batch = ocl_tq_batch;
    ops->iq_idct_batch = ocl_iq_idct_batch;
    ops->tq_iq_batch = ocl_tq_iq_batch;
    ops->tq_dc_batch = ocl_tq_dc_batch;
    ops->iq_dc_batch = ocl_iq_dc_batch;
    ops->sse_batch = ocl_sse_batch;
    ops->interp_plane = ocl_interp_plane;
    ops->ref_upload = ocl_ref_upload;
    ops->satd_batch = ocl_satd_batch;
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

int aeg_be_opencl_create(aeg_be_ops *ops)
{
    ocl_ctx *o;
    void *lib = NULL;
    cl_int st;
    size_t got;

#define LOAD(ret, name, args) do { \
        *(void **)(&p_##name) = dlsym(lib, #name); \
        if (!p_##name) { ocl_load_error(#name); dlclose(lib); return AEG_EBACKEND; } \
    } while (0);

    if (ocl_load(&lib) != AEG_OK)
        return AEG_EBACKEND;
    OCL_SYMS(LOAD)
#undef LOAD

    o = (ocl_ctx *)calloc(1, sizeof(*o));
    if (!o) {
        dlclose(lib);
        return AEG_ENOMEM;
    }
    o->lib = lib;
    /* AEG_DEVICE=n picks the n-th OpenCL device in platform/device order
     * (--device n on the command line sets it).  Without it the first is
     * used, as before, so no existing invocation changes behaviour. */
    {
        cl_platform_id plats[16];
        cl_device_id devs[64];
        cl_uint np = 0, pi, di;
        int idx = 0, want = -1, found = 0;
        const char *ev = getenv("AEG_DEVICE");
        if (ev && *ev)
            want = atoi(ev);
        if (p_clGetPlatformIDs(16, plats, &np) != CL_SUCCESS || !np) {
            ocl_destroy(o);
            return AEG_EBACKEND;
        }
        for (pi = 0; pi < np && !found; pi++) {
            cl_uint nd = 0;
            if (p_clGetDeviceIDs(plats[pi], CL_DEVICE_TYPE_ALL, 64, devs,
                                 &nd) != CL_SUCCESS)
                continue;
            for (di = 0; di < nd; di++, idx++) {
                if (want < 0 || idx == want) {
                    o->platform = plats[pi];
                    o->device = devs[di];
                    found = 1;
                    break;
                }
            }
        }
        if (!found) {
            /* the requested index does not exist: the first device is a
             * better answer than failing, and the log says why */
            if (want >= 0)
                aeg_log(AEG_LOG_WARNING, "aeg: OpenCL device %d not found, "
                        "using device 0", want);
            if (p_clGetPlatformIDs(1, &o->platform, NULL) != CL_SUCCESS ||
                p_clGetDeviceIDs(o->platform, CL_DEVICE_TYPE_ALL, 1, &o->device,
                                 NULL) != CL_SUCCESS) {
                ocl_destroy(o);
                return AEG_EBACKEND;
            }
        }
    }
    p_clGetDeviceInfo(o->device, CL_DEVICE_NAME, sizeof(o->name), o->name, &got);
    p_clGetDeviceInfo(o->device, CL_DEVICE_VENDOR, sizeof(o->vendor), o->vendor, &got);
    p_clGetDeviceInfo(o->device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(o->mem), &o->mem, NULL);
    p_clGetDeviceInfo(o->device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(o->cus), &o->cus, NULL);
    {
        cl_uint t = 0;
        p_clGetDeviceInfo(o->device, CL_DEVICE_TYPE, sizeof(t), &t, NULL);
        o->is_gpu = (t & CL_DEVICE_TYPE_GPU) != 0;
        if (!o->is_gpu && getenv("AEG_REC_DEV") &&
            atoi(getenv("AEG_REC_DEV"))) {
            aeg_log(AEG_LOG_WARNING, "aeg: OpenCL device is a CPU device "
                    "(CL_DEVICE_TYPE_CPU): chroma residency falls back to "
                    "luma-only");
        }
    }
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

int aeg_backend_query_opencl(aeg_device_info *out, int max)
{
    aeg_be_ops ops;
    int n = 0;
    if (max <= 0)
        return 0;
    if (aeg_be_opencl_create(&ops) == AEG_OK) {
        ocl_ctx *o = (ocl_ctx *)ops.priv;
        memset(&out[0], 0, sizeof(out[0]));
        out[0].backend = AEG_BACKEND_OPENCL;
        snprintf(out[0].name, sizeof(out[0].name), "%s", o->name);
        snprintf(out[0].vendor, sizeof(out[0].vendor), "%s", o->vendor);
        out[0].is_gpu = o->is_gpu;
        out[0].global_mem = o->mem;
        out[0].max_compute_units = o->cus;
        n = 1;
        if (ops.destroy)
            ops.destroy(ops.priv);
    }
    return n;
}
