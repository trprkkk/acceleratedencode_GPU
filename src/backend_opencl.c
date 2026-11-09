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
typedef void *cl_event;
typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef intptr_t cl_context_properties;

#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_ALL 0xFFFFFFFF
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
/* the queue this instance's work goes through: out of order, so the device may
 * overlap the upload of one operation with the kernel of the previous one
 * (the dependencies are passed as events, see ocl_write/ocl_launch/ocl_read) */
#define CL_QUEUE_OUT_OF_ORDER_MODE (1 << 0)
#define CL_QUEUE_PROFILING_ENABLE (1 << 1)

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
    X(cl_int, clWaitForEvents, (cl_uint, const void *)) \
    X(cl_int, clReleaseEvent, (void *)) \
    X(cl_int, clFinish, (cl_command_queue)) \
    X(cl_mem, clCreateBuffer, (cl_context, cl_ulong, size_t, void *, cl_int *)) \
    X(cl_int, clReleaseMemObject, (cl_mem)) \
    X(cl_int, clReleaseKernel, (cl_kernel)) \
    X(cl_int, clReleaseProgram, (cl_program)) \
    X(cl_int, clReleaseCommandQueue, (cl_command_queue)) \
    X(cl_int, clReleaseContext, (cl_context)) \
    X(cl_int, clRetainContext, (cl_context)) \
    X(cl_int, clRetainProgram, (cl_program))

#define DECL(ret, name, args) typedef ret (*fn_##name) args; static fn_##name p_##name;
OCL_SYMS(DECL)
#undef DECL

#define NBUF 12

typedef struct ocl_ctx {
    void *lib;
    cl_platform_id platform;
    cl_device_id device;
    cl_context ctx;
    cl_command_queue queue;
    cl_program prog;
    cl_kernel k_sad, k_tq, k_iq, k_tqiq, k_sse, k_intp, k_satd, k_tqdc, k_iqdc;
    cl_mem buf[NBUF];
    size_t buf_sz[NBUF];
    int is_gpu;
    /* the registered reference picture (see ref_upload) */
    int ref_stride, ref_w, ref_h, ref_valid;
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
/* The data flow                                                        */
/* ------------------------------------------------------------------ */
/* One host round trip per operation instead of one per transfer: the uploads
 * are enqueued without waiting, the kernel waits on their events, the reads
 * wait on the kernel's, and a single clFinish() brings the results back.  The
 * queue is out of order (each instance has its own), so the device can overlap
 * this operation's upload with the previous one's kernel on a real GPU; on a
 * runtime that refuses the flag the queue is in order and the events are
 * simply satisfied in the order they were enqueued.
 *
 * An operation that returns a failure code may leave events behind: the ones
 * it enqueued are still owned by the queue, and the next clFinish() releases
 * them, so nothing is leaked and nothing waits. */
#define OCL_EV 16

typedef struct {
    cl_event e[OCL_EV];
    int n;
} ocl_evlist;

static void ocl_ev_release(ocl_evlist *l)
{
    int i;
    for (i = 0; i < l->n; i++)
        if (l->e[i])
            p_clReleaseEvent(l->e[i]);
    l->n = 0;
}

/* The dependencies of the next enqueue are everything this operation has
 * emitted so far, and its own event goes after them.  When the list is full
 * the enqueue waits for all of it and the list restarts: everything emitted is
 * then complete, so the rest of the operation is still ordered after it. */
static cl_event *ocl_emit(ocl_evlist *l, int *nwait, const void **w)
{
    cl_event *out;

    *w = l->n ? (const void *)l->e : NULL;
    *nwait = l->n;
    if (l->n >= OCL_EV) {
        l->n = 0;
        return NULL;
    }
    out = &l->e[l->n];
    l->n++;
    return out;
}

static int ocl_write(ocl_ctx *o, cl_mem m, size_t off, size_t bytes,
                     const void *src, ocl_evlist *ev)
{
    int nw;
    const void *w;
    cl_event *out = ev ? ocl_emit(ev, &nw, &w) : NULL;
    if (p_clEnqueueWriteBuffer(o->queue, m, CL_FALSE, off, bytes, src, nw, w,
                               out) != CL_SUCCESS)
        return AEG_EBACKEND;
    return AEG_OK;
}

static int ocl_launch(ocl_ctx *o, cl_kernel k, size_t n, ocl_evlist *ev)
{
    int nw;
    const void *w;
    cl_event *out = ev ? ocl_emit(ev, &nw, &w) : NULL;
    if (p_clEnqueueNDRangeKernel(o->queue, k, 1, NULL, &n, NULL, nw, w, out) !=
        CL_SUCCESS)
        return AEG_EBACKEND;
    return AEG_OK;
}

static int ocl_read(ocl_ctx *o, cl_mem m, size_t off, size_t bytes, void *dst,
                    ocl_evlist *ev)
{
    int nw;
    const void *w;
    cl_event *out = ev ? ocl_emit(ev, &nw, &w) : NULL;
    if (p_clEnqueueReadBuffer(o->queue, m, CL_FALSE, off, bytes, dst, nw, w,
                              out) != CL_SUCCESS)
        return AEG_EBACKEND;
    return AEG_OK;
}

/* The one synchronisation of an operation: everything enqueued on this queue,
 * and nothing another instance enqueued on its own. */
static int ocl_wait(ocl_ctx *o, ocl_evlist *ev)
{
    cl_int st = p_clFinish(o->queue);
    ocl_ev_release(ev);
    return st == CL_SUCCESS ? AEG_OK : AEG_EBACKEND;
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
    ocl_evlist ev;

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
    ev.n = 0;
    if (ocl_write(o, mcur, 0, sp.cbytes, cur + sp.coff, &ev) != AEG_OK ||
        ocl_write(o, mref, 0, sp.rbytes, ref + sp.roff, &ev) != AEG_OK ||
        ocl_write(o, mmv, 0, (size_t)count * 8, o->pos, &ev) != AEG_OK)
        return ocl_wait(o, &ev);

    p_clSetKernelArg(o->k_sad, 0, sizeof(cl_mem), &mcur);
    p_clSetKernelArg(o->k_sad, 1, sizeof(int), &cstride);
    p_clSetKernelArg(o->k_sad, 2, sizeof(cl_mem), &mref);
    p_clSetKernelArg(o->k_sad, 3, sizeof(int), &rstride);
    p_clSetKernelArg(o->k_sad, 4, sizeof(int), &bw);
    p_clSetKernelArg(o->k_sad, 5, sizeof(int), &bh);
    p_clSetKernelArg(o->k_sad, 6, sizeof(int), &count);
    p_clSetKernelArg(o->k_sad, 7, sizeof(cl_mem), &mmv);
    p_clSetKernelArg(o->k_sad, 8, sizeof(cl_mem), &mout);
    if (ocl_launch(o, o->k_sad, nthreads, &ev) != AEG_OK)
        return ocl_wait(o, &ev);
    if (ocl_read(o, mout, 0, (size_t)count * 4, out, &ev) != AEG_OK)
        return ocl_wait(o, &ev);
    return ocl_wait(o, &ev);
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
    ocl_evlist ev;

    if (nblk <= 0)
        return AEG_OK;
    mres = ocl_buf(o, 0, bytes, CL_MEM_READ_ONLY);
    mlev = ocl_buf(o, 1, bytes, CL_MEM_WRITE_ONLY);
    mnz = ocl_buf(o, 2, (size_t)nblk * 4, CL_MEM_WRITE_ONLY);
    if (!mres || !mlev || !mnz)
        return AEG_EBACKEND;
    ev.n = 0;
    if (ocl_write(o, mres, 0, bytes, res, &ev) != AEG_OK)
        return ocl_wait(o, &ev);
    p_clSetKernelArg(o->k_tq, 0, sizeof(cl_mem), &mres);
    p_clSetKernelArg(o->k_tq, 1, sizeof(int), &nblk);
    p_clSetKernelArg(o->k_tq, 2, sizeof(int), &qbits);
    p_clSetKernelArg(o->k_tq, 3, sizeof(int), &f);
    p_clSetKernelArg(o->k_tq, 4, sizeof(int), &mf0);
    p_clSetKernelArg(o->k_tq, 5, sizeof(int), &mf1);
    p_clSetKernelArg(o->k_tq, 6, sizeof(int), &mf2);
    p_clSetKernelArg(o->k_tq, 7, sizeof(cl_mem), &mlev);
    p_clSetKernelArg(o->k_tq, 8, sizeof(cl_mem), &mnz);
    if (ocl_launch(o, o->k_tq, nthreads, &ev) != AEG_OK)
        return ocl_wait(o, &ev);
    if (ocl_read(o, mlev, 0, bytes, level, &ev) != AEG_OK ||
        (nz && ocl_read(o, mnz, 0, (size_t)nblk * 4, nz, &ev) != AEG_OK))
        return ocl_wait(o, &ev);
    return ocl_wait(o, &ev);
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
    ocl_evlist ev;

    if (nblk <= 0)
        return AEG_OK;
    mlev = ocl_buf(o, 0, bytes, CL_MEM_READ_ONLY);
    mres = ocl_buf(o, 1, bytes, CL_MEM_WRITE_ONLY);
    if (!mlev || !mres)
        return AEG_EBACKEND;
    ev.n = 0;
    if (ocl_write(o, mlev, 0, bytes, level, &ev) != AEG_OK)
        return ocl_wait(o, &ev);
    p_clSetKernelArg(o->k_iq, 0, sizeof(cl_mem), &mlev);
    p_clSetKernelArg(o->k_iq, 1, sizeof(int), &nblk);
    p_clSetKernelArg(o->k_iq, 2, sizeof(int), &sh);
    p_clSetKernelArg(o->k_iq, 3, sizeof(int), &ls0);
    p_clSetKernelArg(o->k_iq, 4, sizeof(int), &ls1);
    p_clSetKernelArg(o->k_iq, 5, sizeof(int), &ls2);
    p_clSetKernelArg(o->k_iq, 6, sizeof(cl_mem), &mres);
    if (ocl_launch(o, o->k_iq, nthreads, &ev) != AEG_OK ||
        ocl_read(o, mres, 0, bytes, res, &ev) != AEG_OK)
        return ocl_wait(o, &ev);
    return ocl_wait(o, &ev);
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
    cl_int st;
    ocl_evlist ev;

    if (nblk <= 0)
        return AEG_OK;
    mres = ocl_buf(o, 0, bytes, CL_MEM_READ_ONLY);
    mlev = ocl_buf(o, 1, bytes, CL_MEM_WRITE_ONLY);
    mnz = ocl_buf(o, 2, (size_t)nblk * 4, CL_MEM_WRITE_ONLY);
    mrec = ocl_buf(o, 3, bytes, CL_MEM_WRITE_ONLY);
    if (!mres || !mlev || !mnz || !mrec)
        return AEG_EBACKEND;
    ev.n = 0;
    if (ocl_write(o, mres, 0, bytes, res, &ev) != AEG_OK)
        return ocl_wait(o, &ev);
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
    if (ocl_launch(o, o->k_tqiq, nthreads, &ev) != AEG_OK)
        return ocl_wait(o, &ev);
    /* one read back per output, all of them ordered after the kernel, and a
     * single clFinish to bring them home */
    if (ocl_read(o, mlev, 0, bytes, level, &ev) != AEG_OK ||
        (nz && ocl_read(o, mnz, 0, (size_t)nblk * 4, nz, &ev) != AEG_OK) ||
        ocl_read(o, mrec, 0, bytes, rec, &ev) != AEG_OK)
        return ocl_wait(o, &ev);
    return ocl_wait(o, &ev);
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
    ocl_evlist ev;

    if (nblk <= 0)
        return AEG_OK;
    mres = ocl_buf(o, 0, bytes, CL_MEM_READ_ONLY);
    mlev = ocl_buf(o, 1, bytes, CL_MEM_WRITE_ONLY);
    mnz = ocl_buf(o, 2, (size_t)nblk * 4, CL_MEM_WRITE_ONLY);
    mdc = ocl_buf(o, 3, (size_t)nblk * 4, CL_MEM_WRITE_ONLY);
    if (!mres || !mlev || !mnz || !mdc)
        return AEG_EBACKEND;
    ev.n = 0;
    if (ocl_write(o, mres, 0, bytes, res, &ev) != AEG_OK)
        return ocl_wait(o, &ev);
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
    if (ocl_launch(o, o->k_tqdc, nthreads, &ev) != AEG_OK)
        return ocl_wait(o, &ev);
    if (ocl_read(o, mlev, 0, bytes, level, &ev) != AEG_OK ||
        (nz && ocl_read(o, mnz, 0, (size_t)nblk * 4, nz, &ev) != AEG_OK) ||
        (dc && ocl_read(o, mdc, 0, (size_t)nblk * 4, dc, &ev) != AEG_OK))
        return ocl_wait(o, &ev);
    return ocl_wait(o, &ev);
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
    ocl_evlist ev;

    if (nblk <= 0)
        return AEG_OK;
    mlev = ocl_buf(o, 0, bytes, CL_MEM_READ_ONLY);
    mdc = ocl_buf(o, 1, (size_t)nblk * 4, CL_MEM_READ_ONLY);
    mres = ocl_buf(o, 2, bytes, CL_MEM_WRITE_ONLY);
    if (!mlev || !mdc || !mres)
        return AEG_EBACKEND;
    ev.n = 0;
    if (ocl_write(o, mlev, 0, bytes, level, &ev) != AEG_OK ||
        ocl_write(o, mdc, 0, (size_t)nblk * 4, dc, &ev) != AEG_OK)
        return ocl_wait(o, &ev);
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
    if (ocl_launch(o, o->k_iqdc, nthreads, &ev) != AEG_OK)
        return ocl_wait(o, &ev);
    if (ocl_read(o, mres, 0, bytes, res, &ev) != AEG_OK)
        return ocl_wait(o, &ev);
    return ocl_wait(o, &ev);
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
    ocl_evlist ev;

    ma = ocl_buf(o, 0, abytes, CL_MEM_READ_ONLY);
    mb = ocl_buf(o, 1, (size_t)sb * h, CL_MEM_READ_ONLY);
    mout = ocl_buf(o, 2, count * 8, CL_MEM_WRITE_ONLY);
    if (!ma || !mb || !mout)
        return AEG_EBACKEND;
    ev.n = 0;
    if (ocl_write(o, ma, 0, abytes, a, &ev) != AEG_OK ||
        ocl_write(o, mb, 0, (size_t)sb * h, b, &ev) != AEG_OK)
        return ocl_wait(o, &ev);
    if (!count) {
        /* an empty batch still has an answer, or the caller keeps the value it
         * happened to have and reports a PSNR from the wrong number */
        *out = 0;
        return ocl_wait(o, &ev);
    }
    tmp = (uint64_t *)malloc(count * 8);
    if (!tmp)
        return AEG_ENOMEM;
    {
        int itw = (int)tw, ith = (int)th, icount = (int)count;
        st |= p_clSetKernelArg(o->k_sse, 0, sizeof(cl_mem), &ma);
        st |= p_clSetKernelArg(o->k_sse, 1, sizeof(int), &sa);
        st |= p_clSetKernelArg(o->k_sse, 2, sizeof(cl_mem), &mb);
        st |= p_clSetKernelArg(o->k_sse, 3, sizeof(int), &sb);
        st |= p_clSetKernelArg(o->k_sse, 4, sizeof(int), &itw);
        st |= p_clSetKernelArg(o->k_sse, 5, sizeof(int), &ith);
        st |= p_clSetKernelArg(o->k_sse, 6, sizeof(int), &icount);
        st |= p_clSetKernelArg(o->k_sse, 7, sizeof(cl_mem), &mout);
    }
    if (st != CL_SUCCESS)
        return AEG_EBACKEND;
    if (ocl_launch(o, o->k_sse, count, &ev) != AEG_OK ||
        ocl_read(o, mout, 0, count * 8, tmp, &ev) != AEG_OK)
        return ocl_wait(o, &ev);
    if (ocl_wait(o, &ev) != AEG_OK) {
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
    ocl_ctx *o = (ocl_ctx *)priv;
    cl_mem mref, mdst;
    size_t rows = (size_t)h + 2 * AEG_PAD;
    size_t count = (size_t)(w + 2 * AEG_PAD) * rows;
    size_t rbytes = (size_t)rstride * rows, dbytes = (size_t)dstride * rows;
    int pad = AEG_PAD;
    cl_int st;
    ocl_evlist ev;

    /* `ref` and `dst` point at the picture origins; the kernel takes the
     * whole padded allocations (rows * stride bytes from pad rows and pad
     * columns before the origin), because its output covers the padding */
    mref = ocl_buf(o, 0, rbytes, CL_MEM_READ_ONLY);
    mdst = ocl_buf(o, 1, dbytes, CL_MEM_WRITE_ONLY);
    if (!mref || !mdst)
        return AEG_EBACKEND;
    ev.n = 0;
    if (ocl_write(o, mref, 0, rbytes,
                  ref - (ptrdiff_t)AEG_PAD * rstride - AEG_PAD, &ev) != AEG_OK)
        return ocl_wait(o, &ev);
    p_clSetKernelArg(o->k_intp, 0, sizeof(cl_mem), &mref);
    p_clSetKernelArg(o->k_intp, 1, sizeof(int), &rstride);
    p_clSetKernelArg(o->k_intp, 2, sizeof(int), &w);
    p_clSetKernelArg(o->k_intp, 3, sizeof(int), &h);
    p_clSetKernelArg(o->k_intp, 4, sizeof(int), &pad);
    p_clSetKernelArg(o->k_intp, 5, sizeof(int), &which);
    p_clSetKernelArg(o->k_intp, 6, sizeof(cl_mem), &mdst);
    p_clSetKernelArg(o->k_intp, 7, sizeof(int), &dstride);
    if (ocl_launch(o, o->k_intp, count, &ev) != AEG_OK)
        return ocl_wait(o, &ev);
    if (ocl_read(o, mdst, 0, dbytes,
                 dst - (ptrdiff_t)AEG_PAD * dstride - AEG_PAD, &ev) != AEG_OK)
        return ocl_wait(o, &ev);
    return ocl_wait(o, &ev);
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
    ocl_evlist ev;
    int k;

    ev.n = 0;
    for (k = 0; k < 4; k++) {
        cl_mem m = ocl_buf(o, 4 + k, bytes, CL_MEM_READ_ONLY);
        if (!m)
            return ocl_wait(o, &ev);
        if (ocl_write(o, m, 0, bytes,
                      planes[k] - (ptrdiff_t)AEG_PAD * pstride - AEG_PAD,
                      &ev) != AEG_OK)
            return ocl_wait(o, &ev);
    }
    o->ref_stride = pstride;
    o->ref_w = w;
    o->ref_h = h;
    o->ref_valid = 1;
    /* the uploads leave on this queue's stream; the searches that read them
     * wait here, so nothing may re-pack the staging buffers in between */
    return ocl_wait(o, &ev);
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
    ocl_evlist ev;

    if (!o->ref_valid || !count)
        return AEG_EBACKEND;
    mcur = ocl_buf(o, 0, cbytes, CL_MEM_READ_ONLY);
    mpos = ocl_buf(o, 1, (size_t)count * 8, CL_MEM_READ_ONLY);
    mout = ocl_buf(o, 2, (size_t)count * 4, CL_MEM_WRITE_ONLY);
    if (!mcur || !mpos || !mout)
        return AEG_EBACKEND;
    for (k = 0; k < 4; k++)
        mref[k] = o->buf[4 + k];
    ev.n = 0;
    if (ocl_write(o, mcur, 0, cbytes,
                  cur - (ptrdiff_t)AEG_PAD * cstride - AEG_PAD, &ev) != AEG_OK ||
        ocl_write(o, mpos, 0, (size_t)count * 8, pos, &ev) != AEG_OK)
        return ocl_wait(o, &ev);
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
    if (ocl_launch(o, o->k_satd, (size_t)count, &ev) != AEG_OK)
        return ocl_wait(o, &ev);
    if (ocl_read(o, mout, 0, (size_t)count * 4, out, &ev) != AEG_OK)
        return ocl_wait(o, &ev);
    return ocl_wait(o, &ev);
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
    if (o->k_sad) p_clReleaseKernel(o->k_sad);
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
    o->k_tq = p_clCreateKernel(o->prog, "tq_batch", &st);
    o->k_iq = p_clCreateKernel(o->prog, "iq_idct_batch", &st);
    o->k_tqiq = p_clCreateKernel(o->prog, "tq_iq_batch", &st);
    o->k_sse = p_clCreateKernel(o->prog, "sse_batch", &st);
    o->k_intp = p_clCreateKernel(o->prog, "interp_plane", &st);
    o->k_satd = p_clCreateKernel(o->prog, "satd_batch", &st);
    o->k_tqdc = p_clCreateKernel(o->prog, "tq_dc_batch", &st);
    o->k_iqdc = p_clCreateKernel(o->prog, "iq_dc_batch", &st);
    return o->k_sad && o->k_tq && o->k_iq && o->k_tqiq && o->k_sse &&
           o->k_intp && o->k_satd && o->k_tqdc && o->k_iqdc ? AEG_OK
                                                             : AEG_EBACKEND;
}

/* One instance's queue.  Out of order, so the device may overlap the upload
 * of one operation with the kernel of the previous one (the dependencies are
 * events, see the data flow above); a runtime that refuses the flag gets an
 * in-order queue, where the events are satisfied in the order they were
 * enqueued and the whole operation is simply sequential. */
static cl_command_queue ocl_queue(cl_context ctx, cl_device_id dev, cl_int *st)
{
    cl_command_queue q;
    cl_ulong props = CL_QUEUE_OUT_OF_ORDER_MODE;
    static _Atomic int ooo = -1;

    if (ooo < 0) {
        const char *ev = getenv("AEG_OCL_OOO");
        ooo = ev ? atoi(ev) != 0 : 1;
    }
    q = ooo ? p_clCreateCommandQueue(ctx, dev, props, st) : NULL;
    if (q && *st == CL_SUCCESS)
        return q;
    *st = CL_SUCCESS;
    q = p_clCreateCommandQueue(ctx, dev, 0, st);
    if (q && *st == CL_SUCCESS)
        aeg_log(AEG_LOG_DEBUG, "aeg: the OpenCL runtime refused an out-of-order "
                "queue, using an in-order one");
    return q;
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
    o->queue = ocl_queue(o->ctx, o->device, &st);
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
    if (p_clGetPlatformIDs(1, &o->platform, NULL) != CL_SUCCESS ||
        p_clGetDeviceIDs(o->platform, CL_DEVICE_TYPE_ALL, 1, &o->device,
                         NULL) != CL_SUCCESS) {
        ocl_destroy(o);
        return AEG_EBACKEND;
    }
    p_clGetDeviceInfo(o->device, CL_DEVICE_NAME, sizeof(o->name), o->name, &got);
    p_clGetDeviceInfo(o->device, CL_DEVICE_VENDOR, sizeof(o->vendor), o->vendor, &got);
    p_clGetDeviceInfo(o->device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(o->mem), &o->mem, NULL);
    p_clGetDeviceInfo(o->device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(o->cus), &o->cus, NULL);
    {
        cl_uint t = 0;
        p_clGetDeviceInfo(o->device, CL_DEVICE_TYPE, sizeof(t), &t, NULL);
        o->is_gpu = (t & CL_DEVICE_TYPE_GPU) != 0;
    }
    o->no_clone = ocl_pocl_unsafe_queues(o->device);

    o->ctx = p_clCreateContext(NULL, 1, &o->device, NULL, NULL, &st);
    if (!o->ctx || st != CL_SUCCESS) {
        ocl_destroy(o);
        return AEG_EBACKEND;
    }
    o->queue = ocl_queue(o->ctx, o->device, &st);
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
