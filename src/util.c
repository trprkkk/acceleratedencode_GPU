/*
 * Allocation, logging, timing and small shared helpers.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdarg.h>
#include <time.h>
#include <unistd.h>
#include "h264.h"

int aeg_log_level = 1;

void aeg_log(int level, const char *fmt, ...)
{
    va_list ap;
    if (level > aeg_log_level)
        return;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
}

void *aeg_alloc_aligned(size_t size, size_t align)
{
    void *p = NULL;
    if (size == 0)
        size = 1;
#if defined(_ISOC11_SOURCE) || __STDC_VERSION__ >= 201112L
    if (posix_memalign(&p, align, size) != 0)
        return NULL;
#else
    p = malloc(size);
#endif
    if (p)
        memset(p, 0, size);
    return p;
}

void *aeg_alloc(size_t size)
{
    return aeg_alloc_aligned(size, 32);
}

void aeg_free(void *p)
{
    free(p);
}

int aeg_cpu_count(void)
{
#if defined(_SC_NPROCESSORS_ONLN)
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
#else
    return 1;
#endif
}

int64_t aeg_now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

const char *aeg_stage_name(int stage)
{
    static const char *names[AEG_STAGE_COUNT] = {
        "motion-est", "sad", "inter-pred", "transform",
        "iquant-idct", "deblock", "intra-pred", "recon", "psnr"
    };
    if (stage < 0 || stage >= AEG_STAGE_COUNT)
        return "?";
    return names[stage];
}

const char *aeg_backend_name(aeg_backend b)
{
    switch (b) {
    case AEG_BACKEND_CPU:    return "cpu";
    case AEG_BACKEND_OPENCL: return "opencl";
    case AEG_BACKEND_CUDA:   return "cuda";
    default:                 return "auto";
    }
}

void aeg_params_default(aeg_params *p)
{
    int i;
    memset(p, 0, sizeof(*p));
    p->fps_num = 30;
    p->fps_den = 1;
    p->qp = 23;
    p->rc_mode = AEG_RC_CQP;
    p->gop_size = 250;
    p->preset = AEG_PRESET_MEDIUM;
    p->backend = AEG_BACKEND_AUTO;
    for (i = 0; i < AEG_STAGE_COUNT; i++)
        p->stage_backend[i] = -1;
    p->aq_mode = 1;
    p->aq_strength = 10;
    p->deblock = 1;
    p->deblock_alpha = 0;
    p->deblock_beta = 0;
    p->chroma_me = 1;
    /* 16x16 is the only inter partitioning the encoder implements; without at
     * least this bit analyse_mb() treats every picture as all-intra. */
    p->partitions = 1;      /* bit 0: 16x16, the only mode implemented */
    p->verbose = 1;
}

/* ------------------------------------------------------------------ */
/* Stage/backend timing accounting shared by encoder and backends      */
/* ------------------------------------------------------------------ */
static const char *g_last_error = "";

void aeg_set_last_error(const char *msg)
{
    g_last_error = msg;
}

const char *aeg_last_error(void)
{
    return g_last_error;
}
