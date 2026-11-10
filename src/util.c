/*
 * Allocation, logging, timing and small shared helpers.
 */
#define _POSIX_C_SOURCE 200809L
#include <pthread.h>
#include <stdarg.h>
#include <time.h>
#include <unistd.h>
#include "h264.h"

int aeg_log_level = AEG_LOG_WARNING;

/* The sink is process-wide.  Messages come from the caller's thread and from
 * the encoder's workers, so the pair (fn, opaque) is read and written under a
 * lock: a message never goes to a new function with the old one's opaque. */
static pthread_mutex_t g_log_lock = PTHREAD_MUTEX_INITIALIZER;
static aeg_log_fn g_log_fn;
static void *g_log_opaque;

void aeg_set_log_callback(aeg_log_fn fn, void *opaque)
{
    pthread_mutex_lock(&g_log_lock);
    g_log_fn = fn;
    g_log_opaque = fn ? opaque : NULL;
    pthread_mutex_unlock(&g_log_lock);
}

void aeg_set_log_level(int level)
{
    aeg_log_level = level < -1 ? -1 : level;
}

int aeg_get_log_level(void)
{
    return aeg_log_level;
}

void aeg_log(int level, const char *fmt, ...)
{
    char msg[1024];
    va_list ap;
    size_t n;
    aeg_log_fn fn;
    void *opaque;

    if (level > aeg_log_level)
        return;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    n = strlen(msg);
    while (n && msg[n - 1] == '\n')
        msg[--n] = 0;
    pthread_mutex_lock(&g_log_lock);
    fn = g_log_fn;
    opaque = g_log_opaque;
    if (!fn) {
        /* one call per line keeps lines from two threads apart */
        fprintf(stderr, "%s\n", msg);
        pthread_mutex_unlock(&g_log_lock);
        return;
    }
    pthread_mutex_unlock(&g_log_lock);
    fn(opaque, level, msg);
}

const char *aeg_version(void)
{
    return AEG_VERSION_STRING;
}

int aeg_version_number(void)
{
    return AEG_VERSION_NUMBER;
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
    p->colorprim = p->transfer = p->colormatrix = -1;
    p->qp_max = 51;
    p->qp = 23;
    p->bit_depth = 8;
    p->crf = 23.0;
    p->qcomp = 60;
    p->rc_mode = AEG_RC_CQP;
    p->gop_size = 250;
    /* 0 would mean "off"; a scene change is common enough that the default is
     * on, with a threshold that ignores ordinary motion */
    p->scenecut = 120;
    p->preset = AEG_PRESET_MEDIUM;
    p->backend = AEG_BACKEND_AUTO;
    p->device = -1;
    p->ssim = 0;
    for (i = 0; i < AEG_STAGE_COUNT; i++)
        p->stage_backend[i] = -1;
    /* Adaptive quantisation (variance AQ) is implemented but off by default:
     * at a fixed rate it *lowers* PSNR on the test set (0.3 dB on the quadrant
     * clip, 0.9 dB on the panning texture at 300 kb/s) because the bit
     * reallocation between flat and busy macroblocks costs more than it buys
     * when the yardstick is PSNR.  See docs/STATUS.md. */
    p->aq_mode = 0;
    p->aq_strength = 10;
    p->transform_8x8 = 1;
    p->cabac = 1;
    /* the rate-distortion refinement of the winning motion vector follows the
     * preset (see aeg_encoder_create) */
    p->refine_mv = -1;
    p->qp_rd = -1;          /* (preset 4) */
    /* B pictures are on by default: three between two anchors, each quantised
     * twelve coarser than the anchors around it.  Both numbers come from the
     * measurement in docs/STATUS.md -- on five test clips at six QPs that pair
     * is between 8% and 21% *cheaper* than x264 medium at matched PSNR, against
     * 10% to 28% more expensive with no B pictures at all.  The offset is large
     * because a B picture is never a reference: its error is thrown away, so
     * the only thing a coarser quantiser costs is the B picture's own
     * appearance.  The sweep that set it (docs/STATUS.md, *Making them pay*)
     * was re-run after the partitioned B types landed, and the optimum moved
     * *up*, from 8 to 12-14, not down: better prediction makes the residual
     * cheaper to throw away, not dearer.  12 is the middle of the measured flat
     * top (g1_60 -2.1%, p2 -1.3%, p0 -3.2% against 8, q4 and p3 neutral).
     * That optimum was an artefact of the synthetic clips: on five real clips
     * (docs/STATUS.md, *Real clips*) 12 costs 8-12 points of BD-rate against
     * x264 and the optimum is 4-6, with the I-picture offset and the adaptive
     * group length on.  4 it is.
     * -1: the preset decides how many -- three up to preset 2, eight (with
     * the adaptive decision picking the length) at presets 3 and 4. */
    p->bframes = -1;
    /* -1: 4, or 2 with the macroblock tree, which already moves the anchors'
     * quantiser down by what the B pictures inherit from them */
    p->b_qp_offset = -1;
    p->i_qp_offset = -1;
    p->direct = -1;
    p->b_adapt = -1;
    p->trellis = -1;
    p->refs = -1;
    p->weightp = -1;
    p->b_pyramid = -1;
    p->mbtree = -1;
    p->rc_lookahead = -1;
    p->deblock = 1;
    p->deblock_alpha = 0;
    p->deblock_beta = 0;
    p->chroma_me = 1;
    /* -1 = "decide from the preset" (aeg_encoder_open): 0 for ultrafast, 1 for
     * everything else.  Leaving the field zeroed silently disabled the
     * quarter-pel refinement at every preset, which cost ~1 dB of PSNR. */
    p->subme = -1;
    /* Inter partitioning mask: bit 0 = 16x16, bit 1 = 16x8, bit 2 = 8x16,
     * bit 3 = P_8x8 (four 8x8 sub-macroblocks with a motion vector each),
     * bit 4 = their further split into 8x4, 4x8 or 4x4 (off: it measured
     * worse, see the P_8x8 candidate).  The mask must not be zero, otherwise
     * analyse_mb() treats every picture as all-intra. */
    p->partitions = 1 | 2 | 4 | 8;
    p->verbose = 1;
}

/* ------------------------------------------------------------------ */
/* Stage/backend timing accounting shared by encoder and backends      */
/* ------------------------------------------------------------------ */
/* per thread: two encoders failing in two threads each see their own */
static _Thread_local const char *g_last_error = "";

void aeg_set_last_error(const char *msg)
{
    g_last_error = msg;
}

const char *aeg_last_error(void)
{
    return g_last_error;
}
