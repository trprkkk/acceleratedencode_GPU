/*
 * api_check - the public API on its own (include/aeg/aeg.h, nothing else):
 *
 *  - the run-time version agrees with the header;
 *  - log messages reach the callback, one line each, filtered by the level;
 *  - bad arguments are refused with a message, and the message is per thread;
 *  - aeg_encoder_get_params() resolves every "-1 = as the preset says";
 *  - aeg_encoder_headers() is exactly what the first IDR begins with;
 *  - strided planes give the same bytes as the packed buffer;
 *  - dts increases strictly, never passes pts, and follows the timebase;
 *  - encoders running in parallel threads give the bytes they give alone;
 *  - bit_depth: 8 and 10 open (10 only in a library built with it), other
 *    depths are refused; a 10-bit encoder writes a High 10 SPS, codes 8-bit
 *    input scaled up exactly (every layout the same stream), uses the two
 *    extra bits of 10-bit input, and returns 16-bit reconstructions.
 *
 * usage: api_check
 */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aeg/aeg.h"

static int failures;

#define CHECK(cond, ...)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                     \
            printf(__VA_ARGS__);                                            \
            printf("\n");                                                   \
            failures++;                                                     \
        }                                                                   \
    } while (0)

/* An odd chroma height and a width that is not a whole number of macroblocks:
 * the crop path, with planes of their own widths. */
enum { W = 200, H = 116, CW = (W + 1) / 2, CH = (H + 1) / 2, NF = 14 };

static uint8_t *clip;   /* NF packed I420 frames */

static size_t frame_bytes(void)
{
    return (size_t)W * H + 2 * (size_t)CW * CH;
}

static void make_clip(void)
{
    size_t fb = frame_bytes();
    int t, x, y;
    unsigned r = 1;
    clip = (uint8_t *)malloc(fb * NF);
    for (t = 0; t < NF; t++) {
        uint8_t *f = clip + fb * t;
        for (y = 0; y < H; y++)
            for (x = 0; x < W; x++) {
                int cx = x + 3 * t, cy = y + 2 * t;
                r = r * 1103515245u + 12345u;
                f[y * W + x] = (uint8_t)(64 + ((cx / 9 + cy / 7) & 1) * 80 +
                                         ((cx * cy) & 31) + ((r >> 16) & 7));
            }
        for (y = 0; y < 2 * CH; y++)
            for (x = 0; x < CW; x++)
                f[W * H + y * CW + x] = (uint8_t)(128 + ((x + t + y) & 15) - 8);
    }
}

/* A growing byte buffer holding a whole stream. */
typedef struct buf {
    uint8_t *d;
    size_t n, cap;
} buf;

static void put(buf *b, const uint8_t *d, size_t n)
{
    if (b->n + n > b->cap) {
        b->cap = 2 * (b->n + n);
        b->d = (uint8_t *)realloc(b->d, b->cap);
    }
    memcpy(b->d + b->n, d, n);
    b->n += n;
}

/* The clip's frame t with flat chroma (value flat_chroma(t)) -- what a 4:4:4,
 * 4:2:2 or grey picture with the same flat chroma converts to exactly. */
static int flat_value = -1;      /* -1: 100 + 3t; else this value */
static int flat_chroma(int t)
{
    return flat_value >= 0 ? flat_value : 100 + 3 * t;
}
static void flat_frame(uint8_t *dst, const uint8_t *f, int t)
{
    memcpy(dst, f, (size_t)W * H);
    memset(dst + W * H, flat_chroma(t), 2 * (size_t)CW * CH);
}

enum {
    IMG_NONE, IMG_NV12, IMG_NV21, IMG_YV12, IMG_I420_10, IMG_I420_16,
    IMG_I444_FLAT, IMG_I422_FLAT, IMG_NV16_FLAT, IMG_I400, IMG_I444_16_FLAT
};

/* Lay frame f (packed I420) out as image mode m in scratch (padded rows
 * of junk the encoder must not read). */
static void fill_image(int m, const uint8_t *f, int t, uint8_t *scratch,
                       aeg_image *im)
{
    const uint8_t *Y = f, *U = f + W * H, *V = U + CW * CH;
    int x, y, i;
    int hi = m == IMG_I420_10 || m == IMG_I420_16 || m == IMG_I444_16_FLAT;
    int bps = hi ? 2 : 1, sh = m == IMG_I420_10 ? 2 : 8;
    int cw = CW, ch = CH, il = 1, np = 3;
    memset(im, 0, sizeof(*im));
    im->depth = m == IMG_I420_10 ? 10 : hi ? 16 : 8;
    switch (m) {
    case IMG_NV12: im->csp = AEG_CSP_NV12; il = 2; np = 2; break;
    case IMG_NV21: im->csp = AEG_CSP_NV21; il = 2; np = 2; break;
    case IMG_YV12: im->csp = AEG_CSP_YV12; break;
    case IMG_I420_10: case IMG_I420_16: im->csp = AEG_CSP_I420; break;
    case IMG_I444_FLAT: case IMG_I444_16_FLAT:
        im->csp = AEG_CSP_I444; cw = W; ch = H; break;
    case IMG_I422_FLAT: im->csp = AEG_CSP_I422; ch = H; break;
    case IMG_NV16_FLAT: im->csp = AEG_CSP_NV16; ch = H; il = 2; np = 2; break;
    case IMG_I400: im->csp = AEG_CSP_I400; np = 1; break;
    }
    im->stride[0] = W * bps + 6;
    for (i = 1; i < np; i++)
        im->stride[i] = cw * il * bps + 10;
    memset(scratch, 0xA7, (size_t)im->stride[0] * H + 2 * (size_t)(W * 2 * bps + 10) * H);
    im->plane[0] = scratch;
    for (i = 1; i < np; i++)
        im->plane[i] = (const uint8_t *)im->plane[i - 1] +
                       (size_t)im->stride[i - 1] * (i == 1 ? H : ch);
    for (y = 0; y < H; y++)
        for (x = 0; x < W; x++) {
            uint8_t *d = (uint8_t *)im->plane[0] + (size_t)y * im->stride[0];
            if (hi)
                ((uint16_t *)d)[x] = (uint16_t)(Y[y * W + x] << sh);
            else
                d[x] = Y[y * W + x];
        }
    for (y = 0; y < ch && np > 1; y++)
        for (x = 0; x < cw; x++) {
            int fl = m >= IMG_I444_FLAT;
            int u = fl ? flat_chroma(t) : U[y * CW + x];
            int v = fl ? flat_chroma(t) : V[y * CW + x];
            if (il == 2) {
                uint8_t *d = (uint8_t *)im->plane[1] + (size_t)y * im->stride[1];
                d[2 * x] = (uint8_t)(m == IMG_NV21 ? v : u);
                d[2 * x + 1] = (uint8_t)(m == IMG_NV21 ? u : v);
            } else {
                int pu = m == IMG_YV12 ? 2 : 1;
                uint8_t *du = (uint8_t *)im->plane[pu] + (size_t)y * im->stride[pu];
                uint8_t *dv = (uint8_t *)im->plane[3 - pu] + (size_t)y * im->stride[3 - pu];
                if (hi) {
                    ((uint16_t *)du)[x] = (uint16_t)(u << sh);
                    ((uint16_t *)dv)[x] = (uint16_t)(v << sh);
                } else {
                    du[x] = (uint8_t)u;
                    dv[x] = (uint8_t)v;
                }
            }
        }
}

typedef struct run {
    aeg_params p;
    int image;           /* feed aeg_image in this IMG_* layout */
    int flat;            /* encode the flat-chroma frames (packed I420) */
    int planes;          /* feed strided planes instead of packed frames */
    int64_t pts_step;
    buf out;
    int64_t pts[NF], dts[NF];
    int n, rc;
} run;

/* Encode the clip; the stream goes to r->out, the timestamps to r->pts/dts. */
static void encode_clip(run *r)
{
    aeg_encoder *enc = aeg_encoder_open(&r->p);
    aeg_packet pkt;
    size_t fb = frame_bytes();
    int t, rc, y, s0 = W + 37, s1 = CW + 11;
    uint8_t *pl[3] = { NULL, NULL, NULL };
    uint8_t *scratch = NULL, *ff = NULL;

    r->n = 0;
    r->rc = 0;
    if (!enc) {
        r->rc = -100;
        return;
    }
    if (r->planes) {
        pl[0] = (uint8_t *)malloc((size_t)s0 * H);
        pl[1] = (uint8_t *)malloc((size_t)s1 * CH);
        pl[2] = (uint8_t *)malloc((size_t)s1 * CH);
    }
    if (r->image)
        scratch = (uint8_t *)malloc((size_t)(W * 2 + 6) * H +
                                    2 * (size_t)(W * 4 + 10) * H);
    if (r->flat)
        ff = (uint8_t *)malloc(fb);
    for (t = 0; t < NF; t++) {
        const uint8_t *f = clip + fb * t;
        int64_t pts = 1000 + t * r->pts_step;
        if (r->flat) {
            flat_frame(ff, f, t);
            f = ff;
        }
        if (r->image) {
            aeg_image im;
            fill_image(r->image, f, t, scratch, &im);
            rc = aeg_encoder_encode_image(enc, &im, pts, &pkt);
        } else if (r->planes) {
            const uint8_t *cp[3];
            int st[3] = { s0, s1, s1 };
            /* fill the padding with junk the encoder must not read */
            memset(pl[0], 0xA5, (size_t)s0 * H);
            memset(pl[1], 0x5A, (size_t)s1 * CH);
            memset(pl[2], 0xC3, (size_t)s1 * CH);
            for (y = 0; y < H; y++)
                memcpy(pl[0] + (size_t)y * s0, f + (size_t)y * W, W);
            for (y = 0; y < CH; y++) {
                memcpy(pl[1] + (size_t)y * s1, f + W * H + (size_t)y * CW, CW);
                memcpy(pl[2] + (size_t)y * s1,
                       f + W * H + CW * CH + (size_t)y * CW, CW);
            }
            cp[0] = pl[0], cp[1] = pl[1], cp[2] = pl[2];
            rc = aeg_encoder_encode_planes(enc, cp, st, pts, &pkt);
        } else
            rc = aeg_encoder_encode(enc, f, fb, pts, &pkt);
        if (rc == AEG_OK && r->n < NF) {
            put(&r->out, pkt.data, pkt.size);
            r->pts[r->n] = pkt.pts;
            r->dts[r->n++] = pkt.dts;
        } else if (rc != AEG_EAGAIN) {
            r->rc = rc;
            break;
        }
    }
    while ((rc = aeg_encoder_flush(enc, &pkt)) == AEG_OK && r->n < NF) {
        put(&r->out, pkt.data, pkt.size);
        r->pts[r->n] = pkt.pts;
        r->dts[r->n++] = pkt.dts;
    }
    if (rc != AEG_EOF && rc != AEG_OK && !r->rc)
        r->rc = rc;
    aeg_encoder_close(enc);
    free(pl[0]);
    free(pl[1]);
    free(pl[2]);
    free(scratch);
    free(ff);
}

static void base_params(aeg_params *p, int preset);

/* Every input layout that converts exactly must give the bytes of the same
 * picture given as I420; and the conversion's argument checks. */
static void check_images(int bit_depth)
{
    static const struct { int mode, flat, fv; const char *name; } cases[] = {
        { IMG_NV12, 0, -1, "NV12" }, { IMG_NV21, 0, -1, "NV21" },
        { IMG_YV12, 0, -1, "YV12" }, { IMG_I420_10, 0, -1, "I420 10-bit" },
        { IMG_I420_16, 0, -1, "I420 16-bit" },
        { IMG_I444_FLAT, 1, -1, "I444 flat chroma" },
        { IMG_I422_FLAT, 1, -1, "I422 flat chroma" },
        { IMG_NV16_FLAT, 1, -1, "NV16 flat chroma" },
        { IMG_I444_16_FLAT, 1, -1, "I444 16-bit flat chroma" },
        { IMG_I400, 1, 128, "I400" },
    };
    int i, inter;
    for (inter = 0; inter <= 1; inter++)
        for (i = 0; i < (int)(sizeof(cases) / sizeof(cases[0])); i++) {
            run a, b;
            memset(&a, 0, sizeof(a));
            memset(&b, 0, sizeof(b));
            flat_value = cases[i].fv;
            base_params(&a.p, 2);
            a.p.interlaced = inter ? 2 : 0;
            a.p.bit_depth = bit_depth;
            a.pts_step = b.pts_step = 1;
            b.p = a.p;
            a.flat = cases[i].flat;
            b.image = cases[i].mode;
            encode_clip(&a);
            encode_clip(&b);
            CHECK(a.rc == 0 && b.rc == 0, "%s: rc %d %d (%s)", cases[i].name,
                  a.rc, b.rc, aeg_last_error());
            CHECK(a.out.n > 0 && a.out.n == b.out.n &&
                  !memcmp(a.out.d, b.out.d, a.out.n),
                  "%s%s, %d-bit: %zu bytes, I420 %zu: not the same stream",
                  cases[i].name, inter ? " (interlaced)" : "", bit_depth,
                  b.out.n, a.out.n);
            free(a.out.d);
            free(b.out.d);
        }
    flat_value = -1;
    if (bit_depth == 8) {
        aeg_params p;
        aeg_encoder *e;
        aeg_packet pkt;
        aeg_image im;
        uint8_t *scratch = (uint8_t *)malloc((size_t)(W * 2 + 6) * H +
                                             2 * (size_t)(W * 4 + 10) * H + 2);
        base_params(&p, 1);
        e = aeg_encoder_open(&p);
        CHECK(e != NULL, "open: %s", aeg_last_error());
        if (e) {
            fill_image(IMG_I420_10, clip, 0, scratch, &im);
            im.csp = 42;
            CHECK(aeg_encoder_encode_image(e, &im, 0, &pkt) == AEG_EINVAL &&
                  strstr(aeg_last_error(), "colour"), "bad csp: '%s'",
                  aeg_last_error());
            im.csp = AEG_CSP_I420;
            im.depth = 17;
            CHECK(aeg_encoder_encode_image(e, &im, 0, &pkt) == AEG_EINVAL,
                  "depth 17 accepted");
            im.depth = 10;
            im.stride[1] = CW;      /* bytes: half a 16-bit row */
            CHECK(aeg_encoder_encode_image(e, &im, 0, &pkt) == AEG_EINVAL &&
                  strstr(aeg_last_error(), "stride"), "short stride: '%s'",
                  aeg_last_error());
            im.stride[1] = 2 * CW + 10;
            im.plane[0] = scratch + 1;
            CHECK(aeg_encoder_encode_image(e, &im, 0, &pkt) == AEG_EINVAL,
                  "unaligned 16-bit plane accepted");
            im.plane[0] = NULL;
            CHECK(aeg_encoder_encode_image(e, &im, 0, &pkt) == AEG_EINVAL,
                  "NULL plane accepted");
            CHECK(aeg_encoder_encode_image(e, NULL, 0, &pkt) == AEG_EINVAL,
                  "NULL image accepted");
            aeg_encoder_close(e);
        }
        free(scratch);
        /* interlaced needs a height that is a multiple of 4 */
        base_params(&p, 1);
        p.height = 118;
        p.interlaced = 1;
        e = aeg_encoder_open(&p);
        CHECK(e == NULL && strstr(aeg_last_error(), "multiple of 4"),
              "interlaced height 118 accepted ('%s')", aeg_last_error());
        aeg_encoder_close(e);
    }
}

static void base_params(aeg_params *p, int preset)
{
    aeg_params_default(p);
    p->width = W;
    p->height = H;
    p->preset = preset;
    p->qp = 27;
    p->backend = AEG_BACKEND_CPU;
}

/* ---------------------------------------------------------------- */

typedef struct logcap {
    int count, bad, max_level;
} logcap;

static void log_cb(void *opaque, int level, const char *msg)
{
    logcap *c = (logcap *)opaque;
    c->count++;
    if (level > c->max_level)
        c->max_level = level;
    if (!msg || strchr(msg, '\n'))
        c->bad++;
}

static void check_version_and_log(void)
{
    logcap c = { 0, 0, -1 };
    aeg_params p;
    aeg_encoder *e;
    int old = aeg_get_log_level();

    CHECK(!strcmp(aeg_version(), AEG_VERSION_STRING), "version %s vs %s",
          aeg_version(), AEG_VERSION_STRING);
    CHECK(aeg_version_number() == AEG_VERSION_NUMBER, "version number");

    aeg_set_log_callback(log_cb, &c);
    aeg_set_log_level(AEG_LOG_INFO);
    base_params(&p, 1);
    e = aeg_encoder_open(&p);
    CHECK(e != NULL, "open: %s", aeg_last_error());
    aeg_encoder_close(e);
    CHECK(c.count > 0, "no message reached the callback");
    CHECK(c.bad == 0, "%d message(s) with a newline or NULL", c.bad);
    CHECK(c.max_level <= AEG_LOG_INFO, "level %d above the threshold",
          c.max_level);

    /* the default threshold: a clean open says nothing, a dubious option
     * (CAVLC with B pictures) is a warning */
    c.count = 0;
    c.max_level = -1;
    aeg_set_log_level(AEG_LOG_WARNING);
    e = aeg_encoder_open(&p);
    aeg_encoder_close(e);
    CHECK(c.count == 0, "%d message(s) from a clean open at the warning level",
          c.count);
    p.cabac = 0;
    p.bframes = 2;
    e = aeg_encoder_open(&p);
    aeg_encoder_close(e);
    CHECK(c.count > 0 && c.max_level == AEG_LOG_WARNING,
          "CAVLC + B pictures: %d message(s), level %d", c.count, c.max_level);
    base_params(&p, 1);

    c.count = 0;
    aeg_set_log_level(-1);
    CHECK(aeg_get_log_level() == -1, "level not -1");
    e = aeg_encoder_open(&p);
    aeg_encoder_close(e);
    CHECK(c.count == 0, "%d message(s) at level -1", c.count);

    aeg_set_log_level(old);
    aeg_set_log_callback(NULL, NULL);
}

static void *error_in_thread(void *arg)
{
    const char **seen = (const char **)arg;
    seen[0] = aeg_last_error();      /* fresh thread: nothing yet */
    CHECK(aeg_encoder_open(NULL) == NULL, "open(NULL) succeeded");
    seen[1] = aeg_last_error();
    return NULL;
}

static void check_errors(void)
{
    aeg_params p;
    aeg_encoder *e;
    aeg_packet pkt;
    const uint8_t *pl[3];
    int st[3] = { W, CW, CW };
    const char *seen[2] = { NULL, NULL };
    pthread_t th;
    int old = aeg_get_log_level();

    aeg_set_log_level(-1);
    base_params(&p, 1);
    p.width = 1;
    CHECK(aeg_encoder_open(&p) == NULL, "open 1xH succeeded");
    CHECK(*aeg_last_error(), "no message for a bad size");

    base_params(&p, 1);
    e = aeg_encoder_open(&p);
    CHECK(e != NULL, "open: %s", aeg_last_error());
    CHECK(aeg_encoder_encode(e, clip, frame_bytes() - 1, 0, &pkt) == AEG_EINVAL,
          "short buffer accepted");
    CHECK(strstr(aeg_last_error(), "small") != NULL, "message '%s'",
          aeg_last_error());
    CHECK(aeg_encoder_encode(e, NULL, frame_bytes(), 0, &pkt) == AEG_EINVAL,
          "NULL buffer accepted");
    CHECK(aeg_encoder_encode(e, clip, frame_bytes(), 0, NULL) == AEG_EINVAL,
          "NULL packet accepted");
    pl[0] = clip;
    pl[1] = clip + W * H;
    pl[2] = NULL;
    CHECK(aeg_encoder_encode_planes(e, pl, st, 0, &pkt) == AEG_EINVAL,
          "NULL plane accepted");
    pl[2] = pl[1] + CW * CH;
    st[1] = CW - 1;
    CHECK(aeg_encoder_encode_planes(e, pl, st, 0, &pkt) == AEG_EINVAL,
          "narrow stride accepted");
    CHECK(strstr(aeg_last_error(), "stride") != NULL, "message '%s'",
          aeg_last_error());
    CHECK(aeg_encoder_headers(e, NULL) == AEG_EINVAL, "headers(NULL)");
    CHECK(aeg_encoder_get_params(e, NULL) == AEG_EINVAL, "get_params(NULL)");
    CHECK(aeg_encoder_get_params(NULL, &p) == AEG_EINVAL, "get_params on NULL");
    /* a refused picture leaves no trace: the stream equals one without it */
    aeg_encoder_close(e);

    /* this thread's message survives another thread's failure */
    aeg_encoder_open(NULL);
    {
        const char *mine = aeg_last_error();
        pthread_create(&th, NULL, error_in_thread, seen);
        pthread_join(th, NULL);
        CHECK(seen[0] && !*seen[0], "a new thread starts with '%s'",
              seen[0] ? seen[0] : "(null)");
        CHECK(seen[1] && *seen[1], "no message in the other thread");
        CHECK(aeg_last_error() == mine, "the other thread's error leaked");
    }
    aeg_set_log_level(old);
}

static void check_params(void)
{
    aeg_params p, q;
    aeg_encoder *e;
    int old = aeg_get_log_level();

    aeg_set_log_level(-1);
    base_params(&p, 3);
    e = aeg_encoder_open(&p);
    CHECK(e && aeg_encoder_get_params(e, &q) == AEG_OK, "get_params");
    if (e) {
        CHECK(q.width == W && q.height == H, "size %dx%d", q.width, q.height);
        CHECK(q.subme >= 0 && q.refine_mv >= 0 && q.refs >= 1 &&
              q.weightp >= 0 && q.b_pyramid >= 0 && q.mbtree >= 0 &&
              q.trellis >= 0 && q.b_adapt >= 0 && q.direct >= 0 &&
              q.b_qp_offset >= 0 && q.i_qp_offset >= 0,
              "unresolved: subme %d refine %d refs %d wp %d pyr %d tree %d "
              "trellis %d badapt %d direct %d bqp %d iqp %d", q.subme,
              q.refine_mv, q.refs, q.weightp, q.b_pyramid, q.mbtree,
              q.trellis, q.b_adapt, q.direct, q.b_qp_offset, q.i_qp_offset);
        CHECK(q.level_idc >= 10 && q.level_idc <= 62, "level %d", q.level_idc);
        CHECK(q.timebase_num == q.fps_den && q.timebase_den == q.fps_num,
              "timebase %d/%d", q.timebase_num, q.timebase_den);
        /* frame_threads: the workers in use (1 = coded serially) */
        CHECK(q.slices >= 1 && q.frame_threads >= 1 && q.frame_threads <= 16,
              "slices %d frame threads %d", q.slices, q.frame_threads);
        aeg_encoder_close(e);
    }
    /* CAVLC codes no B pictures: the resolved value says so */
    base_params(&p, 2);
    p.cabac = 0;
    p.bframes = 3;
    e = aeg_encoder_open(&p);
    CHECK(e && aeg_encoder_get_params(e, &q) == AEG_OK && q.bframes == 0,
          "CAVLC kept %d B pictures", e ? q.bframes : -1);
    aeg_encoder_close(e);
    aeg_set_log_level(old);
}

static void check_headers(int weightp)
{
    aeg_params p;
    aeg_encoder *e;
    aeg_packet hp, pkt;
    uint8_t *hdr;
    size_t hn;
    int rc, t, nal = 0;
    size_t i;
    int old = aeg_get_log_level();

    aeg_set_log_level(-1);
    base_params(&p, 2);
    p.weightp = weightp;
    p.bframes = 0;
    e = aeg_encoder_open(&p);
    CHECK(e != NULL, "open");
    if (!e)
        return;
    CHECK(aeg_encoder_headers(e, &hp) == AEG_OK && hp.size > 8, "headers");
    hn = hp.size;
    hdr = (uint8_t *)malloc(hn);
    memcpy(hdr, hp.data, hn);
    CHECK(hdr[0] == 0 && hdr[1] == 0 && hdr[2] == 0 && hdr[3] == 1 &&
          (hdr[4] & 31) == 7, "does not start with an SPS NAL");
    for (i = 0; i + 3 < hn; i++)
        if (hdr[i] == 0 && hdr[i + 1] == 0 && hdr[i + 2] == 1) {
            int type = hdr[i + 3] & 31;
            CHECK(type == 7 || type == 8, "NAL type %d in the headers", type);
            nal++;
        }
    CHECK(nal == (weightp ? 3 : 2), "%d NAL units (weightp %d)", nal, weightp);
    rc = aeg_encoder_encode(e, clip, frame_bytes(), 0, &pkt);
    CHECK(rc == AEG_OK && pkt.keyframe && pkt.size > hn &&
          !memcmp(pkt.data, hdr, hn),
          "the first IDR does not begin with the headers");
    for (t = 1; t < 3; t++)
        aeg_encoder_encode(e, clip + frame_bytes() * t, frame_bytes(), t, &pkt);
    CHECK(aeg_encoder_headers(e, &hp) == AEG_OK && hp.size == hn &&
          !memcmp(hp.data, hdr, hn), "the headers changed while encoding");
    free(hdr);
    aeg_encoder_close(e);
    aeg_set_log_level(old);
}

static void check_planes_and_dts(int bframes, int rc_mode, int tb)
{
    run a, b;
    int i;

    memset(&a, 0, sizeof(a));
    base_params(&a.p, 2);
    a.p.bframes = bframes;
    a.p.b_adapt = 0;          /* always `bframes`: the dts has to reorder */
    a.p.rc_mode = rc_mode;
    a.p.crf = 24.0;
    a.pts_step = 1;
    if (tb) {
        /* MPEG-TS clock at 30 fps: 3000 ticks a frame */
        a.p.timebase_num = 1;
        a.p.timebase_den = 90000;
        a.pts_step = 3000;
    }
    b = a;
    b.planes = 1;
    encode_clip(&a);
    encode_clip(&b);
    CHECK(a.rc == 0 && b.rc == 0, "encode rc %d / %d", a.rc, b.rc);
    CHECK(a.n == NF && b.n == NF, "%d / %d packets of %d", a.n, b.n, NF);
    CHECK(a.out.n == b.out.n && !memcmp(a.out.d, b.out.d, a.out.n),
          "strided planes differ from packed (bframes %d rc %d): %zu vs %zu",
          bframes, rc_mode, a.out.n, b.out.n);
    for (i = 0; i < a.n; i++) {
        CHECK(a.dts[i] <= a.pts[i], "packet %d: dts %lld > pts %lld", i,
              (long long)a.dts[i], (long long)a.pts[i]);
        CHECK(!i || a.dts[i] > a.dts[i - 1], "packet %d: dts %lld after %lld",
              i, (long long)a.dts[i], (long long)a.dts[i - 1]);
        CHECK(a.dts[i] == 1000 + (i - bframes) * a.pts_step,
              "packet %d: dts %lld, expected %lld (bframes %d)", i,
              (long long)a.dts[i], (long long)(1000 + (i - bframes) * a.pts_step),
              bframes);
        CHECK(a.dts[i] == b.dts[i] && a.pts[i] == b.pts[i], "timestamps differ");
    }
    if (bframes) {
        int reordered = 0;
        for (i = 1; i < a.n; i++)
            reordered |= a.pts[i] < a.pts[i - 1];
        CHECK(reordered, "no picture came out of display order");
    }
    free(a.out.d);
    free(b.out.d);
}

static void *thread_run(void *arg)
{
    encode_clip((run *)arg);
    return NULL;
}

/* the kinds of encoder run side by side: CQP with B pictures, CRF without,
 * and (with a 10-bit build) a 10-bit CRF one -- the two depths share no
 * state either */
#ifdef AEG_HAVE_10BIT
enum { NK = 3 };
#else
enum { NK = 2 };
#endif

static void check_threads(void)
{
    enum { NT = 2 };
    run ref[NK], r[NK * NT];
    pthread_t th[NK * NT];
    int i, k;

    for (k = 0; k < NK; k++) {
        memset(&ref[k], 0, sizeof(ref[k]));
        base_params(&ref[k].p, 2);
        ref[k].pts_step = 1;
        if (k) {
            ref[k].p.rc_mode = AEG_RC_CRF;
            ref[k].p.crf = 25.0;
            ref[k].p.bframes = k == 1 ? 0 : 2;
        }
        ref[k].p.bit_depth = k == 2 ? 10 : 8;
        encode_clip(&ref[k]);
        CHECK(ref[k].rc == 0 && ref[k].n == NF, "reference %d", k);
    }
    for (i = 0; i < NK * NT; i++) {
        r[i] = ref[i % NK];
        memset(&r[i].out, 0, sizeof(r[i].out));
        pthread_create(&th[i], NULL, thread_run, &r[i]);
    }
    for (i = 0; i < NK * NT; i++) {
        pthread_join(th[i], NULL);
        CHECK(r[i].rc == 0 && r[i].out.n == ref[i % NK].out.n &&
              !memcmp(r[i].out.d, ref[i % NK].out.d, r[i].out.n),
              "thread %d: %zu bytes, alone %zu", i, r[i].out.n,
              ref[i % NK].out.n);
        free(r[i].out.d);
    }
    for (k = 0; k < NK; k++)
        free(ref[k].out.d);
}

#ifdef AEG_HAVE_10BIT
/* The SPS's profile_idc: the byte after the first NAL header of type 7. */
static int sps_profile(const uint8_t *d, size_t n)
{
    size_t i;
    for (i = 0; i + 4 < n; i++)
        if (d[i] == 0 && d[i + 1] == 0 && d[i + 2] == 1 && (d[i + 3] & 31) == 7)
            return d[i + 4];
    return -1;
}

/* Sum of squared differences of a W x H plane of 16-bit samples (stride in
 * bytes) against a reference of 16-bit samples. */
static double ssd16(const uint8_t *pl, int stride, const uint16_t *ref)
{
    double s = 0;
    int x, y;
    for (y = 0; y < H; y++)
        for (x = 0; x < W; x++) {
            double d = (double)((const uint16_t *)(pl + (size_t)y * stride))[x] -
                       ref[y * W + x];
            s += d * d;
        }
    return s;
}
#endif

static void check_bit_depth(void)
{
    aeg_params p;
    aeg_encoder *e;
    int d;

    aeg_params_default(&p);
    CHECK(p.bit_depth == 8, "default bit_depth %d", p.bit_depth);
    for (d = 0; d < 4; d++) {
        static const int bad[4] = { 1, 9, 12, -8 };
        base_params(&p, 1);
        p.bit_depth = bad[d];
        e = aeg_encoder_open(&p);
        CHECK(e == NULL && strstr(aeg_last_error(), "bit_depth"),
              "bit_depth %d: opened ('%s')", bad[d], aeg_last_error());
        aeg_encoder_close(e);
    }
    base_params(&p, 1);
    p.bit_depth = 0;                    /* 0 = 8 */
    e = aeg_encoder_open(&p);
    CHECK(e != NULL, "bit_depth 0: %s", aeg_last_error());
    if (e) {
        aeg_params q;
        CHECK(aeg_encoder_get_params(e, &q) == AEG_OK && q.bit_depth == 8,
              "bit_depth 0 resolves to %d", q.bit_depth);
        aeg_encoder_close(e);
    }
#ifdef AEG_HAVE_10BIT
    {
        aeg_params q;
        aeg_packet pkt;
        aeg_encoder *e8;
        const uint8_t *pl[3];
        int st[3], k, x, y, n8 = 0, n10 = 0, rc;
        uint16_t *src = (uint16_t *)malloc(sizeof(uint16_t) * W * H);
        uint8_t *img = (uint8_t *)malloc((size_t)W * H * 3 * 2);
        uint8_t *f8 = (uint8_t *)malloc(frame_bytes());
        double ssd8 = 0, ssd10 = 0, maxv = 0;

        base_params(&p, 2);
        p.bit_depth = 10;
        e = aeg_encoder_open(&p);
        CHECK(e != NULL, "open 10-bit: %s", aeg_last_error());
        if (!e)
            goto out;
        CHECK(aeg_encoder_get_params(e, &q) == AEG_OK && q.bit_depth == 10,
              "get_params: bit_depth %d", q.bit_depth);
        CHECK(aeg_encoder_headers(e, &pkt) == AEG_OK &&
              sps_profile(pkt.data, pkt.size) == 110,
              "10-bit SPS profile_idc %d, expected 110 (High 10)",
              sps_profile(pkt.data, pkt.size));
        aeg_encoder_close(e);

        /* A picture with detail below the 8-bit step: a 10-bit ramp plus
         * texture.  Coded nearly losslessly at both depths, the 10-bit
         * encoder has to come much closer to it than the 8-bit one can. */
        for (y = 0; y < H; y++)
            for (x = 0; x < W; x++)
                src[y * W + x] = (uint16_t)(200 + x * 3 + y + ((x * y) & 7));
        {
            uint16_t *Y = (uint16_t *)img;
            uint16_t *U = Y + W * H, *V = U + CW * CH;
            for (k = 0; k < W * H; k++)
                Y[k] = src[k];
            for (k = 0; k < CW * CH; k++)
                U[k] = V[k] = 512;
            for (k = 0; k < W * H; k++)     /* rounded, for the 8-bit run */
                f8[k] = (uint8_t)((src[k] + 2) >> 2);
            memset(f8 + W * H, 128, 2 * (size_t)CW * CH);
        }
        for (k = 0; k < 2; k++) {
            aeg_image im;
            memset(&im, 0, sizeof(im));
            im.csp = AEG_CSP_I420;
            im.depth = 10;
            im.plane[0] = img;
            im.plane[1] = img + 2 * W * H;
            im.plane[2] = img + 2 * (W * H + CW * CH);
            im.stride[0] = 2 * W;
            im.stride[1] = im.stride[2] = 2 * CW;
            base_params(&p, 2);
            p.bframes = 0;
            p.qp = 0;
            p.bit_depth = k ? 10 : 8;
            e = aeg_encoder_open(&p);
            CHECK(e != NULL, "open: %s", aeg_last_error());
            if (!e)
                continue;
            rc = k ? aeg_encoder_encode_image(e, &im, 0, &pkt)
                   : aeg_encoder_encode(e, f8, frame_bytes(), 0, &pkt);
            if (rc == AEG_EAGAIN)
                rc = aeg_encoder_flush(e, &pkt);
            CHECK(rc == AEG_OK, "%d-bit encode: %d", k ? 10 : 8, rc);
            if (rc == AEG_OK && aeg_encoder_recon(e, pl, st, NULL) == AEG_OK) {
                if (k) {
                    CHECK(st[0] >= 2 * W && st[1] >= 2 * CW && !(st[0] & 1),
                          "10-bit recon strides %d %d", st[0], st[1]);
                    for (y = 0; y < H; y++)
                        for (x = 0; x < W; x++) {
                            int v = ((const uint16_t *)(pl[0] + (size_t)y * st[0]))[x];
                            if (v > maxv)
                                maxv = v;
                        }
                    ssd10 = ssd16(pl[0], st[0], src);
                    n10 = 1;
                } else {
                    double s = 0;
                    for (y = 0; y < H; y++)
                        for (x = 0; x < W; x++) {
                            double dd = (double)(pl[0][(size_t)y * st[0] + x] << 2) -
                                        src[y * W + x];
                            s += dd * dd;
                        }
                    ssd8 = s;
                    n8 = 1;
                }
            }
            aeg_encoder_close(e);
        }
        if (n8 && n10) {
            CHECK(maxv > 255 && maxv <= 1023, "10-bit recon peak %g", maxv);
            /* QP 0: the 8-bit stream cannot do better than its rounding
             * (MSE ~1.25 in 10-bit units); the 10-bit one quantises 4x
             * finer */
            CHECK(ssd10 * 2 < ssd8, "10-bit SSD %g not well below 8-bit %g",
                  ssd10, ssd8);
        }

        /* 10-bit samples above 1023 (the caller's error) are clipped: the
         * same stream as the picture with 1023 there */
        {
            buf o[2] = { { NULL, 0, 0 }, { NULL, 0, 0 } };
            uint16_t *Y = (uint16_t *)img;
            for (k = 0; k < 2; k++) {
                aeg_image im;
                for (y = 0; y < H; y++)
                    for (x = 0; x < W; x++)
                        Y[y * W + x] = (uint16_t)((x / 8 + y / 8) & 1
                                       ? (k ? 0xFFFF : 1023) : src[y * W + x]);
                memset(&im, 0, sizeof(im));
                im.csp = AEG_CSP_I420;
                im.depth = 10;
                im.plane[0] = img;
                im.plane[1] = img + 2 * W * H;
                im.plane[2] = img + 2 * (W * H + CW * CH);
                im.stride[0] = 2 * W;
                im.stride[1] = im.stride[2] = 2 * CW;
                base_params(&p, 2);
                p.qp = 4;
                p.bit_depth = 10;
                e = aeg_encoder_open(&p);
                if (!e)
                    continue;
                if (aeg_encoder_encode_image(e, &im, 0, &pkt) == AEG_OK)
                    put(&o[k], pkt.data, pkt.size);
                while (aeg_encoder_flush(e, &pkt) == AEG_OK)
                    put(&o[k], pkt.data, pkt.size);
                aeg_encoder_close(e);
            }
            CHECK(o[0].n > 0 && o[0].n == o[1].n && !memcmp(o[0].d, o[1].d, o[0].n),
                  "10-bit samples above 1023 not clipped (%zu vs %zu bytes)",
                  o[1].n, o[0].n);
            free(o[0].d);
            free(o[1].d);
        }

        /* 8-bit and 10-bit encoders open at once, coding alternately */
        base_params(&p, 2);
        e8 = aeg_encoder_open(&p);
        p.bit_depth = 10;
        e = aeg_encoder_open(&p);
        CHECK(e8 && e, "open both: %s", aeg_last_error());
        if (e8 && e) {
            run ref8;
            buf o8 = { NULL, 0, 0 };
            memset(&ref8, 0, sizeof(ref8));
            base_params(&ref8.p, 2);
            ref8.pts_step = 1;
            encode_clip(&ref8);
            for (k = 0; k < NF; k++) {
                const uint8_t *fr = clip + frame_bytes() * k;
                if (aeg_encoder_encode(e8, fr, frame_bytes(), 1000 + k, &pkt) == AEG_OK)
                    put(&o8, pkt.data, pkt.size);
                aeg_encoder_encode(e, fr, frame_bytes(), 1000 + k, &pkt);
            }
            while (aeg_encoder_flush(e8, &pkt) == AEG_OK)
                put(&o8, pkt.data, pkt.size);
            CHECK(o8.n == ref8.out.n && !memcmp(o8.d, ref8.out.d, o8.n),
                  "8-bit stream changed by a 10-bit encoder beside it");
            free(o8.d);
            free(ref8.out.d);
        }
        aeg_encoder_close(e8);
        aeg_encoder_close(e);
    out:
        free(src);
        free(img);
        free(f8);
    }
#else
    base_params(&p, 1);
    p.bit_depth = 10;
    e = aeg_encoder_open(&p);
    CHECK(e == NULL && strstr(aeg_last_error(), "bit_depth"),
          "10-bit opened in a library built without it");
    aeg_encoder_close(e);
#endif
}

int main(void)
{
    int old = aeg_get_log_level();
    make_clip();
    check_version_and_log();
    check_errors();
    check_params();
    check_headers(0);
    check_headers(1);
    aeg_set_log_level(-1);
    check_planes_and_dts(3, AEG_RC_CQP, 0);
    check_planes_and_dts(0, AEG_RC_CRF, 0);
    check_planes_and_dts(2, AEG_RC_CRF, 1);
    check_images(8);
#ifdef AEG_HAVE_10BIT
    check_images(10);
#endif
    check_bit_depth();
    check_threads();
    aeg_set_log_level(old);
    free(clip);
    if (failures) {
        printf("api_check: %d failure(s)\n", failures);
        return 1;
    }
    printf("api_check: all checks passed\n");
    return 0;
}
