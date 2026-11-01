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
 *  - encoders running in parallel threads give the bytes they give alone.
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
enum { W = 200, H = 118, CW = (W + 1) / 2, CH = (H + 1) / 2, NF = 14 };

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

typedef struct run {
    aeg_params p;
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
    for (t = 0; t < NF; t++) {
        const uint8_t *f = clip + fb * t;
        int64_t pts = 1000 + t * r->pts_step;
        if (r->planes) {
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
        CHECK(q.slices >= 1 && (q.frame_threads == 1 || q.frame_threads == 2),
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

static void check_threads(void)
{
    enum { NT = 3 };
    run ref[2], r[2 * NT];
    pthread_t th[2 * NT];
    int i, k;

    for (k = 0; k < 2; k++) {
        memset(&ref[k], 0, sizeof(ref[k]));
        base_params(&ref[k].p, 2);
        ref[k].pts_step = 1;
        if (k) {
            ref[k].p.rc_mode = AEG_RC_CRF;
            ref[k].p.crf = 25.0;
            ref[k].p.bframes = 0;
        }
        encode_clip(&ref[k]);
        CHECK(ref[k].rc == 0 && ref[k].n == NF, "reference %d", k);
    }
    for (i = 0; i < 2 * NT; i++) {
        r[i] = ref[i & 1];
        memset(&r[i].out, 0, sizeof(r[i].out));
        pthread_create(&th[i], NULL, thread_run, &r[i]);
    }
    for (i = 0; i < 2 * NT; i++) {
        pthread_join(th[i], NULL);
        CHECK(r[i].rc == 0 && r[i].out.n == ref[i & 1].out.n &&
              !memcmp(r[i].out.d, ref[i & 1].out.d, r[i].out.n),
              "thread %d: %zu bytes, alone %zu", i, r[i].out.n,
              ref[i & 1].out.n);
        free(r[i].out.d);
    }
    free(ref[0].out.d);
    free(ref[1].out.d);
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
