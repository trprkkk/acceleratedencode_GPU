/* Fuzz target: the parsers of aegenc that read files -- the YUV4MPEG2
 * demuxer, the --pass 2 statistics file, the frame-rate syntax -- with what
 * they produce fed on into an encoder as aegenc would.
 *
 * The CLI is included whole (its parsers are static) with main renamed.
 * The first byte picks the parser; the rest is the file. */
#define _GNU_SOURCE
#include <stdio.h>
#define main aegenc_cli_main
#include "../../cli/aegenc.c"
#undef main
#include "fuzz.h"

static FILE *mem_file(const uint8_t *data, size_t size)
{
    /* fmemopen() of 0 bytes is NULL on some C libraries */
    static const uint8_t empty[1] = {0};
    return fmemopen((void *)(size ? data : empty), size ? size : 1, "r");
}

static void fuzz_y4m(const uint8_t *data, size_t size, int force)
{
    input_t in;
    FILE *f = mem_file(data, size);
    aeg_image layout;
    size_t off[3], fsize;
    uint8_t *frame;
    aeg_params p;
    aeg_encoder *e = NULL;
    aeg_packet pkt;
    int csp, depth, k, w, h, guard;

    if (!f)
        return;
    if (in_open(&in, f, force) < 0 || !in.y4m) {
        fclose(f);
        return;
    }
    csp = in.csp >= 0 ? in.csp : AEG_CSP_I420;
    depth = in.depth > 0 ? in.depth : 8;
    w = in.w;
    h = in.h;
    /* the header's size is the caller's to bound: aegenc reads frames of
     * whatever it says; the fuzzer only codes small ones */
    if (w > 64 || h > 64) {
        fclose(f);
        return;
    }
    fsize = frame_layout(csp, depth, w, h, &layout, off);
    frame = (uint8_t *)malloc(fsize ? fsize : 1);
    if (!frame) {
        fclose(f);
        return;
    }
    aeg_params_default(&p);
    p.width = w;
    p.height = h;
    p.preset = 0;
    p.backend = AEG_BACKEND_CPU;
    p.frame_threads = 1;
    p.bit_depth = depth > 8 ? 10 : 8;
    p.interlaced = in.interlaced;
    if (in.fps_num > 0 && in.fps_den > 0) {
        p.fps_num = in.fps_num;
        p.fps_den = in.fps_den;
    }
    p.sar_w = in.sar_w;
    p.sar_h = in.sar_h;
    p.fullrange = in.fullrange;
    aeg_set_log_level(-1);
    e = aeg_encoder_open(&p);
    for (k = 0; k < 4 && in_frame(&in, frame, fsize) == 1; k++)
        if (e)
            encode_frame(e, frame, fsize, &layout, off, k, &pkt);
    if (e) {
        for (guard = 0; guard < 64 && aeg_encoder_flush(e, &pkt) == AEG_OK;
             guard++)
            ;
        aeg_encoder_close(e);
    }
    free(frame);
    fclose(f);
}

static void fuzz_stats(const uint8_t *data, size_t size)
{
    FILE *f = mem_file(data, size);
    aeg_pass1_frame *pass1 = NULL;
    int n = 0, cap = 0, bframes = -1, k, guard;
    aeg_params p;
    aeg_encoder *e;
    aeg_packet pkt;
    uint8_t frame[32 * 32 * 3 / 2];

    if (!f)
        return;
    if (stats_parse(f, "fuzz", 26, &pass1, &n, &cap, &bframes) < 0) {
        fclose(f);
        return;
    }
    fclose(f);
    if (n <= 0) {
        free(pass1);
        return;
    }
    if (n > cap) {
        fprintf(stderr, "fuzz_cli: %d pictures in a table of %d\n", n, cap);
        abort();
    }
    /* what aegenc --pass 2 does with it */
    aeg_set_log_level(-1);
    aeg_params_default(&p);
    p.width = p.height = 32;
    p.preset = 0;
    p.backend = AEG_BACKEND_CPU;
    p.frame_threads = 1;
    p.rc_mode = AEG_RC_ABR;
    p.bitrate_kbps = 300;
    p.bframes = bframes >= 0 ? bframes : 0;
    e = aeg_encoder_open(&p);
    if (e) {
        if (aeg_encoder_set_pass1(e, pass1, n) == AEG_OK) {
            for (k = 0; k < 6; k++) {
                memset(frame, 16 + 9 * k, sizeof(frame));
                if (aeg_encoder_encode(e, frame, sizeof(frame), k, &pkt) < 0 &&
                    aeg_last_error()[0] == 0)
                    break;
            }
            for (guard = 0; guard < 64 && aeg_encoder_flush(e, &pkt) == AEG_OK;
                 guard++)
                ;
        }
        aeg_encoder_close(e);
    }
    free(pass1);
}

static void fuzz_rate(const uint8_t *data, size_t size)
{
    char s[64];
    int num = 0, den = 0;
    size_t n = size < sizeof(s) - 1 ? size : sizeof(s) - 1;
    memcpy(s, data, n);
    s[n] = 0;
    if (parse_rate(s, &num, &den) == 0 && (num <= 0 || den <= 0)) {
        fprintf(stderr, "fuzz_cli: rate '%s' parsed as %d/%d\n", s, num, den);
        abort();
    }
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    if (size < 1)
        return 0;
    switch (data[0] % 5) {
    case 0: fuzz_y4m(data + 1, size - 1, 0); break;
    case 1: fuzz_y4m(data + 1, size - 1, 2); break;
    case 2: fuzz_stats(data + 1, size - 1); break;
    case 3: fuzz_rate(data + 1, size - 1); break;
    default: fuzz_y4m(data + 1, size - 1, 0); break;
    }
    return 0;
}
