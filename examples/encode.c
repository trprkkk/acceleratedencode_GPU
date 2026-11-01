/*
 * encode.c - the smallest complete use of libaeg, through its public header
 * only.
 *
 *   encode_example in.yuv WIDTH HEIGHT out.264 [crf] [fps]
 *
 * Reads raw I420 frames, encodes them at constant quality and writes an
 * Annex-B H.264 stream.  Along the way it shows the parts of the API a real
 * application needs:
 *
 *  - routing the library's messages into the application's own log;
 *  - aeg_encoder_get_params(): what the encoder resolved the options to;
 *  - aeg_encoder_headers(): the SPS/PPS a container keeps out of band;
 *  - aeg_encoder_encode_planes(): input from a frame with padded rows;
 *  - draining with aeg_encoder_flush(), and pts/dts for a muxer.
 *
 * Build against an installed copy:
 *   cc encode.c $(pkg-config --cflags --libs aeg) -o encode_example
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <aeg/aeg.h>

static void on_log(void *opaque, int level, const char *msg)
{
    static const char *names[] = { "error", "warning", "info", "debug" };
    fprintf((FILE *)opaque, "[libaeg %s] %s\n",
            level >= 0 && level <= 3 ? names[level] : "?", msg);
}

static int write_packet(FILE *out, const aeg_packet *pkt, long *frames)
{
    if (fwrite(pkt->data, 1, pkt->size, out) != pkt->size)
        return -1;
    printf("packet %4ld: %c pts %4lld dts %4lld %7zu bytes qp %.2f\n",
           *frames, "IPBb"[pkt->type & 3], (long long)pkt->pts,
           (long long)pkt->dts, pkt->size, pkt->qp_avg);
    (*frames)++;
    return 0;
}

int main(int argc, char **argv)
{
    aeg_params p, used;
    aeg_encoder *enc;
    aeg_packet pkt;
    FILE *in, *out;
    uint8_t *frame;
    const uint8_t *plane[3];
    int stride[3], w, h, cw, ch, rc, y;
    long frames = 0, n = 0;

    if (argc < 5) {
        fprintf(stderr, "usage: %s in.yuv WIDTH HEIGHT out.264 [crf] [fps]\n",
                argv[0]);
        return 2;
    }
    w = atoi(argv[2]);
    h = atoi(argv[3]);
    cw = (w + 1) / 2;
    ch = (h + 1) / 2;
    in = fopen(argv[1], "rb");
    out = fopen(argv[4], "wb");
    if (!in || !out) {
        perror("open");
        return 1;
    }

    printf("libaeg %s (header %s)\n", aeg_version(), AEG_VERSION_STRING);
    aeg_set_log_callback(on_log, stderr);
    aeg_set_log_level(AEG_LOG_INFO);

    aeg_params_default(&p);
    p.width = w;
    p.height = h;
    p.rc_mode = AEG_RC_CRF;
    p.crf = argc > 5 ? atof(argv[5]) : 23.0;
    p.fps_num = argc > 6 ? atoi(argv[6]) : 30;
    p.fps_den = 1;
    p.preset = AEG_PRESET_MEDIUM;
    enc = aeg_encoder_open(&p);
    if (!enc) {
        fprintf(stderr, "aeg_encoder_open: %s\n", aeg_last_error());
        return 1;
    }

    aeg_encoder_get_params(enc, &used);
    printf("level %d.%d, %d B pictures, %s, %d reference(s), subme %d\n",
           used.level_idc / 10, used.level_idc % 10, used.bframes,
           used.cabac ? "CABAC" : "CAVLC", used.refs, used.subme);
    if (aeg_encoder_headers(enc, &pkt) == AEG_OK)
        printf("SPS/PPS: %zu bytes (for an avcC box; every IDR repeats "
               "them in-band)\n", pkt.size);

    /* A frame with padded rows, as a capture device or decoder would hand
     * it over: each plane's stride rounded up to 64 bytes. */
    stride[0] = (w + 63) & ~63;
    stride[1] = stride[2] = (cw + 63) & ~63;
    frame = (uint8_t *)calloc((size_t)stride[0] * h + 2 * (size_t)stride[1] * ch, 1);
    if (!frame)
        return 1;
    plane[0] = frame;
    plane[1] = frame + (size_t)stride[0] * h;
    plane[2] = plane[1] + (size_t)stride[1] * ch;

    for (;;) {
        int ok = 1;
        for (y = 0; y < h && ok; y++)
            ok = fread((uint8_t *)plane[0] + (size_t)y * stride[0], 1, (size_t)w, in) == (size_t)w;
        for (y = 0; y < 2 * ch && ok; y++)
            ok = fread((uint8_t *)plane[1 + y / ch] + (size_t)(y % ch) * stride[1], 1,
                       (size_t)cw, in) == (size_t)cw;
        if (!ok)
            break;
        rc = aeg_encoder_encode_planes(enc, plane, stride, n++, &pkt);
        if (rc == AEG_OK) {
            if (write_packet(out, &pkt, &frames))
                break;
        } else if (rc != AEG_EAGAIN) {
            fprintf(stderr, "encode: %s\n", aeg_last_error());
            break;
        }
    }
    /* the pictures still held for B-picture reordering */
    while ((rc = aeg_encoder_flush(enc, &pkt)) == AEG_OK)
        if (write_packet(out, &pkt, &frames))
            break;

    printf("%ld frames in, %ld packets out\n", n, frames);
    aeg_encoder_close(enc);
    aeg_set_log_callback(NULL, NULL);
    free(frame);
    fclose(in);
    fclose(out);
    return frames == n ? 0 : 1;
}
