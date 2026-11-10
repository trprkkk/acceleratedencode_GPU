/* Print aeg_stats' full accounting for one encode: stage_ns, coding_ns,
 * entropy_ns and the rest, which is what the "ceiling" question needs. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "aeg/aeg.h"

int main(int argc, char **argv)
{
    const char *clip = argc > 1 ? argv[1] : "/tmp/aegbench/src_1280x720_30.yuv";
    int w = argc > 2 ? atoi(argv[2]) : 1280;
    int h = argc > 3 ? atoi(argv[3]) : 720;
    int n = argc > 4 ? atoi(argv[4]) : 30;
    int preset = argc > 5 ? atoi(argv[5]) : 4;
    int qp = argc > 6 ? atoi(argv[6]) : 27;
    aeg_params p;
    aeg_encoder *e;
    aeg_stats st;
    aeg_image img;
    FILE *f = fopen(clip, "rb");
    size_t fs = (size_t)w * h * 3 / 2, got;
    uint8_t *buf;
    int i;

    if (!f) { printf("no clip\n"); return 1; }
    buf = malloc(fs * n);
    if (!buf) { printf("no memory\n"); return 1; }
    got = fread(buf, 1, fs * n, f);
    fclose(f);
    n = (int)(got / fs);
    if (n <= 0) { printf("clip too short\n"); return 1; }
    aeg_params_default(&p);
    p.width = w; p.height = h; p.fps_num = 30; p.fps_den = 1;
    p.preset = preset; p.qp = qp; p.verbose = 0;
    aeg_set_log_level(-1);
    e = aeg_encoder_open(&p);
    if (!e) { printf("open failed\n"); return 1; }
    memset(&img, 0, sizeof(img));
    img.csp = AEG_CSP_I420;
    img.depth = 8;
    img.plane[0] = buf;
    img.plane[1] = buf + (size_t)w * h;
    img.plane[2] = buf + (size_t)w * h + (size_t)w / 2 * h / 2;
    img.stride[0] = w;
    img.stride[1] = w / 2;
    img.stride[2] = w / 2;
    for (i = 0; i < n; i++) {
        aeg_packet pkt;
        int rc = aeg_encoder_encode_image(e, &img, (int64_t)i, &pkt);
        /* AEG_EAGAIN is normal: with a lookahead the packets come later */
        if (rc == AEG_OK)
            aeg_packet_free(&pkt);
        else if (rc != AEG_EAGAIN) {
            fprintf(stderr, "picture %d: rc=%d\n", i, rc);
            break;
        }
        img.plane[0] = buf + (size_t)(i + 1) * fs;
        img.plane[1] = img.plane[0] + (size_t)w * h;
        img.plane[2] = img.plane[0] + (size_t)w * h + (size_t)w / 2 * h / 2;
    }
    /* drain: the lookahead means the pictures left are coded here */
    for (;;) {
        aeg_packet pkt;
        int rc = aeg_encoder_flush(e, &pkt);
        if (rc != AEG_OK)
            break;
        aeg_packet_free(&pkt);
    }
    aeg_encoder_stats(e, &st);
    aeg_encoder_close(e);
    printf("frames %lld  bytes %lld  wall in encode %.3f s (%.1f ms/frame)\n",
           (long long)st.frames, (long long)st.bytes, st.enc_ns / 1e9,
           st.enc_ns / 1e6 / (st.frames ? st.frames : 1));
    printf("  motion-est    %8.1f ms %s\n", st.stage_ns[0] / 1e6,
           st.stage_use_gpu[0] ? "(device)" : "");
    printf("  inter-pred    %8.1f ms %s\n", st.stage_ns[1] / 1e6,
           st.stage_use_gpu[1] ? "(device)" : "");
    printf("  transform     %8.1f ms %s\n", st.stage_ns[2] / 1e6,
           st.stage_use_gpu[2] ? "(device)" : "");
    printf("  iquant-idct   %8.1f ms %s\n", st.stage_ns[3] / 1e6,
           st.stage_use_gpu[3] ? "(device)" : "");
    printf("  deblock       %8.1f ms %s\n", st.stage_ns[4] / 1e6,
           st.stage_use_gpu[4] ? "(device)" : "");
    printf("  psnr          %8.1f ms %s\n", st.stage_ns[5] / 1e6,
           st.stage_use_gpu[5] ? "(device)" : "");
    printf("  coding_ns     %8.1f ms  (RD, gathers, reconstruction)\n",
           st.coding_ns / 1e6);
    printf("  entropy_ns    %8.1f ms  (CABAC/CAVLC)\n", st.entropy_ns / 1e6);
    {
        int64_t other = st.enc_ns - st.stage_ns[0] - st.stage_ns[1] -
                        st.stage_ns[2] - st.stage_ns[3] - st.stage_ns[4] -
                        st.stage_ns[5] - st.coding_ns - st.entropy_ns;
        printf("  the rest     %8.1f ms  (lookahead, rate control)\n", other / 1e6);
    }
    printf("  macroblocks %lld skipped %lld\n", (long long)st.macroblocks,
           (long long)st.skipped);
    return 0;
}
