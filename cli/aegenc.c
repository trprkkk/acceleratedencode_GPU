/*
 * aegenc - command line front end.
 *
 *   aegenc -w 1280 -h 720 -q 23 -i in.yuv -o out.264
 *
 * Input is raw I420 (YUV 4:2:0, 8 bit, planar); output is an H.264 Annex-B
 * elementary stream.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include "aeg/aeg.h"

static void usage(void)
{
    printf(
"aegenc - software H.264 encoder with CUDA/OpenCL accelerated stages\n"
"\n"
"usage: aegenc -w W -h H [-i in.yuv] [-o out.264] [options]\n"
"\n"
"  -w, -h <int>     frame size (multiples of 16; the encoder rounds down)\n"
"  -i <file>        input raw I420 stream (default: stdin)\n"
"  -o <file>        output Annex-B stream (default: stdout)\n"
"  -q <int>         QP 0..51 (default 23)\n"
"  -f <int>         frame rate (default 30)\n"
"  -g <int>         GOP size, 0 = only the first frame is IDR (default 250)\n"
"  -n <int>         encode at most n frames\n"
"  -b <backend>     auto | cpu | opencl | cuda (default auto)\n"
"  -s <stage=be>    force one stage: est,sad,inter,tq,iq,deblock,intra,recon,psnr\n"
"  -p <preset>      0=ultrafast 1=fast 2=medium 3=slow 4=veryslow\n"
"  --no-deblock     disable the in-loop deblocking filter\n"
"  --range <int>    integer motion search range (default: from preset)\n"
"  --subme <int>    subpel refinement 0..2 (default 1)\n"
"  --bitrate <kbps> average-bitrate rate control (ABR) instead of constant QP\n"
"  --aq <n>         adaptive quantisation strength in tenths (0 = off,\n"
"                   default 10 = x264 aq-strength 1.0)\n"
"  --slices <n>     slices per picture (0/1 = one slice; n>1 cuts the\n"
"                   picture into n row ranges, one NAL each)\n"
"  -t <int>         slice threads (0 = auto, 1 = serial; -t N without\n"
"                   --slices uses one slice per thread)\n"
"  --partitions <n> inter partition mask: 1=16x16 2=16x8 4=8x16\n"
"                   8=P_8x8 (8x8 with 8x4/4x8/4x4 sub-partitions)\n"
"                   (default 15; P_8x8 is skipped by preset 0)\n"
"  -v <int>         verbosity (0 quiet, 1 normal, 2 debug)\n"
"  --list-devices   enumerate the compute devices and exit\n"
"  --calibrate      measure the accelerator launch overhead and exit\n"
"  --bench          encode three times (auto/cpu/accelerator) and report\n"
"\n"
"Exit status: 0 on success.\n");
}

static int parse_backend(const char *s)
{
    if (!strcmp(s, "cpu")) return AEG_BACKEND_CPU;
    if (!strcmp(s, "opencl") || !strcmp(s, "cl")) return AEG_BACKEND_OPENCL;
    if (!strcmp(s, "cuda")) return AEG_BACKEND_CUDA;
    return AEG_BACKEND_AUTO;
}

static int parse_stage(const char *s)
{
    static const char *names[] = {
        "est", "sad", "inter", "tq", "iq", "deblock", "intra", "recon", "psnr"
    };
    int i;
    for (i = 0; i < AEG_STAGE_COUNT; i++)
        if (!strcmp(s, names[i]))
            return i;
    return -1;
}

static void list_devices(void)
{
    aeg_device_info info[16];
    int n = aeg_query_devices(AEG_BACKEND_AUTO, info, 16);
    int i;
    printf("compute devices:\n");
    for (i = 0; i < n; i++)
        printf("  [%d] %-7s %-40s %s%s\n", i, aeg_backend_name(info[i].backend),
               info[i].name, info[i].vendor,
               info[i].is_gpu ? "" : " (no discrete GPU)");
    if (n <= 1)
        printf("  (no CUDA/OpenCL device found; the CPU path is used)\n");
}

typedef struct {
    int64_t frames, bytes;
    int64_t ns;
    double psnr;
    aeg_stats st;
} run_result;

int main(int argc, char **argv)
{
    aeg_params p;
    aeg_encoder *enc;
    FILE *in = stdin, *out = stdout;
    int width = 0, height = 0, qp = 23, fps = 30, gop = 250, max_frames = 0;
    int bitrate = 0, aq = -1;
    int backend = AEG_BACKEND_AUTO, verbose = 1, bench = 0;
    int range = 0, subme = -1, partitions = -1;
    const char *inpath = NULL, *outpath = NULL, *dumppath = NULL;
    uint8_t *frame;
    size_t fsize;
    int64_t pts = 0;
    aeg_stats st;
    int i, ret = 0;

    aeg_params_default(&p);

    for (i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-w")) width = atoi(argv[++i]);
        else if (!strcmp(a, "-h")) height = atoi(argv[++i]);
        else if (!strcmp(a, "-i")) inpath = argv[++i];
        else if (!strcmp(a, "-o")) outpath = argv[++i];
        else if (!strcmp(a, "-q")) qp = atoi(argv[++i]);
        else if (!strcmp(a, "-f")) fps = atoi(argv[++i]);
        else if (!strcmp(a, "-g")) gop = atoi(argv[++i]);
        else if (!strcmp(a, "-n")) max_frames = atoi(argv[++i]);
        else if (!strcmp(a, "-b")) backend = parse_backend(argv[++i]);
        else if (!strcmp(a, "-p")) p.preset = atoi(argv[++i]);
        else if (!strcmp(a, "--slices")) p.slices = atoi(argv[++i]);
        else if (!strcmp(a, "-t") || !strcmp(a, "--threads"))
            p.threads = atoi(argv[++i]);
        else if (!strcmp(a, "-v")) verbose = atoi(argv[++i]);
        else if (!strcmp(a, "--range")) range = atoi(argv[++i]);
        else if (!strcmp(a, "--subme")) subme = atoi(argv[++i]);
        else if (!strcmp(a, "--partitions")) partitions = atoi(argv[++i]);
        else if (!strcmp(a, "--bitrate")) bitrate = atoi(argv[++i]);
        else if (!strcmp(a, "--aq")) aq = atoi(argv[++i]);
        else if (!strcmp(a, "--no-deblock")) p.deblock = 0;
        else if (!strcmp(a, "--list-devices")) { list_devices(); return 0; }
        else if (!strcmp(a, "--calibrate")) {
            int r = aeg_calibrate(backend);
            printf("calibration: %s\n", r == 0 ? "ok" : "accelerator unavailable");
            return r == 0 ? 0 : 1;
        } else if (!strcmp(a, "--bench")) bench = 1;
        else if (!strcmp(a, "--dump-recon")) dumppath = argv[++i];
        else if (!strcmp(a, "-s")) {
            char *eq;
            char *arg = argv[++i];
            int stg;
            eq = strchr(arg, '=');
            if (!eq) { fprintf(stderr, "bad -s argument\n"); return 2; }
            *eq = 0;
            stg = parse_stage(arg);
            if (stg < 0) { fprintf(stderr, "unknown stage '%s'\n", arg); return 2; }
            p.stage_backend[stg] = parse_backend(eq + 1);
        } else if (!strcmp(a, "--help") || !strcmp(a, "-help")) { usage(); return 0; }
        else {
            fprintf(stderr, "unknown option '%s'\n", a);
            usage();
            return 2;
        }
    }

    if (!width || !height) {
        fprintf(stderr, "error: -w and -h are required\n");
        return 2;
    }
    if (inpath) {
        in = fopen(inpath, "rb");
        if (!in) { perror(inpath); return 1; }
    }
    if (outpath) {
        out = fopen(outpath, "wb");
        if (!out) { perror(outpath); return 1; }
    }

    p.width = width;
    p.height = height;
    p.qp = qp;
    p.fps_num = fps;
    p.fps_den = 1;
    p.gop_size = gop;
    p.bitrate_kbps = bitrate;
    if (aq >= 0) { p.aq_mode = aq > 0; p.aq_strength = aq; }
    p.rc_mode = bitrate > 0 ? AEG_RC_ABR : AEG_RC_CQP;
    p.backend = backend;
    p.verbose = verbose;
    if (range) p.search_range = range;
    if (subme >= 0) p.subme = subme;
    if (partitions >= 0) p.partitions = partitions;
    aeg_log_level = verbose;

    fsize = (size_t)width * height * 3 / 2;
    frame = (uint8_t *)malloc(fsize);
    if (!frame) { fprintf(stderr, "out of memory\n"); return 1; }

    if (bench) {
        int64_t t[3];
        int be[3] = { AEG_BACKEND_CPU, AEG_BACKEND_AUTO };
        int k;
        for (k = 0; k < 2; k++) {
            aeg_params q = p;
            aeg_encoder *e;
            int64_t t0, tot = 0, bytes = 0, n = 0;
            q.backend = be[k];
            e = aeg_encoder_open(&q);
            if (!e) { fprintf(stderr, "open failed\n"); return 1; }
            fseek(in, 0, SEEK_SET);
            t0 = aeg_now_ns();
            while (n < (max_frames ? max_frames : 30) &&
                   fread(frame, 1, fsize, in) == fsize) {
                aeg_packet pkt;
                if (aeg_encoder_encode(e, frame, fsize, pts++, &pkt) != 0) break;
                bytes += (int64_t)pkt.size;
                n++;
            }
            tot = aeg_now_ns() - t0;
            t[k] = tot;
            (void)bytes;
            if (verbose)
                printf("%-6s %6" PRId64 " frames %8.2f ms/frame %8.0f kb/s\n",
                       aeg_backend_name(q.backend), n, tot / 1e6 / (n ? n : 1),
                       n ? bytes * 8.0 * fps / (tot / 1e9) / 1000.0 : 0.0);
            aeg_encoder_close(e);
        }
        if (t[1] > 0 && t[0] > 0)
            printf("speedup (auto vs cpu): %.2fx\n", (double)t[0] / (double)t[1]);
        free(frame);
        return 0;
    }

    enc = aeg_encoder_open(&p);
    if (!enc) {
        fprintf(stderr, "aeg_encoder_open failed: %s\n", aeg_last_error());
        free(frame);
        return 1;
    }

    {
        int64_t t0 = aeg_now_ns();
        int n = 0;
        while (fread(frame, 1, fsize, in) == fsize) {
            aeg_packet pkt;
            int r = aeg_encoder_encode(enc, frame, fsize, pts++, &pkt);
            if (r != 0) {
                fprintf(stderr, "encode failed: %s\n", aeg_last_error());
                ret = 1;
                break;
            }
            fwrite(pkt.data, 1, pkt.size, out);
            if (dumppath) {
                const uint8_t *pl[3];
                int str[3], y;
                FILE *df = fopen(dumppath, n ? "ab" : "wb");
                if (df) {
                    aeg_encoder_recon(enc, pl, str, NULL);
                    for (y = 0; y < height; y++)
                        fwrite(pl[0] + (size_t)y * str[0], 1, (size_t)width, df);
                    /* planar I420, same layout as the decoder dumps so the two
                     * can be compared byte for byte */
                    for (y = 0; y < height / 2; y++)
                        fwrite(pl[1] + (size_t)y * str[1], 1, (size_t)width / 2, df);
                    for (y = 0; y < height / 2; y++)
                        fwrite(pl[2] + (size_t)y * str[2], 1, (size_t)width / 2, df);
                    fclose(df);
                }
            }
            n++;
            if (max_frames && n >= max_frames)
                break;
        }
        if (verbose) {
            double secs = (aeg_now_ns() - t0) / 1e9;
            aeg_encoder_stats(enc, &st);
            printf("encoded %d frames in %.3f s (%.2f fps), %" PRId64 " bytes\n",
                   n, secs, secs > 0 ? n / secs : 0.0, st.bytes);
            /* the stream's rate is a property of the file and the frame rate,
             * not of how fast this machine happened to encode it */
            printf("bitrate %.1f kb/s at %d fps, PSNR %.2f dB (Y+U+V average)\n",
                   st.frames > 0
                       ? st.bytes * 8.0 * fps / (double)st.frames / 1000.0
                       : 0.0,
                   fps, st.psnr_avg);
            for (i = 0; i < AEG_STAGE_COUNT; i++)
                if (st.stage_ns[i] > 0)
                    printf("  stage %-12s %8.1f ms  %s\n", aeg_stage_name(i),
                           st.stage_ns[i] / 1e6,
                           st.stage_use_gpu[i] ? "gpu" : "cpu");
        }
    }

    aeg_encoder_close(enc);
    free(frame);
    if (in != stdin) fclose(in);
    if (out != stdout) fclose(out);
    return ret;
}
