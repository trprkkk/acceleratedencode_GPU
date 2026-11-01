/*
 * aegenc - command line front end.
 *
 *   aegenc -w 1280 -h 720 -q 23 -i in.yuv -o out.264
 *
 * Input is raw I420 (YUV 4:2:0, 8 bit, planar); output is an H.264 Annex-B
 * elementary stream.
 */
#include <stdio.h>
#include <math.h>
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
"  --scenecut <pct> insert an IDR when a scene change is detected (0 = off,\n"
"                   default 120: inter prediction 20%% worse than intra)\n"
"  --pass <1|2>     two-pass rate control: pass 1 measures each picture, pass 2\n"
"                   spends the bits where they buy the most (needs --bitrate)\n"
"  --stats <file>   statistics file for --pass (default: <output>.stats)\n"
"  --qcomp <pct>    complexity-to-bits exponent of pass 2, in percent\n"
"                   (default 60: bits ~ complexity^0.6)\n"
"  -n <int>         encode at most n frames\n"
"  -b <backend>     auto | cpu | opencl | cuda (default auto)\n"
"  -s <stage=be>    force one stage: est,sad,inter,tq,iq,deblock,intra,recon,psnr\n"
"  -p <preset>      0=ultrafast 1=fast 2=medium 3=slow 4=veryslow\n"
"  --no-8x8dct      use only 4x4 luma transforms (drops High profile)\n"
"  --no-cabac       code the macroblock layer with CAVLC instead of CABAC\n"
"  --no-deblock     disable the in-loop deblocking filter\n"
"  --range <int>    integer motion search range (default: from preset)\n"
"  --subme <int>    subpel refinement 0..2 (default 1)\n"
"  --refine-mv <n>  0 = off, 1 = four quarter-pel neighbours of the winning\n"
"                   vector (the cross), 2 = the four of all eight with the\n"
"                   lowest SATD, 3 = all eight (rate-distortion refined,\n"
"                   slower; default 0, preset 3 uses 1 and preset 4 uses 2)\n"
"  --refine-cand <n> refine every inter candidate's vectors, not only the\n"
"                   winner's: 0 = off (default), 1 = on (costs about a third of\n"
"                   the encode time at preset 4 for 0.3-0.6 points of BD-rate)\n"
"  --b-pyramid <n>  1 = the middle B picture of a group is a reference for\n"
"                   the others (default with B pictures), 0 = off\n"
"  --weightp <n>    explicit weighted prediction of P pictures (fades),\n"
"                   0 = off, 1 = on (default from preset 2)\n"
"  --ref <n>        reference pictures for P pictures, 1 or 2 (default 2 from\n"
"                   preset 3 up, otherwise 1)\n"
"  --bframes <n>    B pictures between two anchors, 0..3 (default 3).  The\n"
"                   stream is then coded out of display order and the encoder\n"
"                   holds n pictures of lookahead, so the first packets appear\n"
"                   after n+1 frames; needs CABAC.  0 codes P pictures only.\n"
"  --b-adapt <n>    1 = --bframes is the most a group gets; the lookahead\n"
"                   picks each group's length from the content by x264's\n"
"                   fast rule (default below preset 3); 0 = always\n"
"                   --bframes B pictures; 2 = every split of the window\n"
"                   priced picture by picture (x264's trellis; default at\n"
"                   presets 3 and 4)\n"
"  --bqp <n>        QP added to every B picture (default 4, 2 with the\n"
"                   macroblock tree; the pyramid's reference B picture gets\n"
"                   the same offset)\n"
"  --mbtree <0|1>   macroblock tree: lower the QP of the blocks later\n"
"                   pictures predict from (default on from preset 1; needs\n"
"                   B pictures)\n"
"  --rc-lookahead <n> pictures the macroblock tree looks past the group\n"
"                   being coded (default 10/20/30/40 at presets 1-4, max 60)\n"
"  --iqp <n>        QP subtracted from I pictures at constant QP (default 3,\n"
"                   x264's ipratio 1.4)\n"
"  --direct <mode>  B direct prediction: spatial (default) or temporal\n"
"  --dump-recon <f> write the reconstruction (coding order, I420)\n"
"  --dump-order <f> write each packet's display index, one per line\n"
"  --trellis <n>    rate-distortion level selection on the luma residual:\n"
"                   0 = off, 1 = on (the default from preset 3 up: 0.15-2.4\n"
"                   points of BD-rate for 12-14%% of the encode time)\n"
"  --bitrate <kbps> average-bitrate rate control (ABR) instead of constant QP\n"
"  --vbv-maxrate <kbps>  VBV/HRD channel rate (implies ABR)\n"
"  --vbv-bufsize <kbits> VBV/HRD buffer size  (implies ABR)\n"
"  --aq <n>         adaptive quantisation strength in tenths (0 = off,\n"
"                   default 10 = x264 aq-strength 1.0)\n"
"  --slices <n>     slices per picture (0/1 = one slice; n>1 cuts the\n"
"                   picture into n row ranges, one NAL each)\n"
"  -t <int>         slice threads (0 = auto, 1 = serial; -t N without\n"
"                   --slices uses one slice per thread)\n"
"  --frame-threads <n> code n pictures at once, each following its\n"
"                   references row by row (same stream as serial);\n"
"                   1 = off, 0 = auto (CPUs + 1, max 8, unless -t 1;\n"
"                   B pictures, constant QP, one slice, CPU backend)\n"
"  --partitions <n> inter partition mask: 1=16x16 2=16x8 4=8x16\n"
"                   8=P_8x8  16=its 8x4/4x8/4x4 sub-partitions\n"
"                   (default 15; P_8x8 is skipped by preset 0, and so\n"
"                   are bits 1-3 for B pictures -- 16x8 and 8x16 are\n"
"                   the only B shapes so far)\n"
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

/* Two-pass rate control, second pass: turn the first pass's measurement of every
 * picture into that picture's QP bias.
 *
 * Pass 1 codes the stream at a fixed QP, so the bits it spends on a picture are
 * a measurement of how hard that picture is.  Scaling them to the target
 * (`bits_i` proportional to `bits_i`) would be exactly a constant QP shift --
 * the whole point of two passes is to spend *unevenly*.  The allocation is
 * therefore bits proportional to `complexity^qcomp` with qcomp < 1 (x264's
 * default 0.6): the hard pictures get a little less than proportional and the
 * easy ones a little more, which is the right direction because distortion per
 * bit falls faster in flat, well-predicted areas than in busy ones.
 *
 * The controller keeps the *rate*; this only says how the bits are shared out.
 * With `t_i ~ b_i^alpha` normalised to the same total, the QP of picture i
 * relative to a constant-QP stream is
 *
 *     QP_i - QP = 6 * (1 - alpha) * (log2 b_i - mean_j log2 b_j)
 *
 * so the bias is mean-zero over the stream and depends only on how unusual a
 * picture is, not on the target bitrate.  It is rounded to whole QP steps and
 * clipped to +-12. */
static void pass2_bias(const int64_t *bits, int n, int qcomp, int *bias_out)
{
    double mean = 0.0, keep = 6.0 * (1.0 - qcomp / 100.0);
    int i;

    if (n <= 0)
        return;
    for (i = 0; i < n; i++)
        mean += log2((double)(bits[i] > 0 ? bits[i] : 1));
    mean /= n;
    for (i = 0; i < n; i++) {
        double l = log2((double)(bits[i] > 0 ? bits[i] : 1));
        double b = keep * (l - mean);
        int bi = (int)(b > 0 ? b + 0.5 : b - 0.5);

        if (bi < -12) bi = -12;
        if (bi > 12) bi = 12;
        bias_out[i] = bi;
    }
}

/* Write one packet to the stream and, if a reconstruction dump was asked for,
 * append this picture's reconstruction to it.  The dump follows the *coding*
 * order when the stream is reordered by B pictures, which is the order the
 * packets come out in; a decoder that outputs in display order has to be
 * re-ordered before the two files are compared. */
/* --dump-order: the display index of every packet, one per line, in the order
 * the packets come out (= the order of the reconstruction dump). */
static FILE *g_order;

static void write_packet(FILE *out, const char *dumppath, aeg_encoder *enc,
                         const aeg_packet *pkt, int *first,
                         int width, int height)
{
    fwrite(pkt->data, 1, pkt->size, out);
    if (g_order)
        fprintf(g_order, "%lld\n", (long long)pkt->pts);
    if (dumppath) {
        const uint8_t *pl[3];
        int str[3], y;
        FILE *df = fopen(dumppath, *first ? "wb" : "ab");

        *first = 0;
        if (!df)
            return;
        aeg_encoder_recon(enc, pl, str, NULL);
        for (y = 0; y < height; y++)
            fwrite(pl[0] + (size_t)y * str[0], 1, (size_t)width, df);
        /* planar I420, same layout as the decoder dumps so the two can be
         * compared byte for byte */
        for (y = 0; y < height / 2; y++)
            fwrite(pl[1] + (size_t)y * str[1], 1, (size_t)width / 2, df);
        for (y = 0; y < height / 2; y++)
            fwrite(pl[2] + (size_t)y * str[2], 1, (size_t)width / 2, df);
        fclose(df);
    }
}

int main(int argc, char **argv)
{
    aeg_params p;
    aeg_encoder *enc;
    FILE *in = stdin, *out = stdout;
    int width = 0, height = 0, qp = 23, fps = 30, gop = 250, max_frames = 0;
    int bitrate = 0, aq = -1;
    int backend = AEG_BACKEND_AUTO, verbose = 1, bench = 0;
    int range = 0, subme = -1, partitions = -1, pass = 0, qcomp = 60;
    int refine_mv = -1, refine_cand = -1, refs = -1, weightp = -1, bpyramid = -1;
    int mbtree = -1, rclookahead = -1;
    int trellis = -1;
    int bframes = -1, bqp = -1000, iqp = -1000, direct = -1, badapt = -1;
    int64_t *pass1_bits = NULL;
    int pass1_n = 0;
    char stats_path[4096];
    const char *inpath = NULL, *outpath = NULL, *dumppath = NULL;
    const char *orderpath = NULL;
    uint8_t *frame;
    size_t fsize;
    int64_t pts = 0;
    int dump_first = 1;
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
        else if (!strcmp(a, "--frame-threads"))
            p.frame_threads = atoi(argv[++i]);
        else if (!strcmp(a, "--scenecut")) p.scenecut = atoi(argv[++i]);
        else if (!strcmp(a, "--pass")) pass = atoi(argv[++i]);
        else if (!strcmp(a, "--qcomp")) qcomp = atoi(argv[++i]);
        else if (!strcmp(a, "--stats")) {
            snprintf(stats_path, sizeof(stats_path), "%s", argv[++i]);
            stats_path[0] = stats_path[0];      /* keep -Wall quiet */
        }
        else if (!strcmp(a, "--vbv-maxrate"))
            p.vbv_maxrate_kbps = atoi(argv[++i]);
        else if (!strcmp(a, "--vbv-bufsize"))
            p.vbv_bufsize_kbits = atoi(argv[++i]);
        else if (!strcmp(a, "-v")) verbose = atoi(argv[++i]);
        else if (!strcmp(a, "--range")) range = atoi(argv[++i]);
        else if (!strcmp(a, "--subme")) subme = atoi(argv[++i]);
        else if (!strcmp(a, "--refine-mv")) refine_mv = atoi(argv[++i]);
        else if (!strcmp(a, "--refine-cand")) refine_cand = atoi(argv[++i]);
        else if (!strcmp(a, "--ref")) refs = atoi(argv[++i]);
        else if (!strcmp(a, "--weightp")) weightp = atoi(argv[++i]);
        else if (!strcmp(a, "--b-pyramid")) bpyramid = atoi(argv[++i]);
        else if (!strcmp(a, "--mbtree")) mbtree = atoi(argv[++i]);
        else if (!strcmp(a, "--rc-lookahead")) rclookahead = atoi(argv[++i]);
        else if (!strcmp(a, "--bframes")) bframes = atoi(argv[++i]);
        else if (!strcmp(a, "--bqp")) bqp = atoi(argv[++i]);
        else if (!strcmp(a, "--iqp")) iqp = atoi(argv[++i]);
        else if (!strcmp(a, "--direct")) {
            const char *v = argv[++i];
            direct = !strcmp(v, "spatial") ? 1 : (!strcmp(v, "temporal") ? 0 : atoi(v));
        }
        else if (!strcmp(a, "--trellis")) trellis = atoi(argv[++i]);
        else if (!strcmp(a, "--partitions")) partitions = atoi(argv[++i]);
        else if (!strcmp(a, "--bitrate")) bitrate = atoi(argv[++i]);
        else if (!strcmp(a, "--aq")) aq = atoi(argv[++i]);
        else if (!strcmp(a, "--no-8x8dct")) p.transform_8x8 = 0;
        else if (!strcmp(a, "--no-cabac")) p.cabac = 0;
        else if (!strcmp(a, "--no-deblock")) p.deblock = 0;
        else if (!strcmp(a, "--list-devices")) { list_devices(); return 0; }
        else if (!strcmp(a, "--calibrate")) {
            int r = aeg_calibrate(backend);
            printf("calibration: %s\n", r == 0 ? "ok" : "accelerator unavailable");
            return r == 0 ? 0 : 1;
        } else if (!strcmp(a, "--bench")) bench = 1;
        else if (!strcmp(a, "--dump-recon")) dumppath = argv[++i];
        else if (!strcmp(a, "--dump-order")) orderpath = argv[++i];
        else if (!strcmp(a, "--b-adapt")) badapt = atoi(argv[++i]);
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

    /* Two-pass looks at one packet per picture, in display order.  With B
     * pictures the packets come out in coding order and the QP bias goes to
     * whichever picture is coded next rather than to the one just submitted,
     * so the measurement of picture i would end up on a different picture.
     * Asking for B pictures *explicitly* in a two-pass run is therefore an
     * error; leaving them at their default just turns them off for the run. */
    if (pass && bframes > 0) {
        fprintf(stderr, "error: --pass and --bframes cannot be combined yet\n");
        return 1;
    }
    if (pass)
        p.bframes = 0;
    if (pass) {
        if (!stats_path[0]) {
            if (!outpath) {
                fprintf(stderr, "--pass needs --stats <file> when writing to stdout\n");
                return 1;
            }
            snprintf(stats_path, sizeof(stats_path), "%s.stats", outpath);
        }
        if (pass == 2 && bitrate <= 0) {
            fprintf(stderr, "--pass 2 needs --bitrate (the first pass only measures)\n");
            return 1;
        }
        if (pass == 2) {
            FILE *sf = fopen(stats_path, "r");
            int cap = 0;

            if (!sf) {
                fprintf(stderr, "cannot read statistics file %s\n", stats_path);
                return 1;
            }
            while (!feof(sf)) {
                long long b;
                if (fscanf(sf, "%lld", &b) != 1)
                    break;
                if (pass1_n == cap) {
                    int ncap = cap ? cap * 2 : 256;
                    int64_t *nb = realloc(pass1_bits, (size_t)ncap * sizeof(*nb));
                    if (!nb) { fprintf(stderr, "out of memory\n"); return 1; }
                    pass1_bits = nb;
                    cap = ncap;
                }
                pass1_bits[pass1_n++] = b;
            }
            fclose(sf);
            if (pass1_n <= 0) {
                fprintf(stderr, "no statistics in %s\n", stats_path);
                return 1;
            }
            if (max_frames && pass1_n < max_frames)
                max_frames = pass1_n;
        }
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
    /* In pass 2 the rate controller stays on -- it is what keeps the stream on
     * target -- and the first pass's measurement arrives per picture as a QP
     * bias (aeg_encoder_set_qp_offset), so the two never fight: the controller
     * owns the level, the bias owns the distribution. */
    p.backend = backend;
    p.verbose = verbose;
    if (range) p.search_range = range;
    if (subme >= 0) p.subme = subme;
    if (refine_mv >= 0) p.refine_mv = refine_mv;
    if (refine_cand >= 0) p.refine_cand = refine_cand;
    if (refs >= 0) p.refs = refs;
    if (weightp >= 0) p.weightp = weightp;
    if (bpyramid >= 0) p.b_pyramid = bpyramid;
    if (mbtree >= 0) p.mbtree = mbtree;
    if (rclookahead >= 0) p.rc_lookahead = rclookahead;
    if (bframes >= 0) p.bframes = bframes;
    if (bqp > -1000) p.b_qp_offset = bqp;
    if (iqp > -1000) p.i_qp_offset = iqp;
    if (direct >= 0) p.direct = direct;
    if (badapt >= 0) p.b_adapt = badapt;
    if (orderpath && !(g_order = fopen(orderpath, "w"))) {
        fprintf(stderr, "cannot open %s\n", orderpath);
        return 1;
    }
    if (trellis >= 0) p.trellis = trellis;
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
            /* A picture is fed per iteration and counted per iteration, not
             * per packet: with B pictures the first `bframes` pictures produce
             * no packet at all (AEG_EAGAIN) and the rest come out one behind,
             * so counting packets would report a fraction of the frames. */
            for (;;) {
                aeg_packet pkt;
                int r;
                if (max_frames && n >= max_frames)
                    break;
                if (fread(frame, 1, fsize, in) != fsize)
                    break;
                r = aeg_encoder_encode(e, frame, fsize, pts++, &pkt);
                n++;
                if (r == AEG_EAGAIN)
                    continue;
                if (r != AEG_OK)
                    break;
                bytes += (int64_t)pkt.size;
            }
            for (;;) {
                aeg_packet pkt;
                if (aeg_encoder_flush(e, &pkt) != AEG_OK)
                    break;
                bytes += (int64_t)pkt.size;
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
        int *bias_of_frame = NULL;

        if (pass == 2) {
            bias_of_frame = malloc((size_t)pass1_n * sizeof(*bias_of_frame));
            if (!bias_of_frame) {
                fprintf(stderr, "out of memory\n");
                aeg_encoder_close(enc);
                return 1;
            }
            pass2_bias(pass1_bits, pass1_n, qcomp, bias_of_frame);
        }
        while (fread(frame, 1, fsize, in) == fsize) {
            aeg_packet pkt;
            int r;

            if (pass == 2)
                aeg_encoder_set_qp_offset(enc, n < pass1_n ? bias_of_frame[n] : 0);
            r = aeg_encoder_encode(enc, frame, fsize, pts++, &pkt);
            if (r == AEG_EAGAIN) {
                /* The picture went into the lookahead (B pictures): it will come
                 * out when the group it belongs to is complete. */
                n++;
                if (max_frames && n >= max_frames)
                    break;
                continue;
            }
            if (r != 0) {
                fprintf(stderr, "encode failed: %s (%d)\n", aeg_last_error(), r);
                ret = 1;
                break;
            }
            write_packet(out, dumppath, enc, &pkt, &dump_first,
                         width, height);
            if (pass == 1) {
                static int cap = 0;

                if (n == cap) {
                    int ncap = cap ? cap * 2 : 256;
                    int64_t *nb = realloc(pass1_bits, (size_t)ncap * sizeof(*nb));
                    if (!nb) {
                        fprintf(stderr, "out of memory\n");
                        ret = 1;
                        break;
                    }
                    pass1_bits = nb;
                    cap = ncap;
                }
                pass1_bits[n] = (int64_t)pkt.size * 8;   /* bits, not bytes */
                pass1_n = n + 1;
            }
            n++;
            if (max_frames && n >= max_frames)
                break;
        }
        /* Drain the lookahead: the pictures still held back have no more input
         * coming, so the encoder codes them and they come out here. */
        for (;;) {
            aeg_packet pkt;
            int r = aeg_encoder_flush(enc, &pkt);

            if (r == AEG_EOF)
                break;
            if (r != AEG_OK) {
                fprintf(stderr, "flush failed: %s (%d)\n", aeg_last_error(), r);
                ret = 1;
                break;
            }
            write_packet(out, dumppath, enc, &pkt, &dump_first,
                         width, height);
        }
        if (pass == 1) {
            FILE *sf = fopen(stats_path, "w");

            if (!sf) {
                fprintf(stderr, "cannot write statistics file %s\n", stats_path);
                ret = 1;
            } else {
                int k;
                for (k = 0; k < pass1_n; k++)
                    fprintf(sf, "%lld\n", (long long)pass1_bits[k]);
                fclose(sf);
                if (verbose)
                    printf("pass 1: measured %d pictures -> %s\n", pass1_n,
                           stats_path);
            }
        }
        free(bias_of_frame);
        if (verbose) {
            double secs = (aeg_now_ns() - t0) / 1e9;
            aeg_encoder_stats(enc, &st);
            printf("encoded %d frames in %.3f s (%.2f fps), %" PRId64 " bytes\n",
                   n, secs, secs > 0 ? n / secs : 0.0, st.bytes);
            /* the stream's rate is a property of the file and the frame rate,
             * not of how fast this machine happened to encode it */
            if (st.macroblocks > 0) {
                static const char *const names[16] = {
                    "skip", "P16x16", "P16x8", "P8x16", "P_8x8",
                    "I4x4", "I16x16", "I_PCM", "I8x8",
                    "B_direct", "B_L0", "B_L1", "B_BI",
                    "B16x8", "B8x16", "B_8x8"
                };
                int k;
                printf("macroblock types: %" PRId64 " skip of %" PRId64
                       " (%.0f%%), %" PRId64 " coded\n",
                       st.skipped, st.macroblocks,
                       100.0 * (double)st.skipped / (double)st.macroblocks,
                       st.macroblocks - st.skipped);
                printf("  breakdown:");
                for (k = 0; k < 16; k++)
                    if (st.mb_types[k])
                        printf(" %s %" PRId64, names[k], st.mb_types[k]);
                printf("\n");
            }
            printf("bitrate %.1f kb/s at %d fps, PSNR %.2f dB (Y+U+V average)\n",
                   st.frames > 0
                       ? st.bytes * 8.0 * fps / (double)st.frames / 1000.0
                       : 0.0,
                   fps, st.psnr_avg);
            if (st.scene_cuts)
                printf("scenecut: %" PRId64 " IDR picture(s) inserted\n",
                       st.scene_cuts);
            if (st.vbv_min_fill_bits > 0.0 || st.vbv_underflows)
                printf("vbv: buffer low water %.1f kbit, %" PRId64
                       " underflow(s), %" PRId64 " picture(s) raised to fit\n",
                       st.vbv_min_fill_bits / 1000.0, st.vbv_underflows,
                       st.vbv_size_clamps);
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
    if (g_order)
        fclose(g_order);
    return ret;
}
