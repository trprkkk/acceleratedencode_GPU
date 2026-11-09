/*
 * aegenc - command line front end.
 *
 *   aegenc -w 1280 -h 720 -q 23 -i in.yuv -o out.264
 *
 * Input is raw I420 (YUV 4:2:0, 8 bit, planar) or YUV4MPEG2 (.y4m, the same
 * samples with a header that carries the size, frame rate and aspect ratio);
 * output is an H.264 Annex-B elementary stream.
 */
#define _POSIX_C_SOURCE 200809L
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
"       aegenc -i in.y4m [-o out.264] [options]\n"
"\n"
"  -w, -h <int>     frame size, any even size (coded as whole macroblocks\n"
"                   and cropped back in the SPS); a .y4m input carries it\n"
"  -i <file>        input: raw I420, or YUV4MPEG2 (recognised by its\n"
"                   header, also on stdin; default: stdin)\n"
"  --demuxer <fmt>  auto (default) | raw | y4m\n"
"  --input-csp <c>  raw input layout: i420 (default) yv12 nv12 nv21 i422\n"
"                   yv16 nv16 i444 yv24 i400; a .y4m header's C tag sets it\n"
"  --input-depth <n> raw input bits per sample, 8 (default) .. 16; above 8\n"
"                   two bytes per sample, little-endian (yuv420p10le etc.)\n"
"                   Anything but 4:2:0 at the coded depth is converted:\n"
"                   chroma resampled to 4:2:0, then dithered to the coded\n"
"                   depth (as x264); 8-bit input to a 10-bit stream is\n"
"                   scaled up exactly\n"
"  --bit-depth <n>  coded bits per sample: 8 (default) or 10 (High 10\n"
"                   profile, CPU only; x264 --output-depth 10)\n"
"  -o <file>        output Annex-B stream (default: stdout)\n"
"  -q <int>         QP 0..51 (default 23)\n"
"  --crf <float>    constant quality instead of constant QP (x264's scale,\n"
"                   -12..51 at 10 bits)\n"
"                   0..51; 18..28 is the usual range).  The QP follows each\n"
"                   picture's complexity (--qcomp), or, with the macroblock\n"
"                   tree, each macroblock's importance\n"
"  -f, --fps <r>    frame rate: 25, 30000/1001 or 29.97 (default 30, or\n"
"                   the .y4m header's)\n"
"  --sar <w:h>      sample aspect ratio to signal (default: none, or the\n"
"                   .y4m header's)\n"
"  --fullrange      the samples are full range 0..255 (signalled in the VUI)\n"
"  --colorprim <c>  colour primaries to signal: bt709 bt470m bt470bg\n"
"                   smpte170m smpte240m film bt2020, or the H.273 number\n"
"  --transfer <t>   transfer: bt709 bt470m bt470bg smpte170m smpte240m\n"
"                   linear srgb bt2020-10 bt2020-12 smpte2084 arib-std-b67\n"
"  --colormatrix <m> matrix: gbr bt709 fcc bt470bg smpte170m smpte240m\n"
"                   ycgco bt2020nc bt2020c\n"
"  --master-display <s>  HDR10 mastering display SEI, x264 syntax:\n"
"                   G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min)\n"
"  --max-cll <m,f>  content light level SEI: max content / frame-average\n"
"                   light level in cd/m2\n"
"  -g <int>         GOP size, 0 = only the first frame is IDR (default 250)\n"
"  --scenecut <pct> insert an IDR when a scene change is detected (0 = off,\n"
"                   default 120: inter prediction 20%% worse than intra)\n"
"  --pass <1|2>     two-pass rate control: pass 1 codes the clip (best with the\n"
"                   same --bitrate) and measures every picture, B pictures\n"
"                   included; pass 2 shares --bitrate out by that measurement\n"
"  --stats <file>   statistics file for --pass (default: <output>.stats)\n"
"  --qcomp <pct>    complexity-to-bits exponent of --crf and of pass 2, in\n"
"                   percent (default 60: bits ~ complexity^0.6)\n"
"  -n <int>         encode at most n frames\n"
"  -b <backend>     auto | cpu | opencl | cuda (default auto)\n"
"  -s <stage=be>    force one stage: est,sad,inter,tq,iq,deblock,intra,recon,psnr\n"
"  -p <preset>      0=ultrafast 1=fast 2=medium 3=slow 4=veryslow\n"
"  --no-8x8dct      use only 4x4 luma transforms (drops High profile)\n"
"  --no-cabac       code the macroblock layer with CAVLC instead of CABAC\n"
"  --no-deblock     disable the in-loop deblocking filter\n"
"  --deblock <a:b>  deblocking strength offsets, -6..6 each (default 0:0)\n"
"  --qpmin <n>, --qpmax <n>  no macroblock below / above this QP (max 1..51)\n"
"                   (defaults 0 / 51)\n"
"  --chroma-qp-offset <n>  chroma QP relative to luma, -12..12 (default 0)\n"
"  --aud            an access unit delimiter before every picture\n"
"  --tff, --bff     interlaced material, top / bottom field first: coded as\n"
"                   frames, signalled as interlaced with that field order\n"
"                   (x264 --fake-interlaced; height a multiple of 4)\n"
"  --field          code interlaced material as field pictures (PAFF): two\n"
"                   pictures per frame, one per field (--tff unless --bff)\n"
"  --paff           frame or field pictures, chosen per frame from the source\n"
"                   (B frames follow their anchor); --tff unless --bff\n"
"  --mbaff          MBAFF frames: frame or field per macroblock pair\n"
"  --paff-mbaff     per frame, field pictures or an MBAFF frame (B frames\n"
"                   follow their anchor); --tff unless --bff\n"
"  --level <l>      force the level (4.1, 41; default: the smallest that\n"
"                   fits)\n"
"  --seek <n>       skip the first n input frames\n"
"  --range <int>    integer motion search range around the zero vector\n"
"                   (default 48, at most 56)\n"
"  --subme <int>    subpel refinement 0..2 (default 1)\n"
"  --refine-mv <n>  0 = off, 1 = four quarter-pel neighbours of the winning\n"
"                   vector (the cross), 2 = the four of all eight with the\n"
"                   lowest SATD, 3 = all eight (rate-distortion refined,\n"
"                   slower; default 0, preset 3 uses 1 and preset 4 uses 2)\n"
"  --refine-cand <n> refine every inter candidate's vectors, not only the\n"
"                   winner's: 0 = off (default), 1 = on (costs about a third of\n"
"                   the encode time at preset 4 for 0.3-0.6 points of BD-rate)\n"
"  --qp-rd <n>      choose each macroblock's QP by rate-distortion (x264's\n"
"                   subme 10): 0 = off, 1 = on (default at preset 4; needs\n"
"                   CABAC; preset 3: -2.2 points of BD-rate for ~20%% time)\n"
"  --b-pyramid <n>  1 = the middle B picture of a group is a reference for\n"
"                   the others (default with B pictures), 0 = off\n"
"  --weightp <n>    explicit weighted prediction of P pictures (fades),\n"
"                   0 = off, 1 = on (default from preset 1)\n"
"  --ref <n>        reference pictures for P pictures, 1..8 (default 8 at\n"
"                   preset 4, 4 at preset 3, otherwise 1)\n"
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
"                   being coded (default 20/20/30/40 at presets 1-4, max 60)\n"
"  --iqp <n>        QP subtracted from I pictures at constant QP (default 3,\n"
"                   x264's ipratio 1.4)\n"
"  --direct <mode>  B direct prediction: spatial (default) or temporal\n"
"  --dump-recon <f> write the reconstruction (coding order, I420; 16-bit\n"
"                   little-endian samples at --bit-depth 10)\n"
"  --dump-order <f> write each packet's display index, one per line\n"
"  --trellis <n>    rate-distortion level selection (x264's meaning): 0 = off,\n"
"                   1 = the winning macroblock only (presets 1-2), 2 = every\n"
"                   candidate of the decision as well (presets 3-4)\n"
"  --bitrate <kbps> average-bitrate rate control (ABR) instead of constant QP\n"
"  --vbv-maxrate <kbps>  VBV/HRD channel rate (implies ABR)\n"
"  --vbv-bufsize <kbits> VBV/HRD buffer size  (implies ABR)\n"
"  --nal-hrd        signal the VBV model as the NAL HRD: hrd_parameters in\n"
"                   the SPS, buffering period and picture timing SEI\n"
"  --open-gop       keyframes are non-IDR I slices (open GOP); recovery\n"
"                   point SEI marks them\n"
"  --info-sei       write the encoder name and options into the stream (a\n"
"                   user data SEI on the first picture, as x264 does)\n"
"  --aq <n>         adaptive quantisation strength in tenths (0 = off,\n"
"                   default off; --aq 10 = x264 aq-strength 1.0)\n"
"  --aq-mode <n>    0 off, 1 variance, 2 auto-variance (default when\n"
"                   --aq is given), 3 auto-variance with dark bias\n"
"  --slices <n>     slices per picture (0/1 = one slice; n>1 cuts the\n"
"                   picture into n row ranges, one NAL each)\n"
"  --slice-max-mbs <n>  no slice holds more than n macroblocks\n"
"  --slice-max-size <n> no slice payload exceeds n bytes\n"
"  --intra-refresh <n>  no IDR after the first picture: code a moving\n"
"                   intra band instead, every macroblock refreshed\n"
"                   once per n frames (implies keyint infinite)\n"
"  -t <int>         slice threads (0 = auto, 1 = serial; -t N without\n"
"                   --slices uses one slice per thread)\n"
"  --frame-threads <n> code n pictures at once, each following its\n"
"                   references row by row (same stream as serial);\n"
"                   1 = off, 0 = auto (CPUs + 1, max 8, unless -t 1;\n"
"                   B pictures, constant QP, one slice)\n"
"  --partitions <n> inter partition mask: 1=16x16 2=16x8 4=8x16\n"
"                   8=P_8x8  16=its 8x4/4x8/4x4 sub-partitions\n"
"                   (default 15; P_8x8 is skipped by preset 0, and so\n"
"                   are bits 1-3 for B pictures -- 16x8 and 8x16 are\n"
"                   the only B shapes so far)\n"
"  -v <int>         verbosity (0 errors only, 1 normal, 2 every picture)\n"
"  --list-devices   enumerate the compute devices and exit\n"
"  --device <n>     use device n of the selected backend (default 0);\n"
"                   also selectable with AEG_DEVICE\n"
"  --qpfile <file>  per-picture QPs: one `frame qp` per line (x264's\n"
"                   format; comments after # and blank lines ignored;\n"
"                   qp -1 leaves that picture to the rate controller)\n"
"  --zones <s,e,q>  give display pictures s..e the QP q (repeatable;\n"
"                   a --qpfile entry wins over a zone)\n"
"  --roi <x,y,w,h,off>  add off to the QP of macroblocks touching the\n"
"                   rectangle (pixel coordinates, repeatable;\n"
"                   negative = finer)\n"
"  --calibrate      measure the accelerator launch overhead and exit\n"
"  --bench          encode three times (auto/cpu/accelerator) and report\n"
"\n"
"Exit status: 0 on success.\n");
}

/* ------------------------------------------------------------------ */
/* Option values                                                       */
/* ------------------------------------------------------------------ */
static char *next_arg(int argc, char **argv, int *i)
{
    if (*i + 1 >= argc) {
        fprintf(stderr, "error: option '%s' needs a value\n", argv[*i]);
        exit(2);
    }
    return argv[++*i];
}

static int int_arg(const char *opt, const char *v)
{
    char *end;
    long n = strtol(v, &end, 10);
    if (end == v || *end || n < -1000000000L || n > 1000000000L) {
        fprintf(stderr, "error: option '%s': '%s' is not a number\n", opt, v);
        exit(2);
    }
    return (int)n;
}

/* 25, 30000/1001, 29.97 (an NTSC-style decimal becomes n*1000/1001) */
static int parse_rate(const char *v, int *num, int *den)
{
    char *end;
    double f;
    long a, b;
    if (strchr(v, '/') || strchr(v, ':')) {
        a = strtol(v, &end, 10);
        if (end == v || (*end != '/' && *end != ':'))
            return -1;
        b = strtol(end + 1, &end, 10);
        if (*end || a <= 0 || b <= 0 || a > 1000000000L || b > 1000000000L)
            return -1;
        *num = (int)a, *den = (int)b;
        return 0;
    }
    f = strtod(v, &end);
    if (end == v || *end || !(f > 0) || f > 1000000)
        return -1;
    if (fabs(f - floor(f + 0.5)) < 1e-9) {
        *num = (int)floor(f + 0.5), *den = 1;
    } else if (fabs(f * 1001.0 / 1000.0 - floor(f * 1001.0 / 1000.0 + 0.5)) < 0.01) {
        *num = (int)floor(f * 1001.0 / 1000.0 + 0.5) * 1000, *den = 1001;
    } else {
        *num = (int)floor(f * 1000.0 + 0.5), *den = 1000;
    }
    return 0;
}

/* an H.273 code point by name or number, -1 = unknown */
static int parse_colour(const char *v, const char *const *names,
                        const int *codes, int n)
{
    char *end;
    long c;
    int i;
    for (i = 0; i < n; i++)
        if (!strcmp(v, names[i]))
            return codes[i];
    c = strtol(v, &end, 10);
    return (end != v && !*end && c >= 0 && c <= 255) ? (int)c : -1;
}

/* ------------------------------------------------------------------ */
/* Input: raw I420 or YUV4MPEG2                                        */
/* ------------------------------------------------------------------ */
typedef struct {
    FILE *f;
    int y4m;
    unsigned char peek[16];
    int npeek, ppeek;
    /* from a .y4m header (0 = not given) */
    int w, h, fps_num, fps_den, sar_w, sar_h, fullrange;
    int interlaced;             /* It = 1 (top first), Ib = 2 */
    int csp, depth;             /* from the C tag (-1 = not given)        */
} input_t;

/* ------------------------------------------------------------------ */
/* Input layouts: planes, their sizes, and the frame as an aeg_image   */
/* ------------------------------------------------------------------ */
static const struct { const char *name; int csp, xs, ys, planes, il; } csps[] = {
    { "i420", AEG_CSP_I420, 1, 1, 3, 1 }, { "yv12", AEG_CSP_YV12, 1, 1, 3, 1 },
    { "nv12", AEG_CSP_NV12, 1, 1, 2, 2 }, { "nv21", AEG_CSP_NV21, 1, 1, 2, 2 },
    { "i422", AEG_CSP_I422, 1, 0, 3, 1 }, { "yv16", AEG_CSP_YV16, 1, 0, 3, 1 },
    { "nv16", AEG_CSP_NV16, 1, 0, 2, 2 }, { "i444", AEG_CSP_I444, 0, 0, 3, 1 },
    { "yv24", AEG_CSP_YV24, 0, 0, 3, 1 }, { "i400", AEG_CSP_I400, 0, 0, 1, 1 },
};

static void le16_to_native(uint8_t *b, size_t n);

/* hand one frame of the input to the encoder */
static int encode_frame(aeg_encoder *e, uint8_t *frame, size_t fsize,
                        const aeg_image *layout, const size_t off[3],
                        int64_t pts, aeg_packet *pkt)
{
    aeg_image im = *layout;
    int i;
    if (im.depth > 8)
        le16_to_native(frame, fsize);
    for (i = 0; i < 3; i++)
        im.plane[i] = im.stride[i] ? frame + off[i] : NULL;
    return aeg_encoder_encode_image(e, &im, pts, pkt);
}

static int csp_index(int csp)
{
    int i;
    for (i = 0; i < (int)(sizeof(csps) / sizeof(csps[0])); i++)
        if (csps[i].csp == csp)
            return i;
    return 0;
}

/* lay out one raw frame of w x h: plane offsets, strides, total bytes */
static size_t frame_layout(int csp, int depth, int w, int h, aeg_image *img,
                           size_t off[3])
{
    int k = csp_index(csp), i, bps = depth > 8 ? 2 : 1;
    size_t total = 0;
    memset(img, 0, sizeof(*img));
    img->csp = csp;
    img->depth = depth;
    for (i = 0; i < csps[k].planes; i++) {
        int pw = i ? (w + csps[k].xs) >> csps[k].xs : w;
        int ph = i ? (h + csps[k].ys) >> csps[k].ys : h;
        img->stride[i] = pw * bps * (i ? csps[k].il : 1);
        off[i] = total;
        total += (size_t)img->stride[i] * ph;
    }
    return total;
}

/* YUV4MPEG2 colour space tag: 420jpeg, 420paldv, 420mpeg2, 420 (all 4:2:0;
 * the siting variants are not resampled, as with x264), 422, 444, mono,
 * each optionally with p9..p16 (mono: 9..16) for the bits per sample */
static int y4m_csp(const char *c, int *csp, int *depth)
{
    const char *rest;
    *depth = 8;
    if (!strncmp(c, "420", 3)) {
        *csp = AEG_CSP_I420;
        rest = c + 3;
        if (!strcmp(rest, "jpeg") || !strcmp(rest, "paldv") ||
            !strcmp(rest, "mpeg2"))
            return 0;
    } else if (!strncmp(c, "422", 3)) {
        *csp = AEG_CSP_I422;
        rest = c + 3;
    } else if (!strncmp(c, "444", 3)) {
        *csp = AEG_CSP_I444;
        rest = c + 3;
        if (!strcmp(rest, "alpha"))
            return -1;
    } else if (!strncmp(c, "mono", 4)) {
        *csp = AEG_CSP_I400;
        rest = c + 4;
        if (*rest && *rest != 'p') {        /* mono16, mono10, ... */
            char *end;
            long d = strtol(rest, &end, 10);
            if (*end || d < 9 || d > 16)
                return -1;
            *depth = (int)d;
            return 0;
        }
    } else
        return -1;
    if (!*rest)
        return 0;
    if (*rest == 'p') {
        char *end;
        long d = strtol(rest + 1, &end, 10);
        if (*end || d < 9 || d > 16)
            return -1;
        *depth = (int)d;
        return 0;
    }
    return -1;
}

/* the 16-bit samples of a file are little-endian; aeg_image wants them in
 * the machine's order */
static void le16_to_native(uint8_t *b, size_t n)
{
    const uint16_t one = 1;
    size_t i;
    if (*(const uint8_t *)&one)
        return;
    for (i = 0; i + 1 < n; i += 2) {
        uint8_t t = b[i];
        b[i] = b[i + 1];
        b[i + 1] = t;
    }
}

static size_t in_read(input_t *in, void *buf, size_t n)
{
    size_t got = 0;
    unsigned char *b = (unsigned char *)buf;
    while (got < n && in->ppeek < in->npeek)
        b[got++] = in->peek[in->ppeek++];
    if (got < n)
        got += fread(b + got, 1, n - got, in->f);
    return got;
}

/* one header line (without its '\n'); -1 at the end of the input */
static int in_line(input_t *in, char *line, int cap)
{
    int n = 0;
    unsigned char c;
    for (;;) {
        if (in_read(in, &c, 1) != 1)
            return n ? n : -1;
        if (c == '\n')
            break;
        if (n < cap - 1)
            line[n++] = (char)c;
    }
    line[n] = 0;
    return n;
}

/* force: 0 auto, 1 raw, 2 y4m */
static int in_open(input_t *in, FILE *f, int force)
{
    char line[1024], *tok, *save = NULL;
    memset(in, 0, sizeof(*in));
    in->f = f;
    in->csp = in->depth = -1;
    in->npeek = (int)fread(in->peek, 1, 9, f);
    if (force == 1 || in->npeek < 9 || memcmp(in->peek, "YUV4MPEG2", 9)) {
        if (force == 2) {
            fprintf(stderr, "error: the input is not YUV4MPEG2\n");
            return -1;
        }
        return 0;
    }
    in->y4m = 1;
    in->ppeek = 9;
    if (in_line(in, line, sizeof(line)) < 0) {
        fprintf(stderr, "error: truncated YUV4MPEG2 header\n");
        return -1;
    }
    for (tok = strtok_r(line, " ", &save); tok; tok = strtok_r(NULL, " ", &save)) {
        switch (tok[0]) {
        case 'W': in->w = atoi(tok + 1); break;
        case 'H': in->h = atoi(tok + 1); break;
        case 'F':
            if (parse_rate(tok + 1, &in->fps_num, &in->fps_den) < 0)
                in->fps_num = in->fps_den = 0;
            break;
        case 'A':
            if (sscanf(tok + 1, "%d:%d", &in->sar_w, &in->sar_h) != 2 ||
                in->sar_w <= 0 || in->sar_h <= 0)
                in->sar_w = in->sar_h = 0;
            break;
        case 'I':
            if (tok[1] == 't')
                in->interlaced = 1;
            else if (tok[1] == 'b')
                in->interlaced = 2;
            else if (tok[1] != 'p' && tok[1] != '?')
                fprintf(stderr, "warning: YUV4MPEG2 field order 'I%c' (mixed) "
                                "is signalled as progressive; --tff/--bff "
                                "override it\n", tok[1]);
            break;
        case 'C':
            if (y4m_csp(tok + 1, &in->csp, &in->depth) < 0) {
                fprintf(stderr, "error: YUV4MPEG2 colour space '%s' is not "
                                "supported (420, 422, 444, mono, with p9..p16)\n",
                        tok + 1);
                return -1;
            }
            break;
        case 'X':
            if (!strcmp(tok, "XCOLORRANGE=FULL"))
                in->fullrange = 1;
            break;
        default:
            break;
        }
    }
    if (in->w <= 0 || in->h <= 0) {
        fprintf(stderr, "error: YUV4MPEG2 header without a frame size\n");
        return -1;
    }
    return 0;
}

/* 1 = a frame, 0 = the end, -1 = a damaged or partial frame at the end */
static int in_frame(input_t *in, uint8_t *frame, size_t fsize)
{
    size_t got;
    if (in->y4m) {
        char line[256];
        int n = in_line(in, line, sizeof(line));
        if (n < 0)
            return 0;
        if (strncmp(line, "FRAME", 5)) {
            fprintf(stderr, "warning: YUV4MPEG2 frame header missing; input "
                            "stops here\n");
            return -1;
        }
    }
    got = in_read(in, frame, fsize);
    if (got == fsize)
        return 1;
    if (got > 0) {
        fprintf(stderr, "warning: the input ends with a partial frame (%zu of "
                        "%zu bytes); it is not coded\n", got, fsize);
        return -1;
    }
    return 0;
}

/* a set option outside its range is an error, not a silent clamp */
static int range_bad(const char *opt, int v, int lo, int hi)
{
    if (v >= lo && v <= hi)
        return 0;
    fprintf(stderr, "error: %s %d is out of range (%d..%d)\n", opt, v, lo, hi);
    return 1;
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

/* x264-style qpfile: lines of `frame qp` (an optional type letter is
 * accepted and ignored).  Returns the table, -1 where no override was given. */
static int *load_qpfile(const char *path, int *n_out)
{
    FILE *f = fopen(path, "r");
    char line[512];
    int *tab = NULL, n = 0, cap = 0;

    *n_out = 0;
    if (!f) {
        fprintf(stderr, "error: cannot open qpfile %s\n", path);
        return NULL;
    }
    while (fgets(line, sizeof(line), f)) {
        char *p = line;
        long frame, qp;
        int k;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '#' || *p == '\n' || *p == '\r' || !*p)
            continue;
        if (sscanf(p, "%ld %ld", &frame, &qp) != 2 || frame < 0 ||
            frame > 100000000L || qp < -1 || qp > 51) {
            fprintf(stderr, "%s: bad line: %s", path, line);
            free(tab);
            fclose(f);
            return NULL;
        }
        if (frame >= cap) {
            int ncap = cap ? cap : 64;
            while (ncap <= frame)
                ncap *= 2;
            {
                int *nt = (int *)realloc(tab, (size_t)ncap * sizeof(int));
                if (!nt) {
                    fprintf(stderr, "out of memory reading %s\n", path);
                    free(tab);
                    fclose(f);
                    return NULL;
                }
                tab = nt;
                for (k = cap; k < ncap; k++)
                    tab[k] = -1;
                cap = ncap;
            }
        }
        tab[frame] = (int)qp;
        if (frame + 1 > n)
            n = (int)frame + 1;
    }
    fclose(f);
    *n_out = n;
    return tab;
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

/* Pass 1's measurement of the picture with display index d (the packets come
 * out in coding order). */
static int pass1_put(aeg_pass1_frame **f, int *n, int *cap, int64_t d,
                     int type, double qp, double qp_base, int64_t bits)
{
    if (d < 0 || d > 100000000)
        return 0;
    if (d >= *cap) {
        int ncap = *cap ? *cap : 256, k;
        aeg_pass1_frame *nf;
        while (ncap <= d)
            ncap *= 2;
        nf = realloc(*f, (size_t)ncap * sizeof(**f));
        if (!nf)
            return -1;
        *f = nf;
        for (k = *cap; k < ncap; k++)
            nf[k].type = 1, nf[k].qp = nf[k].qp_base = 0.0, nf[k].bits = -1;
        *cap = ncap;
    }
    (*f)[d].type = type >= 0 && type <= 3 ? type : 1;
    (*f)[d].qp = qp;
    (*f)[d].qp_base = qp_base;
    (*f)[d].bits = bits;
    if (d + 1 > *n)
        *n = (int)d + 1;
    return 0;
}

/* Write one packet to the stream and, if a reconstruction dump was asked for,
 * append this picture's reconstruction to it.  The dump follows the *coding*
 * order when the stream is reordered by B pictures, which is the order the
 * packets come out in; a decoder that outputs in display order has to be
 * re-ordered before the two files are compared. */
/* --dump-order: the display index of every packet, one per line, in the order
 * the packets come out (= the order of the reconstruction dump). */
static FILE *g_order;

static int g_bps = 1;   /* bytes per reconstructed sample (2 at 10 bits) */

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
        /* planar I420 (yuv420p10le at 10 bits on a little-endian host), the
         * layout the decoder dumps so the two can be compared byte for byte */
        for (y = 0; y < height; y++)
            fwrite(pl[0] + (size_t)y * str[0], g_bps, (size_t)width, df);
        for (y = 0; y < height / 2; y++)
            fwrite(pl[1] + (size_t)y * str[1], g_bps, (size_t)width / 2, df);
        for (y = 0; y < height / 2; y++)
            fwrite(pl[2] + (size_t)y * str[2], g_bps, (size_t)width / 2, df);
        fclose(df);
    }
}

int main(int argc, char **argv)
{
    aeg_params p;
    aeg_encoder *enc;
    FILE *in = stdin, *out = stdout;
    int width = 0, height = 0, qp = 23, gop = 250, max_frames = 0;
    int fps_num = 30, fps_den = 1, fps_set = 0, sar_w = 0, sar_h = 0;
    int fullrange = 0, demux = 0, seek = 0, interlaced = -1, field_coding = 0;
    int md[8] = { 0 }, md_set = 0, max_cll = -1, max_fall = -1;
    unsigned md_max_lum = 0, md_min_lum = 0;
    int in_csp = -1, in_depth = -1;
    aeg_image layout;
    size_t poff[3];
    double crf = -1.0;
    int crf_given = 0, bit_depth = 8;
    double fps;
    input_t inp;
    int bitrate = 0, aq = -1, aq_mode = -1;
    int backend = AEG_BACKEND_AUTO, verbose = 1, bench = 0;
    int range = 0, subme = -1, partitions = -1, pass = 0, qcomp = 60;
    int refine_mv = -1, refine_cand = -1, qp_rd = -1, refs = -1, weightp = -1, bpyramid = -1;
    int mbtree = -1, rclookahead = -1;
    int trellis = -1;
    int bframes = -1, bqp = -1000, iqp = -1000, direct = -1, badapt = -1;
    aeg_pass1_frame *pass1 = NULL;
    int pass1_n = 0, pass1_cap = 0, pass1_bframes = -1;
    char stats_path[4096];
    const char *inpath = NULL, *outpath = NULL, *dumppath = NULL;
    const char *orderpath = NULL;
    const char *qpfilepath = NULL;
    int *qptab = NULL, qpn = 0;
    aeg_zone *zones = NULL;
    int nzones = 0, zonecap = 0;
    aeg_roi *rois = NULL;
    int nrois = 0, roicap = 0;
    uint8_t *frame;
    size_t fsize;
    int64_t pts = 0;
    int dump_first = 1;
    aeg_stats st;
    int i, ret = 0;

    aeg_params_default(&p);

    for (i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-w")) width = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "-h")) height = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "-i")) inpath = next_arg(argc, argv, &i);
        else if (!strcmp(a, "-o")) outpath = next_arg(argc, argv, &i);
        else if (!strcmp(a, "-q")) qp = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "-f") || !strcmp(a, "--fps")) {
            const char *v = next_arg(argc, argv, &i);
            if (parse_rate(v, &fps_num, &fps_den) < 0) {
                fprintf(stderr, "error: bad frame rate '%s'\n", v);
                return 2;
            }
            fps_set = 1;
        }
        else if (!strcmp(a, "--sar")) {
            const char *v = next_arg(argc, argv, &i);
            if (sscanf(v, "%d:%d", &sar_w, &sar_h) != 2 || sar_w <= 0 ||
                sar_h <= 0) {
                fprintf(stderr, "error: bad --sar '%s' (want W:H)\n", v);
                return 2;
            }
        }
        else if (!strcmp(a, "--fullrange")) fullrange = 1;
        else if (!strcmp(a, "--nal-hrd")) p.nal_hrd = 1;
        else if (!strcmp(a, "--aud")) p.aud = 1;
        else if (!strcmp(a, "--input-csp")) {
            const char *v = next_arg(argc, argv, &i);
            int k;
            for (k = 0; k < (int)(sizeof(csps) / sizeof(csps[0])); k++)
                if (!strcmp(v, csps[k].name))
                    in_csp = csps[k].csp;
            if (in_csp < 0) {
                fprintf(stderr, "error: --input-csp %s: i420 yv12 nv12 nv21 "
                                "i422 yv16 nv16 i444 yv24 i400\n", v);
                return 2;
            }
        }
        else if (!strcmp(a, "--input-depth"))
            in_depth = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--bit-depth") || !strcmp(a, "--output-depth"))
            bit_depth = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--tff")) interlaced = 1;
        else if (!strcmp(a, "--bff")) interlaced = 2;
        else if (!strcmp(a, "--field")) field_coding = 1;
        else if (!strcmp(a, "--paff")) field_coding = 2;
        else if (!strcmp(a, "--mbaff")) field_coding = 3;
        else if (!strcmp(a, "--paff-mbaff")) field_coding = 4;
        else if (!strcmp(a, "--qpmin")) p.qp_min = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--qpmax")) p.qp_max = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--chroma-qp-offset"))
            p.chroma_qp_offset = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--seek")) seek = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--deblock")) {
            const char *v = next_arg(argc, argv, &i);
            if (sscanf(v, "%d:%d", &p.deblock_alpha, &p.deblock_beta) != 2 &&
                sscanf(v, "%d,%d", &p.deblock_alpha, &p.deblock_beta) != 2) {
                fprintf(stderr, "error: bad --deblock '%s' (want a:b)\n", v);
                return 2;
            }
        }
        else if (!strcmp(a, "--level")) {
            const char *v = next_arg(argc, argv, &i);
            double l = atof(v);
            p.level_idc = strchr(v, '.') || l < 10 ? (int)(l * 10 + 0.5) : (int)l;
            if (p.level_idc != 10 && p.level_idc != 11 && p.level_idc != 12 &&
                p.level_idc != 13 && p.level_idc != 20 && p.level_idc != 21 &&
                p.level_idc != 22 && p.level_idc != 30 && p.level_idc != 31 &&
                p.level_idc != 32 && p.level_idc != 40 && p.level_idc != 41 &&
                p.level_idc != 42 && p.level_idc != 50 && p.level_idc != 51 &&
                p.level_idc != 52 && p.level_idc != 60 && p.level_idc != 61 &&
                p.level_idc != 62) {
                fprintf(stderr, "error: unknown --level '%s'\n", v);
                return 2;
            }
        }
        else if (!strcmp(a, "--master-display")) {
            const char *v = next_arg(argc, argv, &i);
            const char *r = strstr(v, "R("), *g = strstr(v, "G(");
            const char *b = strstr(v, "B("), *w = strstr(v, "WP(");
            const char *l = strstr(v, "L(");
            unsigned lumax, lumin;
            if (!r || !g || !b || !w || !l ||
                sscanf(r, "R(%d,%d)", &md[0], &md[1]) != 2 ||
                sscanf(g, "G(%d,%d)", &md[2], &md[3]) != 2 ||
                sscanf(b, "B(%d,%d)", &md[4], &md[5]) != 2 ||
                sscanf(w, "WP(%d,%d)", &md[6], &md[7]) != 2 ||
                sscanf(l, "L(%u,%u)", &lumax, &lumin) != 2) {
                fprintf(stderr, "error: --master-display %s: expected "
                        "G(x,y)B(x,y)R(x,y)WP(x,y)L(max,min)\n", v);
                return 2;
            }
            md_max_lum = lumax;
            md_min_lum = lumin;
            md_set = 1;
        }
        else if (!strcmp(a, "--max-cll")) {
            const char *v = next_arg(argc, argv, &i);
            if (sscanf(v, "%d,%d", &max_cll, &max_fall) != 2 ||
                max_cll < 0 || max_cll > 65535 || max_fall < 0 ||
                max_fall > 65535) {
                fprintf(stderr, "error: --max-cll %s: expected max,fall "
                        "(0..65535)\n", v);
                return 2;
            }
        }
        else if (!strcmp(a, "--colorprim") || !strcmp(a, "--transfer") ||
                 !strcmp(a, "--colormatrix")) {
            static const char *const pn[] = { "bt709", "bt470m", "bt470bg",
                "smpte170m", "smpte240m", "film", "bt2020" };
            static const int pc[] = { 1, 4, 5, 6, 7, 8, 9 };
            static const char *const tn[] = { "bt709", "bt470m", "bt470bg",
                "smpte170m", "smpte240m", "linear", "srgb", "iec61966-2-1",
                "bt2020-10", "bt2020-12", "smpte2084", "arib-std-b67" };
            static const int tc[] = { 1, 4, 5, 6, 7, 8, 13, 13, 14, 15, 16, 18 };
            static const char *const mn[] = { "gbr", "bt709", "fcc", "bt470bg",
                "smpte170m", "smpte240m", "ycgco", "bt2020nc", "bt2020c" };
            static const int mc[] = { 0, 1, 4, 5, 6, 7, 8, 9, 10 };
            const char *v = next_arg(argc, argv, &i);
            const int prim = !strcmp(a, "--colorprim");
            const int trc = !strcmp(a, "--transfer");
            int c = prim ? parse_colour(v, pn, pc, 7)
                         : trc ? parse_colour(v, tn, tc, 12)
                               : parse_colour(v, mn, mc, 9);
            if (c < 0) {
                fprintf(stderr, "error: option '%s': unknown value '%s'\n", a, v);
                return 2;
            }
            if (prim) p.colorprim = c;
            else if (trc) p.transfer = c;
            else p.colormatrix = c;
        }
        else if (!strcmp(a, "--demuxer")) {
            const char *v = next_arg(argc, argv, &i);
            demux = !strcmp(v, "raw") ? 1 : !strcmp(v, "y4m") ? 2 :
                    !strcmp(v, "auto") ? 0 : -1;
            if (demux < 0) {
                fprintf(stderr, "error: --demuxer auto|raw|y4m\n");
                return 2;
            }
        }
        else if (!strcmp(a, "-g")) gop = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "-n")) max_frames = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "-b")) backend = parse_backend(next_arg(argc, argv, &i));
        else if (!strcmp(a, "-p")) p.preset = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--slices")) p.slices = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--slice-max-mbs"))
            p.slice_max_mbs = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--slice-max-size"))
            p.slice_max_size = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--intra-refresh"))
            p.intra_refresh = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "-t") || !strcmp(a, "--threads"))
            p.threads = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--frame-threads"))
            p.frame_threads = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--scenecut")) p.scenecut = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--pass")) pass = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--qcomp")) qcomp = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--crf")) {
            const char *v = next_arg(argc, argv, &i);
            char *end;
            crf = strtod(v, &end);
            if (end == v || *end || !(crf >= -12.0 && crf <= 51.0)) {
                fprintf(stderr, "error: --crf %s: expected a number in 0..51 "
                        "(-12..51 at --bit-depth 10)\n", v);
                return 2;
            }
            crf_given = 1;
        }
        else if (!strcmp(a, "--stats")) {
            snprintf(stats_path, sizeof(stats_path), "%s", next_arg(argc, argv, &i));
            stats_path[0] = stats_path[0];      /* keep -Wall quiet */
        }
        else if (!strcmp(a, "--vbv-maxrate"))
            p.vbv_maxrate_kbps = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--vbv-bufsize"))
            p.vbv_bufsize_kbits = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "-v")) verbose = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--range")) range = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--subme")) subme = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--refine-mv")) refine_mv = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--refine-cand")) refine_cand = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--qp-rd")) qp_rd = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--ref")) refs = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--weightp")) weightp = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--b-pyramid")) bpyramid = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--mbtree")) mbtree = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--rc-lookahead")) rclookahead = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--bframes")) bframes = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--bqp")) bqp = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--iqp")) iqp = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--direct")) {
            const char *v = next_arg(argc, argv, &i);
            direct = !strcmp(v, "spatial") || !strcmp(v, "1") ? 1 :
                     !strcmp(v, "temporal") || !strcmp(v, "0") ? 0 : -1;
            if (direct < 0) {
                fprintf(stderr, "error: --direct spatial|temporal\n");
                return 2;
            }
        }
        else if (!strcmp(a, "--trellis")) trellis = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--partitions")) partitions = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--bitrate")) bitrate = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--aq")) aq = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--aq-mode")) aq_mode = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "--no-8x8dct")) p.transform_8x8 = 0;
        else if (!strcmp(a, "--no-cabac")) p.cabac = 0;
        else if (!strcmp(a, "--open-gop")) p.open_gop = 1;
        else if (!strcmp(a, "--no-deblock")) p.deblock = 0;
        else if (!strcmp(a, "--list-devices")) { list_devices(); return 0; }
        else if (!strcmp(a, "--qpfile")) qpfilepath = next_arg(argc, argv, &i);
        else if (!strcmp(a, "--roi")) {
            const char *v = next_arg(argc, argv, &i);
            int x, y, w, h, off;
            if (sscanf(v, "%d,%d,%d,%d,%d", &x, &y, &w, &h, &off) != 5 ||
                w <= 0 || h <= 0 || off < -51 || off > 51) {
                fprintf(stderr, "error: --roi %s: expected x,y,w,h,off "
                        "(w,h > 0, off -51..51)\n", v);
                return 2;
            }
            if (nrois == roicap) {
                int nc = roicap ? roicap * 2 : 4;
                aeg_roi *nr = (aeg_roi *)realloc(rois, (size_t)nc * sizeof(*nr));
                if (!nr) {
                    fprintf(stderr, "out of memory\n");
                    return 1;
                }
                rois = nr;
                roicap = nc;
            }
            rois[nrois].x = x;
            rois[nrois].y = y;
            rois[nrois].w = w;
            rois[nrois].h = h;
            rois[nrois].qp_offset = off;
            nrois++;
        }
        else if (!strcmp(a, "--zones")) {
            const char *v = next_arg(argc, argv, &i);
            int s, e, q;
            if (sscanf(v, "%d,%d,%d", &s, &e, &q) != 3 || s < 0 || e < s ||
                q < 0 || q > 51) {
                fprintf(stderr, "error: --zones %s: expected start,end,qp "
                        "with 0 <= start <= end and qp 0..51\n", v);
                return 2;
            }
            if (nzones == zonecap) {
                int nc = zonecap ? zonecap * 2 : 4;
                aeg_zone *nz = (aeg_zone *)realloc(zones, (size_t)nc * sizeof(*nz));
                if (!nz) {
                    fprintf(stderr, "out of memory\n");
                    return 1;
                }
                zones = nz;
                zonecap = nc;
            }
            zones[nzones].start = s;
            zones[nzones].end = e;
            zones[nzones].qp = q;
            nzones++;
        }
        else if (!strcmp(a, "--device")) {
            char buf[32];
            int d = int_arg(a, next_arg(argc, argv, &i));
            if (d < 0) {
                fprintf(stderr, "error: --device %d: expected 0 or more\n", d);
                return 2;
            }
            snprintf(buf, sizeof(buf), "%d", d);
            setenv("AEG_DEVICE", buf, 1);
        }
        else if (!strcmp(a, "--calibrate")) {
            int r = aeg_calibrate(backend);
            printf("calibration: %s\n", r == 0 ? "ok" : "accelerator unavailable");
            return r == 0 ? 0 : 1;
        } else if (!strcmp(a, "--bench")) bench = 1;
        else if (!strcmp(a, "--dump-recon")) dumppath = next_arg(argc, argv, &i);
        else if (!strcmp(a, "--dump-order")) orderpath = next_arg(argc, argv, &i);
        else if (!strcmp(a, "--b-adapt")) badapt = int_arg(a, next_arg(argc, argv, &i));
        else if (!strcmp(a, "-s")) {
            char *eq;
            char *arg = next_arg(argc, argv, &i);
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

    {
        int bad = 0;
        bad |= range_bad("-q", qp, 0, 51);
        if (bit_depth != 8 && bit_depth != 10) {
            fprintf(stderr, "error: --bit-depth %d: expected 8 or 10\n", bit_depth);
            bad = 1;
        }
        if (crf_given && crf < 0.0 && bit_depth == 8) {
            fprintf(stderr, "error: --crf %g: below 0 needs --bit-depth 10\n", crf);
            bad = 1;
        }
        bad |= range_bad("-p", p.preset, 0, 4);
        bad |= range_bad("-g", gop, 0, 1 << 30);
        bad |= range_bad("-n", max_frames, 0, 1 << 30);
        bad |= range_bad("--slices", p.slices, 0, 1024);
        bad |= range_bad("--slice-max-mbs", p.slice_max_mbs, 0, 1 << 20);
        bad |= range_bad("--slice-max-size", p.slice_max_size, 0, 1 << 30);
        bad |= range_bad("--intra-refresh", p.intra_refresh, 0, 1 << 20);
        bad |= range_bad("-t", p.threads, 0, 256);
        bad |= range_bad("--frame-threads", p.frame_threads, 0, 8);
        bad |= range_bad("--scenecut", p.scenecut, 0, 10000);
        bad |= range_bad("--pass", pass, 0, 2);
        bad |= range_bad("--qcomp", qcomp, 0, 100);
        bad |= range_bad("--bitrate", bitrate, 0, 2000000);
        bad |= range_bad("--vbv-maxrate", p.vbv_maxrate_kbps, 0, 2000000);
        bad |= range_bad("--vbv-bufsize", p.vbv_bufsize_kbits, 0, 2000000);
        bad |= range_bad("-v", verbose, 0, 3);
        bad |= range_bad("--qpmin", p.qp_min, 0, 51);
        /* qp_max 0 is the library's "unset" (a zeroed aeg_params must not pin
         * every macroblock to QP 0), so a range ending at 0 cannot be asked
         * for; refusing it beats silently coding up to QP 51 */
        bad |= range_bad("--qpmax", p.qp_max, 1, 51);
        bad |= range_bad("--chroma-qp-offset", p.chroma_qp_offset, -12, 12);
        bad |= range_bad("--deblock alpha", p.deblock_alpha, -6, 6);
        bad |= range_bad("--deblock beta", p.deblock_beta, -6, 6);
        bad |= range_bad("--seek", seek, 0, 1 << 30);
        if (p.qp_min > p.qp_max) {
            fprintf(stderr, "error: --qpmin %d is above --qpmax %d\n",
                    p.qp_min, p.qp_max);
            bad = 1;
        }
        if (aq >= 0) bad |= range_bad("--aq", aq, 0, 30);
        if (aq_mode >= 0) bad |= range_bad("--aq-mode", aq_mode, 0, 3);
        if (range) bad |= range_bad("--range", range, 1, 512);
        if (subme >= 0) bad |= range_bad("--subme", subme, 0, 9);
        if (refine_mv >= 0) bad |= range_bad("--refine-mv", refine_mv, 0, 3);
        if (refine_cand >= 0) bad |= range_bad("--refine-cand", refine_cand, 0, 1);
        if (qp_rd >= 0) bad |= range_bad("--qp-rd", qp_rd, 0, 1);
        if (refs >= 0) bad |= range_bad("--ref", refs, 1, 8);
        if (weightp >= 0) bad |= range_bad("--weightp", weightp, 0, 1);
        if (bpyramid >= 0) bad |= range_bad("--b-pyramid", bpyramid, 0, 1);
        if (mbtree >= 0) bad |= range_bad("--mbtree", mbtree, 0, 1);
        if (rclookahead >= 0) bad |= range_bad("--rc-lookahead", rclookahead, 0, 60);
        if (bframes >= 0) bad |= range_bad("--bframes", bframes, 0, 3);
        if (badapt >= 0) bad |= range_bad("--b-adapt", badapt, 0, 2);
        if (bqp > -1000) bad |= range_bad("--bqp", bqp, -51, 51);
        if (iqp > -1000) bad |= range_bad("--iqp", iqp, -51, 51);
        if (trellis >= 0) bad |= range_bad("--trellis", trellis, 0, 2);
        if (partitions >= 0) bad |= range_bad("--partitions", partitions, 0, 31);
        if (in_depth >= 0) bad |= range_bad("--input-depth", in_depth, 8, 16);
        if (bad)
            return 2;
    }
    if (inpath) {
        in = fopen(inpath, "rb");
        if (!in) { perror(inpath); return 1; }
    }
    if (in_open(&inp, in, demux) < 0)
        return 1;
    if (inp.y4m) {
        if ((width && width != inp.w) || (height && height != inp.h)) {
            fprintf(stderr, "error: -w/-h %dx%d contradict the YUV4MPEG2 "
                            "header's %dx%d\n", width, height, inp.w, inp.h);
            return 2;
        }
        width = inp.w, height = inp.h;
        if (!fps_set && inp.fps_num > 0)
            fps_num = inp.fps_num, fps_den = inp.fps_den;
        if (!sar_w && inp.sar_w > 0)
            sar_w = inp.sar_w, sar_h = inp.sar_h;
        if (inp.fullrange)
            fullrange = 1;
        if (inp.csp >= 0) {
            if ((in_csp >= 0 && in_csp != inp.csp) ||
                (in_depth >= 0 && in_depth != inp.depth)) {
                fprintf(stderr, "error: --input-csp/--input-depth contradict "
                                "the YUV4MPEG2 header's colour space\n");
                return 2;
            }
            in_csp = inp.csp;
            in_depth = inp.depth;
        }
    }
    if (in_csp < 0) in_csp = AEG_CSP_I420;
    if (in_depth < 0) in_depth = 8;
    fps = (double)fps_num / (double)fps_den;
    if (!width || !height) {
        fprintf(stderr, "error: -w and -h are required (or a .y4m input)\n");
        return 2;
    }
    if (width < 2 || height < 2 || (width & 1) || (height & 1) ||
        width > 16384 || height > 16384) {
        fprintf(stderr, "error: %dx%d: the frame size must be even, 2..16384 "
                        "(4:2:0 is cropped in units of two samples)\n",
                width, height);
        return 2;
    }
    if ((interlaced > 0 || (interlaced < 0 && inp.interlaced)) && (height & 3)) {
        fprintf(stderr, "error: %dx%d: an interlaced stream needs a height "
                        "that is a multiple of 4\n", width, height);
        return 2;
    }
    if (outpath) {
        out = fopen(outpath, "wb");
        if (!out) { perror(outpath); return 1; }
    }

    /* Two-pass keys its measurement by display index (the packet's pts): with
     * B pictures the packets come out in coding order. */
    if (crf_given && (bitrate > 0 || pass)) {
        fprintf(stderr, "error: --crf cannot be combined with %s\n",
                bitrate > 0 ? "--bitrate" : "--pass");
        return 2;
    }
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
            /* "aeg-stats 2" then "display type qp qp_base bits" per picture
             * (type I, P, B or b for a reference B; qp the macroblocks'
             * average, qp_base the picture's QP before their offsets);
             * a file of bare numbers is the first format, one bit count per
             * picture in display order without B pictures */
            FILE *sf = fopen(stats_path, "r");
            char line[256];
            int v2 = 0, k = 0;

            if (!sf) {
                fprintf(stderr, "cannot read statistics file %s\n", stats_path);
                return 1;
            }
            while (fgets(line, sizeof(line), sf)) {
                long long d, b;
                double q, qb;
                int ty;
                char t;

                if (!strncmp(line, "aeg-stats 2", 11)) {
                    const char *kb = strstr(line, "bframes=");
                    v2 = 1;
                    if (kb)
                        pass1_bframes = atoi(kb + 8);
                    continue;
                }
                if (v2) {
                    /* the pictures are numbered from 0 in display order: an
                     * index far past the ones before it is damage, and the
                     * table is allocated up to it */
                    if (sscanf(line, "%lld %c %lf %lf %lld", &d, &t, &q, &qb,
                               &b) != 5 ||
                        !strchr("IPBb", t) || t == 0 || b < 0 || d < 0 ||
                        d > (long long)pass1_n + 65536) {
                        fprintf(stderr, "%s: bad line: %s", stats_path, line);
                        fclose(sf);
                        free(pass1);
                        return 1;
                    }
                    ty = t == 'I' ? 0 : t == 'B' ? 2 : t == 'b' ? 3 : 1;
                } else {
                    if (sscanf(line, "%lld", &b) != 1)
                        break;
                    d = k++;
                    q = qb = qp;
                    ty = 1;
                }
                if (pass1_put(&pass1, &pass1_n, &pass1_cap, d, ty, q, qb, b) < 0) {
                    fprintf(stderr, "out of memory\n");
                    fclose(sf);
                    free(pass1);
                    return 1;
                }
            }
            fclose(sf);
            if (!v2)
                pass1_bframes = 0;   /* the first format is P-only */
            if (pass1_n <= 0) {
                fprintf(stderr, "no statistics in %s\n", stats_path);
                free(pass1);
                return 1;
            }
            /* the plan is for the pictures this pass codes: with fewer (-n)
             * it would share the target out over pictures that never come
             * (+26% coding 10 of 40) */
            if (max_frames && pass1_n < max_frames)
                max_frames = pass1_n;
            else if (max_frames && max_frames < pass1_n)
                pass1_n = max_frames;
        }
    }

    if (qpfilepath) {
        qptab = load_qpfile(qpfilepath, &qpn);
        if (!qptab)
            return 1;
        p.frame_qp = qptab;
        p.n_frame_qp = qpn;
    }
    p.zones = zones;
    p.n_zones = nzones;
    p.rois = rois;
    p.n_rois = nrois;
    p.width = width;
    p.height = height;
    p.qp = qp;
    p.fps_num = fps_num;
    p.fps_den = fps_den;
    p.sar_w = sar_w;
    p.sar_h = sar_h;
    p.fullrange = fullrange;
    if (md_set) {
        int k;
        p.master_display = 1;
        for (k = 0; k < 8; k++)
            p.md_primaries[k] = md[k];
        p.md_max_lum = md_max_lum;
        p.md_min_lum = md_min_lum;
    }
    p.max_cll = max_cll;
    p.max_fall = max_fall;
    p.interlaced = interlaced >= 0 ? interlaced : inp.interlaced;
    p.field_coding = field_coding;
    if (field_coding && !p.interlaced)
        p.interlaced = 1;
    p.gop_size = gop;
    p.bitrate_kbps = bitrate;
    if (aq >= 0) {
        p.aq_strength = aq;
        /* --aq alone keeps the historical variance mode; the auto modes are
         * opt-in. */
        if (aq_mode < 0) p.aq_mode = aq > 0 ? 1 : 0;
    }
    if (aq_mode >= 0) p.aq_mode = aq_mode;
    p.rc_mode = bitrate > 0 ? AEG_RC_ABR : AEG_RC_CQP;
    p.qcomp = qcomp;
    p.bit_depth = bit_depth;
    if (crf_given) {
        p.rc_mode = AEG_RC_CRF;
        p.crf = crf;
    }
    p.backend = backend;
    p.verbose = verbose;
    if (range) p.search_range = range;
    if (subme >= 0) p.subme = subme;
    if (refine_mv >= 0) p.refine_mv = refine_mv;
    if (refine_cand >= 0) p.refine_cand = refine_cand;
    if (qp_rd >= 0) p.qp_rd = qp_rd;
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
    /* -v 0 = errors only, 1 = warnings and the encoder's summary (info),
     * 2 and up = every picture (debug) */
    aeg_set_log_level(verbose ? (verbose + 1 > AEG_LOG_DEBUG ? AEG_LOG_DEBUG
                                                             : verbose + 1)
                              : AEG_LOG_ERROR);

    fsize = frame_layout(in_csp, in_depth, width, height, &layout, poff);
    frame = (uint8_t *)malloc(fsize);
    g_bps = bit_depth > 8 ? 2 : 1;
    if (verbose && (in_csp != AEG_CSP_I420 || in_depth != bit_depth))
        fprintf(stderr, "input: %s, %d bit -> 4:2:0, %d bit%s\n",
                csps[csp_index(in_csp)].name, in_depth, bit_depth,
                in_depth > bit_depth || csps[csp_index(in_csp)].ys == 0 ||
                csps[csp_index(in_csp)].xs == 0 ? " (dithered)" : "");
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
            if (fseek(in, 0, SEEK_SET) != 0 || in_open(&inp, in, demux) < 0) {
                fprintf(stderr, "--bench needs a seekable input file\n");
                return 1;
            }
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
                if (in_frame(&inp, frame, fsize) != 1)
                    break;
                r = encode_frame(e, frame, fsize, &layout, poff, pts++, &pkt);
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
            /* the content bitrate (bits per frame second of the source), not
             * bytes over the time the encoder took: the old expression
             * multiplied by the frame rate *and* divided by the elapsed time,
             * so it reported a number that grew as the encoder got slower and
             * could not be compared between runs. */
            if (verbose)
                printf("%-6s %6" PRId64 " frames %8.2f ms/frame %8.0f kb/s\n",
                       aeg_backend_name(q.backend), n, tot / 1e6 / (n ? n : 1),
                       n ? bytes * 8.0 / (n / (double)fps) / 1000.0 : 0.0);
            aeg_encoder_close(e);
        }
        if (t[1] > 0 && t[0] > 0)
            printf("speedup (auto vs cpu): %.2fx\n", (double)t[0] / (double)t[1]);
        free(frame);
        return 0;
    }

    if (seek > 0) {
        int k;
        for (k = 0; k < seek; k++)
            if (in_frame(&inp, frame, fsize) != 1) {
                fprintf(stderr, "error: --seek %d: the input has only %d "
                                "frame(s)\n", seek, k);
                free(frame);
                return 1;
            }
    }
    enc = aeg_encoder_open(&p);
    if (!enc) {
        fprintf(stderr, "aeg_encoder_open failed: %s\n", aeg_last_error());
        free(frame);
        free(pass1);
        return 1;
    }

    {
        int64_t t0 = aeg_now_ns();
        int n = 0;

        /* the second pass has to code the picture types the first measured
         * (x264 refuses other B-picture settings too) */
        if (pass == 2 && pass1_bframes >= 0) {
            aeg_stats bs;
            aeg_encoder_stats(enc, &bs);
            if (bs.bframes != pass1_bframes) {
                fprintf(stderr, "%s was measured with %d B pictures per group, "
                        "this pass codes %d: use the same options as pass 1\n",
                        stats_path, pass1_bframes, bs.bframes);
                aeg_encoder_close(enc);
                free(frame);
                free(pass1);
                return 1;
            }
        }
        /* the second pass plans every picture from the first's measurement */
        if (pass == 2 && aeg_encoder_set_pass1(enc, pass1, pass1_n) != AEG_OK) {
            fprintf(stderr, "cannot use the statistics in %s: %s\n", stats_path,
                    aeg_last_error());
            aeg_encoder_close(enc);
            free(frame);
            free(pass1);
            return 1;
        }
        while (in_frame(&inp, frame, fsize) == 1) {
            aeg_packet pkt;
            int r;

            r = encode_frame(enc, frame, fsize, &layout, poff, pts++, &pkt);
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
            if (pass == 1 &&
                pass1_put(&pass1, &pass1_n, &pass1_cap, pkt.pts, pkt.type,
                          pkt.qp_avg, pkt.qp_base, (int64_t)pkt.size * 8) < 0) {
                fprintf(stderr, "out of memory\n");
                ret = 1;
                break;
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
            if (pass == 1 &&
                pass1_put(&pass1, &pass1_n, &pass1_cap, pkt.pts, pkt.type,
                          pkt.qp_avg, pkt.qp_base, (int64_t)pkt.size * 8) < 0) {
                fprintf(stderr, "out of memory\n");
                ret = 1;
                break;
            }
        }
        if (pass == 1) {
            FILE *sf = fopen(stats_path, "w");

            if (!sf) {
                fprintf(stderr, "cannot write statistics file %s\n", stats_path);
                ret = 1;
            } else {
                int k;
                aeg_stats bs;
                aeg_encoder_stats(enc, &bs);
                fprintf(sf, "aeg-stats 2 bframes=%d\n", bs.bframes);
                for (k = 0; k < pass1_n; k++)
                    if (pass1[k].bits >= 0)
                        fprintf(sf, "%d %c %.3f %.3f %lld\n", k,
                                "IPBb"[pass1[k].type], pass1[k].qp,
                                pass1[k].qp_base, (long long)pass1[k].bits);
                fclose(sf);
                if (verbose)
                    printf("pass 1: measured %d pictures -> %s\n", pass1_n,
                           stats_path);
            }
        }
        free(pass1);
        pass1 = NULL;
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
            printf("bitrate %.1f kb/s at %g fps, PSNR %.2f dB (Y+U+V average)\n",
                   st.frames > 0
                       ? st.bytes * 8.0 * fps / (double)st.frames / 1000.0
                       : 0.0,
                   fps, st.psnr_avg);
            if (st.scene_cuts)
                printf("scenecut: %" PRId64 " IDR picture(s) inserted\n",
                       st.scene_cuts);
            if (st.vbv_min_fill_bits > 0.0 || st.vbv_underflows)
                printf("vbv: buffer low water %.1f kbit, %" PRId64
                       " underflow(s), %" PRId64 " picture(s) raised to fit, %"
                       PRId64 " coded again\n",
                       st.vbv_min_fill_bits / 1000.0, st.vbv_underflows,
                       st.vbv_size_clamps, st.vbv_recodes);
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
