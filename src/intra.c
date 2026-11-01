/*
 * Intra prediction (H.264 8.3.3): Intra_16x16, Intra_4x4 and 8x8 chroma.
 *
 * Unavailable neighbour samples are substituted as specified in 8.3.3.1:
 * a whole missing side is replaced by the mid-level sample (128 for 8-bit)
 * while missing samples inside an available side are replicated from the
 * closest available sample.
 */
#include "h264.h"

/* ------------------------------------------------------------------ */
/* Intra_16x16 (and 8x8 chroma when size == 8)                         */
/* ------------------------------------------------------------------ */
static void intra_16x16_dc(pixel *pred, int stride, const pixel *top,
                           const pixel *left, int t_av, int l_av)
{
    int x, y, sum = 0, dc;
    if (t_av && l_av) {
        for (x = 0; x < 16; x++) sum += top[x];
        for (y = 0; y < 16; y++) sum += left[y];
        dc = (sum + 16) >> 5;
    } else if (t_av) {
        for (x = 0; x < 16; x++) sum += top[x];
        dc = (sum + 8) >> 4;
    } else if (l_av) {
        for (y = 0; y < 16; y++) sum += left[y];
        dc = (sum + 8) >> 4;
    } else {
        dc = PIXEL_MID;
    }
    for (y = 0; y < 16; y++)
        for (x = 0; x < 16; x++)
            pred[y * stride + x] = (pixel)dc;
}

static void intra_16x16_plane(pixel *pred, int stride, const pixel *top,
                              const pixel *left, int topleft)
{
    int x, y, H = 0, V = 0, a, b, c;

    for (x = 1; x <= 8; x++)
        H += x * (x == 8 ? top[15] - topleft : top[x + 7] - top[7 - x]);
    for (y = 1; y <= 8; y++)
        V += y * (y == 8 ? left[15] - topleft : left[y + 7] - left[7 - y]);

    a = 16 * (left[15] + top[15]);
    b = (5 * H + 32) >> 6;
    c = (5 * V + 32) >> 6;
    for (y = 0; y < 16; y++)
        for (x = 0; x < 16; x++)
            pred[y * stride + x] = aeg_clip_pixel((a + b * (x - 7) + c * (y - 7) + 16) >> 5);
}

void aeg_intra_predict_16x16(pixel *pred, int stride, const pixel *top,
                             const pixel *left, int top_avail, int left_avail,
                             int topleft_avail, int mode)
{
    int x, y;
    switch (mode) {
    case AEG_MB_I_V:  /* vertical */
        for (y = 0; y < 16; y++)
            for (x = 0; x < 16; x++) pred[y * stride + x] = top[x];
        break;
    case AEG_MB_I_H:  /* horizontal */
        for (y = 0; y < 16; y++)
            for (x = 0; x < 16; x++) pred[y * stride + x] = left[y];
        break;
    case AEG_MB_I_DC:
        intra_16x16_dc(pred, stride, top, left, top_avail, left_avail);
        break;
    default:
        intra_16x16_plane(pred, stride, top, left,
                          topleft_avail ? top[-1] : PIXEL_MID);
        break;
    }
}

/* ------------------------------------------------------------------ */
/* Chroma 8x8                                                          */
/* ------------------------------------------------------------------ */
static void intra_chroma_dc(pixel *pred, int stride, const pixel *top,
                            const pixel *left, int t_av, int l_av)
{
    /* H.264 8.3.4.3: unlike the 16x16 luma DC mode, the chroma DC mode gives
     * each of the four 4x4 quadrants its own average. */
    int tl = 0, tr = 0, bl = 0, x, y;
    int dc[4];
    if (t_av && l_av) {
        for (x = 0; x < 4; x++) { tl += top[x] + left[x]; tr += top[4 + x]; bl += left[4 + x]; }
        dc[0] = (tl + 4) >> 3;
        dc[1] = (tr + 2) >> 2;
        dc[2] = (bl + 2) >> 2;
        dc[3] = (tr + bl + 4) >> 3;
    } else if (t_av) {
        for (x = 0; x < 4; x++) { tl += top[x]; tr += top[4 + x]; }
        dc[0] = dc[2] = (tl + 2) >> 2;
        dc[1] = dc[3] = (tr + 2) >> 2;
    } else if (l_av) {
        for (x = 0; x < 4; x++) { tl += left[x]; bl += left[4 + x]; }
        dc[0] = dc[1] = (tl + 2) >> 2;
        dc[2] = dc[3] = (bl + 2) >> 2;
    } else {
        dc[0] = dc[1] = dc[2] = dc[3] = PIXEL_MID;
    }
    for (y = 0; y < 8; y++)
        for (x = 0; x < 8; x++)
            pred[y * stride + x] = (pixel)dc[(y >> 2) * 2 + (x >> 2)];
}

static void intra_chroma_plane(pixel *pred, int stride, const pixel *top,
                               const pixel *left, int topleft)
{
    int x, y, H = 0, V = 0, a, b;
    for (x = 1; x <= 4; x++)
        H += x * (x == 4 ? top[7] - topleft : top[x + 3] - top[3 - x]);
    for (y = 1; y <= 4; y++)
        V += y * (y == 4 ? left[7] - topleft : left[y + 3] - left[3 - y]);
    /* 8x8 plane prediction (H.264 8.3.4.4): the gradients are normalised
     * with (17*H + 16) >> 5 and the interpolation runs from the bottom-right
     * corner sample, exactly like the reference decoder's pred8x8_plane. */
    H = (17 * H + 16) >> 5;
    V = (17 * V + 16) >> 5;
    a = 16 * (left[7] + top[7] + 1) - 3 * (V + H);
    for (y = 0; y < 8; y++) {
        b = a;
        a += V;
        for (x = 0; x < 8; x++)
            pred[y * stride + x] = aeg_clip_pixel((b + x * H) >> 5);
    }
}

void aeg_intra_predict_chroma(pixel *pred, int stride, const pixel *top,
                              const pixel *left, int top_avail, int left_avail,
                              int topleft_avail, int mode)
{
    int x, y;
    switch (mode) {
    case 0:  /* DC (bitstream order: 0=DC, 1=Hor, 2=Ver, 3=Plane) */
        intra_chroma_dc(pred, stride, top, left, top_avail, left_avail);
        break;
    case 1:
        for (y = 0; y < 8; y++)
            for (x = 0; x < 8; x++) pred[y * stride + x] = left[y];
        break;
    case 2:
        for (y = 0; y < 8; y++)
            for (x = 0; x < 8; x++) pred[y * stride + x] = top[x];
        break;
    default:
        intra_chroma_plane(pred, stride, top, left, topleft_avail ? top[-1] : PIXEL_MID);
        break;
    }
}

/* ------------------------------------------------------------------ */
/* Intra_8x8 (High profile, 8.3.3.4)                                   */
/* ------------------------------------------------------------------ */
#define I8_F1(a, b)    (((a) + (b) + 1) >> 1)
#define I8_F2(a, b, c) (((a) + 2 * (b) + (c) + 2) >> 2)
/* The nine predictors below are transcribed from the reference encoder's
 * predict_8x8_* bodies, which spell the two interpolators F1/F2. */
#define F1(a, b)       I8_F1(a, b)
#define F2(a, b, c)    I8_F2(a, b, c)

/* The eight-by-eight prediction modes are the same nine directions as the
 * four-by-four ones, but they read *filtered* reference samples z[]: a low-pass
 * along the neighbouring row and column (which is what stops the 1/4-sample
 * diagonal interpolation of the larger block from ringing) and, at the two
 * ends, a one-sided filter.  The prediction equations below are those of
 * 8.3.3.4.3 with z in place of p.
 *
 * Availability: the caller substitutes an unavailable side with the first
 * sample of the other one (8.3.3.1) -- so an unavailable top-right corner,
 * which this signature has no flag for, is simply the block's last top sample
 * repeated, which is exactly what the filtering then produces as well. */
void aeg_intra_predict_8x8(pixel *pred, int stride, const pixel *top,
                           const pixel *left, int topleft, int mode,
                           int top_avail, int left_avail, int topleft_avail)
{
    int rt[16], rl[8], rlt, zt[16], zl[8], zlt, x, y, i, dc;
    pixel p[64];

    for (x = 0; x < 16; x++)
        rt[x] = top_avail ? top[x] : (left_avail ? left[0] : PIXEL_MID);
    for (y = 0; y < 8; y++)
        rl[y] = left_avail ? left[y] : (top_avail ? top[0] : PIXEL_MID);
    rlt = topleft_avail ? topleft
                        : (top_avail ? top[0] : (left_avail ? left[0] : PIXEL_MID));

    zlt = I8_F2(rt[0], rlt, rl[0]);
    zt[0] = I8_F2(rlt, rt[0], rt[1]);
    for (x = 1; x < 7; x++)
        zt[x] = I8_F2(rt[x - 1], rt[x], rt[x + 1]);
    zt[7] = I8_F2(rt[6], rt[7], rt[8]);
    for (x = 8; x < 15; x++)
        zt[x] = I8_F2(rt[x - 1], rt[x], rt[x + 1]);
    zt[15] = (rt[14] + 3 * rt[15] + 2) >> 2;
    zl[0] = I8_F2(rlt, rl[0], rl[1]);
    for (y = 1; y < 7; y++)
        zl[y] = I8_F2(rl[y - 1], rl[y], rl[y + 1]);
    zl[7] = (rl[6] + 3 * rl[7] + 2) >> 2;

    switch (mode) {
    case 0:                                     /* vertical */
        for (y = 0; y < 8; y++)
            for (x = 0; x < 8; x++)
                p[y * 8 + x] = (pixel)zt[x];
        break;
    case 1:                                     /* horizontal */
        for (y = 0; y < 8; y++)
            for (x = 0; x < 8; x++)
                p[y * 8 + x] = (pixel)zl[y];
        break;
    case 2:                                     /* DC */
        if (top_avail && left_avail) {
            dc = 8;
            for (i = 0; i < 8; i++)
                dc += zt[i] + zl[i];
            dc >>= 4;
        } else if (top_avail) {
            dc = 4;
            for (i = 0; i < 8; i++)
                dc += zt[i];
            dc >>= 3;
        } else if (left_avail) {
            dc = 4;
            for (i = 0; i < 8; i++)
                dc += zl[i];
            dc >>= 3;
        } else {
            dc = PIXEL_MID;
        }
        for (i = 0; i < 64; i++)
            p[i] = (pixel)dc;
        break;
    case 3:                                     /* diagonal down-left */
    p[0] = (pixel)F2(zt[0],zt[1],zt[2]);
    p[1] = (pixel)F2(zt[1],zt[2],zt[3]);
    p[2] = (pixel)F2(zt[2],zt[3],zt[4]);
    p[3] = (pixel)F2(zt[3],zt[4],zt[5]);
    p[4] = (pixel)F2(zt[4],zt[5],zt[6]);
    p[5] = (pixel)F2(zt[5],zt[6],zt[7]);
    p[6] = (pixel)F2(zt[6],zt[7],zt[8]);
    p[7] = (pixel)F2(zt[7],zt[8],zt[9]);
    p[8] = (pixel)F2(zt[1],zt[2],zt[3]);
    p[9] = (pixel)F2(zt[2],zt[3],zt[4]);
    p[10] = (pixel)F2(zt[3],zt[4],zt[5]);
    p[11] = (pixel)F2(zt[4],zt[5],zt[6]);
    p[12] = (pixel)F2(zt[5],zt[6],zt[7]);
    p[13] = (pixel)F2(zt[6],zt[7],zt[8]);
    p[14] = (pixel)F2(zt[7],zt[8],zt[9]);
    p[15] = (pixel)F2(zt[8],zt[9],zt[10]);
    p[16] = (pixel)F2(zt[2],zt[3],zt[4]);
    p[17] = (pixel)F2(zt[3],zt[4],zt[5]);
    p[18] = (pixel)F2(zt[4],zt[5],zt[6]);
    p[19] = (pixel)F2(zt[5],zt[6],zt[7]);
    p[20] = (pixel)F2(zt[6],zt[7],zt[8]);
    p[21] = (pixel)F2(zt[7],zt[8],zt[9]);
    p[22] = (pixel)F2(zt[8],zt[9],zt[10]);
    p[23] = (pixel)F2(zt[9],zt[10],zt[11]);
    p[24] = (pixel)F2(zt[3],zt[4],zt[5]);
    p[25] = (pixel)F2(zt[4],zt[5],zt[6]);
    p[26] = (pixel)F2(zt[5],zt[6],zt[7]);
    p[27] = (pixel)F2(zt[6],zt[7],zt[8]);
    p[28] = (pixel)F2(zt[7],zt[8],zt[9]);
    p[29] = (pixel)F2(zt[8],zt[9],zt[10]);
    p[30] = (pixel)F2(zt[9],zt[10],zt[11]);
    p[31] = (pixel)F2(zt[10],zt[11],zt[12]);
    p[32] = (pixel)F2(zt[4],zt[5],zt[6]);
    p[33] = (pixel)F2(zt[5],zt[6],zt[7]);
    p[34] = (pixel)F2(zt[6],zt[7],zt[8]);
    p[35] = (pixel)F2(zt[7],zt[8],zt[9]);
    p[36] = (pixel)F2(zt[8],zt[9],zt[10]);
    p[37] = (pixel)F2(zt[9],zt[10],zt[11]);
    p[38] = (pixel)F2(zt[10],zt[11],zt[12]);
    p[39] = (pixel)F2(zt[11],zt[12],zt[13]);
    p[40] = (pixel)F2(zt[5],zt[6],zt[7]);
    p[41] = (pixel)F2(zt[6],zt[7],zt[8]);
    p[42] = (pixel)F2(zt[7],zt[8],zt[9]);
    p[43] = (pixel)F2(zt[8],zt[9],zt[10]);
    p[44] = (pixel)F2(zt[9],zt[10],zt[11]);
    p[45] = (pixel)F2(zt[10],zt[11],zt[12]);
    p[46] = (pixel)F2(zt[11],zt[12],zt[13]);
    p[47] = (pixel)F2(zt[12],zt[13],zt[14]);
    p[48] = (pixel)F2(zt[6],zt[7],zt[8]);
    p[49] = (pixel)F2(zt[7],zt[8],zt[9]);
    p[50] = (pixel)F2(zt[8],zt[9],zt[10]);
    p[51] = (pixel)F2(zt[9],zt[10],zt[11]);
    p[52] = (pixel)F2(zt[10],zt[11],zt[12]);
    p[53] = (pixel)F2(zt[11],zt[12],zt[13]);
    p[54] = (pixel)F2(zt[12],zt[13],zt[14]);
    p[55] = (pixel)F2(zt[13],zt[14],zt[15]);
    p[56] = (pixel)F2(zt[7],zt[8],zt[9]);
    p[57] = (pixel)F2(zt[8],zt[9],zt[10]);
    p[58] = (pixel)F2(zt[9],zt[10],zt[11]);
    p[59] = (pixel)F2(zt[10],zt[11],zt[12]);
    p[60] = (pixel)F2(zt[11],zt[12],zt[13]);
    p[61] = (pixel)F2(zt[12],zt[13],zt[14]);
    p[62] = (pixel)F2(zt[13],zt[14],zt[15]);
    p[63] = (pixel)F2(zt[14],zt[15],zt[15]);
    break;
    case 4:                                     /* diagonal down-right */
    p[0] = (pixel)F2(zl[0],zlt,zt[0]);
    p[1] = (pixel)F2(zlt,zt[0],zt[1]);
    p[2] = (pixel)F2(zt[0],zt[1],zt[2]);
    p[3] = (pixel)F2(zt[1],zt[2],zt[3]);
    p[4] = (pixel)F2(zt[2],zt[3],zt[4]);
    p[5] = (pixel)F2(zt[3],zt[4],zt[5]);
    p[6] = (pixel)F2(zt[4],zt[5],zt[6]);
    p[7] = (pixel)F2(zt[5],zt[6],zt[7]);
    p[8] = (pixel)F2(zl[1],zl[0],zlt);
    p[9] = (pixel)F2(zl[0],zlt,zt[0]);
    p[10] = (pixel)F2(zlt,zt[0],zt[1]);
    p[11] = (pixel)F2(zt[0],zt[1],zt[2]);
    p[12] = (pixel)F2(zt[1],zt[2],zt[3]);
    p[13] = (pixel)F2(zt[2],zt[3],zt[4]);
    p[14] = (pixel)F2(zt[3],zt[4],zt[5]);
    p[15] = (pixel)F2(zt[4],zt[5],zt[6]);
    p[16] = (pixel)F2(zl[2],zl[1],zl[0]);
    p[17] = (pixel)F2(zl[1],zl[0],zlt);
    p[18] = (pixel)F2(zl[0],zlt,zt[0]);
    p[19] = (pixel)F2(zlt,zt[0],zt[1]);
    p[20] = (pixel)F2(zt[0],zt[1],zt[2]);
    p[21] = (pixel)F2(zt[1],zt[2],zt[3]);
    p[22] = (pixel)F2(zt[2],zt[3],zt[4]);
    p[23] = (pixel)F2(zt[3],zt[4],zt[5]);
    p[24] = (pixel)F2(zl[3],zl[2],zl[1]);
    p[25] = (pixel)F2(zl[2],zl[1],zl[0]);
    p[26] = (pixel)F2(zl[1],zl[0],zlt);
    p[27] = (pixel)F2(zl[0],zlt,zt[0]);
    p[28] = (pixel)F2(zlt,zt[0],zt[1]);
    p[29] = (pixel)F2(zt[0],zt[1],zt[2]);
    p[30] = (pixel)F2(zt[1],zt[2],zt[3]);
    p[31] = (pixel)F2(zt[2],zt[3],zt[4]);
    p[32] = (pixel)F2(zl[4],zl[3],zl[2]);
    p[33] = (pixel)F2(zl[3],zl[2],zl[1]);
    p[34] = (pixel)F2(zl[2],zl[1],zl[0]);
    p[35] = (pixel)F2(zl[1],zl[0],zlt);
    p[36] = (pixel)F2(zl[0],zlt,zt[0]);
    p[37] = (pixel)F2(zlt,zt[0],zt[1]);
    p[38] = (pixel)F2(zt[0],zt[1],zt[2]);
    p[39] = (pixel)F2(zt[1],zt[2],zt[3]);
    p[40] = (pixel)F2(zl[5],zl[4],zl[3]);
    p[41] = (pixel)F2(zl[4],zl[3],zl[2]);
    p[42] = (pixel)F2(zl[3],zl[2],zl[1]);
    p[43] = (pixel)F2(zl[2],zl[1],zl[0]);
    p[44] = (pixel)F2(zl[1],zl[0],zlt);
    p[45] = (pixel)F2(zl[0],zlt,zt[0]);
    p[46] = (pixel)F2(zlt,zt[0],zt[1]);
    p[47] = (pixel)F2(zt[0],zt[1],zt[2]);
    p[48] = (pixel)F2(zl[6],zl[5],zl[4]);
    p[49] = (pixel)F2(zl[5],zl[4],zl[3]);
    p[50] = (pixel)F2(zl[4],zl[3],zl[2]);
    p[51] = (pixel)F2(zl[3],zl[2],zl[1]);
    p[52] = (pixel)F2(zl[2],zl[1],zl[0]);
    p[53] = (pixel)F2(zl[1],zl[0],zlt);
    p[54] = (pixel)F2(zl[0],zlt,zt[0]);
    p[55] = (pixel)F2(zlt,zt[0],zt[1]);
    p[56] = (pixel)F2(zl[7],zl[6],zl[5]);
    p[57] = (pixel)F2(zl[6],zl[5],zl[4]);
    p[58] = (pixel)F2(zl[5],zl[4],zl[3]);
    p[59] = (pixel)F2(zl[4],zl[3],zl[2]);
    p[60] = (pixel)F2(zl[3],zl[2],zl[1]);
    p[61] = (pixel)F2(zl[2],zl[1],zl[0]);
    p[62] = (pixel)F2(zl[1],zl[0],zlt);
    p[63] = (pixel)F2(zl[0],zlt,zt[0]);
    break;
    case 5:                                     /* vertical-right */
    p[0] = (pixel)F1(zlt,zt[0]);
    p[1] = (pixel)F1(zt[0],zt[1]);
    p[2] = (pixel)F1(zt[1],zt[2]);
    p[3] = (pixel)F1(zt[2],zt[3]);
    p[4] = (pixel)F1(zt[3],zt[4]);
    p[5] = (pixel)F1(zt[4],zt[5]);
    p[6] = (pixel)F1(zt[5],zt[6]);
    p[7] = (pixel)F1(zt[6],zt[7]);
    p[8] = (pixel)F2(zl[0],zlt,zt[0]);
    p[9] = (pixel)F2(zlt,zt[0],zt[1]);
    p[10] = (pixel)F2(zt[0],zt[1],zt[2]);
    p[11] = (pixel)F2(zt[1],zt[2],zt[3]);
    p[12] = (pixel)F2(zt[2],zt[3],zt[4]);
    p[13] = (pixel)F2(zt[3],zt[4],zt[5]);
    p[14] = (pixel)F2(zt[4],zt[5],zt[6]);
    p[15] = (pixel)F2(zt[5],zt[6],zt[7]);
    p[16] = (pixel)F2(zl[1],zl[0],zlt);
    p[17] = (pixel)F1(zlt,zt[0]);
    p[18] = (pixel)F1(zt[0],zt[1]);
    p[19] = (pixel)F1(zt[1],zt[2]);
    p[20] = (pixel)F1(zt[2],zt[3]);
    p[21] = (pixel)F1(zt[3],zt[4]);
    p[22] = (pixel)F1(zt[4],zt[5]);
    p[23] = (pixel)F1(zt[5],zt[6]);
    p[24] = (pixel)F2(zl[2],zl[1],zl[0]);
    p[25] = (pixel)F2(zl[0],zlt,zt[0]);
    p[26] = (pixel)F2(zlt,zt[0],zt[1]);
    p[27] = (pixel)F2(zt[0],zt[1],zt[2]);
    p[28] = (pixel)F2(zt[1],zt[2],zt[3]);
    p[29] = (pixel)F2(zt[2],zt[3],zt[4]);
    p[30] = (pixel)F2(zt[3],zt[4],zt[5]);
    p[31] = (pixel)F2(zt[4],zt[5],zt[6]);
    p[32] = (pixel)F2(zl[3],zl[2],zl[1]);
    p[33] = (pixel)F2(zl[1],zl[0],zlt);
    p[34] = (pixel)F1(zlt,zt[0]);
    p[35] = (pixel)F1(zt[0],zt[1]);
    p[36] = (pixel)F1(zt[1],zt[2]);
    p[37] = (pixel)F1(zt[2],zt[3]);
    p[38] = (pixel)F1(zt[3],zt[4]);
    p[39] = (pixel)F1(zt[4],zt[5]);
    p[40] = (pixel)F2(zl[4],zl[3],zl[2]);
    p[41] = (pixel)F2(zl[2],zl[1],zl[0]);
    p[42] = (pixel)F2(zl[0],zlt,zt[0]);
    p[43] = (pixel)F2(zlt,zt[0],zt[1]);
    p[44] = (pixel)F2(zt[0],zt[1],zt[2]);
    p[45] = (pixel)F2(zt[1],zt[2],zt[3]);
    p[46] = (pixel)F2(zt[2],zt[3],zt[4]);
    p[47] = (pixel)F2(zt[3],zt[4],zt[5]);
    p[48] = (pixel)F2(zl[5],zl[4],zl[3]);
    p[49] = (pixel)F2(zl[3],zl[2],zl[1]);
    p[50] = (pixel)F2(zl[1],zl[0],zlt);
    p[51] = (pixel)F1(zlt,zt[0]);
    p[52] = (pixel)F1(zt[0],zt[1]);
    p[53] = (pixel)F1(zt[1],zt[2]);
    p[54] = (pixel)F1(zt[2],zt[3]);
    p[55] = (pixel)F1(zt[3],zt[4]);
    p[56] = (pixel)F2(zl[6],zl[5],zl[4]);
    p[57] = (pixel)F2(zl[4],zl[3],zl[2]);
    p[58] = (pixel)F2(zl[2],zl[1],zl[0]);
    p[59] = (pixel)F2(zl[0],zlt,zt[0]);
    p[60] = (pixel)F2(zlt,zt[0],zt[1]);
    p[61] = (pixel)F2(zt[0],zt[1],zt[2]);
    p[62] = (pixel)F2(zt[1],zt[2],zt[3]);
    p[63] = (pixel)F2(zt[2],zt[3],zt[4]);
    break;
    case 6:                                     /* horizontal-down */
    p[0] = (pixel)F1(zlt,zl[0]);
    p[1] = (pixel)F2(zl[0],zlt,zt[0]);
    p[2] = (pixel)F2(zt[1],zt[0],zlt);
    p[3] = (pixel)F2(zt[2],zt[1],zt[0]);
    p[4] = (pixel)F2(zt[3],zt[2],zt[1]);
    p[5] = (pixel)F2(zt[4],zt[3],zt[2]);
    p[6] = (pixel)F2(zt[5],zt[4],zt[3]);
    p[7] = (pixel)F2(zt[6],zt[5],zt[4]);
    p[8] = (pixel)F1(zl[0],zl[1]);
    p[9] = (pixel)F2(zlt,zl[0],zl[1]);
    p[10] = (pixel)F1(zlt,zl[0]);
    p[11] = (pixel)F2(zl[0],zlt,zt[0]);
    p[12] = (pixel)F2(zt[1],zt[0],zlt);
    p[13] = (pixel)F2(zt[2],zt[1],zt[0]);
    p[14] = (pixel)F2(zt[3],zt[2],zt[1]);
    p[15] = (pixel)F2(zt[4],zt[3],zt[2]);
    p[16] = (pixel)F1(zl[1],zl[2]);
    p[17] = (pixel)F2(zl[0],zl[1],zl[2]);
    p[18] = (pixel)F1(zl[0],zl[1]);
    p[19] = (pixel)F2(zlt,zl[0],zl[1]);
    p[20] = (pixel)F1(zlt,zl[0]);
    p[21] = (pixel)F2(zl[0],zlt,zt[0]);
    p[22] = (pixel)F2(zt[1],zt[0],zlt);
    p[23] = (pixel)F2(zt[2],zt[1],zt[0]);
    p[24] = (pixel)F1(zl[2],zl[3]);
    p[25] = (pixel)F2(zl[1],zl[2],zl[3]);
    p[26] = (pixel)F1(zl[1],zl[2]);
    p[27] = (pixel)F2(zl[0],zl[1],zl[2]);
    p[28] = (pixel)F1(zl[0],zl[1]);
    p[29] = (pixel)F2(zlt,zl[0],zl[1]);
    p[30] = (pixel)F1(zlt,zl[0]);
    p[31] = (pixel)F2(zl[0],zlt,zt[0]);
    p[32] = (pixel)F1(zl[3],zl[4]);
    p[33] = (pixel)F2(zl[2],zl[3],zl[4]);
    p[34] = (pixel)F1(zl[2],zl[3]);
    p[35] = (pixel)F2(zl[1],zl[2],zl[3]);
    p[36] = (pixel)F1(zl[1],zl[2]);
    p[37] = (pixel)F2(zl[0],zl[1],zl[2]);
    p[38] = (pixel)F1(zl[0],zl[1]);
    p[39] = (pixel)F2(zlt,zl[0],zl[1]);
    p[40] = (pixel)F1(zl[4],zl[5]);
    p[41] = (pixel)F2(zl[3],zl[4],zl[5]);
    p[42] = (pixel)F1(zl[3],zl[4]);
    p[43] = (pixel)F2(zl[2],zl[3],zl[4]);
    p[44] = (pixel)F1(zl[2],zl[3]);
    p[45] = (pixel)F2(zl[1],zl[2],zl[3]);
    p[46] = (pixel)F1(zl[1],zl[2]);
    p[47] = (pixel)F2(zl[0],zl[1],zl[2]);
    p[48] = (pixel)F1(zl[5],zl[6]);
    p[49] = (pixel)F2(zl[4],zl[5],zl[6]);
    p[50] = (pixel)F1(zl[4],zl[5]);
    p[51] = (pixel)F2(zl[3],zl[4],zl[5]);
    p[52] = (pixel)F1(zl[3],zl[4]);
    p[53] = (pixel)F2(zl[2],zl[3],zl[4]);
    p[54] = (pixel)F1(zl[2],zl[3]);
    p[55] = (pixel)F2(zl[1],zl[2],zl[3]);
    p[56] = (pixel)F1(zl[6],zl[7]);
    p[57] = (pixel)F2(zl[5],zl[6],zl[7]);
    p[58] = (pixel)F1(zl[5],zl[6]);
    p[59] = (pixel)F2(zl[4],zl[5],zl[6]);
    p[60] = (pixel)F1(zl[4],zl[5]);
    p[61] = (pixel)F2(zl[3],zl[4],zl[5]);
    p[62] = (pixel)F1(zl[3],zl[4]);
    p[63] = (pixel)F2(zl[2],zl[3],zl[4]);
    break;
    case 7:                                     /* vertical-left */
    p[0] = (pixel)F1(zt[0],zt[1]);
    p[1] = (pixel)F1(zt[1],zt[2]);
    p[2] = (pixel)F1(zt[2],zt[3]);
    p[3] = (pixel)F1(zt[3],zt[4]);
    p[4] = (pixel)F1(zt[4],zt[5]);
    p[5] = (pixel)F1(zt[5],zt[6]);
    p[6] = (pixel)F1(zt[6],zt[7]);
    p[7] = (pixel)F1(zt[7],zt[8]);
    p[8] = (pixel)F2(zt[0],zt[1],zt[2]);
    p[9] = (pixel)F2(zt[1],zt[2],zt[3]);
    p[10] = (pixel)F2(zt[2],zt[3],zt[4]);
    p[11] = (pixel)F2(zt[3],zt[4],zt[5]);
    p[12] = (pixel)F2(zt[4],zt[5],zt[6]);
    p[13] = (pixel)F2(zt[5],zt[6],zt[7]);
    p[14] = (pixel)F2(zt[6],zt[7],zt[8]);
    p[15] = (pixel)F2(zt[7],zt[8],zt[9]);
    p[16] = (pixel)F1(zt[1],zt[2]);
    p[17] = (pixel)F1(zt[2],zt[3]);
    p[18] = (pixel)F1(zt[3],zt[4]);
    p[19] = (pixel)F1(zt[4],zt[5]);
    p[20] = (pixel)F1(zt[5],zt[6]);
    p[21] = (pixel)F1(zt[6],zt[7]);
    p[22] = (pixel)F1(zt[7],zt[8]);
    p[23] = (pixel)F1(zt[8],zt[9]);
    p[24] = (pixel)F2(zt[1],zt[2],zt[3]);
    p[25] = (pixel)F2(zt[2],zt[3],zt[4]);
    p[26] = (pixel)F2(zt[3],zt[4],zt[5]);
    p[27] = (pixel)F2(zt[4],zt[5],zt[6]);
    p[28] = (pixel)F2(zt[5],zt[6],zt[7]);
    p[29] = (pixel)F2(zt[6],zt[7],zt[8]);
    p[30] = (pixel)F2(zt[7],zt[8],zt[9]);
    p[31] = (pixel)F2(zt[8],zt[9],zt[10]);
    p[32] = (pixel)F1(zt[2],zt[3]);
    p[33] = (pixel)F1(zt[3],zt[4]);
    p[34] = (pixel)F1(zt[4],zt[5]);
    p[35] = (pixel)F1(zt[5],zt[6]);
    p[36] = (pixel)F1(zt[6],zt[7]);
    p[37] = (pixel)F1(zt[7],zt[8]);
    p[38] = (pixel)F1(zt[8],zt[9]);
    p[39] = (pixel)F1(zt[9],zt[10]);
    p[40] = (pixel)F2(zt[2],zt[3],zt[4]);
    p[41] = (pixel)F2(zt[3],zt[4],zt[5]);
    p[42] = (pixel)F2(zt[4],zt[5],zt[6]);
    p[43] = (pixel)F2(zt[5],zt[6],zt[7]);
    p[44] = (pixel)F2(zt[6],zt[7],zt[8]);
    p[45] = (pixel)F2(zt[7],zt[8],zt[9]);
    p[46] = (pixel)F2(zt[8],zt[9],zt[10]);
    p[47] = (pixel)F2(zt[9],zt[10],zt[11]);
    p[48] = (pixel)F1(zt[3],zt[4]);
    p[49] = (pixel)F1(zt[4],zt[5]);
    p[50] = (pixel)F1(zt[5],zt[6]);
    p[51] = (pixel)F1(zt[6],zt[7]);
    p[52] = (pixel)F1(zt[7],zt[8]);
    p[53] = (pixel)F1(zt[8],zt[9]);
    p[54] = (pixel)F1(zt[9],zt[10]);
    p[55] = (pixel)F1(zt[10],zt[11]);
    p[56] = (pixel)F2(zt[3],zt[4],zt[5]);
    p[57] = (pixel)F2(zt[4],zt[5],zt[6]);
    p[58] = (pixel)F2(zt[5],zt[6],zt[7]);
    p[59] = (pixel)F2(zt[6],zt[7],zt[8]);
    p[60] = (pixel)F2(zt[7],zt[8],zt[9]);
    p[61] = (pixel)F2(zt[8],zt[9],zt[10]);
    p[62] = (pixel)F2(zt[9],zt[10],zt[11]);
    p[63] = (pixel)F2(zt[10],zt[11],zt[12]);
    break;
    case 8:                                     /* horizontal-up */
    p[0] = (pixel)F1(zl[0],zl[1]);
    p[1] = (pixel)F2(zl[0],zl[1],zl[2]);
    p[2] = (pixel)F1(zl[1],zl[2]);
    p[3] = (pixel)F2(zl[1],zl[2],zl[3]);
    p[4] = (pixel)F1(zl[2],zl[3]);
    p[5] = (pixel)F2(zl[2],zl[3],zl[4]);
    p[6] = (pixel)F1(zl[3],zl[4]);
    p[7] = (pixel)F2(zl[3],zl[4],zl[5]);
    p[8] = (pixel)F1(zl[1],zl[2]);
    p[9] = (pixel)F2(zl[1],zl[2],zl[3]);
    p[10] = (pixel)F1(zl[2],zl[3]);
    p[11] = (pixel)F2(zl[2],zl[3],zl[4]);
    p[12] = (pixel)F1(zl[3],zl[4]);
    p[13] = (pixel)F2(zl[3],zl[4],zl[5]);
    p[14] = (pixel)F1(zl[4],zl[5]);
    p[15] = (pixel)F2(zl[4],zl[5],zl[6]);
    p[16] = (pixel)F1(zl[2],zl[3]);
    p[17] = (pixel)F2(zl[2],zl[3],zl[4]);
    p[18] = (pixel)F1(zl[3],zl[4]);
    p[19] = (pixel)F2(zl[3],zl[4],zl[5]);
    p[20] = (pixel)F1(zl[4],zl[5]);
    p[21] = (pixel)F2(zl[4],zl[5],zl[6]);
    p[22] = (pixel)F1(zl[5],zl[6]);
    p[23] = (pixel)F2(zl[5],zl[6],zl[7]);
    p[24] = (pixel)F1(zl[3],zl[4]);
    p[25] = (pixel)F2(zl[3],zl[4],zl[5]);
    p[26] = (pixel)F1(zl[4],zl[5]);
    p[27] = (pixel)F2(zl[4],zl[5],zl[6]);
    p[28] = (pixel)F1(zl[5],zl[6]);
    p[29] = (pixel)F2(zl[5],zl[6],zl[7]);
    p[30] = (pixel)F1(zl[6],zl[7]);
    p[31] = (pixel)F2(zl[6],zl[7],zl[7]);
    p[32] = (pixel)F1(zl[4],zl[5]);
    p[33] = (pixel)F2(zl[4],zl[5],zl[6]);
    p[34] = (pixel)F1(zl[5],zl[6]);
    p[35] = (pixel)F2(zl[5],zl[6],zl[7]);
    p[36] = (pixel)F1(zl[6],zl[7]);
    p[37] = (pixel)F2(zl[6],zl[7],zl[7]);
    p[38] = (pixel)zl[7];
    p[39] = (pixel)zl[7];
    p[40] = (pixel)F1(zl[5],zl[6]);
    p[41] = (pixel)F2(zl[5],zl[6],zl[7]);
    p[42] = (pixel)F1(zl[6],zl[7]);
    p[43] = (pixel)F2(zl[6],zl[7],zl[7]);
    p[44] = (pixel)zl[7];
    p[45] = (pixel)zl[7];
    p[46] = (pixel)zl[7];
    p[47] = (pixel)zl[7];
    p[48] = (pixel)F1(zl[6],zl[7]);
    p[49] = (pixel)F2(zl[6],zl[7],zl[7]);
    p[50] = (pixel)zl[7];
    p[51] = (pixel)zl[7];
    p[52] = (pixel)zl[7];
    p[53] = (pixel)zl[7];
    p[54] = (pixel)zl[7];
    p[55] = (pixel)zl[7];
    p[56] = (pixel)zl[7];
    p[57] = (pixel)zl[7];
    p[58] = (pixel)zl[7];
    p[59] = (pixel)zl[7];
    p[60] = (pixel)zl[7];
    p[61] = (pixel)zl[7];
    p[62] = (pixel)zl[7];
    p[63] = (pixel)zl[7];
    break;
    default:
        for (i = 0; i < 64; i++)
            p[i] = PIXEL_MID;
        break;
    }

    for (y = 0; y < 8; y++)
        for (x = 0; x < 8; x++)
            pred[y * stride + x] = p[y * 8 + x];
}
#undef F1
#undef F2
#undef I8_F1
#undef I8_F2

/* ------------------------------------------------------------------ */
/* Intra_4x4                                                           */
/* ------------------------------------------------------------------ */
/* `top` holds p[-1][-1] at index 0? No: top[0..7] are p[0][-1]..p[7][-1]
 * and top[-1] is p[-1][-1]; `left` holds p[-1][0..3]. */
void aeg_intra_predict_4x4(pixel *pred, int stride, const pixel *top,
                           const pixel *left, int topleft, int mode,
                           int top_avail, int left_avail, int topleft_avail)
{
    /* tt[0..1] mirror the top-left sample so that negative indices used by
     * the directional modes are well defined. */
    int tt[10], ll[6];
    int x, y;

    (void)topleft_avail;
    tt[0] = tt[1] = topleft;
    ll[0] = ll[1] = topleft;
    for (x = 0; x < 8; x++) tt[2 + x] = top[x];
    for (y = 0; y < 4; y++) ll[2 + y] = left[y];
#define T(i) tt[2 + (i)]
#define L(i) ll[2 + (i)]

    switch (mode) {
    case 0: /* vertical */
        for (y = 0; y < 4; y++)
            for (x = 0; x < 4; x++) pred[y * stride + x] = (pixel)T(x);
        break;
    case 1: /* horizontal */
        for (y = 0; y < 4; y++)
            for (x = 0; x < 4; x++) pred[y * stride + x] = (pixel)L(y);
        break;
    case 2: { /* DC */
        int sum = 0, dc;
        if (top_avail && left_avail) {
            for (x = 0; x < 4; x++) sum += top[x];
            for (y = 0; y < 4; y++) sum += left[y];
            dc = (sum + 4) >> 3;
        } else if (top_avail) {
            for (x = 0; x < 4; x++) sum += top[x];
            dc = (sum + 2) >> 2;
        } else if (left_avail) {
            for (y = 0; y < 4; y++) sum += left[y];
            dc = (sum + 2) >> 2;
        } else {
            dc = PIXEL_MID;
        }
        for (y = 0; y < 4; y++)
            for (x = 0; x < 4; x++) pred[y * stride + x] = (pixel)dc;
        break;
    }
    case 3: /* diagonal down-left */
        for (y = 0; y < 4; y++)
            for (x = 0; x < 4; x++) {
                int z = x + y, v;
                if (z == 6)
                    v = (T(6) + 3 * T(7) + 2) >> 2;
                else
                    v = (T(z) + 2 * T(z + 1) + T(z + 2) + 2) >> 2;
                pred[y * stride + x] = (pixel)v;
            }
        break;
    case 4: /* diagonal down-right: the spec indexes the top row two samples
             * to the left of the block, so for x > y the taps are
             * x-y-2, x-y-1 and x-y (8.3.3.4). */
        for (y = 0; y < 4; y++)
            for (x = 0; x < 4; x++) {
                int v;
                if (x > y)
                    v = (T(x - y - 2) + 2 * T(x - y - 1) + T(x - y) + 2) >> 2;
                else if (x == y)
                    v = (L(0) + 2 * topleft + T(0) + 2) >> 2;
                else if (x == y - 1)
                    v = (topleft + 2 * L(0) + L(1) + 2) >> 2;
                else if (x == y - 2)
                    v = (L(0) + 2 * L(1) + L(2) + 2) >> 2;
                else
                    v = (L(1) + 2 * L(2) + L(3) + 2) >> 2;
                pred[y * stride + x] = (pixel)v;
            }
        break;
    case 5: /* vertical right */
        for (y = 0; y < 4; y++)
            for (x = 0; x < 4; x++) {
                int z = 2 * x - y, v;
                if ((z & 1) == 0 && z >= 0) {
                    int i = z >> 1;
                    v = (T(i - 1) + T(i) + 1) >> 1;
                } else if (z > 0) {
                    int i = (z + 1) >> 1;
                    v = (T(i - 2) + 2 * T(i - 1) + T(i) + 2) >> 2;
                } else if (z == -1) {
                    v = (L(0) + 2 * topleft + T(0) + 2) >> 2;
                } else {
                    int i = y;
                    v = (L(i - 1) + 2 * L(i - 2) + L(i - 3) + 2) >> 2;
                }
                pred[y * stride + x] = (pixel)v;
            }
        break;
    case 6: /* horizontal down */
        for (y = 0; y < 4; y++)
            for (x = 0; x < 4; x++) {
                int z = 2 * y - x, v;
                if ((z & 1) == 0 && z >= 0) {
                    int i = z >> 1;
                    v = (L(i - 1) + L(i) + 1) >> 1;
                } else if (z > 0) {
                    int i = (z + 1) >> 1;
                    v = (L(i - 2) + 2 * L(i - 1) + L(i) + 2) >> 2;
                } else if (z == -1) {
                    v = (T(0) + 2 * topleft + L(0) + 2) >> 2;
                } else {
                    v = (T(x - 1) + 2 * T(x - 2) + T(x - 3) + 2) >> 2;
                }
                pred[y * stride + x] = (pixel)v;
            }
        break;
    case 7: /* vertical-left */
        for (y = 0; y < 4; y++)
            for (x = 0; x < 4; x++) {
                int i = x + (y >> 1), v;
                if ((y & 1) == 0)
                    v = (T(i) + T(i + 1) + 1) >> 1;
                else
                    v = (T(i) + 2 * T(i + 1) + T(i + 2) + 2) >> 2;
                pred[y * stride + x] = (pixel)v;
            }
        break;
    default: /* 8: horizontal-up */
        for (y = 0; y < 4; y++)
            for (x = 0; x < 4; x++) {
                int z = x + 2 * y, v;
                if (z <= 4 && (z & 1) == 0)
                    v = (L(z >> 1) + L((z >> 1) + 1) + 1) >> 1;
                else if (z < 5)
                    v = (L((z - 1) >> 1) + 2 * L(((z - 1) >> 1) + 1) +
                         L(((z - 1) >> 1) + 2) + 2) >> 2;
                else if (z == 5)
                    v = (L(2) + 3 * L(3) + 2) >> 2;
                else
                    v = L(3);
                pred[y * stride + x] = (pixel)v;
            }
        break;
    }
#undef T
#undef L
}

int aeg_intra4x4_pred_mode(const aeg_mb *mb_grid, int mb_stride, int mb_x,
                           int mb_y, int blk, int *available)
{
    static const int blk_x[16] = {0,1,0,1,2,3,2,3,0,1,0,1,2,3,2,3};
    static const int blk_y[16] = {0,0,1,1,0,0,1,1,2,2,3,3,2,2,3,3};
    int i;
    *available = 0;
    for (i = 0; i < 8; i++) {
        int nb = aeg_intra4x4_pred_modes[blk][i];
        int bx = blk_x[blk] + blk_x[nb];
        int by = blk_y[blk] + blk_y[nb];
        int mx = mb_x, my = mb_y;
        const aeg_mb *m;
        if (bx < 0) { mx--; bx += 4; }
        if (by < 0) { my--; by += 4; }
        if (mx < 0 || my < 0)
            continue;
        m = &mb_grid[my * mb_stride + mx];
        if (m->type != AEG_MB_I4x4)
            continue;
        *available = 1;
        return m->intra4[by * 4 + bx];
    }
    /* No usable neighbour: fall back to DC (mode 2) */
    return 2;
}
