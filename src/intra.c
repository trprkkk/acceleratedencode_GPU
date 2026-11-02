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

/* Every directional mode of the two block sizes predicts each sample from one
 * of three things: a reference sample, the 2-tap average of two neighbouring
 * ones (F1) or the 3-tap filter centred on one (F2), with the left column, the
 * corner and the top row read as one line.  aeg_intra*_edge() lays that line
 * out once -- E[] the left column bottom-up, the corner, the top row; G[k] =
 * F2 centred on E[k]; H[k] = F1(E[k], E[k+1]) -- and the predictor of a mode
 * is then n row copies (pred_dir; the two ends of G carry the one-sided
 * filter of the diagonal-down-left corner and of the horizontal-up limit).  A mode decision
 * builds the line once for all nine modes. */
#define IE_G 32
#define IE_H 64

/* The directional modes of an n x n block (n = 4 or 8) from its edge line
 * a[] (E at 0, the corner at E[n]; G and H at IE_G / IE_H): every row is a
 * run of n consecutive entries of one line -- G or H themselves, or for the
 * three modes that alternate between F1 and F2 samples a line interleaved
 * from both -- so each mode is n row copies.  The lines follow from the
 * equations of 8.3.1.2.x / 8.3.2.2.x with x the column and y the row. */
static inline __attribute__((always_inline))
void pred_dir(pixel *pred, int stride, const pixel *a, int n, int mode)
{
    const pixel *G = a + IE_G, *H = a + IE_H;
    pixel w0[32], w1[32];
    int y, i;

#define ROW(y_, src_) memcpy(pred + (y_) * stride, (src_), sizeof(pixel) * (size_t)n)
    switch (mode) {
    case 0:                                         /* vertical */
        for (y = 0; y < n; y++)
            ROW(y, a + n + 1);
        break;
    case 1:                                         /* horizontal */
        for (y = 0; y < n; y++)
            aeg_pixel_fill(pred + y * stride, a[n - 1 - y], (size_t)n);
        break;
    case 3:                                         /* diagonal down-left */
        for (y = 0; y < n; y++)
            ROW(y, G + n + 2 + y);
        break;
    case 4:                                         /* diagonal down-right */
        for (y = 0; y < n; y++)
            ROW(y, G + n - y);
        break;
    case 5:                                         /* vertical-right */
        /* row 2k: w0[i - k], row 2k+1: w1[i - k] (w0/w1 offset by n) */
        for (i = -n; i < n; i++) {
            w0[n + i] = i >= 0 ? H[n + i] : (2 * i + n + 1 >= 0 ? G[n + 1 + 2 * i] : 0);
            w1[n + i] = i >= 1 ? G[n + i] : (2 * i + n >= 0 ? G[n + 2 * i] : 0);
        }
        for (y = 0; y < n; y++)
            ROW(y, ((y & 1) ? w1 : w0) + n - (y >> 1));
        break;
    case 6:                                         /* horizontal-down */
        /* row y: d[j] for j = x - 2y (d offset by 2n) */
        for (i = -2 * (n - 1); i < n; i++) {
            pixel v;
            if (i >= 1)
                v = G[n - 1 + i];
            else if ((i & 1) == 0)
                v = H[n - 1 + i / 2];
            else
                v = G[n + (i - 1) / 2];
            w0[2 * n + i] = v;
        }
        for (y = 0; y < n; y++)
            ROW(y, w0 + 2 * n - 2 * y);
        break;
    case 7:                                         /* vertical-left */
        for (y = 0; y < n; y++)
            ROW(y, (y & 1) ? G + n + 2 + (y >> 1) : H + n + 1 + (y >> 1));
        break;
    default:                                        /* horizontal-up */
        /* row y: u[j] for j = x + 2y */
        for (i = 0; i < 3 * n - 2; i++) {
            pixel v;
            if (i > 2 * n - 3)
                v = a[0];
            else if (i == 2 * n - 3)
                v = G[0];
            else if ((i & 1) == 0)
                v = H[n - 2 - i / 2];
            else
                v = G[n - 2 - (i - 1) / 2];
            w0[i] = v;
        }
        for (y = 0; y < n; y++)
            ROW(y, w0 + 2 * y);
        break;
    }
#undef ROW
}

static void edge_filters(aeg_iedge *ed, int n)
{
    pixel *a = ed->a;
    int k;
    for (k = 1; k < n - 1; k++)
        a[IE_G + k] = (pixel)F2(a[k - 1], a[k], a[k + 1]);
    a[IE_G] = (pixel)F2(a[1], a[0], a[0]);
    a[IE_G + n - 1] = (pixel)F2(a[n - 2], a[n - 1], a[n - 1]);
    for (k = 0; k < n - 1; k++)
        a[IE_H + k] = (pixel)F1(a[k], a[k + 1]);
}

void aeg_intra8x8_edge(aeg_iedge *ed, const pixel *top, const pixel *left,
                       int topleft, int top_avail, int left_avail,
                       int topleft_avail)
{
    int rt[16], rl[8], rlt, x, y, dc;
    pixel *z = ed->a;

    for (x = 0; x < 16; x++)
        rt[x] = top_avail ? top[x] : (left_avail ? left[0] : PIXEL_MID);
    for (y = 0; y < 8; y++)
        rl[y] = left_avail ? left[y] : (top_avail ? top[0] : PIXEL_MID);
    rlt = topleft_avail ? topleft
                        : (top_avail ? top[0] : (left_avail ? left[0] : PIXEL_MID));

    /* E: zl[7..0], z'[-1,-1], zt[0..15] */
    z[8] = (pixel)I8_F2(rt[0], rlt, rl[0]);
    z[9] = (pixel)I8_F2(rlt, rt[0], rt[1]);
    for (x = 1; x < 15; x++)
        z[9 + x] = (pixel)I8_F2(rt[x - 1], rt[x], rt[x + 1]);
    z[24] = (pixel)((rt[14] + 3 * rt[15] + 2) >> 2);
    z[7] = (pixel)I8_F2(rlt, rl[0], rl[1]);
    for (y = 1; y < 7; y++)
        z[7 - y] = (pixel)I8_F2(rl[y - 1], rl[y], rl[y + 1]);
    z[0] = (pixel)((rl[6] + 3 * rl[7] + 2) >> 2);
    edge_filters(ed, 25);

    if (top_avail && left_avail) {
        dc = 8;
        for (x = 0; x < 8; x++)
            dc += z[9 + x] + z[7 - x];
        dc >>= 4;
    } else if (top_avail) {
        dc = 4;
        for (x = 0; x < 8; x++)
            dc += z[9 + x];
        dc >>= 3;
    } else if (left_avail) {
        dc = 4;
        for (x = 0; x < 8; x++)
            dc += z[7 - x];
        dc >>= 3;
    } else {
        dc = PIXEL_MID;
    }
    ed->dc = dc;
}

void aeg_intra8x8_pred(pixel *pred, int stride, const aeg_iedge *ed, int mode)
{
    int y;
    if (mode == 2 || mode < 0 || mode > 8) {
        const int v = mode == 2 ? ed->dc : PIXEL_MID;
        for (y = 0; y < 8; y++)
            aeg_pixel_fill(pred + y * stride, v, 8);
        return;
    }
    pred_dir(pred, stride, ed->a, 8, mode);
}

/* The eight-by-eight prediction modes are the same nine directions as the
 * four-by-four ones, but they read *filtered* reference samples z[]: a low-pass
 * along the neighbouring row and column (which is what stops the 1/4-sample
 * diagonal interpolation of the larger block from ringing) and, at the two
 * ends, a one-sided filter (8.3.2.2.1).
 *
 * Availability: the caller substitutes an unavailable side with the first
 * sample of the other one (8.3.3.1) -- so an unavailable top-right corner,
 * which this signature has no flag for, is simply the block's last top sample
 * repeated, which is exactly what the filtering then produces as well. */
void aeg_intra_predict_8x8(pixel *pred, int stride, const pixel *top,
                           const pixel *left, int topleft, int mode,
                           int top_avail, int left_avail, int topleft_avail)
{
    aeg_iedge ed;
    aeg_intra8x8_edge(&ed, top, left, topleft, top_avail, left_avail,
                      topleft_avail);
    aeg_intra8x8_pred(pred, stride, &ed, mode);
}

/* ------------------------------------------------------------------ */
/* Intra_4x4                                                           */
/* ------------------------------------------------------------------ */
/* `top` holds p[0..7][-1], `left` p[-1][0..3] and `topleft` p[-1][-1]; the
 * caller has already substituted what is unavailable (only DC looks at the
 * flags).  E: left[3..0], the corner, top[0..7]. */
void aeg_intra4x4_edge(aeg_iedge *ed, const pixel *top, const pixel *left,
                       int topleft, int top_avail, int left_avail)
{
    pixel *a = ed->a;
    int i, sum = 0;
    for (i = 0; i < 4; i++)
        a[3 - i] = left[i];
    a[4] = (pixel)topleft;
    for (i = 0; i < 8; i++)
        a[5 + i] = top[i];
    edge_filters(ed, 13);
    if (top_avail && left_avail) {
        for (i = 0; i < 4; i++) sum += top[i] + left[i];
        ed->dc = (sum + 4) >> 3;
    } else if (top_avail) {
        for (i = 0; i < 4; i++) sum += top[i];
        ed->dc = (sum + 2) >> 2;
    } else if (left_avail) {
        for (i = 0; i < 4; i++) sum += left[i];
        ed->dc = (sum + 2) >> 2;
    } else {
        ed->dc = PIXEL_MID;
    }
}

void aeg_intra4x4_pred(pixel *pred, int stride, const aeg_iedge *ed, int mode)
{
    int y;
    if (mode == 2) {
        for (y = 0; y < 4; y++)
            aeg_pixel_fill(pred + y * stride, ed->dc, 4);
        return;
    }
    if (mode < 0 || mode > 8)
        mode = 8;               /* the old switch's default */
    pred_dir(pred, stride, ed->a, 4, mode);
}

void aeg_intra_predict_4x4(pixel *pred, int stride, const pixel *top,
                           const pixel *left, int topleft, int mode,
                           int top_avail, int left_avail, int topleft_avail)
{
    aeg_iedge ed;
    (void)topleft_avail;
    aeg_intra4x4_edge(&ed, top, left, topleft, top_avail, left_avail);
    aeg_intra4x4_pred(pred, stride, &ed, mode);
}

#undef F1
#undef F2
#undef I8_F1
#undef I8_F2

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
