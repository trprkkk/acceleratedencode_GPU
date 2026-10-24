/*
 * Intra prediction (H.264 8.3.3): Intra_16x16, Intra_4x4 and 8x8 chroma.
 *
 * Unavailable neighbour samples are substituted as specified in 8.3.3.1:
 * a whole missing side is replaced by the mid-level sample (128 for 8-bit)
 * while missing samples inside an available side are replicated from the
 * closest available sample.
 */
#include "h264.h"

static inline uint8_t clip255(int v)
{
    return (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
}

/* ------------------------------------------------------------------ */
/* Intra_16x16 (and 8x8 chroma when size == 8)                         */
/* ------------------------------------------------------------------ */
static void intra_16x16_dc(uint8_t *pred, int stride, const uint8_t *top,
                           const uint8_t *left, int t_av, int l_av)
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
        dc = 128;
    }
    for (y = 0; y < 16; y++)
        for (x = 0; x < 16; x++)
            pred[y * stride + x] = (uint8_t)dc;
}

static void intra_16x16_plane(uint8_t *pred, int stride, const uint8_t *top,
                              const uint8_t *left, int topleft)
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
            pred[y * stride + x] = clip255((a + b * (x - 7) + c * (y - 7) + 16) >> 5);
}

void aeg_intra_predict_16x16(uint8_t *pred, int stride, const uint8_t *top,
                             const uint8_t *left, int top_avail, int left_avail,
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
                          topleft_avail ? top[-1] : 128);
        break;
    }
}

/* ------------------------------------------------------------------ */
/* Chroma 8x8                                                          */
/* ------------------------------------------------------------------ */
static void intra_chroma_dc(uint8_t *pred, int stride, const uint8_t *top,
                            const uint8_t *left, int t_av, int l_av)
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
        dc[0] = dc[1] = dc[2] = dc[3] = 128;
    }
    for (y = 0; y < 8; y++)
        for (x = 0; x < 8; x++)
            pred[y * stride + x] = (uint8_t)dc[(y >> 2) * 2 + (x >> 2)];
}

static void intra_chroma_plane(uint8_t *pred, int stride, const uint8_t *top,
                               const uint8_t *left, int topleft)
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
            pred[y * stride + x] = clip255((b + x * H) >> 5);
    }
}

void aeg_intra_predict_chroma(uint8_t *pred, int stride, const uint8_t *top,
                              const uint8_t *left, int top_avail, int left_avail,
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
        intra_chroma_plane(pred, stride, top, left, topleft_avail ? top[-1] : 128);
        break;
    }
}

/* ------------------------------------------------------------------ */
/* Intra_4x4                                                           */
/* ------------------------------------------------------------------ */
/* `top` holds p[-1][-1] at index 0? No: top[0..7] are p[0][-1]..p[7][-1]
 * and top[-1] is p[-1][-1]; `left` holds p[-1][0..3]. */
void aeg_intra_predict_4x4(uint8_t *pred, int stride, const uint8_t *top,
                           const uint8_t *left, int topleft, int mode,
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
            for (x = 0; x < 4; x++) pred[y * stride + x] = (uint8_t)T(x);
        break;
    case 1: /* horizontal */
        for (y = 0; y < 4; y++)
            for (x = 0; x < 4; x++) pred[y * stride + x] = (uint8_t)L(y);
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
            dc = 128;
        }
        for (y = 0; y < 4; y++)
            for (x = 0; x < 4; x++) pred[y * stride + x] = (uint8_t)dc;
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
                pred[y * stride + x] = (uint8_t)v;
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
                pred[y * stride + x] = (uint8_t)v;
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
                pred[y * stride + x] = (uint8_t)v;
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
                pred[y * stride + x] = (uint8_t)v;
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
                pred[y * stride + x] = (uint8_t)v;
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
                pred[y * stride + x] = (uint8_t)v;
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
