/*
 * Lookahead cost estimation for the adaptive B-picture decision.
 *
 * How many B pictures a group should have is a property of the content: on a
 * slow pan or a static shot every B picture is nearly free, on a clip with
 * fast, irregular motion a B picture three frames from either anchor predicts
 * badly and the anchor behind it, four frames from its own reference, predicts
 * worse still.  A fixed group length is right for one of the two and wrong for
 * the other (a 3-B structure costs one of the test clips +20% against P only),
 * so the scheduler asks, group by group, what each structure would cost.
 *
 * The question is answered the way x264's lookahead answers it: on half
 * resolution copies of the *source* pictures (so the answer does not depend on
 * the encode and two passes see the same structure), one 8x8 block per
 * macroblock, each block costed as min(intra, inter) with SATD:
 *
 *   intra  the best of DC / horizontal / vertical / plane from the block's
 *          neighbouring source samples, plus a fixed penalty;
 *   P      an integer-pel hexagon search around the zero vector and the
 *          vectors of the already searched neighbours, plus a vector cost;
 *   B      the best of the list 0 search, the list 1 search, and the average
 *          of the two best predictions (and of the two zero-vector ones).
 *
 * The picture cost is the sum over the macroblocks that are not on the
 * picture's border (their vectors are the least reliable), which is what the
 * decision rule in encoder.c (b_adapt_decide()) and its thresholds expect.
 */
#include "h264.h"
#if defined(__SSE2__)
#include <emmintrin.h>
#endif

#include <stdlib.h>
#include <string.h>

#define LA_PAD AEG_LA_PAD
#define LA_RANGE 16          /* search range, in lowres samples          */
#define LA_LAMBDA 1          /* x264's lambda at its lookahead QP (12)   */
#define LA_INTRA_PENALTY (5 * LA_LAMBDA + 4)

int aeg_lowres_alloc(aeg_lowres *l, int mb_w, int mb_h)
{
    size_t sz;

    l->mb_w = mb_w;
    l->mb_h = mb_h;
    l->w = mb_w * 8;
    l->h = mb_h * 8;
    l->stride = l->w + 2 * LA_PAD;
    sz = (size_t)l->stride * (size_t)(l->h + 2 * LA_PAD);
    l->mem = (uint8_t *)malloc(sz);
    if (!l->mem)
        return -1;
    l->plane = l->mem + (size_t)LA_PAD * l->stride + LA_PAD;
    return 0;
}

void aeg_lowres_free(aeg_lowres *l)
{
    free(l->mem);
    l->mem = l->plane = NULL;
}

/* aeg_lowres_build() and _rows() are in lowres.c: they read the source
 * picture, whose sample type depends on the bit depth. */

static int se_bits(int v)
{
    /* codeNum + 1 has n significant bits; the code is 2n - 1 long */
    unsigned u = (v > 0 ? 2u * (unsigned)v - 1u : 2u * (unsigned)(-v)) + 1u;

    return 2 * (32 - __builtin_clz(u)) - 1;
}

static int mv_cost(int mx, int my, int px, int py)
{
    /* lowres half-sample vectors in the quarter-sample units the coder sends */
    return LA_LAMBDA * (se_bits(2 * (mx - px)) + se_bits(2 * (my - py)));
}

static int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : v > hi ? hi : v;
}

static int median3(int a, int b, int c)
{
    int mx = a > b ? a : b, mn = a < b ? a : b;

    return c > mx ? mx : c < mn ? mn : c;
}

/* The 8x8 prediction at half-sample vector (hx, hy) (bilinear between the
 * integer samples, which is all a cost estimate needs). */
static void get_pred(const aeg_lowres *ref, int bx, int by, int hx, int hy,
                     uint8_t *out)
{
    int ix = hx >> 1, iy = hy >> 1, fx = hx & 1, fy = hy & 1, x, y;
    const uint8_t *p = ref->plane + (ptrdiff_t)(by + iy) * ref->stride + bx + ix;
    int s = ref->stride;

#if defined(__SSE2__)
    {
        const __m128i z = _mm_setzero_si128(), two = _mm_set1_epi16(2);
        for (y = 0; y < 8; y++, p += s) {
            __m128i a = _mm_loadl_epi64((const __m128i *)p), v;
            if (!fx && !fy) {
                v = a;
            } else if (!fy) {
                v = _mm_avg_epu8(a, _mm_loadl_epi64((const __m128i *)(p + 1)));
            } else if (!fx) {
                v = _mm_avg_epu8(a, _mm_loadl_epi64((const __m128i *)(p + s)));
            } else {
                __m128i t = _mm_add_epi16(
                    _mm_add_epi16(_mm_unpacklo_epi8(a, z),
                                  _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(p + 1)), z)),
                    _mm_add_epi16(_mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(p + s)), z),
                                  _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i *)(p + s + 1)), z)));
                t = _mm_srli_epi16(_mm_add_epi16(t, two), 2);
                v = _mm_packus_epi16(t, t);
            }
            _mm_storel_epi64((__m128i *)(out + 8 * y), v);
        }
        (void)x;
        return;
    }
#endif
    for (y = 0; y < 8; y++, p += s)
        for (x = 0; x < 8; x++) {
            int v;
            if (!fx && !fy)
                v = p[x];
            else if (!fy)
                v = (p[x] + p[x + 1] + 1) >> 1;
            else if (!fx)
                v = (p[x] + p[x + s] + 1) >> 1;
            else
                v = (p[x] + p[x + 1] + p[x + s] + p[x + s + 1] + 2) >> 2;
            out[8 * y + x] = (uint8_t)v;
        }
}

/* One block's integer-pel search against `ref`, refined to half samples;
 * returns the cost (SATD plus vector cost), the vector (half-sample units)
 * through mv[].  Candidates and predictor are in half-sample units too. */
static int block_search(const aeg_lowres *cur, const aeg_lowres *ref,
                        int bx, int by, const int (*cand)[2], int ncand,
                        int px, int py, int *mv, int range)
{
    static const int hex[6][2] = {
        {-2, 0}, {-1, 2}, {1, 2}, {2, 0}, {1, -2}, {-1, -2}
    };
    static const int sq[8][2] = {
        {-1, -1}, {0, -1}, {1, -1}, {-1, 0}, {1, 0}, {-1, 1}, {0, 1}, {1, 1}
    };
    const uint8_t *c = cur->plane + (size_t)by * cur->stride + bx;
    int lo_x = -bx - LA_PAD + 1, hi_x = ref->w - bx + LA_PAD - 9;
    int lo_y = -by - LA_PAD + 1, hi_y = ref->h - by + LA_PAD - 9;
    int bmx = 0, bmy = 0, best = 0x7fffffff, i, it;

    if (lo_x < -range) lo_x = -range;
    if (hi_x > range) hi_x = range;
    if (lo_y < -range) lo_y = -range;
    if (hi_y > range) hi_y = range;
#define LA_TRY(X, Y) do {                                                    \
        int tx_ = (X), ty_ = (Y), cst_;                                      \
        if (tx_ < lo_x || tx_ > hi_x || ty_ < lo_y || ty_ > hi_y) break;     \
        cst_ = aeg_satd(8, 8, c, cur->stride,                                \
                        ref->plane + (ptrdiff_t)(by + ty_) * ref->stride +   \
                        bx + tx_, ref->stride) +                             \
               mv_cost(2 * tx_, 2 * ty_, px, py);                            \
        if (cst_ < best) { best = cst_; bmx = tx_; bmy = ty_; }              \
    } while (0)

    LA_TRY(0, 0);
    for (i = 0; i < ncand; i++) {
        int cx = cand[i][0] >> 1, cy = cand[i][1] >> 1;
        if (cx != bmx || cy != bmy)
            LA_TRY(cx, cy);
    }
    /* After a move along hex[d], three of the new hexagon's points (hex[d+2]
     * .. hex[d+4] around the new centre: two points of the old hexagon and the
     * old centre) were measured already, and a point measured before cannot
     * beat the best it has already lost to: only hex[d-1..d+1] are new. */
    {
        int d = -1;
        for (it = 0; it < LA_RANGE; it++) {
            int cx = bmx, cy = bmy, nd = -1;

            for (i = 0; i < 6; i++) {
                int ox = bmx, oy = bmy, r = (i - d + 6) % 6;
                /* in index order, as before, so that ties go the same way */
                if (d >= 0 && r > 1 && r < 5)
                    continue;
                LA_TRY(cx + hex[i][0], cy + hex[i][1]);
                if (bmx != ox || bmy != oy)
                    nd = i;
            }
            if (bmx == cx && bmy == cy)
                break;
            d = nd;
        }
    }
    {
        int cx = bmx, cy = bmy;

        for (i = 0; i < 8; i++)
            LA_TRY(cx + sq[i][0], cy + sq[i][1]);
    }
#undef LA_TRY
    {
        int hx = 2 * bmx, hy = 2 * bmy;
        uint8_t pred[64];

        bmx = hx;
        bmy = hy;
        for (i = 0; i < 8; i++) {
            int tx = hx + sq[i][0], ty = hy + sq[i][1], cst;

            if (tx < 2 * lo_x || tx > 2 * hi_x || ty < 2 * lo_y || ty > 2 * hi_y)
                continue;
            get_pred(ref, bx, by, tx, ty, pred);
            cst = aeg_satd(8, 8, c, cur->stride, pred, 8) +
                  mv_cost(tx, ty, px, py);
            if (cst < best) { best = cst; bmx = tx; bmy = ty; }
        }
    }
    mv[0] = bmx;
    mv[1] = bmy;
    return best;
}

static int la_i8(void)
{
    static int v = -1;
    if (v < 0) {
        const char *ev = getenv("AEG_LRI8");
        v = ev ? atoi(ev) != 0 : 1;
    }
    return v;
}

static int intra_cost(const aeg_lowres *cur, int bx, int by)
{
    const uint8_t *c = cur->plane + (size_t)by * cur->stride + bx;
    const uint8_t *top = c - cur->stride;
    uint8_t pred[64];
    int left[8], i, x, y, best, s;
    int has_t = by > 0, has_l = bx > 0;
    int sum = 0, n = 0, dc;
    uint8_t left8[8];

    for (i = 0; i < 8; i++)
        left8[i] = (uint8_t)(left[i] = c[(ptrdiff_t)i * cur->stride - 1]);
    if (has_t) for (i = 0; i < 8; i++) sum += top[i], n++;
    if (has_l) for (i = 0; i < 8; i++) sum += left[i], n++;
    dc = n ? (sum + n / 2) / n : 128;
    memset(pred, dc, sizeof(pred));
    best = aeg_satd(8, 8, c, cur->stride, pred, 8);
    if (has_t) {
        for (y = 0; y < 8; y++)
            memcpy(pred + 8 * y, top, 8);
        s = aeg_satd(8, 8, c, cur->stride, pred, 8);
        if (s < best) best = s;
    }
    if (has_l) {
        for (y = 0; y < 8; y++)
            memset(pred + 8 * y, left[y], 8);
        s = aeg_satd(8, 8, c, cur->stride, pred, 8);
        if (s < best) best = s;
    }
    if (has_t && has_l) {
        /* the chroma plane predictor, which is the 8x8 one */
        int tl = top[-1], H = 0, V = 0, a, b, cc;

        for (i = 0; i < 4; i++) {
            H += (i + 1) * (top[4 + i] - (i == 3 ? tl : top[2 - i]));
            V += (i + 1) * (left[4 + i] - (i == 3 ? tl : left[2 - i]));
        }
        a = 16 * (left[7] + top[7]);
        b = (34 * H + 32) >> 6;
        cc = (34 * V + 32) >> 6;
        for (y = 0; y < 8; y++)
            for (x = 0; x < 8; x++)
                pred[8 * y + x] = (uint8_t)clampi(
                    (a + b * (x - 3) + cc * (y - 3) + 16) >> 5, 0, 255);
        s = aeg_satd(8, 8, c, cur->stride, pred, 8);
        if (s < best) best = s;
        if (la_i8()) {
            /* x264 (subme > 1): the six directional Intra_8x8 modes on
             * filtered edges as well.  top[8..15] is the top-right, which
             * at the right edge reads the padding.  Without them the
             * estimate of textured blocks with a direction (feathers,
             * branches) is too high, so the macroblock tree propagated too
             * much through them: -0.15% at constant QP, -0.5% at CRF on
             * the long clips, at no measurable cost (AEG_LRI8=0 restores
             * the four plane modes; docs/STATUS.md). */
            int m;
            aeg_iedge ed;
            aeg_intra8x8_edge(&ed, top, left8, tl, 1, 1, 1);
            for (m = 3; m < 9; m++) {
                aeg_intra8x8_pred(pred, 8, &ed, m);
                s = aeg_satd(8, 8, c, cur->stride, pred, 8);
                if (s < best) best = s;
            }
        }
    }
    return best + LA_INTRA_PENALTY;
}

static int bidir_cost(const aeg_lowres *cur, const aeg_lowres *r0,
                      const aeg_lowres *r1, int bx, int by,
                      const int *m0, const int *m1)
{
    const uint8_t *c = cur->plane + (size_t)by * cur->stride + bx;
    uint8_t p0[64], p1[64];
    int i;

    get_pred(r0, bx, by, m0[0], m0[1], p0);
    get_pred(r1, bx, by, m1[0], m1[1], p1);
    for (i = 0; i < 64; i++)
        p0[i] = (uint8_t)((p0[i] + p1[i] + 1) >> 1);
    return aeg_satd(8, 8, c, cur->stride, p0, 8);
}

/* The search of every block of `cur` against one reference, in raster order:
 * its vector (2 ints per macroblock, half-sample units) and cost (SATD plus
 * vector cost).  A block's candidates and predictor are its left, top and
 * top-right neighbours' vectors in the same field, so the field depends on
 * (cur, ref) only -- not on the other list of a B estimate, nor on whether
 * the macroblock tree or the group decision asked -- and the encoder keeps
 * it for every pair it has searched. */
static void aeg_lowres_search_ph(const aeg_lowres *cur, const aeg_lowres *ref, int *mv,
                       int *cost);

/* (timed as a phase, see aeg_ph_switch) */
void aeg_lowres_search(const aeg_lowres *cur, const aeg_lowres *ref, int *mv,
                       int *cost)
{
    const int prev = aeg_ph_push(AEG_PH_LOWRES);

    aeg_lowres_search_ph(cur, ref, mv, cost);
    aeg_ph_pop(prev);
}

static void aeg_lowres_search_ph(const aeg_lowres *cur, const aeg_lowres *ref, int *mv,
                       int *cost)
{
    aeg_lowres_search_seeded(cur, ref, mv, cost, NULL, 1, 1, LA_RANGE);
}

/* The same with an extra candidate per block, seed (2 ints per macroblock,
 * may be NULL) scaled by num / den, and the vectors bounded by +-range. */
static void aeg_lowres_search_seeded_ph(const aeg_lowres *cur, const aeg_lowres *ref,
                              int *mv, int *cost, const int *seed, int num,
                              int den, int range);

/* (timed as a phase, see aeg_ph_switch) */
void aeg_lowres_search_seeded(const aeg_lowres *cur, const aeg_lowres *ref,
                              int *mv, int *cost, const int *seed, int num,
                              int den, int range)
{
    const int prev = aeg_ph_push(AEG_PH_LOWRES);

    aeg_lowres_search_seeded_ph(cur, ref, mv, cost, seed, num, den, range);
    aeg_ph_pop(prev);
}

static void aeg_lowres_search_seeded_ph(const aeg_lowres *cur, const aeg_lowres *ref,
                              int *mv, int *cost, const int *seed, int num,
                              int den, int range)
{
    int mb_w = cur->mb_w, mb_h = cur->mb_h, x, y;

    for (y = 0; y < mb_h; y++)
        for (x = 0; x < mb_w; x++) {
            int i = y * mb_w + x;
            int cand[4][2], nc = 0, px, py;
            int a[2] = {0, 0}, b[2] = {0, 0}, cc[2] = {0, 0};

            if (x > 0) {
                a[0] = mv[2 * (i - 1)]; a[1] = mv[2 * (i - 1) + 1];
                cand[nc][0] = a[0]; cand[nc][1] = a[1]; nc++;
            }
            if (y > 0) {
                b[0] = mv[2 * (i - mb_w)]; b[1] = mv[2 * (i - mb_w) + 1];
                cand[nc][0] = b[0]; cand[nc][1] = b[1]; nc++;
                if (x < mb_w - 1) {
                    cc[0] = mv[2 * (i - mb_w + 1)];
                    cc[1] = mv[2 * (i - mb_w + 1) + 1];
                    cand[nc][0] = cc[0]; cand[nc][1] = cc[1]; nc++;
                }
            }
            if (seed) {
                int sx = seed[2 * i] * num, sy = seed[2 * i + 1] * num;
                cand[nc][0] = (sx + (sx < 0 ? -den / 2 : den / 2)) / den;
                cand[nc][1] = (sy + (sy < 0 ? -den / 2 : den / 2)) / den;
                nc++;
            }
            px = median3(a[0], b[0], cc[0]);
            py = median3(a[1], b[1], cc[1]);
            cost[i] = block_search(cur, ref, 8 * x, 8 * y,
                                   (const int (*)[2])cand, nc, px, py,
                                   mv + 2 * i, range);
        }
}

/* The intra estimate of every block of `cur` (1 int per macroblock). */
static void aeg_lowres_intra_ph(const aeg_lowres *cur, int *icost);

/* (timed as a phase, see aeg_ph_switch) */
void aeg_lowres_intra(const aeg_lowres *cur, int *icost)
{
    const int prev = aeg_ph_push(AEG_PH_LOWRES);

    aeg_lowres_intra_ph(cur, icost);
    aeg_ph_pop(prev);
}

static void aeg_lowres_intra_ph(const aeg_lowres *cur, int *icost)
{
    int x, y;

    for (y = 0; y < cur->mb_h; y++)
        for (x = 0; x < cur->mb_w; x++)
            icost[y * cur->mb_w + x] = intra_cost(cur, 8 * x, 8 * y);
}

/* aeg_lowres_cost() from the fields above (mv1/c1 unused without r1). */
static int64_t aeg_lowres_cost_fields_ph(const aeg_lowres *cur, const aeg_lowres *r0,
                               const aeg_lowres *r1, const int *icost,
                               const int *mv0, const int *c0,
                               const int *mv1, const int *c1, int *intra_mbs);

/* (timed as a phase, see aeg_ph_switch) */
int64_t aeg_lowres_cost_fields(const aeg_lowres *cur, const aeg_lowres *r0,
                               const aeg_lowres *r1, const int *icost,
                               const int *mv0, const int *c0,
                               const int *mv1, const int *c1, int *intra_mbs)
{
    const int prev = aeg_ph_push(AEG_PH_LOWRES);

    int64_t r = aeg_lowres_cost_fields_ph(cur, r0, r1, icost, mv0, c0, mv1, c1, intra_mbs);

    aeg_ph_pop(prev);
    return r;
}

static int64_t aeg_lowres_cost_fields_ph(const aeg_lowres *cur, const aeg_lowres *r0,
                               const aeg_lowres *r1, const int *icost,
                               const int *mv0, const int *c0,
                               const int *mv1, const int *c1, int *intra_mbs)
{
    int mb_w = cur->mb_w, mb_h = cur->mb_h;
    int border = mb_w > 2 && mb_h > 2;
    int64_t total = 0;
    int intra = 0, x, y;

    for (y = 0; y < mb_h; y++)
        for (x = 0; x < mb_w; x++) {
            int i = y * mb_w + x, bx = 8 * x, by = 8 * y;
            int counted = !border ||
                          (x > 0 && x < mb_w - 1 && y > 0 && y < mb_h - 1);
            int cost = c0[i];

            if (r1) {
                static const int z[2] = {0, 0};
                int c;

                if (c1[i] < cost)
                    cost = c1[i];
                c = bidir_cost(cur, r0, r1, bx, by, mv0 + 2 * i, mv1 + 2 * i) +
                    mv_cost(mv0[2 * i], mv0[2 * i + 1], 0, 0) +
                    mv_cost(mv1[2 * i], mv1[2 * i + 1], 0, 0);
                if (c < cost)
                    cost = c;
                if (mv0[2 * i] || mv0[2 * i + 1] || mv1[2 * i] ||
                    mv1[2 * i + 1]) {
                    c = bidir_cost(cur, r0, r1, bx, by, z, z);
                    if (c < cost)
                        cost = c;
                }
            }
            if (icost[i] < cost) {
                cost = icost[i];
                if (counted)
                    intra++;
            }
            if (counted)
                total += cost;
        }
    if (intra_mbs)
        *intra_mbs = intra;
    return total;
}

static int64_t aeg_lowres_cost_ph(const aeg_lowres *cur, const aeg_lowres *r0,
                        const aeg_lowres *r1, int *intra_mbs, int *scratch);

/* (timed as a phase, see aeg_ph_switch) */
int64_t aeg_lowres_cost(const aeg_lowres *cur, const aeg_lowres *r0,
                        const aeg_lowres *r1, int *intra_mbs, int *scratch)
{
    const int prev = aeg_ph_push(AEG_PH_LOWRES);

    int64_t r = aeg_lowres_cost_ph(cur, r0, r1, intra_mbs, scratch);

    aeg_ph_pop(prev);
    return r;
}

static int64_t aeg_lowres_cost_ph(const aeg_lowres *cur, const aeg_lowres *r0,
                        const aeg_lowres *r1, int *intra_mbs, int *scratch)
{
    int n = cur->mb_w * cur->mb_h;
    int *mv0 = scratch, *mv1 = scratch + 2 * n;
    int *f = (int *)malloc(sizeof(int) * 3 * (size_t)n);
    int64_t c;

    if (!f)
        return 0;
    aeg_lowres_intra(cur, f);
    aeg_lowres_search(cur, r0, mv0, f + n);
    if (r1)
        aeg_lowres_search(cur, r1, mv1, f + 2 * n);
    c = aeg_lowres_cost_fields(cur, r0, r1, f, mv0, f + n, mv1, f + 2 * n,
                               intra_mbs);
    free(f);
    return c;
}

/* The list-0 vectors aeg_lowres_cost(cur, r0, NULL, ...) finds, for the
 * block rows y0..y1 only (rows before y0 must already be in `mv`, 2 ints per
 * macroblock): the same search in the same order, without the intra
 * estimate the vectors do not depend on.  Block row y reads `r0` down to
 * lowres row 8 * y + 23 (LA_RANGE below the block, plus the half-sample
 * tap). */
static void aeg_lowres_mv_rows_ph(const aeg_lowres *cur, const aeg_lowres *r0, int *mv,
                        int y0, int y1);

/* (timed as a phase, see aeg_ph_switch) */
void aeg_lowres_mv_rows(const aeg_lowres *cur, const aeg_lowres *r0, int *mv,
                        int y0, int y1)
{
    const int prev = aeg_ph_push(AEG_PH_LOWRES);

    aeg_lowres_mv_rows_ph(cur, r0, mv, y0, y1);
    aeg_ph_pop(prev);
}

static void aeg_lowres_mv_rows_ph(const aeg_lowres *cur, const aeg_lowres *r0, int *mv,
                        int y0, int y1)
{
    int mb_w = cur->mb_w, x, y;

    if (y1 > cur->mb_h - 1)
        y1 = cur->mb_h - 1;
    for (y = y0; y <= y1; y++)
        for (x = 0; x < mb_w; x++) {
            int i = y * mb_w + x, bx = 8 * x, by = 8 * y;
            int cand[3][2], nc = 0, px, py;
            int a[2] = {0, 0}, b[2] = {0, 0}, cc[2] = {0, 0};

            if (x > 0) {
                a[0] = mv[2 * (i - 1)]; a[1] = mv[2 * (i - 1) + 1];
                cand[nc][0] = a[0]; cand[nc][1] = a[1]; nc++;
            }
            if (y > 0) {
                b[0] = mv[2 * (i - mb_w)]; b[1] = mv[2 * (i - mb_w) + 1];
                cand[nc][0] = b[0]; cand[nc][1] = b[1]; nc++;
                if (x < mb_w - 1) {
                    cc[0] = mv[2 * (i - mb_w + 1)];
                    cc[1] = mv[2 * (i - mb_w + 1) + 1];
                    cand[nc][0] = cc[0]; cand[nc][1] = cc[1]; nc++;
                }
            }
            px = median3(a[0], b[0], cc[0]);
            py = median3(a[1], b[1], cc[1]);
            block_search(cur, r0, bx, by, (const int (*)[2])cand, nc, px, py,
                         mv + 2 * i, LA_RANGE);
        }
}

/* Per-macroblock version of aeg_lowres_cost() for the macroblock tree: the
 * intra cost, the best inter cost, which prediction won (0 list 0, 1 list 1,
 * 2 both) and the two vectors of every block, border blocks included.  `mv`
 * holds 4 ints per macroblock (list 0 x/y, list 1 x/y, lowres half-sample
 * units). */
static void aeg_lowres_mbcosts_fields_ph(const aeg_lowres *cur, const aeg_lowres *r0,
                               const aeg_lowres *r1, const int *icost,
                               const int *mv0, const int *c0,
                               const int *mv1, const int *c1,
                               int *intra, int *inter, uint8_t *kind, int *mv);

/* (timed as a phase, see aeg_ph_switch) */
void aeg_lowres_mbcosts_fields(const aeg_lowres *cur, const aeg_lowres *r0,
                               const aeg_lowres *r1, const int *icost,
                               const int *mv0, const int *c0,
                               const int *mv1, const int *c1,
                               int *intra, int *inter, uint8_t *kind, int *mv)
{
    const int prev = aeg_ph_push(AEG_PH_LOWRES);

    aeg_lowres_mbcosts_fields_ph(cur, r0, r1, icost, mv0, c0, mv1, c1, intra, inter, kind, mv);
    aeg_ph_pop(prev);
}

static void aeg_lowres_mbcosts_fields_ph(const aeg_lowres *cur, const aeg_lowres *r0,
                               const aeg_lowres *r1, const int *icost,
                               const int *mv0, const int *c0,
                               const int *mv1, const int *c1,
                               int *intra, int *inter, uint8_t *kind, int *mv)
{
    int mb_w = cur->mb_w, mb_h = cur->mb_h, x, y;

    for (y = 0; y < mb_h; y++)
        for (x = 0; x < mb_w; x++) {
            int i = y * mb_w + x, cost = 0x7fffffff, k = 0;

            if (r0)
                cost = c0[i];
            if (r0 && r1 && c1[i] < cost) {
                cost = c1[i];
                k = 1;
            }
            if (r1) {
                int c = bidir_cost(cur, r0, r1, 8 * x, 8 * y, mv0 + 2 * i,
                                   mv1 + 2 * i) +
                        mv_cost(mv0[2 * i], mv0[2 * i + 1], 0, 0) +
                        mv_cost(mv1[2 * i], mv1[2 * i + 1], 0, 0);
                if (c < cost) {
                    cost = c;
                    k = 2;
                }
            }
            intra[i] = icost[i];
            inter[i] = cost;
            kind[i] = (uint8_t)k;
            mv[4 * i] = r0 ? mv0[2 * i] : 0;
            mv[4 * i + 1] = r0 ? mv0[2 * i + 1] : 0;
            mv[4 * i + 2] = r1 ? mv1[2 * i] : 0;
            mv[4 * i + 3] = r1 ? mv1[2 * i + 1] : 0;
        }
}

static void aeg_lowres_mbcosts_ph(const aeg_lowres *cur, const aeg_lowres *r0,
                        const aeg_lowres *r1, int *intra, int *inter,
                        uint8_t *kind, int *mv, int *scratch);

/* (timed as a phase, see aeg_ph_switch) */
void aeg_lowres_mbcosts(const aeg_lowres *cur, const aeg_lowres *r0,
                        const aeg_lowres *r1, int *intra, int *inter,
                        uint8_t *kind, int *mv, int *scratch)
{
    const int prev = aeg_ph_push(AEG_PH_LOWRES);

    aeg_lowres_mbcosts_ph(cur, r0, r1, intra, inter, kind, mv, scratch);
    aeg_ph_pop(prev);
}

static void aeg_lowres_mbcosts_ph(const aeg_lowres *cur, const aeg_lowres *r0,
                        const aeg_lowres *r1, int *intra, int *inter,
                        uint8_t *kind, int *mv, int *scratch)
{
    int n = cur->mb_w * cur->mb_h;
    int *mv0 = scratch, *mv1 = scratch + 2 * n;
    int *f = (int *)malloc(sizeof(int) * 3 * (size_t)n);

    if (!f)
        return;
    aeg_lowres_intra(cur, f);
    if (r0)
        aeg_lowres_search(cur, r0, mv0, f + n);
    if (r1)
        aeg_lowres_search(cur, r1, mv1, f + 2 * n);
    aeg_lowres_mbcosts_fields(cur, r0, r1, f, mv0, f + n, mv1, f + 2 * n,
                              intra, inter, kind, mv);
    free(f);
}
