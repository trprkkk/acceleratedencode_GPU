/*
 * CABAC: the H.264 arithmetic coder (9.3.4), encoder side, plus the two
 * probability tables it is defined in terms of.
 *
 * This is the arithmetic coder only.  The binarizations and the context
 * assignments of the individual syntax elements live with the syntax writer in
 * encoder.c; nothing here knows what a macroblock is.  (The reason the split
 * matters: the coder has to be exactly right before any of the syntax work can
 * be debugged, and a decoder stub can check it on its own -- tests/cabac_check.c
 * does precisely that, against a decoder written from the decoding process in
 * 9.3.3 rather than from this file.)
 */
#include <string.h>
#include "cabac.h"

/* Table 9-44: the LPS subinterval for each of the 64 probability states and
 * each of the four range classes (range >> 6, in 4..7).  Row 0 is the 0.5
 * state and is never reached by the initialization tables (they start at
 * preCtxState 1 -> state 2). */
const uint8_t aeg_cabac_lps_range[64][4] = {
    { 128, 176, 208, 240},
    { 128, 167, 197, 227},
    { 128, 158, 187, 216},
    { 123, 150, 178, 205},
    { 116, 142, 169, 195},
    { 111, 135, 160, 185},
    { 105, 128, 152, 175},
    { 100, 122, 144, 166},
    {  95, 116, 137, 158},
    {  90, 110, 130, 150},
    {  85, 104, 123, 142},
    {  81,  99, 117, 135},
    {  77,  94, 111, 128},
    {  73,  89, 105, 122},
    {  69,  85, 100, 116},
    {  66,  80,  95, 110},
    {  62,  76,  90, 104},
    {  59,  72,  86,  99},
    {  56,  69,  81,  94},
    {  53,  65,  77,  89},
    {  51,  62,  73,  85},
    {  48,  59,  69,  80},
    {  46,  56,  66,  76},
    {  43,  53,  63,  72},
    {  41,  50,  59,  69},
    {  39,  48,  56,  65},
    {  37,  45,  54,  62},
    {  35,  43,  51,  59},
    {  33,  41,  48,  56},
    {  32,  39,  46,  53},
    {  30,  37,  43,  50},
    {  29,  35,  41,  48},
    {  27,  33,  39,  45},
    {  26,  31,  37,  43},
    {  24,  30,  35,  41},
    {  23,  28,  33,  39},
    {  22,  27,  32,  37},
    {  21,  26,  30,  35},
    {  20,  24,  29,  33},
    {  19,  23,  27,  31},
    {  18,  22,  26,  30},
    {  17,  21,  25,  28},
    {  16,  20,  23,  27},
    {  15,  19,  22,  25},
    {  14,  18,  21,  24},
    {  14,  17,  20,  23},
    {  13,  16,  19,  22},
    {  12,  15,  18,  21},
    {  12,  14,  17,  20},
    {  11,  14,  16,  19},
    {  11,  13,  15,  18},
    {  10,  12,  15,  17},
    {  10,  12,  14,  16},
    {   9,  11,  13,  15},
    {   9,  11,  12,  14},
    {   8,  10,  12,  14},
    {   8,   9,  11,  13},
    {   7,   9,  11,  12},
    {   7,   9,  10,  12},
    {   7,   8,  10,  11},
    {   6,   8,   9,  11},
    {   6,   7,   9,  10},
    {   6,   7,   8,   9},
    {   2,   2,   2,   2},
};

/* Tables 9-45/9-46 in one table: aeg_cabac_trans[state][bin] is the next state
 * after coding `bin`.  Note that the two halves of a pair are not symmetric --
 * an MPS bin moves towards the state where the MPS is most likely (row 1), an
 * LPS bin jumps far away from it. */
const uint8_t aeg_cabac_trans[128][2] = {
    {   2,   1},
    {   0,   3},
    {   4,   0},
    {   1,   5},
    {   6,   2},
    {   3,   7},
    {   8,   4},
    {   5,   9},
    {  10,   4},
    {   5,  11},
    {  12,   8},
    {   9,  13},
    {  14,   8},
    {   9,  15},
    {  16,  10},
    {  11,  17},
    {  18,  12},
    {  13,  19},
    {  20,  14},
    {  15,  21},
    {  22,  16},
    {  17,  23},
    {  24,  18},
    {  19,  25},
    {  26,  18},
    {  19,  27},
    {  28,  22},
    {  23,  29},
    {  30,  22},
    {  23,  31},
    {  32,  24},
    {  25,  33},
    {  34,  26},
    {  27,  35},
    {  36,  26},
    {  27,  37},
    {  38,  30},
    {  31,  39},
    {  40,  30},
    {  31,  41},
    {  42,  32},
    {  33,  43},
    {  44,  32},
    {  33,  45},
    {  46,  36},
    {  37,  47},
    {  48,  36},
    {  37,  49},
    {  50,  38},
    {  39,  51},
    {  52,  38},
    {  39,  53},
    {  54,  42},
    {  43,  55},
    {  56,  42},
    {  43,  57},
    {  58,  44},
    {  45,  59},
    {  60,  44},
    {  45,  61},
    {  62,  46},
    {  47,  63},
    {  64,  48},
    {  49,  65},
    {  66,  48},
    {  49,  67},
    {  68,  50},
    {  51,  69},
    {  70,  52},
    {  53,  71},
    {  72,  52},
    {  53,  73},
    {  74,  54},
    {  55,  75},
    {  76,  54},
    {  55,  77},
    {  78,  56},
    {  57,  79},
    {  80,  58},
    {  59,  81},
    {  82,  58},
    {  59,  83},
    {  84,  60},
    {  61,  85},
    {  86,  60},
    {  61,  87},
    {  88,  60},
    {  61,  89},
    {  90,  62},
    {  63,  91},
    {  92,  64},
    {  65,  93},
    {  94,  64},
    {  65,  95},
    {  96,  66},
    {  67,  97},
    {  98,  66},
    {  67,  99},
    { 100,  66},
    {  67, 101},
    { 102,  68},
    {  69, 103},
    { 104,  68},
    {  69, 105},
    { 106,  70},
    {  71, 107},
    { 108,  70},
    {  71, 109},
    { 110,  70},
    {  71, 111},
    { 112,  72},
    {  73, 113},
    { 114,  72},
    {  73, 115},
    { 116,  72},
    {  73, 117},
    { 118,  74},
    {  75, 119},
    { 120,  74},
    {  75, 121},
    { 122,  74},
    {  75, 123},
    { 124,  76},
    {  77, 125},
    { 124,  76},
    {  77, 125},
    { 126, 126},
    { 127, 127},
};

/* What one bin costs, in 1/256 bit, for each probability state and each bin
 * value: -log2 of the interval that state gives the bin.  This is used only by
 * the rate-distortion pass (`aeg_cabac.cost`), which runs the coder for its
 * contexts and its bit count rather than for its bytes; the stream itself
 * never consults it.  The probability of the LPS is taken as the mean of
 * `rangeTabLPS[pStateIdx][c] / (mean range of class c)` over the four range
 * classes, which is the closest thing to a single number that a range-
 * dependent table has. */
const uint16_t aeg_cabac_bin_cost[128][2] = {
    {  247,  265 },
    {  265,  247 },
    {  233,  281 },
    {  281,  233 },
    {  220,  295 },
    {  295,  220 },
    {  206,  313 },
    {  313,  206 },
    {  192,  333 },
    {  333,  192 },
    {  180,  352 },
    {  352,  180 },
    {  168,  372 },
    {  372,  168 },
    {  158,  390 },
    {  390,  158 },
    {  148,  409 },
    {  409,  148 },
    {  139,  429 },
    {  429,  139 },
    {  130,  449 },
    {  449,  130 },
    {  122,  468 },
    {  468,  122 },
    {  115,  487 },
    {  487,  115 },
    {  108,  506 },
    {  506,  108 },
    {  102,  525 },
    {  525,  102 },
    {   96,  544 },
    {  544,   96 },
    {   90,  565 },
    {  565,   90 },
    {   85,  583 },
    {  583,   85 },
    {   80,  602 },
    {  602,   80 },
    {   76,  623 },
    {  623,   76 },
    {   72,  640 },
    {  640,   72 },
    {   68,  661 },
    {  661,   68 },
    {   64,  678 },
    {  678,   64 },
    {   60,  699 },
    {  699,   60 },
    {   57,  719 },
    {  719,   57 },
    {   54,  737 },
    {  737,   54 },
    {   51,  756 },
    {  756,   51 },
    {   48,  775 },
    {  775,   48 },
    {   46,  795 },
    {  795,   46 },
    {   43,  812 },
    {  812,   43 },
    {   41,  834 },
    {  834,   41 },
    {   39,  851 },
    {  851,   39 },
    {   36,  873 },
    {  873,   36 },
    {   35,  892 },
    {  892,   35 },
    {   33,  911 },
    {  911,   33 },
    {   31,  932 },
    {  932,   31 },
    {   30,  947 },
    {  947,   30 },
    {   28,  966 },
    {  966,   28 },
    {   26,  986 },
    {  986,   26 },
    {   25, 1007 },
    { 1007,   25 },
    {   24, 1023 },
    { 1023,   24 },
    {   23, 1043 },
    { 1043,   23 },
    {   21, 1064 },
    { 1064,   21 },
    {   20, 1086 },
    { 1086,   20 },
    {   19, 1105 },
    { 1105,   19 },
    {   18, 1119 },
    { 1119,   18 },
    {   17, 1140 },
    { 1140,   17 },
    {   16, 1163 },
    { 1163,   16 },
    {   16, 1179 },
    { 1179,   16 },
    {   15, 1197 },
    { 1197,   15 },
    {   14, 1215 },
    { 1215,   14 },
    {   13, 1236 },
    { 1236,   13 },
    {   13, 1248 },
    { 1248,   13 },
    {   12, 1279 },
    { 1279,   12 },
    {   11, 1292 },
    { 1292,   11 },
    {   11, 1312 },
    { 1312,   11 },
    {   10, 1337 },
    { 1337,   10 },
    {   10, 1357 },
    { 1357,   10 },
    {    9, 1365 },
    { 1365,    9 },
    {    9, 1384 },
    { 1384,    9 },
    {    8, 1408 },
    { 1408,    8 },
    {    8, 1429 },
    { 1429,    8 },
    {    7, 1450 },
    { 1450,    7 },
    {    2, 1927 },
    { 1927,    2 },
};

/* How many bits a range needs before it is back in [256, 512), indexed by
 * range >> 3.  The smallest range a decision can produce is 256 - 240 = 16
 * (state 63's LPS against the widest range), so entries 0 and 1 never occur. */
static const uint8_t renorm_shift[64] = {
    6, 5, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};

static void cabac_emit(aeg_cabac *c, uint8_t b)
{
    if (c->len >= c->cap) {
        c->err = 1;
        return;
    }
    c->buf[c->len++] = b;
}

void aeg_cabac_open(aeg_cabac *c, uint8_t *buf, size_t cap)
{
    c->buf = buf;
    c->cap = cap;
    c->len = 0;
    c->err = 0;
    c->low = 0;
    c->range = 0x01FE;
    c->queue = -9;             /* the first 9 bits are not written: the
                                * decoder's codIOffset starts where they end */
    c->outstanding = 0;
    c->prev = 0;
    c->have_prev = 0;
    c->cost = 0;               /* byte-producing coder until asked otherwise */
    c->cost256 = 0;
}

void aeg_cabac_set_contexts(aeg_cabac *c, int intra, int qp, int init_idc)
{
    const int8_t (*tab)[2] = intra ? aeg_cabac_init_I
                                   : aeg_cabac_init_PB[init_idc & 3];
    int i;
    for (i = 0; i < AEG_CABAC_CTX_COUNT; i++) {
        /* preCtxState (9.3.1.1) folded into one byte: the state is
         * 2 * pStateIdx + valMPS, and preCtxState and 127 - preCtxState map to
         * the same pStateIdx with opposite MPS values, so the whole state space
         * is reached by reflecting the pre-state around 63.5. */
        int st = 2 * (((tab[i][0] * qp) >> 4) + tab[i][1]) - 127;
        if (st < 0)
            st = ~st;                   /* |preCtxState - 63| - 1 */
        if (st > 124)
            st = 124 + (st & 1);
        c->state[i] = (uint8_t)st;
    }
}


/* Extracts one byte from `low` once at least eight bits above the pending ones
 * are known.  `queue` counts the bits shifted in since the last extraction, so
 * the byte is the bits [queue + 10, queue + 18) of `low`, the "+ 10" being the
 * nine bits that are still unresolved below it plus its own lowest bit. */
void aeg_cabac_putbyte(aeg_cabac *c)
{
    if (c->queue < 0)
        return;
    {
        uint32_t out = c->low >> (c->queue + 10);
        c->low &= (0x400u << c->queue) - 1u;
        c->queue -= 8;
        if ((out & 0xff) == 0xff) {
            c->outstanding++;       /* a carry would make this 0x00 */
        } else {
            int carry = (int)(out >> 8);
            if (c->have_prev)
                cabac_emit(c, (uint8_t)(c->prev + carry));
            while (c->outstanding > 0) {
                cabac_emit(c, (uint8_t)(carry - 1));
                c->outstanding--;
            }
            c->prev = (uint8_t)out;
            c->have_prev = 1;
        }
    }
}

/* Shifts until the range is back in [256, 512).  One shift is the normal case;
 * the loop exists because a state's LPS range can be as small as 2 (state 0,
 * the 0.5 state), and a decoder normalises with `while (range < 256)` so it
 * would take two shifts of six and one bits for the same input.  Shifting in
 * two steps emits the same bytes as shifting in one, because the byte
 * extraction is incremental. */
void aeg_cabac_renorm(aeg_cabac *c)
{
    while (c->range < 0x100) {
        int shift = renorm_shift[c->range >> 3];
        c->range <<= shift;
        c->low <<= shift;
        c->queue += shift;
        aeg_cabac_putbyte(c);
    }
}

size_t aeg_cabac_flush(aeg_cabac *c)
{
    /* The final end_of_slice_flag = 1 (9.3.4.5).  It is coded against a fixed
     * two-unit LPS interval: the value moves to the top two units of the
     * range, and an odd value inside them is chosen so that the decoder's test
     * (`codIOffset >= range` after `range -= 2`) comes out true. */
    c->low += c->range - 2;
    c->low |= 1;

    /* Nine more bits than the interval needs: the decoder pre-loads nine bits
     * before it decodes anything, so the encoder is nine bits ahead of what it
     * has written.  The shift makes those nine bits writable. */
    c->low <<= 9;
    c->queue += 9;

    /* Write everything that is decided now. */
    while (c->queue >= 0)
        aeg_cabac_putbyte(c);

    /* Align what is left and close with a stop bit: the last byte is then
     * never 0x00, which would look like a start code in the packed NAL. */
    if (c->queue < 0) {
        c->low <<= -c->queue;
        c->queue = 0;
    }
    c->low |= 1u << 10;
    aeg_cabac_putbyte(c);

    if (c->have_prev) {
        cabac_emit(c, c->prev);
        c->have_prev = 0;
    }
    while (c->outstanding > 0) {
        cabac_emit(c, 0xff);
        c->outstanding--;
    }
    return c->len;
}

/* ------------------------------------------------------------------ */
/* Residual block syntax (9.3.2.2)                                     */
/* ------------------------------------------------------------------ */
/* The significance map, the last-coefficient flag, the level state machine and
 * the two escape codes are the largest piece of binarization work, and the
 * part a decoder either reproduces exactly or turns into noise.  It therefore
 * lives here with the coder rather than with the macroblock syntax, and
 * tests/cabac_mb_check.c round-trips it against a transcription of the
 * reference decoder's residual_block_cabac. */
/* Residual categories (ctxBlockCat) of a 4:2:0 picture: the three context
 * offsets each, and the coded_block_flag base (Table 9-40). */
struct cab_cat_ctx { uint16_t sig, last, level; };
static const struct cab_cat_ctx cab_cat[6] = {
    { 105, 166, 227 },  /* 0: Intra_16x16 luma DC */
    { 120, 181, 237 },  /* 1: Intra_16x16 luma AC */
    { 134, 195, 247 },  /* 2: luma 4x4            */
    { 149, 210, 257 },  /* 3: chroma DC           */
    { 152, 213, 266 },  /* 4: chroma AC           */
    { 402, 417, 426 }   /* 5: luma 8x8            */
};

/* The level state machine of 9.3.2.2: from the "node" the previous coefficient
 * left behind, which context codes the level (1 vs >1) and where the node goes
 * next. */
static const uint8_t cab_lvl1[8] = { 1, 2, 3, 4, 0, 0, 0, 0 };
static const uint8_t cab_gt1[8]  = { 5, 5, 5, 5, 6, 7, 8, 9 };
static const uint8_t cab_lvl_next[2][8] = {
    { 1, 2, 3, 3, 4, 5, 6, 7 },     /* after a level of 1     */
    { 4, 4, 4, 4, 5, 6, 7, 7 }      /* after a level above 1  */
};

/* One residual block: coded_block_flag (when the category has one), the
 * significance map with the last-coefficient flag, then the levels backwards
 * from the last significant coefficient.  `coef[i]` is the coefficient at scan
 * position i; `n` of them.  Returns the number of non-zero coefficients, which
 * is what the neighbouring blocks' contexts want. */
int aeg_cabac_residual(aeg_cabac *cb, const int16_t *coef, int n, int cat,
                       int cbf_ctx)
{
    const struct cab_cat_ctx *k = &cab_cat[cat];
    int i, last = -1, count = 0;

    for (i = 0; i < n; i++)
        if (coef[i]) { last = i; count++; }
    if (cbf_ctx >= 0) {
        aeg_cabac_decision(cb, cbf_ctx, count != 0);
        if (!count)
            return 0;
    }
    /* Significance map.  The highest scan position is never signalled -- if the
     * scan gets there it is the last coefficient by definition -- and a 64
     * coefficient block indexes its contexts by scan position through the two
     * tables of 9.3.2.3 instead of the 4x4 neighbourhood rule. */
    if (n == 64) {
        for (i = 0; i < 63; i++) {
            int nz = coef[i] != 0;
            aeg_cabac_decision(cb, k->sig + aeg_cabac_sig8[i], nz);
            if (!nz)
                continue;
            aeg_cabac_decision(cb, k->last + aeg_cabac_last8[i], i == last);
            if (i == last)
                break;
        }
    } else {
        for (i = 0; i < n - 1; i++) {
            int nz = coef[i] != 0;
            aeg_cabac_decision(cb, k->sig + i, nz);
            if (!nz)
                continue;
            aeg_cabac_decision(cb, k->last + i, i == last);
            if (i == last)
                break;
        }
    }
    /* Levels, from the last coefficient back to the first. */
    {
        int node = 0;
        for (i = last; i >= 0; i--) {
            int v = coef[i], abs, neg, cnt, ctx;
            if (!v)
                continue;
            neg = v < 0;
            abs = neg ? -v : v;
            aeg_cabac_decision(cb, k->level + cab_lvl1[node], abs > 1);
            if (abs == 1) {
                node = cab_lvl_next[0][node];
            } else {
                ctx = k->level + cab_gt1[node];
                node = cab_lvl_next[1][node];
                for (cnt = 2; cnt < 15 && cnt < abs; cnt++)
                    aeg_cabac_decision(cb, ctx, 1);
                if (cnt < 15) {
                    aeg_cabac_decision(cb, ctx, 0);
                } else {
                    /* Escape: the level above 14 is sent as an Exp-Golomb code
                     * with bypass bins, offset by 14. */
                    int rest = abs - 14, j = 0;
                    while (rest >= (1 << (j + 1)))
                        j++;
                    for (cnt = 0; cnt < j; cnt++)
                        aeg_cabac_bypass(cb, 1);
                    aeg_cabac_bypass(cb, 0);
                    for (cnt = j - 1; cnt >= 0; cnt--)
                        aeg_cabac_bypass(cb, (rest >> cnt) & 1);
                }
            }
            aeg_cabac_bypass(cb, neg);
        }
    }
    return count;
}


