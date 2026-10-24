/*
 * tests/cabac_mb_check.c -- the CABAC residual block syntax against an
 * independent decoder.
 *
 * The residual block is where a CABAC encoder breaks silently: the coded block
 * flag carries the *pattern* of non-zero coefficients, so a wrong significance
 * context or a wrong level state gives a block that still decodes -- to
 * different coefficients -- and the picture quietly drifts.  This check
 * round-trips blocks of every category through a decoder transcribed from the
 * decoding process of 9.3.2.2 (and from the reference decoder's
 * residual_block_cabac), which reads the stream bit by bit and shares nothing
 * with src/cabac.c but the context tables of the standard.
 *
 * Usage: build/cabac_mb_check [blocks] [seed]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "cabac.h"

/* ------------------------------------------------------------------ */
/* Decoder (9.3.3), the same nine-bit offset reader as cabac_check     */
/* ------------------------------------------------------------------ */
typedef struct {
    const uint8_t *buf;
    size_t   nbits;
    size_t   pos;
    uint32_t range;
    uint32_t offset;
    int      err;
    uint8_t  state[AEG_CABAC_CTX_COUNT];
} dec;

static int dec_bit(dec *d)
{
    size_t byte = d->pos >> 3;
    int bit;

    if (byte >= d->nbits / 8) {
        d->err = 1;
        return 0;
    }
    bit = (d->buf[byte] >> (7 - (d->pos & 7))) & 1;
    d->pos++;
    return bit;
}

static void dec_init(dec *d, const uint8_t *buf, size_t len_bytes,
                     const uint8_t *states)
{
    int i;

    d->buf = buf;
    d->nbits = len_bytes * 8;
    d->pos = 0;
    d->range = 510;
    d->offset = 0;
    d->err = 0;
    memcpy(d->state, states, sizeof d->state);
    for (i = 0; i < 9; i++)
        d->offset = (d->offset << 1) | (uint32_t)dec_bit(d);
}

static int dec_decision(dec *d, int ctx)
{
    int s = d->state[ctx];
    int lps = aeg_cabac_lps_range[s >> 1][(d->range >> 6) - 4];
    int bin;

    d->range -= (uint32_t)lps;
    if (d->offset >= d->range) {
        bin = 1 - (s & 1);
        d->offset -= d->range;
        d->range = (uint32_t)lps;
    } else {
        bin = s & 1;
    }
    d->state[ctx] = aeg_cabac_trans[s][bin];
    while (d->range < 0x100) {
        d->range <<= 1;
        d->offset = (d->offset << 1) | (uint32_t)dec_bit(d);
    }
    return bin;
}

static int dec_bypass(dec *d)
{
    int bin;

    d->offset = (d->offset << 1) | (uint32_t)dec_bit(d);
    if (d->offset >= d->range) {
        d->offset -= d->range;
        bin = 1;
    } else {
        bin = 0;
    }
    return bin;
}

static int dec_terminate(dec *d)
{
    d->range -= 2;
    if (d->offset >= d->range)
        return 1;
    if (d->range < 0x100) {
        d->range <<= 1;
        d->offset = (d->offset << 1) | (uint32_t)dec_bit(d);
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* The residual block as a decoder reads it (9.3.2.2)                  */
/* ------------------------------------------------------------------ */
struct cat_ctx { int sig, last, level; };
static const struct cat_ctx cat[6] = {
    { 105, 166, 227 }, { 120, 181, 237 }, { 134, 195, 247 },
    { 149, 210, 257 }, { 152, 213, 266 }, { 402, 417, 426 }
};
static const int lvl1[8] = { 1, 2, 3, 4, 0, 0, 0, 0 };
static const int gt1[8]  = { 5, 5, 5, 5, 6, 7, 8, 9 };
static const int lvl_next[2][8] = {
    { 1, 2, 3, 3, 4, 5, 6, 7 }, { 4, 4, 4, 4, 5, 6, 7, 7 }
};

/* Returns the number of coefficients decoded, or -1 on a stream that ran out. */
static int dec_residual(dec *d, int16_t *coef, int n, int k, int has_cbf,
                        int cbf_ctx)
{
    int pos[64], cnt = 0, i, lm, node = 0;

    for (i = 0; i < n; i++)
        coef[i] = 0;
    if (has_cbf && !dec_decision(d, cbf_ctx))
        return 0;

    if (n == 64) {
        for (lm = 0; lm < 63; lm++) {
            if (!dec_decision(d, cat[k].sig + aeg_cabac_sig8[lm]))
                continue;
            pos[cnt++] = lm;
            if (dec_decision(d, cat[k].last + aeg_cabac_last8[lm])) {
                lm = 64;
                break;
            }
        }
        if (lm == 63)
            pos[cnt++] = 63;
    } else {
        for (lm = 0; lm < n - 1; lm++) {
            if (!dec_decision(d, cat[k].sig + lm))
                continue;
            pos[cnt++] = lm;
            if (dec_decision(d, cat[k].last + lm)) {
                lm = n;
                break;
            }
        }
        if (lm == n - 1)
            pos[cnt++] = n - 1;
    }
    for (i = cnt - 1; i >= 0; i--) {
        int abs, j = pos[i];

        if (!dec_decision(d, cat[k].level + lvl1[node])) {
            node = lvl_next[0][node];
            abs = 1;
        } else {
            int ctx = cat[k].level + gt1[node];
            node = lvl_next[1][node];
            abs = 2;
            while (abs < 15 && dec_decision(d, ctx))
                abs++;
            if (abs >= 15) {
                int p = 0, t;
                while (dec_bypass(d) && p < 23)
                    p++;
                abs = 1;
                for (t = p; t-- > 0;)
                    abs += abs + dec_bypass(d);
                abs += 14;
            }
        }
        coef[j] = (int16_t)(dec_bypass(d) ? -abs : abs);
    }
    return cnt;
}

/* ------------------------------------------------------------------ */
/* Pseudo-random blocks                                                */
/* ------------------------------------------------------------------ */
static uint32_t rng_state;
static uint32_t rnd(void)
{
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}

/* A quantised block at a plausible QP: mostly zeros, a few small levels and
 * occasionally a large one (which is what exercises the escape code). */
static void make_block(int16_t *coef, int n, int nz)
{
    int i, placed = 0;

    for (i = 0; i < n; i++)
        coef[i] = 0;
    while (placed < nz) {
        int p = (int)(rnd() % (uint32_t)n);
        int v;
        if (coef[p])
            continue;
        v = (int)(rnd() % 100);
        v = v < 60 ? 1 : (v < 85 ? 2 : (v < 95 ? 3 + (int)(rnd() % 5) : 16 + (int)(rnd() % 60)));
        coef[p] = (int16_t)((rnd() & 1) ? v : -v);
        placed++;
    }
}

int main(int argc, char **argv)
{
    static uint8_t out[1 << 16];
    static uint8_t states[AEG_CABAC_CTX_COUNT];
    static int16_t src[64], got[64];
    long blocks = argc > 1 ? atol(argv[1]) : 20000;
    uint32_t seed = argc > 2 ? (uint32_t)strtoul(argv[2], NULL, 0) : 1u;
    long b;
    int bad = 0;

    rng_state = seed ? seed : 1u;
    for (b = 0; b < blocks && bad < 10; b++) {
        aeg_cabac enc;
        dec d;
        int cat_k = (int)(rnd() % 6);
        int n = cat_k == 5 ? 64 : (cat_k == 3 ? 4 : (cat_k == 1 || cat_k == 4 ? 15 : 16));
        int has_cbf = cat_k != 5;             /* luma 8x8 has no coded_block_flag */
        int cbf_ctx = has_cbf ? (int)(rnd() % AEG_CABAC_CTX_COUNT) : -1;
        int nz = (int)(rnd() % 8);
        size_t len;
        int cnt;
        const char *why = NULL;

        if (rnd() % 16 == 0)
            nz = n;                            /* a fully populated block */
        if (nz > n)
            nz = n;                            /* 4-coefficient chroma DC */
        /* An 8x8 luma block has no coded_block_flag: the block only exists at
         * all because its group's pattern bit is set, so an all-zero one can
         * not be coded (and the decoder would read the implied position 63). */
        if (cat_k == 5 && nz == 0)
            nz = 1;
        make_block(src, n, nz);
        if (has_cbf && nz == 0)
            src[0] = 0;                        /* keep the empty case */
        aeg_cabac_open(&enc, out, sizeof out);
        aeg_cabac_set_contexts(&enc, 1, (int)(rnd() % 52), 0);
        memcpy(states, enc.state, sizeof states);
        aeg_cabac_residual(&enc, src, n, cat_k, cbf_ctx);
        aeg_cabac_terminate(&enc, 1);
        len = aeg_cabac_flush(&enc);

        dec_init(&d, out, len, states);
        cnt = dec_residual(&d, got, n, cat_k, has_cbf, cbf_ctx);
        if (cnt != nz) {
            printf("block %ld: cat %d n %d: decoded %d coefficients, sent %d\n",
                   b, cat_k, n, cnt, nz);
            why = "count";
        } else {
            int i;
            for (i = 0; i < n; i++)
                if (src[i] != got[i]) {
                    printf("block %ld: cat %d n %d pos %d: sent %d, decoded %d\n",
                           b, cat_k, n, i, src[i], got[i]);
                    why = "value";
                    break;
                }
        }
        if (!why && !dec_terminate(&d)) {
            printf("block %ld: end-of-slice flag did not decode\n", b);
            why = "terminate";
        }
        if (why) {
            int i;
            bad++;
            printf("  sent:");
            for (i = 0; i < n; i++)
                printf(" %d", src[i]);
            printf("\n  got :");
            for (i = 0; i < n; i++)
                printf(" %d", got[i]);
            printf("\n");
        }
    }
    if (bad) {
        printf("cabac_mb_check: %d failing block(s) from seed %u\n", bad, seed);
        return 1;
    }
    printf("cabac_mb_check OK: %ld residual blocks, all six categories "
           "round-trip (seed %u)\n", blocks, seed);
    return 0;
}
