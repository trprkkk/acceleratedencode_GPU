/*
 * tests/cabac_check.c -- the arithmetic coder against an independent decoder.
 *
 * A CABAC bug is worse than a CAVLC bug: the stream stays syntactically legal
 * and decodes to a different picture, so nothing fails until an external
 * decoder disagrees about pixels.  The coder is therefore checked on its own,
 * before any macroblock syntax exists, against a decoder written from the
 * *decoding* process (9.3.3) and from the bit-level definition of the output,
 * not from src/cabac.c: it reads the produced bytes one bit at a time into a
 * nine-bit codIOffset register and never looks at `low`, `range`, `queue` or
 * the carry mechanism the encoder uses.
 *
 * The stream is a pseudo-random but reproducible mix of
 *   - context-coded bins with states taken from the real initialization tables
 *     (all 460 of them, at every QP) and then left to walk wherever the bins
 *     take them, so every entry of aeg_cabac_trans gets visited;
 *   - bypass bins, which are a different renorm path (no range subtraction);
 *   - runs of the same bin, which drive a context to the 0.5 state and to its
 *     most probable state, where the LPS range is at its largest and the
 *     interval can no longer be normalised in one shift;
 * and it ends with a terminate bin, which must decode as "end of slice".
 *
 * Usage: build/cabac_check [bins] [seeds]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "cabac.h"

/* ------------------------------------------------------------------ */
/* A decoder from 9.3.3: nine-bit offset, bit-by-bit refill            */
/* ------------------------------------------------------------------ */
typedef struct {
    const uint8_t *buf;
    size_t   nbits;
    size_t   pos;
    uint32_t range;
    uint32_t offset;           /* codIOffset, always < range after a decision */
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

/* 9.3.3.2.2 */
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

/* 9.3.3.2.3: a bypass bin is one bit of the interval, so the offset's top bit
 * decides it and the range never changes. */
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

/* 9.3.3.2.4 */
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
/* Deterministic pseudo-random source (xorshift, so a failure is        */
/* reproducible from the seed alone)                                   */
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

#define MAX_BINS (1 << 20)

int main(int argc, char **argv)
{
    static uint8_t out[1 << 20];
    static uint8_t init_state[AEG_CABAC_CTX_COUNT];
    static int bins[MAX_BINS];
    static short ctxs[MAX_BINS];   /* -1 = bypass */
    long n_bins = argc > 1 ? atol(argv[1]) : 200000;
    long seeds = argc > 2 ? atol(argv[2]) : 8;
    long total_bits = 0, total_bytes = 0, worst = 0, total_bypass = 0;
    int s, qp, i;

    if (n_bins < 1 || n_bins > MAX_BINS) {
        printf("cabac_check: bins must be 1..%d\n", MAX_BINS);
        return 1;
    }

    for (s = 0; s < seeds; s++) {
        aeg_cabac enc;
        dec d;
        long n, nbypass = 0;
        uint32_t seed = 0x9e3779b9u ^ (uint32_t)(s * 2654435761u);

        rng_state = seed;
        aeg_cabac_open(&enc, out, sizeof out);

        /* A real slice's context set -- every initialization table, at a QP
         * that walks the whole 0..51 range -- then perturbed, so that the
         * stream does not depend on the (m, n) values alone. */
        qp = (int)(rnd() % 52);
        aeg_cabac_set_contexts(&enc, s & 1, qp, (int)(rnd() % 3));
        for (i = 0; i < AEG_CABAC_CTX_COUNT; i++) {
            if (rnd() % 4 == 0)
                enc.state[i] = (uint8_t)(rnd() & 0x7f);
        }
        memcpy(init_state, enc.state, sizeof init_state);

        n = 0;
        while (n < n_bins) {
            int ctx = (int)(rnd() % AEG_CABAC_CTX_COUNT);
            int kind = (int)(rnd() % 10);
            int bin;
            if (kind == 0) {
                bin = (int)(rnd() & 1);
                aeg_cabac_bypass(&enc, bin);
                ctx = -1;
                nbypass++;
            } else if (kind == 2) {
                /* end_of_slice_flag = 0: after every macroblock but the last */
                aeg_cabac_terminate(&enc, 0);
                bins[n] = 0;
                ctxs[n] = -2;
                n++;
                continue;
            } else if (kind == 1) {
                int run = (int)(rnd() % 40) + 1;
                int val = (int)(rnd() & 1);
                int k;
                for (k = 0; k < run && n < n_bins; k++) {
                    aeg_cabac_decision(&enc, ctx, val);
                    bins[n] = val;
                    ctxs[n] = (short)ctx;
                    n++;
                }
                continue;
            } else {
                /* biased towards the context's MPS, but often enough against
                 * it to reach every state it has */
                bin = (enc.state[ctx] & 1) ^ ((rnd() % 5) == 0);
                aeg_cabac_decision(&enc, ctx, bin);
            }
            bins[n] = bin;
            ctxs[n] = (short)ctx;
            n++;
        }
        aeg_cabac_flush(&enc);
        if (enc.err) {
            printf("cabac_check: seed %d: output buffer overflow\n", s);
            return 1;
        }

        dec_init(&d, out, enc.len, init_state);
        for (i = 0; i < n; i++) {
            int got;
            if (ctxs[i] == -2) {
                if (dec_terminate(&d)) {
                    printf("cabac_check: seed %d: end_of_slice_flag decoded 1 "
                           "where it must be 0 (bin %d)\n", s, i);
                    return 1;
                }
                continue;
            }
            got = ctxs[i] < 0 ? dec_bypass(&d) : dec_decision(&d, ctxs[i]);
            if (got != bins[i]) {
                printf("cabac_check: seed %d: bin %ld (ctx %d) decoded as %d, "
                       "expected %d\n", s, (long)i, ctxs[i], got, bins[i]);
                return 1;
            }
        }
        if (!dec_terminate(&d)) {
            printf("cabac_check: seed %d: end_of_slice_flag did not decode as "
                   "1 after %ld bins\n", s, n);
            return 1;
        }
        if (memcmp(d.state, enc.state, sizeof d.state) != 0) {
            for (i = 0; i < AEG_CABAC_CTX_COUNT; i++)
                if (d.state[i] != enc.state[i])
                    break;
            printf("cabac_check: seed %d: context %d ended at state %d, the "
                   "encoder's is %d (a state update disagrees)\n",
                   s, i, d.state[i], enc.state[i]);
            return 1;
        }
        if (d.err) {
            printf("cabac_check: seed %d: the decoder ran out of bytes after "
                   "%ld bins\n", s, n);
            return 1;
        }

        total_bits += n;
        total_bytes += (long)enc.len;
        total_bypass += nbypass;
        if ((long)enc.len > worst)
            worst = (long)enc.len;
        printf("   seed %2d: %7ld bins, %6ld bypass -> %6zu bytes at qp %2d\n",
               s, n, nbypass, enc.len, qp);
    }

    printf("   cabac round-trip: %ld bins (%ld bypass), %ld bytes, %ld seeds, "
           "worst %ld bytes\n", total_bits, total_bypass, total_bytes, seeds,
           worst);
    printf("cabac_check OK: the arithmetic coder round-trips through the "
           "spec's decoder\n");
    return 0;
}
