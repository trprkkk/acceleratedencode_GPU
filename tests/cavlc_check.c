/*
 * CAVLC round-trip check.
 *
 * Encodes pseudo-random coefficient blocks with aeg_cavlc_write_block() and
 * decodes them again with a transcription of the reference decoder's
 * (FFmpeg h264_cavlc.c:decode_residual) level / run decoding.  Any mismatch is
 * printed with the offending block, which makes writer bugs visible without
 * having to go through a full decoder.
 *
 * build: gcc -O2 -std=c11 -Iinclude -Isrc -o build/cavlc_check tests/cavlc_check.c \
 *            src/cavlc.c src/tables.c src/util.c src/transform.c src/bitstream.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "h264.h"

typedef struct {
    const uint8_t *p;
    int bit, nbits;
} br_t;

static uint32_t br_show(const br_t *b, int n)
{
    uint32_t v = 0;
    int i;
    for (i = 0; i < n; i++) {
        int pos = b->bit + i;
        int bit = pos < b->nbits ? ((b->p[pos >> 3] >> (7 - (pos & 7))) & 1) : 0;
        v = (v << 1) | (uint32_t)bit;
    }
    return v;
}

static void br_skip(br_t *b, int n) { b->bit += n; }

/* longest-prefix VLC lookup over a flat [len][bits] table */
static int vlc_lookup(br_t *b, const uint8_t *len, const uint8_t *bits,
                      int n, int *out_len)
{
    int i, best = -1, best_len = 0;
    for (i = 0; i < n; i++) {
        if (!len[i])
            continue;
        if (len[i] > best_len && br_show(b, len[i]) == bits[i]) {
            best = i;
            best_len = len[i];
        }
    }
    if (best < 0)
        return -1;
    br_skip(b, best_len);
    if (out_len)
        *out_len = best_len;
    return best;
}

static int nc_bucket(int nC) { return nC < 2 ? 0 : (nC < 4 ? 1 : (nC < 8 ? 2 : 3)); }

/* Returns 0 on success; fills out[] with the decoded coefficients. */
static int decode_block(br_t *b, int nC, int kind, int16_t *out, const char *tag)
{
    static const uint8_t zig[16] = { 0,1,4,8,5,2,3,6,9,12,13,10,7,11,14,15 };
    const uint8_t *scan;
    int nscan, max_coeff, is_cdc = 0, total, trailing, i, zeros_total, zeros_left;
    int lvl[16], pos[16];
    int suffix, level_code;

    memset(out, 0, 16 * sizeof(int16_t));
    switch (kind) {
    case 1: scan = zig; nscan = 16; max_coeff = 16; break;
    case 2: scan = zig; nscan = 4;  max_coeff = 4; is_cdc = 1; break;
    case 3: scan = zig + 1; nscan = 15; max_coeff = 15; break;
    default: scan = zig; nscan = 16; max_coeff = 16; break;
    }

    if (is_cdc) {
        total = vlc_lookup(b, aeg_chroma_dc_coeff_token_len,
                           aeg_chroma_dc_coeff_token_bits, 20, NULL);
    } else {
        int buck = nc_bucket(nC);
        total = vlc_lookup(b, aeg_coeff_token_len[buck],
                           aeg_coeff_token_bits[buck], 4 * 17, NULL);
    }
    if (total < 0) {
        printf("%s: unparsable coeff_token\n", tag);
        return -1;
    }
    trailing = total & 3;
    total >>= 2;
    if (total == 0)
        return 0;
    if (total > max_coeff) {
        printf("%s: total_coeff %d > max %d\n", tag, total, max_coeff);
        return -1;
    }

    for (i = 0; i < trailing; i++)
        lvl[i] = br_show(b, 1) ? -1 : 1, br_skip(b, 1);

    suffix = (total > 10 && trailing < 3);
    for (i = trailing; i < total; i++) {
        int prefix = 0;
        while (prefix < 32 && br_show(b, 1) == 0) {
            br_skip(b, 1);
            prefix++;
        }
        br_skip(b, 1);
        if (i == trailing && trailing < 3 && prefix == 14 && suffix == 0) {
            level_code = 14 + (int)br_show(b, 4);
            br_skip(b, 4);
        } else if (prefix < 15) {
            level_code = (prefix << suffix) + (int)br_show(b, suffix);
            br_skip(b, suffix);
        } else {
            int ext = (int)br_show(b, prefix - 3);
            br_skip(b, prefix - 3);
            if (i == trailing && trailing < 3)
                level_code = 30 + ext;
            else
                level_code = (15 << suffix) + ext;
            if (prefix >= 16) {
                if (prefix > 28) {
                    printf("%s: invalid level prefix %d\n", tag, prefix);
                    return -1;
                }
                level_code += (1 << (prefix - 3)) - 4096;
            }
        }
        if (i == trailing && trailing < 3)   /* the decoder adds 2 back */
            level_code += 2;
        {
            int mask = -(level_code & 1);
            int mag = ((2 + level_code) >> 1);
            lvl[i] = ((mag ^ mask) - mask);
        }
        if (suffix == 0)
            suffix = 1;
        else if ((lvl[i] < 0 ? -lvl[i] : lvl[i]) > (3 << (suffix - 1)) && suffix < 6)
            suffix++;
    }

    zeros_total = 0;
    if (total < max_coeff) {
        int idx;
        if (is_cdc) {
            /* the chroma-DC table is stored per total_coeff */
            const uint8_t *len = aeg_chroma_dc_total_zeros_len[total - 1];
            const uint8_t *bits = aeg_chroma_dc_total_zeros_bits[total - 1];
            for (idx = 0; idx <= max_coeff - total; idx++) {
                if (len[idx] && br_show(b, len[idx]) == bits[idx]) {
                    zeros_total = idx;
                    br_skip(b, len[idx]);
                    break;
                }
            }
        } else {
            for (idx = 0; idx <= max_coeff - total; idx++) {
                if (aeg_total_zeros_len[total - 1][idx] &&
                    br_show(b, aeg_total_zeros_len[total - 1][idx]) ==
                        aeg_total_zeros_bits[total - 1][idx]) {
                    zeros_total = idx;
                    br_skip(b, aeg_total_zeros_len[total - 1][idx]);
                    break;
                }
            }
        }
    }

    zeros_left = zeros_total;
    pos[0] = zeros_total + total - 1;
    for (i = 1; i < total; i++) {
        int run = 0;
        if (zeros_left > 0) {
            int z = zeros_left > 7 ? 7 : zeros_left;
            int idx;
            for (idx = 0; idx < (z == 7 ? 15 : zeros_left); idx++) {
                if (aeg_run_before_len[z - 1][idx] &&
                    br_show(b, aeg_run_before_len[z - 1][idx]) ==
                        aeg_run_before_bits[z - 1][idx]) {
                    run = idx;
                    br_skip(b, aeg_run_before_len[z - 1][idx]);
                    break;
                }
            }
            zeros_left -= run;
        }
        pos[i] = pos[i - 1] - run - 1;
    }

    for (i = 0; i < total; i++)
        out[scan[pos[i]]] = (int16_t)lvl[i];
    return 0;
}

static unsigned long rnd_state = 12345;
static int rnd(int mod)
{
    rnd_state = rnd_state * 1103515245u + 12345u;
    return (int)((rnd_state >> 8) % (unsigned)mod);
}

int main(int argc, char **argv)
{
    uint8_t buf[4096];
    int kind, iter, fails = 0, runs = argc > 1 ? atoi(argv[1]) : 2000;

    for (kind = 0; kind <= 3; kind++) {
        for (iter = 0; iter < runs; iter++) {
            int16_t coef[16], back[16];
            aeg_bitwriter bw;
            br_t br;
            int i, nz, nscan = kind == 3 ? 15 : (kind == 2 ? 4 : 16);
            int nC = rnd(20);
            int bwrote;

            memset(coef, 0, sizeof(coef));
            /* a handful of non-zero coefficients, magnitudes up to +-400 */
            nz = 1 + rnd(kind == 2 ? 4 : 6);
            for (i = 0; i < nz; i++) {
                int idx = rnd(nscan);
                int mag = 1 + rnd(400);
                int16_t v = (int16_t)(rnd(2) ? mag : -mag);
                coef[kind == 3 ? aeg_zigzag4x4[idx + 1] : aeg_zigzag4x4[idx]] = v;
            }
            aeg_bw_init(&bw, buf, sizeof(buf), 0);
            aeg_cavlc_write_block(&bw, coef, nC, kind);
            bwrote = (int)aeg_bw_finish(&bw);

            br.p = buf;
            br.bit = 0;
            br.nbits = bwrote * 8;
            if (decode_block(&br, nC, kind, back, "block") < 0) {
                if (++fails < 6) {
                    printf("kind %d nC %d FAIL(parse) coef:", kind, nC);
                    for (i = 0; i < 16; i++) printf(" %d", coef[i]);
                    printf("\n");
                }
                continue;
            }
            if (memcmp(coef, back, sizeof(coef)) != 0) {
                if (++fails < 6) {
                    printf("kind %d nC %d MISMATCH\n  written:", kind, nC);
                    for (i = 0; i < 16; i++) printf(" %d", coef[i]);
                    printf("\n  decoded:");
                    for (i = 0; i < 16; i++) printf(" %d", back[i]);
                    printf("\n");
                }
            }
        }
    }
    printf("%s: %d failures\n", fails ? "FAIL" : "OK", fails);
    return fails ? 1 : 0;
}
