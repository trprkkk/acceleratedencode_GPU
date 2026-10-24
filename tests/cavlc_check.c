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
    int i, L, best = -1, best_len = 0;
    /* prefix code: the shortest matching codeword wins */
    for (L = 1; L <= 16 && best < 0; L++)
        for (i = 0; i < n; i++)
            if (len[i] == L && br_show(b, L) == bits[i]) {
                best = i;
                best_len = L;
                break;
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
/* number of leading zero bits before the terminating one */
static int level_prefix(br_t *b)
{
    int n = 0;
    while (!br_show(b, 1) && n < 32) {
        br_skip(b, 1);
        n++;
    }
    br_skip(b, 1);
    return n;
}

static int decode_block(br_t *b, int nC, int kind, int16_t *out, const char *tag)
{
    /* 4x4 CAVLC scan of the decoder: TRANSPOSE(zig-zag). */
    static const uint8_t dct[16] = { 0,4,1,2,5,8,12,9,6,3,7,10,13,14,11,15 };
    static const uint8_t raster[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
    const uint8_t *scan;
    int nscan, max_coeff, is_cdc = 0, total, trailing, i, zeros_total, zeros_left;
    static const uint8_t dct_tab[16] = { 0,4,1,2,5,8,12,9,6,3,7,10,13,14,11,15 };
    (void)dct_tab;
    int lvl[16], pos[16];
    int suffix, level_code;

    memset(out, 0, 16 * sizeof(int16_t));
    switch (kind) {
    case 1: scan = dct; nscan = 16; max_coeff = 16; break;
    case 2: scan = raster; nscan = 4; max_coeff = 4; is_cdc = 1; break;
    case 3: scan = dct + 1; nscan = 15; max_coeff = 15; break;
    default: scan = dct; nscan = 16; max_coeff = 16; break;
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
    if (getenv("CAVLC_TRACE"))
        printf("  [trace] kind %d nC %d -> total %d trailing %d (bit %d)\n",
               kind, nC, total, trailing, b->bit);
    if (total > max_coeff) {
        printf("%s: total_coeff %d > max %d\n", tag, total, max_coeff);
        return -1;
    }

    /* The reference decoder peeks three bits and skips trailing_ones, because
     * the flag bits follow the trailing-one signs directly. */
    {
        int i3 = (int)br_show(b, 3);
        lvl[0] = 1 - ((i3 & 4) >> 1);
        lvl[1] = 1 - (i3 & 2);
        lvl[2] = 1 - ((i3 & 1) << 1);
        br_skip(b, trailing);
    }

    suffix = (total > 10 && trailing < 3);
    {
        static const int suffix_limit[7] = { 0, 3, 6, 12, 24, 48, 0x7fffffff };
        for (i = trailing; i < total; i++) {
            int prefix = level_prefix(b);
            int bit_at = b->bit;

            if (i == trailing && trailing < 3 && prefix + 1 + suffix <= 9) {
                /* short first-level code: the value is complete inside the
                 * decoder's look-up table, so it carries its own +1 bias. */
                /* levelCode = (prefix << suffixLength) + suffix bits; the
                 * leading one the table folds in cancels the (1<<suffix). */
                int lc = (prefix << suffix) +
                         (suffix ? (int)br_show(b, suffix) : 0);
                int mask, l;
                if (suffix)
                    br_skip(b, suffix);
                mask = -(lc & 1);
                l = (((2 + lc) >> 1) ^ mask) - mask;
                l += ((l >> 31) | 1) & -(trailing < 3);
                suffix = 1 + ((unsigned)(l + 3) > 6U);
                lvl[i] = l;
                continue;
            }

            {
            int is_first = (i == trailing && trailing < 3);
            if (is_first) {
                /* first non-trailing level: suffixLength is 0 or 1 here and
                 * the encoder subtracted the +2 bias, so add it back. */
                if (prefix < 14) {
                    level_code = suffix ? ((prefix << 1) + (int)br_show(b, 1))
                                        : prefix;
                    if (suffix)
                        br_skip(b, 1);
                } else if (prefix == 14) {
                    if (suffix) {
                        level_code = (14 << 1) + (int)br_show(b, 1);
                        br_skip(b, 1);
                    } else {
                        level_code = 14 + (int)br_show(b, 4);
                        br_skip(b, 4);
                    }
                } else {
                    level_code = 30 + (int)br_show(b, prefix - 3);
                    br_skip(b, prefix - 3);
                    if (prefix >= 16)
                        level_code += (1 << (prefix - 3)) - 4096;
                }
                level_code += 2;
                suffix = 2;
            } else {
                if (prefix < 15) {
                    level_code = (prefix << suffix) +
                                 (suffix ? (int)br_show(b, suffix) : 0);
                    if (suffix)
                        br_skip(b, suffix);
                } else {
                    level_code = (15 << suffix) + (int)br_show(b, prefix - 3);
                    br_skip(b, prefix - 3);
                    if (prefix >= 16)
                        level_code += (1 << (prefix - 3)) - 4096;
                }
            }
            if (getenv("CAVLC_TRACE"))
                printf("  [trace]   lvl[%d] code %d prefix %d suffix %d bits %d..%d\n",
                       i, level_code, prefix, suffix, bit_at, b->bit);
            {
                int mask = -(level_code & 1);
                lvl[i] = (((2 + level_code) >> 1) ^ mask) - mask;
            }
            /* the first level pins suffixLength at 2, no magnitude update */
            if (!is_first)
                suffix += (suffix_limit[suffix] + (unsigned)lvl[i] >
                           2U * (unsigned)suffix_limit[suffix]);
            }
        }
    }

    zeros_total = 0;
    if (total < max_coeff) {
        int r;
        if (is_cdc) {
            r = vlc_lookup(b, aeg_chroma_dc_total_zeros_len[total - 1],
                           aeg_chroma_dc_total_zeros_bits[total - 1],
                           max_coeff - total + 1, NULL);
        } else {
            r = vlc_lookup(b, aeg_total_zeros_len[total - 1],
                           aeg_total_zeros_bits[total - 1],
                           max_coeff - total + 1, NULL);
        }
        if (r < 0) {
            printf("%s: unparsable total_zeros (total %d)\n", tag, total);
            return -1;
        }
        zeros_total = r;
    }

    zeros_left = zeros_total;
    pos[0] = zeros_total + total - 1;
    for (i = 1; i < total; i++) {
        int run = 0;
        if (zeros_left > 0) {
            int z = zeros_left > 7 ? 7 : zeros_left;
            int nsym = zeros_left >= 7 ? 16 : 7;
            int r = vlc_lookup(b, aeg_run_before_len[z - 1],
                               aeg_run_before_bits[z - 1], nsym, NULL);
            if (r < 0) {
                int q;
                printf("%s: unparsable run_before (zeros_left %d, z %d) at bit %d, next 16:",
                       tag, zeros_left, z, b->bit);
                for (q = 0; q < 16; q++) { printf(" %u", br_show(b, 1)); br_skip(b, 1); }
                printf("\n");
                return -1;
            }
            run = r;
            zeros_left -= run;
        }
        pos[i] = pos[i - 1] - run - 1;
    }

    for (i = 0; i < total; i++) {
        if (pos[i] < 0 || pos[i] >= nscan) {
            printf("%s: coefficient %d out of range (%d)\n", tag, i, pos[i]);
            return -1;
        }
        out[scan[pos[i]]] = (int16_t)lvl[i];
    }
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
    static const uint8_t dct[16] = { 0,4,1,2,5,8,12,9,6,3,7,10,13,14,11,15 };
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
                int at;
                switch (kind) {
                case 1: at = dct[idx]; break;             /* luma DC: 4x4 scan */
                case 2: at = idx; break;                  /* chroma DC: raster */
                case 3: at = dct[idx + 1]; break;
                default: at = dct[idx]; break;
                }
                coef[at] = v;
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
