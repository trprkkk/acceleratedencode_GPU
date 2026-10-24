/*
 * CAVLC entropy coding of residual blocks (H.264 9.2).
 *
 * The entropy coder is inherently serial: every symbol depends on the
 * neighbouring blocks' "total number of coefficients" counters which in turn
 * depend on previously coded symbols.  It therefore always runs on the CPU --
 * a GPU kernel would need one launch per macroblock, which costs far more
 * than a few dozen table lookups.  The dispatcher models this cost
 * explicitly rather than pretending otherwise.
 */
#include <stdio.h>
#include <stdlib.h>
#include "h264.h"

static const uint8_t scan_dc[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
/* 4x4 CAVLC coefficient scan: the transposed zig-zag.  The decoder builds it
 * with TRANSPOSE(zigzag), i.e. it walks the block column-major, so the scan
 * index of a position is the transpose of the usual zig-zag position. */
static const uint8_t scan_dcT[16] = {0,4,1,2,5,8,12,9,6,3,7,10,13,14,11,15};

static inline int nc_bucket(int nC)
{
    return nC < 2 ? 0 : (nC < 4 ? 1 : (nC < 8 ? 2 : 3));
}

/* ------------------------------------------------------------------ */
/* Core block writer (or cost counter when bw->count_only)             */
/* ------------------------------------------------------------------ */
static void write_block(aeg_bitwriter *bw, const int16_t *coef,
                        const uint8_t *scan, int nscan, int nC,
                        int max_coeff, int is_chroma_dc)
{
    int lvl[16], pos[16];
    int total = 0, i, trailing = 0, zeros_total, zeros_left, suffix;

    for (i = nscan - 1; i >= 0; i--) {
        int v = coef[scan[i]];
        if (v) {
            lvl[total] = v;
            pos[total] = i;
            total++;
        }
    }

    if (total == 0) {
        if (is_chroma_dc)
            aeg_bw_put(bw, aeg_chroma_dc_coeff_token_len[0],
                       aeg_chroma_dc_coeff_token_bits[0]);
        else {
            int b = nc_bucket(nC);
            aeg_bw_put(bw, aeg_coeff_token_len[b][0],
                       aeg_coeff_token_bits[b][0]);
        }
        return;
    }

    for (i = 0; i < total && i < 3; i++) {
        if (lvl[i] == 1 || lvl[i] == -1) trailing++;
        else break;
    }

    if (is_chroma_dc) {
        /* The chroma-DC table is laid out [total_coeff][trailing] with the
         * "empty block" code in slot 0, exactly like the 4x4 tables. */
        int idx = total * 4 + trailing;
        aeg_bw_put(bw, aeg_chroma_dc_coeff_token_len[idx],
                   aeg_chroma_dc_coeff_token_bits[idx]);
    } else {
        int b = nc_bucket(nC);
        int idx = total * 4 + trailing;
        aeg_bw_put(bw, aeg_coeff_token_len[b][idx], aeg_coeff_token_bits[b][idx]);
    }

    for (i = 0; i < trailing; i++)
        aeg_bw_put(bw, 1, lvl[i] < 0 ? 1u : 0u);

    suffix = (total > 10 && trailing < 3) ? 1 : 0;
    for (i = trailing; i < total; i++) {
        /* One level, exactly as H.264 9.2.2 defines it: the decoder rebuilds
         * levelCode and then derives the magnitude/sign from it, so the
         * encoder has to reproduce that mapping bit for bit. */
        int level = lvl[i];
        int neg = level < 0;
        int abs_level = neg ? -level : level;
        /* The first non-trailing coefficient has its own code layout and its
         * escape origin is 30 instead of 15 << suffixLength; the -2 bias only
         * applies when the block has fewer than three trailing ones (the
         * decoder adds 2 back in exactly that case). */
        int first = (i == trailing);
        int bias = first && trailing < 3;
        int level_code = (abs_level << 1) - 2 + (neg ? 1 : 0);
        int prefix, rest;

        if (bias && level_code >= 2)
            level_code -= 2;                 /* decoder adds 2 back */

        prefix = level_code >> suffix;
        if (first) {
            /* First non-trailing level (H.264 tables 9-7/9-8).  With
             * suffixLength 0 the codes are: levelCode < 14 -> levelCode zeroes
             * then a one; 14 <= levelCode < 30 -> prefix 14 plus a 4-bit
             * remainder; otherwise the escape whose origin is 30. */
            if (suffix == 0 && level_code < 14) {
                aeg_bw_put(bw, level_code + 1, 1u);
                goto level_done;
            }
            if (suffix == 0 && level_code < 30) {
                aeg_bw_put(bw, 15, 1u);              /* prefix 14 */
                aeg_bw_put(bw, 4, (uint32_t)(level_code - 14));
                goto level_done;
            }
            if (prefix < 15) {
                aeg_bw_put(bw, prefix + 1 + suffix,
                           (uint32_t)((1u << suffix) |
                                      (level_code & ((1 << suffix) - 1))));
                goto level_done;
            }
            rest = level_code - 30;
        } else {
            if (prefix < 15) {
                aeg_bw_put(bw, prefix + 1 + suffix,
                           (uint32_t)((1u << suffix) |
                                      (level_code & ((1 << suffix) - 1))));
                goto level_done;
            }
            rest = level_code - (15 << suffix);
        }
        {
            int p = 15;
            while (rest >= (1 << (p - 3))) {
                rest -= 1 << (p - 3);
                p++;
            }
            aeg_bw_put(bw, p + 1, 1u);       /* p zeroes then a one */
            aeg_bw_put(bw, p - 3, (uint32_t)(rest & ((1 << (p - 3)) - 1)));
        }
level_done:
        /* H.264 level suffixLength update.  The two steps are independent:
         * a zero suffix becomes 1 first, and only then does the magnitude
         * test (against 3 << (suffixLength-1)) get a chance to bump it again,
         * so a first level with |level| in 4..15 ends at suffixLength 2. */
        if (suffix == 0)
            suffix = 1;
        if (suffix < 6 && abs_level > (3 << (suffix - 1)))
            suffix++;
    }

    zeros_total = pos[0] - (total - 1);
    if (total < max_coeff) {
        if (is_chroma_dc)
            aeg_bw_put(bw, aeg_chroma_dc_total_zeros_len[total - 1][zeros_total],
                       aeg_chroma_dc_total_zeros_bits[total - 1][zeros_total]);
        else
            aeg_bw_put(bw, aeg_total_zeros_len[total - 1][zeros_total],
                       aeg_total_zeros_bits[total - 1][zeros_total]);
    }

    zeros_left = zeros_total;
    for (i = 1; i < total; i++) {
        int run;
        if (zeros_left <= 0)
            break;
        run = pos[i - 1] - pos[i] - 1;   /* pos[] runs from the highest scan
                                         * position down to the lowest */
        {
            int z = zeros_left > 7 ? 7 : zeros_left;
            aeg_bw_put(bw, aeg_run_before_len[z - 1][run],
                       aeg_run_before_bits[z - 1][run]);
        }
        zeros_left -= run;
    }
}

/* ------------------------------------------------------------------ */
/* Public interface                                                    */
/* ------------------------------------------------------------------ */
/* kind: 0 = normal 4x4 (16 coeffs), 1 = luma DC (16), 2 = chroma DC (4),
 *       3 = Intra_16x16 AC (15) */
void aeg_cavlc_write_block(aeg_bitwriter *bw, const int16_t *coef, int nC,
                           int kind)
{
    static uint8_t ac_scan[15];
    int i;
    switch (kind) {
    case 1:
        write_block(bw, coef, scan_dcT, 16, nC, 16, 0);
        break;
    case 2:
        write_block(bw, coef, scan_dc, 4, nC, 4, 1);
        break;
    case 3:
        for (i = 0; i < 15; i++)
            ac_scan[i] = scan_dcT[i + 1];
        write_block(bw, coef, ac_scan, 15, nC, 15, 0);
        break;
    default:
        write_block(bw, coef, scan_dcT, 16, nC, 16, 0);
        break;
    }
}

/* CAVLC residual of one 8x8 luma block (High profile, H.264 7.3.5.3.1.2).
 *
 * The sixty-four coefficients are transmitted as four 4x4 lists, each coded
 * with the ordinary 4x4 syntax -- its own coeff_token, so its own nC, and
 * maxNumCoeff = 16 -- but the lists are not the four quadrants of the block:
 * they interleave the 8x8 scan, list k taking scan positions k, k + 4, k + 8,
 * ...  That keeps the run-length coding of the 4x4 syntax usable on an 8x8
 * block, because every list then holds sixteen coefficients spread evenly over
 * the frequency range.
 *
 * The four lists are written one at a time, in transmission order, because the
 * context of list k depends on the number of coefficients list k-1 turned out
 * to have: the caller therefore reads each list's count back (aeg_cavlc_nz8)
 * and passes it to the neighbouring lists as they come. */
void aeg_cavlc_write_block8(aeg_bitwriter *bw, const int16_t *coef, int sub,
                            int nC)
{
    int16_t list[16];
    int n;
    for (n = 0; n < 16; n++)
        list[n] = coef[aeg_zigzag8x8[sub + 4 * n]];
    if (getenv("AEGDBG")) {
        fprintf(stderr, "ENC sub %d nC %d:", sub, nC);
        for (n = 15; n >= 0; n--)
            if (list[n])
                fprintf(stderr, " [%d]=%d", n, list[n]);
        fprintf(stderr, "\n");
    }
    if (getenv("AEGDBG"))
        fprintf(stderr, "ENC sub %d bits before %d\n", sub,
                (int)(bw->len * 8 + bw->bits));
    write_block(bw, list, scan_dc, 16, nC, 16, 0);
    if (getenv("AEGDBG"))
        fprintf(stderr, "ENC sub %d bits after %d\n", sub,
                (int)(bw->len * 8 + bw->bits));
}

/* Number of non-zero coefficients of one of the four lists (its
 * TotalCoeff), which is what the neighbouring lists use as their context. */
int aeg_cavlc_nz8(const int16_t *coef, int sub)
{
    int n = 0, i;
    for (i = 0; i < 16; i++)
        n += coef[aeg_zigzag8x8[sub + 4 * i]] != 0;
    return n;
}

int aeg_cavlc_nc(int nz_left, int nz_top, int have_left, int have_top)
{
    if (have_left && have_top)
        return (nz_left + nz_top + 1) >> 1;
    if (have_left)
        return nz_left;
    if (have_top)
        return nz_top;
    return 0;
}

/* Number of bits the residual blocks of a macroblock will occupy.
 * Used by the rate-distortion decision; identical arithmetic to the
 * writer above, but nothing is emitted. */
int aeg_cavlc_block_cost(const int16_t *coef, int nC, int kind)
{
    aeg_bitwriter bw;
    aeg_bw_init(&bw, NULL, 0, 0);
    aeg_cavlc_write_block(&bw, coef, nC, kind);
    return (int)(bw.len * 8 + bw.bits);
}

/* Same measurement for one list of an 8x8 block (aeg_cavlc_write_block8). */
int aeg_cavlc_block_cost8(const int16_t *coef, int sub, int nC)
{
    aeg_bitwriter bw;
    aeg_bw_init(&bw, NULL, 0, 0);
    aeg_cavlc_write_block8(&bw, coef, sub, nC);
    return (int)(bw.len * 8 + bw.bits);
}
