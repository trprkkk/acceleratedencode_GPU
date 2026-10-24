/*
 * Consistency check for the derived syntax tables.
 *
 * The coded_block_pattern<->codeNum pairing exists twice: as the decoder
 * table (codeNum -> pattern, aeg_cbp_code_num) and as the encoder table
 * (pattern -> codeNum, aeg_cbp_codenum).  A single wrong entry does not
 * desynchronise the bitstream -- the decoder happily reads *some* legal
 * pattern -- it just decodes a different set of 8x8 luma groups, so the
 * symptom is "the decoder applies residual in the wrong quadrant" rather than
 * a parse error.  That is exactly the bug this check exists to prevent.
 *
 * Exit status 0 means the two tables are exact inverses of each other.
 */
#include <stdio.h>
#include <stdlib.h>

#include "h264.h"

int main(void)
{
    int k, bad = 0;

    for (k = 0; k < 48; k++) {
        int cbp = aeg_cbp_code_num[k];
        int back = aeg_cbp_codenum[cbp];

        if (cbp > 47) {
            printf("aeg_cbp_code_num[%d] = %d is out of range\n", k, cbp);
            bad++;
        }
        if (back != k) {
            printf("cbp %2d (codeNum %2d) maps back to %d\n", cbp, k, back);
            bad++;
        }
    }
    /* Every pattern has to be produced by exactly one codeNum. */
    {
        unsigned seen[48] = { 0 };
        for (k = 0; k < 48; k++) {
            int cbp = aeg_cbp_code_num[k];
            if (cbp < 48)
                seen[cbp]++;
        }
        for (k = 0; k < 48; k++)
            if (seen[k] != 1) {
                printf("cbp %d is produced by %u codeNums\n", k, seen[k]);
                bad++;
            }
    }

    if (bad) {
        printf("syntax_check: %d problem(s)\n", bad);
        return 1;
    }
    printf("syntax_check OK: 48 coded_block_pattern entries are exact inverses\n");
    return 0;
}
