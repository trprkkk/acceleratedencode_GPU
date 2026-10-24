/*
 * synth_probe - writes a hand-built 16x16 H.264 stream containing a single
 * Intra_16x16 macroblock whose only non-zero coefficients are one luma DC
 * level and one chroma DC level.
 *
 * Decoding that stream with an independent decoder (ffmpeg) and comparing the
 * reconstructed samples against the level values pins down the exact
 * normalisation of the DC prediction/transform paths, which is otherwise
 * easy to get subtly wrong.  This is a development/verification tool, not
 * part of the encoder.
 *
 * usage: synth_probe <luma_dc_level> <chroma_dc_level> <qp> out.264
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "h264.h"

int main(int argc, char **argv)
{
    int luma_lvl = argc > 1 ? atoi(argv[1]) : 100;
    int chroma_lvl = argc > 2 ? atoi(argv[2]) : 50;
    int qp = argc > 3 ? atoi(argv[3]) : 26;
    const char *path = argc > 4 ? argv[4] : "synth.264";
    int mb_type = argc > 5 ? atoi(argv[5]) : 19;
    int chroma_mode = argc > 6 ? atoi(argv[6]) : 0;
    int qp_delta = argc > 7 ? atoi(argv[7]) : 0;
    int ac_lvl = argc > 8 ? atoi(argv[8]) : 0;
    aeg_params par;
    uint8_t sps[128], pps[128];
    size_t sps_len, pps_len;
    uint8_t rbsp[4096];
    uint8_t nal[8192];
    size_t rbsp_len;
    aeg_bitwriter bw;
    int16_t coef[16];
    FILE *f;
    int i;

    aeg_params_default(&par);
    par.width = 16;
    par.height = 16;
    par.qp = qp;

    sps_len = aeg_write_sps(sps, sizeof(sps), &par, 1, 1, 8, 30);
    pps_len = aeg_write_pps(pps, sizeof(pps));

    aeg_bw_init(&bw, rbsp, sizeof(rbsp), 0);
    aeg_write_slice_header(&bw, 2 /* I */, 0, 8, 1 /* idr */, 0, qp_delta, 0, 0, 0,
                             0 /* first_mb_in_slice */);

    /* I_16x16 DC prediction, cbp_chroma = 1, cbp_luma = 15 -> mb_type 19 */
    aeg_bw_ue(&bw, (uint32_t)mb_type);
    aeg_bw_ue(&bw, (uint32_t)chroma_mode);
    aeg_bw_se(&bw, 0);            /* mb_qp_delta                            */

    /* luma DC block: 16 coefficients, only the DC non-zero */
    for (i = 0; i < 16; i++)
        coef[i] = (i == 0) ? (int16_t)luma_lvl : 0;
    aeg_cavlc_write_block(&bw, coef, 0, 1);
    /* 16 luma AC blocks; block 0 optionally carries one AC level */
    for (i = 0; i < 16; i++) {
        memset(coef, 0, sizeof(coef));
        if (i == 0 && ac_lvl)
            coef[aeg_zigzag4x4[1]] = (int16_t)ac_lvl;
        aeg_cavlc_write_block(&bw, coef, 0, 3);
    }
    /* chroma DC blocks (cbp_chroma == 1: DC only) */
    for (i = 0; i < 2; i++) {
        memset(coef, 0, sizeof(coef));
        coef[0] = (int16_t)chroma_lvl;
        aeg_cavlc_write_block(&bw, coef, 0, 2);
    }
    rbsp_len = aeg_bw_finish(&bw);
#ifdef DUMP
    { size_t k; fprintf(stderr, "rbsp:"); for (k = 0; k < rbsp_len; k++) fprintf(stderr, " %02x", rbsp[k]); fprintf(stderr, "\n");
      for (k = 0; k < rbsp_len; k++) { int b; for (b = 7; b >= 0; b--) fputc((rbsp[k]>>b)&1 ? '1':'0', stderr); } fputc('\n', stderr); }
#endif

    f = fopen(path, "wb");
    if (!f) {
        perror(path);
        return 1;
    }
    {
        size_t n = aeg_nal_write(nal, sizeof(nal), 3, 7, sps, sps_len, 1);
        fwrite(nal, 1, n, f);
        n = aeg_nal_write(nal, sizeof(nal), 3, 8, pps, pps_len, 0);
        fwrite(nal, 1, n, f);
        n = aeg_nal_write(nal, sizeof(nal), 3, 5, rbsp, rbsp_len, 0);
        fwrite(nal, 1, n, f);
    }
    fclose(f);
    printf("wrote %s (luma_dc=%d chroma_dc=%d qp=%d mb_type=%d chroma=%d)\n",
           path, luma_lvl, chroma_lvl, qp, mb_type, chroma_mode);
    return 0;
}
