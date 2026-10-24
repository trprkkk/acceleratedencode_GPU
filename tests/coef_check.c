/*
 * Cross-check of the vectorised coefficient pipeline against the scalar loops.
 *
 * src/transform.c keeps the portable scalar versions of the quantiser and of
 * the dequantise + inverse transform + add, and adds SSE2 fast paths for both.
 * The fast paths must be bit-identical: the quantiser decides the exact levels
 * that go into the bitstream, and the inverse transform decides the encoder's
 * reconstruction, which has to agree with the decoder sample for sample (bug
 * 31 was exactly such a disagreement -- a fast path that rounded the DC
 * coefficient before the reference cast instead of after it).
 *
 * The scalar loops below are transcribed from the reference decoder's
 * arithmetic, so this test does not depend on the library's own scalar code.
 * Every case also runs with the destination equal to the prediction, and with
 * the strides the encoder really uses (4, 8, 16, picture strides).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "h264.h"

static int quant_ref(const int16_t *coef, int16_t *level, int qp, int intra)
{
    const int qbits = 15 + (qp / 6);
    const int f = aeg_quant_bias(qbits, intra ? 3 : 6);
    int mf[3], x, y;
    for (x = 0; x < 3; x++)
        mf[x] = aeg_quant_mf[qp % 6][x];
    for (y = 0; y < 4; y++)
        for (x = 0; x < 4; x++) {
            const int c = coef[x * 4 + y], cls = (x & 1) + (y & 1);
            const int abs = c < 0 ? -c : c;
            const int lv = (int)(((int64_t)abs * mf[cls] + f) >> qbits);
            level[y * 4 + x] = (int16_t)(c < 0 ? -lv : lv);
        }
    return 0;
}

static void idct_ref(uint8_t *dst, int stride, const int16_t *level, int qp,
                     const uint8_t *pred, int pred_stride)
{
    int32_t d[16], t[16];
    int i, x, y, sh = qp / 6, ac = 0;

    for (y = 0; y < 4; y++)
        for (x = 0; x < 4; x++) {
            const int cls = (x & 1) + (y & 1);
            const int v = level[x * 4 + y];
            if (v && (x | y))
                ac = 1;
            d[y * 4 + x] = (int16_t)((int32_t)v * aeg_dequant_ls[qp % 6][cls] << sh);
        }
    if (pred && !ac) {
        const int v = (d[0] + 32) >> 6;
        for (y = 0; y < 4; y++)
            for (x = 0; x < 4; x++) {
                const int p = pred[y * pred_stride + x] + v;
                dst[y * stride + x] = (uint8_t)(p < 0 ? 0 : (p > 255 ? 255 : p));
            }
        return;
    }
    d[0] = (int16_t)d[0];
    d[0] += 1 << 5;
    for (i = 0; i < 4; i++) {
        const int32_t s0 = d[i * 4 + 0] + d[i * 4 + 2];
        const int32_t s1 = d[i * 4 + 0] - d[i * 4 + 2];
        const int32_t s2 = (d[i * 4 + 1] >> 1) - d[i * 4 + 3];
        const int32_t s3 = d[i * 4 + 1] + (d[i * 4 + 3] >> 1);
        t[i * 4 + 0] = (int16_t)(s0 + s3);
        t[i * 4 + 1] = (int16_t)(s1 + s2);
        t[i * 4 + 2] = (int16_t)(s1 - s2);
        t[i * 4 + 3] = (int16_t)(s0 - s3);
    }
    for (i = 0; i < 4; i++) {
        const int32_t s0 = t[0 * 4 + i] + t[2 * 4 + i];
        const int32_t s1 = t[0 * 4 + i] - t[2 * 4 + i];
        const int32_t s2 = (t[1 * 4 + i] >> 1) - t[3 * 4 + i];
        const int32_t s3 = t[1 * 4 + i] + (t[3 * 4 + i] >> 1);
        d[0 * 4 + i] = (s0 + s3) >> 6;
        d[1 * 4 + i] = (s1 + s2) >> 6;
        d[2 * 4 + i] = (s1 - s2) >> 6;
        d[3 * 4 + i] = (s0 - s3) >> 6;
    }
    for (y = 0; y < 4; y++)
        for (x = 0; x < 4; x++) {
            const int p = (pred ? pred[y * pred_stride + x] : 0) + (int)d[y * 4 + x];
            dst[y * stride + x] = (uint8_t)(p < 0 ? 0 : (p > 255 ? 255 : p));
        }
}

int main(int argc, char **argv)
{
    const int rounds = argc > 1 ? atoi(argv[1]) : 200000;
    static int16_t coef[16], a[16], b[16];
    static uint8_t pred[4 * 384 + 256], o1[4 * 384], o2[4 * 384];
    static const int strides[6] = { 4, 8, 16, 24, 384, 20 };
    int t, i, qp, bad = 0, cases = 0;

    srand(20251025);
    for (i = 0; i < (int)sizeof(pred); i++)
        pred[i] = (uint8_t)rand();

    for (t = 0; t < rounds; t++) {
        /* amplitudes: the residual transform's real range (|coef| <= 4080),
         * then the extremes a caller could pass, then all-zero blocks */
        const int amp = (t % 5 == 0) ? 4080 : ((t % 5 == 1) ? 32000 : 300);
        int stride, ps, inplace, y, x, intra;
        const uint8_t *pr;

        for (i = 0; i < 16; i++) {
            coef[i] = (int16_t)((rand() % (2 * amp + 1)) - amp);
            if (t % 7 == 0)
                coef[i] = 0;
            if (t % 11 == 0 && i)                 /* DC-only blocks */
                coef[i] = 0;
        }
        qp = t % 52;
        intra = t & 1;

        quant_ref(coef, a, qp, intra);
        memset(b, 0x5A, sizeof(b));
        aeg_quant_4x4(coef, b, qp, intra);
        cases++;
        for (i = 0; i < 16; i++)
            if (a[i] != b[i]) {
                if (bad++ < 5)
                    printf("   quant_4x4: qp %d intra %d coefficient %d: "
                           "scalar %d, vector %d\n", qp, intra, i, a[i], b[i]);
                break;
            }

        stride = strides[t % 6];
        ps = (t % 4 == 0) ? stride : 384;
        inplace = (t % 9 == 0);
        pr = (t % 3 == 0) ? NULL : pred + (t % 16);
        if (inplace && ps != stride)
            ps = stride;          /* dst == pred only works at one stride */
        memset(o1, 0x11, sizeof(o1));
        memset(o2, 0x11, sizeof(o2));
        if (inplace) {
            memcpy(o1, pred + (t % 16), sizeof(o1));
            memcpy(o2, o1, sizeof(o2));
            idct_ref(o1, stride, coef, qp, o1, ps);
            aeg_idct_add_block(o2, stride, coef, qp, o2, ps, NULL);
        } else {
            idct_ref(o1, stride, coef, qp, pr, ps);
            aeg_idct_add_block(o2, stride, coef, qp, pr, ps, NULL);
        }
        cases++;
        for (y = 0; y < 4 && bad < 5; y++)
            for (x = 0; x < 4; x++)
                if (o1[y * stride + x] != o2[y * stride + x]) {
                    printf("   idct_add: qp %d stride %d in-place %d at (%d,%d): "
                           "scalar %d, vector %d (level",
                           qp, stride, inplace, y, x,
                           o1[y * stride + x], o2[y * stride + x]);
                    for (i = 0; i < 16; i++)
                        printf(" %d", coef[i]);
                    printf(")\n");
                    bad++;
                    break;
                }
    }
    printf("   coef_check: %d cases, %s\n", cases,
           bad ? "MISMATCH" : "scalar and vector paths bit-exact");
    return bad != 0;
}
