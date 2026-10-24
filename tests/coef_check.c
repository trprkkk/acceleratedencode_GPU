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
 *
 * The last part checks the *batched* entry points the device backends
 * implement (tq_batch, iq_idct_batch, and the residual form of the inverse
 * transform they share) against the same reference.  Those were the ones with
 * no such test until now, and both of them were wrong: the reference C
 * version of iq_idct_batch carried its own copy of the inverse transform that
 * skipped the int16 truncation, the DC rounding and the truncating second
 * pass, and the device kernel was checked against that copy -- so the two
 * agreed with each other and disagreed with the decoder.  Anything that
 * reconstructs a coefficient block gets compared here.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "h264.h"
#include "backend.h"

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

/* The reference decoder's inverse transform, as a signed residual. */
static void idct_ref_res(const int16_t *level, int qp, int32_t res[16])
{
    int32_t d[16], t[16];
    int i, x, y, sh = qp / 6;

    for (y = 0; y < 4; y++)
        for (x = 0; x < 4; x++) {
            const int cls = (x & 1) + (y & 1);
            const int v = level[x * 4 + y];
            d[y * 4 + x] = (int16_t)((int32_t)v * aeg_dequant_ls[qp % 6][cls] << sh);
        }
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
        res[0 * 4 + i] = (s0 + s3) >> 6;
        res[1 * 4 + i] = (s1 + s2) >> 6;
        res[2 * 4 + i] = (s1 - s2) >> 6;
        res[3 * 4 + i] = (s0 - s3) >> 6;
    }
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
    /* ---- the batched entry points the device backends implement ---- */
    {
        aeg_be_ops cpu;
        int16_t resb[16 * 8], levb[16 * 8], leva[16 * 8];
        int16_t resa[16 * 8];
        int32_t nz[8];
        int q;

        aeg_be_cpu_ops(&cpu);
        for (q = 0; q < rounds / 4 && bad < 5; q++) {
            int32_t rres[16];
            int b, i2;
            for (i2 = 0; i2 < 16 * 8; i2++) {
                /* every amplitude class, including the ones that leave int16
                 * once dequantised (bug 8) and the -32768 corner */
                switch (q % 6) {
                case 0: resb[i2] = (int16_t)(rand() % 511 - 255); break;
                case 1: resb[i2] = (int16_t)(rand() % 9 - 4); break;
                case 2: resb[i2] = (int16_t)((rand() % 2) ? 0 : rand() % 8191 - 4096); break;
                case 3: resb[i2] = (int16_t)(q % 12 ? 0 : rand() % 65536 - 32768); break;
                case 4: resb[i2] = (int16_t)(rand() % 33); break;
                default: resb[i2] = (int16_t)(rand() % 2 ? -1 : 0); break;
                }
            }
            qp = q % 52;
            /* forward: transform + quantisation, and the non-zero counts */
            cpu.tq_batch(cpu.priv, resb, 8, qp, q % 3 == 0, levb, nz);
            for (b = 0; b < 8; b++) {
                int16_t coef[16];
                int n = 0;
                aeg_transform_4x4(resb + 16 * b, 4, coef);
                aeg_quant_4x4(coef, leva + 16 * b, qp, q % 3 == 0);
                for (i2 = 0; i2 < 16; i2++)
                    n += leva[16 * b + i2] != 0;
                cases++;
                if (memcmp(leva + 16 * b, levb + 16 * b, 32) || n != nz[b]) {
                    printf("   tq_batch: qp %d blk %d: per-block and batch "
                           "levels differ (nz %d vs %d)\n", qp, b, n, nz[b]);
                    bad++;
                    break;
                }
            }
            /* inverse: the residual, then the same through the batch op */
            cpu.iq_idct_batch(cpu.priv, levb, 8, qp, resa);
            for (b = 0; b < 8; b++) {
                int16_t r2[16];
                aeg_idct_residual(levb + 16 * b, qp, r2);
                idct_ref_res(levb + 16 * b, qp, rres);
                cases += 2;
                for (i2 = 0; i2 < 16; i2++) {
                    if (resa[16 * b + i2] != (int16_t)rres[i2]) {
                        printf("   iq_idct_batch: qp %d blk %d coef %d: batch %d, "
                               "reference %d\n", qp, b, i2,
                               resa[16 * b + i2], (int)rres[i2]);
                        bad++;
                        break;
                    }
                    if (r2[i2] != (int16_t)rres[i2]) {
                        printf("   aeg_idct_residual: qp %d blk %d coef %d: "
                               "%d vs reference %d\n", qp, b, i2,
                               r2[i2], (int)rres[i2]);
                        bad++;
                        break;
                    }
                }
                /* and the residual has to be what the add path adds */
                {
                    uint8_t mid[16], out[16];
                    int16_t r3[16];
                    int x2, clipped = 0;
                    memset(mid, 128, sizeof(mid));
                    aeg_idct_residual(levb + 16 * b, qp, r3);
                    aeg_idct_add_block(out, 4, levb + 16 * b, qp, mid, 4, NULL);
                    for (x2 = 0; x2 < 16; x2++)
                        if (out[x2] == 0 || out[x2] == 255)
                            clipped = 1;
                    if (!clipped) {
                        cases++;
                        for (x2 = 0; x2 < 16; x2++)
                            if ((int)out[x2] - 128 != r3[x2]) {
                                printf("   idct residual: qp %d blk %d coef %d: "
                                       "add-path %d, residual %d\n", qp, b, x2,
                                       (int)out[x2] - 128, r3[x2]);
                                bad++;
                                break;
                            }
                    }
                }
            }
        }
    }
    printf("   coef_check: %d cases, %s\n", cases,
           bad ? "MISMATCH" : "scalar, vector and batched paths bit-exact");
    return bad != 0;
}
