/*
 * Cross-check of the device kernels against the CPU reference.
 *
 * The GPU kernels are a string (src/kernels.h) compiled by the OpenCL runtime
 * or by NVRTC, so nothing in the normal build ever compiles them.  Here the
 * very same text is included as C -- with the OpenCL qualifiers defined away
 * and `get_global_id()` driven by a counter -- and each kernel is called
 * exactly the way its host wrapper calls it, on the same inputs.  The results
 * must match the CPU implementations in src/backend.c and src/transform.c
 * bit for bit: the backends' contract is that moving a stage to the GPU cannot
 * change the bitstream, only its speed.
 *
 * This is what catches a kernel drifting away from the reference (for
 * example a quantiser rounding offset that is updated in one path and not in
 * the other) on a machine with no GPU at all.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "backend.h"

typedef unsigned char uchar;
typedef unsigned long long ulong;
#define __kernel
#define __global
static int aeg_mock_gid;
#define get_global_id(d) (aeg_mock_gid)
#include "kernels_gen.c"

static uint32_t rng_state = 12345u;

static int rnd(int n)
{
    rng_state = rng_state * 1103515245u + 12345u;
    return (int)((rng_state >> 16) % (uint32_t)n);
}

/* Run a kernel once per work item, the way an NDRange launch would. */
static void run_items(int count, void (*body)(int))
{
    int i;
    for (i = 0; i < count; i++) {
        aeg_mock_gid = i;
        body(i);
    }
}

/* ------------------------------------------------------------------ */
/* tq_batch: forward transform + quantisation                          */
/* ------------------------------------------------------------------ */
static const int16_t *g_res;
static int g_nblk, g_qbits, g_f, g_mf0, g_mf1, g_mf2;
static int16_t *g_level;
static int *g_nz;

static void call_tq(int i)
{
    (void)i;
    tq_batch(g_res, g_nblk, g_qbits, g_f, g_mf0, g_mf1, g_mf2, g_level, g_nz);
}

/* ------------------------------------------------------------------ */
/* iq_idct_batch                                                       */
/* ------------------------------------------------------------------ */
static const int16_t *g_qlevel;
static int g_sh, g_ls0, g_ls1, g_ls2;
static int16_t *g_iqres;

static void call_iq(int i)
{
    (void)i;
    iq_idct_batch(g_qlevel, g_nblk, g_sh, g_ls0, g_ls1, g_ls2, g_iqres);
}

/* ------------------------------------------------------------------ */
/* sad_batch                                                           */
/* ------------------------------------------------------------------ */
static const uchar *g_cur, *g_ref;
static int g_cstride, g_rstride, g_bw, g_bh, g_count;
static const short *g_mv;
static int *g_sad;

static void call_sad(int i)
{
    (void)i;
    sad_batch(g_cur, g_cstride, g_ref, g_rstride, g_bw, g_bh, g_count, g_mv,
              g_sad);
}

/* ------------------------------------------------------------------ */
/* sse_batch                                                           */
/* ------------------------------------------------------------------ */
static const uchar *g_a, *g_b;
static int g_sa, g_sb, g_tw, g_th, g_ssecount;
static ulong *g_sse;

static void call_sse(int i)
{
    (void)i;
    sse_batch(g_a, g_sa, g_b, g_sb, g_tw, g_th, g_ssecount, g_sse);
}

int main(void)
{
    aeg_be_ops cpu;
    int fail = 0;
    int t, qp, intra;

    aeg_be_cpu_ops(&cpu);

    /* ---- transform + quantisation, both intra and inter, several QPs --- */
    for (t = 0; t < 2; t++) {
        for (intra = 0; intra <= 1; intra++) {
            qp = 10 + rnd(40);
            g_nblk = 1 + rnd(24);
            g_qbits = 15 + qp / 6;
            g_f = aeg_quant_bias(g_qbits, intra ? 3 : 6);
            g_mf0 = aeg_quant_mf[qp % 6][0];
            g_mf1 = aeg_quant_mf[qp % 6][1];
            g_mf2 = aeg_quant_mf[qp % 6][2];
            g_res = (const int16_t *)malloc((size_t)g_nblk * 16 * 2);
            g_level = (int16_t *)malloc((size_t)g_nblk * 16 * 2);
            g_nz = (int *)malloc((size_t)g_nblk * 4);
            {
                int16_t *ref_level = (int16_t *)malloc((size_t)g_nblk * 16 * 2);
                int *ref_nz = (int *)malloc((size_t)g_nblk * 4);
                int i, bad = 0;
                for (i = 0; i < g_nblk * 16; i++)
                    ((int16_t *)g_res)[i] = (int16_t)(rnd(511) - 255);
                run_items(g_nblk, call_tq);
                cpu.tq_batch(cpu.priv, g_res, g_nblk, qp, intra, ref_level,
                             ref_nz);
                for (i = 0; i < g_nblk * 16; i++)
                    if (g_level[i] != ref_level[i])
                        bad++;
                for (i = 0; i < g_nblk; i++)
                    if (g_nz[i] != ref_nz[i])
                        bad++;
                if (bad) {
                    printf("   kernel_check: tq_batch qp %d intra %d %d of %d "
                           "levels differ\n", qp, intra, bad, g_nblk * 16);
                    fail = 1;
                }
                /* ---- inverse quantisation + inverse transform ---------- */
                g_sh = qp / 6;
                g_ls0 = aeg_dequant_ls[qp % 6][0];
                g_ls1 = aeg_dequant_ls[qp % 6][1];
                g_ls2 = aeg_dequant_ls[qp % 6][2];
                g_qlevel = g_level;
                g_iqres = (int16_t *)malloc((size_t)g_nblk * 16 * 2);
                run_items(g_nblk, call_iq);
                {
                    int16_t *ref_res = (int16_t *)malloc((size_t)g_nblk * 16 * 2);
                    cpu.iq_idct_batch(cpu.priv, g_qlevel, g_nblk, qp, ref_res);
                    bad = 0;
                    for (i = 0; i < g_nblk * 16; i++)
                        if (g_iqres[i] != ref_res[i])
                            bad++;
                    if (bad) {
                        printf("   kernel_check: iq_idct_batch qp %d %d of %d "
                               "samples differ\n", qp, bad, g_nblk * 16);
                        fail = 1;
                    }
                    free(ref_res);
                }
                free(ref_level);
                free(ref_nz);
            }
            free((void *)g_res);
            free(g_level);
            free(g_nz);
            free(g_iqres);
        }
        (void)t;
    }

    /* ---- SAD over a batch of 16x16 blocks ----------------------------- */
    {
        int bw = 16, bh = 16, count = 8, i, bad = 0;
        g_bw = bw;
        g_bh = bh;
        g_count = count;
        g_cstride = bw;
        g_rstride = 64;
        g_cur = (const uchar *)malloc((size_t)bw * bh);
        g_ref = (const uchar *)malloc((size_t)g_rstride * (bh + 32));
        g_mv = (short *)malloc((size_t)count * 2 * sizeof(short));
        g_sad = (int *)malloc((size_t)count * 4);
        {
            int *ref_sad = (int *)malloc((size_t)count * 4);
            int j;
            for (j = 0; j < bw * bh; j++)
                ((uchar *)g_cur)[j] = (uchar)rnd(256);
            for (j = 0; j < g_rstride * (bh + 32); j++)
                ((uchar *)g_ref)[j] = (uchar)rnd(256);
            for (j = 0; j < count; j++) {
                ((short *)g_mv)[2 * j + 0] = (short)rnd(24);
                ((short *)g_mv)[2 * j + 1] = (short)rnd(24);
            }
            run_items(count, call_sad);
            cpu.sad_batch(cpu.priv, g_cur, g_cstride, g_ref, g_rstride, bw, bh,
                          (const int16_t *)g_mv, count, ref_sad);
            for (i = 0; i < count; i++)
                if (g_sad[i] != ref_sad[i])
                    bad++;
            if (bad) {
                printf("   kernel_check: sad_batch %d of %d entries differ\n",
                       bad, count);
                fail = 1;
            }
            free(ref_sad);
        }
        free((void *)g_cur);
        free((void *)g_ref);
        free((void *)g_mv);
        free(g_sad);
    }

    /* ---- SSE over a rectangle ---------------------------------------- */
    {
        int w = 64, h = 32, count, i, bad = 0;
        uint64_t *ref_sse;
        g_sa = w;
        g_sb = w;
        g_tw = w / 8;
        g_th = h / 8;
        g_ssecount = g_tw * g_th;
        count = g_ssecount;
        g_a = (const uchar *)malloc((size_t)g_sa * h);
        g_b = (const uchar *)malloc((size_t)g_sb * h);
        g_sse = (ulong *)malloc((size_t)count * sizeof(ulong));
        ref_sse = (uint64_t *)calloc((size_t)count, sizeof(uint64_t));
        for (i = 0; i < g_sa * h; i++) {
            ((uchar *)g_a)[i] = (uchar)rnd(256);
            ((uchar *)g_b)[i] = (uchar)rnd(256);
        }
        run_items(count, call_sse);
        cpu.sse_batch(cpu.priv, g_a, g_sa, g_b, g_sb, w, h, ref_sse);
        {
            uint64_t total = 0, ref_total = 0;
            for (i = 0; i < count; i++) {
                total += (uint64_t)g_sse[i];
                ref_total += ref_sse[i];
            }
            if (total != ref_total) {
                printf("   kernel_check: sse_batch %llu != %llu\n",
                       (unsigned long long)total, (unsigned long long)ref_total);
                bad = 1;
            }
            if (bad)
                fail = 1;
        }
        free((void *)g_a);
        free((void *)g_b);
        free(g_sse);
        free(ref_sse);
    }

    if (!fail)
        printf("   kernel_check: 4 kernels agree with the CPU reference\n");
    return fail;
}
