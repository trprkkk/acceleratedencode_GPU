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
#define AEG_DEVFN static
static int aeg_mock_gid;
#define get_global_id(d) (aeg_mock_gid)
/* lowres_search is the one kernel with a work-group: run as a group of one
 * item, it walks every diagonal itself, in order, and the barrier between
 * them has nothing to wait for */
#define get_local_id(d) 0
#define get_local_size(d) 1
#define get_group_id(d) (aeg_mock_gid)
#define CLK_GLOBAL_MEM_FENCE 0
#define barrier(f) ((void)0)
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
/* tq_iq_batch: both in one kernel                                     */
/* ------------------------------------------------------------------ */
static int16_t *g_tqrec;

static void call_tqiq(int i)
{
    (void)i;
    tq_iq_batch(g_res, g_nblk, g_qbits, g_f, g_mf0, g_mf1, g_mf2, g_sh, g_ls0,
                g_ls1, g_ls2, g_level, g_nz, g_tqrec);
}

/* ------------------------------------------------------------------ */
/* tq_dc_batch / iq_dc_batch: the DC-split pair (Intra_16x16, chroma)  */
/* ------------------------------------------------------------------ */
static int *g_dc;

static void call_tqdc(int i)
{
    (void)i;
    tq_dc_batch(g_res, g_nblk, g_qbits, g_f, g_mf0, g_mf1, g_mf2, g_level, g_nz,
                g_dc);
}

static void call_iqdc(int i)
{
    (void)i;
    iq_dc_batch(g_qlevel, g_dc, g_nblk, g_sh, g_ls0, g_ls1, g_ls2, g_iqres);
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
/* interp_plane                                                        */
/* ------------------------------------------------------------------ */
static const uchar *g_pref;
static int g_rstride, g_pw, g_ph, g_pwhich;
static uchar *g_pdst;
static int g_pdstride;

static void call_intp(int i)
{
    (void)i;
    /* the host passes the whole padded allocations (see the kernel) */
    interp_plane(g_pref - (size_t)AEG_PAD * g_rstride - AEG_PAD, g_rstride,
                 g_pw, g_ph, AEG_PAD, g_pwhich,
                 g_pdst - (size_t)AEG_PAD * g_pdstride - AEG_PAD, g_pdstride);
}

/* ------------------------------------------------------------------ */
/* satd_batch                                                          */
/* ------------------------------------------------------------------ */
static const uchar *g_scur, *g_s0, *g_s1, *g_s2, *g_s3;
static int g_scstride, g_srstride, g_spad, g_sbw, g_sbh, g_scount;
static const short *g_spos;
static int *g_satd;

static void call_satd(int i)
{
    (void)i;
    satd_batch(g_scur, g_scstride, g_s0, g_s1, g_s2, g_s3, g_srstride, g_spad,
               g_sbw, g_sbh, g_spos, g_scount, g_satd);
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

/* ------------------------------------------------------------------ */
/* lowres_search: the lookahead's field search                         */
/* ------------------------------------------------------------------ */
/* `ref` a shifted, noisy copy of `cur` (so the search has motion to find),
 * padding included; a seed field of arbitrary vectors; kernel against
 * aeg_lowres_search_seeded() */
static int check_lowres(int mb_w, int mb_h, int seeded)
{
    aeg_lowres c, r;
    const int nb = mb_w * mb_h, num = 3, den = 2, range = seeded ? 40 : 16;
    int *seed = (int *)malloc(sizeof(int) * 2 * (size_t)nb);
    int *got = (int *)malloc(sizeof(int) * 3 * (size_t)nb);
    int *want = (int *)malloc(sizeof(int) * 3 * (size_t)nb);
    int x, y, i, bad = 0, dx = rnd(13) - 6, dy = rnd(9) - 4;
    size_t sz;

    if (aeg_lowres_alloc(&c, mb_w, mb_h) || aeg_lowres_alloc(&r, mb_w, mb_h) ||
        !seed || !got || !want) {
        printf("   kernel_check: lowres_search: out of memory\n");
        return 1;
    }
    sz = (size_t)c.stride * (size_t)(c.h + 2 * AEG_LA_PAD);
    for (y = 0; y < c.h + 2 * AEG_LA_PAD; y++)
        for (x = 0; x < c.stride; x++)
            c.mem[(size_t)y * c.stride + x] =
                (uint8_t)(128 + 60 * ((x / 5 + y / 3) & 1) + rnd(40) - 20 +
                          (x * y) % 37);
    for (y = 0; y < r.h + 2 * AEG_LA_PAD; y++)
        for (x = 0; x < r.stride; x++) {
            int sx = x + dx, sy = y + dy, v;
            if (sx < 0) sx = 0;
            if (sx >= c.stride) sx = c.stride - 1;
            if (sy < 0) sy = 0;
            if (sy >= c.h + 2 * AEG_LA_PAD) sy = c.h + 2 * AEG_LA_PAD - 1;
            v = c.mem[(size_t)sy * c.stride + sx] + rnd(9) - 4;
            r.mem[(size_t)y * r.stride + x] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
        }
    for (i = 0; i < 2 * nb; i++)
        seed[i] = rnd(81) - 40;
    aeg_mock_gid = 0;
    lowres_search(c.mem, r.mem, c.stride, AEG_LA_PAD, c.w, c.h, mb_w, mb_h,
                  seed, seeded, num, den, range, AEG_LA_RANGE, got);
    aeg_lowres_search_seeded(&c, &r, want, want + 2 * nb,
                             seeded ? seed : NULL, num, den, range);
    for (i = 0; i < 3 * nb; i++)
        if (got[i] != want[i])
            bad++;
    if (bad)
        printf("   kernel_check: lowres_search %dx%d%s: %d of %d differ\n",
               mb_w, mb_h, seeded ? " seeded" : "", bad, 3 * nb);
    (void)sz;
    aeg_lowres_free(&c);
    aeg_lowres_free(&r);
    free(seed);
    free(got);
    free(want);
    return bad != 0;
}

/* lowres_chains: a ring of pictures in one buffer (slot k at k * slot
 * bytes, as the device keeps them), chains in both directions, seeded from
 * the previous field of the chain or from a given field; every group run in
 * turn against aeg_be_ops.lowres_chains of the CPU */
static int check_chains(int mb_w, int mb_h)
{
    enum { NS = 5 };
    aeg_lowres lr[NS];
    aeg_be_ops cpu;
    aeg_la_pic pics[NS];
    aeg_la_chain chs[4];
    aeg_la_job jobs[8];
    const int nb = mb_w * mb_h;
    int chain_i[16], job_i[40], *seed0, *got, *want, x, y, k, i, bad = 0;
    size_t slot;
    uint8_t *ring;

    aeg_be_cpu_ops(&cpu);
    for (k = 0; k < NS; k++)
        if (aeg_lowres_alloc(&lr[k], mb_w, mb_h))
            return 1;
    slot = (size_t)lr[0].stride * (size_t)(lr[0].h + 2 * AEG_LA_PAD);
    ring = (uint8_t *)malloc(slot * NS);
    seed0 = (int *)malloc(sizeof(int) * 2 * (size_t)nb);
    got = (int *)malloc(sizeof(int) * 3 * (size_t)nb * 8);
    want = (int *)malloc(sizeof(int) * 3 * (size_t)nb * 8);
    if (!ring || !seed0 || !got || !want)
        return 1;
    /* a pattern panning a few samples per picture, plus noise */
    for (k = 0; k < NS; k++) {
        for (y = 0; y < lr[k].h + 2 * AEG_LA_PAD; y++)
            for (x = 0; x < lr[k].stride; x++) {
                int u = x + 3 * k, v = y - 2 * k;
                lr[k].mem[(size_t)y * lr[k].stride + x] =
                    (uint8_t)(110 + 50 * ((u / 5 + v / 4) & 1) +
                              (u * 7 + v * 3) % 29 + rnd(9));
            }
        memcpy(ring + slot * k, lr[k].mem, slot);
        pics[k].plane = lr[k].plane;
        pics[k].slot = (k + 2) % NS;      /* slots need not follow the order */
        pics[k].tag = k;
    }
    for (i = 0; i < 2 * nb; i++)
        seed0[i] = rnd(41) - 20;
    /* chain 0: picture 4 backwards to 3, 2, 1 (seeded after the first);
     * chain 1: picture 0 forwards to 1, 2, given the seed for its first;
     * chain 2: picture 2 to 4, unseeded; chain 3: picture 1 to 0, 3 (the
     * second unseeded) */
    {
        static const int cj[8][4] = { { 3, 0, 1, 1 }, { 2, 1, 2, 1 },
                                      { 1, 1, 3, 2 }, { 1, 1, 3, 2 },
                                      { 2, 1, 2, 1 }, { 4, 0, 1, 1 },
                                      { 0, 0, 1, 1 }, { 3, 0, 1, 1 } };
        static const int cc[4][3] = { { 4, 0, 3 }, { 0, 3, 2 }, { 2, 5, 1 },
                                      { 1, 6, 2 } };
        for (i = 0; i < 8; i++) {
            jobs[i].ref = cj[i][0];
            jobs[i].seeded = cj[i][1] || i == 3;
            jobs[i].num = cj[i][2];
            jobs[i].den = cj[i][3];
            jobs[i].range = 16 + 8 * (cj[i][2] - 1);
            jobs[i].out = want + (size_t)i * 3 * nb;
        }
        for (i = 0; i < 4; i++) {
            chs[i].cur = cc[i][0];
            chs[i].first = cc[i][1];
            chs[i].count = cc[i][2];
            chs[i].seed = i == 1 ? seed0 : NULL;
        }
    }
    cpu.lowres_chains(cpu.priv, pics, NS, NS, chs, 4, jobs, lr[0].stride,
                      lr[0].w, lr[0].h);
    {
        /* the kernel's tables, as the backends pack them */
        uint8_t *planes = (uint8_t *)malloc(slot * NS);
        for (k = 0; k < NS; k++)
            memcpy(planes + slot * (size_t)pics[k].slot, lr[k].mem, slot);
        for (i = 0; i < 4; i++) {
            chain_i[4 * i] = pics[chs[i].cur].slot;
            chain_i[4 * i + 1] = chs[i].first;
            chain_i[4 * i + 2] = chs[i].count;
            chain_i[4 * i + 3] = chs[i].seed ? 0 : -1;
        }
        for (i = 0; i < 8; i++) {
            job_i[5 * i] = pics[jobs[i].ref].slot;
            job_i[5 * i + 1] = jobs[i].seeded;
            job_i[5 * i + 2] = jobs[i].num;
            job_i[5 * i + 3] = jobs[i].den;
            job_i[5 * i + 4] = jobs[i].range;
        }
        memset(got, 0x5a, sizeof(int) * 3 * (size_t)nb * 8);
        for (i = 0; i < 4; i++) {
            aeg_mock_gid = i;
            lowres_chains(planes, (int)slot, lr[0].stride, AEG_LA_PAD, lr[0].w,
                          lr[0].h, mb_w, mb_h, chain_i, job_i, seed0,
                          AEG_LA_RANGE, got);
        }
        free(planes);
    }
    for (i = 0; i < 3 * nb * 8; i++)
        if (got[i] != want[i])
            bad++;
    if (bad)
        printf("   kernel_check: lowres_chains %dx%d: %d of %d differ\n", mb_w,
               mb_h, bad, 3 * nb * 8);
    for (k = 0; k < NS; k++)
        aeg_lowres_free(&lr[k]);
    free(ring);
    free(seed0);
    free(got);
    free(want);
    return bad != 0;
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
                    int cls;
                    /* Four amplitude classes, because the interesting cases of
                     * this kernel are not the ones the quantiser normally
                     * produces: at high QP a dequantised coefficient leaves
                     * int16 and the reference decoder wraps there (bug 8), the
                     * DC rounding happens only in the first pass (bug 7), and
                     * the second pass truncates instead of rounding.  The
                     * kernel that ignored all three agreed with a C reference
                     * that ignored them too (bug 31); class 0 is the one that
                     * shows it. */
                    for (cls = 0; cls < 4; cls++) {
                        for (i = 0; i < g_nblk * 16; i++) {
                            int16_t v;
                            switch (cls) {
                            case 0: v = (int16_t)(rnd(65536) - 32768); break;
                            case 1: v = (int16_t)(rnd(4) ? 0 :
                                                  (rnd(2) ? 32767 : -32767)); break;
                            case 2: v = (int16_t)(i % 16 ? 0 : (rnd(2) ? 32767 : -32768));
                                    break;
                            default: v = (int16_t)(rnd(7) - 3); break;
                            }
                            ((int16_t *)g_qlevel)[i] = v;
                        }
                        run_items(g_nblk, call_iq);
                        cpu.iq_idct_batch(cpu.priv, g_qlevel, g_nblk, qp,
                                          ref_res);
                        bad = 0;
                        for (i = 0; i < g_nblk * 16; i++)
                            if (g_iqres[i] != ref_res[i])
                                bad++;
                        if (bad) {
                            printf("   kernel_check: iq_idct_batch qp %d class "
                                   "%d %d of %d samples differ\n", qp, cls, bad,
                                   g_nblk * 16);
                            fail = 1;
                        }
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

    /* ---- fused forward + inverse: the same levels, counts and residual as
     *      tq_batch followed by iq_idct_batch, at every QP ---------------- */
    for (qp = 0; qp <= 51; qp++) {
        for (intra = 0; intra <= 1; intra++) {
            int16_t *res, *rl, *rr, *cl, *cr;
            int *rn, *cn;
            int i, bad = 0, sparse = qp & 1;

            g_nblk = 1 + rnd(32);
            res = (int16_t *)malloc((size_t)g_nblk * 32);
            rl = (int16_t *)malloc((size_t)g_nblk * 32);
            rr = (int16_t *)malloc((size_t)g_nblk * 32);
            cl = (int16_t *)malloc((size_t)g_nblk * 32);
            cr = (int16_t *)malloc((size_t)g_nblk * 32);
            rn = (int *)malloc((size_t)g_nblk * 4);
            cn = (int *)malloc((size_t)g_nblk * 4);
            g_level = (int16_t *)malloc((size_t)g_nblk * 32);
            g_tqrec = (int16_t *)malloc((size_t)g_nblk * 32);
            g_nz = (int *)malloc((size_t)g_nblk * 4);
            /* full-range residuals, and (odd QPs) mostly small ones, the way
             * a batch mostly looks: empty and DC-only blocks */
            for (i = 0; i < g_nblk * 16; i++)
                res[i] = (int16_t)(sparse ? (rnd(4) ? rnd(5) - 2 : rnd(511) - 255)
                                          : rnd(511) - 255);
            g_res = res;
            g_qbits = 15 + qp / 6;
            g_f = aeg_quant_bias(g_qbits, intra ? 3 : 6);
            g_mf0 = aeg_quant_mf[qp % 6][0];
            g_mf1 = aeg_quant_mf[qp % 6][1];
            g_mf2 = aeg_quant_mf[qp % 6][2];
            g_sh = qp / 6;
            g_ls0 = aeg_dequant_ls[qp % 6][0];
            g_ls1 = aeg_dequant_ls[qp % 6][1];
            g_ls2 = aeg_dequant_ls[qp % 6][2];
            run_items(g_nblk, call_tqiq);
            cpu.tq_batch(cpu.priv, res, g_nblk, qp, intra, rl, rn);
            cpu.iq_idct_batch(cpu.priv, rl, g_nblk, qp, rr);
            cpu.tq_iq_batch(cpu.priv, res, g_nblk, qp, intra, cl, cn, cr);
            for (i = 0; i < g_nblk * 16; i++)
                bad += (g_level[i] != rl[i]) + (g_tqrec[i] != rr[i]) +
                       (cl[i] != rl[i]) + (cr[i] != rr[i]);
            for (i = 0; i < g_nblk; i++)
                bad += (g_nz[i] != rn[i]) + (cn[i] != rn[i]);
            if (bad) {
                printf("   kernel_check: tq_iq_batch qp %d intra %d: %d "
                       "mismatches\n", qp, intra, bad);
                fail = 1;
            }
            free(res); free(rl); free(rr); free(cl); free(cr); free(rn);
            free(cn); free(g_level); free(g_tqrec); free(g_nz);
        }
    }

    /* ---- the DC-split pair against the CPU ops (the per-block steps of
     *      code_luma_i16 / code_chroma_run), at every QP; the given DCs
     *      reach past the int16 range, where the scalar path truncates --- */
    for (qp = 0; qp <= 51; qp++) {
        for (intra = 0; intra <= 1; intra++) {
            int16_t *res, *rl, *rr, *lev;
            int32_t *rn, *rd, *dcin;
            int i, bad = 0, sparse = qp & 1;

            g_nblk = 1 + rnd(32);
            res = (int16_t *)malloc((size_t)g_nblk * 32);
            rl = (int16_t *)malloc((size_t)g_nblk * 32);
            rr = (int16_t *)malloc((size_t)g_nblk * 32);
            lev = (int16_t *)malloc((size_t)g_nblk * 32);
            rn = (int32_t *)malloc((size_t)g_nblk * 4);
            rd = (int32_t *)malloc((size_t)g_nblk * 4);
            dcin = (int32_t *)malloc((size_t)g_nblk * 4);
            g_level = (int16_t *)malloc((size_t)g_nblk * 32);
            g_iqres = (int16_t *)malloc((size_t)g_nblk * 32);
            g_nz = (int *)malloc((size_t)g_nblk * 4);
            g_dc = (int *)malloc((size_t)g_nblk * 4);
            for (i = 0; i < g_nblk * 16; i++)
                res[i] = (int16_t)(sparse ? (rnd(4) ? rnd(5) - 2 : rnd(511) - 255)
                                          : rnd(511) - 255);
            g_res = res;
            g_qbits = 15 + qp / 6;
            g_f = aeg_quant_bias(g_qbits, intra ? 3 : 6);
            g_mf0 = aeg_quant_mf[qp % 6][0];
            g_mf1 = aeg_quant_mf[qp % 6][1];
            g_mf2 = aeg_quant_mf[qp % 6][2];
            g_sh = qp / 6;
            g_ls0 = aeg_dequant_ls[qp % 6][0];
            g_ls1 = aeg_dequant_ls[qp % 6][1];
            g_ls2 = aeg_dequant_ls[qp % 6][2];
            run_items(g_nblk, call_tqdc);
            cpu.tq_dc_batch(cpu.priv, res, g_nblk, qp, intra, rl, rn, rd);
            for (i = 0; i < g_nblk * 16; i++)
                bad += g_level[i] != rl[i];
            for (i = 0; i < g_nblk; i++)
                bad += (g_nz[i] != rn[i]) + (g_dc[i] != rd[i]) +
                       (rl[16 * i] != 0);
            /* the inverse on those levels with DCs of every size */
            for (i = 0; i < g_nblk; i++)
                dcin[i] = g_dc[i] = rnd(4) ? rnd(8001) - 4000
                                           : rnd(80001) - 40000;
            memcpy(lev, rl, (size_t)g_nblk * 32);
            g_qlevel = lev;
            run_items(g_nblk, call_iqdc);
            cpu.iq_dc_batch(cpu.priv, lev, dcin, g_nblk, qp, rr);
            for (i = 0; i < g_nblk * 16; i++)
                bad += g_iqres[i] != rr[i];
            for (i = 0; i < g_nblk; i++) {
                /* and the CPU op against the scalar path's own call */
                uint8_t pred[16], o1[16], o2[16];
                int k;
                for (k = 0; k < 16; k++)
                    pred[k] = (uint8_t)rnd(256);
                memcpy(o1, pred, 16);
                aeg_idct_add_block(o1, 4, lev + 16 * i, qp, pred, 4, &dcin[i]);
                for (k = 0; k < 16; k++) {
                    int v = pred[k] + rr[16 * i + k];
                    o2[k] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
                }
                bad += memcmp(o1, o2, 16) != 0;
            }
            if (bad) {
                printf("   kernel_check: tq_dc/iq_dc qp %d intra %d: %d "
                       "mismatches\n", qp, intra, bad);
                fail = 1;
            }
            free(res); free(rl); free(rr); free(lev); free(rn); free(rd);
            free(dcin); free(g_level); free(g_iqres); free(g_nz); free(g_dc);
        }
    }

    /* ---- SAD over a batch of block pairs ------------------------------ */
    /* The candidates carry both the source block position and the reference
     * displacement: a real batch is a whole picture's search, so the source
     * block differs per candidate (a candidate that ignored the source position
     * would compare every macroblock against the first one's block -- that bug
     * made the parallel search pick vectors that did not match anything). */
    {
        int bw = 16, bh = 16, count = 8, i, bad = 0;
        g_bw = bw;
        g_bh = bh;
        g_count = count;
        g_cstride = 72;
        g_rstride = 64;
        g_cur = (const uchar *)malloc((size_t)g_cstride * (bh + 32));
        g_ref = (const uchar *)malloc((size_t)g_rstride * (bh + 32));
        g_mv = (short *)malloc((size_t)count * 4 * sizeof(short));
        g_sad = (int *)malloc((size_t)count * 4);
        {
            int *ref_sad = (int *)malloc((size_t)count * 4);
            int j;
            for (j = 0; j < g_cstride * (bh + 32); j++)
                ((uchar *)g_cur)[j] = (uchar)rnd(256);
            for (j = 0; j < g_rstride * (bh + 32); j++)
                ((uchar *)g_ref)[j] = (uchar)rnd(256);
            for (j = 0; j < count; j++) {
                ((short *)g_mv)[4 * j + 0] = (short)rnd(16);
                ((short *)g_mv)[4 * j + 1] = (short)rnd(16);
                ((short *)g_mv)[4 * j + 2] = (short)rnd(24);
                ((short *)g_mv)[4 * j + 3] = (short)rnd(24);
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

    /* ---- fractional-pel planes of a reference picture ---------------- */
    {
        int w = 96, h = 64, pad = AEG_PAD, which, x, y;
        size_t stride = (size_t)w + 2 * pad, rows = (size_t)h + 2 * pad;
        uchar *padded = (uchar *)calloc(stride * rows, 1);
        uchar *dst = (uchar *)calloc(stride * rows, 1);
        /* the reference picture as the encoder keeps it: `pad` samples of edge
         * replication on every side, origin `pad` in from the allocation */
        for (y = 0; y < h; y++)
            for (x = 0; x < w; x++)
                padded[(size_t)(y + pad) * stride + pad + x] = (uchar)rnd(256);
        for (y = 0; y < (int)rows; y++)
            for (x = 0; x < (int)stride; x++) {
                int sy = y - pad, sx = x - pad;
                if (sy < 0) sy = 0;
                if (sy > h - 1) sy = h - 1;
                if (sx < 0) sx = 0;
                if (sx > w - 1) sx = w - 1;
                padded[(size_t)y * stride + x] =
                    padded[(size_t)(sy + pad) * stride + pad + sx];
            }
        g_pref = padded + (size_t)pad * stride + pad;
        g_rstride = (int)stride;
        g_pw = w;
        g_ph = h;
        g_pdstride = (int)stride;
        g_pdst = dst + (size_t)pad * stride + pad;
        for (which = 1; which <= 3; which++) {
            int count = (w + 2 * pad) * (h + 2 * pad);
            uchar *exp = (uchar *)calloc(stride * rows, 1);
            g_pwhich = which;
            memset(dst, 0, stride * rows);
            /* reference: the CPU implementation the encoder uses when the
             * stage stays on the CPU, into its own buffer so that the padded
             * picture stays intact for the next plane */
            cpu.interp_plane(cpu.priv, g_pref, (int)stride, w, h, which,
                             exp + (size_t)pad * stride + pad, (int)stride);
            run_items(count, call_intp);
            if (memcmp(dst, exp, stride * rows)) {
                int ndiff = 0;
                for (y = 0; y < (int)(stride * rows); y++)
                    if (dst[y] != exp[y])
                        ndiff++;
                printf("   kernel_check: interp_plane %d: %d samples differ\n",
                       which, ndiff);
                fail = 1;
            }
            free(exp);
        }
        free(padded);
        free(dst);
    }

    /* ---- SATD of fractional-pel candidates --------------------------- */
    {
        int w = 96, h = 64, pad = AEG_PAD, t, bad = 0;
        size_t stride = (size_t)w + 2 * pad, rows = (size_t)h + 2 * pad;
        uchar *padded = (uchar *)calloc(stride * rows, 1);
        uchar *plane[3];
        uchar *cur = (uchar *)calloc(stride * rows, 1);
        const uchar *base[4];
        int *ref_satd;
        short *pos;
        const int sizes[2][2] = { {16, 16}, {8, 8} };

        for (t = 0; t < 3; t++)
            plane[t] = (uchar *)calloc(stride * rows, 1);
        for (t = 0; t < (int)(stride * rows); t++) {
            padded[t] = (uchar)rnd(256);
            cur[t] = (uchar)rnd(256);
        }
        base[0] = padded + (size_t)pad * stride + pad;
        for (t = 0; t < 3; t++)
            base[t + 1] = plane[t] + (size_t)pad * stride + pad;
        for (t = 1; t <= 3; t++)
            cpu.interp_plane(cpu.priv, base[0], (int)stride, w, h, t,
                             (uchar *)base[t], (int)stride);
        cpu.ref_upload(cpu.priv, base, (int)stride, w, h);

        /* the kernel takes whole padded allocations (the backends transfer
         * them in one piece and the kernel shifts by `pad`); the CPU operation
         * takes the picture origin of each.  Both describe the same picture. */
        g_s0 = padded; g_s1 = plane[0]; g_s2 = plane[1]; g_s3 = plane[2];
        g_srstride = (int)stride;
        g_spad = pad;
        g_scur = cur;
        g_scstride = (int)stride;
        for (t = 0; t < 2; t++) {
            int n = 400, i;
            g_sbw = sizes[t][0];
            g_sbh = sizes[t][1];
            g_scount = n;
            bad = 0;
            pos = (short *)malloc((size_t)n * 8);
            g_satd = (int *)malloc((size_t)n * 4);
            ref_satd = (int *)malloc((size_t)n * 4);
            for (i = 0; i < n; i++) {
                int bx = rnd(w - g_sbw + 1), by = rnd(h - g_sbh + 1);
                pos[4 * i + 0] = (short)bx;
                pos[4 * i + 1] = (short)by;
                /* quarter-pel within +-6 samples, both axes and all phases */
                pos[4 * i + 2] = (short)(rnd(25) - 12);
                pos[4 * i + 3] = (short)(rnd(25) - 12);
            }
            g_spos = pos;
            run_items(n, call_satd);
            cpu.satd_batch(cpu.priv, g_scur + (size_t)pad * stride + pad,
                           g_scstride, g_sbw, g_sbh, pos, n, ref_satd);
            for (i = 0; i < n; i++)
                if (g_satd[i] != ref_satd[i])
                    bad++;
            if (bad) {
                printf("   kernel_check: satd_batch %dx%d: %d of %d differ "
                       "(first %d vs %d)\n", g_sbw, g_sbh, bad, n,
                       g_satd[0], ref_satd[0]);
                fail = 1;
            }
            free(pos);
            free(g_satd);
            free(ref_satd);
        }
        free(padded);
        free(cur);
        for (t = 0; t < 3; t++)
            free(plane[t]);
    }

    {
        static const int lsz[][2] = { { 1, 1 }, { 2, 1 }, { 1, 3 }, { 7, 5 },
                                      { 22, 18 }, { 40, 22 } };
        for (t = 0; t < 6; t++) {
            fail |= check_lowres(lsz[t][0], lsz[t][1], 0);
            fail |= check_lowres(lsz[t][0], lsz[t][1], 1);
            fail |= check_chains(lsz[t][0], lsz[t][1]);
        }
    }

    if (!fail)
        printf("   kernel_check: 11 kernels agree with the CPU reference\n");
    return fail;
}
