/*
 * CABAC: the H.264 context-adaptive binary arithmetic coder, encoder side.
 *
 * The coder itself is the standard one (9.3): a 9-bit `range` register, a
 * `low` register whose carry is propagated backwards through the bytes that
 * have already been decided, and one 8-bit probability state per context.
 *
 * The state numbering is that of the standard: one byte per context holding
 * `2 * pStateIdx + valMPS`, with `rangeTabLPS` as a 64-row table indexed by
 * pStateIdx and the range class, and `aeg_cabac_trans[state][bin]` giving the
 * next state for the bin that was just coded (9.3.2.3).
 *
 * Bytes are held back until their carry cannot change any more: of the bytes
 * extracted so far, the newest one is kept in `prev` and only written when the
 * next extraction shows whether it needs the carry; a run of 0xff can never be
 * written directly (a carry would turn it into 0x00), so those are counted in
 * `outstanding`.
 */
#ifndef AEG_CABAC_H
#define AEG_CABAC_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/* 4:2:0, 8-bit: contexts 0..459 (the extra ones are chroma_format_idc 3) */
#define AEG_CABAC_CTX_COUNT 460

typedef struct aeg_cabac {
    uint8_t *buf;              /* output; the slice data, without the emulation
                                * prevention pass (the NAL writer adds that)  */
    size_t   cap;              /* capacity of buf                             */
    size_t   len;              /* bytes decided so far                        */
    int      err;              /* ran out of room                             */
    uint32_t low;              /* the interval's base                         */
    uint32_t range;            /* the interval's width, a 9-bit value         */
    int      queue;            /* bits accumulated since the last extraction  */
    int      outstanding;      /* 0xff bytes waiting for a possible carry     */
    uint8_t  prev;             /* last decided byte, not yet written          */
    int      have_prev;
    uint8_t  state[AEG_CABAC_CTX_COUNT];
    int      trace;            /* per-bin trace to stderr (AEG_CABAC_TRACE) */
    int      cost;             /* 1 = price the bins, do not produce bytes
                                * (the rate-distortion pass)                 */
    uint32_t cost256;          /* the accumulated price, in 1/256 bit        */
} aeg_cabac;

/* Table 9-44 (`rangeTabLPS`), addressed as [state >> 1][(range >> 6) - 4]. */
extern const uint8_t aeg_cabac_lps_range[64][4];
/* Next context state for bin 0 and bin 1, from Tables 9-45 and 9-46. */
extern const uint8_t aeg_cabac_trans[128][2];
/* The price of one bin, in 1/256 bit, as [state][bin]; see cabac.c. */
extern const uint16_t aeg_cabac_bin_cost[128][2];

/* The context initialization values (Table 9-5 for I slices, Tables 9-12 …
 * 9-33 for P/B), first 460 contexts, as the (m, n) pair the spec's
 * preCtxState = Clip3(1, 126, ((m * QP) >> 4) + n) is computed from. */
extern const int8_t aeg_cabac_init_I[460][2];
extern const int8_t aeg_cabac_init_PB[3][460][2];

void aeg_cabac_open(aeg_cabac *c, uint8_t *buf, size_t cap);

/* Sets the 460 contexts for a slice: `intra` selects the I-slice table,
 * otherwise `init_idc` (0..2) selects the P/B one.  Both take the slice's QP
 * (`SliceQPY`, i.e. 26 + pic_init_qp_minus26 + slice_qp_delta). */
void aeg_cabac_set_contexts(aeg_cabac *c, int intra, int qp, int init_idc);

/* Renormalises after a bin; in cabac.c so that the header stays cheap, but the
 * common case is a single shift, so it is also the hottest call in the coder. */
void aeg_cabac_renorm(aeg_cabac *c);

/* The state byte is 0..127; the values 0 and 1 (the 0.5-probability state,
 * which `rangeTabLPS` gives an LPS range of 2) are not reachable -- the
 * initialization clamps the pre-state to 1..126, which maps to 2..127, and the
 * transition table keeps it there -- but the coder handles them anyway, which
 * is what lets the test drive every entry of the transition table. */
static inline void aeg_cabac_decision(aeg_cabac *c, int ctx, int bin)
{
    int s = c->state[ctx];
    int lps;
    if (c->cost) {
        /* Price the bin and advance the context only: the decisions are made
         * one pass before the coder runs, so the rate-distortion pass keeps a
         * shadow of the coder state (initialized for the slice, then fed every
         * chosen macroblock through the same writer) and asks it what a
         * candidate would cost. */
        c->cost256 += aeg_cabac_bin_cost[s][bin & 1];
        c->state[ctx] = aeg_cabac_trans[s][bin & 1];
        return;
    }
    lps = aeg_cabac_lps_range[s >> 1][(c->range >> 6) - 4];
    c->range -= (uint32_t)lps;
    if (bin != (s & 1)) {
        c->low += c->range;
        c->range = (uint32_t)lps;
    }
    if (c->trace)
        fprintf(stderr, "B %d %d %u\n", ctx, bin & 1, c->range);
    c->state[ctx] = aeg_cabac_trans[s][bin & 1];
    if (c->range < 0x100)
        aeg_cabac_renorm(c);
}

/* Writes the byte decided by the last extraction, if any. */
void aeg_cabac_putbyte(aeg_cabac *c);

static inline void aeg_cabac_bypass(aeg_cabac *c, int bin)
{
    if (c->cost) {
        c->cost256 += 256;              /* a bypass bin is one bit, exactly */
        return;
    }
    if (c->trace)
        fprintf(stderr, "Y %d %u %u q%d\n", bin & 1, c->range, c->low, c->queue);
    c->low <<= 1;
    if (bin)
        c->low += c->range;
    c->queue++;
    aeg_cabac_putbyte(c);
}

/* end_of_slice_flag (9.3.4.4): the slice syntax puts one at the end of every
 * macroblock, 0 everywhere except the last. */
static inline void aeg_cabac_terminate(aeg_cabac *c, int end)
{
    if (c->cost) {
        /* A constant per macroblock, so its exact price cannot change a
         * decision; the terminating one is charged the flush's few bits. */
        if (end)
            c->cost256 += 6 * 256;
        return;
    }
    if (c->trace)
        fprintf(stderr, "T %d\n", end & 1);
    if (end)
        return;                 /* aeg_cabac_flush() writes the final flag */
    c->range -= 2;
    if (c->range < 0x100)
        aeg_cabac_renorm(c);
}

/* Codes the terminating end_of_slice_flag = 1, writes the last byte and
 * returns the number of bytes written. */
size_t aeg_cabac_flush(aeg_cabac *c);

/* CABAC context offsets inside a 64-coefficient (8x8) residual block, per scan
 * position (Table 9-42; the last-coefficient map is 9.3.2.3). */
extern const uint8_t aeg_cabac_sig8[63];
extern const uint8_t aeg_cabac_last8[63];

/* One residual block (9.3.2.2): coded_block_flag when `cbf_ctx` is not
 * negative, the significance map, the last-coefficient flag and the levels,
 * from the last significant coefficient backwards.  `coef[i]` is the
 * coefficient at scan position i; `cat` is the ctxBlockCat of 4:2:0 (0 = luma
 * DC, 1 = luma AC, 2 = luma 4x4, 3 = chroma DC, 4 = chroma AC, 5 = luma 8x8).
 * Returns the number of non-zero coefficients. */
int aeg_cabac_residual(aeg_cabac *c, const int16_t *coef, int n, int cat,
                       int cbf_ctx);

/* Price in 1/256 bit of one coefficient of a residual block (see cabac.c):
 * `pos` is the scan position, `n` the block's coefficient count. */
int64_t aeg_cabac_coef_price(const aeg_cabac *c, int cat, int n, int pos,
                             int level, int is_last, int with_sig);
int64_t aeg_cabac_residual_price(const aeg_cabac *st, const int16_t *coef,
                                 int n, int cat, int cbf_ctx);

#endif
