/*
 * CABAC: the H.264 arithmetic coder (9.3.4), encoder side, plus the two
 * probability tables it is defined in terms of.
 *
 * This is the arithmetic coder only.  The binarizations and the context
 * assignments of the individual syntax elements live with the syntax writer in
 * encoder.c; nothing here knows what a macroblock is.  (The reason the split
 * matters: the coder has to be exactly right before any of the syntax work can
 * be debugged, and a decoder stub can check it on its own -- tests/cabac_check.c
 * does precisely that, against a decoder written from the decoding process in
 * 9.3.3 rather than from this file.)
 */
#include "cabac.h"

/* Table 9-44: the LPS subinterval for each of the 64 probability states and
 * each of the four range classes (range >> 6, in 4..7).  Row 0 is the 0.5
 * state and is never reached by the initialization tables (they start at
 * preCtxState 1 -> state 2). */
const uint8_t aeg_cabac_lps_range[64][4] = {
    {  2,  2,  2,  2}, {  6,  7,  8,  9}, {  6,  7,  9, 10}, {  6,  8,  9, 11},
    {  7,  8, 10, 11}, {  7,  9, 10, 12}, {  7,  9, 11, 12}, {  8,  9, 11, 13},
    {  8, 10, 12, 14}, {  9, 11, 12, 14}, {  9, 11, 13, 15}, { 10, 12, 14, 16},
    { 10, 12, 15, 17}, { 11, 13, 15, 18}, { 11, 14, 16, 19}, { 12, 14, 17, 20},
    { 12, 15, 18, 21}, { 13, 16, 19, 22}, { 14, 17, 20, 23}, { 14, 18, 21, 24},
    { 15, 19, 22, 25}, { 16, 20, 23, 27}, { 17, 21, 25, 28}, { 18, 22, 26, 30},
    { 19, 23, 27, 31}, { 20, 24, 29, 33}, { 21, 26, 30, 35}, { 22, 27, 32, 37},
    { 23, 28, 33, 39}, { 24, 30, 35, 41}, { 26, 31, 37, 43}, { 27, 33, 39, 45},
    { 29, 35, 41, 48}, { 30, 37, 43, 50}, { 32, 39, 46, 53}, { 33, 41, 48, 56},
    { 35, 43, 51, 59}, { 37, 45, 54, 62}, { 39, 48, 56, 65}, { 41, 50, 59, 69},
    { 43, 53, 63, 72}, { 46, 56, 66, 76}, { 48, 59, 69, 80}, { 51, 62, 73, 85},
    { 53, 65, 77, 89}, { 56, 69, 81, 94}, { 59, 72, 86, 99}, { 62, 76, 90, 104},
    { 66, 80, 95, 110}, { 69, 85, 100, 116}, { 73, 89, 105, 122}, { 77, 94, 111, 128},
    { 81, 99, 117, 135}, { 85, 104, 123, 142}, { 90, 110, 130, 150}, { 95, 116, 137, 158},
    { 100, 122, 144, 166}, { 105, 128, 152, 175}, { 111, 135, 160, 185}, { 116, 142, 169, 195},
    { 123, 150, 178, 205}, { 128, 158, 187, 216}, { 128, 167, 197, 227}, { 128, 176, 208, 240},
};

/* Tables 9-45/9-46 in one table: aeg_cabac_trans[state][bin] is the next state
 * after coding `bin`.  Note that the two halves of a pair are not symmetric --
 * an MPS bin moves towards the state where the MPS is most likely (row 1), an
 * LPS bin jumps far away from it. */
const uint8_t aeg_cabac_trans[128][2] = {
    {  0,  0}, {  1,  1}, {  2, 50}, { 51,  3},
    {  2, 50}, { 51,  3}, {  4, 52}, { 53,  5},
    {  6, 52}, { 53,  7}, {  8, 52}, { 53,  9},
    { 10, 54}, { 55, 11}, { 12, 54}, { 55, 13},
    { 14, 54}, { 55, 15}, { 16, 56}, { 57, 17},
    { 18, 56}, { 57, 19}, { 20, 56}, { 57, 21},
    { 22, 58}, { 59, 23}, { 24, 58}, { 59, 25},
    { 26, 60}, { 61, 27}, { 28, 60}, { 61, 29},
    { 30, 60}, { 61, 31}, { 32, 62}, { 63, 33},
    { 34, 62}, { 63, 35}, { 36, 64}, { 65, 37},
    { 38, 66}, { 67, 39}, { 40, 66}, { 67, 41},
    { 42, 66}, { 67, 43}, { 44, 68}, { 69, 45},
    { 46, 68}, { 69, 47}, { 48, 70}, { 71, 49},
    { 50, 72}, { 73, 51}, { 52, 72}, { 73, 53},
    { 54, 74}, { 75, 55}, { 56, 74}, { 75, 57},
    { 58, 76}, { 77, 59}, { 60, 78}, { 79, 61},
    { 62, 78}, { 79, 63}, { 64, 80}, { 81, 65},
    { 66, 82}, { 83, 67}, { 68, 82}, { 83, 69},
    { 70, 84}, { 85, 71}, { 72, 84}, { 85, 73},
    { 74, 88}, { 89, 75}, { 76, 88}, { 89, 77},
    { 78, 90}, { 91, 79}, { 80, 90}, { 91, 81},
    { 82, 94}, { 95, 83}, { 84, 94}, { 95, 85},
    { 86, 96}, { 97, 87}, { 88, 96}, { 97, 89},
    { 90, 100}, { 101, 91}, { 92, 100}, { 101, 93},
    { 94, 102}, { 103, 95}, { 96, 104}, { 105, 97},
    { 98, 104}, { 105, 99}, { 100, 108}, { 109, 101},
    { 102, 108}, { 109, 103}, { 104, 110}, { 111, 105},
    { 106, 112}, { 113, 107}, { 108, 114}, { 115, 109},
    { 110, 116}, { 117, 111}, { 112, 118}, { 119, 113},
    { 114, 118}, { 119, 115}, { 116, 122}, { 123, 117},
    { 118, 122}, { 123, 119}, { 120, 124}, { 125, 121},
    { 122, 126}, { 127, 123}, { 124, 127}, { 126, 125},
};

/* How many bits a range needs before it is back in [256, 512), indexed by
 * range >> 3.  The smallest range a decision can produce is 256 - 240 = 16
 * (state 63's LPS against the widest range), so entries 0 and 1 never occur. */
static const uint8_t renorm_shift[64] = {
    6, 5, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};

static void cabac_emit(aeg_cabac *c, uint8_t b)
{
    if (c->len >= c->cap) {
        c->err = 1;
        return;
    }
    c->buf[c->len++] = b;
}

void aeg_cabac_open(aeg_cabac *c, uint8_t *buf, size_t cap)
{
    c->buf = buf;
    c->cap = cap;
    c->len = 0;
    c->err = 0;
    c->low = 0;
    c->range = 0x01FE;
    c->queue = -9;             /* the first 9 bits are not written: the
                                * decoder's codIOffset starts where they end */
    c->outstanding = 0;
    c->prev = 0;
    c->have_prev = 0;
}

void aeg_cabac_set_contexts(aeg_cabac *c, int intra, int qp, int init_idc)
{
    const int8_t (*tab)[2] = intra ? aeg_cabac_init_I
                                   : aeg_cabac_init_PB[init_idc & 3];
    int i;
    for (i = 0; i < AEG_CABAC_CTX_COUNT; i++) {
        int pre = ((tab[i][0] * qp) >> 4) + tab[i][1];
        int p;
        if (pre < 1)
            pre = 1;
        else if (pre > 126)
            pre = 126;
        p = pre < 127 - pre ? pre : 127 - pre;
        /* odd bits: the MPS, taken from the sign of (pre - 63) */
        c->state[i] = (uint8_t)((p << 1) | (pre >> 6));
    }
}

/* Extracts one byte from `low` once at least eight bits above the pending ones
 * are known.  `queue` counts the bits shifted in since the last extraction, so
 * the byte is the bits [queue + 10, queue + 18) of `low`, the "+ 10" being the
 * nine bits that are still unresolved below it plus its own lowest bit. */
void aeg_cabac_putbyte(aeg_cabac *c)
{
    if (c->queue < 0)
        return;
    {
        uint32_t out = c->low >> (c->queue + 10);
        c->low &= (0x400u << c->queue) - 1u;
        c->queue -= 8;
        if ((out & 0xff) == 0xff) {
            c->outstanding++;       /* a carry would make this 0x00 */
        } else {
            int carry = (int)(out >> 8);
            if (c->have_prev)
                cabac_emit(c, (uint8_t)(c->prev + carry));
            while (c->outstanding > 0) {
                cabac_emit(c, (uint8_t)(carry - 1));
                c->outstanding--;
            }
            c->prev = (uint8_t)out;
            c->have_prev = 1;
        }
    }
}

/* Shifts until the range is back in [256, 512).  One shift is the normal case;
 * the loop exists because a state's LPS range can be as small as 2 (state 0,
 * the 0.5 state), and a decoder normalises with `while (range < 256)` so it
 * would take two shifts of six and one bits for the same input.  Shifting in
 * two steps emits the same bytes as shifting in one, because the byte
 * extraction is incremental. */
void aeg_cabac_renorm(aeg_cabac *c)
{
    while (c->range < 0x100) {
        int shift = renorm_shift[c->range >> 3];
        c->range <<= shift;
        c->low <<= shift;
        c->queue += shift;
        aeg_cabac_putbyte(c);
    }
}

size_t aeg_cabac_flush(aeg_cabac *c)
{
    /* The final end_of_slice_flag = 1 (9.3.4.5).  It is coded against a fixed
     * two-unit LPS interval: the value moves to the top two units of the
     * range, and an odd value inside them is chosen so that the decoder's test
     * (`codIOffset >= range` after `range -= 2`) comes out true. */
    c->low += c->range - 2;
    c->low |= 1;

    /* Nine more bits than the interval needs: the decoder pre-loads nine bits
     * before it decodes anything, so the encoder is nine bits ahead of what it
     * has written.  The shift makes those nine bits writable. */
    c->low <<= 9;
    c->queue += 9;

    /* Write everything that is decided now. */
    while (c->queue >= 0)
        aeg_cabac_putbyte(c);

    /* Align what is left and close with a stop bit: the last byte is then
     * never 0x00, which would look like a start code in the packed NAL. */
    if (c->queue < 0) {
        c->low <<= -c->queue;
        c->queue = 0;
    }
    c->low |= 1u << 10;
    aeg_cabac_putbyte(c);

    if (c->have_prev) {
        cabac_emit(c, c->prev);
        c->have_prev = 0;
    }
    while (c->outstanding > 0) {
        cabac_emit(c, 0xff);
        c->outstanding--;
    }
    return c->len;
}
