/* Shared helpers of the fuzz targets (tests/fuzz/README.md).
 *
 * Each target defines LLVMFuzzerTestOneInput(), so it builds both with
 * libFuzzer (clang -fsanitize=fuzzer) and with driver.c, a small standalone
 * runner for compilers without libFuzzer (gcc + ASan/UBSan). */
#ifndef AEG_FUZZ_H
#define AEG_FUZZ_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

/* reads the input front to back; past its end every value is 0 */
typedef struct {
    const uint8_t *p;
    size_t n, pos;
} fz_reader;

static inline void fz_init(fz_reader *r, const uint8_t *p, size_t n)
{
    r->p = p;
    r->n = n;
    r->pos = 0;
}

static inline unsigned fz_u8(fz_reader *r)
{
    return r->pos < r->n ? r->p[r->pos++] : 0;
}

static inline int fz_s16(fz_reader *r)
{
    unsigned a = fz_u8(r), b = fz_u8(r);
    return (int)(int16_t)(uint16_t)(a | b << 8);
}

/* An int option: mostly inside [lo, hi] -- where the interesting code is --
 * sometimes just outside it, sometimes a wild value, sometimes left alone
 * (keep). The library must refuse what it cannot code, never crash. */
static inline int fz_opt(fz_reader *r, int keep, int lo, int hi)
{
    unsigned sel = fz_u8(r);
    if (sel < 96)
        return keep;
    if (sel < 224)
        return lo + (int)(fz_u8(r) % (unsigned)(hi - lo + 1));
    switch (sel & 7) {
    case 0: return lo - 1;
    case 1: return hi + 1;
    case 2: return -1;
    case 3: return 0x7fffffff;
    case 4: return -0x7fffffff - 1;
    default: return fz_s16(r);
    }
}

#endif
