/* Fuzz target: aeg_encoder_encode_image() -- every input layout (colour
 * space, depth 8..16, strides, alignment) into an 8- or 10-bit encoder,
 * progressive or interlaced (which changes the chroma conversion).
 *
 * Each plane is its own allocation of exactly stride x rows bytes, so ASan
 * sees any read past the layout the caller described.  A layout that breaks
 * the documented contract (stride below the width, a misaligned 16-bit plane,
 * an unknown colour space or depth) must be refused with AEG_EINVAL. */
#include <stdio.h>
#include <stdlib.h>
#include "aeg/aeg.h"
#include "fuzz.h"

static const struct { int xs, ys, planes, il; } lay[10] = {
    {1, 1, 3, 1}, {1, 1, 3, 1}, {1, 1, 2, 2}, {1, 1, 2, 2}, {1, 0, 3, 1},
    {1, 0, 3, 1}, {1, 0, 2, 2}, {0, 0, 3, 1}, {0, 0, 3, 1}, {0, 0, 1, 1},
};

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    fz_reader r;
    aeg_params p;
    aeg_encoder *e;
    aeg_image im;
    aeg_packet pkt;
    uint8_t *buf[3] = {NULL, NULL, NULL};
    int w, h, k, i, rc, guard, valid, n;

    aeg_set_log_level(-1);
    fz_init(&r, data, size);
    aeg_params_default(&p);
    w = p.width = 16 + (int)(fz_u8(&r) % 49);
    h = p.height = 16 + (int)(fz_u8(&r) % 49);
    p.preset = 0;
    p.bframes = (int)(fz_u8(&r) % 2);
    p.bit_depth = fz_u8(&r) & 1 ? 10 : 8;
    p.interlaced = (int)(fz_u8(&r) % 3);
    p.backend = AEG_BACKEND_CPU;
    p.frame_threads = 1;
    e = aeg_encoder_open(&p);
    if (!e)
        return 0;
    {
        /* the size it codes (odd dimensions are rounded down) is the one a
         * layout is checked against */
        aeg_params q;
        aeg_encoder_get_params(e, &q);
        w = q.width;
        h = q.height;
    }

    n = 1 + (int)(fz_u8(&r) % 3);
    for (k = 0; k < n; k++) {
        size_t len[3] = {0, 0, 0};
        int csp = fz_opt(&r, AEG_CSP_I420, 0, 9);
        int depth = fz_opt(&r, 8, 8, 16);
        int L = csp >= 0 && csp <= 9 ? csp : 0, bps;
        memset(&im, 0, sizeof(im));
        im.csp = csp;
        im.depth = depth;
        bps = depth > 8 ? 2 : 1;
        valid = csp >= 0 && csp <= 9 && depth >= 8 && depth <= 16;
        for (i = 0; i < 3; i++) {
            int pw = i ? (w + lay[L].xs) >> lay[L].xs : w;
            int ph = i ? (h + lay[L].ys) >> lay[L].ys : h;
            int minst = pw * bps * (i ? lay[L].il : 1);
            unsigned sel = fz_u8(&r);
            size_t j;
            if (i >= lay[L].planes)
                break;
            if (sel < 200)
                im.stride[i] = minst + (int)(fz_u8(&r) % 40);
            else if (sel < 240)
                im.stride[i] = minst - 1 - (int)(fz_u8(&r) % 8);
            else
                im.stride[i] = fz_s16(&r);
            if (im.stride[i] < minst)
                valid = 0;
            /* +1: room to misalign the plane by one byte */
            len[i] = (size_t)(im.stride[i] > 0 ? im.stride[i] : 1) * ph + 1;
            buf[i] = (uint8_t *)malloc(len[i]);
            if (!buf[i])
                goto done;
            for (j = 0; j < len[i]; j++)
                buf[i][j] = r.n ? r.p[(r.pos + j * 3 + (size_t)i) % r.n] : 0;
            im.plane[i] = buf[i];
            if (bps == 2 && (im.stride[i] & 1))
                valid = 0;                      /* odd 16-bit stride */
            if (bps == 2 && fz_u8(&r) > 240) {
                im.plane[i] = buf[i] + 1;       /* misaligned: refused */
                valid = 0;
            } else if (fz_u8(&r) > 250) {
                im.plane[i] = NULL;
                valid = 0;
            }
        }
        rc = aeg_encoder_encode_image(e, &im, k, &pkt);
        if (rc != AEG_OK && rc != AEG_EAGAIN && valid) {
            fprintf(stderr, "fuzz_image: a valid layout was refused (%d: %s)\n",
                    rc, aeg_last_error());
            abort();
        }
        if ((rc == AEG_OK || rc == AEG_EAGAIN) && !valid) {
            fprintf(stderr, "fuzz_image: an invalid layout was accepted: %dx%d "
                    "csp %d depth %d strides %d %d %d planes %p %p %p\n", w, h,
                    im.csp, im.depth, im.stride[0], im.stride[1], im.stride[2],
                    im.plane[0], im.plane[1], im.plane[2]);
            abort();
        }
        for (i = 0; i < 3; i++) {
            free(buf[i]);
            buf[i] = NULL;
        }
    }
    for (guard = 0; guard < 64 && aeg_encoder_flush(e, &pkt) == AEG_OK; guard++)
        ;
done:
    for (i = 0; i < 3; i++)
        free(buf[i]);
    aeg_encoder_close(e);
    return 0;
}
