/* Fuzz target: the library API with arbitrary parameters and pictures.
 *
 * The input picks every aeg_params field (mostly inside its range, sometimes
 * just outside, sometimes wild -- fz_opt), opens an encoder, and if the
 * library accepts the parameters, codes up to six small pictures made of the
 * remaining bytes, drains it and calls the rest of the API on the way.
 * Refusing a parameter set is fine; a crash, a sanitizer report, a hang or a
 * packet that breaks the API contract is a finding. */
#include <stdio.h>
#include <stdlib.h>
#include "aeg/aeg.h"
#include "fuzz.h"

#define FZ_MAXDIM 80        /* keeps one input well under a second */

static void check_packet(const aeg_packet *pkt)
{
    /* every packet is Annex-B: it starts with a start code */
    if (pkt->size < 4 || !pkt->data ||
        !(pkt->data[0] == 0 && pkt->data[1] == 0 &&
          (pkt->data[2] == 1 || (pkt->data[2] == 0 && pkt->data[3] == 1)))) {
        fprintf(stderr, "fuzz_api: packet of %zu bytes without a start code\n",
                pkt->size);
        abort();
    }
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    fz_reader r;
    aeg_params p, q;
    aeg_encoder *e;
    aeg_packet pkt;
    uint8_t *frame = NULL;
    size_t fsize, i;
    int k, nframes, bps, rc, mid, guard;

    aeg_set_log_level(-1);
    fz_init(&r, data, size);
    aeg_params_default(&p);

    p.width = 1 + (int)(fz_u8(&r) % FZ_MAXDIM);
    p.height = 1 + (int)(fz_u8(&r) % FZ_MAXDIM);
    if (fz_u8(&r) > 250) {          /* the sizes the library has to refuse */
        p.width = fz_opt(&r, p.width, -16, 0);
        p.height = fz_opt(&r, p.height, -16, 0);
    }
    p.preset = fz_opt(&r, p.preset, 0, 4);
    p.qp = fz_opt(&r, p.qp, 0, 51);
    p.rc_mode = fz_opt(&r, p.rc_mode, 0, 2);
    p.bitrate_kbps = fz_opt(&r, p.bitrate_kbps, 0, 3000);
    p.crf = fz_opt(&r, (int)p.crf, 0, 51) + fz_u8(&r) / 256.0;
    p.qcomp = fz_opt(&r, p.qcomp, 0, 100);
    p.vbv_maxrate_kbps = fz_opt(&r, p.vbv_maxrate_kbps, 0, 3000);
    p.vbv_bufsize_kbits = fz_opt(&r, p.vbv_bufsize_kbits, 0, 3000);
    p.qp_min = fz_opt(&r, p.qp_min, 0, 51);
    p.qp_max = fz_opt(&r, p.qp_max, 0, 51);
    p.chroma_qp_offset = fz_opt(&r, p.chroma_qp_offset, -12, 12);
    p.gop_size = fz_opt(&r, p.gop_size, 0, 12);
    p.bframes = fz_opt(&r, p.bframes, 0, 16);
    p.b_adapt = fz_opt(&r, p.b_adapt, -1, 2);
    p.b_qp_offset = fz_opt(&r, p.b_qp_offset, -1, 12);
    p.i_qp_offset = fz_opt(&r, p.i_qp_offset, -1, 12);
    p.direct = fz_opt(&r, p.direct, -1, 3);
    p.scenecut = fz_opt(&r, p.scenecut, 0, 100);
    p.slices = fz_opt(&r, p.slices, 0, 6);
    p.threads = fz_opt(&r, p.threads, 0, 3);
    p.frame_threads = fz_opt(&r, p.frame_threads, 0, 3);
    p.aq_mode = fz_opt(&r, p.aq_mode, 0, 1);
    p.aq_strength = fz_opt(&r, p.aq_strength, 0, 30);
    p.transform_8x8 = fz_opt(&r, p.transform_8x8, -1, 1);
    p.cabac = fz_opt(&r, p.cabac, -1, 1);
    p.deblock = fz_opt(&r, p.deblock, 0, 1);
    p.deblock_alpha = fz_opt(&r, p.deblock_alpha, -6, 6);
    p.deblock_beta = fz_opt(&r, p.deblock_beta, -6, 6);
    p.search_range = fz_opt(&r, p.search_range, 0, 64);
    p.subme = fz_opt(&r, p.subme, -1, 9);
    p.refine_mv = fz_opt(&r, p.refine_mv, -1, 2);
    p.refine_cand = fz_opt(&r, p.refine_cand, -1, 2);
    p.qp_rd = fz_opt(&r, p.qp_rd, -1, 1);
    p.refs = fz_opt(&r, p.refs, -1, 16);
    p.weightp = fz_opt(&r, p.weightp, -1, 2);
    p.b_pyramid = fz_opt(&r, p.b_pyramid, -1, 2);
    p.mbtree = fz_opt(&r, p.mbtree, -1, 1);
    p.rc_lookahead = fz_opt(&r, p.rc_lookahead, -1, 40);
    p.trellis = fz_opt(&r, p.trellis, -1, 2);
    p.partitions = fz_opt(&r, p.partitions, -1, 31);
    p.chroma_me = fz_opt(&r, p.chroma_me, -1, 1);
    p.interlaced = fz_opt(&r, p.interlaced, 0, 2);
    p.field_coding = fz_opt(&r, p.field_coding, 0, 4);
    p.bit_depth = fz_opt(&r, p.bit_depth, 0, 10);
    p.level_idc = fz_opt(&r, p.level_idc, 0, 62);
    p.fps_num = fz_opt(&r, p.fps_num, 1, 60000);
    p.fps_den = fz_opt(&r, p.fps_den, 1, 1001);
    p.timebase_num = fz_opt(&r, p.timebase_num, 0, 1001);
    p.timebase_den = fz_opt(&r, p.timebase_den, 0, 60000);
    p.sar_w = fz_opt(&r, p.sar_w, 0, 255);
    p.sar_h = fz_opt(&r, p.sar_h, 0, 255);
    p.fullrange = fz_opt(&r, p.fullrange, 0, 1);
    p.colorprim = fz_opt(&r, p.colorprim, -1, 22);
    p.transfer = fz_opt(&r, p.transfer, -1, 18);
    p.colormatrix = fz_opt(&r, p.colormatrix, -1, 14);
    p.nal_hrd = fz_opt(&r, p.nal_hrd, 0, 1);
    p.info_sei = fz_opt(&r, p.info_sei, 0, 1);
    p.aud = fz_opt(&r, p.aud, 0, 1);
    p.ssim = fz_opt(&r, p.ssim, 0, 1);
    /* the CPU backend: the fuzzer is about the encoder, not a device */
    p.backend = AEG_BACKEND_CPU;
    p.dump_stats = NULL;

    e = aeg_encoder_open(&p);
    if (getenv("AEG_FUZZ_VERBOSE"))
        fprintf(stderr, "fuzz_api: %dx%d preset %d rc %d bf %d il %d/%d "
                "depth %d: %s\n", p.width, p.height, p.preset, p.rc_mode,
                p.bframes, p.interlaced, p.field_coding, p.bit_depth,
                e ? "open" : aeg_last_error());
    if (!e)
        return 0;                   /* refused: fine */
    if (aeg_encoder_get_params(e, &q) != AEG_OK) {
        fprintf(stderr, "fuzz_api: get_params failed on an open encoder\n");
        abort();
    }
    if (aeg_encoder_headers(e, &pkt) == AEG_OK)
        check_packet(&pkt);

    /* aeg_encoder_encode() takes 8-bit samples at any depth; a 10-bit
     * encoder also gets 16-bit pictures through aeg_encoder_encode_image() */
    fsize = (size_t)q.width * q.height +
            2 * (size_t)((q.width + 1) / 2) * ((q.height + 1) / 2);
    frame = (uint8_t *)malloc(2 * fsize);
    if (!frame) {
        aeg_encoder_close(e);
        return 0;
    }
    nframes = 1 + (int)(fz_u8(&r) % 6);
    mid = (int)(fz_u8(&r) % 8);
    for (k = 0; k < nframes; k++) {
        /* the remaining bytes, repeated, are the samples */
        bps = q.bit_depth > 8 && (k & 1) ? 2 : 1;
        for (i = 0; i < fsize * bps; i++)
            frame[i] = r.n ? r.p[(r.pos + i + (size_t)k * 7) % r.n] : (uint8_t)i;
        if (bps == 2)               /* keep 16-bit samples inside 10 bits */
            for (i = 1; i < 2 * fsize; i += 2)
                frame[i] &= 3;
        if (k == mid) {
            aeg_encoder_set_qp(e, fz_opt(&r, 26, 0, 51));
            aeg_encoder_set_qp_offset(e, fz_opt(&r, 0, -12, 12));
        }
        if (bps == 2) {
            aeg_image im;
            int cw = (q.width + 1) / 2, ch = (q.height + 1) / 2;
            memset(&im, 0, sizeof(im));
            im.csp = AEG_CSP_I420;
            im.depth = q.bit_depth;
            im.plane[0] = frame;
            im.plane[1] = frame + (size_t)2 * q.width * q.height;
            im.plane[2] = (const uint8_t *)im.plane[1] + (size_t)2 * cw * ch;
            im.stride[0] = 2 * q.width;
            im.stride[1] = im.stride[2] = 2 * cw;
            rc = aeg_encoder_encode_image(e, &im, k, &pkt);
        } else {
            rc = aeg_encoder_encode(e, frame, fsize, k, &pkt);
        }
        if (rc == AEG_OK)
            check_packet(&pkt);
        else if (rc != AEG_EAGAIN)
            break;
        aeg_encoder_recon(e, NULL, NULL, NULL);
    }
    for (guard = 0; guard < 64; guard++) {
        rc = aeg_encoder_flush(e, &pkt);
        if (rc != AEG_OK)
            break;
        check_packet(&pkt);
    }
    if (guard == 64) {
        fprintf(stderr, "fuzz_api: flush never reached AEG_EOF\n");
        abort();
    }
    {
        aeg_stats st;
        aeg_encoder_stats(e, &st);
    }
    aeg_encoder_close(e);
    free(frame);
    return 0;
}
