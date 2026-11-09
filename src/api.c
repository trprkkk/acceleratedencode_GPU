/*
 * The public encoder API: each call goes to the bit-depth build the encoder
 * was opened with (x264's i_bitdepth, chosen at run time).  An aeg_encoder
 * starts with its aeg_params, whose bit_depth the build has filled in.
 *
 * Compiled once.  Without the 10-bit build (make BIT_DEPTHS=8) a request for
 * 10 bits fails in aeg_encoder_open().
 */
#include "h264.h"
#include "api_impl.h"

#if defined(AEG_HAVE_10BIT) || defined(AEG_HAVE_12BIT)
static int depth_of(const aeg_encoder *enc)
{
    return ((const aeg_params *)(const void *)enc)->bit_depth;
}
#endif

#if defined(AEG_HAVE_12BIT) && defined(AEG_HAVE_10BIT)
#define DISPATCH(enc, fn, ...) \
    (depth_of(enc) == 12 ? fn##_12(__VA_ARGS__) : \
     depth_of(enc) == 10 ? fn##_10(__VA_ARGS__) : fn(__VA_ARGS__))
#elif defined(AEG_HAVE_12BIT)
#define DISPATCH(enc, fn, ...) \
    (depth_of(enc) == 12 ? fn##_12(__VA_ARGS__) : fn(__VA_ARGS__))
#elif defined(AEG_HAVE_10BIT)
#define DISPATCH(enc, fn, ...) \
    (depth_of(enc) == 10 ? fn##_10(__VA_ARGS__) : fn(__VA_ARGS__))
#else
#define DISPATCH(enc, fn, ...) fn(__VA_ARGS__)
#endif

aeg_encoder *aeg_encoder_open(const aeg_params *params)
{
    if (!params)
        return aeg_enc_open(params);
    if (params->bit_depth == 10) {
#ifdef AEG_HAVE_10BIT
        return aeg_enc_open_10(params);
#else
        aeg_set_last_error("bit_depth 10: library built without the 10-bit "
                           "encoder (BIT_DEPTHS=8)");
        return NULL;
#endif
    }
    if (params->bit_depth == 12) {
#ifdef AEG_HAVE_12BIT
        return aeg_enc_open_12(params);
#else
        aeg_set_last_error("bit_depth 12: library built without the 12-bit "
                           "encoder (BIT_DEPTHS=8 10)");
        return NULL;
#endif
    }
    if (params->bit_depth != 0 && params->bit_depth != 8) {
        aeg_set_last_error("bit_depth must be 8, 10 or 12");
        return NULL;
    }
    return aeg_enc_open(params);
}

int aeg_encoder_encode(aeg_encoder *enc, const uint8_t *yuv420p, size_t size,
                       int64_t pts, aeg_packet *pkt)
{
    if (!enc)
        return aeg_enc_encode(enc, yuv420p, size, pts, pkt);
    return DISPATCH(enc, aeg_enc_encode, enc, yuv420p, size, pts, pkt);
}

int aeg_encoder_encode_planes(aeg_encoder *enc, const uint8_t *const plane[3],
                              const int stride[3], int64_t pts, aeg_packet *pkt)
{
    if (!enc)
        return aeg_enc_encode_planes(enc, plane, stride, pts, pkt);
    return DISPATCH(enc, aeg_enc_encode_planes, enc, plane, stride, pts, pkt);
}

int aeg_encoder_encode_image(aeg_encoder *enc, const aeg_image *img,
                             int64_t pts, aeg_packet *pkt)
{
    if (!enc)
        return aeg_enc_encode_image(enc, img, pts, pkt);
    return DISPATCH(enc, aeg_enc_encode_image, enc, img, pts, pkt);
}

int aeg_encoder_flush(aeg_encoder *enc, aeg_packet *pkt)
{
    if (!enc)
        return aeg_enc_flush(enc, pkt);
    return DISPATCH(enc, aeg_enc_flush, enc, pkt);
}

int aeg_encoder_headers(aeg_encoder *enc, aeg_packet *pkt)
{
    if (!enc)
        return aeg_enc_headers(enc, pkt);
    return DISPATCH(enc, aeg_enc_headers, enc, pkt);
}

int aeg_encoder_get_params(aeg_encoder *enc, aeg_params *out)
{
    if (!enc)
        return aeg_enc_get_params(enc, out);
    return DISPATCH(enc, aeg_enc_get_params, enc, out);
}

int aeg_encoder_stats(aeg_encoder *enc, aeg_stats *out)
{
    if (!enc)
        return aeg_enc_stats(enc, out);
    return DISPATCH(enc, aeg_enc_stats, enc, out);
}

int aeg_encoder_recon(aeg_encoder *enc, const uint8_t *plane[3], int stride[3],
                      int mb_qp[])
{
    if (!enc)
        return aeg_enc_recon(enc, plane, stride, mb_qp);
    return DISPATCH(enc, aeg_enc_recon, enc, plane, stride, mb_qp);
}

int aeg_encoder_set_qp(aeg_encoder *enc, int qp)
{
    if (!enc)
        return aeg_enc_set_qp(enc, qp);
    return DISPATCH(enc, aeg_enc_set_qp, enc, qp);
}

int aeg_encoder_set_qp_offset(aeg_encoder *enc, int offset)
{
    if (!enc)
        return aeg_enc_set_qp_offset(enc, offset);
    return DISPATCH(enc, aeg_enc_set_qp_offset, enc, offset);
}

int aeg_encoder_set_pass1(aeg_encoder *enc, const aeg_pass1_frame *f, int n)
{
    if (!enc)
        return aeg_enc_set_pass1(enc, f, n);
    return DISPATCH(enc, aeg_enc_set_pass1, enc, f, n);
}

void aeg_encoder_close(aeg_encoder *enc)
{
    if (!enc)
        return;
#ifdef AEG_HAVE_10BIT
    if (depth_of(enc) == 10) {
        aeg_enc_close_10(enc);
        return;
    }
#endif
    aeg_enc_close(enc);
}
