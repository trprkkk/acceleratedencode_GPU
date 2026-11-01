/*
 * The encoder's entry points as each bit-depth build defines them.
 *
 * encoder.c is compiled once per bit depth (see bitdepth.h); the public
 * aeg_encoder_* functions are api.c's, which pick the build an encoder was
 * opened with.  encoder.c defines aeg_enc_* (the 8-bit build) and, after the
 * Makefile renames every symbol the 10-bit objects define, aeg_enc_*_10.
 */
#ifndef AEG_API_IMPL_H
#define AEG_API_IMPL_H

#include "aeg/aeg.h"

#define AEG_IMPL_DECLARE(S)                                                    \
    aeg_encoder *aeg_enc_open##S(const aeg_params *params);                    \
    int  aeg_enc_encode##S(aeg_encoder *enc, const uint8_t *yuv420p,           \
                           size_t size, int64_t pts, aeg_packet *pkt);         \
    int  aeg_enc_encode_planes##S(aeg_encoder *enc,                            \
                                  const uint8_t *const plane[3],               \
                                  const int stride[3], int64_t pts,            \
                                  aeg_packet *pkt);                            \
    int  aeg_enc_encode_image##S(aeg_encoder *enc, const aeg_image *img,       \
                                 int64_t pts, aeg_packet *pkt);                \
    int  aeg_enc_flush##S(aeg_encoder *enc, aeg_packet *pkt);                  \
    int  aeg_enc_headers##S(aeg_encoder *enc, aeg_packet *pkt);                \
    int  aeg_enc_get_params##S(aeg_encoder *enc, aeg_params *out);             \
    int  aeg_enc_stats##S(aeg_encoder *enc, aeg_stats *out);                   \
    int  aeg_enc_recon##S(aeg_encoder *enc, const uint8_t *plane[3],           \
                          int stride[3], int mb_qp[]);                         \
    int  aeg_enc_set_qp##S(aeg_encoder *enc, int qp);                          \
    int  aeg_enc_set_qp_offset##S(aeg_encoder *enc, int offset);               \
    int  aeg_enc_set_pass1##S(aeg_encoder *enc, const aeg_pass1_frame *f,      \
                              int n);                                          \
    void aeg_enc_close##S(aeg_encoder *enc);

AEG_IMPL_DECLARE()
AEG_IMPL_DECLARE(_10)

/* encoder.c: its definitions of the public names are this build's */
#ifdef AEG_IMPL_RENAME
#define aeg_encoder_open          aeg_enc_open
#define aeg_encoder_encode        aeg_enc_encode
#define aeg_encoder_encode_planes aeg_enc_encode_planes
#define aeg_encoder_encode_image  aeg_enc_encode_image
#define aeg_encoder_flush         aeg_enc_flush
#define aeg_encoder_headers       aeg_enc_headers
#define aeg_encoder_get_params    aeg_enc_get_params
#define aeg_encoder_stats         aeg_enc_stats
#define aeg_encoder_recon         aeg_enc_recon
#define aeg_encoder_set_qp        aeg_enc_set_qp
#define aeg_encoder_set_qp_offset aeg_enc_set_qp_offset
#define aeg_encoder_set_pass1     aeg_enc_set_pass1
#define aeg_encoder_close         aeg_enc_close
#endif

#endif /* AEG_API_IMPL_H */
