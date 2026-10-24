/*
 * Bit writer with RBSP emulation prevention, Exp-Golomb codes and the
 * NAL/SPS/PPS/slice-header syntax used by the encoder.
 */
#include "h264.h"

void aeg_bw_init(aeg_bitwriter *bw, uint8_t *buf, size_t cap, int emu)
{
    bw->buf = buf;
    bw->cap = cap;
    bw->len = 0;
    bw->cache = 0;
    bw->bits = 0;
    bw->err = 0;
    bw->enable_emu = emu;
    bw->count_only = buf ? 0 : 1;
    bw->zeros = 0;
}

/* aeg_bw_put() and the byte writer live in h264.h: they are inline because the
 * RD cost calls them millions of times. */

void aeg_bw_align_zero(aeg_bitwriter *bw)
{
    if (bw->bits)
        aeg_bw_put(bw, 8 - bw->bits, 0);
}

void aeg_bw_align_one(aeg_bitwriter *bw)
{
    int n = 8 - bw->bits;
    aeg_bw_put(bw, n, (1u << (n - 1)));
}

void aeg_bw_ue(aeg_bitwriter *bw, uint32_t val)
{
    uint32_t v = val + 1;
    int n = 0;
    uint32_t t = v;
    while (t > 1) { n++; t >>= 1; }
    aeg_bw_put(bw, n, 0);
    aeg_bw_put(bw, n + 1, v);
}

void aeg_bw_se(aeg_bitwriter *bw, int32_t val)
{
    if (val <= 0)
        aeg_bw_ue(bw, (uint32_t)(-2 * (int64_t)val));
    else
        aeg_bw_ue(bw, (uint32_t)(2 * (int64_t)val - 1));
}

size_t aeg_bw_finish(aeg_bitwriter *bw)
{
    aeg_bw_align_one(bw);
    return bw->len;
}

/* ------------------------------------------------------------------ */
/* NAL units                                                           */
/* ------------------------------------------------------------------ */
size_t aeg_nal_write(uint8_t *out, size_t out_cap, int nal_ref_idc,
                     int nal_type, const uint8_t *rbsp, size_t rbsp_len,
                     int long_start_code)
{
    size_t n = 0, i;
    int zeros = 0;
    if (out_cap < rbsp_len + rbsp_len / 2 + 8)
        return 0;
    if (long_start_code) {
        out[n++] = 0; out[n++] = 0; out[n++] = 0; out[n++] = 1;
    } else {
        out[n++] = 0; out[n++] = 0; out[n++] = 1;
    }
    out[n++] = (uint8_t)((nal_ref_idc << 5) | (nal_type & 0x1f));
    for (i = 0; i < rbsp_len; i++) {
        uint8_t b = rbsp[i];
        if (zeros >= 2 && b <= 3) {
            out[n++] = 0x03;
            zeros = 0;
        }
        out[n++] = b;
        zeros = (b == 0) ? zeros + 1 : 0;
    }
    return n;
}

/* ------------------------------------------------------------------ */
/* SPS / PPS                                                           */
/* ------------------------------------------------------------------ */
/* Pick the smallest level that satisfies the frame size, frame rate and
 * bit rate of the stream (Table A-1). */
static int pick_level(int mb_w, int mb_h, int fps_num, int fps_den,
                      int bitrate_kbps)
{
    /* [level_idc][MaxFS][MaxMBPS][MaxBR(kbit/s)] for Baseline/Main/High   */
    static const struct { int idc, fs, mbs, br; } tab[] = {
        { 10,  99,   1485,    64 }, { 11,  396,   3000,   192 },
        { 12,  396,   6000,   384 }, { 13,  396,  11880,   768 },
        { 20,  396,  11880,  2000 }, { 21,  792,  19800,  4000 },
        { 22, 1620,  40500,  4000 }, { 30, 1620,  40500, 10000 },
        { 31, 3600, 108000, 14000 }, { 40, 5120, 216000, 20000 },
        { 41, 8192, 245760, 20000 }, { 42, 8704, 522240, 50000 },
        { 50, 22080, 589824, 135000 }, { 51, 36864, 983040, 240000 },
    };
    int fs = mb_w * mb_h;
    int64_t mbs = (int64_t)fs * fps_num / fps_den;
    int br = bitrate_kbps > 0 ? bitrate_kbps : 20000;
    size_t i;
    for (i = 0; i < sizeof(tab) / sizeof(tab[0]); i++) {
        if (fs <= tab[i].fs && mbs <= tab[i].mbs && br <= tab[i].br)
            return tab[i].idc;
    }
    return 51;
}

static void write_vui(aeg_bitwriter *bw, const aeg_params *p)
{
    uint32_t units = (uint32_t)(p->fps_den > 0 ? p->fps_den : 1);
    uint32_t scale = (uint32_t)(p->fps_num > 0 ? p->fps_num * 2 : 60);

    aeg_bw_put(bw, 1, 0);            /* aspect_ratio_info_present_flag      */
    aeg_bw_put(bw, 1, 0);            /* overscan_info_present_flag          */
    aeg_bw_put(bw, 1, 1);            /* video_signal_type_present_flag      */
    aeg_bw_put(bw, 3, 5);            /* video_format = unspecified          */
    aeg_bw_put(bw, 1, 0);            /* video_full_range_flag = 0 (TV)      */
    aeg_bw_put(bw, 1, 0);            /* colour_description_present_flag     */
    aeg_bw_put(bw, 1, 0);            /* chroma_loc_info_present_flag        */
    aeg_bw_put(bw, 1, 1);            /* timing_info_present_flag            */
    aeg_bw_put(bw, 32, units);       /* num_units_in_tick                   */
    aeg_bw_put(bw, 32, scale);       /* time_scale (2 ticks per frame)      */
    aeg_bw_put(bw, 1, 1);            /* fixed_frame_rate_flag               */
    aeg_bw_put(bw, 1, 0);            /* nal_hrd_parameters_present_flag     */
    aeg_bw_put(bw, 1, 0);            /* vcl_hrd_parameters_present_flag     */
    aeg_bw_put(bw, 1, 0);            /* pic_struct_present_flag             */
    aeg_bw_put(bw, 1, 1);            /* bitstream_restriction_flag          */
    aeg_bw_put(bw, 1, 1);            /* motion_vectors_over_pic_boundaries  */
    aeg_bw_ue(bw, 0);                /* max_bytes_per_pic_denom             */
    aeg_bw_ue(bw, 0);                /* max_bits_per_mb_denom               */
    aeg_bw_ue(bw, 16);               /* log2_max_mv_length_horizontal       */
    aeg_bw_ue(bw, 16);               /* log2_max_mv_length_vertical         */
    aeg_bw_ue(bw, 0);                /* max_num_reorder_frames              */
    aeg_bw_ue(bw, 1);                /* max_dec_frame_buffering             */
}

size_t aeg_write_sps(uint8_t *rbsp, size_t cap, const aeg_params *p,
                     int mb_w, int mb_h, int frame_num_bits, int level_idc)
{
    aeg_bitwriter bw;
    int coded_w = mb_w * 16, coded_h = mb_h * 16;
    int crop_r = (coded_w - p->width) / 2;
    int crop_b = (coded_h - p->height) / 2;

    if (level_idc <= 0)
        level_idc = pick_level(mb_w, mb_h, p->fps_num, p->fps_den,
                               p->bitrate_kbps);

    /* High profile as soon as the 8x8 transform is on -- it is the profile that
     * introduced transform_size_8x8_flag.  CABAC without it is Main profile;
     * only a stream with neither tool is Baseline (the rest of the syntax stays
     * inside the Baseline tool set, so no further constraints apply). */
    const int high = p->transform_8x8 != 0;
    const int main = !high && p->cabac;

    aeg_bw_init(&bw, rbsp, cap, 0);
    aeg_bw_put(&bw, 8, high ? 100 : (main ? 77 : 66)); /* profile_idc         */
    aeg_bw_put(&bw, 8, (high || main) ? 0x00 : 0xc0); /* constraint_set0/1   */
    aeg_bw_put(&bw, 8, (uint32_t)level_idc);
    aeg_bw_ue(&bw, 0);                   /* seq_parameter_set_id            */
    if (high) {
        aeg_bw_ue(&bw, 1);               /* chroma_format_idc = 4:2:0       */
        aeg_bw_ue(&bw, 0);               /* bit_depth_luma_minus8           */
        aeg_bw_ue(&bw, 0);               /* bit_depth_chroma_minus8         */
        aeg_bw_put(&bw, 1, 0);           /* qpprime_y_zero_transform_bypass */
        aeg_bw_put(&bw, 1, 0);           /* seq_scaling_matrix_present_flag */
    }
    aeg_bw_ue(&bw, (uint32_t)(frame_num_bits - 4));
    aeg_bw_ue(&bw, 2);                   /* pic_order_cnt_type = 2          */
    aeg_bw_ue(&bw, 1);                   /* max_num_ref_frames              */
    aeg_bw_put(&bw, 1, 0);               /* gaps_in_frame_num_value_allowed */
    aeg_bw_ue(&bw, (uint32_t)(mb_w - 1));
    aeg_bw_ue(&bw, (uint32_t)(mb_h - 1));
    aeg_bw_put(&bw, 1, 1);               /* frame_mbs_only_flag             */
    aeg_bw_put(&bw, 1, 1);               /* direct_8x8_inference_flag       */
    if (crop_r || crop_b) {
        aeg_bw_put(&bw, 1, 1);           /* frame_cropping_flag             */
        aeg_bw_ue(&bw, 0);
        aeg_bw_ue(&bw, (uint32_t)crop_r);
        aeg_bw_ue(&bw, 0);
        aeg_bw_ue(&bw, (uint32_t)crop_b);
    } else {
        aeg_bw_put(&bw, 1, 0);
    }
    aeg_bw_put(&bw, 1, 1);               /* vui_parameters_present_flag     */
    write_vui(&bw, p);
    return aeg_bw_finish(&bw);
}

size_t aeg_write_pps(uint8_t *rbsp, size_t cap, int transform_8x8, int cabac)
{
    aeg_bitwriter bw;
    aeg_bw_init(&bw, rbsp, cap, 0);
    aeg_bw_ue(&bw, 0);          /* pic_parameter_set_id                    */
    aeg_bw_ue(&bw, 0);          /* seq_parameter_set_id                    */
    aeg_bw_put(&bw, 1, cabac ? 1u : 0u); /* entropy_coding_mode_flag        */
    aeg_bw_put(&bw, 1, 0);      /* bottom_field_pic_order_in_frame_present */
    aeg_bw_ue(&bw, 0);          /* num_slice_groups_minus1                 */
    aeg_bw_ue(&bw, 0);          /* num_ref_idx_l0_default_active_minus1    */
    aeg_bw_ue(&bw, 0);          /* num_ref_idx_l1_default_active_minus1    */
    aeg_bw_put(&bw, 1, 0);      /* weighted_pred_flag                      */
    aeg_bw_put(&bw, 2, 0);      /* weighted_bipred_idc                     */
    aeg_bw_se(&bw, 0);          /* pic_init_qp_minus26                     */
    aeg_bw_se(&bw, 0);          /* pic_init_qs_minus26                     */
    aeg_bw_se(&bw, 0);          /* chroma_qp_index_offset                  */
    aeg_bw_put(&bw, 1, 1);      /* deblocking_filter_control_present_flag  */
    aeg_bw_put(&bw, 1, 0);      /* constrained_intra_pred_flag             */
    aeg_bw_put(&bw, 1, 0);      /* redundant_pic_cnt_present_flag          */
    /* High profile extensions: only written when the 8x8 transform is on, and
     * then only to switch it on -- the scaling matrices stay flat (absent). */
    if (transform_8x8) {
        aeg_bw_put(&bw, 1, 1);  /* transform_8x8_mode_flag                 */
        aeg_bw_put(&bw, 1, 0);  /* pic_scaling_matrix_present_flag          */
        aeg_bw_se(&bw, 0);      /* second_chroma_qp_index_offset            */
    }
    return aeg_bw_finish(&bw);
}

/* ------------------------------------------------------------------ */
/* Slice header                                                        */
/* ------------------------------------------------------------------ */
/* Written into an in-progress writer so that slice data follows the
 * header without any alignment or trailing bits. */
void aeg_write_slice_header(aeg_bitwriter *bw, int slice_type, int frame_num,
                            int frame_num_bits, int idr, int idr_pic_id,
                            int cabac_init_idc, int slice_qp_delta,
                            int deblock_alpha, int deblock_beta,
                            int deblock_idc, int first_mb)
{
    aeg_bw_ue(bw, (uint32_t)first_mb); /* first_mb_in_slice                  */
    aeg_bw_ue(bw, (uint32_t)slice_type);
    aeg_bw_ue(bw, 0);                 /* pic_parameter_set_id               */
    aeg_bw_put(bw, frame_num_bits, (uint32_t)frame_num);
    if (idr)
        aeg_bw_ue(bw, (uint32_t)idr_pic_id);
    if (slice_type % 5 == 0) {        /* P slice                            */
        aeg_bw_put(bw, 1, 0);         /* num_ref_idx_active_override_flag   */
        aeg_bw_put(bw, 1, 0);         /* ref_pic_list_modification_flag_l0  */
    }
    if (idr) {
        aeg_bw_put(bw, 1, 0);         /* no_output_of_prior_pics_flag       */
        aeg_bw_put(bw, 1, 0);         /* long_term_reference_flag           */
    } else {
        aeg_bw_put(bw, 1, 0);         /* adaptive_ref_pic_marking_mode_flag */
    }
    /* cabac_init_idc is present in every CABAC-coded slice that is not an I
     * slice (7.3.3); it selects the context initialization table. */
    if (cabac_init_idc >= 0 && slice_type % 5 != 2 && slice_type % 5 != 4)
        aeg_bw_ue(bw, (uint32_t)cabac_init_idc);
    aeg_bw_se(bw, slice_qp_delta);    /* slice_qp_delta                     */
    aeg_bw_ue(bw, (uint32_t)deblock_idc);
    if (deblock_idc != 1) {
        aeg_bw_se(bw, deblock_alpha); /* slice_alpha_c0_offset_div2         */
        aeg_bw_se(bw, deblock_beta);  /* slice_beta_offset_div2             */
    }
}
