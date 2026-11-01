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

static void write_vui(aeg_bitwriter *bw, const aeg_params *p, int max_refs)
{
    uint32_t units = (uint32_t)(p->fps_den > 0 ? p->fps_den : 1);
    uint32_t scale = (uint32_t)(p->fps_num > 0 ? p->fps_num * 2 : 60);

    /* Table E-1: the sample aspect ratios that have an index of their own */
    static const unsigned char sar_tab[17][2] = {
        { 0, 0 }, { 1, 1 }, { 12, 11 }, { 10, 11 }, { 16, 11 }, { 40, 33 },
        { 24, 11 }, { 20, 11 }, { 32, 11 }, { 80, 33 }, { 18, 11 }, { 15, 11 },
        { 64, 33 }, { 160, 99 }, { 4, 3 }, { 3, 2 }, { 2, 1 } };
    const int colour = p->colorprim >= 0 || p->transfer >= 0 ||
                       p->colormatrix >= 0;

    if (p->sar_w > 0 && p->sar_h > 0) {
        int sw = p->sar_w, sh = p->sar_h, a = sw, b = sh, idc = 255, i;
        while (b) { int t = a % b; a = b; b = t; }  /* reduce */
        sw /= a, sh /= a;
        while (sw > 65535 || sh > 65535)
            sw = (sw + 1) / 2, sh = (sh + 1) / 2;
        for (i = 1; i < 17; i++)
            if (sar_tab[i][0] == sw && sar_tab[i][1] == sh)
                idc = i;
        aeg_bw_put(bw, 1, 1);        /* aspect_ratio_info_present_flag      */
        aeg_bw_put(bw, 8, (uint32_t)idc);
        if (idc == 255) {            /* Extended_SAR                        */
            aeg_bw_put(bw, 16, (uint32_t)sw);
            aeg_bw_put(bw, 16, (uint32_t)sh);
        }
    } else {
        aeg_bw_put(bw, 1, 0);        /* aspect_ratio_info_present_flag      */
    }
    aeg_bw_put(bw, 1, 0);            /* overscan_info_present_flag          */
    aeg_bw_put(bw, 1, 1);            /* video_signal_type_present_flag      */
    aeg_bw_put(bw, 3, 5);            /* video_format = unspecified          */
    aeg_bw_put(bw, 1, p->fullrange ? 1 : 0); /* video_full_range_flag       */
    aeg_bw_put(bw, 1, colour ? 1 : 0); /* colour_description_present_flag   */
    if (colour) {                    /* 2 = unspecified                     */
        aeg_bw_put(bw, 8, (uint32_t)(p->colorprim >= 0 ? p->colorprim : 2));
        aeg_bw_put(bw, 8, (uint32_t)(p->transfer >= 0 ? p->transfer : 2));
        aeg_bw_put(bw, 8, (uint32_t)(p->colormatrix >= 0 ? p->colormatrix : 2));
    }
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
    /* With B pictures the decoder has to hold pictures back before it can
     * output them, and it must be told how many.  Without B pictures both
     * numbers are 0 and 1 as before, so the SPS of a stream without them is
     * byte for byte what it was. */
    aeg_bw_ue(bw, (uint32_t)(p->bframes > 0 ? p->bframes : 0));
    aeg_bw_ue(bw, (uint32_t)(p->bframes > 0 ? max_refs + p->bframes : 1));
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
    /* pic_order_cnt_type 2 derives the picture order count from frame_num,
     * which counts *decoding* order: with B pictures the two orders differ, so
     * the count has to be sent explicitly (type 0, an 8-bit lsb per slice). */
    const int nrefs = (p->bframes > 0 && p->b_pyramid) ? 3 :
                      ((p->bframes > 0 || p->refs >= 2) ? 2 : 1);
    if (p->bframes > 0) {
        aeg_bw_ue(&bw, 0);               /* pic_order_cnt_type = 0          */
        aeg_bw_ue(&bw, 4);               /* log2_max_pic_order_cnt_lsb_minus4 */
    } else {
        aeg_bw_ue(&bw, 2);               /* pic_order_cnt_type = 2          */
    }
    /* Two reference pictures are alive at once with B pictures (the anchor the
     * B pictures predict backwards from and the one they predict forwards
     * from) and with two P references (list0[1] is the anchor before the
     * previous one).  With max_num_ref_frames = 1 the sliding window drops
     * that picture before the next P picture is decoded, and a decoder then
     * conceals ref_idx 1 with whatever it still has -- a silent mismatch. */
    aeg_bw_ue(&bw, (uint32_t)nrefs);
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
    write_vui(&bw, p, nrefs);
    return aeg_bw_finish(&bw);
}

size_t aeg_write_pps(uint8_t *rbsp, size_t cap, int transform_8x8, int cabac,
                     int weighted_pred)
{
    aeg_bitwriter bw;
    aeg_bw_init(&bw, rbsp, cap, 0);
    /* PPS 1 is the weighted-prediction twin of PPS 0 (see the slice
     * header): only the pictures that use weights pay for the table. */
    aeg_bw_ue(&bw, weighted_pred ? 1u : 0u); /* pic_parameter_set_id      */
    aeg_bw_ue(&bw, 0);          /* seq_parameter_set_id                    */
    aeg_bw_put(&bw, 1, cabac ? 1u : 0u); /* entropy_coding_mode_flag        */
    aeg_bw_put(&bw, 1, 0);      /* bottom_field_pic_order_in_frame_present */
    aeg_bw_ue(&bw, 0);          /* num_slice_groups_minus1                 */
    aeg_bw_ue(&bw, 0);          /* num_ref_idx_l0_default_active_minus1    */
    aeg_bw_ue(&bw, 0);          /* num_ref_idx_l1_default_active_minus1    */
    aeg_bw_put(&bw, 1, weighted_pred ? 1u : 0u); /* weighted_pred_flag */
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
/* pred_weight_table() (7.3.3.2) of a P slice: present whenever the PPS sets
 * weighted_pred_flag, one entry per active list 0 reference, 4:2:0 chroma. */
static void write_pred_weight_table(aeg_bitwriter *bw, const aeg_pwt *t,
                                    int nref)
{
    int i, c;

    aeg_bw_ue(bw, (uint32_t)t->luma_denom);
    aeg_bw_ue(bw, (uint32_t)t->chroma_denom);
    for (i = 0; i < nref; i++) {
        aeg_bw_put(bw, 1, t->lflag[i] ? 1u : 0u);
        if (t->lflag[i]) {
            aeg_bw_se(bw, t->lw[i]);
            aeg_bw_se(bw, t->lo[i]);
        }
        aeg_bw_put(bw, 1, t->cflag[i] ? 1u : 0u);
        if (t->cflag[i])
            for (c = 0; c < 2; c++) {
                aeg_bw_se(bw, t->cw[i][c]);
                aeg_bw_se(bw, t->co[i][c]);
            }
    }
}

/* Written into an in-progress writer so that slice data follows the
 * header without any alignment or trailing bits. */
void aeg_write_slice_header(aeg_bitwriter *bw, int slice_type, int frame_num,
                            int frame_num_bits, int idr, int idr_pic_id,
                            int cabac_init_idc, int slice_qp_delta,
                            int deblock_alpha, int deblock_beta,
                            int deblock_idc, int first_mb, int num_ref_idx_l0_active,
                            const aeg_pwt *pwt)
{
    aeg_bw_ue(bw, (uint32_t)first_mb); /* first_mb_in_slice                  */
    aeg_bw_ue(bw, (uint32_t)slice_type);
    aeg_bw_ue(bw, pwt ? 1u : 0u);    /* pic_parameter_set_id (1 = weighted) */
    aeg_bw_put(bw, frame_num_bits, (uint32_t)frame_num);
    if (idr)
        aeg_bw_ue(bw, (uint32_t)idr_pic_id);
    if (slice_type % 5 == 0) {        /* P slice                            */
        if (num_ref_idx_l0_active > 1) {
            aeg_bw_put(bw, 1, 1);     /* num_ref_idx_active_override_flag   */
            aeg_bw_ue(bw, (uint32_t)(num_ref_idx_l0_active - 1));
        } else {
            aeg_bw_put(bw, 1, 0);     /* num_ref_idx_active_override_flag   */
        }
        aeg_bw_put(bw, 1, 0);         /* ref_pic_list_modification_flag_l0  */
        if (pwt)
            write_pred_weight_table(bw, pwt, num_ref_idx_l0_active);
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

/* The same header for a picture that carries a picture order count, i.e. every
 * picture of a stream with B pictures.  The extra syntax is inserted exactly
 * where 7.3.3 puts it: pic_order_cnt_lsb after idr_pic_id, the direct-mode flag
 * for B slices, then the reference list bookkeeping (one modification flag per
 * list), and finally the reference marking, which a B picture does not have at
 * all because it is not a reference picture (nal_ref_idc = 0). */
void aeg_write_slice_header_poc(aeg_bitwriter *bw, int slice_type_in,
                                int frame_num, int frame_num_bits, int idr,
                                int idr_pic_id, int poc_lsb, int poc_lsb_bits,
                                int is_ref, int cabac_init_idc,
                                int slice_qp_delta, int deblock_alpha,
                                int deblock_beta, int deblock_idc, int first_mb,
                                int num_ref_idx_l0_active, int direct_spatial,
                                const aeg_pwt *pwt, const aeg_refctl *rc)
{
    int nos = slice_type_in % 5;      /* 0 = P, 1 = B, 2 = I */

    aeg_bw_ue(bw, (uint32_t)first_mb); /* first_mb_in_slice                  */
    aeg_bw_ue(bw, (uint32_t)slice_type_in);
    aeg_bw_ue(bw, pwt ? 1u : 0u);    /* pic_parameter_set_id (1 = weighted) */
    aeg_bw_put(bw, frame_num_bits, (uint32_t)frame_num);
    if (idr)
        aeg_bw_ue(bw, (uint32_t)idr_pic_id);
    aeg_bw_put(bw, poc_lsb_bits, (uint32_t)poc_lsb);
    /* bottom_field_pic_order_in_frame_present_flag is 0 in the PPS, so there
     * is no delta_pic_order_cnt_bottom. */
    /* Temporal direct scales the co-located picture's vectors by the picture
     * order counts (8.4.1.2.3, b_direct_mv in the encoder); spatial direct
     * predicts them from the neighbours (8.4.1.2.2, b_direct_spatial). */
    if (nos == 1)
        aeg_bw_put(bw, 1, direct_spatial ? 1u : 0u); /* direct_spatial_mv_pred_flag */
    if (nos == 0 || nos == 1) {
        if (nos == 0 && num_ref_idx_l0_active > 1) {
            aeg_bw_put(bw, 1, 1);     /* num_ref_idx_active_override_flag   */
            aeg_bw_ue(bw, (uint32_t)(num_ref_idx_l0_active - 1));
        } else {
            aeg_bw_put(bw, 1, 0);     /* num_ref_idx_active_override_flag   */
        }
        if (nos == 0 && rc &&
            (num_ref_idx_l0_active > 1 ? rc->n_mod : rc->n_mod_1ref) > 0) {
            int k, n = num_ref_idx_l0_active > 1 ? rc->n_mod : rc->n_mod_1ref;
            aeg_bw_put(bw, 1, 1);     /* ref_pic_list_modification_flag_l0  */
            for (k = 0; k < n; k++) {
                aeg_bw_ue(bw, (uint32_t)rc->mod_idc[k]); /* 0: -, 1: +      */
                aeg_bw_ue(bw, (uint32_t)rc->mod_abs[k]); /* abs_diff_pic_num_minus1 */
            }
            aeg_bw_ue(bw, 3);         /* end of the modifications           */
        } else {
            aeg_bw_put(bw, 1, 0);     /* ref_pic_list_modification_flag_l0  */
        }
        if (nos == 1)
            aeg_bw_put(bw, 1, 0);     /* ref_pic_list_modification_flag_l1  */
    }
    if (nos == 0 && pwt)
        write_pred_weight_table(bw, pwt, num_ref_idx_l0_active);
    if (is_ref) {
        if (idr) {
            aeg_bw_put(bw, 1, 0);     /* no_output_of_prior_pics_flag       */
            aeg_bw_put(bw, 1, 0);     /* long_term_reference_flag           */
        } else if (rc && rc->n_mmco > 0) {
            int k;
            aeg_bw_put(bw, 1, 1);     /* adaptive_ref_pic_marking_mode_flag */
            for (k = 0; k < rc->n_mmco; k++) {
                aeg_bw_ue(bw, 1);     /* mark a short-term picture unused   */
                aeg_bw_ue(bw, (uint32_t)rc->mmco_diff[k]);
            }
            aeg_bw_ue(bw, 0);         /* end of the operations              */
        } else {
            aeg_bw_put(bw, 1, 0);     /* adaptive_ref_pic_marking_mode_flag */
        }
    }
    if (cabac_init_idc >= 0 && nos != 2)
        aeg_bw_ue(bw, (uint32_t)cabac_init_idc);
    aeg_bw_se(bw, slice_qp_delta);    /* slice_qp_delta                     */
    aeg_bw_ue(bw, (uint32_t)deblock_idc);
    if (deblock_idc != 1) {
        aeg_bw_se(bw, deblock_alpha);
        aeg_bw_se(bw, deblock_beta);
    }
}
