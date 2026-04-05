// OpenCL 1.2 FP32 core ops for legacy NVIDIA (no subgroups, no cl_khr_fp16).
// Used with GGML_OPENCL_USE_CLBLAST && GGML_OPENCL_LEGACY_NVIDIA.

//------------------------------------------------------------------------------
// diag_mask_inf (matches kernel_diag_mask_inf layout)
//------------------------------------------------------------------------------
kernel void kernel_legacy_diag_mask_inf_f32(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int n_past
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    int i02 = get_global_id(2);
    int i01 = get_global_id(1);
    int i00 = get_global_id(0);

    if (i00 > n_past + i01) {
        dst[i02*ne01*ne00 + i01*ne00 + i00] = -INFINITY;
    } else {
        dst[i02*ne01*ne00 + i01*ne00 + i00] = src0[i02*ne01*ne00 + i01*ne00 + i00];
    }
}

//------------------------------------------------------------------------------
// RMS norm: one work-item per slice (ne01 x ne02 x ne03); sum over ne00
//------------------------------------------------------------------------------
kernel void kernel_legacy_rms_norm_f32(
        global const float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        float eps
) {
    src0 = (global const float*)((global const char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    int i01 = get_global_id(0);
    int i02 = get_global_id(1);
    int i03 = get_global_id(2);
    if (i01 >= ne01 || i02 >= ne02 || i03 >= ne03) {
        return;
    }
    global const float * x = (global const float *)((global const char *)src0 + i01*nb01 + i02*nb02 + i03*nb03);
    global float * y = (global float *)((global char *)dst + i01*nb01 + i02*nb02 + i03*nb03);
    float sum = 0.0f;
    for (int i = 0; i < ne00; i++) {
        float v = x[i];
        sum += v * v;
    }
    float scale = rsqrt(sum / (float)ne00 + eps);
    for (int i = 0; i < ne00; i++) {
        y[i] = x[i] * scale;
    }
}

//------------------------------------------------------------------------------
// Elementwise (total length n)
//------------------------------------------------------------------------------
kernel void kernel_legacy_add_f32(
        global const float * a, ulong oa,
        global const float * b, ulong ob,
        global float * d, ulong od,
        int n
) {
    a = (global const float*)((global const char*)a + oa);
    b = (global const float*)((global const char*)b + ob);
    d = (global float*)((global char*)d + od);
    int i = get_global_id(0);
    if (i < n) {
        d[i] = a[i] + b[i];
    }
}

kernel void kernel_legacy_mul_f32(
        global const float * a, ulong oa,
        global const float * b, ulong ob,
        global float * d, ulong od,
        int n
) {
    a = (global const float*)((global const char*)a + oa);
    b = (global const float*)((global const char*)b + ob);
    d = (global float*)((global char*)d + od);
    int i = get_global_id(0);
    if (i < n) {
        d[i] = a[i] * b[i];
    }
}

kernel void kernel_legacy_scale_f32(
        global const float * a, ulong oa,
        global float * d, ulong od,
        int n,
        float s,
        float b
) {
    a = (global const float*)((global const char*)a + oa);
    d = (global float*)((global char*)d + od);
    int i = get_global_id(0);
    if (i < n) {
        d[i] = a[i] * s + b;
    }
}

kernel void kernel_legacy_cpy_f32(
        global const float * a, ulong oa,
        global float * d, ulong od,
        int n
) {
    a = (global const float*)((global const char*)a + oa);
    d = (global float*)((global char*)d + od);
    int i = get_global_id(0);
    if (i < n) {
        d[i] = a[i];
    }
}

//------------------------------------------------------------------------------
// Unary SILU / GELU (tanh approximation, same as upstream gelu kernel)
//------------------------------------------------------------------------------
kernel void kernel_legacy_silu_f32(
        global const float * x, ulong ox,
        global float * d, ulong od,
        int n
) {
    x = (global const float*)((global const char*)x + ox);
    d = (global float*)((global char*)d + od);
    int i = get_global_id(0);
    if (i < n) {
        float v = x[i];
        d[i] = v / (1.0f + exp(-v));
    }
}

kernel void kernel_legacy_gelu_f32(
        global const float * x, ulong ox,
        global float * d, ulong od,
        int n
) {
    x = (global const float*)((global const char*)x + ox);
    d = (global float*)((global char*)d + od);
    const float SQRT2_OVER_PI = 0.79788456080286535587989211986876f;
    int i = get_global_id(0);
    if (i < n) {
        float xi = x[i];
        d[i] = 0.5f * xi * (1.0f + tanh(SQRT2_OVER_PI * (xi + 0.044715f * xi * xi * xi)));
    }
}

//------------------------------------------------------------------------------
// Softmax row (FP32 mask only): subgroup-free version of kernel_soft_max
//------------------------------------------------------------------------------
kernel void kernel_legacy_softmax_f32(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        int use_mask,
        global char * src2,
        ulong offset2,
        int use_past,
        global char * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne12,
        int ne13,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        float scale,
        float max_bias,
        float m0,
        float m1,
        int n_head_log2
) {
    // Use global_id with global size (ne01, ne02, ne03) and null local size so row indices are stable.
    int i03 = get_global_id(2);
    int i02 = get_global_id(1);
    int i01 = get_global_id(0);

    int i13 = i03 % ne13;
    int i12 = i02 % ne12;

    global float * psrc0 = (global float *)(src0 + offset0 + i01*nb01 + i02*nb02 + i03*nb03);
    global float * pmask = use_mask ? (global float *)(src1 + offset1 + i01*nb11 + i12*nb12 + i13*nb13) : 0;
    global float * psrc2 = use_past ? (global float *)(src2 + offset2) : 0;
    global float * pdst  = (global float *)(dst + offsetd + i01*nb1 + i02*nb2 + i03*nb3);

    float slope = 1.0f;
    if (max_bias > 0.0f) {
        int h = i02;
        float base = h < n_head_log2 ? m0 : m1;
        int expv = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;
        slope = pow(base, (float)expv);
    }

    float lmax = psrc2 ? psrc2[i02] : -INFINITY;
    for (int i00 = 0; i00 < ne00; i00++) {
        lmax = fmax(lmax, psrc0[i00]*scale + (pmask ? slope*pmask[i00] : 0.0f));
    }

    float lsum = 0.0f;
    for (int i00 = 0; i00 < ne00; i00++) {
        float v = (psrc0[i00]*scale + (pmask ? slope*pmask[i00] : 0.0f)) - lmax;
        float e = exp(v);
        pdst[i00] = e;
        lsum += e;
    }
    if (psrc2) {
        lsum += exp(psrc2[i02] - lmax);
    }
    for (int i00 = 0; i00 < ne00; i00++) {
        pdst[i00] /= lsum;
    }
}

//------------------------------------------------------------------------------
// RoPE FP32: normal (is_neox=0) and NeoX (is_neox=1). YaRN corr dims from host.
//------------------------------------------------------------------------------
static float legacy_rope_yarn_ramp(float low, float high, int i0) {
    float y = ((i0 / 2) - low) / fmax(0.001f, high - low);
    return 1.0f - fmin(1.0f, fmax(0.0f, y));
}

static void legacy_rope_yarn(
    float theta_extrap, float freq_scale, float corr0, float corr1, int i0, float ext_factor, float mscale,
    float * cos_theta, float * sin_theta
) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = legacy_rope_yarn_ramp(corr0, corr1, i0) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    *cos_theta = cos(theta) * mscale;
    *sin_theta = sin(theta) * mscale;
}

kernel void kernel_legacy_rope_f32(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * src2,
        ulong offset2,
        int use_src2,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        int n_dims,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        float corr0,
        float corr1,
        int is_neox
) {
    (void)n_ctx_orig;
    (void)beta_fast;
    (void)beta_slow;

    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    src2 = (global float*)((global char*)src2 + offset2);
    dst = (global float*)((global char*)dst + offsetd);

    int i3 = get_global_id(2);
    int i2 = get_global_id(1);
    int i1b = get_global_id(0);

    if (i1b >= ne01 || i2 >= ne02 || i3 >= ne03) {
        return;
    }

    global int * pos = src1;
    float theta_base = (float) pos[i2];
    float inv_ndims = -1.f/n_dims;

    for (int i0 = 0; i0 < ne0; i0 += 2) {
        if (i0 < n_dims) {
            int ic = i0/2;
            float theta = theta_base * pow(freq_base, inv_ndims*i0);
            float freq_factor = use_src2 ? src2[ic] : 1.0f;
            float ct, st;
            legacy_rope_yarn(theta/freq_factor, freq_scale, corr0, corr1, i0, ext_factor, attn_factor, &ct, &st);

            if (!is_neox) {
                global float * src       = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1b*nb01 + i0*nb00);
                global float * dst_data  = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1b*nb1  + i0*nb0);
                float x0 = src[0];
                float x1 = src[1];
                dst_data[0] = x0*ct - x1*st;
                dst_data[1] = x0*st + x1*ct;
            } else {
                global float * src      = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1b*nb01 + ic*nb00);
                global float * dst_data = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1b*nb1  + ic*nb0);
                float x0 = src[0];
                float x1 = src[n_dims/2];
                dst_data[0]        = x0*ct - x1*st;
                dst_data[n_dims/2] = x0*st + x1*ct;
            }
        } else {
            global float * src      = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1b*nb01 + i0*nb00);
            global float * dst_data = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1b*nb1  + i0*nb0);
            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

//------------------------------------------------------------------------------
// FP16 storage bits -> float (no cl_khr_fp16)
//------------------------------------------------------------------------------
float legacy_f16_as_f32(ushort h) {
    uint s = (uint)(h >> 15);
    uint e = (uint)((h >> 10) & 31u);
    uint m = (uint)(h & 1023u);
    uint u;
    if (e == 0u) {
        if (m == 0u) {
            u = s << 31;
        } else {
            e = 1u;
            while ((m & 0x400u) == 0u) {
                m <<= 1u;
                e--;
            }
            m &= 0x3ffu;
            u = (s << 31) | ((e + (127u - 15u)) << 23) | (m << 13);
        }
    } else if (e == 31u) {
        u = (s << 31) | (255u << 23) | (m << 13);
    } else {
        u = (s << 31) | ((e + (127u - 15u)) << 23) | (m << 13);
    }
    return as_float(u);
}

//------------------------------------------------------------------------------
// Native FP32 GEMM: C (M x N, col-major, ldc) = (A^T) * B
// A is K x M stored col-major (lda = K), B is K x N (ldb = K).
//------------------------------------------------------------------------------
kernel void kernel_legacy_gemm_f32(
        global const float * A, ulong offA,
        global const float * B, ulong offB,
        global float * C, ulong offC,
        int K, int M, int N, int ldc
) {
    A = (global const float *)((global const char *)A + offA);
    B = (global const float *)((global const char *)B + offB);
    C = (global float *)((global char *)C + offC);

    size_t im = get_global_id(0);
    size_t jn = get_global_id(1);
    if (im >= (size_t)M || jn >= (size_t)N) {
        return;
    }
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[(size_t)k + im * (size_t)K] * B[(size_t)k + jn * (size_t)K];
    }
    C[im + jn * (size_t)ldc] = sum;
}

//------------------------------------------------------------------------------
// get_rows (token embedding gather) — FP32 / Q4_0 / Q6_K, no half extension
//------------------------------------------------------------------------------
kernel void kernel_legacy_get_rows_f32(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global void *)((global char *)src0 + offset0);
    src1 = (global int *)((global char *)src1 + offset1);
    dst = (global float *)((global char *)dst + offsetd);

    int i10 = get_group_id(0);
    int i11 = get_group_id(1);
    int i12 = get_group_id(2);

    int r = ((global int *)((global char *)src1 + i12 * nb12 + i11 * nb11 + i10 * nb10))[0];

    int i02 = i11;
    int i03 = i12;

    for (int ind = get_local_id(0); ind < ne00; ind += get_local_size(0)) {
        ((global float *)((global char *)dst + i12 * nb3 + i11 * nb2 + i10 * nb1))[ind] =
            ((global float *)((global char *)src0 + r * nb01 + i02 * nb02 + i03 * nb03))[ind];
    }
}

#define QK4_0_GR 32
struct __attribute__((packed)) block_q4_0_gr {
    ushort d;
    uchar qs[QK4_0_GR / 2];
};

kernel void kernel_legacy_get_rows_q4_0(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global void *)((global char *)src0 + offset0);
    src1 = (global int *)((global char *)src1 + offset1);
    dst = (global float *)((global char *)dst + offsetd);

    int i10 = get_group_id(0);
    int i11 = get_group_id(1);
    int i12 = get_group_id(2);

    int r = ((global int *)((global char *)src1 + i12 * nb12 + i11 * nb11 + i10 * nb10))[0];

    int i02 = i11;
    int i03 = i12;

    global char * row_base = (global char *)src0 + r * nb01 + i02 * nb02 + i03 * nb03;
    global float * drow = (global float *)((global char *)dst + i12 * nb3 + i11 * nb2 + i10 * nb1);

    for (int t = get_local_id(0); t < ne00; t += get_local_size(0)) {
        int ib = t / QK4_0_GR;
        int jrem = t - ib * QK4_0_GR;
        global struct block_q4_0_gr * bl = (global struct block_q4_0_gr *)(row_base + (size_t)ib * sizeof(struct block_q4_0_gr));
        float d = legacy_f16_as_f32(bl->d);
        uchar vui = bl->qs[jrem / 2];
        float v = (jrem & 1) == 0
            ? (((float)(vui & (uchar)15)) - 8.0f) * d
            : (((float)(vui >> 4)) - 8.0f) * d;
        drow[t] = v;
    }
}

#define QK_K_GR 256

typedef struct __attribute__((packed)) {
    uchar ql[128];
    uchar qh[64];
    char scales[16];
    ushort d;
} block_q6_K_gr;

kernel void kernel_legacy_get_rows_q6_k(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global void *)((global char *)src0 + offset0);
    src1 = (global int *)((global char *)src1 + offset1);
    dst = (global float *)((global char *)dst + offsetd);

    int i10 = get_group_id(0);
    int i11 = get_group_id(1);
    int i12 = get_group_id(2);

    int r = ((global int *)((global char *)src1 + i12 * nb12 + i11 * nb11 + i10 * nb10))[0];

    int i02 = i11;
    int i03 = i12;

    if (get_local_id(0) != 0) {
        return;
    }

    global const block_q6_K_gr * x = (global const block_q6_K_gr *)((global char *)src0 + r * nb01 + i02 * nb02 + i03 * nb03);
    global float * y_base = (global float *)((global char *)dst + i12 * nb3 + i11 * nb2 + i10 * nb1);

    const int nb = ne00 / QK_K_GR;

    for (int i = 0; i < nb; i++) {
        const float d_sc = legacy_f16_as_f32(x[i].d);
        global float * y = y_base + i * QK_K_GR;
        global const uchar * ql = x[i].ql;
        global const uchar * qh = x[i].qh;
        global const char * sc = x[i].scales;

        for (int n = 0; n < QK_K_GR; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int q1 = (int)((char)((ql[l + 0] & (uchar)15) | (((qh[l] >> 0) & (uchar)3) << 4)) - 32);
                int q2 = (int)((char)((ql[l + 32] & (uchar)15) | (((qh[l] >> 2) & (uchar)3) << 4)) - 32);
                int q3 = (int)((char)((ql[l + 0] >> 4) | (((qh[l] >> 4) & (uchar)3) << 4)) - 32);
                int q4 = (int)((char)((ql[l + 32] >> 4) | (((qh[l] >> 6) & (uchar)3) << 4)) - 32);
                y[l + 0]  = d_sc * (float)sc[is + 0] * (float)q1;
                y[l + 32] = d_sc * (float)sc[is + 2] * (float)q2;
                y[l + 64] = d_sc * (float)sc[is + 4] * (float)q3;
                y[l + 96] = d_sc * (float)sc[is + 6] * (float)q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}
