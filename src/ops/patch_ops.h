#pragma once

#include <immintrin.h>
#include <algorithm>

namespace cromwell {
namespace ops {

/**
 * @brief AVX2-optimized image patching operations for vision encoder
 *
 * Operations:
 * - Patch embedding (4×4 conv, stride=4)
 * - Patch normalization
 * - Positional embedding addition
 *
 * All operations use AVX2 for 4× speedup over scalar
 */
class PatchOps {
public:
    /**
     * @brief 4×4 patch embedding with AVX2
     * @param input RGB image [H, W, 3] NHWC
     * @param H Image height
     * @param W Image width
     * @param kernel Convolution kernel [4, 4, 3, out_channels] NHWC
     * @param output Patches [H/4, W/4, out_channels] NHWC
     * @param out_channels Number of output channels (96 for stem)
     */
    static void patch_embed_4x4_avx2(
        const float* input,
        int H, int W,
        const float* kernel,
        float* output,
        int out_channels
    );

    /**
     * @brief Flattened patch embedding for ViT
     * @param input Spatial features [H, W, C] NHWC
     * @param H Height
     * @param W Width
     * @param C Channels
     * @param output Flattened patches [H*W, C]
     */
    static void flatten_patches(
        const float* input,
        int H, int W, int C,
        float* output
    );

    /**
     * @brief Add positional embeddings to patches
     * @param patches Input patches [N, dim]
     * @param pos_embed Positional embeddings [N, dim]
     * @param output Output [N, dim]
     * @param N Number of patches
     * @param dim Embedding dimension
     */
    static void add_positional_embeddings(
        const float* patches,
        const float* pos_embed,
        float* output,
        int N,
        int dim
    );

    /**
     * @brief 2D positional embedding generation
     * @param pos_embed Output positional embeddings [H*W, dim]
     * @param H Grid height
     * @param W Grid width
     * @param dim Embedding dimension
     */
    static void generate_2d_pos_embed(
        float* pos_embed,
        int H, int W,
        int dim
    );
};

/**
 * @brief Depthwise separable convolution (MBConv) with AVX2
 */
class DepthwiseConv2D {
public:
    /**
     * @brief Depthwise 3×3 convolution with AVX2
     * @param input Input [H, W, C] NHWC
     * @param H Height
     * @param W Width
     * @param C Channels
     * @param kernel Kernel [3, 3, C] (depthwise)
     * @param output Output [H, W, C] NHWC
     */
    static void depthwise_conv_3x3_avx2(
        const float* input,
        int H, int W, int C,
        const float* kernel,
        float* output
    );

    /**
     * @brief Pointwise 1×1 convolution with AVX2
     * @param input Input [H, W, C_in] NHWC
     * @param H Height
     * @param W Width
     * @param C_in Input channels
     * @param kernel Kernel [C_in, C_out]
     * @param C_out Output channels
     * @param output Output [H, W, C_out] NHWC
     */
    static void pointwise_conv_1x1_avx2(
        const float* input,
        int H, int W, int C_in,
        const float* kernel,
        int C_out,
        float* output
    );
};

/**
 * @brief Squeeze-and-Excitation (SE) attention with AVX2
 */
class SEAttention {
public:
    /**
     * @brief SE attention (channel-wise)
     * @param input Input [H, W, C] NHWC
     * @param H Height
     * @param W Width
     * @param C Channels
     * @param fc1 First FC layer weights [C, C/r]
     * @param fc2 Second FC layer weights [C/r, C]
     * @param reduction Reduction ratio (typically 4)
     * @param output Output [H, W, C] NHWC
     */
    static void se_attention_avx2(
        const float* input,
        int H, int W, int C,
        const float* fc1,
        const float* fc2,
        int reduction,
        float* output
    );
};

/**
 * @brief AVX2 helper functions for patch operations
 */
namespace avx2_utils {
    // Horizontal sum of AVX2 register
    inline float horizontal_sum_ps(__m256 v) {
        __m128 sum = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
        sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
        return _mm_cvtss_f32(sum);
    }

    // Swish activation (SiLU)
    inline __m256 swish_ps(__m256 x) {
        __m256 sigmoid = _mm256_div_ps(_mm256_set1_ps(1.0f),
                                     _mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_exp_ps(_mm256_sub_ps(_mm256_set1_ps(0.0f), x))));
        return _mm256_mul_ps(x, sigmoid);
    }

    // GeLU activation
    inline __m256 gelu_ps(__m256 x) {
        const float sqrt_2_over_pi = 0.7978845608f;
        const float coeff = 0.044715f;
        __m256 x_cubed = _mm256_mul_ps(x, _mm256_mul_ps(x, x));
        __m256 tanh_arg = _mm256_mul_ps(_mm256_fmadd_ps(x_cubed, _mm256_set1_ps(coeff), x), _mm256_set1_ps(sqrt_2_over_pi));
        __m256 tanh_out = _mm256_mul_ps(_mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_exp_ps(tanh_arg)),
                                       _mm256_rsqrt_ps(_mm256_add_ps(_mm256_set1_ps(1.0f), _mm256_exp_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), tanh_arg)))));
        return _mm256_mul_ps(x, _mm256_mul_ps(_mm256_set1_ps(0.5f), _mm256_add_ps(_mm256_set1_ps(1.0f), tanh_out)));
    }

    // Fast exp approximation
    inline __m256 exp_ps(__m256 x) {
        const float log2_e = 1.44269504f;
        __m256 x_log2_e = _mm256_mul_ps(x, _mm256_set1_ps(log2_e));
        __m256i x_i = _mm256_cvttps_epi32(x_log2_e);
        __m256 x_f = _mm256_sub_ps(x_log2_e, _mm256_cvtepi32_ps(x_i));
        __m256 exp_x_f = _mm256_add_ps(_mm256_set1_ps(1.0f),
                                       _mm256_mul_ps(x_f, _mm256_set1_ps(0.693359375f)));  // ln(2)
        int exp_x_i[8];
        _mm256_storeu_si256((__m256i*)exp_x_i, x_i);
        for (int i = 0; i < 8; i++) {
            exp_x_i[i] = (exp_x_i[i] + 127) << 23;
        }
        __m256 result = _mm256_castsi256_ps(_mm256_loadu_si256((__m256i*)exp_x_i));
        return _mm256_mul_ps(result, exp_x_f);
    }
}

} // namespace ops
} // namespace cromwell
