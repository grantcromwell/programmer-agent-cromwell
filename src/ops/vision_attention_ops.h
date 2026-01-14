#pragma once

#include <immintrin.h>

namespace cromwell {
namespace ops {

/**
 * @brief Windowed attention for Vision Transformer with AVX2
 *
 * Windowed attention reduces complexity from O(N²) to O(W² * num_windows)
 * where W is the window size (typically 16×16)
 */
class WindowedAttention {
public:
    /**
     * @brief Windowed multi-head self-attention
     * @param input Input patches [H, W, C] where C = num_heads * head_dim
     * @param H Grid height
     * @param W Grid width
     * @param C Channels (hidden_dim)
     * @param num_heads Number of attention heads (6 for ViT)
     * @param window_size Window size (typically 16)
     * @param qkv_proj QKV projection weights [C, C * 3]
     * @param o_proj Output projection weights [C, C]
     * @param output Output [H, W, C]
     */
    static void windowed_attention_avx2(
        const float* input,
        int H, int W, int C,
        int num_heads,
        int window_size,
        const float* qkv_proj,
        const float* o_proj,
        float* output
    );

    /**
     * @brief Compute attention within a single window
     * @param window Input window [window_size * window_size, C]
     * @param window_size Window dimension
     * @param C Channels
     * @param num_heads Number of attention heads
     * @param head_dim Head dimension (C / num_heads)
     * @param output Output [window_size * window_size, C]
     */
    static void window_attention_avx2(
        const float* window,
        int window_size,
        int C,
        int num_heads,
        int head_dim,
        const float* qkv_proj,
        const float* o_proj,
        float* output
    );

    /**
     * @brief Shifted window attention (Swin-style)
     * @param input Input [H, W, C]
     * @param H Height
     * @param W Width
     * @param C Channels
     * @param shift_size Number of pixels to shift
     * @param num_heads Number of attention heads
     * @param window_size Window size
     * @param qkv_proj QKV projection weights
     * @param o_proj Output projection weights
     * @param output Output [H, W, C]
     */
    static void shifted_window_attention_avx2(
        const float* input,
        int H, int W, int C,
        int shift_size,
        int num_heads,
        int window_size,
        const float* qkv_proj,
        const float* o_proj,
        float* output
    );
};

/**
 * @brief 2D relative positional bias for windowed attention
 */
class RelativePositionBias {
public:
    /**
     * @brief Generate relative positional bias table
     * @param bias_table Output bias table [(2*window_size-1) * (2*window_size-1), num_heads]
     * @param window_size Window size
     * @param num_heads Number of attention heads
     */
    static void generate_bias_table(
        float* bias_table,
        int window_size,
        int num_heads
    );

    /**
     * @brief Add relative bias to attention scores
     * @param attention_scores Attention scores [num_heads, window_size^2, window_size^2]
     * @param bias_table Pre-computed bias table
     * @param window_size Window size
     * @param num_heads Number of attention heads
     */
    static void add_relative_bias(
        float* attention_scores,
        const float* bias_table,
        int window_size,
        int num_heads
    );
};

/**
 * @brief Vision-specific attention utilities with AVX2
 */
namespace vision_attention_utils {

/**
 * @brief Compute QK^T for vision patches with AVX2
 * @param Q Query [N_q, num_heads, head_dim]
 * @param K Key [N_k, num_heads, head_dim]
 * @param scores Output scores [num_heads, N_q, N_k]
 * @param N_q Number of query patches
 * @param N_k Number of key patches
 * @param num_heads Number of attention heads
 * @param head_dim Head dimension
 */
inline void qk_transpose_vision_avx2(
    const float* Q,
    const float* K,
    float* scores,
    int N_q,
    int N_k,
    int num_heads,
    int head_dim
) {
    int dim_per_head = head_dim / 8;  // AVX2 vectors

    for (int h = 0; h < num_heads; h++) {
        for (int i = 0; i < N_q; i++) {
            for (int j = 0; j < N_k; j++) {
                // Dot product Q[i, h] · K[j, h]
                __m256 sum = _mm256_setzero_ps();

                const float* q_ptr = &Q[(i * num_heads + h) * head_dim];
                const float* k_ptr = &K[(j * num_heads + h) * head_dim];

                for (int d = 0; d < dim_per_head; d++) {
                    __m256 q = _mm256_loadu_ps(&q_ptr[d * 8]);
                    __m256 k = _mm256_loadu_ps(&k_ptr[d * 8]);
                    sum = _mm256_fmadd_ps(q, k, sum);
                }

                // Horizontal sum
                float score = 0.0f;
                for (int d = 0; d < 8; d++) {
                    score += ((float*)&sum)[d];
                }
                scores[h * N_q * N_k + i * N_k + j] = score;
            }
        }
    }
}

/**
 * @brief Weighted sum for vision attention with AVX2
 * @param attention_weights Attention weights [N_q, N_k]
 * @param V Value [N_k, num_heads, head_dim]
 * @param output Output [N_q, num_heads, head_dim]
 * @param N_q Number of query positions
 * @param N_k Number of key positions
 * @param num_heads Number of attention heads
 * @param head_dim Head dimension
 */
inline void weighted_sum_vision_avx2(
    const float* attention_weights,
    const float* V,
    float* output,
    int N_q,
    int N_k,
    int num_heads,
    int head_dim
) {
    for (int h = 0; h < num_heads; h++) {
        for (int i = 0; i < N_q; i++) {
            __m256 sum[8] = {_mm256_setzero_ps()};

            for (int j = 0; j < N_k; j++) {
                float weight = attention_weights[i * N_k + j];
                __m256 w = _mm256_set1_ps(weight);

                for (int d = 0; d < 8; d++) {
                    __m256 v = _mm256_loadu_ps(&V[(j * num_heads + h) * head_dim + d * 8]);
                    sum[d] = _mm256_fmadd_ps(w, v, sum[d]);
                }
            }

            // Store results
            for (int d = 0; d < 8; d++) {
                _mm256_storeu_ps(&output[(i * num_heads + h) * head_dim + d * 8], sum[d]);
            }
        }
    }
}

} // namespace vision_attention_utils
} // namespace ops
} // namespace cromwell
