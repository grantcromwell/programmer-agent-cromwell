/**
 * @file attention_ops.h
 * @brief AVX2-optimized attention operations for transformers
 *
 * Implements multi-query attention with:
 * - QK^T computation (attention scores)
 * - Causal masking
 * - Softmax with numerical stability
 * - Attention @ V computation
 *
 * Optimized for AMD Zen 3 architecture with AVX2 support.
 */

#ifndef CROMWELL_OPS_ATTENTION_OPS_H_
#define CROMWELL_OPS_ATTENTION_OPS_H_

#include "avx2_utils.h"
#include "gemm.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace cromwell {
namespace ops {

/**
 * @brief Compute QK^T attention scores
 *
 * @param Q Query tensor [seq_len, num_heads, head_dim]
 * @param K Key tensor [cache_len, num_kv_heads, head_dim]
 * @param attn Output attention scores [seq_len, num_heads, cache_len]
 * @param seq_len Sequence length
 * @param cache_len Cache length
 * @param num_heads Number of query heads
 * @param num_kv_heads Number of key/value heads
 * @param head_dim Head dimension
 *
 * For multi-query attention:
 * - Each query head has independent Q
 * - All query heads share the same K and V
 */
inline void compute_qk_attn(
    const float* Q,
    const float* K,
    float* attn,
    int seq_len,
    int cache_len,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    int kv_repeats = num_heads / num_kv_heads;

    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / kv_repeats;

        for (int t = 0; t < seq_len; t++) {
            const float* q_t = &Q[t * num_heads * head_dim + h * head_dim];

            for (int s = 0; s < cache_len; s++) {
                const float* k_s = &K[s * num_kv_heads * head_dim + kv_h * head_dim];

                // Compute dot product
                __m256 sum = _mm256_setzero_ps();
                int i = 0;

                // Main loop: 8 elements at a time
                for (; i <= head_dim - 8; i += 8) {
                    __m256 q_vec = load8_ps(&q_t[i]);
                    __m256 k_vec = load8_ps(&k_s[i]);
                    sum = fmadd_ps(q_vec, k_vec, sum);
                }

                // Horizontal sum
                float dot = horizontal_sum_ps(sum);

                // Handle remaining elements
                for (; i < head_dim; i++) {
                    dot += q_t[i] * k_s[i];
                }

                attn[t * num_heads * cache_len + h * cache_len + s] = dot;
            }
        }
    }
}

/**
 * @brief Apply causal mask to attention scores
 *
 * Masks future positions (for autoregressive generation).
 * Sets masked positions to -inf.
 *
 * @param attn Attention scores [seq_len, num_heads, cache_len]
 * @param seq_len Sequence length
 * @param cache_len Cache length
 * @param num_heads Number of attention heads
 */
inline void apply_causal_mask(
    float* attn,
    int seq_len,
    int cache_len,
    int num_heads
) {
    constexpr float kNegInf = -1e10f;

    for (int h = 0; h < num_heads; h++) {
        for (int t = 0; t < seq_len; t++) {
            for (int s = t + 1; s < cache_len; s++) {
                attn[t * num_heads * cache_len + h * cache_len + s] = kNegInf;
            }
        }
    }
}

/**
 * @brief Compute softmax in-place
 *
 * Softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
 *
 * @param input Input/output matrix [rows, cols]
 * @param rows Number of rows
 * @param cols Number of columns
 */
inline void softmax_inplace(
    float* input,
    int rows,
    int cols
) {
    for (int r = 0; r < rows; r++) {
        float* row = &input[r * cols];

        // Find max (for numerical stability)
        __m256 max_vec = load8_ps(&row[0]);
        int i = kAVX2Width;

        for (; i <= cols - kAVX2Width; i += kAVX2Width) {
            __m256 curr = load8_ps(&row[i]);
            max_vec = _mm256_max_ps(max_vec, curr);
        }

        // Horizontal max
        float max_val = horizontal_max_ps(max_vec);

        // Handle remaining elements
        for (; i < cols; i++) {
            max_val = std::max(max_val, row[i]);
        }

        // Compute exp(x - max) and sum
        __m256 sum_vec = _mm256_setzero_ps();
        i = 0;

        __m256 max_broadcast = _mm256_set1_ps(max_val);

        for (; i <= cols - kAVX2Width; i += kAVX2Width) {
            __m256 x = load8_ps(&row[i]);
            __m256 shifted = _mm256_sub_ps(x, max_broadcast);

            // Compute exp using approximation
            __m256 exp_x = exp_ps(shifted);

            store8_ps(&row[i], exp_x);
            sum_vec = _mm256_add_ps(sum_vec, exp_x);
        }

        // Horizontal sum
        float sum = horizontal_sum_ps(sum_vec);

        // Handle remaining elements
        for (; i < cols; i++) {
            float exp_val = std::exp(row[i] - max_val);
            row[i] = exp_val;
            sum += exp_val;
        }

        // Normalize
        __m256 sum_inv = _mm256_set1_ps(1.0f / sum);
        i = 0;

        for (; i <= cols - kAVX2Width; i += kAVX2Width) {
            __m256 x = load8_ps(&row[i]);
            store8_ps(&row[i], _mm256_mul_ps(x, sum_inv));
        }

        for (; i < cols; i++) {
            row[i] /= sum;
        }
    }
}

/**
 * @brief Compute attention output: weights @ V
 *
 * @param attn Attention weights [seq_len, num_heads, cache_len]
 * @param V Value tensor [cache_len, num_kv_heads, head_dim]
 * @param output Output tensor [seq_len, num_heads, head_dim]
 * @param seq_len Sequence length
 * @param cache_len Cache length
 * @param num_heads Number of attention heads
 * @param num_kv_heads Number of key/value heads
 * @param head_dim Head dimension
 */
inline void compute_attn_output(
    const float* attn,
    const float* V,
    float* output,
    int seq_len,
    int cache_len,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    int kv_repeats = num_heads / num_kv_heads;

    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / kv_repeats;

        for (int t = 0; t < seq_len; t++) {
            const float* attn_t = &attn[t * num_heads * cache_len + h * cache_len];
            float* out_t = &output[t * num_heads * head_dim + h * head_dim];

            // Initialize output to zero
            std::memset(out_t, 0, head_dim * sizeof(float));

            // Compute weighted sum of values
            for (int s = 0; s < cache_len; s++) {
                const float* v_s = &V[s * num_kv_heads * head_dim + kv_h * head_dim];
                float weight = attn_t[s];

                // out_t += weight * v_s
                int i = 0;

                for (; i <= head_dim - 8; i += 8) {
                    __m256 v_vec = load8_ps(&v_s[i]);
                    __m256 out_vec = load8_ps(&out_t[i]);
                    __m256 weight_vec = _mm256_set1_ps(weight);

                    store8_ps(&out_t[i], fmadd_ps(weight_vec, v_vec, out_vec));
                }

                for (; i < head_dim; i++) {
                    out_t[i] += weight * v_s[i];
                }
            }
        }
    }
}

/**
 * @brief Multi-query attention forward pass
 *
 * Combines all attention operations:
 * 1. QK^T computation
 * 2. Causal masking
 * 3. Softmax
 * 4. Attention @ V
 *
 * @param Q Query tensor [seq_len, num_heads, head_dim]
 * @param K Key tensor [cache_len, num_kv_heads, head_dim]
 * @param V Value tensor [cache_len, num_kv_heads, head_dim]
 * @param output Output tensor [seq_len, num_heads, head_dim]
 * @param buffer Temporary buffer [seq_len, num_heads, cache_len]
 * @param seq_len Sequence length
 * @param cache_len Cache length
 * @param num_heads Number of query heads
 * @param num_kv_heads Number of key/value heads
 * @param head_dim Head dimension
 * @param use_causal_mask Whether to apply causal mask
 */
inline void multi_query_attention_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    float* buffer,
    int seq_len,
    int cache_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    bool use_causal_mask = true
) {
    // 1. Compute QK^T
    compute_qk_attn(
        Q, K, buffer,
        seq_len, cache_len,
        num_heads, num_kv_heads,
        head_dim
    );

    // 2. Apply causal mask (if needed)
    if (use_causal_mask) {
        apply_causal_mask(buffer, seq_len, cache_len, num_heads);
    }

    // 3. Softmax
    softmax_inplace(buffer, seq_len * num_heads, cache_len);

    // 4. Compute attention output
    compute_attn_output(
        buffer, V, output,
        seq_len, cache_len,
        num_heads, num_kv_heads,
        head_dim
    );
}

/**
 * @brief Apply rotary positional embeddings (RoPE)
 *
 * RoPE rotates query and key pairs based on their position.
 *
 * @param Q Query tensor [seq_len, num_heads, head_dim]
 * @param K Key tensor [seq_len, num_kv_heads, head_dim]
 * @param seq_len Sequence length
 * @param num_heads Number of query heads
 * @param num_kv_heads Number of key/value heads
 * @param head_dim Head dimension (must be even)
 * @param base Rotary base frequency (default: 10000)
 */
inline void apply_rotary_embeddings(
    float* Q,
    float* K,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float base = 10000.0f
) {
    assert(head_dim % 2 == 0 && "head_dim must be even for RoPE");

    // Pre-compute frequencies
    std::vector<float> freqs(head_dim / 2);

    for (int i = 0; i < head_dim / 2; i++) {
        freqs[i] = 1.0f / std::pow(base, static_cast<float>(2 * i) / head_dim);
    }

    // Apply RoPE to each position
    for (int pos = 0; pos < seq_len; pos++) {
        // Compute rotation angles for this position
        std::vector<float> cos_theta(head_dim / 2);
        std::vector<float> sin_theta(head_dim / 2);

        for (int i = 0; i < head_dim / 2; i++) {
            float theta = pos * freqs[i];
            cos_theta[i] = std::cos(theta);
            sin_theta[i] = std::sin(theta);
        }

        // Apply to Q
        for (int h = 0; h < num_heads; h++) {
            float* q_pos = &Q[pos * num_heads * head_dim + h * head_dim];

            for (int i = 0; i < head_dim / 2; i++) {
                float x = q_pos[2 * i];
                float y = q_pos[2 * i + 1];

                // Rotate
                q_pos[2 * i] = x * cos_theta[i] - y * sin_theta[i];
                q_pos[2 * i + 1] = x * sin_theta[i] + y * cos_theta[i];
            }
        }

        // Apply to K
        for (int h = 0; h < num_kv_heads; h++) {
            float* k_pos = &K[pos * num_kv_heads * head_dim + h * head_dim];

            for (int i = 0; i < head_dim / 2; i++) {
                float x = k_pos[2 * i];
                float y = k_pos[2 * i + 1];

                // Rotate
                k_pos[2 * i] = x * cos_theta[i] - y * sin_theta[i];
                k_pos[2 * i + 1] = x * sin_theta[i] + y * cos_theta[i];
            }
        }
    }
}

}  // namespace ops
}  // namespace cromwell

#endif  // CROMWELL_OPS_ATTENTION_OPS_H_
