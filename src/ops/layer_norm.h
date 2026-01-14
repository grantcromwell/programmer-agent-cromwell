/**
 * @file layer_norm.h
 * @brief AVX2-optimized normalization layers for transformers
 *
 * Implements:
 * - RMSNorm (Root Mean Square Normalization)
 * - LayerNorm (Layer Normalization)
 *
 * Optimized for AMD Zen 3 architecture with AVX2 support.
 */

#ifndef CROMWELL_OPS_LAYER_NORM_H_
#define CROMWELL_OPS_LAYER_NORM_H_

#include "avx2_utils.h"
#include <cmath>
#include <vector>

namespace cromwell {
namespace ops {

/**
 * @brief RMSNorm: x / sqrt(mean(x^2) + epsilon) * scale
 *
 * @param input Input tensor [batch_size, seq_len, hidden_size]
 * @param output Output tensor [batch_size, seq_len, hidden_size]
 * @param scale Scale parameter [hidden_size]
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param hidden_size Hidden size
 * @param epsilon Epsilon for numerical stability
 */
inline void rmsnorm(
    const float* input,
    float* output,
    const float* scale,
    int batch_size,
    int seq_len,
    int hidden_size,
    float epsilon = 1e-6f
) {
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len; t++) {
            const float* x = &input[(b * seq_len + t) * hidden_size];
            float* y = &output[(b * seq_len + t) * hidden_size];

            // Compute sum of squares
            __m256 sum_sq = _mm256_setzero_ps();
            int i = 0;

            for (; i <= hidden_size - 8; i += 8) {
                __m256 x_vec = load8_ps(&x[i]);
                sum_sq = fmadd_ps(x_vec, x_vec, sum_sq);
            }

            // Horizontal sum
            float mean_sq = horizontal_sum_ps(sum_sq);

            // Handle remaining elements
            for (; i < hidden_size; i++) {
                mean_sq += x[i] * x[i];
            }
            mean_sq /= static_cast<float>(hidden_size);

            // Compute normalization factor
            float norm = 1.0f / std::sqrt(mean_sq + epsilon);
            __m256 norm_vec = _mm256_set1_ps(norm);

            // Apply normalization with scale
            i = 0;
            for (; i <= hidden_size - 8; i += 8) {
                __m256 x_vec = load8_ps(&x[i]);
                __m256 s_vec = load8_ps(&scale[i]);
                __m256 normalized = _mm256_mul_ps(x_vec, norm_vec);
                store8_ps(&y[i], _mm256_mul_ps(normalized, s_vec));
            }

            for (; i < hidden_size; i++) {
                y[i] = (x[i] * norm) * scale[i];
            }
        }
    }
}

/**
 * @brief LayerNorm: (x - mean(x)) / std(x) * scale + bias
 *
 * @param input Input tensor [batch_size, seq_len, hidden_size]
 * @param output Output tensor [batch_size, seq_len, hidden_size]
 * @param scale Scale parameter [hidden_size]
 * @param bias Bias parameter [hidden_size]
 * @param batch_size Batch size
 * @param seq_len Sequence length
 * @param hidden_size Hidden size
 * @param epsilon Epsilon for numerical stability
 */
inline void layernorm(
    const float* input,
    float* output,
    const float* scale,
    const float* bias,
    int batch_size,
    int seq_len,
    int hidden_size,
    float epsilon = 1e-6f
) {
    for (int b = 0; b < batch_size; b++) {
        for (int t = 0; t < seq_len; t++) {
            const float* x = &input[(b * seq_len + t) * hidden_size];
            float* y = &output[(b * seq_len + t) * hidden_size];

            // Compute mean
            __m256 sum_vec = _mm256_setzero_ps();
            int i = 0;

            for (; i <= hidden_size - 8; i += 8) {
                __m256 x_vec = load8_ps(&x[i]);
                sum_vec = _mm256_add_ps(sum_vec, x_vec);
            }

            float mean = horizontal_sum_ps(sum_vec);

            for (; i < hidden_size; i++) {
                mean += x[i];
            }
            mean /= static_cast<float>(hidden_size);

            // Compute variance
            __m256 var_sum = _mm256_setzero_ps();
            i = 0;

            __m256 mean_vec = _mm256_set1_ps(mean);

            for (; i <= hidden_size - 8; i += 8) {
                __m256 x_vec = load8_ps(&x[i]);
                __m256 diff = _mm256_sub_ps(x_vec, mean_vec);
                var_sum = fmadd_ps(diff, diff, var_sum);
            }

            float variance = horizontal_sum_ps(var_sum);

            for (; i < hidden_size; i++) {
                float diff = x[i] - mean;
                variance += diff * diff;
            }
            variance /= static_cast<float>(hidden_size);

            // Compute normalization factor
            float std = std::sqrt(variance + epsilon);
            float inv_std = 1.0f / std;

            __m256 inv_std_vec = _mm256_set1_ps(inv_std);

            // Apply normalization with scale and bias
            i = 0;
            for (; i <= hidden_size - 8; i += 8) {
                __m256 x_vec = load8_ps(&x[i]);
                __m256 s_vec = load8_ps(&scale[i]);
                __m256 b_vec = load8_ps(&bias[i]);

                __m256 normalized = _mm256_mul_ps(_mm256_sub_ps(x_vec, mean_vec), inv_std_vec);
                store8_ps(&y[i], _mm256_add_ps(_mm256_mul_ps(normalized, s_vec), b_vec));
            }

            for (; i < hidden_size; i++) {
                y[i] = ((x[i] - mean) * inv_std) * scale[i] + bias[i];
            }
        }
    }
}

}  // namespace ops
}  // namespace cromwell

#endif  // CROMWELL_OPS_LAYER_NORM_H_
