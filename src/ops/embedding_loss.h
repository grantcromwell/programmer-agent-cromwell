#pragma once

#include <immintrin.h>
#include <cmath>

namespace cromwell {
namespace ops {

/**
 * @brief Embedding loss computation for JEPA training
 *
 * Loss types:
 * - MSE (Mean Squared Error) in embedding space
 * - Cosine similarity loss
 * - Contrastive loss
 *
 * All operations use AVX2 for acceleration
 */
class EmbeddingLoss {
public:
    /**
     * @brief Compute MSE loss in embedding space
     * @param predicted Predicted embeddings [N, dim]
     * @param target Target embeddings [N, dim]
     * @param N Number of tokens
     * @param dim Embedding dimension
     * @param mask Optional mask for which tokens to compute loss [N]
     * @return MSE loss value (scalar)
     */
    static float mse_loss_avx2(
        const float* predicted,
        const float* target,
        int N,
        int dim,
        const bool* mask = nullptr
    );

    /**
     * @brief Compute multi-step JEPA loss with weighting
     * @param predicted Predicted embeddings [N, num_steps, dim]
     * @param target Target embeddings [N, num_steps, dim]
     * @param N Number of tokens
     * @param num_steps Number of prediction steps
     * @param dim Embedding dimension
     * @param step_weights Weight for each prediction step [num_steps]
     * @param mask Optional mask [N]
     * @return Weighted MSE loss value
     */
    static float multi_step_mse_loss_avx2(
        const float* predicted,
        const float* target,
        int N,
        int num_steps,
        int dim,
        const float* step_weights,
        const bool* mask = nullptr
    );

    /**
     * @brief Cosine similarity loss
     * @param predicted Predicted embeddings [N, dim]
     * @param target Target embeddings [N, dim]
     * @param N Number of tokens
     * @param dim Embedding dimension
     * @param mask Optional mask [N]
     * @return Cosine similarity loss (1 - mean cosine similarity)
     */
    static float cosine_loss_avx2(
        const float* predicted,
        const float* target,
        int N,
        int dim,
        const bool* mask = nullptr
    );

    /**
     * @brief Contrastive loss for JEPA
     * @param predicted Predicted embeddings [N, dim]
     * @param target Target embeddings [N, dim]
     * @param N Number of tokens
     * @param dim Embedding dimension
     * @param temperature Temperature for softmax
     * @param mask Optional mask [N]
     * @return Contrastive loss value
     */
    static float contrastive_loss_avx2(
        const float* predicted,
        const float* target,
        int N,
        int dim,
        float temperature = 0.1f,
        const bool* mask = nullptr
    );
};

/**
 * @brief Gradient computation for embedding loss
 */
class EmbeddingLossGrad {
public:
    /**
     * @brief Compute gradients for MSE loss
     * @param predicted Predicted embeddings [N, dim]
     * @param target Target embeddings [N, dim]
     * @param grad_output Output gradients [N, dim]
     * @param N Number of tokens
     * @param dim Embedding dimension
     * @param mask Optional mask [N]
     */
    static void mse_loss_grad_avx2(
        const float* predicted,
        const float* target,
        float* grad_output,
        int N,
        int dim,
        const bool* mask = nullptr
    );

    /**
     * @brief Compute gradients for multi-step MSE loss
     * @param predicted Predicted embeddings [N, num_steps, dim]
     * @param target Target embeddings [N, num_steps, dim]
     * @param grad_output Output gradients [N, num_steps, dim]
     * @param N Number of tokens
     * @param num_steps Number of prediction steps
     * @param dim Embedding dimension
     * @param step_weights Weight for each prediction step [num_steps]
     * @param mask Optional mask [N]
     */
    static void multi_step_mse_loss_grad_avx2(
        const float* predicted,
        const float* target,
        float* grad_output,
        int N,
        int num_steps,
        int dim,
        const float* step_weights,
        const bool* mask = nullptr
    );
};

/**
 * @brief AVX2 utility functions for embedding loss
 */
namespace embedding_loss_utils {

// L2 norm (squared) with AVX2
inline float l2_norm_squared_avx2(const float* x, const float* y, int dim) {
    __m256 sum = _mm256_setzero_ps();
    int dim_avx = dim / 8 * 8;

    for (int i = 0; i < dim_avx; i += 8) {
        __m256 xi = _mm256_loadu_ps(&x[i]);
        __m256 yi = _mm256_loadu_ps(&y[i]);
        __m256 diff = _mm256_sub_ps(xi, yi);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    // Horizontal sum
    float result = 0.0f;
    for (int i = 0; i < 8; i++) {
        result += ((float*)&sum)[i];
    }

    // Handle remaining elements
    for (int i = dim_avx; i < dim; i++) {
        float diff = x[i] - y[i];
        result += diff * diff;
    }

    return result;
}

// Dot product with AVX2
inline float dot_product_avx2(const float* x, const float* y, int dim) {
    __m256 sum = _mm256_setzero_ps();
    int dim_avx = dim / 8 * 8;

    for (int i = 0; i < dim_avx; i += 8) {
        __m256 xi = _mm256_loadu_ps(&x[i]);
        __m256 yi = _mm256_loadu_ps(&y[i]);
        sum = _mm256_fmadd_ps(xi, yi, sum);
    }

    // Horizontal sum
    float result = 0.0f;
    for (int i = 0; i < 8; i++) {
        result += ((float*)&sum)[i];
    }

    // Handle remaining elements
    for (int i = dim_avx; i < dim; i++) {
        result += x[i] * y[i];
    }

    return result;
}

// L2 norm with AVX2
inline float l2_norm_avx2(const float* x, int dim) {
    return std::sqrt(l2_norm_squared_avx2(x, x, dim));
}

// Normalize vector in-place with AVX2
inline void normalize_avx2(float* x, int dim) {
    float norm = l2_norm_avx2(x, dim);
    float inv_norm = 1.0f / (norm + 1e-8f);

    __m256 inv_norm_v = _mm256_set1_ps(inv_norm);
    int dim_avx = dim / 8 * 8;

    for (int i = 0; i < dim_avx; i += 8) {
        __m256 xi = _mm256_loadu_ps(&x[i]);
        _mm256_storeu_ps(&x[i], _mm256_mul_ps(xi, inv_norm_v));
    }

    // Handle remaining elements
    for (int i = dim_avx; i < dim; i++) {
        x[i] *= inv_norm;
    }
}

} // namespace embedding_loss_utils
} // namespace ops
} // namespace cromwell
