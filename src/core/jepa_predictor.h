#pragma once

#include <vector>

namespace cromwell {
namespace vljepa {

/**
 * @brief JEPA (Joint Embedding Predictive Architecture) prediction head
 *
 * Architecture:
 * - Temporal context encoder (4 transformer layers, hidden=512)
 * - Multi-step predictor (z_{t+1}, z_{t+2}, z_{t+3})
 * - Loss: MSE in embedding space
 *
 * Target: ~30M parameters
 */
class JEPAPredictor {
public:
    struct Config {
        int hidden_dim = 512;
        int num_context_layers = 4;
        int num_heads = 8;
        int head_dim = 64;  // hidden_dim / num_heads
        int intermediate_dim = 2048;
        int prediction_steps = 3;
        float step_weights[3] = {1.0f, 0.8f, 0.6f};
    };

    JEPAPredictor(const Config& config);
    ~JEPAPredictor();

    /**
     * @brief Encode context for prediction
     * @param context Joint embeddings [N, hidden_dim]
     * @param N Number of tokens
     * @param encoded_output Context-encoded embeddings [N, hidden_dim]
     */
    void encode_context(
        const float* context,
        int N,
        float* encoded_output
    );

    /**
     * @brief Multi-step embedding prediction
     * @param context Context-encoded embeddings [N, hidden_dim]
     * @param N Number of tokens
     * @param predicted_output Predicted embeddings [N, prediction_steps, hidden_dim]
     */
    void predict(
        const float* context,
        int N,
        float* predicted_output
    );

    /**
     * @brief Compute JEPA loss (MSE in embedding space)
     * @param predicted Predicted embeddings [N, prediction_steps, hidden_dim]
     * @param target Target embeddings [N, prediction_steps, hidden_dim]
     * @param N Number of tokens
     * @param mask Optional mask for which tokens to compute loss [N]
     * @return Loss value (scalar)
     */
    float compute_loss(
        const float* predicted,
        const float* target,
        int N,
        const bool* mask = nullptr
    );

    int hidden_dim() const { return config_.hidden_dim; }
    int prediction_steps() const { return config_.prediction_steps; }

private:
    Config config_;

    // Context encoder (4 transformer layers)
    struct TransformerLayer {
        float* qkv_proj;       // [hidden_dim, hidden_dim * 3]
        float* o_proj;         // [hidden_dim, hidden_dim]
        float* gate_up_proj;   // [intermediate_dim, hidden_dim * 2]
        float* down_proj;      // [hidden_dim, intermediate_dim]
        float* norm1;          // [hidden_dim]
        float* norm2;          // [hidden_dim]
    };
    std::vector<TransformerLayer> context_layers_;

    // Multi-step predictor (shared weights across steps)
    float* predictor_w1_;      // [hidden_dim, hidden_dim]  // Step 1
    float* predictor_w2_;      // [hidden_dim, hidden_dim]  // Step 2
    float* predictor_w3_;      // [hidden_dim, hidden_dim]  // Step 3

    void init_weights();
    void context_layer_forward(
        const float* input,
        int N,
        float* output,
        int layer_idx
    );
};

/**
 * @brief JEPA masking strategy for training
 */
class JEPAMasking {
public:
    enum Strategy {
        BLOCK,      // Block masking for vision
        RANDOM,     // Random masking for language
        SPAN        // Span masking for joint
    };

    struct Config {
        float mask_ratio = 0.15f;
        int block_size = 16;
        int span_length = 32;
    };

    JEPAMasking(const Config& config);
    ~JEPAMasking();

    /**
     * @brief Generate mask for JEPA training
     * @param N Number of tokens
     * @param mask Output mask [N]
     * @param strategy Masking strategy
     * @param H Optional height for block masking
     * @param W Optional width for block masking
     */
    void generate_mask(
        int N,
        bool* mask,
        Strategy strategy = RANDOM,
        int H = 0,
        int W = 0
    );

private:
    Config config_;

    void block_mask(int H, int W, bool* mask);
    void random_mask(int N, bool* mask);
    void span_mask(int N, bool* mask);
};

} // namespace vljepa
} // namespace cromwell
