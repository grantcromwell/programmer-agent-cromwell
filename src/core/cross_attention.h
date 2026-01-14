#pragma once

#include <vector>

namespace cromwell {
namespace vljepa {

/**
 * @brief Bidirectional cross-modal attention for vision-language fusion
 *
 * Architecture:
 * - Projection to common dimension (512)
 * - V→L cross-attention (vision queries language)
 * - L→V cross-attention (language queries vision)
 * - SwiGLU MLP (512 → 2048 → 512)
 *
 * Target: ~40M parameters for 6 blocks
 */
class CrossAttentionBlock {
public:
    struct Config {
        int vision_dim = 384;
        int language_dim = 768;
        int joint_dim = 512;
        int num_heads = 8;
        int intermediate_dim = 2048;
    };

    CrossAttentionBlock(const Config& config);
    ~CrossAttentionBlock();

    /**
     * @brief Forward pass for bidirectional cross-attention
     * @param vision_input Vision embeddings [N_v, vision_dim]
     * @param language_input Language embeddings [N_l, language_dim]
     * @param vision_output Vision-conditioned embeddings [N_v, joint_dim]
     * @param language_output Language-conditioned embeddings [N_l, joint_dim]
     */
    void forward(
        const float* vision_input,
        const float* language_input,
        int N_v, int N_l,
        float* vision_output,
        float* language_output
    );

    int joint_dim() const { return config_.joint_dim; }

private:
    Config config_;

    // Projection layers
    float* vision_proj_;     // [vision_dim, joint_dim]
    float* language_proj_;   // [language_dim, joint_dim]

    // V→L cross-attention
    float* v_to_l_q_;        // [joint_dim, joint_dim]
    float* v_to_l_kv_;       // [joint_dim, joint_dim * 2]
    float* v_to_l_o_;        // [joint_dim, joint_dim]

    // L→V cross-attention
    float* l_to_v_q_;        // [joint_dim, joint_dim]
    float* l_to_v_kv_;       // [joint_dim, joint_dim * 2]
    float* l_to_v_o_;        // [joint_dim, joint_dim]

    // SwiGLU MLP
    float* gate_up_proj_;    // [intermediate_dim, joint_dim * 2]
    float* down_proj_;       // [joint_dim, intermediate_dim]

    // Normalization
    float* norm1_;           // [joint_dim]  // Pre-V→L
    float* norm2_;           // [joint_dim]  // Pre-L→V
    float* norm3_;           // [joint_dim]  // Pre-MLP

    void init_weights();
};

/**
 * @brief Stack of cross-attention blocks for fusion
 */
class CrossModalFusion {
public:
    struct Config {
        int num_blocks = 6;
        int vision_dim = 384;
        int language_dim = 768;
        int joint_dim = 512;
        int num_heads = 8;
        int intermediate_dim = 2048;
    };

    CrossModalFusion(const Config& config);
    ~CrossModalFusion();

    /**
     * @brief Forward pass through all fusion blocks
     * @param vision_input Vision embeddings [N_v, vision_dim]
     * @param language_input Language embeddings [N_l, language_dim]
     * @param joint_output Joint embeddings [N_v + N_l, joint_dim]
     */
    void forward(
        const float* vision_input,
        const float* language_input,
        int N_v, int N_l,
        float* joint_output
    );

    int joint_dim() const { return config_.joint_dim; }

private:
    Config config_;
    std::vector<std::unique_ptr<CrossAttentionBlock>> blocks_;

    // Projection outputs (cached for efficiency)
    float* vision_proj_;     // [N_v, joint_dim]
    float* language_proj_;   // [N_l, joint_dim]
    float* fused_emb_;       // [N_v + N_l, joint_dim]

    int max_tokens_;         // Maximum sequence length for caching
};

} // namespace vljepa
} // namespace cromwell
