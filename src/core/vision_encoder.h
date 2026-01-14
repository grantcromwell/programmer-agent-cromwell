#pragma once

#include <vector>
#include <memory>

namespace cromwell {
namespace vljepa {

/**
 * @brief CNN-ViT hybrid vision encoder for VL-JEPA
 *
 * Architecture:
 * - Stem: 4×4 conv, stride=4, 96 channels
 * - MBConv blocks (×4): Depthwise separable with SE attention
 * - Projection: 192 → 384
 * - ViT blocks (×6): Windowed attention, hidden=384
 *
 * Target: ~50M parameters, ~22 ms for 256×256 image
 */
class VisionEncoder {
public:
    struct Config {
        int stem_channels = 96;
        int mbconv_blocks = 4;
        int vit_blocks = 6;
        int hidden_dim = 384;
        int num_heads = 6;
        int window_size = 16;
        int patch_size = 4;
    };

    VisionEncoder(const Config& config);
    ~VisionEncoder();

    /**
     * @brief Forward pass through vision encoder
     * @param input RGB image [H, W, 3] (NHWC format)
     * @param H Image height
     * @param W Image width
     * @param output Output embeddings [N_patches, hidden_dim]
     */
    void forward(
        const float* input,
        int H, int W,
        float* output
    );

    /**
     * @brief Get number of output patches
     */
    int num_patches(int H, int W) const;

    /**
     * @brief Get output dimension
     */
    int output_dim() const { return config_.hidden_dim; }

private:
    Config config_;

    // Stem layer
    float* stem_conv_;      // [4, 4, 3, stem_channels] NHWC
    float* stem_bn_;        // [stem_channels]

    // MBConv blocks
    struct MBConvBlock {
        float* expand_conv;    // [C_in, C_in * 6]
        float* expand_bn;      // [C_in * 6]
        float* depthwise_conv; // [3, 3, C_in * 6, C_in * 6]
        float* depthwise_bn;   // [C_in * 6]
        float* se_fc1;         // [C_in * 6, C_in * 6 / 4]
        float* se_fc2;         // [C_in * 6 / 4, C_in * 6]
        float* project_conv;   // [C_in * 6, C_out]
        float* project_bn;     // [C_out]
        int C_in, C_out;
    };
    std::vector<MBConvBlock> mbconv_blocks_;

    // Projection layer
    float* proj_conv_;       // [1, 1, 192, hidden_dim]

    // ViT blocks
    struct ViTBlock {
        float* qkv_proj;       // [hidden_dim, hidden_dim * 3]
        float* o_proj;         // [hidden_dim, hidden_dim]
        float* gate_up_proj;   // [hidden_dim * 4, hidden_dim * 2]
        float* down_proj;      // [hidden_dim, hidden_dim * 4]
        float* norm1;          // [hidden_dim]
        float* norm2;          // [hidden_dim]
    };
    std::vector<ViTBlock> vit_blocks_;

    // Positional embeddings
    float* pos_embed_;       // [max_patches, hidden_dim]
    int max_patches_;

    void init_weights();
    void stem_forward(const float* input, int H, int W, float* output);
    void mbconv_forward(const float* input, int H, int W, float* output, int block_idx);
    void vit_forward(const float* input, int N, float* output, int block_idx);
};

} // namespace vljepa
} // namespace cromwell
