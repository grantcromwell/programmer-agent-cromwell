# Cromwell VL-JEPA: Model Architecture Specifications
## Detailed Design for Vision-Language JEPA with Auto-Regressive Generation

**Version**: 2.0
**Date**: 2025-01-14

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      MULTIMODAL INPUT                              │
├─────────────────────────────┬───────────────────────────────────────┤
│       Vision Input          │          Language Input               │
│    [H, W, 3] RGB Image      │      Token IDs [seq_len]             │
└──────────────┬──────────────┴──────────────┬────────────────────────┘
               │                             │
               ▼                             ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│   Vision Encoder         │    │   Language Encoder       │
│   (~50M params)          │    │   (~120M params)         │
│  - CNN Stem              │    │  - Token Embedding       │
│  - MBConv Blocks (×4)    │    │  - RoPE Positional       │
│  - ViT Blocks (×6)       │    │  - Transformer (×12)     │
│  - Output: [N_v, 384]    │    │  - Output: [N_l, 768]    │
└──────────────┬───────────┘    └──────────────┬───────────┘
               │                             │
               └──────────────┬──────────────┘
                              ▼
              ┌──────────────────────────────┐
              │   Cross-Modal Fusion         │
              │   (~40M params)              │
              │  - Projection to 512-dim     │
              │  - Cross-Attention (×6)      │
              │  - Output: [N_v+N_l, 512]    │
              └──────────────┬───────────────┘
                             │
              ┌──────────────┴───────────────┐
              ▼                              ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│   JEPA Prediction Head   │    │   Auto-Regressive Head   │
│   (~30M params)          │    │   (~60M params)          │
│  - Context Encoder (×4)  │    │  - Projection to 768     │
│  - Multi-Step Predictor  │    │  - AR Transformer (×6)   │
│  - Loss: MSE embedding   │    │  - LM Head to vocab      │
└──────────────────────────┘    └──────────────┬───────────┘
                                            │
                                            ▼
                                    Output Tokens
                                  (Generated Code/Text)
```

---

## 2. Vision Encoder (~50M Parameters)

### 2.1 Architecture

```
Input: RGB Image [H, W, 3]
         ↓
┌─────────────────────────────────────┐
│  Stem Layer: Patch Embedding        │
│  - Conv: 4×4, stride=4              │
│  - Channels: 3 → 96                 │
│  - Output: [H/4, W/4, 96]          │
│  - Params: 4×4×3×96 = 4,608        │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  MBConv Blocks (×4)                 │
│  - Depthwise Separable Conv         │
│  - Expansion ratio: 6               │
│  - SE Attention                     │
│  - Output: [H/8, W/8, 192]         │
│  - Params: ~20M total              │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  Projection Layer                   │
│  - Conv: 1×1, stride=1              │
│  - Channels: 192 → 384              │
│  - Output: [H/8, W/8, 384]         │
│  - Params: 192×384 = 73,728        │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  Vision Transformer Blocks (×6)     │
│  - Windowed Attention (16×16)       │
│  - Hidden dim: 384                  │
│  - Heads: 6                         │
│  - MLP: 384 → 1536 → 384           │
│  - Params: ~24M total              │
└──────────────┬──────────────────────┘
               ▼
        Flatten + Positional Embeddings
               ▼
        Output: [N_vision_patches, 384]
```

### 2.2 Component Specifications

**MBConv Block (MobileNetV2-style)**:

```cpp
struct MBConvBlock {
    // Input: [H, W, C_in]
    // Output: [H, W, C_out]

    // 1. Expansion (1x1 conv)
    float* expand_conv;    // [C_in, C_in * 6]
    float* expand_bn;      // [C_in * 6]
    float* expand_bias;    // [C_in * 6]

    // 2. Depthwise (3x3 conv)
    float* depthwise_conv; // [3, 3, C_in * 6, C_in * 6] (groups=C_in*6)
    float* depthwise_bn;   // [C_in * 6]
    float* depthwise_bias; // [C_in * 6]

    // 3. Squeeze-Excitation
    float* se_fc1;         // [C_in * 6, C_in * 6 / 4]
    float* se_fc2;         // [C_in * 6 / 4, C_in * 6]

    // 4. Projection (1x1 conv)
    float* project_conv;   // [C_in * 6, C_out]
    float* project_bn;     // [C_out]
    float* project_bias;   // [C_out]

    // Activation: Swish (SiLU)
    // Residual connection if C_in == C_out
};
```

**Vision Transformer Block**:

```cpp
struct ViTBlock {
    // Windowed Multi-Head Self-Attention
    float* qkv_proj;       // [384, 384*3]  // Q, K, V projection
    float* o_proj;         // [384, 384]    // Output projection
    float* attn_norm;      // [384]         // RMSNorm

    // SwiGLU MLP
    float* gate_up_proj;   // [1536, 384*2]  // Gate and Up fused
    float* down_proj;      // [384, 1536]    // Down projection
    float* mlp_norm;       // [384]         // RMSNorm

    // Window size for attention
    int window_size = 16;

    // Attention heads: 6
    int num_heads = 6;
    int head_dim = 64;  // 384 / 6
};
```

### 2.3 Parameter Count

| Component | Parameters |
|-----------|------------|
| Stem (Conv 4×4) | 4,608 |
| MBConv Blocks (×4) | ~20,000,000 |
| Projection (192 → 384) | 73,728 |
| ViT Blocks (×6) | ~24,000,000 |
| Positional Embeddings | ~6,144 |
| **Total** | **~50,000,000** |

### 2.4 Memory Layout

```cpp
// Vision encoder weights (NHWC for AVX2)
struct VisionEncoderWeights {
    // Stem
    float* stem_conv;       // [4, 4, 3, 96] (NHWC)
    float* stem_bn;         // [96]

    // MBConv blocks (×4)
    float* mbconv_expand[4];      // [1, 1, C_in, C_in*6]
    float* mbconv_depthwise[4];   // [3, 3, C_in*6, C_in*6]
    float* mbconv_se_fc1[4];      // [C_in*6, C_in*6/4]
    float* mbconv_se_fc2[4];      // [C_in*6/4, C_in*6]
    float* mbconv_project[4];     // [1, 1, C_in*6, C_out]

    // Projection
    float* proj_conv;       // [1, 1, 192, 384]

    // ViT blocks (×6)
    float* vit_qkv[6];      // [384, 1152]
    float* vit_o[6];        // [384, 384]
    float* vit_gate_up[6];  // [1536, 768] (SwiGLU fused)
    float* vit_down[6];     // [384, 1536]
    float* vit_norm1[6];    // [384]
    float* vit_norm2[6];    // [384]

    // Positional embeddings
    float* pos_embed;       // [num_patches, 384]

    // All aligned to 32 bytes for AVX2
};
```

---

## 3. Language Encoder (~120M Parameters)

### 3.1 Architecture

```
Input: Token IDs [seq_len]
         ↓
┌─────────────────────────────────────┐
│  Token Embedding Layer              │
│  - Vocab size: 50,000               │
│  - Hidden dim: 768                  │
│  - Output: [seq_len, 768]          │
│  - Params: 50,000 × 768 = 38.4M   │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  Rotary Positional Encoding (RoPE)  │
│  - Applied to Q and K               │
│  - No learned parameters            │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  Transformer Layers (×12)           │
│  For each layer:                    │
│  ┌──────────────────────────────┐  │
│  │ 1. RMSNorm (pre-attention)  │  │
│  │ 2. Multi-Query Attention    │  │
│  │    - 12 Q heads, 3 KV heads  │  │
│  │    - Head dim: 64            │  │
│  │ 3. Residual +1               │  │
│  │ 4. RMSNorm (pre-MLP)         │  │
│  │ 5. SwiGLU MLP                │  │
│  │    - 768 → 3072 → 768        │  │
│  │ 6. Residual +2               │  │
│  └──────────────────────────────┘  │
│  - Params per layer: ~6.5M       │
│  - Total: 12 × 6.5M = 78M        │
└──────────────┬──────────────────────┘
               ▼
        Output: [seq_len, 768]
```

### 3.2 Component Specifications

**Transformer Layer**:

```cpp
struct TransformerLayer {
    // Multi-Query Attention
    // Query: 12 heads, Key/Value: 3 heads (shared)
    float* qkv_proj;       // [768, 768 + 2*192] = [768, 1152]
                           // Q: 768×768, K: 768×192, V: 768×192
    float* o_proj;         // [768, 768]
    float* attn_norm;      // [768]  // RMSNorm

    // SwiGLU MLP
    float* gate_up_proj;   // [3072, 768*2] = [3072, 1536] (fused)
    float* down_proj;      // [768, 3072]
    float* mlp_norm;       // [768]  // RMSNorm

    // Attention heads: 12 query, 3 key-value
    int num_q_heads = 12;
    int num_kv_heads = 3;
    int head_dim = 64;  // 768 / 12
};
```

### 3.3 Parameter Count

| Component | Parameters |
|-----------|------------|
| Token Embeddings | 38,400,000 |
| Per Transformer Layer (×12) | 6,500,000 |
| - QKV Projection | 884,736 |
| - Output Projection | 589,824 |
| - SwiGLU MLP | 4,718,592 |
| - Norm scales | 1,536 |
| **Total** | **~116,400,000** → rounds to **120M** |

---

## 4. Cross-Modal Fusion (~40M Parameters)

### 4.1 Architecture

```
Inputs:
  - Vision: [N_v, 384]
  - Language: [N_l, 768]
         ↓
┌─────────────────────────────────────┐
│  Projection to Common Dimension     │
│  - Vision: Linear(384 → 512)        │
│  - Language: Linear(768 → 512)      │
│  - Params: 384×512 + 768×512 = 590K│
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  Cross-Modal Attention Blocks (×6)  │
│  For each block:                    │
│  ┌──────────────────────────────┐  │
│  │ V→L Cross-Attention          │  │
│  │  - Q_vision, K_lang, V_lang  │  │
│  │  - Output: [N_l, 512]        │  │
│  │ L→V Cross-Attention          │  │
│  │  - Q_lang, K_vision, V_vision│  │
│  │  - Output: [N_v, 512]        │  │
│  │ Concat + Residual             │  │
│  │ SwiGLU MLP (512 → 2048 → 512)│  │
│  └──────────────────────────────┘  │
│  - Params per block: ~6M         │
│  - Total: 6 × 6M = 36M           │
└──────────────┬──────────────────────┘
               ▼
        Output: [N_v + N_l, 512] joint embeddings
```

### 4.2 Component Specifications

**Cross-Attention Block**:

```cpp
struct CrossAttentionBlock {
    // Vision to Language
    float* v_to_l_q;       // [512, 512]  // Vision queries
    float* v_to_l_kv;      // [512, 1024] // Language K, V (concatenated)
    float* v_to_l_o;       // [512, 512]

    // Language to Vision
    float* l_to_v_q;       // [512, 512]  // Language queries
    float* l_to_v_kv;      // [512, 1024] // Vision K, V
    float* l_to_v_o;       // [512, 512]

    // SwiGLU MLP
    float* gate_up_proj;   // [2048, 1024] (fused)
    float* down_proj;      // [512, 2048]

    // Normalization
    float* norm1;          // [512]  // Pre-V→L attention
    float* norm2;          // [512]  // Pre-L→V attention
    float* norm3;          // [512]  // Pre-MLP

    // Attention heads
    int num_heads = 8;
    int head_dim = 64;  // 512 / 8
};
```

### 4.3 Parameter Count

| Component | Parameters |
|-----------|------------|
| Vision Projection (384 → 512) | 196,608 |
| Language Projection (768 → 512) | 393,216 |
| Per Cross-Attention Block (×6) | ~6,000,000 |
| - V→L Attention | ~1,570,000 |
| - L→V Attention | ~1,570,000 |
| - SwiGLU MLP | ~2,620,000 |
| - Norm scales | 1,536 |
| **Total** | **~36,000,000** → rounds to **40M** |

---

## 5. JEPA Prediction Head (~30M Parameters)

### 5.1 Architecture

```
Input: Joint embeddings [N, 512]
         ↓
┌─────────────────────────────────────┐
│  Temporal Context Encoder (×4)      │
│  For each layer:                    │
│  ┌──────────────────────────────┐  │
│  │ 1. RMSNorm (pre-attention)  │  │
│  │ 2. Multi-Head Self-Attention │  │
│  │    - 8 heads, head_dim=64    │  │
│  │ 3. Residual +1               │  │
│  │ 4. RMSNorm (pre-MLP)         │  │
│  │ 5. SwiGLU MLP                │  │
│  │    - 512 → 2048 → 512        │  │
│  │ 6. Residual +2               │  │
│  └──────────────────────────────┘  │
│  - Params per layer: ~5M         │
│  - Total: 4 × 5M = 20M           │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  Multi-Step Predictor               │
│  - Predict z_{t+1}: MLP(512→512)   │
│  - Predict z_{t+2}: MLP(512→512)   │
│  - Predict z_{t+3}: MLP(512→512)   │
│  - Shared weights across steps     │
│  - Params: ~5M                     │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  Embedding Consistency Loss         │
│  - MSE(predicted, target)          │
│  - In embedding space (not tokens) │
└─────────────────────────────────────┘
```

### 5.2 JEPA Prediction Algorithm

```cpp
struct JEPAPredictor {
    // Context encoder (4 transformer layers)
    float* context_qkv[4];     // [512, 1536]
    float* context_o[4];       // [512, 512]
    float* context_gate_up[4]; // [2048, 1024]
    float* context_down[4];    // [512, 2048]

    // Multi-step predictor (shared across steps)
    float* predictor_w1;       // [512, 512]  // Step 1
    float* predictor_w2;       // [512, 512]  // Step 2
    float* predictor_w3;       // [512, 512]  // Step 3

    // Normalization
    float* norm[4];            // [512] per layer

    // Prediction
    void predict(
        const float* context_emb,  // [N, 512]
        float* predicted_emb,      // [N, 3, 512] output
        int N
    );
};

// JEPA loss computation
float jepa_loss(
    const float* predicted,  // [N, 3, 512]
    const float* target,     // [N, 3, 512]
    int N,
    float step_weights[3]    // [1.0, 0.8, 0.6]
);
```

### 5.3 Parameter Count

| Component | Parameters |
|-----------|------------|
| Context Encoder (4 layers) | 20,000,000 |
| Multi-Step Predictor | 5,000,000 |
| Embedding Consistency Heads | 5,000,000 |
| **Total** | **~30,000,000** |

---

## 6. Auto-Regressive Head (~60M Parameters)

### 6.1 Architecture

```
Input: Joint embeddings [N, 512]
         ↓
┌─────────────────────────────────────┐
│  Projection Layer                   │
│  - Linear(512 → 768)                │
│  - Params: 512 × 768 = 393K        │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  Auto-Regressive Transformer (×6)  │
│  For each layer:                    │
│  ┌──────────────────────────────┐  │
│  │ 1. RMSNorm (pre-attention)  │  │
│  │ 2. Causal Self-Attention     │  │
│  │    - MQA: 12 Q, 3 KV heads   │  │
│  │    - Causal mask applied     │  │
│  │ 3. Residual +1               │  │
│  │ 4. RMSNorm (pre-MLP)         │  │
│  │ 5. SwiGLU MLP                │  │
│  │    - 768 → 3072 → 768        │  │
│  │ 6. Residual +2               │  │
│  └──────────────────────────────┘  │
│  - Params per layer: ~6.5M       │
│  - Total: 6 × 6.5M = 39M         │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  Language Model Head                │
│  - Linear(768 → vocab_size)        │
│  - Params: 768 × 50,000 = 38.4M   │
│  - Can tie with input embeddings   │
└──────────────┬──────────────────────┘
               ▼
        Output: Logits [N, vocab_size]
```

### 6.2 Component Specifications

```cpp
struct AutoRegressiveHead {
    // Projection
    float* proj;           // [768, 512]

    // AR Transformer layers (×6)
    float* qkv_proj[6];    // [768, 1152]
    float* o_proj[6];      // [768, 768]
    float* gate_up_proj[6];// [3072, 1536]
    float* down_proj[6];   // [768, 3072]
    float* attn_norm[6];   // [768]
    float* mlp_norm[6];    // [768]

    // LM head
    float* lm_head;        // [vocab_size, 768]

    // Causal mask (pre-computed)
    // bool causal_mask[max_seq_len][max_seq_len];

    // KV cache for fast generation
    struct KVCache {
        float* k_cache;     // [max_seq_len, 3, 64]  // 3 KV heads × 64 dim
        float* v_cache;     // [max_seq_len, 3, 64]
        int current_len;
    } kv_cache[6];  // One per layer
};
```

### 6.3 Parameter Count

| Component | Parameters |
|-----------|------------|
| Projection (512 → 768) | 393,216 |
| AR Transformer (6 layers) | 39,000,000 |
| LM Head (768 → 50K) | 38,400,000 |
| **Total (untied)** | **~78,000,000** |

**Note**: If LM head is tied with input embeddings, subtract 38.4M → **~40M total**

---

## 7. Memory Layout and AVX2 Optimization

### 7.1 Tensor Layout Strategy

**NHWC vs NCHW**:
- **Attention**: NHWC (batch, seq_len, heads, head_dim)
- **Vision**: NHWC (batch, height, width, channels)
- **Dense layers**: Row-major (output_dim, input_dim)

**Rationale**:
- Sequential access along fastest-changing dimension
- Cache-friendly for matrix multiplication
- AVX2-compatible stride (8 floats = 32 bytes)

### 7.2 Alignment Requirements

```cpp
// All memory allocations aligned to 32 bytes
constexpr int kAVX2Alignment = 32;

// Aligned allocation
inline float* aligned_alloc(size_t size) {
    void* ptr = nullptr;
    posix_memalign(&ptr, kAVX2Alignment, size);
    return static_cast<float*>(ptr);
}

// Aligned load/store
__m256 v = _mm256_load_ps(ptr);   // Must be 32-byte aligned
_mm256_store_ps(ptr, v);           // Must be 32-byte aligned
```

### 7.3 Cache Blocking Strategy

**Matrix Multiplication**:
- L3 tiles: 512×512 (fits in 64MB L3)
- L2 tiles: 64×64 (fits in 512KB L2)
- L1 micro-kernel: 8×8 (AVX2 registers)

**Attention Computation**:
- Block QK^T in 64×64 tiles
- Process softmax in blocks
- Accumulate weighted sum in AVX2 registers

---

## 8. Hybrid Training Objective

### 8.1 Loss Function

```
L_total = α × L_JEPA + β × L_LM

Where:
L_JEPA = MSE(z_predicted, z_target)
       = (1/N) Σ ||z_pred - z_target||²

L_LM = CrossEntropy(logits, targets)
     = -(1/N) Σ log P(token_t | token_<t)

Default weights: α = 0.3, β = 0.7
```

### 8.2 JEPA Masking Strategy

```cpp
// Mask sampling for JEPA
struct JEPAMasking {
    float mask_ratio;  // 0.15 (15%)

    void generate_mask(
        bool* mask,           // [N] output
        int N,
        MaskStrategy strategy  // BLOCK, RANDOM, SPAN
    );

    // Block masking (recommended for vision)
    void block_mask(
        bool* mask,
        int H, int W,  // Spatial dimensions
        int block_size = 16
    );
};
```

### 8.3 Multi-Step Prediction

```cpp
// JEPA multi-step prediction
struct MultiStepPredictor {
    int num_steps = 3;

    void predict(
        const float* context,    // [N, 512]
        float* predictions,      // [N, num_steps, 512] output
        int N
    );

    // Weighted loss (recent steps more important)
    float compute_loss(
        const float* predictions,  // [N, num_steps, 512]
        const float* targets,      // [N, num_steps, 512]
        int N
    );
};
```

---

## 9. Summary of Parameter Distribution

| Component | Parameters | Percentage |
|-----------|------------|-------------|
| Vision Encoder | 50M | 16.7% |
| Language Encoder | 120M | 40.0% |
| Cross-Modal Fusion | 40M | 13.3% |
| JEPA Prediction Head | 30M | 10.0% |
| Auto-Regressive Head | 60M | 20.0% |
| **TOTAL** | **300M** | **100%** |

**Memory Footprint** (FP32):
- Parameters: 300M × 4 bytes = **1.2 GB**
- Activations (batch=1, 4096 context): ~80 MB
- KV cache: ~48 MB
- **Total inference memory**: **~1.3 GB**

---

**Document Status**: Updated for VL-JEPA
**Next Document**: 02_HARDWARE_OPTIMIZATION.md
