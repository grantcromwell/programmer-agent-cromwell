# Cromwell Agent: Model Architecture Specifications
## Detailed Design for Decoder-Only Transformer with MQA

**Version**: 1.0
**Date**: 2025-01-14

---

## 1. Architecture Overview

```
Input Token IDs (Long Integer)
         ↓
   Embedding Layer
   (Token Embeddings + RoPE)
         ↓
   Transformer Block × 24
   ┌─────────────────────┐
   │  1. RMSNorm (input) │
   │  2. Self-Attention  │
   │     (MQA + RoPE)    │
   │  3. Residual +1     │
   │  4. RMSNorm (post)  │
   │  5. MLP (SwiGLU)    │
   │  6. Residual +2     │
   └─────────────────────┘
         ↓
   Final RMSNorm
         ↓
   Output Projection
   (to vocabulary logits)
         ↓
   Softmax / Sampling
```

---

## 2. Component Specifications

### 2.1 Embedding Layer

**Purpose**: Convert token IDs to continuous representations

**Specifications**:
```cpp
class EmbeddingLayer {
public:
    // Parameters
    float* token_embeddings;  // [vocab_size, hidden_size]
    int vocab_size = 50000;
    int hidden_size = 2048;

    // Forward pass
    void forward(
        const int64_t* input_ids,  // [batch_size, seq_len]
        float* output,             // [batch_size, seq_len, hidden_size]
        int batch_size,
        int seq_len
    );

    // Memory layout: row-major for cache efficiency
    // token_embeddings[token_id] gives hidden_dim vector
    // Aligned to 32 bytes for AVX2
};
```

**Memory Layout**:
- Shape: `[vocab_size, hidden_size]`
- Order: Row-major (token ID fastest changing)
- Alignment: 32 bytes (AVX2 requirement)
- Size: 50000 × 2048 × 4 bytes = 384 MB

**Optimizations**:
1. **Pre-fetch embeddings**: For common tokens (top 1000)
2. **SIMD gather**: Use AVX2 for batch lookup
3. **Cache alignment**: Align to L1 cache line (64 bytes)

### 2.2 Rotary Positional Embeddings (RoPE)

**Purpose**: Inject position information into queries and keys

**Mathematical Formulation**:

```python
# For a position m and dimension d (head_dim / 2)
theta_m_i = m / (base^(2i/d))

# Rotation matrix for 2D subspaces
R(m, theta) = [
    [cos(theta_m_i), -sin(theta_m_i)],
    [sin(theta_m_i),  cos(theta_m_i)]
]

# Apply to query/key pairs
q_rotated = R @ q
k_rotated = R @ k
```

**Implementation**:

```cpp
struct RoPEConfig {
    int head_dim = 64;           // Must be even for pairing
    float base = 10000.0f;       // Rotary base
    int max_position = 4096;     // Maximum sequence length
};

class RotaryEmbeddings {
public:
    // Pre-computed frequencies
    float* freqs;  // [max_position, head_dim / 2]
    float* inv_freqs;  // [head_dim / 2]

    void initialize(const RoPEConfig& config);

    // Apply rotary embeddings in-place
    void apply(
        float* q,  // [seq_len, num_heads, head_dim]
        float* k,  // [seq_len, num_kv_heads, head_dim]
        int seq_len,
        int num_heads,
        int num_kv_heads,
        int head_dim
    );

private:
    // SIMD-optimized rotation for single 2D subspace
    inline void rotate_2d(
        float& x,
        float& y,
        float cos_theta,
        float sin_theta
    ) {
        // Rotation: [x', y'] = [x*cos - y*sin, x*sin + y*cos]
        float x_new = x * cos_theta - y * sin_theta;
        float y_new = x * sin_theta + y * cos_theta;
        x = x_new;
        y = y_new;
    }
};
```

**Memory Layout**:
- `freqs`: `[max_position, head_dim / 2]` (float32)
- Size: 4096 × 32 × 4 bytes = 512 KB (fits in L2 cache)

**Optimizations**:
1. **Pre-compute frequencies**: Compute once at initialization
2. **SIMD rotation**: Process 4 pairs per AVX2 instruction
3. **Interleave Q/K processing**: Better cache utilization

### 2.3 Multi-Query Attention (MQA)

**Purpose**: Efficient attention with shared key/value projections

**Architecture**:

```
Input: [batch_size, seq_len, hidden_size]
         ↓
    QKV Projection
         ↓
    ┌─────────────────────────────────────────┐
    │ Q: [seq_len, num_heads, head_dim]       │
    │    32 query heads (independent)          │
    │                                          │
    │ K: [seq_len, num_kv_heads, head_dim]    │
    │    4 key-value heads (shared)            │
    │                                          │
    │ V: [seq_len, num_kv_heads, head_dim]    │
    │    4 key-value heads (shared)            │
    └─────────────────────────────────────────┘
         ↓
    Apply RoPE to Q and K
         ↓
    QK^T (Attention Scores)
         ↓
    Causal Mask + Softmax
         ↓
    Attention Weights @ V
         ↓
    Output Projection
         ↓
    [seq_len, hidden_size]
```

**Implementation**:

```cpp
struct MultiQueryAttentionConfig {
    int hidden_size = 2048;
    int num_heads = 32;          // Query heads
    int num_kv_heads = 4;        // Key/Value heads (8:1 ratio)
    int head_dim = 64;           // hidden_size / num_heads
    float attention_dropout = 0.0f;
    bool use_cache = true;
};

class MultiQueryAttention {
public:
    // Parameters
    float* qkv_proj;      // [hidden_size, hidden_size + 2 * (hidden_size / num_heads * num_kv_heads)]
    float* o_proj;        // [hidden_size, hidden_size]

    // KV cache for inference
    KVCache* cache;

    void forward(
        const float* input,       // [seq_len, hidden_size]
        float* output,            // [seq_len, hidden_size]
        int seq_len,
        bool use_causal_mask = true
    );

private:
    // Compute QK^T attention scores
    void compute_qk_attn(
        const float* q,  // [seq_len, num_heads, head_dim]
        const float* k,  // [cache_len, num_kv_heads, head_dim]
        float* attn,     // [seq_len, num_heads, cache_len]
        int seq_len,
        int cache_len
    );

    // Apply causal mask
    void apply_causal_mask(
        float* attn,  // [seq_len, num_heads, cache_len]
        int seq_len,
        int cache_len
    );

    // Compute softmax with numerical stability
    void softmax_inplace(
        float* input,  // [rows, cols]
        int rows,
        int cols
    );

    // Compute attention output: weights @ V
    void compute_attn_output(
        const float* attn,  // [seq_len, num_heads, cache_len]
        const float* v,     // [cache_len, num_kv_heads, head_dim]
        float* output,      // [seq_len, num_heads, head_dim]
        int seq_len,
        int cache_len
    );
};
```

**KV Cache Design**:

```cpp
class KVCache {
public:
    KVCache(int max_batch_size, int max_seq_len, int num_layers, int num_kv_heads, int head_dim);

    // Update cache with new keys and values
    void update(
        int layer_idx,
        const float* k,  // [seq_len, num_kv_heads, head_dim]
        const float* v,  // [seq_len, num_kv_heads, head_dim]
        int seq_len,
        int* new_cache_len  // Output: updated cache length
    );

    // Retrieve cached keys and values
    void get(
        int layer_idx,
        float** k,  // Output: [cache_len, num_kv_heads, head_dim]
        float** v,  // Output: [cache_len, num_kv_heads, head_dim]
        int& cache_len  // Output: current cache length
    );

    // Clear cache (for new sequence)
    void clear();

    // Memory usage per layer: max_seq_len * num_kv_heads * head_dim * 2 * 4 bytes
    // For 4096 cache, 4 heads, 64 dim: 4096 * 4 * 64 * 8 = 8 MB per layer
    // Total for 24 layers: ~192 MB

private:
    std::vector<float*> k_cache_;  // Per-layer key cache
    std::vector<float*> v_cache_;  // Per-layer value cache
    std::vector<int> cache_lengths_;  // Per-layer current length

    int max_batch_size_;
    int max_seq_len_;
    int num_layers_;
    int num_kv_heads_;
    int head_dim_;
};
```

**Memory Layout**:

```
QKV Projection:
- Shape: [hidden_size, qkv_total_dim]
- qkv_total_dim = hidden_size (Q) + 2 * (num_kv_heads * head_dim) (K, V)
- For our config: [2048, 2048 + 2 * (4 * 64)] = [2048, 2560]
- Size: 2048 × 2560 × 4 bytes = 20 MB

Output Projection:
- Shape: [hidden_size, hidden_size]
- Size: 2048 × 2048 × 4 bytes = 16 MB

KV Cache (inference):
- Per layer: max_seq_len × num_kv_heads × head_dim × 2 (K, V) × 4 bytes
- 4096 × 4 × 64 × 8 = 8 MB per layer
- Total (24 layers): ~192 MB
```

### 2.4 RMSNorm

**Purpose**: Normalize hidden states for stable training

**Mathematical Formulation**:

```python
RMSNorm(x) = x / sqrt(mean(x^2) + epsilon) * scale

where:
- mean(x^2) = sum(x^2) / hidden_size
- epsilon = 1e-6 (for numerical stability)
- scale: learned parameter (initialized to 1.0)
```

**Implementation**:

```cpp
class RMSNorm {
public:
    // Parameters
    float* scale;  // [hidden_size]
    int hidden_size;
    float epsilon = 1e-6f;

    void forward(
        const float* input,  // [batch_size, seq_len, hidden_size]
        float* output,       // [batch_size, seq_len, hidden_size]
        int batch_size,
        int seq_len
    );

private:
    // SIMD-optimized normalization for single vector
    void normalize_vector(
        const float* input,  // [hidden_size]
        float* output,       // [hidden_size]
        const float* scale   // [hidden_size]
    );
};
```

**AVX2 Optimization**:

```cpp
void RMSNorm::normalize_vector(
    const float* input,
    float* output,
    const float* scale
) {
    // Compute sum of squares
    __m256 sum_sq = _mm256_setzero_ps();

    int i = 0;
    for (; i <= hidden_size - 8; i += 8) {
        __m256 x = _mm256_load_ps(&input[i]);
        sum_sq = _mm256_fmadd_ps(x, x, sum_sq);  // sum_sq += x * x
    }

    // Horizontal sum
    float sum_array[8];
    _mm256_storeu_ps(sum_array, sum_sq);
    float mean_sq = 0.0f;
    for (int j = 0; j < 8; j++) {
        mean_sq += sum_array[j];
    }

    // Handle remaining elements
    for (; i < hidden_size; i++) {
        mean_sq += input[i] * input[i];
    }
    mean_sq /= static_cast<float>(hidden_size);

    // Compute normalization factor
    float norm_factor = 1.0f / sqrt(mean_sq + epsilon);

    // Apply normalization with scale
    __m256 norm_vec = _mm256_set1_ps(norm_factor);

    i = 0;
    for (; i <= hidden_size - 8; i += 8) {
        __m256 x = _mm256_load_ps(&input[i]);
        __m256 s = _mm256_load_ps(&scale[i]);
        __m256 normalized = _mm256_mul_ps(x, norm_vec);
        _mm256_store_ps(&output[i], _mm256_mul_ps(normalized, s));
    }

    // Handle remaining elements
    for (; i < hidden_size; i++) {
        output[i] = (input[i] * norm_factor) * scale[i];
    }
}
```

**Memory Layout**:
- Shape: `[hidden_size]`
- Size: 2048 × 4 bytes = 8 KB per layer
- Total (24 layers × 2 norms): ~384 KB

### 2.5 MLP with SwiGLU

**Purpose**: Non-linear transformation for expressiveness

**Architecture**:

```
Input: [seq_len, hidden_size]
         ↓
    Gate Projection: X @ W_gate
         ↓
    Up Projection: X @ W_up
         ↓
    Element-wise: Swish(Gate) ⊗ Up
         ↓
    Down Projection: (Swish(Gate) ⊗ Up) @ W_down
         ↓
Output: [seq_len, hidden_size]

Swish(x) = x * sigmoid(x)
```

**Implementation**:

```cpp
class SwiGLUMLP {
public:
    // Parameters
    float* gate_up_proj;  // [2 * intermediate_size, hidden_size] (fused)
    float* down_proj;     // [hidden_size, intermediate_size]

    int hidden_size = 2048;
    int intermediate_size = 5632;  // 2.75 * hidden_size

    void forward(
        const float* input,  // [seq_len, hidden_size]
        float* output,       // [seq_len, hidden_size]
        int seq_len
    );

private:
    // Swish activation: x * sigmoid(x)
    inline float swish(float x) {
        return x * (1.0f / (1.0f + exp(-x)));
    }
};
```

**Memory Layout**:

```
Gate/Up Projection (fused):
- Shape: [2 * intermediate_size, hidden_size]
- Size: 2 × 5632 × 2048 × 4 bytes = 90 MB

Down Projection:
- Shape: [hidden_size, intermediate_size]
- Size: 2048 × 5632 × 4 bytes = 45 MB

Total MLP parameters: ~135 MB per layer
Total for 24 layers: ~3.2 GB
```

**AVX2 Optimization**:

```cpp
void SwiGLUMLP::forward(
    const float* input,
    float* output,
    int seq_len
) {
    // Temporary buffer for intermediate activations
    std::vector<float> gate_up_buf(2 * intermediate_size);
    std::vector<float> activated_buf(intermediate_size);

    for (int t = 0; t < seq_len; t++) {
        const float* x = &input[t * hidden_size];

        // Compute gate and up projections (fused)
        // gate_up_proj is [2 * intermediate_size, hidden_size]
        // We compute: gate = x @ W_gate, up = x @ W_up
        // These are fused in the projection matrix

        for (int i = 0; i < 2 * intermediate_size; i += 8) {
            // Load 8 weights (for 8 outputs)
            __m256 sum = _mm256_setzero_ps();

            for (int j = 0; j < hidden_size; j += 8) {
                __m256 x_vec = _mm256_loadu_ps(&x[j]);
                __m256 w_vec = _mm256_loadu_ps(&gate_up_proj[i * hidden_size + j]);

                // FMA: sum += x * w
                sum = _mm256_fmadd_ps(x_vec, w_vec, sum);
            }

            // Horizontal sum
            float sum_array[8];
            _mm256_storeu_ps(sum_array, sum);

            for (int k = 0; k < 8; k++) {
                gate_up_buf[i + k] = sum_array[k];
            }
        }

        // Split into gate and up, apply Swish to gate, then element-wise multiply
        for (int i = 0; i < intermediate_size; i += 8) {
            __m256 gate = _mm256_load_ps(&gate_up_buf[i]);
            __m256 up = _mm256_load_ps(&gate_up_buf[intermediate_size + i]);

            // Swish(gate) = gate * sigmoid(gate)
            __m256 sigmoid_gate = compute_sigmoid_avx2(gate);
            __m256 activated = _mm256_mul_ps(gate, sigmoid_gate);

            // Element-wise multiply: Swish(gate) * up
            __m256 result = _mm256_mul_ps(activated, up);
            _mm256_store_ps(&activated_buf[i], result);
        }

        // Down projection
        for (int i = 0; i < hidden_size; i += 8) {
            __m256 sum = _mm256_setzero_ps();

            for (int j = 0; j < intermediate_size; j += 8) {
                __m256 act_vec = _mm256_load_ps(&activated_buf[j]);
                __m256 w_vec = _mm256_loadu_ps(&down_proj[i * intermediate_size + j]);

                sum = _mm256_fmadd_ps(act_vec, w_vec, sum);
            }

            float sum_array[8];
            _mm256_storeu_ps(sum_array, sum);

            for (int k = 0; k < 8; k++) {
                output[t * hidden_size + i + k] = sum_array[k];
            }
        }
    }
}
```

### 2.6 Transformer Block

**Purpose**: Combine attention and MLP with residual connections

**Architecture**:

```
Input: [seq_len, hidden_size]
         ↓
    1. RMSNorm(input)
         ↓
    2. Self-Attention
         ↓
    3. Residual: input + attention_output
         ↓
    4. RMSNorm(attention_output)
         ↓
    5. MLP
         ↓
    6. Residual: attention_output + mlp_output
         ↓
Output: [seq_len, hidden_size]
```

**Implementation**:

```cpp
class TransformerBlock {
public:
    // Components
    RMSNorm input_norm;
    RMSNorm post_attention_norm;
    MultiQueryAttention attention;
    SwiGLUMLP mlp;

    int layer_id;
    int hidden_size;

    void forward(
        const float* input,  // [seq_len, hidden_size]
        float* output,       // [seq_len, hidden_size]
        int seq_len,
        bool use_cache = true
    ) {
        // Allocate temporary buffers
        std::vector<float> attn_buf(seq_len * hidden_size);
        std::vector<float> mlp_buf(seq_len * hidden_size);

        // Attention block
        float* attn_input = attn_buf.data();
        input_norm.forward(input, attn_input, 1, seq_len);
        attention.forward(attn_input, attn_buf.data(), seq_len, true);

        // Residual connection
        for (int i = 0; i < seq_len * hidden_size; i++) {
            attn_buf[i] += input[i];
        }

        // MLP block
        float* mlp_input = mlp_buf.data();
        post_attention_norm.forward(attn_buf.data(), mlp_input, 1, seq_len);
        mlp.forward(mlp_input, mlp_buf.data(), seq_len);

        // Residual connection
        for (int i = 0; i < seq_len * hidden_size; i++) {
            output[i] = attn_buf[i] + mlp_buf[i];
        }
    }

    // Memory usage per layer:
    // - Input norm: 8 KB
    // - Attention: 36 MB (projections) + 8 MB (KV cache)
    // - Post norm: 8 KB
    // - MLP: 135 MB
    // Total: ~180 MB per layer
};
```

### 2.7 Full Model

**Purpose**: Stack transformer blocks into complete model

**Implementation**:

```cpp
class CromwellModel {
public:
    // Configuration
    ModelConfig config;

    // Components
    EmbeddingLayer embeddings;
    std::vector<TransformerBlock> layers;  // 24 layers
    RMSNorm final_norm;
    float* lm_head;  // [vocab_size, hidden_size]

    // KV cache for all layers
    std::vector<KVCache> kv_caches;

    void forward(
        const int64_t* input_ids,  // [batch_size, seq_len]
        float* logits,             // [batch_size, seq_len, vocab_size]
        int batch_size,
        int seq_len,
        bool use_cache = true
    ) {
        // Embeddings
        std::vector<float> hidden(batch_size * seq_len * config.hidden_size);
        embeddings.forward(input_ids, hidden.data(), batch_size, seq_len);

        // Transformer layers
        for (int layer_idx = 0; layer_idx < config.num_layers; layer_idx++) {
            std::vector<float> layer_output(batch_size * seq_len * config.hidden_size);

            layers[layer_idx].forward(
                hidden.data(),
                layer_output.data(),
                seq_len,
                use_cache
            );

            hidden = std::move(layer_output);
        }

        // Final normalization
        std::vector<float> normalized(batch_size * seq_len * config.hidden_size);
        final_norm.forward(hidden.data(), normalized.data(), batch_size, seq_len);

        // LM head projection
        // Compute: normalized @ lm_head^T
        // Output: [batch_size, seq_len, vocab_size]
        for (int b = 0; b < batch_size; b++) {
            for (int t = 0; t < seq_len; t++) {
                const float* h = &normalized[(b * seq_len + t) * config.hidden_size];
                float* logits_bt = &logits[(b * seq_len + t) * config.vocab_size];

                // Matrix-vector multiplication
                for (int v = 0; v < config.vocab_size; v += 8) {
                    __m256 sum = _mm256_setzero_ps();

                    for (int i = 0; i < config.hidden_size; i += 8) {
                        __m256 h_vec = _mm256_loadu_ps(&h[i]);
                        __m256 w_vec = _mm256_loadu_ps(&lm_head[v * config.hidden_size + i]);

                        sum = _mm256_fmadd_ps(h_vec, w_vec, sum);
                    }

                    float sum_array[8];
                    _mm256_storeu_ps(sum_array, sum);

                    for (int k = 0; k < 8; k++) {
                        logits_bt[v + k] = sum_array[k];
                    }
                }
            }
        }
    }

    // Memory usage summary:
    // - Embeddings: 384 MB
    // - 24 Transformer layers: ~4.3 GB (180 MB each)
    // - Final norm: 8 KB
    // - LM head: 384 MB
    // - KV cache: 192 MB (inference only)
    // Total: ~5.3 GB (training), ~5.5 GB (inference)
};
```

---

## 3. Parameter Count Summary

| Component | Shape | Parameters | Memory |
|-----------|-------|------------|--------|
| **Embeddings** | [50000, 2048] | 102.4M | 384 MB |
| **Per Layer** | | | |
| - Input norm | [2048] | 2,048 | 8 KB |
| - QKV projection | [2048, 2560] | 5.2M | 20 MB |
| - Output projection | [2048, 2048] | 4.2M | 16 MB |
| - Post norm | [2048] | 2,048 | 8 KB |
| - Gate/Up proj | [11264, 2048] | 23.1M | 90 MB |
| - Down proj | [2048, 5632] | 11.5M | 45 MB |
| **Total per layer** | | 44.0M | 171 MB |
| **24 layers** | | 1,056.0M | 4.1 GB |
| **Final norm** | [2048] | 2,048 | 8 KB |
| **LM head** | [50000, 2048] | 102.4M | 384 MB |
| **TOTAL** | | **1,260.8M** | **4.9 GB** |

**Note**: ~1.26B parameters, close to target of 1.2B

---

## 4. Forward Pass Computation

### 4.1 FLOPs Analysis

**Per token (excluding embeddings):**

```
For each layer (24 total):
1. Attention:
   - QKV projection: 3 * hidden_size^2 = 3 * 2048^2 = 12.6M FLOPs
   - QK^T: seq_len * hidden_size * cache_len (worst case: 4096 * 2048 * 4096) = 34.4B FLOPs
   - Softmax: O(seq_len * cache_len)
   - Attention @ V: seq_len * cache_len * hidden_size = 34.4B FLOPs
   - Output projection: hidden_size^2 = 4.2M FLOPs

2. MLP:
   - Gate/Up: 2 * hidden_size * intermediate_size = 2 * 2048 * 5632 = 23.1M FLOPs
   - Down: hidden_size * intermediate_size = 11.5M FLOPs

Total per token: ~69B FLOPs (worst case with full context)
```

**Optimization**: MQA reduces KV cache memory bandwidth by 8x

### 4.2 Memory Bandwidth Analysis

**Per token generation:**

```
Memory reads:
- Layer weights: 180 MB (must read for each layer)
- KV cache: 8 MB * 24 = 192 MB (accessed during attention)
- Activations: ~50 MB (forward pass buffers)

Total reads per token: ~422 MB

At 50 GB/s memory bandwidth (DDR4-3200):
Theoretical max: 50000 MB/s / 422 MB ≈ 118 tokens/second

Realistic (70% efficiency): ~80 tokens/second

With overhead (Python, batching): Target 50 tokens/second is achievable
```

---

## 5. Initialization Strategy

### 5.1 Weight Initialization

```cpp
void initialize_weights(CromwellModel* model, float scale = 0.02f) {
    // Embeddings: Normal(0, 0.02)
    for (int i = 0; i < model->config.vocab_size * model->config.hidden_size; i++) {
        model->embeddings.token_embeddings[i] = random_normal() * scale;
    }

    // Transformer layers
    for (auto& layer : model->layers) {
        // QKV projection: Kaiming initialization
        initialize_kaiming_uniform(
            layer.attention.qkv_proj,
            model->config.hidden_size,
            model->config.hidden_size + 2 * (model->config.num_kv_heads * model->config.head_dim)
        );

        // Output projection: Xavier initialization (gain = 1 / sqrt(2) for GELU)
        initialize_xavier_uniform(
            layer.attention.o_proj,
            model->config.hidden_size,
            model->config.hidden_size
        );

        // Normalization: Initialize scale to 1.0
        std::fill(layer.input_norm.scale, layer.input_norm.scale + model->config.hidden_size, 1.0f);
        std::fill(layer.post_attention_norm.scale, layer.post_attention_norm.scale + model->config.hidden_size, 1.0f);

        // MLP: Kaiming initialization
        initialize_kaiming_uniform(layer.mlp.gate_up_proj, model->config.intermediate_size, model->config.hidden_size);
        initialize_kaiming_uniform(layer.mlp.down_proj, model->config.hidden_size, model->config.intermediate_size);
    }

    // LM head: Tie with embeddings (share weights)
    std::copy(
        model->embeddings.token_embeddings,
        model->embeddings.token_embeddings + model->config.vocab_size * model->config.hidden_size,
        model->lm_head
    );
}

void initialize_kaiming_uniform(float* weights, int fan_in, int fan_out) {
    float bound = sqrt(6.0f / (fan_in + fan_out));
    for (int i = 0; i < fan_in * fan_out; i++) {
        weights[i] = random_uniform(-bound, bound);
    }
}
```

---

## 6. Next Steps

1. **Implement attention with AVX2** - Critical for performance
2. **Implement RoPE** - Position encoding
3. **Implement RMSNorm** - Normalization
4. **Implement SwiGLU** - Activation
5. **Stack into transformer block** - Full layer
6. **Test forward pass** - Verify correctness
7. **Benchmark** - Measure performance
8. **Optimize hot paths** - Profile and improve

---

**Document Status**: Complete
**Next Document**: 02_HARDWARE_OPTIMIZATION.md
