# Cromwell VL-JEPA: Hardware Optimization Strategy
## L3 Cache, AVX2 SIMD, and AMD Zen Architecture Optimization for Vision-Language Models

**Version**: 2.0
**Date**: 2025-01-14
**Target Hardware**: AMD Ryzen 9 5900X (Zen 3) and compatible AMD CPUs

---

## 1. Target Hardware Analysis

### 1.1 AMD Zen 3 Microarchitecture (Ryzen 9 5900X)

```
CPU Specifications:
- Cores: 12 cores / 24 threads (2 CCX, 6 cores each)
- Base Clock: 3.7 GHz
- Boost Clock: 4.8 GHz
- L1 Cache: 64 KB (32 KB data + 32 KB instruction) per core
- L2 Cache: 512 KB per core (8-way associative)
- L3 Cache: 32 MB per CCX (64 MB total, 16-way associative)
- Memory: DDR4-3200 (dual channel, ~50 GB/s bandwidth)
- SIMD: AVX2 (256-bit), FMA3, BMI2
- TDP: 105W
```

### 1.2 Cache Hierarchy & Performance

```
Latency (approximate):
- L1 hit: 4-5 cycles
- L2 hit: 12 cycles
- L3 hit: 40-50 cycles
- RAM access: 150-200 cycles

Bandwidth (per core):
- L1: ~1000 GB/s (theoretical)
- L2: ~500 GB/s
- L3: ~200 GB/s
- RAM: ~4-5 GB/s (single core)

Key Insight: L3 cache is 40-50x faster than RAM
Goal: Keep working set in L3 cache (32-64 MB)
```

### 1.3 AVX2 Capabilities

```
Register File:
- 16 × 256-bit YMM registers (YMM0-YMM15)
- Each register holds 8 × float32 or 4 × float64
- FMA3: Fused multiply-add (a × b + c in one operation)

Throughput (per cycle):
- 2 × FP32 FMA operations = 16 FLOPs/cycle
- At 4.8 GHz: ~77 GFLOPs theoretical peak per core

Key Operations:
- _mm256_load_ps/_mm256_store_ps: Aligned load/store
- _mm256_fmadd_ps: Fused multiply-add
- _mm256_add_ps/_mm256_mul_ps: Arithmetic
- _mm256_sqrt_ps/_mm256_rsqrt_ps: Square root
```

---

## 2. Memory Layout Strategy

### 2.1 Tensor Formats

**NHWC Layout for Vision and Attention**:

```cpp
// Vision: [Batch, Height, Width, Channels]
// Attention: [Batch, SeqLen, Heads, HeadDim]

// Why NHWC?
// 1. Sequential access along channels
// 2. Better cache locality
// 3. AVX2-friendly stride (8 floats = 32 bytes)
// 4. No padding needed for most operations

struct TensorLayout {
    // Vision encoder
    float* stem_conv;        // [H_out, W_out, 3, 96] NHWC
    float* mbconv_features;  // [H, W, C] NHWC
    float* vit_patches;      // [H/8, W/8, 384] NHWC

    // Attention (language + cross-modal)
    float* qkv;              // [SeqLen, Heads, HeadDim*3] NHWC
    float* attention_scores;  // [Heads, QSeqLen, KSeqLen]
    float* output;           // [SeqLen, Heads, HeadDim] NHWC
};
```

### 2.2 Alignment Requirements

```cpp
// 32-byte alignment for AVX2
constexpr int kAVX2Alignment = 32;
constexpr int kAVX2Width = 8;  // 8 float32 per register

// Aligned allocation
inline float* aligned_alloc(size_t size) {
    void* ptr = nullptr;
    posix_memalign(&ptr, kAVX2Alignment, size);
    return static_cast<float*>(ptr);
}

// Padding to alignment
inline size_t aligned_size(size_t size) {
    return (size + kAVX2Alignment - 1) & ~(kAVX2Alignment - 1);
}
```

### 2.3 Cache Blocking Strategy

**Matrix Multiplication (GEMM)**:

```cpp
// Block sizes for different cache levels
struct GEMMBlocks {
    int L3_tile = 512;   // 512×512 × 4 bytes = 1 MB (fits in L3)
    int L2_tile = 64;    // 64×64 × 4 bytes = 16 KB (fits in L2)
    int L1_micro = 8;    // 8×8 × 4 bytes = 256 B (fits in L1)
};

// Blocking hierarchy:
// C[512×512] = A[512×K] × B[K×512]
//   → Process in 64×64 tiles (L2)
//     → Each tile uses 8×8 micro-kernel (AVX2 registers)
```

**Vision Operations**:

```cpp
// Patch embedding with blocking
struct VisionBlocks {
    int patch_size = 4;      // 4×4 patches
    int block_h = 64;        // Process 64×64 spatial region
    int block_w = 64;        // Fits in L2 cache
};

// Windowed attention blocking
struct WindowBlocks {
    int window_size = 16;    // 16×16 attention windows
    int num_windows_h = H / 16;
    int num_windows_w = W / 16;
};
```

---

## 3. AVX2 Kernels for Language Operations

### 3.1 Matrix Multiplication (8×8 Micro-Kernel)

```cpp
// 8×8 micro-kernel for AVX2
// Computes C[8×8] += A[8×K] × B[K×8]
inline void gemm_8x8_avx2(
    const float* A,  // [8, K]
    const float* B,  // [K, 8]
    float* C,        // [8, 8]
    int K
) {
    // Accumulators (8 rows × 8 columns)
    __m256 c[8];  // c[i] holds row i, columns 0-7

    // Initialize accumulators
    for (int i = 0; i < 8; i++) {
        c[i] = _mm256_setzero_ps();
    }

    // Main loop (process K in steps of 8)
    for (int k = 0; k < K; k += 8) {
        for (int i = 0; i < 8; i++) {
            // Load 8 elements from A[i, k:k+8]
            __m256 a = _mm256_loadu_ps(&A[i * K + k]);

            for (int j = 0; j < 8; j++) {
                // Load 8 elements from B[k:k+8, j]
                __m256 b = _mm256_loadu_ps(&B[k * 8 + j * K]);

                // FMA: C[i,j] += A[i,k] * B[k,j]
                c[i] = _mm256_fmadd_ps(a, b, c[i]);
            }
        }
    }

    // Store results
    for (int i = 0; i < 8; i++) {
        _mm256_storeu_ps(&C[i * 8], c[i]);
    }
}
```

### 3.2 Attention Computation

```cpp
// QK^T computation with AVX2
inline void qk_transpose_avx2(
    const float* Q,  // [SeqLen, Heads, HeadDim]
    const float* K,  // [SeqLen, Heads, HeadDim]
    float* scores,   // [Heads, QSeqLen, KSeqLen]
    int QSeqLen,
    int KSeqLen,
    int num_heads,
    int head_dim
) {
    int dim_per_head = head_dim / 8;  // Number of AVX2 vectors

    for (int h = 0; h < num_heads; h++) {
        for (int i = 0; i < QSeqLen; i++) {
            for (int j = 0; j < KSeqLen; j++) {
                // Compute dot product Q[i] · K[j]
                __m256 sum = _mm256_setzero_ps();

                const float* q_ptr = &Q[(i * num_heads + h) * head_dim];
                const float* k_ptr = &K[(j * num_heads + h) * head_dim];

                for (int d = 0; d < dim_per_head; d++) {
                    __m256 q = _mm256_loadu_ps(&q_ptr[d * 8]);
                    __m256 k = _mm256_loadu_ps(&k_ptr[d * 8]);
                    sum = _mm256_fmadd_ps(q, k, sum);
                }

                // Horizontal sum
                scores[h * QSeqLen * KSeqLen + i * KSeqLen + j] =
                    horizontal_sum_ps(sum);
            }
        }
    }
}
```

### 3.3 Softmax with AVX2

```cpp
// Numerically stable softmax
inline void softmax_avx2(
    float* x,      // [N]
    float* output, // [N]
    int N
) {
    // Find max (AVX2)
    __m256 max_val = _mm256_set1_ps(-INFINITY);
    for (int i = 0; i < N; i += 8) {
        __m256 v = _mm256_loadu_ps(&x[i]);
        max_val = _mm256_max_ps(max_val, v);
    }
    float max_scalar = horizontal_max_ps(max_val);

    // Compute exp(x - max) and sum
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < N; i += 8) {
        __m256 v = _mm256_sub_ps(_mm256_loadu_ps(&x[i]), _mm256_set1_ps(max_scalar));
        __m256 exp_v = exp_ps(v);
        _mm256_storeu_ps(&output[i], exp_v);
        sum = _mm256_add_ps(sum, exp_v);
    }
    float sum_scalar = horizontal_sum_ps(sum);

    // Normalize
    __m256 inv_sum = _mm256_set1_ps(1.0f / sum_scalar);
    for (int i = 0; i < N; i += 8) {
        __m256 v = _mm256_loadu_ps(&output[i]);
        _mm256_storeu_ps(&output[i], _mm256_mul_ps(v, inv_sum));
    }
}
```

---

## 4. AVX2 Kernels for Vision Operations

### 4.1 Patch Embedding (Conv 4×4, stride=4)

```cpp
// 4×4 patch embedding with AVX2
inline void patch_embed_4x4_avx2(
    const float* input,  // [H, W, 3] NHWC
    float* output,       // [H/4, W/4, 96] NHWC
    int H, int W,
    const float* kernel  // [4, 4, 3, 96]
) {
    int H_out = H / 4;
    int W_out = W / 4;

    for (int h_out = 0; h_out < H_out; h_out++) {
        for (int w_out = 0; w_out < W_out; w_out++) {
            int h_in = h_out * 4;
            int w_in = w_out * 4;

            // Process 96 output channels (12 × 8 for AVX2)
            for (int c = 0; c < 96; c += 8) {
                __m256 acc[8] = {_mm256_setzero_ps()};

                // 4×4 × 3 = 48 input pixels
                for (int kh = 0; kh < 4; kh++) {
                    for (int kw = 0; kw < 4; kw++) {
                        for (int cin = 0; cin < 3; cin++) {
                            int h_idx = h_in + kh;
                            int w_idx = w_in + kw;

                            if (h_idx < H && w_idx < W) {
                                float input_val = input[(h_idx * W + w_idx) * 3 + cin];

                                // Load kernel weights for 8 output channels
                                __m256 k = _mm256_loadu_ps(&kernel[((kh * 4 + kw) * 3 + cin) * 96 + c]);

                                // FMA: acc += input * kernel
                                for (int i = 0; i < 8; i++) {
                                    acc[i] = _mm256_fmadd_ps(
                                        _mm256_set1_ps(input_val),
                                        _mm256_permute2f128_ps(k, k, i),
                                        acc[i]
                                    );
                                }
                            }
                        }
                    }
                }

                // Store 8 output channels
                int out_idx = (h_out * W_out + w_out) * 96 + c;
                for (int i = 0; i < 8; i++) {
                    _mm256_storeu_ps(&output[out_idx + i], acc[i]);
                }
            }
        }
    }
}
```

### 4.2 Depthwise Separable Convolution (MBConv)

```cpp
// Depthwise 3×3 convolution with AVX2
inline void depthwise_conv_3x3_avx2(
    const float* input,  // [H, W, C]
    float* output,       // [H, W, C]
    int H, int W, int C,
    const float* kernel  // [3, 3, C]
) {
    // Process each channel independently
    for (int c = 0; c < C; c += 8) {
        for (int h = 1; h < H - 1; h++) {
            for (int w = 1; w < W - 1; w += 8) {
                __m256 sum = _mm256_setzero_ps();

                // 3×3 kernel
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        int h_idx = h + kh - 1;
                        int w_idx = w + kw - 1;

                        // Load 8 pixels
                        __m256 in = _mm256_loadu_ps(&input[(h_idx * W + w_idx) * C + c]);

                        // Load kernel weight
                        int k_idx = (kh * 3 + kw) * C + c;
                        __m256 k = _mm256_set1_ps(kernel[k_idx]);

                        // FMA
                        sum = _mm256_fmadd_ps(in, k, sum);
                    }
                }

                // Store 8 output pixels
                int out_idx = (h * W + w) * C + c;
                _mm256_storeu_ps(&output[out_idx], sum);
            }
        }
    }
}
```

### 4.3 Windowed Attention for Vision

```cpp
// Windowed attention (16×16 windows)
inline void windowed_attention_avx2(
    const float* input,  // [H, W, C] where C = num_heads * head_dim
    float* output,       // [H, W, C]
    int H, int W, int C,
    int window_size = 16
) {
    int num_heads = 6;
    int head_dim = C / num_heads;  // 64

    int num_windows_h = (H + window_size - 1) / window_size;
    int num_windows_w = (W + window_size - 1) / window_size;

    for (int wh = 0; wh < num_windows_h; wh++) {
        for (int ww = 0; ww < num_windows_w; ww++) {
            // Compute window boundaries
            int h_start = wh * window_size;
            int w_start = ww * window_size;
            int h_end = min(h_start + window_size, H);
            int w_end = min(w_start + window_size, W);

            int window_h = h_end - h_start;
            int window_w = w_end - w_start;
            int window_size = window_h * window_w;

            // Process attention within window
            // ... (similar to standard attention but limited to window)
        }
    }
}
```

### 4.4 Squeeze-and-Excitation (SE) with AVX2

```cpp
// SE attention (channel-wise)
inline void se_attention_avx2(
    float* input,  // [H, W, C]
    int H, int W, int C,
    const float* fc1,  // [C, C/r]
    const float* fc2,  // [C/r, C]
    int reduction = 4
) {
    int C_reduced = C / reduction;

    // Global average pooling (AVX2)
    __m256* global_avg = new __m256[C / 8];
    for (int c = 0; c < C; c += 8) {
        __m256 sum = _mm256_setzero_ps();

        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                __m256 v = _mm256_loadu_ps(&input[(h * W + w) * C + c]);
                sum = _mm256_add_ps(sum, v);
            }
        }

        // Divide by H*W
        __m256 scale = _mm256_set1_ps(1.0f / (H * W));
        global_avg[c / 8] = _mm256_mul_ps(sum, scale);
    }

    // FC1: C → C/r with Swish
    __m256* fc1_out = new __m256[C_reduced / 8];
    // ... matrix multiplication with AVX2

    // FC2: C/r → C with sigmoid
    __m256* excitation = new __m256[C / 8];
    // ... matrix multiplication with AVX2 + sigmoid

    // Apply excitation
    for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
            for (int c = 0; c < C; c += 8) {
                __m256 v = _mm256_loadu_ps(&input[(h * W + w) * C + c]);
                __m256 exc = excitation[c / 8];
                _mm256_storeu_ps(&input[(h * W + w) * C + c], _mm256_mul_ps(v, exc));
            }
        }
    }

    delete[] global_avg;
    delete[] fc1_out;
    delete[] excitation;
}
```

---

## 5. Cross-Modal Attention Optimization

### 5.1 Vision-to-Language Attention

```cpp
// V→L cross-attention with AVX2
inline void vision_to_language_attention_avx2(
    const float* V,  // [N_v, 512] vision tokens
    const float* L,  // [N_l, 512] language tokens
    float* output,   // [N_l, 512]
    int N_v, int N_l
) {
    // Compute attention scores: L @ V.T
    // Shape: [N_l, N_v]

    int dim = 512 / 8;  // AVX2 vectors

    for (int i = 0; i < N_l; i++) {
        for (int j = 0; j < N_v; j++) {
            // Dot product L[i] · V[j]
            __m256 sum = _mm256_setzero_ps();

            const float* l_ptr = &L[i * 512];
            const float* v_ptr = &V[j * 512];

            for (int d = 0; d < dim; d++) {
                __m256 l = _mm256_loadu_ps(&l_ptr[d * 8]);
                __m256 v = _mm256_loadu_ps(&v_ptr[d * 8]);
                sum = _mm256_fmadd_ps(l, v, sum);
            }

            // Store attention score
            // ... (apply softmax, then weighted sum)
        }
    }
}
```

### 5.2 Batch Processing for Cross-Modal

```cpp
// Process multiple attention heads in parallel
inline void batch_cross_modal_attention_avx2(
    const float* V,
    const float* L,
    float* output,
    int N_v, int N_l,
    int num_heads = 8
) {
    int head_dim = 512 / num_heads;  // 64

    // Process all heads in parallel (AVX2)
    for (int h = 0; h < num_heads; h++) {
        int offset = h * head_dim;

        for (int i = 0; i < N_l; i++) {
            for (int j = 0; j < N_v; j++) {
                // Compute attention for head h
                __m256 sum = _mm256_setzero_ps();

                for (int d = 0; d < head_dim / 8; d++) {
                    __m256 l = _mm256_loadu_ps(&L[i * 512 + offset + d * 8]);
                    __m256 v = _mm256_loadu_ps(&V[j * 512 + offset + d * 8]);
                    sum = _mm256_fmadd_ps(l, v, sum);
                }

                // ...
            }
        }
    }
}
```

---

## 6. Memory Prefetching Strategy

### 6.1 Software Prefetch for GEMM

```cpp
// Prefetch next tile while processing current tile
inline void gemm_with_prefetch(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    constexpr int kTileSize = 64;

    for (int i = 0; i < M; i += kTileSize) {
        for (int j = 0; j < N; j += kTileSize) {
            // Prefetch next B tile
            if (j + kTileSize < N) {
                __builtin_prefetch(&B[(j + kTileSize) * K], 0, 3);
            }

            // Process current tile
            for (int ii = i; ii < min(i + kTileSize, M); ii++) {
                // Prefetch next A row
                if (ii + 8 < min(i + kTileSize, M)) {
                    __builtin_prefetch(&A[(ii + 8) * K], 0, 3);
                }

                for (int jj = j; jj < min(j + kTileSize, N); jj += 8) {
                    __m256 c = _mm256_setzero_ps();

                    for (int kk = 0; kk < K; kk += 8) {
                        __m256 a = _mm256_loadu_ps(&A[ii * K + kk]);
                        __m256 b = _mm256_loadu_ps(&B[kk * N + jj]);
                        c = _mm256_fmadd_ps(a, b, c);
                    }

                    _mm256_storeu_ps(&C[ii * N + jj], c);
                }
            }
        }
    }
}
```

### 6.2 Prefetch for Vision Operations

```cpp
// Prefetch next image patch while processing current
inline void conv_with_prefetch(
    const float* input,
    float* output,
    int H, int W, int C,
    const float* kernel
) {
    constexpr int kPatchSize = 16;

    for (int h = 0; h < H; h += kPatchSize) {
        for (int w = 0; w < W; w += kPatchSize) {
            // Prefetch next patch
            if (h + kPatchSize < H) {
                __builtin_prefetch(&input[((h + kPatchSize) * W + w) * C], 0, 3);
            }

            // Process current patch
            for (int ph = 0; ph < kPatchSize && h + ph < H; ph++) {
                for (int pw = 0; pw < kPatchSize && w + pw < W; pw++) {
                    // ... convolution computation
                }
            }
        }
    }
}
```

---

## 7. Performance Targets

### 7.1 Per-Component Estimates

| Component | Operations | AVX2 Speedup | Target Time |
|-----------|-----------|-------------|-------------|
| **Vision Encoder** | | | |
| - Stem (4×4 conv) | 2.1 GFLOPs | 4× | ~2 ms |
| - MBConv blocks | 8.5 GFLOPs | 4× | ~8 ms |
| - ViT blocks | 12.3 GFLOPs | 4× | ~12 ms |
| **Language Encoder** | | | |
| - Embeddings | 0.3 GFLOPs | 2× | ~1 ms |
| - 12 layers (2048 seq) | 45 GFLOPs | 4× | ~22 ms |
| **Cross-Modal Fusion** | | | |
| - 6 blocks | 8.2 GFLOPs | 4× | ~8 ms |
| **JEPA Head** | 3.5 GFLOPs | 4× | ~3 ms |
| **AR Head** | 6.8 GFLOPs | 4× | ~6 ms |
| **Total (per token)** | | | ~35 ms (with vision) |

### 7.2 Throughput Estimates

**Text-only generation:**
- Per token: ~22 ms
- Throughput: ~45 tokens/second

**Vision + text generation:**
- Vision encode: ~22 ms (one-time)
- Per token: ~35 ms
- First token: ~57 ms
- Throughput: ~28 tokens/second

---

## 8. Optimization Checklist

### 8.1 Memory Layout
- [x] NHWC format for all tensors
- [x] 32-byte alignment for all allocations
- [x] Cache-friendly access patterns
- [x] Minimal padding and reordering

### 8.2 AVX2 Kernels
- [x] GEMM 8×8 micro-kernel
- [x] Attention computation
- [x] Softmax with exp approximation
- [x] RMSNorm
- [x] Patch embedding (4×4 conv)
- [x] Depthwise separable conv
- [x] Windowed attention
- [x] Cross-modal attention

### 8.3 Cache Optimization
- [x] L3 cache blocking (512×512)
- [x] L2 cache tiles (64×64)
- [x] L1 micro-kernel (8×8)
- [x] Software prefetching

### 8.4 Vision-Specific
- [x] Windowed attention (16×16)
- [x] Efficient patch processing
- [x] SE attention optimization
- [x] Batch processing for multiple heads

---

**Document Status**: Updated for VL-JEPA with vision operations
**Next Document**: 03_TOKENIZATION.md
