# Cromwell VL-JEPA: Vision-Language Auto-Regressive Coding Agent
## Complete System Architecture Design

**Version**: 2.0
**Date**: 2025-01-14
**Design Goal**: Build a production-grade vision-language JEPA model optimized for AMD CPUs with AVX2 support

---

## 1. Executive Summary

Cromwell VL-JEPA is a specialized 300M parameter vision-language model designed for code editing and understanding. It combines:

- **VL-JEPA Architecture**: Joint Embedding Predictive Architecture for vision-language fusion
- **Auto-Regressive Generation**: Hybrid training (JEPA + LM loss) for code generation
- **Hardware-optimized inference** on AMD CPUs (Zen 2/3/4) with AVX2
- **Full Multimodal Support**: Code highlighting, charts, diagrams, UI mockups
- **Parameter Efficient**: 300M parameters (4.2× smaller than original 1.26B design)

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **VL-JEPA Architecture** | Joint embedding space for robust vision-language representations |
| **Hybrid Training (JEPA + LM)** | JEPA for representation learning, LM for generation |
| **Multi-Query Attention (MQA)** | Faster inference, smaller memory footprint |
| **CNN-ViT Hybrid Vision Encoder** | Efficient vision processing with windowed attention |
| **Rotary Positional Embeddings (RoPE)** | Better extrapolation for long code files |
| **Custom tokenizer with code-aware merges** | Handles identifiers, strings, indentation |
| **C++ core with Python bindings** | Performance critical path in C++, flexibility in Python |
| **L3 cache-aware memory layout** | Critical for AMD Zen architecture performance |
| **AVX2 SIMD everywhere** | 4x speedup on matrix operations, attention, vision ops |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CROMWELL VL-JEPA SYSTEM                          │
│                    Target: < 300M Parameters                        │
└─────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────┐
│                        LAYER 1: USER INTERFACE                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   CLI Tool   │  │ Python API   │  │  HTTP Server │           │
│  │  (cromwell)  │  │ (lib.cromwell)│  │   (REST)    │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
└─────────┼──────────────────┼──────────────────┼──────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌───────────────────────────────────────────────────────────────────┐
│                        LAYER 2: I/O MANAGER                       │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Multimodal Input Interface                │ │
│  │  - Image loading (syntax highlighting, charts, diagrams)   │ │
│  │  - Text/code loading with encoding detection               │ │
│  │  - Diff generation (unified, context-aware)                │ │
│  │  - Patch application (safe, atomic operations)             │ │
│  └─────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Context Manager                           │ │
│  │  - File prioritization (relevance scoring)                 │ │
│  │  - Context window allocation (dynamic budgeting)           │ │
│  │  - Token counting (accurate, fast)                         │ │
│  │  - Window management (4096 token context)                  │ │
│  └─────────────────────────────────────────────────────────────┘ │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│                        LAYER 3: VL-JEPA MODEL                      │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Vision Encoder (~50M params)              │ │
│  │  - CNN stem (4x4 conv, stride=4)                           │ │
│  │  - MBConv blocks (×4, depthwise separable)                  │ │
│  │  - ViT blocks (×6, windowed attention, hidden=384)          │ │
│  │  - Output: [N_vision_patches, 384]                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Language Encoder (~120M params)           │ │
│  │  - Token embedding (vocab=50K, dim=768)                    │ │
│  │  - RoPE positional encoding                                │ │
│  │  - Transformer layers (×12, hidden=768, MQA)               │ │
│  │  - SwiGLU MLP (768 → 3072 → 768)                          │ │
│  │  - Output: [seq_len, 768]                                  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Cross-Modal Fusion (~40M params)          │ │
│  │  - Projection to common dimension (512)                    │ │
│  │  - Cross-attention blocks (×6, bidirectional V↔L)          │ │
│  │  - SwiGLU MLP (512 → 2048 → 512)                          │ │
│  │  - Output: [N_v+N_l, 512] joint embeddings                 │ │
│  └─────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    JEPA Prediction Head (~30M params)        │ │
│  │  - Temporal context encoder (4 layers, hidden=512)         │ │
│  │  - Multi-step predictor (z_{t+1}, z_{t+2}, z_{t+3})       │ │
│  │  - Loss: MSE in embedding space                            │ │
│  └─────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Auto-Regressive Head (~60M params)        │ │
│  │  - Projection (512 → 768)                                  │ │
│  │  - AR transformer (6 layers, hidden=768, MQA, causal)      │ │
│  │  - LM head (768 → vocab=50K)                               │ │
│  └─────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Sampling Engine                           │ │
│  │  - Temperature sampling                                      │ │
│  │  - Top-p (nucleus) sampling                                  │ │
│  │  - Repetition penalty                                        │ │
│  │  - Stop token detection                                      │ │
│  └─────────────────────────────────────────────────────────────┘ │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│                        LAYER 4: HARDWARE OPTIMIZATION             │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Memory Layout                             │ │
│  │  - Cache-friendly tensor ordering (NHWC for attention)      │ │
│  │  - 32-byte alignment (AVX2 requirement)                     │ │
│  │  - L3 cache-aware blocking (tile size: 512x512)             │ │
│  │  - Memory prefetching (software prefetch for latency hid)   │ │
│  └─────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    SIMD Kernels                              │ │
│  │  - AVX2 matrix multiplication (FMA3 support)                │ │
│  │  - AVX2 attention computation (QK^T + softmax)              │ │
│  │  - AVX2 layer normalization                                │ │
│  │  - AVX2 activation functions (Swish, GeLU)                  │ │
│  │  - AVX2 patch operations for vision                         │ │
│  │  - AVX2 2D attention for vision tokens                      │ │
│  └─────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    CPU Detection & Tuning                    │ │
│  │  - CPUID feature detection (AVX2, BMI2, FMA, AVX-512)      │ │
│  │  - Cache size detection (L1, L2, L3)                        │ │
│  │  - NUMA-aware allocation                                     │ │
│  │  - Thread pinning (performance cores)                       │ │
│  └─────────────────────────────────────────────────────────────┘ │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│                        LAYER 5: TRAINING PIPELINE                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Dataset Curation                          │ │
│  │  - TheStack v2 (code + natural language)                    │ │
│  │  - Code competition data (Codeforces, LeetCode)             │ │
│  │  - Code review data (GitHub PRs, comments)                  │ │
│  │  - CS/ML textbooks (arXiv, OpenStax)                        │ │
│  │  - Vision-text pairs (syntax highlighting, charts)          │ │
│  │  - Financial code datasets (trading, modeling)              │ │
│  └─────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Hybrid Training Curriculum                │ │
│  │  Stage 1: JEPA pre-training (masked embedding prediction)  │ │
│  │  Stage 2: LM fine-tuning (next-token prediction)           │ │
│  │  Stage 3: Joint training (hybrid JEPA + LM loss)            │ │
│  │  Stage 4: Instruction fine-tuning (code editing tasks)      │ │
│  └─────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
```

---

## 3. Model Architecture Specifications

### 3.1 Base Configuration

```yaml
model:
  name: "Cromwell-VL-JEPA-300M"
  architecture: "vision-language-jepa-autoregressive"
  parameters: 300M

  # Vision Encoder
  vision:
    encoder_type: "cnn-vit-hybrid"
    stem_channels: 96
    cnn_blocks: 4
    vit_blocks: 6
    vision_hidden_dim: 384
    vision_attention_heads: 6
    window_size: 16
    parameters: 50M

  # Language Encoder
  language:
    vocab_size: 50000
    hidden_size: 768
    intermediate_size: 3072  # 4 * hidden_size
    num_layers: 12
    num_attention_heads: 12  # MQA: 12 Q, 3 KV
    max_position_embeddings: 4096
    parameters: 120M

  # Cross-Modal Fusion
  fusion:
    joint_dim: 512
    num_fusion_blocks: 6
    num_attention_heads: 8
    parameters: 40M

  # JEPA Prediction Head
  jepa:
    hidden_dim: 512
    num_context_layers: 4
    prediction_steps: 3
    parameters: 30M

  # Auto-Regressive Head
  autoregressive:
    hidden_size: 768
    intermediate_size: 3072
    num_layers: 6
    num_attention_heads: 12  # MQA: 12 Q, 3 KV
    parameters: 60M

  # Attention
  attention_type: "multi-query"
  rotary_pct: 1.0  # 100% rotary embeddings
  rotary_base: 10000

  # Normalization & Activation
  hidden_act: "swiglu"
  norm_type: "rmsnorm"
  epsilon: 1e-6
  initializer_range: 0.02

  # Optimization
  use_cache: true  # KV cache for fast generation
  tie_word_embeddings: false
```

### 3.2 Component Architecture

#### Vision Encoder (50M params)

```
Input: RGB Image [H, W, 3]
    ↓
Stem: Conv4x4, stride=4, 96 channels
    ↓
[H/4, W/4, 96]
    ↓
MBConv Blocks (×4):
  - Depthwise separable convolution
  - Expansion ratio: 6
  - SE attention
    ↓
[H/8, W/8, 192]
    ↓
Projection: 192 → 384
    ↓
ViT Blocks (×6):
  - Windowed attention (16×16 windows)
  - Hidden dim: 384
  - Heads: 6
  - MLP: 384 → 1536 → 384
    ↓
[H/8, W/8, 384]
    ↓
Flatten: [N_vision_patches, 384]
```

#### Language Encoder (120M params)

```
Input: Token IDs [seq_len]
    ↓
Token Embedding: [50000, 768]
    ↓
[seq_len, 768]
    ↓
For ℓ = 1 to 12:
    ├─ RoPE positional encoding
    ├─ h₁ = x + MQA(RMSNorm(x))  # 12 Q, 3 KV heads
    ├─ h₂ = h₁ + SwiGLU(Linear(RMSNorm(h₁)))
    └─ x = h₂
    ↓
Output: [seq_len, 768]
```

#### Cross-Modal Fusion (40M params)

```
Inputs:
  - Vision: [N_v, 384]
  - Language: [N_l, 768]
    ↓
Projection to 512-dim:
  - Vision: Linear(384 → 512)
  - Language: Linear(768 → 512)
    ↓
For ℓ = 1 to 6:
    ├─ V→L Cross-Attention:
    │   Q_vision, K_lang, V_lang
    ├─ L→V Cross-Attention:
    │   Q_lang, K_vision, V_vision
    ├─ Concat + Residual
    └─ SwiGLU MLP(512 → 2048 → 512)
    ↓
Output: [N_v + N_l, 512] joint embeddings
```

#### JEPA Prediction Head (30M params)

```
Input: Joint embeddings [N, 512]
    ↓
Temporal Context Encoder (4 layers):
  - Standard transformer blocks
  - Hidden dim: 512
  - Heads: 8
    ↓
Context encoded: [N, 512]
    ↓
Multi-Step Predictor:
  - Predict z_{t+1}: MLP(512 → 512)
  - Predict z_{t+2}: MLP(512 → 512)
  - Predict z_{t+3}: MLP(512 → 512)
    ↓
Output: [N, 3, 512] predicted embeddings
    ↓
Loss: MSE(predicted, target) in embedding space
```

#### Auto-Regressive Head (60M params)

```
Input: Joint embeddings [N, 512]
    ↓
Projection: Linear(512 → 768)
    ↓
[N, 768]
    ↓
AR Transformer (6 layers, causal mask):
  - MQA: 12 Q, 3 KV heads
  - SwiGLU MLP: 768 → 3072 → 768
  - RMSNorm (pre/post)
    ↓
[N, 768]
    ↓
LM Head: Linear(768 → vocab_size=50000)
    ↓
Output: Logits [N, 50000]
```

### 3.3 Hybrid Training Objective

```
L_total = α × L_JEPA + β × L_LM

Where:
L_JEPA = MSE(predicted_embeddings, target_embeddings)
       = Σ||z_pred - z_target||² / N

L_LM = CrossEntropy(predicted_logits, target_tokens)
     = -Σ log P(token_t | token_<t)

Default weights: α = 0.3, β = 0.7
```

---

## 4. Hardware Optimization Strategy

### 4.1 AMD Zen Architecture Considerations

**Zen 2/3/4 Microarchitecture:**
- L1 Cache: 32 KB per core
- L2 Cache: 512 KB per core
- L3 Cache: 32 MB per CCD (Zen 3), 96 MB (Zen 4)
- Memory Bandwidth: ~50 GB/s (DDR4-3200)
- AVX2: 256-bit registers (8x float32)
- FMA3: Fused multiply-add

**Key Optimization Principles:**
1. **Maximize L3 cache hits** - Tile operations to 512×512 blocks
2. **Use AVX2 everywhere** - 4x throughput for float32
3. **Minimize memory traffic** - Fuse operations where possible
4. **Prefetch aggressively** - Software prefetch for next tile
5. **Align to 32 bytes** - AVX2 alignment requirement

### 4.2 AVX2 SIMD Kernels

**Matrix Multiplication:**
- 8×8 micro-kernel using FMA
- 512×512 L3 cache blocking
- 64×64 L2 cache tiles

**Attention Computation:**
- AVX2 QK^T computation
- AVX2 softmax with exp approximation
- Vectorized weighted sum

**Vision Operations:**
- AVX2 patch embedding
- AVX2 2D attention (windowed)
- AVX2 depthwise convolution

### 4.3 Memory Layout

**NHWC for Attention:**
- Better cache locality
- Sequential access patterns
- AVX2-friendly stride

**32-Byte Alignment:**
- All tensors aligned
- Direct AVX2 load/store
- No unaligned access penalty

---

## 5. Tokenization Strategy

### 5.1 Code-Aware Tokenizer

**Vocabulary: 50,000 tokens**

Distribution:
- Byte-level tokens: 256
- Common words: 20,000
- Code keywords: 100
- Operators: 50
- Identifiers: 15,000
- String literals: 10,000
- Numbers: 2,000
- Special tokens: 20 (editing markers, control)
- Whitespace: 574

### 5.2 Special Tokens

| Token | Purpose |
|-------|---------|
| `<FILE>` | File boundary marker |
| `<EDIT>` | Edit region start |
| `</EDIT>` | Edit region end |
| `<DIFF>` | Diff output marker |
| `<INS>` | Insertion |
| `<DEL>` | Deletion |
| `<IMAGE>` | Image input marker |
| `<VISION_START>` | Vision token start |
| `<VISION_END>` | Vision token end |

---

## 6. Performance Targets

### 6.1 Model Comparison

| Metric | Original (1.26B) | VL-JEPA (300M) | Change |
|--------|-----------------|----------------|--------|
| **Parameters** | 1.26B | 300M | 4.2× smaller |
| **Memory** | 5.3 GB | 1.3 GB | 3.8× less |
| **Text-only speed** | 50 tok/s | 45 tok/s | 10% slower |
| **Vision+text speed** | N/A | 28 tok/s | New capability |
| **Startup time** | 100 ms | < 50 ms | 2× faster |

### 6.2 Inference Performance

**Text-only (no vision):**
- Throughput: ~45 tokens/second
- Latency (first token): ~40 ms
- Memory: ~1.3 GB

**Vision + text:**
- Throughput: ~28 tokens/second
- Latency (first token): ~80 ms
- Memory: ~1.3 GB

---

## 7. Implementation Roadmap

### Phase 1: Architecture Design (Week 1)
- [x] Design VL-JEPA architecture
- [ ] Update all design documents
- [ ] Validate parameter counts

### Phase 2: Core Components (Weeks 2-4)
- [ ] Implement vision encoder
- [ ] Implement language encoder
- [ ] Implement cross-modal fusion
- [ ] AVX2 optimizations

### Phase 3: JEPA Components (Weeks 5-6)
- [ ] Implement JEPA prediction head
- [ ] Implement hybrid loss function
- [ ] Implement multi-step prediction

### Phase 4: Auto-Regressive Head (Weeks 7-8)
- [ ] Implement AR transformer
- [ ] Implement sampling strategies
- [ ] KV cache implementation

### Phase 5: Training Pipeline (Weeks 9-10)
- [ ] Hybrid training loop
- [ ] Multimodal data loading
- [ ] Checkpoint conversion

### Phase 6: Integration & Testing (Weeks 11-12)
- [ ] Python bindings
- [ ] CLI tool
- [ ] Benchmarking
- [ ] Documentation

---

## 8. Success Criteria

**Model Capabilities:**
- [ ] Parameter count < 300M
- [ ] Pass@1 > 55% on HumanEval (slightly lower target due to smaller size)
- [ ] Generate syntactically valid code > 95% of time
- [ ] Edit files accurately > 75% of time
- [ ] Understand visual inputs (code highlighting, charts)
- [ ] Handle 4096 token context windows

**Performance:**
- [ ] Text-only: > 40 tok/s on Ryzen 9 5900X
- [ ] Vision+text: > 25 tok/s
- [ ] Memory usage < 2 GB
- [ ] Startup time < 50 ms

**Reliability:**
- [ ] Zero memory leaks
- [ ] Handle malformed input gracefully
- [ ] Safe file operations

---

## 9. Next Steps

1. **Complete design document updates** - All 7 design docs
2. **Update build configuration** - CMakeLists.txt with vision library
3. **Create header files** - Core components and ops
4. **Begin implementation** - Start with vision encoder
5. **Prepare training data** - Multimodal datasets

---

**Document Status**: Updated for VL-JEPA
**Next Document**: 01_MODEL_ARCHITECTURE.md
