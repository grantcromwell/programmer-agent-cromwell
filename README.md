# Cromwell: An Auto-Regressive Coding Agent for AMD x86-64 Architecture

## Abstract

Cromwell is a 1.26B parameter decoder-only transformer language model specifically designed for code editing and understanding tasks. This work presents a hardware-conscious implementation optimized for AMD Zen microarchitecture families, featuring AVX2 SIMD acceleration, L3 cache-aware blocked matrix multiplication, and multi-query attention for efficient inference. The model achieves approximately 50 tokens/second on a Ryzen 9 5900X while maintaining a memory footprint of approximately 5.5 GB during inference.

---

## 1. Introduction

Contemporary code editing tasks require language models capable of understanding programming semantics, generating syntactically correct code, and applying precise modifications to existing codebases. Existing solutions often rely on GPU acceleration or suboptimal CPU utilization, limiting deployment in resource-constrained environments.

Cromwell addresses these challenges through:

1. **CPU-Native Architecture**: Purpose-built for x86-64 CPUs with AVX2 instruction sets
2. **Cache-Conscious Design**: L3-aware blocking strategies for matrix operations
3. **Multi-Query Attention**: Reduced KV cache memory footprint for longer context windows
4. **Code-Aware Tokenization**: Byte-Pair Encoding optimized for programming language syntax

---

## 2. Model Architecture

### 2.1 Transformer Configuration

| Component | Specification |
|-----------|---------------|
| Architecture | Decoder-only Transformer |
| Parameters | 1.26B |
| Layers | 24 |
| Hidden Dimension | 2048 |
| FFN Dimension | 5632 (2.75× d_model) |
| Attention Heads | 32 query, 4 key-value (Multi-Query) |
| Head Dimension | 64 |
| Context Window | 4096 tokens |
| Vocabulary Size | 50,000 |
| Positional Encoding | Rotary Positional Embeddings (RoPE) |

### 2.2 Architectural Components

```
Input: x ∈ ℤ^{T}
    ↓
Token Embedding: E ∈ ℝ^{V×d_model}
    ↓
x = E[token] + P  (where P = RoPE positional encoding)
    ↓
For ℓ = 1 to L (L=24):
    ├─ h₁ = x + MQA(RMSNorm(x))
    ├─ h₂ = h₁ + SwiGLU(Linear₁(RMSNorm(h₁)))
    └─ x = h₂
    ↓
Output: RMSNorm(x) @ Eᵀ
    ↓
Logits: ℝ^{T×V}
```

**Multi-Query Attention (MQA):** Single key-value projection shared across all attention heads reduces KV cache memory by a factor of 8 compared to standard multi-head attention.

**SwiGLU Activation:**
```
SwiGLU(x) = (xW₁ ⊙ σ(xW₂))W₃
```
where ⊙ denotes element-wise multiplication and σ is the sigmoid function.

**RMSNorm:**
```
RMSNorm(x) = x / (√(1/n Σᵢ xᵢ² + ε)) · γ
```

---

## 3. Hardware Optimization

### 3.1 Target Architecture

| Specification | Value |
|---------------|-------|
| Target CPU | AMD Ryzen 9 5900X (Zen 3) |
| ISA Extensions | AVX2, FMA, POPCNT, BMI/BMI2 |
| L1 Cache | 32 KB (per core) |
| L2 Cache | 512 KB (per core) |
| L3 Cache | 64 MB (shared) |
| Memory Bandwidth | ~50 GB/s |

### 3.2 AVX2 SIMD Kernels

Core operations implement 256-bit SIMD operations processing 8 single-precision floating-point values per cycle:

- **GEMM Micro-kernel**: 8×8 block processing using FMA instructions
- **Attention**: QK^T computation with vectorized matrix multiplication
- **Softmax**: Numerically stable implementation with AVX2 exponential
- **Layer Normalization**: Vectorized variance computation and scaling

### 3.3 Cache Blocking Strategy

Matrix multiplication employs L3-aware blocking:
- **Block size**: 512×512 elements (2 MB per block)
- **Tile size**: 64×64 elements for L2 optimization
- **Micro-kernel**: 8×8 for register-level optimization

Software prefetching targets:
```cpp
__builtin_prefetch(&A[i+8][k], 0, 3);  // L3 prefetch
__builtin_prefetch(&B[k][j+8], 0, 3);
```

---

## 4. Tokenization

### 4.1 Code-Aware BPE

Vocabulary of 50,000 tokens trained on:
- Python: 40%
- JavaScript/TypeScript: 20%
- C/C++: 15%
- Rust: 10%
- Go: 5%
- Natural language (English): 10%

### 4.2 Special Tokens

| Token | Purpose |
|-------|---------|
| `<FILE>` | File boundary marker |
| `<EDIT>` | Edit region start |
| `</EDIT>` | Edit region end |
| `<DIFF>` | Diff output marker |
| `<INS>` | Insertion |
| `<DEL>` | Deletion |

---

## 5. Performance Characteristics

### 5.1 Inference Performance

| Metric | Value |
|--------|-------|
| Throughput | ~50 tokens/second |
| Latency (first token) | ~80 ms |
| Memory (model + KV cache) | ~5.5 GB @ 4096 context |
| Startup time | < 100 ms |

*Benchmarks conducted on Ryzen 9 5900X, DDR4-3600, single-threaded inference.*

### 5.2 FLOPs Analysis

| Operation | FLOPs per Token | Percentage |
|-----------|-----------------|------------|
| Attention | ~4.2B | 35% |
| MLP (SwiGLU) | ~7.1B | 59% |
| Norm/Embeddings | ~0.7B | 6% |
| **Total** | **~12.0B** | **100%** |

---

## 6. Installation

### 6.1 System Requirements

- **CPU**: x86-64 with AVX2 support (AMD Zen 2/3/4, Intel Skylake+)
- **Compiler**: GCC 9+ or Clang 10+
- **Python**: 3.8+
- **CMake**: 3.18+

### 6.2 Build Procedure

```bash
# Clone repository
git clone https://github.com/grantcromwell/research-agent-cromwell
cd cromwell_agent

# Configure build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release

# Compile
make -j$(nproc)

# Install Python bindings
pip install -e ..
```

### 6.3 CPU Verification

Verify AVX2 support:
```bash
grep avx2 /proc/cpuinfo | head -n 1
```

Expected output should include `avx2` flag.

---

## 7. Usage

### 7.1 Python API

```python
from cromwell import CodeEditor, SamplingConfig

# Initialize model with sampling configuration
config = SamplingConfig(
    temperature=0.2,
    top_p=0.95,
    max_tokens=2048
)

editor = CodeEditor(
    model_path="checkpoints/cromwell-1.2b.bin",
    tokenizer_path="tokenizer/cromwell-50k.json",
    codebase_path="path/to/project",
    config=config
)

# Execute code edit with natural language instruction
result = editor.edit(
    file_path="src/main.py",
    instruction="Add error handling to the parse_config function"
)

# Apply validated diff
if result.success:
    result.apply()
    print(f"Edit applied: {result.stats}")
```

### 7.2 Command-Line Interface

```bash
# Edit file with instruction
cromwell edit src/main.py \
    --instruction "Add type hints to all functions" \
    --model checkpoints/cromwell-1.2b.bin

# Interactive completion
cromwell complete src/main.py \
    --line 42 \
    --context 2048 \
    --max-tokens 256
```

---
research
## 8. Training

Cromwell employs a four-stage training curriculum:

| Stage | Objective | Tokens | Duration |
|-------|-----------|--------|----------|
| 1 | Pre-training (general code) | 150B | Base capability |
| 2 | Pre-training (specific languages) | 50B | Language specialization |
| 3 | Fill-in-Middle (FIM) | 25B | Editing capability |
| 4 | Instruction fine-tuning | 10B | Task alignment |

**Total Training: ~235B tokens**

See [`design/04_TRAINING_PIPELINE.md`](design/04_TRAINING_PIPELINE.md) for complete methodology.

---

## 9. Project Structure

```
cromwell_agent/
├── design/                      # Technical specifications
│   ├── 00_ARCHITECTURE_OVERVIEW.md
│   ├── 01_MODEL_ARCHITECTURE.md
│   ├── 02_HARDWARE_OPTIMIZATION.md
│   ├── 03_TOKENIZATION.md
│   ├── 04_TRAINING_PIPELINE.md
│   ├── 05_INFERENCE.md
│   └── 06_IMPLEMENTATION_ROADMAP.md
│
├── src/
│   ├── ops/                     # Hardware-optimized kernels
│   │   ├── avx2_utils.h         # AVX2 intrinsics
│   │   ├── gemm.h               # Matrix multiplication
│   │   ├── attention_ops.h      # Attention kernels
│   │   └── layer_norm.h         # Normalization
│   │
│   ├── core/                    # Model components
│   ├── inference/               # Generation engine
│   ├── training/                # Training utilities
│   ├── io/                      # File I/O and diff application
│   └── python/                  # Python bindings (pybind11)
│
├── CMakeLists.txt               # Build configuration
└── README.md                    # This document
```

---

## 10. Benchmarking

Reproduce performance benchmarks:

```bash
# Matrix multiplication benchmark
./build/benchmark_gemm --m=2048 --n=2048 --k=2048 --iterations=100

# Attention benchmark
./build/benchmark_attention --batch=1 --seq=4096 --heads=32 --iterations=50

# End-to-end inference benchmark
./build/benchmark_inference --model=checkpoints/cromwell-1.2b.bin --prompt-len=1024
```

---

## 11. References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
2. Shazeer, N. (2019). "Fast Transformer Decoding: One Write-Head is All You Need." *arXiv*. [https://arxiv.org/abs/1911.02150](https://arxiv.org/abs/1911.02150)
3. Su, J., et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position Embedding." *arXiv*. [https://arxiv.org/abs/2104.09864](https://arxiv.org/abs/2104.09864)
4. Touvron, H., et al. (2023). "LLaMA 2: Open Foundation and Fine-Tuned Chat Models." *arXiv*. [https://arxiv.org/abs/2307.09288](https://arxiv.org/abs/2307.09288)
5. AMD Corporation. "AMD Zen 3 Microarchitecture Design Guide." [https://www.amd.com/en/support/tech-docs](https://www.amd.com/en/support/tech-docs)

---

## 12. License

MIT License - see [LICENSE](https://mycromwell.org/license) for details.

---

## 13. Citation

```bibtex
@software{cromwell_agent,
  title={Cromwell: An Auto-Regressive Coding Agent for AMD x86-64 Architecture},
  author={{MyCromwell.org}},
  year={2025},
  url={https://github.com/grantcromwell/programmer-agent-cromwell},
  version={1.0.0},
  note={Property of [MyCromwell.org](https://mycromwell.org)}
}
```
