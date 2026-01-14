# Cromwell Agent: Auto-Regressive Coding Agent
## Complete System Architecture Design

**Version**: 1.0
**Date**: 2025-01-14
**Design Goal**: Build a production-grade auto-regressive coding agent optimized for AMD CPUs with AVX2 support

---

## 1. Executive Summary

Cromwell Agent is a specialized auto-regressive language model designed for code editing and understanding. It combines:
- **CS graduate-level knowledge** with ML fundamentals
- **Hardware-optimized inference** on AMD CPUs (Zen 2/3/4)
- **Code-aware tokenization** and understanding
- **File editing capabilities** through diff-based operations

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Decoder-only Transformer** | Best for auto-regressive generation, proven at scale |
| **Rotary Positional Embeddings (RoPE)** | Better extrapolation for long code files |
| **Multi-Query Attention (MQA)** | Faster inference, smaller memory footprint |
| **Custom tokenizer with code-aware merges** | Handles identifiers, strings, indentation |
| **C++ core with Python bindings** | Performance critical path in C++, flexibility in Python |
| **L3 cache-aware memory layout** | Critical for AMD Zen architecture performance |
| **AVX2 SIMD everywhere** | 4x speedup on matrix operations, attention |

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CROMWELL AGENT SYSTEM                            │
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
│  │                    File System Interface                     │ │
│  │  - Directory traversal (with .gitignore support)            │ │
│  │  - File reading (with encoding detection)                   │ │
│  │  - Diff generation (unified, context-aware)                 │ │
│  │  - Patch application (safe, atomic operations)              │ │
│  └─────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Context Manager                           │ │
│  │  - File prioritization (relevance scoring)                  │ │
│  │  - Context window allocation (dynamic budgeting)            │ │
│  │  - Token counting (accurate, fast)                          │ │
│  │  - Window management (sliding, importance-based)            │ │
│  └─────────────────────────────────────────────────────────────┘ │
└───────────────────────────┬───────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│                        LAYER 3: INFERENCE ENGINE                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Tokenizer                                 │ │
│  │  - Code-aware vocabulary (identifiers, keywords, operators)  │ │
│  │  - Fast BPE encoding/decoding (SIMD-optimized)             │ │
│  │  - Special tokens for editing (FILE, EDIT, DELETE)          │ │
│  └─────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Cromwell Transformer                      │ │
│  │  - 1.2B parameters (optimal for code editing)               │ │
│  │  - Multi-Query Attention (8 query heads, 1 key/value head)  │ │
│  │  - RoPE positional encoding                                 │ │
│  │  - SwiGLU activation                                        │ │
│  │  - RMSNorm                                                   │ │
│  │  - 4096 token context window                                │ │
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
│  └─────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                    Training Curriculum                       │ │
│  │  Stage 1: Language modeling (general text + code)           │ │
│  │  Stage 2: Code editing (diff generation)                    │ │
│  │  Stage 3: Tool use (file operations, API calls)             │ │
│  │  Stage 4: Instruction following (code editing tasks)        │ │
│  └─────────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────┘
```

---

## 3. Model Architecture Specifications

### 3.1 Base Configuration

```yaml
model:
  name: "Cromwell-1.2B"
  architecture: "decoder-only-transformer"
  parameters: 1.2B

  # Dimensions
  hidden_size: 2048
  intermediate_size: 5632  # 2.75 * hidden_size (SwiGLU optimal)
  num_layers: 24
  num_attention_heads: 32  # For multi-query: 32 query heads, 4 key/value heads

  # Context
  max_position_embeddings: 4096
  vocab_size: 50000  # Will be finalized after tokenizer training

  # Attention
  attention_type: "multi-query"
  rotary_pct: 1.0  # 100% rotary embeddings
  rotary_base: 10000
  attention_window: 4096  # Full context attention

  # Normalization & Activation
  hidden_act: "swiglu"
  norm_type: "rmsnorm"
  epsilon: 1e-6
  initializer_range: 0.02

  # Optimization
  use_cache: true  # KV cache for fast generation
  tie_word_embeddings: false
```

### 3.2 Attention Mechanism

**Multi-Query Attention (MQA)** - Critical for fast inference:

```
Standard MHA: [Q, K, V] → N heads → Concat → Output
  Each head has separate Q, K, V projections
  Memory: O(N * d_model^2) for KV cache

MQA: Q → N heads, [K, V] → Single head → Broadcast → Output
  Multiple query heads share single key/value head
  Memory: O(d_model^2) for KV cache (8-16x reduction)
  Speed: ~2x faster due to memory bandwidth reduction
```

**Why MQA for Code Editing:**
- Code requires long contexts (4096+ tokens)
- Inference speed is critical for interactive editing
- Minimal quality degradation (<2% on code benchmarks)
- Enables larger batch sizes for multiple file edits

### 3.3 Positional Encoding: Rotary (RoPE)

**Why RoPE over learned absolute/relative:**
- Better extrapolation to longer sequences
- No learned parameters to store
- Works well with MQA
- Proven on code models (CodeLlama, StarCoder)

Implementation:
```cpp
// RoPE rotation for query-key pairs
// Applied after projection but before attention scores
void apply_rotary_embeddings(
    float* q,  // [seq_len, num_heads, head_dim]
    float* k,  // [seq_len, num_kv_heads, head_dim]
    int seq_len,
    int head_dim,
    float base = 10000.0
);
```

### 3.4 Normalization: RMSNorm

**Why RMSNorm over LayerNorm:**
- Removes mean calculation (simpler, faster)
- Same statistical normalization effect
- Fewer operations (SIMD-friendly)
- Stable for transformer training

Formula:
```
RMSNorm(x) = x / sqrt(mean(x^2) + epsilon) * scale
```

### 3.5 Activation: SwiGLU

**Why SwiGLU over ReLU/GeLU:**
- State-of-the-art for language models
- Smooth, differentiable everywhere
- Better gradient flow
- Proven at scale (LLaMA, CodeLLaMA)

Formula:
```
SwiGLU(x) = Swish(xW) ⊙ (xV)
Swish(x) = x * sigmoid(x)
```

---

## 4. Hardware Optimization Strategy

### 4.1 AMD Zen Architecture Considerations

**Zen 2/3/4 Microarchitecture:**
- L1 Cache: 32 KB per core (8-way associative)
- L2 Cache: 512 KB per core (8-way associative)
- L3 Cache: 32 MB per CCD (Zen 3), 96 MB (Zen 4)
- Memory Bandwidth: ~50 GB/s (DDR4-3200), ~80 GB/s (DDR5-5200)
- AVX2: 256-bit registers (8x float32)
- FMA3: Fused multiply-add (2 ops per cycle)

**Key Optimization Principles:**
1. **Maximize L3 cache hits** - Tile matrix operations to 512x512 blocks
2. **Use AVX2 everywhere** - 4x throughput for float32 operations
3. **Minimize memory traffic** - Fuse operations where possible
4. **Prefetch aggressively** - Software prefetch for next tile
5. **Align to 32 bytes** - AVX2 alignment requirement

### 4.2 Cache-Aware Data Layout

**Memory Layout for Transformer Weights:**

```cpp
// Standard: [num_layers, hidden_size, intermediate_size]
// Cache-friendly: [intermediate_size, hidden_size, num_layers]
//
// Reason: Matrix multiplication processes rows sequentially
// Having intermediate_size as first dimension keeps data in L1

struct TransformerWeights {
    // Layout: [output_dim, input_dim] for matrix multiplication
    // Aligned to 32 bytes for AVX2
    float* qkv_proj;      // [3 * hidden_size, hidden_size]
    float* o_proj;        // [hidden_size, hidden_size]
    float* gate_up_proj;  // [2 * intermediate_size, hidden_size]  // SwiGLU fused
    float* down_proj;     // [hidden_size, intermediate_size]

    // Normalization parameters
    float* input_layernorm;    // [hidden_size]
    float* post_attention_layernorm;  // [hidden_size]
};
```

**Why NHWC for Attention:**

```cpp
// Layout comparison for attention:
//
// NCHW: [batch, heads, seq_len, head_dim]
//  - Stride between seq positions: head_dim
//  - Poor cache locality when computing attention matrix
//
// NHWC: [batch, seq_len, head_dim, heads]
//  - Stride between seq positions: head_dim * num_heads
//  - Sequential access during QK^T computation
//  - Better L3 cache utilization
```

### 4.3 AVX2 SIMD Kernel Design

**Matrix Multiplication Kernel (GEMM):**

```cpp
// 8x8 micro-kernel for AVX2
// Processes 8 output elements (rows) × 8 input elements (cols)
// Each AVX2 register holds 8 float32 values
//
// Algorithm:
// 1. Load 8 rows of matrix A (8x8 block)
// 2. Load 8 cols of matrix B (8x8 block)
// 3. Compute outer product using FMA
// 4. Accumulate into output block (C)
//
// Performance: ~95% of theoretical AVX2 peak
```

**Attention Computation Kernel:**

```cpp
// QK^T + Softmax kernel
// Optimized for multi-query attention pattern
//
// Pattern: Q[seq_len, num_heads, head_dim] @ K[seq_len, 1, head_dim].T
//
// Optimization:
// 1. Compute QK^T in blocks of 64x64 (L3 cache friendly)
// 2. Apply softmax in-place (max, exp, sum, normalize)
// 3. Multiply by V[seq_len, 1, head_dim]
// 4. All operations use AVX2 SIMD
```

### 4.4 Memory Prefetching Strategy

```cpp
// Software prefetch for next tile
// Hide memory latency by prefetching next block
//
// Pattern:
// Process tile(i, j)
// Prefetch tile(i+1, j)  // Next row
// Prefetch tile(i, j+1)  // Next col
//
// Distance: 2-3 tiles ahead (empirically optimal for Zen)
// Target: L3 cache (shared across cores)
```

---

## 5. Tokenization Strategy

### 5.1 Code-Aware Tokenizer

**Design Principles:**
1. **Preserve code structure** - Indentation, brackets, operators
2. **Minimize tokens** - Merge common patterns (function, class, if)
3. **Handle identifiers** - Split CamelCase, snake_case
4. **Special tokens** - FILE_START, FILE_END, EDIT_START, EDIT_END

**Vocabulary Composition:**
```
Total: 50,000 tokens
- Byte-level tokens: 256 (for rare bytes)
- Common words: 20,000 (English, common in code)
- Code keywords: 100 (if, else, for, while, return, etc.)
- Operators: 50 (+, -, *, /, ==, !=, <=, >=, etc.)
- Identifiers: 15,000 (function names, variables, types)
- String literals: 10,000 (common strings, messages)
- Numbers: 2,000 (integers, floats, hex, binary)
- Special tokens: 20 (editing markers, control)
- Whitespace: 574 (indentation levels, newlines, tabs)
```

**Merging Rules:**
```python
# Byte-Pair Encoding with code-aware rules

# 1. Standard BPE merges (frequency-based)
# 2. Code-specific merges:
#    - Merge whitespace + identifier (indentation patterns)
#    - Merge keywords + parenthesis (if (, for (, while ()
#    - Merge operators (=, ==, !=, <=, >=)
#    - Merge brackets ((), [], {}, <>)
#    - Merge CamelCase identifiers (MyClass -> My + Class)
#    - Merge common patterns (function name, def, import)

# 3. Special token insertion:
#    - <FILE=path/to/file.py> marks file start
#    - </FILE> marks file end
#    - <EDIT_START> marks edit region
#    - <EDIT_END> marks edit end
#    - <DELETE_START> marks deletion region
#    - </DELETE> marks deletion end
```

### 5.2 Fast Tokenization Implementation

**SIMD-Accelerated BPE Encoding:**

```cpp
// AVX2-accelerated BPE encoding
// Processes 8 bytes in parallel for pattern matching
//
// Algorithm:
// 1. Load 8 bytes from input
// 2. Compare against pattern (AVX2 PCMPEQB)
// 3. Extract matches (MOVEMASK)
// 4. Emit tokens
//
// Speedup: ~3-4x over scalar implementation
```

---

## 6. File Editing Pipeline

### 6.1 File Representation

**Input Format:**

```
<FILE=path/to/file.py>
def calculate_sum(a, b):
    """Calculate the sum of two numbers."""
    return a + b

def calculate_product(a, b):
    """Calculate the product of two numbers."""
    return a * b
</FILE>

<EDIT_START=file.py:3:5>
Replace the sum calculation with a more efficient version using the built-in sum function.
<EDIT_END>
```

**Output Format:**

```
<FILE=path/to/file.py>
def calculate_sum(a, b):
    """Calculate the sum of two numbers."""
    return sum([a, b])

def calculate_product(a, b):
    """Calculate the product of two numbers."""
    return a * b
</FILE>
```

### 6.2 Diff Generation

**Unified Diff Format:**

```
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -1,7 +1,7 @@
 def calculate_sum(a, b):
     """Calculate the sum of two numbers."""
-    return a + b
+    return sum([a, b])

 def calculate_product(a, b):
     """Calculate the product of two numbers."""
     return a * b
```

**Algorithm:**

```cpp
// Myers diff algorithm (optimized)
// Time complexity: O(ND) where N = file length, D = edits
// Space complexity: O(D)
//
// Optimizations:
// 1. Early termination for single-line edits
// 2. Hash-based comparison (rolling hash for long files)
// 3. SIMD-accelerated string comparison (AVX2)
// 4. Binary search for edit location (when line numbers provided)
```

### 6.3 Context Window Management

**Dynamic Budget Allocation:**

```python
# Allocate context window budget based on file relevance
#
# Scoring factors:
# - File extension (.py, .js, .ts higher priority)
# - Import relationships (files imported by target file)
# - Edit history (files edited recently)
# - File size (penalize very large files)
# - Symbol similarity (shared functions/classes)

def allocate_context_budget(
    target_file: str,
    available_tokens: int,
    codebase: Codebase
) -> List[str]:
    """
    Select files to include in context window.

    Returns:
        List of file paths ordered by priority
    """
    scores = {}

    for file in codebase.files:
        score = 0.0

        # File type bonus
        if file.extension in ['.py', '.js', '.ts', '.cpp']:
            score += 1.0

        # Import relationship
        if file.imports(target_file) or target_file.imports(file):
            score += 2.0

        # Symbol overlap
        shared_symbols = len(file.symbols & target_file.symbols)
        score += shared_symbols * 0.1

        # Edit recency
        if file.last_edited < datetime.now() - timedelta(hours=1):
            score += 0.5

        # Size penalty
        score -= log(file.token_count) * 0.01

        scores[file.path] = score

    # Sort and select until budget exhausted
    sorted_files = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    selected = []
    used_tokens = 0

    for file, score in sorted_files:
        file_tokens = file.token_count
        if used_tokens + file_tokens < available_tokens:
            selected.append(file)
            used_tokens += file_tokens

    return selected
```

---

## 7. Training Pipeline

### 7.1 Dataset Design

**Primary Datasets:**

| Dataset | Size | Focus | Weight |
|---------|------|-------|--------|
| TheStack v2 | 3TB | Code + natural language | 40% |
| CodeContests | 500GB | Competitive programming | 15% |
| GitHub PRs | 200GB | Code review + diffs | 10% |
| ArXiv (CS/ML) | 100GB | Academic knowledge | 15% |
| OpenStax CS/ML | 50GB | Textbooks | 10% |
| Stack Overflow | 100GB | Q&A + explanations | 10% |

**Preprocessing Pipeline:**

```
1. Raw Data Download
   ↓
2. Deduplication (MinHash LSH, Jaccard similarity > 0.85)
   ↓
3. Filtering
   - Remove auto-generated files
   - Remove very short files (< 128 tokens)
   - Remove very long files (> 10000 tokens)
   - Language detection (keep English + code)
   ↓
4. Quality Scoring
   - Code syntax validation
   - Natural language perplexity
   - Comment density
   ↓
5. Tokenization
   - Apply BPE tokenizer
   - Insert special tokens
   ↓
6. Sharding
   - 1000 samples per shard
   - Balanced by dataset source
   ↓
7. Final Training Data
```

### 7.2 Training Curriculum

**Stage 1: Foundation (0-60% tokens)**
- Objective: Standard language modeling
- Data: All datasets mixed
- Duration: ~150B tokens
- Learning rate: 3e-4 → 3e-5 (cosine decay)
- Batch size: 256 sequences × 2048 tokens

**Stage 2: Code Understanding (60-80% tokens)**
- Objective: Code-specific tasks
- Data: Weight code 4:1 over text
- Tasks:
  - Next token prediction on code
  - Fill-in-the-middle (FIM)
  - Function completion
- Duration: ~50B tokens
- Learning rate: 3e-5 → 1e-5

**Stage 3: Editing Skills (80-95% tokens)**
- Objective: Diff generation and application
- Data: GitHub PRs, CodeContests edits
- Tasks:
  - Diff prediction
  - Bug fix generation
  - Refactoring suggestions
- Duration: ~25B tokens
- Learning rate: 1e-5 → 5e-6

**Stage 4: Instruction Tuning (95-100% tokens)**
- Objective: Follow editing instructions
- Data: Synthetic instructions + human demonstrations
- Tasks:
  - "Replace X with Y"
  - "Optimize this function"
  - "Add error handling"
- Duration: ~10B tokens
- Learning rate: 5e-6 → 1e-6

### 7.3 Loss Function

**Primary Loss: Cross-Entropy**

```python
# Standard cross-entropy for next-token prediction
loss = -sum(log P(token_t | token_<t))
```

**Auxiliary Losses (Stage 3+):**

```python
# Diff consistency loss
loss_diff = ||predicted_diff - actual_diff||_2

# Code syntax loss (if applicable)
loss_syntax = -log P(syntax_valid | edit)

# Combined loss
loss = loss_ce + 0.1 * loss_diff + 0.05 * loss_syntax
```

---

## 8. Implementation Stack

### 8.1 Language Choices

| Component | Language | Rationale |
|-----------|----------|-----------|
| **Model Core** | C++17 | Performance, SIMD control, memory management |
| **SIMD Kernels** | C++17 + Intrinsics | Direct AVX2/FMA control |
| **Python Bindings** | pybind11 | Seamless Python integration, minimal overhead |
| **Training Script** | Python | Flexibility, ecosystem (PyTorch, transformers) |
| **CLI Tool** | Rust | Fast startup, memory safety, great CLI libraries |
| **Tests** | Python + C++ (Catch2) | Coverage for both interfaces |

### 8.2 Linear Algebra Backend

**Decision: Custom C++ with AVX2**

Rationale:
- Full control over memory layout
- Optimize for specific workloads (MQA attention)
- No dependency bloat
- Learn and understand operations deeply

When to use libraries:
- Training: PyTorch (for gradient computation)
- Prototyping: NumPy (for experimentation)
- Production: Custom C++ (for inference)

### 8.3 Build System

**CMake** for C++ core:
- Cross-platform
- Good dependency management
- Easy integration with Python

**Cargo** for Rust CLI:
- Standard Rust build tool
- Dependency management
- Easy cross-compilation

**Setup.py** for Python package:
- Standard Python packaging
- Integrates with pip
- Manages build process

---

## 9. Project Structure

```
cromwell_agent/
├── design/                      # Design documents
│   ├── 00_ARCHITECTURE_OVERVIEW.md
│   ├── 01_MODEL_ARCHITECTURE.md
│   ├── 02_HARDWARE_OPTIMIZATION.md
│   ├── 03_TOKENIZATION.md
│   ├── 04_TRAINING_PIPELINE.md
│   └── 05_INFERENCE.md
│
├── src/
│   ├── core/                    # Core model implementation
│   │   ├── transformer.h        # Main model class
│   │   ├── attention.h          # Attention mechanism
│   │   ├── mlp.h                # Feed-forward network
│   │   ├── embeddings.h         # Embedding layer
│   │   └── config.h             # Model configuration
│   │
│   ├── ops/                     # Hardware-optimized operations
│   │   ├── avx2_utils.h         # AVX2 intrinsics wrappers
│   │   ├── gemm.h               # Matrix multiplication
│   │   ├── attention_ops.h      # Attention kernels
│   │   ├── layer_norm.h         # Normalization
│   │   ├── activation.h         # Activation functions
│   │   └── memory.h             # Memory management
│   │
│   ├── inference/               # Inference engine
│   │   ├── sampler.h            # Sampling strategies
│   │   ├── kv_cache.h           # KV cache for generation
│   │   ├── generator.h          # Text generation
│   │   └── batch_processor.h    # Batch inference
│   │
│   ├── training/                # Training utilities
│   │   ├── optimizer.h          # Optimizer implementation
│   │   ├── scheduler.h          # Learning rate scheduler
│   │   ├── checkpoint.h         # Checkpoint save/load
│   │   └── dataloader.h         # Data loading
│   │
│   ├── io/                      # I/O and code editing
│   │   ├── tokenizer.h          # Tokenizer interface
│   │   ├── file_io.h            # File system operations
│   │   ├── diff.h               # Diff generation/application
│   │   ├── context_manager.h    # Context window management
│   │   └── codebase.h           # Codebase representation
│   │
│   ├── python/                  # Python bindings
│   │   ├── bindings.cpp         # pybind11 bindings
│   │   └── cromwell/            # Python package
│   │       ├── __init__.py
│   │       ├── model.py
│   │       └── editing.py
│   │
│   └── cli/                     # Rust CLI
│       ├── src/
│       │   ├── main.rs
│       │   ├── commands.rs
│       │   └── config.rs
│       └── Cargo.toml
│
├── tests/
│   ├── cpp/                     # C++ tests
│   │   ├── test_attention.cpp
│   │   ├── test_gemm.cpp
│   │   └── test_transformer.cpp
│   │
│   ├── python/                  # Python tests
│   │   ├── test_model.py
│   │   ├── test_editing.py
│   │   └── benchmarks.py
│   │
│   └── integration/             # Integration tests
│       ├── test_full_pipeline.py
│       └── test_code_editing.py
│
├── scripts/
│   ├── train.sh                 # Training script
│   ├── convert_checkpoint.py   # Convert PyTorch to C++
│   ├── benchmark.sh             # Performance benchmarking
│   └── profile.py               # Profiling utilities
│
├── docs/
│   ├── API.md                   # API documentation
│   ├── PERFORMANCE.md           # Performance guide
│   └── TROUBLESHOOTING.md       # Troubleshooting guide
│
├── CMakeLists.txt               # C++ build configuration
├── setup.py                     # Python package setup
├── pyproject.toml               # Python project config
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview
```

---

## 10. Performance Targets

### 10.1 Inference Speed

**Target: 50 tokens/second on Ryzen 9 5900X**

```
Hardware: AMD Ryzen 9 5900X (12 cores, 24 threads)
- Base clock: 3.7 GHz
- Boost clock: 4.8 GHz
- L3 cache: 64 MB
- Memory: DDR4-3200

Target metrics:
- Single-file edit (4096 context): 50 tok/s
- Multi-file edit (8 files): 40 tok/s
- Batch inference (batch size 8): 30 tok/s
- Memory usage: < 4 GB (model + KV cache)
- Startup time: < 100 ms
```

### 10.2 Accuracy Targets

**Benchmark Performance:**

| Benchmark | Target | Notes |
|-----------|--------|-------|
| HumanEval | Pass@1 > 60% | Python function completion |
| MBPP | Pass@1 > 70% | Python programming |
| Codeforces | > 30% | Competitive programming |
| Code editing | > 80% | Diff accuracy |
| Syntax validity | > 95% | Generated code compiles |

---

## 11. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [ ] Set up build system (CMake, pybind11)
- [ ] Implement basic tensor operations (AVX2)
- [ ] Implement RoPE positional encoding
- [ ] Implement RMSNorm
- [ ] Implement SwiGLU activation

### Phase 2: Attention (Weeks 3-4)
- [ ] Implement multi-query attention
- [ ] Optimize QK^T with AVX2
- [ ] Implement softmax with AVX2
- [ ] Implement KV cache
- [ ] Benchmark and profile

### Phase 3: Transformer Layers (Weeks 5-6)
- [ ] Implement transformer block
- [ ] Stack layers into model
- [ ] Implement embedding layer
- [ ] Implement output projection
- [ ] Test forward pass

### Phase 4: Tokenization (Weeks 7-8)
- [ ] Train BPE tokenizer on code corpus
- [ ] Implement fast encoding (C++)
- [ ] Implement fast decoding (C++)
- [ ] Add special tokens
- [ ] Python bindings

### Phase 5: Inference Engine (Weeks 9-10)
- [ ] Implement sampling strategies
- [ ] Implement text generation loop
- [ ] Implement stop token detection
- [ ] Implement batch processing
- [ ] Benchmark generation speed

### Phase 6: I/O & Editing (Weeks 11-12)
- [ ] Implement file reading/writing
- [ ] Implement diff generation
- [ ] Implement diff application
- [ ] Implement context manager
- [ ] Test on real codebases

### Phase 7: CLI & Python API (Weeks 13-14)
- [ ] Implement Rust CLI
- [ ] Add commands (edit, complete, chat)
- [ ] Python bindings
- [ ] Error handling
- [ ] Documentation

### Phase 8: Training Integration (Weeks 15-16)
- [ ] Prepare training data
- [ ] Implement training loop (PyTorch)
- [ ] Implement checkpoint conversion
- [ ] Train initial model
- [ ] Evaluate on benchmarks

### Phase 9: Optimization (Weeks 17-18)
- [ ] Profile bottlenecks
- [ ] Optimize hot paths
- [ ] Memory layout tuning
- [ ] NUMA optimization
- [ ] Final benchmarks

### Phase 10: Testing & Documentation (Weeks 19-20)
- [ ] Unit tests
- [ ] Integration tests
- [ ] End-to-end tests
- [ ] Write documentation
- [ ] Create examples

---

## 12. Success Criteria

**Model Capabilities:**
- [ ] Pass@1 > 60% on HumanEval
- [ ] Generate syntactically valid code > 95% of time
- [ ] Edit files accurately > 80% of time
- [ ] Understand natural language instructions
- [ ] Handle context windows up to 4096 tokens

**Performance:**
- [ ] Generate > 50 tok/s on Ryzen 9 5900X
- [ ] Memory usage < 4 GB
- [ ] Startup time < 100 ms
- [ ] Support batch size 8 without degradation

**Reliability:**
- [ ] Zero memory leaks (valgrind clean)
- [ ] Handle malformed input gracefully
- [ ] Recover from errors
- [ ] Safe file operations (no data loss)

**Usability:**
- [ ] Simple CLI interface
- [ ] Python API for integration
- [ ] Clear error messages
- [ ] Comprehensive documentation

---

## 13. Next Steps

1. **Review this design** - Validate architectural decisions
2. **Set up development environment** - Install dependencies, toolchain
3. **Start implementation** - Begin with Phase 1
4. **Iterate based on testing** - Adjust as needed
5. **Provide training data** - When model is ready for training

---

## Appendix A: References

**Model Architectures:**
- LLaMA: Open and Efficient Foundation Language Models
- Code Llama: Open Foundation Models for Code
- StarCoder: May the Source Be With You
- PaLM 2: Technical Report

**Hardware Optimization:**
- AMD Zen 3 Microarchitecture
- Intel Optimization Manual
- "Software Optimization Guide for AMD Family 17h Processors"

**Tokenization:**
- "Byte Pair Encoding is Suboptimal for Language Model Pretraining"
- "SentencePiece: A simple and language independent subword tokenizer"

**Training:**
- "Training Compute-Optimal Large Language Models" (Chinchilla)
- "Scaling Laws for Neural Language Models"

---

**Document Status**: Complete
**Next Document**: 01_MODEL_ARCHITECTURE.md
