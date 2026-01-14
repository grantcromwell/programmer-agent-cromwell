# Cromwell VL-JEPA: Inference & Code Editing
## Multimodal Text Generation, JEPA-Guided Decoding, and File Operations

**Version**: 2.0
**Date**: 2025-01-14

---

## 1. Inference Architecture

```
User Request (Edit file X to do Y, optionally with image)
         ↓
    Context Manager
    (Select relevant files + load image)
         ↓
    ┌─────────────────────┐
    │ Multimodal Encoder   │
    ├─────────────────────┤
    │ Vision Encoder       │  ← Optional (if image provided)
    │ Language Encoder     │
    │ Cross-Modal Fusion   │
    └─────────┬───────────┘
              ↓
    JEPA-Guided Embeddings
    (Improved representations)
         ↓
    Auto-Regressive Head
    (Generate tokens autoregressively)
         ↓
    Sampler
    (Select next token)
         ↓
    Stop Detection
    (Check for completion)
         ↓
    Decoder
    (Convert tokens to text)
         ↓
    Diff Generator
    (Create unified diff)
         ↓
    File Editor
    (Apply changes safely)
         ↓
    Edited File
```

---

## 2. Multimodal Inference

### 2.1 Text-Only Inference

```python
def generate_text_only(
    model,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.95
) -> str:
    """
    Generate text with language encoder only (no vision).

    Throughput: ~45 tokens/second
    """
    # Tokenize prompt
    input_ids = model.tokenizer.encode(prompt)

    # Encode with language encoder
    language_emb = model.language_encoder(input_ids)

    # Generate autoregressively
    generated_ids = model.autoregressive_head.generate(
        context=language_emb,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )

    # Decode
    return model.tokenizer.decode(generated_ids)
```

### 2.2 Vision + Text Inference

```python
def generate_multimodal(
    model,
    image: Image,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.95
) -> str:
    """
    Generate text with vision + language input.

    Throughput: ~28 tokens/second
    First token latency: ~57 ms (vision encode + first step)
    """
    # Process image
    vision_input = model.image_processor.process(image)

    # Encode with vision encoder
    vision_emb = model.vision_encoder(vision_input)  # ~22 ms

    # Tokenize prompt
    input_ids = model.tokenizer.encode(prompt)

    # Encode with language encoder
    language_emb = model.language_encoder(input_ids)  # ~10 ms

    # Cross-modal fusion
    joint_emb = model.cross_modal_fusion(vision_emb, language_emb)  # ~8 ms

    # Generate autoregressively (JEPA-guided)
    generated_ids = model.autoregressive_head.generate(
        context=joint_emb,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        use_jepa_guidance=True
    )

    # Decode
    return model.tokenizer.decode(generated_ids)
```

### 2.3 JEPA-Guided Generation

```python
def generate_with_jepa_guidance(
    model,
    joint_embeddings,
    max_tokens: int,
    temperature: float,
    top_p: float
) -> List[int]:
    """
    Generate tokens with JEPA embedding guidance.

    JEPA embeddings provide better representations,
    improving generation quality.
    """
    generated_tokens = []
    current_emb = joint_embeddings

    for step in range(max_tokens):
        # JEPA predicts next embedding (optional guidance)
        predicted_emb = model.jepa_head.predict(current_emb)

        # Project to language dimension
        lang_emb = model.ar_projection(predicted_emb)

        # Run AR transformer
        logits = model.autoregressive_head(lang_emb)

        # Sample next token
        next_token = sample_token(logits, temperature, top_p)
        generated_tokens.append(next_token)

        # Stop condition
        if next_token == EOS_TOKEN:
            break

        # Update embeddings for next step
        next_emb = model.language_encoder(next_token)
        current_emb = torch.cat([current_emb, next_emb], dim=0)

        # Update KV cache
        model.autoregressive_head.update_kv_cache(next_emb)

    return generated_tokens
```

---

## 3. Sampling Strategies

### 3.1 Temperature Sampling

```cpp
class Sampler {
public:
    enum Strategy {
        GREEDY,       // Always pick argmax
        TEMPERATURE,  // Sample with temperature
        TOP_P,        // Nucleus sampling
        TOP_K,        // Sample from top K
        BEAM_SEARCH,  // Beam search decoding
    };

    Sampler(Strategy strategy = TEMPERATURE, float temperature = 0.8f)
        : strategy_(strategy), temperature_(temperature) {}

    int sample(
        const float* logits,  // [vocab_size]
        int vocab_size,
        std::mt19937& rng
    ) {
        switch (strategy_) {
            case GREEDY:
                return sample_greedy(logits, vocab_size);
            case TEMPERATURE:
                return sample_temperature(logits, vocab_size, temperature_, rng);
            case TOP_P:
                return sample_top_p(logits, vocab_size, 0.9f, rng);
            case TOP_K:
                return sample_top_k(logits, vocab_size, 50, rng);
            case BEAM_SEARCH:
                return sample_beam_search(logits, vocab_size);
            default:
                return sample_greedy(logits, vocab_size);
        }
    }

private:
    int sample_greedy(const float* logits, int vocab_size) {
        return std::max_element(logits, logits + vocab_size) - logits;
    }

    int sample_temperature(const float* logits, int vocab_size,
                          float temp, std::mt19937& rng) {
        // Apply temperature scaling
        std::vector<float> scaled_logits(vocab_size);
        for (int i = 0; i < vocab_size; i++) {
            scaled_logits[i] = logits[i] / temp;
        }

        // Softmax
        std::vector<float> probs = softmax(scaled_logits);

        // Sample from distribution
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        return dist(rng);
    }

    int sample_top_p(const float* logits, int vocab_size,
                     float p, std::mt19937& rng) {
        // Sort by logit value
        std::vector<int> indices(vocab_size);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [logits](int a, int b) { return logits[a] > logits[b]; });

        // Find smallest set with cumulative probability >= p
        std::vector<float> sorted_probs(vocab_size);
        float cumsum = 0.0f;
        int k = 0;
        for (int i = 0; i < vocab_size; i++) {
            sorted_probs[i] = std::exp(logits[indices[i]]);
            cumsum += sorted_probs[i];
            if (cumsum >= p) {
                k = i + 1;
                break;
            }
        }

        // Normalize and sample from top-k
        sorted_probs.resize(k);
        for (int i = 0; i < k; i++) {
            sorted_probs[i] /= cumsum;
        }

        std::discrete_distribution<int> dist(sorted_probs.begin(), sorted_probs.end());
        int sampled_idx = dist(rng);
        return indices[sampled_idx];
    }

    Strategy strategy_;
    float temperature_;
};
```

### 3.2 Beam Search

```cpp
class BeamSearch {
public:
    std::vector<int> search(
        const float* logits,  // [seq_len, vocab_size]
        int seq_len,
        int vocab_size,
        int beam_width = 4,
        int max_tokens = 100
    ) {
        // Initialize beams with top-k tokens from first position
        std::vector<Beam> beams;
        for (int i = 0; i < beam_width; i++) {
            beams.push_back({
                {argmax(&logits[i * vocab_size], vocab_size)},
                logits[i * vocab_size + beams.back().tokens.back()]
            });
        }

        // Generate autoregressively
        for (int step = 1; step < max_tokens; step++) {
            std::vector<Beam> new_beams;

            for (const auto& beam : beams) {
                // Get logits for last token
                // ... (would need model forward pass here)

                // Expand with top-k candidates
                for (int k = 0; k < beam_width; k++) {
                    Beam new_beam = beam;
                    new_beam.tokens.push_back(candidate_token);
                    new_beam.score += candidate_score;
                    new_beams.push_back(new_beam);
                }
            }

            // Keep top-k beams
            std::partial_sort(
                new_beams.begin(),
                new_beams.begin() + beam_width,
                new_beams.end(),
                [](const Beam& a, const Beam& b) { return a.score > b.score; }
            );
            beams = std::vector<Beam>(new_beams.begin(), new_beams.begin() + beam_width);
        }

        // Return best beam
        return std::max_element(beams.begin(), beams.end())->tokens;
    }

private:
    struct Beam {
        std::vector<int> tokens;
        float score;
    };
};
```

---

## 4. KV Cache for Efficient Inference

### 4.1 KV Cache Structure

```cpp
// KV cache for fast autoregressive generation
struct KVCache {
    float* k_cache;     // [max_seq_len, num_kv_heads, head_dim]
    float* v_cache;     // [max_seq_len, num_kv_heads, head_dim]
    int current_len;
    int max_len;
    int num_kv_heads;   // 3 for MQA
    int head_dim;       // 64

    void resize(int max_seq_len) {
        max_len = max_seq_len;
        k_cache = new float[max_len * num_kv_heads * head_dim];
        v_cache = new float[max_len * num_kv_heads * head_dim];
    }

    void append(const float* new_k, const float* new_v) {
        // Append new key/value at current position
        int offset = current_len * num_kv_heads * head_dim;
        std::copy(new_k, new_k + num_kv_heads * head_dim, &k_cache[offset]);
        std::copy(new_v, new_v + num_kv_heads * head_dim, &v_cache[offset]);
        current_len++;
    }

    void reset() {
        current_len = 0;
    }
};
```

### 4.2 Cached Attention

```cpp
// Attention computation with KV cache
void cached_attention(
    const float* query,     // [1, num_q_heads, head_dim] - single token
    const KVCache& cache,   // Contains K, V from previous tokens
    float* output,          // [1, num_q_heads, head_dim]
    int num_q_heads,
    int head_dim
) {
    int seq_len = cache.current_len;

    for (int h = 0; h < num_q_heads; h++) {
        for (int kv_h = 0; kv_h < cache.num_kv_heads; kv_h++) {
            // Compute attention scores for this token vs all previous tokens
            for (int t = 0; t < seq_len; t++) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += query[h * head_dim + d] *
                             cache.k_cache[t * cache.num_kv_heads * cache.head_dim +
                                           kv_h * cache.head_dim + d];
                }

                // Apply softmax (accumulate across all previous tokens)
                // ...
            }

            // Weighted sum of values
            for (int d = 0; d < head_dim; d++) {
                output[h * head_dim + d] = /* weighted sum */;
            }
        }
    }
}
```

---

## 5. Diff Generation

### 5.1 Unified Diff Format

```python
def generate_diff(
    original_code: str,
    edited_code: str,
    file_path: str
) -> str:
    """
    Generate unified diff between original and edited code.

    Uses Myers diff algorithm optimized for code.
    """
    import difflib

    # Split into lines
    original_lines = original_code.splitlines(keepends=True)
    edited_lines = edited_code.splitlines(keepends=True)

    # Generate diff
    diff = difflib.unified_diff(
        original_lines,
        edited_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm=''
    )

    return ''.join(diff)
```

### 5.2 Diff Application

```cpp
// Apply unified diff safely
bool apply_diff(
    const std::string& file_path,
    const std::string& diff_content
) {
    // Parse diff
    auto changes = parse_unified_diff(diff_content);

    // Read original file
    std::string original = read_file(file_path);
    auto lines = split_lines(original);

    // Apply changes in reverse order (to preserve line numbers)
    for (auto it = changes.rbegin(); it != changes.rend(); ++it) {
        if (!apply_change(lines, *it)) {
            return false;
        }
    }

    // Write modified file
    write_file(file_path, join_lines(lines));
    return true;
}

// Safe file writing (atomic operation)
bool write_file_safely(
    const std::string& path,
    const std::string& content
) {
    // Write to temporary file
    std::string temp_path = path + ".tmp";
    std::ofstream out(temp_path);
    out << content;
    out.close();

    // Atomic rename
    return std::rename(temp_path.c_str(), path.c_str()) == 0;
}
```

---

## 6. Performance Targets

### 6.1 Latency Breakdown

**Text-only generation (45 tok/s):**
- Language encode (4096 tokens): ~22 ms
- Per token generation: ~22 ms
  - AR transformer (6 layers): ~12 ms
  - Sampling: ~2 ms
  - KV cache update: ~2 ms
  - Other overhead: ~6 ms

**Vision + text generation (28 tok/s):**
- Vision encode (256×256 image): ~22 ms (one-time)
- Language encode (4096 tokens): ~10 ms
- Cross-modal fusion: ~8 ms
- Per token generation: ~35 ms
  - JEPA prediction: ~3 ms
  - AR transformer (6 layers): ~15 ms
  - Sampling: ~2 ms
  - KV cache update: ~2 ms
  - Other overhead: ~13 ms

### 6.2 Memory Usage

| Component | Memory (batch=1) |
|-----------|------------------|
| Model weights (300M) | 1.2 GB |
| Vision activations | 100 MB |
| Language activations | 150 MB |
| KV cache (4096 context) | 48 MB |
| **Total** | **~1.5 GB** |

---

## 7. Multimodal Examples

### 7.1 Image-to-Code Generation

```python
# Example: Generate plotting code from chart
image = load_image("chart.png")

prompt = """<IMAGE>
Generate matplotlib code to recreate this chart.
The chart shows a line plot with two series."""

result = generate_multimodal(
    model=model,
    image=image,
    prompt=prompt,
    max_tokens=256
)

# Output might be:
# import matplotlib.pyplot as plt
# import numpy as np
#
# x = np.linspace(0, 10, 100)
# y1 = np.sin(x)
# y2 = np.cos(x)
#
# plt.figure(figsize=(10, 6))
# plt.plot(x, y1, label='sin(x)')
# plt.plot(x, y2, label='cos(x)')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.grid(True)
# plt.show()
```

### 7.2 Code Screenshot to Source

```python
# Example: Extract code from screenshot
image = load_image("screenshot.png")

prompt = """<IMAGE>
Extract the Python code from this syntax-highlighted screenshot."""

result = generate_multimodal(
    model=model,
    image=image,
    prompt=prompt,
    max_tokens=512
)
```

---

**Document Status**: Updated for VL-JEPA multimodal inference
**Next Document**: 06_IMPLEMENTATION_ROADMAP.md
