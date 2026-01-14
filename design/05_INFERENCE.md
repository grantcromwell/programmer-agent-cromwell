# Cromwell Agent: Inference & Code Editing
## Text Generation, Sampling, and File Operations

**Version**: 1.0
**Date**: 2025-01-14

---

## 1. Inference Architecture

```
User Request (Edit file X to do Y)
         ↓
    Context Manager
    (Select relevant files)
         ↓
    Tokenizer
    (Encode to tokens)
         ↓
    Cromwell Model
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

## 2. Sampling Strategies

### 2.1 Temperature Sampling

```cpp
class Sampler {
public:
    enum Strategy {
        GREEDY,       // Always pick argmax
        TEMPERATURE,  // Sample with temperature
        TOP_P,        // Nucleus sampling
        TOP_K,        // Sample from top K
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
            default:
                return sample_greedy(logits, vocab_size);
        }
    }

private:
    int sample_greedy(const float* logits, int vocab_size) {
        // Return argmax
        int max_idx = 0;
        float max_val = logits[0];

        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }

        return max_idx;
    }

    int sample_temperature(
        const float* logits,
        int vocab_size,
        float temperature,
        std::mt19937& rng
    ) {
        // Apply temperature
        std::vector<float> scaled_logits(vocab_size);

        for (int i = 0; i < vocab_size; i++) {
            scaled_logits[i] = logits[i] / temperature;
        }

        // Compute softmax
        std::vector<float> probs = softmax(scaled_logits);

        // Sample from categorical distribution
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        return dist(rng);
    }

    int sample_top_p(
        const float* logits,
        int vocab_size,
        float p,
        std::mt19937& rng
    ) {
        // Sort logits in descending order
        std::vector<std::pair<float, int>> sorted(vocab_size);

        for (int i = 0; i < vocab_size; i++) {
            sorted[i] = {logits[i], i};
        }

        std::sort(sorted.begin(), sorted.end(), std::greater<std::pair<float, int>>());

        // Compute cumulative probabilities
        std::vector<float> probs = softmax_logits(sorted);

        float cumsum = 0.0f;
        int cutoff = vocab_size;

        for (int i = 0; i < vocab_size; i++) {
            cumsum += probs[i];

            if (cumsum >= p) {
                cutoff = i + 1;
                break;
            }
        }

        // Sample from top-p tokens
        std::vector<float> top_p_probs(probs.begin(), probs.begin() + cutoff);
        std::discrete_distribution<int> dist(top_p_probs.begin(), top_p_probs.end());

        int idx = dist(rng);
        return sorted[idx].second;
    }

    std::vector<float> softmax_logits(const std::vector<std::pair<float, int>>& sorted) {
        // Find max for numerical stability
        float max_logit = sorted[0].first;

        // Compute exp and sum
        std::vector<float> exp_logits(sorted.size());
        float sum_exp = 0.0f;

        for (size_t i = 0; i < sorted.size(); i++) {
            exp_logits[i] = exp(sorted[i].first - max_logit);
            sum_exp += exp_logits[i];
        }

        // Normalize
        for (size_t i = 0; i < sorted.size(); i++) {
            exp_logits[i] /= sum_exp;
        }

        return exp_logits;
    }

    std::vector<float> softmax(const std::vector<float>& logits) {
        // Find max for numerical stability
        float max_logit = *std::max_element(logits.begin(), logits.end());

        // Compute exp and sum
        std::vector<float> exp_logits(logits.size());
        float sum_exp = 0.0f;

        for (size_t i = 0; i < logits.size(); i++) {
            exp_logits[i] = exp(logits[i] - max_logit);
            sum_exp += exp_logits[i];
        }

        // Normalize
        for (size_t i = 0; i < logits.size(); i++) {
            exp_logits[i] /= sum_exp;
        }

        return exp_logits;
    }

    Strategy strategy_;
    float temperature_;
};
```

### 2.2 Repetition Penalty

```cpp
class RepetitionPenalty {
public:
    RepetitionPenalty(float penalty = 1.0f) : penalty_(penalty) {}

    void apply(
        float* logits,  // [vocab_size]
        int vocab_size,
        const std::vector<int>& generated_tokens
    ) {
        if (penalty_ == 1.0f) {
            return;
        }

        // Count token frequencies
        std::unordered_map<int, int> freq;
        for (int token : generated_tokens) {
            freq[token]++;
        }

        // Apply penalty
        for (const auto& [token, count] : freq) {
            if (token < vocab_size) {
                if (logits[token] > 0) {
                    logits[token] /= penalty_;
                } else {
                    logits[token] *= penalty_;
                }
            }
        }
    }

private:
    float penalty_;
};
```

---

## 3. Text Generation

### 3.1 Generation Loop

```cpp
class TextGenerator {
public:
    TextGenerator(
        CromwellModel* model,
        Tokenizer* tokenizer,
        const Sampler& sampler
    ) : model_(model), tokenizer_(tokenizer), sampler_(sampler) {}

    struct GenerationConfig {
        int max_tokens = 1000;
        float temperature = 0.8f;
        float top_p = 0.9f;
        float repetition_penalty = 1.0f;
        std::vector<int> stop_tokens;
        bool echo_prompt = false;
    };

    std::string generate(
        const std::string& prompt,
        const GenerationConfig& config
    ) {
        // Encode prompt
        std::vector<int> input_ids = tokenizer_->encode(prompt);

        // Initialize KV cache
        model_->clear_kv_cache();

        // Process prompt
        std::vector<int> output_ids;

        if (config.echo_prompt) {
            output_ids = input_ids;
        } else {
            // Only process prompt through model (no generation yet)
            std::vector<float> logits;
            model_->forward({input_ids.data()}, {static_cast<int>(input_ids.size())}, 1, logits);
        }

        // Generate tokens autoregressively
        std::mt19937 rng(std::random_device{}());
        RepetitionPenalty rep_penalty(config.repetition_penalty);

        for (int step = 0; step < config.max_tokens; step++) {
            // Get logits for last token
            std::vector<float> logits;
            model_->forward(
                {&output_ids.back()},
                {1},
                1,
                logits
            );

            // Apply repetition penalty
            rep_penalty.apply(logits.data(), logits.size(), output_ids);

            // Sample next token
            int next_token = sampler_.sample(logits.data(), logits.size(), rng);
            output_ids.push_back(next_token);

            // Check for stop tokens
            if (std::find(config.stop_tokens.begin(), config.stop_tokens.end(), next_token)
                != config.stop_tokens.end()) {
                break;
            }
        }

        // Decode output
        std::string output = tokenizer_->decode(output_ids);

        return output;
    }

private:
    CromwellModel* model_;
    Tokenizer* tokenizer_;
    Sampler sampler_;
};
```

### 3.2 Streaming Generation

```cpp
class StreamingGenerator {
public:
    using TokenCallback = std::function<void(int token, const std::string& text)>;

    std::string generate_streaming(
        const std::string& prompt,
        const GenerationConfig& config,
        TokenCallback callback
    ) {
        std::vector<int> input_ids = tokenizer_->encode(prompt);

        model_->clear_kv_cache();

        // Process prompt
        std::vector<float> logits;
        model_->forward({input_ids.data()}, {static_cast<int>(input_ids.size())}, 1, logits);

        std::vector<int> output_ids;

        // Generate tokens
        std::mt19937 rng(std::random_device{}());

        for (int step = 0; step < config.max_tokens; step++) {
            // Get next token
            model_->forward({&input_ids.back()}, {1}, 1, logits);

            int next_token = sampler_.sample(logits.data(), logits.size(), rng);
            output_ids.push_back(next_token);

            // Decode and callback
            std::string text = tokenizer_->decode(output_ids);
            callback(next_token, text);

            // Check stop tokens
            if (std::find(config.stop_tokens.begin(), config.stop_tokens.end(), next_token)
                != config.stop_tokens.end()) {
                break;
            }
        }

        return tokenizer_->decode(output_ids);
    }
};
```

---

## 4. Code Editing

### 4.1 Context Manager

```cpp
class ContextManager {
public:
    struct ContextConfig {
        int max_tokens = 4096;
        int max_files = 10;
        float file_relevance_threshold = 0.1f;
    };

    std::vector<FileInfo> select_context(
        const std::string& target_file,
        const std::vector<FileInfo>& codebase,
        const ContextConfig& config
    ) {
        // Score files by relevance
        std::vector<std::pair<float, FileInfo>> scored_files;

        for (const auto& file : codebase) {
            float score = compute_relevance(target_file, file);
            scored_files.push_back({score, file});
        }

        // Sort by relevance
        std::sort(scored_files.begin(), scored_files.end(),
                  [](const auto& a, const auto& b) {
                      return a.first > b.first;
                  });

        // Select top files until budget exhausted
        std::vector<FileInfo> selected;
        int used_tokens = 0;

        for (const auto& [score, file] : scored_files) {
            if (score < config.file_relevance_threshold) {
                continue;
            }

            if (used_tokens + file.token_count > config.max_tokens) {
                continue;
            }

            selected.push_back(file);
            used_tokens += file.token_count;

            if (selected.size() >= config.max_files) {
                break;
            }
        }

        return selected;
    }

private:
    float compute_relevance(
        const std::string& target_file,
        const FileInfo& file
    ) {
        float score = 0.0f;

        // File type bonus
        if (has_common_extension(file.path, target_file.path)) {
            score += 1.0f;
        }

        // Import relationship
        if (imports(file.path, target_file.path) ||
            imports(target_file.path, file.path)) {
            score += 2.0f;
        }

        // Symbol overlap
        int shared_symbols = count_shared_symbols(file.symbols, target_file.symbols);
        score += shared_symbols * 0.1f;

        // Directory proximity
        if (same_directory(file.path, target_file.path)) {
            score += 0.5f;
        }

        return score;
    }

    struct FileInfo {
        std::string path;
        std::string content;
        int token_count;
        std::unordered_set<std::string> symbols;
        std::unordered_set<std::string> imports;
    };
};
```

### 4.2 Diff Generation

```cpp
class DiffGenerator {
public:
    struct Diff {
        std::string old_path;
        std::string new_path;
        std::string unified_diff;
    };

    Diff generate_diff(
        const std::string& old_content,
        const std::string& new_content,
        const std::string& file_path
    ) {
        // Split into lines
        std::vector<std::string> old_lines = split_lines(old_content);
        std::vector<std::string> new_lines = split_lines(new_content);

        // Compute edit script using Myers diff algorithm
        auto edit_script = compute_myers_diff(old_lines, new_lines);

        // Generate unified diff
        std::string diff = format_unified_diff(
            old_lines,
            new_lines,
            edit_script,
            file_path
        );

        return {file_path, file_path, diff};
    }

private:
    struct EditOp {
        enum Type { INSERT, DELETE, REPLACE, EQUAL };

        Type type;
        int old_start;
        int old_end;
        int new_start;
        int new_end;
    };

    std::vector<EditOp> compute_myers_diff(
        const std::vector<std::string>& old_lines,
        const std::vector<std::string>& new_lines
    ) {
        // Myers diff algorithm
        // (Implementation simplified for brevity)

        int N = old_lines.size();
        int M = new_lines.size();

        // Maximum possible edit distance
        int MAX = N + M;

        // Trace of furthest reaching D-paths
        std::vector<std::unordered_map<int, int>> V(2 * MAX + 1);

        V[1][1] = 0;

        for (int D = 0; D <= MAX; D++) {
            for (int k = -D; k <= D; k += 2) {
                // Determine whether to move down or right
                bool down = (k == -D || (k != D && V[k - 1] < V[k + 1]));

                int x, y;
                if (down) {
                    x = V[k + 1];
                    y = x - k - 1;
                } else {
                    x = V[k - 1] + 1;
                    y = x - k;
                }

                // Snake as far as possible
                while (x < N && y < M && old_lines[x] == new_lines[y]) {
                    x++;
                    y++;
                }

                V[k] = x;

                // Check if we reached the end
                if (x == N && y == M) {
                    // Backtrack to find edit script
                    return backtrack(V, k, D, old_lines, new_lines);
                }
            }
        }

        // Should not reach here
        return {};
    }

    std::vector<EditOp> backtrack(
        const std::vector<std::unordered_map<int, int>>& V,
        int k,
        int D,
        const std::vector<std::string>& old_lines,
        const std::vector<std::string>& new_lines
    ) {
        std::vector<EditOp> ops;

        int x = old_lines.size();
        int y = new_lines.size();

        for (int d = D; d > 0; d--) {
            bool down = (k == -d || (k != d && V[k - 1 + d] < V[k + 1 + d]));

            int prev_k, prev_x, prev_y;
            if (down) {
                prev_k = k + 1;
                prev_x = V[prev_k + d];
                prev_y = prev_x - prev_k - 1;
            } else {
                prev_k = k - 1;
                prev_x = V[prev_k + d] - 1;
                prev_y = prev_x - prev_k;
            }

            // Check if this was a snake
            while (prev_x > 0 && prev_y > 0 &&
                   old_lines[prev_x - 1] == new_lines[prev_y - 1]) {
                prev_x--;
                prev_y--;
            }

            // Determine edit operation
            EditOp op;
            if (prev_x == x && prev_y == y - 1) {
                op.type = EditOp::INSERT;
                op.new_start = prev_y;
                op.new_end = y;
            } else if (prev_x == x - 1 && prev_y == y) {
                op.type = EditOp::DELETE;
                op.old_start = prev_x;
                op.old_end = x;
            } else if (prev_x == x - 1 && prev_y == y - 1) {
                op.type = EditOp::REPLACE;
                op.old_start = prev_x;
                op.old_end = x;
                op.new_start = prev_y;
                op.new_end = y;
            } else {
                op.type = EditOp::EQUAL;
                op.old_start = prev_x;
                op.old_end = x;
                op.new_start = prev_y;
                op.new_end = y;
            }

            ops.push_back(op);

            x = prev_x;
            y = prev_y;
            k = prev_k;
        }

        std::reverse(ops.begin(), ops.end());
        return ops;
    }

    std::string format_unified_diff(
        const std::vector<std::string>& old_lines,
        const std::vector<std::string>& new_lines,
        const std::vector<EditOp>& ops,
        const std::string& file_path
    ) {
        std::ostringstream diff;

        diff << "--- a/" << file_path << "\n";
        diff << "+++ b/" << file_path << "\n";

        for (const auto& op : ops) {
            if (op.type == EditOp::EQUAL) {
                continue;
            }

            diff << "@@ -" << op.old_start + 1 << "," << op.old_end - op.old_start
                 << " +" << op.new_start + 1 << "," << op.new_end - op.new_start
                 << " @@\n";

            // Print old lines (with -)
            for (int i = op.old_start; i < op.old_end; i++) {
                diff << "-" << old_lines[i] << "\n";
            }

            // Print new lines (with +)
            for (int i = op.new_start; i < op.new_end; i++) {
                diff << "+" << new_lines[i] << "\n";
            }
        }

        return diff.str();
    }

    std::vector<std::string> split_lines(const std::string& text) {
        std::vector<std::string> lines;
        std::istringstream iss(text);
        std::string line;

        while (std::getline(iss, line)) {
            lines.push_back(line);
        }

        return lines;
    }
};
```

### 4.3 Diff Application

```cpp
class DiffApplier {
public:
    std::string apply_diff(
        const std::string& original_content,
        const DiffGenerator::Diff& diff
    ) {
        // Parse unified diff
        auto hunks = parse_unified_diff(diff.unified_diff);

        // Apply each hunk
        std::vector<std::string> lines = split_lines(original_content);

        int line_offset = 0;

        for (const auto& hunk : hunks) {
            // Validate hunk
            if (!validate_hunk(lines, hunk, line_offset)) {
                throw std::runtime_error("Failed to apply diff: hunk validation failed");
            }

            // Apply hunk
            apply_hunk(lines, hunk, line_offset);

            // Update offset
            line_offset += hunk.new_lines.size() - hunk.old_lines.size();
        }

        // Rejoin lines
        return join_lines(lines);
    }

private:
    struct Hunk {
        int old_start;
        int old_count;
        int new_start;
        int new_count;
        std::vector<std::string> old_lines;
        std::vector<std::string> new_lines;
    };

    std::vector<Hunk> parse_unified_diff(const std::string& diff) {
        std::vector<Hunk> hunks;
        std::istringstream iss(diff);
        std::string line;

        while (std::getline(iss, line)) {
            if (line.substr(0, 2) == "@@") {
                Hunk hunk;

                // Parse: @@ -old_start,old_count +new_start,new_count @@
                sscanf(line.c_str(), "@@ -%d,%d +%d,%d @",
                       &hunk.old_start, &hunk.old_count,
                       &hunk.new_start, &hunk.new_count);

                hunk.old_start--;  // Convert to 0-indexed

                // Read hunk lines
                while (std::getline(iss, line)) {
                    if (line.empty() || line[0] == '@') {
                        break;
                    }

                    if (line[0] == '-') {
                        hunk.old_lines.push_back(line.substr(1));
                    } else if (line[0] == '+') {
                        hunk.new_lines.push_back(line.substr(1));
                    } else if (line[0] == ' ') {
                        hunk.old_lines.push_back(line.substr(1));
                        hunk.new_lines.push_back(line.substr(1));
                    }
                }

                hunks.push_back(hunk);
            }
        }

        return hunks;
    }

    bool validate_hunk(
        const std::vector<std::string>& lines,
        const Hunk& hunk,
        int offset
    ) {
        int actual_start = hunk.old_start + offset;

        if (actual_start + hunk.old_count > lines.size()) {
            return false;
        }

        // Check that old lines match
        for (size_t i = 0; i < hunk.old_lines.size(); i++) {
            if (lines[actual_start + i] != hunk.old_lines[i]) {
                return false;
            }
        }

        return true;
    }

    void apply_hunk(
        std::vector<std::string>& lines,
        const Hunk& hunk,
        int offset
    ) {
        int actual_start = hunk.old_start + offset;

        // Remove old lines
        lines.erase(
            lines.begin() + actual_start,
            lines.begin() + actual_start + hunk.old_count
        );

        // Insert new lines
        lines.insert(
            lines.begin() + actual_start,
            hunk.new_lines.begin(),
            hunk.new_lines.end()
        );
    }

    std::string join_lines(const std::vector<std::string>& lines) {
        std::ostringstream oss;

        for (size_t i = 0; i < lines.size(); i++) {
            oss << lines[i];

            if (i < lines.size() - 1) {
                oss << "\n";
            }
        }

        return oss.str();
    }

    std::vector<std::string> split_lines(const std::string& text) {
        std::vector<std::string> lines;
        std::istringstream iss(text);
        std::string line;

        while (std::getline(iss, line)) {
            lines.push_back(line);
        }

        return lines;
    }
};
```

---

## 5. High-Level API

### 5.1 Code Editing Interface

```cpp
class CodeEditor {
public:
    CodeEditor(
        CromwellModel* model,
        Tokenizer* tokenizer,
        const std::string& codebase_path
    ) : model_(model),
        tokenizer_(tokenizer),
        codebase_path_(codebase_path) {
        // Index codebase
        index_codebase();
    }

    struct EditRequest {
        std::string file_path;
        std::string instruction;
        int max_context_tokens = 4096;
        int max_output_tokens = 1000;
    };

    struct EditResult {
        bool success;
        std::string edited_content;
        std::string diff;
        std::string error_message;
    };

    EditResult edit(const EditRequest& request) {
        try {
            // 1. Load file
            std::string file_content = load_file(request.file_path);

            // 2. Select context
            auto context = select_context(request.file_path, request.max_context_tokens);

            // 3. Create prompt
            std::string prompt = create_prompt(
                file_content,
                request.instruction,
                context
            );

            // 4. Generate edit
            TextGenerator::GenerationConfig gen_config;
            gen_config.max_tokens = request.max_output_tokens;
            gen_config.temperature = 0.3f;  // Lower temperature for edits
            gen_config.stop_tokens = {
                tokenizer_->special_tokens().at("<EDIT_END>"),
                tokenizer_->special_tokens().at("</FILE>"),
            };

            TextGenerator generator(model_, tokenizer_, Sampler(Sampler::TEMPERATURE, 0.3f));
            std::string output = generator.generate(prompt, gen_config);

            // 5. Parse output
            auto edit_result = parse_edit_output(output);

            // 6. Generate diff
            DiffGenerator diff_gen;
            auto diff = diff_gen.generate_diff(file_content, edit_result.edited_content, request.file_path);

            // 7. Apply diff (with validation)
            DiffApplier diff_applier;
            std::string edited_content = diff_applier.apply_diff(file_content, diff);

            // 8. Validate syntax (if applicable)
            if (!validate_syntax(edited_content, get_file_extension(request.file_path))) {
                return {
                    false,
                    "",
                    diff.unified_diff,
                    "Generated code has syntax errors"
                };
            }

            return {
                true,
                edited_content,
                diff.unified_diff,
                ""
            };

        } catch (const std::exception& e) {
            return {
                false,
                "",
                "",
                e.what()
            };
        }
    }

private:
    void index_codebase() {
        // Scan codebase directory
        for (const auto& entry : std::filesystem::recursive_directory_iterator(codebase_path_)) {
            if (entry.is_regular_file()) {
                std::string path = entry.path().string();
                std::string content = load_file(path);

                FileInfo info;
                info.path = path;
                info.content = content;
                info.token_count = tokenizer_->encode(content).size();
                info.symbols = extract_symbols(content);
                info.imports = extract_imports(content);

                codebase_.push_back(info);
            }
        }
    }

    std::vector<FileInfo> select_context(
        const std::string& target_file,
        int max_tokens
    ) {
        ContextManager ctx_mgr;
        ContextManager::ContextConfig config;
        config.max_tokens = max_tokens;

        return ctx_mgr.select_context(target_file, codebase_, config);
    }

    std::string create_prompt(
        const std::string& file_content,
        const std::string& instruction,
        const std::vector<FileInfo>& context
    ) {
        std::ostringstream prompt;

        // Add context files
        for (const auto& file : context) {
            prompt << "<FILE=" << file.path << ">\n";
            prompt << file.content << "\n";
            prompt << "</FILE>\n";
        }

        // Add target file
        prompt << "<FILE=" << "TARGET" << ">\n";
        prompt << file_content << "\n";
        prompt << "</FILE>\n";

        // Add instruction
        prompt << "<EDIT_START>\n";
        prompt << instruction << "\n";
        prompt << "<EDIT_END>\n";

        return prompt.str();
    }

    CromwellModel* model_;
    Tokenizer* tokenizer_;
    std::string codebase_path_;
    std::vector<FileInfo> codebase_;
};
```

---

## 6. Performance Optimization

### 6.1 KV Cache Management

```cpp
class KVCacheManager {
public:
    KVCacheManager(int max_cache_size = 100)
        : max_cache_size_(max_cache_size) {}

    void get_or_create_cache(
        const std::string& key,
        int num_layers,
        int max_seq_len,
        int num_kv_heads,
        int head_dim,
        KVCache*& cache
    ) {
        // Check LRU cache
        auto it = cache_map_.find(key);

        if (it != cache_map_.end()) {
            // Move to front (most recently used)
            cache_list_.splice(cache_list_.begin(), cache_list_, it->second);
            cache = it->second->cache.get();
            return;
        }

        // Create new cache
        cache = new KVCache(1, max_seq_len, num_layers, num_kv_heads, head_dim);

        // Add to cache
        auto entry = std::make_unique<CacheEntry>();
        entry->key = key;
        entry->cache.reset(cache);

        cache_list_.push_front(std::move(entry));
        cache_map_[key] = cache_list_.begin();

        // Evict if necessary
        while (cache_map_.size() > max_cache_size_) {
            auto evict = cache_list_.back();
            cache_map_.erase(evict->key);
            cache_list_.pop_back();
        }
    }

    void clear() {
        cache_list_.clear();
        cache_map_.clear();
    }

private:
    struct CacheEntry {
        std::string key;
        std::unique_ptr<KVCache> cache;
    };

    int max_cache_size_;
    std::list<std::unique_ptr<CacheEntry>> cache_list_;
    std::unordered_map<std::string, std::list<std::unique_ptr<CacheEntry>>::iterator> cache_map_;
};
```

---

## 7. Error Handling

```cpp
class CodeEditorException : public std::runtime_error {
public:
    CodeEditorException(const std::string& message)
        : std::runtime_error(message) {}
};

class ValidationException : public CodeEditorException {
public:
    ValidationException(const std::string& message)
        : CodeEditorException("Validation failed: " + message) {}
};

class DiffException : public CodeEditorException {
public:
    DiffException(const std::string& message)
        : CodeEditorException("Diff error: " + message) {}
};
```

---

**Document Status**: Complete
**Next**: Core Implementation Code Samples
