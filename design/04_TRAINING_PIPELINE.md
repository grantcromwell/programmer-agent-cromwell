# Cromwell Agent: Training Pipeline
## Data Curriculum, Loss Functions, and Training Strategy

**Version**: 1.0
**Date**: 2025-01-14

---

## 1. Dataset Design

### 1.1 Primary Data Sources

| Dataset | Size | Format | Focus | Weight |
|---------|------|--------|-------|--------|
| **TheStack v2** | 3TB | Per-file JSON | Code + natural language | 40% |
| **CodeContests** | 500GB | Problem + solution | Competitive programming | 15% |
| **GitHub PRs** | 200GB | Diff + metadata | Code review + edits | 10% |
| **ArXiv CS/ML** | 100GB | PDF→Text | Academic knowledge | 15% |
| **OpenStax CS/ML** | 50GB | Textbook chapters | Educational content | 10% |
| **Stack Overflow** | 100GB | Q&A posts | Explanations | 10% |

**Total**: ~4TB of raw data → ~500GB tokenized

### 1.2 Data Preprocessing Pipeline

```python
# Full preprocessing pipeline
import hashlib
from pathlib import Path
from typing import List, Dict, Any

def preprocess_dataset(
    raw_data_dir: str,
    output_dir: str,
    tokenizer: BPETokenizer,
    max_file_size: int = 100000,  # 100KB max
    min_file_size: int = 100,      # 100B min
) -> None:
    """
    Preprocess raw dataset into training shards.

    Pipeline:
    1. Deduplication (MinHash LSH)
    2. Quality filtering
    3. Tokenization
    4. Sharding
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Deduplication
    print("Step 1: Deduplication...")
    deduped_files = deduplicate_files(raw_data_dir)

    # Step 2: Quality filtering
    print("Step 2: Quality filtering...")
    filtered_files = filter_files(
        deduped_files,
        max_file_size=max_file_size,
        min_file_size=min_file_size,
    )

    # Step 3: Tokenization
    print("Step 3: Tokenization...")
    tokenized_files = tokenize_files(
        filtered_files,
        tokenizer,
        output_dir,
    )

    # Step 4: Sharding
    print("Step 4: Sharding...")
    create_training_shards(
        tokenized_files,
        output_dir,
        samples_per_shard=1000,
        max_tokens_per_sample=10000,
    )

    print(f"Preprocessing complete! Output: {output_dir}")
```

### 1.3 Deduplication Strategy

```python
from datasketch import MinHashLSH
import pickle

def deduplicate_files(
    data_dir: str,
    threshold: float = 0.85,  # Jaccard similarity threshold
    n_perm: int = 128,        # MinHash permutations
) -> List[str]:
    """
    Remove duplicate files using MinHash LSH.

    Algorithm:
    1. Compute MinHash for each file
    2. Build LSH index
    3. Find near-duplicates (Jaccard > threshold)
    4. Keep one representative per duplicate cluster
    """
    # Initialize LSH
    lsh = MinHashLSH(num_perm=n_perm, threshold=threshold)

    # Collect all files
    files = list(Path(data_dir).rglob('*'))
    files = [f for f in files if f.is_file()]

    # Compute MinHash for each file
    minhashes = {}
    for file in files:
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Compute MinHash
        mhash = compute_minhash(content, n_perm)
        minhashes[file] = mhash

        # Add to LSH
        lsh.insert(str(file), mhash)

    # Find duplicates
    duplicates = set()
    for file in files:
        # Find similar files
        similar = lsh.query(minhashes[file])

        if len(similar) > 1:
            # Keep the first one, mark rest as duplicates
            similar = sorted(similar)
            for dup in similar[1:]:
                duplicates.add(dup)

    # Return unique files
    unique_files = [f for f in files if str(f) not in duplicates]

    print(f"Deduplication: {len(files)} → {len(unique_files)} files")

    return unique_files


def compute_minhash(content: str, n_perm: int = 128):
    """Compute MinHash for content."""
    from datasketch import MinHash

    mhash = MinHash(num_perm=n_perm)

    # Split into shingles (words)
    words = content.split()

    for word in words:
        mhash.update(word.encode('utf-8'))

    return mhash
```

### 1.4 Quality Filtering

```python
def filter_files(
    files: List[str],
    max_file_size: int = 100000,
    min_file_size: int = 100,
    min_avg_line_length: int = 5,
    max_avg_line_length: int = 200,
) -> List[str]:
    """
    Filter files based on quality metrics.

    Metrics:
    - File size (remove too large/small)
    - Average line length (remove weird formatting)
    - Character distribution (remove non-text/binary)
    - Language detection (keep English + code)
    - Comment density (for code files)
    """
    filtered = []

    for file in files:
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Check file size
        if len(content) < min_file_size or len(content) > max_file_size:
            continue

        # Check average line length
        lines = content.split('\n')
        avg_line_length = sum(len(line) for line in lines) / len(lines)

        if avg_line_length < min_avg_line_length or avg_line_length > max_avg_line_length:
            continue

        # Check character distribution (for binary files)
        if is_binary(content):
            continue

        # Check language
        if not is_english_or_code(content):
            continue

        filtered.append(file)

    print(f"Filtering: {len(files)} → {len(filtered)} files")

    return filtered


def is_binary(content: str, threshold: float = 0.3) -> bool:
    """Check if content is binary."""
    # Count non-printable characters
    non_printable = sum(1 for c in content if ord(c) < 32 and c not in '\n\r\t')

    return non_printable / len(content) > threshold


def is_english_or_code(content: str) -> bool:
    """Check if content is English or code."""
    # Simple heuristic: check for common English words or code keywords
    english_words = {'the', 'and', 'is', 'in', 'it', 'to', 'of', 'a', 'for'}
    code_keywords = {'def', 'class', 'function', 'import', 'return', 'if', 'else'}

    words = set(content.lower().split())

    has_english = any(word in words for word in english_words)
    has_code = any(word in words for word in code_keywords)

    return has_english or has_code
```

### 1.5 Tokenization and Sharding

```python
import json
import numpy as np

def tokenize_files(
    files: List[str],
    tokenizer: BPETokenizer,
    output_dir: str,
    max_tokens: int = 10000,
) -> List[str]:
    """
    Tokenize files and save to disk.

    Output format:
    {
        'tokens': [1, 2, 3, ...],
        'metadata': {
            'source': str,
            'file_type': str,
            'token_count': int,
        }
    }
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenized_files = []

    for file in files:
        with open(file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Tokenize
        tokens = tokenizer.encode(content)

        # Truncate if too long
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]

        # Save tokenized file
        output_file = output_path / f"{file.name}.json"

        with open(output_file, 'w') as f:
            json.dump({
                'tokens': tokens,
                'metadata': {
                    'source': str(file),
                    'file_type': file.suffix,
                    'token_count': len(tokens),
                }
            }, f)

        tokenized_files.append(str(output_file))

    return tokenized_files


def create_training_shards(
    tokenized_files: List[str],
    output_dir: str,
    samples_per_shard: int = 1000,
    max_tokens_per_sample: int = 10000,
) -> None:
    """
    Create training shards for efficient data loading.

    Each shard contains:
    - Multiple samples (tokenized files)
    - Balanced by data source
    - Fixed size for efficient loading
    """
    output_path = Path(output_dir)
    shards_dir = output_path / 'shards'
    shards_dir.mkdir(parents=True, exist_ok=True)

    # Group by data source
    by_source = {}
    for file in tokenized_files:
        with open(file, 'r') as f:
            data = json.load(f)

        source = data['metadata']['source']

        if source not in by_source:
            by_source[source] = []

        by_source[source].append(data)

    # Create shards
    shard_id = 0
    samples_in_shard = 0

    current_shard = {
        'samples': [],
        'metadata': {
            'shard_id': shard_id,
            'sample_count': 0,
        }
    }

    # Round-robin sampling from each source
    sources = list(by_source.keys())
    source_indices = {s: 0 for s in sources}

    while True:
        # Check if we have more data
        has_data = False
        for source in sources:
            if source_indices[source] < len(by_source[source]):
                has_data = True
                break

        if not has_data:
            break

        # Sample from each source
        for source in sources:
            if source_indices[source] >= len(by_source[source]):
                continue

            # Get sample
            sample = by_source[source][source_indices[source]]
            source_indices[source] += 1

            current_shard['samples'].append(sample)
            samples_in_shard += 1

            # Check if shard is full
            if samples_in_shard >= samples_per_shard:
                # Save shard
                shard_file = shards_dir / f"shard_{shard_id:06d}.json"

                with open(shard_file, 'w') as f:
                    current_shard['metadata']['sample_count'] = samples_in_shard
                    json.dump(current_shard, f)

                # Reset for next shard
                shard_id += 1
                samples_in_shard = 0
                current_shard = {
                    'samples': [],
                    'metadata': {
                        'shard_id': shard_id,
                        'sample_count': 0,
                    }
                }

    # Save final shard (if not empty)
    if samples_in_shard > 0:
        shard_file = shards_dir / f"shard_{shard_id:06d}.json"

        with open(shard_file, 'w') as f:
            current_shard['metadata']['sample_count'] = samples_in_shard
            json.dump(current_shard, f)

    print(f"Created {shard_id + 1} shards")
```

---

## 2. Training Curriculum

### 2.1 Four-Stage Training

```
Stage 1: Foundation (0-150B tokens)
  Objective: General language modeling
  Data: All datasets mixed
  Duration: ~150B tokens

Stage 2: Code Understanding (150-200B tokens)
  Objective: Code-specific tasks
  Data: Code-weighted (4:1 ratio)
  Duration: ~50B tokens

Stage 3: Editing Skills (200-225B tokens)
  Objective: Diff generation
  Data: GitHub PRs, CodeContests
  Duration: ~25B tokens

Stage 4: Instruction Tuning (225-235B tokens)
  Objective: Follow editing instructions
  Data: Synthetic instructions + demonstrations
  Duration: ~10B tokens

Total: ~235B tokens
```

### 2.2 Stage 1: Foundation

**Objective**: Standard next-token prediction

```python
def foundation_loss(
    logits: torch.Tensor,  # [batch_size, seq_len, vocab_size]
    targets: torch.Tensor,  # [batch_size, seq_len]
) -> torch.Tensor:
    """
    Standard cross-entropy loss.

    L = -sum(log P(token_t | token_<t))
    """
    # Flatten
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)

    # Cross-entropy
    loss = F.cross_entropy(logits_flat, targets_flat, reduction='mean')

    return loss
```

**Learning Rate Schedule**:

```python
def get_lr_schedule(
    step: int,
    warmup_steps: int = 2000,
    max_lr: float = 3e-4,
    min_lr: float = 3e-5,
    total_steps: int = 150000,  # 150B tokens / 1M tokens per batch
) -> float:
    """
    Cosine decay with warmup.

    Formula:
    - Warmup: lr = max_lr * step / warmup_steps
    - Decay: lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + cos(pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    """
    if step < warmup_steps:
        # Linear warmup
        return max_lr * step / warmup_steps
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
```

### 2.3 Stage 2: Code Understanding

**Objectives**:
1. Next-token prediction (continued)
2. Fill-in-the-middle (FIM)
3. Function completion

```python
def fim_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    prefix_len: int,
    middle_len: int,
    suffix_len: int,
) -> torch.Tensor:
    """
    Fill-in-the-middle loss.

    Format: [prefix, suffix, middle]
    Task: Predict middle given prefix and suffix
    """
    # We only care about predicting the middle part
    middle_logits = logits[:, prefix_len + suffix_len:prefix_len + suffix_len + middle_len, :]
    middle_targets = targets[:, prefix_len + suffix_len:prefix_len + suffix_len + middle_len]

    # Cross-entropy on middle only
    loss = F.cross_entropy(
        middle_logits.view(-1, middle_logits.size(-1)),
        middle_targets.view(-1),
        reduction='mean'
    )

    return loss


def create_fim_sample(tokens: List[int], fim_rate: float = 0.5) -> Dict[str, Any]:
    """
    Create fill-in-the-middle sample.

    Algorithm:
    1. With probability fim_rate, apply FIM
    2. Split tokens into prefix, middle, suffix
    3. Permute: [prefix, suffix, middle]
    """
    if random.random() > fim_rate:
        # Return normal sample
        return {
            'tokens': tokens,
            'type': 'standard',
        }

    # Split point (random)
    split1 = random.randint(len(tokens) // 4, len(tokens) // 2)
    split2 = random.randint(len(tokens) // 2, 3 * len(tokens) // 4)

    prefix = tokens[:split1]
    middle = tokens[split1:split2]
    suffix = tokens[split2:]

    # Permute
    fim_tokens = prefix + suffix + middle

    return {
        'tokens': fim_tokens,
        'type': 'fim',
        'prefix_len': len(prefix),
        'middle_len': len(middle),
        'suffix_len': len(suffix),
    }
```

### 2.4 Stage 3: Editing Skills

**Objectives**:
1. Diff prediction
2. Bug fix generation
3. Refactoring

```python
def diff_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    diff_mask: torch.Tensor,  # [batch_size, seq_len]
) -> torch.Tensor:
    """
    Diff-specific loss.

    We care more about predicting the edited regions correctly.
    """
    # Standard cross-entropy
    ce_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        reduction='none'
    ).view(targets.size())

    # Weight diff regions higher
    diff_weight = 2.0
    weights = torch.ones_like(ce_loss)
    weights[diff_mask] = diff_weight

    # Weighted loss
    weighted_loss = (ce_loss * weights).sum() / weights.sum()

    return weighted_loss


def create_diff_sample(
    before_file: str,
    after_file: str,
    tokenizer: BPETokenizer,
) -> Dict[str, Any]:
    """
    Create diff-based training sample.

    Format:
    Input: <FILE=path> [before content] <EDIT_START> [instruction] <EDIT_END>
    Output: [after content]
    """
    # Generate diff
    diff = unified_diff(before_file, after_file)

    # Tokenize
    before_tokens = tokenizer.encode(before_file)
    after_tokens = tokenizer.encode(after_file)
    diff_tokens = tokenizer.encode(diff)

    # Create sample
    sample = {
        'input': before_tokens + diff_tokens,
        'target': after_tokens,
        'type': 'diff',
    }

    return sample
```

### 2.5 Stage 4: Instruction Tuning

**Objectives**:
1. Follow editing instructions
2. Generate valid diffs
3. Maintain code correctness

```python
def instruction_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    instruction_mask: torch.Tensor,
    syntax_validity: torch.Tensor,  # [batch_size] (0 or 1)
) -> torch.Tensor:
    """
    Instruction following loss.

    Components:
    1. Cross-entropy on output
    2. Bonus for syntax validity
    3. Penalty for malformed output
    """
    # Standard cross-entropy
    ce_loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        reduction='mean'
    )

    # Syntax validity bonus
    validity_bonus = -0.1 * syntax_validity.mean()

    # Combined loss
    total_loss = ce_loss + validity_bonus

    return total_loss


def create_instruction_sample(
    instruction: str,
    file_content: str,
    edited_content: str,
    tokenizer: BPETokenizer,
) -> Dict[str, Any]:
    """
    Create instruction-following sample.

    Format:
    Input: <FILE=path> [file content] <EDIT_START> [instruction] <EDIT_END>
    Output: [edited content]
    """
    # Tokenize
    file_tokens = tokenizer.encode(file_content)
    instruction_tokens = tokenizer.encode(instruction)
    edited_tokens = tokenizer.encode(edited_content)

    # Create sample
    sample = {
        'input': file_tokens + instruction_tokens,
        'target': edited_tokens,
        'type': 'instruction',
    }

    return sample
```

---

## 3. Training Configuration

### 3.1 Hyperparameters

```yaml
training:
  # Batch size
  batch_size: 256  # Number of sequences
  sequence_length: 2048  # Tokens per sequence
  tokens_per_batch: 524288  # batch_size * sequence_length

  # Optimizer
  optimizer: "adamw"
  learning_rate: 3.0e-4
  weight_decay: 0.1
  beta1: 0.9
  beta2: 0.95
  epsilon: 1.0e-8

  # Learning rate schedule
  warmup_steps: 2000
  min_lr: 3.0e-5
  lr_decay: "cosine"

  # Training duration
  max_steps: 235000  # 235B tokens / 1M tokens per batch
  save_steps: 5000
  eval_steps: 1000
  logging_steps: 100

  # Precision
  precision: "bf16"  # Brain float 16

  # Distributed training
  num_nodes: 4
  gpus_per_node: 8
  total_gpus: 32

  # Gradient clipping
  max_grad_norm: 1.0

  # Checkpointing
  checkpoint_dir: "checkpoints"
  keep_last_n_checkpoints: 3
```

### 3.2 Training Script

```python
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

def train_model(
    model: CromwellModel,
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: Dict[str, Any],
):
    """
    Main training loop.
    """
    # Setup distributed training
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    # Move model to GPU
    model = model.to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(config['beta1'], config['beta2']),
        eps=config['epsilon'],
    )

    # Data loaders
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
    )

    # Training loop
    global_step = 0

    for epoch in range(config['num_epochs']):
        train_sampler.set_epoch(epoch)

        for batch in train_loader:
            # Move to GPU
            input_ids = batch['input_ids'].to(local_rank)
            targets = batch['targets'].to(local_rank)

            # Forward pass
            logits = model(input_ids)

            # Compute loss
            loss = compute_loss(logits, targets, batch, global_step)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['max_grad_norm']
            )

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

            # Logging
            if global_step % config['logging_steps'] == 0:
                lr = get_lr_schedule(global_step)
                print(f"Step {global_step}: loss={loss.item():.4f}, lr={lr:.2e}")

            # Evaluation
            if global_step % config['eval_steps'] == 0:
                eval_loss = evaluate(model, val_loader, local_rank)
                print(f"Eval loss: {eval_loss:.4f}")

            # Checkpointing
            if global_step % config['save_steps'] == 0:
                save_checkpoint(model, optimizer, global_step, config)

            global_step += 1

            # Check completion
            if global_step >= config['max_steps']:
                break

        if global_step >= config['max_steps']:
            break

    # Save final model
    save_checkpoint(model, optimizer, global_step, config, final=True)
```

### 3.3 Evaluation

```python
def evaluate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: int,
) -> float:
    """
    Evaluate model on validation set.
    """
    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            # Move to GPU
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)

            # Forward pass
            logits = model(input_ids)

            # Compute loss
            loss = compute_loss(logits, targets, batch, 0)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches

    model.train()

    return avg_loss
```

---

## 4. Synthetic Data Generation

### 4.1 Instruction Generation

```python
def generate_editing_instructions(
    code_corpus: List[str],
    num_instructions: int = 100000,
) -> List[Dict[str, str]]:
    """
    Generate synthetic editing instructions.

    Templates:
    - "Replace X with Y"
    - "Optimize this function"
    - "Add error handling"
    - "Refactor to use Z"
    - "Fix the bug in..."
    """
    instructions = []

    templates = [
        "Replace {old_code} with {new_code}",
        "Optimize the {entity} for better performance",
        "Add error handling to {entity}",
        "Refactor {entity} to use {method}",
        "Fix the bug in {entity}",
        "Add documentation to {entity}",
        "Rename {entity} to {new_name}",
        "Extract {entity} into a separate function",
        "Add type hints to {entity}",
        "Simplify {entity}",
    ]

    for _ in range(num_instructions):
        # Sample random code file
        code = random.choice(code_corpus)

        # Parse code to extract entities
        entities = extract_entities(code)

        if not entities:
            continue

        # Sample entity
        entity = random.choice(entities)

        # Generate instruction
        template = random.choice(templates)

        instruction = template.format(
            old_code=entity['code'],
            new_code=generate_alternative(entity['code']),
            entity=entity['name'],
            method=random.choice(['list comprehension', 'generator', 'map', 'lambda']),
            new_name=f"new_{entity['name']}",
        )

        # Apply edit to get target
        edited_code = apply_edit(code, instruction)

        instructions.append({
            'instruction': instruction,
            'code': code,
            'edited_code': edited_code,
        })

    return instructions
```

---

## 5. Training Infrastructure

### 5.1 Distributed Training Setup

```bash
# Launch training on 32 GPUs (4 nodes, 8 GPUs each)
torchrun \
  --nproc_per_node=8 \
  --nnodes=4 \
  --node_rank=$RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=29500 \
  train.py \
  --config configs/training.yaml
```

### 5.2 Monitoring

```python
# Weights & Biases integration
import wandb

def setup_wandb(config: Dict[str, Any]):
    wandb.init(
        project="cromwell-agent",
        config=config,
    )

def log_metrics(step: int, metrics: Dict[str, float]):
    wandb.log(metrics, step=step)
```

---

## 6. Expected Training Curve

```
Loss vs. Tokens:
Stage 1 (0-150B):
  - Initial loss: ~10.0
  - Mid-stage loss: ~3.5
  - Final loss: ~2.8

Stage 2 (150-200B):
  - Initial loss: ~2.8
  - Final loss: ~2.3

Stage 3 (200-225B):
  - Initial loss: ~2.3
  - Final loss: ~2.0

Stage 4 (225-235B):
  - Initial loss: ~2.0
  - Final loss: ~1.8

Expected perplexity: exp(1.8) ≈ 6.0
```

---

## 7. Next Steps

1. **Collect datasets** - Download and preprocess
2. **Train tokenizer** - Learn vocabulary
3. **Setup training infrastructure** - Distributed training
4. **Run training** - Monitor and iterate
5. **Evaluate on benchmarks** - Measure performance
6. **Convert to C++** - Production deployment

---

**Document Status**: Complete
**Next Document**: 05_INFERENCE.md
