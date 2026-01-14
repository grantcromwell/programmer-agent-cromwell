# Cromwell VL-JEPA: Training Pipeline
## Hybrid JEPA + LM Training for Vision-Language Models

**Version**: 2.0
**Date**: 2025-01-14

---

## 1. Dataset Design

### 1.1 Primary Data Sources (Updated for VL-JEPA)

| Dataset | Size | Format | Focus | Weight |
|---------|------|--------|-------|--------|
| **TheStack v2** | 3TB | Per-file JSON | Code + natural language | 30% |
| **CodeContests** | 500GB | Problem + solution | Competitive programming | 10% |
| **GitHub PRs** | 200GB | Diff + metadata | Code review + edits | 10% |
| **ArXiv CS/ML** | 100GB | PDF→Text | Academic knowledge | 10% |
| **OpenStax CS/ML** | 50GB | Textbook chapters | Educational content | 5% |
| **Stack Overflow** | 100GB | Q&A posts | Explanations | 5% |
| **Vision-Text Pairs** | 500GB | Image+Text | **NEW: Multimodal data** | 25% |
| **Financial Code** | 300GB | Trading/Modeling | **NEW: Domain specific** | 5% |

**Vision-Text Pairs Breakdown**:
- Code screenshots with syntax highlighting → Source code
- Charts/graphs → Plotting code (matplotlib, plotly)
- UI mockups → Frontend code (HTML/CSS/React)
- Financial statements → Extraction code
- Documentation images → LaTeX/Markdown

**Total**: ~5TB of raw data → ~600GB tokenized

### 1.2 Hybrid Training Objective

```python
# Hybrid loss function for VL-JEPA
def hybrid_loss(
    model_outputs,
    batch,
    alpha=0.3,  # JEPA weight
    beta=0.7    # LM weight
):
    """
    Compute combined JEPA + LM loss.

    JEPA Loss: MSE in embedding space (representation learning)
    LM Loss: Cross-entropy in token space (generation)
    """
    # JEPA loss (masked embedding prediction)
    jepa_loss = compute_jepa_loss(
        predicted_embeddings=model_outputs['predicted_emb'],  # [N_masked, 512]
        target_embeddings=model_outputs['target_emb'],          # [N_masked, 512]
        mask=batch['mask']                                     # [N]
    )

    # LM loss (next token prediction)
    lm_loss = compute_lm_loss(
        predicted_logits=model_outputs['logits'],  # [seq_len, vocab_size]
        target_tokens=batch['target_tokens']        # [seq_len]
    )

    # Combined loss
    total_loss = alpha * jepa_loss + beta * lm_loss

    return {
        'total_loss': total_loss,
        'jepa_loss': jepa_loss,
        'lm_loss': lm_loss
    }
```

### 1.3 JEPA Masking Strategy

```python
# JEPA masking for vision-language inputs
class JEPAMasking:
    def __init__(self, mask_ratio=0.15):
        self.mask_ratio = mask_ratio
        self.strategies = {
            'vision': 'block',      # Block masking for images
            'language': 'random',   # Random masking for text
            'joint': 'span'         # Span masking for fusion
        }

    def generate_mask(self, inputs, modality='vision'):
        """
        Generate mask for JEPA training.

        Vision: Block masking (16×16 blocks)
        Language: Random masking (15% of tokens)
        Joint: Span masking (contiguous regions)
        """
        if modality == 'vision':
            return self._block_mask(inputs['height'], inputs['width'])
        elif modality == 'language':
            return self._random_mask(inputs['seq_len'])
        else:
            return self._span_mask(inputs['seq_len'])

    def _block_mask(self, H, W, block_size=16):
        """Block masking for vision tokens."""
        num_blocks_h = H // block_size
        num_blocks_w = W // block_size
        num_blocks = num_blocks_h * num_blocks_w
        num_mask = int(num_blocks * self.mask_ratio)

        mask = torch.zeros(num_blocks, dtype=torch.bool)
        mask_idx = torch.randperm(num_blocks)[:num_mask]
        mask[mask_idx] = True

        return mask.view(num_blocks_h, num_blocks_w)

    def _random_mask(self, seq_len):
        """Random masking for language tokens."""
        num_mask = int(seq_len * self.mask_ratio)
        mask = torch.zeros(seq_len, dtype=torch.bool)
        mask_idx = torch.randperm(seq_len)[:num_mask]
        mask[mask_idx] = True
        return mask

    def _span_mask(self, seq_len, span_length=32):
        """Span masking for joint embeddings."""
        num_spans = max(1, int(seq_len * self.mask_ratio / span_length))
        mask = torch.zeros(seq_len, dtype=torch.bool)

        for _ in range(num_spans):
            start = torch.randint(0, max(1, seq_len - span_length), (1,)).item()
            mask[start:start+span_length] = True

        return mask
```

---

## 2. Training Curriculum

### 2.1 Four-Stage Training

```
Stage 1: JEPA Pre-training (0-40% of tokens)
├── Objective: Masked embedding prediction
├── Data: All datasets (vision + language)
├── Loss: Pure JEPA (MSE in embedding space)
├── Duration: ~80B tokens
├── Learning rate: 3e-4 → 3e-5 (cosine decay)
└── Focus: Learn robust joint representations

Stage 2: Language Model Fine-tuning (40-70% of tokens)
├── Objective: Next token prediction
├── Data: Code-weighted (4:1 code:text)
├── Loss: Pure LM (cross-entropy)
├── Duration: ~90B tokens
├── Learning rate: 3e-5 → 1e-5
└── Focus: Learn generation capability

Stage 3: Hybrid Joint Training (70-90% of tokens)
├── Objective: Combined JEPA + LM
├── Data: Balanced mix
├── Loss: α=0.3 × L_JEPA + β=0.7 × L_LM
├── Duration: ~60B tokens
├── Learning rate: 1e-5 → 5e-6
└── Focus: Joint optimization

Stage 4: Instruction Fine-tuning (90-100% of tokens)
├── Objective: Task-specific performance
├── Data: Code editing tasks, multimodal QA
├── Loss: LM loss (task-specific)
├── Duration: ~20B tokens
├── Learning rate: 5e-6 → 1e-6
└── Focus: Downstream task performance
```

**Total Training: ~250B tokens**

### 2.2 Training Configuration

```yaml
training:
  batch_size: 256
  micro_batch_size: 4
  gradient_accumulation_steps: 64

  optimizers:
    type: adamw
    lr: 3.0e-4
    beta1: 0.9
    beta2: 0.95
    weight_decay: 0.1
    grad_clip: 1.0

  scheduler:
    type: cosine_decay
    warmup_tokens: 5B
    min_lr: 1.0e-6

  # JEPA-specific
  jepa:
    mask_ratio: 0.15
    prediction_steps: 3
    step_weights: [1.0, 0.8, 0.6]
    embedding_dim: 512

  # LM-specific
  lm:
    context_length: 4096
    vocab_size: 50000
    tie_embeddings: false

  # Hybrid
  hybrid:
    alpha: 0.3  # JEPA weight
    beta: 0.7   # LM weight
```

---

## 3. Data Loading

### 3.1 Multimodal DataLoader

```python
class MultimodalDataLoader:
    def __init__(self, datasets, tokenizer, image_processor):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.datasets = datasets

    def __getitem__(self, idx):
        """
        Load a multimodal sample.

        Returns:
        - vision_input: RGB image [H, W, 3] or None
        - language_input: Token IDs [seq_len]
        - target_tokens: Target token IDs for LM loss
        - mask: Mask for JEPA loss
        """
        sample = self.datasets[idx]

        # Process vision (if present)
        vision_input = None
        if 'image' in sample:
            vision_input = self.image_processor.process(
                sample['image'],
                size=(256, 256),
                normalize=True
            )

        # Process language
        language_input = self.tokenizer.encode(
            sample['text'],
            max_length=4096,
            truncation=True
        )

        # Create mask for JEPA
        mask = self._create_mask(language_input, vision_input)

        return {
            'vision_input': vision_input,
            'language_input': language_input,
            'target_tokens': language_input[1:],  # Shifted for LM
            'mask': mask
        }

    def _create_mask(self, lang_tokens, vision_input=None):
        """Create JEPA mask for hybrid inputs."""
        # Language masking
        lang_mask = self._random_mask(len(lang_tokens))

        # Vision masking (if applicable)
        if vision_input is not None:
            H, W = vision_input.shape[1:3]
            vision_mask = self._block_mask(H, W)
        else:
            vision_mask = None

        return {
            'language': lang_mask,
            'vision': vision_mask
        }
```

### 3.2 Image Processing Pipeline

```python
class ImageProcessor:
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size
        self.mean = [0.485, 0.456, 0.406]  # ImageNet stats
        self.std = [0.229, 0.224, 0.225]

    def process(self, image, size=None, normalize=True):
        """
        Process image for vision encoder.

        Pipeline:
        1. Resize to target size
        2. Convert to RGB
        3. Normalize
        4. Convert to tensor [C, H, W]
        """
        from PIL import Image
        import torchvision.transforms as T

        transforms = []
        if size is not None:
            transforms.append(T.Resize(size))
        transforms.append(T.ToTensor())
        if normalize:
            transforms.append(T.Normalize(self.mean, self.std))

        transform = T.Compose(transforms)
        return transform(image)
```

---

## 4. Training Loop

### 4.1 Hybrid Training Step

```python
def hybrid_train_step(model, batch, optimizer, scaler=None):
    """
    Single training step with hybrid JEPA + LM loss.

    Args:
        model: VL-JEPA model
        batch: Multimodal batch
        optimizer: AdamW optimizer
        scaler: GradScaler for mixed precision (optional)
    """
    # Forward pass
    with torch.cuda.amp.autocast() if scaler is not None else nullcontext():
        outputs = model(
            vision_input=batch['vision_input'],
            language_input=batch['language_input'],
            mask=batch['mask']
        )

        # Compute losses
        losses = compute_hybrid_loss(outputs, batch)

    # Backward pass
    optimizer.zero_grad()

    if scaler is not None:
        scaler.scale(losses['total_loss']).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        losses['total_loss'].backward()
        optimizer.step()

    return losses
```

### 4.2 Evaluation Metrics

```python
def evaluate_model(model, eval_dataloader):
    """
    Evaluate model on multimodal tasks.

    Metrics:
    - Text-only: Perplexity, HumanEval Pass@1
    - Vision+text: Image-to-code accuracy
    - JEPA: Embedding prediction MSE
    """
    model.eval()

    metrics = {
        'perplexity': [],
        'jepa_mse': [],
        'code_accuracy': [],
        'multimodal_accuracy': []
    }

    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)

            # Compute metrics
            metrics['perplexity'].append(compute_perplexity(outputs))
            metrics['jepa_mse'].append(compute_jepa_mse(outputs, batch))
            # ... other metrics

    # Aggregate
    return {k: torch.stack(v).mean().item() for k, v in metrics.items()}
```

---

## 5. Checkpointing

### 5.1 Checkpoint Format

```python
# Checkpoint structure
checkpoint = {
    # Model weights
    'vision_encoder': vision_encoder.state_dict(),
    'language_encoder': language_encoder.state_dict(),
    'cross_modal_fusion': fusion.state_dict(),
    'jepa_head': jepa_head.state_dict(),
    'autoregressive_head': ar_head.state_dict(),

    # Optimizer state
    'optimizer': optimizer.state_dict(),

    # Training state
    'step': global_step,
    'epoch': epoch,
    'tokens_seen': tokens_seen,

    # Metrics
    'train_loss': train_loss,
    'eval_metrics': eval_metrics,

    # Config
    'config': config,
    'tokenizer': tokenizer_state,
}
```

### 5.2 Conversion from PyTorch to C++

```python
# Convert PyTorch checkpoint to C++ format
def convert_checkpoint_to_cpp(pytorch_checkpoint, output_path):
    """
    Convert PyTorch checkpoint to C++ loadable format.

    Saves weights in binary format with metadata.
    """
    import numpy as np

    cpp_weights = {}

    for name, tensor in pytorch_checkpoint.items():
        # Convert to numpy
        np_array = tensor.cpu().numpy().astype(np.float32)

        # Save to file
        tensor_path = output_path / f"{name}.bin"
        np_array.tofile(tensor_path)

        # Store metadata
        cpp_weights[name] = {
            'path': str(tensor_path),
            'shape': list(np_array.shape),
            'dtype': 'float32'
        }

    # Save metadata
    import json
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(cpp_weights, f, indent=2)
```

---

## 6. Distributed Training

### 6.1 Data Parallel Training

```python
# Distributed training setup
import torch.distributed as dist

def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank

def distributed_train_step(model, batch, optimizer, local_rank):
    """Training step for distributed training."""
    # Forward pass
    outputs = model(**batch)

    # Compute loss (scaled by world size)
    loss = compute_hybrid_loss(outputs, batch)['total_loss']
    loss = loss / dist.get_world_size()

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Gradient averaging
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)

    optimizer.step()

    return loss.item() * dist.get_world_size()
```

---

**Document Status**: Updated for VL-JEPA hybrid training
**Next Document**: 05_INFERENCE.md
