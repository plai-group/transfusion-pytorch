# Transfusion-PyTorch Agent Instructions

## Overview
This is a PyTorch implementation of Transfusion (Meta AI, 2024) - a unified multi-modal model that combines autoregressive text generation with flow matching for continuous modalities (images, audio, embeddings). The codebase substitutes diffusion with flow matching inspired by Flux.

## Core Architecture

### Unified Token Representation
- **Text tokens**: `torch.long` tensors (discrete, autoregressive)
- **Modality tokens**: `torch.float` tensors (continuous, flow-matched)
- **Multi-modality**: Use tuples `(modality_type: int, tensor: Float)` to specify which modality

Example data format:
```python
# Single modality
[randint(0, 256, (16,)), randn(4, 384), randint(0, 256, (8,))]

# Multi-modality (explicit type indexing)
[randint(0, 256, (16,)), (0, randn(4, 384)), (1, randn(6, 192))]
```

### Key Components in `transfusion_pytorch/transfusion.py`

1. **Transfusion class** (line ~1226): Main model combining transformer backbone with modality encoding/decoding
   - `forward_text()`: Autoregressive text loss (cross-entropy)
   - `forward_modality()`: Flow matching loss for continuous modalities
   - `sample()`: Unified sampling for mixed text-modality sequences
   
2. **Flow Matching**: Uses `torchdiffeq.odeint` for sampling (not discrete diffusion steps)
   - Training: Learn to predict flow from noise to data at random times t ∈ [0,1]
   - Sampling: Integrate ODE from t=0 (noise) to t=1 (data)
   - Times for previously decoded modalities fixed at 0.5 (line ~215)

3. **Special tokens**:
   - `sos_id`/`eos_id`: Sentence start/end
   - `som_ids[i]`/`eom_ids[i]`: Start/End of modality (per modality type)
   - `meta_id`: For modality shape metadata (e.g., "32,32" for 32×32 images)

### Modality Encoding/Decoding
- Optional `modality_encoder`/`modality_decoder` parameters handle raw data (e.g., images → latents)
- `channel_first_latent=True`: Rearranges `(b, c, h, w) → (b, h, w, c) → (b, h*w, c)` for transformer
- `add_pos_emb=True`: Injects axial positional embeddings for 2D/3D modalities
- `pre_post_transformer_enc_dec`: Optional UNet-style encoder/decoder around transformer

## Development Workflows

### Training Scripts Pattern
All `train_*.py` scripts follow this structure:
1. Define model with `Transfusion(**config)` 
2. Create EMA model: `ema_model = model.create_ema(beta=0.99)`
3. Use special collate: `dataloader = model.create_dataloader(dataset, batch_size=16)`
4. Training loop: `loss = model(batch)` (automatically handles mixed modalities)
5. Sampling: `ema_model.sample(prompt=..., max_length=...)`

Key utilities:
- `print_modality_sample(sample)`: Pretty-print structure of generated sample
- `divisible_by(step, N)`: Common helper for periodic evaluation

### Testing
- Tests in `tests/test_transfusion.py` use `@pytest.mark.parametrize` extensively
- Key tested configurations: `cache_kv`, `use_flex_attn`, `num_residual_streams`, `reconstruction_loss_weight`
- Mock data generation: `partial(randint, 0, num_tokens)` for text, `randn(...)` for modalities

### Running Examples
```bash
pip install .[examples]  # Install with example dependencies
python train_mnist.py    # Text-to-image or captioning (toggle IMAGE_AFTER_TEXT)
python train_text_only.py # Pure language modeling on enwik8
python train_toy.py      # Minimal multimodal example
```

## Project Conventions

### Type Annotations
- Uses `jaxtyping` + `beartype` for tensor shape checking
- Enable runtime checking: `export TYPECHECK=1`
- Custom types: `Float['b n d']`, `Int['b seq']`, etc. (see lines 62-67)
- `@typecheck` decorator wraps functions when `TYPECHECK=1`

### Code Style
- Ein notation documented at top of `transfusion.py` (lines 4-14)
- `einops` for reshaping: `rearrange`, `repeat`, `reduce`, `einsum`
- `einx` for advanced operations (e.g., `einx.where`, `einx.less`)
- Ruff configured with line-length=1000, ignores F722 (jaxtyping), F401, F821, E402

### Loss Breakdown
- `LossBreakdown` NamedTuple (line ~96): Returns `total`, `text`, `flow`, optional `velocity`/`recon`
- Velocity consistency loss: Requires passing `velocity_consistency_ema_model` during training
- Reconstruction loss: Optional, requires decoder and `reconstruction_loss_weight > 0`

### Attention Mechanisms
- Supports PyTorch flex attention for memory efficiency (`use_flex_attn=True`, requires CUDA)
- Custom attention masking: `transfusion_attn_mask()` allows modalities to attend to their full sequence
- Causal masking for text, bidirectional within modality segments

## Key Implementation Details

### Modality Shape Handling
- Language model generates shape metadata as text (e.g., "32,32")
- `default_to_modality_shape_fn()` (line ~197): Parses "h,w,..." strings to tuples
- Falls back to `modality_default_shape` if invalid or not generated
- `fallback_to_default_shape_if_invalid=True`: Always use default (useful for early training)

### Batching Mixed Sequences
- Sequences vary in length and modality order → custom `collate_fn` returns list of lists
- `create_dataloader()` (line ~327): Automatically uses correct collate function
- Padding handled internally per modality type

### Flex Attention Compilation
- Auto-compiles if available: `flex_attention = torch.compile(flex_attention)` (line ~72)
- Graceful fallback to standard attention if not available
- Tests skip flex attention tests when CUDA unavailable

## Common Patterns

### Text-only pretraining then multimodal finetuning
```python
# Stage 1: Text only
model(text_tokens)  # text_tokens: Int['b seq']

# Stage 2: Add modalities
model(mixed_data)   # mixed_data: list of [text, modality, text, ...]
```

### Conditional generation
```python
# Text → image
model.sample(prompt=text_label, max_length=384)

# Image → text (captioning)  
model.sample(prompt=image, max_length=384)
```

### Multiple modalities in one model
```python
model = Transfusion(
    dim_latent=(384, 192),  # Two modality types
    modality_default_shape=((32, 32), (64,)),  # Images and audio
    ...
)
```

## Dependencies
Core: `torch`, `einops`, `einx`, `beartype`, `jaxtyping`, `torchdiffeq`, `ema-pytorch`, `hyper-connections`
Examples: `diffusers`, `datasets`, `adam-atan2-pytorch`

## Adding TODOs for AI Agents

### Where to Add TODOs
1. **In-code comments** (discoverable via `grep_search`):
   ```python
   # TODO: Add support for 3D modalities (videos)
   # FIXME: Velocity consistency loss unstable with very small delta_time
   # OPTIMIZE: Batch processing for multi-modal sequences could be more efficient
   ```

2. **GitHub Issues**: Create issues with labels like `good-first-issue`, `ai-agent-friendly`, `enhancement`
   - Include specific file references and line numbers
   - Provide context about expected behavior
   - Link to related test cases

3. **Project-level TODO file**: Create `TODO.md` in project root for larger initiatives:
   ```markdown
   ## High Priority
   - [ ] Add video modality support (3D flow matching)
   - [ ] Implement CFG (Classifier-Free Guidance) for conditional generation
   
   ## Implementation Notes
   - Video: Extend `modality_num_dim` to support 3, update `ContinuousAxialPositionalEmbedding`
   - CFG: Add unconditional training mode, modify sampling to blend conditional/unconditional flows
   ```

4. **Test files**: Mark incomplete test coverage:
   ```python
   # TODO: Add test for mixed 0D/2D modalities (embeddings + images)
   @pytest.mark.skip(reason="Not yet implemented")
   def test_mixed_dimensionality():
       pass
   ```

### TODO Formatting for Best AI Discovery
- **Be specific**: Include file paths, function names, line numbers
- **Provide context**: Explain why, not just what
- **Link examples**: Reference similar existing code
- **Specify constraints**: Mention compatibility requirements, performance targets

Example good TODO:
```python
# TODO: Extend forward_modality() (line ~1946) to support batched multi-resolution
# Currently assumes all samples in batch have same modality shape
# See train_mnist.py for single-resolution example
# Challenge: Flex attention mask needs dynamic per-sample resolution handling
```

## Resources
- Main paper: "Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model" (Zhou et al. 2024)
- Related: Flow matching (replaces discrete diffusion), MonoFormer, Consistency Flow Matching
- README contains all citations and architecture inspirations
