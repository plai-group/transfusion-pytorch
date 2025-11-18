# Transfusion-PyTorch TODO List

## High Priority Features

### [ ] Add Video Modality Support (3D Flow Matching)
**Files**: `transfusion_pytorch/transfusion.py`
**Context**: Currently supports 0D (embeddings), 1D (audio), 2D (images). Videos need 3D support.

**Implementation Notes**:
- Extend `modality_num_dim` to support `3` for (frames, height, width)
- Update `ContinuousAxialPositionalEmbedding` initialization (line ~1335) to handle 3 axes
- Add example in `train_video.py` using a simple video dataset
- Test with `test_video_modality()` in `tests/test_transfusion.py`

**References**:
- See `train_mnist.py` for 2D image handling pattern
- `modality_num_dim = 2` initialization around line ~1310

---

### [ ] Implement Classifier-Free Guidance (CFG)
**Files**: `transfusion_pytorch/transfusion.py` (lines ~1946, ~1581)
**Context**: Enable better conditional generation quality via CFG during sampling

**Implementation Notes**:
- Add `cfg_scale` parameter to `sample()` method (line ~1581)
- Modify `forward_modality()` (line ~1946) to support unconditional training mode
- During sampling, blend conditional and unconditional flow predictions: `flow = uncond_flow + cfg_scale * (cond_flow - uncond_flow)`
- Add `unconditional_prob` parameter to training (e.g., 0.1 dropout of conditioning)
- Store special "null" conditioning token for unconditional generation

**References**:
- Standard CFG implementation pattern from diffusion models
- `sample()` method already has temperature/min_p for text, add similar for modalities

---

### [ ] Batched Multi-Resolution Support
**Files**: `transfusion_pytorch/transfusion.py` (line ~1946)
**Context**: Currently assumes all samples in batch have same modality shape

**Implementation Notes**:
- Extend `forward_modality()` to handle variable shapes per batch item
- Update flex attention mask generation in `transfusion_attn_mask()` (line ~342)
- Each modality instance needs its own shape metadata and padding strategy
- Requires per-sample dynamic masking in flex attention

**Challenge**: Flex attention mask needs dynamic per-sample resolution handling, may require nested masking or bucketed batching

**References**:
- `train_mnist.py` shows single-resolution pattern
- `modality_positions_to_tensor()` (line ~366) for position tracking

---

## Medium Priority Enhancements

### [ ] Add Mixed Precision (AMP) Training Support
**Files**: All `train_*.py` scripts
**Context**: Enable faster training with torch.amp for modern GPUs

**Implementation Notes**:
- Add `torch.cuda.amp.autocast()` context managers in training loops
- Use `GradScaler` for loss scaling
- Test that flow matching predictions remain stable with fp16
- Add `use_amp` flag to training scripts

**References**:
- PyTorch AMP documentation: https://pytorch.org/docs/stable/amp.html
- Check velocity consistency loss stability with mixed precision

---

### [ ] Optimize Memory Usage for Long Sequences
**Files**: `transfusion_pytorch/transfusion.py` (Transformer class, line ~1007)
**Context**: Large multi-modal sequences (text + multiple images) can OOM

**Implementation Notes**:
- Implement gradient checkpointing option in Transformer blocks
- Add `use_checkpoint` parameter to `Transformer.__init__()`
- Use `torch.utils.checkpoint.checkpoint()` for attention + FFN layers
- Trade compute for memory (2x slower but ~50% less memory)

**References**:
- PyTorch checkpoint docs: https://pytorch.org/docs/stable/checkpoint.html
- `cache_kv` parameter already exists for inference optimization

---

### [ ] Add Gradio Demo Interface
**Files**: Create `demo.py` in project root
**Context**: Easy web interface for testing text-to-image and image-to-text

**Implementation Notes**:
- Use `gradio` library for UI
- Load pretrained checkpoint (add checkpoint saving to training scripts)
- Two modes: text→image generation, image→text captioning
- Display `print_modality_sample()` output for debugging
- Add image upload widget and text input box

**Example Structure**:
```python
import gradio as gr
from transfusion_pytorch import Transfusion

def generate(text_prompt):
    sample = model.sample(prompt=text_prompt, max_length=384)
    return extract_image(sample)

demo = gr.Interface(fn=generate, inputs="text", outputs="image")
```

---

## Testing & Documentation

### [ ] Add Test for Mixed Dimensionality (0D + 2D)
**Files**: `tests/test_transfusion.py`
**Context**: No test for mixing embeddings (0D) with images (2D) in same sequence

**Implementation**:
```python
def test_mixed_0d_2d_modalities():
    model = Transfusion(
        num_text_tokens=256,
        dim_latent=(384, 384),
        modality_default_shape=((), (4, 4)),  # 0D and 2D
        modality_num_dim=(0, 2),
        add_pos_emb=(False, True),  # Only 2D needs pos emb
        transformer=dict(dim=64, depth=2)
    )
    
    mixed_data = [
        [randint(0, 256, (16,)), (0, randn(384)), (1, randn(4, 4, 384))]
    ]
    
    loss = model(mixed_data)
    loss.backward()
    
    sample = model.sample(max_length=128)
    assert len(sample) >= 2
```

---

### [ ] Document Velocity Consistency Loss
**Files**: `README.md`, `.github/copilot-instructions.md`
**Context**: Velocity consistency is mentioned but not well explained

**Add Section**:
```markdown
### Velocity Consistency Loss
Inspired by Consistency Flow Matching (Yang et al. 2024), this optional loss enforces
flow predictions to be consistent at different time steps.

Usage:
```python
ema_model = model.create_ema()
loss = model(batch, velocity_consistency_ema_model=ema_model)
```

The EMA model provides stable flow predictions at t+δt for consistency training.
```

---

### [ ] Benchmark Performance vs Diffusion
**Files**: Create `benchmarks/` directory
**Context**: Quantify speedup of flow matching vs discrete diffusion

**Implementation**:
- Compare sampling speed (flow matching should be faster with fewer ODE steps)
- Measure FID scores on MNIST/CIFAR10
- Track training convergence speed
- Document findings in `benchmarks/results.md`

---

## Code Quality

### [ ] Add Pre-commit Hooks
**Files**: `.pre-commit-config.yaml` (create new)
**Context**: Enforce code quality automatically

**Add**:
- `ruff` formatting/linting
- `pytest` on changed test files
- Type checking with `beartype` (already in code)

---

### [ ] Refactor Attention Mask Logic
**Files**: `transfusion_pytorch/transfusion.py` (lines ~336-360)
**Context**: `transfusion_attn_mask()` is complex, could be more modular

**Implementation Notes**:
- Extract causal mask logic into separate function
- Extract modality mask logic into separate function
- Add unit tests for mask construction
- Add docstrings explaining mask semantics

---

## Nice to Have

### [ ] Support for Audio Modality with Waveform Encoder
**Files**: Create `train_audio.py`, add example audio encoder/decoder
**Context**: Demonstrate 1D modality usage beyond embeddings

**Dataset**: LibriSpeech or similar
**Encoder**: Simple 1D conv encoder
**Task**: Text-to-speech or speech-to-text

---

### [ ] Add Wandb Logging Integration
**Files**: All `train_*.py` scripts
**Context**: Better experiment tracking than print statements

**Add**:
- `wandb.init()` with config logging
- Log loss breakdown (text, flow, velocity, recon)
- Log generated samples periodically
- Track gradient norms and learning rate

---

### [ ] Implement Multi-GPU Training (DDP)
**Files**: Create `train_ddp.py` example
**Context**: Scale training to multiple GPUs

**Implementation Notes**:
- Use `torch.nn.parallel.DistributedDataParallel`
- Handle EMA model synchronization across ranks
- Update all training scripts with DDP support flag
- Test on 2-GPU setup minimum

---

## Notes for AI Agents

When implementing TODOs:
1. **Always add tests** - Every feature needs a test in `tests/test_transfusion.py`
2. **Follow existing patterns** - Look at similar code before implementing
3. **Update documentation** - README and copilot-instructions.md should reflect changes
4. **Run full test suite** - `pytest tests/` before submitting
5. **Check type annotations** - Set `TYPECHECK=1` and verify no errors
6. **Preserve backward compatibility** - Add parameters with sensible defaults

Use `grep_search` to find related code patterns before implementing new features.
