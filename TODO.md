# Transfusion-PyTorch TODO List

## High Priority Features

### [ ] Add ability to sample (multi-)modality only samples
**Files**: `transfusion_pytorch/transfusion.py`
**Context**: Currently only supports sampling if text modality is present; does not support sampling multiple modalities without text padding

**Implementation Notes**:
- Add example in `train_modalities_only.py` using a simple multimodal dataset (without text)

**References**:
