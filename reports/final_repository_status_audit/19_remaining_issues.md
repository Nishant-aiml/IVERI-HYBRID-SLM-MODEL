# Final Repository Status Audit — Remaining Issues & Action Plan

This document outlines all identified remaining issues, bugs, and gaps in the IVERI Core codebase. It is divided into blocking bugs (which will prevent training or lead to corrupt models) and code quality/architectural gaps.

---

## 1. Blocking Bugs (Immediate Actions Required)

### Issue 1.1: Missing `nn` Import in Pretrain Runner
* **Location**: `training/pretrain_runner.py` (Line ~386, `_assert_finite_tensors` function signature/definition)
* **Symptom**: The function signature references `nn.Module`, but the `nn` module is never imported from `torch` (only `torch` is imported). Running pretraining will crash as soon as this function is called.
* **Fix**: Add `import torch.nn as nn` to imports in `pretrain_runner.py`.

### Issue 1.2: Invalid Format Strings in SFT Runner Logging
* **Location**: `training/sft_runner.py` (Lines 264 & 395)
* **Symptom**: String format calls use `%.2%` or `%.2%%`. In Python's formatting syntax, `%.2%` is invalid and will raise a `ValueError` or `TypeError` during evaluation logging.
* **Fix**: Change to `%.2f%%` to correctly print decimal percentages.

### Issue 1.3: SFT Loss Masking Defect
* **Location**: `data/pipeline/dataloader.py` (`SFTByteDataset`) and `training/sft_runner.py`
* **Symptom**: `SFTByteDataset.__getitem__` returns a tuple of `(x, y)` only. The SFT runner expects a 3-element tuple `(x, y, loss_mask)`. When the dataset does not return a mask, the runner defaults to `loss_mask = torch.ones_like(y)`. As a result, prompt tokens are not masked and contribute to the cross-entropy loss, which is highly detrimental to instruction-following tuning.
* **Fix**: Modify `SFTByteDataset` to compute and return a binary `loss_mask` (0 for prompt bytes, 1 for response bytes).

---

## 2. Architectural & Code Quality Gaps

### Issue 2.1: Quadruplicated Training Loops
* **Location**: `training/pretrain_runner.py`, `training/sft_runner.py`, `training/coding_runner.py`, `training/preference_runner.py`
* **Symptom**: Each runner duplicates the basic forward, backward, optimizer step, gradient clipping, and mixed-precision scaling logic. The `Trainer` class defined in `training/trainer.py` is bypassed. This violates the DRY (Don't Repeat Yourself) principle and makes maintenance difficult.
* **Fix**: Refactor all runners to use `Trainer.train_epoch()` for the step-based training loop.

### Issue 2.2: Duplicate Dataloader Modules
* **Location**: `data/dataloader.py` vs `data/pipeline/dataloader.py`
* **Symptom**: Two overlapping dataloader modules exist in the same repository. `data/dataloader.py` contains `ByteDataset` and `StreamingByteDataset`, while `data/pipeline/dataloader.py` contains `PretrainByteDataset`, `SFTByteDataset`, and `CodingByteDataset`.
* **Fix**: Merge all datasets and dataloader factory functions into a single structured module under `data/dataloader.py` and delete the redundant pipeline copy.

### Issue 2.3: Missing Mamba Baseline
* **Location**: `baselines/` directory
* **Symptom**: The project roadmap requires benchmarking IVERI against both a standard Transformer baseline and a pure Mamba baseline. Only `baseline_transformer.py` is present. No Mamba baseline exists.
* **Fix**: Implement `baselines/baseline_mamba.py` (or `tiny_mamba.py`) wrapping the custom Mamba2 blocks without Attention/MoE.

### Issue 2.4: Absence of KV-Cache in Inference
* **Location**: `inference/engine.py`
* **Symptom**: The generation engine runs the full model forward pass on the entire sequence at every step. This makes generation slow, scaling poorly as sequence length increases.
* **Fix**: Update the model's components and inference engine to support incremental state updating (KV-caching for Attention, recurrent state passing for Mamba2).

---

## 3. Dependency Cleanup

### Issue 3.1: Misleading requirements.txt
* **Location**: `requirements.txt`
* **Symptom**: Multiple heavy CUDA/deep learning packages are listed (`mamba-ssm`, `flash-attn`, `rotary-emb`, `bitsandbytes`, `triton`, `tokenizers`) but are not actually used by the code. The model uses pure PyTorch fallbacks instead.
* **Fix**: Strip unused packages from `requirements.txt` to avoid unnecessary installation compilation failures on unsupported platforms (like Windows).
