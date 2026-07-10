# Final Repository Status Audit — Risk Assessment

## Critical Risks (Blocking)

### RISK-1: No Training Validation 🔴
**Impact**: Cannot verify that the architecture learns anything  
**Description**: The entire architecture, data pipeline, training loop, evaluation framework, and research infrastructure exist — but the model has never been trained on real data. The fundamental question "Does IVERI's BLT+Titans+Mamba2+MoR+MoE architecture actually learn?" remains unanswered.  
**Mitigation**: Execute a 1000-step pretraining run on TinyStories and verify loss convergence before any further development.

### RISK-2: Runtime Crash Bugs 🔴
**Impact**: First real training run will crash  
**Description**: Two confirmed crash bugs exist:
1. `pretrain_runner.py:386` — references `nn.Module` but `nn` is not imported
2. `sft_runner.py:264,395` — invalid format string `%.2%` will crash Python's string formatter  
**Mitigation**: Fix both bugs before first training run.

### RISK-3: SFT Loss Masking Broken 🔴
**Impact**: SFT training will train on prompt tokens, defeating its purpose  
**Description**: `SFTByteDataset` does not return loss masks. The SFT runner defaults to `loss_mask = ones`, meaning all tokens (including prompt tokens) contribute to loss.  
**Mitigation**: Implement loss mask generation in `SFTByteDataset`.

## High Risks

### RISK-4: Parameter Count Mismatch 🟡
**Impact**: Hardware planning and comparison claims affected  
**Description**: Default config produces 36.6M parameters, not 10M as spec states for "nano".  
**Mitigation**: Either adjust config to produce 10M or update spec to reflect actual parameter count.

### RISK-5: No KV-Cache in Inference 🟡
**Impact**: Inference will be extremely slow for real use  
**Description**: The inference engine re-processes the entire context at every generation step. No KV-cache is implemented.  
**Mitigation**: Implement KV-cache for Mamba2 state and attention caches.

### RISK-6: Missing Mamba Baseline 🟡
**Impact**: Cannot make comparative architecture claims  
**Description**: Only a transformer baseline exists. The spec requires both transformer AND Mamba baselines.  
**Mitigation**: Implement `baselines/tiny_mamba.py`.

### RISK-7: Misleading requirements.txt 🟡
**Impact**: Installation failures and confusion  
**Description**: 6 of 15 listed packages (mamba-ssm, flash-attn, rotary-emb, bitsandbytes, triton, tokenizers) are not actually used.  
**Mitigation**: Remove unused packages from requirements.txt.

## Medium Risks

### RISK-8: Duplicated Training Loops 🟡
**Impact**: Maintenance burden, bugs must be fixed in 4 places  
**Description**: 4 independent training loops exist (Trainer + 3 runners), each implementing their own forward/backward/optimizer step.  
**Mitigation**: Refactor runners to delegate to `Trainer.train_epoch()`.

### RISK-9: Duplicate Dataloaders 🟡
**Impact**: Confusion about which dataloader is canonical  
**Description**: `data/dataloader.py` and `data/pipeline/dataloader.py` have overlapping functionality.  
**Mitigation**: Designate one as canonical and deprecate the other.

### RISK-10: Cross-Platform Risk 🟡
**Impact**: Linux training may behave differently  
**Description**: Developed and tested exclusively on Windows. Custom fallback implementations may produce different results on Linux with native CUDA kernels.  
**Mitigation**: Test on Linux before any production training.
