# Final Repository Status Audit — Dependency Audit

## Requirements File Analysis

### Listed in requirements.txt vs Actually Used

| Package | Listed | Actually Used | Verdict |
|---|---|---|---|
| `torch>=2.3.0` | ✅ | ✅ Core framework | ✅ CORRECT |
| `mamba-ssm` | ✅ | ❌ Custom implementation used instead | ⚠️ MISLEADING |
| `flash-attn>=2.5` | ✅ | ❌ PyTorch SDPA used instead | ⚠️ MISLEADING |
| `einops` | ✅ | ✅ Used in model layers | ✅ CORRECT |
| `rotary-emb` | ✅ | ❌ Custom `model/rope.py` used instead | ⚠️ MISLEADING |
| `bitsandbytes` | ✅ | ❌ Never imported | ⚠️ MISLEADING |
| `triton` | ✅ | ❌ No Triton kernels written | ⚠️ MISLEADING |
| `accelerate` | ✅ | ⚠️ Listed but distributed module uses custom code | ⚠️ UNCLEAR |
| `transformers` | ✅ | ⚠️ May be used for dataset loading utilities | ⚠️ UNCLEAR |
| `datasets` | ✅ | ✅ Used in data pipeline (HuggingFace loader) | ✅ CORRECT |
| `tokenizers` | ✅ | ❌ Not used (byte-level, no tokenizer) | ⚠️ MISLEADING |
| `wandb` | ✅ | ✅ Used in logging infrastructure | ✅ CORRECT |
| `lm-eval` | ✅ | ⚠️ Referenced in evaluation but never called | ⚠️ UNCLEAR |
| `numpy` | ✅ | ✅ Used throughout | ✅ CORRECT |
| `tqdm` | ✅ | ✅ Used in training loops | ✅ CORRECT |

### Summary
- **6 of 15 listed packages are NOT actually used** (mamba-ssm, flash-attn, rotary-emb, bitsandbytes, triton, tokenizers)
- The requirements.txt was copied from the master spec and never updated to reflect the actual implementation

## Circular Import Check

No circular imports detected. The import hierarchy is clean:
```
core/ → (no model/ imports)
configs/ → core/
model/ → configs/, core/
training/ → model/, configs/, core/
inference/ → model/, core/
evaluation/ → model/, training/
research/ → training/, evaluation/, model/, configs/
```

## Dead Code / Orphan Files

| File | Status |
|---|---|
| `test_imports.py` (root) | Test fixture, not dead code |
| `graphify.py` (root) | One-off script, dead |
| `scratch_diag.py` (root) | One-off script, dead |
| `debug_run.py` (root) | One-off script, dead |
| `research/campaign_lock.json` | Generated lock file, can be deleted |

## Verdict

**The requirements.txt needs to be cleaned up.** 40% of listed packages are not actually used. This creates confusion about dependencies and makes installation harder (some of these packages like `mamba-ssm` and `flash-attn` are notoriously difficult to install).
