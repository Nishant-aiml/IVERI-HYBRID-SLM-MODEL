# Final Repository Status Audit — Integration Validation

## End-to-End Data Flow

```
Data Pipeline          → Verified in tests (test_data_pipeline.py: 34 passed)
  ↓
Byte Encoding          → Verified: text_to_byte_ids() produces collision-free IDs
  ↓
BLT Entropy Model      → Verified: entropy_model(raw_bytes) → (B, S, 1)
  ↓
Dynamic Patcher        → Verified: compute_boundaries() → boundary_map (B, S) bool
  ↓
BLT Encoder            → Verified: encode_with_boundaries() → latent_patches (B, P, D)
  ↓
Patch Entropy          → Verified: boundary-aggregated entropy → (B, P, 1)
  ↓
Titans Memory          → Verified: forward_with_injection() → (B, P, D)
  ↓
Backbone Blocks × L    → Verified: MoR → Mamba2×6 → Attention → MoE → RMSNorm
  ↓
BLT Decoder            → Verified: decode_with_boundaries() → logits (B, S, 259)
  ↓
Loss (Cross-Entropy)   → Verified: loss.backward() produces gradients
  ↓
Optimizer (AdamW)      → Verified: step() updates parameters
  ↓
Checkpoint             → Verified: save/load bitwise identical
  ↓
Evaluation             → ⚠️ IMPLEMENTED but never run with real data
  ↓
Replay                 → ⚠️ IMPLEMENTED but fails on legacy DB (by design)
  ↓
Publication            → ❌ No publication artifacts with real data
  ↓
Inference              → ⚠️ Works mechanically but produces random bytes (untrained)
```

## Integration Issues Found

1. **Duplicate Dataloaders**: `data/dataloader.py` AND `data/pipeline/dataloader.py` coexist with overlapping functionality. Which one is canonical is unclear.

2. **SFT Loss Masking Gap**: `SFTByteDataset` in `data/pipeline/dataloader.py` returns `(x, y)` tuples. The SFT runner expects optional `(x, y, loss_mask)`. When loss_mask is absent, the runner defaults to `loss_mask = ones`, meaning **prompt tokens are never actually masked** during SFT training. This defeats the purpose of SFT.

3. **4 Duplicated Training Loops**: `pretrain_runner.py`, `sft_runner.py`, `coding_runner.py`, and `preference_runner.py` each implement their own forward/backward/optimizer step logic. The `Trainer` class is created but its `train_epoch()` method is bypassed.

4. **Train.py only supports pretraining**: The main CLI entry point `train.py` calls `run_pretraining()`. No CLI exists for SFT, coding, or preference stages.

## Verdict

**Integration is structurally complete but never end-to-end tested with real data.** All components connect mathematically. The critical gap is that the system has never processed a real dataset through the full pipeline and produced a trained model.
