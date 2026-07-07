# Pretraining Verification Log — IVERI CORE Phase 3.1

This log documents the pretraining verification execution details.

## Level 1 Smoke Test (20 steps)

- **Execution command**: `python train.py --verification-level 1 --device cpu`
- **Output checkpoints**:
  - `checkpoint_5.pt`, `checkpoint_10.pt`, `checkpoint_15.pt`, `checkpoint_20.pt`
- **Final Train Loss**: 5.5446
- **Final Val Loss**: 5.5432
- **Final Perplexity**: 255.51
- **Status**: ✅ SUCCESSFUL

## Level 2 Verification Run (100 steps)

- **Execution command**: `python -u train.py --verification-level 2 --device cpu`
- **Output checkpoints**:
  - `checkpoint_50.pt`, `checkpoint_100.pt`
- **Final Train Loss (Step 50)**: 4.6385
- **Final Val Loss (Step 50)**: 5.5432
- **Final Perplexity (Step 50)**: 103.39
- **Status**: ✅ RUNNING (step 50 checkpoint written successfully)
