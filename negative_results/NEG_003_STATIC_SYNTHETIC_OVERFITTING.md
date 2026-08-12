# Negative Result NEG-003: Static Synthetic Batch Overfitting

**Experiment ID:** `EXP-0005-v1`  
**Finding:** Instantiating synthetic input/target tensors once outside the training loop caused the model to memorize the fixed 4,096-token tensor, driving loss down to `0.0049` at Step 1,500. Re-instantiating fresh random tensors on script restart caused an abrupt loss jump to `17.14`.

**Impact:**
- Hardware characterization (1,232.3 tok/s FP16, 1,722.6 MB VRAM, 0 OOMs) is 100% unaffected and empirically valid.
- Proves the necessity of per-step dynamic token batching (`get_next_batch()`) for true cross-entropy loss convergence metrics.
