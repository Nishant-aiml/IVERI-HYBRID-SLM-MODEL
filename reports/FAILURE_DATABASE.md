# IVERI CORE — Failure Mode & Experiment Discard Registry

| Experiment ID | Failure Cause | Observed Symptom | Root Cause Analysis | Corrective Action | Result |
|---|---|---|---|---|---|
| `EXP-0008` | LR Warmup Instability | Loss Spike @ Step 15 | Learning rate scaled too aggressively before optimizer state stabilization | Added 100-step linear warmup | **RESOLVED** |
| `EXP-0012` | FP16 Overflow | NaN Loss in Mamba scan | Unscaled FP16 state updates in selective scan kernel | Switched to PyTorch Native Autocast FP16/BF16 handler | **RESOLVED** |
