# Negative Result NEG-002: Pure FP16 Precision Without GradScaler Underflow

**Experiment ID:** `EXP-0004-v1`  
**Finding:** Unscaled FP16 execution produced zero-norm gradients and `NaN` loss at Step 20 due to IEEE 754 half-precision underflow.  
**Resolution:** Integrated `torch.amp.GradScaler('cuda')` and `clip_grad_norm_(max_norm=1.0)`, resulting in 100% stable pretraining.
