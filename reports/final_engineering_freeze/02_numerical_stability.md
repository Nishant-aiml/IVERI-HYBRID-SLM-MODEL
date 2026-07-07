# Numerical Stability Report

**Audit Date:** 2026-07-07  
**Auditor:** Independent AI Architect (Antigravity)  
**Protocol:** Phase 6.4 — Final Engineering Freeze Audit

---

## 1. Forward Pass Stability

| Test | Result | Details | Status |
|------|--------|---------|--------|
| NaN in logits | None detected | 5 consecutive forward passes | PASS |
| Inf in logits | None detected | 5 consecutive forward passes | PASS |
| Loss is finite | All 5 steps finite | `[5.5605, 5.5699, 5.5635, 5.5799, 5.5781]` | PASS |
| Empty sequence | No crash | Handles gracefully | PASS |
| Single byte | No crash | Handles gracefully | PASS |

---

## 2. Gradient Health

| Test | Result | Details | Status |
|------|--------|---------|--------|
| Gradient norms bounded | Yes | `[18.69, 13.13, 10.83, 11.07, 7.68]` | PASS |
| Gradient norm decreasing | Trend downward | 18.69 → 7.68 over 5 steps | PASS |
| Backward pass completes | Yes | `loss=-0.0021` | PASS |

---

## 3. Mamba2 A_log Stability

The `A_log` parameter in Mamba2 blocks must satisfy `A < 0` (i.e., the exponential of A_log must be negative for stable state-space dynamics).

| Check | Value | Status |
|-------|-------|--------|
| `A_log` min | -147.7420 | PASS (< 0) |
| `A_log` max | -2.7249 | PASS (< 0) |

**Verdict:** All A_log values are strictly negative — SSM dynamics are stable.

---

## 4. Weight Norms

| Check | Value | Status |
|-------|-------|--------|
| Max weight norm | 73.29 | PASS (bounded) |
| Max weight location | `backbone.blocks.0.sub_block.mamba_blocks.0.A_log` | Expected |

---

## 5. Mixed Precision (AMP FP16)

| Test | Result | Status |
|------|--------|--------|
| AMP FP16 forward | No NaN | PASS |
| FutureWarning | `torch.cuda.amp.autocast` deprecated | LOW (cosmetic) |

> **Note:** The `torch.cuda.amp.autocast(dtype=torch.float16)` call should be migrated to `torch.amp.autocast('cuda', dtype=torch.float16)` to suppress the FutureWarning.

---

## 6. Memory Leak Test

| Metric | Value | Status |
|--------|-------|--------|
| Delta over 20 iterations | 0.3 MB | PASS (< 10 MB threshold) |
| Gradient accumulation | `total_loss=-0.0019` | PASS |

---

## Overall Numerical Stability Verdict: **PASS**
