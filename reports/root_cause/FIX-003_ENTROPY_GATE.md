# FIX-003_ENTROPY_GATE.md
## FIX-003: Backbone Entropy-Gate Bug — Report

---

## Summary

| Field | Value |
|---|---|
| Bug ID | BUG-05 |
| Fix commit | `b5889f1` |
| Baseline | `7012134f` (`IVERI_BASELINE_FROZEN`) |
| Type | Code-path bug fix — restores intended entropy-gate behavior |
| **Decision** | **PENDING** — awaiting ABL-C result (task-6126) |

---

## 1. Bug Description

**Location**: `model/backbone.py:116`

```python
# BEFORE (buggy):
if self.use_moe and is_complex:
    ... MoE experts ...
else:
    moe_out = self.dense_ffn(norm_x)   # ← CRASH when use_moe=True
x = x + moe_out
```

When `use_moe=True`, `self.dense_ffn` is **never created** (line 68 in `__init__`
only creates it in the `else` branch of `if self.use_moe`). So the `else` branch above
always crashes with `AttributeError: 'BackboneSubBlock' object has no attribute 'dense_ffn'`
when `use_moe=True` and `is_complex=False`.

`is_complex` is `False` when `entropy.mean() < 0.20`. This occurs reliably when
`use_blt=False` (producing zero-entropy patches), and may also occur during
normal training for low-entropy patches.

---

## 2. Establishing Intended Behavior

From the architecture documentation and code context:

1. **`use_blt=False` crash** (confirmed): With no BLT, all patches have `patch_entropy=0`
   → `is_complex = (0.0 >= 0.20) = False` → crash. This definitively shows the `else`
   branch was never tested with `use_moe=True`.

2. **Entropy gate design intent**: The `is_complex` check at line 100 also gates the
   attention block:
   ```python
   if is_complex:
       x = x + self.attention(...)
   ```
   This is the documented "Option C entropy gating" — high-entropy patches go through
   the full computation path (attention + MoE), low-entropy patches skip both.

3. **Correct `else` behavior**: When `is_complex=False`, the FFN should be **skipped
   entirely** (pass through unchanged), not replaced with `dense_ffn`. The entropy gate
   is designed to reduce computation for low-complexity inputs.

4. **`dense_ffn` is the ablation path**: It only exists when `use_moe=False`
   (ablation baseline mode), as confirmed by `__init__` lines 64-68.

---

## 3. Fix

```python
# AFTER (fixed):
norm_x = self.moe_norm(x)
if is_complex:
    if self.use_moe:
        dispatch_weights, dispatch_indices, aux_loss = self.moe_router(...)
        ...
        x = x + moe_out
    else:
        # Ablation path (use_moe=False): use dense FFN
        if use_ckpt:
            moe_out = checkpoint(self.dense_ffn, ...)
        else:
            moe_out = self.dense_ffn(norm_x)
        x = x + moe_out
# else: is_complex=False — entropy gate skips FFN entirely; x passes through unchanged.
```

The fix separates the `is_complex` entropy gate from the `use_moe` config flag.
When `is_complex=False`:
- The FFN (whether MoE or dense) is **not called**
- `x` passes through unchanged (correct entropy-gate behavior)
- No `AttributeError` possible

When `is_complex=True`:
- If `use_moe=True`: MoE experts are invoked (production path)
- If `use_moe=False`: `dense_ffn` is invoked (ablation path)

---

## 4. Impact on Training Semantics

### Production training (use_blt=True, use_moe=True)
Previously: low-entropy patches triggered the buggy `else` branch → crash.
In practice, this means the training must always have had high enough mean entropy
(`>= 0.20`) to avoid the crash during normal training runs. The bug was latent.

After fix: low-entropy patches skip FFN (intended). No semantic change for
patches that already had `is_complex=True`.

**Potential training behavior change**: If any low-entropy patches occurred during
EXP-020 training, they previously crashed (or were avoided somehow). After fix,
they cleanly skip the FFN. This may slightly change the quality distribution.

### Ablation (use_blt=False)
Previously: always crashes (entropy=0 → is_complex=False).
After fix: runs correctly, skipping the FFN for all patches.
**ABL-C is now enabled.**

---

## 5. Measured Results

**Pending task-6126.** Will be updated with:
- ABL-C (no BLT) quality result
- Comparison to full IVERI
- Acceptance decision

---

## 6. Secondary Effect: ABL-C Unblocked

FIX-003 directly enables ABL-C — running IVERI without BLT to isolate the BLT
contribution from the Mamba backbone contribution. This is the experiment needed
to answer the most important remaining research question:

> **Is the quality deficit (IVERI 3.112 vs Transformer 2.705) caused by BLT, or
> by the Mamba-2 backbone?**

The ABL-C result in task-6126 will determine this.
