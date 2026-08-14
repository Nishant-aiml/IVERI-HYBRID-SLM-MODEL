# FIX-002_MOR_SYNC.md
## FIX-002: MoR GPU/CPU Synchronization — Report

---

## Summary

| Field | Value |
|---|---|
| Bug ID | BUG-02 |
| Fix commit | `b5889f1` |
| Baseline | `7012134f` (`IVERI_BASELINE_FROZEN`) |
| Type | Compute optimization removal — zero mathematical change |
| **Decision** | **PENDING** — awaiting validation results (task-6126) |

---

## 1. Bug Description

**Location**: `model/mor/recursion.py:171`

```python
for step in range(self.max_depth):
    active_mask = depths > step   # (B, P)

    if not active_mask.any():     # ← GPU→CPU sync here
        break                     # ← variable execution depth
```

`active_mask.any()` evaluates a GPU tensor to a Python boolean. This requires a
CUDA synchronization every recursion step. With `max_depth=4`, this produces at
most 4 GPU-CPU syncs per forward pass. This is far fewer than Titans BUG-01
(796 syncs per backward), but creates a different problem: **bimodal latency**.

When all patches reach their assigned depth early (e.g., max depth = 1), the loop
exits at step 0 after 1 sync. When patches have high depth (e.g., max depth = 4),
all 4 steps execute. This creates two distinct execution-time clusters, producing
the bimodal latency distribution observed in EXP-021 (CV=46%).

---

## 2. Mathematical Argument for Safe Removal

**Key fact**: `torch.where(active_mask.unsqueeze(-1), x_step, x)` at line 183 already
handles inactive patches correctly without the `break`:
- When `active_mask[b, p] = False`: `torch.where` selects `x[b, p]` unchanged
- When `active_mask` is all-False: `x_step` is computed but **every element is
  discarded** by `torch.where` — the output is identical to `x`

Therefore, removing the `break` produces **bit-exact identical outputs** for all inputs.
The only cost is that `alpha_t`, `norm_x`, and `x_next` are computed for steps where
all `active_mask` values are False — pure wasted compute, not a semantic change.

At `max_depth=4`, the maximum wasted iterations after early termination is 3
(if all patches have depth=1 and the loop runs 4 times instead of 1). Since
`max_depth` is fixed and small, this overhead is bounded and acceptable.

---

## 3. Fix

```python
# REMOVED:
if not active_mask.any():
    break
```

The comment in the replacement explains the justification inline in the code.

---

## 4. Expected Outcomes

### Bimodality
**Expected ELIMINATED**: With fixed `max_depth=4` iterations, every forward pass
executes the same computation regardless of input entropy. The bimodal latency
distribution should collapse to a unimodal distribution.

### Timing
**Expected: small improvement or neutral.** The `.any()` is only 1 GPU-CPU sync
per recursion step (up to 4 per forward pass), not per patch. This is far fewer
than the Titans BUG-01 syncs (P per step). Given FIX-001 showed no improvement
from 796 syncs, removing 4 syncs is unlikely to produce a large timing change.
Any improvement will come from the forward pass only — backward is dominated by
second-order Titans autograd.

---

## 5. Measured Results

**Pending task-6126.** Will be updated with:
- Forward latency: pre vs post CV (bimodality measure)
- Training step: pre vs post
- Quality: 5-seed vs baseline
- Acceptance decision

---

## 6. _update_telemetry note

`_update_telemetry(depths, ...)` at line 156 performs `depths.detach().cpu()` before
the loop — this is a single GPU-CPU transfer, called once per forward, not in the hot
path of the loop. This is acceptable telemetry overhead and is NOT changed by FIX-002.
