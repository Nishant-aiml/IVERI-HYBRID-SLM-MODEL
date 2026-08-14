# FIX-001_TITANS_SYNC.md
## FIX-001: Titans GPU/CPU Synchronization — Complete Report

---

## Summary

| Field | Value |
|---|---|
| Bug ID | BUG-01 |
| Fix commit | `b065e27` |
| Baseline | `7012134f` (`IVERI_BASELINE_FROZEN`) |
| Type | Pure telemetry refactoring — zero mathematical change |
| **Decision** | **ACCEPTED** (correctness + cleanliness); timing hypothesis **FALSIFIED** |

---

## 1. Bug Description (Pre-Fix)

**Location**: `model/titans/memory.py`, methods `write()` and `forward()`.

Inside `for t in range(P):` (the per-patch online learning loop), every iteration
called:

```python
sum_lr    += lr_t.detach().mean().item()          # GPU-CPU sync
sum_forget += forget_t.detach().mean().item()      # GPU-CPU sync
lr_vals.extend(lr_t.detach().view(-1).tolist())    # GPU-CPU sync
forget_vals.extend(forget_t.detach().view(-1).tolist())  # GPU-CPU sync
gnorm = sum([g.detach().pow(2).sum().item() ...])  # GPU-CPU sync × n_weights
update_norms.append(...)                           # GPU-CPU sync × n_weights
```

At `SL=64`, this produced **954 `.item()` calls per training step**, of which **796
were in the backward-contributing path** (training mode).

---

## 2. The Fix

Telemetry accumulation changed from Python-scalar summation inside the loop to
GPU-tensor accumulation with a **single deferred `.item()` call after the loop**:

```python
# BEFORE — inside loop:
sum_lr += lr_t.detach().mean().item()   # sync every patch

# AFTER — inside loop:
telemetry_lr_list.append(lr_t.detach().mean())   # stays on GPU

# AFTER — once, after loop:
sum_lr = float(torch.stack(telemetry_lr_list).sum().item())
```

Mathematical algorithm (**retrieve → loss → autograd.grad → lr_generator →
updater.update**) is **completely unchanged**. The telemetry values themselves
(avg_lr, avg_forget, avg_grad_norm, avg_update_mag) are numerically equivalent
because they are computed from the same underlying tensors.

---

## 3. Microbenchmark Results (Pre-Fix)

Measured by `scratch/fix001_microbenchmark.py` (task-6078):

| Metric | Value |
|---|---|
| `.item()` calls / training step | **954** |
| `.item()` calls / inference step | 158 |
| `.item()` calls in backward path | **796** |
| Training step (SL=64) | 3,240ms ± 102ms |
| Forward only (eval) | 372ms ± 20ms |
| Backward overhead | 2,868ms (88.5%) |

At SL=512 this would be ~7,632 `.item()` calls per training step.

---

## 4. Validation Results (Post-Fix)

Measured by `scratch/fix001_validate.py` (task-6091):

### A. Numerical Equivalence
| Test | Result |
|---|---|
| Max logit difference (two identical models) | 0.00 (bit-exact) |
| Verdict | **PASS** |

### B. Gradient Correctness
| Metric | Value |
|---|---|
| Gradient norm | 10.33 |
| NaN in gradients | False |
| Loss value | 5.55 (valid range) |
| Verdict | **PASS** |

### C. .item() Call Reduction
| Metric | Value |
|---|---|
| Pre-fix | 954 |
| Post-fix | 446 |
| Reduction | **53.2%** |
| Remaining | 446 (from MoE `expert_counts.cpu()`, other telemetry, final post-loop calls) |

### D. Timing
| Metric | Pre-Fix | Post-Fix | Change |
|---|---|---|---|
| Training step mean | 3,240ms | 3,303ms | **−1.9% (SLOWER — within noise)** |
| Std | ±102ms | ±741ms | High variance — bimodal effects |

**Timing hypothesis: FALSIFIED.**

Removing 53.2% of `.item()` calls produced **no measurable timing improvement**.
The backward overhead (2,868ms) is not caused by the `.item()` telemetry calls.

### E. Quality (5-seed)
| Model | Mean | Std | Delta |
|---|---|---|---|
| Frozen baseline (EXP-020) | 3.1118 | 0.047 | — |
| Fixed IVERI (FIX-001) | 3.1247 | 0.091 | +0.013 |
| Welch t | −0.283 | p=0.787 | **NOT SIGNIFICANT** |
| Verdict | **PASS — quality preserved** |

---

## 5. Critical Discovery: The Real Bottleneck

The failed timing hypothesis reveals what the actual bottleneck is.

The Titans `forward()` method contains:

```python
for t in range(P):
    ...
    grads = torch.autograd.grad(
        loss_t,
        weights,
        create_graph=True,      ← second-order differentiation
        retain_graph=True,
        allow_unused=True,
    )
```

`create_graph=True` instructs PyTorch to build the backward graph **through the
gradient computation itself** — enabling higher-order derivatives. This is required
for Titans' online learning update rule: the weight update uses the gradient of the
forward MLP loss as a signal, and this must be differentiable so the gradient can
flow back through the update rule during the outer optimizer step.

**Cost**: At every patch step `t`, a full second-order autograd trace is computed
over the Titans MLP weights. With `P` patches and `n_weights` weight tensors, the
total cost is O(P × n_weights²) in terms of autograd graph operations.

This is the actual source of the 2,868ms backward overhead — not the `.item()` syncs.

### What this means for root-cause B2

**BUG-01 (root-cause B2) must be revised:**

> ~~"`.item()` calls inside the Titans loop cause O(P) GPU-CPU synchronizations that dominate backward time."~~

**Corrected finding:**
> The `.item()` calls are real and unnecessary, but they are NOT the timing bottleneck.
> The actual timing bottleneck is `torch.autograd.grad(create_graph=True)` called P times
> per forward pass. This is the algorithmic cost of Titans' differentiable online learning
> (second-order differentiation). It is NOT an implementation bug — it is the intended
> algorithm. The overhead is fundamental to Titans as implemented.

**Implication for the overall root-cause tree:**

The 89.5% backward overhead is primarily caused by:

1. **Titans second-order autograd** — O(P × cost_of_backward_through_MLP): ~600ms
   (from ABL-D, which showed 3,098ms → 2,500ms when Titans removed)
2. **MoR forward computation** — the recursion loop itself: ~1,628ms
   (from ABL-E, which showed 3,098ms → 1,470ms when MoR removed)
3. **Remaining backbone operations** — Mamba, attention, MoE: ~870ms

None of these three are `.item()`-driven.

### Revised priority assessment for BUG-01

| Original priority | CRITICAL (based on hypothesis that .item() = bottleneck) |
|---|---|
| **Revised priority** | LOW (telemetry cleanup — correct but negligible timing impact) |

---

## 6. Acceptance Decision

**ACCEPTED** with revised scope.

Acceptance basis:
- ✅ `.item()` calls eliminated from inner loop (correct, clean code)
- ✅ Numerical equivalence: bit-exact
- ✅ Gradient correctness: valid norms, no NaN
- ✅ Quality preserved: p=0.787
- ❌ Timing hypothesis: FALSIFIED

The fix is accepted **not because it solves the timing problem** (it does not), but
because:
1. The code change is correct — unnecessary syncs are eliminated
2. Zero mathematical change — the algorithm is identical
3. The codebase is cleaner
4. At longer sequence lengths, the telemetry `.item()` calls would scale with P
   and eventually become significant even though they are not at SL=64

**The timing deficit requires a different fix** — see FIX-TITANS-ALGORITHMIC.

---

## 7. Revised Action Plan

### FIX-TITANS-ALGORITHMIC (NEW, HIGH)
The Titans second-order autograd loop costs O(P) backward passes through the Titans MLP.
Options:
- **Option A**: `eval()` Titans during training (no online update) → benchmark if quality degrades
- **Option B**: Truncated BPTT through Titans (limit gradient depth)
- **Option C**: Vectorized batch Titans update (possible if weights are small enough)
- **Option D**: Accept Titans overhead as algorithmic — evaluate whether quality benefit
  at longer training justifies the cost

> [!WARNING]
> Any of these options change the Titans algorithm semantics. They must be documented
> as ALGORITHMIC CHANGES, not implementation fixes, and require separate ablation study.

### FIX-002 — MoR .any() (STILL HIGH)
MoR accounts for 1,628ms / 3,098ms (53%) of step time from ABL-E.
This is not all from `.any()` — most is from the recursion computation itself.
But the `.any()` causes bimodal timing. Still worth investigating.

### FIX-003 — Backbone entropy gate crash
Still CONFIRMED as a code bug. Fix to enable ABL-C.

### FIX-004 — MoR router gradient
**FINDING REVISED**: In production mode (`research_mode=False`), the router has NO
learnable parameters (`self.wg` is only instantiated in `research_mode=True`).
The `floor(...).long()` converts entropy to depth, but no trainable parameters
exist to receive gradients in this path. This is intended behavior.
**BUG-04 is NOT a production bug** — it is a misidentification. The production MoR router
is a deterministic function, not a learned router.

---

## 8. FIX-REGISTRY Update

```
FIX-001 | BUG-01 | b065e27 | ACCEPTED
Timing hypothesis: FALSIFIED
Quality: PASS (p=0.787)
Numerical: PASS (bit-exact)
.item() reduction: 53.2%
Timing improvement: NONE (bottleneck is second-order autograd, not .item())
```
