# CORRECTED_ROOT_CAUSE_SUMMARY.md
## IVERI Core — Corrected Root-Cause Summary
## Status after FIX-001, FIX-002, FIX-003, BUG-04 investigation

---

> [!IMPORTANT]
> This document supersedes `29_ROOT_CAUSE_TREE.md` and `30_ROOT_CAUSE_RANKING.md`
> for all items that have been corrected by experimental evidence.
> Original documents are preserved (immutable historical record).

---

## Corrected Findings

### BUG-01 → FIX-001: ACCEPTED but timing hypothesis FALSIFIED

**Original claim**: `.item()` calls inside Titans loop cause O(P) GPU-CPU syncs
that drive the 89.5% backward overhead.

**What was found**:
- `.item()` calls confirmed: **954 per training step** (pre-fix), 796 in backward path
- After removing all inner-loop `.item()` calls (−53.2% reduction): **no timing improvement**
- Training step: 3,240ms → 3,303ms (within noise; std jumped to 741ms)

**Corrected root-cause B2**:
The backward overhead is caused by `torch.autograd.grad(create_graph=True)` called
**P times per forward pass** in the Titans loop. This builds a second-order autograd
graph through the Titans MLP at every patch step — required for the differentiable
online learning update. This is the **algorithmic cost of Titans' meta-learning**, not
a telemetry bug.

**What it means**:
- Titans backward cannot be fixed by telemetry refactoring
- Reducing Titans overhead requires **algorithmic change** (truncated BPTT, vectorized
  update, or disabling online learning during training)
- Any such change modifies the Titans algorithm semantics — not a simple bug fix

**FIX-001 status**: ACCEPTED (code is cleaner; `.item()` elimination is correct even
if timing-neutral at this sequence length)

---

### BUG-04 → REJECTED — FALSE POSITIVE

**Original forensic claim**: `torch.floor(...).long()` blocks gradients; router never learns.

**Corrected finding after reading `router.py` in full**:
The production MoR router (`research_mode=False`, the default in all experiments)
has **no trainable parameters at all**. `self.wg` is only instantiated when
`research_mode=True`. The production path at line 87 is a pure deterministic function
(entropy → discrete depth), documented explicitly as "Option C Production Mode."
There are no parameters to receive gradients. Zero gradient flow is correct.

**Decision**: **REJECTED. False positive in the forensic audit.** The audit inferred
a gradient-blocking bug without first verifying that any trainable parameters existed
at the reported location. No code change. No fix entry.

(`research_mode=True`, which does have `self.wg`, is not used in any experiment to date.)

---

### BUG-02 → FIX-002: APPLIED, PARTIAL RESULTS

**Original claim**: `active_mask.any()` forces GPU-CPU sync per recursion step;
53% of training step time is from MoR (ABL-E ablation).

**Fix applied**: Removed `if not active_mask.any(): break` from recursion loop.
`torch.where` already handles inactive patches correctly. Execution depth is now
fixed at `max_depth` — output is bit-exact identical.

**Partial measured results (task-6126)**:
- Forward NaN/Inf: PASS (no NaN, no Inf)
- Eval forward: 372ms → 364ms (−2.1%, within noise)
- Training step: 3,240ms → 3,203ms (−1.1%, within noise)
- CV (eval forward): 5.4% → 6.5% — **bimodality not present in either pre or post-fix eval mode**
- Bimodality (CV>15%): False both before and after

**Clarification on expected impact**: FIX-002 eliminates 4 GPU-CPU syncs per forward
pass (one per recursion step). Given FIX-001 showed no improvement from 796 syncs,
4 additional syncs were never going to produce a large speedup. The primary benefit
is deterministic execution depth, not mean latency reduction. The bimodal pattern
was not present in eval-mode measurements; it may only manifest at training resolution
or at longer sequence lengths. Quality result pending.

---

### BUG-05 → FIX-003: APPLIED, ABL-C RUNNING

**Original finding**: CONFIRMED. `backbone.py:116` crashes when `use_moe=True` and
`entropy < 0.20` (always with `use_blt=False`).

**Fix**: Separated `is_complex` entropy gate from `use_moe` flag. When `is_complex=False`,
FFN is skipped entirely (correct entropy-gate behavior). When `is_complex=True`, MoE or
dense_ffn is used based on `use_moe` flag.

**Measured so far**: `use_blt=False` smoke test **PASS** — no crash, valid logit range.
ABL-C quality result pending (task-6126 still running).

---

### BUG-03: CONFIRMED, NOT YET FIXED

**BLT encoder O(B×S²) dense pooling matrix** (`blt/encoder.py:129`) causes
quadratic memory growth at longer sequences. At SL=64 impact is manageable.
Long-context testing remains blocked until FIX-005 is implemented.

**Status**: PLANNED after ABL-C results are reviewed.

---

## Revised Bottleneck Hierarchy

| Component | Latency contribution | Root cause | Fixable without algorithm change? |
|---|---|---|---|
| Titans backward | ~600ms (from ABL-D) | Second-order autograd × P patches | NO — algorithmic |
| MoR recursion | ~1,628ms (from ABL-E) | Recursion computation at each step | Partial (FIX-002 removes 4 syncs; does not eliminate compute) |
| Mamba+rest | ~870ms remaining | Normal backprop | NO |
| Titans `.item()` | 0ms (FIX-001 confirmed) | Was telemetry; not the bottleneck | DONE (FIX-001) |
| MoR `.any()` | Unknown (<< 1% expected) | 4 GPU-CPU syncs per forward | DONE (FIX-002) |
| BLT encoder | ~5.7ms forward, much more backward | O(S²) matrix | FIX-005 (planned) |

---

## Revised Research Questions

After bugs are fully tested:

### Q1 (was open, now partially answered)
**What causes the quality deficit?**
- Titans: NOT the cause (ABL-D: quality SIMILAR without Titans)
- MoE: NOT the cause (ABL-F: quality IDENTICAL without MoE)
- MoR: NOT the cause (ABL-E: quality SIMILAR without MoR)
- BLT: UNKNOWN — ABL-C now running

### Q2 (was open, now clearer)
**What causes the latency?**
- IVERI training step is ~3,240ms vs Transformer ~36ms (~90x slower)
- Confirmed: Titans second-order autograd = ~600ms = algorithmic, not fixable
- Confirmed: MoR recursion computation = ~1,628ms = dominated by the computation, not the sync
- Confirmed: BLT = only 5.7ms forward (< 2%)
- IVERI is slow primarily because of Titans (second-order) + MoR (deep recursion)

### Q3 (new, from FIX-001 discovery)
**Is Titans' second-order online learning worthwhile?**
ABL-D shows removing Titans: quality SIMILAR (p=0.670), latency SIMILAR without second-order cost.
At current scale/training budget: Titans appears to add overhead with no measurable quality benefit.
Whether this changes with longer training or real data: UNKNOWN.

### Q4 (new)
**Is the MoR recursion compute justified?**
ABL-E shows removing MoR: quality SLIGHTLY WORSE (p=0.386, not significant), saves 53% of step time.
The recursion compute appears to provide marginal quality benefit at this scale.

---

## Current Git History (remediation branch)

```
b5889f1  FIX-002+FIX-003: MoR sync + backbone entropy gate
b065e27  FIX-001-titans-sync: move .item()/.tolist() outside per-patch loop
7012134  [FROZEN BASELINE] Phase 10 + Forensic Audit
```
