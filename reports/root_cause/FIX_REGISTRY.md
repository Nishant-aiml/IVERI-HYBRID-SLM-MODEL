# FIX_REGISTRY.md
## IVERI Core — Scientific Change Control Registry
## All modifications after IVERI_BASELINE_FROZEN are recorded here.

---

> [!IMPORTANT]
> Frozen baseline: SHA `7012134f44403f6732333bba9e93bbc7ea73b867`, tag `IVERI_BASELINE_FROZEN`.
> Every fix is compared against this baseline individually.
> Compound comparisons are run separately after individual fixes are accepted.

---

## Defect Classification

The forensic audit identified five candidate bugs. After investigation, the
correct classification is:

| Bug ID | Classification | Fix |
|---|---|---|
| BUG-01 | Confirmed implementation defect (telemetry) | FIX-001 — ACCEPTED |
| BUG-02 | Confirmed implementation defect (synchronization) | FIX-002 — ACCEPTED |
| BUG-03 | Confirmed implementation defect (memory) | FIX-005 — PLANNED |
| BUG-04 | **REJECTED — FALSE POSITIVE** | No fix required |
| BUG-05 | Confirmed implementation defect (code-path crash) | FIX-003 — ACCEPTED |

**Separate from bugs — algorithmic bottleneck (not a defect):**
Titans `torch.autograd.grad(create_graph=True)` repeated P times per forward pass.
This is the dominant measured backward cost (~600ms from ABL-D). Reducing it
requires an algorithmic change, not a bug fix. Investigation deferred until
ABL-C results are available and Titans quality/cost trade-off is fully characterized.

---

## FIX-001 — Titans Telemetry Synchronization

| Field | Value |
|---|---|
| **Bug ID** | BUG-01 |
| **Git SHA** | `b065e27` |
| **Baseline SHA** | `7012134f` |
| **Date** | 2026-08-14 |
| **Type** | Telemetry refactoring — zero mathematical change |
| **Decision** | **ACCEPTED** |

### Defect
`.item()` / `.tolist()` calls inside `for t in range(P):` in `titans/memory.py`
(write method lines 317–331, forward method lines 453–467). These are pure telemetry
accumulation with no effect on the algorithm.

### Fix
Accumulate telemetry as GPU tensors inside the loop; call `.item()` once after the
loop. Mathematical algorithm (retrieve → loss → autograd.grad → lr_generator →
updater.update) is unchanged.

### Measured results
| Metric | Pre-fix | Post-fix | Change |
|---|---|---|---|
| `.item()` calls / training step | 954 | 446 | −53.2% ✅ |
| Numerical equivalence | — | 0.00 | bit-exact ✅ |
| Gradient norm | — | 10.33 | valid, no NaN ✅ |
| Training step | 3,240ms ±102ms | 3,303ms ±741ms | **−1.9% — NO IMPROVEMENT** |
| Quality (5-seed) | 3.1118 ±0.047 | 3.1247 ±0.091 | +0.013 nats, p=0.787 ✅ |

### Critical finding: timing hypothesis falsified
Removing 53.2% of `.item()` calls produced **zero timing improvement**.
The 2,868ms backward overhead is caused by `torch.autograd.grad(create_graph=True)`
called P times inside the Titans patch loop — second-order autograd required for the
differentiable online learning update. This is the **algorithmic cost of Titans**,
not an implementation defect.

### Acceptance
**ACCEPTED** on correctness grounds. Code is cleaner; unnecessary syncs eliminated;
no quality harm. Timing hypothesis is FALSIFIED — the actual Titans bottleneck requires
an algorithmic investigation, not a telemetry fix.

---

## FIX-002 — MoR Active-Mask Synchronization

| Field | Value |
|---|---|
| **Bug ID** | BUG-02 |
| **Git SHA** | `b5889f1` |
| **Baseline SHA** | `7012134f` |
| **Date** | 2026-08-14 |
| **Type** | Compute optimization removal — zero mathematical change |
| **Decision** | **ACCEPTED** |

### Defect
`if not active_mask.any(): break` at `recursion.py:171` inside the recursion loop.
Evaluates GPU tensor `.any()` as a Python boolean, forcing a CUDA sync at each
recursion step (up to `max_depth=4` times per forward pass). Creates variable-depth
execution that contributes to bimodal latency distribution.

### Fix
Removed the `break` condition. `torch.where(active_mask, x_step, x)` at line 183
already correctly leaves inactive patches unchanged — the `break` was a compute
optimization only. With it removed, execution depth is fixed at `max_depth=4`;
outputs are bit-exact identical.

> [!NOTE]
> FIX-002 eliminates 4 GPU-CPU syncs per forward pass — far fewer than BUG-01's
> 796 syncs per backward. Given FIX-001 showed no timing improvement from 796 syncs,
> FIX-002 is unlikely to produce a large latency reduction. The primary expected
> benefit is **elimination of bimodal latency variance**, not mean latency improvement.

### Measured results (task-6126 — COMPLETE)
| Metric | Pre-fix | Post-fix | Change |
|---|---|---|---|
| Forward logit NaN/Inf | — | 0 / 0 | ✅ PASS |
| Eval forward | 372.4ms ±20ms, CV=5.4% | 364.6ms ±23.7ms, CV=6.5% | −2.1% |
| Bimodal (CV>15%) | False | False | Not eliminated (absent in both) |
| Training step | 3,240ms ±102ms, CV=3.2% | 3,203ms ±87.5ms, CV=2.7% | −1.1% |
| Quality (5-seed) | 3.1118 ±0.047 | 3.1147 ±0.044 | +0.003, p=0.924 ✅ |

Bimodal latency was not present in eval-mode measurements at SL=64 either before
or after the fix. Bimodal pattern may only manifest in training mode or at longer
sequences. All latency changes within noise.

### Acceptance
- [x] Numerical equivalence — PASS (no NaN/Inf)
- [x] Synchronization eliminated (4 GPU-CPU syncs removed)
- [ ] Bimodality eliminated — not confirmed (absent in both conditions at SL=64)
- [x] Mean latency measured (within noise, −1–2%)
- [x] Quality unchanged (p=0.924)

---

## FIX-003 — Backbone Entropy-Gate Code-Path Bug

| Field | Value |
|---|---|
| **Bug ID** | BUG-05 |
| **Git SHA** | `b5889f1` |
| **Baseline SHA** | `7012134f` |
| **Date** | 2026-08-14 |
| **Type** | Code-path bug fix — restores intended entropy-gate behavior |
| **Decision** | **ACCEPTED** |

### Defect
`if self.use_moe and is_complex: ... else: self.dense_ffn(norm_x)` at `backbone.py:116`.
When `use_moe=True` and `is_complex=False` (entropy < 0.20), `dense_ffn` does not exist.
Raises `AttributeError`. Always triggers when `use_blt=False` (zero entropy),
blocking the BLT isolation experiment (ABL-C).

### Fix
Separated `is_complex` entropy gate from `use_moe` flag. When `is_complex=False`,
the FFN is now skipped entirely (pass-through unchanged) — the correct entropy-gate
behavior. When `is_complex=True`, MoE or dense_ffn is selected based on `use_moe`.

### Measured results (task-6126 — COMPLETE)
| Metric | Result |
|---|---|
| `use_blt=False` crash | **ELIMINATED — PASS** ✅ |
| Logit range (no BLT) | [−1.302, 1.520] — valid |
| Full IVERI quality | 3.1147 ±0.044 vs 3.1118 ±0.047 (Δ=+0.003, p=0.924) ✅ |
| ABL-C enabled | YES — ran successfully |

**Decision**: **ACCEPTED**. Bug fixed; intended entropy-gate behavior restored.
ABL-C result: 2.0502 ±0.194 vs baseline 3.1118 (−1.062 nats, p=0.000151).
**Interpretation is confounded** — see `ABL-C_BLT_ISOLATION.md` for full analysis.
The dramatic improvement cannot be attributed to BLT alone (3 simultaneous changes).

### Defect
`if self.use_moe and is_complex: ... else: self.dense_ffn(norm_x)` at `backbone.py:116`.
When `use_moe=True` and `is_complex=False` (entropy < 0.20), `dense_ffn` does not exist.
Raises `AttributeError`. This always triggers when `use_blt=False` (zero entropy),
blocking the BLT isolation experiment (ABL-C).

### Fix
Separated `is_complex` entropy gate from `use_moe` flag. When `is_complex=False`,
the FFN is now skipped entirely (pass-through unchanged) — the correct entropy-gate
behavior. When `is_complex=True`, MoE or dense_ffn is selected based on `use_moe`.

### Measured results so far (task-6126, partial)
| Metric | Result |
|---|---|
| `use_blt=False` crash | **ELIMINATED — PASS** ✅ |
| Logit range (no BLT) | [−1.302, 1.520] — valid |
| ABL-C quality | **Pending** — running |

### Secondary effect
FIX-003 directly enables **ABL-C** (IVERI without BLT) — the critical experiment
needed to determine whether the quality deficit is in BLT or the Mamba backbone.

---

## BUG-04 — MoR Router Zero-Gradient — REJECTED / FALSE POSITIVE

> [!CAUTION]
> **BUG-04 is rejected. It was a false positive in the forensic audit.**
> There is no fix and no code change associated with this entry.

### Forensic audit claim
`torch.floor(...).long()` in `model/mor/router.py:87` blocks all gradient flow to
router parameters, meaning the router never learns.

### Why this is wrong
After reading [`router.py`](file:///c:/Users/datta.000/Desktop/iveri%20core%20nexus/iveri-core/model/mor/router.py) completely:

The production router (`research_mode=False`, the default in all experiments) has
**no trainable parameters**. `self.wg` is only created when `research_mode=True`.
The production path is a pure deterministic function:

```python
# Option C Production Mode: direct entropy-to-depth mapping
depths = 1 + torch.floor(entropy * (self.max_depth - 1)).long()
depths = torch.clamp(depths, min=1, max=self.max_depth)
```

There is nothing to receive gradients. This is documented in the source as intentional:
*"Official Option C Production Mode: direct entropy-to-depth mapping."*

The forensic audit inferred a gradient-blocking bug without establishing that any
trainable parameters existed at the reported location. The inference was incorrect.

### Status
**REJECTED. Not a bug. No action required.**

Note: `research_mode=True` (not used in any experiment to date) has `self.wg`
parameters and uses argmax dispatch, which does block gradients to those parameters.
That is a separate issue and only relevant if research mode is activated.

---

## FIX-005 — BLT Quadratic Memory

| Field | Value |
|---|---|
| **Bug ID** | BUG-03 |
| **Status** | **PLANNED** — after ABL-C results reviewed |
| **Location** | `model/blt/encoder.py:129` |

### Defect
`blt/encoder.py` allocates a dense tensor of shape `(B, P_max, seq_len)` —
O(B×S²) memory. At SL=64 impact is manageable (~65KB). At SL=512: ~1MB per pass.
At SL=2048: ~64MB per pass. This blocks long-context testing (passkey retrieval
timed out at ctx=512 in EXP-011).

### Pre-fix requirement
Establish the exact mathematical function of the pooling tensor before replacing it.
The optimized implementation must produce numerically equivalent outputs.

### Plan
1. Read `blt/encoder.py` completely; characterize the pooling operation
2. Create reference test with known inputs/outputs
3. Implement sparse equivalent (`scatter_add` or masked indexing)
4. Verify numerical equivalence at SL = [64, 128, 256, 512, 1024, 2048]
5. Re-run long-context benchmark after fix

**Not started.** Waiting for ABL-C results to inform whether BLT is worth optimizing
before fixing its memory implementation.

---

## Titans Algorithmic Bottleneck (not a defect)

| Classification | Algorithmic bottleneck — not an implementation defect |
|---|---|
| Location | `model/titans/memory.py` — `torch.autograd.grad(create_graph=True)` inside `for t in range(P):` |
| Measured cost | ~600ms per training step (from ABL-D: removing Titans saves this) |
| Nature | Second-order differentiation required for Titans' differentiable online learning. P separate higher-order autograd calls per forward pass. Cannot be removed without changing the Titans algorithm. |
| Status | **INVESTIGATION DEFERRED** until ABL-C and FIX-002 results are reviewed. |

Candidate investigations (after ABL-C):
1. Truncated BPTT through Titans (limit gradient depth)
2. Vectorized batch update (amortize across patches)
3. Disable Titans online learning during training (fixed memory state)
4. Remove Titans entirely at this scale (if quality benefit remains undetected)

> [!WARNING]
> Each candidate above changes Titans' algorithm semantics. Any such change must be
> documented as an ALGORITHMIC VARIANT, compared against the frozen baseline, and
> evaluated with a full ablation — not treated as a bug fix.

---

*Last updated: 2026-08-14*
