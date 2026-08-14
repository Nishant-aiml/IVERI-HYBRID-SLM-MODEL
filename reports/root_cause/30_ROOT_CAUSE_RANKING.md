# 30_ROOT_CAUSE_RANKING.md
## IVERI Core — Root Cause Priority Ranking
## Based on: experimental evidence + forensic code audit
## Date: 2026-08-14

---

## Ranking Criteria

| Criterion | Weight |
|---|---|
| Impact | Effect size on loss, latency, or stability |
| Evidence strength | Experiment + code line vs. inference |
| Reproducibility | 100% / likely / conditional |
| Fixability | Code change without architectural redesign? |

---

## CRITICAL

### BUG-01: Titans .item() calls in per-patch loop → 89.5% backward overhead
| Field | Value |
|---|---|
| Location | `model/titans/memory.py:317–341, 453–509` |
| Effect | Every backward pass executes O(P) CPU-GPU synchronizations via `.item()` and `.tolist()`. Backward pass = **2,937ms of 3,281ms total step time (89.5%)**. |
| Evidence | Deep profiler: training step 3,281ms, forward 344.7ms. Code: multiple `.item()` calls inside `for t in range(P):` loop. |
| Reproducibility | 100% |
| Fix | Move all `.detach().item()` telemetry collection **outside** the patch loop. Log scalars only once after the loop completes. **Zero semantic change to the algorithm.** |
| Expected impact | Estimated 60–80% reduction in training step time. |
| Risk | LOW — pure telemetry refactoring, no mathematical change. |

---

### BUG-02: MoR .any() inside recursion loop → dominant forward latency + bimodal timing
| Field | Value |
|---|---|
| Location | `model/mor/recursion.py:171` — `if not active_mask.any(): break` |
| Effect | Causes CUDA synchronization on every recursion step. MoR is responsible for **53% of total training step time** (1,628ms saved when MoR removed, ABL-E). Also causes bimodal forward latency (CV=46%). |
| Evidence | ABL-E: removing MoR → 1,470ms vs 3,098ms. Code: `.any()` on GPU tensor evaluated as Python bool. |
| Reproducibility | 100% |
| Fix | Option A (minimal): Remove the `break` condition entirely — always execute `max_depth` iterations. Adds predictability, removes bimodality, eliminates sync. Option B: Pre-compute batch termination condition on GPU without Python evaluation. |
| Expected impact | Elimination of bimodal latency. Estimated 30–50% reduction in forward latency. |
| Risk | MEDIUM — removes early exit optimization, may slightly increase compute for low-entropy inputs. |

---

### BUG-04: MoR router gradient blocking → router never learns
| Field | Value |
|---|---|
| Location | `model/mor/router.py:87` — `torch.floor(...).long()` |
| Effect | Gradient to `self.wg` (router parameters) = **exactly zero** at every step. Router uses a fixed linear mapping of entropy → depth. The 356K router parameters are updated only through gradient leakage from the entropy model, not through the intended routing objective. |
| Evidence | Code: `.long()` is non-differentiable. Gradient flow audit: MoR router status = BLOCKED. Ablation: removing MoR (non-learning) costs only +0.053 nats (p=0.386 — not significant). |
| Reproducibility | 100% |
| Fix | Replace `floor(...).long()` with straight-through estimator (STE): use floor in forward, pass gradient through as if it were identity. Or use Gumbel-Softmax relaxation for differentiable discrete routing. |
| Expected impact | Router parameters may learn useful routing patterns. Quality improvement: UNKNOWN until measured — could be 0 to significant. |
| Risk | MEDIUM — changes optimization landscape. Must validate via ablation after fix. |

---

## HIGH

### BUG-03: BLT encoder O(B×S²) dense pooling matrix → memory explosion at long context
| Field | Value |
|---|---|
| Location | `model/blt/encoder.py:129` |
| Effect | Creates dense boolean/float matrix of shape (B, P_max, seq_len). At SL=512, B=4: ~1MB active. At SL=2048: ~64MB. Causes long-context timeout (EXP-011: >25 min at ctx=512). |
| Evidence | Code inspection. EXP-011 timeout. VRAM overhead (336MB vs 206MB for Transformer). |
| Reproducibility | 100% |
| Fix | Replace dense boolean matrix with sparse indexing + `scatter_add` reduction. Reduces memory from O(B×S²) to O(B×S). |
| Expected impact | Long-context capability becomes testable. VRAM reduction. |
| Risk | MEDIUM — changes tensor operations but not mathematical result if implemented correctly. |

---

### BUG-05: MoE entropy gate silently disables MoE when entropy < 0.20 → use_blt=False crash
| Field | Value |
|---|---|
| Location | `model/backbone.py:116` — `if self.use_moe and is_complex:` |
| Effect | When patch_entropy < 0.20, both attention AND MoE are skipped, falling to `else: self.dense_ffn(norm_x)`. But `dense_ffn` doesn't exist when `use_moe=True`. This crashes when `use_blt=False` (zero entropy), and silently removes MoE routing for low-entropy patches during normal training. |
| Evidence | ABL-C crash: `'BackboneSubBlock' object has no attribute 'dense_ffn'`. |
| Reproducibility | 100% when `use_blt=False`. Conditional when `use_blt=True` and entropy < 0.20. |
| Fix | Add fallback: when `is_complex=False` and `use_moe=True`, use the top-1 expert (simplest dispatch) rather than calling `dense_ffn`. Or separate the attention gate from the MoE gate. |
| Expected impact | Enables ABL-C (BLT ablation), fixes the code path. May change quality if low-entropy patches are common. |
| Risk | LOW — bug fix, not redesign. |

---

## MEDIUM

### SCALE-01: Titans/MoR/MoE components provide no measurable quality benefit at 10M params / 10 steps
| Field | Value |
|---|---|
| Evidence | Ablations show SIMILAR loss with or without Titans (p=0.670), SIMILAR with or without MoR (p=0.386), IDENTICAL with or without MoE (p=0.998). All three components add overhead, none add quality. |
| Nature | May be scale issue (components need larger models), training budget issue (10 steps is insufficient to see benefits), or fundamental mismatch between components and byte-level task. |
| Status | HYPOTHESIS — "scale issue" cannot be confirmed without larger-scale experiment. |
| Action | Do not remove components permanently. Fix the bugs, test with longer training, evaluate whether benefit emerges. |

### EXPERIMENTAL-01: 10 training steps is insufficient to characterize architecture behavior
| Field | Value |
|---|---|
| Evidence | EXP-020 uses 10 steps. This measures early optimization dynamics, not convergence quality. |
| Action | Training dynamics experiment (training_dynamics.py, currently staged) — must be run to determine whether Transformer consistently dominates or curves cross. |

---

## LOW

### IMPL-01: Titans telemetry incorrectly reports 0ms (telemetry key mismatch)
| Field | Value |
|---|---|
| Evidence | `titans_time_seconds` key exists in backbone's `t_start/titans_time` measurement but is NOT passed to `_update_telemetry_data` in the same way as other telemetry keys. Result: telemetry always shows 0ms for Titans. |
| Impact | Reporting only — no effect on training or results. |
| Fix | Pass `titans_time` to telemetry dict in `Backbone._update_telemetry_data()`. |

---

## Summary Priority Table

| Priority | Bug ID | Component | Effect | Fix Complexity | Run Next? |
|---|---|---|---|---|---|
| 🔴 CRITICAL | BUG-01 | Titans .item() | 89.5% backward overhead | LOW | YES — first fix |
| 🔴 CRITICAL | BUG-02 | MoR .any() | 53% step time, bimodality | MEDIUM | YES — second fix |
| 🔴 CRITICAL | BUG-04 | MoR router gradient | Router never learns | MEDIUM | YES — third fix |
| 🟠 HIGH | BUG-03 | BLT encoder O(S²) | Memory/long-context failure | MEDIUM | YES — enables long-ctx test |
| 🟠 HIGH | BUG-05 | MoE entropy gate | use_blt=False crash, BLT ablation blocked | LOW | YES — enables ABL-C |
| 🟡 MEDIUM | SCALE-01 | All novel components | No measured quality benefit | N/A | Investigate after bugs fixed |
| 🟡 MEDIUM | EXPERIMENTAL-01 | Training budget | 10-step results insufficient | N/A | Run training_dynamics.py |
| 🟢 LOW | IMPL-01 | Titans telemetry key | 0ms reporting in telemetry | LOW | Eventually |
