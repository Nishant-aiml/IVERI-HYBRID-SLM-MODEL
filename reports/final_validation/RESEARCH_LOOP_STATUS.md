# IVERI Core — Validated Research Loop & Next Steps
## STATUS: Living document — updated as experiments complete
## Based on: final_experiment_registry.json (schema 1.1), EXP-019, EXP-020, EXP-021

---

## The Correct Research Loop (No Architecture Changes Until This Is Complete)

```
CURRENT IVERI (frozen at git SHA 7012134f)
     |
     v
FULL VALIDATION (in progress)
     |
     v
FIND ACTUAL FAILURE (identified: LR fairness, latency overhead, component attribution)
     |
     v
IDENTIFY ROOT CAUSE (profiling in progress via EXP-021)
     |
     v
FORMULATE HYPOTHESIS (one per intervention)
     |
     v
ONE TARGETED MODIFICATION (only after all validation complete)
     |
     v
ABLATION
     |
     v
RE-BENCHMARK against frozen IVERI-v1
     |
     v
ACCEPT / REJECT MODIFICATION
     |
     v
FINAL VALIDATION
```

---

## Problem 1 — LR Fairness (EXP-020, IN PROGRESS)

### What the validation data already shows (Phase 1 of EXP-020):

| Transformer LR | Val mean loss | n_valid | Status |
|---|---|---|---|
| 1e-4 | 3.2188 | 3/3 | Too slow to converge in 10 steps |
| **3e-4** | **2.4858** | **3/3** | **SELECTED by validation protocol** |
| 1e-3 | 2.5352 | 3/3 | Close second |
| **3e-3** | **4.3852** | **3/3** | **Used in EXP-019 — worst of all LRs** |

> [!CAUTION]
> **This is a significant finding.** EXP-019 used lr=3e-3 for the Transformer, which the
> validation sweep identifies as the **worst-performing LR** across all four candidates.
> The Transformer's val_mean at lr=3e-3 (4.39) is nearly 2x worse than at lr=3e-4 (2.49).
> This means EXP-019 compared IVERI against a Transformer trained at a sub-optimal LR.
> The EXP-019 IVERI advantage (3.061 vs 5.762) must be interpreted with this caveat.

### What this means for the research conclusion:
- EXP-019 does NOT establish IVERI superiority over a fairly-tuned Transformer.
- EXP-020 Phase 2 (test set with lr=3e-4 Transformer) will determine whether the
  advantage persists after fair LR selection.
- If Transformer at lr=3e-4 achieves test loss near IVERI's 3.06 → **advantage vanishes**.
- If Transformer at lr=3e-4 still loses significantly → **advantage survives LR fairness**.

### EXP-020 Status: RUNNING (Phase 2 in progress)

---

## Problem 2 — Latency Overhead (EXP-021, COMPLETE)

### Measured results:
| Measurement | Value |
|---|---|
| IVERI forward pass | 524.53 ms (CV=46% — bimodal!) |
| Transformer forward pass | 23.61 ms (CV=10% — stable) |
| Forward-only ratio | **22.2x** |
| Training-step ratio (EXP-019) | **~85x** |
| Backward/optimizer overhead ratio | ~3.9x additional beyond forward |
| Latency at sl=32 | 23.5x ratio |
| Latency at sl=256 | 93.4x ratio |

### Root cause hypothesis (NOT YET PROVEN — requires instrumentation):
Python-sequential Titans recurrent update processes each byte individually.
At sl=64: 64 bytes × Python_loop_overhead >> CUDA kernel cost.

### Next step for Problem 2:
Add `skip_memory_update: bool = False` to `IVERIModel.forward()`.
Run timing with `skip_memory_update=True` vs `False`.
The difference = Titans execution cost.
**Do NOT do this yet — finish validation campaign first.**

### Bimodality finding (NEW — not in prior reports):
IVERI forward times split into two clusters:
- Fast: runs 1–6, mean ~363 ms
- Slow: runs 7–10, mean ~874 ms
This bimodality is unexplained. Candidate causes:
1. MoR dynamic routing varying depth between calls
2. Python interpreter JIT behavior
3. GPU memory paging
**This must be investigated before claiming IVERI timing results are reproducible.**

---

## Problem 3 — Unknown Source of Quality Improvement

### Minimum ablation ladder required:

| Experiment | Config | Purpose |
|---|---|---|
| A — Raw-byte Transformer | Transformer, raw bytes, vocab=259 | Already done (EXP-019 Transformer side) |
| B — BLT Transformer | Transformer + BLT/entropy patching | Isolates BLT effect from architecture |
| C — IVERI without BLT | IVERI, fixed-length patches | Isolates backbone from BLT |
| D — IVERI without Titans | IVERI, Titans disabled/zeroed | Isolates Titans contribution |
| E — IVERI without MoR | IVERI, MoR depth=1 (no dynamic routing) | Isolates MoR contribution |
| F — IVERI without MoE | IVERI, single expert | Isolates MoE contribution |
| G — Full IVERI | Current config | Already done (EXP-019 IVERI side) |

**Priority order** (minimum for conference paper):
1. D (IVERI without Titans) — most important because Titans is the latency bottleneck
2. A vs B (BLT effect on Transformer) — resolves BLT/architecture confound
3. C (IVERI without BLT) — completes the BLT confound isolation

Ablations E and F are desirable but can be noted as future work for a conference paper.

### Status: ALL BLOCKED — script not written

### To unlock:
```
# For ablation D (IVERI without Titans):
# Modify configs/base_config.py to add titans_enabled=False flag
# Then in model/iveri_core.py: if not titans_enabled: skip Titans block
# Run: python scratch/ablation_study.py --ablation no_titans --seeds 42 43 44
```

---

## Current Campaign State (2026-08-14)

| Area | Status |
|---|---|
| Architecture implementation | VERIFIED |
| Parameter-matched baseline | EXECUTED (EXP-019) |
| IVERI preliminary loss result | MEASURED (preliminary) |
| Transformer comparison | CONDITIONAL — LR unfair in EXP-019 |
| Statistical evidence | PRELIMINARY — n small, LR untuned |
| LR fairness | IN PROGRESS (EXP-020 running) |
| Component attribution | REQUIRED — BLOCKED |
| Long-context claim | UNVERIFIED — contradicted at ctx=512 |
| Latency root cause | PARTIAL — forward measured, Titans not isolated |
| Bimodality explanation | UNKNOWN — new finding from EXP-021 |
| Independent reproduction | REQUIRED — BLOCKED |
| Final scientific conclusion | PENDING |
| **IEEE Readiness** | **RESEARCH VALIDATION REQUIRED** |

---

## One Rule

**Do not change the IVERI architecture until all of the above experiments are complete.**

Changing IVERI now would:
1. Invalidate EXP-019 as a baseline for the modified version
2. Mix the "IVERI as-built" results with "IVERI modified" results
3. Destroy the clean research narrative

When modification time comes:
- Freeze current IVERI as **IVERI-v1** (git tag: research-freeze-v1)
- Create **IVERI-v2** with ONE targeted change (e.g., vectorized Titans)
- Compare v2 against frozen v1
- Accept or reject the modification based on measured delta

---

## Immediate Next Experiments (in order)

1. **EXP-020** (RUNNING): Wait for completion. Record whether IVERI advantage
   persists after Transformer trained at lr=3e-4.

2. **EXP-021b** (NEXT): Add `skip_memory_update` to IVERIModel.
   Measure: forward with Titans vs without Titans.
   Result determines what % of 524ms comes from Titans.

3. **EXP-013** (NEXT): IVERI without Titans — ablation D above.
   Same 5 seeds. Determines Titans quality contribution.

4. **EXP-022** (NEXT): BLT-patched Transformer — ablation B above.
   Resolves BLT/architecture confound.

5. **EXP-018** (AFTER ABOVE): Clean-room reproduction.
   Follow README_REPRODUCIBILITY.md from scratch.
