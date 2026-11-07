# FINAL RESEARCH VALIDATION CERTIFICATE
## IVERI Core — Hybrid Byte-Level Small Language Model
## Git SHA: 7012134f44403f6732333bba9e93bbc7ea73b867
## Certificate Updated: 2026-08-14T07:00:00Z (post EXP-020 completion)

---

## CLASSIFICATION

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CLASSIFICATION:   RESEARCH VALIDATION REQUIRED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Validated Campaign Status

| Area | Status | Source |
|---|---|---|
| Architecture implementation | VERIFIED | Source code inspection, EXP-001–003 |
| Parameter-matched baseline executed | VERIFIED | EXP-019, delta=3.07% |
| Primary quality comparison (EXP-019) | WITHDRAWN AS COMPARATIVE CLAIM | EXP-019 Transformer at worst LR |
| LR-fair quality comparison (EXP-020) | COMPLETE — Transformer wins | EXP-020 raw JSON |
| IVERI LR 3e-3 (used in EXP-019) | Selected according to the validation-based hyperparameter-selection protocol (lowest val_mean=3.010 across preregistered grid) | CONFIRMED (EXP-020) |
| Statistical evidence of quality advantage | FALSIFIED under fair conditions | EXP-020 |
| Training stability (IVERI 0/5 diverged at lr=3e-3) | MEASURED | EXP-019 |
| Latency overhead (forward: 22x, step: 85x) | MEASURED | EXP-021 |
| Latency bimodality (CV=46%) | MEASURED — cause unknown | EXP-021 |
| Component attribution | INCOMPLETE — ablations BLOCKED | EXP-013/014 |
| Long-context retention | CONTRADICTED at ctx=512 | EXP-010 |
| Independent reproduction | INCOMPLETE | EXP-018 |
| Final scientific conclusion | PENDING | — |

---

## Primary Finding (Post EXP-020)

> **Under a preregistered LR fairness protocol (validation-selected LR for each model), a parameter-matched Pre-LN Dense Transformer (lr=3e-4) achieved lower test cross-entropy loss than IVERI-Nano (lr=3e-3) after 10 training steps on a synthetic byte-level corpus: Transformer 2.705 ± 0.176 vs. IVERI 3.112 ± 0.047 (Welch t=4.994, p=0.005, Cohen's d=3.157, n=5).**

> **The IVERI quality advantage observed in EXP-019 (IVERI 3.061 vs. Transformer 5.762) was largely or entirely attributable to the Transformer being evaluated at lr=3e-3, which the validation sweep confirmed as the worst-performing LR for the Transformer.**

> **IVERI currently demonstrates no quality advantage over a fairly-tuned Transformer under the evaluated early-training protocol. IVERI additionally incurs ~85x wall-clock overhead per training step and failed the evaluated 512-token passkey retention test.**

---

## What Is Established

| Claim | Status |
|---|---|
| IVERI implementation works and trains stably | VERIFIED |
| IVERI operates in 340 MB VRAM on RTX 3050 | VERIFIED |
| IVERI achieves lower loss than Transformer at lr=3e-3 (same LR for both) | MEASURED (EXP-019) |
| IVERI lr=3e-3 is confirmed optimal by validation | CONFIRMED (EXP-020) |
| Under fair LR tuning, Transformer achieves lower loss than IVERI | MEASURED (EXP-020) |
| IVERI incurs 22x forward / 85x training-step overhead | MEASURED (EXP-019, EXP-021) |
| IVERI forward-pass latency is bimodal (CV=46%) | MEASURED (EXP-021) |
| IVERI overhead scales super-linearly with sequence length | MEASURED (EXP-021) |

## What Is Not Established

| Claim | Reason |
|---|---|
| IVERI is better than Transformer | Falsified by EXP-020 under fair conditions |
| IVERI is computationally efficient | Directly contradicted by EXP-019/021 |
| Any component causes the quality difference | No ablations run |
| IVERI retains information in long context | Contradicted at ctx=512 |
| Results generalize to real text | Synthetic corpus only |

---

## Certificate Unlock Conditions

The certificate CANNOT advance beyond RESEARCH VALIDATION REQUIRED until:

| # | Requirement | Status |
|---|---|---|
| 1 | EXP-013: IVERI without Titans — isolates Titans quality contribution | BLOCKED |
| 2 | EXP-022: BLT-patched Transformer — resolves input representation confound | BLOCKED |
| 3 | EXP-021b: Titans timing isolation (`skip_memory_update` flag) | BLOCKED |
| 4 | Explanation of latency bimodality (CV=46%) | UNKNOWN |
| 5 | Long-context investigation at progressive ctx (256→512→1024) | BLOCKED |
| 6 | EXP-018: Independent clean-room reproduction | BLOCKED |
| 7 | Novelty audit against prior byte-level / hybrid SLM work | INCOMPLETE |

---

## Research Program Going Forward

The falsification campaign has produced a clear, honest result: under fair conditions, IVERI does not currently outperform the Transformer on the evaluated task. This is a scientifically valid outcome — it refines the research question from:

> ~~"Does IVERI outperform a Transformer?"~~

to:

> **"Under what conditions, if any, does the IVERI hybrid architecture provide an advantage? What is causing the quality gap vs. a fair Transformer baseline? Can Titans memory be implemented without the observed overhead while preserving the learning dynamics that differ between the two architectures?"**

The correct next experiments in order are:
1. Titans timing isolation (EXP-021b) — add `skip_memory_update` flag
2. IVERI without Titans ablation (EXP-013) — does removing Titans close the quality gap?
3. BLT-patched Transformer (EXP-022) — does BLT alone explain representational differences?
4. Progressive long-context test (ctx=256→512→1024→2048) — where exactly does retention fail?

**Architecture must remain frozen at git SHA `7012134f` until all of the above are complete.**

---

*Issued by: IVERI Core Scientific Validation Campaign*
*Authority: final_experiment_registry.json (schema 1.1)*
*Status: RESEARCH VALIDATION REQUIRED — certificate will be re-evaluated after EXP-013, EXP-021b, EXP-022, EXP-018*
