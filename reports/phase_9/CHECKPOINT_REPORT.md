# Phase 9 — Checkpoint Report

**Date:** 2026-07-11

---

## Isolated Checkpoint Round-Trip (EXECUTED)

**Config:** Reduced nano (hidden_dim=64, 1 layer, 2 experts) for CPU speed  
**Method:** `save_checkpoint` → fresh model → `load_checkpoint` → compare logits

| Metric | Value |
|--------|-------|
| Max logit difference | **0.0** |
| Tolerance | 1e-4 |
| Result | **PASS** |

---

## Phase 9 Campaign Checkpoint Resume (NOT EXECUTED)

`verify_checkpoint_resume()` exists in `scripts/run_phase9_campaign.py` but was **not reached** because campaign failed at 9.2.

The method tests:
- 5-step training loop
- Checkpoint save at step 10
- Load into fresh model
- Logit equivalence < 1e-4

**Status:** Code present, not executed in campaign.

---

## Milestone Checkpoint IVERI_10M_v1.0.pt (NOT VALID)

Campaign `run_campaign()` would save checkpoint at step 2000 **without training 2000 steps** if it reached 9.10. Current campaign never reaches this code path.

---

## Verdict

Checkpoint **infrastructure works** (training/checkpointing.py).  
Phase 9 **campaign checkpoint validation not evidenced**.
