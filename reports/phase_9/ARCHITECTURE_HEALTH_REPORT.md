# Phase 9 — Architecture Health Report

**Date:** 2026-07-11  
**Script:** `scripts/architecture_health.py`  
**Device:** CPU

---

## Execution Result: FAILED

MoE load balancing assertion failed before script could print "ALL SUBSYSTEMS HEALTHY".

---

## Measured Telemetry (last step)

| Subsystem | Metric | Value | Gate | Result |
|-----------|--------|-------|------|--------|
| MoE | Expert histogram | [950, 2122] | All experts > 0 | PASS |
| MoE | Max expert load | **69.08%** | < 60% | **FAIL** |
| MoR | Avg recursion depth | > 0 | > 0 | PASS |
| Titans | Read count | > 0 | > 0 | PASS |
| Titans | Write count | logged | — | — |
| Gradients | Active tensors | > 0 | > 0 | PASS |
| Gradients | NaN/Inf | 0 | 0 | PASS |

---

## Model Init Telemetry (forward-only, no training)

From isolated `IVERIModel` forward pass:

```
BLT entropy: 0.999996
MoR depth: 3.0
MoE hist: [677, 859]
Titans reads: 128, writes: 128
```

All subsystems **active** on forward pass.

---

## Phase 9 Campaign Architecture Health Score

**NOT COMPUTED FROM REAL DATA.**

`run_phase9_campaign.py` writes hardcoded:
```json
{"score": 98.5}
```

This is **invalid** for scientific use.

---

## Recommendation

1. Investigate whether 60% MoE threshold is appropriate for 2-expert nano config
2. Integrate `architecture_health.py` logic into campaign with measured score formula
3. Do not use hardcoded 98.5
