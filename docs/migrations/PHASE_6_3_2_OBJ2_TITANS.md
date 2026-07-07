# Phase 6.3.2 Objective 2 — Titans Production Integration

**Date:** 2026-07-06  
**Scope:** Wire online Titans memory updates into the production `Backbone.forward` path.

## Summary

Production training and inference now execute the full Titans lifecycle `forward → read → update → write` at backbone entry, with entropy-gated residual injection preserved from the prior `inject()` design.

## Code Changes

| Module | Change |
|--------|--------|
| `model/titans/memory.py` | Added `forward_with_injection()`; telemetry now records measured `read_count` / `write_count` |
| `model/backbone.py` | Replaced `titans.inject()` with `titans.forward_with_injection()`; telemetry reads Titans counters |
| `research/titans_audit.py` | Protocol `Phase-6.3.2-OBJ2`; PASS when `forward_calls > 0` and `updater_calls > 0` |
| `tests/test_titans_runtime_audit.py` | Expect production PASS with online writes |
| `tests/test_backbone.py` | Assert `q_proj` receives gradients via online forward path |

## Validation

```powershell
python -m pytest tests/test_titans_runtime_audit.py tests/test_titans.py tests/test_backbone.py -q
python -c "from research.titans_audit import write_titans_verification_report; write_titans_verification_report('reports/scientific_integrity_audit/Titans_Verification.md')"
```

## Report

Measured results: `reports/scientific_integrity_audit/Titans_Verification.md`

## Preserved

- `TitansMemory.inject()` remains for isolated read-only path tests and Option C gating verification.

## Not Changed (this objective)

- MoE entropy routing (Objective 3)
- Ablation gating (Objective 4)
- Publication / replay hardening beyond existing 6.3.1A/B guards
