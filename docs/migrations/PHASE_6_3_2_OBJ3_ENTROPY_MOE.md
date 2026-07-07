# Phase 6.3.2 Objective 3 — Entropy-Conditioned MoE Routing

**Date:** 2026-07-06  
**Scope:** Wire BLT patch entropy into SparseMoERouter gating (Patent 3).

## Summary

Patch entropy now biases MoE expert gating logits via `w_entropy`, propagates through `BackboneSubBlock` and `RecursionEngine`, and measurably changes routing when hidden states are held fixed.

## Code Changes

| Module | Change |
|--------|--------|
| `model/moe/router.py` | Added `w_entropy`; logits = `wg(x) + w_entropy(entropy)` |
| `model/backbone.py` | `BackboneSubBlock` passes `entropy` to `moe_router` |
| `model/mor/recursion.py` | Stop stripping `entropy` from block kwargs |
| `research/entropy_routing_audit.py` | Protocol `Phase-6.3.2-OBJ3`; fixed-hidden PASS criteria |
| `tests/test_entropy_routing_audit.py` | Expect PASS with routing changes |
| `tests/test_moe_integration.py` | Entropy routing invariance test |

## Validation

```powershell
python -m pytest tests/test_entropy_routing_audit.py tests/test_moe_integration.py tests/test_backbone.py -q
python -c "from research.entropy_routing_audit import write_entropy_routing_report; write_entropy_routing_report('reports/scientific_integrity_audit/Entropy_Routing_Report.md')"
```

## Report

Measured results: `reports/scientific_integrity_audit/Entropy_Routing_Report.md`

## Not Changed (this objective)

- Ablation flag `use_entropy_routing` physical disable (Objective 4)
- Publication / replay hardening beyond existing 6.3.1A/B guards
