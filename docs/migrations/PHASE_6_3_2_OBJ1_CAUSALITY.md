# Phase 6.3.2 Objective 1 — BLT Causality Restoration

**Date:** 2026-07-06  
**Scope:** Model architecture corrections for autoregressive byte modeling only.

## Summary

Restored end-to-end causality in the BLT front-end and decoder. Future-byte perturbations no longer change logits at earlier positions beyond numerical tolerance (`atol=1e-6`, `rtol=1e-5`).

## Code Changes

| Module | Change |
|--------|--------|
| `model/blt/entropy_model.py` | Causal Conv1d: `padding=0` + left pad `kernel_size-1` |
| `model/blt/encoder.py` | Within-patch causal self-attention (mask future byte keys) |
| `model/blt/decoder.py` | Cross-attention keys limited to patches with `patch_end <= query_index` |
| `research/causality_probe.py` | `allclose` pass tolerance; protocol `Phase-6.3.2-OBJ1` |

## Validation

```powershell
python -m pytest tests/test_causality_runtime.py tests/test_blt.py -q
python -c "from research.causality_probe import write_causality_report; write_causality_report('reports/scientific_integrity_audit/Causality_Report.md')"
```

## Report

Measured results: `reports/scientific_integrity_audit/Causality_Report.md`

## Not Changed (this objective)

- Titans production path (Objective 2)
- MoE entropy routing (Objective 3)
- Ablation gating (Objective 4)
- Publication / replay hardening beyond existing 6.3.1A/B guards
