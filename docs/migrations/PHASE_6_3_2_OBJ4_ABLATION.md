# Phase 6.3.2 Objective 4 — Real Ablation Framework

**Date:** 2026-07-06  
**Scope:** Physical component disabling via `ModelConfig` boolean gates.

## Summary

Ablation flags now remove components from the forward path instead of only shrinking hyperparameters. Campaign overrides apply to `cfg.model` and fail loudly on unknown fields.

## Code Changes

| Module | Change |
|--------|--------|
| `configs/base_config.py` | Added `use_titans`, `use_blt`, `use_mor`, `use_moe`, `use_entropy_routing`; `apply_ablation_overrides()` |
| `model/backbone.py` | Conditional Titans, MoR bypass, dense FFN when MoE ablated |
| `model/iveri_core.py` | BLT bypass path with `byte_embed` + `bypass_lm_head` |
| `model/moe/router.py` | Respects `use_entropy_routing` |
| `research/baselines.py` | Sets boolean flags instead of hyperparameter weakening |
| `research/ablation.py` | Added `no_entropy_routing` |
| `research/campaign_runner.py` | Uses `apply_ablation_overrides` |
| `research/ablation_audit.py` | Runtime proof harness |
| `tests/test_ablation_runtime.py` | Physical ablation tests |

## Validation

```powershell
python -m pytest tests/test_ablation_runtime.py tests/test_research.py tests/test_production_campaign.py -q
python -c "from research.ablation_audit import write_ablation_verification_report; write_ablation_verification_report('reports/scientific_integrity_audit/Ablation_Verification.md')"
```

## Report

Measured results: `reports/scientific_integrity_audit/Ablation_Verification.md`

## Not Changed (this objective)

- Publication / replay hardening (Objectives 5–6)
- Byte vocabulary collisions (Objective 7)
