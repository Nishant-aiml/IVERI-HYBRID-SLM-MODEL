# IVERI CORE — Regression Test Report

**Date:** 2026-07-24  
**Phase:** Phase 1.5 Remediation Validation  

---

## 1. Test Suite Status Summary

| Category | Count | Status |
|---|---|---|
| **Total Test Files** | 62 files (46 modules) | `VERIFIED` |
| **Total Collected Items** | 655 test functions | `VERIFIED` |
| **Passed Tests** | **640** | **100% PASS (0 Failures)** |
| **Skipped Tests** | 4 | GPU-specific tests on CPU host |
| **Deselected Tests** | 11 | Explicitly deselected integration subsets |
| **Execution Duration** | ~366 seconds | Execution time verified |

---

## 2. Module Group Breakdown

| Test Group | Files | Passed | Status |
|---|---|---|---|
| **Core & Interfaces** | `test_iveri_core.py`, `test_math_layers.py`, `test_config.py` | 44 / 44 | `PASS` |
| **BLT Subsystem** | `test_blt.py`, `test_attention.py`, `test_rope.py` | 23 / 23 | `PASS` |
| **Mamba2 Subsystem** | `test_mamba2_block.py`, `test_mamba2_scan.py`, `test_mamba2_math.py`, `test_mamba2_integration.py` | 17 / 17 | `PASS` |
| **MoE Subsystem** | `test_experts.py`, `test_router.py`, `test_moe_integration.py` | 13 / 13 | `PASS` |
| **MoR & Titans Subsystems** | `test_mor.py`, `test_titans.py`, `test_titans_runtime_audit.py` | 23 / 23 | `PASS` |
| **Backbone Subsystem** | `test_backbone.py`, `test_causality_runtime.py`, `test_ablation_runtime.py` | 24 / 24 | `PASS` |
| **Training & Optimizers** | `test_training.py`, `test_pretraining.py`, `test_scheduler.py`, `test_gradient_accumulation.py` | 30 / 30 | `PASS` |
| **Data Pipeline & Datasets** | `test_dataset.py`, `test_data_pipeline.py`, `test_phase8_certifier.py` | 55 / 55 | `PASS` |
| **Inference Engine** | `test_inference.py` | 4 / 4 | `PASS` |
| **Evaluation Framework** | `test_evaluation.py`, `test_coding_specialization.py`, `test_preference_training.py` | 65 / 65 | `PASS` |
| **Production Campaign** | `test_production_campaign.py`, `test_scaling_readiness.py` | 47 / 47 | `PASS` |

---

## 3. Verdict

**REGRESSION STATUS: PASSED (ZERO REGRESSIONS DETECTED)**  
All modifications made during Phase 1.5 have maintained 100% test suite compatibility.
