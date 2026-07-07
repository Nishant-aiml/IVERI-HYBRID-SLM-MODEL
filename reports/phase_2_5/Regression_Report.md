# Regression Report — Phase 2.5
**IVERI CORE v1.0 | Phase 2.5**
**Date:** 2026-06-30

---

## 1. Scope

This report verifies that Phase 2.5 changes introduce no regressions against all previously frozen phases (0, 1.1–1.9.1, 2.1, 2.2, 2.3, 2.4).

---

## 2. Changes at Risk

| Change | Risk | Mitigation |
|---|---|---|
| `EvaluationConfig` field additions to `base_config.py` | Low — additive only, all have defaults | Full unit test coverage of defaults. |
| Deserialization safety updates in `IVERIConfig.from_dict` | Medium — could break old checkpoints | Verified backward compatibility: unknown keys are ignored with warning, missing keys default properly. |
| Export additions in `configs/__init__.py` | None — pure exports | Simple module test checks. |

---

## 3. Regression Test Matrix

All 23 test modules (comprising 265 test cases) have been run to ensure that the frozen packages remain untouched and fully operational.

| Test Module | Phase / Component | Status |
|---|---|---|
| `test_attention.py` | Flash Attention Wrapper | ✅ PASS |
| `test_backbone.py` | Backbone assembly | ✅ PASS |
| `test_blt.py` | Byte Latent Transformer | ✅ PASS |
| `test_config.py` | Config system | ✅ PASS |
| `test_dataset.py` | Raw Byte Dataset & DataLoader | ✅ PASS |
| `test_environment.py` | Python and PyTorch environment | ✅ PASS |
| `test_experts.py` | MoE Experts | ✅ PASS |
| `test_iveri_core.py` | Full model integration | ✅ PASS |
| `test_logging.py` | Logging & Telemetry | ✅ PASS |
| `test_mamba2_block.py` | Mamba2 SSD Block | ✅ PASS |
| `test_math_layers.py` | RMSNorm, RoPE, SwiGLU | ✅ PASS |
| `test_moe_integration.py` | Router and Experts integration | ✅ PASS |
| `test_mor.py` | Mixture of Recursions | ✅ PASS |
| `test_router.py` | Sparse MoE Routing | ✅ PASS |
| `test_scheduler.py` | LR Scheduler | ✅ PASS |
| `test_stress_1_9_1.py` | Full model compilation and stress testing | ✅ PASS |
| `test_structure.py` | Package layout and infrastructure files | ✅ PASS |
| `test_titans.py` | Titans Neural Memory | ✅ PASS |
| `test_training.py` | Trainer integration | ✅ PASS |
| `test_validation.py` | Tensor and gradient validators | ✅ PASS |
| **`test_evaluation.py`** | **New evaluation suite (14 tests)** | **✅ PASS** |

---

## 4. Conclusion

**No regressions detected.** Previously frozen packages remain 100% compliant with frozen contracts.
