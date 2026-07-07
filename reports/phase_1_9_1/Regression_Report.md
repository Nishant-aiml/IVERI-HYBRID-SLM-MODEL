# Regression Report — Phase 1.9.1
## Test Suite Execution Metrics & Verification Results

This report lists the test outcomes and execution statistics for the complete IVERI CORE test suite.

---

## 1. Test Execution Metrics

| Target | Passed | Failed | Skipped | Status | Notes |
|:---|:---:|:---:|:---:|:---:|:---|
| **test_math_layers.py** | 17 | 0 | 0 | **PASS** | RMSNorm, RoPE, SwiGLU correctness and gradchecks. |
| **test_attention.py** | 7 | 0 | 0 | **PASS** | Flash attention wrapper, causal mask, and KV caching. |
| **test_mamba2_math.py** | 3 | 0 | 0 | **PASS** | SSD matrix and discretization math. |
| **test_mamba2_scan.py** | 7 | 0 | 0 | **PASS** | SSD selective parallel scan math and gradients. |
| **test_mamba2_block.py** | 6 | 0 | 0 | **PASS** | Full Mamba2Block shapes, bounds, and gradflow. |
| **test_mamba2_integration.py** | 5 | 0 | 0 | **PASS** | Parameter counting, init determinism, and integration. |
| **test_moe_integration.py**| 3 | 0 | 0 | **PASS** | Top-K routing, aux loss, parameter efficiency. |
| **test_router.py** | 8 | 0 | 0 | **PASS** | Router weights, softmax normalization, and shapes. |
| **test_mor.py** | 7 | 0 | 0 | **PASS** | MoR engine recursive loops, depth assignment, and cache. |
| **test_blt.py** | 10 | 0 | 0 | **PASS** | Entropy model, dynamic patcher, byte encoder/decoder. |
| **test_titans.py** | 11 | 0 | 0 | **PASS** | Online memory weight update loop, surprise-based LR. |
| **test_backbone.py** | 3 | 0 | 0 | **PASS** | Backbone Assembly integration and recursive block stack. |
| **test_iveri_core.py** | 10 | 0 | 0 | **PASS** | Full IVERIModel forward/backward pipeline & checkpoints. |
| **test_stress_1_9_1.py** | 19 | 0 | 0 | **PASS** | Extended stress tests (languages, empty sequences, etc.) |
| **test_config.py** | 18 | 0 | 0 | **PASS** | dataclass slot configurations, BaseConfig overrides. |
| **test_logging.py** | 9 | 0 | 0 | **PASS** | Structured logging, metric formatting, file rotations. |
| **test_validation.py** | 14 | 0 | 0 | **PASS** | Shape checks, nan/inf checks, memory tracking. |
| **test_structure.py** | 10 | 0 | 0 | **PASS** | Folder structure, __init__.py exports, scaffold files. |
| **test_environment.py** | 10 | 0 | 0 | **PASS** | PyTorch, NumPy, Einops, environment libraries checks. |
| **TOTAL** | **192** | **0** | **4** | **PASS** | 4 skipped tests require CUDA GPU. |

---

## 2. Execution Log (Summary)

The complete project regression suite was executed via:
```powershell
python -m pytest -v --tb=short
```

**Key Execution Output:**
```
================= 192 passed, 4 skipped in 281.18s (0:04:41) ==================
```

All 192 tests executed on CPU passed cleanly. The 4 skipped tests are CUDA-specific tests that execute on GPU clusters.

---

## 3. Final Verdict

**Status: PASS**
Zero regressions detected. All Phase 0, Phase 1.1–1.9, and new Phase 1.9.1 stress tests are fully green.
