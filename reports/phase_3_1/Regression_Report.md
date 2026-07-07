# Regression Report — IVERI CORE Phase 3.1

This report documents the full codebase regression test suite verification outcomes.

## Test Summary

- **Total Tests Executed**: 339
- **Passed**: 339
- **Failed**: 0
- **Skipped**: 4 (CUDA-only tests skipped due to CPU execution environment)
- **Status**: ✅ PASSED

## Tested Modules

1. **Pretraining Pipeline (`tests/test_pretraining.py`)**: 11 new tests verified dataset loader, curriculum scheduler, baseline transformer, loss monitor, convergence analyzer, checkpoint selector, evaluator, generation inspector, failure recovery, numerical stability, and pretraining runner.
2. **Structure (`tests/test_structure.py`)**: 10 tests verified directory and package file locations, package initializations, baseline files locations, and structure conformity.
3. **Core Modules (`tests/test_iveri_core.py`, etc.)**: 318 tests verified backbone, Mamba2, Flash Attention, Titans Memory, and MoE experts.

## Conclusion

No regressions were introduced. All tests remain green.
