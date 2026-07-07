# Regression Testing Report

This report confirms that the implementation of Phase 3.0 has not introduced any regressions.

## 1. Testing Summary

- **Total Tests Executed**: 328
- **Passed**: 328
- **Failed**: 0
- **Skipped**: 4 (distributed tests that require multiple MPI processes or GPU setups not present in local test run)
- **Time Elapsed**: 347.94s (~5m47s)

## 2. Frozen Phases Verified

All pre-existing regression tests for earlier phases passed successfully:

- **Phase 0 (Foundation)**: conftest config roundtrips and validations passed.
- **Phase 1.1-1.9.1 (Architecture)**: Attention, backbone, BLT encoder/decoder, mamba2, Titans memory, and Mixture of Recursions tests passed.
- **Phase 2.1-2.6 (Training & Environment)**: LR scheduler, dataset loaders, logging, trainer, distributed trainer, and evaluation wrapper tests passed.

This confirms that the new dataset pipeline code coexists cleanly with all pre-existing parts of the codebase, preserving backward compatibility.
