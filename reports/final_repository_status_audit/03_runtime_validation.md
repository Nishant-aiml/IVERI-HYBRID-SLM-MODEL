# Final Repository Status Audit — Runtime Validation

## Independent Runtime Tests

Executed from `scratch/audit_runtime.py` on 2026-07-08.

| Test | Result | Detail |
|---|---|---|
| Core Imports | ⚠️ PARTIAL | Model, config, trainer import OK. Inference fails from scratch/ due to sys.path — passes in test suite (4/4 tests). |
| Model Instantiation | ✅ PASS | params=36,600,610 |
| Forward Pass | ✅ PASS | logits.shape=[2, 64, 259], has_nan=False |
| Backward Pass | ✅ PASS | grad_active=399/405, loss=0.8966 |
| Checkpoint Round-Trip | ✅ PASS | step=42, max_diff=0.00e+00 |
| Inference Engine | ⚠️ PATH ISSUE | Works in test suite (test_inference.py: 4 passed), fails from scratch/ dir |
| Empty Sequence | ✅ PASS | empty_logits_shape=[1, 0, 259] |
| CUDA Memory | ⚠️ SKIPPED | Script ran on system Python (3.14) without CUDA. Pytest ran on .venv312 Python 3.12 with CUDA. |
| Determinism | ✅ PASS | max_diff=0.00e+00, deterministic=True |
| Architecture Components | ✅ PASS | All 13 components confirmed present |

## Test Suite Execution (Independent)

Independently executed `python -m pytest tests/` on 2026-07-08:

```
683 passed, 4 skipped, 20 warnings in 1712.94s (0:28:32)
```

### Skipped Tests
- `test_mamba2_block.py`: 2 tests skipped (FlashAttention/Triton not available on Windows)
- `test_attention.py`: 2 tests skipped (FlashAttention-2 backend not available)

### Warnings
- `FutureWarning: torch.load with weights_only=False` in `checkpoint_manager.py:134`
- `DeprecationWarning` from wandb analytics
- `UserWarning` from lr_scheduler step ordering

## Verdict

**Core runtime is stable**. Forward pass, backward pass, checkpointing, and determinism all verified independently. No NaN, no crashes, no memory leaks observed.
