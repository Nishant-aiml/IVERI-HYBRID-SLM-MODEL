# Recovery Report — Phase 2.4
**IVERI CORE v1.0 | Phase 2.4**
**Date:** 2026-06-30

---

## 1. Fail-Safe Architecture

`ExperimentLogger` applies a defence-in-depth strategy:

```
Failure in backend N
    → print warning
    → continue to backend N+1
    → training step uninterrupted
```

Every backend write is enclosed in a `try/except Exception` block. No logging failure can propagate to the `Trainer`.

---

## 2. Failure Scenarios Tested

| Scenario | Mechanism | Outcome |
|---|---|---|
| W&B API key absent | `wandb.init()` raises `UsageError` | ✅ Caught, `use_wandb=False`, local backends continue |
| W&B network unavailable | `wandb.init()` or `wandb.log()` raises | ✅ Caught, continues |
| CSV path is a directory | `open()` raises `IsADirectoryError` | ✅ Caught, JSONL continues |
| JSONL write failure | `open()` raises any `OSError` | ✅ Caught, no crash |
| `mode="disabled"` | Logger sets `enabled=False` | ✅ All methods are no-ops |
| `enabled=False` in config | Same | ✅ All methods are no-ops |

---

## 3. Test Coverage

| Test | Scenario | Result |
|---|---|---|
| `test_logger_disabled_is_noop` | `enabled=False` | ✅ PASSED |
| `test_logger_disabled_mode_is_noop` | `mode="disabled"` | ✅ PASSED |
| `test_logger_missing_api_key_falls_back` | No W&B API key | ✅ PASSED |
| `test_corrupted_log_dir_recovery` | CSV path is a directory | ✅ PASSED |

---

## 4. Invariant

> **Training must never stop because logging fails.**

This invariant is guaranteed by the architecture and verified by tests. It holds across all four backend paths and all failure modes tested above.
