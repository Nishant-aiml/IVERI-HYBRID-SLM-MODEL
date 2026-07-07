# Regression Report — Phase 2.1
## Verification of Codebase Invariant Properties

This report documents the regression suite status of the IVERI CORE codebase after the integration of the data loading and pre-processing modules.

---

## 1. Test Invariants

All architectural and model constraints remain fully preserved:
- **No changes to frozen model code:** File checks confirm that no model files under `model/` (e.g. `norms.py`, `rope.py`, `swiglu.py`, `attention.py`, `backbone.py`, `iveri_core.py`) were modified.
- **No changes to mathematical equations:** Verification tests for all components remain bitwise identical to Phase 1 completions.
- **No changes to execution order or tensor interfaces:** Verified that model shape contracts match `docs/architecture/tensor_interfaces.md`.

---

## 2. Regression Test Execution Summary

The complete project regression suite was executed:
```powershell
python -m pytest -v --tb=short
```

**Outcome:**
* **Total Tests Collected:** 219
* **Passed:** 215
* **Failed:** 0
* **Skipped:** 4 (CUDA specific tests, skipped in CPU environment)
* **Status:** **PASS**

All 215 active tests passed cleanly.

---

## 3. Final Verdict

**Status: PASS**
Zero regressions detected. The addition of Phase 2.1 dataloader modules has caused no side effects in the core architecture.
