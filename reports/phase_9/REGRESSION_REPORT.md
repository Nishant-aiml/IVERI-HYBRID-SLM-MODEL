# Phase 9 — Regression Report

**Date:** 2026-07-11  
**Command:** `python -m pytest tests/ -v --tb=line -q`  
**Duration:** 592.6 seconds

---

## Summary

| Metric | Value |
|--------|-------|
| Passed | **634** |
| Failed | **1** |
| Skipped | 4 |
| Deselected | 11 |
| Phase 9 tests | 3/3 passed |

---

## Failure

```
FAILED tests/test_phase8_integration.py::test_full_pipeline_integration
AssertionError: assert 0 == 10
  DownloadResult.num_docs for tinystories download
```

**Impact on Phase 9:** Phase 8 full pipeline not verified; certified data cannot be produced until download works.

---

## Phase 9 Test Coverage

```
tests/test_phase9_campaign.py::test_environment_lock PASSED
tests/test_phase9_campaign.py::test_model_init_validation PASSED
tests/test_phase9_campaign.py::test_stability_early_warnings_and_failure_db PASSED
```

**Gap:** No regression test executes `CampaignOrchestrator.run_campaign()` end-to-end.

---

## Phase 8 Submodule Tests (targeted run)

```
test_phase8_integration FAILED (1)
test_phase8_certifier PASSED
test_phase8_fingerprint PASSED
test_phase8_curriculum PASSED
```

---

## Verdict

Regression suite is **stable for Phase 7 infrastructure** but **not Phase 9 complete**. One Phase 8 integration failure blocks data pipeline confidence.
