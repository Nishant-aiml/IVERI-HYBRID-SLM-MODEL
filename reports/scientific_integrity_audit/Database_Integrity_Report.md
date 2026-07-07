# Database Integrity Report

**Audit Date:** 2026-07-06  
**Repository:** IVERI CORE  
**Mode:** Read-only — no code modified

---

## Verdict: **FAIL**

SQLite schema in `research/experiment_registry.py` is well-shaped but **not enforced** by campaign logic. Failed experiments can appear `COMPLETED`. Mock metrics are indistinguishable from real metrics. Several tables are never populated. Foreign keys are declared but not enabled.

---

## Schema Overview

**File:** `research/experiment_registry.py` — `ExperimentRegistry._init_db()`

| Table | Purpose | Populated by campaign? |
|-------|---------|------------------------|
| `experiments` | Run registry | Yes |
| `metrics` | Training metrics | Yes (mostly via `_log_mock_metrics`) |
| `hardware` | GPU/energy telemetry | Rarely |
| `datasets` | Dataset hash FK | **No** |
| `checkpoints` | Checkpoint registry | Partial (DB archive only) |
| `failures` | Failure log | Yes (but status still COMPLETED) |
| `artifacts` | Generic artifacts | Partial |
| `notes` | Annotations | Rarely |
| `paper_assets` | Figures/tables | Yes |
| `benchmark_registry` | Benchmark definitions | **No** |
| `benchmark_runs` | Benchmark scores | Yes (hardcoded scores) |
| `benchmark_integrity` | Hash validation flags | Yes (all `True`) |
| `benchmark_artifacts` | Eval outputs | Partial |
| `publication_runs` | Report compilation | Partial |
| `release_manifests` | Release metadata | Yes |

---

## Issue D1: Failed experiments marked COMPLETED

| Field | Value |
|-------|-------|
| **Severity** | CRITICAL |
| **Impact** | Registry reports success while `failures` table has entries |
| **Likelihood** | Certain on training exception |
| **Evidence** | `_attempt_real_pretraining` logs failure (lines 601–609) then returns `False`; caller calls `_log_mock_metrics` and `update_experiment_status(exp_id, "COMPLETED")` (lines 396–410) |
| **Root Cause** | No status invariant linking failures to experiment status |
| **Recommended Fix** | Set `FAILED` when `log_failure` called; block certificate if any `FAILED` or any failure row for campaign experiments |
| **Estimated Effort** | 1 day |

---

## Issue D2: No mock/synthetic provenance column on metrics

| Field | Value |
|-------|-------|
| **Severity** | CRITICAL |
| **Impact** | Cannot distinguish `2.4 * (0.9**i)` synthetic loss from real training |
| **Evidence** | `log_metrics()` schema: `experiment_id, step, train_loss, val_loss, perplexity, accuracy` — no `source` or `provenance` field |
| **Recommended Fix A** | Add `metric_source TEXT` column with CHECK constraint |
| **Recommended Fix B** | Separate `mock_metrics` table quarantined from publication queries |
| **Recommendation** | Fix A |

---

## Issue D3: INSERT OR REPLACE can overwrite experiment metadata

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Impact** | Re-running campaign with same `experiment_id` silently replaces prior run |
| **Evidence** | `register_experiment()` line 232: `INSERT OR REPLACE INTO experiments` |
| **Recommendation** | Use `INSERT` with conflict error, or version increment on replace |

---

## Issue D4: Foreign keys not enforced

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Impact** | Orphan `metrics`/`benchmark_runs` rows possible |
| **Evidence** | Schema declares `FOREIGN KEY` but no `PRAGMA foreign_keys=ON` in connection setup |
| **Recommendation** | Enable FK enforcement in `_get_connection()` |

---

## Issue D5: benchmark_registry never populated

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Impact** | `benchmark_runs.benchmark_id` references free-text strings (`"HumanEval"`) without registry row |
| **Evidence** | `campaign_runner.py` lines 314–322 logs runs with `benchmark_id=b_name`; no `register_benchmark()` call in campaign path |
| **Recommendation** | Register benchmarks before logging runs; enforce FK |

---

## Issue D6: datasets table unused

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Impact** | Dataset hashes in manifests not FK-linked to experiments |
| **Evidence** | `datasets` table created in schema; no campaign writer found |
| **Recommendation** | `register_dataset()` per campaign stage with hash; link via experiment tags or junction table |

---

## Issue D7: No schema versioning

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Impact** | Ad-hoc migrations (e.g. `golden.py` ALTER TABLE) risk inconsistency |
| **Evidence** | Only `CREATE TABLE IF NOT EXISTS`; no `schema_version` table |
| **Recommendation** | Add migration framework with version stamp |

---

## Issue D8: benchmark_integrity stores booleans, not hashes

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Impact** | Cannot audit actual prompt/dataset SHA-256 from DB |
| **Evidence** | `benchmark_integrity` columns: `prompt_hash_ok INTEGER` etc.; campaign sets all `True` (lines 323–329) without computation |
| **Recommendation** | Store actual hash values; compute via `BenchmarkIntegrityFramework` |

---

## Status State Machine (Actual)

```
register_experiment(status="PENDING")
  → training attempt
  → [failure logged to failures table]
  → _log_mock_metrics()
  → update_experiment_status("COMPLETED")   # always, unless health PAUSED
```

**Missing transitions:** `FAILED`, `MOCK_COMPLETED`, `PARTIAL`

---

## Files Audited

- `research/experiment_registry.py`
- `research/campaign_runner.py`
- `research/experiment_scheduler.py` (has `FAILED` status — unused by campaign stages)
- `tests/test_phase_6_2.py`

---

## Overall Assessment

Database integrity objective **not met**. Schema supports scientific provenance but campaign logic undermines it by marking failed/mock runs as completed and logging unverifiable benchmark integrity flags.
