# Phase 6.3.1B — Database Integrity Migration Notes

**Date:** 2026-07-06  
**Scope:** `experiments.db` write-path hardening (no architecture or training changes)

## Summary

Phase 6.3.1B adds fail-closed integrity rules on every registry write, an auditable write log, foreign-key validation, schema checks, and removes remaining hardcoded benchmark values from publication outputs.

## Schema Changes

### New table: `db_write_audit`

| Column        | Type  | Description                          |
|---------------|-------|--------------------------------------|
| audit_id      | TEXT  | Primary key (`table:key:timestamp`)  |
| table_name    | TEXT  | Target table                         |
| operation     | TEXT  | INSERT / UPDATE / UPSERT / …         |
| record_key    | TEXT  | Primary record identifier            |
| payload_json  | TEXT  | Serialized write payload             |
| timestamp     | REAL  | Unix timestamp                       |

### New index

```sql
CREATE UNIQUE INDEX IF NOT EXISTS idx_metrics_experiment_step
ON metrics(experiment_id, step);
```

Ensures each metric row maps to exactly one `(experiment_id, step)` pair.

## Behaviour Changes

| Rule | Before 6.3.1B | After 6.3.1B |
|------|---------------|--------------|
| Duplicate Run UUID | `INSERT OR REPLACE` overwrote rows | `INSERT` only; duplicate `experiment_id` raises `RegistryIntegrityError` |
| `FAILED → COMPLETED` | Allowed via `update_experiment_status` | **Blocked** |
| `PENDING → COMPLETED` | Allowed | **Blocked** (must pass through `RUNNING`) |
| MEASURED metric overwrite | Any label could insert at same step | Non-`MEASURED` cannot replace `MEASURED` |
| MEASURED benchmark overwrite | `INSERT OR REPLACE` | Non-`MEASURED` cannot replace `MEASURED` |
| Child rows without parent | FK errors at SQLite level only | Explicit pre-write FK validation + audit |
| Publication benchmark registry | Hardcoded HumanEval / Needle scores | Loaded exclusively from `benchmark_registry` table |
| Release manifest benchmarks | Hardcoded version map | Loaded from `benchmark_registry` + `experiments` |

## Status Transition Graph

```
PENDING  → RUNNING | FAILED
RUNNING  → COMPLETED | SUCCESS | FAILED
FAILED   → (terminal)
COMPLETED → (terminal)
SUCCESS  → (terminal)
```

Initial registration may still set `status='COMPLETED'` at insert time for test seeding when `provenance_label='MEASURED'`.

## Campaign Runner Adjustment (orchestration only)

Before each training dispatch, status is set to `RUNNING`. This satisfies the `PENDING → COMPLETED` guard without modifying training code.

## Upgrading an Existing Database

1. Back up `experiments.db`.
2. Open the database with any code path that constructs `ExperimentRegistry` — `_init_db()` will:
   - create `db_write_audit` if missing
   - add `idx_metrics_experiment_step` if missing
   - run `validate_schema()`
3. **Duplicate metrics:** Legacy DBs are auto-deduplicated on registry init (keeping the earliest row per `(experiment_id, step)`). Manual dedup SQL is still available if needed:

```sql
-- Example dedup keeping lowest rowid per (experiment_id, step)
DELETE FROM metrics
WHERE rowid NOT IN (
  SELECT MIN(rowid) FROM metrics GROUP BY experiment_id, step
);
```

4. Re-run `replay_campaign.py` to confirm provenance chains.

## New Module

- `research/registry_integrity.py` — validators, audit helper, schema definitions

## New Tests

- `tests/test_phase_6_3_1b_integrity.py` — transition guards, overwrite protection, audit trail, FK checks, DB-only publication

## Breaking Changes for Callers

- Calling `register_experiment()` twice with the same `experiment_id` now raises `RegistryIntegrityError`.
- `update_experiment_status(id, "COMPLETED")` requires current status `RUNNING` (or register with `status="COMPLETED"` at insert).
- `PublicationManager.generate_benchmark_registry()` requires pre-populated `benchmark_registry` rows.

## Rollback

Restore the pre-6.3.1B `research/experiment_registry.py` and remove `db_write_audit` / the metrics unique index if a full rollback is required. No training or model files are affected.
