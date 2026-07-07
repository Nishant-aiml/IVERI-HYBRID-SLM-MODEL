# Phase 6.3.2 Objective 6 — Replay Integrity (Fail-Closed)

**Date:** 2026-07-06  
**Scope:** Harden campaign replay across checkpoints, reports, figures, tables, cards, and lineage.

## Summary

Replay now pre-flights registry integrity before publication, verifies H1–H10 claim provenance chains with MEASURED metrics, rejects mock figure placeholders, and exits non-zero on any failure. Reviewer scorecard reports honest pass/fail labels.

## Code Changes

| Module | Change |
|--------|--------|
| `research/replay_integrity.py` | Registry, claim-chain, and figure verification gates |
| `replay_campaign.py` | Pre-flight registry check; DB-sourced checkpoint hash; fail-closed exit |
| `research/replay_audit.py` | Runtime audit harness + report generator |
| `tests/test_replay_integrity_audit.py` | OBJ6 audit tests |

## Validation

```powershell
python -m pytest tests/test_phase_6_3.py tests/test_replay_integrity_audit.py tests/test_production_campaign.py -q
python -c "from research.replay_audit import write_replay_integrity_report; write_replay_integrity_report('reports/scientific_integrity_audit/Replay_Integrity_Report.md')"
```

## Report

Measured results: `reports/scientific_integrity_audit/Replay_Integrity_Report.md`

## Not Changed (this objective)

- Byte vocabulary collisions (Objective 7)
- Documentation sync across README/CHANGELOG (Objective 8)
