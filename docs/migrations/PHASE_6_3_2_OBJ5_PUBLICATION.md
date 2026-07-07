# Phase 6.3.2 Objective 5 — Publication Integrity (Fail-Closed)

**Date:** 2026-07-06  
**Scope:** Block publication when evidence is missing, synthetic, or from failed runs.

## Summary

Publication pipeline now fails closed on non-MEASURED provenance, registry failure rows, and missing metrics. Mock metric fallback was removed from campaign dispatch. Figure generation no longer writes placeholder files without matplotlib.

## Code Changes

| Module | Change |
|--------|--------|
| `research/publication_manager.py` | Block on `failures` table; certificate/final report gated |
| `research/paper_figures.py` | Fail closed without matplotlib (no mock placeholders) |
| `replay_campaign.py` | Always verify claim provenance chain before success |
| `research/publication_audit.py` | Runtime gate probes + report generator |
| `tests/test_publication_integrity_audit.py` | OBJ5 audit tests |

## Validation

```powershell
python -m pytest tests/test_publication_integrity_audit.py tests/test_phase_6_3_1b_integrity.py tests/test_phase_6_3.py -q
python -c "from research.publication_audit import write_publication_integrity_report; write_publication_integrity_report('reports/scientific_integrity_audit/Publication_Integrity_Report.md')"
```

## Report

Measured results: `reports/scientific_integrity_audit/Publication_Integrity_Report.md`

## Not Changed (this objective)

- Full training→DB bridge for all runners (partial via `_record_measured_training_outcome`)
- Replay checksum hardening (Objective 6)
