# Statistics Consistency Report (Phase 6.3.1G)

**Generated:** 2026-07-07T06:09:05Z  
**Protocol:** Phase-6.3.1G  

## Executive Verdict

**Single statistics pipeline:** `PASS`

## Canonical Methods

All seven methods must flow through `ResearchStatisticalValidator.compute_paired_hypothesis_statistics()`:

- `shapiro_wilk`
- `paired_t_test`
- `wilcoxon`
- `holm_bonferroni`
- `bootstrap`
- `cohens_d`
- `cliffs_delta`

**Bundle covers all methods:** `True`
**Golden bundle self-test:** `True`

## Consumer Audit

| File | Uses canonical bundle | Forbidden inline calls |
|------|:---------------------:|:----------------------:|
| `research/compare_runs.py` | True | none |
| `research/publication_manager.py` | True | none |
| `research/generate_reports.py` | True | none |

## Duplicate Calculation Detection

No duplicated inline statistics calculations detected in consumer modules.

## Raw JSON

```json
{
  "protocol_version": "Phase-6.3.1G",
  "timestamp_utc": "2026-07-07T06:09:05Z",
  "production_verdict": "PASS",
  "canonical_methods": [
    "shapiro_wilk",
    "paired_t_test",
    "wilcoxon",
    "holm_bonferroni",
    "bootstrap",
    "cohens_d",
    "cliffs_delta"
  ],
  "bundle_covers_all_methods": true,
  "golden_bundle_ok": true,
  "consumers": [
    {
      "rel_path": "research/compare_runs.py",
      "uses_canonical_bundle": true,
      "forbidden_inline_calls": []
    },
    {
      "rel_path": "research/publication_manager.py",
      "uses_canonical_bundle": true,
      "forbidden_inline_calls": []
    },
    {
      "rel_path": "research/generate_reports.py",
      "uses_canonical_bundle": true,
      "forbidden_inline_calls": []
    }
  ],
  "duplicate_calculation_violations": []
}
```
