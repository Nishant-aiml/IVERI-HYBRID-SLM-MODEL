# IVERI Core — Evidence Integrity Report
Generated: 2026-07-26T07:04:37Z  |  Auditor: Automated IEEE/JMLR/USPTO Auditor

> **Policy**: ZERO FABRICATION. Every number from executed experiments.
> **Warning**: This report describes the actual state of evidence, not a desired state.

## Artifact Inventory
Total artifacts found: 642

| Filename | Path | Size (B) | SHA256 | Status |
|----------|------|----------|--------|--------|
| tier1_ablations_and_failures.json | `scientific_campaigns\stage_c\phase_5_6\tier1_ablations_and_failures.json` | 10788 | `61f6c006c771` | WARN |
| latency_breakdown.json | `scientific_campaigns\stage_c\profiling\latency_breakdown.json` | 433 | `a98274acc7ad` | OK |
| lr_warmup_sweep_results.json | `scientific_campaigns\stage_c\phase_7\lr_warmup_sweep_results.json` | 2275 | `0329955c94e9` | OK |

## Validation Issues

| File | Issue | Severity | Resolution |
|------|-------|----------|------------|
| tier1_ablations_and_failures.json | PYTHON_NaN x5 | INFO | Auto-corrected (null substitution) |
| tier1_ablations_and_failures.json | PYTHON_Infinity x25 | INFO | Auto-corrected (null substitution) |

## Missing Expected Artifacts

The following experiment results were NOT found anywhere in the repository:

| Missing Artifact | Required For | Experiment Command |
|-----------------|-------------|-------------------|
| `phase_4/titans_context_sweep.json` | CLAIM-005, CLAIM-006 | `python scratch/stage_c_titans_context_sweep.py` |
| `router_analysis/router_entropy_log.json` | Paper-1, Patent-1 | `python scratch/stage_c_router_instrumentation.py` |
| `long_training/loss_curves.json` | CLAIM-004 resolution | `python scratch/stage_c_long_run.py` |
| `phase_3/adaptive_mor_results.json` | Patent-3 enablement | Implement v0.92 first |
| `stage_a/` any results | Historical baseline | Not found in repository |
| `stage_b/` any results | Historical baseline | Not found in repository |