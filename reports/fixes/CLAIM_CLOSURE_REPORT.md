# IVERI CORE — Unsupported Claims Closure Report

**Date:** 2026-07-24  
**Audit Reference:** IEEE + Patent Readiness Audit  

---

## 1. Executive Summary

This document records the systematic resolution of all scientific, engineering, and documentation claims identified as unsupported, false, or unverified during the repository audit.

Every flagged claim has been remediated:
1. **False claims** have been purged or corrected to match empirical codebase metrics.
2. **Unsubstantiated claims** have been relabeled as `"Engineering Verification Only"` or `"Conceptual / Preliminary"`.
3. **Placeholder reports & figures** have been marked with explicit non-execution headers (`"Scientific Status: NOT GENERATED"`).

---

## 2. Claim Resolution Matrix

| Claim Description | Document / Location | Previous Audit Status | Resolution Applied | Verification Evidence |
|---|---|---|---|---|
| "VALID (100% PASS)" | `reports/phase_10/BENCHMARK_SUMMARY.md` | **FALSE CLAIM** | Replaced with `"Scientific Status: NOT GENERATED"` until campaign completion | `BENCHMARK_SUMMARY.md` line 5 |
| "Mean ± Std" = 0.0000 across seeds | `reports/phase_10/04_training_curves.md` | **FALSE CLAIM** | Replaced with `"Single-seed engineering run only"` header | `04_training_curves.md` line 5 |
| "Seeds Processed: 3" | `reports/phase_10/BENCHMARK_SUMMARY.md` | **UNSUPPORTED** | Relabeled to reflect single-seed pilot | `BENCHMARK_SUMMARY.md` line 7 |
| "683 pass" | `task.md` | **FALSE CLAIM** | Updated to exact pytest output: `640 pass, 4 skip, 11 deselected` | `task.md` line 12 |
| "Paired t-test p = 0.0003" | `research_log/RESEARCH_LOG.md` | **UNSUPPORTED** | Relabeled as qualitative pilot observation, missing raw logs | `RESEARCH_LOG.md` line 579 |
| "IVERI 22.96 vs Transformer 56.22" | `research_log/RESEARCH_LOG.md` | **UNSUPPORTED** | Flagged as 100-step micro-test observation on mock data | `RESEARCH_LOG.md` line 422 |
| "HumanEval pass@1: 100%" | `research_log/RESEARCH_LOG.md` | **UNSUPPORTED** | Relabeled to specify synthetic 10-prompt subset test | `RESEARCH_LOG.md` line 500 |
| "Reproducibility Score: 100.0%" | `research_log/RESEARCH_LOG.md` | **FALSE CLAIM** | Removed self-assessed score; replaced with raw reproduction instructions | `RESEARCH_LOG.md` line 582 |
| "Paper Bundle Status: Ready" | `reports/phase_10/BENCHMARK_SUMMARY.md` | **FALSE CLAIM** | Replaced with `"Paper Bundle Status: IN PROGRESS (Awaiting Real Campaign)"` | `BENCHMARK_SUMMARY.md` line 14 |
| Version mismatch (1.0.0 vs 0.1.0 vs 0.2.0) | `VERSION`, `pyproject.toml`, `README.md` | **CONTRADICTION** | Synchronized all version files to `1.0.0` | `VERSION`, `pyproject.toml`, `README.md` |
| Python minimum version (3.11+ vs 3.10+) | `QUICKSTART.md`, `pyproject.toml` | **CONTRADICTION** | Aligned QUICKSTART.md to state Python `>= 3.10` | `QUICKSTART.md` line 15 |
| Golden checkpoint 0-byte file | `reports/phase_3_6/Paper_Figures/checkpoint_golden.pt` | **CRITICAL BUG** | File purged; loaders updated to reject 0-byte files | `checkpoint_audit.md` |

---

## 3. Mandatory Disclaimers Applied

All Phase 10 report cards, paper figures, and baseline data tables have been updated with the following mandatory disclaimers:

```markdown
> [!IMPORTANT]
> **Engineering Verification Only**: The metrics and tables in this report represent micro-benchmarks and engineering pipeline validation runs on toy datasets (~142 KB). They are NOT suitable for scientific conclusions or comparative architectural claims. Full multi-seed campaigns on certified datasets are pending Phase 3B execution.
```

---

## 4. Verification & Audit Trail

- **Pytest Verification:** `python -m pytest tests/ -q` $\to$ **640 passed, 4 skipped, 11 deselected**.
- **Architecture Health:** `scripts/architecture_health.py` $\to$ **All 5 subsystems healthy**.
- **Zero Placeholder Leak Guarantee:** Automated script checks that no file claims `100% PASS` or `p < 0.05` without an accompanying raw CSV observation log.
