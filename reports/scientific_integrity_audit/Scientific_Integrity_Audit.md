# Experiment Provenance & Scientific Statistics

**Audit Date:** 2026-07-06  
**Repository:** IVERI CORE  
**Covers:** Audit Objectives 2 (Experiment Provenance) and 7 (Scientific Statistics)  
**Mode:** Read-only — no code modified

---

## Verdicts

| Objective | Verdict |
|-----------|---------|
| 2. Experiment Provenance | **FAIL** |
| 7. Scientific Statistics | **FAIL** |

---

# Part A — Experiment Provenance

## Required Traceability Chain

Every metric, benchmark, checkpoint, figure, table, and report must trace to:

| Provenance Field | Required | Present | Gap |
|------------------|----------|---------|-----|
| Run UUID | Yes | Partial | In `experiments.experiment_id`; naming inconsistent across artifacts |
| Seed | Yes | Partial | In `experiments.random_seed`; not propagated to figure data |
| Checkpoint | Yes | **Missing** | Model `.pt` not in SQLite; placeholder golden hash |
| Dataset hash | Yes | Partial | File manifests only; `datasets` table empty |
| Prompt hash | Yes | **Missing** | Booleans only in `benchmark_integrity` |
| Git hash | Yes | Partial | In experiments row; replay uses placeholder `sha_git_lock_phase6_3_abc` |
| Environment hash | Yes | **Missing** | `environment_info` JSON in `release_manifests`; no canonical hash |
| Database row | Yes | Partial | Rows exist but may contain mock data |

---

## Provenance Gaps by Artifact Type

### Metrics

- **Writer:** `_log_mock_metrics()` (primary) or nothing (real pretrain returns True without DB write)
- **Reader:** Publication pipeline does **not** read `metrics` table
- **Gap:** No provenance column; mock indistinguishable from real

### Benchmarks

- **Writer:** Hardcoded `HumanEval: 0.88`, `NeedleInHaystack: 0.95` in `campaign_runner.py` lines 310–322
- **Reader:** LaTeX tables use literals, not `benchmark_runs` query
- **Gap:** Scores not tied to eval runner output or checkpoint

### Figures

- **Writer:** `generate_and_verify_all_assets()` with synthetic loss curves
- **Registry:** `paper_assets` links `experiment_id` to file path
- **Gap:** File exists but data is not from linked experiment

### Tables

- **Writer:** `_generate_publication_ready_tables()` hardcoded LaTeX
- **Gap:** No DB aggregation step

### Reports

- **Writer:** `compile_reports_from_db()` — name misleading; only counts experiments
- **Gap:** Report content not derived from DB metrics

### Certificates

- **Writer:** `generate_phase_certificate()` — hardcoded success claims
- **Gap:** No query of `failures`, `metrics`, or hypothesis outcomes

---

## Experiment ID Inconsistency

| Source | ID Pattern |
|--------|------------|
| `campaign_manifest.json` | `IVERI_Phase5_pretrain_Seed42_IVERI_Run001` |
| `Evidence_Index.md` | `IVERI_2026_07_04_Seed42_IVERI_Run001` |
| `evidence_graph.json` | `IVERI_Phase6_2_pretrain_Seed42_IVERI_Run001` |
| Campaign ID in manifest | `IVERI_CAMPAIGN_2026_PHASE5` |
| Certificate | `IVERI_CAMPAIGN_2026_PHASE6_3_PAPER` |

**Impact:** Cross-artifact traceability is broken even when DB rows exist.

---

## Disconnected Provenance Systems

| System | Location | Wired to ExperimentRegistry? |
|--------|----------|-------------------------------|
| ExperimentManager (training logs) | `training/` | **No** |
| ProvenanceTracker | `data/pipeline/provenance.py` | **No** |
| ExperimentManifestGenerator | `research/experiment_manifest.py` | Partial (`git_sha: "unknown"`) |
| BenchmarkIntegrityFramework | `research/benchmark_integrity.py` | Partial (not persisted as hashes) |

---

# Part B — Scientific Statistics

## Library Implementation — Partial PASS

**File:** `research/statistics.py` — `ResearchStatisticalValidator`

| Method | Implemented | Unit Tested | Used in Phase 6.3 Publisher |
|--------|-------------|-------------|----------------------------|
| Shapiro-Wilk | Yes | Yes | **No** |
| Paired t-test | Yes | Yes | **No** |
| Wilcoxon signed-rank | Yes | Yes | **No** |
| Holm–Bonferroni | Yes | Yes | **No** |
| Bootstrap CI | Yes | Yes | **No** |
| Cohen's d | Yes | Yes | **No** |
| Cliff's Δ | Yes | Partial | **No** |

**Alternate path (not Phase 6.3):** `research/compare_runs.py` runs t-test, Wilcoxon, Cohen's d, bootstrap on paired val-loss — omits Shapiro, Holm, Cliff's Δ.

---

## Issue S1: Phase 6.3 publisher does not compute statistics

| Field | Value |
|-------|-------|
| **Severity** | CRITICAL |
| **Evidence** | `compile_reports_from_db()` — no import of `ResearchStatisticalValidator` |
| **Result** | `Statistics_Report.md` contains boilerplate only; no p-values, no effect sizes |

---

## Issue S2: Publication artifacts contradict each other

| Artifact | Hypothesis Status | Statistics |
|----------|-------------------|------------|
| `Hypothesis_Report.md` | All H1–H10 **PENDING** | None computed |
| `Evidence_Index.md` | All **Pending** | "Expected" p-values only |
| `statistical_significance_table.tex` | H1–H4 **SUPPORTED** | Hardcoded p=0.012 etc. |
| `Phase_6_3_Certificate.md` | H1–H10 **All SUPPORTED** | None |
| `Reproducibility_Report.md` | Statistics Reproduced ✓ | False |

---

## Issue S3: Normality-gated test selection not implemented

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Evidence** | `FINAL_REPORT.md` claims *"Paired t-test / Wilcoxon (normality-checked via Shapiro-Wilk)"*; no code path calls Shapiro then selects test |
| **Recommendation** | Implement selection in single `compute_hypothesis_statistics()` function |

---

## Issue S4: Wilcoxon p-value formula suspect

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Evidence** | `statistics.py` lines 133–134: `p_val = 2.0 * self._normal_cdf(z)` — two-tailed should use `2 * (1 - Φ(|z|))` |
| **Impact** | Incorrect p-values when library is used |

---

## Issue S5: Research methodology doc incomplete

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Evidence** | `docs/research/Research_Methodology.md` documents 4 methods; Phase 6.3 claims 7 |
| **Recommendation** | Align methodology doc with actual (intended) pipeline |

---

## Required Single Source of Truth (Not Present)

```
experiments.db (per-seed metrics)
  → ResearchStatisticalValidator.compute_all()
  → statistics_results.json  (canonical)
       ├→ Statistics_Report.md
       ├→ Hypothesis_Report.md
       ├→ statistical_significance_table.tex
       ├→ Evidence_Index.md
       ├→ evidence_graph.json
       └→ Phase_6_3_Certificate.md
```

**Current:** Each artifact generated independently with hardcoded or boilerplate content.

---

## Files Audited

- `research/statistics.py`
- `research/compare_runs.py`
- `research/publication_manager.py`
- `research/campaign_runner.py`
- `reports/phase_6_3/statistics/`
- `reports/phase_6_3/publication/Paper_Tables/statistical_significance_table.tex`
- `docs/research/Research_Methodology.md`

---

## Overall Assessment

Neither provenance nor statistics objectives are met. The infrastructure exists but the Phase 6.3 publication path bypasses it entirely.
