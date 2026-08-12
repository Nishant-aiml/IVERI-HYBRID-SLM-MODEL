# IVERI CORE — Audit Issue Registry

**Status:** ACTIVE REMEDIATION  
**Last Updated:** 2026-07-24  
**Audit Source:** IEEE + Patent Readiness Audit, 18 Risk Assessment Audit, Source Code Audit, Test Audit

---

## 1. Issue Overview

| Issue ID | Severity | Category | Summary | Status |
|---|---|---|---|---|
| `ISSUE-ENG-001` | 🔴 CRITICAL | Engineering | Golden Checkpoint file `checkpoint_golden.pt` is 0 bytes | OPEN |
| `ISSUE-ENG-002` | 🔴 CRITICAL | Engineering | Datasets in `data/raw/` and `data/certified/` are toy mock files (~142 KB total) | OPEN |
| `ISSUE-ENG-003` | 🟡 HIGH | Engineering | Parameter count mismatch: default config produces 36.6M/455K vs 10M spec | OPEN |
| `ISSUE-ENG-004` | 🟡 HIGH | Engineering | Requirements.txt lists unused packages (`mamba-ssm`, `flash-attn`, `rotary-emb`) | OPEN |
| `ISSUE-SCI-001` | 🔴 CRITICAL | Scientific | `BENCHMARK_SUMMARY.md` claims "100% PASS" while 7 Phase 10 reports are 2-line stubs | OPEN |
| `ISSUE-SCI-002` | 🔴 CRITICAL | Scientific | `04_training_curves.md` reports `std ± 0.0000` across seeds (single-seed / duplicated data) | OPEN |
| `ISSUE-SCI-003` | 🔴 CRITICAL | Scientific | 12 of 13 publication figures in `paper/paper_figures/` are placeholder `y=5-x` plots | OPEN |
| `ISSUE-SCI-004` | 🔴 CRITICAL | Scientific | `table4_ablation.csv` and `table5_statistics.csv` contain only 1 row each | OPEN |
| `ISSUE-SCI-005` | 🔴 CRITICAL | Scientific | `RESEARCH_LOG.md` claims p=0.0003, ECE=0.0452, HumanEval 100% without raw data files | OPEN |
| `ISSUE-DOC-001` | 🟡 HIGH | Documentation | Test count claim in `task.md` ("683 pass") contradicts actual `pytest` output (640 pass) | OPEN |
| `ISSUE-DOC-002` | 🟡 HIGH | Documentation | Version mismatch (`VERSION`: 1.0.0, `pyproject.toml`: 0.1.0, `README.md`: 0.2.0-byte-vocab) | OPEN |
| `ISSUE-DOC-003` | 🟡 HIGH | Documentation | `QUICKSTART.md` claims Python 3.11+ minimum while `pyproject.toml` defines `>=3.10` | OPEN |
| `ISSUE-BENCH-001`| 🔴 CRITICAL | Benchmark | No execution logs for multi-seed baseline runs exist in `experiments/` | OPEN |

---

## 2. Detailed Issue Records

### ISSUE-ENG-001: Golden Checkpoint 0-Byte File
- **Discovery Source:** Artifact Audit (`reports/phase_3_6/Paper_Figures/checkpoint_golden.pt`)
- **Severity:** CRITICAL
- **Root Cause:** Placeholder checkpoint file was created as an empty string/file during Phase 3.6 script execution.
- **Affected Files:** `reports/phase_3_6/Paper_Figures/checkpoint_golden.pt`, `research/experiments_run.db`
- **Reproduction Steps:** Run `Get-ChildItem reports\phase_3_6\Paper_Figures\checkpoint_golden.pt` -> length is 0.
- **Proposed Fix:** Delete 0-byte file or replace with valid saved PyTorch checkpoint dictionary containing state_dict, step, config, and random seed. Flag all 0-byte checkpoint loaders to raise `ValidationError`.

### ISSUE-ENG-002: Mock Dataset Scale
- **Discovery Source:** Dataset Audit (`data/curriculum/phase_a_manifest.json`)
- **Severity:** CRITICAL
- **Root Cause:** Data prep scripts were run with micro synthetic generators, producing ~20 KB per dataset (142 KB total).
- **Affected Files:** `data/raw/`, `data/certified/`, `data/curriculum/phase_a_manifest.json`
- **Reproduction Steps:** Read `phase_a_manifest.json` -> `total_certified_gb = 0.000142`.
- **Proposed Fix:** Execute Phase 8 data pipeline (`data/pipeline/certification.py`) on full approved datasets and enforce Dataset Scale Gate (`STATUS = ENGINEERING VALIDATION ONLY` for <1 GB data).

### ISSUE-SCI-001: Fake Benchmark Summary Claims
- **Discovery Source:** IEEE Audit (`reports/phase_10/BENCHMARK_SUMMARY.md`)
- **Severity:** CRITICAL
- **Root Cause:** Automated report script wrote "VALID (100% PASS)" despite 7 underlying reports being 2-line placeholder stubs.
- **Affected Files:** `reports/phase_10/BENCHMARK_SUMMARY.md`, `reports/phase_10/BENCHMARK_CERTIFICATE.md`
- **Reproduction Steps:** View `05_efficiency.md`, `06_ablation.md`, `07_statistics.md` -> 2 lines each.
- **Proposed Fix:** Replace fake summary cards with honest status headers ("Scientific Status: NOT GENERATED") until real campaign completion.

### ISSUE-SCI-002: Multi-Seed Zero Standard Deviation
- **Discovery Source:** IEEE Audit (`reports/phase_10/04_training_curves.md`)
- **Severity:** CRITICAL
- **Root Cause:** Report script formatted single-seed values into a `Mean ± Std` template without running seeds 123 or 3407.
- **Affected Files:** `reports/phase_10/04_training_curves.md`, `reports/phase_10/phase10_summary.json`
- **Reproduction Steps:** View `04_training_curves.md` -> `Transformer: 5.5906 ± 0.0000`.
- **Proposed Fix:** Replace false multi-seed claims with `"Single-seed engineering run only"` until Phase 3B multi-seed runs finish.

### ISSUE-SCI-003: Placeholder Publication Figures
- **Discovery Source:** Image Audit (`paper/paper_figures/`)
- **Severity:** CRITICAL
- **Root Cause:** `research/paper_figures.py` generated `y = 5-x` straight lines for figures 1, 3–12.
- **Affected Files:** `paper/paper_figures/fig01_benchmark_chart.png` through `fig12_benchmark_chart.png`
- **Reproduction Steps:** View `fig01_benchmark_chart.png` -> straight diagonal line labeled "Publication Figure 1".
- **Proposed Fix:** Replace placeholder plots with visual notice: `"Placeholder Removed: Experiment not executed"`. Generate real matplotlib figures only from measured CSV logs.

### ISSUE-DOC-001: Contradictory Test Counts
- **Discovery Source:** Documentation Audit (`task.md`)
- **Severity:** HIGH
- **Root Cause:** Documentation claimed 683 passing tests, whereas pytest collects 655 items (640 pass, 4 skip, 11 deselected).
- **Affected Files:** `task.md`, `README.md`
- **Reproduction Steps:** Run `pytest tests/ -q` -> `640 passed, 4 skipped, 11 deselected`.
- **Proposed Fix:** Synchronize all docs to state exact pytest numbers (`640 pass, 4 skip, 11 deselected`).
