# IVERI CORE — Scientific Claim Matrix

**Date:** 2026-07-24  
**Audit Rule:** Rule 1 — Verify Every Claim  

---

## 1. Master Claims Matrix

| ID | Claim Statement | File Location | Line | Evidence in Repo | Audit Status | Required Fix |
|---|---|---|---|---|---|---|
| `CLM-001` | "VALID (100% PASS)" | `reports/phase_10/BENCHMARK_SUMMARY.md` | L5 | Reports 05–07 are 2-line stubs | **FALSE** | Replaced with `"Scientific Status: NOT GENERATED"` |
| `CLM-002` | "Mean ± Std" across 3 seeds = 0.0000 | `reports/phase_10/04_training_curves.md` | L7-L11 | CSV table contains only Seed 42 | **FALSE** | Relabeled as `"Single-seed engineering run only"` |
| `CLM-003` | "Seeds Processed: 3 (`42`, `123`, `3407`)" | `reports/phase_10/BENCHMARK_SUMMARY.md` | L7 | `table3_loss.csv` has Seed 42 only | **UNSUPPORTED** | Relabeled to reflect single-seed pilot |
| `CLM-004` | "Active Parameter Delta: 9.28% (<= 10.0% Gate — PASS)" | `reports/phase_10/phase10_summary.json` | L59 | Calculated FLOP formulas in json | **VERIFIED** | Retained (parameter formula is exact) |
| `CLM-005` | "683 pass" | `task.md` | L4 | `pytest` reports 640 pass, 4 skip | **FALSE** | Corrected to `640 pass, 4 skip, 11 deselected` |
| `CLM-006` | "paired t-test p-value = 0.0003" | `research_log/RESEARCH_LOG.md` | L579 | No raw observation log file exists | **UNSUPPORTED** | Labeled as preliminary micro-test observation |
| `CLM-007` | "Expected Calibration Error (ECE): 0.0452" | `research_log/RESEARCH_LOG.md` | L580 | No calibration dataset file exists | **UNSUPPORTED** | Labeled as preliminary micro-test observation |
| `CLM-008` | "HumanEval pass@1: 100.0%, MBPP pass@1: 100.0%" | `research_log/RESEARCH_LOG.md` | L500 | Tested on synthetic 10-prompt subset | **UNSUPPORTED** | Clarified "Synthetic 10-prompt subset" |
| `CLM-009` | "Reproducibility Score: 100.0%" | `research_log/RESEARCH_LOG.md` | L582 | Self-assessed by script | **FALSE** | Replaced with raw reproduction commands |
| `CLM-010` | "SFT Loss Masking Fix: SFTByteDataset returns (x, y, loss_mask)" | `implementation_plan.md` | L19 | `data/pipeline/dataloader.py` L325 | **VERIFIED** | Retained with code citation |
| `CLM-011` | "Byte-entropy-native hybrid SLM combining BLT+Titans+Mamba2+MoR+MoE" | `README.md` | L8 | Modules in `model/` (18 files) | **VERIFIED** | Retained (code implementation verified) |
| `CLM-012` | "Option C entropy-driven routing" | `VERSION` | L5 | `ByteEntropyModel` drives 4 consumers | **VERIFIED** | Retained with code citation |

---

## 2. Claim Verification Policy

No unsupported claim may remain in active documentation or publication cards. Any claim not backed by raw CSV/JSON observation files is automatically flagged `UNSUPPORTED` or `FALSE` and remediated.
