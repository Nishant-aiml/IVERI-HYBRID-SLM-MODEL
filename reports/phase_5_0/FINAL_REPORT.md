# IVERI CORE — Phase 5.0 Final Report Index

**Campaign ID:** IVERI_CAMPAIGN_2026_PHASE5  
**Generated:** 2026-07-05T03:41:15Z UTC  
**Protocol Version:** Phase5.0-v1.0  
**Output Directory:** `C:\Users\datta.000\Desktop\iveri core nexus\iveri-core\reports\phase_5_0`  

> This document is the master entry point for the Phase 5.0 empirical campaign.
> Every section below links to one of the 17 component scientific reports.
> All metrics in linked reports originate exclusively from `experiments.db` — no synthetic values.

---

## Report Index

| Report | Phase | Contents | Status |
|---|---|---|---|
| [Training_Report.md](./Training_Report.md) | Phase B Pretraining | Training loss, gradient norms, convergence telemetry | ✓ Present |
| [Baseline_Report.md](./Baseline_Report.md) | Phase B Baselines | Matched FLOPs/parameter Transformer, Mamba2, Hybrid comparisons | ✓ Present |
| [Ablation_Report.md](./Ablation_Report.md) | Phase C Ablations | Component ablations: Titans, BLT, MoR, MoE, Entropy Routing | ✓ Present |
| [Instruction_Report.md](./Instruction_Report.md) | Phase D — SFT | Stage 2 instruction tuning: MMLU-lite, IFEval, QA correctness | ✓ Present |
| [Coding_Report.md](./Coding_Report.md) | Phase D — Coding | Stage 3A coding: HumanEval, MBPP, LiveCodeBench pass@1/pass@5 | ✓ Present |
| [Alignment_Report.md](./Alignment_Report.md) | Phase D — Alignment | Stage 4 DPO/SimPO/IPO win rates and reward margins | ✓ Present |
| [Calibration_Report.md](./Calibration_Report.md) | Phase E — Calibration | ECE, MCE, Brier score, confidence histograms | ✓ Present |
| [Efficiency_Report.md](./Efficiency_Report.md) | Phase E — Efficiency | TTFT, tokens/sec/GPU, tokens/sec/TFLOP, Watts/token | ✓ Present |
| [Energy_Report.md](./Energy_Report.md) | Phase E — Energy & Carbon | kWh total, kgCO2e emissions, power efficiency | ✓ Present |
| [Long_Context_Report.md](./Long_Context_Report.md) | Phase E — Long Context | Needle-in-a-Haystack 2K–128K, multi-needle retrieval, 95% CI | ✓ Present |
| [Statistics_Report.md](./Statistics_Report.md) | Phase E — Statistics | Paired t-tests, Wilcoxon, Holm-Bonferroni, Cohen's d, Cliff's Δ | ✓ Present |
| [Hypothesis_Report.md](./Hypothesis_Report.md) | Phase E — Hypotheses | H1–H10 SUPPORTED / REFUTED / INCONCLUSIVE verdict table | ✓ Present |
| [Architecture_Validation_Report.md](./Architecture_Validation_Report.md) | Phase B+C Arch Validation | Structural validation decisions, scaling projections 35M–3B | ✓ Present |
| [Reproducibility_Report.md](./Reproducibility_Report.md) | Reproducibility | Environment lock, seeds, tokenizer parameters, verification checklist | ✓ Present |
| [Campaign_Report.md](./Campaign_Report.md) | Campaign Overview | Master execution statistics: runs, failures, durations, hardware | ✓ Present |
| [Evidence_Index.md](./Evidence_Index.md) | Evidence Index | Maps H1–H10 to figures, tables, experiment IDs, p-values | ✓ Present |
| [Executive_Summary.md](./Executive_Summary.md) | Executive Summary | High-level findings and strategic scaling recommendations | ✓ Present |

---

## Phase Execution Summary

| Phase | Description | Reports |
|---|---|---|
| **Phase A** | Pilot Verification Gate | Reproducibility_Report.md, Campaign_Report.md |
| **Phase B** | Production Pretraining (4 models × 5 seeds) | Training_Report.md, Baseline_Report.md, Architecture_Validation_Report.md |
| **Phase C** | Architecture Ablations (5 variants × 5 seeds) | Ablation_Report.md |
| **Phase D** | Downstream Specialization (SFT → Coding → Alignment) | Instruction_Report.md, Coding_Report.md, Alignment_Report.md |
| **Phase E** | Scientific Evaluation & Publication | All remaining reports, Statistical analysis, FINAL_REPORT.md |

---

## Statistical Validation Summary

- **Statistical Tests:** Paired t-test / Wilcoxon Signed-Rank (normality-checked via Shapiro-Wilk)
- **Multiple Comparison Correction:** Holm–Bonferroni applied across all benchmarks
- **Effect Size:** Cohen's d and Cliff's Δ with 95% bootstrap confidence intervals
- **Seeds:** N = 5 independent random seeds per model
- **Hypothesis Status:** See [Hypothesis_Report.md](./Hypothesis_Report.md) for H1–H10 verdicts

---

## Reproducibility

To replay this campaign from `experiments.db`:

```bash
python replay_campaign.py --reviewer-mode
```

Expected output: `Replication status: APPROVED`

---

## Publication Assets

- **Tables:** `Paper_Tables/` — 6 LaTeX table snippets
- **Figures:** `Paper_Figures/` — Loss curves and benchmark comparison plots
- **Reproducibility Archive:** `reproducibility_package.zip`
- **Campaign Manifest:** `campaign_manifest.json`
- **Publication Manifest:** `publication_manifest.json`
- **Evidence Graph:** `evidence_graph.json`
- **Artifact DAG:** `artifact_dag.json`
- **Campaign Certificate:** `Campaign_Certificate.md`
