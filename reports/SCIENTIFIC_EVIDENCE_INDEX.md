# IVERI CORE — Scientific Evidence Index

**Date:** 2026-07-24  
**Master Traceability Index:** Code $\to$ Experiment ID $\to$ Dataset $\to$ Checkpoint $\to$ Figure $\to$ Table  

---

## 1. Master Evidence Traceability Matrix

| Claim / Conclusion | Experiment ID | Dataset Provenance | Checkpoint SHA-256 | Empirical Figure | Empirical Table | Verdict |
|---|---|---|---|---|---|---|
| **Parameter Matching** | `EXP-0001` | Phase 8 Certified (`tinystories`) | `checkpoint_registry.json` | N/A | `table1_parameters.csv` | `VERIFIED` |
| **Transformer Loss** | `EXP-0001..0003` | Phase 8 Certified (`tinystories`) | `checkpoint_lineage.json` | `fig01_loss_curves.png` | `table3_loss.csv` | `VERIFIED` |
| **Mamba2 Loss** | `EXP-0004..0006` | Phase 8 Certified (`tinystories`) | `checkpoint_lineage.json` | `fig01_loss_curves.png` | `table3_loss.csv` | `VERIFIED` |
| **Transformer+MoE Loss** | `EXP-0007..0009` | Phase 8 Certified (`tinystories`) | `checkpoint_lineage.json` | `fig01_loss_curves.png` | `table3_loss.csv` | `VERIFIED` |
| **Transformer+Mamba Loss**| `EXP-0010..0012` | Phase 8 Certified (`tinystories`) | `checkpoint_lineage.json` | `fig01_loss_curves.png` | `table3_loss.csv` | `VERIFIED` |
| **IVERI Hybrid Loss** | `EXP-0013..0015` | Phase 8 Certified (`tinystories`) | `checkpoint_lineage.json` | `fig01_loss_curves.png` | `table3_loss.csv` | `VERIFIED` |
| **Component Ablation** | `EXP-0016..0022` | Phase 8 Certified (`tinystories`) | `checkpoint_lineage.json` | `fig04_ablation_waterfall.png` | `table4_ablation.csv` | `VERIFIED` |
| **Statistical Significance** | All Experiments | Phase 8 Certified (`tinystories`) | `experiment_registry.json` | `fig08_statistical_ci.png` | `table5_statistics.csv` | `VERIFIED` |

---

## 2. Integrity Guarantee

Every cell in this index traces directly to an immutable Experiment ID in `reports/experiment_registry.json` and a PyTorch checkpoint in `reports/checkpoint_lineage.json`.
