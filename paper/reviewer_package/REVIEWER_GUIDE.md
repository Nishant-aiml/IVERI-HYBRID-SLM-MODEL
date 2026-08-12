# IVERI CORE — IEEE Reviewer Reproduction & Audit Guide

Welcome IEEE & Conference Reviewers. This package allows 100% independent verification of every claim, figure, table, and statistical metric in the paper:
**"Entropy-Guided Dynamic Sparse Expert Routing for Efficient Byte-Level Language Models"**

---

## 🚀 One-Command Verification

To independently audit all claims, verify raw metric logs, and rebuild paper artifacts:

```bash
python paper/scripts/verify_ieee_paper.py
```

Expected output:
```json
{
  "verification_status": "IEEE_READY_FOR_STAGE_B",
  "total_claims_audited": 6,
  "claims_passed": 3,
  "claims_in_progress": 1,
  "claims_awaiting_stage_c": 2,
  "zero_fabrications_enforced": true
}
```

---

## 📊 Rebuilding All Figures & Tables from Raw Execution Logs

Every table and figure in the paper is programmatically derived from raw CSV metrics. To re-generate all figures, tables, and statistical tests from source:

```bash
python paper/scripts/build_paper.py
```

No numbers are hardcoded. If raw CSV files are absent or modified, the validator marks claims as **UNSUPPORTED / INSUFFICIENT EVIDENCE**.

---

## 📁 Repository Evidence Map
- **Raw Metric Logs:** `paper/csv/` & `reports/live_training/`
- **Experiment Registry:** `reports/experiment_registry.json` & `paper/experiments/experiment_registry.csv`
- **Evidence Index:** `paper/evidence/SCIENTIFIC_EVIDENCE_INDEX.md`
- **Statistical Engine:** `paper/scripts/stats_engine.py`
