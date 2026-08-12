# IVERI CORE — IEEE Publication Package

**Date:** 2026-07-24  
**Status:** REPRODUCIBLE PUBLICATION BUNDLE  

---

## 1. Bundle Metadata
- **Architecture Version:** `Option C (v1.0.0)`
- **Git SHA:** `2f31069`
- **Environment:** PyTorch 2.5.1+cu121, Python 3.10+
- **Experiment Registry:** `reports/experiment_registry.json`
- **Checkpoint Lineage:** `reports/checkpoint_lineage.json`

---

## 2. Threats to Validity Analysis

### Internal Validity
- **Seed Control:** Evaluated across 3 seeds (`42`, `123`, `3407`) to control for initialization variance.
- **Gradient Stability:** Monitored via gradient health CSVs (`zero_grad_ratio`, `global_grad_norm`).

### External Validity
- **Domain Scope:** Validated on byte-level natural language (`Wikipedia`, `FineWeb-Edu`), code (`The Stack`), and narrative (`TinyStories`).
- **Hardware Portability:** Tested across CPU and CUDA GPU execution backends.
