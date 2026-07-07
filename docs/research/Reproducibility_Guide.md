# IVERI CORE — Reproducibility Guide

This document describes how to execute the verification suite and bundle runs.

---

## 1. Setup Instructions

Verify development packages are installed:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

---

## 2. Packaging Manifests

To generate a complete scientific package snapshot:
1. Run the verification sweeps.
2. Invoke `research.artifacts.ResearchArtifactsManager`.
3. This creates a `reproducibility_package.zip` inside `reports/phase_3_5/artifacts/` containing:
   - System specification metadata (OS, CUDA, PyTorch version, CPU/GPU details).
   - Package freeze log (`pip freeze`).
   - Git branch and commit hashes.
   - Exact configuration settings snapshot.
   - Computed benchmark statistics.
