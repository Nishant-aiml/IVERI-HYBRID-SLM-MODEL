# IVERI CORE — Experiment Guidelines

This document details protocol controls for running runs and preventing configuration leakage.

---

## 1. Random Seeds

To prevent data selection bias, all experiments must run over exactly the 5 random seeds:
- **Seed 42**
- **Seed 123**
- **Seed 3407**
- **Seed 2026**
- **Seed 9999**

---

## 2. Environment Controls

To guarantee complete reproducibility:
- **Git State Logs:** The active Git commit SHA and current branch name must be written to all telemetry files.
- **Hardware Telemetry:** Log host RSS, peak VRAM allocations, GPU wattage draws, and climate cost estimations.
- **Config Hashing:** A SHA-256 hash of the fully resolved model configuration dictionary must accompany all registered metrics.
- **No In-place Overwrites:** Metrics must reside under timestamp-versioned files inside `logs/`.
