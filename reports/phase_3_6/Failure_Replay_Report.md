# IVERI CORE — Failure Replay Report

This document outlines the failure capture and replay architecture.

### Replay Design
- **RNG Serialization:** Saves state details for random, numpy, CPU pytorch, and all CUDA devices.
- **Traceback Preservation:** Records stack traces alongside configurations.
- **Deterministic Replay API:** Restores parameters, seeds, and executes identical batches.
