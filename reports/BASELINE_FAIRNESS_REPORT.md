# IVERI CORE — Baseline Fairness Audit Report

**Date:** 2026-07-24  
**Audit Verdict:** **PASS (Compute-Matched & Hyperparameter Parity Verified)**  

---

## Hyperparameter Parity Matrix

| Parameter | Transformer | Mamba2 | Transformer+MoE | Transformer+Mamba | IVERI Hybrid | Parity Status |
|---|---|---|---|---|---|---|
| **Optimizer** | AdamW | AdamW | AdamW | AdamW | AdamW | `PASS` |
| **Learning Rate** | 5e-4 | 5e-4 | 5e-4 | 5e-4 | 5e-4 | `PASS` |
| **Scheduler** | Cosine Warmup | Cosine Warmup | Cosine Warmup | Cosine Warmup | Cosine Warmup | `PASS` |
| **Tokenizer** | Raw Byte (259) | Raw Byte (259) | Raw Byte (259) | Raw Byte (259) | Raw Byte (259) | `PASS` |
| **Active Params** | ~10M | ~10M | ~10M | ~10M | ~10M | `PASS` |
| **Sequence Length**| 512 | 512 | 512 | 512 | 512 | `PASS` |
| **Step Budget** | 100 | 100 | 100 | 100 | 100 | `PASS` |

*IVERI CORE — Baseline Fairness Auditor.*
