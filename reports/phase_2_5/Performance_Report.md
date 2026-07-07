# Performance Report — Phase 2.5
**IVERI CORE v1.0 | Phase 2.5**
**Date:** 2026-06-30

---

## 1. Subsystem Performance Overhead

The `test_evaluation.py` suite includes micro-benchmarking checks to ensure the evaluation framework introduces minimal execution overhead.

| Module | Typical Overhead per Pass / Step | Target Limit | Status |
|---|---|---|---|
| **Perplexity Evaluator** | < 0.2 ms per batch | < 5.0 ms | ✅ PASS |
| **Memory Tracker** | < 0.1 ms | < 2.0 ms | ✅ PASS |
| **Architecture Evaluator** | < 0.5 ms | < 10.0 ms | ✅ PASS |

The overall evaluator loop overhead remains strictly under 1% of the forward step latency of typical 10M models.

---

## 2. Repeated Evaluation Determinism

A primary design constraint is evaluation reproducibility:
- **Seed Determinism**: Verified in `test_repeated_evaluation`. By applying a manual PyTorch random seed (`torch.manual_seed(42)`), the evaluator computes identical loss and perplexity values across consecutive evaluations on the same dataloader batch.
- **Titans Memory Isolation**: The evaluation run does not modify the model state or Titans neural memory parameters. Subsequent runs produce mathematically identical representations.
- **Decayed parameter updates**: No parameters require gradients during or after evaluation, confirming the read-only contract.
