# Architecture Report — Phase 2.5
**IVERI CORE v1.0 | Phase 2.5**
**Date:** 2026-06-30

---

## 1. Subsystem Telemetry Map

The `ArchitectureEvaluator` extracts telemetry metrics across all major subsystems in the IVERI CORE model:

| Subsystem | Telemetry Metrics | Distribution / Format |
|---|---|---|
| **BLT** | average byte entropy, average patch entropy, compression ratio, boundary frequency, average/median patch count. | Binned histograms for patch size distributions. |
| **Mamba2** | hidden state norms (mean/std/max), state update norm, state variance, token-level throughput. | Scalar values and binned state norm histograms. |
| **Flash Attention** | attention backend type, latency, activation memory. | Latency profiles. |
| **MoE** | routing counts, unused experts, imbalance ratio, Shannon entropy, collapse score, auxiliary load balance loss. | Expert utilization histograms. |
| **MoR** | average depth, median depth, max depth, p95 depth, FLOPs saved ratio. | Recursion depth binned histograms. |
| **Titans** | memory reads/writes, update norm, retrieval norm, learning-rate, forget-rate, gate activation. | Learning-rate, forget-rate, and gate histograms. |
| **Backbone** | layer count, layer latency list, residual L2 norms, activation norms, peak VRAM. | Layer-wise lists. |

---

## 2. Metrics Verification

Telemetry extraction is validated under `test_architecture_statistics` in the test suite. All metrics are collected dynamically from the model forward dictionary outputs without structural modifications to model packages.
- Binned histograms use `numpy.histogram` with deterministic bins.
- Entropy and collapse indicators utilize Shannon entropy calculations to identify structural degradation early.
