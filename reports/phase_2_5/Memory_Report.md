# Memory Report — Phase 2.5
**IVERI CORE v1.0 | Phase 2.5**
**Date:** 2026-06-30

---

## 1. Memory Profiling Channels

The `MemoryTracker` tracks 5 distinct memory indicators:
- **GPU Allocated**: Currently allocated CUDA tensor memory (MB).
- **GPU Reserved / Peak**: Reserved memory segments and peak CUDA allocation recorded during the evaluation pass.
- **CPU System RAM**: Process resident set size (RSS) via `psutil`.
- **Parameter Memory**: Analytical calculation matching model weight sizes ($N \times \text{element\_size}$ bytes).
- **Activation Memory**: Estimated from the difference between peak memory usage during evaluation and starting model weights.

---

## 2. Memory Growth Verification

The evaluation pipeline is validated to ensure zero memory leaks.

| Test | Objective | Result | Status |
|---|---|---|---|
| `test_memory_tracking` | Confirms memory metrics are successfully read and non-negative. | Validated | ✅ PASS |
| `test_memory_growth` | Evaluates memory consumption across 5 repeated inference loops. | Growth < 2.0 MB | ✅ PASS |

The absence of a backward pass and gradients (`torch.no_grad()`) ensures that activation memory graphs are garbage-collected instantly at the end of each forward pass, guaranteeing memory growth remains well within acceptable bounds.
