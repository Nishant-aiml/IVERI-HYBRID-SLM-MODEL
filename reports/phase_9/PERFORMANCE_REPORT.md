# Phase 9 — Performance Report

**Date:** 2026-07-11  
**Hardware:** CPU only (PyTorch 2.12.1+cpu, CUDA unavailable)

---

## Model Forward Pass (nano 10M config)

Measured via `IVERIModel(get_nano_config())` forward on `(2, 64)` random bytes:

| Metric | Value |
|--------|-------|
| Total parameters | 9,347,282 |
| Logits shape | `[2, 64, 259]` |
| BLT avg byte entropy | 0.999996 |
| MoR avg recursion depth | 3.0 |
| MoE expert histogram | [677, 859] |
| Titans reads / writes | 128 / 128 |

Telemetry includes: `forward_latency_seconds`, `backward_latency_seconds`, `flops_per_module`, `activation_memory_mb`, `peak_vram_mb` (0 on CPU).

---

## Architecture Health Script (2 training steps)

| Step | Loss | Aux Loss |
|------|------|----------|
| 1 | 5.8349 | 0.1426 |
| 2 | 5.3566 | 0.1401 |

**Result:** Loss decreased. Script failed on MoE load balance gate (69.1% > 60%).

---

## Phase 9 Campaign

**Not executed past 9.2** — no pilot throughput, tokens/sec, or milestone metrics available.

---

## Limitations

- No GPU measurements (VRAM, CUDA utilization, tokens/sec on RTX 3050)
- No extended training run metrics
- Phase 7 report (`reports/phase_7/06_real_training.md`) claims 100-step CPU run with final loss 3.08 — not re-verified in this audit
