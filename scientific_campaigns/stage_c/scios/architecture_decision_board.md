# SciOS Architecture Decision Board
*Updated: 2026-07-26T07:23:07Z*

| Decision ID | Feature | Technical Benefit | Resource Cost | Status | Version | Evidence |
|---|---|---|---|---|---|---|
| DEC-001 | Disable Gradient Checkpointing for seq_len < 128 | +41.7% Throughput (2012ms vs 2852ms per step) | +472MB VRAM (987MB vs 515MB, well within 4096MB 3050 limit) | **ACCEPTED** | v0.91 | `latency_10x.json (10 reps, 95% CI verified)` |
| DEC-002 | Dynamic MoR Depth Control (Adaptive MoR v0.92) | +100% Throughput, -50% VRAM, -80% Gradient Explosions | Modest controller parameter overhead (~2K params) | **PLANNED** | v0.92 | `tier1_ablations_and_failures.json (No_MoR variant evidence)` |
| DEC-003 | Gradient Clipping Within MoR Recursion Stack | Eliminates MoR-induced recursive gradient amplification | Zero runtime overhead | **ACCEPTED** | v0.91A | `lr_warmup_sweep_results.json (Explosion invariance proof)` |
