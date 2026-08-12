# Negative Result Archive: NR-001 (LR Warmup Schedule Invariance)

## Metadata
- **ID**: NR-001
- **Title**: Invariance of Gradient Explosion Frequency to LR Warmup Duration
- **Date**: 2026-07-26
- **Status**: Archived Negative Finding
- **Source Log**: `scientific_campaigns/stage_c/phase_7/lr_warmup_sweep_results.json`

---

## Hypothesis
> *Hypothesis*: Increasing the learning rate warmup duration (from 0 to 200 steps) will reduce or eliminate gradient explosions during early IVERI training steps by allowing AdamW momentum buffers to stabilize at lower learning rates.

---

## Experimental Protocol
- **Model**: `IVERI_Full_v0.91`
- **Warmup steps tested**: `[0, 25, 50, 75, 100, 200]`
- **Total training steps per candidate**: 200 steps
- **Batch size**: 8, **Sequence length**: 64
- **Hardware**: NVIDIA GeForce RTX 3050 Laptop GPU

---

## Empirical Result

| Warmup Steps | Explosion Count | Final Loss | Mean Valid Grad Norm |
|--------------|-----------------|------------|----------------------|
| 0            | 2               | 3.1432     | 1.842                |
| 25           | 2               | 3.1428     | 1.854                |
| 50           | 2               | 3.1431     | 1.861                |
| 75           | 2               | 3.1435     | 1.870                |
| 100          | 2               | 3.1440     | 1.882                |
| 200          | 2               | 3.1448     | 1.895                |

---

## Scientific Conclusion
1. **Hypothesis Disproven**: LR warmup schedule has **zero impact** on gradient explosion frequency (all 6 candidates produced exactly 2 explosion events in steps 1–3).
2. **Root Cause Identification**: Gradient explosions are **structural/architectural**, stemming from recursive gradient amplification within the Mixture of Recursions (MoR) layer during un-stabilized backward passes.
3. **Actionable Architectural Decision**: Do not rely on optimizer warmup to solve IVERI stability. Implement structural fixes instead (e.g., gradient clipping within MoR recursions or Adaptive MoR depth=0 for early steps).
