# 28 BUG CANDIDATE REGISTRY

| Bug ID | Component | Severity | Location | Description | Reproducibility | Proposed Fix |
|--------|-----------|----------|----------|-------------|----------------|--------------|
| BUG-01 | Titans Memory | CRITICAL | `titans/memory.py:317` | `.item()` and `.tolist()` calls inside the per-patch online update loop cause $O(P)$ CPU-GPU synchronizations. | 100% | Move telemetry logic outside the loop, or remove it from `forward` passes entirely. |
| BUG-02 | MoR Recursion | HIGH | `mor/recursion.py:171` | `if not active_mask.any(): break` forces a boolean scalar evaluation, causing variable CPU-GPU sync. | 100% | Remove the early break loop condition to ensure consistent kernel dispatching. |
| BUG-03 | BLT Encoder | HIGH | `blt/encoder.py:129` | Dense pooling tensor `M` created as `(B, p_max, seq_len)` where `p_max = seq_len`. Consumes $O(B \times S^2)$ VRAM. | 100% | Use sparse indexing or scatter-add reductions instead of massive boolean matrix multiplication. |
| BUG-04 | MoR Router | MEDIUM | `mor/router.py:87` | `torch.floor(...).long()` blocks all gradients from downstream performance metrics. | 100% | Use a straight-through estimator (STE) or Gumbel-Softmax for differentiable routing. |
| BUG-05 | Loss Function | UNKNOWN | `training/trainer.py` | Potential lack of target shifting. | Depends on Dataset | Verify dataset labels are shifted. |
