# Checkpoint Comparison Report — Phase 2.5
**IVERI CORE v1.0 | Phase 2.5**
**Date:** 2026-06-30

---

## 1. Compatibility Check Invariants

The `CheckpointComparator` validates structural compatibility before analyzing weight or metric deltas.

| Invariant Check | Action on Mismatch | Verification Test |
|---|---|---|
| **Architecture Version** | Flags as `NOT DIRECTLY COMPARABLE` | `test_checkpoint_version_mismatch` |
| **Model Parameters Size** | Flags as `NOT DIRECTLY COMPARABLE` | `test_checkpoint_hash_mismatch` |
| **Model Config Shapes** | Flags as `NOT DIRECTLY COMPARABLE` | `test_checkpoint_hash_mismatch` |

If checkpoints are marked as incompatible:
- A `NOT DIRECTLY COMPARABLE` status is written to the report.
- The comparison reasons are appended to a detailed mismatch reasons list.
- Standard loss, perplexity, and parameter weight deltas are skipped.

---

## 2. Delta Calculations (Compatible Checkpoints)

For structurally compatible checkpoints:
- **Metrics Diff**: Calculates change in validation cross-entropy loss and epoch/step indices.
- **Weight Norm Delta**: Computes the cumulative L2 norm difference of weights:
  $$\Delta W_{L2} = \sqrt{\sum_l \|W_l^B - W_l^A\|_2^2}$$
- **Layer-wise Delta**: Resolves layer-wise L2 norm weight differences.
