# 04 GRADIENT FLOW AUDIT

## BLT Encoder/Decoder
- **Status**: GRADIENT ACTIVE
- **Evidence**: Gradients flow cleanly from CE loss through decoder, cross-attention, latent patches, and mean-pooling in the encoder. 

## Entropy Model
- **Status**: PARTIAL
- **Evidence**: The `entropy` tensor flows to two destinations:
  1. `model/titans/memory.py`: `gate = torch.sigmoid(self.entropy_proj(patch_entropy))`. This path maintains full gradients.
  2. `model/mor/router.py`: `depths = 1 + torch.floor(entropy * (self.max_depth - 1)).long()`. This path is **blocked** by `.long()` and `floor()`.
  
## BLT Patcher
- **Status**: BLOCKED
- **Evidence**: Boundary maps are typically boolean logic maps (thresholding entropy or hardcoded limits), severing gradients to dynamic sequence partitioning.

## Titans
- **Status**: GRADIENT ACTIVE
- **Evidence**: `torch.autograd.grad(..., create_graph=True)` allows the backward pass to differentiate through the forward inner-loop updates.

## Mamba2
- **Status**: GRADIENT ACTIVE
- **Evidence**: Standard operations.

## MoR Router
- **Status**: BLOCKED (Production) / PARTIAL (Research)
- **Evidence**: 
  - *Production*: Uses `floor` and `.long()` on entropy to derive depths.
  - *Research*: Computes `probs` using softmax, then extracts `dispatch_indices` via `argmax`. However, it returns **only** `dispatch_indices` and discards `dispatch_weights`. Thus, gradients never flow back to the router parameters (`self.wg`).

## Loss Computation
- **Status**: POTENTIALLY PARTIAL
- **Evidence**: `trainer.py` uses `cross_entropy(flat_logits, flat_targets)`. If the dataset does not pre-shift the targets by 1, the model is penalized for not predicting `x[t]` given `x[:t]`, which is impossible. This must be validated in dataset logic.
