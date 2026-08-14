# 02 ACTUAL COMPUTATION GRAPH

## Forward Pass Trace

1. **Input**: `raw_bytes` of shape `(B, S)`
2. **Entropy Model (`model/blt/entropy_model.py`)**:
   - `x` -> Embed -> Conv1D -> Predictor -> `entropy` `(B, S, 1)`
   - *Gradient Flow*: Receives gradients only from downstream differentiable uses (Titans injection).
3. **BLT Patcher (`model/blt/patcher.py`)**:
   - Converts `entropy` to `boundary_map` `(B, S)`.
   - *Gradient Flow*: Typically non-differentiable thresholding (BLOCKED).
4. **BLT Encoder (`model/blt/encoder.py`)**:
   - Computes latent patches by mean-pooling byte embeddings according to `boundary_map`.
   - Creates a massive dense pooling matrix `M` of shape `(B, p_max, S)` where `p_max = S`.
   - *Output*: `latent_patches` `(B, p_max, D)`
   - *Gradient Flow*: ACTIVE (back to bytes, but not to boundaries).
5. **MoR Router (`model/mor/router.py`)**:
   - Maps `entropy` to `depths` via `1 + floor(entropy * (max_depth - 1))`.
   - *Output*: `dispatch_indices` `(B, P, 1)`
   - *Gradient Flow*: BLOCKED by `.long()` and `floor()`.
6. **MoR Recursion Engine (`model/mor/recursion.py`)**:
   - Iterates `for step in range(max_depth)`.
   - Computes `active_mask = depths > step`.
   - Executes CPU sync: `if not active_mask.any(): break`
   - *Output*: Recursively updated `x` `(B, P, D)`
   - *Gradient Flow*: ACTIVE through `torch.where`.
7. **Titans Memory (`model/titans/memory.py`)**:
   - `forward_with_injection` calls `forward`.
   - Iterates in Python `for t in range(P)`.
   - At each step, performs functional update with `torch.autograd.grad`.
   - Fetches telemetry via `.item()` and `.tolist()`, causing extreme CPU stalling.
   - *Output*: `retrieved` `(B, P, D)` gated by `sigmoid(entropy_proj)`.
   - *Gradient Flow*: ACTIVE.
8. **BLT Decoder (`model/blt/decoder.py`)**:
   - Uses `raw_bytes` as queries to cross-attend to `latent_patches`.
   - *Output*: `logits` `(B, S, V)`
   - *Gradient Flow*: ACTIVE.
9. **Loss (`training/trainer.py`)**:
   - `cross_entropy(flat_logits, flat_targets)`
