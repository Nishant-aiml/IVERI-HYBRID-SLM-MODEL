# 16 MIXTURE OF RECURSIONS (MoR) AUDIT

## General Assessment
The MoR Recursion engine correctly executes recursive passes, but introduces bimodal latency and gradient flow issues depending on the operating mode.

## Answers to Key Questions
1. **Does it loop over tokens/patches?**
   No. It loops over recursion depth (`max_depth`). Operations inside the loop are vectorized over batches and patches via `active_mask`.
2. **What is the actual recursion mechanism?**
   It repeatedly passes the hidden state `x` through the base block (e.g., Mamba2). At each step, a boolean mask (`active_mask`) is evaluated. If a patch's assigned depth is less than the current step, it bypasses the block via `torch.where`.
3. **Is it adding Python overhead per patch?**
   Not per patch, but it adds massive overhead per *step*. Line 171 executes `if not active_mask.any(): break`. This `.any()` forces a blocking CPU-GPU synchronization. 
4. **Is it causing the bimodal latency?**
   YES. The early-exit condition `active_mask.any()` causes synchronization latency that varies directly with the maximum patch depth within the batch. This perfectly explains the CV=46% bimodal latency overhead.
5. **What is the actual compute vs. what is claimed?**
   While compute VRAM and FLOPs are saved using `torch.where`, the latency is severely penalized by the synchronization. Additionally, in "Production Option C", the `router.py` converts `entropy` to depth using `torch.floor(...).long()`. This operation is strictly non-differentiable, completely blocking gradient flow to the router and upstream entropy predictors regarding recursion decisions.
