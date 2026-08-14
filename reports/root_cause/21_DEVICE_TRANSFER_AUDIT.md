# 21 DEVICE TRANSFER AUDIT

## model/titans/memory.py

### Hotpath Transfers
Inside the `for t in range(P):` loop in BOTH `write()` and `forward()`:
- `sum_lr += lr_t.detach().mean().item()`
- `sum_forget += forget_t.detach().mean().item()`
- `lr_vals.extend(lr_t.detach().view(-1).tolist())`
- `forget_vals.extend(forget_t.detach().view(-1).tolist())`
- `gnorm = sum([g.detach().pow(2).sum().item() ...]) ** 0.5`
- `mag = sum([(w_new - w_old).detach().pow(2).sum().item()]) ** 0.5`

**Impact**: 
These lines execute on *every single patch* in the sequence, for every forward pass and write step. `.item()` and `.tolist()` force the GPU to synchronize with the CPU, completely blocking async kernel dispatch. This alone explains the 85x training overhead and 22x forward overhead.
**Elimination**:
YES. Telemetry should be accumulated as tensors (e.g., `sum_lr += lr_t.detach().mean()`) and only moved to CPU `.item()` *once* at the very end of the batch, or entirely deferred/disabled in production.

## model/mor/recursion.py

### Hotpath Transfers
Inside the `for step in range(self.max_depth):` loop:
- `if not active_mask.any(): break`

**Impact**:
Evaluating the truth value of `.any()` requires transferring the boolean scalar from GPU to CPU, stalling the CPU until the GPU finishes computing `active_mask`. Because this happens at every recursion step, it introduces severe synchronization latency. Furthermore, because batches with lower max-depth exit the loop earlier, this stall is variable, causing the **46% coefficient of variation (bimodal latency)** observed in experiments.
**Elimination**:
YES. Either remove the early break and execute all steps masked, or compute the max depth of the batch dynamically using a custom optimized kernel. Removing the break entirely is fastest for latency stability.
