# 22 PYTHON HOTPATH AUDIT

## model/titans/memory.py

### `write` method (line 283)
- **Loops over**: `P` (Patch sequence length) via `for t in range(P):`
- **Why it exists**: Performs sequential online associative memory updates per patch.
- **Cost**: High. Autograd and MLP computations in a python loop.
- **Vectorizable?**: Yes, parallel prefix scans or chunked linear attention formulations could vectorize this, but it requires changing the exact update semantics.

### `forward` method (line 413)
- **Loops over**: `P` (Patch sequence length) via `for t in range(P):`
- **Why it exists**: Fused read-and-update step for the backbone forward pass.
- **Cost**: EXTREMELY HIGH. Contains multiple `.item()` calls.
- **Vectorizable?**: Yes, with chunked memory updates.

## model/mor/recursion.py

### `forward` method (line 166)
- **Loops over**: `max_depth` via `for step in range(self.max_depth):`
- **Why it exists**: Iteratively applies the residual block to patches that require deeper computation.
- **Cost**: Low to Medium. `max_depth` is typically small (e.g. 8).
- **Vectorizable?**: The loop itself is unrolled over steps, which is semantically necessary. However, the condition check inside it is expensive.
