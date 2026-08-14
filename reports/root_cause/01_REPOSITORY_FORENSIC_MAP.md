# 01 REPOSITORY FORENSIC MAP

## Directory Audit

### `model/titans/`
- `memory.py`: Implements `TitansMemory`. Contains critical performance bugs (Python loop over patches, `.item()` inside loop).
- `updater.py`: Implements `MemoryUpdater`. Fully differentiable update equations.
- `lr_gen.py`: Generator for learning rates.

### `model/mor/`
- `recursion.py`: Implements `RecursionEngine`. Contains a critical GPU synchronization bug (`.any()` on tensor in an `if` statement).
- `router.py`: Implements `RecursionDepthRouter`. Blocks gradients in production mode (`Option C`).
- `adaptive_router.py`: Adaptive components.
- `dynamic_controller.py`: Dynamic recursion limiters.
- `kv_cache.py`: Standard KV caching for recursive steps.

### `model/blt/`
- `encoder.py`: Implements `BLTByteEncoder`. Contains a memory bloat bug (O(S^2) dense pooling tensor `M`).
- `decoder.py`: Implements `BLTByteDecoder`. Cross-attention back to bytes.
- `entropy_model.py`: Generates patch entropy.
- `patcher.py`: Transforms byte boundaries.

### `model/mamba2/`
- `block.py`: `Mamba2Block` implementation. Standard and efficient.
- `scan.py`: Selective SSD scan.
- `math.py`: Mathematics helpers.

### `core/`
- Core interfaces and constants.

### `training/`
- `trainer.py`: Orchestrates training. Shows potential issues with target shifting (depends on dataset).
- `loss_mask.py`: Standard masking tools.

## Red Flags
1. Python loops in `titans/memory.py` `forward` and `write` methods.
2. Tensor `.item()` and `.tolist()` calls inside the hot loop.
3. GPU-CPU sync via `active_mask.any()` in `mor/recursion.py`.
4. Large dense tensor creation `(B, seq_len, seq_len)` in `blt/encoder.py`.
5. Non-differentiable operations blocking gradient flow in `mor/router.py`.
