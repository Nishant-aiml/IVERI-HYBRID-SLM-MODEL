# Flash Attention Wrapper Architecture

This document details the unified, backend-independent attention subsystem implementation in IVERI CORE.

---

## 1. Unified Interface Layout

The `FlashAttentionWrapper` serves as an abstraction layer that isolates the model's backbone logic from specific attention backend dependencies:

```
                  Input x (B, S, D)
                         │
                         ▼
                  Linear QKV Projection (3 * D)
                         │
                         ▼
                  Multi-Head Reshaping (B, H, S, d_head)
                         │
                         ▼
             Is CUDA & flash_attn installed?
              ├──► YES: FlashAttention-2 Backend
              └──► NO:  PyTorch SDPA Backend (optimized fallback)
                         │
                         ▼
                  Transpose & Concatenate Output (B, S, D)
                         │
                         ▼
                  Linear Output Projection (D)
```

---

## 2. Dynamic Backend Dispatcher

*   **PyTorch Scaled Dot-Product Attention (SDPA):** Dispatched via `F.scaled_dot_product_attention`. PyTorch dynamically chooses FlashAttention, MemoryEfficientAttention, or C++ vectorized math kernels depending on hardware capability.
*   **FlashAttention-2:** Dispatched via `flash_attn.flash_attn_func`. Input heads are transposed to `(B, S, H, D)` before computing, and mapped back to standard multi-head dimensions.

---

## 3. Key-Value (KV) Cache updates

Incremental decoding requires preserving past keys and values. The caching interface is kept simple and mutable:
*   Pass `kv_cache` as a dictionary containing `"key"` and `"value"` tensor arrays.
*   The wrapper mutates `kv_cache` in-place by appending new token keys/values along the sequence dimension.
*   This avoids signature mismatching with standard PyTorch module execution, keeping the return contract of `forward` identical.
