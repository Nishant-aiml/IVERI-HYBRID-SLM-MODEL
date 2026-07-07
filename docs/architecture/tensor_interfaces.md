# Tensor Interfaces & Signature Contracts

This document formalizes the tensor shape expectations and signature contracts for each block in IVERI CORE.

---

## 1. Global Dimensions Reference

| Dimension Slug | Symbol | Meaning |
|---|---|---|
| `B` | Batch Size | Sequence batches processed in parallel. |
| `S` | Seq Len (Bytes) | Length of raw byte sequences. |
| `P` | Patches Len | Segmented latent representations sequence length. |
| `D` | Hidden Dim | Feature dimension width. |
| `H` | Heads Count | Subdivisions of the attention layer space. |
| `E` | Experts Count | Total size of FFN experts pool. |
| `K` | Active Experts | Number of experts selected per token. |

---

## 2. Component Signatures

### ByteEntropyModel
*   **Input:** `(B, S)` of type `torch.int64` (raw byte indices in `[0, 255]`).
*   **Output:** `(B, S, 1)` of type `torch.float32` — normalized Shannon entropy in `[0.0, 1.0]` per byte position.
*   **Note:** This is the single entropy source (Option C) that drives all 4 downstream consumers.

### DynamicPatcher
*   **Input:**
    *   `raw_bytes`: `(B, S)` of type `torch.int64`.
    *   `entropy`: `(B, S, 1)` or `(B, S)` of type `torch.float32` (from ByteEntropyModel).
*   **Output:** `boundary_map`: `(B, S)` of type `torch.bool` — `True` at positions that begin a new patch.

### Patch Entropy Aggregation (Mean Pooling)
*   **Inputs:**
    *   `byte_entropy`: `(B, S, 1)` float32.
    *   `boundary_map`: `(B, S)` bool.
*   **Output:** `patch_entropy`: `(B, P, 1)` float32 — mean-pooled byte entropy within each patch boundary.
*   **Used by:** `RecursionDepthRouter` (MoR), `SparseMoERouter` (via backbone), and `TitansMemory.inject()`.

### BLT Encoder
*   **Input:** `(B, S)` of type `torch.int64` (raw byte indices in `[0, 255]`).
*   **Boundary Map:** `(B, S)` of type `torch.bool` (indicating patch starts).
*   **Output:** `(B, P, D)` of type `torch.float32` (latent patched representation).

### Backbone Layer Module
*   **Input:** `(B, P, D)` representation tensors.
*   **Output:** `(B, P, D)` updated representation tensors.

### Sparse MoE FFN Router
*   **Input:** `(B, P, D)` token vectors (or flattened `(B*P, D)`).
*   **Output:**
    *   `weights`: `(B, P, K)` or `(B*P, K)` — selected expert routing weights (softmax-normalized, top-k only).
    *   `indices`: `(B, P, K)` or `(B*P, K)` — selected expert identifiers.
    *   `aux_loss`: scalar `torch.float32` — Shazeer load-balancing loss.

### MoR RecursionDepthRouter
*   **Input:**
    *   `x`: `(B, P, D)` hidden representations.
    *   `entropy`: `(B, P, 1)` or `(B, P)` patch entropy (required in Option C production mode).
*   **Output:**
    *   `dispatch_weights`: `(B, P, 1)` float32 — routing confidence (ones in production mode).
    *   `dispatch_indices`: `(B, P, 1)` int64 — 0-indexed assigned depth per patch.
*   **Depth Formula:** `D_p = 1 + floor(E_p × (max_depth − 1))`, yielding values in `[1, max_depth]`.

### MoR RecursionEngine
*   **Input:**
    *   `x`: `(B, P, D)` hidden representations.
    *   `depths`: `(B, P)` or `(B, P, 1)` int64 — depth assignment per patch (from `RecursionDepthRouter`).
*   **Output:** `(B, P, D)` updated hidden representations after selective recursion.

### Titans Memory Module
*   **Input (forward):** `(B, P, D)` query and context keys.
*   **Output (forward):** `(B, P, D)` retrieved memory representation.
*   **Input (inject):**
    *   `x`: `(B, P, D)` patch representations.
    *   `patch_entropy`: `(B, P, 1)` or `(B, P)` float32 entropy gate.
*   **Output (inject):** `(B, P, D)` — entropy-gated memory-injected representation.

### BLT Decoder
*   **Input:**
    *   `latent_patches`: `(B, P, D)` of type `torch.float32`.
    *   `boundary_map`: `(B, S)` of type `torch.bool`.
    *   `raw_bytes`: `(B, S)` of type `torch.int64`.
*   **Output:** `(B, S, 256)` of type `torch.float32` — next-byte prediction logits.

### IVERIModel (Full Pipeline)
*   **Input:** `raw_bytes` — `(B, S)` of type `torch.int64` (raw byte indices in `[0, 255]`).
*   **Outputs (return_dict=True):**
    *   `logits`: `(B, S, 256)` float32 — next-byte prediction logits.
    *   `byte_entropy`: `(B, S, 1)` float32 — per-byte Shannon entropy from `ByteEntropyModel`.
    *   `patch_entropy`: `(B, P, 1)` float32 — mean-pooled patch-level entropy.
    *   `boundary_map`: `(B, S)` bool — patch boundary positions from `DynamicPatcher`.
    *   `aux_loss`: scalar float32 — MoE load-balancing auxiliary loss.
    *   `telemetry`: dict — runtime diagnostics (backbone, memory, routing stats).
*   **Output (return_dict=False):** `logits` — `(B, S, 256)` float32 only.
*   **Note:** Empty sequence (`S=0`) returns all tensors with correct zero-sized shapes and `aux_loss=0.0`.

---

## 3. Full Pipeline Shape Flow Summary

```
Raw Bytes (B, S) int64
    │
    ▼
ByteEntropyModel → byte_entropy (B, S, 1) float32
    │
    ├──► DynamicPatcher(byte_entropy) → boundary_map (B, S) bool
    │
    ├──► BLTByteEncoder(raw_bytes, boundary_map) → latent_patches (B, P, D) float32
    │
    └──► Patch Entropy Aggregation(byte_entropy, boundary_map) → patch_entropy (B, P, 1) float32
              │
              ▼
         Backbone Blocks × L
         ├── TitansMemory.inject(latent_patches, patch_entropy) → (B, P, D)
         ├── RecursionDepthRouter(x, patch_entropy) → depths (B, P, 1) int
         ├── RecursionEngine(x, depths) → (B, P, D)
         │    └── [MoR block per step]
         │         ├── Mamba2Block(x) → (B, P, D)
         │         ├── FlashAttentionWrapper(x) → (B, P, D)
         │         └── SparseMoERouter(x) → weights (B*P, K), indices (B*P, K)
         │
         ▼
    Backbone Out (B, P, D) float32
         │
         ▼
    BLTByteDecoder(latent_patches, boundary_map, raw_bytes) → logits (B, S, 256) float32
```

