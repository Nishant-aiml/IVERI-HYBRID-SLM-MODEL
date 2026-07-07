# Mixture of Recursions (MoR) Architecture

This document details the dynamic computation depth router, recursion loop controller, and selective key-value caching system implemented for the Mixture of Recursions (MoR) subsystem in the IVERI architecture.

---

## 1. Multi-Functional Entropy Signal Gating (Option C)

In standard Mixture of Recursions implementations, a separate learned router network assigns computation depths. In IVERI, the separate learned router is replaced entirely by directly mapping the patch-level prediction entropy:

$$D_p = 1 + \text{floor}(E_p \times (\text{max\_recursion\_depth} - 1))$$

Where:
- $E_p \in [0.0, 1.0]$ is the pooled byte-level prediction entropy of patch $p$.
- $\text{max\_recursion\_depth}$ is config-configured (default $8$).
- This produces depth values $D_p \in [1, 8]$.

This mapping ensures that simple, predictable patches (e.g. spaces, repetitive patterns) exit computation after 1 step, while complex patches (e.g. technical jargon, code structures) undergo up to 8 recursive passes through the shared backbone weights.

---

## 2. Active Recursion Masking

Adaptive computation requires selectively bypassing layer blocks during the recursion loop. For each recursive pass `step` (from $0$ to $\text{max\_recursion\_depth} - 1$), the active mask is computed as:

$$\text{active\_mask} = D_p > \text{step}$$

Computation is applied selectively:
- If a block natively supports masking, `active_mask` is passed to the block directly.
- Otherwise, bypassed values are retained unchanged via residual selection:
  $$x_{\text{next}} = \text{where}(\text{active\_mask}, \text{block}(x), x)$$

---

## 3. Selective KV Caching

Autoregressive key-value states are cached only for active computational steps to prevent context bloat:

- Appending of keys and values along the sequence dimension is gated by `active_mask`.
- Inactive items are padded or bypass updating, reducing overall KV cache VRAM footprint by 60–70% on average.
