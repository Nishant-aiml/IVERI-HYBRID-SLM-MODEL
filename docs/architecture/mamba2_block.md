# Mamba2 Block Assembly Architecture

This document details the complete module layout, input parameter projections, causal 1D convolutions, and gated elementwise scaling of the Mamba2Block layer.

---

## 1. Block Diagram & Execution Flow

```
Input x (B, S, D)
     │
     ├──► in_proj (nn.Linear) ──► Projected tensor (B, S, 3*d_inner + 2*d_state)
     │                                │
     │     ┌──────────────────────────┴─────────────────────────┐
     │     ▼ [x, delta, B, C] paths                             ▼ [gate] path
     │   Causal Conv1d (kernel=4)                               │
     │     │                                                    │
     │     ▼                                                    │
     │   Reshape & Transpose (Head Layout)                      │
     │     │                                                    │
     │     ▼                                                    │
     │   selective_ssd_scan                                     │
     │     │                                                    │
     │     ▼                                                    │
     │   Reshape (Sequence Layout)                              │
     │     │                                                    │
     │     └──────────────────────────┬─────────────────────────┘
     │                                ▼
     │                      Multiply Gated (SiLU gate)
     │                                │
     │                                ▼
     └──► Residual sum ◄─────── out_proj (nn.Linear)
```

---

## 2. Block Component Specifications

1.  **Input Projection:**
    Projects input $X \in \mathbb{R}^{B \times S \times D}$ to intermediate dimension $3 \cdot D_{\text{inner}} + 2 \cdot D_{\text{state}}$ using a single linear layer:
    $$[x, \text{gate}, \delta, B, C] = \text{in\_proj}(X)$$
2.  **Causal 1D Convolution:**
    Concatenates $x, \delta, B, C$ along the channel dimension, pads the sequence dimension by $3$ on the left, and runs a depthwise 1D convolution:
    $$[x_{\text{conv}}, \delta_{\text{conv}}, B_{\text{conv}}, C_{\text{conv}}] = \text{Conv1d}(\text{Pad}(x \oplus \delta \oplus B \oplus C))$$
3.  **Parameter Discretization:**
    $$\delta_{\text{param}} = \text{Softplus}(\delta_{\text{conv}} + dt_{\text{bias}})$$
4.  **Structured State Space Duality (SSD) Scan:**
    The parameters are reshaped to the multi-head layout and passed to the selective scan recurrence layer:
    $$Y_{\text{scan}}, h_{\text{final}} = \text{selective\_ssd\_scan}(x_{\text{heads}}, \delta_{\text{heads}}, A, B_{\text{heads}}, C_{\text{heads}})$$
5.  **Multiplicative Gating:**
    $$Y_{\text{gated}} = Y_{\text{scan}} \odot \text{SiLU}(\text{gate})$$
6.  **Output Projection:**
    $$\text{output} = \text{out\_proj}(Y_{\text{gated}})$$
