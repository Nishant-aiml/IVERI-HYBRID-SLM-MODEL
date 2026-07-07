# Mamba2 Structured State Space Duality (SSD) Specification

This document details the mathematical framework, discretization methods, and block assembly properties of the Structured State Space Duality (SSD) implementation in IVERI CORE.

---

## 1. Structured State Space Duality Mathematics

Mamba2 unifies state space models (SSMs) and attention mechanisms under the Structured State Space Duality (SSD) framework.

### 1.1 Model Discretization
Given continuous parameters $A$ (negative values for stability) and time-step values $\Delta_t$:

*   **ZOH Discretization for A:**
    $$\bar{A}_t = \exp(\Delta_t \cdot A)$$
*   **ZOH Discretization for B:**
    $$\bar{B}_t = \frac{\exp(\Delta_t \cdot A) - 1}{A} \cdot B_t$$
    *(In recurrent scan, a first-order Euler approximation $\bar{B}_t = \Delta_t \cdot B_t$ is commonly used for computational speed).*

### 1.2 State Space Recurrence
The discretized system evolves sequence step by step:
$$h_t = \bar{A}_t \odot h_{t-1} + (x_t \odot \Delta_t) \otimes B_t$$
$$y_t = h_t \cdot C_t$$

Where:
*   $x_t \in \mathbb{R}^{B \times H \times D_{\text{head}}}$: Sequence inputs.
*   $\Delta_t \in \mathbb{R}^{B \times H \times D_{\text{head}}}$: Time steps.
*   $A \in \mathbb{R}^{H \times D_{\text{head}}}$: Decay transitions.
*   $B_t, C_t \in \mathbb{R}^{B \times H \times D_{\text{state}}}$: State projection parameters.
*   $h_t \in \mathbb{R}^{B \times H \times D_{\text{head}} \times D_{\text{state}}}$: Hidden state tensor.
*   $y_t \in \mathbb{R}^{B \times H \times D_{\text{head}}}$: Discretized outputs.

---

## 2. Block Architecture Specification

```
Input x (B, S, D)
  │
  ├──► in_proj (projects to x, gate, delta, B, C channels)
  │      │
  │      ▼
  │    Depthwise Conv1D (kernel_size=4, causal padding)
  │      │
  │      ▼
  │    Softplus dt_bias discretization of delta
  │      │
  │      ▼
  │    selective_ssd_scan (sequential recurrence state propagation)
  │      │
  │      ▼
  │    Gated Multiplicative Scaling (gate SiLU)
  │      │
  │      ▼
  └──► out_proj (projects back to hidden dimension D)
```

---

## 3. Parameter Dimensions
For IVERI v1 Nano configuration:
*   `hidden_dim` ($D$) = 256.
*   `num_heads` ($H$) = 4.
*   `expand` factor = 2.
*   `d_inner` = 512.
*   `d_head` = 128.
*   `d_state` = 16.
