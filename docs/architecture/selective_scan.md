# Mamba2 Selective Scan Architecture

This document details the mathematical logic, shape layout, tensor contraction, and causal properties of the selective Structured State Space Duality (SSD) scan recurrence algorithm.

---

## 1. Recurrence Mathematical Formulation

Structured State Space Duality (SSD) computes time-varying sequence propagation using a linear state space recurrence relation:

1.  **Transition Discretization:**
    $$\bar{A}_t = \exp(\Delta_t \cdot A)$$
    Where $A \in \mathbb{R}^{H \times D_{\text{head}}}$ and $\Delta_t \in \mathbb{R}^{B \times H \times D_{\text{head}}}$.

2.  **Input Discretization Outer Product:**
    $$x'_t = x_t \odot \Delta_t$$
    $$\bar{B}_t = x'_t \otimes B_t$$
    Where $B_t \in \mathbb{R}^{B \times H \times D_{\text{state}}}$, $x_t \in \mathbb{R}^{B \times H \times D_{\text{head}}}$, and the outer product $\bar{B}_t$ has shape $(B, H, D_{\text{head}}, D_{\text{state}})$.

3.  **State Recurrent Propagation:**
    $$h_t = \bar{A}_t \odot h_{t-1} + \bar{B}_t$$
    Where $h_t \in \mathbb{R}^{B \times H \times D_{\text{head}} \times D_{\text{state}}}$ is the state tensor at sequence position $t$.

4.  **Causal State projection:**
    $$y_t = h_t \cdot C_t = \sum_{d=1}^{D_{\text{state}}} h_{t, \cdot, \cdot, \cdot, d} \cdot C_{t, \cdot, \cdot, d}$$
    Where $C_t \in \mathbb{R}^{B \times H \times D_{\text{state}}}$.

---

## 2. Shape Layout & Contractions Flow

```
Input x (B, H, S, D_head) ──┐
                            ├──► x_t * delta_t ──► otimes B_t ──► input_term (B, H, D_head, D_state)
Delta delta (B, H, S, D_head) ┘                                                   │
                                                                                 ▼
                                                      state_t = A_bar_t * state_{t-1} + input_term
                                                                                 │
                                                                                 ▼
Output y_t (B, H, D_head) ◄────────────────────────────────────────────── state_t . C_t
```

At each sequence step $t$:
*   State update step contracts elementwise multiplication of $\bar{A}_t$ along $D_{\text{head}}$, updating state tensor of shape $(B, H, D_{\text{head}}, D_{\text{state}})$.
*   Output step contracts output parameter $C_t$ along $D_{\text{state}}$ to return step output $y_t$ of shape $(B, H, D_{\text{head}})$.
