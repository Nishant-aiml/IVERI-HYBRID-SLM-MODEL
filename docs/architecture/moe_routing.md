# Mixture of Experts (MoE) Routing Architecture

This document details the mathematical formulation, token dispatch logistics, and capacity-based dropping policy implemented for the Sparse Mixture of Experts (MoE) layers in the IVERI architecture.

---

## 1. Routing & Gating Mathematical Derivations

### 1.1 Sparse Top-k Gating
Let $x \in \mathbb{R}^{D}$ be the hidden state representation of a single token. Gating coefficients are computed as:

1.  **Linear Projection to Experts Space:**
    $$H(x) = x \cdot W_g$$
    Where $W_g \in \mathbb{R}^{D \times E}$, and $E$ is the total number of experts.

2.  **Noisy Exploration Gating:**
    $$H_{\text{noise}}(x) = H(x) + \epsilon \cdot \text{Softplus}(x \cdot W_{\text{noise}})$$
    Where $W_{\text{noise}} \in \mathbb{R}^{D \times E}$ is the learnable noise weighting matrix, and $\epsilon \sim \mathcal{N}(0, I)$ is Standard Normal noise. Noise injection is active only during specific exploratory training passes to promote expert routing diversity.

3.  **Top-k Filtering:**
    $$\text{KeepTopK}(v, k)_i = \begin{cases} v_i & \text{if } v_i \text{ is in the top } k \text{ elements of } v \\ -\infty & \text{otherwise} \end{cases}$$
    By default, $k=2$, mapping each token to the two best-matching experts.

4.  **Softmax Routing Normalization:**
    $$G(x) = \text{Softmax}(\text{KeepTopK}(H_{\text{noise}}(x), k))$$

---

## 2. Load-Balancing Load Invariance Loss

To prevent expert collapse (where the router sends all tokens to a single expert, leaving others underutilized), we compute a balancing loss:

1.  **Fraction of gates per expert ($f_i$):**
    $$f_i = \frac{1}{N} \sum_{j=1}^N \mathbb{I}(\text{Expert } i \text{ is selected for token } j)$$

2.  **Fraction of routing weight per expert ($P_i$):**
    $$P_i = \frac{1}{N} \sum_{j=1}^N G(x_j)_i$$

3.  **Auxiliary Loss ($L_{\text{aux}}$):**
    $$L_{\text{aux}} = E \cdot \sum_{i=1}^E f_i \cdot P_i$$

---

## 3. Expert Capacity & Token Dropping Logistics

To prevent execution bottlenecks caused by load imbalance, we bound the computation capacity of each expert:

$$\text{Capacity} = \left\lceil \frac{N \times K}{E} \times \text{Capacity Factor} \right\rceil$$

Where:
- $N$ is total tokens in the sequence.
- $K$ is active experts selected per token (default $2$).
- $E$ is total experts (default $4$).
- **Capacity Factor (CF):** Defaults to $1.25$.

### Token Dropping Flow
```
Token Inputs x (B, S, D)
     │
     ├──► Gating Router ──► [Weights & Indices]
     │
     └──► Dispatcher (Gather input tokens matching active experts)
             │
             ├──► Under Capacity? ──► Execute SwiGLUFFN expert
             │
             └──► Exceeds Capacity? ─► DROP Token (Bypass expert computation, propagate residual stream)
```

Tokens that exceed the capacity limit of their target expert are **dropped** from FFN execution. The output contribution of the expert pathway for dropped tokens is set to zero, so they proceed solely through the residual connection block.
