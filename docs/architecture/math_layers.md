# Mathematical Derivation & Design Notes — Phase 1.1 Core Math Layers

This document logs the mathematical definitions, implementation constraints, and design justifications for the foundational mathematical primitives built in Phase 1.1.

---

## 1. RMSNorm (Root Mean Square Layer Normalization)

### Derivation
Unlike standard LayerNorm which calculates both mean and variance, RMSNorm simplifies by ignoring the mean subtraction step:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{j=1}^d x_j^2 + \epsilon}} \otimes \gamma$$

Where:
- $x \in \mathbb{R}^d$ is the layer input.
- $\gamma \in \mathbb{R}^d$ is a learnable scaling parameter.
- $\epsilon$ is a small constant preventing division by zero.

By avoiding mean subtraction, RMSNorm reduces computational overhead by 10–50% per normalization pass while retaining similar regularization capability, as scale invariance is preserved.

### Mixed-Precision Safety
To ensure numerical stability in half-precision training (FP16/BF16), squaring large features can overflow the float bounds. We cast input values to `float32` prior to taking the power and mean, then cast back to the input dtype before scaling by weight.

---

## 2. Rotary Position Embeddings (RoPE)

### Derivation
Rotary Position Embeddings multiply query and key vectors by a rotary matrix $R_{\Theta, m}^d$, applying rotation to 2D feature coordinates:

$$\begin{pmatrix} x'_{2i} \\ x'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$

Where:
$$\theta_i = 10000^{-2(i-1)/d}$$

In vector form, we optimize this using the `rotate_half` operation:

$$\text{RoPE}(x) = x \odot \cos(m\Theta) + \text{rotate\_half}(x) \odot \sin(m\Theta)$$

$$\text{rotate\_half}(x) = \begin{pmatrix} -x_{d/2:} \\ x_{:d/2} \end{pmatrix}$$

### Caching Strategy
Cosine and sine states are cached up to `max_seq_len` to avoid re-generating position indices during every forward pass. If a sequence length exceeds the cached threshold, the cache dynamically expands on-the-fly and registers the updated buffers without resetting the model state.

---

## 3. SwiGLU (Swish Gated Linear Unit)

### Derivation
SwiGLU is a gated activation variant where the input is split and multiplied by its activated projection:

$$\text{SwiGLU}(x) = \text{Swish}(x W_g) \otimes (x W_v)$$

Where:
- $\text{Swish}(x) = x \cdot \text{Sigmoid}(x)$ (also known as SiLU in PyTorch).
- $W_g, W_v \in \mathbb{R}^{d \times d_{\text{ff}}}$ are projection parameters.

The Feed-Forward variant projects this back to hidden space:

$$\text{SwiGLUFFN}(x) = \text{SwiGLU}(x) W_o$$

Following the LLaMA standard, we compute the intermediate dimension size $d_{\text{ff}}$ as:

$$d_{\text{ff}} = \left\lfloor \frac{2}{3} \cdot 4d \right\rceil \implies \text{rounded to nearest multiple of 256 for tensor alignment.}$$

### Ablation Cleanliness
To support future comparative studies against other activations (e.g. GELU or ReLU²), SwiGLU is kept in `swiglu.py` separate from any general activations.

---

## 4. Known Limitations
- **CUDA Compilation:** Flash-Attention/Triton specific low-level kernels will run when target modules are built in Phase 1.4, but local testing is fully supported on CPU.
