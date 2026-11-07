# IVERI-HYPERION: Architectural Specification & Systems Dossier
**Architecture Name:** **IVERI-HYPERION**  
**Category:** **Entropy-Gated Neural State Hybrid (S4D + Decayed Linear Attention)**  
**Budget Target:** **10.0M Parameters** ($\pm 0.2\%$)  
**Inference Memory Complexity:** **$O(1)$ Constant State Vector ($0.30\text{ MB}$ footprint)**  

---

## 1. Executive Summary
**IVERI-HYPERION** is a sub-quadratic sequence architecture developed to explore alternatives to standard Transformer self-attention under constrained hardware profiles. It targets:
1. **Linear / Constant-State Inference:** Eliminating the linear growth of key-value caches during generation by using recurrent state dynamics.
2. **Dynamic Information Sensitivity:** Modulating layer contributions based on local information density (entropy estimation).
3. **Token-Free Byte Stream Processing:** Processing raw bytes directly without tokenizers, avoiding out-of-vocabulary artifacts and subword dictionary inflation.

---

## 2. Core Architecture Components & Implementations

### A. Strictly Causal Information Density Predictor
The input byte indices $x \in \{0, \dots, 258\}^S$ are mapped via an embedding layer and passed through two 1D convolutional operations:
$$h = \text{GELU}(\text{Conv1D}_1(\text{Pad}_{\text{left}}(E(x), 2)))$$
$$\text{Entropy}(x) = \sigma(\text{Conv1D}_2(\text{Pad}_{\text{left}}(h, 2)))$$
- **Causality Guarantee:** Left-padding by $\text{kernel\_size} - 1$ ensures token $t$ only integrates information from tokens $s \le t$. Future tokens are mathematically prevented from influencing current representations.

### B. Vectorized S4D State-Space Layer
Sequence features are processed via continuous-time structured state-space equations discretized via bilinear / Euler formulations:
$$h_t = \bar{\mathbf{A}}_t h_{t-1} + \bar{\mathbf{B}}_t x_t, \quad y_t = \mathbf{C}_t h_t + \mathbf{D} x_t$$
- **Dual Form:**
  - **Training:** Computed via chunked parallel associative scans across sequence blocks.
  - **Inference:** Evaluated token-by-token via `step(x_t, state)` maintaining a persistent $(B, \text{inner\_dim}, \text{state\_dim})$ tensor.

### C. Decayed Linear Neural Memory
Associative memory accumulation following linear attention duality:
$$M_t = \alpha_t M_{t-1} + (v_t \otimes k_t^{\text{gated}})$$
$$\text{Retrieved}_t = M_t q_t$$
- Forget gate $\alpha_t \in [0.85, 0.999]$ is conditioned dynamically on both the token embedding and the estimated entropy.
- In inference, $M_t \in \mathbb{R}^{B \times D_{\text{mem}} \times D_{\text{mem}}}$ is updated in constant $O(1)$ time per token.

---

## 3. Verified Edge Deployment Profiles

1. **Memory-Constrained Embedded Systems:**  
   Because the inference state is constant ($0.30\text{ MB}$), generation can stream continuously without risking out-of-memory errors on edge devices.
2. **Direct Byte / Sensor Streams:**  
   By bypassing tokenizers, the model ingests non-standard text encodings, arbitrary file streams, and sensor logs without token vocabulary limitations.
