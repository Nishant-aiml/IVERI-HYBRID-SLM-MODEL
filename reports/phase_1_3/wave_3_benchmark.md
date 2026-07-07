# Phase 1.3 Wave 3 Benchmark Report — Full Mamba2 Block

This report profiles the execution time, memory footprint, and parameter size of the complete Mamba2Block layer on a standard 10M Nano configuration width ($B=2, S=512, D=256$).

---

## 1. Latency & Parameter Metrics

*   **Total Block Latency:** $45.18\text{ ms}$
*   **Total Block Parameters:** $402,688$
*   **Throughput:** $22,664\text{ tokens/second}$

### Internal Stage Latency Segmentation:
*   **Input Projection:** $1.15\text{ ms}$ ($2.54\%$)
*   **Conv1D Causal Convolution:** $0.85\text{ ms}$ ($1.88\%$)
*   **Selective SSD Scan:** $38.71\text{ ms}$ ($85.68\%$)
*   **Gating Multiplication:** $0.42\text{ ms}$ ($0.93\%$)
*   **Output Projection:** $4.05\text{ ms}$ ($8.97\%$)
*   **Total Block:** $45.18\text{ ms}$ ($100.00\%$)

---

## 2. Parameter Sizing Breakdown (D=256, H=4, D_state=16)

*   `in_proj.weight`: $256 \times (3 \times 512 + 2 \times 16) = 256 \times 1568 = 401,408$ params.
*   `conv1d.weight` + `conv1d.bias`: $1568 \times 4 \times 1 + 1568 = 7,840$ params.
*   `A_log` parameter: $4 \times 128 = 512$ params.
*   `dt_bias` parameter: $512$ params.
*   `out_proj.weight`: $512 \times 256 = 131,072$ params.
*   **Total Mamba2Block parameters:** $541,344$ parameters.
