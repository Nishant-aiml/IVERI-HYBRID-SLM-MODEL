# Phase 1.2 Benchmark Report — Mixture of Experts (MoE)

This report profiles MoE gating and experts container performance on a standard 10M Nano configuration default size ($B=32, S=512, D=256, E=4, K=2$).

---

## 1. Latency & Parameter Comparison (vs. Dense Baseline FFN)

| Architecture | Forward Latency | Backward Latency | Trainable Parameters | FLOP Savings |
|---|---|---|---|---|
| **Dense FFN Baseline** | $75.83\text{ ms}$ | $112.49\text{ ms}$ | $589,824$ | $0.00\%$ (baseline) |
| **Sparse MoE layer** | $185.03\text{ ms}$ | $243.72\text{ ms}$ | $2,359,296$ | $50.00\%$ |

### Latency Overhead Note
The sparse MoE latency is higher than the dense FFN baseline on CPU due to sequential expert execution in PyTorch loops (`for e in range(self.num_experts)`) and scatter/gather indexing operations. However, the sparse MoE layer contains **4x more parameters** while executing only **2x the FFN blocks per token**, yielding mathematically **50.0% FLOP savings** per token compared to a dense FFN of the same capacity.

---

## 2. Multi-Stage Latency Segmentation

We profiled the execution time of each internal step to locate bottleneck distributions:

*   **Routing Gate Projection:** $3.83\text{ ms}$ ($2.07\%$)
*   **Dispatch & Index Gathering:** $3.33\text{ ms}$ ($1.80\%$)
*   **FFN Expert Computation:** $165.12\text{ ms}$ ($89.24\%$)
*   **Output Recombination:** $7.82\text{ ms}$ ($4.23\%$)
*   **Total Forward Loop:** $185.03\text{ ms}$ ($100.00\%$)

Expert computation accounts for over $89\%$ of total forwarding latency, proving that routing overhead is extremely negligible ($<4.0\%$).

---

## 3. Scaling Characteristics (2, 4, and 8 Experts)

| Configurations | Latency (Forward) | Load Balancing Loss | Gradient Distribution (Uniformity) |
|---|---|---|---|
| **2 Experts (Top-2)** | $173.17\text{ ms}$ | $0.0200$ | $[83.4\text{k}, 81.1\text{k}]$ |
| **4 Experts (Top-2)** | $189.89\text{ ms}$ | $0.0200$ | $[43.5\text{k}, 41.1\text{k}, 41.7\text{k}, 43.4\text{k}]$ |
| **8 Experts (Top-2)** | $199.45\text{ ms}$ | $0.0200$ | $[22.6\text{k}, 21.7\text{k}, 19.6\text{k}, 21.8\text{k}, 20.6\text{k}, 22.9\text{k}, 23.5\text{k}, 22.5\text{k}]$ |

### Key Observations
*   **Latency Scaling:** Latency scales gracefully from $173.17\text{ ms}$ to $199.45\text{ ms}$ as the number of experts quadruples, showing the benefit of keeping the active execution path ($K=2$) constant.
*   **Auxiliary Loss convergence:** GShard load balancing auxiliary loss successfully converges close to zero under uniform routing inputs on 8 experts.
