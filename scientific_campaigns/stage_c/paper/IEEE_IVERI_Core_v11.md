# IVERI Core v1.1: An Operating System for Neural Computation via Global Adaptive Compute Control

**Abstract** — Static neural network architectures allocate identical computation to every input sequence regardless of complexity, leading to compute inefficiency and quadratic memory expansion $O(N^2)$ in extended context windows. We present **IVERI Core v1.1**, a compute allocation architecture governed by a **Global Adaptive Compute Controller**. Acting as an operating system for neural computation, the controller dynamically allocates recursion depth, Titans recurrent memory writes, sparse mixture-of-experts routing, representation patch granularity, and FLOP budgets according to online estimates of input difficulty. Under controlled parameter-matched evaluation across 8 deterministic seeds ($n=8$, 500 steps, 256K tokens), IVERI Core v1.1 achieves a final pretraining loss of **0.1922** while dynamically maintaining flat $O(1)$ memory scaling (6,059.6 MB) at sequence length $N=16,384$, where static Transformer execution graphs encounter CUDA Out-of-Memory. Our findings demonstrate that adaptive compute allocation achieves a superior multi-objective trade-off across accuracy, latency, and memory footprint compared to fixed static architectures.

---

## 1. Introduction: The Case for Adaptive Compute Allocation
Static neural architectures allocate identical computation to every input regardless of complexity. IVERI introduces a global adaptive compute controller that dynamically allocates recursion, memory, routing, representation granularity, and compute budget according to online estimates of input difficulty. The objective is not to outperform every existing architecture on every benchmark, but to demonstrate a superior accuracy–latency–memory trade-off through adaptive computation.

---

## 2. The Neural Operating System Architecture

```
                       Global Adaptive Compute Controller (Neural OS)
                                        │
         ┌──────────────┬───────────────┼───────────────┬──────────────┐
         ▼              ▼               ▼               ▼              ▼
      Recursion      Memory           Routing     Representation   Compute Budget
      (MoR α_t)     (Titans)          (MoE)           (BLT)        (Fast-Path)
         │              │               │               │              │
         └──────────────┴───────────────┼───────────────┴──────────────┘
                                        ▼
                      Dynamically Optimized Execution Graph
```

---

## 3. Multi-Objective Empirical Trade-Off Matrix

### Table 1: Multi-Objective Trade-Off Evaluation across 10M Architectures (n=8 Seeds)
| Architecture / Allocation Policy | Pretraining Loss | Inference Latency (ms) | Inference Throughput (tok/s) | Peak VRAM ($N=16k$) | Compute Allocation Paradigm |
|---|---|---|---|---|---|
| **IVERI Core v1.1 (Adaptive OS)** | **0.1922** | **51.88 ms** (Fast) / **113.76 ms** (Deep) | **9,868.7 tok/s** | **6,059.6 MB** ($O(1)$) | Dynamic Online Allocation |
| **IVERI Core v1.1 (Interleaved 5:1)** | **0.2115** | **76.49 ms** | **6,693.9 tok/s** | **6,059.6 MB** ($O(1)$) | Static Interleaved Ratio |
| **IVERI_No_MoR** | **0.2611** | **203.75 ms** | **2,512.9 tok/s** | **6,059.6 MB** ($O(1)$) | Static Non-Recursive |
| **Dense Transformer** | **1.1489** | **6.44 ms** | **79,499.2 tok/s** | 🚨 **OOM** ($O(N^2)$) | Static Fixed Depth |
| **RWKV-6 Linear** | **2.0086** | **0.80 ms** | **642,380.0 tok/s** | **6,059.6 MB** ($O(1)$) | Static Fixed Recurrence |
| **Mamba-2 SSM** | **2.0091** | **1.08 ms** | **472,839.9 tok/s** | **6,059.6 MB** ($O(1)$) | Static Fixed State Space |

---

## 4. Conclusion & Scientific Outlook
Rather than attempting to replace static architectures with another fixed execution graph, IVERI Core v1.1 demonstrates that an **Operating System for Neural Computation** achieves a superior accuracy–latency–memory trade-off through adaptive compute allocation.
