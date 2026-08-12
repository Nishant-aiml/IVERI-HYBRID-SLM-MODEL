# IVERI CORE — Experiment Execution (02_EXPERIMENT_EXECUTION.md)
**Profiling-First & Evidence-Gated Architecture Optimization**

---

## Profiling-First Workflow

No architectural modification occurs without preceding profiling evidence:

```
[1] Profiling ──> [2] Locate Bottleneck ──> [3] Hypothesis ──> [4] Optimization ──> [5] Benchmark
                                                                                          │
[10] Patent/Paper <── [9] Decision Database <── [8] Stats Audit <── [7] Threshold Audit <─┘
```

---

## Predefined Acceptance Criteria Thresholds

An architectural change is accepted into a new model version ONLY if it satisfies at least one threshold:

$$\text{Acceptance Condition: } \Delta\text{Throughput} \ge +10\% \quad \text{OR} \quad \Delta\text{Loss} \le -3.0\% \quad \text{OR} \quad \Delta\text{VRAM} \le -10.0\%$$

If none of these thresholds are met, the modification is **REJECTED & REVERTED**.

---

## Signal-Driven Adaptive MoR Policy

Rather than hardcoded rules, MoR depth is dynamically allocated via a learned/estimated complexity signal:

$$\text{Depth}(x) = f\Big(\text{seq\_len}(x), \, \text{Router\_Confidence}(x), \, \text{Entropy}(x), \, \text{Memory\_State}(x)\Big)$$

- **Simple Tokens (High Confidence, Low Entropy):** Depth = 0 (Skip recursion, fast path)
- **Moderate Tokens:** Depth = 1–2
- **Complex Tokens (Low Confidence, High Entropy):** Depth = 4 (Full recursive reasoning)

---

## Comprehensive Router Instrumentation Metrics

Every training & inference step logs:
1. Router Entropy ($H$)
2. Expert Allocation Probability Distribution
3. Expert Utilization & Load Balance Ratio
4. Dead Experts Count ($\text{weight} < 10^{-8}$)
5. Token Allocation Matrix
6. Routing Variance across steps
7. Routing Confidence ($\max p_i$)
8. Router Gradient Norms & Latency Breakdown

---

## Complete Inference Benchmark Suite

In addition to pretraining, evaluate inference capabilities:
- **Time-to-First-Token (TTFT / Cold Start)**
- **Streaming Generation Throughput (tok/s)**
- **Peak VRAM & KV-Cache Growth**
- **Energy Consumption per Token (Joules/token)**
- **Long-Sequence Generation Stability**
