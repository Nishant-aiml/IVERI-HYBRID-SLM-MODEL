# IVERI CORE — Stage C Research Program (01_RESEARCH_PROGRAM.md)

## Official Project Thesis Statement

> **"Engineer and experimentally validate IVERI until it demonstrates repeatable, statistically significant advantages over Transformers on well-defined tasks and metrics."**

We treat every proposed architectural advantage as an unproven hypothesis to be validated empirically:
1. **Adaptive Computation (Hypothesis H-ADAPT):** Plausible advantage if IVERI learns to allocate compute dynamically based on token difficulty, outperforming dense Transformers in compute efficiency.
2. **Long-Context Memory (Hypothesis H-MEMORY):** Plausible advantage if Titans-style persistent memory scales better than Transformer KV-caches at sequence length $\ge 512$.
3. **Byte-Level Native Processing (Hypothesis H-BYTE):** Plausible advantage if BLT matching consistently matches tokenized models while improving robustness to noisy and multilingual text.

---

## 6 Optimization Phases

```
[Phase 1] Training Stability ──> [Phase 2] Router Optimization ──> [Phase 3] Adaptive MoR
                                                                           │
[Phase 6] Fair Benchmark <────── [Phase 5] Hotspot Profiling <───── [Phase 4] Titans Context Sweep
```

### Phase 1 — Training Stability
- LR Warmup sensitivity sweep (`Warmup = [0, 25, 50, 75, 100, 200]` steps)
- Gradient clipping sweep ($1.0, 0.5, 0.25$)
- Weight decay & AdamW $\beta_1, \beta_2$ tuning

### Phase 2 — Router Instrumentation & Optimization
- Log routing entropy, gating probabilities, expert utilization, load balancing loss, dead experts count, routing variance over time
- Temperature sweep & gating threshold tuning

### Phase 3 — Signal-Driven Adaptive MoR
- Dynamic complexity recursion depth: $f(\text{seq\_len}, \text{confidence}, \text{entropy}, \text{memory\_state})$

### Phase 4 — Titans Memory Context Sweep
- Context length sweep: $\text{seq\_len} \in [64, 128, 256, 512, 1024, 2048, 4096, 8192]$

### Phase 5 — Full Subsystem Flame-Graph Profiling
- Isolate per-module latency breakdown: BLT Patcher, Router, Titans, MoR, Decoder, CUDA memory transfers

### Phase 6 — Fair Architecture Benchmarking
- Compare IVERI v0.91, v0.91A, v0.92, v0.93, v1.0 against Dense Transformer, Mamba2, Static MoE, Dense BLT across equal token and compute budgets
