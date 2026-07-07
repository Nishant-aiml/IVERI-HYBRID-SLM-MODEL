# IVERI CORE — Phase Dependency Graph

> This document defines the dependency relationships between all implementation
> phases and steps. Before starting any phase or step, verify that all
> dependencies have successfully passed their exit gates.

## Graph

```mermaid
graph TD
    P0["Phase 0: Foundation"] --> P1_1["1.1: RMSNorm + RoPE + SwiGLU"]
    P0 --> P1_2["1.2: MoE FFN"]
    P0 --> P1_3["1.3: Mamba2 Block"]
    P0 --> P1_4["1.4: Flash Attention"]
    P0 --> P1_5["1.5: MoR Router"]
    P0 --> P1_6["1.6: BLT Components"]
    P0 --> P1_7["1.7: Titans Memory"]

    P1_1 --> P1_8["1.8: Backbone Block"]
    P1_2 --> P1_8
    P1_3 --> P1_8
    P1_4 --> P1_8
    P1_5 --> P1_8

    P1_6 --> P1_9["1.9: Full Model Assembly"]
    P1_7 --> P1_9
    P1_8 --> P1_9

    P1_9 --> P2_1["2.1: Data Pipeline"]
    P1_9 --> P2_2["2.2: Training Loop"]

    P2_1 --> P2_3["2.3: Train 5M, 100 Steps"]
    P2_2 --> P2_3

    P2_3 --> P2_4["2.4: Train 20M, 1000 Steps"]

    P2_4 --> P3["Phase 3: First Benchmark"]
    P3 --> P4["Phase 4: Scale Incrementally"]
    P4 --> P5["Phase 5: Instruction Tuning"]
    P5 --> P6["Phase 6: Research Stage"]
```

## Dependency Table

| Step | Depends On | Description |
|------|-----------|-------------|
| **Phase 0** | — | Foundation & Infrastructure |
| **1.1** | Phase 0 | RMSNorm, RoPE, SwiGLU |
| **1.2** | Phase 0 | MoE Router + Expert FFNs |
| **1.3** | Phase 0 | Mamba2 SSM Block |
| **1.4** | Phase 0 | Flash Attention Wrapper |
| **1.5** | Phase 0 | MoR Router + Recursion Engine |
| **1.6** | Phase 0 | BLT Entropy Model, Patcher, Encoder, Decoder |
| **1.7** | Phase 0 | Titans Memory, Updater, LR Generator |
| **1.8** | 1.1, 1.2, 1.3, 1.4, 1.5 | Backbone Block Assembly |
| **1.9** | 1.6, 1.7, 1.8 | Full Model Assembly |
| **2.1** | 1.9 | Data Pipeline (TinyStories) |
| **2.2** | 1.9 | Training Loop |
| **2.3** | 2.1, 2.2 | Train 5M Model, 100 Steps |
| **2.4** | 2.3 | Train 20M Model, 1000 Steps |
| **Phase 3** | 2.4 | First Benchmark vs Baselines |
| **Phase 4** | Phase 3 | Scale to 50M, 123M |
| **Phase 5** | Phase 4 | Instruction Tuning + Chat |
| **Phase 6** | Phase 5 | Research Publications + Patents |

## Rules

1. **Never skip dependencies.** Every dependency must have passed its exit gate.
2. **Steps within Phase 1 (1.1–1.7)** can be built in parallel — they only depend on Phase 0.
3. **Step 1.8** is the first integration point — it requires 1.1 through 1.5.
4. **Step 1.9** is the full integration — it requires everything in Phase 1.
5. **Phase 2** cannot begin until 1.9 passes its sanity check.
