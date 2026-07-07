# Component Build Order — Phase 1

This document specifies the sequence in which individual model components will be built during Phase 1.

---

## Build Steps

```mermaid
gantt
    title Phase 1 Build Sequence
    dateFormat  X
    axisFormat %d
    
    section Core Math Layers
    1.1: RMSNorm, RoPE, SwiGLU :active, step1_1, 0, 3
    
    section Specialized Blocks
    1.2: Sparse MoE Layer       :step1_2, after step1_1, 3d
    1.3: Mamba2 Block           :step1_3, after step1_1, 3d
    1.4: Flash Attention Wrapper:step1_4, after step1_1, 2d
    1.5: Mixture of Recursions  :step1_5, after step1_1, 3d
    
    section BLT & Titans
    1.6: BLT Tokenization & Encoder :step1_6, after step1_1, 4d
    1.7: Titans Memory Module   :step1_7, after step1_1, 4d
    
    section Assembly
    1.8: Backbone Block Assembly:step1_8, after step1_2 step1_3 step1_4 step1_5, 3d
    1.9: Full Model Integration :step1_9, after step1_6 step1_7 step1_8, 3d
```

### Detailed Order

1.  **Step 1.1 — Basic Layers (RMSNorm, RoPE, SwiGLU):** High-efficiency normalization, rotational position embeddings, and activation function.
2.  **Step 1.2 — MoE Experts:** Experts FFN blocks + routing logic.
3.  **Step 1.3 — Mamba2:** State Space Model layers.
4.  **Step 1.4 — Attention:** Wrapped scaled-dot product and Flash Attention routines.
5.  **Step 1.5 — MoR Router:** Boundary routers controlling recursion loops.
6.  **Step 1.6 — BLT Components:** Entropy predictors, patchers, byte encoders, and decoders.
7.  **Step 1.7 — Titans Memory:** Linear memory banks + updating kernels.
8.  **Step 1.8 — Backbone Integration:** Aggregation of normalized sub-layers into the backbone block.
9.  **Step 1.9 — Assembly:** Hooking the BLT Encoder/Decoder to the backbone to form the compile-ready `IVERI` model.
