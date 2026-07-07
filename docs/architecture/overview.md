# IVERI CORE — Architectural Overview

This document provides a high-level overview of the IVERI CORE hybrid SLM architecture.

## 1. Vision & Core Philosophy

IVERI (Byte-Entropy-Native Hybrid Model) is designed to operate directly on raw bytes, bypassing tokenizers completely. The architecture merges several cutting-edge ML research paradigms into a cohesive, resource-optimized, high-throughput system:

*   **Byte Latent Transformer (BLT):** Groups variable-length byte groups into structured latent patches based on a dynamic entropy boundary model.
*   **Titans Neural Memory:** Enhances long-term context retention using a fast-update linear memory module (MLP-updater).
*   **Mamba2 State-Space Model:** Accelerates sequence modeling via high-throughput structured state spaces.
*   **Mixture of Recursions (MoR):** Dynamically loops and routes representation vectors to adapt depth based on task complexity.
*   **Mixture of Experts (MoE):** Allocates sparse computation through sparse routing mechanisms.

---

## 2. Global Pipeline

```
[Raw Bytes Input] 
       │
       ▼
 [Entropy Model] ──(Dynamic Boundary Patcher)──> [Byte Latent Patches]
                                                          │
                                                          ▼
                                                  [BLT Byte Encoder]
                                                          │
                                                          ▼
                                                [Backbone Blocks × L]
                                                ├── Titans Neural Memory
                                                ├── Mamba2 SSM
                                                ├── Flash Attention
                                                └── Sparse MoE FFN (with MoR Routing)
                                                          │
                                                          ▼
                                                  [BLT Byte Decoder]
                                                          │
                                                          ▼
                                                  [Predict Next Byte]
```
