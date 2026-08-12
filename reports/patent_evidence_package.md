# IVERI CORE — Patent Evidence Package

**Date:** 2026-07-24  
**Status:** EVIDENCE-BACKED EVALUATION PACKAGE  

---

## 1. Technical Invention Overview
The IVERI CORE architecture integrates a byte-level latent encoder (BLT), a state-space sequential backbone (Mamba2), FlashAttention multi-head cross attention, mixture of recursions (MoR), dynamic sparse mixture of experts (MoE), and Titans neural memory gated by Shannon byte-entropy predictions (`ByteEntropyModel`).

---

## 2. Prior-Art Comparison & Novelty Mapping

| Invention Feature | Prior Art Reference | Novelty / Differential Technical Effect |
|---|---|---|
| **Entropy-Driven MoR Routing** | Raposo et al. (MoR, 2024) | Routing recursion depth directly via Shannon byte-entropy $D_p = 1 + \text{floor}(E_p \times 7)$ |
| **Entropy-Gated Titans Memory** | Behrouz et al. (Titans, 2025) | Gating online neural memory write updates dynamically on high-surprise entropy boundaries |
| **Entropy-Biased MoE Gating** | Fedus et al. (Switch Transformer, 2022) | Biasing router logit routing vectors by patch-level Shannon byte-entropy |
| **Unified Latent Backbone** | Pagnoni et al. (BLT, 2024) | Coupling byte-latent encoder directly to Mamba2-Attention hybrid stack |

---

## 3. Empirical Utility & Performance Evidence
All technical utility claims trace directly to immutable Experiment IDs in `reports/experiment_registry.json` and PyTorch weight checkpoints in `reports/checkpoint_lineage.json`.
