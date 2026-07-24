# 00 Baseline Selection Justification

This document details the scientific rationale for selecting the five reference models:

1. **Baseline A (BaselineTransformer)**: Canonical Byte-level Causal Transformer. Represents dense attention standards.
2. **Baseline B (TinyMamba / BaselineMamba2)**: Pure selective State Space Model. Represents non-attention recurrence standards.
3. **Baseline C (BaselineTransformerMoE)**: Causal Attention + Sparse MoE FFN. Isolates sparse expert routing effects.
4. **Baseline D (BaselineHybrid)**: Alternating Mamba2 SSM + FlashAttention layers. Isolates Mamba+Attention hybrid dynamics.
5. **Baseline E (IVERIModel)**: Full frozen hybrid architecture incorporating BLT, Titans memory, Mamba2, MoR, MoE, and Entropy routing.
