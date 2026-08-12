# IVERI CORE — Model Card

**Model Name:** IVERI Hybrid (Byte-Entropy Recurrent State-Space Transformer)  
**Architecture:** BLT Byte Segmentation + Mamba2 State-Space + Titans Neural Memory + Entropy-Guided MoE  
**Vocabulary:** Raw Bytes (259 Vocab: 256 byte values + 3 special tokens)  
**License:** Apache 2.0  
**Version:** 2.0.0-rc1  

---

## Model Description
IVERI is a byte-entropy-native hybrid architecture designed for sub-word-free language modeling. It dynamically allocates computational capacity using byte-entropy routing across recurrent state-space (Mamba2) and neural memory (Titans) layers.

---

## Intended Use
- **Primary Uses:** Sub-word-free language modeling, long-context byte processing, multi-lingual text modeling, code generation, and low-resource NLP research.
- **Out of Scope:** High-stakes medical diagnosis or unmonitored automated execution without human-in-the-loop validation.
