# IVERI CORE — Architectural Decision Log

| Date | Design Component | Reason | Empirical Evidence | Decision |
|---|---|---|---|---|
| 2026-07-24 | Byte Segmentation | Eliminate sub-word BPE tokenizer bias | BLT Paper & 259-vocab entropy analysis | **APPROVED** |
| 2026-07-24 | Mamba2 + Titans Hybrid | Combine fast SSM scan with long-context memory | Passkey retrieval benchmark (+17.2%) | **APPROVED** |
| 2026-07-25 | Entropy-Guided MoE | Dynamic FLOP allocation based on sequence entropy | BPB reduction on high-entropy byte spans | **APPROVED** |
