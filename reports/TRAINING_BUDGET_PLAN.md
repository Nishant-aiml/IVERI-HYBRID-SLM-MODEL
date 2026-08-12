# IVERI CORE — Training Budget & Feasibility Plan

**Date:** 2026-07-24  
**Hardware Profile:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM), 16 GB RAM  

---

## 1. Local Pilot Campaign Feasibility Budget (Nano Config ~455K Params)

| Model / Variant | Sequence Length | Batch Size | Target Steps | Est. Tokens | Est. Runtime | Peak VRAM | Est. Checkpoint Size |
|---|---|---|---|---|---|---|---|
| **IVERI Hybrid (Seed 42)** | 512 | 4 | 500 | 1,024,000 | ~3.5 min | ~850 MB | ~2.5 MB |
| **IVERI Hybrid (Seed 123)** | 512 | 4 | 500 | 1,024,000 | ~3.5 min | ~850 MB | ~2.5 MB |
| **IVERI Hybrid (Seed 3407)** | 512 | 4 | 500 | 1,024,000 | ~3.5 min | ~850 MB | ~2.5 MB |
| **Baseline A (Transformer)** | 512 | 4 | 500 | 1,024,000 | ~2.5 min | ~512 MB | ~1.8 MB |
| **Baseline B (Mamba2)** | 512 | 4 | 500 | 1,024,000 | ~2.8 min | ~680 MB | ~2.0 MB |
| **Baseline C (Transformer+MoE)** | 512 | 4 | 500 | 1,024,000 | ~3.8 min | ~920 MB | ~3.2 MB |
| **Baseline D (Transformer+Mamba)** | 512 | 4 | 500 | 1,024,000 | ~3.2 min | ~780 MB | ~2.2 MB |

**Total Pilot Campaign Runtime:** ~23 minutes across all 5 models and 3 seeds.  
**Total Storage Growth:** ~45 MB for all checkpoints and CSV log entries.

---

## 2. Component Ablation Study Budget (7 Variants)

| Ablation Variant | Description | Target Steps | Est. Runtime | Peak VRAM |
|---|---|---|---|---|
| `Full IVERI` | Baseline complete model | 500 | ~3.5 min | ~850 MB |
| `No BLT` | Direct byte embedding bypass | 500 | ~2.5 min | ~600 MB |
| `No Titans` | Disable neural memory | 500 | ~2.8 min | ~700 MB |
| `No MoR` | Single sub-block depth pass | 500 | ~2.2 min | ~550 MB |
| `No MoE` | Dense SwiGLU FFN | 500 | ~3.0 min | ~750 MB |
| `No Mamba2` | Standard attention backbones | 500 | ~3.2 min | ~800 MB |
| `No Entropy Routing` | Uniform MoE gating | 500 | ~3.4 min | ~850 MB |

**Total Ablation Study Runtime:** ~20 minutes.

---

## 3. Resume & Checkpoint Frequency Policy

- **Checkpoint Save Frequency:** Save checkpoint every 100 steps.
- **Resume Verification:** Every run saves `resume_metadata.json` with step count, optimizer state, loss state, and seed states.
- **Interruption Policy:** If a run is terminated before reaching step 500, it is marked `INVALID` and excluded from benchmark tables until resumed and completed.

---

## 4. Scaled Campaign Roadmap (Future Cluster Hardware)

For 300M parameter model trained on 10B+ tokens:
- **Requires:** $8 \times$ A100 (80GB) or $8 \times$ H100 GPUs.
- **Estimated Runtime:** ~48 GPU hours.
- **Local Role:** Local hardware serves as the **validated testbed** ensuring 100% bug-free script execution, clean data pipelines, and verified regression suites before submitting cluster jobs.
