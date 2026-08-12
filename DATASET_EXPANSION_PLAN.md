# IVERI CORE — Dataset Expansion Plan

**Date:** 2026-07-24  
**Pipeline Base:** Phase 8 Certified Dataset Pipeline (`data/pipeline/`)  

---

## 1. Executive Summary

This document defines the structured progression from validation-scale engineering testbeds (Phase A) to publication-grade research datasets (Phase B & Phase C).

The repository enforces a strict **Dataset Scale Gate**:
> Validation-scale datasets (< 1 GB) are strictly reserved for engineering pipeline validation, regression tests, and architecture health verification. No scientific benchmark, IEEE publication bundle, or patent package may be generated from a validation-scale dataset.

---

## 2. Expansion Roadmap

### Phase A: Engineering Validation Scale (Current Baseline)
- **Datasets:** `tinystories`, `fineweb_edu`, `wikipedia`, `finemath`, `dclm` (Mock Shards / Subsets)
- **Target Storage:** ~142 KB to 50 MB
- **Target Tokens:** ~1M tokens
- **Target Hardware:** Local Host CPU / RTX 3050 Laptop GPU (4 GB VRAM)
- **Scope:** Unit testing, dataloader integration, loss masking verification, architecture health checks, trainer resume tests.
- **Scientific Status:** `ENGINEERING VALIDATION ONLY` (Not suitable for publication claims).

### Phase B: Research Validation Scale (Local Pilot Campaign)
- **Datasets:**
  - `TinyStories` (Complete dataset ~500 MB)
  - `FineWeb-Edu` (Certified 1 GB sample shard)
  - `Wikipedia` (English dump sample ~200 MB)
  - `FineMath` (Selected mathematical reasoning subset ~100 MB)
  - `DCLM` (Filtered web text subset ~100 MB)
- **Target Storage:** **~1.9 GB total certified data**
- **Target Tokens:** ~500M tokens
- **Target Hardware:** RTX 3050 Laptop GPU (4 GB VRAM), 16 GB RAM
- **Scope:** Multi-seed baseline comparisons (Seeds 42, 123, 3407), 7-variant component ablations, loss convergence curves, throughput and VRAM profiling.
- **Scientific Status:** `VALID FOR PILOT RESEARCH BENCHMARKS` (Controlled comparative baseline claims).

### Phase C: Production Scaled Pretraining (Cluster Acceleration)
- **Datasets:** Full certified FineWeb-Edu 10B+ token corpus + RedPajama/The Stack v2.
- **Target Storage:** **~50+ GB total certified data**
- **Target Tokens:** ~10B+ tokens
- **Target Hardware:** Multi-GPU Cluster ($8 \times$ A100 / H100)
- **Scope:** 300M parameter foundation model pretraining, SFT instruction tuning, SimPO/DPO alignment, HumanEval/MBPP code benchmarks.
- **Scientific Status:** `PUBLICATION-GRADE FOUNDATION MODEL`.

---

## 3. Phase Transition Checklist

| Transition | Requirements | Verification Artifact | Status |
|---|---|---|---|
| **Phase A $\to$ Phase B** | Phase 1.5 Audit Closure PASS; 1.9 GB dataset download & certification; 10 dataset gates PASS | `TRAINING_READINESS_CERTIFICATE.md` | `READY FOR EXECUTION` |
| **Phase B $\to$ Phase C** | Phase 3B pilot campaign completion; statistical significance (p < 0.05) verified; GPU cluster allocation | `SCIENTIFIC_FREEZE_CERTIFICATE.md` | `FUTURE ROADMAP` |
