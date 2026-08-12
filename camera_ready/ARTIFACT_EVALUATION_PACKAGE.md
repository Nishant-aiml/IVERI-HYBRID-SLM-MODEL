# Phase 26: Artifact Evaluation & Reproducibility Package

**Target Review Standard**: IEEE / NeurIPS / ICLR Artifact Evaluation Badge (Available & Reproduced)  
**Audit Date**: 2026-07-30 19:36:00 UTC  
**Hardware Specification**: NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)  
**Status**: 🏆 **ARTIFACT EVALUATION PACKAGE SEALED**  

---

## 📦 Artifact Package Components

1. **One-Command Reproduction Script**:
   - Executable path: [`camera_ready/Reproduce.sh`](file:///c:/Users/datta.000/Desktop/iveri%20core%20nexus/iveri-core/camera_ready/Reproduce.sh)
   - Runs full benchmarking suite, statistical Welch t-tests, and generates publication tables in < 60 seconds.

2. **Docker Container Environment**:
   - File path: [`camera_ready/Dockerfile`](file:///c:/Users/datta.000/Desktop/iveri%20core%20nexus/iveri-core/camera_ready/Dockerfile)
   - Container Image: `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`

3. **Master Verification Scripts**:
   - [`scratch/execute_phase25_benchmark_framework.py`](file:///c:/Users/datta.000/Desktop/iveri%20core%20nexus/iveri-core/scratch/execute_phase25_benchmark_framework.py)
   - [`scratch/execute_all_tier_matched_comparisons.py`](file:///c:/Users/datta.000/Desktop/iveri%20core%20nexus/iveri-core/scratch/execute_all_tier_matched_comparisons.py)
   - [`scratch/fix_exact_35m_fair_comparison.py`](file:///c:/Users/datta.000/Desktop/iveri%20core%20nexus/iveri-core/scratch/fix_exact_35m_fair_comparison.py)

4. **Trained Checkpoint**:
   - Checkpoint path: [`training_campaign/checkpoints/iveri_medium_45m_checkpoint.pt`](file:///c:/Users/datta.000/Desktop/iveri%20core%20nexus/iveri-core/training_campaign/checkpoints/iveri_medium_45m_checkpoint.pt) (**69,153,298 Parameters / 69.15M**).
