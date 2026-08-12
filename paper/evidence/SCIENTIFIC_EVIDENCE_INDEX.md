# IVERI CORE — Scientific Evidence Index & Zero-Trust Audit
**Paper Topic:** *Entropy-Guided Dynamic Sparse Expert Routing for Efficient Byte-Level Language Models*  
**Last Build:** `2026-07-25T13:06:20Z`  
**Policy:** Zero-Trust Audit — No manual numbers, no estimations, no unverified claims.

---

## 📌 Scientific Claims & Evidence Status

| Claim ID | Scientific Claim | Evidence Level | Source Experiment | Raw Artifact | Verification Command | Status |
|:---|:---|:---:|:---:|:---|:---|:---:|
| **CLAIM-01** | RTX 3050 hardware supports batch=64 FP16 pretraining without OOM | Direct Empirical | `EXP-0002`, `EXP-0004` | `reports/EMPIRICAL_GPU_BENCHMARK.json` | `python paper/scripts/verify_ieee_paper.py --claim CLAIM-01` | **PASS** |
| **CLAIM-02** | FP16 + GradScaler provides 8.15x speedup over FP32 on RTX 3050 Tensor Cores | Direct Empirical | `EXP-0003` | `reports/OPTIMIZATION_BENCHMARKS.json` | `python paper/scripts/verify_ieee_paper.py --claim CLAIM-02` | **PASS** |
| **CLAIM-03** | 100-step pretraining exhibits monotonic loss convergence (5.56 -> 0.81) | Direct Empirical | `EXP-0004` | `reports/PILOT_FP16_BENCHMARK.json` | `python paper/scripts/verify_ieee_paper.py --claim CLAIM-03` | **PASS** |
| **CLAIM-04** | Entropy-guided dynamic routing reduces FLOPs vs Dense BLT baseline | Pending Campaign | `EXP-0005` | `reports/live_training/stage_b_10m_loss.csv` | `python paper/scripts/verify_ieee_paper.py --claim CLAIM-04` | **IN_PROGRESS** |
| **CLAIM-05** | Titans neural memory improves long-context retrieval > 128k bytes | Pending Campaign | Baseline Suite | `evaluation/results/long_context.json` | `python paper/scripts/verify_ieee_paper.py --claim CLAIM-05` | **AWAITING_STAGE_C** |
| **CLAIM-06** | Mixture of Recursions (MoR) lowers step latency by 18-25% | Pending Campaign | Ablation Suite | `experiments/ablations/mor_ablation.csv` | `python paper/scripts/verify_ieee_paper.py --claim CLAIM-06` | **AWAITING_STAGE_C** |

---

## 🔬 Registered Empirical Experiments (`EXP-0001` to `EXP-0004`)

### `EXP-0001`: Stage A (Nano CPU Baseline)
- **Git Commit:** `2f31069`
- **Hardware:** `NVIDIA GeForce RTX 3050 Laptop GPU` (Driver: `610.47`)
- **Throughput:** `16.8 tok/sec` | **Peak VRAM:** `0.0 MB`
- **Loss:** `5.54 -> 2.97`
- **Notes:** Initial 36-step CPU baseline run establishing monotonic loss reduction.

### `EXP-0002`: Hardware Characterization Sweep
- **Git Commit:** `2f31069`
- **Hardware:** `NVIDIA GeForce RTX 3050 Laptop GPU` (Driver: `610.47`)
- **Throughput:** `984.8 tok/sec` | **Peak VRAM:** `2853.5 MB`
- **Loss:** `5.5 -> 4.09`
- **Notes:** Empirical hardware characterization sweep across 10 batch x seq_len combos.

### `EXP-0003`: Mixed Precision Benchmarking
- **Git Commit:** `2f31069`
- **Hardware:** `NVIDIA GeForce RTX 3050 Laptop GPU` (Driver: `610.47`)
- **Throughput:** `4167.3 tok/sec` | **Peak VRAM:** `189.0 MB`
- **Loss:** `5.5 -> 4.12`
- **Notes:** Discovered 8.15x FP16 speedup over FP32 on RTX 3050 Tensor Cores.

### `EXP-0004`: Stage A (100-Step Stable FP16 Pilot)
- **Git Commit:** `2f31069`
- **Hardware:** `NVIDIA GeForce RTX 3050 Laptop GPU` (Driver: `610.47`)
- **Throughput:** `1149.3 tok/sec` | **Peak VRAM:** `1722.6 MB`
- **Loss:** `5.5603 -> 0.8143`
- **Notes:** End-to-end 100-step stable FP16 pilot run. Monotonic loss 5.56 -> 0.81, zero NaNs, 1.72GB peak VRAM.

