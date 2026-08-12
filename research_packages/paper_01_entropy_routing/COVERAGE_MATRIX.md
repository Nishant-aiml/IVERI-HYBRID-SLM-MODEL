# Coverage Matrix: paper_01_entropy_routing
**Paper Topic:** *Entropy-Guided Dynamic Sparse Expert Routing for Efficient Byte-Level Language Models*

| Requirement / Component | Evidence Source | Status | Confidence |
|:---|:---|:---:|:---:|
| **Source Code** | `model/iveri_core.py` & `model/blt/` | ✅ Complete | High |
| **Unit Tests (640/640)** | `tests/` | ✅ Complete | High |
| **Hardware Sweep** | `reports/HARDWARE_CHARACTERIZATION_SWEEP.json` | ✅ Complete | High |
| **FP16 Speedup Proof** | `reports/OPTIMIZATION_BENCHMARKS.json` | ✅ Complete | High |
| **100-Step Pilot** | `reports/PILOT_FP16_BENCHMARK.json` | ✅ Complete | High |
| **Stage B 10M Tokens** | `reports/live_training/stage_b_10m_loss.csv` | 🟡 In Progress | Medium |
| **3-Seed Statistical Runs** | `scientific_campaigns/stage_c/` | 🔴 Pending Stage C | Awaiting |
| **Compute-Matched Baselines** | `scientific_campaigns/stage_c/` | 🔴 Pending Stage C | Awaiting |
| **Component Ablation Suite** | `scientific_campaigns/stage_c/` | 🔴 Pending Stage C | Awaiting |
| **Reviewer Verification** | `proof/verify_everything.py` | 🟡 Active | Medium |
