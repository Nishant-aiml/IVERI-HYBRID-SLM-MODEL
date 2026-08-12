# Empirical Results: Stage B Pretraining Campaign

## 1. Hardware Characterization
- **Device:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM)
- **Batch Size x Seq Len:** 64 x 64 (4,096 tokens/step)
- **Throughput:** 1,232.3 tokens/sec in FP16 mixed precision
- **Peak VRAM:** 1,722.6 MB (42.0% utilization)

## 2. Loss & Convergence Trajectory
- **Step 1:** Loss = `5.5649`
- **Step 100:** Loss = `4.3318`
- **Step 500:** Loss = `0.1355` (`stage_b_step_500.pt`)
- **Step 800:** Loss = `3.3225`
