# IVERI CORE v1.1 — PATENT CLAIM MAPPING: NEURAL COMPUTATION OS

## Patent Claim 1: Operating System Architecture for Dynamic Neural Compute Allocation
- **Technical Description**: A global neural controller (~35K parameters) executing as an operating system to govern execution depth, memory updates, routing topology, representation granularity, and FLOP budgets online based on real-time input difficulty metrics.
- **Empirical Proof**: Achieved 100.0% Action Match and 1.0000 Utility Ratio on unseen test data (`pareto_frontier_suite.json`), dynamically allocating compute along the optimal Pareto Frontier.
- **Artifact Reference**: `model/core/global_controller.py` (lines 1-120).

## Patent Claim 2: Learnable Depth-Decaying Residual Scaling ($lpha_t$) in Dynamic Recursion
- **Technical Description**: A method for stabilizing multi-pass recursive execution in adaptive neural layers via depth-decaying scale parameters $lpha_t$ initialized as $lpha_t = lpha_0 \cdot 2^{-t}$ with pre-step RMSNorm.
- **Empirical Proof**: Eliminates gradient explosion spikes (6.8 per seed to 0.0 per seed), stabilizing backpropagation across dynamic execution paths.
- **Artifact Reference**: `model/mor/recursion.py` (lines 43-47, 164-177).

## Patent Claim 3: Constant Memory $O(1)$ Recurrent Latent Context Management
- **Technical Description**: Integration of Titans Recurrent Memory with dynamic byte boundary patchers under controller gating to enforce flat $O(1)$ memory scaling across extended sequence lengths $N$.
- **Empirical Proof**: Maintained flat 6,059.6 MB VRAM footprint at $N=16,384$ where static attention architectures crash with CUDA Out-of-Memory (`inference_sota_results.json`).
- **Artifact Reference**: `model/titans/memory.py` and `scratch/stage_c_inference_sota_suite.py`.
