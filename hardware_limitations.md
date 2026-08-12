# IVERI CORE — Hardware Limitations & Operational Constraints

**Date:** 2026-07-24  
**Primary Execution Host:** Local Development Environment  

---

## 1. System Specifications

| Resource | Specification | Notes / Constraints |
|---|---|---|
| **GPU Model** | NVIDIA GeForce RTX 3050 Laptop GPU | Ampere Architecture |
| **VRAM Capacity** | **4,096 MB (4 GB GDDR6)** | Severe memory ceiling for LLM training |
| **CUDA Cores** | 2,048 cores | FP32 / FP16 Tensor Core acceleration |
| **CUDA Driver / Runtime** | CUDA 12.1 / PyTorch 2.5.1+cu121 | Full autograd & amp support |
| **Host CPU** | Intel x86_64 Family 6 Model 154 (14 cores) | High multi-threading capacity |
| **System RAM** | **16.0 GB DDR4/DDR5** | Dataset streaming & loader buffer limit |
| **Host OS** | Windows 11 64-bit | Local fallback paths enabled |

---

## 2. Operational Constraints & Parameter Budgeting

### Memory Ceilings (4 GB VRAM limit)
- **Maximum Batch Size per GPU Step:** $B=1$ to $B=4$ depending on sequence length.
- **Maximum Sequence Length ($S$):**
  - For $S = 512$ bytes: Batch Size $B \le 8$
  - For $S = 1024$ bytes: Batch Size $B \le 4$
  - For $S = 4096$ bytes: Batch Size $B = 1$ (AMP FP16/BF16 required)
- **Model Parameter Budget (Local Training):**
  - **Nano / Micro Tier (~455K to 10M parameters):** Fits comfortably in VRAM ($< 1.5\text{ GB}$).
  - **Scaled 300M Tier:** Exceeds 4 GB VRAM for full AdamW optimizer state ($300\text{M} \times 16\text{ bytes} \approx 4.8\text{ GB}$). Requires gradient accumulation, CPU offloading, or cluster acceleration.

---

## 3. Feasibility Scope (Local vs Scaled Execution)

| Task | Local Hardware (RTX 3050 4GB) | Scaled / Cluster Hardware |
|---|---|---|
| **Architecture Verification** | ✅ **FEASIBLE** (Full speed) | ✅ Feasible |
| **Unit & Integration Testing** | ✅ **FEASIBLE** (640 tests pass in ~6 min) | ✅ Feasible |
| **Phase 8 Data Certification Pipeline** | ✅ **FEASIBLE** (CPU multi-processing) | ✅ Feasible |
| **Phase 3A Infrastructure Validation** | ✅ **FEASIBLE** (Nano config) | ✅ Feasible |
| **Small-Scale Pretraining & Baselines** | ✅ **FEASIBLE** (Nano config, $S=512$, $B=4$) | ✅ Feasible |
| **Ablation Sanity Checks** | ✅ **FEASIBLE** (Nano config, 7 variants) | ✅ Feasible |
| **Full 300M 10B+ Token Campaign** | ❌ **LOCAL LIMIT EXCEEDED** (Requires GPU scaling) | ✅ Required for production release |

---

## 4. Hardware Fallback & Safety Guarantees

1. **OOM Protection:** Trainer uses automatic fallback to `CPU` or reduced batch size if `torch.cuda.OutOfMemoryError` is encountered.
2. **Deterministic Seed Control:** Seeds `42`, `123`, `3407` strictly maintained across PyTorch CPU, PyTorch CUDA, NumPy, and Python standard random engines.
3. **No False Scaling Claims:** All local experiment logs will record GPU model (`RTX 3050 Laptop GPU`) and VRAM usage. Local runs are designated as **Pilot / Small-Scale Engineering Campaigns**.
