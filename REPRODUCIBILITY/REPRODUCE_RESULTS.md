# IVERI-HYPERION: Exact Replication & Reproducibility Guide

This document provides step-by-step instructions to replicate all empirical results, benchmarks, and architectural analyses reported in the manuscript:
**"IVERI-HYPERION: A Sub-Quadratic Hybrid Sequence Model with Associative Memory and Entropy Gating for Raw-Byte Language Modeling"**.

---

## 1. Environment & Setup

Ensure Python 3.10+ (tested on Python 3.12.10) with CUDA-enabled PyTorch 2.5+:

```bash
git clone https://github.com/Nishant-aiml/IVERI-HYBRID-SLM-MODEL.git
cd IVERI-HYBRID-SLM-MODEL
pip install -r requirements.txt
pip install -e .
```

Verify GPU availability and dependencies:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 2. Model Architecture

The canonical IVERI-HYPERION architecture is implemented in:
- `model/v1/iveri_v1.py`: Complete implementation including:
  - `VectorizedMambaLayer`: S4D continuous state-space recurrence with chunked parallel associative scan and Triton GPU kernel fallback.
  - `FastLinearTitansMemory`: Decayed associative neural matrix memory ($M_t \in \mathbb{R}^{64 \times 64}$) with Hebbian update and token-level $O(1)$ recurrent step.
  - `CausalEntropyPredictor`: 3-tap depthwise causal 1D convolution predicting information surprise.
  - `EntropyGatedSwiGLU`: Compute-modulated feed-forward network with token-level gating.
  - `IVERI_v1`: Complete 4-layer parameter-matched 9.985M parameter model ($d=406, N=16, d_{\text{mem}}=64, k=4$).

---

## 3. Replicating Empirical Benchmarks

### A. TABLE II: Multi-Seed 200-Step Pre-Training Benchmark (10M Parameters)
Evaluates 4 parameter-matched architectures (IVERI-HYPERION, Dense Transformer, Pure Mamba, Linear Attention) across random seeds 42, 43, 44 under tuned learning rates:
```bash
python SHOWCASE_PROFESSOR/benchmark_all_10m_architectures.py
```
- **Output:** Raw validation loss, perplexity, parameter counts, and throughput printed to console and logged to `reports/final_validation/four_way_10m_benchmark_results.json` and `FINAL_CONCLUDED_BENCHMARK.json`.

### B. TABLE III: Post-Training Component Intervention Matrix (Ablations)
Evaluates the sensitivity of the converged 1,000-step checkpoint to component zeroing probes:
```bash
python SHOWCASE_PROFESSOR/run_proof_campaign.py --proof 3
```
- Evaluates: baseline vs. w/o Titans memory (output projection clamped to 0), w/o causal entropy gating, w/o causal 1D convolution, and reduced memory rank.

### C. TABLE IV: Extended 1,000-Step Single-Seed Pre-Training Trajectory
Runs the continuous 1,000-step trajectory comparing IVERI-HYPERION against the Dense Transformer on raw-byte TinyStories:
```bash
python SHOWCASE_PROFESSOR/phase4_real_english_head_to_head_benchmark.py
```
- **Output:** Step-by-step convergence curves logged to `reports/capability/head_to_head_real_english_results.json`.

### D. TABLE V & Physical Recurrent Buffer Measurement:
Profiles autoregressive generation memory delta (MB) over 500 decoding steps:
```bash
python SHOWCASE_PROFESSOR/run_proof_campaign.py --proof 5
```
- Verifies constant $0.000000\text{ MB}$ net allocation growth during 500 generation steps.

### E. Passkey Needle Retrieval Limitation Experiment:
Evaluates synthetic passkey extraction at context length 512:
```bash
python SHOWCASE_PROFESSOR/benchmark_passkey_needle.py
```
- Reproduces the reported out-of-distribution language prior interference where zero-shot synthetic passkey retrieval fails.

---

## 4. Authenticated Raw Evaluation Logs

Pre-computed raw JSON benchmark logs are versioned in the repository:
1. `reports/final_validation/FINAL_CONCLUDED_BENCHMARK.json`: Canonical summary of the 3-seed 200-step benchmark.
2. `reports/final_validation/four_way_10m_benchmark_results.json`: Full 4-way comparison metrics.
3. `reports/proof_campaign/comprehensive_proof_results.json`: Complete 7-proof forensic test suite output.
4. `reports/capability/head_to_head_real_english_results.json`: 1,000-step head-to-head trajectory data.
