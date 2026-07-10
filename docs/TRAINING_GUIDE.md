# IVERI CORE — Training Guide

A comprehensive guide to training IVERI CORE models at all scales.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Configuration System](#2-configuration-system)
3. [Data Pipeline](#3-data-pipeline)
4. [Training Modes](#4-training-modes)
5. [Phase Training Schedule](#5-phase-training-schedule)
6. [Monitoring & Metrics](#6-monitoring--metrics)
7. [Checkpointing & Recovery](#7-checkpointing--recovery)
8. [Optimization & Performance](#8-optimization--performance)
9. [Debugging & Diagnostics](#9-debugging--diagnostics)
10. [Scaling Guide](#10-scaling-guide)

---

## 1. Architecture Overview

IVERI CORE is a byte-entropy-native hybrid Small Language Model (SLM) combining:

| Component | Role |
|---|---|
| **BLT** | Byte-Level Tokenizer — dynamic entropy-based patching |
| **Titans** | Neural Memory Module — online weight updates, no context limit |
| **Mamba2** | Linear recurrence backbone — efficient long-range state |
| **MoR** | Mixture of Recursion — adaptive depth routing per token |
| **MoE** | Mixture of Experts — sparse activation for efficiency |

**Model Sizes:**

| Preset | Params | Hidden | Layers | Best For |
|---|---|---|---|---|
| nano | ~10M | 256 | 4 | CI / smoke tests |
| small | ~50M | 512 | 8 | Research iteration |
| medium | ~150M | 768 | 12 | Baseline training |
| large | ~400M | 1024 | 18 | Production (GPU) |

---

## 2. Configuration System

All training is controlled by `IVERIConfig` from `configs/base_config.py`.

### Using a preset:

```python
from configs.base_config import get_nano_config, get_small_config

config = get_nano_config()
config.training.max_steps = 10000
config.hardware.device = "cuda"
```

### Key training parameters:

```python
config.training.batch_size = 8         # samples per step
config.training.seq_len = 512          # context window bytes
config.training.learning_rate = 3e-4   # peak LR
config.training.warmup_steps = 200     # cosine warmup
config.training.grad_clip = 1.0        # gradient clipping norm
config.training.gradient_accumulation = 4  # effective batch = batch_size * accum
```

### Mixed precision (GPU only):

```python
config.hardware.mixed_precision = "fp16"  # "none" for CPU, "fp16" for GPU
```

### Logging:

```python
config.logging.mode = "wandb"          # "disabled", "local", "wandb"
config.logging.eval_every = 100        # validation frequency
config.logging.save_every = 500        # checkpoint frequency
```

---

## 3. Data Pipeline

### 3.1 Pre-training Data

IVERI CORE is pre-trained on byte sequences using `PretrainByteDataset`.

**Prepare TinyStories** (default corpus, ~500 MB):

```bash
python data/prepare_tinystories.py
# Saves to: data/tinystories/train_*.bin, val_*.bin
```

**Custom corpus**: Place raw `.bin` byte files in a directory and update
`config.data.train_path`.

### 3.2 SFT / Instruction Data

For Instruction Fine-Tuning (Phase 3.2), use `SFTByteDataset`:

```python
from training.sft_dataset import SFTByteDataset
dataset = SFTByteDataset(config, samples=my_samples)
```

Samples must follow the alpaca/chatml format — see `data/pipeline/sft_validator.py`
for schema validation rules.

### 3.3 Coding Specialization Data

For Phase 3.3 coding specialization, datasets are loaded via `CodingDatasetLoader`:

```python
from training.coding_dataset import CodingDatasetLoader
loader = CodingDatasetLoader(config)
train_ds = loader.load("the_stack_v2_deep", split="train")
```

---

## 4. Training Modes

### 4.1 Pre-training

```bash
python train.py --verification-level 3   # 1,000 steps
python train.py --verification-level 4   # 100,000 steps (full)
```

Or programmatically:

```python
from training.pretrain_runner import run_pretraining
results = run_pretraining(config, verification_level=3)
```

### 4.2 Instruction Fine-Tuning (SFT)

```bash
python -c "from training.sft_runner import run_sft; from configs.base_config import get_small_config; run_sft(get_small_config())"
```

### 4.3 Coding Specialization

```bash
python -c "
from training.coding_runner import run_coding
from configs.base_config import get_small_config
config = get_small_config()
config.coding.enabled = True
run_coding(config, verification_level=3)
"
```

### 4.4 Dry Run (architecture validation)

```bash
python train.py --dry-run
```

Validates config, model creation, and one forward pass without training.

---

## 5. Phase Training Schedule

| Phase | Description | Verification Level | Data |
|---|---|---|---|
| 1 — Pre-train | Byte-level language modelling | 3 (1k steps) → 4 (100k) | TinyStories |
| 2 — SFT | Instruction following | 3 (1k steps) | Dolly, OpenOrca |
| 3A — Coding | Code generation | 3 (1k steps) | Stack v2, LeetCode |

**Recommended pipeline:**

```bash
# 1. Prepare data
python data/prepare_tinystories.py

# 2. Pre-train pilot
python train.py --verification-level 3

# 3. Full pre-train (GPU recommended)
python train.py --verification-level 4

# 4. SFT
# python -m training.sft_runner (set sft_checkpoint in config)

# 5. Coding specialization
# python -m training.coding_runner (set sft_checkpoint in config)
```

---

## 6. Monitoring & Metrics

### Live Dashboard

```bash
python scripts/training_dashboard.py
```

Displays: step speed, ETA, loss curve, expert utilization, Titans update count.

### WandB Integration

Set `config.logging.mode = "wandb"` and ensure `wandb login` has been run.

Key metrics tracked:
- `train/loss`, `train/perplexity` — language modelling objective
- `val/loss`, `val/perplexity` — validation performance
- `moe/expert_utilization` — expert load balance
- `titans/update_count` — memory update frequency
- `blt/entropy_mean` — patch complexity distribution
- `grad/norm_mean` — gradient health
- `perf/tokens_per_sec` — throughput

---

## 7. Checkpointing & Recovery

### Automatic Saving

Checkpoints are saved every `config.logging.save_every` steps to:
```
logs/<run_name>/checkpoint_<step>.pt
```

### Resume Training

```bash
python train.py --resume logs/iveri_stage1_lvl3/checkpoint_1000.pt
```

Or programmatically:

```python
trainer.resume_from_checkpoint(checkpoint_path)
```

### Emergency Checkpoints

On `DivergenceError`, the trainer automatically saves:
```
logs/<run_name>/emergency_checkpoint_<step>.pt
```

### Verify Resume Fidelity

```bash
python scripts/verify_resume.py --checkpoint logs/.../checkpoint_N.pt
```

---

## 8. Optimization & Performance

### Gradient Accumulation

Simulate larger batches on limited VRAM:

```python
config.training.gradient_accumulation = 8  # effective_batch = 8 * batch_size
```

### Mixed Precision (GPU)

```python
config.hardware.mixed_precision = "fp16"  # ~2x speedup, ~2x VRAM savings
```

### Gradient Checkpointing

Reduces peak VRAM at cost of ~30% compute:

```python
config.training.use_gradient_checkpointing = True
```

### Profiling

```bash
python scripts/profile_training_step.py   # per-step timing breakdown
python scripts/profile_memory.py          # VRAM usage by component
```

---

## 9. Debugging & Diagnostics

### Instability Tracker

The trainer monitors layer-wise activation norms and raises `DivergenceError`
if instability is detected. Debug logs are saved to:
```
logs/debug_diagnostics.json
```

### Ablation Flags

Disable components for debugging:

```bash
python train.py --disable-titans --disable-moe
python train.py --disable-blt --use-baseline-transformer
```

### Architecture Health

```bash
python scripts/architecture_health.py
```

Verifies all modules receive gradients, expert utilization is balanced,
Titans memory updates are occurring, and MoR depth routing is non-uniform.

---

## 10. Scaling Guide

| Config | Params | GPU | Batch | Steps | Time |
|---|---|---|---|---|---|
| nano | 10M | CPU / T4 | 8 | 100k | ~4h CPU |
| small | 50M | T4 / A100 | 16 | 500k | ~12h A100 |
| medium | 150M | A100 80GB | 32 | 1M | ~2d A100 |
| large | 400M | A100 80GB ×4 | 64 | 2M | ~1w 4×A100 |

For detailed VRAM/FLOP estimates, see:

```bash
python scripts/count_params.py --config large
cat configs/parameter_breakdown.json
```

---

_IVERI CORE Training Guide — Apache-2.0 License._
