# IVERI CORE — Quick Start Guide

> Get IVERI CORE running on your machine in under 10 minutes.

---

## Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.11+ | 3.12+ |
| RAM | 8 GB | 16 GB |
| Storage | 2 GB | 10 GB |
| GPU | CPU (slow) | CUDA 11.8+ |

---

## 1. Clone & Install

```bash
git clone https://github.com/your-org/iveri-core
cd iveri-core
pip install -e ".[dev]"
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

---

## 2. Prepare Training Data

Download and tokenise the TinyStories dataset (≈500 MB):

```bash
python data/prepare_tinystories.py
```

Expected output:
```
Downloading TinyStories...
Saving shard 0 -> data/tinystories/train_0.bin  (47 MB)
...
Dataset ready: 8 shards, 476 MB total
```

---

## 3. Verify the Installation

Run a quick architecture check (no GPU needed):

```bash
python scripts/regression_suite.py
```

Expected: `ALL CHECKS PASSED` in under 60 seconds.

---

## 4. Train a 10M Nano Model (CPU)

```bash
python train.py --verification-level 1
```

| Level | Steps | Time (CPU) | Use Case |
|---|---|---|---|
| 1 | 20 | ~30s | Quick smoke test |
| 2 | 100 | ~3 min | Dev iteration |
| 3 | 1,000 | ~20 min | Pilot run |
| 4 | 100,000 | ~hours | Full pre-train |

---

## 5. Generate Text

```bash
python scripts/generate.py \
    --checkpoint logs/iveri_stage1_lvl3/checkpoint_1000.pt \
    --prompt "Once upon a time" \
    --max-new-bytes 128 \
    --temperature 0.8
```

---

## 6. Resume Training

If training was interrupted, resume from the latest checkpoint:

```bash
python train.py --resume logs/iveri_stage1_lvl3/checkpoint_1000.pt
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: No module named 'torch'` | torch not installed | `pip install torch` |
| `CUDA out of memory` | Batch too large | `--batch-size 2` or use CPU |
| `AssertionError: warmup_steps` | Config mismatch | Delete `config_snapshot.json` |
| WandB prompt | Not logged in | `wandb login` or `WANDB_MODE=disabled` |

---

## Next Steps

- Read the full [Training Guide](docs/TRAINING_GUIDE.md)
- Review the [Architecture Overview](docs/architecture/overview.md)
- Run the [Phase 7.9 E2E Validation](scripts/e2e_validation.py)

---

_IVERI CORE — Byte-entropy-native hybrid SLM. Apache-2.0 License._
