# 00_BASELINE_FREEZE.md
## IVERI Core — Scientific Baseline Freeze
## This document is IMMUTABLE after creation. All modifications must be COMPARED AGAINST this.

---

## Git Identification

| Field | Value |
|---|---|
| **Git SHA** | `7012134f44403f6732333bba9e93bbc7ea73b867` |
| **Git tag** | `IVERI_BASELINE_FROZEN` |
| **Branch** | `main` |
| **Commit message** | Phase 10 + Forensic Audit: complete benchmark execution, evidence package |
| **Freeze timestamp** | 2026-08-14T07:05:00Z |

---

## Hardware Environment

| Field | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM | 4.29 GB |
| CUDA | 12.1 |
| OS | Windows-11-10.0.26200-SP0 |
| Python | 3.12.10 [MSC v.1943 64 bit] |
| PyTorch | 2.5.1+cu121 |

---

## Model Configuration (IVERI-Nano — Frozen)

| Parameter | Value | Source |
|---|---|---|
| hidden_dim | 268 | `configs/base_config.py::ModelConfig` |
| num_layers | 4 | " |
| num_heads | 4 | " |
| num_experts | 2 | " |
| titans_memory_dim | 128 | " |
| use_blt | True | " |
| use_titans | True | " |
| use_mor | True | " |
| use_moe | True | " |
| mamba_ratio | (default) | " |
| Vocabulary | 259 (bytes 0–255 + BOS/PAD/EOS) | `core/constants.py` |
| Total trainable params | **10,029,874** | `sum(p.numel())` verified EXP-019 |

---

## Frozen Experimental Results (Canonical — Do Not Modify)

### EXP-019 (Primary parameter-matched comparison, lr=3e-3 for BOTH)

| Model | Params | LR | Mean test loss | Std | Diverged |
|---|---|---|---|---|---|
| IVERI-Nano | 10,029,874 | 3e-3 | 3.061 | 0.057 | 0/5 |
| DenseTransformer Pre-LN | 9,722,083 | 3e-3 | 5.762 | 1.435 | 1/5 |

**Status: WITHDRAWN as comparative claim — Transformer LR was sub-optimal.**

### EXP-020 (LR-fair comparison, validation-selected LRs)

| Model | Params | LR | Val basis | Mean test loss | Std | Diverged |
|---|---|---|---|---|---|---|
| IVERI-Nano | 10,029,874 | 3e-3 | Best of [1e-4,3e-4,1e-3,3e-3] | **3.112** | 0.047 | 0/5 |
| DenseTransformer Pre-LN | 9,722,083 | **3e-4** | Best of [1e-4,3e-4,1e-3,3e-3] | **2.705** | 0.176 | 0/5 |

Welch t=4.994, p=0.005, Cohen's d=3.157. **Transformer wins by 0.407 nats.**

**IVERI LR 3e-3** was selected according to the validation-based hyperparameter-selection protocol.

### EXP-021 (Component timing, forward-pass only)

| Measurement | Value |
|---|---|
| IVERI forward pass | 524.53 ms ± 242.86 ms (CV=46%, bimodal) |
| Transformer forward | 23.61 ms ± 2.35 ms |
| Forward-only ratio | **22.2x** |
| Training-step ratio (EXP-019) | **~85x** |
| Seq-len scaling IVERI | ~4.8ms/byte (linear) |
| Seq-len scaling Transformer | ~flat at this range |
| IVERI VRAM | 336 MB |
| Transformer VRAM | 206 MB |

---

## Training Configuration (Frozen)

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| IVERI LR | 3e-3 (validation-selected) |
| Transformer LR | 3e-4 (validation-selected) |
| LR schedule | CosineAnnealingLR(T_max=steps) |
| Grad clip | 1.0 |
| Batch size | 4 |
| Sequence length | 64 |
| Steps | 10 |
| Mixed precision | torch.amp.autocast("cuda") |
| Seeds | [42, 43, 44, 45, 46] (test) |

---

## Corpus (Frozen)

| Field | Value |
|---|---|
| Corpus hash | `4ac89968d4a4a22f` |
| Total length | 10,120 bytes |
| Content | 40× repeat of 5 English sentences |
| Train bytes | 6,072 (60%) |
| Val bytes | 2,024 (20%) |
| Test bytes | 2,024 (20%) |
| Type | **Synthetic** — not a real text distribution |

---

## Open Scientific Questions (as of freeze)

1. Why does IVERI achieve higher loss than a fairly-tuned Transformer?
2. What fraction of the 22x–85x overhead comes from Titans?
3. What causes the bimodal latency (CV=46%)?
4. Which components contribute positively to quality?
5. Which components contribute negatively to quality or overhead?
6. Why did the 512-token passkey test fail?
7. Does IVERI's performance relative to Transformer change with longer training?
8. Is the quality deficit architectural, implementation-level, or scale-related?

---

## Rule

**All future experiments must report: (a) the frozen baseline result for comparison, (b) what changed from the baseline, (c) whether the change is a controlled single-variable intervention.**

Any change to the IVERI config, training procedure, or evaluation that is NOT recorded in FIX_REGISTRY.md invalidates the experiment.
