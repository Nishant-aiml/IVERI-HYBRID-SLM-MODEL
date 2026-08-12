# Scientific Anomaly Investigation: Loss Discontinuity at Step 1894

**Investigation Date:** 2026-07-26  
**Investigated Artifact:** `reports/live_training/stage_b_10m_loss.csv`  
**Status:** **ROOT CAUSE IDENTIFIED & DOCUMENTED**

---

## 🔍 Observed Anomaly
During Stage B execution (`EXP-0005`), cross-entropy loss exhibited the following trajectory:

| Step Range | Loss Range | Behavior |
|:---:|:---:|:---|
| Step 1 – 1,500 | `5.5649 → 0.0049` | Monotonic decay down to near-zero loss |
| **Step 1,894 (Resume)** | **`17.1404`** | **Abrupt discontinuity jump (+17.1355 nats)** |
| Step 1,894 – 2,050 | `17.1404 → 0.7661` | Rapid re-convergence down to sub-1.0 loss |

---

## 🎯 Root Cause Analysis

### 1. Static Tensor Instantiation Outside the Loop
In `scratch/run_stage_b_10m.py` and `scratch/resume_stage_b_10m.py`, synthetic input and target tensors were initialized **once outside the step loop**:

```python
# INITIAL IMPLEMENTATION (BUG):
x = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), device=device, dtype=torch.long)
y = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), device=device, dtype=torch.long)

for step in range(start_step, total_steps + 1):
    out = model(x)  # <--- Same static tensor evaluated every step!
    loss = F.cross_entropy(out, y)
```

### 2. Overfitting Mechanism
- Because `x` and `y` remained fixed across steps 1–1,500, the 9.35M parameter model memorized the exact static 4,096-token sequence, driving cross-entropy loss down to **`0.0049`**.
- When the background process was restarted at Step 1,894 following a server restart, `resume_stage_b_final.py` re-seeded PyTorch, generating a **new, unseen random tensor**.
- The model evaluated on this new tensor produced an initial loss of **`17.1404`**, before rapidly memorizing the new tensor down to `0.7661` at Step 2,050.

---

## 🛡️ Scientific & Engineering Takeaways

### 1. Hardware Evidence Remains 100% Valid
- **Throughput (1,232.3 tok/s), VRAM (1,722.6 MB), FP16 GradScaler stability, and Zero-OOM guarantees** are completely independent of tensor randomness and remain **100% empirically verified**.

### 2. Required Fix for Stage C Pretraining
- For language modeling convergence evidence, every batch must contain **freshly sampled tokens** from the streaming dataset iterator or dynamic per-step `torch.randint`:

```python
# CORRECTED IMPLEMENTATION FOR STACK/REAL DATASET:
for step in range(start_step, total_steps + 1):
    x, y = get_next_batch(dataset_stream)  # Dynamic fresh tokens per step!
    out = model(x)
    loss = F.cross_entropy(out, y)
```

---

## 📋 Governance Action
1. Documented in `negative_results/NEG_003_STATIC_SYNTHETIC_OVERFITTING.md`.
2. Flagged `EXP-0005` in `experiment_registry.json` as `HARDWARE_BENCHMARK_ONLY` (throughput/VRAM validated; convergence trajectory subject to dynamic dataset re-run in Stage C).
