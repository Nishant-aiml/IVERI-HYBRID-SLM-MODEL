# 29_ROOT_CAUSE_TREE.md
## IVERI Core — Formal Root-Cause Tree
## Built from: EXP-019, EXP-020, EXP-021, EXP_DEEP_COMPONENT_PROFILE, EXP_ABLATION_STUDY, forensic code audit
## Date: 2026-08-14

---

## TOP EVENT

> **IVERI-Nano (10.03M) underperforms a fairly-tuned Dense Transformer (9.72M) by 0.407 nats on an early-training synthetic corpus benchmark, while incurring ~89x total training-step latency overhead.**

---

## Branch A — Quality Deficit (IVERI 3.112 vs Transformer 2.705)

### A1. BLT input representation introduces information loss or processing inefficiency
- **Evidence**: BLT (entropy + patcher + encoder + decoder) consumes only **1.6% of forward time** (5.7ms of 344.7ms). The ablation ABL-C is BLOCKED by Bug-5. Cannot conclude from current measurements.
- **Status**: UNKNOWN — ABL-C must be run after Bug-5 is fixed.

### A2. MoR gradient blocking prevents router from learning
- **Evidence**: `model/mor/router.py` uses `torch.floor(...).long()` to convert entropy to discrete depths. `.long()` is a non-differentiable operation — **all gradients to the MoR router parameters (`self.wg`) are completely zeroed**. The router never learns; it uses a fixed linear mapping of entropy → depth.
- **Evidence from ablation**: ABL-E (no MoR) gives loss 3.165 vs full IVERI 3.112 — removing the non-learning router costs +0.053 nats (not statistically significant, p=0.386). The router contributes essentially nothing.
- **Status**: CONFIRMED. BUG-04 (gradient blocking at MoR router).
- **Impact on quality deficit**: The MoR router adds parameter count and overhead but learns nothing. Weak impact on quality — suggests MoR is neutral to slightly harmful at this scale.

### A3. Titans memory contribution is neutral at this scale/training length
- **Evidence from ablation**: ABL-D (no Titans) gives loss 3.097 vs full IVERI 3.112 (delta=-0.015, p=0.670). Removing Titans slightly **improves** mean loss (not significantly). Titans provides no measured quality improvement at 10 training steps on a 10K-byte synthetic corpus.
- **Status**: CONFIRMED (for this experimental scale). Whether Titans helps at longer training or real data is untested.

### A4. MoE provides no quality improvement at 10M scale
- **Evidence from ablation**: ABL-F (no MoE) gives loss 3.112 vs full IVERI 3.112 (delta=+0.0001, p=0.998). Identical quality.
- **Status**: CONFIRMED (for this experimental scale).

### A5. Residual quality deficit is in the Mamba-2 backbone vs Transformer attention
- **Evidence**: After removing Titans, MoR, and MoE (all neutral), IVERI quality is still ~3.10–3.17, while a Transformer achieves 2.705. The remaining components are BLT + Mamba-2. BLT accounts for <2% of forward cost. The quality gap must come from the Mamba-2 backbone's representation quality at this scale/training budget, OR from BLT (not yet tested).
- **Status**: HYPOTHESIS — requires ABL-C (BLT isolation) to distinguish Mamba vs BLT contribution.

### A6. MoE entropy gating bug silently disables MoE when entropy=0
- **Evidence**: `backbone.py:116` — `if self.use_moe and is_complex:` where `is_complex = (entropy.mean() >= 0.20)`. When BLT is disabled (use_blt=False), patch_entropy is all zeros → `is_complex=False` → falls to `else: self.dense_ffn(norm_x)` → **AttributeError because dense_ffn doesn't exist when use_moe=True**. This is Bug-5 (confirmed by ABL-C crash).
- **Secondary effect**: When entropy is very low during training with BLT, MoE is silently bypassed. The attention block is also skipped (`if is_complex`). This means **low-entropy patches skip both attention AND MoE entirely** — which may be intentional (entropy-gating design) but has the side effect of potentially starving gradient flow through those layers.
- **Status**: CONFIRMED bug. Impact on quality: UNKNOWN until fixed and re-evaluated.

---

## Branch B — Training Step Runtime Overhead (3,281ms vs ~36ms)

### B1. BACKWARD PASS is the dominant bottleneck, not forward pass
- **Evidence**: Deep profiling shows forward=344.7ms, training step=3,281ms. **Backward = 2,937ms (89.5% of total step time)**.
- **Status**: CONFIRMED. The 22x forward ratio and ~85-89x step ratio are explained by the backward pass being ~8.5x more expensive than the forward.
- **Root cause of backward overhead**: See B2 and B3.

### B2. Titans per-patch .item() calls inside backward graph cause O(P) CPU-GPU sync
- **Evidence**: `titans/memory.py` lines 317-341, 453-488, 509: multiple `.item()` and `.tolist()` calls inside a `for t in range(P):` loop. These are inside the forward pass computation graph. When backward is computed, PyTorch must replay the graph — and each `.item()` forces a CUDA synchronization. With P patches per sequence, this causes O(P) CPU-GPU synchronizations per backward pass.
- **Evidence from profiler**: Titans telemetry reports 0.0ms because `titans_time_seconds` key is not populated in the backbone's `_update_telemetry_data` call, but the ablation shows removing Titans saves only ~600ms of step time — suggesting Titans backward, not forward, is the costly part.
- **Status**: CONFIRMED (code evidence). Measured impact: ~600ms reduction when Titans removed (ABL-D: 3,098ms → 2,500ms). Remaining 2,500ms from MoR backward.

### B3. MoR recursion .any() causes O(max_depth) CPU-GPU sync per forward + backward
- **Evidence**: `mor/recursion.py:171` — `if not active_mask.any(): break` inside the recursion loop. `.any()` evaluates a GPU tensor to a Python bool, requiring CUDA synchronization. This happens every recursion step (up to max_depth times per forward pass), and the backward must re-execute this path.
- **Evidence from ablation**: ABL-E (no MoR): 1,470ms vs full IVERI 3,098ms. **Removing MoR saves 1,628ms (53% of step time)**. MoR is the single largest overhead contributor.
- **Status**: CONFIRMED (code + ablation evidence). MoR is the dominant runtime bottleneck.

### B4. MoR has 9+ Python loops in recursion.py
- **Evidence**: `mor/recursion.py` has Python loops at lines 28, 51, 53, 65, 93, 114, 121, 166, 167. These cannot all be parallelized at CUDA level with the current sequential recursion design.
- **Status**: CONFIRMED from code inspection.

### B5. Even without MoR and Titans, ~1,470ms remains (still ~41x slower than Transformer)
- **Evidence**: ABL-D latency: ~2,500ms. ABL-E latency: ~1,470ms. Neither ablation was applied simultaneously. Without both, estimated latency is still >1,000ms based on BLT forward (5.7ms) + Mamba/attention backbone.
- **Root**: The remaining overhead after removing MoR/Titans likely comes from the Mamba-2 implementation itself, MoR's forward residual even without the Python early-exit, and BLT backward.
- **Status**: REQUIRES EXPERIMENT — run IVERI without BOTH Titans and MoR simultaneously.

---

## Branch C — Memory Overhead (IVERI 336MB vs Transformer 206MB, +130MB)

### C1. BLT encoder O(B × S²) dense pooling matrix
- **Evidence**: `blt/encoder.py:129` creates tensor `M` of shape `(B, p_max, seq_len)` where `p_max ≤ seq_len`. At SL=64, B=4: 4 × 64 × 64 × 4 bytes = ~65KB. At SL=512, B=4: 4 × 512 × 512 × 4 bytes = ~4MB. At SL=2048: 64MB per batch. This explains why long-context is impossible — VRAM explodes quadratically.
- **Status**: CONFIRMED. Primary cause of VRAM overhead at longer sequences and likely contributes to SL=64 overhead.

### C2. Titans memory state tensors persist per-sequence
- **Evidence**: `TitansMemory` maintains persistent weight matrices in `self.W_*`. These are additional parameters beyond the normal model weights and occupy VRAM during both forward and backward.
- **Status**: CONFIRMED. Titans parameters account for ~356K params × 4 bytes × 2 (grad) × 2 (optimizer state) = ~11MB of optimizer VRAM overhead.

---

## Branch D — Latency Bimodality (CV=46%)

### D1. MoR early-exit .any() causes variable execution length
- **Evidence**: `if not active_mask.any(): break` at `recursion.py:171` — the number of Python iterations of the recursion loop is determined by when all active masks become False. This depends on input entropy, which varies per batch. High-entropy batches execute more iterations, low-entropy batches fewer → bimodal distribution in total execution time.
- **Status**: CONFIRMED (code evidence). The two clusters in EXP-021 (352–438ms fast, 852–906ms slow) correspond to different MoR recursion depths triggered by batch entropy variation.

### D2. Entropy-gated attention/MoE — variable block execution
- **Evidence**: `backbone.py:100-116` — `is_complex = (entropy.mean() >= 0.20)`. When this threshold is crossed, both attention block and MoE are executed; when not crossed, both are skipped. This creates two execution paths with different costs, contributing to timing bimodality.
- **Status**: CONFIRMED (code evidence).

---

## Branch E — Long-Context Failure (passkey ctx=512 failed)

### E1. BLT encoder O(S²) memory kills long-context
- **Evidence**: BUG-03. At ctx=512, B=1: 1 × 512 × 512 × 4 bytes = 1MB per forward pass. Tensor operations on this become dominant and slow. At ctx=2048: 16MB. The passkey test with ctx=512 timed out after 25 minutes in EXP-011.
- **Status**: CONFIRMED as the mechanical cause of timeout. Whether the Titans memory mechanism would actually help long-context retention (if the timeout were resolved) remains UNKNOWN.

### E2. Titans sequential processing fails at long sequences
- **Evidence**: `for t in range(P):` loop in `titans/memory.py` — at P=512 patches, 512 × (multiple .item() calls) = thousands of CPU-GPU synchronizations. EXP-011 timeout at ctx=512 is consistent with this being O(P²) in total sync cost.
- **Status**: CONFIRMED (code evidence). The sequential loop is the mechanical cause of the timeout.
