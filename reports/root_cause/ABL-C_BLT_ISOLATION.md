# ABL-C_BLT_ISOLATION.md
## ABL-C: IVERI Without BLT — Complete Report

---

> [!CAUTION]
> **The ABL-C result (2.0502 vs 3.1118) cannot be interpreted as "removing BLT
> improves IVERI quality."** The ablation conflates at least three simultaneous
> changes. See Section 3 before drawing any conclusion.

---

## 1. Motivation

FIX-003 (backbone entropy-gate bug fix) unblocked ABL-C — the first opportunity
to run IVERI without its BLT front-end. This was the highest-priority open
research question: **does the quality deficit vs the Transformer come from BLT, or
from the Mamba-2 backbone?**

---

## 2. Measured Results

| Model | Mean loss | Std | n | Params | Δ vs baseline | Welch t | p |
|---|---|---|---|---|---|---|---|
| Frozen baseline (EXP-020) | 3.1118 | 0.047 | 5 | 10.03M | 0 | — | — |
| IVERI + fixes | 3.1147 | 0.044 | 5 | 10.03M | +0.003 | −0.099 | 0.924 |
| **ABL-C (no BLT)** | **2.0502** | **0.194** | **5** | **8.96M** | **−1.062** | **11.875** | **0.000151** |
| Tuned Transformer (EXP-020) | 2.7051 | 0.176 | 5 | 9.72M | −0.407 | — | — |

**Individual ABL-C seeds**: 1.9282, 1.8377, 2.1706, 2.3209, 1.9936

The numerical result is striking: removing BLT correlates with a **−1.062 nat
improvement**, highly statistically significant (p=0.000151, t=11.875).

---

## 3. Why This Result Cannot Be Interpreted at Face Value

`use_blt=False` does not cleanly remove only the BLT front-end. It simultaneously
changes at least three things:

### 3.1 Parameter count changes — NOT a matched comparison

| Component removed | Parameters removed |
|---|---|
| `ByteEntropyModel` | ~350K |
| `DynamicPatcher` | ~90K |
| `BLTByteEncoder` (embed + attention + proj) | ~500K |
| `BLTByteDecoder` | ~130K |
| **Total removed** | **~1,073,063** |

ABL-C: 8,956,811 params vs Full IVERI: 10,029,874 params = **−10.7% parameter reduction**.

A model with 10.7% fewer parameters achieving better loss is not surprising — smaller
models overfit less on a small synthetic corpus over 10 training steps. The parameter
budget freed from BLT is not redistributed elsewhere; it simply disappears.

**Conclusion**: This is not a parameter-matched comparison. The result could be
partly or entirely explained by the reduced parameter count on this small corpus.

### 3.2 MoE and attention are permanently disabled in ABL-C

From `iveri_core.py:155-156`:
```python
patch_entropy = torch.zeros(batch_size, seq_len, 1, ...)   # always zero
```

From `backbone.py` (post FIX-003):
```python
is_complex = (entropy.mean() >= 0.20)   # → always False when entropy=0
if is_complex:
    x = x + self.attention(...)     # ← ALWAYS SKIPPED in ABL-C
if is_complex:
    ...moe_experts(...)             # ← ALWAYS SKIPPED in ABL-C
```

ABL-C is therefore not testing "Mamba backbone without BLT." It is testing:
> **Mamba backbone without BLT, without attention, and without MoE.**

All three components are disabled simultaneously by the zero-entropy path.

This means ABL-C tests a strictly simpler model than intended. The improvement
could come from removing BLT, from removing attention, or from removing MoE —
or from any combination of the three.

### 3.3 Different prediction task

With BLT:
- `BLTByteEncoder` compresses variable-length byte patches into patch embeddings
- Backbone processes **P_max patches** (fewer than SL=64 when patching is active)
- `BLTByteDecoder` upsamples patch representations back to byte-level logits

Without BLT (ABL-C):
- `self.byte_embed` produces one embedding per raw byte
- Backbone processes **SL=64 byte embeddings** (1:1 with input)
- `self.bypass_lm_head` projects directly to logit space

These are different prediction tasks:
- Full IVERI: predict next byte via compressed patch representation → decompress
- ABL-C: predict next byte directly from per-byte representation

The lower loss in ABL-C may simply reflect that the direct byte-level prediction
task is easier for this corpus at this training budget than the patch-encode-decode
route.

---

## 4. What the Result Does and Does Not Establish

### What it establishes
1. IVERI with the BLT pipeline achieves 3.11 loss at 10 steps on this corpus.
2. IVERI without BLT, without MoE, without attention, with 10.7% fewer parameters,
   on direct byte prediction, achieves 2.05 loss at the same condition.
3. ABL-C is faster: 2,879ms/step vs ~3,200ms/step (BLT backward also has cost).
4. The entropy-gate FIX-003 is working correctly (no crash, valid output).

### What it does NOT establish
1. That BLT alone causes the quality deficit.
2. That removing BLT from IVERI would improve quality in a parameter-matched setting.
3. That the 2.05 result is directly comparable to the Transformer's 2.705 result
   (those two models process input identically — raw token embeddings — unlike ABL-C).
4. That IVERI would beat the Transformer if BLT were removed.

---

## 5. Required Follow-Up Experiments

To isolate BLT's contribution, the following ablations are needed in order:

### EXP-ABL-C2: Fix the MoE/attention gating for ABL-C
Patch the entropy gate to run MoE and attention even when `entropy=0`:
```python
is_complex = True   # override — run all components regardless of entropy
```
Then re-run ABL-C. This tests "no BLT but same compute path as full IVERI."
A loss **still better** than 3.11 would more cleanly implicate BLT.
A loss **worse** than current ABL-C would confirm MoE/attention are helping.

### EXP-ABL-C3: Parameter-matched BLT ablation
Add ~1.07M parameters to the backbone (e.g., extra MLP depth or wider dimensions)
to compensate for BLT removal. Then compare fairly:
- Full IVERI (10.03M, with BLT)
- IVERI-no-BLT (10.03M, without BLT, backbone expanded)

### EXP-ABL-C4: BLT encoder bypass (keep decoder)
Use `byte_embed` for input (no entropy patching) but keep the BLT decoder.
Isolates encoder contribution from decoder contribution.

---

## 6. Preliminary Interpretation (Provisional)

**What the ABL-C result is consistent with:**
- BLT's encoding pipeline is making the prediction task harder at this scale/training budget
- The patch compression may lose information that matters for the 10-step regime
- Or: the direct byte-embedding baseline is simply easier to optimize at 10 steps

**What remains unknown:**
- Whether BLT helps with longer training (100+ steps, where compression benefits may appear)
- Whether BLT helps with longer sequences (where byte-level direct embedding becomes expensive)
- The isolated effect of BLT with all other components held constant

---

## 7. FIX-002 and FIX-003 Acceptance Decisions

### FIX-002 — ACCEPTED
| Check | Result |
|---|---|
| Numerical equivalence | PASS (no NaN/Inf) |
| Synchronization eliminated | 4 GPU-CPU syncs removed |
| Bimodality eliminated | Not confirmed — bimodal pattern absent in both pre and post-fix eval-mode measurements |
| Mean latency change | −2.1% eval forward, −1.1% training step — within noise |
| Quality (from full IVERI run) | 3.1147 ±0.044 vs 3.1118 ±0.047 (Δ=+0.003, p=0.924) PASS |

**Decision**: ACCEPTED. Removes 4 unnecessary GPU-CPU syncs. Execution depth is now
deterministic. No quality harm. Mean latency change is within noise, consistent with
the bimodal artifact being absent at this sequence length in eval mode. FIX-001's
lesson applies here too: the `.any()` syncs were never the dominant bottleneck.

### FIX-003 — ACCEPTED
| Check | Result |
|---|---|
| `use_blt=False` crash eliminated | PASS — runs without error |
| Logit range valid | [−1.302, 1.520] — within normal range |
| ABL-C unblocked | YES — experiment executed successfully |
| Quality (full IVERI) | 3.1147 ±0.044 vs 3.1118 ±0.047 — SIMILAR (p=0.924) |

**Decision**: ACCEPTED. Bug fixed; intended entropy-gate behavior restored; ABL-C unblocked.

---

## 8. Summary of Open Research Questions After ABL-C

| Question | Status |
|---|---|
| Does BLT cause the quality deficit? | **INCONCLUSIVE** — ABL-C confounds 3 changes; needs EXP-ABL-C2 and EXP-ABL-C3 |
| Does the Mamba backbone cause the deficit? | UNKNOWN — not yet isolated |
| Does Titans justify its algorithmic cost? | LIKELY NO at this scale, but needs longer training |
| What is IVERI's true quality after ALL bugs fixed? | **3.1147 ±0.044 — statistically identical to frozen baseline** |
| Does fixed IVERI beat the Transformer? | NO — Transformer still at 2.705, IVERI at 3.11 |
