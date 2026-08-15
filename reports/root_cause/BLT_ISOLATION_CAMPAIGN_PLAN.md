# EXP-ABL-C2_C3_C4_PLAN.md
## BLT Isolation Campaign — Experiment Plan

---

## Context

ABL-C (task-6126) produced a dramatic result: −1.062 nats (p=0.000151) when BLT
was removed. The result is confounded by three simultaneous changes:

| Confounder | ABL-C change | Required isolation experiment |
|---|---|---|
| Parameter count | 10.03M → 8.96M (−10.7%) | **ABL-C3** — param-matched no-BLT |
| Attention/MoE gating | Active → Disabled (entropy=0 always) | **ABL-C2** — force_complex=True |
| Prediction task | Patch encode-decode → Direct byte prediction | Inherent to no-BLT; cannot be fully controlled |

---

## EXP-ABL-C2 (running — task-6180)

**Question**: Was the ABL-C improvement caused by disabling attention + MoE?

**Setup**:
- `use_blt=False`, `force_complex=True` (entropy=1.0 passed to backbone)
- All backbone components active: Mamba, attention, MoR, MoE, Titans
- Same 8,956,811 parameters as ABL-C
- Same LR (3e-3), same 5 seeds, same 10 steps

**What the result tells us**:

| ABL-C2 result | Interpretation |
|---|---|
| ≈ 2.05 (ABL-C) | Gating NOT the source; BLT or parameters explain the gap |
| ≈ 3.11 (full IVERI) | Disabling gating explains the improvement — adding back attention/MoE recovers full cost |
| Between 2.05 and 3.11 | Partial contribution from gating |

**Commit**: `de2954c` (force_complex kwarg addition, no architecture change)

---

## EXP-ABL-C3 (running — task-6180)

**Question**: Was the ABL-C improvement caused by having 10.7% fewer parameters?

**Setup**:
- `use_blt=False`, zero entropy (same gating as ABL-C)
- `hidden_dim` chosen by grid search to match ~10,029,874 parameters
- Parameters allocated proportionally via hidden_dim increase (not arbitrary padding)
- LR selected by 3-candidate validation sweep [1e-3, 3e-3, 5e-3]
- Same 5 seeds, same 10 steps

**What the result tells us**:

| ABL-C3 result | Interpretation |
|---|---|
| ≈ 2.05 (ABL-C) | Parameter count NOT masking improvement; no-BLT advantage is real at any param count |
| >> ABL-C (e.g. 2.8) | Smaller model benefits from reduced overfitting on 10-step corpus |
| ≈ 3.11 (full IVERI) | At matched params, no-BLT advantage disappears — parameter budget drives result |

---

## EXP-ABL-C4 (planned — after C2/C3 results reviewed)

**Question**: Does BLT improve or hurt quality when the comparison is parameter-matched and
computation-matched?

**Setup**:
Two models, same parameter budget (~10M), same backbone:
- **BLT arm**: full IVERI (10.03M) as already measured
- **no-BLT arm**: ABL-C3 result

This is the cleanest possible comparison: same parameter count, same backbone family,
same training protocol. The only difference is whether BLT patching is applied.

**Decision criteria**:

| Result | Conclusion |
|---|---|
| BLT arm better | BLT provides a quality benefit when the model has sufficient capacity |
| no-BLT arm better | BLT imposes too much reconstruction difficulty at this scale |
| Not significantly different | BLT is neutral at 10M parameters |

**Prerequisite**: ABL-C3 result and its LR must be confirmed before designing C4.

> [!NOTE]
> C4 uses the already-measured full IVERI as the BLT arm. No new BLT experiment
> is needed. The comparison is: full IVERI (EXP-020 baseline) vs ABL-C3.
> This means C4 depends only on having ABL-C3 results — it is essentially free.

---

## Decision Tree After C2/C3

```
ABL-C: 2.05 (confounded)
    │
    ├── ABL-C2 ≈ 2.05 AND ABL-C3 ≈ 2.05
    │       → Gating and params are NOT the cause
    │       → Conclusion: BLT is the primary contributor to quality deficit
    │       → Next: investigate WHY (C4 + longer training with BLT)
    │
    ├── ABL-C2 >> 2.05 (e.g. 2.8–3.0)
    │       → Disabling attention/MoE was a major factor
    │       → ABL-C result was partly from disabling components
    │       → Next: check if full attention/MoE is worth the cost
    │
    ├── ABL-C3 >> 2.05 (e.g. 2.5–3.0)
    │       → Parameter count was a major factor
    │       → Smaller model overfits less on 10-step corpus
    │       → ABL-C result was partly a parameter-count artifact
    │       → Next: longer training needed to separate overfitting from architecture
    │
    └── All configurations ≈ 3.11 (no improvement vs full IVERI)
            → Something else in the BLT path provides quality (impossible given ABL-C data)
```

---

## Research Tree Status

```
IVERI
  │
  ▼
Fair Transformer comparison (EXP-020)
  │
  ▼
Transformer wins at 10M (3.11 vs 2.71)
  │
  ▼
Root-cause analysis
  │
  ├── Implementation defects
  │     FIX-001 (Titans .item) — ACCEPTED, timing hypothesis falsified
  │     FIX-002 (MoR .any()) — ACCEPTED, bimodal absent at SL=64
  │     FIX-003 (entropy gate crash) — ACCEPTED, ABL-C unblocked
  │     BUG-04 (router gradient) — REJECTED FALSE POSITIVE
  │
  └── Architectural investigation
        ABL-C (no BLT) — 2.05, confounded (3 simultaneous changes)
          │
          ├── ABL-C2 (force attn/MoE) — RUNNING
          └── ABL-C3 (param-matched) — RUNNING
                    │
                    ▼
              ABL-C4 (BLT vs no-BLT, param-matched) — PLANNED
                    │
                    ▼
          Is BLT actually helping at 10M?
```

---

## Fixed Elements (Do Not Change)

- Frozen baseline: `7012134f` (`IVERI_BASELINE_FROZEN`)
- Corpus: same 400-byte synthetic text × 40 repeats
- Training: AdamW lr=3e-3, CosineAnnealingLR, 10 steps, SL=64, BS=4
- Evaluation: same 5 seeds [42,43,44,45,46], test split on fixed corpus
- All prior experiment results: immutable

---

*Created: 2026-08-14*
