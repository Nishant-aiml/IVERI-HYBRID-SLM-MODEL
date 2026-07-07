# Architecture Consistency Report

**Audit Date:** 2026-07-06  
**Repository:** IVERI CORE  
**Mode:** Read-only — no code modified

---

## Verdict: **FAIL**

The frozen research specification (`IVERI_PROJECT_MASTER.md`, architecture docs) describes capabilities that the implementation does not fully realize. Engineering tests pass component-level gates; integrated behavior diverges from documented architecture on causality, Titans online memory, MoE entropy routing, and ablation isolation.

---

## Component Consistency Matrix

| Component | Spec (Master Doc) | Implementation | Consistent? |
|-----------|-------------------|----------------|-----------|
| BLT byte input | Raw bytes, entropy patches | Implemented | **Partial** — causality violations |
| BLT-D parallel decode | Multiple bytes per pass | Decoder outputs per-byte logits | **Partial** |
| Titans online memory | Weights update per token | `inject()` read-only in production | **No** |
| Mamba2 backbone | 92% compute, linear | Implemented in `backbone.py` | **Yes** |
| Flash Attention | 8% compute, causal | `is_causal=True` at patch level | **Partial** |
| MoR adaptive depth | Entropy-conditioned depth | `router.route(x, entropy)` | **Yes** |
| MoE 4 experts, top-2 | Standard routing | Implemented | **Yes** |
| MoE entropy routing | Patent 3 claim | Not implemented | **No** |
| 18 backbone blocks | At 300M scale | Default `num_layers=6` (nano) | **Yes** (scale config) |
| 300M parameters | v1.0 Mini target | Nano config ~36M measured | **Yes** (scale target) |

---

## Pipeline Consistency (Spec vs Code)

### Forward path — `model/iveri_core.py`

```
Spec:  bytes → entropy → patch → encode → Titans(update) → backbone×N → decode → logits

Code:  bytes → entropy → patch → encode → backbone(Titans inject/read) → decode → logits
                              ↑ non-causal entropy
                              ↑ within-patch bidirectional encoder
                                                          ↑ decoder cross-attn all patches
```

### Titans — spec vs production

| Aspect | Spec | `backbone.py` line 242 |
|--------|------|------------------------|
| Online update | Per-token weight write | `titans.inject(x, entropy)` |
| Memory persistence | Across sequence | `read()` parallel, no update |
| LR generator | Per-token dynamic LR | Not called in inject path |

### Entropy routing — spec vs production

| Router | Entropy input? |
|--------|----------------|
| MoR | Yes |
| Titans gate | Yes |
| MoE | **No** (`moe_router(norm_x)` only) |

---

## Causality vs Autoregressive Objective

The model is trained for next-byte prediction (`cross_entropy(logits, targets)`), but:

1. Entropy CNN sees future bytes (symmetric conv)
2. Encoder attends bidirectionally within patches
3. Decoder attends to all patches

**Impact:** Implementation is inconsistent with causal language modeling specification unless patches are strictly size-1 (defeating BLT purpose).

---

## Ablation vs Architecture Freeze

`AblationSuite` docstring claims component disable *"without changing frozen architecture source code"*. Result: ablations change config scalars only — architecture code paths unchanged. This is **inconsistent** with scientific ablation requirements in `IVERI_DATA_PIPELINE_COMPLETE.md` Stage 5.

---

## Parameter Budget (Master Doc vs Nano Config)

| Component | Master (300M) | Nano (`base_config.py`) | Notes |
|-----------|---------------|-------------------------|-------|
| num_layers | 18 | 6 | Documented scale difference |
| hidden_dim | 768 (mini) | 256 | Expected for nano |
| Titans | 15M | `titans_memory_dim=256` | Present but read-only |

Architecture **structure** matches; **behavior** does not for Titans and MoE entropy.

---

## Tests vs Specification

| Spec requirement | Test exists? | Passes? |
|------------------|--------------|---------|
| Forward/backward no NaN | `scripts/sanity_check.py`, unit tests | Yes |
| MoR depth diversity | `test_mor.py` | Yes |
| Titans gradient flow | `test_titans.py` | Yes (isolated `forward()`) |
| Titans in backbone online update | — | **No test** |
| End-to-end causality | — | **No test** |
| MoE entropy routing | — | **No test** |
| Ablation component removal | — | **No test** |

---

## Issue Register

### AC1: Titans specified as online memory; production uses static read

- **Severity:** CRITICAL
- **Files:** `model/titans/memory.py`, `model/backbone.py`
- **See:** `Titans_Verification.md`

### AC2: MoE entropy routing in patents/spec; not in code

- **Severity:** CRITICAL
- **Files:** `model/moe/router.py`, `model/backbone.py`
- **See:** `Entropy_Routing_Report.md`

### AC3: BLT stack non-causal for causal LM task

- **Severity:** CRITICAL
- **Files:** `model/blt/entropy_model.py`, `encoder.py`, `decoder.py`
- **See:** `Causality_Report.md`

### AC4: Class name IVERICore vs IVERIModel

- **Severity:** LOW (documentation)
- **Evidence:** Master doc uses `IVERICore`; code uses `IVERIModel`

---

## Architecture Docs vs Code — Partial PASS

`docs/architecture/overview.md` pipeline diagram matches module layout. ADR directory (`docs/decisions/`) is empty despite README describing ADR format.

---

## Recommended Actions (Await Approval)

| Priority | Action | Touches frozen arch? |
|----------|--------|----------------------|
| P0 | Add end-to-end causality test; document expected FAIL until fixed | Test only |
| P0 | Wire Titans `forward()` in training path | Yes |
| P1 | Add entropy input to MoE router OR revise H1/patent claims | Yes / docs |
| P1 | Causal encoder/decoder masks | Yes |
| P2 | Ablation flags in config with forward gating | Yes |

---

## Overall Assessment

Architecture **module inventory** is consistent with the frozen design. Architecture **behavioral semantics** are not. The implementation passes engineering validation but fails scientific consistency validation against the research specification.
