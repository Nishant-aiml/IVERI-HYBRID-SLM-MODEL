# IVERI CORE — Phase 1.5 Validation & Verification Report

## Mixture of Recursions (MoR) Validation

---

## 1. Unit & Integration Testing Results
All test suites run and pass successfully on CPU:

*   `test_router_entropy_mapping` — Validates Option C prediction entropy mapping:
    *   $E_p = 0.0 \to D_p = 1$ (Index 0)
    *   $E_p = 0.5 \to D_p = 4$ (Index 3)
    *   $E_p = 1.0 \to D_p = 8$ (Index 7)
*   `test_router_learned_mode` — Confirms learned gating (ablation research mode) correctly maps inputs to valid logits and top-1 argmax indices.
*   `test_recursion_engine_execution` — Verifies that `RecursionEngine` performs correct looping. Evaluates output elements match active counts (e.g. element with depth 4 is executed exactly 4 times).
*   `test_mor_gradflow` — Confirms gradient backpropagation through the recursion engine to the wrapped block parameters.
*   `test_selective_kv_cache` — Validates that keys/values are updated/appended along the sequence dimension selectively based on the boolean active mask.
*   `test_router_validation_checks` — Confirms shape mismatches and missing entropy inputs are rejected with descriptive errors (`ShapeError` / `ValueError`).

---

## 2. Telemetry and Statistics Verification
`RecursionEngine` telemetry was validated:
- **Average Depth Calculation:** For sample depths of `[1, 2, 4, 8, 8]`, average depth was computed exactly as `4.6`.
- **Max Depth Frequency:** Exactly `40.0%` of elements reached the maximum recursion depth.
- **Skipped Computation %:** Computations skipped was computed as `(17 / 40) * 100.0 = 42.5%`, proving significant FLOPS reduction under adaptive recursion.

---

## 3. Numerical Stability & Robustness
The system was stress-tested under edge-case configurations:
*   **NaN/Inf Robustness:** Verified that NaN inputs compile without infinite loops and propagate safely.
*   **Clamping Safety:** Negative entropy (e.g., `-0.5`) and overflow entropy (e.g., `1.5`, `99.0`) clamp safely to valid depth levels without raising errors.
*   **Extreme Dimensions:** Tested under batch size 1 and sequence length 1.
*   **Seed Determinism:** Confirmed that seed-driven initialization yields reproducible output weights and indices.

---

## 4. Regression Status
All test suites from previous phases:
*   Phase 0 (Infrastructure)
*   Phase 1.1 (Norms, RoPE, SwiGLU)
*   Phase 1.2 (Mixture of Experts)
*   Phase 1.3 (Mamba2)
*   Phase 1.4 (Flash Attention Wrapper)

remained completely unaffected, preserving the green build status.
