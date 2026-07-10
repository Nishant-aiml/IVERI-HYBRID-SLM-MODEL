# Phase 7.5 -- Debugging Report

## Summary
To prevent silent or unlocalized training crashes in Phase 7/Pilot campaigns, command line overrides and high-frequency numerical stability monitoring were added to the core pretraining pipeline.

---

## 1. CLI Component Ablation Flags
Command-line arguments were added to `train.py` to allow direct ablation testing of individual model components without code changes:
- `--disable-titans`: Sets `config.model.use_titans = False`.
- `--disable-blt`: Sets `config.model.use_blt = False`.
- `--disable-mor`: Sets `config.model.use_mor = False`.
- `--disable-moe`: Sets `config.model.use_moe = False`.
- `--disable-entropy`: Sets `config.model.use_entropy_routing = False`.

**Verification**: Runs with all features disabled completed successfully on micro workloads:
`python train.py --dry-run --disable-titans --disable-blt --disable-moe --disable-mor --disable-entropy`

---

## 2. Layer-wise Diagnostics
The new diagnostics tracker `training/instability_tracker.py` hooks into training execution step-by-step:
- **Hidden state norms**: Active forward hooks capture L2 norm of block output states.
- **Gradient norms**: Computes parameter-level gradient magnitude during optimizer passes.
- **File**: Dumped continuously to `logs/debug_diagnostics.json`.

---

## 3. Divergence Recovery & Early Crashes
A strict divergence protection threshold is set at `1e4`. If any hidden activation norm or backpropagated gradient norm exceeds `1e4` (or turns into a `NaN`/`Inf` value):
1. Warnings are logged.
2. A full dump of the current layer norms and gradient states is written to `logs/divergence_report.json`.
3. A `DivergenceError` is raised, causing training to terminate immediately before corrupting other saved checkpoint history.

---

## 4. Phase 7.x Regression Suite Execution
A post-modification test run of `scripts/regression_suite.py` was executed:
- **Imports**: Pass
- **Forward/Backward**: Pass
- **Gradient Flow**: Pass (53 active tensors)
- **Checkpoint save/load round-trip**: Pass (Bitwise identical, max diff: 0.00e+00)
- **Inference text generation**: Pass
- **Trainer single step**: Pass
- **Verdict**: ✅ **Diagnostics and ablation flags verified successfully**
