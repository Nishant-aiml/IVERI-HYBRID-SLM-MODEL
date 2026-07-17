# Phase 9 — Independent Verification Report (v1.0)

**Auditor role:** Independent Principal Engineer / Scientific Auditor  
**Date:** 2026-07-11  
**Method:** Ground-up inspection. No prior reports trusted as truth.  
**Environment measured:** Windows 11, Python 3.14.4, PyTorch 2.12.1+cpu, CUDA unavailable

---

## Executive Verdict

**Phase 9 is NOT complete.**

A single orchestrator scaffold exists (`scripts/run_phase9_campaign.py`) with 3 trivial unit tests. The full campaign **fails at Phase 9.2** because certified datasets do not exist. Subsystems 9.5–9.11 are missing or contain hardcoded mock outputs.

| Subsystem | Exists in Code | Executes Successfully | Produces Real Evidence | Verdict |
|-----------|---------------|----------------------|------------------------|---------|
| 9.1 Environment Lock | Yes | Yes (partial) | Yes (measured) | **PARTIAL** |
| 9.2 Dataset Verification | Yes | **No** | No | **NOT IMPLEMENTED** |
| 9.3 Model Initialization | Yes | Yes | Yes (measured) | **PASS** |
| 9.4 Sanity Training | Yes | Not reached in campaign | No campaign evidence | **NOT VERIFIED** |
| 9.5 Pilot Training | No | No | No | **NOT IMPLEMENTED** |
| 9.6 Live Architecture Validation | Partial | Fails MoE balance check | Partial telemetry | **NOT IMPLEMENTED** |
| 9.7 Stability & Recovery | Partial | Warnings only; no injection | No | **NOT IMPLEMENTED** |
| 9.8 Checkpoint Resume | Yes | Yes (isolated test) | Yes (diff=0.0) | **PASS** (not in campaign) |
| 9.9 Evaluation | Partial | Greedy only in campaign | No stored benchmark outputs | **NOT IMPLEMENTED** |
| 9.10 Extended Training | Stub | No | No | **NOT IMPLEMENTED** |
| 9.11 Certification | Stub | Writes mock files | **Fake** | **NOT IMPLEMENTED** |

---

## Repository Inspection Summary

### What exists and works (pre-Phase-9 infrastructure)

- **Model (`model/iveri_core.py`)**: 9,347,282 parameters at nano config (~10M target). All subsystems emit telemetry on forward pass.
- **Training (`training/trainer.py`, `training/checkpointing.py`)**: Checkpoint round-trip verified — max logit diff **0.0** on CPU.
- **Phase 8 pipeline modules**: certification, freeze, fingerprint, curriculum, registry — present and partially tested.
- **Inference sampling**: `inference/sampling.py` supports temperature, top-k, top-p.
- **Failure analyzer**: `training/failure_analyzer.py` classifies OOM/NaN/network errors — **not wired into Phase 9 campaign**.

### Critical gaps

1. **`data/certified/` directory does not exist** — no frozen training data for recipe `iveri_v1`.
2. **`reports/phase_9/` did not exist** before this audit (only partial `reports/phase_9_audit/` from failed run).
3. **Phase 9 campaign writes mock release artifacts** when it reaches 9.11:
   - `{"fingerprint": "mock"}`
   - `mock-sha256-hash`
   - `{"val_loss": 2.45}` (hardcoded)
   - `{"score": 98.5}` (hardcoded)
4. **Milestone training is simulated** — comment in source: `# Simulate milestones pretraining`.
5. **`pilot_passed` flag is never set** — no `run_pilot_training()` method exists.
6. **Only 3 Phase 9 tests** — environment lock, model init (with prerequisites faked), stability warnings.

### Bugs found during audit

| ID | Severity | Description | Status |
|----|----------|-------------|--------|
| P9-BUG-001 | High | Eval dataset YAML specs (`arc`, `gsm8k`, etc.) have `quality_threshold: null` causing `float(None)` crash in registry | **FIXED** in `data/pipeline/registry.py` |
| P9-BUG-002 | Critical | No certified datasets — Phase 9.2 blocks entire campaign | **OPEN** — requires Phase 8 pipeline execution |
| P9-BUG-003 | Critical | Phase 9.11 release bundle uses mock/hardcoded values | **OPEN** |
| P9-BUG-004 | High | `architecture_health.py` fails: MoE max expert load 69.1% > 60% threshold | **OPEN** |
| P9-BUG-005 | Medium | Default git commit fallback `mock-git-sha-7a6c98` if git subprocess fails | **OPEN** |
| P9-BUG-006 | Medium | `test_phase8_integration.py` fails: download returns 0 docs | **OPEN** |
| P9-BUG-007 | Low | `datetime.utcnow()` deprecated in Phase 9 script | **OPEN** |

---

## Measured Evidence

### Phase 9.1 — Environment Lock (EXECUTED)

```
python scripts/run_phase9_campaign.py --artifacts-dir reports/phase_9_audit
```

Result: 9.1 PASS, 9.2 FAIL.

Measured `campaign_manifest.json`:
- Python: 3.14.4
- PyTorch: 2.12.1+cpu
- CUDA: false
- Git commit: 4d1e203173c112ff3730c07fa3d0586b4e50ca0a (real, not mock on this machine)

### Phase 9.2 — Dataset Verification (FAILED)

```
ERROR: Certified file not found for dataset 'tinystories' at data\certified\tinystories\train.jsonl
```

Registry after fix: **11/11** specs discovered (was 6/11 before fix).

### Phase 9.3 — Model Initialization (EXECUTED in isolation)

```
Total params: 9,347,282
BLT entropy: 0.999996
MoR depth: 3.0
MoE hist: [677, 859]
Titans reads: 128, writes: 128
```

All subsystem telemetry keys present.

### Phase 9.8 — Checkpoint Resume (EXECUTED in isolation)

```
Checkpoint max logit diff: 0.0
Resume OK: True
```

### Test Suite (EXECUTED)

```
634 passed, 1 failed, 4 skipped, 11 deselected (592.6s)
FAILED: tests/test_phase8_integration.py::test_full_pipeline_integration
Phase 9 tests: 3/3 passed (trivial coverage only)
```

### Architecture Health Script (EXECUTED)

```
scripts/architecture_health.py → FAILED
AssertionError: Expert routing imbalance: max expert has 69.1% load
```

Subsystems ran (loss decreased 5.83 → 5.36) but MoE balance gate failed.

---

## Documentation Mismatch

`implementation_plan.md` defines:
- Phase 8 = Scale architecture (35M–500M)
- Phase 9 = Scientific benchmarking and research

The user's Phase 9 specification (pretraining campaign freeze audit) **does not match** the implementation plan. The codebase reflects a **custom Phase 9 pretraining campaign** concept in `run_phase9_campaign.py`, but it is incomplete.

---

## Acceptance Gate Results

| Gate | Result |
|------|--------|
| Environment lock passes | PARTIAL (no CUDA/GPU on audit machine) |
| Dataset verification passes | **FAIL** |
| Certified dataset used | **FAIL** — none exist |
| Model initializes correctly | **PASS** |
| BLT/Titans/MoE/MoR/Mamba2 active | **PASS** (telemetry) |
| Sanity training succeeds | NOT REACHED |
| Pilot training succeeds | **NOT IMPLEMENTED** |
| Architecture Health from real telemetry | **FAIL** (script fails MoE gate) |
| Stability recovery works | **NOT VERIFIED** |
| Checkpoint save/load works | **PASS** (isolated) |
| Canonical release bundle complete | **FAIL** (mock placeholders) |
| Phase 9 tests pass | **PASS** (3 trivial tests only) |
| Regression tests pass | **634/635** (1 failure) |

---

## Recommendation

**Do not freeze Phase 9. Do not approve for Phase 10.**

The repository has a solid Phase 7 architecture and Phase 8 pipeline scaffolding, but Phase 9 as specified is a **partial scaffold with fake certification outputs**, not a completed pretraining campaign.
