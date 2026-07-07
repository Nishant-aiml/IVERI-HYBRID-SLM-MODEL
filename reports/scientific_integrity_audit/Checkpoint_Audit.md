# Checkpoint Integrity Audit

**Audit Date:** 2026-07-06  
**Repository:** IVERI CORE  
**Mode:** Read-only — no code modified

---

## Verdict: **FAIL**

Multiple checkpoint systems exist in parallel and are **not connected**. Training saves `.pt` files via `CheckpointSelector` (JSON index). Campaign registers only a SQLite DB archive as a checkpoint. Golden/paper promotion machinery exists but is not invoked by the campaign. Placeholder hashes appear in manifests and freeze documents.

---

## Checkpoint Subsystems

| Subsystem | File | Role | Connected to campaign? |
|-----------|------|------|------------------------|
| Training save/load | `training/checkpointing.py` | Atomic `.pt` save, arch version check | Partial (pretrain only) |
| Local selection | `training/model_selection.py` | `CheckpointSelector` JSON ranking | Pretrain path only |
| Baseline manager | `research/checkpoint_manager.py` | SHA-256 on save, registry JSON | No |
| Golden lifecycle | `research/golden.py` | Candidate→Golden→Paper promotion | **No** |
| Campaign lock | `research/campaign_lock.py` | Frozen checkpoint hashes | Manifest only |
| SQLite registry | `research/experiment_registry.py` | `checkpoints` table | DB archive only |

---

## Issue C1: Training checkpoints not registered in SQLite

| Field | Value |
|-------|-------|
| **Severity** | CRITICAL |
| **Impact** | Paper claims cannot trace to model weight files in DB |
| **Likelihood** | Certain |
| **Evidence** | `pretrain_runner.py` uses `CheckpointSelector` at `logs/checkpoint_history.json`; no `register_checkpoint()` for `.pt` files |
| **Recommended Fix** | After each `save_checkpoint()`, call `registry.register_checkpoint()` with file SHA-256 |
| **Estimated Effort** | 1–2 days |

---

## Issue C2: Campaign registers DB archive, not model weights

| Field | Value |
|-------|-------|
| **Severity** | CRITICAL |
| **Impact** | "Checkpoint" in registry may mean `experiments_PHASE6_3_FINAL.db`, not trained model |
| **Evidence** | `CampaignRunner._archive_database()` registers `archives/experiments_PHASE6_3_FINAL.db` as `db_archive_phase6_3` |
| **Recommendation** | Distinguish `checkpoint_type`: `MODEL_WEIGHTS` vs `DB_SNAPSHOT` |

---

## Issue C3: Golden checkpoint hash is placeholder

| Field | Value |
|-------|-------|
| **Severity** | CRITICAL |
| **Impact** | Freeze document and manifests reference non-existent golden hash |
| **Likelihood** | Certain |
| **Evidence** | `campaign_runner.py` line 174: `checkpoint_hashes = {"golden_base": "hash_golden_phase6_1"}` |
| **Recommendation** | Compute SHA-256 of actual golden `.pt` via `GoldenCheckpointManager` |

---

## Issue C4: GoldenCheckpointManager not invoked by campaign

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Impact** | Promotion policy (candidate→golden→paper) never executed in Phase 6.3 path |
| **Evidence** | `campaign_runner.run_campaign()` imports `GoldenCheckpointManager` but does not call `promote_checkpoint()` |
| **Recommendation** | Wire selection policy after pretrain stage completes |

---

## Issue C5: Golden comparison uses mock fallbacks

| Field | Value |
|-------|-------|
| **Severity** | HIGH |
| **Impact** | Golden regression tests can pass with hardcoded metrics |
| **Evidence** | `research/golden.py` `compare_to_golden()` injects hardcoded `ttft_sec`, `humaneval_pass_rate` when DB partial |
| **Recommendation** | Fail comparison when required metrics missing |

---

## Issue C6: training/checkpointing.py has no file hash

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Impact** | Integrity relies on `architecture_version` string match only |
| **Evidence** | `save_checkpoint()` / `load_checkpoint()` — no SHA-256 of file contents |
| **Recommendation** | Add `file_hash` to checkpoint dict; verify on load |

---

## Issue C7: Hash mismatch loads anyway

| Field | Value |
|-------|-------|
| **Severity** | MEDIUM |
| **Impact** | Corrupted or swapped checkpoints may load |
| **Evidence** | `research/checkpoint_manager.py` `load_checkpoint()` warns on hash mismatch but still loads state dict |
| **Recommendation** | Fail closed on hash mismatch for paper profile |

---

## Selection & Promotion Policy (Documented vs Implemented)

| Policy | Documented | Implemented |
|--------|------------|-------------|
| Best val_loss checkpoint | `CheckpointSelector` | Pretrain only |
| Golden promotion | `golden.py` | Not in campaign |
| Paper checkpoint lock | `campaign_lock.py` | Hash in manifest only |
| Reproducibility hash in certificate | Phase 6.3 freeze | Placeholder strings |

---

## Files Audited

- `training/checkpointing.py`
- `training/model_selection.py`
- `research/checkpoint_manager.py`
- `research/golden.py`
- `research/campaign_lock.py`
- `research/campaign_runner.py`

---

## Overall Assessment

Checkpoint integrity objective **not met**. Cannot prove that published benchmark numbers derive from a specific, hashed, reproducible model checkpoint.
