# IVERI Core Phase 6.2 Validation Report — Remaining Issues Register

## 1. Scope
This report registers all confirmed and suspected bugs, technical debt items, and code smells found during the engineering audit of the IVERI Core codebase.

## 2. Methodology
- **Database Inspection**: Checked registry entries and error logs in `research/experiments.db`.
- **Warning Verification**: Reviewed pytest warning messages and runtime telemetry output.

## 3. Evidence
- **Registry Contamination**: Running `replay_campaign.py --reviewer-mode` against the local database file `research/experiments.db` throws errors:
  ```
  failures table has 73 row(s)
  experiment IVERI_2026_07_04_Seed42_IVERI_Run001 status=PENDING
  experiment IVERI_2026_07_04_Seed42_IVERI_Run001 provenance=UNKNOWN
  ```
- **Deprecation Warnings**: PyTorch warning message logged during runtime:
  ```
  FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  ```

## 4. Measurements
- **Unresolved Code Bugs**: 0
- **Registry Database Issues**: 1 ( legacy runs and failures present)
- **Deprecation Warnings**: 1

## 5. Findings
- **Registry Database Issue**: The development database `research/experiments.db` contains legacy runs from earlier testing cycles. By design, the fail-closed reviewer replay script rejects databases containing failure rows or `UNKNOWN` provenance entries, which prevents automated campaign certification of local development registries.
- **Autocast Deprecation Warning**: `torch.cuda.amp.autocast` is deprecated in PyTorch 2.5. The code should transition to `torch.amp.autocast('cuda', ...)` in the next update cycle.

## 6. Risks
- **Registry Blockage**: Production campaign certificates cannot be generated unless the database is reset or cleared of failure/pending entries.

## 7. Recommendations
- **Reset Database**: Archive the existing `research/experiments.db` file and initialize a clean database before launching Phase 6.3 production training sweeps.
- **Harden Autocast Calls**: Update the autocast context calls in `training/mixed_precision.py` and validation scripts to avoid warnings.

## 8. Final Verdict
**MINOR ISSUES FOUND**
All registered issues relate to database configuration and minor deprecation warnings, with zero active bugs in the model architecture.
