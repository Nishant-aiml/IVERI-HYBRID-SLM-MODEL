# Publication Integrity Report (Phase 6.3.2 OBJ5)

**Generated:** 2026-07-09T03:05:18Z  
**Protocol:** Phase-6.3.2-OBJ5  

## Executive Verdict

**Fail-closed publication framework:** `PASS`

- Mock metrics helper removed from campaign: `True`
- Campaign marks FAILED on training error: `True`

## Gate Probes

| Gate | Passed | Detail |
|------|:------:|--------|
| non_measured_blocked | True | Publication blocked: non-measured experiment provenance present (1). |
| failed_runs_blocked | True | Publication blocked: failed runs exist (1). |
| failure_records_blocked | True | Publication blocked: 1 failure record(s) in registry. |
| measured_pipeline_passes | True | exp=exp_good_c2c33503 |

## Proof: Fail-Closed Publication

1. PublicationManager._assert_integrity_for_publication requires COMPLETED + MEASURED rows.
2. Non-MEASURED metrics and benchmark_runs raise RuntimeError before report generation.
3. failures table rows block publication and certificate signing.
4. CampaignRunner no longer defines _log_mock_metrics synthetic loss fallback.
5. Failed training attempts set experiment status FAILED instead of COMPLETED.

## Raw JSON

```json
{
  "protocol_version": "Phase-6.3.2-OBJ5",
  "timestamp_utc": "2026-07-09T03:05:18Z",
  "production_verdict": "PASS",
  "mock_metrics_path_removed": true,
  "campaign_marks_failed_on_training_error": true,
  "gates": [
    {
      "gate_name": "non_measured_blocked",
      "passed": true,
      "detail": "Publication blocked: non-measured experiment provenance present (1)."
    },
    {
      "gate_name": "failed_runs_blocked",
      "passed": true,
      "detail": "Publication blocked: failed runs exist (1)."
    },
    {
      "gate_name": "failure_records_blocked",
      "passed": true,
      "detail": "Publication blocked: 1 failure record(s) in registry."
    },
    {
      "gate_name": "measured_pipeline_passes",
      "passed": true,
      "detail": "exp=exp_good_c2c33503"
    }
  ],
  "presence_proof": [
    "PublicationManager._assert_integrity_for_publication requires COMPLETED + MEASURED rows.",
    "Non-MEASURED metrics and benchmark_runs raise RuntimeError before report generation.",
    "failures table rows block publication and certificate signing.",
    "CampaignRunner no longer defines _log_mock_metrics synthetic loss fallback.",
    "Failed training attempts set experiment status FAILED instead of COMPLETED."
  ]
}
```
