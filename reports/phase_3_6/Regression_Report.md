# IVERI CORE — Regression Report

This report checks the current candidate metrics against the active Golden Checkpoint values.

### Summary
- **Highest Severity Level:** FATAL

### Metrics Breakdowns
- **loss:** Golden: 1.5, New: 1.55, Change: 3.33%, Severity: WARNING
- **perplexity:** Golden: 4.5, New: 4.62, Change: 2.67%, Severity: WARNING
- **ttft_sec:** Golden: 0.12, New: 0.13, Change: 8.33%, Severity: INFO
- **decode_speed_tps:** Golden: 300.0, New: 290.0, Change: -3.33%, Severity: FATAL
- **vram_peak_mb:** Skipped (missing telemetry logs)
- **energy_per_token_j:** Skipped (missing telemetry logs)
- **calibration_ece:** Skipped (missing telemetry logs)
- **humaneval_pass_rate:** Skipped (missing telemetry logs)
- **instruction_score:** Skipped (missing telemetry logs)
