# SciOS Research Debt Registry
*Updated: 2026-07-26T07:23:07Z*

| Debt ID | Problem Description | Priority | Affected Papers | Affected Patents | Required Experiment | Est. GPU Hours |
|---|---|---|---|---|---|---|
| DEBT-001 | Titans memory evaluation missing for long sequences (seq_len >= 512). | **CRITICAL** | Paper 2 | Patent 2 | `stage_c_titans_context_sweep.py` | 8.0h |
| DEBT-002 | Router entropy logs not saved during training. | **HIGH** | Paper 3 | Patent 1 | `stage_c_router_instrumentation.py` | 2.0h |
| DEBT-003 | Step-controlled baseline comparison is confounded by 75x throughput mismatch. | **CRITICAL** | Paper 1, Paper 5 | Patent 4 | `stage_c_wallclock_comparison.py` | 4.0h |
