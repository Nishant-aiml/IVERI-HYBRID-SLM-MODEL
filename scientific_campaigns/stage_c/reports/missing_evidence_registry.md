# IVERI Core — Missing Evidence Registry
Generated: 2026-07-26T07:04:37Z

> This registry documents ALL evidence that does not yet exist.
> No paper claim, patent claim, or research conclusion may reference these items
> until the corresponding experiment has been executed and validated.

| ID | Priority | Status | Blocks | Experiment | Expected Output |
|----|----------|--------|--------|------------|-----------------|
| MISSING-001 | COMPLETED | DONE | CLAIM-003 | Phase-7 warmup sweep | phase_7/lr_warmup_sweep_results.json |
| MISSING-002 | CRITICAL | NOT_STARTED | CLAIM-005,CLAIM-006,Paper-2,Patent-2 | stage_c_titans_context_sweep.py | phase_4/titans_context_sweep.json |
| MISSING-003 | HIGH | NOT_STARTED | Paper-1,Patent-1 | stage_c_router_instrumentation.py | router_analysis/router_entropy_log.json |
| MISSING-004 | CRITICAL | NOT_STARTED | CLAIM-004,All papers | stage_c_long_run.py (5000 steps) | long_training/loss_curves.json |
| MISSING-005 | CRITICAL | NOT_STARTED | Patent-3 all claims | Implement Adaptive MoR v0.92 | phase_3/adaptive_mor_results.json |
| MISSING-006 | HIGH | NOT_STARTED | REV-002 | Re-run with n>=8 seeds | all phase_5_6 files expanded |
| MISSING-007 | HIGH | NOT_STARTED | REV-004 | 10x profiling repetitions | profiling/latency_repeated.json |
| MISSING-008 | MODERATE | NOT_STARTED | REV-007 | Log parameter counts + FLOPs | architecture/model_specs.json |

## Most Critical Missing Experiment

**MISSING-002: Titans Context Length Sweep** is the single most important missing experiment.
IVERI's entire architectural proposition rests on long-context performance. Without this,
the project cannot demonstrate its core architectural hypothesis.

**Command to execute** (once sweep script is implemented):
```bash
python scratch/stage_c_titans_context_sweep.py
```

**Expected content**: Loss, throughput, VRAM for seq_len=[64,128,256,512,1024,2048] with
Titans enabled vs disabled, with 3+ seeds each.