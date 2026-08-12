# SciOS Reviewer Attack Matrix
*Updated: 2026-07-26T07:23:07Z*

| ID | Question | Experiment | Status | Confidence | Paper Section |
|---|---|---|---|---|---|
| REV-001 | Step-controlled comparison vs Dense baseline is confounded because Dense runs 75x faster per step. | EXP-004 (Stage C Compute Accounting) | **OPEN** | HIGH (Problem identified, compute-matched run scheduled) | Paper 1 Section V: Baselines & Compute Accounting |
| REV-002 | n=3 seeds is statistically underpowered for Welch t-tests and Cohen's d. | EXP-005 (n=8 Extended Seeds Run) | **IN_PROGRESS** | HIGH (stage_c_extended_runs.py actively executing 8 seeds) | Paper 1 Section IV: Statistical Methodology |
| REV-003 | 50 training steps is in transient phase, not reflecting long-run convergence. | EXP-006 (500-Step Extended Training) | **IN_PROGRESS** | HIGH (stage_c_extended_runs.py actively executing 500 steps) | Paper 1 Section IV-C: Convergence Analysis |
| REV-004 | Single profiling measurement is non-reproducible due to GPU thermal/kernel startup variance. | EXP-007 (10x Profiling Repetitions) | **CLOSED** | VERIFIED (10 reps completed: with_ckpt=2852ms+-480ms, wo_ckpt=2012ms+-61ms, overhead=1.417x) | Paper 1 Section IV-B: Profiling & Hardware Latency |
| REV-005 | Titans memory long-context advantage is completely untested (seq_len=64 only). | EXP-008 (Titans Context Length Sweep 64->8192) | **OPEN** | CRITICAL_MISSING (Required for Paper 2 & Patent 2) | Paper 2 Section III: Context Scaling |
