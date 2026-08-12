# IVERI Core — Reviewer Objection Report (Adversarial Mode)
Generated: 2026-07-26T07:04:37Z

> **Purpose**: Simulate peer review rejection. Every objection below is a
> scientifically valid criticism. The goal is to resolve these BEFORE submission.

Total objections: 8 | Cannot rebut: 4 | Partial: 4

## REV-001 [FATAL] — IEEE TPAMI

**Objection**: The compute comparison is unfair. Dense Transformer runs 75x faster. At the same wall-clock time, Dense completes ~3,750 steps vs IVERI's 50 steps. The loss comparison at 50 steps is meaningless — of course Dense converges faster per step when it's 75x cheaper per step.

**Rebuttal Status**: CANNOT_REBUT
**Current Evidence**: Cannot rebut — this objection is CORRECT. The current experimental design does not control for compute budget.
**Resolution Required**: Execute compute-controlled comparison: train both models for T seconds of wall-clock time (e.g., 600 seconds). Dense will take ~45,000 steps; IVERI ~400 steps. Compare final loss.
**Blocking**: CLAIM-004, Paper-4, Any journal submission

## REV-002 [MAJOR] — IEEE TNNLS

**Objection**: n=3 seeds is statistically underpowered. For a medium effect size (d=0.5), two-sample t-test at n=3 has approximately 8% power. You cannot claim statistical significance or insignificance from 3 seeds.

**Rebuttal Status**: PARTIAL
**Current Evidence**: ACKNOWLEDGED in our analysis. All claims are marked preliminary pending larger-N experiments.
**Resolution Required**: Increase to n>=8 seeds per model for reliable inference. For n=8, power reaches ~50% for medium effects.
**Blocking**: CLAIM-002, CLAIM-005, Any conference submission

## REV-003 [MAJOR] — JMLR

**Objection**: 50 training steps is not enough to assess convergence. Language models typically show very different loss curves at steps 50 vs 5000. All loss comparisons are in the transient phase.

**Rebuttal Status**: CANNOT_REBUT
**Current Evidence**: Cannot rebut — acknowledged. Experiments are labeled 'preliminary short-run'.
**Resolution Required**: Long training run: 5000 steps at seq_len=64 and 512. This is MISSING-004.
**Blocking**: CLAIM-002, CLAIM-004, All papers

## REV-004 [MAJOR] — ACM ASPLOS

**Objection**: The profiling baseline is a single measurement. Hardware variability, thermal throttling, CUDA kernel startup effects, and memory fragmentation can cause 10-30% variance in GPU timing on laptop GPUs. One profiling run cannot establish a reliable throughput claim.

**Rebuttal Status**: CANNOT_REBUT
**Current Evidence**: Cannot rebut fully. Profiling data is from a single measurement.
**Resolution Required**: Repeat profiling 10 times per configuration, report mean and std. Warm up GPU first.
**Blocking**: CLAIM-001 as paper claim (OK as engineering note)

## REV-005 [MAJOR] — IEEE TPAMI

**Objection**: Where is the long-context evaluation? IVERI claims to excel at long sequences via Titans and MoR. All experiments use seq_len=64 — approximately 5-10 tokens of actual content. This is the opposite of the architectural use case.

**Rebuttal Status**: CANNOT_REBUT
**Current Evidence**: Cannot rebut — acknowledged. Long-context experiments are not yet executed (MISSING-002).
**Resolution Required**: Execute Titans context sweep: seq_len = [64, 128, 256, 512, 1024, 2048] with Titans vs No-Titans. This is the central experiment.
**Blocking**: CLAIM-005, CLAIM-006, Paper-2, Patent-2, All primary claims

## REV-006 [MODERATE] — JMLR

**Objection**: The gradient explosion count is a noisy metric. 1 vs 5 explosions across 3 seeds could easily be random initialization variance, not a structural difference. Is this metric reproducible?

**Rebuttal Status**: PARTIAL
**Current Evidence**: Partially supported: warmup sweep shows explosions are NOT warmup-dependent (all 6 candidates identical). MoR attribution is suggestive but not confirmed.
**Resolution Required**: Run 10 seeds per model to establish reliable explosion rate. Test specifically: does removing only MoR reduce explosions while keeping BLT and Titans?
**Blocking**: CLAIM-003 (partial resolution only)

## REV-007 [MODERATE] — IEEE TNNLS

**Objection**: The parameter count comparison is not documented. Are IVERI and Dense Transformer models truly parameter-matched? The VRAM difference (828MB vs 394MB) is 2.1x, which suggests the models are not comparable.

**Rebuttal Status**: PARTIAL
**Current Evidence**: VRAM difference is documented but parameter count is not explicitly logged in experimental artifacts. VRAM includes activation memory and optimizer states which vary by architecture.
**Resolution Required**: Log and report: total_parameters, trainable_parameters, non-embedding_parameters, FLOP_count for all models.
**Blocking**: CLAIM-004 (must establish fair comparison)

## REV-008 [MODERATE] — USPTO Examiner

**Objection**: Patent claims for MoR adaptive depth scheduling lack experimental demonstration of the adaptive mechanism. Current experiments only test static depth=0 (No_MoR) and static depth=full (IVERI_Full). The patented dynamic depth control is not tested.

**Rebuttal Status**: PARTIAL
**Current Evidence**: CORRECT — Adaptive MoR v0.92 has not been implemented or tested. Patent claim 1 is NOT YET ENABLED.
**Resolution Required**: Implement Adaptive MoR in iveri/model/ and run benchmarks comparing fixed-depth vs adaptive-depth. This is MISSING-005.
**Blocking**: Patent-3 all claims
