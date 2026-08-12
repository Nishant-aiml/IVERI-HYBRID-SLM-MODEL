# IVERI Core — Claim Validation Report
Generated: 2026-07-26T07:04:37Z

| Claim ID | Evidence Status | Statistical Status | IEEE Status | Patent Status | Key Blocker |
|----------|----------------|-------------------|-------------|---------------|-------------|
| CLAIM-001 | SUPPORTED | SINGLE_MEASUREMENT (no replication across seeds/co... | ENGINEERING_OBSERVATION — | PARTIALLY_ENABLED — demon | Single profiling run. Requires 5+ repetitions for CI. |
| CLAIM-002 | PARTIALLY_SUPPORTED | INCONCLUSIVE (n=3 too small to distinguish)... | NOT_READY — n=3 insuffici | PARTIALLY_ENABLED — throu | n=3 underpowered |
| CLAIM-003 | SUPPORTED | DETERMINISTIC (all 6 warmup candidates identical —... | PARTIALLY_READY — warmup  | PARTIALLY_ENABLED — suppo | Single-seed warmup sweep (not 3 seeds per warmup value) |
| CLAIM-004 | OBSERVATION_NOT_CONCLUSION | Loss difference highly significant (Welch p<0.001,... | NOT_READY — reviewers wil | NOT_ENABLED for superiori | Compute-controlled comparison required (same FLOPs or same wall-clock time) |
| CLAIM-005 | PARTIALLY_SUPPORTED (at seq_len=64 only) | NOT_SIGNIFICANT (negligible effect)... | NOT_READY — Titans hypoth | NOT_ENABLED — Titans pate | CRITICAL: Long-context sweep (seq_len 64->8192) not executed |
| CLAIM-006 | MISSING_EVIDENCE | NOT_TESTED... | NOT_READY | NOT_ENABLED | Titans context sweep not executed |

## Detailed Claim Analysis

### CLAIM-001: RQ7-PROFILING

**Measurement**: Disabling gradient checkpointing for seq_len=64 increases throughput by 31.6% (154 -> 203 tok/s). Backward pass accounts for 70% of wall-clock.

**Correct Scientific Framing**: Disabling gradient checkpointing for seq_len=64 increases throughput by 31.6% (154 -> 203 tok/s). Backward pass accounts for 70% of wall-clock.

**Evidence Status**: SUPPORTED
**Statistical Status**: SINGLE_MEASUREMENT (no replication across seeds/configs — engineering observation, not statistical claim)
**Confidence**: HIGH for seq_len=64. LOW for generalization to other seq_len.
**IEEE Status**: ENGINEERING_OBSERVATION — valid for methods section, not a primary result
**Patent Status**: PARTIALLY_ENABLED — demonstrates VRAM-throughput tradeoff control

**Blockers**: Single profiling run. Requires 5+ repetitions for CI.

---

### CLAIM-002: RQ3-MoR

**Measurement**: At seq_len=64/50 steps: Removing MoR gives 108% throughput gain and 49% VRAM reduction. Loss difference: No-MoR=3.1365 vs Full=3.1425 (delta=-0.0060, Welch p=0.514, Cohen's d=0.737 [medium]).

**Correct Scientific Framing**: Under this specific experimental condition (50 steps, seq_len=64, n=3), removing MoR improves throughput substantially without statistically detectable loss change. Whether MoR provides benefit under long-context or long-training conditions REMAINS UNTESTED.

**Evidence Status**: PARTIALLY_SUPPORTED
**Statistical Status**: INCONCLUSIVE (n=3 too small to distinguish)
**Confidence**: LOW — n=3 seeds, 50 steps, seq_len=64 only. Cannot generalize to long-context.
**IEEE Status**: NOT_READY — n=3 insufficient. Requires n>=8, longer training, longer context.
**Patent Status**: PARTIALLY_ENABLED — throughput evidence exists. Need long-context benefit evidence to fully enable adaptive MoR claim.

**Blockers**: n=3 underpowered | seq_len=64 only | 50 steps only | Need seq_len>=512 experiments

---

### CLAIM-003: RQ6-STABILITY

**Measurement**: Gradient explosions are NOT caused by LR schedule: all 6 warmup candidates (0-200 steps) produce identical explosion counts. MoR removal reduces total explosions from 5 to 1 across 3 seeds.

**Correct Scientific Framing**: The gradient explosion frequency is invariant to LR warmup schedule across 6 candidates. MoR removal reduces explosions, but the attribution to MoR specifically requires controlled experiments removing only MoR's recursive gradient path.

**Evidence Status**: SUPPORTED
**Statistical Status**: DETERMINISTIC (all 6 warmup candidates identical — no statistical test needed for invariance). MoR contribution is suggestive (n=3) but not proven.
**Confidence**: HIGH for warmup-invariance finding. MEDIUM for MoR attribution.
**IEEE Status**: PARTIALLY_READY — warmup invariance is clean. MoR attribution needs stronger experiment design.
**Patent Status**: PARTIALLY_ENABLED — supports claim of architectural stability through gradient control.

**Blockers**: Single-seed warmup sweep (not 3 seeds per warmup value) | MoR attribution not isolated

---

### CLAIM-004: RQ5-BASELINE

**Measurement**: Dense Baseline achieves loss 1.3664 vs IVERI Full 3.1425 at 50 steps. Dense runs 75x faster (13016 vs 173 tok/s).

**Correct Scientific Framing**: At the same NUMBER OF STEPS (50), Dense achieves lower loss by 1.7761. At the same WALL-CLOCK TIME, Dense would complete ~75x more steps, making step-controlled comparison misleading. FLOP-controlled or wall-clock-controlled comparison is required for any architectural conclusion.

**Evidence Status**: OBSERVATION_NOT_CONCLUSION
**Statistical Status**: Loss difference highly significant (Welch p<0.001, d=150.0445) BUT comparison is CONFOUNDED by compute budget disparity (75x throughput difference).
**Confidence**: HIGH for measurement. LOW for interpretation as architectural inferiority.
**IEEE Status**: NOT_READY — reviewers will immediately flag uncontrolled compute comparison.
**Patent Status**: NOT_ENABLED for superiority claims. Dense baseline results must be framed carefully.

**Blockers**: Compute-controlled comparison required (same FLOPs or same wall-clock time)

---

### CLAIM-005: RQ2-TITANS

**Measurement**: At seq_len=64: Titans adds 17% VRAM overhead with 57% throughput cost. Loss difference: -0.0006 (p=0.926, d=0.0989).

**Correct Scientific Framing**: Titans adds measurable VRAM and latency overhead at seq_len=64 with no observed loss benefit. Whether Titans provides benefit at longer sequences (the hypothesis) REMAINS COMPLETELY UNTESTED.

**Evidence Status**: PARTIALLY_SUPPORTED (at seq_len=64 only)
**Statistical Status**: NOT_SIGNIFICANT (negligible effect)
**Confidence**: MEDIUM — VRAM/throughput observation is clear. Loss difference significance is inconclusive at n=3.
**IEEE Status**: NOT_READY — Titans hypothesis requires long-context experiments.
**Patent Status**: NOT_ENABLED — Titans patent requires demonstration of long-context memory advantage.

**Blockers**: CRITICAL: Long-context sweep (seq_len 64->8192) not executed | n=3 underpowered

---

### CLAIM-006: RQ1-LONG-CONTEXT

**Measurement**: IVERI Full outperforms Dense Transformer at long-context tasks (seq_len >= 1024).

**Correct Scientific Framing**: This is an architectural hypothesis. No experiment has been executed to test it.

**Evidence Status**: MISSING_EVIDENCE
**Statistical Status**: NOT_TESTED
**Confidence**: ZERO — no experiment executed
**IEEE Status**: NOT_READY
**Patent Status**: NOT_ENABLED

**Blockers**: Titans context sweep not executed | Long training run not executed

---
