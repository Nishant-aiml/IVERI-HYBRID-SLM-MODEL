# IVERI Core — IEEE Readiness Report
Generated: 2026-07-26T07:04:37Z

| Paper Target | Score | Status | Fatal Blockers |
|-------------|-------|--------|----------------|
| IEEE TPAMI (journal) | 2/10 | NOT_READY | n=3, no long-context, confounded comparison, no convergence evidence |
| IEEE TNNLS (journal) | 3/10 | NOT_READY | n=3, no long-context, short training only |
| IEEE Transactions (short paper) | 4/10 | NOT_READY | n=3, compute-budget confounding |
| IEEE Conference (CVPR/ICLR workshop) | 5/10 | BORDERLINE | n=3 acceptable with caveats, need stronger framing |
| EMNLP/ACL Findings | 4/10 | NOT_READY | No downstream NLP task evaluation |

## What Would Make This IEEE-Ready

1. **n >= 8 seeds** per model (currently n=3)
2. **Compute-controlled baseline** (FLOP-matched or wall-clock-matched)
3. **Long-context experiment** (seq_len=512–8192, MISSING-002)
4. **Long training run** (5000+ steps, MISSING-004)
5. **Downstream task evaluation** (perplexity on PTB/WikiText, or accuracy on classification)
6. **Multiple profiling repetitions** (MISSING-007)
7. **Parameter count documentation** (MISSING-008)

## Earliest Possible IEEE Conference Submission

If MISSING-002, MISSING-004, MISSING-006 are executed: approximately **4-6 weeks**.
If only MISSING-005 is executed: **Patent-3 (MoR) could reach IEEE workshop level**.