# IVERI CORE — Pre-Registered Hypothesis Registry

**Date:** 2026-07-25  
**Policy:** Pre-registered statistical hypotheses with strict acceptance/rejection criteria ($p < 0.05$, Cohen's $d > 0.5$).

---

## Hypothesis 1 (H1: Entropy Routing Efficiency)
- **Statement:** Byte-entropy dynamic routing significantly reduces bits-per-byte (BPB) loss compared to dense Transformer attention under equal FLOPs.
- **Acceptance Criteria:** Paired $t$-test $p < 0.05$, Cohen's $d > 0.50$, BPB reduction $\ge 5\%$.
- **Rejection Action:** Downgrade entropy routing weight to $0.0$.

## Hypothesis 2 (H2: Titans Memory Retention)
- **Statement:** Test-time neural memory updates improve passkey retrieval accuracy over 32,768-token contexts by $\ge 15\%$ over baseline Mamba2.
- **Acceptance Criteria:** Passkey accuracy $\ge 95.0\%$, $p < 0.01$.

## Hypothesis 3 (H3: Mixture of Recursion Scaling)
- **Statement:** MoR dynamic depth allocation reduces total parameter update FLOPs by $\ge 10\%$ while maintaining perplexity parity ($\pm 1.0\%$).
- **Acceptance Criteria:** FLOP reduction $\ge 10\%$, Perplexity delta $\le +0.05$.
