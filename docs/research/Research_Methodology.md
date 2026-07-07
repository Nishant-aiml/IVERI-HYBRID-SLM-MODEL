# IVERI CORE — Research Methodology

This document outlines the rigorous scientific methodology to compare IVERI CORE against parameter- and compute-matched control models.

---

## 1. Baseline Selection and Parity Bounds

Empirical comparisons of neural architectures require matched computational parameters:

1. **Parameter-Matched Parity:**
   Every model (Vanilla Transformer, Pure Mamba2, alternating Hybrid, and IVERI) must be instantiated with identical hidden dimension width, MLP/FFN expansion ratios, and layer stack counts. All comparison configurations must be checked and audited at startup.

2. **FLOP-Matched Parity:**
   Analytical FLOP calculations per token are used to match execution speed profiles. Models must operate within identical total training token budgets (same epochs and batch sizes).

---

## 2. Statistical Significance Testing

We reject the null hypothesis of equivalence only when significance bounds are satisfied:

- **paired Student's t-test:** Assesses whether differences in seed losses or perplexities are statistically significant. A two-tailed $p$-value $< 0.05$ is required for support.
- **Wilcoxon Signed-Rank Test:** A non-parametric audit verifying that ranking order of metric changes is consistent across runs.
- **Cohen's d Effect Size:** Quantifies the magnitude of improvement:
  $$d = \frac{\mu_{\text{model}} - \mu_{\text{baseline}}}{\sigma_{\text{pooled}}}$$
  An effect size $d > 0.8$ represents a strong improvement.
- **Bootstrap 95% Confidence Intervals:** Sampled with replacement over 1,000 runs to bound variance boundaries.
