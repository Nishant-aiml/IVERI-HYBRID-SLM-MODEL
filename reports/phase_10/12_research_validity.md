# 12 Research Validity & Threats Analysis

This report categorizes all potential validity threats according to standard scientific methodology:

## 1. Internal Validity
- **Seed Variance**: Evaluated across 3 seeds (`42`, `123`, `3407`) to control for initialization noise.
- **Gradient Stability**: Monitored via `training_stability.csv` to ensure no gradient explosions or NaNs distorted convergence.

## 2. External Validity
- **Domain Scope**: Evaluated on byte-level natural language (Wikipedia, FineWeb-Edu), code (The Stack), and synthetic narrative (TinyStories).
- **Architecture Transfer**: Findings apply to byte-level autoregressive models; BPE tokenized variants require separate calibration.

## 3. Construct Validity
- **Compute Match Fidelity**: Active parameters matched within 9.28% (≤ 10.0% gate) to prevent FLOP disparity artifacts.
- **Metric Quality**: BPB and Perplexity calculated directly from raw byte cross-entropy loss.

## 4. Conclusion Validity
- **Statistical Power**: Paired t-tests, Wilcoxon signed-rank tests, and Cohen's d effect size calculations with Holm-Bonferroni correction.
