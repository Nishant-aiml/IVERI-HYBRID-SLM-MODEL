# Research Hypothesis: Entropy-Guided Dynamic Expert Routing

**Primary Hypothesis ($H_1$):** Local byte-sequence Shannon entropy $\mathcal{H}(b_t)$ correlates strongly with the required neural capacity needed to predict upcoming bytes. Low-entropy sequences (predictable repetitions, whitespace, boilerplate code) can be processed by a sparse subset of expert layers ($k=1$), whereas high-entropy boundary transitions require full expert capacity ($k=4$).

**Null Hypothesis ($H_0$):** Dynamic entropy-conditioned expert routing yields no statistically significant reduction in FLOPs or step latency compared to static top-k routing at equivalent perplexity.
