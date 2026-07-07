# IVERI CORE — Hypothesis Register

This register establishes the null ($H_0$) and alternative ($H_1$) hypotheses for the ten core architectural claims of the project.

---

## Hypothesis Register List

### H1: Entropy Routing Efficiency
- **$H_0$:** Entropy routing does not improve perplexity compared to vanilla hybrid baselines under matched parameter budgets.
- **$H_1$:** Entropy routing reduces perplexity with statistical significance ($p < 0.05$).

### H2: Titans Long-Context Retrieval
- **$H_0$:** Titans neural memory does not reduce long-context degradation over sequence lengths.
- **$H_1$:** Titans memory increases Needle-in-a-Haystack recall accuracy beyond 32k tokens.

### H3: Mixture of Recursion (MoR)
- **$H_0$:** MoR recursion depth does not reduce inference latency without affecting output quality.
- **$H_1$:** MoR recursion improves inference throughput while maintaining validation perplexity.

### H4: BLT Compression
- **$H_0$:** BLT patching does not increase throughput compared to standard byte models.
- **$H_1$:** BLT patching improves decoding speed (tokens/sec).

### H5: Convergence vs. Transformer
- **$H_0$:** IVERI does not converge faster than matched Transformer baselines.
- **$H_1$:** IVERI achieves lower training loss in fewer steps compared to matched Transformers.

### H6: Convergence vs. Mamba2
- **$H_0$:** IVERI does not converge faster than matched Mamba2 baselines.
- **$H_1$:** IVERI achieves lower loss than pure Mamba2 configurations.

### H7: Byte-Level Entropy
- **$H_0$:** IVERI does not achieve a lower Bits-Per-Byte (BPB) profile than baselines.
- **$H_1$:** IVERI achieves a statistically lower BPB.

### H8: SFT Coding Retention
- **$H_0$:** Coding specialization degrades general instruction-following capabilities.
- **$H_1$:** Coding specialized models maintain $\ge 95\%$ of their general instruction pass rates.

### H9: Preference Optimization Safety
- **$H_0$:** Preference alignment (DPO/SimPO) triggers mode collapse or severe quality drops.
- **$H_1$:** Preference optimization improves human alignment win rates without regressing SFT metrics.

### H10: Scale Predictability
- **$H_0$:** Model performance scaling does not follow a predictable power-law.
- **$H_1$:** validation loss vs parameter count matches a power-law curve ($R^2 \ge 0.95$).
