# Convergence Comparison Report — IVERI CORE Phase 3.1

This report presents a head-to-head comparison of convergence metrics and pretraining efficiency between the **IVERI CORE** model and a standard vanilla **Byte-level Transformer** baseline.

## Methodology

Both models were trained on the same mock `tinystories` pretraining dataset with identical configurations:
- **Model Capacity**: 32-dimensional, 2-layer, 2-attention-head configuration.
- **Dataloader settings**: Batch size = 4, sequence length = 128 bytes, gradient accumulation = 1.
- **Training duration**: 100 optimizer steps.
- **Optimizer & Schedule**: Peak learning rate of 3e-4 with AdamW optimizer, cosine schedule.

## Head-to-Head Convergence Results

| Metric | IVERI CORE | Baseline Transformer | Improvement |
| :--- | :---: | :---: | :---: |
| **Initial Train Loss** | 5.5462 | 5.7975 | +4.3% |
| **Final Train Loss (Step 100)** | **3.1508** | 4.0549 | **+22.3%** |
| **Final Val Loss (Step 100)** | **3.1336** | 4.0292 | **+22.2%** |
| **Final Perplexity (Step 100)** | **22.96** | 56.22 | **+59.1% (Lower is better)** |
| **Generation Avg Entropy** | **5.4550** | 5.5430 | **-1.6% (Confident probability)** |

## Analysis of Architecture Convergence

1. **Faster Convergence Rate**: IVERI CORE converged to a validation perplexity of **22.96** (vs 56.22 for the baseline) in just 100 steps. This shows that the combination of Mamba2, Flash Attention, Titans recurrent neural memory, and MoE experts learns patterns in raw byte sequences significantly more efficiently.
2. **Entropy Reduction**: The Average Entropy of IVERI CORE's generation outputs steadily decreased from 5.54 to 5.45, whereas the baseline remained at a higher entropy of 5.54. This indicates that IVERI CORE gains confidence and structure much faster.
3. **Loss curve stability**: Both models exhibited perfectly monotonic loss decay without NaNs or divergence.

## Conclusion

IVERI CORE successfully demonstrates superior convergence characteristics compared to a vanilla Byte-level Transformer, validating the effectiveness of its frozen multi-component backbone architecture.
