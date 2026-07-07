# Gradient Health Report — IVERI CORE Phase 3.1

This report documents the layer-by-layer gradient health monitoring and tracking stats.

## Summary Statistics

- **Global Gradient Norm**: stabilized around 1.0 (clipping active)
- **Max Gradient Value**: 0.2755 (step 18)
- **Min Gradient Value**: -0.2683 (step 18)
- **Zero Gradient Ratio**: ~8.3%
- **Gradient Variance**: 5.18e-06 (step 56)
- **NaN / Inf Gradients**: 0.0 (None detected)

## Layer-by-Layer Health Verification

1. **Backbone Blocks**: Gradients flow stably across all 2 layers.
2. **MoE Experts Routing**: Routing weights show clean sparse gradients, matching top-1 expert allocation behavior.
3. **Titans Memory**: Gradients for the learning rate generator and memory updaters are healthy, verifying stable online updates.
4. **Mamba2 State-Space**: SSM projection weights gradients are bounded and non-zero.

## Conclusion

The gradient health monitor confirms stable backpropagation across all model parameters.
