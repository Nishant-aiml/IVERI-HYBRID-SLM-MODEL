# Phase 26: Supplementary Mathematical Appendices

**Manuscript Title**: *IVERI Core v1.1: A Byte-Latent Hybrid Neural Operating System with Titans Recurrent Memory and Dynamic Mixture-of-Recursions*  
**Audit Date**: 2026-07-30 19:36:00 UTC  

---

## Appendix A: Formal Derivation of Titans Recurrent Memory Updates

The Titans Recurrent Memory module updates its internal memory matrix M_t in R^{d x d} using a surprise-gated gradient descent rule:

$$S_t = 
abla_{M_{t-1}} \ell(M_{t-1} x_t, y_t)$$

$$M_t = (1 - \eta_t) M_{t-1} + \eta_t S_t x_t^T$$

Where:
- S_t is the instantaneous surprise gradient.
- \eta_t = \sigma(W_\eta x_t + b_\eta) is the dynamic learning rate gate (\eta_t in (0, 1)).
- x_t is the current input representation vector.

---

## Appendix B: Convergence Proof for Unrolled Recursions with alpha_0 = 0.05

Let h^{(r)} denote the hidden state at recursion depth r in [1, R]. The residual recurrence relation is defined as:

$$h^{(r+1)} = h^{(r)} + lpha_0 f_{MoR}(h^{(r)})$$

Assuming f_MoR is L-Lipschitz continuous (||f_MoR(a) - f_MoR(b)|| <= L ||a - b||), the Jacobian norm is bounded by:

$$||
abla_{h^{(r)}} h^{(r+1)}|| \le 1 + lpha_0 L$$

For R recursion steps, the total gradient expansion factor satisfies:

$$||
abla_{h^{(0)}} h^{(R)}|| \le (1 + lpha_0 L)^R pprox e^{lpha_0 L R}$$

Choosing lpha_0 = 0.05 ensures that for R <= 4 and L pprox 1.2, e^{0.05 	imes 1.2 	imes 4} = e^{0.24} pprox 1.27, guaranteeing that gradient norms remain strictly bounded without exploding.
