# 25 THREATS TO VALIDITY
**STATUS: FINAL**

1. **LR threat**: Transformer grad_norms of 34-55 vs IVERI's 7-9 suggests lr=3e-3 is sub-optimal for the Transformer. A tuned LR for the Transformer might close the loss gap significantly.
2. **Data threat**: All results on a 10,120-char synthetic English corpus repeated 40x. Results may not generalize to real text distributions.
3. **Steps threat**: 10 training steps is preliminary. Long-run behavior is unknown.
4. **Attribution threat**: IVERI's lower loss cannot be attributed to any specific component (Titans/MoR/BLT/Mamba2) without ablations.
5. **Representation threat**: IVERI uses entropy BLT patching internally; the Transformer sees raw bytes. Architecture and input representation are confounded.
6. **Power threat**: Sample size of n=4 stable Transformer seeds yields df=3 for hypothesis testing, limiting the statistical strength.
