# 24 LIMITATIONS
**STATUS: FINAL**

1. **LR Fairness Threat**: The Transformer grad_norms of 34-55 vs IVERI's 7-9 suggests lr=3e-3 is sub-optimal for the Transformer. A tuned LR for the Transformer might close the loss gap significantly.
2. **Data Threat**: All results are based on a 10,120-char synthetic English corpus repeated 40x. Results may not generalize to real text distributions.
3. **Training Steps Threat**: 10 training steps is extremely preliminary. Long-run behavior is unknown.
4. **Attribution Threat**: IVERI's lower loss cannot be attributed to any specific component (Titans/MoR/BLT/Mamba2) without independent ablation studies.
5. **Input Representation Threat**: IVERI uses entropy BLT patching internally; the Transformer sees raw bytes. Architecture and input representation are confounded.
6. **Latency Overhead**: Python-level recurrent loop causes ~85x slowdown vs Transformer.
7. **Long Context**: Unable to test beyond 512 bytes due to timeouts.
