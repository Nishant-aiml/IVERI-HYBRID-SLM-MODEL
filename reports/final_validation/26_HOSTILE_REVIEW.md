# 26 HOSTILE REVIEW
**STATUS: FINAL**

### Q1: Is the comparison to the Transformer baseline fair given the parameter matching?
**ANSWER:** The comparison satisfies the preregistered parameter-matching criterion of <=5% relative difference.

### Q2: How can you claim efficiency when the latency is ~85x slower than a standard transformer?
**ANSWER:** We do not claim computational efficiency. IVERI demonstrates improved early convergence quality at this scale (lower loss at step 10, 0/5 seeds diverged vs 1/5) while currently incurring substantial implementation overhead (~85x step latency vs the matched Transformer). This overhead arises from Python-level sequential execution of the Titans recurrent state update, which is an engineering limitation rather than an algorithmic complexity property.

### Q3: Why does the model fail to scale beyond a trivial 512 context length despite the theoretical O(1) memory of Titans?
**ANSWER:** The python-level recurrent processing bottlenecks context scaling, causing hard timeouts beyond 512 bytes on the tested hardware. The theoretical claims require CUDA kernel fusion to be evaluated empirically.

### Q4: Does the same LR (3e-3) for both architectures unfairly penalize the Transformer?
**ANSWER:** Yes, this is a significant threat to validity. The Transformer grad_norms of 34-55 vs IVERI's 7-9 suggests lr=3e-3 is sub-optimal for the Transformer. A tuned LR for the Transformer might close the loss gap significantly.

### Q5: Is the comparison fair if IVERI uses BLT patching while the Transformer processes raw bytes?
**ANSWER:** This confounds the architectural improvements with input representation improvements. Since IVERI uses entropy BLT patching internally while the Transformer sees raw bytes, the lower loss cannot be cleanly attributed to the recurrent architecture alone.
