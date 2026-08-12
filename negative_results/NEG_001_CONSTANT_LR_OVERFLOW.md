# Negative Result NEG-001: Un-decayed Constant Learning Rate Softmax Overflow

**Experiment ID:** `EXP-0005-v1`  
**Finding:** Running `lr=1e-3` without a cosine decay schedule caused loss divergence (`NaN`) around Step 300 when cross-entropy logits reached near-zero values.  
**Resolution:** Implemented Cosine Annealing Learning Rate Schedule ($3 	imes 10^{-4} 	o 3 	imes 10^{-5}$) with linear warmup, completely eliminating FP16 gradient overflow.
