# IVERI CORE — Pre-Registered Experimental Protocol

**Date:** 2026-07-25  
**Status:** **FROZEN & SIGNED**  

---

## 1. Frozen Experimental Parameters
- **Vocabulary Size:** Raw Bytes (259 Vocab: 256 bytes + 3 special tokens).
- **Sequence Length:** 512 (Nano), 2,048 (Micro), 32,768 (Long Context Evaluation).
- **Optimizer:** AdamW ($eta_1=0.9, eta_2=0.95, \epsilon=10^{-8}$, weight decay $=0.1$).
- **Learning Rate Schedule:** Cosine decay with 1,000-step linear warmup ($LR_{\min} = 10^{-5}$).
- **Random Seeds:** `[42, 1337, 2026]` (Minimum 3 seeds per baseline run).
