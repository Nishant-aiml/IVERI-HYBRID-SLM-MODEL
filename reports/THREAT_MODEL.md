# IVERI CORE — Threat Model & Security Architecture

**Scope:** Production & Open-Source Security Protocol  

---

## Threat Vectors & Safeguards
1. **Adversarial Unicode Attacks:** Malformed multi-byte UTF-8 sequences targeting tokenizer crash $	o$ *Handled natively by raw byte fallback (259 vocab).*
2. **Prompt Injection / Jailbreaks:** Indirect prompt injection via dynamic context $	o$ *Scrubbed by SafetyFilter & PII remover.*
3. **Memory Poisoning (Titans State):** Adversarial byte sequences designed to corrupt recurrent neural memory $	o$ *Bound by memory decay rate $lpha$ and gradient norm clipping.*
4. **Checkpoint Tampering:** Modifying `.pt` model weights $	o$ *Protected by cryptographic `.FROZEN` sentinels and SHA-256 manifest verification.*
