# Model Card — IVERI CORE (10M Nano)

## Model Details
- **Architecture Version:** v1.0.0-Nano
- **Parameter Count:** 10,480,256 parameters (10M class)
- **Base Components:** BLT Latent Transformer, Mamba2 Backbone, MoR Router, Titans Memory Module, MoE FFN.
- **Layers:** 6 backbone layers (SSM to Attention ratio: 6:1)
- **Vocabulary:** Byte-native (vocab size = 256 + tokens)
- **Training Checkpoint ID:** `ckpt_IVERI_2026_Phase6_1_Seed42_IVERI_Run001`
- **License:** Apache-2.0

## Intended Use
Designed for high-throughput, low-latency small language model research, testing token-free byte-level encoding, Titans neural memory scaling, and selective Mixture of Recursions routing.

## Capabilities
- Multi-needle long context recall (up to 128K context window).
- Native handling of non-English text and source code.
