# IVERI CORE

> The world's first byte-entropy-native hybrid SLM combining BLT + Titans + Mamba2 + MoR + MoE — solving token waste, context limits, memory bloat, and inference cost in a single unified architecture.

## Architecture

```
INPUT
  Raw bytes
  → BLT Entropy Model (scores byte complexity)
  → Dynamic Patcher (groups bytes by entropy)
  → BLT Local Encoder (bytes → patch vectors)

MEMORY
  → Titans Neural Memory Module
    (deep MLP, updates weights online, no context limit)

BACKBONE (18 blocks at full scale)
  Each block:
  ┌────────────────────────────────┐
  │ MoR Router                     │  assigns recursion depth per token
  │ Mamba2 × 6   [linear O(n)]    │  92% of compute, no KV cache
  │ Flash Attention × 1            │  in-context recall, 8% of compute
  │ MoE FFN (4 experts, 2 active) │  50% parameter efficiency
  │ RMSNorm + SwiGLU + RoPE       │
  └────────────────────────────────┘

OUTPUT
  → BLT Local Decoder (patch vectors → raw bytes)
  → BLT-D: parallel byte generation
```

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 0** | Project Foundation & Infrastructure | ✅ Complete & Frozen |
| **Phase 1** | Core Architecture (BLT, Titans, Mamba2, MoR, MoE, Backbone) | ✅ Complete & Frozen (`0.2.0-byte-vocab`) |
| **Phase 2** | Training Infrastructure (data, trainer, logging, evaluation) | ✅ Complete |
| **Phase 3** | Benchmarking, SFT, coding, alignment, orchestration (3.1–3.6) | ✅ Complete |
| **Phase 4** | Production campaign runner & publication packaging | ✅ Complete |
| **Phase 6.3** | Paper-profile campaign artifacts & reviewer bundle | ✅ Engineering complete |
| **Phase 6.3.2** | Scientific integrity restoration (OBJ1–8) | ✅ Complete |
| **Phase 6.3.3** | Engineering stabilization (tests, inference, deployment) | ✅ Complete |

See `docs/phases/INDEX.md` for per-phase reports and `reports/scientific_integrity_audit/` for measured integrity gates.

### Phase 6.3.2 Scientific Integrity (OBJ1–8)

| Objective | Topic | Report |
|-----------|-------|--------|
| 1 | BLT end-to-end causality | `Causality_Report.md` |
| 2 | Titans production integration | `Titans_Verification.md` |
| 3 | Entropy-conditioned MoE routing | `Entropy_Routing_Report.md` |
| 4 | Physical ablation framework | `Ablation_Verification.md` |
| 5 | Publication fail-closed gates | `Publication_Integrity_Report.md` |
| 6 | Replay integrity | `Replay_Integrity_Report.md` |
| 7 | Collision-free byte vocabulary | `Byte_Vocabulary_Report.md` |
| 8 | Documentation sync | `Documentation_Sync_Report.md` |

### Byte Vocabulary (v0.2.0)

Content UTF-8 bytes map 1:1 to token IDs **0–255**. Structural specials are collision-free: **BOS=256**, **PAD=257**, **EOS=258** (`BYTE_VOCAB_SIZE=259`). See `docs/migrations/PHASE_6_3_2_OBJ7_BYTE_VOCAB.md`.

## Inference (Phase 6.3.3)

Production byte-level inference via the `inference/` package:

```bash
# CLI
python -m inference.cli --prompt "Explain quicksort." --device cpu

# Benchmark (writes reports/phase_6_3_3/inference_benchmark.json)
python scripts/run_inference_benchmark.py --device cpu
```

See `docs/deployment/INFERENCE.md` for checkpoint loading, streaming, and deployment checklist.

### Stage 3B proprietary data

Ingest proprietary JSON (university papers, GATE, placement Q&A) with:

```bash
python scripts/ingest_stage3b.py --validate-only   # check records
python scripts/ingest_stage3b.py                 # process to data/processed/stage3b/
```

Format: `data/proprietary/FORMAT.md`. Raw assets are not shipped in this repository.

## Setup

### Prerequisites

- Python ≥ 3.10
- NVIDIA GPU with CUDA (RTX 3050 4GB minimum for development)

### Installation

```bash
# Clone the repository
cd iveri-core

# Install in development mode
pip install -e .

# Install dev dependencies
pip install -r requirements-dev.txt

# GPU-specific packages (requires CUDA toolkit)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install mamba-ssm --no-build-isolation
pip install flash-attn --no-build-isolation
```

### Verify Installation

```bash
# Run all tests
python -m pytest tests/ -v

# Run quality checks
python quality/run_all.py

# Verify config system
python -c "from configs.base_config import get_base_config; c = get_base_config(); print(c)"
```

## Project Structure

```
iveri-core/
├── configs/        Configuration system (dataclasses)
├── core/           Infrastructure (registry, factory, interfaces, exceptions)
├── model/          Model components (created per Phase 1 step)
│   ├── blt/        Byte Latent Transformer
│   ├── mamba2/     Mamba2 SSM blocks
│   ├── mor/        Mixture of Recursions
│   ├── titans/     Titans Neural Memory
│   └── moe/        Mixture of Experts
├── data/           Data loading & preprocessing
├── training/       Training infrastructure
├── evaluation/     Benchmarking & evaluation
├── inference/      Production inference (CLI, streaming, benchmark)
├── research/       Campaign orchestration, audits, publication
├── baselines/      Baseline models for comparison
├── utils/          Logging, validation utilities
├── scripts/        Entry point scripts
├── tests/          Test suite
├── quality/        Local CI quality checks
├── docs/           Documentation
├── experiments/    Experiment tracking
├── research_log/   Master research log
└── reports/        Phase completion reports
```

## Scale Versions

| Version | Parameters | Purpose |
|---------|-----------|---------|
| v0.1 Nano | 10M | Verify architecture |
| v0.2 Core | 50M | Verify full stack |
| v1.0 Mini | 300M | Full proof of concept |
| v2.0 Base | 1B | First competitive product |
| v3.0 Pro | 7B | Genuinely competitive |
| v4.0 Max | 70B | Frontier domain champion |

## License

Apache-2.0

---

*IVERI CORE v0.1.0 — Architecture Revision 0.2.0-byte-vocab*
