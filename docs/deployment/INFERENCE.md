# IVERI CORE — Inference Deployment Guide

Phase 6.3.3 production inference uses the `inference/` package. Architecture is frozen; this guide covers deployment only.

## Prerequisites

- Python ≥ 3.10
- `pip install -e .` from repository root
- Optional: CUDA GPU for faster generation

## Quick start (CPU)

```bash
python -m inference.cli --prompt "Explain merge sort." --device cpu --max-new-tokens 64
```

Streaming output:

```bash
python -m inference.cli --prompt "Hello" --device cpu --stream
```

## Checkpoint loading

```bash
python -m inference.cli \
  --checkpoint path/to/checkpoint.pt \
  --prompt "Your prompt" \
  --device cuda
```

Checkpoints must match `ARCHITECTURE_VERSION=0.2.0-byte-vocab` (`BYTE_VOCAB_SIZE=259`).

## Python API

```python
from configs.base_config import get_base_config
from inference import InferenceEngine, load_inference_model

cfg = get_base_config()
cfg.hardware.device = "cpu"
model = load_inference_model("checkpoints/final.pt", config=cfg, device="cpu")
engine = InferenceEngine(model)

result = engine.generate("Explain B-trees.", max_new_tokens=128)
print(result.text, result.tokens_per_second)

for chunk in engine.stream("Once upon a time"):
    print(chunk.text_delta, end="", flush=True)
```

## Benchmarking

Measured CPU benchmark (nano-reduced profile for dev hardware):

```bash
python scripts/run_inference_benchmark.py --device cpu --output reports/phase_6_3_3/inference_benchmark.json
```

On CUDA:

```bash
python scripts/run_inference_benchmark.py --device cuda
```

Results include `avg_latency_seconds`, `avg_tokens_per_second`, and `peak_vram_mb` (GPU only). All benchmark outputs are tagged `provenance: MEASURED`.

## Logging

Inference does not initialize W&B. Training logging fallbacks are documented in `reports/phase_6_3_3/Logging_Report.md`.

## Stage 3B proprietary data

Domain-specific tuning requires proprietary JSON under `data/proprietary/`. See `data/proprietary/FORMAT.md` and:

```bash
python scripts/ingest_stage3b.py --validate-only
```

## Production checklist

1. Load a **measured** checkpoint from the experiment registry (not mock metrics).
2. Verify `BYTE_VOCAB_SIZE=259` and byte-level decode in your integration tests.
3. Run `python -m pytest tests/test_inference.py` after deployment image build.
4. Monitor latency, peak memory, and invalid UTF-8 rates under real prompts.
5. Do not publish benchmark claims until `PublicationManager` integrity gates pass with `MEASURED` provenance.

## Related reports

- `reports/phase_6_3_3/Inference_Architecture.md`
- `reports/phase_6_3_3/Performance_Report.md`
- `reports/phase_6_3_3/Deployment_Report.md`
