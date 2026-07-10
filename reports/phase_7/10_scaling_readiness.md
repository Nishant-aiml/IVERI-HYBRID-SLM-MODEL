# Phase 7.10 -- IVERI Scaling Readiness Report

Generated: 2026-07-10 12:34:19 UTC

## Summary

| Preset | Params | Config | Init | Forward | NaN? | VRAM Train | Issues |
|--------|--------|--------|------|---------|------|-----------|--------|
| large_150m | 140.8M | [PASS] | [PASS] | [PASS] | no | 2084 MB | 0 |
| max_500m | 552.3M | [PASS] | [PASS] | [PASS] | no | 7425 MB | 0 |
| medium_70m | 68.1M | [PASS] | [PASS] | [PASS] | no | 1319 MB | 0 |
| nano_10m | 9.3M | [PASS] | [PASS] | [PASS] | no | 1763 MB | 0 |
| small_35m | 35.5M | [PASS] | [PASS] | [PASS] | no | 1293 MB | 0 |
| xlarge_300m | 302.2M | [PASS] | [PASS] | [PASS] | no | 4137 MB | 0 |

## Verdict: ALL PRESETS PASS

## Detailed Results

### large_150m

- **Parameters**: 140.8M total
  - Trainable: 140,760,574
- **Architecture**: D=768 L=5 H=12 Experts=4 (K=1) Depth=8 TitansDim=192
- **Training**: batch=4 x accum=32 = eff_batch=128 | seq=512 | steps=50000
- **FLOPs/token**: 0.1003 GFLOPs (estimate)

**VRAM Estimates (FP16 inference, FP32 training):**

| Mode | VRAM | GPU Compatibility |
|------|------|-------------------|
| Model weights (fp16) | 268 MB | -- |
| Inference (B=1) | 371 MB | 4GB OK |
| Training (B=4) | 2084 MB | 8GB GPU |

**Forward Pass**: [PASS]
- Output shape: [1, 64, 259]
- Has NaN: False
- Latency (B=1, S=64): 1137.7 ms

**No issues found.**

---

### max_500m

- **Parameters**: 552.3M total
  - Trainable: 552,296,390
- **Architecture**: D=1280 L=7 H=20 Experts=4 (K=1) Depth=16 TitansDim=320
- **Training**: batch=1 x accum=128 = eff_batch=128 | seq=512 | steps=50000
- **FLOPs/token**: 0.3899 GFLOPs (estimate)

**VRAM Estimates (FP16 inference, FP32 training):**

| Mode | VRAM | GPU Compatibility |
|------|------|-------------------|
| Model weights (fp16) | 1053 MB | -- |
| Inference (B=1) | 1079 MB | 4GB OK |
| Training (B=1) | 7425 MB | 8GB GPU |

**Forward Pass**: [PASS]
- Output shape: [1, 64, 259]
- Has NaN: False
- Latency (B=1, S=64): 4454.2 ms

**No issues found.**

---

### medium_70m

- **Parameters**: 68.1M total
  - Trainable: 68,085,182
- **Architecture**: D=512 L=5 H=8 Experts=4 (K=1) Depth=8 TitansDim=128
- **Training**: batch=8 x accum=16 = eff_batch=128 | seq=512 | steps=50000
- **FLOPs/token**: 0.0446 GFLOPs (estimate)

**VRAM Estimates (FP16 inference, FP32 training):**

| Mode | VRAM | GPU Compatibility |
|------|------|-------------------|
| Model weights (fp16) | 130 MB | -- |
| Inference (B=1) | 335 MB | 4GB OK |
| Training (B=8) | 1319 MB | 8GB GPU |

**Forward Pass**: [PASS]
- Output shape: [1, 64, 259]
- Has NaN: False
- Latency (B=1, S=64): 715.1 ms

**No issues found.**

---

### nano_10m

- **Parameters**: 9.3M total
  - Trainable: 9,347,282
- **Architecture**: D=256 L=4 H=4 Experts=2 (K=1) Depth=4 TitansDim=64
- **Training**: batch=32 x accum=4 = eff_batch=128 | seq=512 | steps=50000
- **FLOPs/token**: 0.0105 GFLOPs (estimate)

**VRAM Estimates (FP16 inference, FP32 training):**

| Mode | VRAM | GPU Compatibility |
|------|------|-------------------|
| Model weights (fp16) | 18 MB | -- |
| Inference (B=1) | 837 MB | 4GB OK |
| Training (B=32) | 1763 MB | 8GB GPU |

**Forward Pass**: [PASS]
- Output shape: [1, 64, 259]
- Has NaN: False
- Latency (B=1, S=64): 154.1 ms

**No issues found.**

---

### small_35m

- **Parameters**: 35.5M total
  - Trainable: 35,507,614
- **Architecture**: D=384 L=5 H=6 Experts=4 (K=1) Depth=6 TitansDim=96
- **Training**: batch=16 x accum=8 = eff_batch=128 | seq=512 | steps=50000
- **FLOPs/token**: 0.0251 GFLOPs (estimate)

**VRAM Estimates (FP16 inference, FP32 training):**

| Mode | VRAM | GPU Compatibility |
|------|------|-------------------|
| Model weights (fp16) | 68 MB | -- |
| Inference (B=1) | 477 MB | 4GB OK |
| Training (B=16) | 1293 MB | 8GB GPU |

**Forward Pass**: [PASS]
- Output shape: [1, 64, 259]
- Has NaN: False
- Latency (B=1, S=64): 475.4 ms

**No issues found.**

---

### xlarge_300m

- **Parameters**: 302.2M total
  - Trainable: 302,175,970
- **Architecture**: D=1024 L=6 H=16 Experts=4 (K=1) Depth=12 TitansDim=256
- **Training**: batch=2 x accum=64 = eff_batch=128 | seq=512 | steps=50000
- **FLOPs/token**: 0.2139 GFLOPs (estimate)

**VRAM Estimates (FP16 inference, FP32 training):**

| Mode | VRAM | GPU Compatibility |
|------|------|-------------------|
| Model weights (fp16) | 576 MB | -- |
| Inference (B=1) | 628 MB | 4GB OK |
| Training (B=2) | 4137 MB | 8GB GPU |

**Forward Pass**: [PASS]
- Output shape: [1, 64, 259]
- Has NaN: False
- Latency (B=1, S=64): 3301.9 ms

**No issues found.**

---
