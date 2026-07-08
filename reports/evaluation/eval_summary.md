# IVERI CORE — Evaluation Report
**Generated:** 2026-07-08T11:08:47
**Reproduction Seed:** 42

## 1. Run Metadata
| Attribute | Value |
|---|---|
| Git Commit | `f2adb39` |
| Architecture Version | `0.2.0-byte-vocab` |
| Device / Hardware | `cpu` |
| Dtype | `float32` |
| PyTorch Version | `2.5.1+cu121` |
| CUDA Version | `12.1` |
| Evaluation Duration | `1.36s` |

## 2. Language Modeling Performance
| Metric | Value |
|---|---|
| Cross Entropy Loss | `5.5671` |
| Perplexity | `261.6827` |
| Total Tokens evaluated | `60` |
| Total Batches processed | `2` |

## 3. Generative Decoding Performance
| Metric | Value |
|---|---|
| Latency (s) | `0.1796s` |
| Throughput (bytes/s) | `44.54 bytes/sec` |
| Average length | `4.0 bytes` |
| Early exit ratio | `0.00%` |

## 4. Inference Performance Benchmarks
| Metric | Latency / Throughput |
|---|---|
| Warmup Latency | `74.65 ms` |
| Average Latency | `65.30 ms` |
| Median Latency | `65.97 ms` |
| P90 / P95 / P99 | `69.86 / 70.35 / 70.74 ms` |
| Min / Max Latency | `59.08 / 70.84 ms` |
| Samples / sec | `30.63 samples/sec` |
| Tokens / sec | `459.44 tokens/sec` |
| Estimated FLOPs | `8.79e+07 FLOPs` |
| Model Parameter Count | `488,492` |

## 5. Memory Consumption Benchmarks
| Resource | Allocated | Reserved / Peak |
|---|---|---|
| GPU Memory | `16.2 MB` | `24.0 MB / 16.2 MB` |
| CPU System RAM | `430.2 MB` | `1217.5 MB` |
| Parameter Memory | `1.86 MB` | - |
| Activation Memory (Est) | `0.00 MB` | - |
| Memory Fragmentation Ratio | `47.69%` | - |
| Memory Growth Delta | `164.97 MB` | - |

## 6. Architecture Subsystem Telemetry
### Mixture of Recursions (MoR)
- Average Depth: `1.00`
- Median Depth: `1.00`
- 95th Percentile Depth: `1.00`
- Maximum Depth: `1`
- FLOPs Saved Ratio: `50.00%`

### Mixture of Experts (MoE)
- Expert utilization histogram: `[16, 44]`
- Unused experts count: `0`
- Max load / Min load: `44.0 / 16.0`
- Imbalance Ratio: `0.933`
- Routing Entropy: `0.580`
- Expert Collapse Score: `0.163`

### Byte Latent Transformer (BLT)
- Average Byte Entropy: `1.000`
- Average Patch Entropy: `1.000`
- Average / Median Patch Count: `512.0 / 512.0`
- Compression Ratio (seq_len / patches): `1.00`
- Boundary Frequency: `100.00%`

### Titans Gated Neural Memory
- Gated Memory reads / writes: `60 / 0`
- Memory Update Norm: `0.0000`
- Memory Retrieval Norm: `0.0000`

### Mamba2 Structured State Space Duality (SSD)
- Hidden State Norm (mean): `7.9998`
- State Update Norm: `1.0000`
- State Variance: `1.1787e-09`
- SSM throughput: `776.81 tokens/sec`

### Attention & Backbone stack
- Attention Backend Selected: `SDPA (PyTorch)`
- Backbone layer latency (list of layer runtimes): `[0.036524650000501424]`
- Backbone residual norm (mean): `7.9997`
- Backbone activation norm (mean): `7.9998`
