# IVERI CORE — Evaluation Report
**Generated:** 2026-07-17T13:01:45
**Reproduction Seed:** 42

## 1. Run Metadata
| Attribute | Value |
|---|---|
| Git Commit | `4d1e203` |
| Architecture Version | `0.2.0-byte-vocab` |
| Device / Hardware | `cpu` |
| Dtype | `float32` |
| PyTorch Version | `2.5.1+cu121` |
| CUDA Version | `12.1` |
| Evaluation Duration | `0.63s` |

## 2. Language Modeling Performance
| Metric | Value |
|---|---|
| Cross Entropy Loss | `5.5639` |
| Perplexity | `260.8499` |
| Total Tokens evaluated | `60` |
| Total Batches processed | `2` |

## 3. Generative Decoding Performance
| Metric | Value |
|---|---|
| Latency (s) | `0.1253s` |
| Throughput (bytes/s) | `63.82 bytes/sec` |
| Average length | `4.0 bytes` |
| Early exit ratio | `0.00%` |

## 4. Inference Performance Benchmarks
| Metric | Latency / Throughput |
|---|---|
| Warmup Latency | `31.80 ms` |
| Average Latency | `32.87 ms` |
| Median Latency | `33.21 ms` |
| P90 / P95 / P99 | `33.82 / 33.90 / 33.96 ms` |
| Min / Max Latency | `31.41 / 33.97 ms` |
| Samples / sec | `60.85 samples/sec` |
| Tokens / sec | `912.80 tokens/sec` |
| Estimated FLOPs | `5.50e+07 FLOPs` |
| Model Parameter Count | `305,612` |

## 5. Memory Consumption Benchmarks
| Resource | Allocated | Reserved / Peak |
|---|---|---|
| GPU Memory | `16.2 MB` | `24.0 MB / 16.2 MB` |
| CPU System RAM | `1184.4 MB` | `1187.0 MB` |
| Parameter Memory | `1.17 MB` | - |
| Activation Memory (Est) | `0.00 MB` | - |
| Memory Fragmentation Ratio | `47.69%` | - |
| Memory Growth Delta | `2.67 MB` | - |

## 6. Architecture Subsystem Telemetry
### Mixture of Recursions (MoR)
- Average Depth: `1.00`
- Median Depth: `1.00`
- 95th Percentile Depth: `1.00`
- Maximum Depth: `1`
- FLOPs Saved Ratio: `50.00%`

### Mixture of Experts (MoE)
- Expert utilization histogram: `[42, 18]`
- Unused experts count: `0`
- Max load / Min load: `42.0 / 18.0`
- Imbalance Ratio: `0.800`
- Routing Entropy: `0.611`
- Expert Collapse Score: `0.119`

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
- State Variance: `3.4584e-10`
- SSM throughput: `1399.00 tokens/sec`

### Attention & Backbone stack
- Attention Backend Selected: `SDPA (PyTorch)`
- Backbone layer latency (list of layer runtimes): `[0.013358250005694572]`
- Backbone residual norm (mean): `7.9996`
- Backbone activation norm (mean): `7.9998`
