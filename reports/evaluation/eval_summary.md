# IVERI CORE — Evaluation Report
**Generated:** 2026-07-09T08:31:35
**Reproduction Seed:** 42

## 1. Run Metadata
| Attribute | Value |
|---|---|
| Git Commit | `07e8eb6` |
| Architecture Version | `0.2.0-byte-vocab` |
| Device / Hardware | `cpu` |
| Dtype | `float32` |
| PyTorch Version | `2.12.1+cpu` |
| CUDA Version | `N/A` |
| Evaluation Duration | `0.60s` |

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
| Latency (s) | `0.1444s` |
| Throughput (bytes/s) | `55.39 bytes/sec` |
| Average length | `4.0 bytes` |
| Early exit ratio | `0.00%` |

## 4. Inference Performance Benchmarks
| Metric | Latency / Throughput |
|---|---|
| Warmup Latency | `33.01 ms` |
| Average Latency | `30.72 ms` |
| Median Latency | `33.08 ms` |
| P90 / P95 / P99 | `34.29 / 34.44 / 34.56 ms` |
| Min / Max Latency | `24.48 / 34.59 ms` |
| Samples / sec | `65.11 samples/sec` |
| Tokens / sec | `976.64 tokens/sec` |
| Estimated FLOPs | `5.50e+07 FLOPs` |
| Model Parameter Count | `305,612` |

## 5. Memory Consumption Benchmarks
| Resource | Allocated | Reserved / Peak |
|---|---|---|
| GPU Memory | `0.0 MB` | `0.0 MB / 0.0 MB` |
| CPU System RAM | `588.1 MB` | `590.5 MB` |
| Parameter Memory | `1.17 MB` | - |
| Activation Memory (Est) | `0.00 MB` | - |
| Memory Fragmentation Ratio | `0.00%` | - |
| Memory Growth Delta | `3.89 MB` | - |

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
- State Variance: `3.3702e-10`
- SSM throughput: `1647.56 tokens/sec`

### Attention & Backbone stack
- Attention Backend Selected: `SDPA (PyTorch)`
- Backbone layer latency (list of layer runtimes): `[0.012515550013631582]`
- Backbone residual norm (mean): `7.9996`
- Backbone activation norm (mean): `7.9998`
