# IVERI CORE — Evaluation Report
**Generated:** 2026-08-21T18:47:09
**Reproduction Seed:** 42

## 1. Run Metadata
| Attribute | Value |
|---|---|
| Git Commit | `4c1182d` |
| Architecture Version | `0.2.0-byte-vocab` |
| Device / Hardware | `cpu` |
| Dtype | `float32` |
| PyTorch Version | `2.5.1+cu121` |
| CUDA Version | `12.1` |
| Evaluation Duration | `0.45s` |

## 2. Language Modeling Performance
| Metric | Value |
|---|---|
| Cross Entropy Loss | `5.5554` |
| Perplexity | `258.6379` |
| Total Tokens evaluated | `60` |
| Total Batches processed | `2` |

## 3. Generative Decoding Performance
| Metric | Value |
|---|---|
| Latency (s) | `0.1017s` |
| Throughput (bytes/s) | `78.65 bytes/sec` |
| Average length | `4.0 bytes` |
| Early exit ratio | `0.00%` |

## 4. Inference Performance Benchmarks
| Metric | Latency / Throughput |
|---|---|
| Warmup Latency | `24.92 ms` |
| Average Latency | `24.08 ms` |
| Median Latency | `24.21 ms` |
| P90 / P95 / P99 | `24.73 / 24.79 / 24.84 ms` |
| Min / Max Latency | `23.17 / 24.86 ms` |
| Samples / sec | `83.06 samples/sec` |
| Tokens / sec | `1245.95 tokens/sec` |
| Estimated FLOPs | `5.50e+07 FLOPs` |
| Model Parameter Count | `305,742` |

## 5. Memory Consumption Benchmarks
| Resource | Allocated | Reserved / Peak |
|---|---|---|
| GPU Memory | `16.2 MB` | `24.0 MB / 16.2 MB` |
| CPU System RAM | `1191.3 MB` | `1192.7 MB` |
| Parameter Memory | `1.17 MB` | - |
| Activation Memory (Est) | `0.00 MB` | - |
| Memory Fragmentation Ratio | `47.69%` | - |
| Memory Growth Delta | `3.76 MB` | - |

## 6. Architecture Subsystem Telemetry
### Mixture of Recursions (MoR)
- Average Depth: `1.00`
- Median Depth: `1.00`
- 95th Percentile Depth: `1.00`
- Maximum Depth: `1`
- FLOPs Saved Ratio: `50.00%`

### Mixture of Experts (MoE)
- Expert utilization histogram: `[55, 65]`
- Unused experts count: `0`
- Max load / Min load: `65.0 / 55.0`
- Imbalance Ratio: `0.167`
- Routing Entropy: `0.690`
- Expert Collapse Score: `0.005`

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
- Hidden State Norm (mean): `7.9378`
- State Update Norm: `0.9998`
- State Variance: `8.9701e-06`
- SSM throughput: `1309.12 tokens/sec`

### Attention & Backbone stack
- Attention Backend Selected: `SDPA (PyTorch)`
- Backbone layer latency (list of layer runtimes): `[0.017964250000659376]`
- Backbone residual norm (mean): `7.9361`
- Backbone activation norm (mean): `7.9378`
