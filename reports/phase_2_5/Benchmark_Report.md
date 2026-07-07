# Benchmark Report — Phase 2.5
**IVERI CORE v1.0 | Phase 2.5**
**Date:** 2026-06-30

---

## 1. Latency Profile

The `InferenceBenchmark` evaluates forward execution latency across a configurable number of iterations:
- **Warmup Latency**: Discarded from stats to prevent cold-start noise.
- **Latency Percentiles**: Computes mean, median, p50, p90, p95, p99, minimum, and maximum latencies in milliseconds using `numpy.percentile`.
- **Barrier Synchronization**: Enforces CUDA barrier synchronization (`torch.cuda.synchronize()`) before and after timed sections when running on GPU.

---

## 2. Throughput Metrics

Throughput is logged across five channels to provide comprehensive performance evaluation:
- **Samples/sec**: Sequence throughput rate.
- **Tokens/sec**: Token processing speed.
- **Bytes/sec**: Raw byte rate (1 token = 1 byte).
- **Patches/sec**: Average patch rate resolved from dynamic patch boundaries.
- **Docs/sec**: Document throughput.

---

## 3. Resource Telemetry

System resource utilization is captured via `psutil` and optional NVML bindings:
- **CPU Utilization**: Average CPU percentage across the benchmark run.
- **GPU Utilization**: Average GPU core percentage (via `pynvml`).
- **RAM/VRAM**: Peak RAM and VRAM allocation.

---

## 4. FLOPs Estimation

The benchmark computes the analytical floating-point operations (FLOPs) performed during the run.
- Standard forward pass FLOPs = $2 \times \text{parameter\_count} \times \text{batch\_size} \times \text{sequence\_length} \times \text{iterations}$.
- This metric acts as an architectural indicator for computing hardware FLOPS metrics.
