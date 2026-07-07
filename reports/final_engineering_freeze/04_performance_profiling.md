# Performance Profiling Report

**Audit Date:** 2026-07-07  
**Auditor:** Independent AI Architect (Antigravity)  
**Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4.0 GB VRAM)  
**Protocol:** Phase 6.4 — Final Engineering Freeze Audit

---

## 1. VRAM Profile (Nano Config — 8.8M params)

| Metric | Value | Status |
|--------|-------|--------|
| Peak VRAM | 175.1 MB | PASS |
| Current VRAM | 172.9 MB | PASS |
| Reserved VRAM | 256.0 MB | PASS |
| Fragmentation | 83.1 MB (32.4%) | ACCEPTABLE |

> The nano model easily fits within the 4 GB VRAM budget. The production 10M-param config (with longer sequences) will require more.

---

## 2. Throughput (Nano Config, B=1, S=64)

| Metric | Value | Notes |
|--------|-------|-------|
| Forward pass | 0.736 s | Single batch |
| Forward + backward | 7.138 s | Including gradient computation |
| Throughput | 87 bytes/sec | At batch=1, seq=64 |
| Bytes per step | 64 | Nano config |

---

## 3. Component-Level Timing (Forward Pass Breakdown)

| Component | Time (s) | % of Forward | Status |
|-----------|---------|-------------|--------|
| Titans memory | 0.0448 | 6.1% | PASS |
| Backbone blocks (total) | 0.6224 | 84.6% | PRIMARY BOTTLENECK |
| Block 0 | 0.3135 | 42.6% | — |
| Block 1 | 0.3089 | 42.0% | — |
| Other (BLT, etc.) | 0.0688 | 9.3% | PASS |

### Analysis

The backbone blocks dominate runtime at 84.6%. Within each block, the Mamba2 `selective_ssd_scan()` is the primary hotspot (previously profiled at ~162s for production-size sequences). This is expected — the pure PyTorch fallback for the Mamba2 scan is computationally expensive without custom CUDA kernels.

### Optimization Recommendations

1. **Custom CUDA kernel for `selective_ssd_scan()`** — The chunked parallel scan algorithm should be implemented as a CUDA extension for 10-50x speedup.
2. **Reduce sequence length during development** — Use `max_seq_len=256` or `512` for iteration speed.
3. **Gradient checkpointing** — Already implemented. Should be enabled for production training.
4. **Gradient accumulation** — Verified working. Use `accumulation_steps=4-8` to increase effective batch size without VRAM increase.

---

## 4. Parameter Budget

| Metric | Value | Status |
|--------|-------|--------|
| Total parameters | 8,821,458 | PASS |
| Trainable parameters | 8,821,458 (100%) | PASS |
| Frozen parameters | 0 | Expected for pretraining |

---

## 5. Memory Stability Under Load

| Test | Metric | Value | Status |
|------|--------|-------|--------|
| Memory leak check | VRAM delta over 20 iterations | 0.3 MB | PASS |
| Gradient accumulation | Accumulates correctly | `total_loss=-0.0019` | PASS |

---

## Overall Performance Verdict: **PASS (with optimization recommendations)**

> The architecture is correct and all subsystems function as specified. Performance is constrained by the pure PyTorch Mamba2 scan implementation. A custom CUDA kernel is the primary optimization target for production training throughput.
