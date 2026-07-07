# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Inference benchmark utilities."""

from __future__ import annotations

import time
from typing import Any

import torch

from inference.engine import InferenceEngine


def benchmark_inference(
    engine: InferenceEngine,
    prompt: str = "Hello, IVERI",
    *,
    warmup: int = 2,
    runs: int = 5,
) -> dict[str, Any]:
    """Measure TTFT proxy, throughput, and peak memory."""
    device = engine.device
    for _ in range(warmup):
        engine.generate(prompt, max_new_tokens=8)

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    latencies: list[float] = []
    tps_list: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        result = engine.generate(prompt, max_new_tokens=32)
        latencies.append(time.perf_counter() - t0)
        tps_list.append(result.tokens_per_second)

    peak_vram_mb = 0.0
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    return {
        "runs": runs,
        "avg_latency_seconds": sum(latencies) / len(latencies),
        "avg_tokens_per_second": sum(tps_list) / len(tps_list),
        "peak_vram_mb": peak_vram_mb,
    }
