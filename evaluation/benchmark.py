# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Inference performance benchmarking for IVERI CORE (Phase 2.5).

Measures forward execution latencies, throughput metrics, CPU/GPU utilization,
and estimates FLOPs for model configurations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    PSUTIL_AVAILABLE = False


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Dataclass holding inference performance benchmark metrics."""

    warmup_latency_ms: float
    latency_mean_ms: float
    latency_median_ms: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_min_ms: float
    latency_max_ms: float
    samples_per_sec: float
    tokens_per_sec: float
    bytes_per_sec: float
    patches_per_sec: float
    docs_per_sec: float
    cpu_utilization_pct: float
    gpu_utilization_pct: float
    vram_used_mb: float
    ram_used_mb: float
    estimated_flops: float
    parameter_count: int


class InferenceBenchmark:
    """Benchmarks inference performance, latencies, and resource utilization."""

    def __init__(self, model: nn.Module) -> None:
        """Initialize the InferenceBenchmark.

        Args:
            model: IVERI model instance.
        """
        self.model = model
        self.param_count = sum(p.numel() for p in model.parameters())

    def _get_gpu_utilization(self) -> float:
        """Query GPU utilization using NVML, falling back to 0.0 if unavailable."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return float(rates.gpu)
        except Exception:
            return 0.0

    def run(
        self,
        input_ids: torch.Tensor,
        iterations: int = 50,
        warmup_iterations: int = 5,
        device: torch.device | None = None,
    ) -> BenchmarkResult:
        """Execute the benchmark.

        Args:
            input_ids: Input tensor of shape (B, S).
            iterations: Number of timed forward passes to execute.
            warmup_iterations: Number of un-timed warmup passes.
            device: Accelerator device to use.

        Returns:
            BenchmarkResult container.
        """
        self.model.eval()
        device_resolved = device or next(self.model.parameters()).device
        input_ids = input_ids.to(device_resolved)
        batch_size, seq_len = input_ids.shape

        # Warmup phase
        warmup_start = time.perf_counter()
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = self.model(input_ids, return_dict=True)
            if device_resolved.type == "cuda":
                torch.cuda.synchronize()
        warmup_latency_ms = ((time.perf_counter() - warmup_start) / max(1, warmup_iterations)) * 1000.0

        # Start utilization polling
        cpu_start_pct = psutil.cpu_percent(interval=None) if PSUTIL_AVAILABLE else 0.0
        gpu_start_pct = self._get_gpu_utilization()

        latencies_ms = []

        with torch.no_grad():
            for _ in range(iterations):
                if device_resolved.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                # Perform forward pass
                outputs = self.model(input_ids, return_dict=True)

                if device_resolved.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)

        # End utilization polling
        cpu_end_pct = psutil.cpu_percent(interval=None) if PSUTIL_AVAILABLE else 0.0
        gpu_end_pct = self._get_gpu_utilization()

        # Compute latency stats
        latencies = np.array(latencies_ms)
        mean_lat = float(np.mean(latencies))
        median_lat = float(np.median(latencies))
        p50 = float(np.percentile(latencies, 50))
        p90 = float(np.percentile(latencies, 90))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))
        min_lat = float(np.min(latencies))
        max_lat = float(np.max(latencies))

        # Throughput calculations (average latency in seconds)
        avg_sec = mean_lat / 1000.0
        samples_per_sec = batch_size / avg_sec if avg_sec > 0 else 0.0
        tokens_per_sec = (batch_size * seq_len) / avg_sec if avg_sec > 0 else 0.0
        bytes_per_sec = tokens_per_sec  # in raw byte model, 1 token = 1 byte
        docs_per_sec = samples_per_sec  # 1 sequence = 1 document

        # Extract patch counts from model forward telemetry if available
        patches_per_sec = 0.0
        if isinstance(outputs, dict) and "telemetry" in outputs:
            telemetry = outputs["telemetry"]
            if telemetry and "average_patch_length" in telemetry:
                avg_patch_len = telemetry["average_patch_length"]
                if avg_patch_len > 0:
                    patches_per_sec = tokens_per_sec / avg_patch_len

        # Resource stats
        cpu_util = (cpu_start_pct + cpu_end_pct) / 2.0
        gpu_util = (gpu_start_pct + gpu_end_pct) / 2.0

        vram_mb = 0.0
        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        ram_mb = 0.0
        if PSUTIL_AVAILABLE:
            import os
            proc = psutil.Process(os.getpid())
            ram_mb = proc.memory_info().rss / (1024 * 1024)

        # FLOPs estimate: standard forward pass is 2 * params per token
        # Estimated FLOPs total in the benchmark = 2 * param_count * batch * seq * iterations
        estimated_flops = 2.0 * self.param_count * batch_size * seq_len * iterations

        return BenchmarkResult(
            warmup_latency_ms=warmup_latency_ms,
            latency_mean_ms=mean_lat,
            latency_median_ms=median_lat,
            latency_p50_ms=p50,
            latency_p90_ms=p90,
            latency_p95_ms=p95,
            latency_p99_ms=p99,
            latency_min_ms=min_lat,
            latency_max_ms=max_lat,
            samples_per_sec=samples_per_sec,
            tokens_per_sec=tokens_per_sec,
            bytes_per_sec=bytes_per_sec,
            patches_per_sec=patches_per_sec,
            docs_per_sec=docs_per_sec,
            cpu_utilization_pct=cpu_util,
            gpu_utilization_pct=gpu_util,
            vram_used_mb=vram_mb,
            ram_used_mb=ram_mb,
            estimated_flops=estimated_flops,
            parameter_count=self.param_count,
        )
