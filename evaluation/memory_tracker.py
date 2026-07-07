# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Memory tracking utilities for evaluating CPU and GPU utilization (Phase 2.5)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    PSUTIL_AVAILABLE = False


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """Dataclass holding detailed memory utilization metrics."""

    gpu_allocated_mb: float
    gpu_reserved_mb: float
    gpu_peak_mb: float
    cpu_ram_mb: float
    cpu_peak_ram_mb: float
    parameter_mb: float
    activation_mb: float
    checkpoint_mb: float
    fragmentation_ratio: float
    growth_mb: float


class MemoryTracker:
    """Context manager for tracking memory consumption, fragmentation, and growth."""

    def __init__(self, model: nn.Module | None = None) -> None:
        """Initialize the MemoryTracker.

        Args:
            model: Optional model to estimate parameter and activation memory.
        """
        self.model = model
        self.start_gpu_allocated = 0.0
        self.start_gpu_peak = 0.0
        self.start_cpu_ram = 0.0
        self.end_gpu_allocated = 0.0
        self.end_gpu_peak = 0.0
        self.end_cpu_ram = 0.0
        self.end_cpu_peak = 0.0

    def __enter__(self) -> MemoryTracker:
        """Record starting memory snapshots."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self.start_gpu_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            self.start_gpu_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            self.start_gpu_allocated = 0.0
            self.start_gpu_peak = 0.0

        self.start_cpu_ram = self._get_cpu_ram()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Record final memory snapshots."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.end_gpu_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
            self.end_gpu_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            self.end_gpu_allocated = 0.0
            self.end_gpu_peak = 0.0

        self.end_cpu_ram = self._get_cpu_ram()
        self.end_cpu_peak = self._get_peak_cpu_ram()

    def _get_cpu_ram(self) -> float:
        """Get current CPU RAM usage in MB."""
        if not PSUTIL_AVAILABLE:
            return 0.0
        proc = psutil.Process(os.getpid())
        return float(proc.memory_info().rss) / (1024 * 1024)

    def _get_peak_cpu_ram(self) -> float:
        """Get peak CPU RAM usage in MB."""
        if not PSUTIL_AVAILABLE:
            return 0.0
        proc = psutil.Process(os.getpid())
        info = proc.memory_info()
        if hasattr(info, "peak_wset"):
            return float(info.peak_wset) / (1024 * 1024)  # type: ignore[attr-defined]

        # Non-Windows POSIX fallback
        try:
            import resource  # type: ignore[import-not-found,unused-import]

            maxrss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)  # type: ignore[attr-defined]
            if sys.platform != "darwin":
                # Linux ru_maxrss is in KB
                return maxrss / 1024.0
            # macOS ru_maxrss is in bytes
            return maxrss / (1024 * 1024.0)
        except Exception:
            return float(info.rss) / (1024 * 1024)

    def get_snapshot(self) -> MemorySnapshot:
        """Compute the final MemorySnapshot.

        Returns:
            MemorySnapshot container.
        """
        gpu_alloc = 0.0
        gpu_res = 0.0
        gpu_peak = 0.0
        frag_ratio = 0.0

        if torch.cuda.is_available():
            gpu_alloc = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_res = torch.cuda.memory_reserved() / (1024 * 1024)
            gpu_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
            if gpu_alloc > 0:
                frag_ratio = (gpu_res - gpu_alloc) / gpu_alloc

        curr_cpu_ram = self._get_cpu_ram()
        peak_cpu_ram = max(self.end_cpu_peak, curr_cpu_ram)

        # Param memory
        param_mb = 0.0
        if self.model is not None:
            param_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
            param_mb = param_bytes / (1024 * 1024)

        # Estimate activations (difference between peak during run and starting model weights)
        activation_mb = max(0.0, self.end_gpu_peak - max(self.start_gpu_allocated, param_mb))

        # Checkpoint size estimation
        checkpoint_mb = param_mb * 1.25  # weights + optimizer state ratio

        # Memory growth across inference
        growth = max(0.0, self.end_cpu_ram - self.start_cpu_ram)

        return MemorySnapshot(
            gpu_allocated_mb=gpu_alloc,
            gpu_reserved_mb=gpu_res,
            gpu_peak_mb=gpu_peak,
            cpu_ram_mb=curr_cpu_ram,
            cpu_peak_ram_mb=peak_cpu_ram,
            parameter_mb=param_mb,
            activation_mb=activation_mb,
            checkpoint_mb=checkpoint_mb,
            fragmentation_ratio=frag_ratio,
            growth_mb=growth,
        )
