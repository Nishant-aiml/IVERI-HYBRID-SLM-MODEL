# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Convergence and throughput analysis for IVERI CORE pretraining.

Computes mathematical convergence rates (slope, rolling variance, best/worst bounds,
loss reduction %) and resource throughput (FLOPs/sec, CPU/GPU utilization).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class ConvergenceAnalyzer:
    """Computes mathematical convergence and throughput metrics during training."""

    def __init__(self, window_size: int = 50) -> None:
        self.window_size = window_size
        self.losses: deque[float] = deque(maxlen=window_size)
        self.step_times: deque[float] = deque(maxlen=window_size)
        self.tokens_processed: deque[int] = deque(maxlen=window_size)
        self.patches_processed: deque[int] = deque(maxlen=window_size)

        # Performance history
        self.start_time = time.perf_counter()
        self.best_loss = float("inf")
        self.worst_loss = float("-inf")
        self.initial_loss: float | None = None

    def update(self, loss: float, step_time: float, num_tokens: int, num_patches: int = 0) -> None:
        """Register a new step for rolling metrics."""
        self.losses.append(loss)
        self.step_times.append(step_time)
        self.tokens_processed.append(num_tokens)
        self.patches_processed.append(num_patches)

        if self.initial_loss is None:
            self.initial_loss = loss
        self.best_loss = min(self.best_loss, loss)
        self.worst_loss = max(self.worst_loss, loss)

    def analyze(self) -> dict[str, float]:
        """Compute convergence metrics over the sliding window."""
        if not self.losses:
            return {}

        loss_arr = np.array(self.losses)
        n = len(loss_arr)

        # 1. Slope (linear fit)
        if n >= 2:
            x = np.arange(n)
            slope, _ = np.polyfit(x, loss_arr, 1)
        else:
            slope = 0.0

        # 2. Rolling stats
        rolling_var = float(np.var(loss_arr)) if n >= 1 else 0.0
        rolling_std = float(np.std(loss_arr)) if n >= 1 else 0.0
        moving_best = float(np.min(loss_arr))
        moving_worst = float(np.max(loss_arr))

        # 3. Improvement rates
        improvement_pct = 0.0
        loss_reduction_pct = 0.0
        if self.initial_loss and self.initial_loss > 0:
            loss_reduction_pct = ((self.initial_loss - loss_arr[-1]) / self.initial_loss) * 100.0

        if n >= 10:
            avg_early = np.mean(loss_arr[:max(1, n//5)])
            avg_late = np.mean(loss_arr[-max(1, n//5):])
            if avg_early > 0:
                improvement_pct = ((avg_early - avg_late) / avg_early) * 100.0

        # 4. Tokens until plateau estimate
        # Simple heuristic: how many tokens at current rate until std falls below threshold
        tokens_until_plateau = -1.0
        if slope < 0 and rolling_std > 1e-4:
            # steps = (target_std - current_std) / slope
            # For estimate, we say tokens to drop loss by 1.0
            steps_needed = max(0.0, -1.0 / slope)
            tokens_until_plateau = steps_needed * np.mean(self.tokens_processed)

        return {
            "convergence/loss_slope": slope,
            "convergence/rolling_variance": rolling_var,
            "convergence/rolling_std": rolling_std,
            "convergence/moving_best": moving_best,
            "convergence/moving_worst": moving_worst,
            "convergence/improvement_pct": improvement_pct,
            "convergence/loss_reduction_pct": loss_reduction_pct,
            "convergence/tokens_until_plateau": tokens_until_plateau,
        }

    def compute_throughput(
        self,
        model: nn.Module,
        batch_size: int,
        seq_len: int,
        last_step_time: float,
    ) -> dict[str, float]:
        """Compute training throughput and hardware resource metrics."""
        elapsed = last_step_time
        if elapsed <= 0:
            elapsed = 1e-6

        samples_per_sec = batch_size / elapsed
        bytes_per_sec = (batch_size * seq_len) / elapsed
        tokens_per_sec = bytes_per_sec  # since raw bytes are tokens

        # Estimate patches/sec if patcher is present
        patches_per_sec = 0.0
        if hasattr(model, "patcher") and self.patches_processed:
            # Use actual processed patches from the batch
            patches_per_sec = self.patches_processed[-1] / elapsed
        else:
            # Fallback based on typical patch size config
            avg_patch_len = 4.0
            if hasattr(model, "config") and hasattr(model.config.model, "blt"):
                avg_patch_len = (model.config.model.blt.patch_size_min + model.config.model.blt.patch_size_max) / 2.0
            patches_per_sec = (batch_size * (seq_len / avg_patch_len)) / elapsed

        # Compute parameter count
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Heuristic FLOPs/sec: 6 * parameters * tokens_per_sec (standard for training)
        flops_per_sec = 6.0 * param_count * tokens_per_sec

        # CPU/GPU utilization
        cpu_util = 0.0
        gpu_util = 0.0

        if _PSUTIL_AVAILABLE:
            cpu_util = psutil.cpu_percent()

        if torch.cuda.is_available():
            # CUDA device utilization
            try:
                gpu_util = float(torch.cuda.utilization())
            except Exception:
                gpu_util = 0.0

        return {
            "performance/samples_per_sec": samples_per_sec,
            "performance/bytes_per_sec": bytes_per_sec,
            "performance/tokens_per_sec": tokens_per_sec,
            "performance/patches_per_sec": patches_per_sec,
            "performance/flops_per_sec": flops_per_sec,
            "performance/cpu_utilization_pct": cpu_util,
            "performance/gpu_utilization_pct": gpu_util,
            "performance/wall_clock_seconds": time.perf_counter() - self.start_time,
        }
