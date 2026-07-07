# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Loss and numerical health monitoring infrastructure for IVERI CORE pretraining.

Tracks loss convergence, gradient health (norms, variance, NaNs), and activation
health (mean, std, dead units) across key modules without early stopping.
"""

from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LossMonitor:
    """Monitors loss trends, gradient health, and activation distributions."""

    def __init__(self, config: Any, log_dir: str | Path = "logs") -> None:
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "gradient_health.csv"

        # Tracking states
        self.loss_history: list[float] = []
        self.val_loss_history: list[float] = []
        self.ema_loss: float | None = None
        self.ema_alpha = 0.1
        self.best_loss = float("inf")

        # Activation health storage (reset every step)
        self.activation_stats: dict[str, dict[str, float]] = {}
        self._hooks: list[Any] = []

    def register_activation_hooks(self, model: nn.Module) -> None:
        """Register forward hooks on BLT, Mamba, Attention, MoE, and Titans submodules."""
        self.remove_hooks()
        self.activation_stats.clear()

        def make_hook(name: str):
            def hook(module: nn.Module, inp: Any, out: Any) -> None:
                # Handle tuple/dict outputs
                if isinstance(out, tuple):
                    tensor = out[0]
                elif isinstance(out, dict):
                    tensor = out.get("logits", next(iter(out.values())))
                else:
                    tensor = out

                if not isinstance(tensor, torch.Tensor):
                    return

                t_flat = tensor.detach().float()
                # Compute stats
                mean_val = t_flat.mean().item()
                std_val = t_flat.std().item()
                max_val = t_flat.max().item()
                min_val = t_flat.min().item()
                nan_count = torch.isnan(t_flat).sum().item()
                total_els = t_flat.numel()

                # Dead activations: check how many elements are exactly 0.0 or near zero
                dead_count = (t_flat.abs() < 1e-6).sum().item()
                dead_ratio = dead_count / max(1, total_els)

                self.activation_stats[name] = {
                    "mean": mean_val,
                    "std": std_val,
                    "max": max_val,
                    "min": min_val,
                    "nans": float(nan_count),
                    "dead_ratio": dead_ratio,
                }
            return hook

        # Traverse model and attach hooks to matching names
        for name, module in model.named_modules():
            cls_name = module.__class__.__name__.lower()
            # Target BLT, Mamba, Attention, MoE, and Titans
            is_target = any(
                target in cls_name
                for target in ["blt", "mamba", "attention", "moe", "titans", "entropy"]
            )
            if is_target:
                h = module.register_forward_hook(make_hook(name))
                self._hooks.append(h)

        logger.info(f"Registered {len(self._hooks)} activation health hooks.")

    def remove_hooks(self) -> None:
        """Remove all registered activation hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def update_loss(self, loss: float, is_val: bool = False) -> dict[str, Any]:
        """Update loss metrics and check for anomalies."""
        if math.isnan(loss) or math.isinf(loss):
            logger.error("[LossMonitor] CRITICAL WARNING: Loss is NaN or Inf!")

        metrics: dict[str, Any] = {}
        if is_val:
            self.val_loss_history.append(loss)
            if loss < self.best_loss:
                self.best_loss = loss
        else:
            self.loss_history.append(loss)
            if self.ema_loss is None:
                self.ema_loss = loss
            else:
                self.ema_loss = self.ema_alpha * loss + (1 - self.ema_alpha) * self.ema_loss

            # Check for loss spikes: sudden jump > 2.5x EMA loss
            if len(self.loss_history) > 10 and loss > 2.5 * self.ema_loss:
                logger.warning(
                    f"[LossMonitor] WARNING: Loss spike detected! Current: {loss:.4f}, EMA: {self.ema_loss:.4f}"
                )

            # Check for plateaus: loss std-dev of last 50 steps is very small (< 0.005)
            if len(self.loss_history) >= 50:
                recent = np.array(self.loss_history[-50:])
                std_recent = float(np.std(recent))
                if std_recent < 0.005:
                    logger.warning(
                        f"[LossMonitor] WARNING: Loss plateau detected (recent std: {std_recent:.5f})."
                    )

        return metrics

    def track_gradient_health(self, model: nn.Module, step: int) -> dict[str, float]:
        """Check all gradients for NaNs, extreme values, and compute norms."""
        grad_norms = []
        grad_maxs = []
        grad_mins = []
        total_grads = 0
        zero_grads = 0
        nan_grads = 0
        grad_vals = []

        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                g = p.grad.detach().float()
                total_grads += g.numel()
                g_norm = g.norm(2).item()
                grad_norms.append(g_norm)
                grad_maxs.append(g.max().item())
                grad_mins.append(g.min().item())

                # Zero gradients
                zero_grads += (g == 0.0).sum().item()

                # NaNs
                nans = torch.isnan(g).sum().item()
                nan_grads += nans

                # Save a subset of grads to compute variance
                if g.numel() > 0:
                    grad_vals.append(g.view(-1)[:min(100, g.numel())].cpu().numpy())

        # Combine sampled grads for variance
        if grad_vals:
            all_sampled = np.concatenate(grad_vals)
            grad_var = float(np.var(all_sampled))
        else:
            grad_var = 0.0

        global_grad_norm = math.sqrt(sum(n**2 for n in grad_norms)) if grad_norms else 0.0
        max_grad = max(grad_maxs) if grad_maxs else 0.0
        min_grad = min(grad_mins) if grad_mins else 0.0

        # Gradient explosion warning
        if global_grad_norm > 15.0:
            logger.warning(
                f"[LossMonitor] WARNING: High gradient norm detected! Global Norm: {global_grad_norm:.4f}"
            )

        # NaN gradient warning
        if nan_grads > 0:
            logger.error(f"[LossMonitor] CRITICAL WARNING: {nan_grads} NaN gradient elements detected!")

        # Compile health report
        health = {
            "step": float(step),
            "global_grad_norm": global_grad_norm,
            "max_grad": max_grad,
            "min_grad": min_grad,
            "zero_grad_ratio": zero_grads / max(1, total_grads),
            "nan_grad_count": float(nan_grads),
            "grad_variance": grad_var,
        }

        # Write to gradient health CSV
        self._write_health_csv(health)

        return health

    def _write_health_csv(self, health: dict[str, float]) -> None:
        """Append health statistics to gradient_health.csv."""
        flat = {k: f"{v:.6f}" for k, v in health.items()}
        file_exists = self.csv_path.exists()
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(flat.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(flat)
