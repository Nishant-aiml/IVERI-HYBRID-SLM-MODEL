# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Instability and divergence diagnostics tracker for IVERI CORE training.

Hooks into backbone block activations and parameters to log hidden state and
gradient norms, detecting anomalies/NaNs before divergence occurs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DivergenceError(ValueError):
    """Raised when training instability or parameter divergence is detected."""
    pass


class InstabilityTracker:
    """Monitors hidden states and gradients to detect training divergence."""

    def __init__(self, model: nn.Module, log_dir: str | Path, threshold: float = 1e4) -> None:
        """Initialize the InstabilityTracker.

        Args:
            model: Model to monitor.
            log_dir: Directory where diagnostics logs will be written.
            threshold: Value threshold for triggering divergence error.
        """
        self.model = model
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold

        self.diagnostics_file = self.log_dir / "debug_diagnostics.json"
        self.report_file = self.log_dir / "divergence_report.json"

        # Internal state tracking
        self.history: list[dict[str, Any]] = []
        self._forward_norms: dict[str, float] = {}
        self._hooks: list[Any] = []

        # Register forward hooks on backbone blocks
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Register forward hooks on BackboneBlocks to monitor hidden state norms."""
        # Find the backbone blocks
        backbone = getattr(self.model, "backbone", None)
        if backbone is not None and hasattr(backbone, "blocks"):
            for idx, block in enumerate(backbone.blocks):
                name = f"backbone.blocks.{idx}"
                hook = block.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)
                logger.debug(f"Registered instability tracking forward hook on {name}")

    def _make_hook(self, name: str) -> Any:
        """Helper to create a hook closure with block name reference."""
        def hook_fn(module: nn.Module, inputs: Any, output: torch.Tensor) -> None:
            if isinstance(output, torch.Tensor):
                # Compute activation L2 norm safely on CPU
                norm_val = output.detach().pow(2).sum(-1).sqrt().mean().item()
                self._forward_norms[name] = norm_val
        return hook_fn

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def step(self, step_idx: int) -> dict[str, Any]:
        """Collect layer norms and gradient norms for the current step.

        Args:
            step_idx: Global training step index.

        Returns:
            Dictionary of diagnostics metrics.
        """
        step_data: dict[str, Any] = {
            "step": step_idx,
            "hidden_state_norms": dict(self._forward_norms),
            "gradient_norms": {},
            "warnings": []
        }

        # Check hidden state norms for NaN/Inf/Divergence
        for name, norm in step_data["hidden_state_norms"].items():
            if not torch.isfinite(torch.tensor(norm)):
                msg = f"NaN/Inf hidden state norm detected at {name}: value={norm}"
                step_data["warnings"].append(msg)
                logger.warning(msg)
            elif norm > self.threshold:
                msg = f"Hidden state norm exceeded threshold ({self.threshold}) at {name}: value={norm:.2f}"
                step_data["warnings"].append(msg)
                logger.warning(msg)

        # Collect parameter gradient norms
        for param_name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.detach().pow(2).sum().sqrt().item()
                step_data["gradient_norms"][param_name] = grad_norm

                if not torch.isfinite(torch.tensor(grad_norm)):
                    msg = f"NaN/Inf gradient norm detected at parameter {param_name}"
                    step_data["warnings"].append(msg)
                    logger.warning(msg)
                elif grad_norm > self.threshold:
                    msg = f"Gradient norm exceeded threshold ({self.threshold}) at {param_name}: value={grad_norm:.2f}"
                    step_data["warnings"].append(msg)
                    logger.warning(msg)

        self.history.append(step_data)
        
        # Keep memory size bounded to last 1000 steps
        if len(self.history) > 1000:
            self.history.pop(0)

        # Dump history to diagnostics file
        try:
            with open(self.diagnostics_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to write instability diagnostics: {e}")

        # Check if we should trigger a divergence crash
        if step_data["warnings"]:
            self._write_divergence_report(step_data)
            raise DivergenceError(
                f"Training instability detected at step {step_idx}. "
                f"Details saved to {self.report_file}. Warnings: {step_data['warnings']}"
            )

        return step_data

    def _write_divergence_report(self, step_data: dict[str, Any]) -> None:
        """Write a detailed diagnostic report prior to training termination."""
        report = {
            "title": "IVERI CORE Training Divergence Report",
            "step": step_data["step"],
            "primary_warnings": step_data["warnings"],
            "layer_norms": step_data["hidden_state_norms"],
            "gradient_norms": step_data["gradient_norms"],
        }
        try:
            with open(self.report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4)
            logger.info(f"Divergence report written to {self.report_file}")
        except Exception as e:
            logger.error(f"Failed to write divergence report: {e}")
