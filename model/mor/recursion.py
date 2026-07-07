# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Mixture of Recursions (MoR) Recursion Engine.

Implements the recursion loop controller that recursively applies modular block
layers to latent patch elements based on assigned recursion depths.
"""

from __future__ import annotations

import inspect
import typing

import torch
from torch import nn

from configs.base_config import IVERIConfig
from core.interfaces import BaseModule
from core.registry import register
from utils.validation import validate_shape


@register("recursion_engine")
class RecursionEngine(BaseModule):
    """Recursion loop controller for Mixture of Recursions.

    Wraps a core model block (or block sequence) and repeatedly applies it
    to sequence patch elements. Computes active sequence masks per step to allow
    completed tokens to bypass execution, saving compute and VRAM.
    """

    stats: dict[str, typing.Any]

    def __init__(self, block: nn.Module, config: IVERIConfig) -> None:
        """Initialize the recursion engine.

        Args:
            block: The module layer to run recursively.
            config: Configuration object containing model parameters.
        """
        super().__init__()
        self.block = block
        self.config = config
        self.max_depth = config.model.max_recursion_depth
        self.hidden_dim = config.model.hidden_dim

        # Telemetry statistics counters
        self.reset_statistics()

    def reset_statistics(self) -> None:
        """Clear all collected recursion telemetry statistics."""
        self.stats = {
            "total_calls": 0,
            "total_patches": 0,
            "total_computations_run": 0,
            "total_computations_skipped": 0,
            "depth_histogram": {d: 0 for d in range(1, self.max_depth + 1)},
        }

    def get_statistics(self) -> dict[str, object]:
        """Compile and return current recursion statistics.

        Returns:
            A dictionary containing compiled statistics:
                average_depth: Mean recursion depth assigned.
                skipped_pct: Percentage of computations skipped.
                depth_histogram: Frequency of each recursion depth.
                max_depth_frequency: Percentage of patches reaching max depth.
        """
        total_patches = self.stats["total_patches"]
        if total_patches == 0:
            return {
                "average_depth": 0.0,
                "skipped_pct": 0.0,
                "depth_histogram": self.stats["depth_histogram"].copy(),
                "max_depth_frequency": 0.0,
            }

        total_runs = self.stats["total_computations_run"]
        total_skips = self.stats["total_computations_skipped"]
        total_comp_slots = total_runs + total_skips
        skipped_pct = (total_skips / total_comp_slots) * 100.0 if total_comp_slots > 0 else 0.0

        # Calculate average depth from histogram
        depth_sum = sum(d * count for d, count in self.stats["depth_histogram"].items())
        average_depth = depth_sum / total_patches

        max_depth_count = self.stats["depth_histogram"].get(self.max_depth, 0)
        max_depth_frequency = (max_depth_count / total_patches) * 100.0

        return {
            "average_depth": average_depth,
            "skipped_pct": skipped_pct,
            "depth_histogram": self.stats["depth_histogram"].copy(),
            "max_depth_frequency": max_depth_frequency,
        }

    def _update_telemetry(self, depths: torch.Tensor, batch_size: int, seq_len: int) -> None:
        """Update telemetry stats based on the batch depths.

        Args:
            depths: Assigned depths tensor of shape (B, P) or (B, P, 1).
            batch_size: Batch size of input.
            seq_len: Patch sequence length of input.
        """
        # Detach and move to CPU for stats tracking
        depths_cpu = depths.detach().cpu()

        self.stats["total_calls"] += 1
        self.stats["total_patches"] += batch_size * seq_len

        # Compute histogram values
        for d in range(1, self.max_depth + 1):
            count = int((depths_cpu == d).sum().item())
            self.stats["depth_histogram"][d] += count

            # Computations executed: elements with depth > step (step from 0 to max_depth-1)
            # For a given depth d: it undergoes exactly d executions and (max_depth - d) bypasses
            self.stats["total_computations_run"] += count * d
            self.stats["total_computations_skipped"] += count * (self.max_depth - d)

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Execute recursive forward pass over the hidden representation x.

        Args:
            x: Input patched representations of shape (B, P, D).
            **kwargs: Extra arguments including:
                depths (torch.Tensor): Recursion depth assignments of shape (B, P) or (B, P, 1).

        Returns:
            Updated representation tensor of shape (B, P, D).
        """
        validate_shape(x, (-1, -1, self.hidden_dim), name="x")

        # Retrieve and validate depths
        depths_val = kwargs.get("depths")
        if depths_val is None or not isinstance(depths_val, torch.Tensor):
            raise ValueError("depths must be provided as a torch.Tensor in kwargs")

        depths: torch.Tensor = depths_val

        # Standardize depths shape to (B, P)
        if depths.ndim == 3:
            depths = depths.squeeze(-1)
        validate_shape(depths, (x.shape[0], x.shape[1]), name="depths")

        # Update stats
        self._update_telemetry(depths, x.shape[0], x.shape[1])

        # Prepare block arguments; depths are consumed by the engine, entropy flows to MoE.
        block_kwargs = dict(kwargs)
        block_kwargs.pop("depths", None)

        # Inspect if block's forward method accepts active_mask directly
        sig = inspect.signature(self.block.forward)
        block_accepts_mask = "active_mask" in sig.parameters

        for step in range(self.max_depth):
            # active_mask is True for patches that require further computation
            active_mask = depths > step  # (B, P)

            # If all elements are inactive, we can stop the recursion loop early
            if not active_mask.any():
                break

            if block_accepts_mask:
                # Target block handles masking internally (e.g. Mamba2/Attention cache filtration)
                x_next = self.block(x, active_mask=active_mask, **block_kwargs)
            else:
                # External Masking: execute block and conditionally update active indices
                x_next = self.block(x, **block_kwargs)
                x_next = torch.where(active_mask.unsqueeze(-1), x_next, x)

            # Preserve gradients but apply bypassing
            x = x_next

        validate_shape(x, (depths.shape[0], depths.shape[1], self.hidden_dim), name="output x")
        return x
