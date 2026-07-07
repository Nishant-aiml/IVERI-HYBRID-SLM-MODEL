# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Mixed precision utilities for IVERI CORE training.

Encapsulates PyTorch Automatic Mixed Precision (AMP) autocasting and gradient scaling
into a unified PrecisionHandler class supporting FP16, BF16, and FP32.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch.amp import GradScaler, autocast


class PrecisionHandler:
    """Manages AMP context selection and gradient scaling for backward passes.

    Supports FP16 (with GradScaler), BF16 (no scaling), and FP32 (no scaling/autocast).
    Handles state dictionary serialization to support checkpoint resume functionality.
    """

    def __init__(
        self,
        precision: str = "fp16",
        device: str = "cuda",
    ) -> None:
        """Initialize the precision handler.

        Args:
            precision: Precision mode, one of 'fp16', 'bf16', 'fp32'.
            device: Target device name (e.g. 'cuda', 'cpu').
        """
        self.precision = precision.lower()
        self.device_type = "cuda" if "cuda" in device else "cpu"

        # Resolve dtype
        if self.precision == "fp16":
            self.dtype = torch.float16
        elif self.precision == "bf16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        # Initialize GradScaler only for CUDA FP16
        self.use_scaler = self.precision == "fp16" and self.device_type == "cuda"
        self.scaler = GradScaler("cuda") if self.use_scaler else None

    def autocast_context(self) -> Any:
        """Return the appropriate autocast context manager for mixed precision.

        Returns:
            A context manager (autocast or nullcontext).
        """
        if self.precision in ("fp16", "bf16") and self.device_type == "cuda":
            return autocast(device_type="cuda", dtype=self.dtype)
        # For CPU BF16 support:
        elif self.precision == "bf16" and self.device_type == "cpu":
            return autocast(device_type="cpu", dtype=torch.bfloat16)
        else:
            return nullcontext()

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale the loss if gradient scaling is enabled.

        Args:
            loss: The computed loss tensor.

        Returns:
            Scaled loss tensor.
        """
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def step_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        max_norm: float | None = None,
    ) -> None:
        """Unscale gradients, optionally clip them, and step the optimizer.

        Args:
            optimizer: The PyTorch optimizer.
            max_norm: Optional value for gradient clipping norm.
        """
        if self.scaler is not None:
            # Unscale gradients before clipping
            self.scaler.unscale_(optimizer)
            if max_norm is not None:
                # Get all parameters that require grad and have grads
                params = []
                for group in optimizer.param_groups:
                    for p in group["params"]:
                        if p.requires_grad and p.grad is not None:
                            params.append(p)
                if params:
                    torch.nn.utils.clip_grad_norm_(params, max_norm)
            # Step scaler
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # Standard step with optional clipping
            if max_norm is not None:
                params = []
                for group in optimizer.param_groups:
                    for p in group["params"]:
                        if p.requires_grad and p.grad is not None:
                            params.append(p)
                if params:
                    torch.nn.utils.clip_grad_norm_(params, max_norm)
            optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        """Get the state dict of the GradScaler.

        Returns:
            Dictionary containing scaler state.
        """
        if self.scaler is not None:
            return self.scaler.state_dict()
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the state dict into the GradScaler.

        Args:
            state_dict: Dictionary containing scaler state.
        """
        if self.scaler is not None and state_dict:
            self.scaler.load_state_dict(state_dict)
