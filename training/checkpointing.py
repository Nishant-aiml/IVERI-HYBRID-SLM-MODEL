# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint saving and loading utilities for IVERI CORE.

Manages serialization of model weights, optimizer/scheduler state, mixed precision scaling,
and verification of configuration/architecture versions on loading.
"""

from __future__ import annotations

import pathlib
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from configs.base_config import IVERIConfig
from core.constants import ARCHITECTURE_VERSION, IVERI_VERSION
from core.exceptions import CheckpointError


def save_checkpoint(
    path: str | pathlib.Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    step: int = 0,
    epoch: int = 0,
    metrics: dict[str, Any] | None = None,
    config: IVERIConfig | None = None,
) -> None:
    """Save training checkpoint including model state, optimizer, scheduler, and metadata.

    Args:
        path: Output file path.
        model: PyTorch model.
        optimizer: Optional optimizer to serialize.
        scheduler: Optional learning rate scheduler to serialize.
        scaler: Optional GradScaler to serialize.
        step: Current training step index.
        epoch: Current training epoch index.
        metrics: Dictionary of tracking metrics.
        config: Configuration class dictionary.
    """
    path_obj = pathlib.Path(path)
    # Ensure parent folders exist
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Capture current random seeds for reproducibility
    seeds = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        seeds["torch_cuda"] = torch.cuda.get_rng_state_all()

    checkpoint = {
        "iveri_version": IVERI_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "step": step,
        "epoch": epoch,
        "metrics": metrics or {},
        "config": config.to_dict() if config is not None else None,
        "seeds": seeds,
        # State dicts
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
    }

    # Save to temp file first to prevent corruption during disk interrupts
    temp_path = path_obj.with_suffix(".tmp")
    try:
        torch.save(checkpoint, temp_path)
        if temp_path.exists():
            temp_path.replace(path_obj)
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise CheckpointError(f"Failed to write checkpoint file: {e}") from e


def load_checkpoint(
    path: str | pathlib.Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
) -> dict[str, Any]:
    """Load checkpoint weights and optimizer states with compatibility checking.

    Args:
        path: Path to the checkpoint file.
        model: PyTorch model to populate.
        optimizer: Optional optimizer to populate.
        scheduler: Optional scheduler to populate.
        scaler: Optional mixed precision scaler to populate.

    Returns:
        Dictionary of loaded metadata keys (step, epoch, metrics, config, seeds).
    """
    path_obj = pathlib.Path(path)
    if not path_obj.exists() or not path_obj.is_file():
        raise CheckpointError(f"Checkpoint file not found: {path}")

    try:
        checkpoint = torch.load(path_obj, map_location="cpu", weights_only=False)
    except Exception as e:
        raise CheckpointError(f"Failed to load checkpoint file structure: {e}") from e

    # Compatibility check
    arch = checkpoint.get("architecture_version", "")
    if arch != ARCHITECTURE_VERSION:
        raise CheckpointError(
            f"Checkpoint architecture mismatch! Checkpoint: '{arch}', "
            f"Expected: '{ARCHITECTURE_VERSION}'"
        )

    # Restore model state dict
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except Exception as e:
        raise CheckpointError(f"Model state dictionary mismatch: {e}") from e

    # Restore optimizer state
    if optimizer is not None:
        opt_state = checkpoint.get("optimizer_state_dict")
        if opt_state is not None:
            try:
                optimizer.load_state_dict(opt_state)
            except Exception as e:
                raise CheckpointError(f"Optimizer state dictionary loading failed: {e}") from e

    # Restore scheduler state
    if scheduler is not None:
        sched_state = checkpoint.get("scheduler_state_dict")
        if sched_state is not None:
            try:
                scheduler.load_state_dict(sched_state)
                scheduler.step(scheduler.last_epoch)
            except Exception as e:
                raise CheckpointError(f"Scheduler state loading failed: {e}") from e

    # Restore scaler state
    if scaler is not None:
        scal_state = checkpoint.get("scaler_state_dict")
        if scal_state is not None:
            try:
                scaler.load_state_dict(scal_state)
            except Exception as e:
                raise CheckpointError(f"Scaler state loading failed: {e}") from e

    # Restore random state if found
    seeds = checkpoint.get("seeds")
    if seeds is not None:
        try:
            random.setstate(seeds["python"])
            np.random.set_state(seeds["numpy"])
            torch.random.set_rng_state(seeds["torch"])
            if torch.cuda.is_available() and "torch_cuda" in seeds:
                torch.cuda.set_rng_state_all(seeds["torch_cuda"])
        except Exception:
            # Non-fatal: do not crash if random seed restore fails
            pass

    return {
        "step": checkpoint.get("step", 0),
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
        "config": checkpoint.get("config"),
    }
