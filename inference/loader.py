# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint and model loading for inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from configs.base_config import IVERIConfig, get_base_config
from model.iveri_core import IVERIModel

logger = logging.getLogger(__name__)


def load_inference_model(
    checkpoint_path: str | Path | None = None,
    *,
    config: IVERIConfig | None = None,
    device: str | None = None,
    dtype: torch.dtype | None = None,
) -> IVERIModel:
    """Load IVERIModel for inference from optional checkpoint."""
    cfg = config or get_base_config()
    if device is not None:
        cfg.hardware.device = device

    model = IVERIModel(cfg)
    target_device = torch.device(cfg.hardware.device)

    if dtype is None:
        if cfg.hardware.mixed_precision == "bf16":
            dtype = torch.bfloat16
        elif cfg.hardware.mixed_precision == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32

    model.to(device=target_device)
    model.eval()

    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        meta = model.load_checkpoint(path)
        logger.info("Loaded checkpoint %s (step=%s)", path, meta.get("step"))

    return model
