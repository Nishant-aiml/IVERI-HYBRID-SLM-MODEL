# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Execution Backend abstraction auto-tuning training hyperparameters per hardware tier."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ExecutionBackend:
    """Configures hardware resource limits and optimization overrides for various nodes."""

    def __init__(self, backend_name: str = "local") -> None:
        self.backend_name = backend_name.lower()
        self.backends = {
            "local": {
                "num_workers": 0,
                "pin_memory": False,
                "gradient_accumulation": 8,
                "precision": "fp32",
                "checkpoint_interval_steps": 1000,
            },
            "rtx3050": {
                "num_workers": 0,
                "pin_memory": False,
                "gradient_accumulation": 16,  # accumulate to fit within 4/8GB VRAM
                "precision": "fp16",
                "checkpoint_interval_steps": 500,
            },
            "colab": {
                "num_workers": 2,
                "pin_memory": True,
                "gradient_accumulation": 4,
                "precision": "fp16",
                "checkpoint_interval_steps": 1000,
            },
            "kaggle": {
                "num_workers": 4,
                "pin_memory": True,
                "gradient_accumulation": 4,
                "precision": "bf16",
                "checkpoint_interval_steps": 2000,
            },
            "vast": {
                "num_workers": 8,
                "pin_memory": True,
                "gradient_accumulation": 2,
                "precision": "bf16",
                "checkpoint_interval_steps": 5000,
            },
            "lambda": {
                "num_workers": 8,
                "pin_memory": True,
                "gradient_accumulation": 2,
                "precision": "bf16",
                "checkpoint_interval_steps": 5000,
            }
        }

        if self.backend_name not in self.backends:
            logger.warning(f"Unknown execution backend '{self.backend_name}'. Defaulting to 'local'.")
            self.backend_name = "local"

    def get_hardware_overrides(self) -> dict[str, Any]:
        """Fetch backend-specific hardware settings overrides."""
        return self.backends[self.backend_name]
