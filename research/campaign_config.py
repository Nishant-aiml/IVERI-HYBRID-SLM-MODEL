# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Declarative Campaign Configuration profiles defining experiment scales and sweeps."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class CampaignConfig:
    """Manages validation profiles: verification, pilot, full, and paper."""

    def __init__(self, profile_name: str = "pilot") -> None:
        self.profile_name = profile_name.lower()
        self.profiles = {
            "verification": {
                "max_steps": 100,
                "seeds": [42],
                "models": ["iveri", "transformer"],
                "benchmarks": ["perplexity"],
                "precision": "fp32",
                "checkpoint_interval": 50,
            },
            "pilot": {
                "max_steps": 5000,
                "seeds": [42, 123],
                "models": ["iveri", "transformer", "mamba2"],
                "benchmarks": ["perplexity", "humaneval"],
                "precision": "fp16",
                "checkpoint_interval": 1000,
            },
            "full": {
                "max_steps": 100000,
                "seeds": [42, 123, 3407, 2026, 9999],
                "models": ["iveri", "transformer", "mamba2", "hybrid"],
                "benchmarks": ["perplexity", "humaneval", "mbpp", "gsm8k", "longbench"],
                "precision": "bf16",
                "checkpoint_interval": 5000,
            },
            "paper": {
                "max_steps": 300000,  # Full Phase 5 pretraining steps
                "seeds": [42, 123, 3407, 2026, 9999],
                "models": ["iveri", "transformer", "mamba2", "hybrid"],
                "benchmarks": [
                    "perplexity",
                    "humaneval",
                    "mbpp",
                    "livecodebench",
                    "longbench",
                    "gsm8k",
                    "calibration",
                    "efficiency",
                    "energy",
                    "safety",
                    "ifeval",
                ],
                "precision": "bf16",
                "checkpoint_interval": 10000,  # Required by CostEstimator
            }
        }

        if self.profile_name not in self.profiles:
            logger.warning(f"Unknown campaign profile '{self.profile_name}'. Defaulting to 'pilot'.")
            self.profile_name = "pilot"

    def get_profile(self) -> dict[str, Any]:
        """Fetch declarative hyperparameter dictionary of selected profile."""
        return self.profiles[self.profile_name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "settings": self.get_profile(),
        }
