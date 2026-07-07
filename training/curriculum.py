# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Curriculum scheduler for dataset mixture management in IVERI CORE pretraining.

Orchestrates dataset mixing strategies over training steps, allowing transitions
from simple English stories to complex web text and engineering datasets.
"""

from __future__ import annotations

import logging
from typing import Any

from configs.base_config import IVERIConfig
from data.pipeline.mixer import MixingStrategy

logger = logging.getLogger(__name__)


def _get_val(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class CurriculumScheduler:
    """Controls the active dataset weights and mixture ratios dynamically over steps."""

    def __init__(self, config: IVERIConfig) -> None:
        self.config = config
        data_pipeline = getattr(config, "data_pipeline", {})
        mixing_cfg = _get_val(data_pipeline, "mixing", {})
        self.strategy = _get_val(mixing_cfg, "strategy", "constant")

        # Default weights for Stage 1 datasets
        # For Phase 3.1, only TinyStories is active (1.0), others are 0.0.
        self.base_weights = {
            "tinystories": 1.0,
            "fineweb_edu": 0.0,
            "dclm_baseline": 0.0,
            "wikipedia": 0.0,
            "finemath": 0.0,
            "the_stack_v2_python": 0.0,
        }

    def get_mixture_weights(self, step: int) -> dict[str, float]:
        """Compute the active dataset mixing weights for the current step."""
        if self.strategy == "constant":
            return self.base_weights.copy()

        elif self.strategy == "linear" or self.strategy == "weighted_random":
            # Return configured base weights
            return self.base_weights.copy()

        elif self.strategy == "curriculum":
            # Transition weights over steps
            data_pipeline = getattr(self.config, "data_pipeline", {})
            mixing_cfg = _get_val(data_pipeline, "mixing", {})
            start_step = _get_val(mixing_cfg, "curriculum_start_step", 0)
            end_step = _get_val(mixing_cfg, "curriculum_end_step", 50000)

            if step <= start_step:
                # 100% TinyStories
                return {"tinystories": 1.0, "fineweb_edu": 0.0, "dclm_baseline": 0.0}
            elif step >= end_step:
                # Fully transitioned to target mixture weights (e.g. FineWeb/DCLM)
                # But for this phase (TinyStories-only freeze), we enforce TinyStories=1.0
                return self.base_weights.copy()
            else:
                # Interpolate weights linearly
                fraction = (step - start_step) / (end_step - start_step)
                # We interpolate between 1.0 (start) and base_weights["tinystories"] (end)
                ts_weight = 1.0 - (1.0 - self.base_weights.get("tinystories", 1.0)) * fraction
                weights = {"tinystories": ts_weight}

                # Scale other datasets proportionally
                for name, target_w in self.base_weights.items():
                    if name != "tinystories":
                        weights[name] = target_w * fraction
                return weights

        elif self.strategy == "temperature":
            # Scale weights by temperature
            data_pipeline = getattr(self.config, "data_pipeline", {})
            mixing_cfg = _get_val(data_pipeline, "mixing", {})
            temp = _get_val(mixing_cfg, "temperature", 1.0)
            raw_weights = self.base_weights.copy()
            # Softmax or normalized scaling based on temperature
            sum_w = sum(w ** (1.0 / temp) for w in raw_weights.values() if w > 0)
            if sum_w <= 0:
                return self.base_weights.copy()
            return {k: (v ** (1.0 / temp) / sum_w if v > 0 else 0.0) for k, v in raw_weights.items()}

        else:
            logger.warning(f"Unknown mixing strategy '{self.strategy}'. Falling back to constant.")
            return self.base_weights.copy()
