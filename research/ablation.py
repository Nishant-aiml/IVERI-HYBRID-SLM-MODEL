# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Ablation Suite configuration for Stage 5 structural evaluations."""

from __future__ import annotations

import logging
from typing import Any

import torch.nn as nn

from configs.base_config import IVERIConfig
from research.baselines import BaselineManager

logger = logging.getLogger(__name__)


class AblationSuite:
    """Manages structural ablated variants of the IVERI CORE architecture.

    Uses ``ModelConfig`` boolean gates to physically remove components from the
    forward path while preserving the full model when all flags are True.
    """

    def __init__(self, config: IVERIConfig) -> None:
        self.config = config
        self.baseline_manager = BaselineManager(config)
        self.components = [
            "full",
            "no_titans",
            "no_mor",
            "no_moe",
            "no_blt",
            "no_entropy_routing",
        ]

    def get_ablated_model(self, component: str) -> nn.Module:
        """Instantiate and return the ablated model model block.

        Args:
            component: Name of the ablation component.

        Returns:
            nn.Module: Model instance.
        """
        comp_lower = component.lower()
        if comp_lower == "full":
            from model.iveri_core import IVERIModel
            return IVERIModel(self.config)

        if comp_lower in {"no_titans", "titans"}:
            return self.baseline_manager.build_ablated_variant("titans")
        if comp_lower in {"no_mor", "mor"}:
            return self.baseline_manager.build_ablated_variant("mor")
        if comp_lower in {"no_moe", "moe"}:
            return self.baseline_manager.build_ablated_variant("moe")
        if comp_lower in {"no_blt", "blt"}:
            return self.baseline_manager.build_ablated_variant("blt")
        if comp_lower in {"no_entropy_routing", "entropy_routing"}:
            return self.baseline_manager.build_ablated_variant("entropy_routing")

        raise ValueError(f"Unsupported ablation target component: {component}")

    def run_ablation_evaluation(
        self,
        component: str,
        eval_fn: Any,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Orchestrate evaluation of a specific ablated model variant.

        Args:
            component: Component name to evaluate.
            eval_fn: Callback function to execute evaluation.
            *args: Position arguments passed to eval_fn.
            **kwargs: Keyword arguments passed to eval_fn.

        Returns:
            dict[str, Any]: Extracted evaluation metrics.
        """
        logger.info(f"Setting up ablation target: {component}...")
        model = self.get_ablated_model(component)
        # Execute validation callback passing ablated model as first argument
        metrics = eval_fn(model, *args, **kwargs)
        logger.info(f"Completed ablation evaluation for {component}.")
        return metrics
