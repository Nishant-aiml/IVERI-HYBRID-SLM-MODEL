# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Dataset mixing and curriculum learning utilities.

Supports weighted random sampling, temperature-scaled sampling, round-robin,
and dynamic curriculum scheduling to interpolate dataset mix weights over step progress.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class MixingStrategy(Enum):
    """Enumeration of available mixing strategies."""

    WEIGHTED_RANDOM = "weighted_random"
    ROUND_ROBIN = "round_robin"
    TEMPERATURE = "temperature"
    CURRICULUM = "curriculum"


STAGE1_MIXING_WEIGHTS = {
    "tinystories": 0.05,
    "fineweb_edu": 0.35,
    "dclm_baseline": 0.25,
    "wikipedia": 0.10,
    "finemath": 0.10,
    "the_stack_v2_python": 0.15,
}

STAGE2_MIXING_WEIGHTS = {
    "magpie_pro": 0.30,
    "tulu3_sft": 0.25,
    "openhermes": 0.20,
    "wildchat": 0.10,
    "code_feedback": 0.10,
    "numinamath": 0.05,
}

STAGE3A_MIXING_WEIGHTS = {
    "the_stack_v2_deep": 0.30,
    "nemotron_competitive": 0.20,
    "leetcode": 0.20,
    "opencode_instruct": 0.15,
    "codeforces": 0.15,
}


class DatasetMixer:
    """Combines multiple datasets based on weights and strategies."""

    def __init__(
        self,
        strategy: MixingStrategy = MixingStrategy.WEIGHTED_RANDOM,
        temperature: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.strategy = strategy
        self.temperature = temperature
        self.seed = seed
        self.schedule_fn: Callable[[int], dict[str, float]] | None = None

    def validate_weights(self, weights: dict[str, float]) -> bool:
        """Validate that mixing weights are positive and sum close to 1.0."""
        if not weights:
            return False
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-4:
            logger.warning(f"Mixing weights sum to {total}, expected 1.0. Readjusting.")
        return all(w >= 0.0 for w in weights.values())

    def _normalize(self, weights: dict[str, float]) -> dict[str, float]:
        """Normalize weights to sum exactly to 1.0."""
        total = sum(weights.values())
        if total == 0:
            return {k: 1.0 / len(weights) for k in weights}
        return {k: v / total for k, v in weights.items()}

    def mix(
        self,
        datasets: dict[str, list[Any]],
        weights: dict[str, float],
        total_samples: int,
        seed: int | None = None,
    ) -> list[Any]:
        """Combine datasets using weighted random sampling."""
        if not datasets:
            return []

        norm_weights = self._normalize(weights)
        self.validate_weights(norm_weights)

        rng = random.Random(seed if seed is not None else self.seed)
        mixed = []

        keys = list(datasets.keys())
        probs = [norm_weights.get(k, 0.0) for k in keys]

        # Filter out empty datasets
        active_indices = [i for i, k in enumerate(keys) if datasets[k]]
        if not active_indices:
            return []

        # Re-normalize probs for active datasets
        active_keys = [keys[i] for i in active_indices]
        active_probs = [probs[i] for i in active_indices]
        total_active_prob = sum(active_probs)
        if total_active_prob == 0:
            active_probs = [1.0 / len(active_keys)] * len(active_keys)
        else:
            active_probs = [p / total_active_prob for p in active_probs]

        # Track read indices per dataset
        read_idx = {k: 0 for k in active_keys}
        # Shuffle individual datasets to ensure random order
        shuffled_ds = {}
        for k in active_keys:
            shuffled_ds[k] = datasets[k].copy()
            rng.shuffle(shuffled_ds[k])

        for _ in range(total_samples):
            # Select dataset key
            k = rng.choices(active_keys, weights=active_probs, k=1)[0]
            ds = shuffled_ds[k]

            idx = read_idx[k]
            mixed.append(ds[idx % len(ds)])
            read_idx[k] += 1

        return mixed

    def weighted_sample(
        self, datasets: dict[str, list[Any]], weights: dict[str, float], n: int
    ) -> list[Any]:
        """Alias for mix."""
        return self.mix(datasets, weights, n)

    def round_robin(self, datasets: dict[str, list[Any]], n_samples: int) -> list[Any]:
        """Combine datasets sequentially one sample at a time."""
        if not datasets:
            return []

        keys = [k for k in datasets if datasets[k]]
        if not keys:
            return []

        mixed = []
        read_idx = {k: 0 for k in keys}

        for i in range(n_samples):
            k = keys[i % len(keys)]
            ds = datasets[k]
            idx = read_idx[k]
            mixed.append(ds[idx % len(ds)])
            read_idx[k] += 1

        return mixed

    def temperature_sample(
        self,
        datasets: dict[str, list[Any]],
        weights: dict[str, float],
        n_samples: int,
        temperature: float = 1.0,
    ) -> list[Any]:
        """Scale weights by temperature to balance small and large datasets."""
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be > 0, got {temperature}")

        # Scale weights: p_i = (w_i ^ (1/T)) / sum(w_j ^ (1/T))
        scaled_weights = {}
        for k, w in weights.items():
            scaled_weights[k] = w ** (1.0 / temperature)

        return self.mix(datasets, scaled_weights, n_samples)

    def curriculum_mix(
        self,
        datasets: dict[str, list[Any]],
        weights_start: dict[str, float],
        weights_end: dict[str, float],
        current_step: int,
        total_steps: int,
        n_samples: int,
    ) -> list[Any]:
        """Linearly interpolate mixing weights between start and end configs."""
        if total_steps <= 0:
            return self.mix(datasets, weights_end, n_samples)

        alpha = min(1.0, max(0.0, current_step / total_steps))

        current_weights = {}
        all_keys = set(weights_start.keys()) | set(weights_end.keys())
        for k in all_keys:
            w_start = weights_start.get(k, 0.0)
            w_end = weights_end.get(k, 0.0)
            current_weights[k] = w_start + alpha * (w_end - w_start)

        return self.mix(datasets, current_weights, n_samples)

    def set_schedule(self, step_fn: Callable[[int], dict[str, float]]) -> None:
        """Register a dynamic callback function to compute weights per training step."""
        self.schedule_fn = step_fn
