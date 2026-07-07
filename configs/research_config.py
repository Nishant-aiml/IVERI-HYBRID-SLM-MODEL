# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Configuration parameters for Stage 5 Research Validation and Benchmarking campaign."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from core.exceptions import ConfigError

logger = logging.getLogger(__name__)


@dataclass
class ResearchConfig:
    """Parameters defining the scientific validation, baseline comparisons, and statistical audits."""

    enabled: bool = False
    experiment_name: str = "iveri_research"
    random_seeds: list[int] = field(default_factory=lambda: [42, 123, 3407, 2026, 9999])
    num_runs: int = 5
    compare_transformer: bool = True
    compare_mamba: bool = True
    compare_hybrid: bool = True
    compute_budget: float = 1e12  # FLOP threshold for resource capping
    flop_matching: bool = True
    parameter_matching: bool = True
    report_directory: str = "reports/phase_3_5/"
    save_tables: bool = True
    save_figures: bool = True
    latex_tables: bool = True
    csv_tables: bool = True
    json_results: bool = True
    confidence_interval: float = 0.95
    statistical_tests: list[str] = field(
        default_factory=lambda: ["paired_t", "wilcoxon"]
    )
    profiling_enabled: bool = True

    def __post_init__(self) -> None:
        """Validate input parameter bounds."""
        if not self.random_seeds:
            raise ConfigError("random_seeds list cannot be empty.")
        if self.num_runs <= 0:
            raise ConfigError("num_runs must be a positive integer.")
        if self.confidence_interval <= 0.0 or self.confidence_interval >= 1.0:
            raise ConfigError("confidence_interval must be strictly between 0.0 and 1.0.")
        if self.compute_budget <= 0:
            raise ConfigError("compute_budget must be a positive float.")
