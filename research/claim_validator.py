# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Claim Validator and Research Integrity scorecard module."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ClaimValidator:
    """Classifies claims and computes Reproducibility and Research Integrity scorecards."""

    def __init__(self) -> None:
        pass

    def validate_claim(
        self,
        claim_description: str,
        p_value: float | None,
        delta_pct: float,
        num_seeds: int,
    ) -> str:
        """Classify claims based on significance testing and metrics logs.

        Returns one of: "SUPPORTED", "LIKELY", "HYPOTHESIS", "UNVERIFIED", or "REFUTED".
        """
        # If the metric performance degrades significantly, the claim is REFUTED
        if delta_pct < -0.05:
            return "REFUTED"

        # If no statistical testing was run, the statement is classified as UNVERIFIED or HYPOTHESIS
        if p_value is None:
            if abs(delta_pct) < 1e-9:
                return "HYPOTHESIS"
            return "UNVERIFIED"

        if num_seeds >= 5 and p_value < 0.05 and delta_pct > 0.0:
            return "SUPPORTED"
        elif p_value < 0.10 and delta_pct > 0.0:
            return "LIKELY"
        elif p_value >= 0.10:
            if delta_pct <= 0.0:
                return "REFUTED"
            return "HYPOTHESIS"
        
        if delta_pct <= 0.0:
            return "REFUTED"
        return "UNVERIFIED"

    def calculate_reproducibility_score(
        self,
        git_sha: str,
        config_hash: str,
        seed_count: int,
        checksums_ok: bool,
        env_captured: bool,
    ) -> float:
        """Compute the Reproducibility Score (0-100%).

        Weights:
        - env_captured: 20%
        - git_sha presence: 20%
        - config_hash presence: 20%
        - seed_count matches requirement (>=5): 20%
        - dataset checksums verified: 20%
        """
        score = 0.0
        if env_captured:
            score += 20.0
        if git_sha and git_sha != "unknown":
            score += 20.0
        if config_hash and config_hash != "unknown":
            score += 20.0
        if seed_count >= 5:
            score += 20.0
        elif seed_count > 0:
            score += 10.0
        if checksums_ok:
            score += 20.0

        return score

    def calculate_research_integrity_score(
        self,
        baseline_coverage_ok: bool,
        completed_ablations: int,
        total_ablations: int,
        calibration_completed: bool,
        seeds_run: int,
        statistical_significance_run: bool,
    ) -> float:
        """Compute the Research Integrity Score (0-100%).

        Weights:
        - baseline_coverage_ok (comparing 3 baselines): 25%
        - completed_ablations fraction: 25%
        - calibration_completed check: 20%
        - seeds_run (z-scale >=5): 15%
        - statistical_significance_run check: 15%
        """
        score = 0.0
        if baseline_coverage_ok:
            score += 25.0

        ablation_ratio = (completed_ablations / total_ablations) if total_ablations > 0 else 0.0
        score += 25.0 * min(1.0, ablation_ratio)

        if calibration_completed:
            score += 20.0
        if seeds_run >= 5:
            score += 15.0
        elif seeds_run > 0:
            score += 7.5
        if statistical_significance_run:
            score += 15.0

        return score
