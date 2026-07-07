# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""IVERI Project Stage 5 Research Validation and Benchmarking infrastructure."""

from __future__ import annotations

from research.baselines import BaselineManager, BaselineMamba2, BaselineHybrid
from research.checkpoint_manager import BaselineCheckpointManager
from research.ablation import AblationSuite
from research.benchmark_research import ResearchBenchmarkRunner
from research.benchmark_engineering import EngineeringBenchmarkRunner
from research.flops import FlopProfiler
from research.profiler import MemoryProfiler
from research.energy_profiler import EnergyProfiler
from research.calibration import ConfidenceCalibrator
from research.scaling import ScalingAnalyzer
from research.statistics import ResearchStatisticalValidator
from research.claim_validator import ClaimValidator
from research.hypothesis import ResearchHypothesisEngine
from research.paper_figures import PaperFigureGenerator
from research.paper_tables import PaperTableGenerator
from research.paper_summary import PaperSummaryGenerator
from research.artifacts import ResearchArtifactsManager

__all__ = [
    "BaselineManager",
    "BaselineMamba2",
    "BaselineHybrid",
    "BaselineCheckpointManager",
    "AblationSuite",
    "ResearchBenchmarkRunner",
    "EngineeringBenchmarkRunner",
    "FlopProfiler",
    "MemoryProfiler",
    "EnergyProfiler",
    "ConfidenceCalibrator",
    "ScalingAnalyzer",
    "ResearchStatisticalValidator",
    "ClaimValidator",
    "ResearchHypothesisEngine",
    "PaperFigureGenerator",
    "PaperTableGenerator",
    "PaperSummaryGenerator",
    "ResearchArtifactsManager",
]
