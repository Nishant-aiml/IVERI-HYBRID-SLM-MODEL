# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Evaluation pipeline and benchmark infrastructure package exports (Phase 2.5)."""

from __future__ import annotations

from evaluation.arch_eval import ArchitectureEvaluator
from evaluation.benchmark import InferenceBenchmark
from evaluation.checkpoint_compare import CheckpointComparator
from evaluation.distributed_evaluator import DistributedEvaluator
from evaluation.evaluator import Evaluator
from evaluation.generation import GenerationEvaluator
from evaluation.memory_tracker import MemoryTracker
from evaluation.perplexity import PerplexityEvaluator
from evaluation.report_generator import ReportGenerator
from evaluation.sft_evaluator import SFTEvaluator
from evaluation.prompt_suite import PromptSuite, EvalPrompt
from evaluation.response_inspector import ResponseInspector, InspectionResult
from evaluation.coding_evaluator import CodingEvaluator
from evaluation.coding_prompt_suite import CodingPromptSuite, CodeEvalPrompt
from evaluation.code_inspector import CodeInspector, CodeInspectionResult
from evaluation.instruction_retention import InstructionRetentionEvaluator
from evaluation.humaneval_benchmark import HumanEvalBenchmark
from evaluation.mbpp_benchmark import MBPPBenchmark
from evaluation.code_execution import CodeExecutor, CodeExecutionResult
from evaluation.code_quality_analyzer import CodeQualityAnalyzer, CodeQualityResult
from evaluation.security_scanner import SecurityScanner, SecurityScanResult, SecurityBatchResult
from evaluation.contamination_checker import ContaminationChecker, ContaminationReport
from evaluation.alignment_evaluator import AlignmentEvaluator
from evaluation.alignment_prompt_suite import AlignmentPromptSuite, AlignmentEvalPrompt
from evaluation.alignment_inspector import AlignmentInspector, AlignmentInspectionResult
from evaluation.preference_benchmark import PreferenceBenchmarkRunner

__all__ = [
    "Evaluator",
    "DistributedEvaluator",
    "PerplexityEvaluator",
    "GenerationEvaluator",
    "InferenceBenchmark",
    "MemoryTracker",
    "ArchitectureEvaluator",
    "CheckpointComparator",
    "ReportGenerator",
    # Phase 3.2 — SFT Evaluation
    "SFTEvaluator",
    "PromptSuite",
    "EvalPrompt",
    "ResponseInspector",
    "InspectionResult",
    # Phase 3.3 — Coding Specialization
    "CodingEvaluator",
    "CodingPromptSuite",
    "CodeEvalPrompt",
    "CodeInspector",
    "CodeInspectionResult",
    "InstructionRetentionEvaluator",
    "HumanEvalBenchmark",
    "MBPPBenchmark",
    "CodeExecutor",
    "CodeExecutionResult",
    "CodeQualityAnalyzer",
    "CodeQualityResult",
    "SecurityScanner",
    "SecurityScanResult",
    "SecurityBatchResult",
    "ContaminationChecker",
    "ContaminationReport",
    # Phase 3.4 — Preference Alignment
    "AlignmentEvaluator",
    "AlignmentPromptSuite",
    "AlignmentEvalPrompt",
    "AlignmentInspector",
    "AlignmentInspectionResult",
    "PreferenceBenchmarkRunner",
]


