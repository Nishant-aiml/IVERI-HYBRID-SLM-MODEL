# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Phase 3.3 coding specialization pipeline.

Kept in a separate module (following the pattern of :mod:`configs.instruction_config`
and :mod:`configs.distributed_config`) so that base_config.py remains modular.

Examples
--------
>>> from configs.coding_config import CodingConfig
>>> cfg = CodingConfig()
>>> cfg.enabled
False
>>> cfg.generation_temperature
0.2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.exceptions import ConfigError


@dataclass(frozen=False, slots=True)
class CodingConfig:
    """Configuration for the Phase 3.3 coding specialization pipeline.

    All fields default to safe / disabled values so that existing
    pretraining and SFT runs are unaffected when this config is present.

    Attributes
    ----------
    enabled:
        Master switch for the coding pipeline.  ``False`` = disabled (default).
    datasets:
        Ordered list of SFT-format dataset names for coding instruction tuning.
        Each name must match a registry entry in ``data/dataset_specs/coding.yaml``.
    pretrain_dataset:
        Name of the raw-code pretrain dataset (``format: pretrain`` in coding.yaml).
        Trained with full-sequence loss.  Empty string = skip pretrain mix.
    languages:
        Language filter for the code datasets.  Empty list = no filtering.
    curriculum_stages:
        Number of curriculum stages (1–3).  Default 3.
    train_on_prompt:
        If ``False`` (default), loss is computed only on code-response bytes.
    max_turns:
        Maximum dialogue turns per sample.  0 = unlimited.
    code_prefix:
        Byte prefix for code tasks in formatted sequences.
    solution_prefix:
        Byte prefix for code solutions / responses.
    language_prefix:
        Prefix for injected language header.
    include_language_header:
        If ``True``, inject a language-header line before code blocks.
    include_explanation:
        If ``True``, include explanation text after code solutions.
    max_code_bytes:
        Maximum bytes per code sample before truncation.
    evaluation_frequency:
        Evaluate coding metrics every N steps.  0 = use logging.eval_every.
    generation_frequency:
        Run qualitative generation on code prompt suite every N steps.
        0 = end of run only.
    max_new_bytes:
        Maximum new bytes to generate per prompt during evaluation.
    generation_temperature:
        Sampling temperature for code generation.  Lower values (0.2) produce
        more deterministic, syntactically valid code.
    generation_top_k:
        Top-K filter for code generation.  Lower values (20) reduce hallucination.
    sft_checkpoint:
        Path to the Phase 3.2 SFT checkpoint to load before coding fine-tuning.
        Empty string = train from random init (not recommended).
    verification_steps:
        Number of steps for quick verification runs.  Maps to verification_level
        (20 / 100 / 1000 / 100k / 1M) in the coding runner.

    Catastrophic Forgetting (Feedback-#1)
    --------------------------------------
    instruction_retention_enabled:
        If ``True``, run Phase 3.2 PromptSuite after every evaluation to check
        whether instruction-following ability is retained.
    instruction_retention_threshold:
        Maximum allowed perplexity delta (absolute) vs baseline before a
        checkpoint is flagged as having instruction regression.  Default 0.15
        (~15% relative degradation).

    Coding Benchmarks (Feedback-#2)
    --------------------------------
    run_humaneval:
        If ``True``, run HumanEval pass@1 evaluation when benchmarks are invoked.
        Requires ``datasets`` library and network access (or local cache).
    run_mbpp:
        If ``True``, run MBPP pass@1 evaluation when benchmarks are invoked.

    Code Execution (Feedback-#5)
    -----------------------------
    run_execution_eval:
        If ``True``, execute generated Python code in a sandboxed subprocess to
        compute ``compile_success_ratio`` and ``execution_success_ratio``.
    execution_timeout_sec:
        Subprocess timeout in seconds for code execution.

    Contamination (Feedback-#4)
    ----------------------------
    run_contamination_check:
        If ``True``, run the contamination checker at startup to fingerprint-
        compare benchmark prompts against training data files.
    """

    # ── Core switches ──────────────────────────────────────────────────
    enabled: bool = False

    # ── Dataset configuration ──────────────────────────────────────────
    datasets: list[str] = field(
        default_factory=lambda: ["nemotron_competitive", "leetcode"]
    )
    pretrain_dataset: str = "the_stack_v2_deep"
    languages: list[str] = field(
        default_factory=lambda: ["python", "javascript", "cpp", "java", "rust"]
    )

    # ── Curriculum ─────────────────────────────────────────────────────
    curriculum_stages: int = 3

    # ── Training behaviour ─────────────────────────────────────────────
    train_on_prompt: bool = False
    max_turns: int = 0

    # ── Formatting prefixes ────────────────────────────────────────────
    code_prefix: str = "### Code Task:\n"
    solution_prefix: str = "### Solution:\n"
    language_prefix: str = "### Language: "
    include_language_header: bool = True
    include_explanation: bool = True
    max_code_bytes: int = 2048

    # ── Evaluation schedule ────────────────────────────────────────────
    evaluation_frequency: int = 0
    generation_frequency: int = 0

    # ── Generation parameters ──────────────────────────────────────────
    max_new_bytes: int = 256
    generation_temperature: float = 0.2
    generation_top_k: int = 20

    # ── Checkpointing ──────────────────────────────────────────────────
    sft_checkpoint: str = ""
    verification_steps: int = 100

    # ── Catastrophic forgetting (Feedback #1) ─────────────────────────
    instruction_retention_enabled: bool = True
    instruction_retention_threshold: float = 0.15

    # ── Coding benchmarks (Feedback #2) ───────────────────────────────
    run_humaneval: bool = False
    run_mbpp: bool = False

    # ── Code execution (Feedback #5) ──────────────────────────────────
    run_execution_eval: bool = False
    execution_timeout_sec: float = 5.0

    # ── Contamination check (Feedback #4) ─────────────────────────────
    run_contamination_check: bool = True

    def __post_init__(self) -> None:
        if self.curriculum_stages not in (1, 2, 3):
            raise ConfigError(
                f"curriculum_stages must be 1, 2, or 3, got {self.curriculum_stages}"
            )
        if self.max_turns < 0:
            raise ConfigError(f"max_turns must be >= 0, got {self.max_turns}")
        if not (0.0 <= self.generation_temperature <= 10.0):
            raise ConfigError(
                f"generation_temperature must be in [0, 10], "
                f"got {self.generation_temperature}"
            )
        if self.generation_top_k < 0:
            raise ConfigError(
                f"generation_top_k must be >= 0, got {self.generation_top_k}"
            )
        if self.max_new_bytes <= 0:
            raise ConfigError(
                f"max_new_bytes must be > 0, got {self.max_new_bytes}"
            )
        if self.verification_steps <= 0:
            raise ConfigError(
                f"verification_steps must be > 0, got {self.verification_steps}"
            )
        if self.max_code_bytes <= 0:
            raise ConfigError(
                f"max_code_bytes must be > 0, got {self.max_code_bytes}"
            )
        if not (0.0 <= self.instruction_retention_threshold <= 1.0):
            raise ConfigError(
                f"instruction_retention_threshold must be in [0, 1], "
                f"got {self.instruction_retention_threshold}"
            )
        if self.execution_timeout_sec <= 0:
            raise ConfigError(
                f"execution_timeout_sec must be > 0, "
                f"got {self.execution_timeout_sec}"
            )

    def to_formatter_dict(self) -> dict[str, Any]:
        """Convert to a dict compatible with :class:`~training.code_formatter.CodeFormatterConfig`.

        Returns
        -------
        dict[str, Any]
        """
        return {
            "code_prefix": self.code_prefix,
            "solution_prefix": self.solution_prefix,
            "language_prefix": self.language_prefix,
            "include_language_header": self.include_language_header,
            "include_explanation": self.include_explanation,
            "max_turns": self.max_turns,
        }
