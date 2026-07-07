# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Phase 3.2 instruction tuning (SFT) pipeline.

Kept in a separate module (following the pattern of :mod:`configs.distributed_config`
and :mod:`configs.data_pipeline_config`) so that base_config.py remains modular.

Examples
--------
>>> from configs.instruction_config import InstructionConfig
>>> cfg = InstructionConfig()
>>> cfg.enabled
False
>>> cfg.train_on_prompt
False
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.exceptions import ConfigError


@dataclass(frozen=False, slots=True)
class InstructionConfig:
    """Configuration for the SFT instruction tuning pipeline.

    All fields default to safe / disabled values so that existing
    pretraining runs are unaffected when this config is present.

    Attributes
    ----------
    enabled:
        Master switch for the SFT pipeline.  ``False`` = disabled (default).
    datasets:
        Ordered list of dataset names to use for SFT training.
        Each name must match a registry entry in ``data/dataset_specs/instruction.yaml``.
    conversation_template:
        Formatter template style.  One of ``"alpaca"``, ``"chat"``.
    train_on_prompt:
        If ``False`` (default), loss is computed only on assistant response bytes.
        If ``True``, loss is computed on all bytes (including prompt).
    packing:
        If ``True``, pack multiple short samples into a single sequence window.
        Reserved for future use.
    max_turns:
        Maximum number of dialogue turns to retain per sample.  0 = unlimited.
    ignore_prompt_loss:
        Alias for ``not train_on_prompt``.  Kept for API symmetry.
    prompt_prefix:
        Token prepended before user instructions in formatted sequences.
    assistant_prefix:
        Token prepended before assistant responses.
    system_prefix:
        Token prepended before system messages.
    user_prefix:
        Token prepended before user messages in chat format.
    evaluation_frequency:
        Evaluate SFT validation metrics every N steps.  0 = use logging.eval_every.
    generation_frequency:
        Run qualitative generation on prompt suite every N steps.  0 = end of run only.
    max_new_bytes:
        Maximum new bytes to generate per evaluation prompt.
    generation_temperature:
        Sampling temperature for qualitative generation.
    generation_top_k:
        Top-K filter for generation.
    pretrained_checkpoint:
        Path to the Phase 3.1 checkpoint to load before SFT fine-tuning.
        Empty string = train from random init (not recommended for SFT).
    verification_steps:
        Number of steps for quick verification runs.  Maps to verification_level
        (20 / 100 / 1000) in the SFT runner.
    """

    enabled: bool = False
    datasets: list[str] = field(default_factory=lambda: ["magpie_pro", "tulu3_sft"])
    conversation_template: str = "alpaca"
    train_on_prompt: bool = False
    packing: bool = False
    max_turns: int = 0
    ignore_prompt_loss: bool = True
    prompt_prefix: str = "### Instruction:\n"
    assistant_prefix: str = "### Response:\n"
    system_prefix: str = "### System:\n"
    user_prefix: str = "### User:\n"
    evaluation_frequency: int = 0
    generation_frequency: int = 0
    max_new_bytes: int = 128
    generation_temperature: float = 0.8
    generation_top_k: int = 50
    pretrained_checkpoint: str = ""
    verification_steps: int = 100

    def __post_init__(self) -> None:
        valid_templates = {"alpaca", "chat", "multi_turn"}
        if self.conversation_template not in valid_templates:
            raise ConfigError(
                f"conversation_template must be one of {sorted(valid_templates)}, "
                f"got '{self.conversation_template}'"
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

    def to_formatter_dict(self) -> dict[str, Any]:
        """Convert to FormatterConfig-compatible dict for ConversationFormatter.

        Returns
        -------
        dict[str, Any]
            Dict with keys matching :class:`~training.conversation_formatter.FormatterConfig`.
        """
        return {
            "system_prefix": self.system_prefix,
            "prompt_prefix": self.prompt_prefix,
            "assistant_prefix": self.assistant_prefix,
            "user_prefix": self.user_prefix,
            "max_turns": self.max_turns,
        }
