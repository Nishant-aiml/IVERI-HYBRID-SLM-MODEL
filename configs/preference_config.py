# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Phase 3.4 preference optimization and response alignment.

Kept in a separate module (following the pattern of :mod:`configs.instruction_config`
and :mod:`configs.coding_config`) so that base_config.py remains modular.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.exceptions import ConfigError


@dataclass(frozen=False, slots=True)
class PreferenceConfig:
    """Configuration for the Phase 3.4 preference optimization pipeline.

    All fields default to safe / disabled values so that existing pretraining,
    SFT, and coding runs are unaffected when this config is present.

    Attributes
    ----------
    enabled:
        Master switch for preference optimization. ``False`` = disabled (default).
    algorithm:
        Alignment algorithm to run. One of ``"dpo"``, ``"ipo"``, ``"simpo"``,
        or ``"conservative_dpo"``. Default is ``"dpo"``.
    beta:
        DPO/IPO/SimPO preference temperature parameter. Default ``0.1``.
    reference_checkpoint:
        Path to the starting SFT/Coding checkpoint to load for the reference model.
        Also used to initialize the policy model unless overridden.
    reference_device:
        Device to load the reference model on (``"cpu"`` or ``"cuda"``).
        Offloading to CPU helps run preference optimization on smaller GPUs.
    loss_type:
        Type of preference loss function. Typically ``"sigmoid"`` for DPO.
    label_smoothing:
        Smoothing parameter for Conservative DPO. Default ``0.1``.
    ipo_gamma:
        Margin threshold for IPO / SimPO (default ``2.0``).
    max_sequence_length:
        Maximum sequence length for inputs. Default ``512``.
    datasets:
        Ordered list of Stage 4 preference datasets. Must match entries in
        ``data/dataset_specs/preference.yaml``.
    save_preference_scores:
        Whether to save policy/reference logps and rewards to checkpoints.
    generate_alignment_reports:
        Whether to write JSON/Markdown alignment reports at the end of training.
    evaluation_frequency:
        Run preference evaluations every N steps. 0 = use logging.eval_every.
    checkpoint_frequency:
        Save checkpoints every N steps. 0 = use logging.save_every.
    rejection_threshold:
        Maximum validation loss threshold before triggering policy rejection.
    chosen_reward_weight:
        Weight multiplier for chosen reward logging. Default ``1.0``.
    rejected_reward_weight:
        Weight multiplier for rejected reward logging. Default ``1.0``.
    deterministic_mode:
        Enforce strict reproducibility settings. Default ``True``.
    verification_steps:
        Number of steps for quick verification runs. Maps to verification levels.
    """

    # ── Core switches ──────────────────────────────────────────────────
    enabled: bool = False
    algorithm: str = "dpo"
    beta: float = 0.1

    # ── Reference & Checkpointing ──────────────────────────────────────
    reference_checkpoint: str = ""
    reference_device: str = "cpu"
    loss_type: str = "sigmoid"

    # ── Hyperparameters ────────────────────────────────────────────────
    label_smoothing: float = 0.1
    ipo_gamma: float = 2.0
    max_sequence_length: int = 512

    # ── Dataset & Formatter ────────────────────────────────────────────
    datasets: list[str] = field(
        default_factory=lambda: ["ultrafeedback", "tulu3_pref"]
    )

    # ── Logging & Telemetry ─────────────────────────────────────────────
    save_preference_scores: bool = True
    generate_alignment_reports: bool = True
    evaluation_frequency: int = 0
    checkpoint_frequency: int = 0
    rejection_threshold: float = 15.0

    # ── Metrics & Constraints ──────────────────────────────────────────
    chosen_reward_weight: float = 1.0
    rejected_reward_weight: float = 1.0
    deterministic_mode: bool = True
    verification_steps: int = 100

    def __post_init__(self) -> None:
        valid_algorithms = {"dpo", "ipo", "simpo", "conservative_dpo"}
        if self.algorithm.lower() not in valid_algorithms:
            raise ConfigError(
                f"algorithm must be one of {sorted(valid_algorithms)}, "
                f"got '{self.algorithm}'"
            )

        valid_devices = {"cpu", "cuda"}
        if self.reference_device.lower() not in valid_devices:
            raise ConfigError(
                f"reference_device must be one of {sorted(valid_devices)}, "
                f"got '{self.reference_device}'"
            )

        if self.beta <= 0.0:
            raise ConfigError(f"beta must be positive, got {self.beta}")

        if not (0.0 <= self.label_smoothing <= 0.5):
            raise ConfigError(
                f"label_smoothing must be in [0.0, 0.5], got {self.label_smoothing}"
            )

        if self.max_sequence_length <= 0:
            raise ConfigError(
                f"max_sequence_length must be > 0, got {self.max_sequence_length}"
            )

        if self.verification_steps <= 0:
            raise ConfigError(
                f"verification_steps must be > 0, got {self.verification_steps}"
            )

        if self.ipo_gamma < 0.0:
            raise ConfigError(f"ipo_gamma must be >= 0.0, got {self.ipo_gamma}")
