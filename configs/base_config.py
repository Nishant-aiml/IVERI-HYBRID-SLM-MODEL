# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Master configuration for the IVERI CORE architecture.

All hyperparameters, hardware settings, and logging knobs live here as
strongly-typed, validated :mod:`dataclasses`.  Defaults correspond to
the **10M-parameter nano prototype** (``hidden_dim=256``,
``num_layers=6``, ``num_heads=4``), which is the first model built.

Hierarchy
---------
.. code-block:: text

    IVERIConfig
    ├── ModelConfig
    │   └── BLTConfig
    ├── TrainingConfig
    ├── HardwareConfig
    └── LoggingConfig

Quick Start
-----------
>>> from configs.base_config import IVERIConfig, get_base_config
>>> cfg = get_base_config()                          # nano defaults
>>> cfg = get_base_config(model={"hidden_dim": 512}) # partial override
>>> cfg.save("run_config.json")
>>> cfg2 = IVERIConfig.load("run_config.json")
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from configs.distributed_config import DistributedConfig
from configs.data_pipeline_config import DataPipelineConfig
from configs.instruction_config import InstructionConfig
from configs.coding_config import CodingConfig
from configs.preference_config import PreferenceConfig
from configs.research_config import ResearchConfig
from core.exceptions import ConfigError

# ── Allowed values ─────────────────────────────────────────────────────────

_VALID_MIXED_PRECISION: frozenset[str] = frozenset({"fp16", "bf16", "fp32"})
"""Accepted values for :pyattr:`HardwareConfig.mixed_precision`."""

_VALID_LOG_LEVELS: frozenset[str] = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
"""Standard Python logging levels accepted by :pyattr:`LoggingConfig.log_level`."""


# ── BLT sub-config ─────────────────────────────────────────────────────────


@dataclass(frozen=False, slots=True)
class BLTConfig:
    """Configuration for the Byte Latent Transformer (BLT) patcher.

    Controls how raw bytes are grouped into variable-length patches
    based on the entropy model's output.

    Attributes
    ----------
    patch_size_min:
        Minimum bytes per patch.  Must be ≥ 1.
    patch_size_max:
        Maximum bytes per patch.  Must be ≥ ``patch_size_min``.
    entropy_threshold:
        Entropy score above which a patch boundary is placed.
        Must be > 0.
    """

    patch_size_min: int = 1
    patch_size_max: int = 8
    entropy_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.patch_size_min < 1:
            raise ConfigError(f"patch_size_min must be >= 1, got {self.patch_size_min}")
        if self.patch_size_max < self.patch_size_min:
            raise ConfigError(
                f"patch_size_max ({self.patch_size_max}) must be "
                f">= patch_size_min ({self.patch_size_min})"
            )
        if self.entropy_threshold <= 0:
            raise ConfigError(f"entropy_threshold must be > 0, got {self.entropy_threshold}")


# ── Model config ───────────────────────────────────────────────────────────


@dataclass(frozen=False, slots=True)
class ModelConfig:
    """Model architecture hyperparameters.

    Defaults are sized for the **10M nano prototype**:
    ``hidden_dim=256``, ``num_layers=6``, ``num_heads=4``.

    Attributes
    ----------
    hidden_dim:
        Width of the backbone feature vectors.  Must be divisible by
        ``num_heads``.
    num_layers:
        Number of backbone blocks.
    num_heads:
        Attention heads for Flash Attention layers.
    mamba_ratio:
        Mamba2 SSM blocks per attention block within each backbone block.
    num_experts:
        Total MoE expert networks.
    num_active_experts:
        Experts activated per token.  Must be ≤ ``num_experts``.
    max_recursion_depth:
        Maximum recursion depth for MoR (Mixture of Recursions).
    titans_memory_dim:
        Hidden width of the Titans neural-memory MLP.
    dropout:
        Dropout probability.  Set to 0.0 for research-grade defaults.
    blt:
        Byte Latent Transformer patcher settings.
    """

    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    mamba_ratio: int = 1
    num_experts: int = 2
    num_active_experts: int = 1
    max_recursion_depth: int = 4
    titans_memory_dim: int = 64
    dropout: float = 0.0
    blt: BLTConfig = field(default_factory=BLTConfig)
    # Phase 6.3.2 OBJ4 — physical ablation gates (default True preserves production path)
    use_titans: bool = True
    use_blt: bool = True
    use_mor: bool = True
    use_moe: bool = True
    use_entropy_routing: bool = True
    moe_capacity_factor: float = 1.25


    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ConfigError(f"hidden_dim must be > 0, got {self.hidden_dim}")
        if self.num_heads <= 0:
            raise ConfigError(f"num_heads must be > 0, got {self.num_heads}")
        if self.hidden_dim % self.num_heads != 0:
            raise ConfigError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.num_layers <= 0:
            raise ConfigError(f"num_layers must be > 0, got {self.num_layers}")
        if self.num_experts <= 0:
            raise ConfigError(f"num_experts must be > 0, got {self.num_experts}")
        if self.num_active_experts <= 0:
            raise ConfigError(f"num_active_experts must be > 0, got {self.num_active_experts}")
        if self.num_active_experts > self.num_experts:
            raise ConfigError(
                f"num_active_experts ({self.num_active_experts}) must be "
                f"<= num_experts ({self.num_experts})"
            )
        if self.max_recursion_depth <= 0:
            raise ConfigError(f"max_recursion_depth must be > 0, got {self.max_recursion_depth}")
        if self.titans_memory_dim <= 0:
            raise ConfigError(f"titans_memory_dim must be > 0, got {self.titans_memory_dim}")
        if self.mamba_ratio <= 0:
            raise ConfigError(f"mamba_ratio must be > 0, got {self.mamba_ratio}")
        if not (0.0 <= self.dropout < 1.0):
            raise ConfigError(f"dropout must be in [0.0, 1.0), got {self.dropout}")


# ── Training config ────────────────────────────────────────────────────────


@dataclass(frozen=False, slots=True)
class TrainingConfig:
    """Training loop hyperparameters.

    Attributes
    ----------
    batch_size:
        Per-device micro-batch size.
    gradient_accumulation:
        Number of micro-steps before an optimiser step.
        Effective batch = ``batch_size * gradient_accumulation``.
    learning_rate:
        Peak learning rate for the cosine schedule.
    min_lr:
        Floor learning rate at the end of decay.  Must be ≤ ``learning_rate``.
    warmup_steps:
        Linear warm-up steps before cosine decay begins.
    max_steps:
        Total optimiser steps (not gradient-accumulation steps).
    weight_decay:
        AdamW decoupled weight-decay coefficient.
    grad_clip:
        Maximum gradient norm for clipping.  Must be > 0.
    seq_len:
        Sequence length in bytes.
    """

    batch_size: int = 32
    gradient_accumulation: int = 4
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 1000
    max_steps: int = 50_000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    seq_len: int = 512
    scheduler_type: str = "cosine"
    scheduler_power: float = 1.0
    scheduler_step_size: int = 1000
    scheduler_gamma: float = 0.1

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ConfigError(f"batch_size must be > 0, got {self.batch_size}")
        if self.gradient_accumulation <= 0:
            raise ConfigError(
                f"gradient_accumulation must be > 0, got {self.gradient_accumulation}"
            )
        if self.learning_rate <= 0:
            raise ConfigError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.min_lr <= 0:
            raise ConfigError(f"min_lr must be > 0, got {self.min_lr}")
        if self.min_lr > self.learning_rate:
            raise ConfigError(
                f"min_lr ({self.min_lr}) must be <= " f"learning_rate ({self.learning_rate})"
            )
        if self.warmup_steps < 0:
            raise ConfigError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.max_steps <= 0:
            raise ConfigError(f"max_steps must be > 0, got {self.max_steps}")
        if self.weight_decay < 0:
            raise ConfigError(f"weight_decay must be >= 0, got {self.weight_decay}")
        if self.grad_clip <= 0:
            raise ConfigError(f"grad_clip must be > 0, got {self.grad_clip}")
        if self.seq_len <= 0:
            raise ConfigError(f"seq_len must be > 0, got {self.seq_len}")
        if self.scheduler_type not in (
            "constant",
            "linear",
            "cosine",
            "polynomial",
            "step",
            "exponential",
        ):
            raise ConfigError(f"Invalid scheduler_type: {self.scheduler_type}")
        if self.scheduler_power <= 0:
            raise ConfigError(f"scheduler_power must be > 0, got {self.scheduler_power}")
        if self.scheduler_step_size <= 0:
            raise ConfigError(f"scheduler_step_size must be > 0, got {self.scheduler_step_size}")
        if not (0.0 < self.scheduler_gamma <= 1.0):
            raise ConfigError(f"scheduler_gamma must be in (0, 1], got {self.scheduler_gamma}")


# ── Hardware config ────────────────────────────────────────────────────────


@dataclass(frozen=False, slots=True)
class HardwareConfig:
    """Device and precision settings.

    Attributes
    ----------
    mixed_precision:
        Floating-point format: ``'fp16'``, ``'bf16'``, or ``'fp32'``.
    gradient_checkpointing:
        Whether to trade compute for VRAM via activation checkpointing.
    device:
        PyTorch device string (``'cuda'``, ``'cpu'``, ``'cuda:0'``, …).
    num_workers:
        DataLoader worker processes.
    """

    mixed_precision: str = "fp16"
    gradient_checkpointing: bool = True
    device: str = "cuda"
    num_workers: int = 4

    def __post_init__(self) -> None:
        if self.mixed_precision not in _VALID_MIXED_PRECISION:
            raise ConfigError(
                f"mixed_precision must be one of {sorted(_VALID_MIXED_PRECISION)}, "
                f"got '{self.mixed_precision}'"
            )
        if self.num_workers < 0:
            raise ConfigError(f"num_workers must be >= 0, got {self.num_workers}")


# ── Logging config ─────────────────────────────────────────────────────────


@dataclass(frozen=False, slots=True)
class LoggingConfig:
    """Logging, evaluation, and checkpoint cadence.

    Attributes
    ----------
    log_every:
        Log training metrics every *N* optimiser steps.
    eval_every:
        Run evaluation every *N* optimiser steps.
    save_every:
        Save a checkpoint every *N* optimiser steps.
    wandb_project:
        Weights & Biases project slug.
    log_dir:
        Directory for local log files and TensorBoard events.
    log_level:
        Python logging level string.
    enabled:
        Master switch for the logging system.
    project:
        W&B project name.
    entity:
        W&B entity (team or username).
    mode:
        W&B run mode: ``'online'``, ``'offline'``, or ``'disabled'``.
    offline:
        Force offline mode regardless of ``mode``.
    save_dir:
        Root directory for local log artefacts.
    tensorboard:
        Enable TensorBoard ``SummaryWriter``.
    csv:
        Enable CSV metric appending.
    json:
        Enable JSONL metric appending.
    frequency:
        Global logging frequency fallback (steps).
    gradient_logging:
        Log per-layer gradient norms and histograms.
    memory_logging:
        Log GPU and CPU memory telemetry.
    telemetry_logging:
        Log architecture-specific telemetry from model forward pass.
    run_name:
        Human-readable run label (auto-generated if None).
    run_id:
        Unique run identifier for resume support (auto-generated if None).
    resume:
        W&B resume mode: ``'allow'``, ``'must'``, ``'never'``, or None.
    tags:
        Optional list of string tags attached to the run.
    notes:
        Optional free-text notes attached to the run.
    log_frequency:
        Log scalar metrics every *N* steps.
    checkpoint_frequency:
        Save checkpoints every *N* steps.
    system_monitor_interval:
        System resource polling interval in seconds.
    """

    log_every: int = 10
    eval_every: int = 500
    save_every: int = 1000
    wandb_project: str = "iveri-core"
    log_dir: str = "logs"
    log_level: str = "INFO"

    enabled: bool = True
    project: str = "iveri-core"
    entity: str | None = None
    mode: str = "online"
    offline: bool = False
    save_dir: str = "logs"
    tensorboard: bool = False
    csv: bool = True
    json: bool = True
    frequency: int = 10
    gradient_logging: bool = True
    memory_logging: bool = True
    telemetry_logging: bool = True

    run_name: str | None = None
    run_id: str | None = None
    resume: str | None = None
    tags: list | None = None
    notes: str | None = None
    log_frequency: int = 10
    checkpoint_frequency: int = 1000
    system_monitor_interval: float = 5.0

    def __post_init__(self) -> None:
        if self.log_every <= 0:
            raise ConfigError(f"log_every must be > 0, got {self.log_every}")
        if self.eval_every <= 0:
            raise ConfigError(f"eval_every must be > 0, got {self.eval_every}")
        if self.save_every <= 0:
            raise ConfigError(f"save_every must be > 0, got {self.save_every}")
        if self.log_level not in _VALID_LOG_LEVELS:
            raise ConfigError(
                f"log_level must be one of {sorted(_VALID_LOG_LEVELS)}, " f"got '{self.log_level}'"
            )
        if self.mode not in ("online", "offline", "disabled"):
            raise ConfigError(f"mode must be one of ('online', 'offline', 'disabled'), got '{self.mode}'")
        if self.frequency <= 0:
            raise ConfigError(f"frequency must be > 0, got {self.frequency}")
        if self.log_frequency <= 0:
            raise ConfigError(f"log_frequency must be > 0, got {self.log_frequency}")
        if self.checkpoint_frequency <= 0:
            raise ConfigError(f"checkpoint_frequency must be > 0, got {self.checkpoint_frequency}")
        if self.system_monitor_interval <= 0:
            raise ConfigError(f"system_monitor_interval must be > 0, got {self.system_monitor_interval}")
        if self.resume is not None and self.resume not in ("allow", "must", "never"):
            raise ConfigError(
                f"resume must be one of ('allow', 'must', 'never') or None, got '{self.resume}'"
            )# ── Evaluation config ───────────────────────────────────────────────────────


@dataclass(frozen=False, slots=True)
class EvaluationConfig:
    """Settings for the evaluation pipeline and benchmark infrastructure.

    Attributes
    ----------
    enabled:
        Master switch for evaluation.
    batch_size:
        Evaluation batch size.
    max_eval_batches:
        Maximum number of validation dataset batches to evaluate (0 for full dataset).
    benchmark_iterations:
        Number of iterations to run during inference throughput benchmark.
    warmup_iterations:
        Number of warmup iterations for benchmarking.
    memory_tracking:
        Whether to track system memory utilization.
    throughput_tracking:
        Whether to benchmark inference throughput.
    architecture_tracking:
        Whether to collect and analyze architecture-specific telemetry.
    compare_checkpoints:
        Whether checkpoint comparison mode is enabled.
    generate_reports:
        Whether to output generated reports (Markdown, JSON, CSV).
    save_predictions:
        Whether to save text predictions.
    save_metrics:
        Whether to write metrics data to log files.
    save_logits:
        Whether to save output logits.
    save_attention_maps:
        Whether to save model attention maps.
    save_hidden_states:
        Whether to save intermediate hidden states.
    save_architecture_stats:
        Whether to save detailed distribution stats.
    generation_enabled:
        Whether to run generation benchmarks.
    generation_max_new_bytes:
        Max new tokens/bytes to generate during inference.
    generation_temperature:
        Temperature for stochastic token decoding.
    generation_top_k:
        Top-k parameter for decoding.
    generation_top_p:
        Top-p parameter for nucleus decoding.
    report_dir:
        Directory for evaluation reports.
    metrics_dir:
        Directory for storing metric logs.
    """

    enabled: bool = True
    batch_size: int = 16
    max_eval_batches: int = 100
    benchmark_iterations: int = 50
    warmup_iterations: int = 5
    memory_tracking: bool = True
    throughput_tracking: bool = True
    architecture_tracking: bool = True
    compare_checkpoints: bool = False
    generate_reports: bool = True
    save_predictions: bool = False
    save_metrics: bool = True
    save_logits: bool = False
    save_attention_maps: bool = False
    save_hidden_states: bool = False
    save_architecture_stats: bool = True
    generation_enabled: bool = True
    generation_max_new_bytes: int = 32
    generation_temperature: float = 0.7
    generation_top_k: int = 50
    generation_top_p: float = 0.9
    report_dir: str = "reports/evaluation"
    metrics_dir: str = "logs/eval_metrics"

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ConfigError(f"batch_size must be > 0, got {self.batch_size}")
        if self.max_eval_batches < 0:
            raise ConfigError(f"max_eval_batches must be >= 0, got {self.max_eval_batches}")
        if self.benchmark_iterations <= 0:
            raise ConfigError(f"benchmark_iterations must be > 0, got {self.benchmark_iterations}")
        if self.warmup_iterations < 0:
            raise ConfigError(f"warmup_iterations must be >= 0, got {self.warmup_iterations}")
        if self.generation_max_new_bytes <= 0:
            raise ConfigError(f"generation_max_new_bytes must be > 0, got {self.generation_max_new_bytes}")
        if self.generation_temperature < 0.0:
            raise ConfigError(f"generation_temperature must be >= 0.0, got {self.generation_temperature}")
        if self.generation_top_k < 0:
            raise ConfigError(f"generation_top_k must be >= 0, got {self.generation_top_k}")
        if not (0.0 <= self.generation_top_p <= 1.0):
            raise ConfigError(f"generation_top_p must be between 0.0 and 1.0, got {self.generation_top_p}")


# ── Master config ──────────────────────────────────────────────────────────


@dataclass(frozen=False, slots=True)
class IVERIConfig:
    """Top-level configuration for the full IVERI CORE system.

    Composes all sub-configs and provides serialisation / deserialisation
    helpers for experiment reproducibility.

    Examples
    --------
    >>> cfg = IVERIConfig()
    >>> cfg.model.hidden_dim
    256
    >>> cfg.save("experiment_01.json")
    >>> loaded = IVERIConfig.load("experiment_01.json")
    >>> assert cfg.to_dict() == loaded.to_dict()
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    data_pipeline: DataPipelineConfig = field(default_factory=DataPipelineConfig)
    # Phase 3.2 — instruction tuning (SFT). Disabled by default.
    instruction: InstructionConfig = field(default_factory=InstructionConfig)
    # Phase 3.3 — coding specialization. Disabled by default.
    coding: CodingConfig = field(default_factory=CodingConfig)
    # Phase 3.4 — preference optimization (DPO/SimPO). Disabled by default.
    preference: PreferenceConfig = field(default_factory=PreferenceConfig)
    # Stage 5 — research validation campaign. Disabled by default.
    research: ResearchConfig = field(default_factory=ResearchConfig)

    def __post_init__(self) -> None:
        """Run cross-field validation after initialisation."""
        self.validate()

    # ── Validation ─────────────────────────────────────────────────────

    def validate(self) -> None:
        """Run all cross-field validation checks.

        Individual sub-config ``__post_init__`` methods already verify
        per-section invariants.  This method checks constraints that
        span multiple sections.

        Raises
        ------
        ConfigError
            If any cross-field invariant is violated.
        """
        # Sub-config __post_init__ validators have already fired during
        # construction.  Add any cross-section checks below.

        # Effective batch size sanity (warn-level, no raise — just guard
        # against clearly erroneous combos).
        effective_batch = self.training.batch_size * self.training.gradient_accumulation
        if effective_batch > 4096:
            raise ConfigError(
                f"Effective batch size ({effective_batch} = "
                f"batch_size({self.training.batch_size}) × "
                f"gradient_accumulation({self.training.gradient_accumulation})) "
                f"exceeds 4096.  This is almost certainly unintentional for a "
                f"research prototype.  Reduce one or both values."
            )

        # Warm-up must not exceed total steps.
        if self.training.warmup_steps >= self.training.max_steps:
            raise ConfigError(
                f"warmup_steps ({self.training.warmup_steps}) must be < "
                f"max_steps ({self.training.max_steps})"
            )

    # ── Serialisation ──────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Recursively convert the config tree to plain dicts.

        :class:`~pathlib.Path` objects are converted to POSIX strings so
        the result is JSON-serialisable.

        Returns
        -------
        dict[str, Any]
            Nested dictionary matching the dataclass hierarchy.
        """
        from typing import cast

        return cast(dict[str, Any], _dataclass_to_dict(self))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IVERIConfig:
        """Reconstruct an :class:`IVERIConfig` from a nested dict.

        Parameters
        ----------
        data:
            Dictionary previously produced by :meth:`to_dict` or parsed
            from a JSON file.

        Returns
        -------
        IVERIConfig
            Fully validated configuration instance.

        Raises
        ------
        ConfigError
            If ``data`` contains invalid values.
        """
        import copy
        import warnings
        from dataclasses import fields

        data_copy = copy.deepcopy(data)

        def _instantiate_dataclass_safe(dc_cls: type, dc_data: dict[str, Any]) -> Any:
            known_fields = {f.name for f in fields(dc_cls)}
            filtered_data = {}
            for k, v in dc_data.items():
                if k in known_fields:
                    filtered_data[k] = v
                else:
                    warnings.warn(
                        f"Unknown configuration key '{k}' for {dc_cls.__name__} ignored.",
                        UserWarning,
                        stacklevel=2,
                    )
            return dc_cls(**filtered_data)

        # Handle BLT nested sub-config separately as it is nested under model
        model_data = data_copy.get("model", {})
        blt_data = model_data.pop("blt", {})
        blt_cfg = _instantiate_dataclass_safe(BLTConfig, blt_data) if blt_data else BLTConfig()

        model_cfg = _instantiate_dataclass_safe(ModelConfig, model_data)
        model_cfg.blt = blt_cfg

        training_cfg = (
            _instantiate_dataclass_safe(TrainingConfig, data_copy["training"])
            if "training" in data_copy
            else TrainingConfig()
        )
        hardware_cfg = (
            _instantiate_dataclass_safe(HardwareConfig, data_copy["hardware"])
            if "hardware" in data_copy
            else HardwareConfig()
        )
        logging_cfg = (
            _instantiate_dataclass_safe(LoggingConfig, data_copy["logging"])
            if "logging" in data_copy
            else LoggingConfig()
        )
        evaluation_cfg = (
            _instantiate_dataclass_safe(EvaluationConfig, data_copy["evaluation"])
            if "evaluation" in data_copy
            else EvaluationConfig()
        )
        distributed_cfg = (
            _instantiate_dataclass_safe(DistributedConfig, data_copy["distributed"])
            if "distributed" in data_copy
            else DistributedConfig()
        )
        data_pipeline_cfg = (
            _instantiate_dataclass_safe(DataPipelineConfig, data_copy["data_pipeline"])
            if "data_pipeline" in data_copy
            else DataPipelineConfig()
        )

        instruction_cfg = (
            _instantiate_dataclass_safe(InstructionConfig, data_copy["instruction"])
            if "instruction" in data_copy
            else InstructionConfig()
        )

        coding_cfg = (
            _instantiate_dataclass_safe(CodingConfig, data_copy["coding"])
            if "coding" in data_copy
            else CodingConfig()
        )

        preference_cfg = (
            _instantiate_dataclass_safe(PreferenceConfig, data_copy["preference"])
            if "preference" in data_copy
            else PreferenceConfig()
        )

        research_cfg = (
            _instantiate_dataclass_safe(ResearchConfig, data_copy["research"])
            if "research" in data_copy
            else ResearchConfig()
        )

        return cls(
            model=model_cfg,
            training=training_cfg,
            hardware=hardware_cfg,
            logging=logging_cfg,
            evaluation=evaluation_cfg,
            distributed=distributed_cfg,
            data_pipeline=data_pipeline_cfg,
            instruction=instruction_cfg,
            coding=coding_cfg,
            preference=preference_cfg,
            research=research_cfg,
        )

    # ── File I/O ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Persist the config as a pretty-printed JSON file.

        Parameters
        ----------
        path:
            Destination file path.  Parent directories are created
            automatically.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> IVERIConfig:
        """Load and validate a config from a JSON or YAML file.

        Parameters
        ----------
        path:
            Source file path.

        Returns
        -------
        IVERIConfig
            Fully validated configuration instance.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ConfigError
            If the file contents fail validation.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            except ImportError:
                raise ImportError("PyYAML is required to load configuration from YAML files.")
        else:
            raw = json.loads(path.read_text(encoding="utf-8"))
            
        return cls.from_dict(raw)


    # ── Pretty printing ────────────────────────────────────────────────

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__}("]
        for f in fields(self):
            value = getattr(self, f.name)
            lines.append(f"  {f.name}={value!r},")
        lines.append(")")
        return "\n".join(lines)


# ── Private helpers ────────────────────────────────────────────────────────


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert a dataclass tree to plain dicts.

    Handles nested dataclasses, :class:`~pathlib.Path`, and primitive
    types.  Non-dataclass values are returned as-is.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        result: dict[str, Any] = {}
        for f in fields(obj):
            result[f.name] = _dataclass_to_dict(getattr(obj, f.name))
        return result
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    return obj


# ── Factory function ──────────────────────────────────────────────────────


def get_base_config(**overrides: Any) -> IVERIConfig:
    """Create a base :class:`IVERIConfig` with optional overrides.

    Each keyword argument corresponds to a top-level section
    (``model``, ``training``, ``hardware``, ``logging``) and should
    be a :class:`dict` of field-name → value mappings that override
    the defaults.

    Parameters
    ----------
    **overrides:
        Section-level dicts.  E.g.
        ``get_base_config(model={"hidden_dim": 512})``.

    Returns
    -------
    IVERIConfig
        Fully validated configuration with overrides applied.

    Examples
    --------
    >>> nano = get_base_config()  # pure defaults (10M nano)
    >>> small = get_base_config(
    ...     model={"hidden_dim": 512, "num_layers": 12, "num_heads": 8},
    ... )
    """
    base = IVERIConfig().to_dict()

    for section, section_overrides in overrides.items():
        if section not in base:
            raise ConfigError(
                f"Unknown config section '{section}'. " f"Valid sections: {sorted(base.keys())}"
            )
        if not isinstance(section_overrides, dict):
            raise ConfigError(
                f"Override for section '{section}' must be a dict, "
                f"got {type(section_overrides).__name__}"
            )
        base[section].update(section_overrides)

    return IVERIConfig.from_dict(base)


def get_nano_config(**overrides: Any) -> IVERIConfig:
    """Create a 10M Nano IVERIConfig.

    Nano scale: hidden_dim=256, num_layers=4, num_heads=4, mamba_ratio=1,
    num_experts=2, num_active_experts=1, titans_memory_dim=64, max_recursion_depth=4.
    """
    model_defaults = {
        "hidden_dim": 256,
        "num_layers": 4,
        "num_heads": 4,
        "mamba_ratio": 1,
        "num_experts": 2,
        "num_active_experts": 1,
        "titans_memory_dim": 64,
        "max_recursion_depth": 4,
    }
    if "model" in overrides:
        model_defaults.update(overrides.pop("model"))
    return get_base_config(model=model_defaults, **overrides)


def get_small_config(**overrides: Any) -> IVERIConfig:
    """Create a 35M Small IVERIConfig.

    Small scale: hidden_dim=384, num_layers=5, num_heads=6, mamba_ratio=1,
    num_experts=4, num_active_experts=1, titans_memory_dim=96, max_recursion_depth=6.
    """
    model_defaults = {
        "hidden_dim": 384,
        "num_layers": 5,
        "num_heads": 6,
        "mamba_ratio": 1,
        "num_experts": 4,
        "num_active_experts": 1,
        "titans_memory_dim": 96,
        "max_recursion_depth": 6,
    }
    if "model" in overrides:
        model_defaults.update(overrides.pop("model"))
    return get_base_config(model=model_defaults, **overrides)


def get_medium_config(**overrides: Any) -> IVERIConfig:
    """Create a 70M Medium IVERIConfig.

    Medium scale: hidden_dim=512, num_layers=5, num_heads=8, mamba_ratio=1,
    num_experts=4, num_active_experts=1, titans_memory_dim=128, max_recursion_depth=8.
    """
    model_defaults = {
        "hidden_dim": 512,
        "num_layers": 5,
        "num_heads": 8,
        "mamba_ratio": 1,
        "num_experts": 4,
        "num_active_experts": 1,
        "titans_memory_dim": 128,
        "max_recursion_depth": 8,
    }
    if "model" in overrides:
        model_defaults.update(overrides.pop("model"))
    return get_base_config(model=model_defaults, **overrides)


def get_large_config(**overrides: Any) -> IVERIConfig:
    """Create a 150M Large IVERIConfig.

    Large scale: hidden_dim=768, num_layers=5, num_heads=12, mamba_ratio=1,
    num_experts=4, num_active_experts=1, titans_memory_dim=192, max_recursion_depth=8.
    """
    model_defaults = {
        "hidden_dim": 768,
        "num_layers": 5,
        "num_heads": 12,
        "mamba_ratio": 1,
        "num_experts": 4,
        "num_active_experts": 1,
        "titans_memory_dim": 192,
        "max_recursion_depth": 8,
    }
    if "model" in overrides:
        model_defaults.update(overrides.pop("model"))
    return get_base_config(model=model_defaults, **overrides)


def get_xlarge_config(**overrides: Any) -> IVERIConfig:
    """Create a 300M XLarge IVERIConfig.

    XLarge scale: hidden_dim=1024, num_layers=6, num_heads=16, mamba_ratio=1,
    num_experts=4, num_active_experts=1, titans_memory_dim=256, max_recursion_depth=12.
    """
    model_defaults = {
        "hidden_dim": 1024,
        "num_layers": 6,
        "num_heads": 16,
        "mamba_ratio": 1,
        "num_experts": 4,
        "num_active_experts": 1,
        "titans_memory_dim": 256,
        "max_recursion_depth": 12,
    }
    if "model" in overrides:
        model_defaults.update(overrides.pop("model"))
    return get_base_config(model=model_defaults, **overrides)


def get_max_config(**overrides: Any) -> IVERIConfig:
    """Create a 500M Max IVERIConfig.

    Max scale: hidden_dim=1280, num_layers=7, num_heads=20, mamba_ratio=1,
    num_experts=4, num_active_experts=1, titans_memory_dim=320, max_recursion_depth=16.
    """
    model_defaults = {
        "hidden_dim": 1280,
        "num_layers": 7,
        "num_heads": 20,
        "mamba_ratio": 1,
        "num_experts": 4,
        "num_active_experts": 1,
        "titans_memory_dim": 320,
        "max_recursion_depth": 16,
    }
    if "model" in overrides:
        model_defaults.update(overrides.pop("model"))
    return get_base_config(model=model_defaults, **overrides)


def apply_ablation_overrides(cfg: IVERIConfig, overrides: dict[str, bool]) -> None:
    """Apply component ablation flags to ``cfg.model`` and re-validate.

    Raises
    ------
    ConfigError
        If an override field is not defined on :class:`ModelConfig`.
    """
    for field, value in overrides.items():
        if not hasattr(cfg.model, field):
            raise ConfigError(
                f"Unknown ablation field '{field}'. Expected a boolean flag on ModelConfig "
                f"(e.g. use_titans, use_blt, use_mor, use_moe, use_entropy_routing)."
            )
        setattr(cfg.model, field, value)
    cfg.validate()

