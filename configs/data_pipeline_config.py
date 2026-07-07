# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Hierarchical configuration for the IVERI CORE data engineering pipeline.

Organizes all settings for download, quality filtering, deduplication,
splitting, registry, PII removal, language detection, dataset mixing,
and reporting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, cast

from core.exceptions import ConfigError

# Optional yaml import for yaml serialization
try:
    import yaml

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


@dataclass(frozen=False, slots=True)
class DownloadConfig:
    """Configuration for downloading datasets from Hugging Face Hub."""

    raw_data_dir: str = "data/raw"
    hf_cache_dir: str | None = None
    hf_token: str | None = None
    resume_downloads: bool = True
    verify_checksums: bool = True
    max_retries: int = 3
    timeout_seconds: int = 300

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ConfigError(f"max_retries must be >= 0, got {self.max_retries}")
        if self.timeout_seconds <= 0:
            raise ConfigError(f"timeout_seconds must be > 0, got {self.timeout_seconds}")


@dataclass(frozen=False, slots=True)
class QualityConfig:
    """Configuration for quality filtering of text documents."""

    min_doc_chars: int = 100
    max_doc_chars: int = 100_000
    min_alpha_ratio: float = 0.5
    max_avg_line_length: int = 1000
    max_rep_ratio: float = 0.2
    remove_control_chars: bool = True
    normalize_unicode: bool = True
    repair_broken_utf8: bool = True
    filter_excessive_emoji: bool = True
    max_emoji_ratio: float = 0.1
    filter_excessive_punctuation: bool = True
    max_punct_ratio: float = 0.3
    filter_html_garbage: bool = True
    max_html_ratio: float = 0.2

    def __post_init__(self) -> None:
        if self.min_doc_chars < 0:
            raise ConfigError(f"min_doc_chars must be >= 0, got {self.min_doc_chars}")
        if self.max_doc_chars < self.min_doc_chars:
            raise ConfigError(
                f"max_doc_chars ({self.max_doc_chars}) must be >= min_doc_chars ({self.min_doc_chars})"
            )
        if not (0.0 <= self.min_alpha_ratio <= 1.0):
            raise ConfigError(f"min_alpha_ratio must be in [0.0, 1.0], got {self.min_alpha_ratio}")
        if self.max_avg_line_length <= 0:
            raise ConfigError(f"max_avg_line_length must be > 0, got {self.max_avg_line_length}")
        if not (0.0 <= self.max_rep_ratio <= 1.0):
            raise ConfigError(f"max_rep_ratio must be in [0.0, 1.0], got {self.max_rep_ratio}")
        if not (0.0 <= self.max_emoji_ratio <= 1.0):
            raise ConfigError(f"max_emoji_ratio must be in [0.0, 1.0], got {self.max_emoji_ratio}")
        if not (0.0 <= self.max_punct_ratio <= 1.0):
            raise ConfigError(f"max_punct_ratio must be in [0.0, 1.0], got {self.max_punct_ratio}")
        if not (0.0 <= self.max_html_ratio <= 1.0):
            raise ConfigError(f"max_html_ratio must be in [0.0, 1.0], got {self.max_html_ratio}")


@dataclass(frozen=False, slots=True)
class DedupConfig:
    """Configuration for exact and near-duplicate detection."""

    exact_dedup_enabled: bool = True
    near_dedup_enabled: bool = True
    near_dedup_threshold: float = 0.8
    near_dedup_num_perm: int = 128
    compute_sha256: bool = True

    def __post_init__(self) -> None:
        if not (0.0 < self.near_dedup_threshold <= 1.0):
            raise ConfigError(
                f"near_dedup_threshold must be in (0.0, 1.0], got {self.near_dedup_threshold}"
            )
        if self.near_dedup_num_perm <= 0:
            raise ConfigError(f"near_dedup_num_perm must be > 0, got {self.near_dedup_num_perm}")


@dataclass(frozen=False, slots=True)
class SplitConfig:
    """Configuration for splitting data into train, val, and test partitions."""

    train_ratio: float = 0.98
    val_ratio: float = 0.01
    test_ratio: float = 0.01
    seed: int = 42
    small_dataset_train_ratio: float = 0.90
    small_dataset_val_ratio: float = 0.05
    small_dataset_test_ratio: float = 0.05
    small_dataset_threshold: int = 10_000

    def __post_init__(self) -> None:
        # Check that main ratios sum to approx 1.0
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-5:
            raise ConfigError(f"Ratios must sum to 1.0, got {total}")
        # Check that small dataset ratios sum to approx 1.0
        small_total = (
            self.small_dataset_train_ratio
            + self.small_dataset_val_ratio
            + self.small_dataset_test_ratio
        )
        if abs(small_total - 1.0) > 1e-5:
            raise ConfigError(f"Small dataset ratios must sum to 1.0, got {small_total}")
        if self.small_dataset_threshold <= 0:
            raise ConfigError(
                f"small_dataset_threshold must be > 0, got {self.small_dataset_threshold}"
            )


@dataclass(frozen=False, slots=True)
class RegistryConfig:
    """Configuration for the plugin-based dataset registry."""

    spec_dir: str = "data/dataset_specs"
    auto_discover: bool = True
    validate_on_load: bool = True


@dataclass(frozen=False, slots=True)
class PIIConfig:
    """Configuration for PII and credential scrubbing."""

    enabled: bool = True
    replacement: str = "[REDACTED]"
    remove_emails: bool = True
    remove_phones: bool = True
    remove_aadhaar: bool = True
    remove_pan: bool = True
    remove_ip: bool = True
    remove_credit_cards: bool = True
    remove_urls: bool = False
    remove_github_tokens: bool = True
    remove_aws_keys: bool = True
    remove_openai_keys: bool = True
    remove_bearer_tokens: bool = True
    remove_rsa_keys: bool = True
    remove_jwt_tokens: bool = True


@dataclass(frozen=False, slots=True)
class LanguageConfig:
    """Configuration for language identification and filtering."""

    allowed_languages: list[str] = field(default_factory=lambda: ["en"])
    reject_languages: list[str] = field(default_factory=list)
    short_text_min_chars: int = 20
    stage1_languages: list[str] = field(default_factory=lambda: ["en"])
    stage3b_languages: list[str] = field(default_factory=lambda: ["en", "hi"])

    def __post_init__(self) -> None:
        if self.short_text_min_chars < 0:
            raise ConfigError(f"short_text_min_chars must be >= 0, got {self.short_text_min_chars}")


@dataclass(frozen=False, slots=True)
class MixingConfig:
    """Configuration for mixing datasets for different training stages."""

    strategy: str = "weighted_random"
    temperature: float = 1.0
    curriculum_start_step: int = 0
    curriculum_end_step: int = 50_000

    def __post_init__(self) -> None:
        if self.strategy not in ("weighted_random", "round_robin", "temperature", "curriculum"):
            raise ConfigError(f"Invalid strategy: {self.strategy}")
        if self.temperature <= 0.0:
            raise ConfigError(f"temperature must be > 0.0, got {self.temperature}")
        if self.curriculum_start_step < 0:
            raise ConfigError(
                f"curriculum_start_step must be >= 0, got {self.curriculum_start_step}"
            )
        if self.curriculum_end_step < self.curriculum_start_step:
            raise ConfigError(
                f"curriculum_end_step ({self.curriculum_end_step}) must be >= "
                f"curriculum_start_step ({self.curriculum_start_step})"
            )


@dataclass(frozen=False, slots=True)
class ReportConfig:
    """Configuration for generating pipeline summaries and stats reports."""

    report_dir: str = "reports/phase_3_0"
    processed_data_dir: str = "data/processed"
    splits_dir: str = "data/splits"
    proprietary_dir: str = "data/proprietary"
    generate_reports: bool = True
    generate_histograms: bool = True
    generate_composition_json: bool = True


@dataclass(frozen=False, slots=True)
class DataPipelineConfig:
    """Master configuration for the Phase 3.0 Data Engineering Pipeline."""

    download: DownloadConfig = field(default_factory=DownloadConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    dedup: DedupConfig = field(default_factory=DedupConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    registry: RegistryConfig = field(default_factory=RegistryConfig)
    pii: PIIConfig = field(default_factory=PIIConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    mixing: MixingConfig = field(default_factory=MixingConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration recursively to a dict."""
        return {
            "download": {f.name: getattr(self.download, f.name) for f in fields(self.download)},
            "quality": {f.name: getattr(self.quality, f.name) for f in fields(self.quality)},
            "dedup": {f.name: getattr(self.dedup, f.name) for f in fields(self.dedup)},
            "split": {f.name: getattr(self.split, f.name) for f in fields(self.split)},
            "registry": {f.name: getattr(self.registry, f.name) for f in fields(self.registry)},
            "pii": {f.name: getattr(self.pii, f.name) for f in fields(self.pii)},
            "language": {f.name: getattr(self.language, f.name) for f in fields(self.language)},
            "mixing": {f.name: getattr(self.mixing, f.name) for f in fields(self.mixing)},
            "report": {f.name: getattr(self.report, f.name) for f in fields(self.report)},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DataPipelineConfig:
        """Deserialize configuration recursively from a dict."""
        download = DownloadConfig(**d.get("download", {}))
        quality = QualityConfig(**d.get("quality", {}))
        dedup = DedupConfig(**d.get("dedup", {}))
        split = SplitConfig(**d.get("split", {}))
        registry = RegistryConfig(**d.get("registry", {}))
        pii = PIIConfig(**d.get("pii", {}))
        language = LanguageConfig(**d.get("language", {}))
        mixing = MixingConfig(**d.get("mixing", {}))
        report = ReportConfig(**d.get("report", {}))
        return cls(
            download=download,
            quality=quality,
            dedup=dedup,
            split=split,
            registry=registry,
            pii=pii,
            language=language,
            mixing=mixing,
            report=report,
        )

    def to_yaml(self) -> str:
        """Serialize configuration to a YAML string."""
        if not _YAML_AVAILABLE:
            raise ImportError("PyYAML is not installed. Run 'pip install pyyaml'")
        return cast(str, yaml.dump(self.to_dict(), sort_keys=False))

    @classmethod
    def from_yaml(cls, yaml_str: str) -> DataPipelineConfig:
        """Deserialize configuration from a YAML string."""
        if not _YAML_AVAILABLE:
            raise ImportError("PyYAML is not installed. Run 'pip install pyyaml'")
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        """Save configuration to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, path: str | Path) -> DataPipelineConfig:
        """Load configuration from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
