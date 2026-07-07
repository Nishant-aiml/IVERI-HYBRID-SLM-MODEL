# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Plugin-based dataset registry for IVERI CORE.

Loads dataset specifications from YAML files in data/dataset_specs/.
Supports dynamic discovery, registration, and validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

# Optional yaml import, with fallback
try:
    import yaml

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass(frozen=False, slots=True)
class DatasetEntry:
    """Detailed specifications and metadata for a single dataset."""

    name: str
    hf_id: str | None = None
    hf_config: str | None = None
    hf_split: str = "train"
    priority: str = "A"  # S, A, or B
    license: str = "unknown"
    format: str = "pretrain"  # pretrain, sft, preference
    description: str = ""
    size_estimate_gb: float = 0.1
    text_field: str = "text"
    mixing_weight: float = 0.0
    stage: str | int = "1"
    stage_name: str = "foundation_pretraining"
    format_type: str = "raw"  # raw, messages, conversations, alpaca, dpo
    requires_hf_token: bool = False
    attribution_required: bool = False
    license_filter_field: str | None = None
    license_filter_values: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    source: str = "huggingface"  # huggingface, local
    path: str | None = None


class DataRegistry:
    """Registry class for discovery and lookup of dataset specs."""

    def __init__(
        self,
        spec_dir: str | Path = "data/dataset_specs",
        auto_discover: bool = True,
        validate: bool = True,
    ) -> None:
        self.spec_dir = Path(spec_dir)
        self.datasets: dict[str, DatasetEntry] = {}
        self._validate_on_register = validate

        if auto_discover:
            self.load_all_specs()

    def discover_plugins(self, spec_dir: Path) -> list[Path]:
        """Find all YAML files within the spec directory."""
        if not spec_dir.exists():
            return []
        return sorted(list(spec_dir.glob("*.yaml")) + list(spec_dir.glob("*.yml")))

    def load_yaml(self, yaml_path: Path) -> list[DatasetEntry]:
        """Parse a YAML spec file and extract DatasetEntry objects."""
        if not _YAML_AVAILABLE:
            logger.warning("PyYAML not available. Cannot parse YAML specs. Using empty specs.")
            return []

        try:
            with open(yaml_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to read or parse YAML spec at {yaml_path}: {e}")
            return []

        if not data or not isinstance(data, dict):
            return []

        stage = data.get("stage", "1")
        stage_name = data.get("stage_name", "unknown")
        raw_datasets = data.get("datasets", {})

        entries = []
        for name, spec in raw_datasets.items():
            if not isinstance(spec, dict):
                continue
            # Merge with top-level stage metadata
            spec_copy = spec.copy()
            spec_copy["name"] = name
            spec_copy["stage"] = stage
            spec_copy["stage_name"] = stage_name
            try:
                entry = DatasetEntry(**spec_copy)
                entries.append(entry)
            except Exception as e:
                logger.error(f"Failed to instantiate DatasetEntry for '{name}' in {yaml_path}: {e}")

        return entries

    def register(self, entry: DatasetEntry) -> None:
        """Add a dataset entry to the registry."""
        if entry.name in self.datasets:
            raise ValueError(f"Dataset '{entry.name}' is already registered.")

        if self._validate_on_register:
            self.validate(entry)

        self.datasets[entry.name] = entry
        logger.debug(f"Registered dataset '{entry.name}' successfully.")

    def validate(self, entry: DatasetEntry) -> None:
        """Validate fields on the DatasetEntry."""
        if not entry.name:
            raise ValueError("Dataset entry must have a valid 'name'.")
        if entry.source == "huggingface" and not entry.hf_id:
            raise ValueError(f"HuggingFace dataset '{entry.name}' must have a valid 'hf_id'.")
        if entry.source == "local" and not entry.path:
            raise ValueError(f"Local dataset '{entry.name}' must specify a 'path'.")
        if entry.priority not in ("S", "A", "B"):
            raise ValueError(f"Invalid priority '{entry.priority}' for dataset '{entry.name}'.")
        if entry.format not in ("pretrain", "sft", "preference"):
            raise ValueError(f"Invalid format '{entry.format}' for dataset '{entry.name}'.")

    def get(self, name: str) -> DatasetEntry:
        """Retrieve dataset entry by name."""
        if name not in self.datasets:
            raise KeyError(f"Dataset '{name}' not found in registry.")
        return self.datasets[name]

    def list_by_stage(self, stage: str | int) -> list[DatasetEntry]:
        """Filter dataset entries by training stage."""
        stage_str = str(stage)
        return [e for e in self.datasets.values() if str(e.stage) == stage_str]

    def list_by_priority(self, priority: str) -> list[DatasetEntry]:
        """Filter dataset entries by priority (S, A, B)."""
        return [e for e in self.datasets.values() if e.priority == priority]

    def all(self) -> dict[str, DatasetEntry]:
        """Return dict of all registered datasets."""
        return self.datasets.copy()

    def load_all_specs(self) -> int:
        """Scan and load all spec files from the configured directory."""
        yaml_files = self.discover_plugins(self.spec_dir)
        count = 0
        for yaml_path in yaml_files:
            entries = self.load_yaml(yaml_path)
            for entry in entries:
                try:
                    self.register(entry)
                    count += 1
                except Exception as e:
                    logger.error(f"Error registering entry '{entry.name}': {e}")
        logger.info(f"Loaded {count} dataset specifications from {self.spec_dir}.")
        return count


# ── Global default registry ──────────────────────────────────────────────────

_DEFAULT_REGISTRY: DataRegistry | None = None


def get_default_registry() -> DataRegistry:
    """Retrieve or create the global default DataRegistry."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = DataRegistry()
    return _DEFAULT_REGISTRY
