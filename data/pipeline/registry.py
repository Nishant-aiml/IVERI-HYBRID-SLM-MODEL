# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""YAML-discovered dataset registry for Phase 8.

Discovers dataset configs dynamically from YAML files located in configs/datasets/.
Prevents hardcoded lists in configuration code.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from configs.phase8_config import DatasetBudget

logger = logging.getLogger(__name__)


class DatasetRegistry:
    """Discovers and validates dataset specs from YAML files."""

    def __init__(self, specs_dir: Path | str = "configs/datasets") -> None:
        self.specs_dir = Path(specs_dir)
        self._cache: dict[str, DatasetBudget] = {}

    def discover(self) -> dict[str, DatasetBudget]:
        """Discover and load all enabled dataset YAML specifications."""
        if self._cache:
            return self._cache

        if not self.specs_dir.exists():
            logger.warning("Specs directory %s does not exist.", self.specs_dir)
            return {}

        import yaml

        discovered: dict[str, DatasetBudget] = {}
        for path in sorted(self.specs_dir.glob("*.yaml")):
            try:
                content = path.read_text(encoding="utf-8")
                raw = yaml.safe_load(content)
                if not raw:
                    continue
                
                # Check schema keys
                name = raw.get("name")
                if not name:
                    logger.warning("Skipping spec %s: missing 'name' field.", path)
                    continue

                if not raw.get("enabled", True):
                    logger.debug("Skipping disabled dataset: %s", name)
                    continue

                # Eval-only specs may omit quality fields; avoid float(None) crashes.
                raw_quality_backend = raw.get("quality_backend")
                raw_quality_threshold = raw.get("quality_threshold")
                quality_backend = (
                    "none" if raw_quality_backend is None and raw.get("stage") == "eval_only"
                    else (raw_quality_backend or "fast")
                )
                quality_threshold = (
                    0.0 if raw_quality_threshold is None else float(raw_quality_threshold)
                )

                # Instantiate spec
                spec = DatasetBudget(
                    name=name,
                    hf_id=raw.get("hf_id", ""),
                    hf_revision=raw.get("hf_revision", "main"),
                    hf_config=raw.get("hf_config"),
                    license=raw.get("license", "unknown"),
                    license_url=raw.get("license_url", ""),
                    source_url=raw.get("source_url", ""),
                    text_field=raw.get("text_field", "text"),
                    priority=raw.get("priority", 99),
                    stage=raw.get("stage", "pretrain"),
                    max_storage_gb=float(raw.get("max_storage_gb", 1.0)),
                    curriculum_phases=raw.get("curriculum_phases", []),
                    source_type=raw.get("source_type", "huggingface"),
                    enabled=raw.get("enabled", True),
                    quality_backend=quality_backend,
                    quality_threshold=quality_threshold,
                )
                discovered[name] = spec
            except Exception as exc:
                logger.error("Failed to load dataset spec %s: %s", path, exc)

        self._cache = discovered
        logger.info("Discovered %d dataset specifications.", len(discovered))
        return discovered

    def get(self, name: str) -> DatasetBudget:
        """Get dataset spec by name. Raises KeyError if not found."""
        specs = self.discover()
        if name not in specs:
            raise KeyError(f"Dataset '{name}' not found in registry.")
        return specs[name]

    def list_pretrain(self) -> list[DatasetBudget]:
        """List pretrain datasets."""
        return [s for s in self.discover().values() if s.stage == "pretrain"]

    def list_eval_only(self) -> list[DatasetBudget]:
        """List evaluation-only datasets."""
        return [s for s in self.discover().values() if s.stage == "eval_only"]

    def list_by_priority(self) -> list[DatasetBudget]:
        """List all discovered datasets ordered by priority."""
        return sorted(self.discover().values(), key=lambda x: x.priority)

    def validate_all(self) -> dict[str, list[str]]:
        """Validate all YAML files against required schema. Returns name -> error list."""
        errors: dict[str, list[str]] = {}
        for path in self.specs_dir.glob("*.yaml"):
            name = path.stem
            errs = []
            try:
                import yaml
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
                if not raw:
                    errs.append("Empty spec file")
                    errors[name] = errs
                    continue
                if "name" not in raw:
                    errs.append("Missing field: name")
                if "license" not in raw:
                    errs.append("Missing field: license")
                if "priority" not in raw:
                    errs.append("Missing field: priority")
                elif not isinstance(raw["priority"], int):
                    errs.append("Field priority must be an integer")
            except Exception as e:
                errs.append(f"YAML parse error: {e}")
            if errs:
                errors[name] = errs
        return errors
