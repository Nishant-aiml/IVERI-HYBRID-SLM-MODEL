# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Download cache manager for Phase 8 data engineering.

Prevents re-downloading previously verified and certified datasets.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from configs.phase8_config import DatasetBudget

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A registered entry in the local download cache."""

    dataset_name: str
    hf_revision: str
    download_date: str
    size_bytes: int
    sha256_sample: str | None
    completed: bool


class DownloadCacheManager:
    """Manages cached HuggingFace/local datasets to prevent redundant downloads."""

    def __init__(self, cache_dir: str | Path = ".hf_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.cache_dir / "cache_registry.json"
        self.registry: dict[str, dict[str, Any]] = self._load_registry()

    def _load_registry(self) -> dict[str, dict[str, Any]]:
        if not self.registry_path.exists():
            return {}
        try:
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load cache registry: %s", exc)
            return {}

    def _save_registry(self) -> None:
        try:
            self.registry_path.write_text(
                json.dumps(self.registry, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Failed to save cache registry: %s", exc)

    def is_cached(self, spec: DatasetBudget) -> bool:
        """Return True if the dataset is cached with the correct revision."""
        entry_data = self.registry.get(spec.name)
        if not entry_data:
            return False
        # Verify the revision match to guarantee version lock
        revision_match = entry_data.get("hf_revision") == spec.hf_revision
        completed = entry_data.get("completed", False)
        return revision_match and completed

    def get_cache_path(self, spec: DatasetBudget) -> Path | None:
        """Return the path to the cached dataset if it is valid."""
        if not self.is_cached(spec):
            return None
        path = self.cache_dir / spec.name
        return path if path.exists() else None

    def register_download(self, spec: DatasetBudget, size_bytes: int, sha256_sample: str | None = None) -> None:
        """Record a completed download in the cache registry."""
        self.registry[spec.name] = {
            "dataset_name": spec.name,
            "hf_revision": spec.hf_revision,
            "download_date": datetime.now().isoformat(),
            "size_bytes": size_bytes,
            "sha256_sample": sha256_sample,
            "completed": True,
        }
        self._save_registry()
        logger.info("Registered dataset '%s' in download cache.", spec.name)

    def verify_cache_integrity(self, spec: DatasetBudget) -> bool:
        """Check if cache contains valid data for the given spec."""
        cache_path = self.get_cache_path(spec)
        if not cache_path:
            return False
        # Verify directory has jsonl files and size is non-zero
        files = list(cache_path.glob("*.jsonl"))
        if not files:
            return False
        total_size = sum(f.stat().st_size for f in files)
        entry = self.registry.get(spec.name, {})
        return total_size > 0 and total_size == entry.get("size_bytes", 0)
