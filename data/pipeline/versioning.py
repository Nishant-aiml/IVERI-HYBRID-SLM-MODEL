# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Dataset version management, checksumming, and manifest generation.

Guarantees complete reproducibility of experiments by storing hash-stamps and
configuration snapshots alongside processed datasets.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=False, slots=True)
class VersionInfo:
    """Attributes defining a saved dataset version."""

    version_id: str
    dataset_name: str
    created_at: str
    config_hash: str
    content_hash: str
    pipeline_hash: str
    document_count: int = 0
    byte_count: int = 0
    stage: str = "unknown"
    source_datasets: list[str] = field(default_factory=list)
    processing_steps: list[str] = field(default_factory=list)


@dataclass(frozen=False, slots=True)
class ManifestEntry:
    """Representing an entry in the master data manifest."""

    dataset_name: str
    version: str
    license: str
    sha256: str
    pipeline_version: str
    creation_time: str
    document_count: int
    byte_count: int
    stage: str
    source: str
    mixing_weight: float


class DatasetVersioner:
    """Manages version info, SHA256/MD5 hashing, and manifest files."""

    def __init__(self, pipeline_version: str = "3.0.0") -> None:
        self.pipeline_version = pipeline_version

    def compute_file_sha256(self, path: Path) -> str:
        """Compute the SHA256 checksum of a single file."""
        sha256_hash = hashlib.sha256()
        with open(path, "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def compute_content_hash(self, data_dir: Path) -> str:
        """Compute combined SHA256 of all files in a directory (sorted by path)."""
        if not data_dir.exists():
            return ""

        # Find files recursively and sort to ensure deterministic hashing
        files = sorted(
            [f for f in data_dir.glob("**/*") if f.is_file() and f.name != "VERSION.json"]
        )

        sha256_hash = hashlib.sha256()
        for f in files:
            # Hash relative path to prevent local path absolute variation
            rel_path = f.relative_to(data_dir).as_posix()
            sha256_hash.update(rel_path.encode("utf-8"))

            # Hash file contents
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    def compute_pipeline_hash(self, config: dict[str, Any]) -> str:
        """Compute MD5 of a configuration dictionary, sorting keys for determinism."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode("utf-8")).hexdigest()

    def create_version(
        self,
        name: str,
        data_path: str | Path,
        config: dict[str, Any],
        document_count: int = 0,
        byte_count: int = 0,
        stage: str = "unknown",
        source_datasets: list[str] | None = None,
        processing_steps: list[str] | None = None,
    ) -> VersionInfo:
        """Create a new VERSION.json file in the target directory."""
        data_path = Path(data_path)
        data_path.mkdir(parents=True, exist_ok=True)

        config_hash = self.compute_pipeline_hash(config)
        content_hash = self.compute_content_hash(data_path)

        # Combined version ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"v_{timestamp}_{content_hash[:8]}"

        info = VersionInfo(
            version_id=version_id,
            dataset_name=name,
            created_at=datetime.now().isoformat(),
            config_hash=config_hash,
            content_hash=content_hash,
            pipeline_hash=config_hash,
            document_count=document_count,
            byte_count=byte_count,
            stage=str(stage),
            source_datasets=source_datasets or [name],
            processing_steps=processing_steps or [],
        )

        # Save to disk
        version_file = data_path / "VERSION.json"
        with open(version_file, "w", encoding="utf-8") as f:
            # Serialize the dataclass fields
            serializable = {
                "version_id": info.version_id,
                "dataset_name": info.dataset_name,
                "created_at": info.created_at,
                "config_hash": info.config_hash,
                "content_hash": info.content_hash,
                "pipeline_hash": info.pipeline_hash,
                "document_count": info.document_count,
                "byte_count": info.byte_count,
                "stage": info.stage,
                "source_datasets": info.source_datasets,
                "processing_steps": info.processing_steps,
            }
            json.dump(serializable, f, indent=4)

        logger.info(f"Saved dataset version metadata to {version_file}")
        return info

    def load_version(self, data_path: str | Path) -> VersionInfo:
        """Load version info from VERSION.json in target directory."""
        data_path = Path(data_path)
        version_file = data_path / "VERSION.json"
        if not version_file.exists():
            raise FileNotFoundError(f"VERSION.json not found in {data_path}")

        with open(version_file, encoding="utf-8") as f:
            data = json.load(f)

        return VersionInfo(
            version_id=data["version_id"],
            dataset_name=data["dataset_name"],
            created_at=data["created_at"],
            config_hash=data["config_hash"],
            content_hash=data["content_hash"],
            pipeline_hash=data["pipeline_hash"],
            document_count=data.get("document_count", 0),
            byte_count=data.get("byte_count", 0),
            stage=data.get("stage", "unknown"),
            source_datasets=data.get("source_datasets", []),
            processing_steps=data.get("processing_steps", []),
        )

    def assert_version_exists(self, data_path: str | Path) -> None:
        """Verify that VERSION.json is present in the target directory."""
        data_path = Path(data_path)
        version_file = data_path / "VERSION.json"
        if not version_file.exists():
            raise AssertionError(f"VERSION.json is missing in {data_path}")

    def write_manifest(self, output_dir: Path, entries: list[ManifestEntry]) -> Path:
        """Save a list of manifest entries to manifest.json."""
        manifest_path = output_dir / "manifest.json"
        data = []
        for e in entries:
            data.append(
                {
                    "dataset_name": e.dataset_name,
                    "version": e.version,
                    "license": e.license,
                    "sha256": e.sha256,
                    "pipeline_version": e.pipeline_version,
                    "creation_time": e.creation_time,
                    "document_count": e.document_count,
                    "byte_count": e.byte_count,
                    "stage": e.stage,
                    "source": e.source,
                    "mixing_weight": e.mixing_weight,
                }
            )
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Wrote master manifest to {manifest_path}")
        return manifest_path

    def load_manifest(self, manifest_path: Path) -> list[ManifestEntry]:
        """Load manifest entries from manifest.json."""
        if not manifest_path.exists():
            return []
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        entries = []
        for d in data:
            entries.append(
                ManifestEntry(
                    dataset_name=d["dataset_name"],
                    version=d["version"],
                    license=d["license"],
                    sha256=d["sha256"],
                    pipeline_version=d["pipeline_version"],
                    creation_time=d["creation_time"],
                    document_count=d["document_count"],
                    byte_count=d["byte_count"],
                    stage=d["stage"],
                    source=d["source"],
                    mixing_weight=d["mixing_weight"],
                )
            )
        return entries
