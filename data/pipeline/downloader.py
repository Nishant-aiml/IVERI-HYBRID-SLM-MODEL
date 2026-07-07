# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Dataset downloading and verification for the IVERI CORE pipeline.

Downloads HuggingFace datasets with retry loops and exponential backoffs,
resuming previous failures, writing metadata.json, and computing
MD5 + SHA256 hashes of the resulting directory to ensure data integrity.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Conditional imports for circular type checking
if TYPE_CHECKING:
    from data.pipeline.data_registry import DataRegistry, DatasetEntry

# Optional datasets import
import os
if os.environ.get("IVERI_DISABLE_HF", "0") == "1":
    _HF_AVAILABLE = False
    load_dataset = None  # type: ignore[assignment]
else:
    try:
        from datasets import load_dataset

        _HF_AVAILABLE = True
    except (ImportError, Exception):
        _HF_AVAILABLE = False
        load_dataset = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass(frozen=False, slots=True)
class DownloadMetadata:
    """Lineage metadata representing a downloaded dataset folder."""

    name: str
    hf_id: str | None
    downloaded_at: str
    num_rows: int | str
    sha256: str | None
    md5: str | None
    download_size_bytes: int | None
    status: str  # success, failed, resumed
    error_message: str | None = None


class DatasetDownloader:
    """Downloads HF datasets and manages metadata/checksums."""

    def __init__(
        self,
        save_dir: str | Path = "data/raw",
        cache_dir: str | Path | None = None,
        hf_token: str | None = None,
        max_retries: int = 3,
        verify_checksums: bool = True,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.hf_token = hf_token
        self.max_retries = max_retries
        self.verify_checksums = verify_checksums
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def is_downloaded(self, name: str) -> bool:
        """Return True if metadata exists and was successfully completed."""
        meta = self._load_metadata(name)
        return meta is not None and meta.status == "success"

    def _compute_dir_md5(self, path: Path) -> str:
        """Compute MD5 of all files in directory."""
        if not path.exists():
            return ""
        files = sorted([f for f in path.glob("**/*") if f.is_file() and f.name != "metadata.json"])
        md5_hash = hashlib.md5()
        for f in files:
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    md5_hash.update(chunk)
        return md5_hash.hexdigest()

    def _compute_dir_sha256(self, path: Path) -> str:
        """Compute SHA256 of all files in directory."""
        if not path.exists():
            return ""
        files = sorted([f for f in path.glob("**/*") if f.is_file() and f.name != "metadata.json"])
        sha256_hash = hashlib.sha256()
        for f in files:
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _load_metadata(self, name: str) -> DownloadMetadata | None:
        """Load metadata file if it exists."""
        meta_file = self.save_dir / name / "metadata.json"
        if not meta_file.exists():
            return None
        try:
            with open(meta_file, encoding="utf-8") as f:
                d = json.load(f)
            return DownloadMetadata(
                name=d["name"],
                hf_id=d["hf_id"],
                downloaded_at=d["downloaded_at"],
                num_rows=d["num_rows"],
                sha256=d.get("sha256"),
                md5=d.get("md5"),
                download_size_bytes=d.get("download_size_bytes"),
                status=d["status"],
                error_message=d.get("error_message"),
            )
        except Exception as e:
            logger.error(f"Failed to load downloader metadata for {name}: {e}")
            return None

    def _write_metadata(self, name: str, meta: DownloadMetadata) -> None:
        """Save metadata file."""
        meta_dir = self.save_dir / name
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_file = meta_dir / "metadata.json"
        d = {
            "name": meta.name,
            "hf_id": meta.hf_id,
            "downloaded_at": meta.downloaded_at,
            "num_rows": meta.num_rows,
            "sha256": meta.sha256,
            "md5": meta.md5,
            "download_size_bytes": meta.download_size_bytes,
            "status": meta.status,
            "error_message": meta.error_message,
        }
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=4)

    def _load_hf_dataset(self, hf_id: str, hf_config: str | None, split: str) -> Any:
        """Load dataset from HuggingFace hub (isolated for testing/mocking)."""
        if not _HF_AVAILABLE or load_dataset is None:
            raise ImportError(
                "HuggingFace datasets library is not installed. "
                "Run 'pip install datasets' to download."
            )
        kwargs: dict[str, Any] = {}
        if self.cache_dir:
            kwargs["cache_dir"] = str(self.cache_dir)
        if self.hf_token:
            kwargs["token"] = self.hf_token

        if hf_config:
            return load_dataset(hf_id, hf_config, split=split, **kwargs)
        return load_dataset(hf_id, split=split, **kwargs)

    def download(self, name: str, entry: DatasetEntry, force: bool = False) -> DownloadMetadata:
        """Download dataset based on specification."""
        if not force and self.is_downloaded(name):
            logger.info(f"Dataset '{name}' already downloaded. Skipping.")
            return self._load_metadata(name)  # type: ignore[return-value]

        dest_dir = self.save_dir / name
        logger.info(f"Downloading dataset '{name}' from HF ID: {entry.hf_id} to {dest_dir}...")

        if entry.source == "local":
            # For local files, we just check if it exists and write success metadata
            path_exists = entry.path and Path(entry.path).exists()
            status = "success" if path_exists else "failed"
            error = None if path_exists else f"Local path '{entry.path}' not found."
            meta = DownloadMetadata(
                name=name,
                hf_id=None,
                downloaded_at=datetime.now().isoformat(),
                num_rows="unknown",
                sha256=(
                    self._compute_dir_sha256(Path(entry.path))
                    if (path_exists and entry.path)
                    else None
                ),
                md5=(
                    self._compute_dir_md5(Path(entry.path))
                    if (path_exists and entry.path)
                    else None
                ),
                download_size_bytes=None,
                status=status,
                error_message=error,
            )
            self._write_metadata(name, meta)
            return meta

        # HF download with retry logic
        retries = 0
        backoff = 2.0
        last_error = ""

        while retries <= self.max_retries:
            try:
                ds = self._load_hf_dataset(entry.hf_id or "", entry.hf_config, entry.hf_split)
                # Save to disk
                ds.save_to_disk(str(dest_dir))
                logger.info(f"Successfully saved '{name}' to {dest_dir}")

                # Compute checksums
                md5 = self._compute_dir_md5(dest_dir) if self.verify_checksums else None
                sha256 = self._compute_dir_sha256(dest_dir) if self.verify_checksums else None

                # Compute byte size
                size_bytes = sum(f.stat().st_size for f in dest_dir.glob("**/*") if f.is_file())

                meta = DownloadMetadata(
                    name=name,
                    hf_id=entry.hf_id,
                    downloaded_at=datetime.now().isoformat(),
                    num_rows=len(ds),
                    sha256=sha256,
                    md5=md5,
                    download_size_bytes=size_bytes,
                    status="success",
                )
                self._write_metadata(name, meta)
                return meta
            except Exception as e:
                retries += 1
                last_error = str(e)
                if retries <= self.max_retries:
                    logger.warning(
                        f"Download failed for '{name}' (attempt {retries}/{self.max_retries+1}): {e}. "
                        f"Retrying in {backoff} seconds..."
                    )
                    time.sleep(backoff)
                    backoff *= 2.0
                else:
                    logger.error(
                        f"Download failed for '{name}' after {self.max_retries+1} attempts."
                    )

        meta = DownloadMetadata(
            name=name,
            hf_id=entry.hf_id,
            downloaded_at=datetime.now().isoformat(),
            num_rows=0,
            sha256=None,
            md5=None,
            download_size_bytes=None,
            status="failed",
            error_message=last_error,
        )
        self._write_metadata(name, meta)
        return meta

    def download_by_registry(
        self, name: str, registry: DataRegistry, force: bool = False
    ) -> DownloadMetadata:
        """Download dataset look up by name in registry."""
        entry = registry.get(name)
        return self.download(name, entry, force=force)

    def download_all_by_stage(
        self,
        stage: str | int,
        registry: DataRegistry,
        force: bool = False,
    ) -> dict[str, DownloadMetadata]:
        """Download all datasets specified for a stage."""
        entries = registry.list_by_stage(stage)
        results = {}
        for entry in entries:
            results[entry.name] = self.download(entry.name, entry, force=force)
        return results

    def resume(self, name: str, registry: DataRegistry) -> DownloadMetadata:
        """Resume a failed download."""
        meta = self._load_metadata(name)
        if meta and meta.status == "success":
            logger.info(f"Dataset '{name}' is already successfully downloaded.")
            return meta

        logger.info(f"Resuming download for dataset '{name}'...")
        return self.download_by_registry(name, registry, force=True)
