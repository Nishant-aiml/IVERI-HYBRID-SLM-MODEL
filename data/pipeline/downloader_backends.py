# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Multi-source downloader backends for Phase 8.

Abstracts the download source (HuggingFace, Local Folder, HTTP, Kaggle, S3).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from configs.phase8_config import DatasetBudget
    from data.pipeline.streaming import DownloadResult

logger = logging.getLogger(__name__)


class DownloaderBackend(ABC):
    """Abstract base class for download backends."""

    @abstractmethod
    def download(
        self,
        spec: DatasetBudget,
        output_dir: Path,
        max_storage_gb: float,
        split: str = "train",
        quality_filter: Any = None,
        resume: bool = True,
    ) -> DownloadResult:
        """Download (stream) dataset to output_dir up to max_storage_gb budget."""
        pass


class HuggingFaceBackend(DownloaderBackend):
    """Streams data from HuggingFace Hub."""

    def __init__(self, downloader: Any) -> None:
        self.downloader = downloader

    def download(
        self,
        spec: DatasetBudget,
        output_dir: Path,
        max_storage_gb: float,
        split: str = "train",
        quality_filter: Any = None,
        resume: bool = True,
    ) -> DownloadResult:
        # Delegate to downloader
        return self.downloader._download_huggingface(
            spec=spec,
            output_dir=output_dir,
            max_storage_gb=max_storage_gb,
            split=split,
            quality_filter=quality_filter,
            resume=resume,
            start_time=0.0,
        )


class LocalFolderBackend(DownloaderBackend):
    """Reads data from a local folder."""

    def __init__(self, downloader: Any) -> None:
        self.downloader = downloader

    def download(
        self,
        spec: DatasetBudget,
        output_dir: Path,
        max_storage_gb: float,
        split: str = "train",
        quality_filter: Any = None,
        resume: bool = True,
    ) -> DownloadResult:
        return self.downloader._download_local(
            spec=spec,
            output_dir=output_dir,
            max_storage_gb=max_storage_gb,
            quality_filter=quality_filter,
            resume=resume,
            start_time=0.0,
        )


class HTTPBackend(DownloaderBackend):
    """Placeholder for downloading via HTTP."""

    def download(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("HTTP backend is not implemented.")


class KaggleBackend(DownloaderBackend):
    """Placeholder for downloading via Kaggle API."""

    def download(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Kaggle backend is not implemented.")


class S3Backend(DownloaderBackend):
    """Placeholder for downloading via AWS S3."""

    def download(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("S3 backend is not implemented.")
