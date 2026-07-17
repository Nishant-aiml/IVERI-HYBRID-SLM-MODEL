# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Resume-capable streaming downloader for Phase 8 dataset acquisition.

Streams documents from HuggingFace datasets one shard at a time, applies
in-flight quality filtering, respects per-dataset storage budgets, and saves
progress checkpoints so interrupted downloads can be resumed. Never performs
a full dataset download — always uses streaming=True.

Offline mode: checks data/raw/<dataset>/ then .hf_cache/ for cached data;
raises OfflineModeError with a descriptive message if neither is found.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from configs.phase8_config import DatasetBudget, Phase8Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional HuggingFace datasets import
# ---------------------------------------------------------------------------
import os
if os.environ.get("IVERI_DISABLE_HF", "0") == "1" or os.name == "nt":
    hf_datasets = None  # type: ignore[assignment]
    _HF_AVAILABLE = False
else:
    try:
        import datasets as hf_datasets

        _HF_AVAILABLE = True
    except ImportError:
        hf_datasets = None  # type: ignore[assignment]
        _HF_AVAILABLE = False
        logger.warning(
            "HuggingFace 'datasets' package not installed. "
            "Streaming downloads are disabled. Run 'pip install datasets' to enable."
        )


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class OfflineModeError(Exception):
    """Raised in offline mode when a dataset is not locally cached."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DownloadProgress:
    """Checkpoint state for a resumable streaming download."""

    dataset: str
    started_at: str
    last_updated: str
    shards_completed: list[int]
    rows_written: int
    bytes_written: int
    status: str  # "in_progress" | "complete" | "budget_hit"

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation."""
        return {
            "dataset": self.dataset,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "shards_completed": self.shards_completed,
            "rows_written": self.rows_written,
            "bytes_written": self.bytes_written,
            "status": self.status,
        }

    def to_json(self) -> str:
        """Return a JSON string representation."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class DownloadResult:
    """Summary of a completed (or budget-stopped) streaming download."""

    dataset_name: str
    num_docs: int
    num_bytes: int
    num_filtered: int
    num_budget_stops: int
    elapsed_sec: float
    budget_hit: bool
    output_path: Path
    completed: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation."""
        return {
            "dataset_name": self.dataset_name,
            "num_docs": self.num_docs,
            "num_bytes": self.num_bytes,
            "num_filtered": self.num_filtered,
            "num_budget_stops": self.num_budget_stops,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "budget_hit": self.budget_hit,
            "output_path": str(self.output_path),
            "completed": self.completed,
            "error": self.error,
        }

    def to_json(self) -> str:
        """Return a JSON string representation."""
        return json.dumps(self.to_dict(), indent=2)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROGRESS_FILENAME = ".download_progress.json"
_BUDGET_CHECK_INTERVAL = 1000  # rows between budget checks
_MAX_RETRY_ATTEMPTS = 3
_RETRY_DELAY_SEC = 5.0


# ---------------------------------------------------------------------------
# StreamingDownloader
# ---------------------------------------------------------------------------

class StreamingDownloader:
    """Stream from HuggingFace (or local), filter in-flight, stop at budget, support resume."""

    def __init__(self, config: Phase8Config) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_to_budget(
        self,
        spec: DatasetBudget,
        output_dir: Path,
        max_storage_gb: float,
        split: str = "train",
        quality_filter: Any = None,
        resume: bool = True,
        source_type: str = "huggingface",
    ) -> DownloadResult:
        """Stream from a backend, filter in-flight, stop at budget.

        Args:
            spec: DatasetBudget specification (hf_id, revision, config, …).
            output_dir: Directory for raw JSONL output.
            max_storage_gb: Hard storage cap — streaming stops when reached.
            split: Dataset split to stream (default "train").
            quality_filter: Optional QualityFilter instance for in-flight filtering.
            resume: If True, continue an interrupted download.
            source_type: Backend to use — ``"huggingface"`` (default) or ``"local"``.

        Returns:
            DownloadResult summarising what was written.
        """
        start_time = time.monotonic()
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{spec.name}.jsonl"

        # --- Offline mode guard ---
        if self.config.offline_mode:
            return self._handle_offline(spec, output_dir, output_file, start_time)

        # Route to backend
        if source_type == "local":
            return self._download_local(
                spec=spec,
                output_dir=output_dir,
                output_file=output_file,
                max_storage_gb=max_storage_gb,
                quality_filter=quality_filter,
                resume=resume,
                start_time=start_time,
            )
        # Default: huggingface
        return self._download_huggingface(
            spec=spec,
            output_dir=output_dir,
            output_file=output_file,
            max_storage_gb=max_storage_gb,
            split=split,
            quality_filter=quality_filter,
            resume=resume,
            start_time=start_time,
        )

    # ------------------------------------------------------------------
    # Offline mode
    # ------------------------------------------------------------------

    def _handle_offline(
        self,
        spec: DatasetBudget,
        output_dir: Path,
        output_file: Path,
        start_time: float,
    ) -> DownloadResult:
        """Return cached result or raise OfflineModeError."""
        # Check 1: raw staging dir
        if output_file.exists() and output_file.stat().st_size > 0:
            logger.info("Offline mode: using cached raw data for '%s' at %s.", spec.name, output_file)
            progress = self._load_progress(output_dir, spec.name)
            if progress:
                rows = progress.rows_written
                bts = progress.bytes_written
            else:
                try:
                    with open(output_file, encoding="utf-8") as f:
                        rows = sum(1 for line in f if line.strip())
                except Exception:
                    rows = 0
                bts = output_file.stat().st_size
            return DownloadResult(
                dataset_name=spec.name,
                num_docs=rows,
                num_bytes=bts,
                num_filtered=0,
                num_budget_stops=0,
                elapsed_sec=time.monotonic() - start_time,
                budget_hit=False,
                output_path=output_file,
                completed=True,
            )

        # Check 2: HF cache directory
        cache_dir = Path(self.config.cache_dir)
        cache_candidates = list(cache_dir.rglob(f"*{spec.name}*.jsonl")) if cache_dir.exists() else []
        if cache_candidates:
            cached_path = cache_candidates[0]
            logger.info("Offline mode: found HF cache hit for '%s' at %s.", spec.name, cached_path)
            bts = cached_path.stat().st_size
            return DownloadResult(
                dataset_name=spec.name,
                num_docs=0,  # Unknown without parsing
                num_bytes=bts,
                num_filtered=0,
                num_budget_stops=0,
                elapsed_sec=time.monotonic() - start_time,
                budget_hit=False,
                output_path=cached_path,
                completed=True,
            )

        raise OfflineModeError(
            f"Offline mode is active but dataset '{spec.name}' was not found locally.\n"
            f"  Checked raw staging: {output_file}\n"
            f"  Checked HF cache:    {cache_dir}\n"
            "Set IVERI_OFFLINE=0 or run with internet access to download the dataset."
        )

    # ------------------------------------------------------------------
    # HuggingFace backend
    # ------------------------------------------------------------------

    def _download_huggingface(
        self,
        spec: DatasetBudget,
        output_dir: Path,
        output_file: Path,
        max_storage_gb: float,
        split: str,
        quality_filter: Any,
        resume: bool,
        start_time: float,
    ) -> DownloadResult:
        """Download via HuggingFace streaming API."""
        if not _HF_AVAILABLE:
            logger.warning("HuggingFace datasets unavailable; returning empty result for '%s'.", spec.name)
            return DownloadResult(
                dataset_name=spec.name,
                num_docs=0,
                num_bytes=0,
                num_filtered=0,
                num_budget_stops=0,
                elapsed_sec=time.monotonic() - start_time,
                budget_hit=False,
                output_path=output_file,
                completed=False,
                error="datasets package not installed",
            )

        # Resume logic
        progress, skip_rows, file_mode = self._setup_resume(output_dir, spec.name, resume)
        max_storage_bytes = int(max_storage_gb * (1024 ** 3))
        num_docs = progress.rows_written
        num_bytes = progress.bytes_written
        num_filtered = 0
        num_budget_stops = 0
        budget_hit = False

        for attempt in range(1, _MAX_RETRY_ATTEMPTS + 1):
            try:
                result = self._stream_hf_once(
                    spec=spec,
                    split=split,
                    output_file=output_file,
                    file_mode=file_mode,
                    skip_rows=skip_rows,
                    max_storage_bytes=max_storage_bytes,
                    quality_filter=quality_filter,
                    progress=progress,
                    start_docs=num_docs,
                    start_bytes=num_bytes,
                )
                num_docs = result["num_docs"]
                num_bytes = result["num_bytes"]
                num_filtered = result["num_filtered"]
                num_budget_stops = result["num_budget_stops"]
                budget_hit = result["budget_hit"]
                progress.status = "budget_hit" if budget_hit else "complete"
                progress.rows_written = num_docs
                progress.bytes_written = num_bytes
                progress.last_updated = datetime.now().isoformat()
                self._save_progress(output_dir, progress)
                break

            except (ConnectionError, OSError) as exc:
                logger.warning(
                    "Download attempt %d/%d for '%s' failed: %s",
                    attempt, _MAX_RETRY_ATTEMPTS, spec.name, exc,
                )
                if attempt < _MAX_RETRY_ATTEMPTS:
                    logger.info("Retrying in %.0f seconds…", _RETRY_DELAY_SEC)
                    time.sleep(_RETRY_DELAY_SEC)
                    file_mode = "a"
                    skip_rows = num_docs
                else:
                    elapsed = time.monotonic() - start_time
                    return DownloadResult(
                        dataset_name=spec.name,
                        num_docs=num_docs,
                        num_bytes=num_bytes,
                        num_filtered=num_filtered,
                        num_budget_stops=num_budget_stops,
                        elapsed_sec=elapsed,
                        budget_hit=budget_hit,
                        output_path=output_file,
                        completed=False,
                        error=str(exc),
                    )
            except Exception as exc:
                logger.error("Unexpected error downloading '%s': %s", spec.name, exc, exc_info=True)
                elapsed = time.monotonic() - start_time
                return DownloadResult(
                    dataset_name=spec.name,
                    num_docs=num_docs,
                    num_bytes=num_bytes,
                    num_filtered=num_filtered,
                    num_budget_stops=num_budget_stops,
                    elapsed_sec=elapsed,
                    budget_hit=budget_hit,
                    output_path=output_file,
                    completed=False,
                    error=str(exc),
                )

        elapsed = time.monotonic() - start_time
        logger.info(
            "Download '%s': %d docs, %.2f MB, %d filtered, budget_hit=%s, elapsed=%.1fs",
            spec.name, num_docs, num_bytes / (1024 ** 2), num_filtered, budget_hit, elapsed,
        )
        return DownloadResult(
            dataset_name=spec.name,
            num_docs=num_docs,
            num_bytes=num_bytes,
            num_filtered=num_filtered,
            num_budget_stops=num_budget_stops,
            elapsed_sec=elapsed,
            budget_hit=budget_hit,
            output_path=output_file,
            completed=not budget_hit,
        )

    # ------------------------------------------------------------------
    # Local backend
    # ------------------------------------------------------------------

    def _download_local(
        self,
        spec: DatasetBudget,
        output_dir: Path,
        output_file: Path,
        max_storage_gb: float,
        quality_filter: Any,
        resume: bool,
        start_time: float,
    ) -> DownloadResult:
        """Read from a local JSONL source file and copy to the staging area.

        The local source path is resolved as:
            ``data/raw/{spec.name}/{spec.name}.jsonl``  (within the raw_data_dir)
        """
        source_path = (Path(self.config.raw_data_dir) / spec.name / f"{spec.name}.jsonl").resolve()
        output_file = output_file.resolve()
        if not source_path.exists():
            logger.warning("Local source not found for '%s': %s", spec.name, source_path)
            return DownloadResult(
                dataset_name=spec.name,
                num_docs=0,
                num_bytes=0,
                num_filtered=0,
                num_budget_stops=0,
                elapsed_sec=time.monotonic() - start_time,
                budget_hit=False,
                output_path=output_file,
                completed=False,
                error=f"Local source not found: {source_path}",
            )

        # Source and staging target are the same file — skip copy (opening "w" would truncate).
        if source_path == output_file:
            num_docs = 0
            num_bytes = 0
            with open(source_path, encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = row.get(spec.text_field, "")
                    if isinstance(text, str) and text.strip():
                        num_docs += 1
                        num_bytes += self._estimate_size(text)
            elapsed = time.monotonic() - start_time
            logger.info(
                "Local source '%s' already staged at %s: %d docs, %.2f MB",
                spec.name,
                output_file,
                num_docs,
                num_bytes / (1024 ** 2),
            )
            return DownloadResult(
                dataset_name=spec.name,
                num_docs=num_docs,
                num_bytes=num_bytes,
                num_filtered=0,
                num_budget_stops=0,
                elapsed_sec=elapsed,
                budget_hit=False,
                output_path=output_file,
                completed=True,
            )

        _, skip_rows, file_mode = self._setup_resume(output_dir, spec.name, resume)
        max_storage_bytes = int(max_storage_gb * (1024 ** 3))
        num_docs = skip_rows
        num_bytes = 0
        num_filtered = 0
        budget_hit = False

        try:
            with (
                open(source_path, encoding="utf-8") as fin,
                open(output_file, file_mode, encoding="utf-8") as fout,
            ):
                for line_idx, line in enumerate(fin):
                    if line_idx < skip_rows:
                        continue
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        num_filtered += 1
                        continue

                    text = row.get(spec.text_field, "")
                    if not isinstance(text, str) or not text.strip():
                        num_filtered += 1
                        continue

                    if quality_filter is not None:
                        try:
                            if not quality_filter.min_length_filter(text):
                                num_filtered += 1
                                continue
                        except Exception:
                            pass

                    line_bytes = self._estimate_size(text)
                    fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    num_docs += 1
                    num_bytes += line_bytes

                    if num_bytes >= max_storage_bytes:
                        budget_hit = True
                        break

        except OSError as exc:
            logger.error("Local read failed for '%s': %s", spec.name, exc)
            return DownloadResult(
                dataset_name=spec.name,
                num_docs=num_docs,
                num_bytes=num_bytes,
                num_filtered=num_filtered,
                num_budget_stops=1 if budget_hit else 0,
                elapsed_sec=time.monotonic() - start_time,
                budget_hit=budget_hit,
                output_path=output_file,
                completed=False,
                error=str(exc),
            )

        elapsed = time.monotonic() - start_time
        logger.info(
            "Local copy '%s': %d docs, %.2f MB, %d filtered, elapsed=%.1fs",
            spec.name, num_docs, num_bytes / (1024 ** 2), num_filtered, elapsed,
        )
        return DownloadResult(
            dataset_name=spec.name,
            num_docs=num_docs,
            num_bytes=num_bytes,
            num_filtered=num_filtered,
            num_budget_stops=1 if budget_hit else 0,
            elapsed_sec=elapsed,
            budget_hit=budget_hit,
            output_path=output_file,
            completed=not budget_hit,
        )

    # ------------------------------------------------------------------
    # Internal HF streaming implementation
    # ------------------------------------------------------------------

    def _stream_hf_once(
        self,
        spec: DatasetBudget,
        split: str,
        output_file: Path,
        file_mode: str,
        skip_rows: int,
        max_storage_bytes: int,
        quality_filter: Any,
        progress: DownloadProgress,
        start_docs: int,
        start_bytes: int,
    ) -> dict[str, Any]:
        """Single HF streaming pass. Returns counters dict."""
        load_kwargs: dict[str, Any] = {
            "path": spec.hf_id,
            "split": split,
            "streaming": True,
            "trust_remote_code": False,
        }
        if spec.hf_config:
            load_kwargs["name"] = spec.hf_config
        if spec.hf_revision and spec.hf_revision != "main":
            load_kwargs["revision"] = spec.hf_revision

        dataset = hf_datasets.load_dataset(**load_kwargs)  # type: ignore[union-attr]

        num_docs = start_docs
        num_bytes = start_bytes
        num_filtered = 0
        num_budget_stops = 0
        budget_hit = False
        row_idx = 0

        with open(output_file, file_mode, encoding="utf-8") as fout:
            for row in dataset:
                if row_idx < skip_rows:
                    row_idx += 1
                    continue

                text = row.get(spec.text_field, "")
                if not isinstance(text, str) or not text.strip():
                    num_filtered += 1
                    row_idx += 1
                    continue

                if quality_filter is not None:
                    try:
                        if not quality_filter.min_length_filter(text):
                            num_filtered += 1
                            row_idx += 1
                            continue
                        if not quality_filter.alpha_ratio_filter(text):
                            num_filtered += 1
                            row_idx += 1
                            continue
                    except Exception as qexc:
                        logger.debug("Quality filter error on row %d: %s", row_idx, qexc)

                line_bytes = self._estimate_size(text)
                fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                num_docs += 1
                num_bytes += line_bytes
                row_idx += 1

                if num_bytes >= max_storage_bytes:
                    logger.info(
                        "Budget reached for '%s': %.2f MB >= cap %.2f MB.",
                        spec.name,
                        num_bytes / (1024 ** 2),
                        max_storage_bytes / (1024 ** 2),
                    )
                    budget_hit = True
                    num_budget_stops += 1
                    progress.rows_written = num_docs
                    progress.bytes_written = num_bytes
                    progress.last_updated = datetime.now().isoformat()
                    self._save_progress(output_file.parent, progress)
                    break

                if num_docs % _BUDGET_CHECK_INTERVAL == 0:
                    progress.rows_written = num_docs
                    progress.bytes_written = num_bytes
                    progress.last_updated = datetime.now().isoformat()
                    self._save_progress(output_file.parent, progress)

        return {
            "num_docs": num_docs,
            "num_bytes": num_bytes,
            "num_filtered": num_filtered,
            "num_budget_stops": num_budget_stops,
            "budget_hit": budget_hit,
        }

    # ------------------------------------------------------------------
    # Resume helpers
    # ------------------------------------------------------------------

    def _setup_resume(
        self, output_dir: Path, dataset_name: str, resume: bool
    ) -> tuple[DownloadProgress, int, str]:
        """Return *(progress, skip_rows, file_mode)* for a download run."""
        now_str = datetime.now().isoformat()
        if resume:
            progress = self._load_progress(output_dir, dataset_name)
            if progress and progress.status not in ("complete",):
                skip_rows = progress.rows_written
                file_mode = "a"
                logger.info(
                    "Resuming '%s' from row %d (%.2f MB written).",
                    dataset_name,
                    skip_rows,
                    progress.bytes_written / (1024 ** 2),
                )
                return progress, skip_rows, file_mode

        fresh_progress = DownloadProgress(
            dataset=dataset_name,
            started_at=now_str,
            last_updated=now_str,
            shards_completed=[],
            rows_written=0,
            bytes_written=0,
            status="in_progress",
        )
        return fresh_progress, 0, "w"

    def _load_progress(self, output_dir: Path, dataset_name: str) -> DownloadProgress | None:
        """Load existing *.download_progress.json* if present and valid."""
        progress_file = output_dir / _PROGRESS_FILENAME
        if not progress_file.exists():
            return None
        try:
            data = json.loads(progress_file.read_text(encoding="utf-8"))
            if data.get("dataset") != dataset_name:
                return None
            return DownloadProgress(
                dataset=data["dataset"],
                started_at=data.get("started_at", ""),
                last_updated=data.get("last_updated", ""),
                shards_completed=data.get("shards_completed", []),
                rows_written=data.get("rows_written", 0),
                bytes_written=data.get("bytes_written", 0),
                status=data.get("status", "in_progress"),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Could not parse progress file %s: %s", progress_file, exc)
            return None

    def _save_progress(self, output_dir: Path, progress: DownloadProgress) -> None:
        """Save *.download_progress.json* after each checkpoint."""
        progress_file = output_dir / _PROGRESS_FILENAME
        try:
            progress_file.write_text(progress.to_json(), encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not save progress checkpoint: %s", exc)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_size(text: str) -> int:
        """Estimate bytes when encoded as a UTF-8 JSONL line."""
        return len(json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")) + 1
