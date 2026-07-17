# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Parallel download manager for downloading multiple datasets concurrently in Phase 8.

Uses ThreadPoolExecutor to download datasets in priority order, respecting
the storage budget constraint at all times.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from configs.phase8_config import DatasetBudget, Phase8Config
    from data.pipeline.storage import StorageManager
    from data.pipeline.streaming import DownloadResult, StreamingDownloader

logger = logging.getLogger(__name__)


class ParallelDownloadManager:
    """Manages parallel dataset downloading using a ThreadPoolExecutor."""

    def __init__(self, config: Phase8Config, max_concurrent: int = 2) -> None:
        self.config = config
        self.max_concurrent = max_concurrent

    def download_all_by_priority(
        self,
        specs: list[DatasetBudget],
        downloader: StreamingDownloader,
        storage_manager: StorageManager,
        base_output_dir: Path,
    ) -> dict[str, DownloadResult]:
        """Download multiple datasets in priority order, max_concurrent at a time."""
        # Sort by priority ascending (1 = highest priority)
        sorted_specs = sorted(specs, key=lambda x: x.priority)
        results: dict[str, DownloadResult] = {}

        budget = storage_manager.compute_budget()
        logger.info(
            "Starting parallel download manager. Usable budget: %.2f GB",
            budget.usable_gb,
        )

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # We process them in batches or submit them sequentially as workers free up
            futures = {}
            for spec in sorted_specs:
                if not spec.enabled:
                    logger.info("Dataset '%s' is disabled, skipping.", spec.name)
                    continue

                # Pre-flight check: do we have disk budget?
                summary = storage_manager.get_summary()
                if summary.total_used_gb >= budget.usable_gb:
                    logger.warning("Total usable storage budget exhausted! Skipping remaining downloads.")
                    break

                output_dir = base_output_dir / spec.name
                max_cap = min(spec.max_storage_gb, budget.usable_gb - summary.total_used_gb)

                if max_cap <= 0.0:
                    logger.warning("Insufficient storage budget remaining to download '%s'.", spec.name)
                    break

                logger.info(
                    "Submitting download for '%s' (cap=%.2f GB, priority=%d)",
                    spec.name, max_cap, spec.priority,
                )

                # Submit to thread pool
                future = executor.submit(
                    downloader.download_to_budget,
                    spec=spec,
                    output_dir=output_dir,
                    max_storage_gb=max_cap,
                    split="train",
                    resume=self.config.resume_downloads,
                    source_type=spec.source_type,
                )
                futures[future] = spec.name

            for future in as_completed(futures):
                name = futures[future]
                try:
                    res = future.result()
                    results[name] = res
                    logger.info("Download completed for '%s'. Status: %s", name, "success" if res.completed else "failed/budget_stop")
                except Exception as exc:
                    logger.error("Download thread for '%s' raised exception: %s", name, exc, exc_info=True)

        return results
