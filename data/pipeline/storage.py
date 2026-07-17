# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Storage management, budget tracking, and disk usage reporting for Phase 8.

Tracks per-dataset disk usage across raw/processed/certified directories,
enforces dynamic storage budgets computed from available free space, and
generates structured JSON + Markdown reports. Raises EvalContaminationError
if evaluation-only datasets attempt to write into the certified training dir.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from configs.phase8_config import Phase8Config

logger = logging.getLogger(__name__)


class EvalContaminationError(Exception):
    """Raised when an eval-only dataset attempts to write to the certified training dir."""


@dataclass
class DatasetUsage:
    """Measured disk usage for a single dataset across pipeline stages."""

    name: str
    raw_gb: float
    processed_gb: float
    certified_gb: float
    total_gb: float

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation."""
        return {
            "name": self.name,
            "raw_gb": round(self.raw_gb, 4),
            "processed_gb": round(self.processed_gb, 4),
            "certified_gb": round(self.certified_gb, 4),
            "total_gb": round(self.total_gb, 4),
        }

    def to_json(self) -> str:
        """Return a JSON string representation."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class StorageBudget:
    """Dynamically computed storage budget for the current run."""

    total_free_gb: float
    reserved_gb: float
    usable_gb: float
    per_dataset_caps: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation."""
        return {
            "total_free_gb": round(self.total_free_gb, 4),
            "reserved_gb": round(self.reserved_gb, 4),
            "usable_gb": round(self.usable_gb, 4),
            "per_dataset_caps": {k: round(v, 4) for k, v in self.per_dataset_caps.items()},
        }

    def to_json(self) -> str:
        """Return a JSON string representation."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class StorageSummary:
    """Aggregate storage summary across all datasets."""

    datasets: list[DatasetUsage]
    total_used_gb: float
    budget_gb: float
    free_gb: float
    status: str  # "OK" | "WARNING" | "FULL"

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation."""
        return {
            "generated_at": datetime.now().isoformat(),
            "total_free_gb": round(self.free_gb, 4),
            "total_used_gb": round(self.total_used_gb, 4),
            "budget_gb": round(self.budget_gb, 4),
            "status": self.status,
            "datasets": [d.to_dict() for d in self.datasets],
        }

    def to_json(self) -> str:
        """Return a JSON string representation."""
        return json.dumps(self.to_dict(), indent=2)


def _dir_size_gb(path: Path) -> float:
    """Recursively sum file sizes under *path* and return gigabytes."""
    if not path.exists():
        return 0.0
    total_bytes = 0
    try:
        for root, _dirs, files in os.walk(path):
            for fname in files:
                try:
                    total_bytes += os.path.getsize(os.path.join(root, fname))
                except OSError:
                    pass
    except OSError as exc:
        logger.warning("Could not walk directory %s: %s", path, exc)
    return total_bytes / (1024 ** 3)


class StorageManager:
    """Tracks disk usage, enforces budgets, and prevents eval contamination."""

    def __init__(self, config: Phase8Config, base_dir: str | Path = ".") -> None:
        self.config = config
        self.base_dir = Path(base_dir)

    # ------------------------------------------------------------------
    # Budget / free-space helpers
    # ------------------------------------------------------------------

    def check_free_space(self, required_gb: float) -> tuple[bool, float]:
        """Return *(has_space, free_gb)*.  Never raises."""
        try:
            import shutil
            free_bytes = shutil.disk_usage(str(self.base_dir)).free
            free_gb = free_bytes / (1024 ** 3)
            has_space = (free_gb - required_gb) >= self.config.min_free_gb
            return has_space, free_gb
        except Exception as exc:
            logger.warning("Could not check free space: %s", exc)
            return False, 0.0

    def compute_budget(self) -> StorageBudget:
        """Compute dynamic budget: free_gb * (1 - reserve_fraction) - min_free_gb."""
        try:
            import shutil
            usage = shutil.disk_usage(str(self.base_dir))
            free_gb = usage.free / (1024 ** 3)
        except Exception as exc:
            logger.warning("disk_usage failed, defaulting to 0 GB free: %s", exc)
            free_gb = 0.0

        reserved_gb = max(free_gb * self.config.reserve_fraction, self.config.min_free_gb)
        usable_gb = max(free_gb - reserved_gb, 0.0)

        # Override with explicit cap if provided
        if self.config.max_total_budget_gb is not None:
            usable_gb = min(usable_gb, self.config.max_total_budget_gb)

        per_caps: dict[str, float] = {}
        for name, spec in self.config.dataset_budgets.items():
            per_caps[name] = min(spec.max_storage_gb, usable_gb)

        logger.info(
            "Storage budget: free=%.2f GB, reserved=%.2f GB, usable=%.2f GB",
            free_gb, reserved_gb, usable_gb,
        )
        return StorageBudget(
            total_free_gb=free_gb,
            reserved_gb=reserved_gb,
            usable_gb=usable_gb,
            per_dataset_caps=per_caps,
        )

    # ------------------------------------------------------------------
    # Usage measurement
    # ------------------------------------------------------------------

    def get_dataset_usage(self, dataset_name: str) -> DatasetUsage:
        """Measure actual disk usage of raw/processed/certified dirs for one dataset."""
        raw_path = self.base_dir / self.config.raw_data_dir / dataset_name
        processed_path = self.base_dir / self.config.processed_data_dir / dataset_name
        certified_path = self.base_dir / self.config.certified_data_dir / dataset_name

        raw_gb = _dir_size_gb(raw_path)
        processed_gb = _dir_size_gb(processed_path)
        certified_gb = _dir_size_gb(certified_path)

        return DatasetUsage(
            name=dataset_name,
            raw_gb=raw_gb,
            processed_gb=processed_gb,
            certified_gb=certified_gb,
            total_gb=raw_gb + processed_gb + certified_gb,
        )

    def get_summary(self) -> StorageSummary:
        """Get usage summary across all registered datasets."""
        usages: list[DatasetUsage] = []
        for name in self.config.dataset_budgets:
            try:
                usages.append(self.get_dataset_usage(name))
            except Exception as exc:
                logger.warning("Could not measure usage for dataset '%s': %s", name, exc)

        total_used_gb = sum(u.total_gb for u in usages)
        budget = self.compute_budget()
        _, free_gb = self.check_free_space(0.0)

        utilization = (total_used_gb / budget.usable_gb) if budget.usable_gb > 0 else 0.0
        if utilization >= 1.0:
            status = "FULL"
        elif utilization >= 0.85:
            status = "WARNING"
        else:
            status = "OK"

        return StorageSummary(
            datasets=usages,
            total_used_gb=total_used_gb,
            budget_gb=budget.usable_gb,
            free_gb=free_gb,
            status=status,
        )

    # ------------------------------------------------------------------
    # Budget enforcement
    # ------------------------------------------------------------------

    def check_budget(self, dataset_name: str, current_bytes: int) -> bool:
        """Return True if *current_bytes* is within the per-dataset cap."""
        spec = self.config.dataset_budgets.get(dataset_name)
        if spec is None:
            logger.warning("No budget spec found for dataset '%s'; allowing.", dataset_name)
            return True
        current_gb = current_bytes / (1024 ** 3)
        within = current_gb < spec.max_storage_gb
        if not within:
            logger.info(
                "Budget hit for '%s': %.3f GB >= cap %.3f GB",
                dataset_name, current_gb, spec.max_storage_gb,
            )
        return within

    # ------------------------------------------------------------------
    # Contamination guard
    # ------------------------------------------------------------------

    def validate_path_isolation(self, path: Path, dataset_name: str) -> None:
        """Raise EvalContaminationError if an eval dataset writes to the certified dir.

        Evaluation-only datasets must never enter the certified training directory.
        """
        spec = self.config.dataset_budgets.get(dataset_name)
        if spec is None:
            return  # Unknown dataset — no contamination rule applied

        if spec.stage != "eval_only":
            return  # Pretrain datasets may write anywhere

        certified_abs = (self.base_dir / self.config.certified_data_dir).resolve()
        try:
            path.resolve().relative_to(certified_abs)
        except ValueError:
            return  # Path is NOT under certified dir — safe

        raise EvalContaminationError(
            f"Eval-only dataset '{dataset_name}' attempted to write to certified "
            f"training directory '{certified_abs}'. Aborting to prevent contamination."
        )

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def save_storage_report(
        self,
        output_dir: Path,
        summary: StorageSummary | None = None,
    ) -> Path:
        """Write *storage_report.json* and *storage_report.md*; return the Markdown path."""
        output_dir.mkdir(parents=True, exist_ok=True)

        if summary is None:
            summary = self.get_summary()

        # --- JSON ---
        json_path = output_dir / "storage_report.json"
        try:
            json_path.write_text(summary.to_json(), encoding="utf-8")
            logger.info("Wrote storage JSON report to %s", json_path)
        except OSError as exc:
            logger.error("Failed to write storage JSON report: %s", exc)

        # --- Markdown ---
        md_path = output_dir / "storage_report.md"
        try:
            lines = [
                "# IVERI CORE — Phase 8 Storage Report",
                "",
                f"**Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"**Status:** {summary.status}",
                "",
                "## Budget Summary",
                "",
                f"| Metric | Value |",
                "|---|---|",
                f"| Total Free (disk) | {summary.free_gb:.2f} GB |",
                f"| Usable Budget | {summary.budget_gb:.2f} GB |",
                f"| Total Used | {summary.total_used_gb:.2f} GB |",
                f"| Remaining | {max(summary.budget_gb - summary.total_used_gb, 0.0):.2f} GB |",
                "",
                "## Per-Dataset Usage",
                "",
                "| Dataset | Raw (GB) | Processed (GB) | Certified (GB) | Total (GB) |",
                "|---|---|---|---|---|",
            ]
            for d in sorted(summary.datasets, key=lambda x: x.name):
                lines.append(
                    f"| {d.name} | {d.raw_gb:.4f} | {d.processed_gb:.4f} "
                    f"| {d.certified_gb:.4f} | {d.total_gb:.4f} |"
                )
            md_path.write_text("\n".join(lines), encoding="utf-8")
            logger.info("Wrote storage Markdown report to %s", md_path)
        except OSError as exc:
            logger.error("Failed to write storage Markdown report: %s", exc)

        return md_path
