# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Dataset statistics, histogram, and composition report generator.

Generates comprehensive reports (Markdown, JSON, CSV) analyzing document length
distributions, language proportions, license breakdowns, and duplicate rates.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=False, slots=True)
class LengthHistogram:
    """Document length distribution histogram bins."""

    bins: list[int]
    counts: list[int]
    bin_labels: list[str]


@dataclass(frozen=False, slots=True)
class DatasetReport:
    """Detailed summary report for a single dataset."""

    name: str
    stage: str
    num_documents: int
    total_bytes: int
    total_gb: float
    avg_bytes: float
    median_bytes: float
    min_bytes: int
    max_bytes: int
    p25_bytes: float
    p75_bytes: float
    p95_bytes: float
    p99_bytes: float
    language_distribution: dict[str, int]
    license: str
    mixing_weight: float
    duplicate_rate: float | None = None
    filter_rate: float | None = None
    creation_time: str = ""
    length_histogram: LengthHistogram | None = None


@dataclass(frozen=False, slots=True)
class PipelineSummaryReport:
    """Overall summary across all datasets in the pipeline."""

    total_datasets: int
    total_documents: int
    total_bytes: int
    total_gb: float
    stage_breakdown: dict[str, int]
    license_breakdown: dict[str, int]
    creation_time: str
    dataset_reports: list[DatasetReport] = field(default_factory=list)


class DatasetStatisticsGenerator:
    """Computes stats and serializes reports in multiple formats."""

    def generate_histogram(self, sizes: list[int], n_bins: int = 10) -> LengthHistogram:
        """Create a length histogram from document sizes."""
        if not sizes:
            return LengthHistogram(bins=[], counts=[], bin_labels=[])

        sizes_arr = np.array(sizes)
        counts, bins_edges = np.histogram(sizes_arr, bins=n_bins)

        bins = [int(x) for x in bins_edges]
        counts_list = [int(x) for x in counts]

        labels = []
        for i in range(len(bins) - 1):
            labels.append(f"{bins[i]:,} - {bins[i+1]:,}")

        return LengthHistogram(bins=bins, counts=counts_list, bin_labels=labels)

    def generate(
        self,
        name: str,
        texts: list[str],
        stage: str,
        license_str: str = "unknown",
        mixing_weight: float = 0.0,
        duplicate_rate: float | None = None,
        filter_rate: float | None = None,
        languages: list[str] | None = None,
    ) -> DatasetReport:
        """Analyze a list of documents and return a DatasetReport."""
        num_docs = len(texts)
        if num_docs == 0:
            return DatasetReport(
                name=name,
                stage=str(stage),
                num_documents=0,
                total_bytes=0,
                total_gb=0.0,
                avg_bytes=0.0,
                median_bytes=0.0,
                min_bytes=0,
                max_bytes=0,
                p25_bytes=0.0,
                p75_bytes=0.0,
                p95_bytes=0.0,
                p99_bytes=0.0,
                language_distribution={},
                license=license_str,
                mixing_weight=mixing_weight,
                duplicate_rate=duplicate_rate,
                filter_rate=filter_rate,
                creation_time=datetime.now().isoformat(),
                length_histogram=None,
            )

        sizes = [len(t.encode("utf-8")) for t in texts]
        sizes_arr = np.array(sizes)
        total_b = int(np.sum(sizes_arr))

        # Language distribution
        lang_dist: dict[str, int] = {}
        if languages:
            for lang in languages:
                lang_dist[lang] = lang_dist.get(lang, 0) + 1
        else:
            lang_dist = {"en": num_docs}  # default assumption

        hist = self.generate_histogram(sizes)

        return DatasetReport(
            name=name,
            stage=str(stage),
            num_documents=num_docs,
            total_bytes=total_b,
            total_gb=total_b / (1024**3),
            avg_bytes=float(np.mean(sizes_arr)),
            median_bytes=float(np.median(sizes_arr)),
            min_bytes=int(np.min(sizes_arr)),
            max_bytes=int(np.max(sizes_arr)),
            p25_bytes=float(np.percentile(sizes_arr, 25)),
            p75_bytes=float(np.percentile(sizes_arr, 75)),
            p95_bytes=float(np.percentile(sizes_arr, 95)),
            p99_bytes=float(np.percentile(sizes_arr, 99)),
            language_distribution=lang_dist,
            license=license_str,
            mixing_weight=mixing_weight,
            duplicate_rate=duplicate_rate,
            filter_rate=filter_rate,
            creation_time=datetime.now().isoformat(),
            length_histogram=hist,
        )

    def save_markdown(self, report: DatasetReport, path: str | Path) -> None:
        """Write single dataset report as Markdown."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"# Dataset Quality Report: {report.name}",
            "",
            f"- **Stage**: {report.stage}",
            f"- **License**: {report.license}",
            f"- **Mixing Weight**: {report.mixing_weight:.2%}",
            f"- **Created At**: {report.creation_time}",
            "",
            "## Summary Metrics",
            "",
            f"- **Total Documents**: {report.num_documents:,}",
            f"- **Total Size**: {report.total_bytes:,} bytes ({report.total_gb:.4f} GB)",
            f"- **Avg Document Size**: {report.avg_bytes:.1f} bytes",
            f"- **Median Document Size**: {report.median_bytes:.1f} bytes",
            f"- **Min/Max Size**: {report.min_bytes:,} / {report.max_bytes:,} bytes",
            "",
            "## Size Percentiles (Bytes)",
            "",
            f"- **25th Percentile**: {report.p25_bytes:.1f}",
            f"- **75th Percentile**: {report.p75_bytes:.1f}",
            f"- **95th Percentile**: {report.p95_bytes:.1f}",
            f"- **99th Percentile**: {report.p99_bytes:.1f}",
            "",
        ]

        if report.duplicate_rate is not None:
            lines.append(f"- **Duplicate Rate**: {report.duplicate_rate:.2%}")
        if report.filter_rate is not None:
            lines.append(f"- **Filtered Out Rate**: {report.filter_rate:.2%}")

        if report.length_histogram:
            lines.extend(
                [
                    "",
                    "## Length Distribution Histogram",
                    "",
                    "| Bin Range (Bytes) | Count | Percentage |",
                    "|---|---|---|",
                ]
            )
            hist = report.length_histogram
            total_counts = sum(hist.counts)
            for label, count in zip(hist.bin_labels, hist.counts, strict=False):
                pct = (count / total_counts) if total_counts > 0 else 0.0
                lines.append(f"| {label} | {count:,} | {pct:.2%} |")

        p.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Wrote Markdown report to {p}")

    def save_json(self, report: DatasetReport | PipelineSummaryReport, path: str | Path) -> None:
        """Write report or pipeline summary as JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        def make_serializable(obj: Any) -> Any:
            if dataclass_fields := getattr(obj, "__dataclass_fields__", None):
                return {k: make_serializable(getattr(obj, k)) for k in dataclass_fields}
            elif isinstance(obj, list):
                return [make_serializable(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            return obj

        with open(p, "w", encoding="utf-8") as f:
            json.dump(make_serializable(report), f, indent=4)
        logger.info(f"Wrote JSON report to {p}")

    def save_csv(self, report: DatasetReport, path: str | Path) -> None:
        """Write summary metrics to a CSV file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            writer.writerow(["dataset_name", report.name])
            writer.writerow(["stage", report.stage])
            writer.writerow(["num_documents", report.num_documents])
            writer.writerow(["total_bytes", report.total_bytes])
            writer.writerow(["total_gb", report.total_gb])
            writer.writerow(["avg_bytes", report.avg_bytes])
            writer.writerow(["median_bytes", report.median_bytes])
            writer.writerow(["min_bytes", report.min_bytes])
            writer.writerow(["max_bytes", report.max_bytes])
            writer.writerow(["p25_bytes", report.p25_bytes])
            writer.writerow(["p75_bytes", report.p75_bytes])
            writer.writerow(["p95_bytes", report.p95_bytes])
            writer.writerow(["p99_bytes", report.p99_bytes])
            writer.writerow(["license", report.license])
            writer.writerow(["mixing_weight", report.mixing_weight])
            if report.duplicate_rate is not None:
                writer.writerow(["duplicate_rate", report.duplicate_rate])
            if report.filter_rate is not None:
                writer.writerow(["filter_rate", report.filter_rate])

        logger.info(f"Wrote CSV report to {p}")

    def generate_pipeline_summary(
        self, dataset_reports: list[DatasetReport]
    ) -> PipelineSummaryReport:
        """Aggregate multiple dataset reports into a single pipeline summary."""
        total_ds = len(dataset_reports)
        total_docs = sum(r.num_documents for r in dataset_reports)
        total_b = sum(r.total_bytes for r in dataset_reports)

        stage_breakdown: dict[str, int] = {}
        license_breakdown: dict[str, int] = {}

        for r in dataset_reports:
            stage_breakdown[r.stage] = stage_breakdown.get(r.stage, 0) + r.num_documents
            license_breakdown[r.license] = license_breakdown.get(r.license, 0) + r.num_documents

        return PipelineSummaryReport(
            total_datasets=total_ds,
            total_documents=total_docs,
            total_bytes=total_b,
            total_gb=total_b / (1024**3),
            stage_breakdown=stage_breakdown,
            license_breakdown=license_breakdown,
            creation_time=datetime.now().isoformat(),
            dataset_reports=dataset_reports,
        )

    def save_summary_markdown(self, summary: PipelineSummaryReport, path: str | Path) -> None:
        """Write pipeline summary report as Markdown."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# IVERI CORE — Data Pipeline Master Summary",
            "",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Combined Sizing Metrics",
            "",
            f"- **Total Datasets**: {summary.total_datasets}",
            f"- **Total Documents**: {summary.total_documents:,}",
            f"- **Total Size**: {summary.total_bytes:,} bytes ({summary.total_gb:.4f} GB)",
            "",
            "## Dataset Breakdown Table",
            "",
            "| Dataset Name | Stage | Documents | Size (GB) | License | Weight |",
            "|---|---|---|---|---|---|",
        ]

        for r in sorted(summary.dataset_reports, key=lambda x: x.name):
            lines.append(
                f"| {r.name} | {r.stage} | {r.num_documents:,} | {r.total_gb:.4f} | {r.license} | {r.mixing_weight:.2%} |"
            )

        lines.extend(
            [
                "",
                "## Stage Composition Breakdown",
                "",
                "| Stage | Documents | Percentage |",
                "|---|---|---|",
            ]
        )
        for stg, count in sorted(summary.stage_breakdown.items()):
            pct = count / summary.total_documents if summary.total_documents > 0 else 0.0
            lines.append(f"| Stage {stg} | {count:,} | {pct:.2%} |")

        lines.extend(
            [
                "",
                "## License Breakdown",
                "",
                "| License | Documents | Percentage |",
                "|---|---|---|",
            ]
        )
        for lic, count in sorted(summary.license_breakdown.items()):
            pct = count / summary.total_documents if summary.total_documents > 0 else 0.0
            lines.append(f"| {lic} | {count:,} | {pct:.2%} |")

        p.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Wrote pipeline summary Markdown report to {p}")

    def generate_composition_json(self, dataset_reports: list[DatasetReport]) -> dict[str, Any]:
        """Generate formatted data points for visualization (pie chart)."""
        data_points = []
        total_bytes = sum(r.total_bytes for r in dataset_reports)

        for r in dataset_reports:
            data_points.append(
                {
                    "label": r.name,
                    "value": r.total_bytes,
                    "percentage": (r.total_bytes / total_bytes) if total_bytes > 0 else 0.0,
                    "license": r.license,
                    "stage": r.stage,
                }
            )

        return {
            "title": "Dataset Byte Composition Breakdown",
            "total_bytes": total_bytes,
            "data": data_points,
            "generated_at": datetime.now().isoformat(),
        }
