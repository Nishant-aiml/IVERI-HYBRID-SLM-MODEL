# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Statistics pipeline consistency audit (Phase 6.3.1G).

Verifies that Shapiro, t-test, Wilcoxon, Holm, bootstrap, Cohen's d, and
Cliff's Δ are computed through a single canonical pipeline consumed by all
research reports and comparators.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from research.statistics import CANONICAL_STATISTICS_METHODS, ResearchStatisticalValidator

REPO_ROOT = Path(__file__).resolve().parents[1]

CONSUMER_FILES = (
    "research/compare_runs.py",
    "research/publication_manager.py",
    "research/generate_reports.py",
)

FORBIDDEN_INLINE_PATTERNS = (
    r"\.compute_paired_t_test\s*\(",
    r"\.compute_wilcoxon_signed_rank\s*\(",
    r"\.compute_cohens_d\s*\(",
    r"\.compute_bootstrap_confidence_interval\s*\(",
    r"\.compute_shapiro_wilk\s*\(",
    r"\.apply_holm_bonferroni\s*\(",
    r"\.compute_cliffs_delta\s*\(",
)


@dataclass
class ConsumerProbe:
    rel_path: str
    uses_canonical_bundle: bool
    forbidden_inline_calls: list[str] = field(default_factory=list)


@dataclass
class StatisticsConsistencyResult:
    protocol_version: str = "Phase-6.3.1G"
    timestamp_utc: str = ""
    production_verdict: str = "UNKNOWN"
    canonical_methods: list[str] = field(default_factory=list)
    bundle_covers_all_methods: bool = False
    golden_bundle_ok: bool = False
    consumers: list[ConsumerProbe] = field(default_factory=list)
    duplicate_calculation_violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _scan_consumer_file(rel_path: str) -> ConsumerProbe:
    text = (REPO_ROOT / rel_path).read_text(encoding="utf-8")
    uses_bundle = "compute_paired_hypothesis_statistics" in text
    forbidden: list[str] = []
    if rel_path != "research/statistics.py":
        for pattern in FORBIDDEN_INLINE_PATTERNS:
            if re.search(pattern, text):
                forbidden.append(pattern)
    return ConsumerProbe(rel_path, uses_bundle, forbidden)


def _verify_bundle_methods() -> bool:
    validator = ResearchStatisticalValidator()
    baseline = [1.12, 1.10, 1.09, 1.11, 1.08]
    treatment = [1.05, 1.04, 1.03, 1.06, 1.02]
    bundle = validator.compute_paired_hypothesis_statistics(
        baseline, treatment, metric_name="val_loss"
    )
    if bundle.get("status") != "OK":
        return False

    required_keys = {
        "shapiro_wilk",
        "paired_t_test",
        "wilcoxon",
        "cohens_d",
        "cliffs_delta",
        "bootstrap_95_ci",
        "primary_p_value",
    }
    if not required_keys.issubset(bundle.keys()):
        return False

    holm = validator.apply_holm_bonferroni({"val_loss": bundle["primary_p_value"]})
    return "val_loss" in holm


def _detect_duplicate_calculations() -> list[str]:
    """Flag files that mix canonical bundle with forbidden inline statistics calls."""
    violations: list[str] = []
    for rel in CONSUMER_FILES:
        probe = _scan_consumer_file(rel)
        if probe.forbidden_inline_calls:
            violations.append(
                f"{rel}: inline statistics calls outside canonical pipeline "
                f"({', '.join(probe.forbidden_inline_calls)})"
            )
    return violations


def run_statistics_consistency_audit() -> StatisticsConsistencyResult:
    consumers = [_scan_consumer_file(p) for p in CONSUMER_FILES]
    bundle_ok = _verify_bundle_methods()
    duplicates = _detect_duplicate_calculations()
    all_use_bundle = all(c.uses_canonical_bundle for c in consumers)
    no_inline = len(duplicates) == 0
    verdict = "PASS" if bundle_ok and all_use_bundle and no_inline else "FAIL"
    return StatisticsConsistencyResult(
        timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        production_verdict=verdict,
        canonical_methods=list(CANONICAL_STATISTICS_METHODS),
        bundle_covers_all_methods=bundle_ok,
        golden_bundle_ok=bundle_ok,
        consumers=consumers,
        duplicate_calculation_violations=duplicates,
    )


def render_statistics_consistency_report(result: StatisticsConsistencyResult) -> str:
    lines = [
        "# Statistics Consistency Report (Phase 6.3.1G)",
        "",
        f"**Generated:** {result.timestamp_utc}  ",
        f"**Protocol:** {result.protocol_version}  ",
        "",
        "## Executive Verdict",
        "",
        f"**Single statistics pipeline:** `{result.production_verdict}`",
        "",
        "## Canonical Methods",
        "",
        "All seven methods must flow through "
        "`ResearchStatisticalValidator.compute_paired_hypothesis_statistics()`:",
        "",
    ]
    for method in result.canonical_methods:
        lines.append(f"- `{method}`")
    lines.extend(
        [
            "",
            f"**Bundle covers all methods:** `{result.bundle_covers_all_methods}`",
            f"**Golden bundle self-test:** `{result.golden_bundle_ok}`",
            "",
            "## Consumer Audit",
            "",
            "| File | Uses canonical bundle | Forbidden inline calls |",
            "|------|:---------------------:|:----------------------:|",
        ]
    )
    for c in result.consumers:
        inline = "none" if not c.forbidden_inline_calls else f"{len(c.forbidden_inline_calls)}"
        lines.append(
            f"| `{c.rel_path}` | {c.uses_canonical_bundle} | {inline} |"
        )

    lines.extend(["", "## Duplicate Calculation Detection", ""])
    if not result.duplicate_calculation_violations:
        lines.append("No duplicated inline statistics calculations detected in consumer modules.")
    else:
        for v in result.duplicate_calculation_violations:
            lines.append(f"- {v}")

    lines.extend(
        [
            "",
            "## Raw JSON",
            "",
            "```json",
            json.dumps(result.to_dict(), indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_statistics_consistency_report(output_path: str | Path) -> StatisticsConsistencyResult:
    result = run_statistics_consistency_audit()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_statistics_consistency_report(result), encoding="utf-8")
    return result
