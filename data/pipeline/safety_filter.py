# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Research-grade safety filter for Phase 8 IVERI CORE data pipeline.

Designed to minimize false positives on legitimate educational content.
Research mode targets only clearly harmful content (CSAM indicators,
explicit self-harm instructions, weapons synthesis). Production mode
adds detection of explicit adult content and hate speech.

IMPORTANT: CSAM indicator matches are counted only — the matched content
is NEVER logged. Only the count is surfaced in reports.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SafetyResult:
    """Result of a safety evaluation on a single document."""

    is_safe: bool
    flags: list[str]      # Names of triggered patterns
    risk_level: str       # "none" | "low" | "medium" | "high" | "critical"
    mode: str             # "research" or "production"

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation."""
        return {
            "is_safe": self.is_safe,
            "flags": self.flags,
            "risk_level": self.risk_level,
            "mode": self.mode,
        }

    def to_json(self) -> str:
        """Return a JSON string representation."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class SafetyReport:
    """Aggregate safety statistics for a batch of documents."""

    total_docs: int
    kept: int
    removed: int
    removal_rate: float
    risk_distribution: dict[str, int]   # risk_level -> count
    pattern_frequencies: dict[str, int]  # pattern_name -> count

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation."""
        return {
            "generated_at": datetime.now().isoformat(),
            "total_docs": self.total_docs,
            "kept": self.kept,
            "removed": self.removed,
            "removal_rate": round(self.removal_rate, 6),
            "risk_distribution": self.risk_distribution,
            "pattern_frequencies": self.pattern_frequencies,
        }

    def to_json(self) -> str:
        """Return a JSON string representation."""
        return json.dumps(self.to_dict(), indent=2)

    def save_report(self, path: Path) -> None:
        """Write both *safety_report.json* and *safety_report.md* to *path* directory."""
        path.mkdir(parents=True, exist_ok=True)

        # JSON
        json_path = path / "safety_report.json"
        try:
            json_path.write_text(self.to_json(), encoding="utf-8")
            logger.info("Wrote safety JSON report to %s", json_path)
        except OSError as exc:
            logger.error("Failed to write safety JSON report: %s", exc)

        # Markdown
        md_path = path / "safety_report.md"
        try:
            lines = [
                "# IVERI CORE — Phase 8 Safety Filter Report",
                "",
                f"**Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "## Summary",
                "",
                f"| Metric | Value |",
                "|---|---|",
                f"| Total Documents | {self.total_docs:,} |",
                f"| Kept | {self.kept:,} |",
                f"| Removed | {self.removed:,} |",
                f"| Removal Rate | {self.removal_rate:.2%} |",
                "",
                "## Risk Distribution",
                "",
                "| Risk Level | Count |",
                "|---|---|",
            ]
            for level in ("none", "low", "medium", "high", "critical"):
                count = self.risk_distribution.get(level, 0)
                lines.append(f"| {level} | {count:,} |")

            lines.extend([
                "",
                "## Pattern Frequencies",
                "",
                "| Pattern | Matches |",
                "|---|---|",
            ])
            for pattern_name, count in sorted(
                self.pattern_frequencies.items(), key=lambda x: -x[1]
            ):
                # Never show CSAM pattern names — replace with redacted label
                display_name = "[redacted]" if "csam" in pattern_name.lower() else pattern_name
                lines.append(f"| {display_name} | {count:,} |")

            md_path.write_text("\n".join(lines), encoding="utf-8")
            logger.info("Wrote safety Markdown report to %s", md_path)
        except OSError as exc:
            logger.error("Failed to write safety Markdown report: %s", exc)


class SafetyFilter:
    """Research-grade content safety filter with minimized false positives.

    In *research* mode only CRITICAL_PATTERNS are active (very high precision,
    minimal FP rate).  In *production* mode PRODUCTION_PATTERNS are added.
    """

    # High-precision critical patterns — active in both modes
    CRITICAL_PATTERNS: dict[str, str] = {
        "csam_indicator": r"(?i)\b(child pornography|cp [0-9]|loli[^n])\b",
        "self_harm_instruction": (
            r"(?i)(step[s]? to (commit|attempt) suicide|how to kill yourself"
            r"|detailed (method|guide|instruction).{0,30}(suicide|self.harm))"
        ),
        "weapons_synthesis": (
            r"(?i)(synthesize|manufacture|make|produce).{0,50}"
            r"(sarin|vx nerve|ricin|anthrax|novichok|mustard gas)"
        ),
    }

    # Medium-confidence patterns — production mode only
    PRODUCTION_PATTERNS: dict[str, str] = {
        "explicit_adult": (
            r"(?i)\b(hardcore porn|explicit sex scene|graphic sexual).{0,30}\b"
        ),
        "hate_speech_directive": (
            r"(?i)(all (jews|muslims|blacks|whites|gays) (should|must|deserve to) (die|be killed|be exterminated))"
        ),
        "doxxing_template": (
            r"(?i)(home address|phone number|ssn).{0,30}(of|for).{0,30}(harass|stalk|attack|harm)"
        ),
    }

    # Risk level assignments per pattern group
    _PATTERN_RISK: dict[str, str] = {
        "csam_indicator": "critical",
        "self_harm_instruction": "high",
        "weapons_synthesis": "high",
        "explicit_adult": "medium",
        "hate_speech_directive": "medium",
        "doxxing_template": "medium",
    }

    def __init__(self, mode: str = "research") -> None:
        """Initialise the safety filter.

        Args:
            mode: ``"research"`` (critical patterns only) or
                  ``"production"`` (critical + production patterns).
        """
        if mode not in ("research", "production"):
            raise ValueError(f"mode must be 'research' or 'production', got {mode!r}")
        self.mode = mode

        active_patterns = dict(self.CRITICAL_PATTERNS)
        if mode == "production":
            active_patterns.update(self.PRODUCTION_PATTERNS)

        self._compiled: dict[str, re.Pattern] = {}
        for name, pattern in active_patterns.items():
            try:
                self._compiled[name] = re.compile(pattern, re.DOTALL)
            except re.error as exc:
                logger.warning("Failed to compile safety pattern '%s': %s", name, exc)

    def is_safe(self, text: str) -> SafetyResult:
        """Evaluate a single document for safety.

        Returns a :class:`SafetyResult` with flags and risk level.
        CSAM match content is NEVER logged — only counted.
        """
        flags: list[str] = []
        max_risk = "none"

        risk_order = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}

        for name, pattern in self._compiled.items():
            try:
                if pattern.search(text):
                    flags.append(name)
                    risk = self._PATTERN_RISK.get(name, "medium")
                    if risk_order.get(risk, 0) > risk_order.get(max_risk, 0):
                        max_risk = risk
            except Exception as exc:
                logger.debug("Safety pattern '%s' raised exception: %s", name, exc)

        is_safe_doc = len(flags) == 0
        return SafetyResult(
            is_safe=is_safe_doc,
            flags=flags,
            risk_level=max_risk,
            mode=self.mode,
        )

    def filter_batch(
        self,
        docs: list[dict],
        text_field: str = "text",
    ) -> tuple[list[dict], SafetyReport]:
        """Filter a batch of documents, returning kept docs and a report.

        Removed documents are counted with reason but content is not stored.
        CSAM indicator content is NEVER logged.

        Args:
            docs: List of dicts, each expected to have a ``text_field`` key.
            text_field: Key to extract text from each document dict.

        Returns:
            *(kept_docs, SafetyReport)*
        """
        kept: list[dict] = []
        risk_dist: dict[str, int] = {
            "none": 0, "low": 0, "medium": 0, "high": 0, "critical": 0
        }
        pattern_freq: dict[str, int] = {name: 0 for name in self._compiled}
        removed = 0

        for doc in docs:
            text = doc.get(text_field, "")
            if not isinstance(text, str):
                text = ""

            result = self.is_safe(text)
            risk_dist[result.risk_level] = risk_dist.get(result.risk_level, 0) + 1

            for flag in result.flags:
                pattern_freq[flag] = pattern_freq.get(flag, 0) + 1

            if result.is_safe:
                kept.append(doc)
            else:
                removed += 1
                # Only log pattern names and risk level — NEVER the content
                safe_flags = [
                    "[redacted]" if "csam" in f.lower() else f
                    for f in result.flags
                ]
                logger.debug(
                    "Removed document: risk=%s, flags=%s", result.risk_level, safe_flags
                )

        total = len(docs)
        report = SafetyReport(
            total_docs=total,
            kept=len(kept),
            removed=removed,
            removal_rate=(removed / total) if total > 0 else 0.0,
            risk_distribution=risk_dist,
            pattern_frequencies=pattern_freq,
        )

        logger.info(
            "SafetyFilter [%s]: kept %d/%d (removed %d, %.2f%%)",
            self.mode, len(kept), total, removed,
            report.removal_rate * 100,
        )
        return kept, report
