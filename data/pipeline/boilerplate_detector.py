# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Boilerplate detection and removal for Phase 8 data engineering.

Detects cookie banners, copyright notices, headers, footers, and navigation links.
Provides options to strip boilerplate or reject documents entirely.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BoilerplateResult:
    """Result of boilerplate analysis on a single document."""

    has_boilerplate: bool
    boilerplate_ratio: float
    patterns_found: list[str]
    cleaned_text: str


@dataclass
class BoilerplateReport:
    """Aggregate boilerplate statistics for a batch."""

    total_docs: int
    docs_with_boilerplate: int
    boilerplate_rate: float
    fully_removed: int
    pattern_frequencies: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Convert to plain dictionary."""
        return {
            "generated_at": datetime.now().isoformat(),
            "total_docs": self.total_docs,
            "docs_with_boilerplate": self.docs_with_boilerplate,
            "boilerplate_rate": round(self.boilerplate_rate, 6),
            "fully_removed": self.fully_removed,
            "pattern_frequencies": self.pattern_frequencies,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class BoilerplateDetector:
    """Detects and strips template/boilerplate text (headers, footers, cookie notices)."""

    HEADER_PATTERNS = [
        r"(?i)^(cookies?|privacy policy|terms of (use|service|conditions))",
        r"(?i)^(copyright \d{4}|all rights reserved)",
        r"(?i)^(posted (by|in)|filed under|tags?:)",
        r"(?i)^(share (this|on)|like (this|us)|follow us)",
    ]
    FOOTER_PATTERNS = [
        r"(?i)(subscribe|newsletter|sign up).{0,60}(email|free)",
        r"(?i)(read more|continue reading|click here)",
        r"(?i)(\d+ (comments?|shares?|likes?))",
        r"(?i)(about the author|written by)",
    ]
    NAVIGATION_PATTERNS = [
        r"(?i)^(home|about|contact|sitemap|privacy)\s*[|>]\s*",
        r"(?i)(skip to (main content|navigation))",
    ]

    def __init__(self) -> None:
        self._compiled: dict[str, list[re.Pattern]] = {
            "header": [re.compile(p) for p in self.HEADER_PATTERNS],
            "footer": [re.compile(p) for p in self.FOOTER_PATTERNS],
            "navigation": [re.compile(p) for p in self.NAVIGATION_PATTERNS],
        }

    def detect(self, text: str, strip: bool = True) -> BoilerplateResult:
        """Analyze a document for boilerplate patterns. Optionally strip matches."""
        lines = text.split("\n")
        cleaned_lines: list[str] = []
        patterns_found: list[str] = []
        boilerplate_chars = 0
        total_chars = len(text) if text else 1

        for line in lines:
            line_strip = line.strip()
            if not line_strip:
                cleaned_lines.append(line)
                continue

            matched = False
            for group, patterns in self._compiled.items():
                for pat in patterns:
                    if pat.search(line_strip):
                        patterns_found.append(group)
                        boilerplate_chars += len(line)
                        matched = True
                        break
                if matched:
                    break

            if not matched or not strip:
                cleaned_lines.append(line)

        cleaned_text = "\n".join(cleaned_lines) if strip else text
        ratio = boilerplate_chars / total_chars

        return BoilerplateResult(
            has_boilerplate=len(patterns_found) > 0,
            boilerplate_ratio=ratio,
            patterns_found=list(set(patterns_found)),
            cleaned_text=cleaned_text,
        )

    def filter_batch(
        self,
        docs: list[dict],
        text_field: str = "text",
        max_boilerplate_ratio: float = 0.5,
    ) -> tuple[list[dict], BoilerplateReport]:
        """Filter and strip boilerplate from a batch of documents."""
        kept: list[dict] = []
        docs_with_bp = 0
        fully_removed = 0
        pattern_freq: dict[str, int] = {"header": 0, "footer": 0, "navigation": 0}

        for doc in docs:
            text = doc.get(text_field, "")
            if not isinstance(text, str):
                text = ""

            res = self.detect(text, strip=True)
            if res.has_boilerplate:
                docs_with_bp += 1
                for pat in res.patterns_found:
                    pattern_freq[pat] = pattern_freq.get(pat, 0) + 1

            if res.boilerplate_ratio > max_boilerplate_ratio:
                fully_removed += 1
                logger.debug(
                    "Removed document with high boilerplate ratio: %.2f",
                    res.boilerplate_ratio,
                )
            else:
                doc_copy = dict(doc)
                doc_copy[text_field] = res.cleaned_text
                kept.append(doc_copy)

        total = len(docs)
        rate = docs_with_bp / total if total > 0 else 0.0
        report = BoilerplateReport(
            total_docs=total,
            docs_with_boilerplate=docs_with_bp,
            boilerplate_rate=rate,
            fully_removed=fully_removed,
            pattern_frequencies=pattern_freq,
        )

        logger.info(
            "BoilerplateDetector: kept %d/%d (removed %d, stripped %d)",
            len(kept), total, fully_removed, docs_with_bp - fully_removed,
        )
        return kept, report
