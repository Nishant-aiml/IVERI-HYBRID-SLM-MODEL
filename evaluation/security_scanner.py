# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Pattern-based security scanner for IVERI CORE Phase 3.3 coding specialization.

Identifies potential vulnerabilities in generated code snippets (e.g. eval, exec,
pickle, shell execution, hardcoded credentials) for logging and telemetry.
Reporting only; does not filter training sequences.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Security warning patterns (Feedback #7)
SECURITY_PATTERNS: dict[str, str] = {
    "eval_usage": r"\beval\s*\(",
    "exec_usage": r"\bexec\s*\(",
    "pickle_loads": r"pickle\s*\.\s*loads\s*\(",
    "subprocess_shell": r"subprocess\s*\.\s*[A-Za-z_0-9]+\s*\(.*shell\s*=\s*True",
    "os_system": r"os\s*\.\s*system\s*\(",
    "hardcoded_secrets": r"(?i)(password|secret|api_key|token|passwd|private_key)\s*=\s*['\"][^'\"]{8,}['\"]",
    "dangerous_import": r"import\s+(pickle|socket|pty|ctypes)\b",
}


@dataclass
class SecurityScanResult:
    """Security status of a single code block.

    Attributes
    ----------
    flags:
        Mapping of warning pattern identifier → True (found) or False.
    flagged_patterns:
        List of all warning identifiers found in the code.
    total_flags:
        Total number of warning flags.
    is_flagged:
        True if at least one warning flag was raised.
    risk_level:
        One of ``\"none\"`` (0 flags), ``\"low\"`` (1 dangerous import),
        ``\"medium\"`` (1-2 flags), or ``\"high\"`` (pickle, shell, credentials, or 3+ flags).
    """

    flags: dict[str, bool] = field(default_factory=dict)
    flagged_patterns: list[str] = field(default_factory=list)
    total_flags: int = 0
    is_flagged: bool = False
    risk_level: str = "none"


@dataclass
class SecurityBatchResult:
    """Aggregated security metrics over a batch of code blocks.

    Attributes
    ----------
    total_scanned:
        Total code samples checked.
    flagged_count:
        Number of flagged samples.
    flagged_ratio:
        Ratio of flagged samples to total scanned.
    pattern_frequencies:
        Occurrences count of each flagged pattern across the batch.
    risk_distribution:
        Occurrences count of each risk level (none, low, medium, high).
    """

    total_scanned: int
    flagged_count: int
    flagged_ratio: float
    pattern_frequencies: dict[str, int]
    risk_distribution: dict[str, int]


class SecurityScanner:
    """Scans code for dangerous patterns and potential vulnerabilities."""

    def scan(self, code: str) -> SecurityScanResult:
        """Scan a code string for security issues.

        Parameters
        ----------
        code:
            Source code to scan.

        Returns
        -------
        SecurityScanResult
        """
        flags: dict[str, bool] = {}
        flagged_patterns: list[str] = []

        for name, pattern in SECURITY_PATTERNS.items():
            found = bool(re.search(pattern, code))
            flags[name] = found
            if found:
                flagged_patterns.append(name)

        total_flags = len(flagged_patterns)
        is_flagged = total_flags > 0

        # Assess risk level
        risk_level = "none"
        if total_flags == 0:
            risk_level = "none"
        elif total_flags == 1 and flagged_patterns[0] == "dangerous_import":
            risk_level = "low"
        elif total_flags >= 3 or any(
            p in flagged_patterns
            for p in ("pickle_loads", "subprocess_shell", "hardcoded_secrets")
        ):
            risk_level = "high"
        else:
            risk_level = "medium"

        return SecurityScanResult(
            flags=flags,
            flagged_patterns=flagged_patterns,
            total_flags=total_flags,
            is_flagged=is_flagged,
            risk_level=risk_level,
        )

    def scan_bytes(self, code_bytes: bytes) -> SecurityScanResult:
        """Decode and scan a byte sequence."""
        return self.scan(code_bytes.decode("utf-8", errors="replace"))

    def scan_batch(self, code_snippets: list[str]) -> SecurityBatchResult:
        """Scan a batch of code snippets and return aggregated telemetry.

        Parameters
        ----------
        code_snippets:
            List of source code strings.

        Returns
        -------
        SecurityBatchResult
        """
        total = len(code_snippets)
        if total == 0:
            return SecurityBatchResult(
                total_scanned=0,
                flagged_count=0,
                flagged_ratio=0.0,
                pattern_frequencies={k: 0 for k in SECURITY_PATTERNS},
                risk_distribution={"none": 0, "low": 0, "medium": 0, "high": 0},
            )

        results = [self.scan(s) for s in code_snippets]
        flagged_count = sum(1 for r in results if r.is_flagged)

        pattern_freqs: dict[str, int] = {k: 0 for k in SECURITY_PATTERNS}
        risk_dist: dict[str, int] = {"none": 0, "low": 0, "medium": 0, "high": 0}

        for r in results:
            risk_dist[r.risk_level] += 1
            for pat in r.flagged_patterns:
                pattern_freqs[pat] += 1

        return SecurityBatchResult(
            total_scanned=total,
            flagged_count=flagged_count,
            flagged_ratio=flagged_count / total,
            pattern_frequencies=pattern_freqs,
            risk_distribution=risk_dist,
        )
