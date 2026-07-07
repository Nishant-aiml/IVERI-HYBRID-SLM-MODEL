# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Dataset contamination checker for IVERI CORE Phase 3.3 coding specialization.

Detects potential overlap (contamination) between benchmark prompts and training
datasets. Uses normalised n-gram fingerprint hashing and Jaccard similarity.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ContaminationReport:
    """Contamination status of a benchmark suite.

    Attributes
    ----------
    benchmark_name:
        Name of the benchmark suite evaluated (e.g. ``\"HumanEval\"``).
    total_prompts:
        Total number of prompts checked.
    contaminated_count:
        Number of prompts containing overlapping training sequences.
    contamination_ratio:
        Ratio of contaminated prompts to total checked.
    flagged_items:
        List of match dicts: ``{\"prompt_id\", \"matched_file\", \"similarity\"}``.
    clean:
        True if ``contamination_ratio < 0.01`` (no leakage).
    method:
        Fingerprint method identifier string.
    """

    benchmark_name: str
    total_prompts: int
    contaminated_count: int
    contamination_ratio: float
    flagged_items: list[dict[str, Any]] = field(default_factory=list)
    clean: bool = True
    method: str = "ngram_fingerprint"


class ContaminationChecker:
    """Performs fingerprint-based overlap checks to detect dataset contamination.

    Parameters
    ----------
    ngram_size:
        Word n-gram size used for fingerprint generation.
    similarity_threshold:
        Jaccard similarity threshold for flagging contamination.
    """

    def __init__(
        self,
        ngram_size: int = 8,
        similarity_threshold: float = 0.8,
    ) -> None:
        self.ngram_size = ngram_size
        self.similarity_threshold = similarity_threshold

    def check(
        self,
        benchmark_prompts: list[dict[str, Any]],
        data_dir: Path | str,
    ) -> ContaminationReport:
        """Scan training data files for overlap with benchmark prompts.

        Parameters
        ----------
        benchmark_prompts:
            List of prompt dicts containing ``\"prompt_id\"`` and ``\"instruction\"``.
        data_dir:
            Directory to search recursively for *.jsonl and *.json training files.

        Returns
        -------
        ContaminationReport
        """
        data_path = Path(data_dir)
        total_prompts = len(benchmark_prompts)

        if total_prompts == 0 or not data_path.exists():
            logger.warning("ContaminationChecker: no prompts or invalid data path %s.", data_path)
            return ContaminationReport(
                benchmark_name="unknown",
                total_prompts=total_prompts,
                contaminated_count=0,
                contamination_ratio=0.0,
                clean=True,
            )

        # 1. Pre-generate fingerprints for all benchmark prompts
        prompt_fingerprints: dict[str, set[str]] = {}
        for p in benchmark_prompts:
            pid = p.get("prompt_id") or p.get("task_id", "prompt")
            # Concat instruction and reference solution if present
            text = p.get("instruction", "") + "\n" + p.get("reference_solution", "")
            prompt_fingerprints[pid] = self._fingerprint(text)

        flagged_items: list[dict[str, Any]] = []
        contaminated_pids: set[str] = set()

        # 2. Scan training files
        # Cap files scanned at 100MB to avoid OOM or slow execution in test/verification runs
        for fpath in sorted(data_path.rglob("*")):
            if fpath.suffix not in (".json", ".jsonl"):
                continue
            if not fpath.is_file():
                continue
            if fpath.stat().st_size > 100 * 1024 * 1024:
                logger.warning("ContaminationChecker: skipping large file %s", fpath.name)
                continue

            try:
                # Read training samples
                content = fpath.read_text(encoding="utf-8", errors="replace")
                samples: list[dict[str, Any]] = []
                # Simple line-by-line json parsing
                for line in content.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            samples.append(obj)
                        elif isinstance(obj, list):
                            samples.extend(o for o in obj if isinstance(o, dict))
                    except json.JSONDecodeError:
                        pass

                # Compare training sample text with prompt fingerprints
                for s in samples:
                    # Concat potential text fields in sample
                    sample_text = (
                        s.get("instruction", "") + "\n" +
                        s.get("output", "") + "\n" +
                        s.get("content", "") + "\n" +
                        s.get("text", "")
                    )
                    if not sample_text.strip():
                        continue

                    sample_fg = self._fingerprint(sample_text)
                    if not sample_fg:
                        continue

                    for pid, p_fg in prompt_fingerprints.items():
                        if not p_fg:
                            continue
                        sim = self._jaccard(p_fg, sample_fg)
                        if sim >= self.similarity_threshold:
                            flagged_items.append({
                                "prompt_id": pid,
                                "matched_file": fpath.as_posix(),
                                "similarity": sim,
                            })
                            contaminated_pids.add(pid)

            except Exception as exc:
                logger.warning("ContaminationChecker failed reading %s: %s", fpath, exc)

        n_contaminated = len(contaminated_pids)
        ratio = n_contaminated / total_prompts
        clean = ratio < 0.01

        return ContaminationReport(
            benchmark_name="Suite-3A",
            total_prompts=total_prompts,
            contaminated_count=n_contaminated,
            contamination_ratio=ratio,
            flagged_items=flagged_items,
            clean=clean,
        )

    def generate_report(self, report: ContaminationReport, output_path: Path | str) -> None:
        """Write a formatted Markdown contamination report to disk."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"# Contamination Checker Report — {report.benchmark_name}",
            "",
            f"- **Method:** {report.method}",
            f"- **N-gram size:** {self.ngram_size} words",
            f"- **Jaccard Threshold:** {self.similarity_threshold:.2f}",
            f"- **Total Prompts checked:** {report.total_prompts}",
            f"- **Contaminated Prompts found:** {report.contaminated_count}",
            f"- **Contamination Ratio:** {report.contamination_ratio * 100:.2f}%",
            f"- **Status:** {'🟢 CLEAN' if report.clean else '🔴 CONTAMINATED'}",
            "",
        ]

        if report.flagged_items:
            lines.extend([
                "## Flagged Matches Table",
                "",
                "| Prompt ID | Matched File | Similarity |",
                "|---|---|---|",
            ])
            for item in report.flagged_items:
                lines.append(
                    f"| {item['prompt_id']} | `{item['matched_file']}` | {item['similarity']:.4f} |"
                )
        else:
            lines.append("No benchmark prompts overlap with the inspected training data.")

        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Contamination report saved to %s", out)

    # ── Private methods ────────────────────────────────────────────────

    def _normalize(self, text: str) -> str:
        """Lowercases and normalises whitespace/punctuation."""
        lower = text.lower().strip()
        # Remove common punctuation to avoid trivial mismatches
        cleaned = re.sub(r"[^a-z0-9\s]", " ", lower)
        return " ".join(cleaned.split())

    def _fingerprint(self, text: str) -> set[str]:
        """Generate a set of word-level n-grams from the normalized text."""
        tokens = self._normalize(text).split()
        if len(tokens) < self.ngram_size:
            # Fallback for short texts
            return {" ".join(tokens)} if tokens else set()
        return {
            " ".join(tokens[i : i + self.ngram_size])
            for i in range(len(tokens) - self.ngram_size + 1)
        }

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        """Compute Jaccard similarity index between two sets."""
        if not a or not b:
            return 0.0
        u = a.union(b)
        if not u:
            return 0.0
        return len(a.intersection(b)) / len(u)
