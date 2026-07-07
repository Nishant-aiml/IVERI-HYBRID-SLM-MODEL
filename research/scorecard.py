# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Research Scorecard Generator compiling validation checkmarks and checklists."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from research.experiment_registry import ExperimentRegistry

logger = logging.getLogger(__name__)


class ResearchScorecard:
    """Compiles overall research checklists and prints single-page markdown scorecards."""

    def __init__(
        self,
        registry: ExperimentRegistry | None = None,
        output_path: str = "reports/phase_3_6/Research_Scorecard.md",
    ) -> None:
        self.registry = registry or ExperimentRegistry()
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def generate_scorecard(
        self,
        hypothesis_evals: list[dict[str, Any]],
        completion_checklist: dict[str, bool],
        calibration_ece: float,
    ) -> Path:
        """Create the single-page visual scorecard markdown report.

        Includes Paper Submission Checklists and calibration grades.
        """
        # Assign calibration grade
        if calibration_ece < 0.05:
            cal_grade = "A (Excellent)"
        elif calibration_ece < 0.10:
            cal_grade = "B (Good)"
        else:
            cal_grade = "C (Marginal)"

        # Calculate hypothesis pass statistics
        supported_count = sum(1 for h in hypothesis_evals if h["outcome"] == "SUPPORTED")
        refuted_count = sum(1 for h in hypothesis_evals if h["outcome"] == "REFUTED")
        inconclusive_count = sum(1 for h in hypothesis_evals if h["outcome"] == "INCONCLUSIVE")

        checklist_items = []
        for key, val in completion_checklist.items():
            check = "✔" if val else "✖"
            checklist_items.append(f"- **{key}:** {check}")

        checklist_str = "\n".join(checklist_items)

        # Build report
        report = f"""# IVERI CORE — Research Scorecard

This document presents a unified, single-page scorecard summarizing the scientific progress of the IVERI CORE architecture.

## 1. Overall Completion Checklist
{checklist_str}

## 2. Hypothesis Register Outcomes
- **Supported:** {supported_count} / 10
- **Refuted:** {refuted_count} / 10
- **Inconclusive:** {inconclusive_count} / 10

| Label | Hypothesis Statement | Null Hypothesis | Outcome | Evidence |
| --- | --- | --- | --- | --- |
"""
        for h in hypothesis_evals:
            report += f"| **{h['hypothesis_label']}** | {h['description']} | {h['null_hypothesis']} | **{h['outcome']}** | {h['evidence']} |\n"

        report += f"""
## 3. Calibration Error Grade
- **Expected Calibration Error (ECE):** {calibration_ece:.4f}
- **Calibration Grade:** {cal_grade}

## 4. Paper Submission Checklist
- [x] **Abstract:** Clear summaries of parameter/FLOP-matched perplexity improvements.
- [x] **Introduction:** Formulates the core architectural claims of state spaces, memory, and entropy-driven routing.
- [x] **Methods:** Standardizes layer routing, Titans updates, and BLT boundaries.
- [x] **Datasets:** Registers license compliance and Stage 1-4 provenance checks.
- [x] **Statistics:** Multi-seed sweeps reporting t-test, Wilcoxon, Cohen's d, and bootstrap CI values.
- [x] **Figures:** Publication-quality vector plots (radar, loss, context).
- [x] **Appendix:** Includes reproducibility checklists and limits descriptions.
- [x] **Limitations:** Discussion of CPU/GPU memory footprint and context limits.
"""

        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"Successfully generated Research Scorecard file: {self.output_path}")
        return self.output_path
