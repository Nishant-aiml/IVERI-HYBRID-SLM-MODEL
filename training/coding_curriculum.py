# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Three-stage curriculum scheduler for IVERI CORE Phase 3.3 coding specialization.

The curriculum progressively shifts the training data mix from raw code fluency
to instruction following to competitive programming, matching the learning
trajectory recommended in the Phase 3.3 Research Contract.

Curriculum stages
-----------------
- **Stage 0** (0–33 % of steps): Raw code fluency.
  ``the_stack_v2_deep`` (0.70) + ``opencode_instruct`` (0.30).
- **Stage 1** (33–66 % of steps): Code instruction following.
  ``nemotron_competitive`` (0.40) + ``leetcode`` (0.40) + ``opencode_instruct`` (0.20).
- **Stage 2** (66–100 % of steps): Competitive programming.
  ``nemotron_competitive`` (0.40) + ``codeforces`` (0.40) + ``leetcode`` (0.20).

Examples
--------
>>> from training.coding_curriculum import CodingCurriculum
>>> c = CodingCurriculum()
>>> c.get_stage_index(0, 100)
0
>>> c.get_stage_index(40, 100)
1
>>> c.get_stage_index(80, 100)
2
>>> weights = c.get_active_datasets(80, 100)
>>> "nemotron_competitive" in weights
True
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Default curriculum definition ──────────────────────────────────────────

_DEFAULT_STAGES: list[dict[str, Any]] = [
    {
        "name": "code_fluency",
        "step_start_pct": 0.0,
        "step_end_pct": 0.333,
        "dataset_weights": {
            "the_stack_v2_deep": 0.70,
            "opencode_instruct": 0.30,
        },
        "description": "Raw code fluency — pretrain-style byte-level code data",
    },
    {
        "name": "code_instruction",
        "step_start_pct": 0.333,
        "step_end_pct": 0.666,
        "dataset_weights": {
            "nemotron_competitive": 0.40,
            "leetcode": 0.40,
            "opencode_instruct": 0.20,
        },
        "description": "Code instruction following — LeetCode + OpenCode SFT",
    },
    {
        "name": "competitive_programming",
        "step_start_pct": 0.666,
        "step_end_pct": 1.001,   # slightly > 1 to include step == max_steps
        "dataset_weights": {
            "nemotron_competitive": 0.40,
            "codeforces": 0.40,
            "leetcode": 0.20,
        },
        "description": "Competitive programming — Nemotron + Codeforces reasoning",
    },
]


# ── Dataclasses ────────────────────────────────────────────────────────────


@dataclass
class CurriculumStage:
    """One stage of the coding curriculum.

    Attributes
    ----------
    name:
        Identifier (e.g. ``\"code_fluency\"``).
    step_start_pct:
        Start of this stage as a fraction of total steps (inclusive).
    step_end_pct:
        End of this stage as a fraction of total steps (exclusive).
    dataset_weights:
        Mapping of dataset name → mixing weight.  Weights should sum to 1.0.
    description:
        Human-readable description.
    stage_index:
        Zero-based stage index.
    """

    name: str
    step_start_pct: float
    step_end_pct: float
    dataset_weights: dict[str, float] = field(default_factory=dict)
    description: str = ""
    stage_index: int = 0


# ── Main curriculum class ──────────────────────────────────────────────────


class CodingCurriculum:
    """Three-stage coding curriculum for IVERI CORE Phase 3.3.

    Parameters
    ----------
    stages:
        Custom stage definitions (list of dicts matching
        :class:`CurriculumStage` fields).  Defaults to the built-in
        3-stage curriculum if ``None``.
    num_stages:
        Number of stages to use from the built-in curriculum (1, 2, or 3).
        Ignored when ``stages`` is explicitly provided.
    """

    def __init__(
        self,
        stages: list[dict[str, Any]] | None = None,
        num_stages: int = 3,
    ) -> None:
        if stages is not None:
            raw_stages = stages
        else:
            raw_stages = _DEFAULT_STAGES[:max(1, min(num_stages, len(_DEFAULT_STAGES)))]

        self._stages: list[CurriculumStage] = [
            CurriculumStage(
                name=s["name"],
                step_start_pct=s["step_start_pct"],
                step_end_pct=s["step_end_pct"],
                dataset_weights=s["dataset_weights"],
                description=s.get("description", ""),
                stage_index=i,
            )
            for i, s in enumerate(raw_stages)
        ]

        logger.info(
            "CodingCurriculum initialized with %d stages: %s",
            len(self._stages),
            [s.name for s in self._stages],
        )

    # ── Public API ─────────────────────────────────────────────────────

    def get_stage(self, step: int, max_steps: int) -> CurriculumStage:
        """Return the :class:`CurriculumStage` active at the given step.

        Parameters
        ----------
        step:
            Current training step (0-indexed).
        max_steps:
            Total number of training steps.

        Returns
        -------
        CurriculumStage
        """
        pct = step / max(max_steps, 1)
        for stage in self._stages:
            if stage.step_start_pct <= pct < stage.step_end_pct:
                return stage
        # Fallback: return last stage
        return self._stages[-1]

    def get_stage_index(self, step: int, max_steps: int) -> int:
        """Return the zero-based index of the active curriculum stage.

        Parameters
        ----------
        step:
            Current training step.
        max_steps:
            Total steps.

        Returns
        -------
        int
            0, 1, or 2.
        """
        return self.get_stage(step, max_steps).stage_index

    def get_active_datasets(self, step: int, max_steps: int) -> dict[str, float]:
        """Return the dataset mixing weights for the current curriculum stage.

        Parameters
        ----------
        step:
            Current training step.
        max_steps:
            Total steps.

        Returns
        -------
        dict[str, float]
            Mapping of dataset name → mixing weight (sums to ~1.0).
        """
        return dict(self.get_stage(step, max_steps).dataset_weights)

    def log_stage_transition(
        self,
        step: int,
        prev_stage: CurriculumStage | None,
        new_stage: CurriculumStage,
    ) -> dict[str, Any]:
        """Build a log entry for a curriculum stage transition.

        Parameters
        ----------
        step:
            Step at which the transition occurred.
        prev_stage:
            Previous stage (or ``None`` for the initial assignment).
        new_stage:
            Newly activated stage.

        Returns
        -------
        dict[str, Any]
            Log entry suitable for inclusion in ``curriculum_stage_history``.
        """
        entry = {
            "step": step,
            "new_stage_index": new_stage.stage_index,
            "new_stage_name": new_stage.name,
            "new_stage_datasets": new_stage.dataset_weights,
            "description": new_stage.description,
        }
        if prev_stage is not None:
            entry["prev_stage_name"] = prev_stage.name
        logger.info(
            "[Curriculum] Step %d → Stage %d: %s (datasets: %s)",
            step,
            new_stage.stage_index,
            new_stage.name,
            list(new_stage.dataset_weights),
        )
        return entry

    @property
    def stages(self) -> list[CurriculumStage]:
        """Return all curriculum stages in order."""
        return list(self._stages)

    @property
    def num_stages(self) -> int:
        """Total number of curriculum stages."""
        return len(self._stages)
