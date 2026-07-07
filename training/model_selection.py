# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Model checkpoint selection and history index manager for IVERI CORE pretraining.

Ranks checkpoints based on convergence metrics and maintains a detailed history
index JSON file without performing any deletion operations.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CheckpointSelector:
    """Manages index registry of saved checkpoints and ranks them by performance."""

    def __init__(self, log_dir: str | Path = "logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.log_dir / "checkpoint_history.json"
        self.checkpoints: list[dict[str, Any]] = []
        self._load_history()

    def _load_history(self) -> None:
        """Load index log if it exists on disk."""
        if self.history_path.exists():
            try:
                with open(self.history_path, encoding="utf-8") as f:
                    self.checkpoints = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load checkpoint history index: {e}")

    def _save_history(self) -> None:
        """Write index log back to disk."""
        try:
            with open(self.history_path, "w", encoding="utf-8") as f:
                json.dump(self.checkpoints, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save checkpoint history index: {e}")

    def register_checkpoint(
        self,
        path: str | Path,
        step: int,
        train_loss: float,
        val_loss: float,
        perplexity: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a new saved checkpoint in the history index."""
        chk_path = Path(path).resolve()
        entry = {
            "path": str(chk_path.as_posix()),
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "perplexity": perplexity,
            "metadata": metadata or {},
        }
        # Update or append
        existing = next((c for c in self.checkpoints if c["step"] == step), None)
        if existing:
            self.checkpoints.remove(existing)
        self.checkpoints.append(entry)
        self._save_history()

    def get_best_checkpoint(self, metric: str = "val_loss") -> dict[str, Any] | None:
        """Return the best checkpoint entry based on the specified metric."""
        if not self.checkpoints:
            return None

        # Filter out NaN or invalid metrics
        valid_chks = []
        for c in self.checkpoints:
            val = c.get(metric)
            if val is not None and not math.isnan(val) and not math.isinf(val):
                valid_chks.append(c)

        if not valid_chks:
            return None

        if metric in ("val_loss", "train_loss", "perplexity"):
            # Lower is better
            return min(valid_chks, key=lambda x: x[metric])
        else:
            # Higher is better default fallback
            return max(valid_chks, key=lambda x: x[metric])

    def get_latest_checkpoint(self) -> dict[str, Any] | None:
        """Return the latest checkpoint entry based on step number."""
        if not self.checkpoints:
            return None
        return max(self.checkpoints, key=lambda x: x["step"])

    def get_history(self) -> list[dict[str, Any]]:
        """Return the complete checkpoint history list."""
        return sorted(self.checkpoints, key=lambda x: x["step"])


# ── SFT-specific checkpoint selector ─────────────────────────────────────


class SFTCheckpointSelector(CheckpointSelector):
    """Checkpoint selector extended for SFT Phase 3.2 ranking.

    Adds support for ranking checkpoints by ``response_quality_score``
    (higher is better) in addition to the standard ``val_loss`` /
    ``perplexity`` metrics inherited from :class:`CheckpointSelector`.

    All existing behaviour is preserved.  This class is a strict superset
    of the base class.

    Parameters
    ----------
    log_dir:
        Directory to store checkpoint history JSON.
    """

    #: Metrics where higher is better (all others: lower is better).
    _HIGHER_IS_BETTER: frozenset[str] = frozenset(
        {
            "response_quality_score",
            "top1_accuracy",
            "top5_accuracy",
            "keyword_ratio",
            "valid_ratio",
            "avg_entropy",
        }
    )

    def register_checkpoint(
        self,
        path: str | Path,
        step: int,
        train_loss: float,
        val_loss: float,
        perplexity: float,
        metadata: dict[str, Any] | None = None,
        response_quality_score: float | None = None,
    ) -> None:
        """Register a checkpoint with optional SFT quality score.

        Parameters
        ----------
        path:
            Checkpoint file path.
        step:
            Training step.
        train_loss:
            Training loss.
        val_loss:
            Validation loss.
        perplexity:
            Validation perplexity.
        metadata:
            Arbitrary metadata dict.
        response_quality_score:
            Mean quality score from :class:`~evaluation.response_inspector.ResponseInspector`.
            ``None`` = not yet computed.
        """
        meta = dict(metadata or {})
        if response_quality_score is not None:
            meta["response_quality_score"] = response_quality_score

        super().register_checkpoint(
            path=path,
            step=step,
            train_loss=train_loss,
            val_loss=val_loss,
            perplexity=perplexity,
            metadata=meta,
        )
        # Promote response_quality_score to top-level for ranking
        if response_quality_score is not None:
            for chk in self.checkpoints:
                if chk["step"] == step:
                    chk["response_quality_score"] = response_quality_score
            self._save_history()

    def get_best_checkpoint(self, metric: str = "val_loss") -> dict[str, Any] | None:
        """Return the best checkpoint for the given metric.

        Parameters
        ----------
        metric:
            Metric name.  Supports all base metrics plus SFT-specific metrics
            defined in :attr:`_HIGHER_IS_BETTER`.

        Returns
        -------
        dict[str, Any] | None
        """
        if not self.checkpoints:
            return None

        valid_chks = []
        for c in self.checkpoints:
            val = c.get(metric) or c.get("metadata", {}).get(metric)
            if val is not None and not math.isnan(val) and not math.isinf(val):
                valid_chks.append((c, val))

        if not valid_chks:
            return None

        if metric in self._HIGHER_IS_BETTER:
            return max(valid_chks, key=lambda x: x[1])[0]
        return min(valid_chks, key=lambda x: x[1])[0]

    def get_best_sft_checkpoint(self) -> dict[str, Any] | None:
        """Return the best checkpoint ranked jointly by val_loss and quality.

        Uses a combined score: ``joint = val_loss - 0.5 * quality_score``.
        Lower joint score is better.

        Returns
        -------
        dict[str, Any] | None
        """
        if not self.checkpoints:
            return None

        scored = []
        for c in self.checkpoints:
            vl = c.get("val_loss")
            qs = c.get("response_quality_score", c.get("metadata", {}).get("response_quality_score", 0.0))
            if vl is not None and not math.isnan(vl):
                joint = vl - 0.5 * (qs or 0.0)
                scored.append((c, joint))

        if not scored:
            return None
        return min(scored, key=lambda x: x[1])[0]


class CodingCheckpointSelector(SFTCheckpointSelector):
    """Checkpoint selector extended for Phase 3.3 Coding Specialization.

    Adds code_quality_score, syntax_valid_ratio, and execution metrics to
    the list of higher-is-better metrics, and enforces instruction
    retention checks (checkpoints with regression can be filtered out).
    """

    _HIGHER_IS_BETTER: frozenset[str] = frozenset(
        {
            *SFTCheckpointSelector._HIGHER_IS_BETTER,
            "code_quality_score",
            "syntax_valid_ratio",
            "compile_success_ratio",
            "execution_success_ratio",
            "humaneval_pass_at_1",
            "mbpp_pass_at_1",
        }
    )

    def register_checkpoint(
        self,
        path: str | Path,
        step: int,
        train_loss: float,
        val_loss: float,
        perplexity: float,
        metadata: dict[str, Any] | None = None,
        code_quality_score: float | None = None,
        syntax_valid_ratio: float | None = None,
        instruction_retention_ok: bool = True,
    ) -> None:
        """Register a checkpoint with optional coding quality and instruction check."""
        meta = dict(metadata or {})
        if code_quality_score is not None:
            meta["code_quality_score"] = code_quality_score
        if syntax_valid_ratio is not None:
            meta["syntax_valid_ratio"] = syntax_valid_ratio

        meta["instruction_retention_ok"] = instruction_retention_ok
        if not instruction_retention_ok:
            meta["instruction_regression"] = True

        super().register_checkpoint(
            path=path,
            step=step,
            train_loss=train_loss,
            val_loss=val_loss,
            perplexity=perplexity,
            metadata=meta,
        )

        # Promote metrics to top-level for sorting/retrieval
        for chk in self.checkpoints:
            if chk["step"] == step:
                if code_quality_score is not None:
                    chk["code_quality_score"] = code_quality_score
                if syntax_valid_ratio is not None:
                    chk["syntax_valid_ratio"] = syntax_valid_ratio
                chk["instruction_retention_ok"] = instruction_retention_ok
                if not instruction_retention_ok:
                    chk["instruction_regression"] = True
        self._save_history()

    def get_best_coding_checkpoint(self) -> dict[str, Any] | None:
        """Return the best checkpoint ranked jointly by loss, code quality and syntax.

        Excludes checkpoints that have failed instruction retention checks.
        Combined score: ``joint = val_loss - 0.5 * code_quality_score - 0.3 * syntax_valid_ratio``.
        Lower joint score is better.

        Returns
        -------
        dict[str, Any] | None
        """
        if not self.checkpoints:
            return None

        # Filter out checkpoints with instruction regression
        valid_candidates = []
        for c in self.checkpoints:
            meta = c.get("metadata", {})
            if meta.get("instruction_regression", False) or not c.get("instruction_retention_ok", True):
                continue
            valid_candidates.append(c)

        # Fallback to all checkpoints if all regressed (graceful recovery)
        candidates = valid_candidates if valid_candidates else self.checkpoints

        scored = []
        for c in candidates:
            vl = c.get("val_loss")
            cq = c.get("code_quality_score", c.get("metadata", {}).get("code_quality_score", 0.0))
            sv = c.get("syntax_valid_ratio", c.get("metadata", {}).get("syntax_valid_ratio", 0.0))

            if vl is not None and not math.isnan(vl):
                # Normalize defaults if None
                cq_val = cq if cq is not None else 0.0
                sv_val = sv if sv is not None else 0.0
                joint = vl - 0.5 * cq_val - 0.3 * sv_val
                scored.append((c, joint))

        if not scored:
            return None
        return min(scored, key=lambda x: x[1])[0]


class PreferenceCheckpointSelector(CodingCheckpointSelector):
    """Checkpoint selector for Phase 3.4 Preference Optimization.

    Ranks checkpoints prioritising:
    Preference Accuracy -> Reward Margin -> Val Loss -> Instruction Retention -> Coding Retention.
    """

    _HIGHER_IS_BETTER: frozenset[str] = frozenset(
        {
            *CodingCheckpointSelector._HIGHER_IS_BETTER,
            "preference_accuracy",
            "reward_margin",
        }
    )

    def register_checkpoint(
        self,
        path: str | Path,
        step: int,
        train_loss: float,
        val_loss: float,
        perplexity: float,
        metadata: dict[str, Any] | None = None,
        preference_accuracy: float | None = None,
        reward_margin: float | None = None,
        instruction_retention_ok: bool = True,
        coding_retention_ok: bool = True,
    ) -> None:
        """Register a checkpoint with preference accuracy, reward margin and retentions."""
        meta = dict(metadata or {})
        if preference_accuracy is not None:
            meta["preference_accuracy"] = preference_accuracy
        if reward_margin is not None:
            meta["reward_margin"] = reward_margin

        meta["instruction_retention_ok"] = instruction_retention_ok
        meta["coding_retention_ok"] = coding_retention_ok
        
        if not instruction_retention_ok:
            meta["instruction_regression"] = True
        if not coding_retention_ok:
            meta["coding_regression"] = True

        super().register_checkpoint(
            path=path,
            step=step,
            train_loss=train_loss,
            val_loss=val_loss,
            perplexity=perplexity,
            metadata=meta,
            instruction_retention_ok=instruction_retention_ok,
        )

        # Promote metrics to top level
        for chk in self.checkpoints:
            if chk["step"] == step:
                if preference_accuracy is not None:
                    chk["preference_accuracy"] = preference_accuracy
                if reward_margin is not None:
                    chk["reward_margin"] = reward_margin
                chk["instruction_retention_ok"] = instruction_retention_ok
                chk["coding_retention_ok"] = coding_retention_ok
                if not instruction_retention_ok:
                    chk["instruction_regression"] = True
                if not coding_retention_ok:
                    chk["coding_regression"] = True
        self._save_history()

    def get_best_preference_checkpoint(self) -> dict[str, Any] | None:
        """Return the best checkpoint ranked jointly by loss, accuracy, and margin.

        Filters out checkpoints that failed instruction or coding retention checks.
        Combined score: ``joint = val_loss - 2.0 * preference_accuracy - 0.5 * reward_margin``.
        Lower joint score is better.

        Returns
        -------
        dict[str, Any] | None
        """
        if not self.checkpoints:
            return None

        # Filter out checkpoints with regression
        valid_candidates = []
        for c in self.checkpoints:
            meta = c.get("metadata", {})
            has_instruction_regression = meta.get("instruction_regression", False) or not c.get("instruction_retention_ok", True)
            has_coding_regression = meta.get("coding_regression", False) or not c.get("coding_retention_ok", True)
            if has_instruction_regression or has_coding_regression:
                continue
            valid_candidates.append(c)

        # Fallback if all regressed
        candidates = valid_candidates if valid_candidates else self.checkpoints

        scored = []
        for c in candidates:
            vl = c.get("val_loss")
            pref_acc = c.get("preference_accuracy", c.get("metadata", {}).get("preference_accuracy", 0.5))
            reward_margin = c.get("reward_margin", c.get("metadata", {}).get("reward_margin", 0.0))

            if vl is not None and not math.isnan(vl):
                pref_acc_val = pref_acc if pref_acc is not None else 0.5
                reward_margin_val = reward_margin if reward_margin is not None else 0.0
                joint = vl - 2.0 * pref_acc_val - 0.5 * reward_margin_val
                scored.append((c, joint))

        if not scored:
            return None
        return min(scored, key=lambda x: x[1])[0]



