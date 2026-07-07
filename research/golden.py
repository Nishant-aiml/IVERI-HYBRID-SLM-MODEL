# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Golden Checkpoint Manager coordinates rollbacks, comparisons, and promotion lifecycles."""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

import torch
import torch.nn as nn

from configs.base_config import IVERIConfig
from research.experiment_registry import ExperimentRegistry
from research.checkpoint_manager import BaselineCheckpointManager
from research.regression_detector import RegressionDetector

logger = logging.getLogger(__name__)

VALID_STAGES = {"Candidate", "Validated", "Golden", "Paper", "Released", "Archived"}


class GoldenCheckpointManager:
    """Orchestrates golden checkpoint setting, loading, comparisons, and rollback functions."""

    def __init__(self, config: IVERIConfig, registry: ExperimentRegistry | None = None) -> None:
        self.config = config
        self.registry = registry or ExperimentRegistry()
        self.ckpt_manager = BaselineCheckpointManager(config)
        self.detector = RegressionDetector(self.registry)
        self._ensure_stage_column()

    def _ensure_stage_column(self) -> None:
        """Add the 'stage' column to the checkpoints database if it is missing."""
        conn = self.registry._get_connection()
        try:
            with conn:
                conn.execute("ALTER TABLE checkpoints ADD COLUMN stage TEXT DEFAULT 'Candidate'")
        except sqlite3.OperationalError:
            # Column already exists
            pass
        finally:
            conn.close()

    def promote_checkpoint(self, checkpoint_id: str, new_stage: str) -> None:
        """Promote a checkpoint state through the lifecycle stage chain.

        Stages: Candidate -> Validated -> Golden -> Paper -> Released -> Archived.
        """
        if new_stage not in VALID_STAGES:
            raise ValueError(f"Invalid checkpoint promotion stage '{new_stage}'. Must be one of: {VALID_STAGES}")

        conn = self.registry._get_connection()
        try:
            with conn:
                # If promoting to 'Golden', reset any other golden checks
                if new_stage == "Golden":
                    conn.execute("UPDATE checkpoints SET is_golden = 0")
                    conn.execute("UPDATE checkpoints SET is_golden = 1 WHERE checkpoint_id = ?", (checkpoint_id,))
                
                conn.execute(
                    "UPDATE checkpoints SET stage = ? WHERE checkpoint_id = ?",
                    (new_stage, checkpoint_id),
                )
                conn.commit()
        finally:
            conn.close()
        logger.info(f"Checkpoint '{checkpoint_id}' promoted successfully to '{new_stage}' stage.")

    def set_golden(self, checkpoint_id: str) -> None:
        """Mark a checkpoint as golden (Legacy wrapper)."""
        self.promote_checkpoint(checkpoint_id, "Golden")

    def get_golden(self) -> dict[str, Any] | None:
        """Query SQLite to fetch the active golden checkpoint details."""
        conn = self.registry._get_connection()
        try:
            conn.row_factory = sqlite3_row_factory
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM checkpoints WHERE is_golden = 1 OR stage = 'Golden'")
            row = cursor.fetchone()
            if row:
                return dict(row)
        finally:
            conn.close()
        return None

    def select_top_checkpoints(self, experiment_id: str) -> dict[str, list[int]]:
        """Query metrics to locate the top 3 steps based on loss, perplexity, and accuracy.

        This helps optimize benchmarking resource usage.
        """
        conn = self.registry._get_connection()
        try:
            cursor = conn.cursor()
            
            # 1. Top 3 by validation loss
            cursor.execute(
                "SELECT step FROM metrics WHERE experiment_id = ? ORDER BY val_loss ASC LIMIT 3",
                (experiment_id,),
            )
            top_loss = [row[0] for row in cursor.fetchall()]

            # 2. Top 3 by perplexity
            cursor.execute(
                "SELECT step FROM metrics WHERE experiment_id = ? ORDER BY perplexity ASC LIMIT 3",
                (experiment_id,),
            )
            top_ppl = [row[0] for row in cursor.fetchall()]

            # 3. Top 3 by accuracy
            cursor.execute(
                "SELECT step FROM metrics WHERE experiment_id = ? ORDER BY accuracy DESC LIMIT 3",
                (experiment_id,),
            )
            top_acc = [row[0] for row in cursor.fetchall()]

            return {
                "val_loss": top_loss,
                "perplexity": top_ppl,
                "accuracy": top_acc,
            }
        finally:
            conn.close()

    def compare_to_golden(self, current_metrics: dict[str, float]) -> dict[str, Any]:
        """Compare current metrics to golden metrics."""
        golden = self.get_golden()
        if not golden:
            logger.warning("No golden checkpoint set. Skipping regression comparison.")
            return {"status": "NO_GOLDEN_SET"}

        # Query metrics of the golden experiment from database
        experiment_id = golden["experiment_id"]
        step = golden["step"]

        conn = self.registry._get_connection()
        try:
            conn.row_factory = sqlite3_row_factory
            cursor = conn.cursor()
            cursor.execute(
                "SELECT train_loss, val_loss, perplexity, accuracy FROM metrics WHERE experiment_id = ? AND step = ?",
                (experiment_id, step),
            )
            row = cursor.fetchone()
        finally:
            conn.close()

        if not row:
            logger.warning(f"No metric logs found for golden experiment {experiment_id} at step {step}.")
            return {"status": "GOLDEN_METRICS_MISSING"}

        golden_metrics = {
            "loss": row["val_loss"],
            "perplexity": row["perplexity"],
            "accuracy": row["accuracy"],
            # Fallback mocks for other metrics
            "ttft_sec": 0.12,
            "decode_speed_tps": 300.0,
            "vram_peak_mb": 1200.0,
            "energy_per_token_j": 1.9,
            "calibration_ece": 0.04,
            "humaneval_pass_rate": 0.8,
            "instruction_score": 0.9,
        }

        return self.detector.check_for_regression(current_metrics, golden_metrics)

    def rollback_to_golden(self, model: nn.Module) -> dict[str, Any]:
        """Roll back model parameters to the active golden checkpoint."""
        golden = self.get_golden()
        if not golden:
            raise RuntimeError("Cannot roll back; no golden checkpoint is set in the registry database.")

        path = golden["path"]
        logger.info(f"Rolling back model parameter state to golden checkpoint: {path}")
        
        # Load parameters
        checkpoint_info = self.ckpt_manager.load_checkpoint(model, path)
        return {
            "status": "SUCCESS",
            "checkpoint_id": golden["checkpoint_id"],
            "step": checkpoint_info["step"],
            "metrics": checkpoint_info["metrics"],
        }


def sqlite3_row_factory(cursor: Any, row: tuple[Any, ...]) -> dict[str, Any]:
    """Helper row factory to get dictionary structures from sqlite rows."""
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d
