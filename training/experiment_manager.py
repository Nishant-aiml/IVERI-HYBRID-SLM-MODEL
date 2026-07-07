# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Experiment management, naming, git version logging, and resume support for IVERI.

Maintains strict metadata snapshots and manages experiment subdirectories to
guarantee reproducibility across multiple pretraining runs.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from configs.base_config import IVERIConfig
from training.logger import _git_info, _system_info

logger = logging.getLogger(__name__)


class ExperimentManager:
    """Manages directory structures, config snapshots, git metadata, and resume checkpoints."""

    def __init__(self, config: IVERIConfig, run_name: str | None = None) -> None:
        self.config = config

        # Derive naming
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_name = run_name or config.logging.run_name or f"iveri_pretrain_{timestamp}"

        # Setup paths
        self.experiment_dir = Path(config.logging.save_dir) / self.run_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.snapshot_path = self.experiment_dir / "config_snapshot.json"
        self.metadata_path = self.experiment_dir / "experiment_metadata.json"
        self.resume_path = self.experiment_dir / "resume_metadata.json"

    def setup_run(self, seed: int = 42, dataset_version: str = "", dataset_hash: str = "") -> None:
        """Initialize directory, save configuration snapshot, and log git commit info."""
        # 1. Save config snapshot
        with open(self.snapshot_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=4)

        # 2. Gather git and system info
        git_info = _git_info()
        sys_info = _system_info()

        metadata = {
            "run_name": self.run_name,
            "seed": seed,
            "dataset_version": dataset_version,
            "dataset_hash": dataset_hash,
            "git_commit": git_info.get("git_commit", ""),
            "git_branch": git_info.get("git_branch", ""),
            "system_info": sys_info,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        # 3. Save experiment metadata
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

        logger.info(f"Initialized experiment run '{self.run_name}' at {self.experiment_dir}")

    def save_resume_metadata(self, step: int, epoch: int, checkpoint_file: Path) -> None:
        """Save step and checkpoint path to resume_metadata.json."""
        resume_info = {
            "step": step,
            "epoch": epoch,
            "latest_checkpoint": str(checkpoint_file.resolve().as_posix()),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(self.resume_path, "w", encoding="utf-8") as f:
            json.dump(resume_info, f, indent=4)

    def load_resume_metadata(self) -> dict[str, Any] | None:
        """Load step and checkpoint path from resume_metadata.json if it exists."""
        if not self.resume_path.exists():
            return None
        try:
            with open(self.resume_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load resume metadata: {e}")
            return None

    @staticmethod
    def set_seed(seed: int) -> None:
        """Set all random seeds for exact reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Enforce deterministic algorithms (caution: some models might not support this fully, but good default)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info(f"Reproducibility seeds initialized to: {seed}")

    def log_sft_metadata(
        self,
        dataset_mixture: list[str] | None = None,
        conversation_template: str = "alpaca",
        train_on_prompt: bool = False,
        dataset_versions: dict[str, str] | None = None,
        pretrained_checkpoint: str = "",
        extra: dict | None = None,
    ) -> None:
        """Save SFT-specific run metadata to experiment directory.

        Tracks dataset mixture, conversation template, masking configuration,
        and dataset versions for full reproducibility of SFT runs.

        Parameters
        ----------
        dataset_mixture:
            List of dataset names used in this SFT run.
        conversation_template:
            Formatter template name (``"alpaca"``, ``"chat"``).
        train_on_prompt:
            Whether loss was computed on prompt bytes.
        dataset_versions:
            Mapping of dataset name → version string.
        pretrained_checkpoint:
            Path to the pretrained checkpoint loaded before SFT.
        extra:
            Additional arbitrary metadata to record.
        """
        sft_meta = {
            "training_stage": "Stage 2 SFT",
            "dataset_mixture": dataset_mixture or [],
            "conversation_template": conversation_template,
            "train_on_prompt": train_on_prompt,
            "ignore_prompt_loss": not train_on_prompt,
            "dataset_versions": dataset_versions or {},
            "pretrained_checkpoint": pretrained_checkpoint,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if extra:
            sft_meta.update(extra)

        sft_path = self.experiment_dir / "sft_metadata.json"
        try:
            with open(sft_path, "w", encoding="utf-8") as f:
                json.dump(sft_meta, f, indent=4)
            logger.info(f"SFT metadata saved to {sft_path}")
        except Exception as exc:
            logger.error(f"Failed to save SFT metadata: {exc}")

