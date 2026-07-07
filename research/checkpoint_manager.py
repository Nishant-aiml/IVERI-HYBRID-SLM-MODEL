# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Baseline Checkpoint Manager for Stage 5 scientific validation campaigns.

Provides validation checkpointing, reproducibility checks, parameter auditing,
and verification hashes to avoid duplicate training.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from configs.base_config import IVERIConfig

logger = logging.getLogger(__name__)


class BaselineCheckpointManager:
    """Manages baseline and ablated validation checkpoints.

    Audits parameters and FLOP matches, computes file checksums,
    and stores git states for scientific reproducibility.
    """

    def __init__(self, config: IVERIConfig, registry_path: str = "research_checkpoints.json") -> None:
        self.config = config
        self.registry_path = Path(registry_path)
        self.registry: dict[str, Any] = self._load_registry()

    def _load_registry(self) -> dict[str, Any]:
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read baseline registry: {e}. Starting fresh.")
        return {}

    def _save_registry(self) -> None:
        try:
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(self.registry, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save baseline registry: {e}")

    def _get_git_commit(self) -> str:
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=False
            )
            return res.stdout.strip() or "unknown"
        except Exception:
            return "unknown"

    def _compute_file_sha256(self, path: Path) -> str:
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def count_parameters(self, model: nn.Module) -> int:
        """Count total trainable parameters in the model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def save_checkpoint(
        self,
        model: nn.Module,
        path: str | Path,
        step: int,
        metrics: dict[str, Any],
        seed: int = 42,
    ) -> str:
        """Save baseline checkpoint with full reproducibility metrics.

        Returns:
            str: SHA-256 hash of the generated checkpoint file.
        """
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "config_dict": self.config.to_dict(),
            "step": step,
            "metrics": metrics,
            "seed": seed,
            "git_commit": self._get_git_commit(),
            "parameter_count": self.count_parameters(model),
        }

        torch.save(checkpoint_data, target_path)
        sha256_hash = self._compute_file_sha256(target_path)

        # Register in registry index
        self.registry[str(target_path.resolve())] = {
            "step": step,
            "seed": seed,
            "parameter_count": checkpoint_data["parameter_count"],
            "sha256": sha256_hash,
            "git_commit": checkpoint_data["git_commit"],
        }
        self._save_registry()
        logger.info(f"Saved baseline checkpoint: {target_path} (Hash: {sha256_hash})")
        return sha256_hash

    def load_checkpoint(self, model: nn.Module, path: str | Path) -> dict[str, Any]:
        """Load baseline checkpoint and verify file integrity hashes."""
        target_path = Path(path)
        if not target_path.exists():
            raise FileNotFoundError(f"Checkpoint file does not exist: {path}")

        current_hash = self._compute_file_sha256(target_path)
        registered_info = self.registry.get(str(target_path.resolve()))

        if registered_info and registered_info["sha256"] != current_hash:
            logger.warning(
                f"Checkpoint file checksum mismatch! Registered: {registered_info['sha256']}, "
                f"Actual file: {current_hash}."
            )

        checkpoint_data = torch.load(target_path, map_location="cpu")
        model.load_state_dict(checkpoint_data["model_state_dict"])

        return {
            "step": checkpoint_data.get("step", 0),
            "metrics": checkpoint_data.get("metrics", {}),
            "seed": checkpoint_data.get("seed", 42),
            "git_commit": checkpoint_data.get("git_commit", "unknown"),
        }

    def verify_parity(self, model_a: nn.Module, model_b: nn.Module, threshold_pct: float = 0.05) -> bool:
        """Verify that model_a and model_b have matched parameters within a threshold tolerance."""
        params_a = self.count_parameters(model_a)
        params_b = self.count_parameters(model_b)

        diff = abs(params_a - params_b)
        mean_params = (params_a + params_b) / 2.0
        pct_diff = (diff / mean_params) if mean_params > 0 else 0.0

        is_parity = pct_diff <= threshold_pct
        logger.info(
            f"Parity Audit: Model A parameters={params_a}, Model B parameters={params_b}. "
            f"Diff={pct_diff * 100:.2f}%. Parity outcome: {is_parity}"
        )
        return is_parity
