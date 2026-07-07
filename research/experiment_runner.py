# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Experiment runner managing training and evaluation sweeps for Stage 5 validation."""

from __future__ import annotations

import json
import logging
import time
import subprocess
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.base_config import IVERIConfig
from training.trainer import Trainer

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Manages single-run executions under strict parity constraints.

    Captures hardware info, git status, and metrics log files.
    """

    def __init__(self, config: IVERIConfig) -> None:
        self.config = config

    def _get_git_state(self) -> dict[str, str]:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
            ).stdout.strip() or "unknown"
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=False
            ).stdout.strip() or "unknown"
            return {"commit": commit, "branch": branch}
        except Exception:
            return {"commit": "unknown", "branch": "unknown"}

    def run_experiment(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        max_steps: int = 100,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Execute a matched training run and return evaluation results.

        Args:
            model: PyTorch model block (IVERIModel or BaselineTransformer/SSM).
            train_loader: DataLoader supplying training examples.
            val_loader: DataLoader supplying evaluation examples.
            max_steps: Step budget.
            seed: Initial random seed.

        Returns:
            dict[str, Any]: Collected training and evaluation stats.
        """
        logger.info(f"Setting seed: {seed} and starting experiment run...")
        # Set all standard seeds
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        start_time = time.perf_counter()
        device = torch.device(self.config.hardware.device)
        model.to(device)

        # Standard baseline optimization using AdamW
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        criterion = nn.CrossEntropyLoss()

        train_losses: list[float] = []
        model.train()

        steps_executed = 0
        epoch = 0

        # Simple mini-batch training loop matching training/trainer.py style
        while steps_executed < max_steps:
            for batch in train_loader:
                if steps_executed >= max_steps:
                    break

                optimizer.zero_grad()
                if isinstance(batch, (tuple, list)):
                    # Handle SFT masked pairs vs pretrain raw inputs
                    if len(batch) >= 2 and isinstance(batch[1], torch.Tensor):
                        inputs, targets = batch[0].to(device), batch[1].to(device)
                    else:
                        inputs = batch[0].to(device)
                        targets = inputs.clone()
                else:
                    inputs = batch.to(device)
                    targets = inputs.clone()

                # Model forward pass
                outputs = model(inputs)
                if isinstance(outputs, dict):
                    logits = outputs["logits"]
                else:
                    logits = outputs

                # Standard flattened CrossEntropy
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                steps_executed += 1

            epoch += 1

        end_time = time.perf_counter()
        runtime = end_time - start_time

        # Run final validation perplexity
        val_loss = 0.0
        val_ppl = 0.0
        model.eval()
        if val_loader:
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, (tuple, list)):
                        inputs = batch[0].to(device)
                        targets = batch[1].to(device) if len(batch) >= 2 else inputs
                    else:
                        inputs = batch.to(device)
                        targets = inputs.clone()
                    outputs = model(inputs)
                    logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                    v_loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                    val_losses.append(v_loss.item())
            val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
            val_ppl = torch.exp(torch.tensor(val_loss)).item()

        git_state = self._get_git_state()

        metrics = {
            "train_loss": sum(train_losses) / len(train_losses) if train_losses else 0.0,
            "val_loss": val_loss,
            "perplexity": val_ppl,
            "runtime_seconds": runtime,
            "steps": steps_executed,
            "seed": seed,
            "git_commit": git_state["commit"],
            "git_branch": git_state["branch"],
            "config_hash": hashlib.sha256(json.dumps(self.config.to_dict(), default=str).encode()).hexdigest(),
        }

        logger.info(f"Finished experiment run. Val Loss: {val_loss:.4f}, Perplexity: {val_ppl:.4f}")
        return metrics
import hashlib
