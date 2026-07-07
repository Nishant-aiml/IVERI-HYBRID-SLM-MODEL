# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint comparison capabilities for evaluating relative model quality (Phase 2.5).

Compares configurations, parameter shapes, training steps, and performance metrics between two checkpoints.
Marks them as NOT DIRECTLY COMPARABLE if shape or architecture version mismatches occur.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

import torch


class CheckpointComparator:
    """Compares configuration, metadata, and weights between two checkpoints."""

    def __init__(self) -> None:
        """Initialize the CheckpointComparator."""
        pass

    def _hash_dict(self, d: dict[str, Any] | None) -> str:
        """Compute MD5 hash of a dictionary for quick config comparison."""
        if not d:
            return ""
        # Sort keys to ensure deterministic serialization
        serialized = json.dumps(d, sort_keys=True, default=str)
        return hashlib.md5(serialized.encode("utf-8")).hexdigest()

    def compare(
        self,
        path_a: str | pathlib.Path,
        path_b: str | pathlib.Path,
    ) -> dict[str, Any]:
        """Perform comparison of two checkpoint files.

        Args:
            path_a: File path to checkpoint A.
            path_b: File path to checkpoint B.

        Returns:
            Comparison metrics dictionary.
        """
        path_a = pathlib.Path(path_a)
        path_b = pathlib.Path(path_b)

        if not path_a.exists():
            raise FileNotFoundError(f"Checkpoint A not found: {path_a}")
        if not path_b.exists():
            raise FileNotFoundError(f"Checkpoint B not found: {path_b}")

        # Load checkpoints on CPU to avoid device mismatch
        ckpt_a = torch.load(path_a, map_location="cpu", weights_only=False)
        ckpt_b = torch.load(path_b, map_location="cpu", weights_only=False)

        # Extract config dicts
        cfg_a = ckpt_a.get("config") or ckpt_a.get("config_dict", {})
        cfg_b = ckpt_b.get("config") or ckpt_b.get("config_dict", {})

        hash_a = self._hash_dict(cfg_a)
        hash_b = self._hash_dict(cfg_b)

        # Count parameters in state dicts
        def get_param_count(state_dict: dict[str, Any]) -> int:
            return sum(tensor.numel() for tensor in state_dict.values() if isinstance(tensor, torch.Tensor))

        state_a = ckpt_a.get("model_state_dict", {})
        state_b = ckpt_b.get("model_state_dict", {})

        params_a = get_param_count(state_a)
        params_b = get_param_count(state_b)

        arch_a = ckpt_a.get("architecture_version", "N/A")
        arch_b = ckpt_b.get("architecture_version", "N/A")

        git_a = ckpt_a.get("git_commit", cfg_a.get("logging", {}).get("git_commit", "N/A"))
        git_b = ckpt_b.get("git_commit", cfg_b.get("logging", {}).get("git_commit", "N/A"))

        # Determine comparability: check structural/shape invariants
        comparable = True
        mismatch_reasons: list[str] = []

        if arch_a != arch_b:
            comparable = False
            mismatch_reasons.append(f"Architecture version mismatch: A='{arch_a}', B='{arch_b}'")

        if params_a != params_b:
            comparable = False
            mismatch_reasons.append(f"Parameter count mismatch: A={params_a:,}, B={params_b:,}")

        # Compare model shape parameters explicitly if config is stored
        if cfg_a and cfg_b:
            model_cfg_a = cfg_a.get("model", {})
            model_cfg_b = cfg_b.get("model", {})
            for key in ["hidden_dim", "num_layers", "num_heads", "num_experts", "max_recursion_depth"]:
                val_a = model_cfg_a.get(key)
                val_b = model_cfg_b.get(key)
                if val_a != val_b:
                    comparable = False
                    mismatch_reasons.append(f"Structural config key '{key}' mismatch: A={val_a}, B={val_b}")

        # Construct comparison output
        comp_status = "COMPARABLE" if comparable else "NOT DIRECTLY COMPARABLE"

        result: dict[str, Any] = {
            "status": comp_status,
            "comparable": comparable,
            "mismatch_reasons": mismatch_reasons,
            "checkpoint_a": {
                "path": str(path_a),
                "step": ckpt_a.get("step", 0),
                "epoch": ckpt_a.get("epoch", 0),
                "parameter_count": params_a,
                "architecture_version": arch_a,
                "git_commit": git_a,
                "config_hash": hash_a,
            },
            "checkpoint_b": {
                "path": str(path_b),
                "step": ckpt_b.get("step", 0),
                "epoch": ckpt_b.get("epoch", 0),
                "parameter_count": params_b,
                "architecture_version": arch_b,
                "git_commit": git_b,
                "config_hash": hash_b,
            },
            "diffs": {},
        }

        # Calculate deltas only if structures are comparable
        if comparable:
            metrics_a = ckpt_a.get("metrics", {})
            metrics_b = ckpt_b.get("metrics", {})

            # Standard loss/perplexity delta (B - A)
            loss_a = metrics_a.get("loss") or metrics_a.get("train_loss") or metrics_a.get("val_loss") or 0.0
            loss_b = metrics_b.get("loss") or metrics_b.get("train_loss") or metrics_b.get("val_loss") or 0.0

            loss_diff = loss_b - loss_a

            # Compute parameter weights L2 difference
            weight_diffs = {}
            total_weight_diff_sq = 0.0
            for name, tensor_a in state_a.items():
                if name in state_b:
                    tensor_b = state_b[name]
                    if tensor_a.shape == tensor_b.shape:
                        diff_norm = (tensor_b.float() - tensor_a.float()).pow(2).sum().sqrt().item()
                        weight_diffs[name] = diff_norm
                        total_weight_diff_sq += diff_norm**2

            result["diffs"] = {
                "loss_diff": loss_diff,
                "step_diff": ckpt_b.get("step", 0) - ckpt_a.get("step", 0),
                "epoch_diff": ckpt_b.get("epoch", 0) - ckpt_a.get("epoch", 0),
                "weight_diff_l2_norm": float(total_weight_diff_sq**0.5),
                "per_layer_weight_diff": weight_diffs,
            }

        return result
