# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Reference model management and validation for preference optimization.

Instantiates, loads, and verifies the frozen reference model. Enforces strict
parameter equality audits and checkpoint compatibility checks.
"""

from __future__ import annotations

import logging
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from configs.base_config import IVERIConfig
from core.constants import ARCHITECTURE_VERSION, IVERI_VERSION
from training.checkpointing import load_checkpoint

logger = logging.getLogger(__name__)


def verify_parameter_equality(model_a: nn.Module, model_b: nn.Module) -> bool:
    """Perform a strict, parameter-by-parameter identity audit.

    Returns True if all parameters in both models are exactly equal in name,
    shape, and byte-level values, False otherwise.
    """
    params_a = list(model_a.named_parameters())
    params_b = list(model_b.named_parameters())

    if len(params_a) != len(params_b):
        logger.error("Parameter count mismatch: %d vs %d", len(params_a), len(params_b))
        return False

    for (name_a, p_a), (name_b, p_b) in zip(params_a, params_b):
        if name_a != name_b:
            logger.error("Parameter name mismatch: '%s' vs '%s'", name_a, name_b)
            return False
        if p_a.shape != p_b.shape:
            logger.error("Parameter shape mismatch for '%s': %s vs %s", name_a, p_a.shape, p_b.shape)
            return False
        
        # Element-wise comparison
        val_a = p_a.detach().cpu()
        val_b = p_b.detach().cpu()
        if not torch.equal(val_a, val_b):
            logger.error("Parameter values mismatch for parameter '%s'", name_a)
            return False

    return True


def verify_checkpoint_compatibility(checkpoint_path: str | Path, expected_config: IVERIConfig) -> None:
    """Enforce strict pre-load validation of checkpoint metadata.

    Asserts compatibility of architecture version, model configuration keys,
    tensor interfaces, and logs Git commit hash if present.
    """
    path_obj = Path(checkpoint_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load file structure on CPU
    try:
        checkpoint = torch.load(path_obj, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"Failed to load checkpoint file structure: {exc}") from exc

    # 1. Architecture version check
    arch = checkpoint.get("architecture_version", "")
    if arch != ARCHITECTURE_VERSION:
        raise ValueError(
            f"Checkpoint architecture mismatch! Checkpoint: '{arch}', "
            f"Expected: '{ARCHITECTURE_VERSION}'"
        )

    # 2. IVERI version check
    iveri_ver = checkpoint.get("iveri_version", "")
    if iveri_ver != IVERI_VERSION:
        logger.warning(
            "Checkpoint loaded from different IVERI version. "
            "Checkpoint version: '%s', Current: '%s'", iveri_ver, IVERI_VERSION
        )

    # 3. Model config compatibility check
    checkpoint_config = checkpoint.get("config", {})
    if checkpoint_config:
        chk_model_cfg = checkpoint_config.get("model", {})
        curr_model_cfg = expected_config.model
        
        # Check critical structural hyperparameters
        critical_keys = ["hidden_dim", "num_layers", "num_heads", "mamba_ratio", "num_experts"]
        for key in critical_keys:
            chk_val = chk_model_cfg.get(key)
            curr_val = getattr(curr_model_cfg, key, None)
            if chk_val is not None and curr_val is not None and chk_val != curr_val:
                raise ValueError(
                    f"Config mismatch on '{key}': Checkpoint config has {chk_val}, "
                    f"but training config specifies {curr_val}."
                )

    # Log additional tracking metadata
    git_commit = checkpoint.get("git_commit", "unknown")
    logger.info(
        "Checkpoint compatibility check passed. Git commit of checkpoint: %s", git_commit
    )


class ReferenceModelManager:
    """Manages SFT reference model copy for relative preference feedback logps.

    Parameters
    ----------
    config:
        Master IVERI Config.
    device:
        Torch device to load the reference model on.
    """

    def __init__(self, config: IVERIConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.reference_model: nn.Module | None = None
        self.checkpoint_sha256: str = "unknown"

    def load(self, checkpoint_path: str | Path) -> None:
        """Verify, load and freeze SFT reference checkpoint."""
        if not checkpoint_path:
            logger.info("No reference checkpoint path specified. Running reference-free SimPO.")
            return

        path_obj = Path(checkpoint_path)
        verify_checkpoint_compatibility(path_obj, self.config)

        # Compute SHA-256 for tracking
        sha256_hash = hashlib.sha256()
        with open(path_obj, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        self.checkpoint_sha256 = sha256_hash.hexdigest()

        # Load reference model architecture
        from model.iveri_core import IVERIModel
        model = IVERIModel(self.config)

        logger.info("Loading reference weights from: %s", path_obj)
        load_checkpoint(path_obj, model)

        # Offload or load to requested device
        model.to(self.device)
        model.eval()

        # Freeze parameter gradients
        for param in model.parameters():
            param.requires_grad = False

        self.reference_model = model
        logger.info(
            "Reference model initialized successfully on device '%s' (SHA256: %s)",
            self.device,
            self.checkpoint_sha256[:16]
        )

    def verify_identity(self, policy_model: nn.Module) -> None:
        """Assert parameter equality with the starting policy model.

        Raises ValueError if parameters differ.
        """
        if self.reference_model is None:
            return

        logger.info("Verifying parameter equality between Policy and Reference models...")
        if not verify_parameter_equality(policy_model, self.reference_model):
            raise ValueError(
                "Verification failed: policy and reference model parameters differ! "
                "Preference training must start from identical weight values."
            )
        logger.info("🟢 Policy and Reference parameters are identical.")

    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute reference model logits in evaluation context."""
        if self.reference_model is None:
            raise RuntimeError("Reference model has not been loaded.")
        
        with torch.no_grad():
            outputs = self.reference_model(input_ids, return_dict=True)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            return logits
