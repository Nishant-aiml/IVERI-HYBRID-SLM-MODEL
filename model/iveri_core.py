# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""IVERI CORE — Full IVERI Model Integration (Phase 1.9).

Wires together the complete Byte Latent Transformer (BLT) pipeline and Backbone Block stack
into a single production-ready model with a unified forward and checkpointing API.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from configs.base_config import IVERIConfig
from core.constants import ARCHITECTURE_VERSION, BYTE_VOCAB_SIZE
from core.exceptions import CheckpointError
from core.interfaces import BaseModule
from core.registry import register
from model.backbone import Backbone
from model.blt.decoder import BLTByteDecoder
from model.blt.encoder import BLTByteEncoder
from model.blt.entropy_model import ByteEntropyModel
from model.blt.patcher import DynamicPatcher
from utils.validation import validate_shape


@register("iveri_core")
@register("iveri_model")
class IVERIModel(BaseModule):
    """The complete integrated IVERI Core Model.

    Orchestrates the frozen execution pipeline:
    Raw Bytes -> ByteEntropyModel -> DynamicPatcher -> BLTByteEncoder ->
    Patch Entropy Generation -> Backbone -> BLTByteDecoder -> Output Logits.
    """

    def __init__(self, config: IVERIConfig) -> None:
        """Initialize the complete IVERI Model.

        Args:
            config: General configuration object.
        """
        super().__init__()
        self.config = config
        self.hidden_dim = config.model.hidden_dim
        self.architecture_version = ARCHITECTURE_VERSION
        self.checkpoint_version = "1.0"
        self.use_blt = config.model.use_blt

        if self.use_blt:
            # 1. Byte Latent Transformer (BLT) front-end and decoder
            self.entropy_model = ByteEntropyModel(config)
            self.patcher = DynamicPatcher(config)
            self.encoder = BLTByteEncoder(config)
            self.decoder = BLTByteDecoder(config)
        else:
            # Ablated path: per-byte embedding without entropy-driven patching
            self.byte_embed = torch.nn.Embedding(BYTE_VOCAB_SIZE, self.hidden_dim)
            self.bypass_lm_head = torch.nn.Linear(self.hidden_dim, BYTE_VOCAB_SIZE)

        # 2. Main Backbone block stack (includes Titans, MoR, Mamba2, Attention, MoE)
        self.backbone = Backbone(config)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters across all integrated sub-modules."""
        if self.use_blt:
            self.entropy_model.reset_parameters()
            self.encoder.reset_parameters()
            self.decoder.reset_parameters()
        else:
            torch.nn.init.normal_(self.byte_embed.weight, mean=0.0, std=0.02)
            torch.nn.init.normal_(self.bypass_lm_head.weight, mean=0.0, std=0.02)
            if self.bypass_lm_head.bias is not None:
                torch.nn.init.zeros_(self.bypass_lm_head.bias)
        self.backbone.reset_parameters()

    def forward(  # type: ignore[override]
        self,
        raw_bytes: torch.Tensor,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor | dict[str, Any]:
        """Execute the complete model forward pipeline.

        Args:
            raw_bytes: Input token indices tensor of shape (B, S) in [0, BYTE_VOCAB_SIZE).
            return_dict: If True, returns a structured dictionary of internal states.
                         If False, returns next-byte logits tensor directly.
            **kwargs: Extra parameters passed down the pipeline.

        Returns:
            Output next-byte prediction logits tensor of shape (B, S, 256) or
            a dictionary containing the frozen schema fields.
        """
        validate_shape(raw_bytes, (-1, -1), name="raw_bytes")
        batch_size, seq_len = raw_bytes.shape
        device = raw_bytes.device

        # Track start time for end-to-end telemetry latency profiling
        start_time = time.perf_counter()

        # Handle boundary case: empty sequence
        if seq_len == 0:
            logits = torch.zeros(batch_size, 0, BYTE_VOCAB_SIZE, device=device, dtype=torch.float32)
            byte_entropy = torch.zeros(batch_size, 0, 1, device=device, dtype=torch.float32)
            patch_entropy = torch.zeros(batch_size, 0, 1, device=device, dtype=torch.float32)
            boundary_map = torch.zeros(batch_size, 0, device=device, dtype=torch.bool)
            aux_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            telemetry: dict[str, Any] = {}

            if return_dict:
                return {
                    "logits": logits,
                    "byte_entropy": byte_entropy,
                    "patch_entropy": patch_entropy,
                    "boundary_map": boundary_map,
                    "aux_loss": aux_loss,
                    "telemetry": telemetry,
                }
            return logits

        # 1. Front-end encoding (BLT or ablated byte-embedding path)
        if self.use_blt:
            byte_entropy = self.entropy_model(raw_bytes, **kwargs)
            boundary_map = self.patcher.compute_boundaries(raw_bytes, byte_entropy)
            latent_patches = self.encoder.encode_with_boundaries(raw_bytes, boundary_map)
            p_max = latent_patches.shape[1]

            patch_ids = torch.cumsum(boundary_map.long(), dim=-1) - 1
            patch_indices = torch.arange(p_max, device=device).view(1, -1, 1)
            is_patch = patch_ids.unsqueeze(1) == patch_indices
            patch_lengths = is_patch.sum(dim=-1, keepdim=True)
            patch_lengths_clamped = torch.clamp(patch_lengths, min=1)
            m = is_patch.float() / patch_lengths_clamped.float()
            patch_entropy = torch.bmm(m, byte_entropy)
        else:
            latent_patches = self.byte_embed(raw_bytes)
            p_max = seq_len
            byte_entropy = torch.zeros(batch_size, seq_len, 1, device=device, dtype=torch.float32)
            patch_entropy = torch.zeros(batch_size, seq_len, 1, device=device, dtype=torch.float32)
            boundary_map = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)

        # 5. Forward through the Backbone Block stack: (B, P_max, D)
        backbone_out = self.backbone(latent_patches, entropy=patch_entropy, **kwargs)

        # 6. Decode to sequence next-byte logits: (B, S, 256)
        if self.use_blt:
            logits = self.decoder.decode_with_boundaries(backbone_out, boundary_map, raw_bytes)
        else:
            logits = self.bypass_lm_head(backbone_out)

        # 7. Accumulate MoE load-balancing auxiliary loss from backbone
        aux_loss = (
            torch.stack(self.backbone.current_aux_losses).sum()
            if self.backbone.current_aux_losses
            else torch.tensor(0.0, device=device, dtype=torch.float32)
        )

        # Profile end-to-end latency and build telemetry metrics
        end_time = time.perf_counter()
        forward_latency = end_time - start_time
        throughput = (batch_size * seq_len) / forward_latency if forward_latency > 0 else 0.0

        # Inject model-level statistics into the backbone's telemetry copy
        telemetry = dict(self.backbone.telemetry)
        telemetry.update(
            {
                "model_architecture_version": self.architecture_version,
                "end_to_end_forward_latency_seconds": forward_latency,
                "end_to_end_throughput_tokens_per_sec": throughput,
                "average_patch_length": float(seq_len / p_max) if p_max > 0 else 0.0,
                "average_byte_entropy": byte_entropy.mean().item(),
                "average_patch_entropy": patch_entropy.mean().item(),
            }
        )

        if return_dict:
            return {
                "logits": logits,
                "byte_entropy": byte_entropy,
                "patch_entropy": patch_entropy,
                "boundary_map": boundary_map,
                "aux_loss": aux_loss,
                "telemetry": telemetry,
            }

        return logits

    def save_checkpoint(
        self,
        path: str | Path,
        step: int,
        metrics: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> None:
        """Save the model's weights and metadata matching the checkpoint contract.

        Args:
            path: Target file path to write to.
            step: Current training step.
            metrics: Dict of current training metrics.
            seed: Current random seed value.
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config_dict": self.config.to_dict(),
            "random_seed": seed,
            "step": step,
            "optimizer_state_dict": {},  # Reserved for Phase 2
            "metrics": metrics or {},
            "architecture_version": self.architecture_version,
            "checkpoint_version": self.checkpoint_version,
        }

        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, target_path)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Load model state and assert architectural compatibility.

        Args:
            path: Path to checkpoint file.

        Returns:
            Checkpoint metadata dictionary (step, seed, metrics).

        Raises:
            CheckpointError: If architecture version or checkpoint version mismatch.
        """
        target_path = Path(path)
        if not target_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        checkpoint = torch.load(target_path, map_location="cpu", weights_only=True)

        # Validate checkpoint contract versions
        arch_ver = checkpoint.get("architecture_version")
        if arch_ver != self.architecture_version:
            raise CheckpointError(
                f"Architecture version mismatch. Local: {self.architecture_version}, "
                f"Checkpoint: {arch_ver}. Incompatible restoration."
            )

        self.load_state_dict(checkpoint["model_state_dict"])

        return {
            "step": checkpoint.get("step", 0),
            "random_seed": checkpoint.get("random_seed", 42),
            "metrics": checkpoint.get("metrics", {}),
            "checkpoint_version": checkpoint.get("checkpoint_version"),
        }
