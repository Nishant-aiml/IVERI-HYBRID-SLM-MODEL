# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Vanilla byte-level Transformer baseline matching parameter count of IVERI CORE nano.

Used as a control baseline to compare training convergence metrics and stack correctness.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from configs.base_config import IVERIConfig
from core.interfaces import BaseModule
from core.registry import register
from utils.validation import validate_shape


@register("baseline_transformer")
class BaselineTransformer(BaseModule):
    """A vanilla autoregressive Byte-level Transformer baseline.

    Uses standard embedding, sinusoidal or learned position encodings,
    and standard PyTorch Transformer blocks. Vocab size is fixed to 256.
    """

    def __init__(self, config: IVERIConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_dim = config.model.hidden_dim
        self.architecture_version = "vanilla-transformer-baseline"
        self.checkpoint_version = "1.0"

        # Byte embeddings: vocab is 256 (all raw byte values)
        self.token_embeddings = nn.Embedding(256, self.hidden_dim)

        # Simple learned absolute position embeddings
        self.pos_embeddings = nn.Embedding(config.training.seq_len, self.hidden_dim)

        # Encoder blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=config.model.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=config.model.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.model.num_layers,
        )

        # Output lm head
        self.lm_head = nn.Linear(self.hidden_dim, 256)

        # Dropout
        self.dropout = nn.Dropout(config.model.dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        if self.lm_head.bias is not None:
            nn.init.zeros_(self.lm_head.bias)

    def _generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(
        self,
        raw_bytes: torch.Tensor,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor | dict[str, Any]:
        """Execute the baseline model forward pass.

        Args:
            raw_bytes: Input raw byte indices of shape (B, S).
            return_dict: Whether to return dict or raw logits.
        """
        validate_shape(raw_bytes, (-1, -1), name="raw_bytes")
        batch_size, seq_len = raw_bytes.shape
        device = raw_bytes.device

        start_time = time.perf_counter()

        if seq_len == 0:
            logits = torch.zeros(batch_size, 0, 256, device=device, dtype=torch.float32)
            if return_dict:
                return {
                    "logits": logits,
                    "aux_loss": torch.tensor(0.0, device=device, dtype=torch.float32),
                    "telemetry": {},
                }
            return logits

        # Embed tokens and add positions
        x = self.token_embeddings(raw_bytes)  # (B, S, D)
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0)  # (1, S)
        x = x + self.pos_embeddings(positions)
        x = self.dropout(x)

        # Generate causal mask
        mask = self._generate_square_subsequent_mask(seq_len, device)

        # Forward through transformer
        h = self.transformer(x, mask=mask, is_causal=True)  # (B, S, D)

        # LM Head
        logits = self.lm_head(h)  # (B, S, 256)

        end_time = time.perf_counter()
        latency = end_time - start_time
        throughput = (batch_size * seq_len) / latency if latency > 0 else 0.0

        if return_dict:
            return {
                "logits": logits,
                "aux_loss": torch.tensor(0.0, device=device, dtype=torch.float32),
                "telemetry": {
                    "model_architecture_version": self.architecture_version,
                    "end_to_end_forward_latency_seconds": latency,
                    "end_to_end_throughput_tokens_per_sec": throughput,
                },
            }

        return logits

    def save_checkpoint(
        self,
        path: str | Path,
        step: int,
        metrics: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> None:
        """Save standard checkpoint dictionary."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config_dict": self.config.to_dict(),
            "random_seed": seed,
            "step": step,
            "optimizer_state_dict": {},
            "metrics": metrics or {},
            "architecture_version": self.architecture_version,
            "checkpoint_version": self.checkpoint_version,
        }
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, target_path)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Load checkpoint and verify architectural compatibility."""
        target_path = Path(path)
        if not target_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        checkpoint = torch.load(target_path, map_location="cpu")
        arch_ver = checkpoint.get("architecture_version")
        if arch_ver != self.architecture_version:
            raise ValueError(
                f"Architecture mismatch. Expected: {self.architecture_version}, Got: {arch_ver}"
            )

        self.load_state_dict(checkpoint["model_state_dict"])
        return {
            "step": checkpoint.get("step", 0),
            "random_seed": checkpoint.get("random_seed", 42),
            "metrics": checkpoint.get("metrics", {}),
            "checkpoint_version": checkpoint.get("checkpoint_version"),
        }
