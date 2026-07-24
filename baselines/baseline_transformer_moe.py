# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Vanilla byte-level Transformer + MoE baseline matching parameter/FLOP counts of IVERI CORE.

Used as a control baseline to compare training convergence and sparse routing behavior
against standard Transformer and full IVERI hybrid models.
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
from model.moe.experts import MoEExperts
from model.moe.router import SparseMoERouter
from model.norms import RMSNorm
from utils.validation import validate_shape


class TransformerMoEBlock(nn.Module):
    """Transformer block with Causal Multi-Head Attention and Sparse MoE FFN."""

    def __init__(self, config: IVERIConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_dim = config.model.hidden_dim
        self.num_heads = config.model.num_heads

        # Norms
        self.attn_norm = RMSNorm(self.hidden_dim)
        self.ffn_norm = RMSNorm(self.hidden_dim)

        # Multi-Head Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=config.model.dropout,
            batch_first=True,
        )

        # Sparse MoE Router & Experts
        self.router = SparseMoERouter(config)
        self.experts = MoEExperts(config)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through attention and sparse MoE block.

        Returns:
            x: Output tensor of shape (B, S, D).
            aux_loss: Scalar MoE load balancing auxiliary loss.
        """
        # 1. Attention with residual
        norm_x = self.attn_norm(x)
        attn_out, _ = self.attn(
            norm_x, norm_x, norm_x, attn_mask=attn_mask, is_causal=True
        )
        x = x + attn_out

        # 2. Sparse MoE FFN with residual
        norm_ffn = self.ffn_norm(x)
        weights, indices, aux_loss = self.router(norm_ffn)
        moe_out, _ = self.experts(norm_ffn, weights=weights, indices=indices)
        x = x + moe_out

        return x, aux_loss


@register("baseline_transformer_moe")
class BaselineTransformerMoE(BaseModule):
    """A Byte-level Transformer baseline augmented with Sparse Mixture-of-Experts (MoE).

    Removes BLT dynamic patching, Titans memory, and MoR depth routing, but preserves
    sparse MoE feed-forward layers to isolate the specific impact of MoE vs. Titans/BLT/MoR.
    """

    architecture_version: str = "transformer-moe-baseline-v1"
    checkpoint_version: str = "1.0"

    def __init__(self, config: IVERIConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_dim = config.model.hidden_dim
        self.vocab_size = 256

        # Token & position embeddings
        self.token_embeddings = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.pos_embeddings = nn.Embedding(config.training.seq_len, self.hidden_dim)
        self.dropout = nn.Dropout(config.model.dropout)

        # Transformer + MoE block stack
        self.blocks = nn.ModuleList([
            TransformerMoEBlock(config) for _ in range(config.model.num_layers)
        ])

        # Final norm & LM head
        self.final_norm = RMSNorm(self.hidden_dim)
        self.lm_head = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def _generate_square_subsequent_mask(
        self, sz: int, device: torch.device
    ) -> torch.Tensor:
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
        """Forward pass for Transformer + MoE baseline."""
        validate_shape(raw_bytes, (-1, -1), name="raw_bytes")
        batch_size, seq_len = raw_bytes.shape
        device = raw_bytes.device

        t0 = time.perf_counter()

        if seq_len == 0:
            logits = torch.zeros(batch_size, 0, self.vocab_size, device=device, dtype=torch.float32)
            if return_dict:
                return {
                    "logits": logits,
                    "aux_loss": torch.tensor(0.0, device=device, dtype=torch.float32),
                    "telemetry": {},
                }
            return logits

        raw_bytes = raw_bytes.clamp(0, self.vocab_size - 1)
        x = self.token_embeddings(raw_bytes)
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0)
        x = x + self.pos_embeddings(positions)
        x = self.dropout(x)

        attn_mask = self._generate_square_subsequent_mask(seq_len, device)

        total_aux_loss = torch.tensor(0.0, device=device)
        for block in self.blocks:
            x, aux_loss = block(x, attn_mask=attn_mask)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.final_norm(x)
        logits = self.lm_head(x)

        latency = time.perf_counter() - t0
        throughput = (batch_size * seq_len) / latency if latency > 0 else 0.0

        if return_dict:
            return {
                "logits": logits,
                "aux_loss": total_aux_loss,
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
        """Save standard checkpoint."""
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
        """Load checkpoint."""
        target_path = Path(path)
        if not target_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(target_path, map_location="cpu", weights_only=True)
        arch_ver = checkpoint.get("architecture_version")
        if arch_ver != self.architecture_version:
            raise ValueError(
                f"Architecture mismatch. Expected: {self.architecture_version!r}, Got: {arch_ver!r}"
            )

        self.load_state_dict(checkpoint["model_state_dict"])
        return {
            "step": checkpoint.get("step", 0),
            "random_seed": checkpoint.get("random_seed", 42),
            "metrics": checkpoint.get("metrics", {}),
            "checkpoint_version": checkpoint.get("checkpoint_version"),
        }
