# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Pure-Mamba2 byte-level baseline matching parameter count of IVERI CORE nano.

Used as a control baseline to compare training convergence and architecture
ablation: IVERI (BLT + Titans + Mamba2 + MoR + MoE) vs. a plain Mamba2 stack
with no attention, no MoE, no MoR, no BLT patching.

Architecture
------------
Embedding (256 bytes → hidden_dim)
  ↓
Mamba2Block × num_layers   (linear-time selective SSM, no attention)
  ↓
RMSNorm
  ↓
LM Head (hidden_dim → 259)    ← same 259-token vocab as IVERI

Parameter count target: comparable to IVERI nano (~36M at hidden_dim=256,
num_layers=6).  Adjust hidden_dim via config if needed.

Examples
--------
>>> from configs.base_config import get_base_config
>>> from baselines.tiny_mamba import TinyMamba
>>> cfg = get_base_config()
>>> m = TinyMamba(cfg)
>>> import torch
>>> out = m(torch.randint(0, 256, (2, 64)), return_dict=True)
>>> out["logits"].shape
torch.Size([2, 64, 259])
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from configs.base_config import IVERIConfig
from core.byte_vocab import BYTE_VOCAB_SIZE
from core.interfaces import BaseModule
from core.registry import register
from model.mamba2.block import Mamba2Block
from model.norms import RMSNorm
from utils.validation import validate_shape


@register("tiny_mamba")
class TinyMamba(BaseModule):
    """Byte-level pure-Mamba2 baseline for ablation against IVERI CORE.

    Deliberately removes every IVERI-specific architectural component
    (BLT dynamic patching, Titans persistent memory, MoR recursive depth
    routing, MoE sparse FFNs) so that comparisons isolate the value added
    by each component.

    Parameters
    ----------
    config:
        Master IVERI configuration.  Uses ``config.model.hidden_dim``,
        ``config.model.num_layers``, ``config.model.dropout``,
        ``config.model.num_heads`` (for SSM heads), and
        ``config.training.seq_len``.
    """

    architecture_version: str = "tiny-mamba-baseline-v1"
    checkpoint_version: str = "1.0"

    def __init__(self, config: IVERIConfig) -> None:
        super().__init__()
        self.config = config

        d_model: int = config.model.hidden_dim
        n_layers: int = config.model.num_layers
        dropout: float = getattr(config.model, "dropout", 0.0)
        # SSM heads: use same ratio as IVERI backbone (num_heads from config)
        n_heads: int = config.model.num_heads

        self.d_model = d_model
        self.vocab_size = BYTE_VOCAB_SIZE  # 259

        # ── Input embedding ────────────────────────────────────────────────
        self.token_embedding = nn.Embedding(self.vocab_size, d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # ── Mamba2 stack ───────────────────────────────────────────────────
        # Each Mamba2Block uses the full config for hidden_dim, num_heads, etc.
        self.blocks = nn.ModuleList([
            Mamba2Block(config)
            for _ in range(n_layers)
        ])

        # ── Final norm + LM head ───────────────────────────────────────────
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)

        self._init_weights()

    # ── Initialisation ─────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        """Scaled normal initialisation (same as IVERI backbone)."""
        std = 0.02 / math.sqrt(2 * self.config.model.num_layers)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        raw_bytes: torch.Tensor,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor | dict[str, Any]:
        """Run the Mamba2 baseline forward pass.

        Parameters
        ----------
        raw_bytes:
            Integer byte IDs of shape ``(B, S)``.  Values in ``[0, 258]``.
        return_dict:
            If ``True``, return a dict with keys ``logits``, ``aux_loss``,
            and ``telemetry`` (compatible with IVERI forward contract).

        Returns
        -------
        torch.Tensor | dict[str, Any]
            Logits of shape ``(B, S, vocab_size)`` or a result dict.
        """
        validate_shape(raw_bytes, (-1, -1), name="raw_bytes")
        B, S = raw_bytes.shape
        device = raw_bytes.device
        t0 = time.perf_counter()

        # Handle empty sequences
        if S == 0:
            logits = torch.zeros(B, 0, self.vocab_size, device=device, dtype=torch.float32)
            if return_dict:
                return {"logits": logits, "aux_loss": torch.tensor(0.0, device=device), "telemetry": {}}
            return logits

        # Clamp inputs to valid vocab range
        raw_bytes = raw_bytes.clamp(0, self.vocab_size - 1)

        # Embed → (B, S, D)
        h = self.token_embedding(raw_bytes)
        h = self.embed_dropout(h)

        # Pass through Mamba2 blocks with residual connections
        for block in self.blocks:
            h = h + block(h)

        # Final norm
        h = self.final_norm(h)

        # LM head → (B, S, vocab_size)
        logits = self.lm_head(h)

        latency = time.perf_counter() - t0
        throughput = (B * S) / latency if latency > 0 else 0.0

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

    # ── Checkpointing ──────────────────────────────────────────────────────

    def save_checkpoint(
        self,
        path: str | Path,
        step: int,
        metrics: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> None:
        """Save a standard IVERI-compatible checkpoint."""
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
        """Load checkpoint with architecture version validation."""
        target_path = Path(path)
        if not target_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(target_path, map_location="cpu", weights_only=True)
        arch_ver = checkpoint.get("architecture_version")
        if arch_ver != self.architecture_version:
            raise ValueError(
                f"Architecture mismatch. Expected: {self.architecture_version!r}, "
                f"Got: {arch_ver!r}"
            )

        self.load_state_dict(checkpoint["model_state_dict"])
        return {
            "step": checkpoint.get("step", 0),
            "random_seed": checkpoint.get("random_seed", 42),
            "metrics": checkpoint.get("metrics", {}),
            "checkpoint_version": checkpoint.get("checkpoint_version"),
        }

    # ── Utilities ─────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_summary(self) -> dict[str, int]:
        """Return a per-component parameter count summary."""
        return {
            "token_embedding": self.token_embedding.weight.numel(),
            "mamba2_blocks": sum(
                p.numel() for b in self.blocks for p in b.parameters()
            ),
            "final_norm": sum(p.numel() for p in self.final_norm.parameters()),
            "total": self.count_parameters(),
        }
