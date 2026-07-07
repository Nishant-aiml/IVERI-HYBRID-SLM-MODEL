# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Vanilla baseline models matching parameter counts and compute profiles of IVERI CORE."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from configs.base_config import IVERIConfig
from core.interfaces import BaseModule
from core.registry import register
from baselines.baseline_transformer import BaselineTransformer
from model.attention import FlashAttentionWrapper
from model.mamba2 import Mamba2Block
from model.norms import RMSNorm
from utils.validation import validate_shape

logger = logging.getLogger(__name__)


@register("baseline_mamba2")
class BaselineMamba2(BaseModule):
    """A pure Mamba2 autoregressive Byte-level model baseline.

    Uses standard embedding, RMSNorm layer normalization, and a stack of Mamba2 blocks.
    """

    def __init__(self, config: IVERIConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_dim = config.model.hidden_dim
        self.architecture_version = "vanilla-mamba2-baseline"
        self.checkpoint_version = "1.0"

        # Byte embeddings: vocab is 256
        self.token_embeddings = nn.Embedding(256, self.hidden_dim)

        # Mamba2 SSM blocks stack
        self.mamba_layers = nn.ModuleList([
            Mamba2Block(config) for _ in range(config.model.num_layers)
        ])
        self.norms = nn.ModuleList([
            RMSNorm(self.hidden_dim) for _ in range(config.model.num_layers)
        ])

        # Output head
        self.lm_head = nn.Linear(self.hidden_dim, 256)
        self.dropout = nn.Dropout(config.model.dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        if self.lm_head.bias is not None:
            nn.init.zeros_(self.lm_head.bias)

    def forward(
        self,
        raw_bytes: torch.Tensor,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor | dict[str, Any]:
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

        x = self.token_embeddings(raw_bytes)
        x = self.dropout(x)

        # Forward through layers
        for layer, norm in zip(self.mamba_layers, self.norms, strict=False):
            x = x + layer(norm(x))

        logits = self.lm_head(x)

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
        target_path = Path(path)
        if not target_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        checkpoint = torch.load(target_path, map_location="cpu")
        arch_ver = checkpoint.get("architecture_version")
        if arch_ver != self.architecture_version:
            raise ValueError(f"Architecture mismatch. Expected {self.architecture_version}, got {arch_ver}")
        self.load_state_dict(checkpoint["model_state_dict"])
        return {
            "step": checkpoint.get("step", 0),
            "random_seed": checkpoint.get("random_seed", 42),
            "metrics": checkpoint.get("metrics", {}),
            "checkpoint_version": checkpoint.get("checkpoint_version"),
        }


@register("baseline_hybrid")
class BaselineHybrid(BaseModule):
    """An alternating Mamba2 and FlashAttention hybrid baseline.

    Contains alternating layers: Mamba2 Block -> RMSNorm -> FlashAttention -> RMSNorm.
    """

    def __init__(self, config: IVERIConfig) -> None:
        super().__init__()
        self.config = config
        self.hidden_dim = config.model.hidden_dim
        self.architecture_version = "vanilla-hybrid-baseline"
        self.checkpoint_version = "1.0"

        # Embeddings
        self.token_embeddings = nn.Embedding(256, self.hidden_dim)
        self.pos_embeddings = nn.Embedding(config.training.seq_len, self.hidden_dim)

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Build alternating layers: num_layers of Mamba2 blocks and num_layers of Attention blocks
        for i in range(config.model.num_layers):
            # Mamba layer
            self.layers.append(Mamba2Block(config))
            self.norms.append(RMSNorm(self.hidden_dim))
            # Attention layer
            self.layers.append(FlashAttentionWrapper(config))
            self.norms.append(RMSNorm(self.hidden_dim))

        self.lm_head = nn.Linear(self.hidden_dim, 256)
        self.dropout = nn.Dropout(config.model.dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        if self.lm_head.bias is not None:
            nn.init.zeros_(self.lm_head.bias)

    def forward(
        self,
        raw_bytes: torch.Tensor,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor | dict[str, Any]:
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

        x = self.token_embeddings(raw_bytes)
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0)
        x = x + self.pos_embeddings(positions)
        x = self.dropout(x)

        for layer, norm in zip(self.layers, self.norms, strict=False):
            x = x + layer(norm(x))

        logits = self.lm_head(x)

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
        target_path = Path(path)
        if not target_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        checkpoint = torch.load(target_path, map_location="cpu")
        arch_ver = checkpoint.get("architecture_version")
        if arch_ver != self.architecture_version:
            raise ValueError(f"Architecture mismatch. Expected {self.architecture_version}, got {arch_ver}")
        self.load_state_dict(checkpoint["model_state_dict"])
        return {
            "step": checkpoint.get("step", 0),
            "random_seed": checkpoint.get("random_seed", 42),
            "metrics": checkpoint.get("metrics", {}),
            "checkpoint_version": checkpoint.get("checkpoint_version"),
        }


class BaselineManager:
    """Manager to create parameter/FLOP-matched baseline and ablated models."""

    def __init__(self, config: IVERIConfig) -> None:
        self.config = config

    def build_transformer_baseline(self) -> BaselineTransformer:
        """Create parameter/FLOP-matched Transformer baseline."""
        return BaselineTransformer(self.config)

    def build_mamba_baseline(self) -> BaselineMamba2:
        """Create parameter/FLOP-matched pure Mamba2 baseline."""
        return BaselineMamba2(self.config)

    def build_hybrid_baseline(self) -> BaselineHybrid:
        """Create parameter/FLOP-matched Mamba-Attention hybrid baseline."""
        return BaselineHybrid(self.config)

    def build_ablated_variant(self, component: str) -> nn.Module:
        """Create an ablated IVERI model by disabling a component flag.

        Supported components to disable:
        - ``titans``: skip Titans memory in backbone forward.
        - ``mor``: single sub-block pass (no recursion engine).
        - ``moe``: dense SwiGLU FFN instead of sparse MoE.
        - ``blt``: byte embedding bypass without entropy patching/BLT codec.
        - ``entropy_routing``: MoE gating ignores patch entropy bias.
        """
        import copy
        from model.iveri_core import IVERIModel

        ablated_config = copy.deepcopy(self.config)

        key = component.lower()
        if key == "titans":
            ablated_config.model.use_titans = False
        elif key == "mor":
            ablated_config.model.use_mor = False
        elif key == "moe":
            ablated_config.model.use_moe = False
        elif key == "blt":
            ablated_config.model.use_blt = False
        elif key in {"entropy_routing", "entropy"}:
            ablated_config.model.use_entropy_routing = False
        else:
            raise ValueError(f"Unknown ablation component: {component}")

        ablated_config.validate()
        return IVERIModel(ablated_config)
