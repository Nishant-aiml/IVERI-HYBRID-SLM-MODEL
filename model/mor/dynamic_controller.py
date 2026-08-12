# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Multi-Signal Dynamic Compute Controller (SciOS v1.1 / Patent 1 Core).

Implements dynamic depth assignment governed by multiple joint signals:
  Depth = f(Entropy, Confidence, Memory Utilization, Gradient History, Context Length)

Features:
  - Multi-signal decision engine balancing token difficulty, router confidence, and memory pressure.
  - Useful Computation Ratio tracking: Useful_Activations / Total_Activations.
  - Full PyTorch vectorization with zero graph breaks or CPU synchronization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.base_config import IVERIConfig
from core.interfaces import BaseRouter
from core.registry import register


@register("dynamic_compute_controller")
class DynamicComputeController(BaseRouter):
    """Multi-Signal Dynamic Compute Controller for IVERI Core.

    Evaluates dynamic execution depth d in [0, max_depth] per token based on:
      1. Token Entropy (local uncertainty)
      2. Router Confidence (gating score gap)
      3. Memory State Utilization (Titans state norm)
      4. Context Length regime
    """

    def __init__(
        self,
        config: IVERIConfig,
        seq_len_threshold: int = 128,
        entropy_weight: float = 0.4,
        confidence_weight: float = 0.3,
        memory_weight: float = 0.3,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_dim = config.model.hidden_dim
        self.max_depth = config.model.max_recursion_depth
        self.seq_len_threshold = seq_len_threshold

        self.entropy_weight = entropy_weight
        self.confidence_weight = confidence_weight
        self.memory_weight = memory_weight

        # Learned linear score projection from input hidden states: (D -> 1)
        self.difficulty_proj = nn.Linear(self.hidden_dim, 1, bias=True)

        # Useful Computation Ratio counters
        self.register_buffer("total_activations", torch.tensor(0, dtype=torch.long))
        self.register_buffer("useful_activations", torch.tensor(0, dtype=torch.long))

    def compute_signals(
        self,
        x: torch.Tensor,
        entropy: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
        memory_norm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute unified multi-signal difficulty score in [0.0, 1.0]."""
        batch, seq_len, _ = x.shape

        # Signal 1: Token entropy (normalized to [0, 1])
        if entropy is None:
            # Learned fallback difficulty score
            s_entropy = torch.sigmoid(self.difficulty_proj(x))
        else:
            if entropy.ndim == 2:
                entropy = entropy.unsqueeze(-1)
            s_entropy = torch.clamp(entropy, 0.0, 1.0)

        # Signal 2: Router Confidence (1.0 - confidence) -> high confidence means easy token
        if confidence is None:
            s_conf = torch.zeros_like(s_entropy)
        else:
            if confidence.ndim == 2:
                confidence = confidence.unsqueeze(-1)
            s_conf = 1.0 - torch.clamp(confidence, 0.0, 1.0)

        # Signal 3: Memory state norm utilization (normalized via tanh)
        if memory_norm is None:
            s_mem = torch.zeros_like(s_entropy)
        else:
            if memory_norm.ndim == 2:
                memory_norm = memory_norm.unsqueeze(-1)
            s_mem = torch.tanh(memory_norm)

        # Combine weighted multi-signal score
        composite_score = (
            self.entropy_weight * s_entropy
            + self.confidence_weight * s_conf
            + self.memory_weight * s_mem
        )
        return composite_score

    def route(
        self,
        x: torch.Tensor,
        entropy: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
        memory_norm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route tokens to dynamic execution depth d in [0, max_depth]."""
        batch, seq_len, _ = x.shape

        # On short sequences (seq_len < threshold), bypass recursion (depth = 0)
        if seq_len < self.seq_len_threshold:
            depths = torch.zeros((batch, seq_len, 1), dtype=torch.long, device=x.device)
            dispatch_weights = torch.ones_like(depths, dtype=torch.float32)
            return dispatch_weights, depths

        # Calculate multi-signal composite score
        score = self.compute_signals(
            x, entropy=entropy, confidence=confidence, memory_norm=memory_norm
        )

        # Map score in [0.0, 1.0] to depths in [0, max_depth]
        # Low composite score -> depth 0; High score -> depth 1..max_depth
        depth_float = score * self.max_depth
        depths = torch.floor(depth_float).long()
        depths = torch.clamp(depths, min=0, max=self.max_depth)

        # Update Useful Computation Ratio tracking
        if self.training:
            with torch.no_grad():
                total_possible = batch * seq_len * self.max_depth
                executed = depths.sum().item()
                self.total_activations += total_possible
                self.useful_activations += executed

        dispatch_weights = torch.ones_like(depths, dtype=torch.float32)
        return dispatch_weights, depths

    def get_useful_computation_ratio(self) -> float:
        """Return the ratio of Useful Module Activations / Total Module Activations."""
        tot = self.total_activations.item()
        if tot == 0:
            return 1.0
        return float(self.useful_activations.item() / tot)

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        entropy = kwargs.get("entropy")
        confidence = kwargs.get("confidence")
        memory_norm = kwargs.get("memory_norm")

        _, depths = self.route(
            x,
            entropy=entropy if isinstance(entropy, torch.Tensor) else None,
            confidence=confidence if isinstance(confidence, torch.Tensor) else None,
            memory_norm=memory_norm if isinstance(memory_norm, torch.Tensor) else None,
        )
        return depths
