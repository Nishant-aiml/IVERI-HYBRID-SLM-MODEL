# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Flop Profiler for Stage 5 structural and comparative compute profiling."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from configs.base_config import IVERIConfig

logger = logging.getLogger(__name__)


class FlopProfiler:
    """Calculates analytical forward and backward FLOP counts for model configurations."""

    def __init__(self, config: IVERIConfig) -> None:
        self.config = config
        self.hidden_dim = config.model.hidden_dim
        self.seq_len = config.training.seq_len
        self.num_layers = config.model.num_layers
        self.num_heads = config.model.num_heads
        self.d_head = self.hidden_dim // self.num_heads if self.num_heads > 0 else 0

    def estimate_attention_flops(self) -> float:
        """Estimate FLOPs for single-layer self-attention forward pass per token.

        Calculates projections and Scaled Dot Product Attention.
        FLOPs = QKV projections (2 * 3 * D^2) + Attention score Matrix Mult (2 * S * D)
              + Softmax/Scale (negligible) + Attention value Matrix Mult (2 * S * D)
              + Output projection (2 * D^2)
        """
        # projections: QKV (3 projections of size D x D) -> 2 * 3 * D^2
        proj_flops = 6 * (self.hidden_dim ** 2)
        # attention maps: Q x K -> 2 * seq_len * d_head * num_heads -> 2 * seq_len * D
        attn_map_flops = 2 * self.seq_len * self.hidden_dim
        # attention values: weights x V -> 2 * seq_len * d_head * num_heads -> 2 * seq_len * D
        attn_val_flops = 2 * self.seq_len * self.hidden_dim
        # out projection: 2 * D^2
        out_proj_flops = 2 * (self.hidden_dim ** 2)

        return (proj_flops + attn_map_flops + attn_val_flops + out_proj_flops) * self.seq_len

    def estimate_mamba_flops(self) -> float:
        """Estimate FLOPs for single-layer Mamba2 block forward pass."""
        # Mamba2 SSD math is linear in seq_len.
        # projections (expansion ratio = 2): Q, K, V, X, B, C, dt -> ~ 2 * 2.5 * D^2
        # state space convolution and scan: linear scaling
        proj_flops = 5 * (self.hidden_dim ** 2)
        state_dim = 16  # standard SSD state dimension
        scan_flops = 2 * self.seq_len * self.hidden_dim * state_dim
        return (proj_flops + scan_flops) * self.seq_len

    def estimate_moe_flops(self) -> float:
        """Estimate FLOPs for single-layer Mixture of Experts forward pass."""
        # FFN FLOPs per expert: SwiGLU has 3 projections of size D x (FFN_dim)
        # FFN_dim is rounded standard (e.g. 4 * D * 2/3) -> 8/3 * D
        # projections: gate (2 * D * FFN_dim) + up (2 * D * FFN_dim) + down (2 * FFN_dim * D)
        ffn_dim = int(self.hidden_dim * 8 / 3)
        ffn_flops = 2 * self.hidden_dim * ffn_dim * 3

        # Router projections: D -> num_experts
        router_flops = 2 * self.hidden_dim * self.config.model.num_experts

        # Scale by active experts routing ratio
        active_ratio = self.config.model.num_active_experts / self.config.model.num_experts
        moe_flops = (ffn_flops * active_ratio) + router_flops
        return moe_flops * self.seq_len

    def estimate_titans_flops(self) -> float:
        """Estimate FLOPs for single-layer Titans neural memory forward pass."""
        # Memory retrieval and memory update loops.
        # MLP width is titans_memory_dim
        memory_dim = self.config.model.titans_memory_dim
        if memory_dim <= 0:
            return 0.0
        # retrieval projections: 2 * D * memory_dim + 2 * memory_dim * D
        retrieval_flops = 4 * self.hidden_dim * memory_dim
        # memory update MLP forward: D -> memory_dim -> D
        update_flops = 4 * self.hidden_dim * memory_dim
        return (retrieval_flops + update_flops) * self.seq_len

    def estimate_blt_flops(self) -> float:
        """Estimate FLOPs for Byte Latent Transformer patcher forward pass."""
        # Byte embeddings (256 -> D) and patch reconstruction MLP.
        patch_flops = 2 * self.hidden_dim * self.hidden_dim
        return patch_flops * self.seq_len

    def calculate_forward_flops(self) -> float:
        """Calculate total FLOPs for model forward pass."""
        # Sum layers
        layer_flops = 0.0

        # Sub-layer allocations
        layer_flops += self.estimate_mamba_flops() * self.config.model.mamba_ratio
        layer_flops += self.estimate_attention_flops()
        layer_flops += self.estimate_moe_flops()

        total_layer_flops = layer_flops * self.num_layers
        # Global Titans memory at entry
        total_layer_flops += self.estimate_titans_flops()
        # BLT byte encoder
        total_layer_flops += self.estimate_blt_flops()

        return total_layer_flops

    def calculate_total_training_flops(self, num_tokens: int) -> float:
        """Calculate total training FLOPs (forward + backward) for a token budget.

        Standard convention: backward pass takes 2x the FLOPs of the forward pass.
        Total FLOPs = 3 * forward_flops * (num_tokens / seq_len)
        """
        forward_flops = self.calculate_forward_flops()
        num_sequences = num_tokens / self.seq_len
        return 3.0 * forward_flops * num_sequences

    def get_tflops(self, forward_flops: float, runtime_sec: float) -> float:
        """Calculate hardware execution speed in TFLOPs/sec."""
        if runtime_sec <= 0:
            return 0.0
        return (forward_flops / 1e12) / runtime_sec
