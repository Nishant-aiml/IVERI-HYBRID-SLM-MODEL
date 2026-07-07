# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""IVERI CORE — Backbone Block Assembly (Phase 1.8).

Assembles Mamba2, Flash Attention, MoE, MoR, and Titans Neural Memory into a
unified, differentiable, and modular backbone block stack.
"""

from __future__ import annotations

import time
import typing
from typing import Any

import torch
import torch.nn as nn
import torch.utils.checkpoint

from configs.base_config import IVERIConfig
from core.interfaces import BaseModule
from core.registry import register
from model.attention import FlashAttentionWrapper
from model.mamba2 import Mamba2Block
from model.moe import MoEExperts, SparseMoERouter
from model.mor import RecursionDepthRouter, RecursionEngine
from model.norms import RMSNorm
from model.swiglu import SwiGLUFFN
from model.titans import TitansMemory
from utils.validation import validate_shape


class BackboneSubBlock(nn.Module):
    """Core layer block containing sequential sub-layers:
    Mamba2 (mamba_ratio times) -> Flash Attention -> MoE FFN.

    Applies Pre-LN (RMSNorm) and residual connections around each layer.
    """

    def __init__(self, config: IVERIConfig) -> None:
        """Initialize BackboneSubBlock.

        Args:
            config: General configuration object.
        """
        super().__init__()
        self.config = config
        self.mamba_ratio = config.model.mamba_ratio
        self.hidden_dim = config.model.hidden_dim
        self.num_experts = config.model.num_experts
        self.use_moe = config.model.use_moe

        # Mamba2 SSM blocks (mamba_ratio blocks)
        self.mamba_blocks = nn.ModuleList([Mamba2Block(config) for _ in range(self.mamba_ratio)])
        self.mamba_norms = nn.ModuleList(
            [RMSNorm(self.hidden_dim) for _ in range(self.mamba_ratio)]
        )

        # Flash Attention wrapper block
        self.attention = FlashAttentionWrapper(config)
        self.attn_norm = RMSNorm(self.hidden_dim)

        # MoE FFN expert container (or dense FFN when ablated)
        if self.use_moe:
            self.moe_router = SparseMoERouter(config)
            self.moe_experts = MoEExperts(config)
        else:
            self.dense_ffn = SwiGLUFFN(dim=self.hidden_dim, bias=False)
        self.moe_norm = RMSNorm(self.hidden_dim)

        # Container to accumulate routing load-balancing losses across recursion steps
        self.current_aux_losses: list[torch.Tensor] = []

        # Expert utilization tracking
        self.expert_counts = torch.zeros(self.num_experts, dtype=torch.long)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Execute sub-layers with residuals and Pre-LN.

        Args:
            x: Input token representations of shape (B, P, D).
            **kwargs: Extra arguments for compatibility.

        Returns:
            Output representations of shape (B, P, D).
        """
        # Run Mamba2 blocks
        use_ckpt = self.config.hardware.gradient_checkpointing and torch.is_grad_enabled()

        for mamba, norm in zip(self.mamba_blocks, self.mamba_norms, strict=False):
            if use_ckpt:
                # Helper function for checkpointing the combined norm and block pass
                def run_mamba(h: torch.Tensor) -> torch.Tensor:
                    return mamba(norm(h))
                x = x + torch.utils.checkpoint.checkpoint(run_mamba, x, use_reentrant=False)
            else:
                x = x + mamba(norm(x))

        # Run Attention wrapper block
        if use_ckpt:
            # Helper function for checkpointing attention
            def run_attn(h: torch.Tensor) -> torch.Tensor:
                return self.attention(self.attn_norm(h))
            x = x + torch.utils.checkpoint.checkpoint(run_attn, x, use_reentrant=False)
        else:
            x = x + self.attention(self.attn_norm(x))

        # Run MoE FFN block (or dense FFN ablation path)
        norm_x = self.moe_norm(x)
        entropy = kwargs.get("entropy")
        if self.use_moe:
            dispatch_weights, dispatch_indices, aux_loss = self.moe_router(
                norm_x, entropy=entropy if isinstance(entropy, torch.Tensor) else None
            )
            self.current_aux_losses.append(aux_loss)

            self.expert_counts += torch.bincount(
                dispatch_indices.detach().view(-1), minlength=self.num_experts
            ).cpu()

            if use_ckpt:
                def run_experts(h: torch.Tensor, w: torch.Tensor, idxs: torch.Tensor) -> torch.Tensor:
                    moe_out, _ = self.moe_experts(h, w, idxs)
                    return moe_out
                moe_out = torch.utils.checkpoint.checkpoint(
                    run_experts, norm_x, dispatch_weights, dispatch_indices, use_reentrant=False
                )
            else:
                moe_out, _ = self.moe_experts(norm_x, dispatch_weights, dispatch_indices)
        else:
            if use_ckpt:
                moe_out = torch.utils.checkpoint.checkpoint(self.dense_ffn, norm_x, use_reentrant=False)
            else:
                moe_out = self.dense_ffn(norm_x)
        x = x + moe_out

        return x


@register("backbone_block")
class BackboneBlock(BaseModule):
    """A single layer of the IVERI Backbone.

    Wraps BackboneSubBlock in a RecursionEngine to execute Mixture of Recursions (MoR)
    based on patch entropy.
    """

    def __init__(self, config: IVERIConfig) -> None:
        """Initialize BackboneBlock.

        Args:
            config: General configuration object.
        """
        super().__init__()
        self.config = config
        self.hidden_dim = config.model.hidden_dim
        self.use_mor = config.model.use_mor

        # Router mapping entropy to depth (Option C)
        self.router = RecursionDepthRouter(config)

        # Sub-layer block wrapped in RecursionEngine
        self.sub_block = BackboneSubBlock(config)
        self.recursion_engine = RecursionEngine(self.sub_block, config)

        # Final layer norm
        self.norm = RMSNorm(self.hidden_dim)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Execute block with MoR recursion loops.

        Args:
            x: Input token representations of shape (B, P, D).
            **kwargs: Extra arguments including:
                entropy: Patch entropy tensor of shape (B, P, 1) or (B, P).

        Returns:
            Output representations of shape (B, P, D).
        """
        entropy = kwargs.get("entropy")
        if entropy is None:
            raise ValueError("entropy tensor is required for BackboneBlock forward pass.")

        # Clear auxiliary losses and reset recursion stats
        self.sub_block.current_aux_losses.clear()
        if self.use_mor:
            self.recursion_engine.reset_statistics()

        if self.use_mor:
            # Compute depths using Option C entropy mapping
            _, dispatch_indices = self.router.route(x, entropy=entropy)
            depths = dispatch_indices + 1
            x = self.recursion_engine(x, depths=depths, **kwargs)
        else:
            x = self.sub_block(x, **kwargs)

        # Final block norm
        x = self.norm(x)

        return x


@register("backbone")
class Backbone(BaseModule):
    """The full assembled IVERI CORE Backbone block stack.

    Orchestrates the sequence: Titans Memory -> BackboneBlock x L.
    Accumulates MoE load-balancing auxiliary losses and compiles telemetry statistics.
    """

    def __init__(self, config: IVERIConfig) -> None:
        """Initialize Backbone.

        Args:
            config: General configuration object.
        """
        super().__init__()
        self.config = config
        self.hidden_dim = config.model.hidden_dim
        self.num_layers = config.model.num_layers
        self.use_titans = config.model.use_titans

        # Global Titans memory at backbone entry (optional when ablated)
        self.titans = TitansMemory(config) if self.use_titans else None

        # Backbone layers
        self.blocks = nn.ModuleList([BackboneBlock(config) for _ in range(self.num_layers)])

        # State containers
        self.current_aux_losses: list[torch.Tensor] = []
        self.telemetry: dict[str, Any] = {}

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Forward pass of the full Backbone.

        Args:
            x: Input representations of shape (B, P, D).
            **kwargs: Extra arguments including:
                entropy: Patch entropy tensor of shape (B, P, 1) or (B, P).

        Returns:
            Output representations of shape (B, P, D).
        """
        entropy = kwargs.get("entropy")
        if entropy is None:
            raise ValueError("entropy tensor is required for Backbone.")

        validate_shape(x, (-1, -1, self.hidden_dim), name="x")

        # Record start time for forward latency
        start_time = time.perf_counter()

        # Input residual baseline for residual norm tracking
        x_in = x.clone()

        # 1. Global Titans Memory (online forward + entropy-gated injection at backbone entry)
        t_start = time.perf_counter()
        if self.use_titans:
            assert self.titans is not None
            x = self.titans.forward_with_injection(x, entropy)
        titans_time = time.perf_counter() - t_start

        # 2. Block stack execution
        self.current_aux_losses.clear()
        block_times: list[float] = []

        # Reset sub-block expert count statistics for this forward pass
        for block in self.blocks:
            assert isinstance(block, BackboneBlock)
            block.sub_block.expert_counts.zero_()

        for block in self.blocks:
            assert isinstance(block, BackboneBlock)
            b_start = time.perf_counter()
            x = block(x, **kwargs)
            block_times.append(time.perf_counter() - b_start)

            # Gather auxiliary losses from MoE router
            self.current_aux_losses.extend(block.sub_block.current_aux_losses)

        # Calculate forward latency
        forward_latency = time.perf_counter() - start_time

        # Update telemetry data
        self._update_telemetry_data(
            x_in=x_in,
            x_out=x,
            entropy=entropy,
            forward_latency=forward_latency,
            titans_time=titans_time,
            block_times=block_times,
        )

        return x

    def _update_telemetry_data(
        self,
        x_in: torch.Tensor,
        x_out: torch.Tensor,
        entropy: torch.Tensor,
        forward_latency: float,
        titans_time: float,
        block_times: list[float],
    ) -> None:
        """Compile and format telemetry stats for the forward pass.

        Args:
            x_in: Input tensor to the backbone (pre-Titans).
            x_out: Final output tensor from the backbone blocks.
            entropy: Input entropy tensor of shape (B, P, 1).
            forward_latency: Total runtime of forward pass in seconds.
            titans_time: Runtime of Titans memory block in seconds.
            block_times: Runtimes of individual blocks in seconds.
        """
        B, P, D = x_out.shape

        # 1. Parameter counts
        total_params = sum(p.numel() for p in self.parameters())
        titans_params = sum(p.numel() for p in self.titans.parameters()) if self.titans else 0
        block_params = [sum(p.numel() for p in block.parameters()) for block in self.blocks]

        # 2. FLOPs estimation (analytical MAC-based)
        # Mamba2: in_proj (2*B*P*D*(6D+32)), conv1d (8*B*P*(4D+32)), SSD scan (64*B*P*D), out_proj (4*B*P*D^2)
        mamba_ratio = self.config.model.mamba_ratio
        mamba_flops = mamba_ratio * (
            2 * B * P * D * (6 * D + 32)
            + 8 * B * P * (4 * D + 32)
            + 64 * B * P * D
            + 4 * B * P * D * D
        )

        # Attention: qkv_proj (6*B*P*D^2), attention computation (4*B*P^2*D), out_proj (2*B*P*D^2)
        attn_flops = 8 * B * P * D * D + 4 * B * P * P * D

        # MoE: router gating (2*B*P*D*E), SwiGLU active (32*B*P*K*D^2)
        E = self.config.model.num_experts
        K = self.config.model.num_active_experts
        moe_flops = 2 * B * P * D * E + 32 * B * P * K * D * D

        # Titans: projections (6*B*P*D^2), MLP updates (8*B*P*D*titans_memory_dim)
        mem_dim = self.config.model.titans_memory_dim
        titans_flops = 6 * B * P * D * D + 8 * B * P * D * mem_dim

        # Per-layer FLOPs (block sub-layers)
        layer_flops = mamba_flops + attn_flops + moe_flops

        # 3. Activation memory estimation (analytical bytes)
        # RMSNorm + Mamba2 + Attention + MoE FFN stored activations
        dtype_bytes = 4 if x_out.dtype == torch.float32 else 2
        activation_mem_per_block = B * P * D * (mamba_ratio * 4 + 6 + 2 * K) * dtype_bytes

        # 4. Hidden representation L2 norms
        hidden_norm = x_out.detach().pow(2).sum(-1).sqrt().mean().item()
        residual_norm = (x_out - x_in).detach().pow(2).sum(-1).sqrt().mean().item()

        # 5. Entropy stats
        entropy_cpu = entropy.detach().cpu()
        entropy_stats = {
            "mean": entropy_cpu.mean().item(),
            "min": entropy_cpu.min().item(),
            "max": entropy_cpu.max().item(),
            "std": entropy_cpu.std().item() if entropy_cpu.numel() > 1 else 0.0,
        }

        # 6. Aggregated MoR stats
        depths_list = []
        for block in self.blocks:
            assert isinstance(block, BackboneBlock)
            if block.use_mor:
                stats = block.recursion_engine.get_statistics()
                depths_list.append(typing.cast(float, stats["average_depth"]))
            else:
                depths_list.append(1.0)
        avg_depth = sum(depths_list) / len(depths_list) if len(depths_list) > 0 else 0.0

        # 7. Expert utilization histogram
        expert_hist = torch.zeros(E, dtype=torch.long)
        for block in self.blocks:
            assert isinstance(block, BackboneBlock)
            if block.sub_block.use_moe:
                expert_hist += block.sub_block.expert_counts
        expert_utilization = expert_hist.tolist()

        # 8. Titans read/write tracking (measured from TitansMemory telemetry)
        titans_tel = getattr(self.titans, "telemetry", {}) or {} if self.titans else {}
        titans_reads = int(titans_tel.get("read_count", titans_tel.get("update_count", 0)))
        titans_writes = int(titans_tel.get("write_count", titans_tel.get("update_count", 0)))
        avg_mem_update = float(titans_tel.get("average_update_magnitude", 0.0))

        # 9. VRAM tracking
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024**2)
            avg_vram = torch.cuda.memory_allocated() / (1024**2)
        else:
            try:
                import os

                import psutil

                process = psutil.Process(os.getpid())
                avg_vram = process.memory_info().rss / (1024**2)
                peak_vram = avg_vram
            except ImportError:
                avg_vram = 0.0
                peak_vram = 0.0

        # Calculate throughput (patches processed per second)
        throughput = (B * P) / forward_latency if forward_latency > 0 else 0.0

        # 10. Gradient norms per module (computed after backward passes, defaults to 0.0)
        grad_norms = {
            "titans": (
                sum(
                    p.grad.detach().pow(2).sum().item()
                    for p in self.titans.parameters()
                    if p.grad is not None
                )
                ** 0.5
                if self.titans
                else 0.0
            ),
            "blocks": sum(
                p.grad.detach().pow(2).sum().item()
                for p in self.blocks.parameters()
                if p.grad is not None
            )
            ** 0.5,
        }

        self.telemetry = {
            "total_parameters": total_params,
            "parameters_per_module": {
                "titans": titans_params,
                "blocks": sum(block_params),
                "per_layer_parameter_count": block_params,
            },
            "flops_per_module": {
                "titans": titans_flops,
                "blocks": layer_flops * self.num_layers,
                "per_layer_flops": [layer_flops] * self.num_layers,
            },
            "runtime_per_module": {
                "titans": titans_time,
                "blocks": sum(block_times),
                "per_layer_runtime": block_times,
            },
            "peak_vram_mb": peak_vram,
            "average_vram_mb": avg_vram,
            "activation_memory_mb": (activation_mem_per_block * self.num_layers) / (1024**2),
            "per_layer_activation_memory_mb": [activation_mem_per_block / (1024**2)]
            * self.num_layers,
            "hidden_state_norm": hidden_norm,
            "residual_norm": residual_norm,
            "gradient_norm_per_module": grad_norms,
            "entropy_statistics": entropy_stats,
            "average_recursion_depth": avg_depth,
            "expert_utilization_histogram": expert_utilization,
            "titans_read_count": titans_reads,
            "titans_write_count": titans_writes,
            "average_memory_update_magnitude": avg_mem_update,
            "average_throughput_tokens_per_sec": throughput,
            "forward_latency_seconds": forward_latency,
            "backward_latency_seconds": 0.0,  # Filled during training updates
        }
