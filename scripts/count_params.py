# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Calculate exact parameter distribution and resource requirements for IVERI presets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

# Ensure root is on sys.path
sys.path.append(str(Path(__file__).parent.parent))

from configs.base_config import (
    get_nano_config,
    get_small_config,
    get_medium_config,
    get_large_config,
    get_xlarge_config,
    get_max_config,
    IVERIConfig,
)
from model.iveri_core import IVERIModel


def count_parameters_by_module(model: IVERIModel) -> dict[str, int]:
    """Break down model parameters by architectural components."""
    breakdown = {
        "blt_encoder_decoder": 0,
        "entropy_model": 0,
        "titans_neural_memory": 0,
        "mamba2_ssm": 0,
        "flash_attention": 0,
        "moe_experts": 0,
        "moe_router": 0,
        "norms_and_other": 0,
    }
    
    for name, p in model.named_parameters():
        numel = p.numel()
        if "entropy_model" in name:
            breakdown["entropy_model"] += numel
        elif "encoder" in name or "decoder" in name:
            breakdown["blt_encoder_decoder"] += numel
        elif "titans" in name:
            breakdown["titans_neural_memory"] += numel
        elif "mamba" in name:
            breakdown["mamba2_ssm"] += numel
        elif "attention" in name or "attn" in name:
            breakdown["flash_attention"] += numel
        elif "moe_router" in name or "router" in name:
            breakdown["moe_router"] += numel
        elif "moe_experts" in name or "experts" in name:
            breakdown["moe_experts"] += numel
        else:
            breakdown["norms_and_other"] += numel
            
    return breakdown


def estimate_metrics(preset_name: str, config: IVERIConfig, total_params: int, module_params: dict[str, int]) -> dict[str, Any]:
    """Estimate training/inference VRAM, FLOPs, and cache sizes for a preset."""
    # Hyperparameters
    D = config.model.hidden_dim
    L = config.model.num_layers
    E_tot = config.model.num_experts
    E_act = config.model.num_active_experts
    S = config.training.seq_len
    B = config.training.batch_size
    
    # Active parameters (per token forward pass)
    # BLT + Titans + Mamba + Attention are fully active.
    # MoE experts are partially active: E_act / E_tot parameters are active.
    inactive_expert_params = module_params["moe_experts"] * (1 - E_act / E_tot)
    active_params = total_params - inactive_expert_params
    
    # 1. Compute FLOPs per token
    # Forward pass: ~2 FLOPs per active parameter (multiply-accumulate)
    # Backward pass: ~4 FLOPs per active parameter (gradients wrt inputs and weights)
    # Total training FLOPs per token = ~6 * active_parameters
    forward_flops_per_token = 2 * active_params
    backward_flops_per_token = 4 * active_params
    training_flops_per_token = 6 * active_params
    
    # 2. KV Cache and State Cache sizes (in bytes, assuming FP16/BF16 - 2 bytes/element)
    # Attention KV Cache: 2 * L * B * S * D * 2 bytes (2 for K and V, 2 for FP16)
    attn_kv_cache_bytes = 2 * L * B * S * D * 2
    
    # Mamba state cache: state_dim = 128, conv_dim = 4
    # Mamba2 SSM blocks: L * mamba_ratio.
    # Each Mamba2 block has state of size: B * D_inner * d_state (FP16)
    # conv state size: B * D_inner * d_conv (FP16)
    mamba_ratio = config.model.mamba_ratio
    d_state = 128
    d_conv = 4
    mamba_d_inner = D * 2  # standard Mamba2 expansion
    mamba_cache_bytes = L * mamba_ratio * B * mamba_d_inner * (d_state + d_conv) * 2
    
    # Titans Neural Memory state: MLP weights or hidden states
    titans_memory_dim = config.model.titans_memory_dim
    titans_cache_bytes = L * B * titans_memory_dim * D * 2
    
    total_cache_bytes = attn_kv_cache_bytes + mamba_cache_bytes + titans_cache_bytes
    
    # 3. VRAM Estimations (in MB)
    # Model weights: FP16 (2 bytes per parameter)
    weights_vram_mb = (total_params * 2) / (1024**2)
    
    # Optimizer memory: AdamW in FP32 requires 8 bytes per parameter (mom + var),
    # plus FP32 master weights (4 bytes) and FP32 gradients (4 bytes) = 16 bytes per parameter.
    optimizer_vram_mb = (total_params * 16) / (1024**2)
    
    # Activation Memory (rough heuristic based on activation footprint with gradient checkpointing)
    # With gradient checkpointing: ~ L * B * S * D * 12 bytes
    # Without gradient checkpointing: ~ L * B * S * D * 34 bytes
    activation_gc_mb = (L * B * S * D * 12) / (1024**2)
    activation_nogc_mb = (L * B * S * D * 34) / (1024**2)
    
    # KV Cache in MB
    cache_vram_mb = total_cache_bytes / (1024**2)
    
    # Training VRAM: Weights + Gradients + Optimizer States + Activations (GC)
    # Weights (FP16) + Gradients (FP16, 2 bytes) + Optimizer (16 bytes) + Activations (GC)
    training_vram_est_mb = weights_vram_mb + (total_params * 2 / 1024**2) + optimizer_vram_mb + activation_gc_mb
    
    # Inference VRAM: Weights + KV Cache
    inference_vram_est_mb = weights_vram_mb + cache_vram_mb
    
    return {
        "model_parameters": {
            "total": total_params,
            "active_per_token": int(active_params),
            "breakdown": module_params,
        },
        "flops_per_token": {
            "forward": forward_flops_per_token,
            "backward": backward_flops_per_token,
            "total_training": training_flops_per_token,
        },
        "memory_caches_bytes": {
            "attention_kv_cache": attn_kv_cache_bytes,
            "mamba_state_cache": mamba_cache_bytes,
            "titans_neural_memory_cache": titans_cache_bytes,
            "total_inference_cache": total_cache_bytes,
        },
        "vram_estimate_mb": {
            "model_weights_fp16": round(weights_vram_mb, 2),
            "optimizer_states_adamw": round(optimizer_vram_mb, 2),
            "activations_with_gc": round(activation_gc_mb, 2),
            "activations_no_gc": round(activation_nogc_mb, 2),
            "inference_kv_cache": round(cache_vram_mb, 2),
            "estimated_training_footprint": round(training_vram_est_mb, 2),
            "estimated_inference_footprint": round(inference_vram_est_mb, 2),
        },
    }


def main() -> None:
    presets = {
        "nano": get_nano_config(),
        "small": get_small_config(),
        "medium": get_medium_config(),
        "large": get_large_config(),
        "xlarge": get_xlarge_config(),
        "max": get_max_config(),
    }
    
    breakdown_data = {}
    
    print("IVERI Configuration Presets Scaling Calculator")
    print("==============================================")
    
    for name, config in presets.items():
        print(f"Instantiating model for preset '{name}'...")
        # Force CPU and disable mixed precision for counting
        config.hardware.device = "cpu"
        config.hardware.mixed_precision = "fp32"
        
        model = IVERIModel(config)
        total_params = sum(p.numel() for p in model.parameters())
        module_params = count_parameters_by_module(model)
        
        preset_metrics = estimate_metrics(name, config, total_params, module_params)
        breakdown_data[name] = preset_metrics
        
        print(f"Preset '{name}':")
        print(f"  Total Params: {total_params:,}")
        print(f"  Active Params: {preset_metrics['model_parameters']['active_per_token']:,}")
        print(f"  Training VRAM: {preset_metrics['vram_estimate_mb']['estimated_training_footprint']:.2f} MB")
        print(f"  Inference VRAM: {preset_metrics['vram_estimate_mb']['estimated_inference_footprint']:.2f} MB")
        print("-" * 46)
        
    out_path = Path("configs/parameter_breakdown.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(breakdown_data, indent=2), encoding="utf-8")
    print(f"Breakdown report written to: {out_path}")


if __name__ == "__main__":
    main()
