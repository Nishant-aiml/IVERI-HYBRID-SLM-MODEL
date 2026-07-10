# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Memory profiling utility for the IVERI CORE architecture.

Profiles the memory footprint of model weights, activation maps, optimizer
states, and evaluates the VRAM reduction from gradient checkpointing.
"""

from __future__ import annotations

import gc
import torch
import torch.nn as nn
from configs.base_config import get_base_config
from model.iveri_core import IVERIModel


def get_memory_stats() -> dict[str, float]:
    """Get active GPU VRAM usage or fall back to system memory if on CPU."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return {
            "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
            "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
        }
    else:
        # Fall back to heuristic using CPU tracking if CUDA is not available
        import os
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / (1024**2)
            return {"allocated_mb": mem_mb, "max_allocated_mb": mem_mb}
        except ImportError:
            return {"allocated_mb": 0.0, "max_allocated_mb": 0.0}


def profile_model_memory() -> None:
    """Run model memory profiling under different configurations."""
    print("=" * 70)
    print("                    IVERI CORE MEMORY PROFILE")
    print("=" * 70)

    config = get_base_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target profiling device: {device}")

    # Scale down config for CPU profiling if CUDA is not available
    if device.type == "cpu":
        config.model.hidden_dim = 32
        config.model.num_layers = 2
        config.model.num_heads = 2
        config.model.num_experts = 2
        config.model.num_active_experts = 1
        config.model.titans_memory_dim = 16

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 1. Base footprint (initial VRAM state)
    init_mem = get_memory_stats()["allocated_mb"]
    print(f"Initial baseline memory footprint: {init_mem:.2f} MB")

    # 2. Model initialization footprint
    model = IVERIModel(config).to(device)
    gc.collect()
    model_mem = get_memory_stats()["allocated_mb"] - init_mem
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters memory: {model_mem:.2f} MB (Params count: {num_params:,})")

    # 3. Forward pass activations profile (without gradient checkpointing)
    config.hardware.gradient_checkpointing = False
    inputs = torch.randint(0, 256, (4, 128), device=device)
    
    gc.collect()
    fwd_start_mem = get_memory_stats()["allocated_mb"]
    outputs = model(inputs, return_dict=True)
    fwd_end_mem = get_memory_stats()["allocated_mb"]
    act_mem_no_ckpt = fwd_end_mem - fwd_start_mem
    print(f"Activation memory (Gradient Checkpointing OFF): {act_mem_no_ckpt:.2f} MB")

    # Backward pass VRAM peak
    loss = outputs["logits"].sum()
    loss.backward()
    post_grad_mem = get_memory_stats()["allocated_mb"]
    print(f"Peak VRAM during backward pass: {get_memory_stats()['max_allocated_mb']:.2f} MB")
    
    # 4. Activation memory profile with gradient checkpointing
    # Clear gradients and reset model
    model.zero_grad(set_to_none=True)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    config.hardware.gradient_checkpointing = True
    # Re-instantiate model with checkpointing enabled
    model_ckpt = IVERIModel(config).to(device)
    
    gc.collect()
    fwd_ckpt_start_mem = get_memory_stats()["allocated_mb"]
    outputs_ckpt = model_ckpt(inputs, return_dict=True)
    fwd_ckpt_end_mem = get_memory_stats()["allocated_mb"]
    act_mem_ckpt = fwd_ckpt_end_mem - fwd_ckpt_start_mem
    print(f"Activation memory (Gradient Checkpointing ON): {act_mem_ckpt:.2f} MB")

    # VRAM reduction comparison
    reduction = act_mem_no_ckpt - act_mem_ckpt
    pct = (reduction / max(1e-5, act_mem_no_ckpt)) * 100
    print(f"Activation VRAM reduction from checkpointing: {reduction:.2f} MB ({pct:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    profile_model_memory()
