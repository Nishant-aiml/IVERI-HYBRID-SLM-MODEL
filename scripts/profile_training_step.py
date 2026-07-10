# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Execution timing profile utility for IVERI CORE training steps.

Profiles execution times of individual training step phases: dataloading,
forward pass, loss computation, backward pass, and optimizer steps.
"""

from __future__ import annotations

import time
import torch
from configs.base_config import get_base_config
from model.iveri_core import IVERIModel
from training.optimizer import get_optimizer
from training.mixed_precision import PrecisionHandler


def profile_training_step() -> None:
    """Run a timed step profile of the model training loop."""
    print("=" * 70)
    print("                IVERI CORE TRAINING STEP TIME PROFILE")
    print("=" * 70)

    config = get_base_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target profiling device: {device}")

    # Scale down config for CPU timing to keep execution fast
    if device.type == "cpu":
        config.model.hidden_dim = 32
        config.model.num_layers = 2
        config.model.num_heads = 2
        config.model.num_experts = 2
        config.model.num_active_experts = 1
        config.model.titans_memory_dim = 16

    # Instantiate model, optimizer, precision handler
    model = IVERIModel(config).to(device)
    optimizer = get_optimizer(model, config.training.learning_rate, config.training.weight_decay)
    precision_handler = PrecisionHandler(config.hardware.mixed_precision, device.type)

    # Inputs
    inputs = torch.randint(0, 256, (4, 128), device=device)
    targets = torch.randint(0, 256, (4, 128), device=device)

    # Warmup step
    outputs = model(inputs, return_dict=True)
    loss = outputs["logits"].sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Profile iterations
    num_steps = 5
    fwd_times = []
    loss_times = []
    bwd_times = []
    opt_times = []

    for _ in range(num_steps):
        # 1. Forward Pass
        t0 = time.perf_counter()
        with precision_handler.autocast_context():
            outputs = model(inputs, return_dict=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        fwd_times.append(t1 - t0)

        # 2. Loss Computation
        t0 = time.perf_counter()
        logits = outputs["logits"]
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = targets.view(-1)
        loss = torch.nn.functional.cross_entropy(flat_logits, flat_targets)
        aux_loss = outputs.get("aux_loss", torch.tensor(0.0, device=device))
        composite_loss = loss + 0.01 * aux_loss
        scaled_loss = precision_handler.scale_loss(composite_loss)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        loss_times.append(t1 - t0)

        # 3. Backward Pass
        t0 = time.perf_counter()
        scaled_loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        bwd_times.append(t1 - t0)

        # 4. Optimizer Step
        t0 = time.perf_counter()
        precision_handler.step_optimizer(optimizer, config.training.grad_clip)
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        opt_times.append(t1 - t0)

    # Compute averages
    avg_fwd = sum(fwd_times) / num_steps
    avg_loss = sum(loss_times) / num_steps
    avg_bwd = sum(bwd_times) / num_steps
    avg_opt = sum(opt_times) / num_steps
    avg_total = avg_fwd + avg_loss + avg_bwd + avg_opt

    # Percentages
    pct_fwd = (avg_fwd / avg_total) * 100
    pct_loss = (avg_loss / avg_total) * 100
    pct_bwd = (avg_bwd / avg_total) * 100
    pct_opt = (avg_opt / avg_total) * 100

    print(f"Total step execution time: {avg_total*1000:.2f} ms")
    print(f"  Forward Pass:    {avg_fwd*1000:.2f} ms ({pct_fwd:.1f}%)")
    print(f"  Loss Evaluation: {avg_loss*1000:.2f} ms ({pct_loss:.1f}%)")
    print(f"  Backward Pass:   {avg_bwd*1000:.2f} ms ({pct_bwd:.1f}%)")
    print(f"  Optimizer step:  {avg_opt*1000:.2f} ms ({pct_opt:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    profile_training_step()
