# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Mathematical verification of gradient accumulation correctness.

Verifies that accumulating gradients over N steps of batch size B is
equivalent to a single step with batch size B * N.
"""

from __future__ import annotations

import copy
import torch
from configs.base_config import get_base_config
from model.iveri_core import IVERIModel


def test_gradient_accumulation_correctness() -> None:
    """Compare gradient accumulation with single large batch gradient."""
    config = get_base_config()
    
    # Scale down configuration to run quickly and reliably on CPU
    config.model.use_blt = False
    config.model.use_titans = False
    config.model.use_moe = False
    config.model.hidden_dim = 32
    config.model.num_layers = 2
    config.model.num_heads = 2
    config.model.num_experts = 2
    config.model.num_active_experts = 1
    config.model.titans_memory_dim = 16




    device = torch.device("cpu")
    
    # Instantiate two identical models
    model_large = IVERIModel(config).to(device)
    model_accum = copy.deepcopy(model_large)

    # Disable dropout to ensure deterministic gradient comparison
    model_large.eval()
    model_accum.eval()

    # Inputs

    inputs_1 = torch.randint(0, 256, (2, 64), device=device)
    targets_1 = torch.randint(0, 256, (2, 64), device=device)
    inputs_2 = torch.randint(0, 256, (2, 64), device=device)
    targets_2 = torch.randint(0, 256, (2, 64), device=device)

    # Large batch combines inputs_1 and inputs_2
    inputs_large = torch.cat([inputs_1, inputs_2], dim=0)
    targets_large = torch.cat([targets_1, targets_2], dim=0)

    # 1. Run large batch pass
    model_large.zero_grad(set_to_none=True)
    outputs_large = model_large(inputs_large, return_dict=True)
    logits_large = outputs_large["logits"]
    loss_large = torch.nn.functional.cross_entropy(
        logits_large.view(-1, logits_large.size(-1)), targets_large.view(-1)
    )
    loss_large.backward()

    # 2. Run accumulated passes (N=2 steps of batch size 2)
    model_accum.zero_grad(set_to_none=True)
    
    # Step A
    outputs_1 = model_accum(inputs_1, return_dict=True)
    logits_1 = outputs_1["logits"]
    loss_1 = torch.nn.functional.cross_entropy(
        logits_1.view(-1, logits_1.size(-1)), targets_1.view(-1)
    )
    # Scale by N=2 for correct gradient accumulation averaging
    (loss_1 / 2.0).backward()

    # Step B
    outputs_2 = model_accum(inputs_2, return_dict=True)
    logits_2 = outputs_2["logits"]
    loss_2 = torch.nn.functional.cross_entropy(
        logits_2.view(-1, logits_2.size(-1)), targets_2.view(-1)
    )
    # Scale by N=2
    (loss_2 / 2.0).backward()

    # 3. Assert gradients are mathematically equivalent
    max_diff = 0.0
    for (name_l, param_l), (name_a, param_a) in zip(
        model_large.named_parameters(), model_accum.named_parameters(), strict=True
    ):
        if param_l.requires_grad:
            if param_l.grad is None and param_a.grad is None:
                continue
            assert param_l.grad is not None and param_a.grad is not None
            diff = torch.abs(param_l.grad - param_a.grad).max().item()
            if diff > 1e-4:
                print(f"Parameter '{name_l}' diff: {diff:.2e}")
            max_diff = max(max_diff, diff)


    # Threshold for Float32 equivalence
    assert max_diff < 1e-4, f"Gradients diverged beyond tolerance: diff={max_diff:.2e}"
    print("=" * 70)
    print("Gradient accumulation validation: SUCCESS (Max diff < 1e-4)")
    print("=" * 70)





if __name__ == "__main__":
    test_gradient_accumulation_correctness()
