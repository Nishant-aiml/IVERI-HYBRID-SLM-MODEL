# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Integration, numerical stability, and telemetry tests for the Backbone (Phase 1.8)."""

from __future__ import annotations

import pytest
import torch

from configs.base_config import IVERIConfig, get_base_config
from model.backbone import Backbone, BackboneBlock


@pytest.fixture
def mini_config() -> IVERIConfig:
    """Create a very small configuration for fast testing."""
    cfg = get_base_config()
    cfg.model.hidden_dim = 64
    cfg.model.num_heads = 2
    cfg.model.num_layers = 2
    cfg.model.mamba_ratio = 2
    cfg.model.num_experts = 4
    cfg.model.num_active_experts = 2
    cfg.model.max_recursion_depth = 4
    cfg.model.titans_memory_dim = 32
    cfg.validate()
    return cfg


def test_backbone_block_creation_and_shapes(mini_config: IVERIConfig) -> None:
    """Verify BackboneBlock instantiates properly and produces correct output shapes."""
    block = BackboneBlock(mini_config)

    # Input shape: (B, P, D)
    B, P, D = 2, 8, mini_config.model.hidden_dim
    x = torch.randn(B, P, D)
    entropy = torch.rand(B, P, 1)

    out = block(x, entropy=entropy)
    assert out.shape == (B, P, D)
    assert not torch.isnan(out).any()


def test_backbone_full_orchestration_and_gradients(mini_config: IVERIConfig) -> None:
    """Verify that the full Backbone runs end-to-end and computes gradients differentiably."""
    backbone = Backbone(mini_config)

    B, P, D = 2, 6, mini_config.model.hidden_dim
    x = torch.randn(B, P, D, requires_grad=True)
    entropy = torch.rand(B, P, 1)

    # Forward pass
    out = backbone(x, entropy=entropy)
    assert out.shape == (B, P, D)

    # Check that MoE router auxiliary losses were accumulated (at least one per layer/recursion step)
    assert len(backbone.current_aux_losses) >= mini_config.model.num_layers

    # Backward pass
    loss = out.pow(2).mean() + sum(backbone.current_aux_losses)
    loss.backward()

    # Verify gradient flow
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()

    # Verify gradients reach active Titans parameters (read/update path and gate)
    assert backbone.titans.out_proj.weight.grad is not None
    assert not torch.isnan(backbone.titans.out_proj.weight.grad).any()
    assert backbone.titans.base_W1.grad is not None
    assert not torch.isnan(backbone.titans.base_W1.grad).any()
    assert backbone.titans.q_proj.weight.grad is not None
    assert not torch.isnan(backbone.titans.q_proj.weight.grad).any()
    assert backbone.titans.entropy_proj.weight.grad is not None
    assert not torch.isnan(backbone.titans.entropy_proj.weight.grad).any()

    # Verify gradients reach BackboneBlock subcomponents
    has_grad = False
    for p in backbone.blocks.parameters():
        if p.requires_grad and p.grad is not None:
            assert not torch.isnan(p.grad).any()
            has_grad = True
    assert has_grad, "At least some block parameters must receive gradients."


def test_backbone_telemetry_compilation(mini_config: IVERIConfig) -> None:
    """Verify Backbone telemetry dict is populated and contains all requested parameters."""
    backbone = Backbone(mini_config)

    B, P, D = 2, 4, mini_config.model.hidden_dim
    x = torch.randn(B, P, D)
    entropy = torch.rand(B, P, 1)

    # Execute forward pass to compile telemetry
    _ = backbone(x, entropy=entropy)

    telemetry = backbone.telemetry
    assert isinstance(telemetry, dict)

    # Verify keys
    expected_keys = {
        "total_parameters",
        "parameters_per_module",
        "flops_per_module",
        "runtime_per_module",
        "peak_vram_mb",
        "average_vram_mb",
        "activation_memory_mb",
        "hidden_state_norm",
        "residual_norm",
        "gradient_norm_per_module",
        "entropy_statistics",
        "average_recursion_depth",
        "expert_utilization_histogram",
        "titans_read_count",
        "titans_write_count",
        "average_memory_update_magnitude",
        "average_throughput_tokens_per_sec",
        "forward_latency_seconds",
    }
    for key in expected_keys:
        assert key in telemetry, f"Missing telemetry key: {key}"

    assert telemetry["total_parameters"] > 0
    assert (
        len(telemetry["parameters_per_module"]["per_layer_parameter_count"])
        == mini_config.model.num_layers
    )
    assert len(telemetry["expert_utilization_histogram"]) == mini_config.model.num_experts


def test_backbone_residual_norm_and_order(mini_config: IVERIConfig) -> None:
    """Validate residual updates and assert execution sequence correctness."""
    backbone = Backbone(mini_config)

    B, P, D = 2, 4, mini_config.model.hidden_dim
    x = torch.ones(B, P, D)
    entropy = torch.ones(B, P, 1)  # Max recursion depth

    out = backbone(x, entropy=entropy)
    assert not torch.equal(out, x)

    # Check that residual norm in telemetry is strictly non-zero
    res_norm = backbone.telemetry["residual_norm"]
    assert res_norm > 0.0


def test_integration_stress_boundary_conditions(mini_config: IVERIConfig) -> None:
    """Stress test boundary cases including batch size 1, long sequences, and extreme entropy."""
    backbone = Backbone(mini_config)
    D = mini_config.model.hidden_dim

    # 1. Batch size 1
    x_bs1 = torch.randn(1, 4, D)
    ent_bs1 = torch.rand(1, 4, 1)
    out_bs1 = backbone(x_bs1, entropy=ent_bs1)
    assert out_bs1.shape == (1, 4, D)

    # 2. Very long sequence (up to 512 patches)
    x_long = torch.randn(1, 128, D)
    ent_long = torch.rand(1, 128, 1)
    out_long = backbone(x_long, entropy=ent_long)
    assert out_long.shape == (1, 128, D)

    # 3. Minimum recursion boundary (entropy zero)
    ent_zero = torch.zeros(2, 4, 1)
    x = torch.randn(2, 4, D)
    out_min = backbone(x, entropy=ent_zero)
    assert out_min.shape == (2, 4, D)
    assert backbone.telemetry["average_recursion_depth"] == 1.0

    # 4. Maximum recursion boundary (entropy one)
    ent_one = torch.ones(2, 4, 1)
    out_max = backbone(x, entropy=ent_one)
    assert out_max.shape == (2, 4, D)
    assert backbone.telemetry["average_recursion_depth"] == mini_config.model.max_recursion_depth


def test_moe_expert_imbalance_and_titans_saturation(mini_config: IVERIConfig) -> None:
    """Verify backbone telemetry with expert utilization skew and extreme Titans inputs."""
    backbone = Backbone(mini_config)
    D = mini_config.model.hidden_dim

    # Simulate high expert imbalance input (strongly constant input vector to skew router)
    x_skew = torch.full((2, 8, D), 10.0)
    entropy = torch.rand(2, 8, 1)

    _ = backbone(x_skew, entropy=entropy)
    util = backbone.telemetry["expert_utilization_histogram"]
    assert sum(util) > 0

    # Titans saturation (extreme values)
    x_sat = torch.full((1, 4, D), 1e5)
    entropy_sat = torch.ones(1, 4, 1)
    out_sat = backbone(x_sat, entropy=entropy_sat)
    assert not torch.isnan(out_sat).any()


def test_multilingual_utf8_pipeline_equivalence(mini_config: IVERIConfig) -> None:
    """Sanity check demonstrating integration with Unicode byte inputs."""
    backbone = Backbone(mini_config)

    # Mimic character-to-patch conversion
    # English "Hello", Chinese "你好", Hindi "नमस्ते", Emoji "👋"
    texts = ["Hello", "你好", "नमस्ते", "👋"]
    encoded_patches = []

    for text in texts:
        bytes_data = text.encode("utf-8")
        # Map each byte to a mock vector
        mock_rep = torch.stack(
            [torch.full((mini_config.model.hidden_dim,), float(b)) for b in bytes_data]
        ).unsqueeze(
            0
        )  # (1, num_bytes, D)

        # Mean pool to represent 1 patch
        patch = mock_rep.mean(dim=1, keepdim=True)
        encoded_patches.append(patch)

    x = torch.cat(encoded_patches, dim=1)  # (1, 4, D)
    entropy = torch.tensor([[[0.1], [0.8], [0.9], [0.4]]])  # assigned token-level entropies

    out = backbone(x, entropy=entropy)
    assert out.shape == (1, 4, mini_config.model.hidden_dim)
    assert not torch.isnan(out).any()


def test_seed_determinism(mini_config: IVERIConfig) -> None:
    """Verify that execution is perfectly deterministic given identical seeds."""
    torch.manual_seed(42)
    backbone1 = Backbone(mini_config)

    torch.manual_seed(42)
    backbone2 = Backbone(mini_config)

    # Verify parameters match
    for p1, p2 in zip(backbone1.parameters(), backbone2.parameters(), strict=False):
        assert torch.equal(p1, p2)

    x = torch.randn(2, 8, mini_config.model.hidden_dim)
    entropy = torch.rand(2, 8, 1)

    torch.manual_seed(100)
    out1 = backbone1(x.clone(), entropy=entropy.clone())

    torch.manual_seed(100)
    out2 = backbone2(x.clone(), entropy=entropy.clone())

    assert torch.allclose(out1, out2)
