# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Phase 7.10 tests: verify all model presets instantiate and forward-pass cleanly.

These tests are marked @pytest.mark.slow because the larger presets (150M+)
take several seconds on CPU. The nano and small presets run in the standard suite.
"""

from __future__ import annotations

import pathlib

import pytest
import torch

from configs.base_config import IVERIConfig
from model.iveri_core import IVERIModel

PRESETS_DIR = pathlib.Path(__file__).parent.parent / "configs" / "presets"

# Fast presets run in the standard suite (< 5s each on CPU)
FAST_PRESETS = ["nano_10m", "small_35m"]

# Slow presets are explicitly excluded from standard suite runs
SLOW_PRESETS = ["medium_70m", "large_150m", "xlarge_300m", "max_500m"]


def _load_and_instantiate(preset_name: str, device: str = "cpu") -> tuple[IVERIModel, IVERIConfig]:
    """Load a preset and instantiate the model."""
    preset_path = PRESETS_DIR / f"{preset_name}.yaml"
    assert preset_path.exists(), f"Preset not found: {preset_path}"
    config = IVERIConfig.load(preset_path)
    config.hardware.device = device
    config.hardware.mixed_precision = "none"
    model = IVERIModel(config).to(device)
    return model, config


@pytest.mark.parametrize("preset_name", FAST_PRESETS)
def test_preset_forward_pass(preset_name: str) -> None:
    """Verify fast presets instantiate and produce valid logits."""
    model, config = _load_and_instantiate(preset_name)
    model.eval()

    B, S = 1, 32
    x = torch.randint(0, 256, (B, S))
    with torch.no_grad():
        out = model(x)

    assert isinstance(out, dict), f"Expected dict output, got {type(out)}"
    assert "logits" in out, f"No 'logits' key in output: {list(out.keys())}"
    logits = out["logits"]

    assert logits.shape[0] == B
    assert logits.shape[1] == S
    assert logits.ndim == 3
    assert not torch.isnan(logits).any(), f"NaN in logits for {preset_name}"
    assert not torch.isinf(logits).any(), f"Inf in logits for {preset_name}"


@pytest.mark.parametrize("preset_name", FAST_PRESETS)
def test_preset_parameter_count(preset_name: str) -> None:
    """Verify fast presets have sensible parameter counts."""
    model, config = _load_and_instantiate(preset_name)
    total_params = sum(p.numel() for p in model.parameters())

    # Nano should be ~9-12M, Small ~30-40M
    expected_ranges = {
        "nano_10m": (5_000_000, 15_000_000),
        "small_35m": (25_000_000, 50_000_000),
    }
    lo, hi = expected_ranges.get(preset_name, (1_000_000, 1_000_000_000))
    assert lo <= total_params <= hi, (
        f"{preset_name}: param count {total_params:,} out of expected range [{lo:,}, {hi:,}]"
    )


@pytest.mark.parametrize("preset_name", FAST_PRESETS)
def test_preset_gradient_flow(preset_name: str) -> None:
    """Verify gradients flow through the model for fast presets."""
    model, config = _load_and_instantiate(preset_name)
    model.train()

    B, S = 1, 16
    x = torch.randint(0, 256, (B, S))
    targets = torch.randint(0, 256, (B, S))

    out = model(x)
    logits = out["logits"]  # (B, S, vocab)

    # Cross-entropy loss
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        targets.view(-1),
    )
    loss.backward()

    # At least some parameters should have gradients
    grad_params = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    assert len(grad_params) > 0, f"No gradients for {preset_name}"
    assert not any(torch.isnan(p.grad).any() for p in grad_params), f"NaN gradients in {preset_name}"


@pytest.mark.parametrize("preset_name", FAST_PRESETS)
def test_preset_config_constraints(preset_name: str) -> None:
    """Verify preset configs satisfy all architecture constraints."""
    preset_path = PRESETS_DIR / f"{preset_name}.yaml"
    config = IVERIConfig.load(preset_path)

    assert config.model.hidden_dim % config.model.num_heads == 0, (
        f"{preset_name}: hidden_dim ({config.model.hidden_dim}) not divisible by "
        f"num_heads ({config.model.num_heads})"
    )
    assert config.model.num_active_experts <= config.model.num_experts, (
        f"{preset_name}: num_active_experts ({config.model.num_active_experts}) > "
        f"num_experts ({config.model.num_experts})"
    )
    assert config.model.titans_memory_dim <= config.model.hidden_dim, (
        f"{preset_name}: titans_memory_dim ({config.model.titans_memory_dim}) > "
        f"hidden_dim ({config.model.hidden_dim})"
    )
    assert config.training.warmup_steps < config.training.max_steps, (
        f"{preset_name}: warmup_steps ({config.training.warmup_steps}) >= "
        f"max_steps ({config.training.max_steps})"
    )
    eff_batch = config.training.batch_size * config.training.gradient_accumulation
    assert eff_batch <= 4096, (
        f"{preset_name}: effective batch size ({eff_batch}) exceeds limit 4096"
    )


# ── Slow preset tests (run with: pytest -m slow) ──────────────────────────────


@pytest.mark.slow
@pytest.mark.parametrize("preset_name", SLOW_PRESETS)
def test_slow_preset_forward_pass(preset_name: str) -> None:
    """Verify slow presets (70M-500M) produce valid logits. Run explicitly."""
    model, config = _load_and_instantiate(preset_name)
    model.eval()

    B, S = 1, 16  # Smaller sequence for CPU speed
    x = torch.randint(0, 256, (B, S))
    with torch.no_grad():
        out = model(x)

    logits = out["logits"]
    assert logits.shape[0] == B
    assert logits.shape[1] == S
    assert not torch.isnan(logits).any(), f"NaN in logits for {preset_name}"


@pytest.mark.slow
@pytest.mark.parametrize("preset_name", SLOW_PRESETS)
def test_slow_preset_config_constraints(preset_name: str) -> None:
    """Verify slow preset configs satisfy architecture constraints."""
    preset_path = PRESETS_DIR / f"{preset_name}.yaml"
    config = IVERIConfig.load(preset_path)

    assert config.model.hidden_dim % config.model.num_heads == 0
    assert config.model.num_active_experts <= config.model.num_experts
    assert config.model.titans_memory_dim <= config.model.hidden_dim
    assert config.training.warmup_steps < config.training.max_steps
