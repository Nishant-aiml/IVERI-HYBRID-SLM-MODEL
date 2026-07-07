# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the configuration system (IVERIConfig and sub-configs)."""

from __future__ import annotations

from pathlib import Path

import pytest

from configs.base_config import IVERIConfig, get_base_config
from core.exceptions import ConfigError


def test_default_config_creation(base_config: IVERIConfig) -> None:
    """Test that default config can be instantiated successfully."""
    assert base_config is not None
    assert isinstance(base_config, IVERIConfig)


def test_default_values_match_nano(base_config: IVERIConfig) -> None:
    """Test that default values match the 10M Nano architecture specification."""
    assert base_config.model.hidden_dim == 256
    assert base_config.model.num_layers == 6
    assert base_config.model.num_heads == 4
    assert base_config.model.mamba_ratio == 6
    assert base_config.model.num_experts == 4
    assert base_config.model.num_active_experts == 2
    assert base_config.model.max_recursion_depth == 8
    assert base_config.model.titans_memory_dim == 128


def test_nested_config_access(base_config: IVERIConfig) -> None:
    """Test that nested config values can be accessed correctly."""
    assert base_config.model.blt.patch_size_min == 1
    assert base_config.model.blt.patch_size_max == 8
    assert base_config.model.blt.entropy_threshold == 0.5


def test_to_dict_returns_dict(base_config: IVERIConfig) -> None:
    """Test that serialization to dict returns the correct structure."""
    d = base_config.to_dict()
    assert isinstance(d, dict)
    assert "model" in d
    assert "training" in d
    assert "hardware" in d
    assert "logging" in d
    assert d["model"]["hidden_dim"] == 256
    assert d["model"]["blt"]["patch_size_min"] == 1


def test_from_dict_roundtrip(base_config: IVERIConfig) -> None:
    """Test that dict serialization and deserialization is lossless."""
    d = base_config.to_dict()
    reconstructed = IVERIConfig.from_dict(d)
    assert reconstructed.to_dict() == d


def test_save_load_roundtrip(base_config: IVERIConfig, tmp_dir: Path) -> None:
    """Test that saving and loading JSON config files works correctly."""
    config_path = tmp_dir / "test_config.json"
    base_config.save(config_path)
    assert config_path.exists()

    loaded = IVERIConfig.load(config_path)
    assert loaded.to_dict() == base_config.to_dict()


def test_get_base_config_no_overrides() -> None:
    """Test that get_base_config with no overrides returns default config."""
    assert get_base_config().to_dict() == IVERIConfig().to_dict()


def test_get_base_config_with_overrides() -> None:
    """Test overrides in get_base_config."""
    cfg = get_base_config(
        model={"hidden_dim": 512, "num_layers": 12, "num_heads": 8},
        training={"learning_rate": 1e-4},
    )
    assert cfg.model.hidden_dim == 512
    assert cfg.model.num_layers == 12
    assert cfg.model.num_heads == 8
    assert cfg.training.learning_rate == 1e-4
    # Unoverridden values should retain defaults
    assert cfg.model.mamba_ratio == 6


def test_invalid_hidden_dim_zero() -> None:
    """Test that hidden_dim <= 0 raises ConfigError."""
    with pytest.raises(ConfigError):
        get_base_config(model={"hidden_dim": 0})


def test_invalid_hidden_dim_not_divisible_by_heads() -> None:
    """Test that hidden_dim not divisible by num_heads raises ConfigError."""
    with pytest.raises(ConfigError):
        get_base_config(model={"hidden_dim": 255, "num_heads": 4})


def test_invalid_active_experts_exceeds_total() -> None:
    """Test that num_active_experts > num_experts raises ConfigError."""
    with pytest.raises(ConfigError):
        get_base_config(model={"num_experts": 4, "num_active_experts": 5})


def test_invalid_learning_rate_zero() -> None:
    """Test that learning_rate <= 0 raises ConfigError."""
    with pytest.raises(ConfigError):
        get_base_config(training={"learning_rate": 0})


def test_invalid_min_lr_exceeds_lr() -> None:
    """Test that min_lr > learning_rate raises ConfigError."""
    with pytest.raises(ConfigError):
        get_base_config(training={"learning_rate": 1e-4, "min_lr": 2e-4})


def test_invalid_mixed_precision() -> None:
    """Test that invalid mixed_precision value raises ConfigError."""
    with pytest.raises(ConfigError):
        get_base_config(hardware={"mixed_precision": "fp64"})


def test_invalid_log_level() -> None:
    """Test that invalid log_level raises ConfigError."""
    with pytest.raises(ConfigError):
        get_base_config(logging={"log_level": "VERBOSE"})


def test_invalid_patch_size() -> None:
    """Test that patch_size_max < patch_size_min raises ConfigError."""
    with pytest.raises(ConfigError):
        get_base_config(model={"blt": {"patch_size_min": 5, "patch_size_max": 3}})


def test_effective_batch_size_limit() -> None:
    """Test that effective batch size > 4096 raises ConfigError."""
    with pytest.raises(ConfigError):
        get_base_config(training={"batch_size": 256, "gradient_accumulation": 32})


def test_warmup_exceeds_max_steps() -> None:
    """Test that warmup_steps >= max_steps raises ConfigError."""
    with pytest.raises(ConfigError):
        get_base_config(training={"warmup_steps": 50000, "max_steps": 50000})
