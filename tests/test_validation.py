# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the validation utilities suite (utils/validation/)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from configs.base_config import get_base_config
from core.exceptions import ShapeError, ValidationError
from utils.validation.configs import (
    validate_architecture_consistency,
    validate_config,
    validate_device_compatibility,
)
from utils.validation.gradients import (
    check_gradient_flow,
    gradient_stats,
    validate_gradient_norms,
)
from utils.validation.memory import (
    MemoryTracker,
    estimate_model_memory,
    get_gpu_memory_usage,
)
from utils.validation.tensors import (
    check_nan_inf,
    tensor_stats,
    validate_dtype,
    validate_shape,
)

# --- Tensor Validation Tests -----------------------------------------------


def test_validate_shape_correct() -> None:
    """Test that correct shape validation passes without error."""
    t = torch.randn(2, 3, 4)
    validate_shape(t, (2, 3, 4))  # Should pass silently


def test_validate_shape_wildcard() -> None:
    """Test that wildcard dimensions (-1) match any size."""
    t = torch.randn(5, 10, 15)
    validate_shape(t, (-1, 10, -1))  # Should pass silently


def test_validate_shape_mismatch() -> None:
    """Test that shape mismatches raise ShapeError."""
    t = torch.randn(2, 3, 4)
    with pytest.raises(ShapeError):
        validate_shape(t, (2, 3, 5))


def test_check_nan_inf_clean() -> None:
    """Test that clean tensors pass NaN/Inf check."""
    t = torch.randn(10, 10)
    check_nan_inf(t)  # Should pass silently


def test_check_nan_inf_with_nan() -> None:
    """Test that tensors containing NaN raise ValidationError."""
    t = torch.randn(10, 10)
    t[5, 5] = float("nan")
    with pytest.raises(ValidationError) as excinfo:
        check_nan_inf(t)
    assert "NaN" in str(excinfo.value)


def test_check_nan_inf_with_inf() -> None:
    """Test that tensors containing Inf raise ValidationError."""
    t = torch.randn(10, 10)
    t[2, 3] = float("inf")
    with pytest.raises(ValidationError) as excinfo:
        check_nan_inf(t)
    assert "Inf" in str(excinfo.value)


def test_validate_dtype_correct() -> None:
    """Test that correct dtype matches pass without error."""
    t = torch.randn(5, 5, dtype=torch.float32)
    validate_dtype(t, torch.float32)  # Should pass silently


def test_validate_dtype_mismatch() -> None:
    """Test that mismatched dtype raises ValidationError."""
    t = torch.randn(5, 5, dtype=torch.float32)
    with pytest.raises(ValidationError):
        validate_dtype(t, torch.int64)


def test_tensor_stats() -> None:
    """Test tensor statistics computation."""
    t = torch.tensor([1.0, 2.0, float("nan"), float("inf"), -float("inf")])
    stats = tensor_stats(t)
    assert stats["num_nan"] == 1
    assert stats["num_inf"] == 2
    # Finite values should be used for stats calculation
    assert stats["min"] == 1.0
    assert stats["max"] == 2.0


# --- Gradient Validation Tests ---------------------------------------------


def test_gradient_stats_and_flow_with_simple_model() -> None:
    """Test gradient stats and flow utilities using a toy module."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1),
    )
    x = torch.randn(5, 10)
    y = model(x)
    loss = y.mean()
    loss.backward()

    # Check validation passes
    validate_gradient_norms(model, max_norm=100.0)

    # Check flow verification
    flow = check_gradient_flow(model.named_parameters())
    assert all(flow.values())

    # Check stats extraction
    stats = gradient_stats(model)
    assert len(stats) > 0
    for _param_name, param_stats in stats.items():
        assert "norm" in param_stats
        assert param_stats["norm"] >= 0.0


# --- Config Validation Tests -----------------------------------------------


def test_validate_config_valid() -> None:
    """Test that valid configuration passes validation checks."""
    config = get_base_config()
    warnings = validate_config(config)
    assert isinstance(warnings, list)
    assert len(warnings) == 0


def test_validate_architecture_consistency_valid() -> None:
    """Test architectural validation with valid configs."""
    config = get_base_config()
    warnings = validate_architecture_consistency(config)
    assert isinstance(warnings, list)
    assert len(warnings) == 0


def test_validate_device_compatibility() -> None:
    """Verify that device checks run without throwing unexpected errors."""
    config = get_base_config()
    validate_device_compatibility(config)  # Should pass or warn gracefully


# --- Memory Validation Tests -----------------------------------------------


def test_estimate_model_memory() -> None:
    """Test parameter/memory estimation logic."""
    estimates = estimate_model_memory(10_000_000, dtype_bytes=4)
    assert estimates["params_mb"] > 0
    assert estimates["gradients_mb"] > 0
    assert estimates["optimizer_mb"] > 0
    assert estimates["total_mb"] > 0


def test_memory_tracker_cpu() -> None:
    """Test that MemoryTracker operates gracefully in CPU contexts."""
    with MemoryTracker() as tracker:
        t = torch.randn(100, 100)
        res = t @ t
        assert res is not None

    summary = tracker.summary()
    assert "delta_mb" in summary
    assert "peak_mb" in summary


def test_gpu_memory_usage_no_crash() -> None:
    """Verify that GPU memory query returns dictionary and does not crash."""
    usage = get_gpu_memory_usage()
    assert isinstance(usage, dict)
    assert "allocated" in usage
    assert "reserved" in usage
    assert "peak" in usage
