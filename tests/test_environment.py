# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Verification tests for the development environment, imports, and PyTorch status."""

from __future__ import annotations

import sys


def test_python_version() -> None:
    """Test that the environment runs Python 3.10 or higher."""
    assert sys.version_info >= (3, 10), "IVERI CORE requires Python >= 3.10"


def test_torch_import() -> None:
    """Test that PyTorch can be imported."""
    import torch

    assert torch is not None


def test_numpy_import() -> None:
    """Test that NumPy can be imported."""
    import numpy

    assert numpy is not None


def test_einops_import() -> None:
    """Test that einops can be imported."""
    import einops

    assert einops is not None


def test_tqdm_import() -> None:
    """Test that tqdm can be imported."""
    import tqdm

    assert tqdm is not None


def test_core_import() -> None:
    """Test imports from the core package."""
    from core import IVERI_VERSION, BaseModule, ComponentRegistry, register

    assert IVERI_VERSION == "1.0.0"
    assert ComponentRegistry is not None
    assert register is not None
    assert BaseModule is not None


def test_config_import() -> None:
    """Test imports from the configs package."""
    from configs import IVERIConfig, get_base_config

    assert IVERIConfig is not None
    assert get_base_config is not None


def test_utils_logging_import() -> None:
    """Test imports from logging utility."""
    from utils.logging import get_logger, get_training_logger

    assert get_logger is not None
    assert get_training_logger is not None


def test_utils_validation_import() -> None:
    """Test imports from validation utilities."""
    from utils.validation import check_nan_inf, validate_config, validate_shape

    assert validate_shape is not None
    assert check_nan_inf is not None
    assert validate_config is not None


def test_cuda_availability_reported() -> None:
    """Check if CUDA is available, and print status."""
    import torch

    available = torch.cuda.is_available()
    print(f"\nCUDA Available: {available}")
    if available:
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
