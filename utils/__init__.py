"""IVERI CORE — Utilities Package.

Provides structured logging, tensor/gradient validation, config checking,
and GPU memory tracking helpers used throughout the project.

Public API
----------
>>> from utils import get_logger
>>> logger = get_logger(__name__)

>>> from utils import validate_shape, check_nan_inf, validate_config
"""

from __future__ import annotations

from utils.logging import get_logger, get_training_logger
from utils.validation import (
    MemoryTracker,
    check_gradient_flow,
    check_nan_inf,
    estimate_model_memory,
    get_gpu_memory_usage,
    gradient_stats,
    tensor_stats,
    validate_architecture_consistency,
    validate_config,
    validate_device_compatibility,
    validate_dtype,
    validate_gradient_norms,
    validate_shape,
)

__all__ = [
    # Logging
    "get_logger",
    "get_training_logger",
    # Tensor validation
    "validate_shape",
    "check_nan_inf",
    "validate_dtype",
    "tensor_stats",
    # Gradient validation
    "validate_gradient_norms",
    "check_gradient_flow",
    "gradient_stats",
    # Config validation
    "validate_config",
    "validate_device_compatibility",
    "validate_architecture_consistency",
    # Memory tracking
    "get_gpu_memory_usage",
    "estimate_model_memory",
    "MemoryTracker",
]
