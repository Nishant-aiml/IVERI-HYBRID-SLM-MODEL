"""IVERI CORE — Validation Sub-package.

Re-exports every public validation utility so that callers can write::

    from utils.validation import validate_shape, check_nan_inf

instead of reaching into individual sub-modules.
"""

from __future__ import annotations

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

__all__ = [
    # tensors
    "validate_shape",
    "check_nan_inf",
    "validate_dtype",
    "tensor_stats",
    # gradients
    "validate_gradient_norms",
    "check_gradient_flow",
    "gradient_stats",
    # configs
    "validate_config",
    "validate_device_compatibility",
    "validate_architecture_consistency",
    # memory
    "get_gpu_memory_usage",
    "estimate_model_memory",
    "MemoryTracker",
]
