"""Tensor validation utilities for the IVERI CORE project.

Provides runtime assertions for tensor shapes, data-types, and numerical
health (NaN / Inf detection) that are used throughout forward passes and
data pipelines to catch errors early.

Typical usage::

    from utils.validation.tensors import validate_shape, check_nan_inf

    validate_shape(x, (batch, -1, 256), name="encoder_input")
    check_nan_inf(x, name="encoder_input")
"""

from __future__ import annotations

import torch

from core.exceptions import ShapeError, ValidationError

# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------


def validate_shape(
    tensor: torch.Tensor,
    expected: tuple[int, ...],
    name: str = "tensor",
) -> None:
    """Assert that *tensor* has the *expected* shape.

    A value of ``-1`` in any position of *expected* acts as a **wildcard**
    that matches any size in that dimension.

    Args:
        tensor: The tensor whose shape is checked.
        expected: Desired shape tuple.  Use ``-1`` for "any size".
        name: Human-readable label included in the error message.

    Raises:
        ShapeError: If the number of dimensions differs or any
            non-wildcard dimension does not match.

    Examples:
        >>> import torch
        >>> validate_shape(torch.zeros(2, 3, 256), (2, -1, 256))  # OK
        >>> validate_shape(torch.zeros(2, 3), (2, 4))
        Traceback (most recent call last):
        ...
        core.exceptions.ShapeError: ...
    """
    actual = tensor.shape

    if len(actual) != len(expected):
        raise ShapeError(
            f"'{name}' has {len(actual)} dimensions, expected {len(expected)}.",
            details=f"actual={tuple(actual)}, expected={expected}",
        )

    mismatches: list[str] = []
    for dim_idx, (act, exp) in enumerate(zip(actual, expected, strict=False)):
        if exp != -1 and act != exp:
            mismatches.append(f"dim {dim_idx}: got {act}, expected {exp}")

    if mismatches:
        raise ShapeError(
            f"'{name}' shape mismatch: {', '.join(mismatches)}.",
            details=f"actual={tuple(actual)}, expected={expected}",
        )


# ---------------------------------------------------------------------------
# Numerical health
# ---------------------------------------------------------------------------


def check_nan_inf(
    tensor: torch.Tensor,
    name: str = "tensor",
) -> None:
    """Raise if *tensor* contains any NaN or Inf values.

    The error message includes both the count and percentage of
    problematic elements so that the caller can gauge severity.

    Args:
        tensor: The tensor to inspect.
        name: Human-readable label for the error message.

    Raises:
        ValidationError: If one or more elements are NaN or ±Inf.
    """
    num_elements = tensor.numel()
    if num_elements == 0:
        return

    nan_count = int(torch.isnan(tensor).sum().item())
    inf_count = int(torch.isinf(tensor).sum().item())

    issues: list[str] = []
    if nan_count > 0:
        pct = 100.0 * nan_count / num_elements
        issues.append(f"{nan_count} NaN ({pct:.2f}%)")
    if inf_count > 0:
        pct = 100.0 * inf_count / num_elements
        issues.append(f"{inf_count} Inf ({pct:.2f}%)")

    if issues:
        raise ValidationError(
            f"'{name}' contains invalid values: {', '.join(issues)}.",
            details=f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}",
        )


# ---------------------------------------------------------------------------
# Dtype validation
# ---------------------------------------------------------------------------


def validate_dtype(
    tensor: torch.Tensor,
    expected: torch.dtype,
    name: str = "tensor",
) -> None:
    """Assert that *tensor* has the *expected* dtype.

    Args:
        tensor: The tensor to check.
        expected: The required :class:`torch.dtype`.
        name: Human-readable label for the error message.

    Raises:
        ValidationError: If the dtype does not match.
    """
    if tensor.dtype != expected:
        raise ValidationError(
            f"'{name}' dtype mismatch: got {tensor.dtype}, expected {expected}.",
        )


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    """Compute summary statistics for *tensor*.

    Args:
        tensor: The tensor to summarise.  Must be a floating-point type
            for ``mean``/``std``; integer tensors are cast to float
            internally.

    Returns:
        A dictionary with keys ``min``, ``max``, ``mean``, ``std``,
        ``num_nan``, and ``num_inf``.
    """
    t = tensor.detach().float()
    nan_mask = torch.isnan(t)
    inf_mask = torch.isinf(t)
    finite_mask = ~(nan_mask | inf_mask)
    finite_t = t[finite_mask]

    if finite_t.numel() > 0:
        t_min = float(finite_t.min().item())
        t_max = float(finite_t.max().item())
        t_mean = float(finite_t.mean().item())
        t_std = float(finite_t.std().item()) if finite_t.numel() > 1 else 0.0
    else:
        t_min = float("nan")
        t_max = float("nan")
        t_mean = float("nan")
        t_std = 0.0

    return {
        "min": t_min,
        "max": t_max,
        "mean": t_mean,
        "std": t_std,
        "num_nan": float(nan_mask.sum().item()),
        "num_inf": float(inf_mask.sum().item()),
    }
