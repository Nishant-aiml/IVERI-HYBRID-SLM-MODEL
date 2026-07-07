"""Gradient validation utilities for the IVERI CORE project.

Provides runtime checks for gradient health during training: norm bounds,
flow verification across all parameters, and per-parameter summary
statistics.  These utilities are designed to be called inside training
loops (e.g. after ``loss.backward()``) and are intentionally cheap
enough for every-step use.

Typical usage::

    from utils.validation.gradients import validate_gradient_norms

    norms = validate_gradient_norms(model, max_norm=100.0)
"""

from __future__ import annotations

from collections.abc import Iterator

from torch import nn

from core.exceptions import ValidationError

# ---------------------------------------------------------------------------
# Gradient norm validation
# ---------------------------------------------------------------------------


def validate_gradient_norms(
    model: nn.Module,
    max_norm: float = 100.0,
    min_norm: float = 1e-10,
) -> dict[str, float]:
    """Check that every parameter gradient norm is within bounds.

    Iterates over all parameters in *model* that have a non-``None``
    ``.grad`` attribute, computes the L2 norm, and raises if any norm
    exceeds *max_norm* or falls below *min_norm*.

    Args:
        model: The :class:`nn.Module` to inspect.
        max_norm: Upper bound on the gradient L2 norm.
        min_norm: Lower bound on the gradient L2 norm (detects
            vanishing gradients).

    Returns:
        A dictionary mapping ``parameter_name → gradient_norm`` for
        every parameter that has a gradient.

    Raises:
        ValidationError: If any gradient norm is outside the
            ``[min_norm, max_norm]`` range.
    """
    norms: dict[str, float] = {}
    violations: list[str] = []

    for param_name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad_norm = float(param.grad.data.norm(2).item())
        norms[param_name] = grad_norm

        if grad_norm > max_norm:
            violations.append(f"{param_name}: norm={grad_norm:.4e} > max_norm={max_norm:.4e}")
        elif grad_norm < min_norm:
            violations.append(f"{param_name}: norm={grad_norm:.4e} < min_norm={min_norm:.4e}")

    if violations:
        raise ValidationError(
            f"Gradient norm violations detected ({len(violations)} parameters).",
            details="\n".join(violations),
        )

    return norms


# ---------------------------------------------------------------------------
# Gradient flow verification
# ---------------------------------------------------------------------------


def check_gradient_flow(
    named_parameters: Iterator[tuple[str, nn.Parameter]],
) -> dict[str, bool]:
    """Verify that gradients reach every parameter.

    This is useful for catching dead branches or frozen parameters that
    should be trainable.

    Args:
        named_parameters: An iterator of ``(name, parameter)`` pairs,
            typically from ``model.named_parameters()``.

    Returns:
        A dictionary mapping ``parameter_name → has_gradient`` where
        ``has_gradient`` is ``True`` when ``.grad`` is not ``None`` and
        contains at least one non-zero element.
    """
    flow: dict[str, bool] = {}

    for name, param in named_parameters:
        if not param.requires_grad:
            continue

        if param.grad is None:
            flow[name] = False
        else:
            flow[name] = bool(param.grad.abs().sum().item() > 0.0)

    return flow


# ---------------------------------------------------------------------------
# Per-parameter gradient statistics
# ---------------------------------------------------------------------------


def gradient_stats(
    model: nn.Module,
) -> dict[str, dict[str, float]]:
    """Compute per-parameter gradient statistics.

    Parameters whose ``.grad`` is ``None`` are silently skipped.

    Args:
        model: The :class:`nn.Module` to inspect.

    Returns:
        A nested dictionary::

            {
                "layer.weight": {
                    "norm": 0.42,
                    "min": -0.03,
                    "max":  0.05,
                    "mean": 0.001,
                    "std":  0.012,
                },
                ...
            }
    """
    stats: dict[str, dict[str, float]] = {}

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad = param.grad.data.float()
        stats[name] = {
            "norm": float(grad.norm(2).item()),
            "min": float(grad.min().item()),
            "max": float(grad.max().item()),
            "mean": float(grad.mean().item()),
            "std": float(grad.std().item()) if grad.numel() > 1 else 0.0,
        }

    return stats
