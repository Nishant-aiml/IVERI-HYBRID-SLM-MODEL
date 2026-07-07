"""Configuration validation utilities for the IVERI CORE project.

Validates :class:`IVERIConfig` (or duck-typed config objects) for internal
consistency, device compatibility, and architectural soundness **before**
any model is instantiated — catching mis-configurations early.

The validators accept any object whose attributes match the expected
config schema so that they work with both the frozen dataclass hierarchy
and plain namespace objects used in tests.

Typical usage::

    from utils.validation.configs import validate_config

    warnings = validate_config(config)
    if warnings:
        for w in warnings:
            logger.warning(w)
"""

from __future__ import annotations

import logging

import torch

from core.exceptions import ConfigError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Master validator
# ---------------------------------------------------------------------------


def validate_config(config: object) -> list[str]:
    """Run every consistency check against *config*.

    Delegates to :func:`validate_device_compatibility` and
    :func:`validate_architecture_consistency`, aggregating all
    warnings.  Fatal issues are raised immediately as
    :class:`ConfigError`.

    Args:
        config: A configuration object whose attributes conform to the
            IVERI config schema (e.g. ``config.model.hidden_dim``).

    Returns:
        A list of non-fatal warning strings.  An empty list indicates
        the configuration is fully valid.

    Raises:
        ConfigError: If a fatal misconfiguration is detected (e.g.
            ``hidden_dim <= 0``).
    """
    warnings: list[str] = []

    validate_device_compatibility(config)
    warnings.extend(validate_architecture_consistency(config))

    return warnings


# ---------------------------------------------------------------------------
# Device compatibility
# ---------------------------------------------------------------------------


def validate_device_compatibility(config: object) -> None:
    """Check that the requested device is available on this machine.

    Issues a log-level warning (and mutates nothing) when CUDA is
    requested but unavailable, rather than raising — this allows the
    caller to fall back gracefully.

    Args:
        config: Configuration object.  Looks for
            ``config.hardware.device`` (string).

    Raises:
        ConfigError: If ``config`` has no ``hardware`` attribute or
            ``hardware`` has no ``device`` attribute.
    """
    hardware = getattr(config, "hardware", None)
    if hardware is None:
        raise ConfigError(
            "Config is missing 'hardware' section.",
        )

    device: str | None = getattr(hardware, "device", None)
    if device is None:
        raise ConfigError(
            "Config 'hardware' section is missing 'device' field.",
        )

    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(
            "CUDA device '%s' requested but CUDA is not available. "
            "Training will likely fall back to CPU.",
            device,
        )

    if device.startswith("mps") and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        logger.warning(
            "MPS device '%s' requested but MPS is not available.",
            device,
        )


# ---------------------------------------------------------------------------
# Architecture consistency
# ---------------------------------------------------------------------------


def validate_architecture_consistency(config: object) -> list[str]:
    """Check that architecture hyper-parameters are mutually consistent.

    Performs the following checks (when the relevant attributes exist):

    * ``hidden_dim`` must be positive.
    * ``hidden_dim`` must be divisible by ``num_heads``.
    * ``mamba_ratio`` must be > 0 if present.
    * ``num_active_experts <= num_experts`` (MoE sanity).
    * ``num_layers`` must be positive.

    Unknown or missing attributes are silently skipped so that this
    function stays forward-compatible as the config schema evolves.

    Args:
        config: Configuration object.  Looks for ``config.model.*``.

    Returns:
        A list of non-fatal warning strings.

    Raises:
        ConfigError: If a fatal constraint is violated (e.g.
            ``hidden_dim <= 0``).
    """
    warnings: list[str] = []
    model = getattr(config, "model", None)

    if model is None:
        raise ConfigError("Config is missing 'model' section.")

    # --- hidden_dim --------------------------------------------------------
    hidden_dim: int | None = getattr(model, "hidden_dim", None)
    if hidden_dim is not None and hidden_dim <= 0:
        raise ConfigError(
            f"hidden_dim must be positive, got {hidden_dim}.",
        )

    # --- num_heads ---------------------------------------------------------
    num_heads: int | None = getattr(model, "num_heads", None)
    if hidden_dim is not None and num_heads is not None:
        if num_heads <= 0:
            raise ConfigError(
                f"num_heads must be positive, got {num_heads}.",
            )
        if hidden_dim % num_heads != 0:
            raise ConfigError(
                f"hidden_dim ({hidden_dim}) must be divisible by " f"num_heads ({num_heads}).",
            )

    # --- num_layers --------------------------------------------------------
    num_layers: int | None = getattr(model, "num_layers", None)
    if num_layers is not None and num_layers <= 0:
        raise ConfigError(
            f"num_layers must be positive, got {num_layers}.",
        )

    # --- mamba_ratio -------------------------------------------------------
    mamba_ratio: float | None = getattr(model, "mamba_ratio", None)
    if mamba_ratio is not None and mamba_ratio <= 0:
        raise ConfigError(
            f"mamba_ratio must be > 0, got {mamba_ratio}.",
        )

    # --- MoE: num_experts / num_active_experts -----------------------------
    num_experts: int | None = getattr(model, "num_experts", None)
    num_active: int | None = getattr(model, "num_active_experts", None)

    if num_experts is not None and num_experts <= 0:
        raise ConfigError(
            f"num_experts must be positive, got {num_experts}.",
        )

    if num_experts is not None and num_active is not None:
        if num_active <= 0:
            raise ConfigError(
                f"num_active_experts must be positive, got {num_active}.",
            )
        if num_active > num_experts:
            warnings.append(
                f"num_active_experts ({num_active}) exceeds "
                f"num_experts ({num_experts}). This is likely a "
                f"misconfiguration — all experts will always be active."
            )

    # --- head_dim consistency warning --------------------------------------
    if hidden_dim is not None and num_heads is not None:
        head_dim = hidden_dim // num_heads
        if head_dim < 8:
            warnings.append(
                f"head_dim ({head_dim}) is unusually small. "
                f"Consider increasing hidden_dim or reducing num_heads."
            )

    return warnings
