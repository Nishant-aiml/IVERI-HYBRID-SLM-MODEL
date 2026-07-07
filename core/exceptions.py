"""Custom exception hierarchy for the IVERI CORE project.

Every project-specific error inherits from :class:`IVERIError` so that
callers can catch the entire family with a single ``except IVERIError``
clause, or handle individual failure modes (config, shape, registry, …)
with finer granularity.

Each concrete exception ships with a sensible *default_message* and
accepts an optional *details* string for additional context that gets
appended automatically.
"""

from __future__ import annotations


class IVERIError(Exception):
    """Base exception for all IVERI CORE errors.

    Parameters
    ----------
    message:
        Human-readable error description.  Falls back to
        *default_message* when ``None``.
    details:
        Optional extra context (tensor shapes, config paths, etc.)
        that is appended to the rendered message.
    """

    default_message: str = "An error occurred in IVERI CORE."

    def __init__(
        self,
        message: str | None = None,
        details: str | None = None,
    ) -> None:
        self.details = details
        resolved = message or self.default_message
        if details:
            resolved = f"{resolved} | Details: {details}"
        super().__init__(resolved)


# --- Configuration --------------------------------------------------------


class ConfigError(IVERIError):
    """Raised when a configuration value is missing or invalid."""

    default_message: str = "Configuration validation failed."


# --- Tensor shapes --------------------------------------------------------


class ShapeError(IVERIError):
    """Raised when tensor dimensions do not match the expected shape."""

    default_message: str = "Tensor shape mismatch detected."


# --- Component registry ---------------------------------------------------


class RegistryError(IVERIError):
    """Raised on component registry conflicts (duplicate or not found)."""

    default_message: str = "Component registry error."


# --- General validation ---------------------------------------------------


class ValidationError(IVERIError):
    """Raised when a general validation check fails."""

    default_message: str = "Validation failed."


# --- Training convergence -------------------------------------------------


class ConvergenceError(IVERIError):
    """Raised when training fails to converge (NaN loss, gradient explosion, etc.)."""

    default_message: str = "Training convergence failure detected."


# --- Memory / VRAM --------------------------------------------------------


class MemoryError(IVERIError):  # noqa: A001 – intentional shadow of builtin
    """Raised when a VRAM or system-memory budget is exceeded."""

    default_message: str = "Memory budget exceeded."


# --- Checkpointing -------------------------------------------------------


class CheckpointError(IVERIError):
    """Raised when saving or loading a checkpoint fails."""

    default_message: str = "Checkpoint save/load failure."
