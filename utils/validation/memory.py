"""GPU memory tracking utilities for the IVERI CORE project.

Provides three levels of memory instrumentation:

1. **Snapshot** — :func:`get_gpu_memory_usage` returns current CUDA
   memory counters in megabytes.
2. **Estimation** — :func:`estimate_model_memory` predicts VRAM
   requirements from a parameter count *before* the model is created.
3. **Context-managed tracking** — :class:`MemoryTracker` measures the
   GPU memory delta incurred by a block of code.

All functions degrade gracefully on CPU-only machines (returning zeros).

Typical usage::

    from utils.validation.memory import MemoryTracker

    with MemoryTracker() as mt:
        model = build_model(config)
    print(mt.summary())
"""

from __future__ import annotations

from types import TracebackType

import torch

# ---------------------------------------------------------------------------
# Bytes ↔ MB conversion factor
# ---------------------------------------------------------------------------
_BYTES_PER_MB: float = 1024.0 * 1024.0


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def get_gpu_memory_usage() -> dict[str, float]:
    """Return current CUDA memory usage in megabytes.

    Returns:
        A dictionary with keys ``allocated``, ``reserved``, and
        ``peak`` (all in MB).  Values are ``0.0`` when CUDA is not
        available.
    """
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "peak": 0.0}

    return {
        "allocated": torch.cuda.memory_allocated() / _BYTES_PER_MB,
        "reserved": torch.cuda.memory_reserved() / _BYTES_PER_MB,
        "peak": torch.cuda.max_memory_allocated() / _BYTES_PER_MB,
    }


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------


def estimate_model_memory(
    param_count: int,
    dtype_bytes: int = 4,
) -> dict[str, float]:
    """Estimate VRAM required to train a model with *param_count* parameters.

    The estimate accounts for:

    * **Parameters** — ``param_count × dtype_bytes``
    * **Gradients** — same size as parameters (one gradient per param)
    * **Optimizer state** — Adam stores two momentum buffers per param,
      so 2× the parameter size.

    Activations and batch-dependent buffers are **not** included; this
    is a *lower-bound* estimate.

    Args:
        param_count: Total number of learnable parameters.
        dtype_bytes: Bytes per element (default 4 for ``float32``).

    Returns:
        A dictionary with keys ``params_mb``, ``gradients_mb``,
        ``optimizer_mb``, and ``total_mb``.
    """
    params_mb = (param_count * dtype_bytes) / _BYTES_PER_MB
    gradients_mb = params_mb  # one gradient per parameter
    optimizer_mb = params_mb * 2.0  # Adam: first & second moment

    return {
        "params_mb": params_mb,
        "gradients_mb": gradients_mb,
        "optimizer_mb": optimizer_mb,
        "total_mb": params_mb + gradients_mb + optimizer_mb,
    }


# ---------------------------------------------------------------------------
# Context-managed tracker
# ---------------------------------------------------------------------------


class MemoryTracker:
    """Context manager that measures GPU memory consumed by a code block.

    On CPU-only systems every property returns ``0.0`` so callers never
    need to gate usage on CUDA availability.

    Example::

        with MemoryTracker() as mt:
            model = MyModel(config).cuda()
        print(f"Allocated {mt.allocated_delta:.1f} MB")
    """

    def __init__(self) -> None:
        self._cuda_available: bool = torch.cuda.is_available()
        self._start_allocated: float = 0.0
        self._end_allocated: float = 0.0
        self._peak_during: float = 0.0

    # -- Context protocol ---------------------------------------------------

    def __enter__(self) -> MemoryTracker:
        """Record starting memory and reset the peak tracker.

        Returns:
            ``self`` so that the tracker can be used in ``with … as mt:``
            blocks.
        """
        if self._cuda_available:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            self._start_allocated = torch.cuda.memory_allocated() / _BYTES_PER_MB
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Record ending memory and compute peak.

        Args:
            exc_type: Exception type, if any.
            exc_val: Exception value, if any.
            exc_tb: Traceback, if any.
        """
        if self._cuda_available:
            torch.cuda.synchronize()
            self._end_allocated = torch.cuda.memory_allocated() / _BYTES_PER_MB
            self._peak_during = torch.cuda.max_memory_allocated() / _BYTES_PER_MB

    # -- Public properties --------------------------------------------------

    @property
    def allocated_delta(self) -> float:
        """Net change in allocated GPU memory (MB).

        Returns:
            Positive value when allocation grew, negative when it shrank,
            ``0.0`` on CPU.
        """
        return self._end_allocated - self._start_allocated

    @property
    def peak_during(self) -> float:
        """Peak GPU memory allocated during the tracked block (MB).

        Returns:
            Peak allocation in MB, or ``0.0`` on CPU.
        """
        return self._peak_during

    def summary(self) -> dict[str, float]:
        """Return a summary dictionary of the memory measurement.

        Returns:
            A dictionary with keys ``start_mb``, ``end_mb``,
            ``delta_mb``, and ``peak_mb``.
        """
        return {
            "start_mb": self._start_allocated,
            "end_mb": self._end_allocated,
            "delta_mb": self.allocated_delta,
            "peak_mb": self._peak_during,
        }
