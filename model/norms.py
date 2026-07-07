# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Root Mean Square Layer Normalization (RMSNorm).

Provides the :class:`RMSNorm` module — a computationally cheaper alternative
to standard Layer Normalization that omits the mean-centering step while
retaining comparable training stability and downstream accuracy.

Reference
---------
Zhang, B. & Sennrich, R. (2019). "Root Mean Square Layer Normalization."
*Advances in Neural Information Processing Systems (NeurIPS) 32*.

The implementation follows the LLaMA-style pattern: normalization math is
always executed in FP32 for numerical safety, allowing the module to be
used seamlessly with FP16 and BF16 mixed-precision training.
"""

from __future__ import annotations

import torch

from core.interfaces import BaseModule
from core.registry import register


@register("rmsnorm")
class RMSNorm(BaseModule):
    r"""Root Mean Square Layer Normalization.

    Normalizes the input by its root-mean-square statistic and scales the
    result by a learnable weight vector :math:`\gamma`.

    .. math::

        \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}

        \text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma

    Unlike standard Layer Normalization, RMSNorm does **not** subtract the
    mean and does **not** include a learnable bias term.  This makes it
    ~10–15 % faster while achieving equivalent model quality.

    Parameters
    ----------
    dim : int
        Feature dimension (last axis) of the input tensors.
    eps : float, optional
        Small constant added inside the square-root for numerical stability.
        Default: ``1e-6``.

    Attributes
    ----------
    weight : torch.nn.Parameter
        Learnable scale parameter :math:`\gamma` of shape ``(dim,)``,
        initialized to ones.

    Examples
    --------
    >>> norm = RMSNorm(dim=512)
    >>> x = torch.randn(2, 128, 512)
    >>> out = norm(x)
    >>> out.shape
    torch.Size([2, 128, 512])
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Apply RMS normalization to the input tensor.

        The normalization computation is performed in FP32 regardless of
        the input dtype to prevent overflow when squaring FP16/BF16 values.
        The output is cast back to the original input dtype before return.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of arbitrary shape whose last dimension equals
            ``dim``.
        **kwargs : object
            Ignored; accepted for interface compatibility with
            :class:`~core.interfaces.BaseModule`.

        Returns
        -------
        torch.Tensor
            Normalized tensor with the same shape and dtype as *x*.
        """
        input_dtype = x.dtype
        x_float = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        norm = x_float * torch.rsqrt(
            x_float.pow(2).mean(-1, keepdim=True) + self.eps,
        )
        return (norm * self.weight.to(norm.dtype)).to(input_dtype)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset_parameters(self) -> None:
        """Re-initialize :attr:`weight` to all ones."""
        torch.nn.init.ones_(self.weight)

    def extra_repr(self) -> str:
        """Return a summary string for the module's ``repr``."""
        return f"dim={self.dim}, eps={self.eps}"
