"""Abstract base classes (interfaces) for IVERI CORE neural modules.

Every concrete component in the architecture — routers, memory banks,
encoders, decoders — inherits from one of the ABCs defined here.  This
guarantees a uniform contract across the codebase and lets the factory /
registry machinery instantiate components generically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseModule(nn.Module, ABC):
    """Root abstract module that all IVERI components extend.

    Enforces a standard ``forward`` signature and provides hooks for
    parameter (re-)initialisation and human-readable ``repr`` output.
    """

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Run the module's primary computation.

        Parameters
        ----------
        x:
            Input tensor — semantics are component-specific.
        **kwargs:
            Arbitrary keyword arguments for component-specific options
            (masks, caches, step indices, etc.).

        Returns
        -------
        torch.Tensor
            Transformed output tensor.
        """
        ...

    def reset_parameters(self) -> None:
        """Re-initialise learnable parameters to their starting values.

        The default implementation is a no-op; subclasses should override
        this when they own parameters that need deterministic init.
        """

    def extra_repr(self) -> str:
        """Return a string inserted into the module's ``repr``.

        Override in subclasses to surface key hyper-parameters (e.g.
        ``hidden_dim=256, num_heads=4``).
        """
        return ""


class BaseRouter(BaseModule):
    """Interface for mixture-of-experts / routing components.

    Routers decide *which* expert(s) process each token.  The canonical
    return contract is a ``(dispatch_weights, dispatch_indices)`` tuple.
    """

    @abstractmethod
    def route(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute expert routing decisions.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, seq_len, hidden_dim)``.

        Returns
        -------
        dispatch_weights:
            Soft weights assigned to each selected expert,
            shape ``(batch, seq_len, top_k)``.
        dispatch_indices:
            Integer expert indices for each token,
            shape ``(batch, seq_len, top_k)``.
        """
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Apply the full route-and-combine pipeline."""
        ...


class BaseMemory(BaseModule):
    """Interface for external / persistent memory modules (e.g. Titans)."""

    @abstractmethod
    def read(self, query: torch.Tensor) -> torch.Tensor:
        """Retrieve information from the memory bank.

        Parameters
        ----------
        query:
            Query tensor used to address / attend over memory slots.

        Returns
        -------
        torch.Tensor
            Retrieved memory content.
        """
        ...

    @abstractmethod
    def write(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Store information into the memory bank.

        Parameters
        ----------
        key:
            Key tensor for addressing.
        value:
            Value tensor to be written.
        """
        ...

    def reset_memory(self) -> None:
        """Clear all stored memory state.

        The default implementation is a no-op; subclasses that maintain
        buffers must override.
        """

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Combined read-process-write step used during training."""
        ...


class BaseEncoder(BaseModule):
    """Interface for encoder components (e.g. BLT byte-to-patch encoder)."""

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input into a latent representation.

        Parameters
        ----------
        x:
            Raw input tensor (e.g. byte embeddings).

        Returns
        -------
        torch.Tensor
            Encoded latent tensor.
        """
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Alias that delegates to :meth:`encode` by default."""
        ...


class BaseDecoder(BaseModule):
    """Interface for decoder components (e.g. BLT patch-to-byte decoder)."""

    @abstractmethod
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode a latent representation back to the output space.

        Parameters
        ----------
        x:
            Latent tensor to decode.

        Returns
        -------
        torch.Tensor
            Decoded output tensor.
        """
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Alias that delegates to :meth:`decode` by default."""
        ...
