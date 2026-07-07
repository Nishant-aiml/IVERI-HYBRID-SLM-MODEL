# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Sparse Mixture of Experts Router.

Implements the top-k sparsely-gated routing mechanism from:

    *Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer*
    Shazeer et al. (2017). arXiv:1701.06538

Features:
- Noisy top-k selection for exploration.
- Softmax routing weights normalization.
- Shazeer load balancing auxiliary loss to prevent expert collapse.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from configs.base_config import IVERIConfig
from core.interfaces import BaseRouter
from core.registry import register


@register("moe_router")
class SparseMoERouter(BaseRouter):
    """Sparsely-Gated Mixture of Experts Router with Load Balancing.

    Projects inputs to expert gating logit space, selects the top-k experts,
    and normalizes their routing weights via softmax. Computes an auxiliary
    loss during training to enforce uniform expert utilization.
    """

    def __init__(self, config: IVERIConfig) -> None:
        """Initialize the sparse MoE router.

        Parameters
        ----------
        config : IVERIConfig
            Project configuration containing model hyperparameters.
        """
        super().__init__()
        self.config = config
        self.hidden_dim = config.model.hidden_dim
        self.num_experts = config.model.num_experts
        self.use_entropy_routing = config.model.use_entropy_routing
        self.num_active_experts = config.model.num_active_experts

        # Retrieve fallback hyperparameters if not present in default config
        self.moe_noise_epsilon = float(getattr(config.model, "moe_noise_epsilon", 1.0))
        self.moe_aux_loss_coef = float(getattr(config.model, "moe_aux_loss_coef", 0.01))
        self.noise_enabled = bool(getattr(config.model, "moe_noise_enabled", False))

        # Gating projections
        self.wg = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.w_noise = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        # Patent 3 / Option C: patch entropy biases expert gating logits per token
        self.w_entropy = nn.Linear(1, self.num_experts, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset projection layers using normal initializations."""
        nn.init.normal_(self.wg.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w_noise.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w_entropy.weight, mean=0.0, std=0.02)

    def _flatten_inputs(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, ...], int | None, int | None, int]:
        """Flatten (B, S, D) inputs to (N, D) for gating."""
        orig_shape = x.shape
        if len(orig_shape) == 3:
            b, s, d = orig_shape
            return x.view(-1, d), orig_shape, b, s, d
        return x, orig_shape, None, None, orig_shape[-1]

    def _entropy_logit_bias(
        self,
        entropy: torch.Tensor | None,
        token_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Map patch entropy to per-expert logit bias of shape (N, E)."""
        if entropy is None:
            return torch.zeros(token_count, self.num_experts, device=device, dtype=dtype)

        if entropy.ndim == 2:
            entropy = entropy.unsqueeze(-1)
        if entropy.ndim != 3 or entropy.shape[-1] != 1:
            raise ValueError("entropy must have shape (B, S) or (B, S, 1)")

        entropy_flat = entropy.reshape(-1, 1)
        if entropy_flat.shape[0] != token_count:
            raise ValueError(
                f"entropy token count {entropy_flat.shape[0]} does not match "
                f"hidden token count {token_count}"
            )
        return self.w_entropy(entropy_flat.to(device=device, dtype=dtype))

    def _gating_logits(
        self,
        x: torch.Tensor,
        entropy: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[int, ...], int | None, int | None, int]:
        """Compute raw gating logits including optional entropy conditioning."""
        x_flat, orig_shape, b, s, d = self._flatten_inputs(x)
        logits = self.wg(x_flat)
        if self.use_entropy_routing:
            logits = logits + self._entropy_logit_bias(
                entropy, x_flat.shape[0], x_flat.device, x_flat.dtype
            )

        if self.noise_enabled and self.training:
            noise_scale = F.softplus(self.w_noise(x_flat))
            noise = torch.randn_like(logits) * noise_scale * self.moe_noise_epsilon
            logits = logits + noise

        return logits, orig_shape, b, s, d

    def route(
        self,
        x: torch.Tensor,
        entropy: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing weights and expert indices.

        Parameters
        ----------
        x : torch.Tensor
            Input hidden representations of shape ``(B, S, D)`` or ``(N, D)``.

        Returns
        -------
        dispatch_weights : torch.Tensor
            Softmax routing weights of shape ``(B, S, K)`` or ``(N, K)``.
        dispatch_indices : torch.Tensor
            Expert indices of shape ``(B, S, K)`` or ``(N, K)``.
        """
        logits, orig_shape, b, s, _ = self._gating_logits(x, entropy=entropy)

        k = self.num_active_experts
        topk_values, topk_indices = torch.topk(logits, k, dim=-1)

        masked_logits = torch.full_like(logits, float("-inf"))
        masked_logits.scatter_(dim=-1, index=topk_indices, src=topk_values)
        dispatch_weights = F.softmax(masked_logits, dim=-1)
        dispatch_weights_selected = dispatch_weights.gather(dim=-1, index=topk_indices)

        if len(orig_shape) == 3:
            assert b is not None and s is not None
            dispatch_weights_selected = dispatch_weights_selected.view(b, s, k)
            topk_indices = topk_indices.view(b, s, k)

        return dispatch_weights_selected, topk_indices

    def _compute_auxiliary_loss(
        self,
        logits: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Shazeer load balancing auxiliary loss.

        Parameters
        ----------
        logits : torch.Tensor
            Raw or noisy logits of shape ``(N, E)``.
        topk_indices : torch.Tensor
            Selected indices of shape ``(N, K)``.

        Returns
        -------
        torch.Tensor
            Load-balancing loss scalar.
        """
        n, e = logits.shape

        # P_i: Fraction of routing weight per expert (softmax weights before filtering)
        probs = F.softmax(logits, dim=-1)
        p_i = probs.mean(dim=0)  # Shape (E,)

        # f_i: Fraction of tokens routed to each expert
        # Create hot mask for selected indices
        mask = torch.zeros_like(logits)
        mask.scatter_(dim=-1, index=topk_indices, value=1.0)
        f_i = mask.mean(dim=0)  # Shape (E,)

        # Shazeer aux loss: E * sum(f_i * p_i)
        loss = e * torch.sum(f_i * p_i)
        return loss * self.moe_aux_loss_coef

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gating routing step computing weights, indices, and auxiliary loss.

        Parameters
        ----------
        x : torch.Tensor
            Input hidden representations of shape ``(B, S, D)`` or ``(N, D)``.
        **kwargs : object
            Optional ``entropy`` tensor of shape ``(B, S, 1)`` or ``(B, S)`` for
            entropy-conditioned expert gating (Patent 3 / Option C).

        Returns
        -------
        dispatch_weights : torch.Tensor
            Routing weights, shape ``(B, S, K)`` or ``(N, K)``.
        dispatch_indices : torch.Tensor
            Selected expert indices, shape ``(B, S, K)`` or ``(N, K)``.
        aux_loss : torch.Tensor
            Scalar load-balancing loss tensor.
        """
        entropy_val = kwargs.get("entropy")
        entropy = entropy_val if isinstance(entropy_val, torch.Tensor) else None

        logits, orig_shape, b, s, _ = self._gating_logits(x, entropy=entropy)

        k = self.num_active_experts
        topk_values, topk_indices = torch.topk(logits, k, dim=-1)

        masked_logits = torch.full_like(logits, float("-inf"))
        masked_logits.scatter_(dim=-1, index=topk_indices, src=topk_values)
        dispatch_weights = F.softmax(masked_logits, dim=-1)
        dispatch_weights_selected = dispatch_weights.gather(dim=-1, index=topk_indices)

        aux_loss = self._compute_auxiliary_loss(logits, topk_indices)

        if len(orig_shape) == 3:
            assert b is not None and s is not None
            dispatch_weights_selected = dispatch_weights_selected.view(b, s, k)
            topk_indices = topk_indices.view(b, s, k)

        return dispatch_weights_selected, topk_indices, aux_loss

    def extra_repr(self) -> str:
        """Return summary of the router attributes."""
        return (
            f"hidden_dim={self.hidden_dim}, num_experts={self.num_experts}, "
            f"num_active_experts={self.num_active_experts}, "
            f"aux_loss_coef={self.moe_aux_loss_coef}"
        )
