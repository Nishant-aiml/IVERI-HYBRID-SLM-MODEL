# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Titans Neural Memory Module.

Implements the long-term neural memory component of the Titans architecture,
performing test-time online parameter updates on a local associative MLP.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.base_config import IVERIConfig, get_base_config
from core.interfaces import BaseMemory
from core.registry import register
from model.titans.lr_gen import MemoryLearningRateGenerator
from model.titans.updater import MemoryUpdater
from utils.validation import validate_shape


@register("titans_memory")
class TitansMemory(BaseMemory):
    """Long-term neural memory using online test-time weight updates.

    Constructs a local multi-layer perceptron (MLP) mapping keys to values.
    MLP parameters act as the long-term memory state, which is updated sequentially
    per step during the forward pass.
    """

    def __init__(self, config_or_dim: IVERIConfig | int) -> None:
        """Initialize the Titans neural memory module.

        Args:
            config_or_dim: Config object or hidden dimension integer.
        """
        super().__init__()
        if isinstance(config_or_dim, int):
            self.config = get_base_config()
            self.hidden_dim = config_or_dim
            self.config.model.hidden_dim = config_or_dim
        else:
            self.config = config_or_dim
            self.hidden_dim = self.config.model.hidden_dim

        self.memory_dim = getattr(self.config.model, "titans_memory_dim", 128)

        # Projections to get queries, keys, and values from the input patch vectors
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        # Output projection for retrieved memory vector
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        # Projection to scale patch_entropy to compute the gate factor for injection
        self.entropy_proj = nn.Linear(1, 1)

        # Learning Rate and Forget Rate generator
        self.lr_generator = MemoryLearningRateGenerator(self.config)

        # Memory Updater
        self.updater = MemoryUpdater(self.config)

        # Define base parameters of the memory MLP (to be expanded per sequence)
        self.base_W1 = nn.Parameter(torch.empty(self.hidden_dim, self.memory_dim))
        self.base_b1 = nn.Parameter(torch.empty(1, self.memory_dim))
        self.base_W2 = nn.Parameter(torch.empty(self.memory_dim, self.hidden_dim))
        self.base_b2 = nn.Parameter(torch.empty(1, self.hidden_dim))

        # Current memory instance state (for persistence across calls if needed)
        self.current_weights: list[torch.Tensor] | None = None
        self.current_surprise: list[torch.Tensor] | None = None

        # Telemetry storage
        self.telemetry: dict[str, object] = {}

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset learnable parameters to standard initialization values."""
        # Initialize projections
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        nn.init.constant_(self.entropy_proj.weight, 1.0)
        nn.init.zeros_(self.entropy_proj.bias)

        # Initialize base memory weights
        std1 = 1.0 / math.sqrt(self.hidden_dim)
        std2 = 1.0 / math.sqrt(self.memory_dim)
        nn.init.normal_(self.base_W1, mean=0.0, std=std1)
        nn.init.zeros_(self.base_b1)
        nn.init.normal_(self.base_W2, mean=0.0, std=std2)
        nn.init.zeros_(self.base_b2)

        self.reset_memory()

    def reset_memory(self) -> None:
        """Clear all stored online memory state."""
        self.current_weights = None
        self.current_surprise = None
        self.telemetry = {}

    def _initialize_memory_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Initialize online memory weights and surprise states for the batch.

        Args:
            batch_size: The batch size B.
            device: Device to initialize parameters on.
            dtype: Floating point type.
        """
        # Expand base weights to shape (B, ...) and clone to separate graphs
        W1 = (
            self.base_W1.to(device=device, dtype=dtype)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .clone()
        )
        b1 = (
            self.base_b1.to(device=device, dtype=dtype)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .clone()
        )
        W2 = (
            self.base_W2.to(device=device, dtype=dtype)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .clone()
        )
        b2 = (
            self.base_b2.to(device=device, dtype=dtype)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .clone()
        )

        self.current_weights = [W1, b1, W2, b2]

        # Surprise state accumulates momentum (initialized to zero)
        s_W1 = torch.zeros_like(W1)
        s_b1 = torch.zeros_like(b1)
        s_W2 = torch.zeros_like(W2)
        s_b2 = torch.zeros_like(b2)

        self.current_surprise = [s_W1, s_b1, s_W2, s_b2]

    def _forward_mlp(
        self,
        x: torch.Tensor,
        weights: list[torch.Tensor],
    ) -> torch.Tensor:
        """Helper to run functional forward pass of the associative memory MLP.

        Args:
            x: Input keys or queries of shape (B, 1, D) or (B, D).
            weights: List of parameter tensors [W1, b1, W2, b2].

        Returns:
            Output value of shape (B, 1, D) or (B, D).
        """
        W1, b1, W2, b2 = weights
        # If input has 3 dimensions (B, 1, D)
        is_3d = x.dim() == 3
        if not is_3d:
            x = x.unsqueeze(1)

        # First layer
        h = torch.bmm(x, W1) + b1
        h = F.silu(h)

        # Second layer
        out = torch.bmm(h, W2) + b2

        if not is_3d:
            out = out.squeeze(1)
        return out

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """Retrieve information from the memory bank.

        Args:
            query: Query tensor of shape (B, P, D) or (B, D).

        Returns:
            Retrieved memory representation of matching shape.
        """
        device = query.device
        dtype = query.dtype
        is_3d = query.dim() == 3

        B = query.size(0)

        if self.current_weights is None or self.current_weights[0].size(0) != B:
            self._initialize_memory_state(B, device, dtype)

        weights = self.current_weights
        assert weights is not None

        if is_3d:
            # Multi-step parallel read (no weight updates occur during read)
            # We can process all P steps in parallel for speed if we don't update weights
            P = query.size(1)
            # Reshape query to run fully batched bmm
            # current_weights has batch-expanded tensors W1 of shape (B, D, H)
            # To process P queries in parallel, we can replicate the weights per patch or reshape
            # Replicating weights:
            W1, b1, W2, b2 = weights
            W1_rep = (
                W1.unsqueeze(1)
                .expand(-1, P, -1, -1)
                .reshape(B * P, self.hidden_dim, self.memory_dim)
            )
            b1_rep = b1.unsqueeze(1).expand(-1, P, -1, -1).reshape(B * P, 1, self.memory_dim)
            W2_rep = (
                W2.unsqueeze(1)
                .expand(-1, P, -1, -1)
                .reshape(B * P, self.memory_dim, self.hidden_dim)
            )
            b2_rep = b2.unsqueeze(1).expand(-1, P, -1, -1).reshape(B * P, 1, self.hidden_dim)

            q_flat = query.view(B * P, 1, self.hidden_dim)
            h = torch.bmm(q_flat, W1_rep) + b1_rep
            h = F.silu(h)
            out_flat = torch.bmm(h, W2_rep) + b2_rep
            out = out_flat.view(B, P, self.hidden_dim)
        else:
            out = self._forward_mlp(query, weights)

        res = self.out_proj(out)
        assert isinstance(res, torch.Tensor)
        return res

    def write(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Store information into the memory bank.

        Args:
            key: Key tensor of shape (B, P, D) or (B, D).
            value: Value tensor to be written of shape (B, P, D) or (B, D).
        """
        device = key.device
        dtype = key.dtype
        is_3d = key.dim() == 3

        B = key.size(0)

        if self.current_weights is None:
            self._initialize_memory_state(B, device, dtype)

        # Ensure not None for type checker
        weights = self.current_weights
        surprise = self.current_surprise
        assert weights is not None
        assert surprise is not None

        # Prepare 3D view
        if not is_3d:
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)

        P = key.size(1)

        create_graph = torch.is_grad_enabled()

        # Metrics for telemetry
        sum_lr = 0.0
        sum_forget = 0.0
        sum_grad_norm = 0.0
        sum_update_mag = 0.0
        lr_vals = []
        forget_vals = []

        # Iterate sequentially over step dimension to perform online updates
        for t in range(P):
            k_t = key[:, t : t + 1, :]
            v_t = value[:, t : t + 1, :]

            # 1. Forward pass on key to predict value
            v_pred = self._forward_mlp(k_t, weights)

            # 2. Compute local reconstruction loss (normalized by dimension)
            loss = 0.5 * (v_pred - v_t).pow(2).mean()

            # 3. Compute gradients of the loss with respect to parameters
            grads = torch.autograd.grad(
                loss,
                weights,
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=True,
            )

            # 4. Generate learning rate and forget rate
            # Use k_t as conditioning context
            lr_t, forget_t = self.lr_generator(k_t)

            # 5. Update weights functionalities
            old_weights = weights
            new_weights, new_surprise = self.updater.update(
                weights, list(grads), surprise, lr_t, forget_t
            )
            self.current_weights = new_weights
            self.current_surprise = new_surprise
            weights = new_weights
            surprise = new_surprise

            # Accumulate telemetry metrics
            sum_lr += lr_t.detach().mean().item()
            sum_forget += forget_t.detach().mean().item()
            lr_vals.extend(lr_t.detach().view(-1).tolist())
            forget_vals.extend(forget_t.detach().view(-1).tolist())

            # Gradient norm
            gnorm = sum([g.detach().pow(2).sum().item() for g in grads if g is not None]) ** 0.5
            sum_grad_norm += gnorm

            # Update magnitude (L2 distance old vs new weights)
            update_norms = []
            for w_new, w_old in zip(weights, old_weights, strict=False):
                update_norms.append((w_new - w_old).detach().pow(2).sum().item())
            mag = sum(update_norms) ** 0.5
            sum_update_mag += mag

        # Calculate final telemetry stats
        if P > 0:
            avg_lr = sum_lr / P
            avg_forget = sum_forget / P
            avg_grad_norm = sum_grad_norm / P
            avg_update_mag = sum_update_mag / P

            # Weight norms
            w_norms = [w.detach().pow(2).sum().item() for w in weights]
            w_norm = sum(w_norms) ** 0.5

            # Memory saturation: ratio of parameters whose absolute value is above 0.5
            total_params = sum([w.numel() for w in weights])
            params_saturated = sum([(w.detach().abs() > 0.5).sum().item() for w in weights])
            saturation = params_saturated / total_params if total_params > 0 else 0.0

            # Generate histograms (simple bin counts)
            lr_hist, _ = torch.histogram(torch.tensor(lr_vals), bins=5, range=(0.0, 0.1))
            forget_hist, _ = torch.histogram(torch.tensor(forget_vals), bins=5, range=(0.0, 0.1))

            self.telemetry = {
                "update_count": B * P,
                "read_count": B * P,
                "write_count": B * P,
                "avg_learning_rate": avg_lr,
                "learning_rate_histogram": lr_hist.tolist(),
                "avg_forget_rate": avg_forget,
                "forget_rate_histogram": forget_hist.tolist(),
                "memory_saturation": saturation,
                "memory_weight_norm": w_norm,
                "average_gradient_norm": avg_grad_norm,
                "average_update_magnitude": avg_update_mag,
            }

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Combined read-process-write step.

        Processes the input sequence: generates queries, keys, and values,
        reads memory, and performs sequential online test-time updates.

        Args:
            x: Input patched representations of shape (B, P, D).
            **kwargs: Extra arguments.

        Returns:
            Retrieved representations of shape (B, P, D).
        """
        validate_shape(x, (-1, -1, self.hidden_dim), name="input x")
        device = x.device
        dtype = x.dtype
        B, P, D = x.shape

        # Initialize fresh memory state for this forward pass (unless explicitly persistent)
        self._initialize_memory_state(B, device, dtype)

        # Ensure not None for type checker
        weights = self.current_weights
        surprise = self.current_surprise
        assert weights is not None
        assert surprise is not None

        # Project input to queries, keys, and values
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        retrieved_list = []

        create_graph = torch.is_grad_enabled()
        online_learning = create_graph

        # Metrics for telemetry
        sum_lr = 0.0
        sum_forget = 0.0
        sum_grad_norm = 0.0
        sum_update_mag = 0.0
        lr_vals = []
        forget_vals = []

        # Run sequential loop for online learning
        for t in range(P):
            q_t = q[:, t : t + 1, :]
            k_t = k[:, t : t + 1, :]
            v_t = v[:, t : t + 1, :]

            # A. Retrieve output using current memory state
            y_t = self._forward_mlp(q_t, weights)
            retrieved_list.append(y_t)

            if not online_learning:
                continue

            # B. Compute loss on key-value association (normalized by dimension)
            v_pred = self._forward_mlp(k_t, weights)
            loss_t = 0.5 * (v_pred - v_t).pow(2).mean()

            # C. Compute gradients of loss with respect to parameters
            grads = torch.autograd.grad(
                loss_t,
                weights,
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=True,
            )

            # D. Dynamic step rates
            # Condition rate generator on key
            lr_t, forget_t = self.lr_generator(k_t)

            # E. Functional update
            old_weights = weights
            new_weights, new_surprise = self.updater.update(
                weights, list(grads), surprise, lr_t, forget_t
            )
            self.current_weights = new_weights
            self.current_surprise = new_surprise
            weights = new_weights
            surprise = new_surprise

            # Accumulate telemetry metrics if requested or for logging
            sum_lr += lr_t.detach().mean().item()
            sum_forget += forget_t.detach().mean().item()
            lr_vals.extend(lr_t.detach().view(-1).tolist())
            forget_vals.extend(forget_t.detach().view(-1).tolist())

            # Gradient norm
            gnorm = sum([g.detach().pow(2).sum().item() for g in grads if g is not None]) ** 0.5
            sum_grad_norm += gnorm

            # Update magnitude (L2 distance old vs new weights)
            update_norms = []
            for w_new, w_old in zip(weights, old_weights, strict=False):
                update_norms.append((w_new - w_old).detach().pow(2).sum().item())
            mag = sum(update_norms) ** 0.5
            sum_update_mag += mag

        # Stack outputs to shape (B, P, D)
        retrieved = torch.cat(retrieved_list, dim=1)
        res_retrieved = self.out_proj(retrieved)
        assert isinstance(res_retrieved, torch.Tensor)

        # Calculate final telemetry stats
        if P > 0:
            if online_learning:
                avg_lr = sum_lr / P
                avg_forget = sum_forget / P
                avg_grad_norm = sum_grad_norm / P
                avg_update_mag = sum_update_mag / P

                # Weight norms
                w_norms = [w.detach().pow(2).sum().item() for w in weights]
                w_norm = sum(w_norms) ** 0.5

                # Memory saturation: ratio of parameters whose absolute value is above 0.5
                total_params = sum([w.numel() for w in weights])
                params_saturated = sum([(w.detach().abs() > 0.5).sum().item() for w in weights])
                saturation = params_saturated / total_params if total_params > 0 else 0.0

                # Generate histograms (simple bin counts)
                lr_hist, _ = torch.histogram(torch.tensor(lr_vals), bins=5, range=(0.0, 0.1))
                forget_hist, _ = torch.histogram(torch.tensor(forget_vals), bins=5, range=(0.0, 0.1))

                self.telemetry = {
                    "update_count": B * P,
                    "read_count": B * P,
                    "write_count": B * P,
                    "avg_learning_rate": avg_lr,
                    "learning_rate_histogram": lr_hist.tolist(),
                    "avg_forget_rate": avg_forget,
                    "forget_rate_histogram": forget_hist.tolist(),
                    "memory_saturation": saturation,
                    "memory_weight_norm": w_norm,
                    "average_gradient_norm": avg_grad_norm,
                    "average_update_magnitude": avg_update_mag,
                }
            else:
                w_norms = [w.detach().pow(2).sum().item() for w in weights]
                w_norm = sum(w_norms) ** 0.5
                self.telemetry = {
                    "update_count": 0,
                    "read_count": B * P,
                    "write_count": 0,
                    "avg_learning_rate": 0.0,
                    "learning_rate_histogram": [0, 0, 0, 0, 0],
                    "avg_forget_rate": 0.0,
                    "forget_rate_histogram": [0, 0, 0, 0, 0],
                    "memory_saturation": 0.0,
                    "memory_weight_norm": w_norm,
                    "average_gradient_norm": 0.0,
                    "average_update_magnitude": 0.0,
                }

        return res_retrieved

    def forward_with_injection(self, x: torch.Tensor, patch_entropy: torch.Tensor) -> torch.Tensor:
        """Production path: online forward (read/update/write) + entropy-gated injection.

        Combines ``forward()`` sequential memory updates with Option C entropy gating
        used by ``inject()``. Intended for backbone training and production integration.

        Args:
            x: Patch representations of shape (B, P, D).
            patch_entropy: Entropy tensor of shape (B, P, 1) or (B, P).

        Returns:
            Gated sequence representation tensor of shape (B, P, D).
        """
        validate_shape(x, (-1, -1, self.hidden_dim), name="input x")
        if patch_entropy.dim() == 2:
            patch_entropy = patch_entropy.unsqueeze(-1)
        validate_shape(patch_entropy, (-1, -1, 1), name="patch_entropy")

        retrieved = self.forward(x)
        gate = torch.sigmoid(self.entropy_proj(patch_entropy))
        out = x + retrieved * gate
        assert isinstance(out, torch.Tensor)
        return out

    def inject(self, x: torch.Tensor, patch_entropy: torch.Tensor) -> torch.Tensor:
        """Inject global memory representations into patch-level sequence representations.

        Implements Option C Entropy-Gated Injection:
        x = x + retrieved * gate
        where gate = sigmoid(entropy_proj(patch_entropy)).

        Args:
            x: Patch representations of shape (B, P, D).
            patch_entropy: Entropy tensor of shape (B, P, 1) or (B, P).

        Returns:
            Gated sequence representation tensor of shape (B, P, D).
        """
        validate_shape(x, (-1, -1, self.hidden_dim), name="input x")

        # Consumes official entropy tensor
        if patch_entropy.dim() == 2:
            patch_entropy = patch_entropy.unsqueeze(-1)
        validate_shape(patch_entropy, (-1, -1, 1), name="patch_entropy")

        # Read from memory to get retrieved sequence representation
        retrieved = self.read(x)

        # Compute entropy gate
        gate = torch.sigmoid(self.entropy_proj(patch_entropy))

        # Memory injection gated by patch entropy
        out = x + retrieved * gate
        assert isinstance(out, torch.Tensor)
        return out
