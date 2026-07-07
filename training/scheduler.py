# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Learning rate scheduler infrastructure for IVERI CORE.

Provides a unified step-based learning rate scheduler supporting 7 core strategies
(constant, linear, cosine, polynomial, step, exponential, and cosine warmup + decay)
and a SchedulerFactory for configuration-driven initialization.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from configs.base_config import IVERIConfig


class IVERIScheduler(LRScheduler):
    """Unified Step-Based Learning Rate Scheduler for IVERI CORE.

    Supports Constant, Linear, Cosine, Polynomial, Step, and Exponential decay,
    each with optional linear warmup.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        scheduler_type: str = "cosine",
        learning_rate: float = 3e-4,
        min_lr: float = 3e-5,
        warmup_steps: int = 1000,
        max_steps: int = 50000,
        power: float = 1.0,
        step_size: int = 1000,
        gamma: float = 0.1,
        initial_lr: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """Initialize the scheduler.

        Args:
            optimizer: PyTorch optimizer instance.
            scheduler_type: Scheduling strategy, one of 'constant', 'linear',
                'cosine', 'polynomial', 'step', 'exponential'.
            learning_rate: Peak/Target learning rate (applied after warmup).
            min_lr: Floor learning rate.
            warmup_steps: Number of linear warmup steps.
            max_steps: Total steps for decay duration.
            power: Power coefficient for polynomial decay.
            step_size: Step interval for step decay.
            gamma: Decay rate coefficient for step and exponential decay.
            initial_lr: Base learning rate starting value for warmup.
            last_epoch: The index of last epoch (step). Default: -1.
        """
        self.scheduler_type = scheduler_type.lower()
        self.peak_lr = learning_rate
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.power = power
        self.step_size = step_size
        self.gamma = gamma
        self.initial_lr = initial_lr

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float | torch.Tensor]:
        """Compute learning rate for the current step (self.last_epoch).

        Returns:
            List of learning rates for each parameter group in the optimizer.
        """
        step = self.last_epoch

        # If step is negative (pre-training initialization), return peak LR
        if step < 0:
            return [self.peak_lr for _ in self.base_lrs]

        # 1. Warmup Phase
        if step < self.warmup_steps and self.warmup_steps > 0:
            ratio = float(step) / float(self.warmup_steps)
            lr = self.initial_lr + (self.peak_lr - self.initial_lr) * ratio
            return [lr for _ in self.base_lrs]

        # 2. Post-Warmup / Decay Phase
        t_curr = step - self.warmup_steps
        t_decay = max(1, self.max_steps - self.warmup_steps)

        if self.scheduler_type == "constant":
            lr = self.peak_lr

        elif self.scheduler_type == "cosine":
            if t_curr >= t_decay:
                lr = self.min_lr
            else:
                progress = float(t_curr) / float(t_decay)
                cos_val = math.cos(math.pi * progress)
                lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1.0 + cos_val)

        elif self.scheduler_type == "linear":
            if t_curr >= t_decay:
                lr = self.min_lr
            else:
                progress = float(t_curr) / float(t_decay)
                lr = self.peak_lr - (self.peak_lr - self.min_lr) * progress

        elif self.scheduler_type == "polynomial":
            if t_curr >= t_decay:
                lr = self.min_lr
            else:
                progress = float(t_curr) / float(t_decay)
                lr = self.min_lr + (self.peak_lr - self.min_lr) * ((1.0 - progress) ** self.power)

        elif self.scheduler_type == "step":
            decay_count = t_curr // self.step_size
            lr = max(self.min_lr, self.peak_lr * (self.gamma**decay_count))

        elif self.scheduler_type == "exponential":
            lr = max(self.min_lr, self.peak_lr * (self.gamma**t_curr))

        else:
            # Fallback
            lr = self.peak_lr

        # Verify learning rates remain valid and positive
        if math.isnan(lr) or math.isinf(lr):
            lr = self.min_lr
        lr = max(0.0, lr)

        return [lr for _ in self.base_lrs]

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load scheduler state and immediately apply learning rates to the optimizer.

        Args:
            state_dict: Serialized scheduler state dictionary.
        """
        super().load_state_dict(state_dict)
        self.step(self.last_epoch)


class SchedulerFactory:
    """Factory to build learning rate schedulers from configuration schema."""

    @staticmethod
    def create_scheduler(
        optimizer: Optimizer,
        config: IVERIConfig,
        warmup_ratio: float | None = None,
        initial_lr: float = 0.0,
    ) -> IVERIScheduler:
        """Instantiate and validate a scheduler from the training config.

        Args:
            optimizer: The optimizer to schedule.
            config: Full configuration dataclass object.
            warmup_ratio: Optional ratio of max_steps for warmup calculation
                (overrides config warmup_steps).
            initial_lr: Initial learning rate at start of warmup.

        Returns:
            IVERIScheduler instance.
        """
        warmup_steps = config.training.warmup_steps
        if warmup_ratio is not None:
            warmup_steps = int(config.training.max_steps * warmup_ratio)

        # Validate warmup boundaries
        if warmup_steps >= config.training.max_steps:
            warmup_steps = max(0, config.training.max_steps - 1)

        return IVERIScheduler(
            optimizer=optimizer,
            scheduler_type=config.training.scheduler_type,
            learning_rate=config.training.learning_rate,
            min_lr=config.training.min_lr,
            warmup_steps=warmup_steps,
            max_steps=config.training.max_steps,
            power=config.training.scheduler_power,
            step_size=config.training.scheduler_step_size,
            gamma=config.training.scheduler_gamma,
            initial_lr=initial_lr,
        )
