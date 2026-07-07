# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Unit and integration tests for IVERI CORE learning rate scheduler (Phase 2.3)."""

from __future__ import annotations

import math

import pytest
import torch

from configs.base_config import get_base_config
from core.exceptions import ConfigError
from training.scheduler import IVERIScheduler, SchedulerFactory


class DummyOptimizer(torch.optim.Optimizer):
    """Dummy optimizer to satisfy scheduler parameter groups."""

    def __init__(self, lr: float = 3e-4) -> None:
        defaults = {"lr": lr}
        # Fake parameter to satisfy base constructor
        self.param = torch.nn.Parameter(torch.zeros(1))
        super().__init__([self.param], defaults)


# --- Scheduler Configuration and Factory Tests ---


def test_scheduler_factory_defaults() -> None:
    """Verify SchedulerFactory constructs scheduler with correct default overrides."""
    opt = DummyOptimizer()
    cfg = get_base_config()
    scheduler = SchedulerFactory.create_scheduler(opt, cfg)

    assert isinstance(scheduler, IVERIScheduler)
    assert scheduler.peak_lr == cfg.training.learning_rate
    assert scheduler.min_lr == cfg.training.min_lr
    assert scheduler.warmup_steps == cfg.training.warmup_steps
    assert scheduler.max_steps == cfg.training.max_steps


def test_scheduler_factory_ratio_override() -> None:
    """Verify SchedulerFactory supports warmup_ratio override."""
    opt = DummyOptimizer()
    cfg = get_base_config()
    cfg.training.max_steps = 10000
    scheduler = SchedulerFactory.create_scheduler(opt, cfg, warmup_ratio=0.1)

    assert scheduler.warmup_steps == 1000


def test_invalid_scheduler_config_type() -> None:
    """Verify config system rejects unsupported scheduler strategies."""
    with pytest.raises(ConfigError):
        get_base_config(training={"scheduler_type": "unknown"})


# --- Strategy Progression Tests ---


def test_constant_lr_progression() -> None:
    """Verify constant scheduling strategy stays at target learning rate."""
    opt = DummyOptimizer(lr=3e-4)
    cfg = get_base_config(training={"scheduler_type": "constant", "warmup_steps": 0})
    scheduler = SchedulerFactory.create_scheduler(opt, cfg)

    # Step through sequence
    for _ in range(5):
        scheduler.step()
        assert math.isclose(opt.param_groups[0]["lr"], 3e-4)


def test_linear_warmup_and_cosine_decay() -> None:
    """Verify correct linear warmup ramp followed by cosine decay progression."""
    peak_lr = 1.0
    min_lr = 0.1
    warmup_steps = 10
    max_steps = 100

    opt = DummyOptimizer(lr=peak_lr)
    scheduler = IVERIScheduler(
        optimizer=opt,
        scheduler_type="cosine",
        learning_rate=peak_lr,
        min_lr=min_lr,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        initial_lr=0.0,
    )

    # 1. Warmup progression: linear step-based scaling
    # Step 0 (initial_lr)
    assert math.isclose(opt.param_groups[0]["lr"], 0.0)

    # Step 5 (halfway through warmup)
    scheduler.step()  # step=1
    scheduler.step()  # step=2
    scheduler.step()  # step=3
    scheduler.step()  # step=4
    scheduler.step()  # step=5
    assert math.isclose(opt.param_groups[0]["lr"], 0.5, abs_tol=1e-5)

    # Step 10 (peak_lr reached)
    for _ in range(5):
        scheduler.step()
    assert math.isclose(opt.param_groups[0]["lr"], 1.0, abs_tol=1e-5)

    # 2. Cosine Decay Phase (halfway step 55 -> progress 0.5)
    # Cosine(progress = 0.5) = Cosine(pi/2) = 0.0
    # Expected: min_lr + 0.5 * (peak_lr - min_lr) * 1.0 = 0.1 + 0.45 = 0.55
    for _ in range(45):
        scheduler.step()
    assert math.isclose(opt.param_groups[0]["lr"], 0.55, abs_tol=1e-5)

    # Step 100+ (min_lr limit reached and clamped)
    for _ in range(50):
        scheduler.step()
    assert math.isclose(opt.param_groups[0]["lr"], 0.1, abs_tol=1e-5)


def test_linear_decay_progression() -> None:
    """Verify linear decay progress matches slope exactly."""
    peak = 1.0
    floor = 0.0
    opt = DummyOptimizer(lr=peak)
    scheduler = IVERIScheduler(
        optimizer=opt,
        scheduler_type="linear",
        learning_rate=peak,
        min_lr=floor,
        warmup_steps=0,
        max_steps=10,
    )

    # Step 0 (Peak)
    assert math.isclose(opt.param_groups[0]["lr"], 1.0)

    # Step 5 (Halfway)
    for _ in range(5):
        scheduler.step()
    assert math.isclose(opt.param_groups[0]["lr"], 0.5, abs_tol=1e-5)

    # Step 10 (Floor)
    for _ in range(5):
        scheduler.step()
    assert math.isclose(opt.param_groups[0]["lr"], 0.0, abs_tol=1e-5)


def test_polynomial_decay_progression() -> None:
    """Verify polynomial decay values with power constraints."""
    opt = DummyOptimizer(lr=1.0)
    scheduler = IVERIScheduler(
        optimizer=opt,
        scheduler_type="polynomial",
        learning_rate=1.0,
        min_lr=0.0,
        warmup_steps=0,
        max_steps=10,
        power=2.0,
    )

    # Step 5 (Halfway, progress = 0.5)
    # Expected: (1 - 0.5) ^ 2.0 = 0.25
    for _ in range(5):
        scheduler.step()
    assert math.isclose(opt.param_groups[0]["lr"], 0.25, abs_tol=1e-5)


def test_step_decay_intervals() -> None:
    """Verify step decay decreases at fixed step count intervals."""
    opt = DummyOptimizer(lr=1.0)
    scheduler = IVERIScheduler(
        optimizer=opt,
        scheduler_type="step",
        learning_rate=1.0,
        min_lr=0.01,
        warmup_steps=0,
        step_size=10,
        gamma=0.5,
    )

    # Steps 0 to 9: 1.0
    for _ in range(9):
        assert math.isclose(opt.param_groups[0]["lr"], 1.0)
        scheduler.step()

    # Step 10: decays to 0.5
    scheduler.step()
    assert math.isclose(opt.param_groups[0]["lr"], 0.5)


def test_exponential_decay() -> None:
    """Verify exponential decay progression follows continuous scaling."""
    opt = DummyOptimizer(lr=1.0)
    scheduler = IVERIScheduler(
        optimizer=opt,
        scheduler_type="exponential",
        learning_rate=1.0,
        min_lr=0.01,
        warmup_steps=0,
        gamma=0.9,
    )

    # Step 1: 0.9
    scheduler.step()
    assert math.isclose(opt.param_groups[0]["lr"], 0.9)

    # Step 2: 0.81
    scheduler.step()
    assert math.isclose(opt.param_groups[0]["lr"], 0.81)


# --- Serialization & State Restoration Tests ---


def test_scheduler_state_dict_roundtrip() -> None:
    """Verify saving and restoring scheduler state dictionary."""
    opt1 = DummyOptimizer(lr=1.0)
    sched1 = IVERIScheduler(opt1, scheduler_type="cosine", warmup_steps=5, max_steps=10)

    # Step sched1 through part of warmup
    sched1.step()
    sched1.step()

    # Capture state
    state = sched1.state_dict()

    # Create new optimizer and scheduler, restore state
    opt2 = DummyOptimizer(lr=1.0)
    sched2 = IVERIScheduler(opt2, scheduler_type="cosine", warmup_steps=5, max_steps=10)

    sched2.load_state_dict(state)

    # Verify steps sync (last_epoch) and rates align
    assert sched2.last_epoch == sched1.last_epoch
    assert math.isclose(opt2.param_groups[0]["lr"], opt1.param_groups[0]["lr"])


# --- Stress & Long Horizon Tests ---


def test_scheduler_zero_warmup() -> None:
    """Verify scheduler functions correctly with zero warmup steps."""
    opt = DummyOptimizer(lr=1.0)
    _scheduler = IVERIScheduler(
        optimizer=opt,
        scheduler_type="cosine",
        learning_rate=1.0,
        min_lr=0.0,
        warmup_steps=0,
        max_steps=10,
    )

    # First step should begin decay immediately (no 0.0 warmup)
    assert opt.param_groups[0]["lr"] > 0.9


def test_scheduler_long_horizon_simulation() -> None:
    """Simulate a long training horizon (100,000 steps) to verify stability."""
    opt = DummyOptimizer(lr=3e-4)
    scheduler = IVERIScheduler(
        optimizer=opt,
        scheduler_type="cosine",
        learning_rate=3e-4,
        min_lr=3e-5,
        warmup_steps=5000,
        max_steps=100000,
    )

    # Step 100,000 times
    for _ in range(100005):
        scheduler.step()

    # Verify no NaNs, Infs, or negative rates, and it reaches min_lr
    lr = opt.param_groups[0]["lr"]
    assert not math.isnan(lr)
    assert not math.isinf(lr)
    assert lr >= 3e-5
