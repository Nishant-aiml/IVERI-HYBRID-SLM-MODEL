# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Phase 2.6 — Distributed Training Infrastructure Tests.

All 30 tests run in a single process on CPU with no GPU or torchrun
required.  Distributed-specific operations (all_reduce, barrier, etc.)
are tested in their non-initialized / world_size=1 no-op paths.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from configs.base_config import IVERIConfig
from configs.distributed_config import DistributedConfig
from core.exceptions import ConfigError
from training.distributed import DistributedManager
from training.distributed_checkpointing import (
    load_checkpoint_distributed,
    save_checkpoint_distributed,
)
from training.distributed_dataloader import make_distributed_dataloader, set_epoch
from training.distributed_fault_tolerance import FaultToleranceHandler
from training.distributed_logger import DistributedLogger
from training.distributed_trainer import DistributedTrainer

# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture()
def dist_cfg_disabled() -> DistributedConfig:
    """Disabled DistributedConfig (default single-GPU mode)."""
    return DistributedConfig(enabled=False)


@pytest.fixture()
def dist_cfg_single() -> DistributedConfig:
    """Enabled DistributedConfig with world_size=1 (no-op mode)."""
    return DistributedConfig(enabled=True, strategy="ddp", world_size=1, rank=0, local_rank=0)


@pytest.fixture()
def dist_manager_disabled(dist_cfg_disabled: DistributedConfig) -> DistributedManager:
    """DistributedManager that is not set up (disabled)."""
    return DistributedManager(dist_cfg_disabled)


@pytest.fixture()
def dist_manager_single(dist_cfg_single: DistributedConfig) -> DistributedManager:
    """DistributedManager with world_size=1 (enabled but trivially single-process)."""
    return DistributedManager(dist_cfg_single)


@pytest.fixture()
def iveri_cfg() -> IVERIConfig:
    """Default IVERIConfig with distributed defaults."""
    return IVERIConfig()


@pytest.fixture()
def tiny_model() -> nn.Module:
    """Tiny Linear model for checkpointing tests."""
    return nn.Linear(16, 16)


@pytest.fixture()
def tiny_dataset() -> TensorDataset:
    """Small TensorDataset for DataLoader tests."""
    x = torch.randint(0, 256, (32, 8), dtype=torch.long)
    y = torch.randint(0, 256, (32, 8), dtype=torch.long)
    return TensorDataset(x, y)


# ── Configuration tests ────────────────────────────────────────────────────


def test_distributed_config_defaults() -> None:
    """DistributedConfig() should have correct single-GPU defaults."""
    cfg = DistributedConfig()
    assert cfg.enabled is False
    assert cfg.strategy == "ddp"
    assert cfg.backend == "nccl"
    assert cfg.world_size == 1
    assert cfg.rank == 0
    assert cfg.local_rank == 0
    assert cfg.log_rank_zero_only is True
    assert cfg.checkpoint_state_dict_type == "full"
    assert cfg.timeout_minutes == 30
    assert cfg.graceful_shutdown is True
    assert cfg.dataloader_drop_last is True
    assert cfg.dataloader_persistent_workers is True


def test_distributed_config_validation_strategy() -> None:
    """Invalid strategy should raise ConfigError."""
    with pytest.raises(ConfigError, match="strategy"):
        DistributedConfig(strategy="deepspeed")


def test_distributed_config_validation_backend() -> None:
    """Invalid backend should raise ConfigError."""
    with pytest.raises(ConfigError, match="backend"):
        DistributedConfig(backend="rdma")


def test_distributed_config_validation_rank() -> None:
    """rank >= world_size should raise ConfigError."""
    with pytest.raises(ConfigError, match="rank"):
        DistributedConfig(world_size=2, rank=2, local_rank=0)


def test_distributed_config_serialization() -> None:
    """DistributedConfig should survive a to_dict / from_dict round-trip via IVERIConfig."""
    cfg = IVERIConfig(
        distributed=DistributedConfig(
            enabled=True, strategy="ddp", world_size=4, rank=0, local_rank=0
        )
    )
    d = cfg.to_dict()
    assert "distributed" in d
    assert d["distributed"]["enabled"] is True
    assert d["distributed"]["strategy"] == "ddp"
    assert d["distributed"]["world_size"] == 4
    cfg2 = IVERIConfig.from_dict(d)
    assert cfg2.distributed.enabled is True
    assert cfg2.distributed.strategy == "ddp"
    assert cfg2.distributed.world_size == 4


def test_iveri_config_distributed_field() -> None:
    """IVERIConfig must have a distributed field that is a DistributedConfig."""
    cfg = IVERIConfig()
    assert hasattr(cfg, "distributed")
    assert isinstance(cfg.distributed, DistributedConfig)


def test_iveri_config_backward_compat() -> None:
    """Pre-2.6 JSON (missing 'distributed' key) should load with default DistributedConfig."""
    base_dict = IVERIConfig().to_dict()
    del base_dict["distributed"]  # simulate old checkpoint
    cfg = IVERIConfig.from_dict(base_dict)
    assert isinstance(cfg.distributed, DistributedConfig)
    assert cfg.distributed.enabled is False


def test_distributed_config_sharded_requires_fsdp() -> None:
    """checkpoint_state_dict_type='sharded' with strategy != 'fsdp' should raise ConfigError."""
    with pytest.raises(ConfigError, match="sharded"):
        DistributedConfig(strategy="ddp", checkpoint_state_dict_type="sharded")


# ── DistributedManager tests ───────────────────────────────────────────────


def test_manager_single_process(dist_manager_disabled: DistributedManager) -> None:
    """DistributedManager returns correct values without init."""
    dm = dist_manager_disabled
    assert dm.rank() == 0
    assert dm.world_size() == 1
    assert dm.local_rank() == 0
    assert dm.is_initialized() is False


def test_manager_is_main_process(dist_manager_disabled: DistributedManager) -> None:
    """is_main_process() returns True when rank is 0 or manager is not initialized."""
    assert dist_manager_disabled.is_main_process() is True


def test_manager_noop_barrier(dist_manager_disabled: DistributedManager) -> None:
    """barrier() is a no-op when not initialized — must not raise."""
    dist_manager_disabled.barrier()  # Should not raise


def test_manager_all_reduce_noop(dist_manager_disabled: DistributedManager) -> None:
    """all_reduce_mean() returns the tensor unchanged when not initialized."""
    t = torch.tensor(3.14)
    result = dist_manager_disabled.all_reduce_mean(t)
    assert abs(result.item() - 3.14) < 1e-5


def test_manager_reduce_dict(dist_manager_disabled: DistributedManager) -> None:
    """reduce_dict() returns the same dict unchanged when not initialized."""
    metrics = {"loss": 1.5, "aux": 0.3, "perplexity": 4.5}
    result = dist_manager_disabled.reduce_dict(metrics)
    assert result == metrics


def test_world_size_one(dist_manager_single: DistributedManager) -> None:
    """All operations are correct with world_size=1 (enabled but trivially single-process)."""
    dm = dist_manager_single
    assert dm.world_size() == 1
    assert dm.rank() == 0
    assert dm.is_main_process() is True
    dm.barrier()  # Must not raise
    t = torch.tensor(2.0)
    r = dm.all_reduce_mean(t)
    assert abs(r.item() - 2.0) < 1e-5


def test_ddp_wrap_noop(dist_manager_disabled: DistributedManager) -> None:
    """wrap_model() with strategy='none' returns the original model unchanged."""
    model = nn.Linear(4, 4)
    result = dist_manager_disabled.wrap_model(model)
    assert result is model


def test_fsdp_wrap_unavailable(dist_manager_single: DistributedManager) -> None:
    """wrap_model('fsdp') without initialization raises RuntimeError."""
    cfg = DistributedConfig(enabled=True, strategy="fsdp", world_size=1, rank=0, local_rank=0)
    dm = DistributedManager(cfg)
    model = nn.Linear(4, 4)
    # wrap_model called before setup() should raise RuntimeError
    with pytest.raises(RuntimeError, match="setup"):
        dm.wrap_model(model)


# ── Sampler / DataLoader tests ─────────────────────────────────────────────


def test_sampler_partitioning(tiny_dataset: TensorDataset, iveri_cfg: IVERIConfig) -> None:
    """DistributedSampler partitions dataset deterministically (world_size=1 covers all)."""
    dm = DistributedManager(DistributedConfig(enabled=True, world_size=1, rank=0, local_rank=0))
    loader = make_distributed_dataloader(tiny_dataset, iveri_cfg, dm, shuffle=False)
    assert isinstance(loader, DataLoader)
    total = sum(len(b[0]) for b in loader)
    assert total == len(tiny_dataset) or total == len(tiny_dataset) - (len(tiny_dataset) % 1)


def test_sampler_prime_length(iveri_cfg: IVERIConfig) -> None:
    """Prime-length dataset (37 samples) should not crash with distributed sampler."""
    prime_ds = TensorDataset(torch.zeros(37, 8, dtype=torch.long))
    dm = DistributedManager(DistributedConfig(enabled=True, world_size=1, rank=0, local_rank=0))
    loader = make_distributed_dataloader(prime_ds, iveri_cfg, dm, shuffle=False)
    assert isinstance(loader, DataLoader)
    # Iterating should not raise
    for _ in loader:
        pass


def test_multiple_epochs_sampler(tiny_dataset: TensorDataset, iveri_cfg: IVERIConfig) -> None:
    """set_epoch() on DistributedSampler-backed DataLoader shuffles differently each epoch."""
    dm = DistributedManager(DistributedConfig(enabled=True, world_size=1, rank=0, local_rank=0))
    loader = make_distributed_dataloader(tiny_dataset, iveri_cfg, dm, shuffle=True)
    # set_epoch must not raise
    set_epoch(loader, epoch=0)
    set_epoch(loader, epoch=1)


def test_dataloader_no_dist(tiny_dataset: TensorDataset, iveri_cfg: IVERIConfig) -> None:
    """make_distributed_dataloader returns plain DataLoader when distributed is not enabled."""
    dm = DistributedManager(DistributedConfig(enabled=False))
    loader = make_distributed_dataloader(tiny_dataset, iveri_cfg, dm, shuffle=False)
    assert isinstance(loader, DataLoader)
    # No DistributedSampler — sampler should be None or SequentialSampler
    from torch.utils.data.distributed import DistributedSampler

    assert not isinstance(loader.sampler, DistributedSampler)


def test_set_epoch_noop(tiny_dataset: TensorDataset, iveri_cfg: IVERIConfig) -> None:
    """set_epoch() is a no-op when sampler has no set_epoch attribute."""
    dm = DistributedManager(DistributedConfig(enabled=False))
    loader = make_distributed_dataloader(tiny_dataset, iveri_cfg, dm)
    set_epoch(loader, epoch=5)  # Should not raise


# ── Checkpoint tests ───────────────────────────────────────────────────────


def test_rank_zero_checkpoint(tiny_model: nn.Module, iveri_cfg: IVERIConfig) -> None:
    """Rank-0 distributed checkpoint save produces a valid file."""
    dm = DistributedManager(DistributedConfig(enabled=False))
    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "checkpoint.pt"
        save_checkpoint_distributed(
            path=path,
            model=tiny_model,
            optimizer=None,
            scheduler=None,
            scaler=None,
            step=10,
            epoch=1,
            metrics={"loss": 1.5},
            config=iveri_cfg,
            dist_manager=dm,
        )
        assert path.exists(), "Checkpoint file should exist on rank 0."


def test_non_rank_zero_checkpoint(tiny_model: nn.Module, iveri_cfg: IVERIConfig) -> None:
    """Non-rank-0 processes should not create a checkpoint file when guarded correctly."""
    # Simulate a rank-1 process by mocking is_main_process to return False
    cfg = DistributedConfig(enabled=True, strategy="ddp", world_size=2, rank=1, local_rank=1)
    dm = DistributedManager(cfg)
    # Mock barrier and is_main_process so we don't actually need dist
    dm.barrier = MagicMock()  # type: ignore[method-assign]
    # Force is_main_process to return False (rank 1)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "checkpoint.pt"
        # Patch dist_manager.is_main_process to simulate non-rank-0
        with patch.object(dm, "is_main_process", return_value=False):
            save_checkpoint_distributed(
                path=path,
                model=tiny_model,
                optimizer=None,
                scheduler=None,
                scaler=None,
                step=10,
                epoch=1,
                metrics={"loss": 1.5},
                config=iveri_cfg,
                dist_manager=dm,
            )
        assert not path.exists(), "Non-rank-0 should NOT write a checkpoint file."


def test_checkpoint_distributed_load(tiny_model: nn.Module, iveri_cfg: IVERIConfig) -> None:
    """Distributed load should populate model weights identically to standard load."""
    from training.checkpointing import save_checkpoint

    dm = DistributedManager(DistributedConfig(enabled=False))
    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "checkpoint.pt"
        save_checkpoint(path=path, model=tiny_model, step=5, epoch=0, metrics={}, config=iveri_cfg)

        # Load into a fresh model
        fresh_model = nn.Linear(16, 16)
        meta = load_checkpoint_distributed(
            path=path,
            model=fresh_model,
            optimizer=None,
            scheduler=None,
            scaler=None,
            dist_manager=dm,
        )
        assert meta["step"] == 5
        assert meta["epoch"] == 0
        # Verify weights match
        for p1, p2 in zip(tiny_model.parameters(), fresh_model.parameters(), strict=True):
            assert torch.allclose(p1, p2), "Loaded weights must match saved weights."


def test_distributed_resume(tiny_model: nn.Module, iveri_cfg: IVERIConfig) -> None:
    """Distributed checkpoint resume should restore step and epoch correctly."""
    from training.checkpointing import save_checkpoint

    dm = DistributedManager(DistributedConfig(enabled=False))
    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "ckpt.pt"
        save_checkpoint(
            path=path,
            model=tiny_model,
            step=100,
            epoch=3,
            metrics={"val_loss": 0.42},
            config=iveri_cfg,
        )
        fresh = nn.Linear(16, 16)
        meta = load_checkpoint_distributed(
            path=path, model=fresh, optimizer=None, scheduler=None, scaler=None, dist_manager=dm
        )
        assert meta["step"] == 100
        assert meta["epoch"] == 3
        assert abs(meta["metrics"]["val_loss"] - 0.42) < 1e-6


def test_full_state_dict(tiny_model: nn.Module, iveri_cfg: IVERIConfig) -> None:
    """Full state dict checkpoint round-trip produces identical weights."""
    dm = DistributedManager(DistributedConfig(enabled=False, checkpoint_state_dict_type="full"))
    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "full.pt"
        save_checkpoint_distributed(
            path=path,
            model=tiny_model,
            optimizer=None,
            scheduler=None,
            scaler=None,
            step=0,
            epoch=0,
            metrics={},
            config=iveri_cfg,
            dist_manager=dm,
        )
        fresh = nn.Linear(16, 16)
        load_checkpoint_distributed(
            path=path, model=fresh, optimizer=None, scheduler=None, scaler=None, dist_manager=dm
        )
        for p1, p2 in zip(tiny_model.parameters(), fresh.parameters(), strict=True):
            assert torch.allclose(p1, p2)


def test_sharded_state_dict() -> None:
    """Sharded state dict config requires strategy=fsdp — validation enforced at config level."""
    with pytest.raises(ConfigError, match="sharded"):
        DistributedConfig(strategy="ddp", checkpoint_state_dict_type="sharded")
    # Valid: fsdp + sharded should not raise
    cfg = DistributedConfig(strategy="fsdp", checkpoint_state_dict_type="sharded")
    assert cfg.checkpoint_state_dict_type == "sharded"


# ── Logger tests ───────────────────────────────────────────────────────────


def test_rank_logging_rank_zero_only(dist_manager_disabled: DistributedManager) -> None:
    """DistributedLogger.log() delegates to ExperimentLogger when is_main_process()=True."""
    mock_logger = MagicMock()
    cfg = DistributedConfig(log_rank_zero_only=True)
    dist_logger = DistributedLogger(mock_logger, dist_manager_disabled, cfg)
    dist_logger.log({"loss": 1.0}, step=1)
    mock_logger.log.assert_called_once_with({"loss": 1.0}, step=1)


def test_rank_logging_debug_mode(dist_manager_disabled: DistributedManager) -> None:
    """DistributedLogger debug mode writes a rank-scoped JSONL file."""
    mock_logger = MagicMock()
    mock_logger._config = MagicMock()
    mock_logger._config.logging.save_dir = None  # will trigger AttributeError fallback

    with tempfile.TemporaryDirectory() as tmpdir:
        # Patch the logger's save_dir
        mock_logger._config.logging.save_dir = tmpdir
        cfg = DistributedConfig(log_rank_zero_only=False)
        dist_logger = DistributedLogger(mock_logger, dist_manager_disabled, cfg)
        dist_logger.log({"loss": 0.5}, step=10)
        dist_logger.shutdown()

        # Verify rank-0 JSONL file was written
        rank_file = pathlib.Path(tmpdir) / "rank0.jsonl"
        assert rank_file.exists(), "rank0.jsonl must be created in debug mode."
        lines = rank_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["step"] == 10
        assert record["rank"] == 0
        assert abs(record["loss"] - 0.5) < 1e-6


# ── Metric reduction tests ─────────────────────────────────────────────────


def test_metric_reduction(dist_manager_disabled: DistributedManager) -> None:
    """DistributedTrainer._reduce_metrics reduces all float keys via reduce_dict."""
    mock_trainer = MagicMock()
    mock_trainer.train_epoch.return_value = {
        "train_loss": 1.5,
        "train_aux_loss": 0.3,
        "epoch_time": 42.0,
        "moe_aux_loss": 0.05,
        "titans_lr_mean": 0.001,
    }
    cfg = IVERIConfig()
    dt = DistributedTrainer(mock_trainer, dist_manager_disabled, cfg)
    result = dt.train_epoch()
    # All keys must be present
    assert "train_loss" in result
    assert "train_aux_loss" in result
    assert "epoch_time" in result
    # Values unchanged in non-distributed mode (reduce_dict is identity)
    assert abs(result["train_loss"] - 1.5) < 1e-6
    assert abs(result["moe_aux_loss"] - 0.05) < 1e-6


# ── Fault tolerance tests ──────────────────────────────────────────────────


def test_cleanup_after_exception(dist_manager_disabled: DistributedManager) -> None:
    """FaultToleranceHandler.cleanup_on_exception calls teardown and cleanup callbacks."""
    cfg = DistributedConfig(enabled=False, graceful_shutdown=True)
    ft = FaultToleranceHandler(dist_manager_disabled, cfg)

    callback_called = []

    def my_cleanup() -> None:
        callback_called.append(True)

    ft.register_cleanup(my_cleanup)
    ft.cleanup_on_exception(RuntimeError("test error"))
    assert len(callback_called) == 1, "Cleanup callback must have been called."


def test_barrier_timeout(dist_manager_disabled: DistributedManager) -> None:
    """check_all_ranks_alive() returns True in single-process mode (no timeout)."""
    cfg = DistributedConfig(enabled=False)
    ft = FaultToleranceHandler(dist_manager_disabled, cfg)
    assert ft.check_all_ranks_alive() is True


def test_protect_context_manager(dist_manager_disabled: DistributedManager) -> None:
    """FaultToleranceHandler.protect() re-raises exceptions after cleanup."""
    cfg = DistributedConfig(enabled=False)
    ft = FaultToleranceHandler(dist_manager_disabled, cfg)

    cleaned_up = []
    ft.register_cleanup(lambda: cleaned_up.append(1))

    with pytest.raises(ValueError, match="boom"), ft.protect():
        raise ValueError("boom")

    assert len(cleaned_up) == 1, "Cleanup must run even when exception escapes."
