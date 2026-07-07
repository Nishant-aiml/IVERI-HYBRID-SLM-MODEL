# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402

"""Pytest shared fixtures and configuration for IVERI CORE tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.base_config import IVERIConfig, get_base_config


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Fixture that returns the absolute path to the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def base_config() -> IVERIConfig:
    """Fixture that returns a fresh instance of the default 10M nano config."""
    cfg = get_base_config()
    # E2E tests use in-process mock datasets; workers break on Windows / Python 3.14+.
    cfg.hardware.num_workers = 0
    if not torch.cuda.is_available():
        cfg.hardware.device = "cpu"
        cfg.hardware.mixed_precision = "fp32"
    return cfg


@pytest.fixture(scope="session")
def device() -> str:
    """Fixture that returns the default device ('cuda' if available, else 'cpu')."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(autouse=True)
def seed() -> int:
    """Fixture that sets a deterministic random seed before every test."""
    torch.manual_seed(42)
    return 42


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Fixture that creates and returns a temporary directory for testing."""
    test_dir = tmp_path / "iveri_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir
