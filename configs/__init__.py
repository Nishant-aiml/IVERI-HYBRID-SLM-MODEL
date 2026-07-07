# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""IVERI CORE — Configuration Package.

Public API
----------
>>> from configs import IVERIConfig, get_base_config
>>> config = get_base_config()                          # 10M nano defaults
>>> config = get_base_config(model={"hidden_dim": 512}) # override model dims

Sub-configs are also importable for type annotations::

    from configs import (
        BLTConfig,
        HardwareConfig,
        IVERIConfig,
        LoggingConfig,
        ModelConfig,
        TrainingConfig,
        EvaluationConfig,
        get_base_config,
    )
"""

from __future__ import annotations

from configs.base_config import (
    BLTConfig,
    EvaluationConfig,
    HardwareConfig,
    IVERIConfig,
    LoggingConfig,
    ModelConfig,
    TrainingConfig,
    get_base_config,
)
from configs.distributed_config import DistributedConfig
from configs.data_pipeline_config import DataPipelineConfig

__all__ = [
    "BLTConfig",
    "DistributedConfig",
    "DataPipelineConfig",
    "EvaluationConfig",
    "HardwareConfig",
    "IVERIConfig",
    "LoggingConfig",
    "ModelConfig",
    "TrainingConfig",
    "get_base_config",
]
