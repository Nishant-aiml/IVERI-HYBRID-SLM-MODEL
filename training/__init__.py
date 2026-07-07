# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Training infrastructure — trainer, optimizer, checkpointing, mixed precision, scheduler, logger.

Phase 2.6 additions: distributed training infrastructure (DDP/FSDP).
"""

from __future__ import annotations

from training.trainer import Trainer
from training.optimizer import get_optimizer
from training.checkpointing import save_checkpoint, load_checkpoint
from training.mixed_precision import PrecisionHandler
from training.scheduler import IVERIScheduler, SchedulerFactory
from training.logger import ExperimentLogger
from training.distributed import DistributedManager
from training.distributed_trainer import DistributedTrainer
from training.distributed_checkpointing import (
    save_checkpoint_distributed,
    load_checkpoint_distributed,
)
from training.distributed_logger import DistributedLogger
from training.distributed_dataloader import make_distributed_dataloader, set_epoch
from training.distributed_fault_tolerance import FaultToleranceHandler
from training.conversation_formatter import ConversationFormatter, FormatterConfig
from training.loss_mask import LossMaskBuilder, MaskStrategy
from training.sft_dataset import SFTByteDataset, make_sft_dataloader
from training.instruction_dataset import InstructionDatasetLoader
from training.sft_runner import run_sft
from training.model_selection import SFTCheckpointSelector, CodingCheckpointSelector
from training.coding_dataset import CodingDatasetLoader
from training.code_formatter import CodeFormatter, CodeFormatterConfig
from training.coding_curriculum import CodingCurriculum, CurriculumStage
from training.coding_runner import run_coding
from configs.preference_config import PreferenceConfig
from training.preference_dataset import PreferenceDatasetLoader, PreferenceByteDataset, make_preference_dataloader
from training.preference_formatter import PreferenceFormatter, FormattedPreferencePair
from training.preference_loss import PreferenceLoss, compute_logps
from training.reference_model import ReferenceModelManager, verify_parameter_equality, verify_checkpoint_compatibility
from training.preference_runner import run_preference_training
from training.model_selection import PreferenceCheckpointSelector

__all__ = [
    # Phase 2.2 — frozen
    "Trainer",
    "get_optimizer",
    "save_checkpoint",
    "load_checkpoint",
    "PrecisionHandler",
    "IVERIScheduler",
    "SchedulerFactory",
    "ExperimentLogger",
    # Phase 2.6 — distributed
    "DistributedManager",
    "DistributedTrainer",
    "save_checkpoint_distributed",
    "load_checkpoint_distributed",
    "DistributedLogger",
    "make_distributed_dataloader",
    "set_epoch",
    "FaultToleranceHandler",
    # Phase 3.2 — SFT
    "ConversationFormatter",
    "FormatterConfig",
    "LossMaskBuilder",
    "MaskStrategy",
    "SFTByteDataset",
    "make_sft_dataloader",
    "InstructionDatasetLoader",
    "run_sft",
    "SFTCheckpointSelector",
    # Phase 3.3 — Coding Specialization
    "CodingCheckpointSelector",
    "CodingDatasetLoader",
    "CodeFormatter",
    "CodeFormatterConfig",
    "CodingCurriculum",
    "CurriculumStage",
    "run_coding",
    # Phase 3.4 — Preference Alignment
    "PreferenceConfig",
    "PreferenceDatasetLoader",
    "PreferenceByteDataset",
    "make_preference_dataloader",
    "PreferenceFormatter",
    "FormattedPreferencePair",
    "PreferenceLoss",
    "compute_logps",
    "ReferenceModelManager",
    "verify_parameter_equality",
    "verify_checkpoint_compatibility",
    "run_preference_training",
    "PreferenceCheckpointSelector",
]


