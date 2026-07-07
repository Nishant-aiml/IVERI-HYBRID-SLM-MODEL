"""IVERI CORE — foundational infrastructure package.

Re-exports all public symbols so that downstream code can write::

    from core import BYTE_VOCAB_SIZE, IVERIError, BaseModule, register

instead of reaching into individual sub-modules.
"""

from __future__ import annotations

# -- Constants -------------------------------------------------------------
from core.constants import (
    ARCHITECTURE_VERSION,
    BOS_BYTE,
    BUILD_VERSION,
    BYTE_VOCAB_SIZE,
    CONTENT_LOGITS_SIZE,
    CURRENT_PHASE,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DOCUMENT_VERSION,
    EOS_BYTE,
    IVERI_VERSION,
    LEGACY_BOS_BYTE,
    LEGACY_EOS_BYTE,
    LEGACY_PAD_BYTE,
    LEGACY_SPECIAL_BYTE_IDS,
    NUM_SPECIAL_BYTES,
    PAD_BYTE,
    PROJECT_NAME,
    RAW_BYTE_VOCAB_SIZE,
    RESEARCH_VERSION,
    SPECIAL_BYTE_IDS,
    WANDB_PROJECT,
)

# -- Exceptions ------------------------------------------------------------
from core.exceptions import (
    CheckpointError,
    ConfigError,
    ConvergenceError,
    IVERIError,
    MemoryError,
    RegistryError,
    ShapeError,
    ValidationError,
)

# -- Factory ---------------------------------------------------------------
from core.factory import (
    build_component,
    build_model,
    count_parameters,
    count_parameters_by_module,
)

# -- Abstract interfaces ---------------------------------------------------
from core.interfaces import (
    BaseDecoder,
    BaseEncoder,
    BaseMemory,
    BaseModule,
    BaseRouter,
)

# -- Registry --------------------------------------------------------------
from core.registry import ComponentRegistry, register

__all__: list[str] = [
    # constants
    "ARCHITECTURE_VERSION",
    "BUILD_VERSION",
    "BYTE_VOCAB_SIZE",
    "BOS_BYTE",
    "CONTENT_LOGITS_SIZE",
    "CURRENT_PHASE",
    "DEFAULT_DEVICE",
    "DEFAULT_DTYPE",
    "DOCUMENT_VERSION",
    "EOS_BYTE",
    "IVERI_VERSION",
    "LEGACY_BOS_BYTE",
    "LEGACY_EOS_BYTE",
    "LEGACY_PAD_BYTE",
    "LEGACY_SPECIAL_BYTE_IDS",
    "NUM_SPECIAL_BYTES",
    "PAD_BYTE",
    "PROJECT_NAME",
    "RAW_BYTE_VOCAB_SIZE",
    "RESEARCH_VERSION",
    "SPECIAL_BYTE_IDS",
    "WANDB_PROJECT",
    # exceptions
    "CheckpointError",
    "ConfigError",
    "ConvergenceError",
    "IVERIError",
    "MemoryError",
    "RegistryError",
    "ShapeError",
    "ValidationError",
    # interfaces
    "BaseDecoder",
    "BaseEncoder",
    "BaseMemory",
    "BaseModule",
    "BaseRouter",
    # registry
    "ComponentRegistry",
    "register",
    # factory
    "build_component",
    "build_model",
    "count_parameters",
    "count_parameters_by_module",
]
