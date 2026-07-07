"""Project-wide constants for the IVERI CORE architecture.

Centralises all magic numbers, version strings, vocabulary definitions,
and default device/dtype settings so that every downstream module can
import them from a single authoritative source.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Version strings
# ---------------------------------------------------------------------------
IVERI_VERSION: str = "1.0.0"
"""Semantic version of the IVERI CORE package."""

ARCHITECTURE_VERSION: str = "0.2.0-byte-vocab"
"""Architecture revision tag (BLT + Titans + Mamba2 + MoR + MoE)."""

RESEARCH_VERSION: str = "1.0.0"
"""Research milestone version aligned with publication targets."""

BUILD_VERSION: str = "1.0.0"
"""CI/CD build version for reproducibility."""

DOCUMENT_VERSION: str = "2.0"
"""Version of the IVERI CORE design document this code tracks."""

# ---------------------------------------------------------------------------
# Byte vocabulary (raw-byte input — no BPE tokenizer)
# ---------------------------------------------------------------------------
RAW_BYTE_VOCAB_SIZE: int = 256
"""Content byte vocabulary: UTF-8 payload bytes map 1:1 to IDs 0–255."""

NUM_SPECIAL_BYTES: int = 3
"""Count of structural specials outside the raw byte range."""

BYTE_VOCAB_SIZE: int = RAW_BYTE_VOCAB_SIZE + NUM_SPECIAL_BYTES
"""Full embedding vocabulary: 256 content bytes + 3 collision-free specials."""

CONTENT_LOGITS_SIZE: int = RAW_BYTE_VOCAB_SIZE
"""Next-byte prediction head width (content bytes only)."""

# Extended IDs — disjoint from raw bytes 0–255 (Phase 6.3.2 OBJ7).
BOS_BYTE: int = RAW_BYTE_VOCAB_SIZE + 0
"""Beginning-of-sequence token index (256)."""

PAD_BYTE: int = RAW_BYTE_VOCAB_SIZE + 1
"""Padding token index (257)."""

EOS_BYTE: int = RAW_BYTE_VOCAB_SIZE + 2
"""End-of-sequence token index (258)."""

SPECIAL_BYTE_IDS: frozenset[int] = frozenset({BOS_BYTE, PAD_BYTE, EOS_BYTE})

# Pre-v0.2.0 legacy assignments (collided with NUL / U+0001 / U+0002).
LEGACY_PAD_BYTE: int = 0
LEGACY_BOS_BYTE: int = 1
LEGACY_EOS_BYTE: int = 2
LEGACY_SPECIAL_BYTE_IDS: frozenset[int] = frozenset(
    {LEGACY_PAD_BYTE, LEGACY_BOS_BYTE, LEGACY_EOS_BYTE}
)

# ---------------------------------------------------------------------------
# Device / dtype defaults
# ---------------------------------------------------------------------------
DEFAULT_DEVICE: str = "cuda"
"""Default accelerator device string (overridable via config)."""

DEFAULT_DTYPE: torch.dtype = torch.float32
"""Default floating-point dtype for parameters and activations."""

# ---------------------------------------------------------------------------
# Architecture / project metadata
# ---------------------------------------------------------------------------
PROJECT_NAME: str = "IVERI CORE"
"""Human-readable project name used in logging and W&B."""

WANDB_PROJECT: str = "iveri-core"
"""Weights & Biases project slug for experiment tracking."""

# ---------------------------------------------------------------------------
# Phase tracking
# ---------------------------------------------------------------------------
CURRENT_PHASE: int = 6
"""Active development phase (6 = research campaign + scientific integrity)."""
