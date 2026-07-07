# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""IVERI CORE — Model Components Package.

This package contains all model architecture components:
- blt/      : Byte Latent Transformer (entropy model, patcher, encoder, decoder)
- mamba2/   : Mamba2 State Space Model blocks
- mor/      : Mixture of Recursions (router, recursion engine, KV cache)
- titans/   : Titans Neural Memory (memory MLP, online updater, LR generator)
- moe/      : Mixture of Experts (router, expert FFNs)
- attention : Flash Attention wrapper
- norms     : RMSNorm implementation
- backbone  : Full backbone block assembly
- iveri_core: Complete model assembly
"""

from __future__ import annotations

from model.attention import FlashAttentionWrapper
from model.backbone import Backbone, BackboneBlock
from model.blt import BLTByteDecoder, BLTByteEncoder, ByteEntropyModel, DynamicPatcher
from model.iveri_core import IVERIModel
from model.mamba2 import Mamba2Block
from model.moe import MoEExperts, SparseMoERouter
from model.mor import RecursionDepthRouter, RecursionEngine
from model.norms import RMSNorm
from model.rope import RotaryEmbedding, apply_rotary_emb
from model.swiglu import SwiGLU, SwiGLUFFN
from model.titans import MemoryLearningRateGenerator, MemoryUpdater, TitansMemory

__all__ = [
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rotary_emb",
    "SwiGLU",
    "SwiGLUFFN",
    "SparseMoERouter",
    "MoEExperts",
    "Mamba2Block",
    "FlashAttentionWrapper",
    "RecursionDepthRouter",
    "RecursionEngine",
    "ByteEntropyModel",
    "DynamicPatcher",
    "BLTByteEncoder",
    "BLTByteDecoder",
    "TitansMemory",
    "MemoryLearningRateGenerator",
    "MemoryUpdater",
    "Backbone",
    "BackboneBlock",
    "IVERIModel",
]
