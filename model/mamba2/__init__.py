# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Mamba2 Structured State Space Duality block exports."""

from __future__ import annotations

from model.mamba2.block import Mamba2Block
from model.mamba2.math import compute_ssd_matrix, discretize_parameters
from model.mamba2.scan import selective_ssd_scan

__all__ = [
    "Mamba2Block",
    "selective_ssd_scan",
    "discretize_parameters",
    "compute_ssd_matrix",
]
