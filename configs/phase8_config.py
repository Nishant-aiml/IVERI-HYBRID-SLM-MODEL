# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Phase 8 Dataset Engineering configuration.

Storage budgets are computed dynamically based on available disk space.
Datasets are downloaded in priority order; if budget exhausted, pipeline
stops gracefully with whatever datasets were successfully downloaded.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DatasetBudget:
    name: str
    hf_id: str
    hf_revision: str              # PINNED revision — never "latest"
    hf_config: str | None
    license: str                  # SPDX: "CC-BY-4.0", "ODC-By", etc.
    license_url: str
    source_url: str
    text_field: str               # Primary text field name
    priority: int                 # 1 = highest priority (download first)
    stage: str                    # "pretrain" or "eval_only"
    max_storage_gb: float         # Hard cap — streaming stops here
    curriculum_phases: list[str]  # ["A", "B", "C"]
    source_type: str = "huggingface" # huggingface | local | zip | http | kaggle
    enabled: bool = True
    quality_backend: str = "fast"
    quality_threshold: float = 0.4
    download_date: str | None = None
    sha256_sample: str | None = None   # SHA-256 of first 1000 docs
    verified: bool = False


@dataclass  
class Phase8Config:
    """Master configuration for Phase 8 dataset engineering."""

    # Base directories
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed/stage1"
    certified_data_dir: str = "data/certified"
    curriculum_dir: str = "data/curriculum"
    evaluation_data_dir: str = "data/evaluation"
    reports_dir: str = "reports/phase_8"
    
    # Storage constraints
    min_free_gb: float = 15.0          # Minimum free disk to keep (never touch)
    reserve_fraction: float = 0.20     # Reserve 20% of free space on top
    max_total_budget_gb: float | None = None  # None = auto-compute from disk

    # Directory for YAML dataset specs
    datasets_specs_dir: str = "configs/datasets"

    # Quality scoring backend
    default_quality_backend: str = "fast"  # "fast" | "perplexity" | "llm" | "custom"

    # Recipe system
    recipes_dir: str = "configs/recipes"
    default_recipe: str = "iveri_v1"

    # Cache
    cache_dir: str = ".hf_cache"
    use_cache: bool = True

    # Parallel downloads
    max_concurrent_downloads: int = 2

    # Lineage tracking
    enable_lineage: bool = True
    lineage_dir: str = "data/certified"

    # Pipeline behavior
    streaming_mode: bool = True             # Always stream, never full download
    eval_contamination_check: bool = True   # Always run contamination check
    safety_mode: str = "research"           # "research" (permissive) | "production" (strict)
    offline_mode: bool = False              # Set True or IVERI_OFFLINE=1 env var
    resume_downloads: bool = True           # Resume interrupted downloads
    
    # Quality thresholds
    min_doc_chars: int = 50
    max_doc_chars: int = 100_000
    min_alpha_ratio: float = 0.5
    min_quality_score: float = 0.4         # Keep docs above this quality score
    near_dedup_threshold: float = 0.8      # MinHash LSH threshold
    language_confidence: float = 0.8       # Min confidence for language detection

    def __post_init__(self) -> None:
        import os
        if os.environ.get("IVERI_OFFLINE", "0") == "1":
            self.offline_mode = True

    def compute_dynamic_budget(self, base_dir: str | Path = ".") -> float:
        """Compute usable storage budget from current free disk space."""
        free_bytes = shutil.disk_usage(str(base_dir)).free
        free_gb = free_bytes / (1024**3)
        reserved_gb = max(free_gb * self.reserve_fraction, self.min_free_gb)
        usable_gb = max(free_gb - reserved_gb, 0.0)
        if self.max_total_budget_gb is not None:
            usable_gb = min(usable_gb, self.max_total_budget_gb)
        return usable_gb

    def to_dict(self) -> dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Phase8Config:
        return cls(**data)
