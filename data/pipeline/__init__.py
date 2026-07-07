# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""IVERI CORE Data Pipeline — Stage 0 Data Engineering Infrastructure.

This package implements the complete data engineering pipeline:
  download → validate → encode → filter → deduplicate → split → version → report
"""

from __future__ import annotations

# Expose byte counter
from data.pipeline.byte_counter import ByteCounter, ByteStats

# Expose byte_encoder
from data.pipeline.byte_encoder import ByteEncoder, ByteEncoderConfig, ByteEncoderStats

# Expose registry and downloader first
from data.pipeline.data_registry import DataRegistry, DatasetEntry, get_default_registry

# Expose dataloaders
from data.pipeline.dataloader import (
    BaseByteDataset,
    CodingByteDataset,
    PretrainByteDataset,
    SFTByteDataset,
    get_coding_dataloader,
    get_pretrain_dataloader,
    get_sft_dataloader,
)

# Expose deduplicator
from data.pipeline.deduplication import DeduplicationConfig, DeduplicationReport, Deduplicator
from data.pipeline.downloader import DatasetDownloader, DownloadMetadata

# Expose language detector
from data.pipeline.language_detector import (
    ALLOWED_LANGUAGES_STAGE1,
    ALLOWED_LANGUAGES_STAGE3B,
    LanguageDetectionReport,
    LanguageDetector,
)
from data.pipeline.license_checker import LicenseChecker, LicenseInfo, LicenseReport

# Expose mixer
from data.pipeline.mixer import (
    STAGE1_MIXING_WEIGHTS,
    STAGE2_MIXING_WEIGHTS,
    STAGE3A_MIXING_WEIGHTS,
    DatasetMixer,
    MixingStrategy,
)

# Expose PII remover
from data.pipeline.pii_remover import PIIRemover, PIIReport
from data.pipeline.provenance import ProcessingStep, ProvenanceRecord, ProvenanceTracker
from data.pipeline.proprietary_ingest import (
    ProprietaryIngestReport,
    ProprietaryRecord,
    count_proprietary_records,
    ingest_stage3b,
)

# Expose quality filters
from data.pipeline.quality_filter import QualityFilter, QualityFilterConfig, QualityReport

# Expose SFT validator
from data.pipeline.sft_validator import SFTValidationReport, SFTValidator

# Expose splitter
from data.pipeline.splitter import DatasetSplitter, SplitReport

# Expose statistics
from data.pipeline.statistics import (
    DatasetReport,
    DatasetStatisticsGenerator,
    LengthHistogram,
    PipelineSummaryReport,
)
from data.pipeline.versioning import DatasetVersioner, ManifestEntry, VersionInfo

# Optional dependency check
try:
    import datasketch  # noqa: F401

    DEDUP_AVAILABLE = True
except ImportError:
    DEDUP_AVAILABLE = False

try:
    import langdetect  # noqa: F401

    LANGUAGE_DETECTION_AVAILABLE = True
except ImportError:
    LANGUAGE_DETECTION_AVAILABLE = False

__all__ = [
    "DataRegistry",
    "DatasetEntry",
    "get_default_registry",
    "DatasetDownloader",
    "DownloadMetadata",
    "LicenseChecker",
    "LicenseInfo",
    "LicenseReport",
    "ProcessingStep",
    "ProvenanceRecord",
    "ProvenanceTracker",
    "ProprietaryIngestReport",
    "ProprietaryRecord",
    "count_proprietary_records",
    "ingest_stage3b",
    "DatasetVersioner",
    "ManifestEntry",
    "VersionInfo",
    "BaseByteDataset",
    "CodingByteDataset",
    "PretrainByteDataset",
    "SFTByteDataset",
    "get_coding_dataloader",
    "get_pretrain_dataloader",
    "get_sft_dataloader",
    "ByteEncoder",
    "ByteEncoderConfig",
    "ByteEncoderStats",
    "QualityFilter",
    "QualityFilterConfig",
    "QualityReport",
    "PIIRemover",
    "PIIReport",
    "DatasetSplitter",
    "SplitReport",
    "ByteCounter",
    "ByteStats",
    "STAGE1_MIXING_WEIGHTS",
    "STAGE2_MIXING_WEIGHTS",
    "STAGE3A_MIXING_WEIGHTS",
    "DatasetMixer",
    "MixingStrategy",
    "SFTValidationReport",
    "SFTValidator",
    "DatasetReport",
    "DatasetStatisticsGenerator",
    "LengthHistogram",
    "PipelineSummaryReport",
    "DeduplicationConfig",
    "DeduplicationReport",
    "Deduplicator",
    "ALLOWED_LANGUAGES_STAGE1",
    "ALLOWED_LANGUAGES_STAGE3B",
    "LanguageDetectionReport",
    "LanguageDetector",
]
