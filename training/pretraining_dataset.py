# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Training dataset assembly loader for IVERI CORE pretraining.

Loads datasets using the Phase 3.0 registry and strictly validates license,
integrity (SHA256), version, and pipeline metadata before loading.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from configs.base_config import IVERIConfig
from data.pipeline.data_registry import DataRegistry, DatasetEntry
from data.pipeline.dataloader import PretrainByteDataset
from data.pipeline.license_checker import LicenseChecker
from data.pipeline.versioning import DatasetVersioner

logger = logging.getLogger(__name__)


def _get_val(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class PretrainingDatasetLoader:
    """Loads and validates pretraining datasets through the data pipeline."""

    def __init__(self, config: IVERIConfig, registry: DataRegistry | None = None) -> None:
        self.config = config
        data_pipeline = getattr(config, "data_pipeline", {})
        reg_cfg = _get_val(data_pipeline, "registry", {})
        
        self.registry = registry or DataRegistry(
            spec_dir=_get_val(reg_cfg, "spec_dir", "data/dataset_specs"),
            auto_discover=_get_val(reg_cfg, "auto_discover", True),
        )
        self.versioner = DatasetVersioner()
        self.license_checker = LicenseChecker(self.registry)

    def load(self, name: str = "tinystories", split: str = "train") -> PretrainByteDataset:
        """Verify ingestion constraints and load the PretrainByteDataset.

        Strict verification chain:
        download -> SHA256 -> manifest -> license -> stats -> pipeline version -> processed -> training.
        """
        # 1. Look up in registry
        entry: DatasetEntry = self.registry.get(name)

        # 2. Verify license metadata
        if not self.license_checker.verify(name, use_case="research"):
            raise ValueError(
                f"License '{entry.license}' for dataset '{name}' is not compatible with research pretraining."
            )

        # 3. Determine paths
        # Preprocessed stage directories: e.g. data/processed/stage1/tinystories
        data_pipeline = getattr(self.config, "data_pipeline", {})
        report_cfg = _get_val(data_pipeline, "report", {})
        processed_base = Path(_get_val(report_cfg, "processed_data_dir", "data/processed"))
        processed_dir = processed_base / f"stage{entry.stage}" / name

        # Fallback to standard processed folder if stage subfolder not present
        if not processed_dir.exists():
            processed_dir = processed_base / name

        # In test/mock mode, if directories do not exist, we raise FileNotFoundError
        # but allow test suite to mock or construct it on the fly.
        if not processed_dir.exists():
            raise FileNotFoundError(
                f"Processed dataset directory '{processed_dir}' not found. "
                "Ensure Stage 0 data pipeline has run and processed the dataset."
            )

        # 4. Verify VERSION.json & pipeline version
        version_info = self.versioner.load_version(processed_dir)
        if version_info.stage != str(entry.stage):
            raise ValueError(
                f"Dataset stage mismatch. Expected: {entry.stage}, Got: {version_info.stage}"
            )

        # Verify pipeline version
        manifest_file = processed_base / "manifest.json"
        if manifest_file.exists():
            manifest_entries = self.versioner.load_manifest(manifest_file)
            matching_entry = next((e for e in manifest_entries if e.dataset_name == name), None)
            if matching_entry:
                # 5. Verify SHA256
                # Compute current processed directory content hash
                current_hash = self.versioner.compute_content_hash(processed_dir)
                if current_hash != version_info.content_hash:
                    raise ValueError(
                        f"Dataset content hash mismatch. Manifest says {version_info.content_hash}, "
                        f"but computed {current_hash}."
                    )
                if matching_entry.sha256 != current_hash:
                    raise ValueError(
                        f"Dataset SHA256 in manifest ({matching_entry.sha256}) does not match "
                        f"computed hash ({current_hash})."
                    )
                # 6. Verify pipeline version match
                if matching_entry.pipeline_version != self.versioner.pipeline_version:
                    raise ValueError(
                        f"Pipeline version mismatch. Expected: {self.versioner.pipeline_version}, "
                        f"Got: {matching_entry.pipeline_version}"
                    )

        # 7. Verify byte statistics
        if version_info.byte_count <= 0 or version_info.document_count <= 0:
            raise ValueError(
                f"Dataset '{name}' statistics show empty bytes/documents: "
                f"bytes={version_info.byte_count}, docs={version_info.document_count}"
            )

        logger.info(
            f"Successfully verified pretraining dataset '{name}' (version: {version_info.version_id}, "
            f"bytes: {version_info.byte_count}). Ingesting for pretraining."
        )

        # Return standard PretrainByteDataset pointing to processed folder
        return PretrainByteDataset(
            data_path=processed_dir,
            seq_len=self.config.training.seq_len,
            split=split,
            text_field=entry.text_field,
        )
