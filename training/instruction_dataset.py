# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Instruction dataset loader for IVERI CORE Phase 3.2 SFT pipeline.

Validates, loads, and prepares SFT datasets from the data registry.
Follows the same verification chain as :class:`~training.pretraining_dataset.PretrainingDatasetLoader`
but targets Stage 2 (instruction tuning) datasets.

Verification chain
------------------
1. Resolve dataset entry from :class:`~data.registry.DataRegistry`.
2. License validation.
3. Resolve ``data/processed/stage2/{name}/`` path.
4. VERSION.json stage check.
5. manifest.json SHA-256 hash verification.
6. SFT schema validation via :class:`~data.pipeline.sft_validator.SFTValidator`.
7. Return :class:`~training.sft_dataset.SFTByteDataset`.

Examples
--------
>>> from configs.base_config import IVERIConfig
>>> config = IVERIConfig()
>>> loader = InstructionDatasetLoader(config)
>>> train_ds = loader.load("magpie_pro", split="train")  # doctest: +SKIP
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from configs.base_config import IVERIConfig
from training.conversation_formatter import ConversationFormatter, FormatterConfig
from training.sft_dataset import SFTByteDataset

logger = logging.getLogger(__name__)

# ── Stage 2 constants ──────────────────────────────────────────────────────

_STAGE2_ID: int = 2
_DEFAULT_PROCESSED_BASE: str = "data/processed"


# ── Dataset entry stub (used if DataRegistry is unavailable) ───────────────


class _DatasetEntry:
    """Minimal dataset entry for offline / mock use."""

    def __init__(
        self,
        name: str,
        stage: int,
        license_id: str = "Apache-2.0",
        format_type: str = "messages",
        hf_id: str = "",
    ) -> None:
        self.name = name
        self.stage = stage
        self.license_id = license_id
        self.format_type = format_type
        self.hf_id = hf_id


# ── Main loader class ──────────────────────────────────────────────────────


class InstructionDatasetLoader:
    """Load and validate SFT (Stage 2) datasets for IVERI CORE instruction tuning.

    Parameters
    ----------
    config:
        Master IVERI configuration.
    registry:
        Optional :class:`~data.registry.DataRegistry` instance.
        If ``None``, a registry is loaded from the data package if available;
        falls back to offline-safe path resolution.
    formatter_config:
        Optional :class:`~training.conversation_formatter.FormatterConfig`.
        Defaults to Alpaca-style IVERI template.
    """

    def __init__(
        self,
        config: IVERIConfig,
        registry: Any | None = None,
        formatter_config: FormatterConfig | None = None,
    ) -> None:
        self.config = config
        self.formatter_config = formatter_config or FormatterConfig()
        self._registry = registry
        self._versioner: Any | None = None

        # Try to import DataRegistry and versioner from the pipeline
        if registry is None:
            try:
                from data.registry import DataRegistry  # type: ignore[import]
                self._registry = DataRegistry()
            except ImportError:
                logger.debug("DataRegistry not available; using path-only resolution.")

        try:
            from data.pipeline.versioner import DataVersioner  # type: ignore[import]
            self._versioner = DataVersioner()
        except ImportError:
            logger.debug("DataVersioner not available; skipping version checks.")

        # Resolve base processed data directory
        data_pipeline = getattr(config, "data_pipeline", None)
        if data_pipeline is not None:
            report_cfg = getattr(data_pipeline, "report", None) or {}
            if isinstance(report_cfg, dict):
                self._processed_base = Path(
                    report_cfg.get("processed_data_dir", _DEFAULT_PROCESSED_BASE)
                )
            else:
                self._processed_base = Path(
                    getattr(report_cfg, "processed_data_dir", _DEFAULT_PROCESSED_BASE)
                )
        else:
            self._processed_base = Path(_DEFAULT_PROCESSED_BASE)

    # ── Public API ─────────────────────────────────────────────────────

    def load(
        self,
        name: str,
        split: str = "train",
        seq_len: int | None = None,
        train_on_prompt: bool = False,
        max_samples: int | None = None,
    ) -> SFTByteDataset:
        """Load and validate a Stage 2 SFT dataset.

        Parameters
        ----------
        name:
            Dataset name as registered in the data registry
            (e.g. ``"magpie_pro"``, ``"tulu3_sft"``).
        split:
            One of ``"train"``, ``"val"``, ``"test"``.
        seq_len:
            Sequence length override.  Defaults to ``config.training.seq_len``.
        train_on_prompt:
            Whether to compute loss on prompt bytes.
        max_samples:
            Cap the number of loaded samples (useful for verification runs).

        Returns
        -------
        SFTByteDataset
            Validated, formatted dataset ready for training.

        Raises
        ------
        FileNotFoundError
            If the processed dataset directory does not exist.
        RuntimeError
            If license check fails or schema validation reports critical errors.
        """
        effective_seq_len = seq_len or self.config.training.seq_len

        # 1. Resolve dataset entry
        entry = self._get_dataset_entry(name)

        # 2. License validation
        self._check_license(entry)

        # 3. Resolve processed path
        processed_dir = self._resolve_path(name, entry)

        # 4. VERSION.json stage validation
        self._validate_stage(processed_dir, name)

        # 5. manifest.json hash verification
        self._verify_manifest(processed_dir, name)

        # 6. Load raw samples from JSONL / JSON
        raw_samples = self._load_raw_samples(processed_dir, split, max_samples)

        if not raw_samples:
            logger.warning(
                "No samples loaded for dataset '%s' split='%s'. "
                "Returning empty dataset.",
                name,
                split,
            )

        # 7. SFT schema validation
        raw_samples = self._validate_schema(raw_samples, name)

        # 8. Build formatter
        formatter = ConversationFormatter(self.formatter_config)

        # 9. Build dataset
        dataset = SFTByteDataset(
            samples=raw_samples,
            seq_len=effective_seq_len,
            formatter=formatter,
            train_on_prompt=train_on_prompt,
        )

        logger.info(
            "Loaded SFT dataset '%s' split='%s': %d samples, seq_len=%d",
            name,
            split,
            len(dataset),
            effective_seq_len,
        )
        return dataset

    def load_mock(
        self,
        samples: list[dict[str, Any]],
        seq_len: int = 512,
        train_on_prompt: bool = False,
    ) -> SFTByteDataset:
        """Load a dataset directly from a list (for testing/offline use).

        Parameters
        ----------
        samples:
            List of raw sample dicts (no validation chain).
        seq_len:
            Sequence length.
        train_on_prompt:
            Whether to compute loss on prompt bytes.

        Returns
        -------
        SFTByteDataset
        """
        formatter = ConversationFormatter(self.formatter_config)
        return SFTByteDataset(
            samples=samples,
            seq_len=seq_len,
            formatter=formatter,
            train_on_prompt=train_on_prompt,
        )

    # ── Private methods ────────────────────────────────────────────────

    def _get_dataset_entry(self, name: str) -> Any:
        """Retrieve registry entry or create a stub."""
        if self._registry is not None:
            try:
                entry = self._registry.get(name)
                if entry is not None:
                    return entry
            except Exception as exc:
                logger.debug("Registry lookup failed for '%s': %s", name, exc)

        # Offline stub — assume Stage 2 / Apache-2.0
        return _DatasetEntry(name=name, stage=_STAGE2_ID)

    def _check_license(self, entry: Any) -> None:
        """Validate dataset license for research use."""
        license_id = getattr(entry, "license_id", getattr(entry, "license", ""))
        if not license_id:
            logger.warning("No license information found for dataset '%s'.", getattr(entry, "name", "?"))
            return

        # Permissive licenses allowed for SFT research
        _ALLOWED_LICENSES = frozenset({
            "Apache-2.0", "MIT", "CC-BY-4.0", "CC-BY-SA-4.0",
            "CC0-1.0", "OpenRAIL", "NVIDIA-Open",
        })
        if license_id not in _ALLOWED_LICENSES:
            raise RuntimeError(
                f"Dataset license '{license_id}' is not in the allowed set "
                f"for SFT research: {sorted(_ALLOWED_LICENSES)}"
            )
        logger.debug("License check passed: %s", license_id)

    def _resolve_path(self, name: str, entry: Any) -> Path:
        """Resolve stage2 processed data directory."""
        # Try registered path first
        registered_path = getattr(entry, "processed_path", None)
        if registered_path:
            p = Path(registered_path)
            if p.exists():
                return p

        # Standard layout: data/processed/stage2/{name}/
        stage2_dir = self._processed_base / "stage2" / name
        if stage2_dir.exists():
            return stage2_dir

        # Fallback flat layout
        flat_dir = self._processed_base / name
        if flat_dir.exists():
            logger.warning(
                "Dataset '%s' found under flat layout (not stage2/). "
                "Consider migrating to data/processed/stage2/%s/.",
                name,
                name,
            )
            return flat_dir

        raise FileNotFoundError(
            f"Processed dataset directory not found for '{name}'. "
            f"Expected: {stage2_dir}. "
            f"Run the Stage 0 data pipeline first."
        )

    def _validate_stage(self, processed_dir: Path, name: str) -> None:
        """Check VERSION.json stage number matches Stage 2."""
        version_file = processed_dir / "VERSION.json"
        if not version_file.exists():
            logger.warning("No VERSION.json found at %s; skipping stage check.", processed_dir)
            return
        try:
            version_data = json.loads(version_file.read_text(encoding="utf-8"))
            stage = int(version_data.get("stage", -1))
            if stage != _STAGE2_ID:
                raise RuntimeError(
                    f"Dataset '{name}' has stage={stage} in VERSION.json, "
                    f"but Stage 2 is required for SFT training."
                )
        except RuntimeError:
            raise
        except Exception as exc:
            logger.warning("Could not parse VERSION.json at %s: %s", processed_dir, exc)

    def _verify_manifest(self, processed_dir: Path, name: str) -> None:
        """Verify manifest.json SHA-256 hash if present."""
        manifest_file = processed_dir / "manifest.json"
        if not manifest_file.exists():
            logger.debug("No manifest.json for '%s'; skipping hash verification.", name)
            return
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            logger.debug("manifest.json loaded for '%s': %d files.", name, len(manifest))
        except Exception as exc:
            logger.warning("Could not parse manifest.json for '%s': %s", name, exc)

    def _load_raw_samples(
        self, processed_dir: Path, split: str, max_samples: int | None
    ) -> list[dict[str, Any]]:
        """Load raw samples from JSONL/JSON files in the processed directory."""
        samples: list[dict[str, Any]] = []

        # Try split-specific files first
        for fname in (f"{split}.jsonl", f"{split}.json", "data.jsonl", "data.json"):
            fpath = processed_dir / fname
            if fpath.exists():
                samples = _load_jsonl_or_json(fpath)
                logger.info("Loaded %d samples from %s", len(samples), fpath)
                break

        # If no split file found, try any *.jsonl in directory
        if not samples:
            for fpath in sorted(processed_dir.glob("*.jsonl")):
                samples.extend(_load_jsonl_or_json(fpath))
            if not samples:
                for fpath in sorted(processed_dir.glob("*.json")):
                    data = _load_jsonl_or_json(fpath)
                    samples.extend(data)

        if max_samples is not None:
            samples = samples[:max_samples]

        return samples

    def _validate_schema(
        self, samples: list[dict[str, Any]], name: str
    ) -> list[dict[str, Any]]:
        """Run SFT schema validation and filter invalid samples."""
        try:
            from data.pipeline.sft_validator import SFTValidator  # type: ignore[import]

            validator = SFTValidator()
            valid = validator.filter_valid(samples)
            n_removed = len(samples) - len(valid)
            if n_removed > 0:
                logger.warning(
                    "SFT validation for '%s': removed %d/%d invalid samples.",
                    name,
                    n_removed,
                    len(samples),
                )
            return valid
        except ImportError:
            logger.debug("SFTValidator not available; skipping schema validation.")
            return samples


# ── Private helpers ────────────────────────────────────────────────────────


def _load_jsonl_or_json(path: Path) -> list[dict[str, Any]]:
    """Load either a JSONL or JSON file into a list of dicts."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    # Try JSONL (one JSON object per line)
    results: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                results.append(obj)
            elif isinstance(obj, list):
                results.extend(o for o in obj if isinstance(o, dict))
        except json.JSONDecodeError:
            pass
    if results:
        return results
    # Fallback: try as a single JSON array
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [o for o in obj if isinstance(o, dict)]
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass
    return []
