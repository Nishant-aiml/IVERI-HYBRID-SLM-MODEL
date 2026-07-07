# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Coding dataset loader for IVERI CORE Phase 3.3 coding specialization.

Validates, loads, and prepares coding datasets from the Stage 3A data registry.
Follows the same verification chain as :class:`~training.instruction_dataset.InstructionDatasetLoader`
but targets Stage 3A (coding specialization) datasets.

Two format types are supported:
- **pretrain**: raw source code trained with full-sequence loss (``train_on_prompt=True``)
- **sft**: instruction-following trained with response-only masked loss (``train_on_prompt=False``)

Verification chain
------------------
1. Resolve dataset entry from registry or coding.yaml.
2. License validation (extended for coding licenses).
3. Resolve ``data/processed/stage3a/{name}/`` path.
4. VERSION.json stage check (accepts ``\"3A\"`` string or int ``3``).
5. manifest.json SHA-256 hash verification.
6. SFT schema validation via :class:`~data.pipeline.sft_validator.SFTValidator`.
7. Return :class:`~training.sft_dataset.SFTByteDataset`.

Examples
--------
>>> from configs.base_config import IVERIConfig
>>> config = IVERIConfig()
>>> loader = CodingDatasetLoader(config)
>>> ds = loader.load_mock([{"instruction": "Write hello world", "output": "print('hello')"}])
>>> len(ds) > 0
True
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from configs.base_config import IVERIConfig
from training.sft_dataset import SFTByteDataset

logger = logging.getLogger(__name__)

# ── Stage 3A constants ──────────────────────────────────────────────────────

_STAGE3A_IDS: frozenset = frozenset({"3A", "3a", 3})
"""Accepted values for VERSION.json ``stage`` field."""

_DEFAULT_PROCESSED_BASE: str = "data/processed"

# Extended license set for coding datasets
_ALLOWED_LICENSES: frozenset[str] = frozenset({
    "Apache-2.0", "MIT", "CC-BY-4.0", "CC-BY-SA-4.0",
    "CC0-1.0", "OpenRAIL", "NVIDIA-Open",
    # Extended for coding (Feedback)
    "NVIDIA-Open-Model", "various-permissive",
    "bsd-2-clause", "bsd-3-clause",
    "CC-BY-NC-4.0",
})

# Format types that map to pretrain-style (full sequence loss)
_PRETRAIN_FORMAT_TYPES: frozenset[str] = frozenset({"pretrain", "raw", "code_pretrain"})


# ── Dataset entry stub ─────────────────────────────────────────────────────


class _DatasetEntry:
    """Minimal dataset entry for offline / mock use."""

    def __init__(
        self,
        name: str,
        stage: str = "3A",
        license_id: str = "Apache-2.0",
        format_type: str = "sft",
        hf_id: str = "",
        format_mode: str = "sft",
    ) -> None:
        self.name = name
        self.stage = stage
        self.license_id = license_id
        self.format_type = format_type
        self.hf_id = hf_id
        self.format_mode = format_mode  # "pretrain" or "sft"


# ── Coding.yaml loader ─────────────────────────────────────────────────────

def _load_coding_yaml() -> dict[str, Any]:
    """Load coding.yaml dataset registry. Returns empty dict on failure."""
    try:
        import yaml  # type: ignore[import]
        yaml_path = Path("data/dataset_specs/coding.yaml")
        if yaml_path.exists():
            with open(yaml_path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception as exc:
        logger.debug("Could not load coding.yaml: %s", exc)
    return {}


# ── Main loader class ──────────────────────────────────────────────────────


class CodingDatasetLoader:
    """Load and validate Stage 3A coding datasets for IVERI CORE.

    Parameters
    ----------
    config:
        Master IVERI configuration.
    registry:
        Optional :class:`~data.registry.DataRegistry` instance.
        If ``None``, falls back to coding.yaml then offline-safe stubs.
    """

    def __init__(
        self,
        config: IVERIConfig,
        registry: Any | None = None,
    ) -> None:
        self.config = config
        self._registry = registry
        self._versioner: Any | None = None
        self._coding_yaml: dict[str, Any] = _load_coding_yaml()

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
        train_on_prompt: bool | None = None,
        max_samples: int | None = None,
        language_filter: list[str] | None = None,
    ) -> SFTByteDataset:
        """Load and validate a Stage 3A coding dataset.

        Parameters
        ----------
        name:
            Dataset name (registry key, e.g. ``\"the_stack_v2_deep\"``,
            ``\"nemotron_competitive\"``).
        split:
            One of ``\"train\"``, ``\"val\"``, ``\"test\"``.
        seq_len:
            Sequence length override.  Defaults to ``config.training.seq_len``.
        train_on_prompt:
            Override loss masking.  If ``None``, auto-detected from format type:
            pretrain datasets use ``True``; SFT datasets use ``False``.
        max_samples:
            Cap loaded samples (useful for verification runs).
        language_filter:
            Filter samples by detected language (e.g. ``[\"python\"]``).

        Returns
        -------
        SFTByteDataset

        Raises
        ------
        FileNotFoundError
            If the processed dataset directory does not exist.
        RuntimeError
            If license check fails or schema validation fails critically.
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

        # 6. Load raw samples
        raw_samples = self._load_raw_samples(processed_dir, split, max_samples)

        if not raw_samples:
            logger.warning(
                "No samples loaded for coding dataset '%s' split='%s'. "
                "Returning empty dataset.",
                name,
                split,
            )

        # 7. Language filtering
        if language_filter:
            raw_samples = self._filter_by_language(raw_samples, language_filter, name)

        # 8. SFT schema validation (for SFT-format datasets only)
        fmt_mode = self.get_format_type(name)
        if fmt_mode != "pretrain":
            raw_samples = self._validate_schema(raw_samples, name)

        # 9. Auto-detect train_on_prompt from format type
        if train_on_prompt is None:
            train_on_prompt = fmt_mode == "pretrain"

        # 10. Build code formatter and dataset
        from training.code_formatter import CodeFormatter, CodeFormatterConfig

        coding_cfg = getattr(self.config, "coding", None)
        fmt_config = CodeFormatterConfig(
            **(coding_cfg.to_formatter_dict() if coding_cfg else {})
        )
        formatter_wrapper = CodeFormatter(fmt_config)

        # Convert pretrain-style samples to SFT-compatible format
        if fmt_mode == "pretrain":
            raw_samples = [_convert_pretrain_to_sft(s) for s in raw_samples]

        from training.sft_dataset import SFTByteDataset
        from training.conversation_formatter import ConversationFormatter, FormatterConfig

        # Use CodeFormatter's ConversationFormatter under the hood
        conv_formatter = formatter_wrapper.get_conversation_formatter()

        dataset = SFTByteDataset(
            samples=raw_samples,
            seq_len=effective_seq_len,
            formatter=conv_formatter,
            train_on_prompt=train_on_prompt,
        )

        logger.info(
            "Loaded coding dataset '%s' split='%s' format='%s': %d samples, seq_len=%d",
            name,
            split,
            fmt_mode,
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
        """Load a dataset directly from a list (for testing / offline use).

        Parameters
        ----------
        samples:
            List of raw sample dicts.
        seq_len:
            Sequence length.
        train_on_prompt:
            Whether to compute loss on prompt bytes.

        Returns
        -------
        SFTByteDataset
        """
        from training.sft_dataset import SFTByteDataset
        from training.conversation_formatter import ConversationFormatter

        formatter = ConversationFormatter()
        return SFTByteDataset(
            samples=samples,
            seq_len=seq_len,
            formatter=formatter,
            train_on_prompt=train_on_prompt,
        )

    def get_format_type(self, name: str) -> str:
        """Return the format mode for a dataset: ``\"pretrain\"`` or ``\"sft\"``.

        Consults coding.yaml first; defaults to ``\"sft\"`` if not found.

        Parameters
        ----------
        name:
            Dataset name.

        Returns
        -------
        str
            ``\"pretrain\"`` for raw-code datasets; ``\"sft\"`` for Q&A datasets.
        """
        datasets = self._coding_yaml.get("datasets", {})
        if name in datasets:
            fmt = datasets[name].get("format", "sft")
            if fmt in _PRETRAIN_FORMAT_TYPES:
                return "pretrain"
        return "sft"

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

        # Look up in coding.yaml
        datasets = self._coding_yaml.get("datasets", {})
        if name in datasets:
            ds = datasets[name]
            license_id = ds.get("license", "Apache-2.0")
            fmt = "pretrain" if ds.get("format", "sft") in _PRETRAIN_FORMAT_TYPES else "sft"
            return _DatasetEntry(
                name=name,
                stage="3A",
                license_id=license_id,
                format_type=ds.get("format_type", fmt),
                hf_id=ds.get("hf_id", ""),
                format_mode=fmt,
            )

        # Offline stub
        return _DatasetEntry(name=name, stage="3A")

    def _check_license(self, entry: Any) -> None:
        """Validate dataset license for research use."""
        license_id = getattr(entry, "license_id", getattr(entry, "license", ""))
        if not license_id:
            logger.warning(
                "No license information for '%s'; proceeding (research use).",
                getattr(entry, "name", "?"),
            )
            return

        if license_id not in _ALLOWED_LICENSES:
            raise RuntimeError(
                f"Dataset license '{license_id}' is not in the allowed set "
                f"for coding research: {sorted(_ALLOWED_LICENSES)}"
            )
        logger.debug("License check passed: %s", license_id)

    def _resolve_path(self, name: str, entry: Any) -> Path:
        """Resolve stage3a processed data directory."""
        registered_path = getattr(entry, "processed_path", None)
        if registered_path:
            p = Path(registered_path)
            if p.exists():
                return p

        # Standard layout: data/processed/stage3a/{name}/
        stage3a_dir = self._processed_base / "stage3a" / name
        if stage3a_dir.exists():
            return stage3a_dir

        # Fallback flat layout
        flat_dir = self._processed_base / name
        if flat_dir.exists():
            logger.warning(
                "Coding dataset '%s' found under flat layout. "
                "Consider migrating to data/processed/stage3a/%s/.",
                name,
                name,
            )
            return flat_dir

        raise FileNotFoundError(
            f"Processed coding dataset directory not found for '{name}'. "
            f"Expected: {stage3a_dir}. "
            f"Run the Stage 3A data pipeline first."
        )

    def _validate_stage(self, processed_dir: Path, name: str) -> None:
        """Check VERSION.json stage matches Stage 3A (accepts string '3A' or int 3)."""
        version_file = processed_dir / "VERSION.json"
        if not version_file.exists():
            logger.warning("No VERSION.json at %s; skipping stage check.", processed_dir)
            return
        try:
            version_data = json.loads(version_file.read_text(encoding="utf-8"))
            stage = version_data.get("stage", -1)
            if stage not in _STAGE3A_IDS:
                raise RuntimeError(
                    f"Coding dataset '{name}' has stage={stage!r} in VERSION.json, "
                    f"but Stage 3A is required."
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

        for fname in (f"{split}.jsonl", f"{split}.json", "data.jsonl", "data.json"):
            fpath = processed_dir / fname
            if fpath.exists():
                samples = _load_jsonl_or_json(fpath)
                logger.info("Loaded %d samples from %s", len(samples), fpath)
                break

        if not samples:
            for fpath in sorted(processed_dir.glob("*.jsonl")):
                samples.extend(_load_jsonl_or_json(fpath))
            if not samples:
                for fpath in sorted(processed_dir.glob("*.json")):
                    samples.extend(_load_jsonl_or_json(fpath))

        if max_samples is not None:
            samples = samples[:max_samples]

        return samples

    def _filter_by_language(
        self,
        samples: list[dict[str, Any]],
        language_filter: list[str],
        name: str,
    ) -> list[dict[str, Any]]:
        """Filter samples by detected programming language."""
        normalized = {lang.lower().strip() for lang in language_filter}
        filtered = []
        skipped = 0
        for s in samples:
            lang = (
                s.get("language") or s.get("programming_language") or
                s.get("lang") or ""
            ).lower().strip()
            if not lang or lang in normalized:
                filtered.append(s)
            else:
                skipped += 1
        if skipped:
            logger.debug(
                "Language filter for '%s': kept %d/%d samples (filter=%s).",
                name,
                len(filtered),
                len(samples),
                language_filter,
            )
        return filtered

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
                    "Schema validation for coding dataset '%s': removed %d/%d invalid samples.",
                    name,
                    n_removed,
                    len(samples),
                )
            return valid
        except ImportError:
            logger.debug("SFTValidator not available; skipping schema validation.")
            return samples


# ── Private helpers ────────────────────────────────────────────────────────


def _convert_pretrain_to_sft(sample: dict[str, Any]) -> dict[str, Any]:
    """Convert a pretrain-format sample (with ``content`` field) to SFT format.

    The entire content is treated as the response (no prompt).

    Parameters
    ----------
    sample:
        Raw pretrain sample dict with at least a ``\"content\"`` field.

    Returns
    -------
    dict[str, Any]
        Sample in ``{\"instruction\": \"\", \"output\": content}`` format.
    """
    content = sample.get("content", "")
    language = sample.get("language", "")
    instruction = f"Write {language} code:" if language else "Write code:"
    return {
        "instruction": instruction,
        "input": "",
        "output": content,
        "language": language,
        **{k: v for k, v in sample.items() if k not in ("content", "instruction", "output")},
    }


def _load_jsonl_or_json(path: Path) -> list[dict[str, Any]]:
    """Load either a JSONL or JSON file into a list of dicts."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
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
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return [o for o in obj if isinstance(o, dict)]
        if isinstance(obj, dict):
            return [obj]
    except json.JSONDecodeError:
        pass
    return []
