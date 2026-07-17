# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Manifest-based curriculum builder for IVERI CORE training phases.

Builds curriculum manifests (JSON) that reference certified datasets by path.
NO DATA DUPLICATION — certified datasets are stored once and referenced by path.
Supports YAML recipe loading for flexible phase configuration.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Optional PyYAML import
try:
    import yaml as _yaml

    _YAML_AVAILABLE = True
except ImportError:
    _yaml = None  # type: ignore[assignment]
    _YAML_AVAILABLE = False


# ── Data Models ───────────────────────────────────────────────────────────────


@dataclass(frozen=False, slots=True)
class CurriculumRecipe:
    """A named curriculum recipe loaded from a YAML file."""

    name: str
    version: str
    description: str
    phase: str                        # "A", "B", "C", or custom
    datasets: dict[str, float]        # dataset_name -> weight (not normalized)
    min_datasets_required: int = 2
    fallback_recipe: str | None = None

    def normalized_weights(self) -> dict[str, float]:
        """Return weights that sum to 1.0."""
        total = sum(self.datasets.values())
        if total == 0:
            return {k: 0.0 for k in self.datasets}
        return {k: v / total for k, v in self.datasets.items()}


@dataclass(frozen=False, slots=True)
class CurriculumManifest:
    """Manifest describing a curriculum phase with dataset weights and paths."""

    version: str                              # "1.0"
    phase: str                                # "A", "B", or "C"
    description: str                          # Human-readable description
    created_at: str
    weights: dict[str, float]                 # dataset_name -> weight (sum to 1.0)
    certified_paths: dict[str, str]           # dataset_name -> absolute path
    freeze_hashes: dict[str, str]             # dataset_name -> freeze cert hash
    available_datasets: list[str]             # datasets actually available
    total_certified_gb: float
    curriculum_manifest_hash: str             # SHA-256 of this manifest JSON

    def adjusted_weights(self) -> dict[str, float]:
        """Return weights renormalized to only available_datasets (sum to 1.0)."""
        avail = {k: v for k, v in self.weights.items() if k in self.available_datasets}
        total = sum(avail.values())
        if total == 0:
            return avail
        return {k: v / total for k, v in avail.items()}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "version": self.version,
            "phase": self.phase,
            "description": self.description,
            "created_at": self.created_at,
            "weights": self.weights,
            "certified_paths": self.certified_paths,
            "freeze_hashes": self.freeze_hashes,
            "available_datasets": self.available_datasets,
            "total_certified_gb": self.total_certified_gb,
            "curriculum_manifest_hash": self.curriculum_manifest_hash,
        }

    def save(self, path: Path) -> None:
        """Save manifest JSON to path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)
        logger.info(f"Curriculum manifest saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "CurriculumManifest":
        """Load CurriculumManifest from JSON file."""
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            version=data.get("version", "1.0"),
            phase=data.get("phase", "A"),
            description=data.get("description", ""),
            created_at=data.get("created_at", ""),
            weights=data.get("weights", {}),
            certified_paths=data.get("certified_paths", {}),
            freeze_hashes=data.get("freeze_hashes", {}),
            available_datasets=data.get("available_datasets", []),
            total_certified_gb=data.get("total_certified_gb", 0.0),
            curriculum_manifest_hash=data.get("curriculum_manifest_hash", ""),
        )


@dataclass(frozen=False, slots=True)
class CurriculumResult:
    """Result of building a curriculum manifest for a phase."""

    phase: str
    manifest: CurriculumManifest
    manifest_path: Path
    available_datasets: list[str]
    missing_datasets: list[str]       # weights specified but data not found
    success: bool
    message: str


# ── Recipe Loader ─────────────────────────────────────────────────────────────


class RecipeLoader:
    """Loads and validates curriculum recipes from YAML files."""

    def load(self, recipe_name: str, recipes_dir: Path) -> CurriculumRecipe:
        """Load a YAML recipe file from recipes_dir/{recipe_name}.yaml.

        Falls back to a minimal default recipe if PyYAML is unavailable.
        """
        recipes_dir = Path(recipes_dir)
        recipe_path = recipes_dir / f"{recipe_name}.yaml"

        if not recipe_path.exists():
            raise FileNotFoundError(f"Recipe file not found: {recipe_path}")

        if not _YAML_AVAILABLE:
            logger.warning(
                "PyYAML not installed — loading recipe from YAML is unavailable. "
                "Run 'pip install pyyaml' to enable. Returning empty recipe."
            )
            return CurriculumRecipe(
                name=recipe_name,
                version="0.0",
                description="PyYAML not available",
                phase="A",
                datasets={},
            )

        with open(recipe_path, encoding="utf-8") as f:
            raw: dict[str, Any] = _yaml.safe_load(f) or {}

        return CurriculumRecipe(
            name=raw.get("name", recipe_name),
            version=str(raw.get("version", "1.0")),
            description=raw.get("description", ""),
            phase=str(raw.get("phase", "A")),
            datasets=raw.get("datasets", {}),
            min_datasets_required=int(raw.get("min_datasets_required", 2)),
            fallback_recipe=raw.get("fallback_recipe"),
        )

    def list_available(self, recipes_dir: Path) -> list[str]:
        """List all .yaml files in recipes_dir (without extension)."""
        recipes_dir = Path(recipes_dir)
        if not recipes_dir.exists():
            return []
        return sorted(p.stem for p in recipes_dir.glob("*.yaml"))

    def validate(self, recipe: CurriculumRecipe) -> list[str]:
        """Validate a CurriculumRecipe. Returns list of error strings (empty = valid)."""
        errors: list[str] = []

        if not recipe.name:
            errors.append("Recipe 'name' is required.")

        if not recipe.phase:
            errors.append("Recipe 'phase' is required (e.g., 'A', 'B', 'C').")

        if not recipe.datasets:
            errors.append("Recipe 'datasets' must be a non-empty dict of {name: weight}.")
        else:
            for ds_name, weight in recipe.datasets.items():
                if not isinstance(weight, (int, float)) or weight < 0:
                    errors.append(f"Dataset '{ds_name}' has invalid weight: {weight!r} (must be >= 0).")

            total = sum(recipe.datasets.values())
            if total <= 0:
                errors.append("Sum of dataset weights must be > 0.")

        if recipe.min_datasets_required < 1:
            errors.append("'min_datasets_required' must be >= 1.")

        return errors


# ── Curriculum Builder ────────────────────────────────────────────────────────


class CurriculumBuilder:
    """Builds curriculum manifests for training phases A, B, and C.

    No data is copied — only manifests (JSON pointers) are created.
    """

    def __init__(
        self,
        config: Any,
        certified_base_dir: Path,
        curriculum_dir: Path,
    ) -> None:
        self._config = config
        self.certified_base_dir = Path(certified_base_dir)
        self.curriculum_dir = Path(curriculum_dir)
        self.curriculum_dir.mkdir(parents=True, exist_ok=True)
        self._recipe_loader = RecipeLoader()

    def build_phase(
        self,
        phase: str,
        weights: dict[str, float],
        dataset_specs: dict[str, Any],
    ) -> CurriculumResult:
        """Build curriculum manifest for a phase. No data copied.

        Steps:
        1. Find which datasets are actually available (certified_base_dir/{name}/ exists)
        2. Build CurriculumManifest with available datasets
        3. Log which datasets are missing
        4. Save manifest JSON
        5. Return CurriculumResult
        """
        available: list[str] = []
        missing: list[str] = []
        certified_paths: dict[str, str] = {}
        freeze_hashes: dict[str, str] = {}
        total_bytes = 0

        for ds_name in weights:
            ds_path = self.certified_base_dir / ds_name
            if ds_path.exists():
                available.append(ds_name)
                certified_paths[ds_name] = str(ds_path.resolve())
                # Load freeze hash if available
                freeze_file = ds_path / "FREEZE_CERTIFICATE.json"
                if freeze_file.exists():
                    try:
                        with open(freeze_file, encoding="utf-8") as f:
                            cert_data = json.load(f)
                        freeze_hashes[ds_name] = cert_data.get("sha256_manifest_hash", "")
                    except Exception as e:
                        logger.warning(f"Could not read freeze cert for '{ds_name}': {e}")
                # Estimate size
                for jl in ds_path.glob("**/*.jsonl"):
                    total_bytes += jl.stat().st_size
            else:
                missing.append(ds_name)
                logger.warning(
                    f"Curriculum phase {phase}: dataset '{ds_name}' not found at "
                    f"{ds_path} — excluded from manifest."
                )

        # Normalize weights to available datasets
        avail_weights = {k: v for k, v in weights.items() if k in available}
        total_w = sum(avail_weights.values())
        norm_weights = {k: v / total_w for k, v in avail_weights.items()} if total_w > 0 else avail_weights
        total_certified_gb = total_bytes / (1024 ** 3)

        # Build manifest (hash computed after serialization)
        manifest = CurriculumManifest(
            version="1.0",
            phase=phase,
            description=f"IVERI Phase {phase} curriculum manifest",
            created_at=datetime.now(timezone.utc).isoformat(),
            weights=norm_weights,
            certified_paths=certified_paths,
            freeze_hashes=freeze_hashes,
            available_datasets=available,
            total_certified_gb=total_certified_gb,
            curriculum_manifest_hash="",  # placeholder — computed below
        )

        # Compute hash of manifest content (before writing hash field itself)
        manifest_dict = manifest.to_dict()
        manifest_json_str = json.dumps(manifest_dict, sort_keys=True)
        manifest.curriculum_manifest_hash = hashlib.sha256(
            manifest_json_str.encode("utf-8")
        ).hexdigest()

        # Save manifest
        manifest_path = self.curriculum_dir / f"phase_{phase.lower()}_manifest.json"
        manifest.save(manifest_path)

        success = len(available) > 0
        message = (
            f"Phase {phase}: {len(available)}/{len(weights)} datasets available. "
            f"Missing: {missing}" if missing else
            f"Phase {phase}: all {len(available)} datasets available."
        )
        logger.info(message)

        return CurriculumResult(
            phase=phase,
            manifest=manifest,
            manifest_path=manifest_path,
            available_datasets=available,
            missing_datasets=missing,
            success=success,
            message=message,
        )

    def build_all(self, config: Any) -> dict[str, CurriculumResult]:
        """Build manifests for phases A, B, and C from config."""
        results: dict[str, CurriculumResult] = {}

        # Extract phase configs from config object
        phase_configs = getattr(config, "curriculum_phases", {})
        if not phase_configs:
            # Fallback: look for phase_a_weights, phase_b_weights, phase_c_weights attrs
            for phase in ("A", "B", "C"):
                phase_key = f"phase_{phase.lower()}_weights"
                weights = getattr(config, phase_key, None)
                if weights:
                    results[phase] = self.build_phase(
                        phase=phase,
                        weights=weights,
                        dataset_specs={},
                    )
        else:
            for phase, phase_cfg in phase_configs.items():
                weights = getattr(phase_cfg, "weights", {}) if not isinstance(phase_cfg, dict) else phase_cfg.get("weights", {})
                results[phase] = self.build_phase(
                    phase=phase,
                    weights=weights,
                    dataset_specs={},
                )

        return results

    def build_from_recipe(
        self,
        recipe: CurriculumRecipe,
    ) -> CurriculumResult:
        """Build a curriculum manifest from a loaded CurriculumRecipe."""
        return self.build_phase(
            phase=recipe.phase,
            weights=recipe.normalized_weights(),
            dataset_specs={},
        )

    def verify_manifest(self, manifest_path: Path) -> tuple[bool, str]:
        """Verify manifest integrity (hash check, paths exist)."""
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            return False, f"Manifest file not found: {manifest_path}"

        try:
            manifest = CurriculumManifest.load(manifest_path)
        except Exception as e:
            return False, f"Failed to load manifest: {e}"

        # Verify stored hash
        stored_hash = manifest.curriculum_manifest_hash
        check = manifest.to_dict()
        check["curriculum_manifest_hash"] = ""
        computed = hashlib.sha256(
            json.dumps(check, sort_keys=True).encode("utf-8")
        ).hexdigest()

        if stored_hash and stored_hash != computed:
            return (
                False,
                f"Manifest hash mismatch: stored={stored_hash[:16]}... "
                f"computed={computed[:16]}...",
            )

        # Verify certified paths exist
        missing = [
            ds for ds, path in manifest.certified_paths.items()
            if not Path(path).exists()
        ]
        if missing:
            return False, f"Missing certified paths for datasets: {missing}"

        return True, f"Manifest valid — {len(manifest.available_datasets)} datasets, hash OK."

    def generate_curriculum_report(
        self,
        results: dict[str, CurriculumResult],
        output_path: Path,
    ) -> None:
        """Write reports/phase_8/07_curriculum_report.md and .json."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = [
            "# IVERI CORE — Phase 8 Curriculum Report",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Phase Summary",
            "",
            "| Phase | Available Datasets | Missing Datasets | Total GB | Success |",
            "|---|---|---|---|---|",
        ]

        report_data: dict[str, Any] = {"phases": {}}

        for phase, result in sorted(results.items()):
            m = result.manifest
            status = "✅" if result.success else "❌"
            lines.append(
                f"| {phase} | {len(result.available_datasets)} "
                f"| {len(result.missing_datasets)} "
                f"| {m.total_certified_gb:.4f} "
                f"| {status} |"
            )
            report_data["phases"][phase] = {
                "available": result.available_datasets,
                "missing": result.missing_datasets,
                "total_certified_gb": m.total_certified_gb,
                "success": result.success,
                "manifest_path": str(result.manifest_path),
                "message": result.message,
            }

        for phase, result in sorted(results.items()):
            m = result.manifest
            lines += [
                "",
                f"## Phase {phase}",
                "",
                f"**Status**: {'✅ Available' if result.success else '❌ No data'}",
                f"**Manifest**: `{result.manifest_path}`",
                f"**Message**: {result.message}",
                "",
                "### Dataset Weights",
                "",
                "| Dataset | Weight | Certified Path | Available |",
                "|---|---|---|---|",
            ]
            for ds, w in m.weights.items():
                avail = "✅" if ds in m.available_datasets else "❌"
                cert_path = m.certified_paths.get(ds, "—")
                lines.append(f"| {ds} | {w:.4f} | `{cert_path}` | {avail} |")

            if result.missing_datasets:
                lines += [
                    "",
                    f"> ⚠️ Missing datasets: {', '.join(result.missing_datasets)}",
                ]

        output_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Curriculum report written to {output_path}")

        # Write JSON counterpart
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4)
        logger.info(f"Curriculum report JSON written to {json_path}")
