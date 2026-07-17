# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Immutability enforcement for IVERI CORE certified datasets.

Writes a cryptographically signed freeze certificate alongside a .FROZEN
sentinel file to prevent accidental modification of certified datasets.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=False, slots=True)
class FreezeCertificate:
    """Immutable record of a certified dataset freeze event."""

    version: str                       # e.g., "IVERI_DATASET_v1.0"
    dataset_name: str
    frozen_at: str                     # ISO timestamp
    frozen_by: str                     # "Phase8-CertificationPipeline-v1"
    certified_path: str
    sha256_manifest_hash: str          # Hash of the SHA-256 manifest file
    doc_count: int
    byte_count: int
    certification_stages: int          # 24
    pipeline_version: str              # "1.0.0"
    # Extended provenance fields
    git_commit: str | None = None      # from subprocess git rev-parse HEAD
    python_version: str = ""           # from sys.version
    hf_revision: str | None = None
    license: str = ""
    license_url: str = ""
    config_hash: str = ""              # SHA-256 of config dict
    recipe_hash: str | None = None     # SHA-256 of recipe YAML
    stages_passed: int = 24

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "version": self.version,
            "dataset_name": self.dataset_name,
            "frozen_at": self.frozen_at,
            "frozen_by": self.frozen_by,
            "certified_path": self.certified_path,
            "sha256_manifest_hash": self.sha256_manifest_hash,
            "doc_count": self.doc_count,
            "byte_count": self.byte_count,
            "certification_stages": self.certification_stages,
            "pipeline_version": self.pipeline_version,
            "git_commit": self.git_commit,
            "python_version": self.python_version,
            "hf_revision": self.hf_revision,
            "license": self.license,
            "license_url": self.license_url,
            "config_hash": self.config_hash,
            "recipe_hash": self.recipe_hash,
            "stages_passed": self.stages_passed,
        }

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=4, sort_keys=True)

    def compute_certificate_hash(self) -> str:
        """SHA-256 of the freeze certificate JSON itself."""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()


class FreezeManager:
    """Manages freeze certificates and frozen-dataset sentinels."""

    FREEZE_FILE = "FREEZE_CERTIFICATE.json"
    LOCK_FILE = ".FROZEN"  # Presence of this file = dataset is frozen

    def freeze(
        self,
        dataset_path: Path,
        version: str,
        doc_count: int,
        byte_count: int,
        sha256_manifest_hash: str,
        license: str = "",
        license_url: str = "",
        config_hash: str = "",
        recipe_hash: str | None = None,
        hf_revision: str | None = None,
    ) -> FreezeCertificate:
        """Write FREEZE_CERTIFICATE.json and .FROZEN lock file.

        Populates git_commit via subprocess, python_version via sys.version.
        """
        dataset_path = Path(dataset_path)
        dataset_path.mkdir(parents=True, exist_ok=True)

        git_commit = self._get_git_commit()
        python_version = sys.version

        cert = FreezeCertificate(
            version=version,
            dataset_name=dataset_path.name,
            frozen_at=datetime.now(timezone.utc).isoformat(),
            frozen_by="Phase8-CertificationPipeline-v1",
            certified_path=str(dataset_path.resolve()),
            sha256_manifest_hash=sha256_manifest_hash,
            doc_count=doc_count,
            byte_count=byte_count,
            certification_stages=24,
            pipeline_version="1.0.0",
            git_commit=git_commit,
            python_version=python_version,
            hf_revision=hf_revision,
            license=license,
            license_url=license_url,
            config_hash=config_hash,
            recipe_hash=recipe_hash,
            stages_passed=24,
        )

        # Write FREEZE_CERTIFICATE.json
        freeze_path = dataset_path / self.FREEZE_FILE
        cert_json = cert.to_json()
        freeze_path.write_text(cert_json, encoding="utf-8")
        logger.info(f"Wrote freeze certificate to {freeze_path}")

        # Write .FROZEN sentinel (contains certificate hash)
        cert_hash = cert.compute_certificate_hash()
        lock_path = dataset_path / self.LOCK_FILE
        lock_path.write_text(
            f"FROZEN\ncertificate_hash={cert_hash}\nfrozen_at={cert.frozen_at}\n",
            encoding="utf-8",
        )
        logger.info(f"Dataset frozen: {lock_path}")

        return cert

    def is_frozen(self, dataset_path: Path) -> bool:
        """Check if .FROZEN file exists in dataset_path."""
        return (Path(dataset_path) / self.LOCK_FILE).exists()

    def verify_frozen(self, dataset_path: Path) -> tuple[bool, str]:
        """Verify freeze certificate integrity. Returns (valid, message)."""
        dataset_path = Path(dataset_path)
        freeze_file = dataset_path / self.FREEZE_FILE
        lock_file = dataset_path / self.LOCK_FILE

        if not lock_file.exists():
            return False, f"Dataset is not frozen (.FROZEN not found in {dataset_path})"

        if not freeze_file.exists():
            return False, f"Freeze certificate missing: {freeze_file}"

        try:
            cert_json = freeze_file.read_text(encoding="utf-8")
            data = json.loads(cert_json)

            # Reconstruct FreezeCertificate to recompute hash
            cert = FreezeCertificate(
                version=data.get("version", ""),
                dataset_name=data.get("dataset_name", ""),
                frozen_at=data.get("frozen_at", ""),
                frozen_by=data.get("frozen_by", ""),
                certified_path=data.get("certified_path", ""),
                sha256_manifest_hash=data.get("sha256_manifest_hash", ""),
                doc_count=data.get("doc_count", 0),
                byte_count=data.get("byte_count", 0),
                certification_stages=data.get("certification_stages", 24),
                pipeline_version=data.get("pipeline_version", "1.0.0"),
                git_commit=data.get("git_commit"),
                python_version=data.get("python_version", ""),
                hf_revision=data.get("hf_revision"),
                license=data.get("license", ""),
                license_url=data.get("license_url", ""),
                config_hash=data.get("config_hash", ""),
                recipe_hash=data.get("recipe_hash"),
                stages_passed=data.get("stages_passed", 24),
            )

            expected_hash = cert.compute_certificate_hash()
            lock_contents = lock_file.read_text(encoding="utf-8")

            if f"certificate_hash={expected_hash}" in lock_contents:
                return True, f"Freeze certificate valid (hash={expected_hash[:16]}...)"
            else:
                return False, "Freeze certificate hash mismatch — dataset may have been tampered with."

        except Exception as e:
            return False, f"Error verifying freeze certificate: {e}"

    def require_not_frozen(self, dataset_path: Path, operation: str) -> None:
        """Raise RuntimeError if dataset is frozen and operation would modify it."""
        if self.is_frozen(dataset_path):
            raise RuntimeError(
                f"Operation '{operation}' is not permitted: dataset at '{dataset_path}' is frozen. "
                f"Remove '{self.LOCK_FILE}' and '{self.FREEZE_FILE}' to unfreeze (destructive action)."
            )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _get_git_commit() -> str | None:
        """Retrieve the current git HEAD commit hash via subprocess."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logger.debug(f"Could not retrieve git commit: {e}")
        return None
