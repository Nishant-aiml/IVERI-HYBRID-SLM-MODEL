# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Artifacts Dependency Graph tracking validation files, reports, and paper dependencies."""

from __future__ import annotations

import logging
import hashlib
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ArtifactsGraphManager:
    """Tracks research file pipelines and validates hash signatures and paths."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.dependencies: dict[str, list[str]] = {}

    def add_artifact(self, name: str, file_path: str | Path, depends_on: list[str] | None = None) -> None:
        """Register an artifact with its location and source dependencies."""
        self.nodes[name] = {
            "path": Path(file_path),
            "expected_hash": None,
        }
        self.dependencies[name] = depends_on or []

    def set_expected_hash(self, name: str, expected_hash: str) -> None:
        if name in self.nodes:
            self.nodes[name]["expected_hash"] = expected_hash

    def verify_integrity(self) -> dict[str, Any]:
        """Checks if all artifact paths exist and verify checksum hashes.

        Returns:
            dict[str, Any]: Verification outcome and lists of missing/mismatched assets.
        """
        missing_files = []
        hash_mismatches = []
        violations = []

        for name, node in self.nodes.items():
            path = node["path"]
            if not path.exists():
                missing_files.append(name)
                violations.append(f"Missing file: {name} (Expected at: {path})")
                continue

            # Verify checksum hash if specified
            expected = node["expected_hash"]
            if expected:
                try:
                    with open(path, "rb") as f:
                        file_hash = hashlib.sha256(f.read()).hexdigest()
                    if file_hash != expected:
                        hash_mismatches.append(name)
                        violations.append(f"Hash mismatch for {name}: Expected {expected}, got {file_hash}")
                except Exception as e:
                    violations.append(f"Failed to read hash for {name}: {e}")

        # Check dependency resolution (dependent files should exist if target exists)
        for name, deps in self.dependencies.items():
            path = self.nodes[name]["path"]
            if path.exists():
                for dep in deps:
                    dep_path = self.nodes[dep]["path"]
                    if not dep_path.exists():
                        violations.append(f"Dependency violation: Artifact '{name}' exists but its dependency '{dep}' is missing.")

        return {
            "ok": len(violations) == 0,
            "violations": violations,
            "missing_files": missing_files,
            "hash_mismatches": hash_mismatches,
        }

    def render_graph(self) -> str:
        """Render a clean text layout of the dependency graph."""
        lines = ["# Research Artifact Dependency Graph\n"]
        for name, deps in self.dependencies.items():
            path = self.nodes[name]["path"]
            status = "✔ EXISTS" if path.exists() else "✖ MISSING"
            lines.append(f"- **{name}** ({status})")
            if deps:
                for dep in deps:
                    lines.append(f"  └── depends on: {dep}")
        return "\n".join(lines)
