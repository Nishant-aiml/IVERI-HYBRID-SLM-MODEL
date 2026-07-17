# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Tests for dataset fingerprinting and integrity checks."""

from __future__ import annotations

from pathlib import Path
import pytest

from data.pipeline.fingerprint import DatasetFingerprintGenerator


def test_fingerprint_generation_and_integrity(tmp_path: Path) -> None:
    """Test generating a fingerprint and verifying it remains identical on identical data."""
    generator = DatasetFingerprintGenerator()

    docs = [
        {"text": "This is sample document 1.", "quality_score": 0.8},
        {"text": "This is sample document 2.", "quality_score": 0.4},
    ]

    fprint_1 = generator.generate(
        dataset_name="test_fprint",
        version="v1",
        docs=docs,
        sha256_manifest="manifest-sha-abc",
        duplicates_removed=1,
    )

    assert fprint_1.doc_count == 2
    assert fprint_1.byte_count > 0
    assert fprint_1.fingerprint_hash != ""

    # Re-generating on identical data should produce identical hash
    fprint_2 = generator.generate(
        dataset_name="test_fprint",
        version="v1",
        docs=docs,
        sha256_manifest="manifest-sha-abc",
        duplicates_removed=1,
    )
    assert fprint_1.fingerprint_hash == fprint_2.fingerprint_hash

    # Modified data should produce different hash
    docs_modified = [
        {"text": "This is sample document 1.", "quality_score": 0.8},
        {"text": "This is a modified sample document 2.", "quality_score": 0.4},
    ]
    fprint_mod = generator.generate(
        dataset_name="test_fprint",
        version="v1",
        docs=docs_modified,
        sha256_manifest="manifest-sha-abc",
        duplicates_removed=1,
    )
    assert fprint_1.fingerprint_hash != fprint_mod.fingerprint_hash

    # Test file writing and verification
    fprint_file = tmp_path / "fingerprint.json"
    fprint_file.write_text(fprint_1.to_json(), encoding="utf-8")
    assert generator.verify(tmp_path, fprint_1)
