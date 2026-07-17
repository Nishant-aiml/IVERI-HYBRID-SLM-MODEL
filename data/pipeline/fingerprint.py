# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Dataset fingerprinting utility for Phase 8 data certification.

Uniquely fingerprints dataset instances based on comprehensive statistical
metrics to guarantee reproducibility and immutability.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DatasetFingerprint:
    """Statistical fingerprint uniquely identifying a dataset instance."""

    dataset_name: str
    version: str
    doc_count: int
    byte_count: int
    avg_doc_length: float
    median_doc_length: float
    byte_entropy_mean: float
    char_entropy_mean: float
    vocabulary_size: int            # Unique byte bigrams as proxy
    duplicates_removed: int
    quality_histogram: dict[str, int]
    sha256_manifest: str
    fingerprint_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Convert fingerprint to nested dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "version": self.version,
            "doc_count": self.doc_count,
            "byte_count": self.byte_count,
            "avg_doc_length": round(self.avg_doc_length, 4),
            "median_doc_length": round(self.median_doc_length, 4),
            "byte_entropy_mean": round(self.byte_entropy_mean, 4),
            "char_entropy_mean": round(self.char_entropy_mean, 4),
            "vocabulary_size": self.vocabulary_size,
            "duplicates_removed": self.duplicates_removed,
            "quality_histogram": self.quality_histogram,
            "sha256_manifest": self.sha256_manifest,
            "fingerprint_hash": self.fingerprint_hash,
        }

    def to_json(self) -> str:
        """Convert fingerprint to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class DatasetFingerprintGenerator:
    """Generates and verifies dataset statistical fingerprints."""

    @staticmethod
    def compute_hash(data_dict: dict[str, Any]) -> str:
        """Compute stable SHA-256 hash of a dictionary."""
        serialized = json.dumps(data_dict, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def generate(
        self,
        dataset_name: str,
        version: str,
        docs: list[dict],
        sha256_manifest: str,
        duplicates_removed: int = 0,
        text_field: str = "text",
    ) -> DatasetFingerprint:
        """Generate a DatasetFingerprint from certified in-memory documents."""
        doc_count = len(docs)
        if doc_count == 0:
            return DatasetFingerprint(
                dataset_name=dataset_name,
                version=version,
                doc_count=0,
                byte_count=0,
                avg_doc_length=0.0,
                median_doc_length=0.0,
                byte_entropy_mean=0.0,
                char_entropy_mean=0.0,
                vocabulary_size=0,
                duplicates_removed=duplicates_removed,
                quality_histogram={},
                sha256_manifest=sha256_manifest,
                fingerprint_hash="",
            )

        lengths = []
        total_bytes = 0
        bigrams_seen: set[bytes] = set()
        byte_entropies = []
        char_entropies = []
        quality_counts = {
            "0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0
        }

        # Lazy imports of helper functions
        from data.pipeline.entropy_analyzer import _shannon_entropy

        for doc in docs:
            text = doc.get(text_field, "")
            if not isinstance(text, str):
                text = ""

            encoded = text.encode("utf-8")
            total_bytes += len(encoded)
            lengths.append(len(text))

            # Compute entropies
            byte_entropies.append(_shannon_entropy(encoded))
            char_entropies.append(_shannon_entropy(text))

            # Extract unique bigrams as vocabulary proxy
            for i in range(len(encoded) - 1):
                bigrams_seen.add(encoded[i : i + 2])

            # Classify quality
            qs = doc.get("quality_score", 1.0)
            if qs < 0.2:
                quality_counts["0.0-0.2"] += 1
            elif qs < 0.4:
                quality_counts["0.2-0.4"] += 1
            elif qs < 0.6:
                quality_counts["0.4-0.6"] += 1
            elif qs < 0.8:
                quality_counts["0.6-0.8"] += 1
            else:
                quality_counts["0.8-1.0"] += 1

        lengths.sort()
        avg_len = total_bytes / doc_count
        median_len = lengths[doc_count // 2]
        mean_byte_ent = sum(byte_entropies) / doc_count
        mean_char_ent = sum(char_entropies) / doc_count

        raw_dict = {
            "dataset_name": dataset_name,
            "version": version,
            "doc_count": doc_count,
            "byte_count": total_bytes,
            "avg_doc_length": round(avg_len, 4),
            "median_doc_length": round(median_len, 4),
            "byte_entropy_mean": round(mean_byte_ent, 4),
            "char_entropy_mean": round(mean_char_ent, 4),
            "vocabulary_size": len(bigrams_seen),
            "duplicates_removed": duplicates_removed,
            "quality_histogram": quality_counts,
            "sha256_manifest": sha256_manifest,
        }
        fprint_hash = self.compute_hash(raw_dict)

        return DatasetFingerprint(
            dataset_name=dataset_name,
            version=version,
            doc_count=doc_count,
            byte_count=total_bytes,
            avg_doc_length=avg_len,
            median_doc_length=median_len,
            byte_entropy_mean=mean_byte_ent,
            char_entropy_mean=mean_char_ent,
            vocabulary_size=len(bigrams_seen),
            duplicates_removed=duplicates_removed,
            quality_histogram=quality_counts,
            sha256_manifest=sha256_manifest,
            fingerprint_hash=fprint_hash,
        )

    def verify(self, path: Path, expected: DatasetFingerprint) -> bool:
        """Verify fingerprint file matches expected fingerprint hash."""
        fprint_path = path / "fingerprint.json"
        if not fprint_path.exists():
            logger.warning("Fingerprint file not found: %s", fprint_path)
            return False
        try:
            with open(fprint_path, encoding="utf-8") as f:
                data = json.load(f)
            hash_actual = data.get("fingerprint_hash", "")
            return hash_actual == expected.fingerprint_hash
        except Exception as e:
            logger.error("Fingerprint verification failed: %s", e)
            return False
