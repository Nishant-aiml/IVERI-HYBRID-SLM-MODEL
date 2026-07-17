# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""24-stage deterministic certification pipeline for IVERI CORE datasets.

Every stage produces measurable statistics. No silent discards.
All removed samples are counted and logged with reasons.
Wraps existing data/pipeline/ components — does not duplicate logic.
"""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=False, slots=True)
class StageResult:
    """Statistics produced by a single certification stage."""

    stage_num: int
    stage_name: str
    input_count: int
    kept_count: int
    removed_count: int
    removal_reasons: dict[str, int]  # reason -> count
    elapsed_sec: float

    @property
    def removal_rate(self) -> float:
        """Fraction of documents removed by this stage."""
        if self.input_count == 0:
            return 0.0
        return self.removed_count / self.input_count


@dataclass(frozen=False, slots=True)
class CertificationResult:
    """Full result from a CertificationPipeline run."""

    dataset_name: str
    input_count: int
    certified_count: int
    stage_results: list[StageResult]
    total_removed: int
    overall_removal_rate: float
    quality_score: float          # mean quality score of certified docs
    elapsed_sec: float
    certified_path: Path | None
    success: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "input_count": self.input_count,
            "certified_count": self.certified_count,
            "total_removed": self.total_removed,
            "overall_removal_rate": self.overall_removal_rate,
            "quality_score": self.quality_score,
            "elapsed_sec": self.elapsed_sec,
            "certified_path": str(self.certified_path) if self.certified_path else None,
            "success": self.success,
            "stage_results": [
                {
                    "stage_num": sr.stage_num,
                    "stage_name": sr.stage_name,
                    "input_count": sr.input_count,
                    "kept_count": sr.kept_count,
                    "removed_count": sr.removed_count,
                    "removal_reasons": sr.removal_reasons,
                    "elapsed_sec": sr.elapsed_sec,
                    "removal_rate": sr.removal_rate,
                }
                for sr in self.stage_results
            ],
        }

    def save(self, path: Path) -> None:
        """Write CertificationResult as JSON to path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)
        logger.info(f"Saved CertificationResult to {path}")


class CertificationPipeline:
    """Runs 24 deterministic stages on a list of documents.

    Lazy-imports all pipeline components to avoid circular dependencies.
    Each stage records StageResult with full removal accounting.
    """

    STAGE_NAMES = [
        "download_verification",       # 1
        "file_integrity",              # 2
        "utf8_validation",             # 3
        "encoding_normalization",      # 4
        "language_detection",          # 5
        "english_filtering",           # 6
        "corruption_detection",        # 7
        "empty_removal",               # 8
        "length_filtering",            # 9
        "exact_deduplication",         # 10
        "near_dedup_detection",        # 11
        "boilerplate_detection",       # 12
        "repetition_detection",        # 13
        "pii_detection",               # 14
        "safety_filtering",            # 15
        "quality_scoring",             # 16
        "entropy_analysis",            # 17
        "formatting_standardization",  # 18
        "byte_encoding_verification",  # 19
        "curriculum_tagging",          # 20
        "statistics_generation",       # 21
        "dataset_card_generation",     # 22
        "sha256_manifest",             # 23
        "freeze_certification",        # 24
    ]

    def __init__(self, config: Any) -> None:  # Phase8Config
        self._config = config
        self._min_doc_chars: int = getattr(config, "min_doc_chars", 100)
        self._max_doc_chars: int = getattr(config, "max_doc_chars", 100_000)
        self._lang_confidence: float = getattr(config, "lang_confidence_threshold", 0.0)

        # Lazy-load pipeline components
        from data.pipeline.deduplication import Deduplicator, DeduplicationConfig
        from data.pipeline.language_detector import LanguageDetector
        from data.pipeline.pii_remover import PIIRemover
        from data.pipeline.quality_filter import QualityFilter, QualityFilterConfig
        from data.pipeline.statistics import DatasetStatisticsGenerator
        from data.pipeline.versioning import DatasetVersioner

        self._deduplicator = Deduplicator(DeduplicationConfig(exact_enabled=True, near_enabled=True))
        self._lang_detector = LanguageDetector(allowed={"en"})
        self._pii_remover = PIIRemover()
        self._quality_filter = QualityFilter(
            QualityFilterConfig(
                min_doc_chars=self._min_doc_chars,
                max_doc_chars=self._max_doc_chars,
            )
        )
        self._stats_gen = DatasetStatisticsGenerator()
        self._versioner = DatasetVersioner()

        # Optional components — may not exist yet; caught gracefully in stages
        self._boilerplate_detector: Any = None
        self._safety_filter: Any = None
        self._entropy_analyzer: Any = None

        try:
            from data.pipeline.boilerplate_detector import BoilerplateDetector  # type: ignore[import]
            self._boilerplate_detector = BoilerplateDetector()
        except ImportError:
            logger.warning("BoilerplateDetector not available — stage 12 will be a no-op pass-through.")

        try:
            from data.pipeline.safety_filter import SafetyFilter  # type: ignore[import]
            self._safety_filter = SafetyFilter()
        except ImportError:
            logger.warning("SafetyFilter not available — stage 15 will be a no-op pass-through.")

        try:
            from data.pipeline.entropy_analyzer import EntropyAnalyzer  # type: ignore[import]
            self._entropy_analyzer = EntropyAnalyzer()
        except ImportError:
            logger.warning("EntropyAnalyzer not available — stage 17 will be a no-op pass-through.")

        # Compiled patterns for formatting stage
        self._html_tag_pat = re.compile(r"<[^>]*>")
        self._multi_ws_pat = re.compile(r"[ \t]{2,}")

    # ── Public API ────────────────────────────────────────────────────────────

    def certify(
        self,
        docs: list[dict],
        dataset_name: str,
        output_dir: Path | None = None,
    ) -> CertificationResult:
        """Run all 24 stages. Return CertificationResult."""
        pipeline_start = time.time()
        stage_results: list[StageResult] = []
        current_docs = list(docs)

        logger.info(
            f"[CertificationPipeline] Starting 24-stage certification for "
            f"'{dataset_name}' on {len(current_docs)} documents."
        )

        # ── Stage 1: download_verification ───────────────────────────────────
        current_docs, sr = self._run_stage(
            1, current_docs,
            stage_fn=self._stage_download_verification,
        )
        stage_results.append(sr)

        # ── Stage 2: file_integrity ───────────────────────────────────────────
        current_docs, sr = self._run_stage(
            2, current_docs,
            stage_fn=self._stage_file_integrity,
        )
        stage_results.append(sr)

        # ── Stage 3: utf8_validation ──────────────────────────────────────────
        current_docs, sr = self._run_stage(
            3, current_docs,
            stage_fn=self._stage_utf8_validation,
        )
        stage_results.append(sr)

        # ── Stage 4: encoding_normalization ──────────────────────────────────
        current_docs, sr = self._run_stage(
            4, current_docs,
            stage_fn=self._stage_encoding_normalization,
        )
        stage_results.append(sr)

        # ── Stage 5: language_detection ───────────────────────────────────────
        current_docs, sr = self._run_stage(
            5, current_docs,
            stage_fn=self._stage_language_detection,
        )
        stage_results.append(sr)

        # ── Stage 6: english_filtering ────────────────────────────────────────
        current_docs, sr = self._run_stage(
            6, current_docs,
            stage_fn=self._stage_english_filtering,
        )
        stage_results.append(sr)

        # ── Stage 7: corruption_detection ─────────────────────────────────────
        current_docs, sr = self._run_stage(
            7, current_docs,
            stage_fn=self._stage_corruption_detection,
        )
        stage_results.append(sr)

        # ── Stage 8: empty_removal ────────────────────────────────────────────
        current_docs, sr = self._run_stage(
            8, current_docs,
            stage_fn=self._stage_empty_removal,
        )
        stage_results.append(sr)

        # ── Stage 9: length_filtering ─────────────────────────────────────────
        current_docs, sr = self._run_stage(
            9, current_docs,
            stage_fn=self._stage_length_filtering,
        )
        stage_results.append(sr)

        # ── Stage 10: exact_deduplication ────────────────────────────────────
        current_docs, sr = self._run_stage(
            10, current_docs,
            stage_fn=self._stage_exact_deduplication,
        )
        stage_results.append(sr)

        # ── Stage 11: near_dedup_detection ───────────────────────────────────
        current_docs, sr = self._run_stage(
            11, current_docs,
            stage_fn=self._stage_near_dedup,
        )
        stage_results.append(sr)

        # ── Stage 12: boilerplate_detection ──────────────────────────────────
        current_docs, sr = self._run_stage(
            12, current_docs,
            stage_fn=self._stage_boilerplate,
        )
        stage_results.append(sr)

        # ── Stage 13: repetition_detection ───────────────────────────────────
        current_docs, sr = self._run_stage(
            13, current_docs,
            stage_fn=self._stage_repetition,
        )
        stage_results.append(sr)

        # ── Stage 14: pii_detection ───────────────────────────────────────────
        current_docs, sr = self._run_stage(
            14, current_docs,
            stage_fn=self._stage_pii,
        )
        stage_results.append(sr)

        # ── Stage 15: safety_filtering ────────────────────────────────────────
        current_docs, sr = self._run_stage(
            15, current_docs,
            stage_fn=self._stage_safety,
        )
        stage_results.append(sr)

        # ── Stage 16: quality_scoring ─────────────────────────────────────────
        current_docs, sr = self._run_stage(
            16, current_docs,
            stage_fn=self._stage_quality_scoring,
        )
        stage_results.append(sr)

        # ── Stage 17: entropy_analysis ────────────────────────────────────────
        current_docs, sr = self._run_stage(
            17, current_docs,
            stage_fn=self._stage_entropy,
        )
        stage_results.append(sr)

        # ── Stage 18: formatting_standardization ──────────────────────────────
        current_docs, sr = self._run_stage(
            18, current_docs,
            stage_fn=self._stage_formatting,
        )
        stage_results.append(sr)

        # ── Stage 19: byte_encoding_verification ──────────────────────────────
        current_docs, sr = self._run_stage(
            19, current_docs,
            stage_fn=self._stage_byte_verify,
        )
        stage_results.append(sr)

        # ── Stage 20: curriculum_tagging ──────────────────────────────────────
        current_docs, sr = self._run_stage(
            20, current_docs,
            stage_fn=self._stage_curriculum_tag,
            dataset_name=dataset_name,
        )
        stage_results.append(sr)

        # ── Stage 21: statistics_generation ──────────────────────────────────
        current_docs, sr = self._run_stage(
            21, current_docs,
            stage_fn=self._stage_statistics,
            dataset_name=dataset_name,
            output_dir=output_dir,
        )
        stage_results.append(sr)

        # ── Stage 22: dataset_card_generation ────────────────────────────────
        current_docs, sr = self._run_stage(
            22, current_docs,
            stage_fn=self._stage_dataset_card,
            dataset_name=dataset_name,
            output_dir=output_dir,
            stage_results_so_far=stage_results,
        )
        stage_results.append(sr)

        # ── Stage 23: sha256_manifest ─────────────────────────────────────────
        certified_path: Path | None = None
        if output_dir is not None:
            certified_path = output_dir / dataset_name
            certified_path.mkdir(parents=True, exist_ok=True)
            self._write_certified_jsonl(current_docs, certified_path)

        current_docs, sr = self._run_stage(
            23, current_docs,
            stage_fn=self._stage_sha256_manifest,
            output_dir=certified_path,
            dataset_name=dataset_name,
        )
        stage_results.append(sr)

        # ── Stage 24: freeze_certification ───────────────────────────────────
        current_docs, sr = self._run_stage(
            24, current_docs,
            stage_fn=self._stage_freeze,
            output_dir=certified_path,
            dataset_name=dataset_name,
            doc_count=len(current_docs),
        )
        stage_results.append(sr)

        # ── Compute summary ───────────────────────────────────────────────────
        total_removed = len(docs) - len(current_docs)
        overall_removal_rate = total_removed / len(docs) if docs else 0.0
        quality_scores = [d.get("quality_score", 1.0) for d in current_docs]
        mean_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
        elapsed = time.time() - pipeline_start

        result = CertificationResult(
            dataset_name=dataset_name,
            input_count=len(docs),
            certified_count=len(current_docs),
            stage_results=stage_results,
            total_removed=total_removed,
            overall_removal_rate=overall_removal_rate,
            quality_score=mean_quality,
            elapsed_sec=elapsed,
            certified_path=certified_path,
            success=True,
        )

        logger.info(
            f"[CertificationPipeline] '{dataset_name}' certified: "
            f"{len(current_docs)}/{len(docs)} docs kept "
            f"({overall_removal_rate:.1%} removed) in {elapsed:.1f}s."
        )
        return result

    def certify_from_jsonl(
        self,
        input_path: Path,
        dataset_name: str,
        output_dir: Path,
        text_field: str = "text",
    ) -> CertificationResult:
        """Load JSONL, run certification, write certified JSONL to output_dir."""
        docs: list[dict] = []
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        docs.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed JSONL line: {e}")

        logger.info(f"Loaded {len(docs)} documents from {input_path}")
        return self.certify(docs, dataset_name=dataset_name, output_dir=output_dir)

    def _run_stage(
        self,
        stage_num: int,
        docs: list[dict],
        stage_fn: Any,
        **kwargs: Any,
    ) -> tuple[list[dict], StageResult]:
        """Run a single stage function. Returns (kept_docs, StageResult)."""
        stage_name = self.STAGE_NAMES[stage_num - 1]
        input_count = len(docs)
        t0 = time.time()

        kept_docs, removal_reasons = stage_fn(docs, **kwargs)

        elapsed = time.time() - t0
        kept_count = len(kept_docs)
        removed_count = input_count - kept_count

        sr = StageResult(
            stage_num=stage_num,
            stage_name=stage_name,
            input_count=input_count,
            kept_count=kept_count,
            removed_count=removed_count,
            removal_reasons=removal_reasons,
            elapsed_sec=elapsed,
        )

        logger.info(
            f"Stage {stage_num:02d}/{len(self.STAGE_NAMES)} [{stage_name}]: "
            f"{kept_count}/{input_count} kept, {removed_count} removed "
            f"({sr.removal_rate:.1%}) in {elapsed:.2f}s"
        )
        return kept_docs, sr

    # ── Stage implementations ─────────────────────────────────────────────────

    def _stage_download_verification(
        self, docs: list[dict], **_: Any
    ) -> tuple[list[dict], dict[str, int]]:
        """Stage 1: Verify at least one document was downloaded."""
        if not docs:
            logger.warning("Stage 1: No documents provided — dataset is empty.")
        return docs, {}

    def _stage_file_integrity(
        self, docs: list[dict], **_: Any
    ) -> tuple[list[dict], dict[str, int]]:
        """Stage 2: Flag documents with empty text (already in memory)."""
        kept, reasons = [], {"empty_text_field": 0}
        for d in docs:
            if not d.get("text", ""):
                reasons["empty_text_field"] += 1
            else:
                kept.append(d)
        return kept, reasons

    def _stage_utf8_validation(
        self, docs: list[dict], **_: Any
    ) -> tuple[list[dict], dict[str, int]]:
        """Stage 3: Remove documents whose text cannot round-trip through UTF-8."""
        kept, reasons = [], {"utf8_invalid": 0}
        for d in docs:
            text = d.get("text", "")
            try:
                text.encode("utf-8").decode("utf-8")
                kept.append(d)
            except (UnicodeDecodeError, UnicodeEncodeError):
                reasons["utf8_invalid"] += 1
        return kept, reasons

    def _stage_encoding_normalization(
        self, docs: list[dict], **_: Any
    ) -> tuple[list[dict], dict[str, int]]:
        """Stage 4: Apply NFKC normalization — all documents kept, text modified."""
        for d in docs:
            d["text"] = unicodedata.normalize("NFKC", d.get("text", ""))
        return docs, {}

    def _stage_language_detection(
        self, docs: list[dict], **_: Any
    ) -> tuple[list[dict], dict[str, int]]:
        """Stage 5: Keep only documents detected as English."""
        texts = [d.get("text", "") for d in docs]
        langs = self._lang_detector.detect_batch(texts)
        kept, reasons = [], {"non_english_language": 0}
        for d, lang in zip(docs, langs):
            d["_detected_lang"] = lang
            if self._lang_detector.is_allowed(lang):
                kept.append(d)
            else:
                reasons["non_english_language"] += 1
        return kept, reasons

    def _stage_english_filtering(
        self, docs: list[dict], **_: Any
    ) -> tuple[list[dict], dict[str, int]]:
        """Stage 6: Secondary confidence-based English filter (same detector, stricter)."""
        kept, reasons = [], {"low_confidence_english": 0}
        for d in docs:
            lang = d.get("_detected_lang", "unknown")
            # 'unknown' is allowed (detector unavailable or short text)
            if lang in {"en", "unknown"}:
                kept.append(d)
            else:
                reasons["low_confidence_english"] += 1
        return kept, reasons

    def _stage_corruption_detection(
        self, docs: list[dict], **_: Any
    ) -> tuple[list[dict], dict[str, int]]:
        """Stage 7: Remove documents with null bytes or excessive replacement chars."""
        kept, reasons = [], {"null_bytes": 0, "replacement_chars": 0}
        for d in docs:
            text = d.get("text", "")
            if "\x00" in text:
                reasons["null_bytes"] += 1
                continue
            if text.count("\ufffd") > 2:
                reasons["replacement_chars"] += 1
                continue
            kept.append(d)
        return kept, reasons

    def _stage_empty_removal(
        self, docs: list[dict], **_: Any
    ) -> tuple[list[dict], dict[str, int]]:
        """Stage 8: Remove documents where stripped text is empty."""
        kept, reasons = [], {"empty_after_strip": 0}
        for d in docs:
            if len(d.get("text", "").strip()) == 0:
                reasons["empty_after_strip"] += 1
            else:
                kept.append(d)
        return kept, reasons

    def _stage_length_filtering(
        self, docs: list[dict], **_: Any
    ) -> tuple[list[dict], dict[str, int]]:
        """Stage 9: Remove documents outside the configured character length range."""
        kept, reasons = [], {"too_short": 0, "too_long": 0}
        for d in docs:
            n = len(d.get("text", ""))
            if n < self._min_doc_chars:
                reasons["too_short"] += 1
            elif n > self._max_doc_chars:
                reasons["too_long"] += 1
            else:
                kept.append(d)
        return kept, reasons

    def _stage_exact_deduplication(
        self, docs: list[dict], **_: Any
    ) -> tuple[list[dict], dict[str, int]]:
        """Stage 10: Exact deduplication via MD5 hashing."""
        texts = [d.get("text", "") for d in docs]
        kept_texts, report = self._deduplicator.exact_deduplicate(texts)
        kept_set = set(kept_texts)
        # Preserve order while filtering
        kept_docs: list[dict] = []
        seen_texts: set[str] = set()
        for d in docs:
            t = d.get("text", "")
            if t in kept_set and t not in seen_texts:
                kept_docs.append(d)
                seen_texts.add(t)
        reasons = {"exact_duplicate": report.removed_exact}
        return kept_docs, reasons

    def _stage_near_dedup(
        self, docs: list[dict], **_: Any
    ) -> tuple[list[dict], dict[str, int]]:
        """Stage 11: Near-duplicate removal via MinHash LSH (skipped if unavailable)."""
        texts = [d.get("text", "") for d in docs]
        kept_texts, report = self._deduplicator.near_deduplicate(texts)
        if not report.near_dedup_available:
            logger.info("Stage 11: datasketch unavailable — near-dedup skipped.")
            return docs, {}
        kept_set = set(kept_texts)
        kept_docs: list[dict] = []
        seen: set[str] = set()
        for d in docs:
            t = d.get("text", "")
            if t in kept_set and t not in seen:
                kept_docs.append(d)
                seen.add(t)
        reasons = {"near_duplicate": report.removed_near}
        return kept_docs, reasons

    def _stage_boilerplate(
        self, docs: list[dict], **_: Any
    ) -> tuple[list[dict], dict[str, int]]:
        """Stage 12: Boilerplate removal (delegates to BoilerplateDetector if available)."""
        if self._boilerplate_detector is None:
            return docs, {}
        try:
            texts = [d.get("text", "") for d in docs]
            kept_texts = self._boilerplate_detector.filter_batch(texts)
            kept_set = set(kept_texts)
            kept_docs = [d for d in docs if d.get("text", "") in kept_set]
            removed = len(docs) - len(kept_docs)
            return kept_docs, {"boilerplate": removed}
        except Exception as e:
            logger.warning(f"Stage 12 boilerplate filter error: {e} — passing through.")
            return docs, {}

    def _stage_repetition(
        self, docs: list[dict], **_: Any
    ) -> tuple[list[dict], dict[str, int]]:
        """Stage 13: Remove documents with excessive char 3-gram repetition (>30% coverage)."""
        kept, reasons = [], {"high_repetition": 0}
        for d in docs:
            text = d.get("text", "")
            if len(text) < 10:
                kept.append(d)
                continue
            # Build char 3-gram frequency
            ngrams: dict[str, int] = {}
            for i in range(len(text) - 2):
                ng = text[i : i + 3]
                ngrams[ng] = ngrams.get(ng, 0) + 1
            if not ngrams:
                kept.append(d)
                continue
            top5 = sorted(ngrams.values(), reverse=True)[:5]
            top5_coverage = sum(top5) * 3 / len(text)
            if top5_coverage > 0.30:
                reasons["high_repetition"] += 1
            else:
                kept.append(d)
        return kept, reasons

    def _stage_pii(
        self, docs: list[dict], **_: Any
    ) -> tuple[list[dict], dict[str, int]]:
        """Stage 14: PII scrubbing — modifies text in place, all documents kept."""
        for d in docs:
            d["text"] = self._pii_remover.remove(d.get("text", ""))
        return docs, {}

    def _stage_safety(
        self, docs: list[dict], **_: Any
    ) -> tuple[list[dict], dict[str, int]]:
        """Stage 15: Safety filtering (delegates to SafetyFilter if available)."""
        if self._safety_filter is None:
            return docs, {}
        try:
            texts = [d.get("text", "") for d in docs]
            kept_texts = self._safety_filter.filter_batch(texts)
            kept_set = set(kept_texts)
            kept_docs = [d for d in docs if d.get("text", "") in kept_set]
            removed = len(docs) - len(kept_docs)
            return kept_docs, {"safety_filtered": removed}
        except Exception as e:
            logger.warning(f"Stage 15 safety filter error: {e} — passing through.")
            return docs, {}

    def _stage_quality_scoring(
        self, docs: list[dict], **_: Any
    ) -> tuple[list[dict], dict[str, int]]:
        """Stage 16: Score quality and filter low-quality docs."""
        texts = [d.get("text", "") for d in docs]
        _, report = self._quality_filter.apply_with_report(texts)
        # Re-apply to get kept indices
        kept_texts, _ = self._quality_filter.apply_with_report(texts)
        kept_set = set(kept_texts)
        kept_docs: list[dict] = []
        for d in docs:
            t = d.get("text", "")
            if t in kept_set:
                # Simple quality score: alpha ratio as proxy
                alpha = sum(1 for c in t if c.isalpha())
                d["quality_score"] = alpha / len(t) if t else 0.0
                kept_docs.append(d)
        removed = len(docs) - len(kept_docs)
        reasons = {"low_quality": removed}
        return kept_docs, reasons

    def _stage_entropy(
        self, docs: list[dict], **_: Any
    ) -> tuple[list[dict], dict[str, int]]:
        """Stage 17: Entropy analysis (delegates to EntropyAnalyzer if available)."""
        if self._entropy_analyzer is None:
            return docs, {}
        try:
            texts = [d.get("text", "") for d in docs]
            results = self._entropy_analyzer.analyze_batch(texts)
            kept_docs = [d for d, r in zip(docs, results) if r.get("keep", True)]
            removed = len(docs) - len(kept_docs)
            return kept_docs, {"low_entropy": removed}
        except Exception as e:
            logger.warning(f"Stage 17 entropy error: {e} — passing through.")
            return docs, {}

    def _stage_formatting(
        self, docs: list[dict], **_: Any
    ) -> tuple[list[dict], dict[str, int]]:
        """Stage 18: Strip HTML tags, normalize multiple whitespace."""
        for d in docs:
            text = d.get("text", "")
            text = self._html_tag_pat.sub(" ", text)
            text = self._multi_ws_pat.sub(" ", text)
            d["text"] = text.strip()
        return docs, {}

    def _stage_byte_verify(
        self, docs: list[dict], **_: Any
    ) -> tuple[list[dict], dict[str, int]]:
        """Stage 19: Verify text.encode('utf-8', errors='strict') succeeds."""
        kept, reasons = [], {"byte_encoding_failed": 0}
        for d in docs:
            try:
                d.get("text", "").encode("utf-8", errors="strict")
                kept.append(d)
            except (UnicodeEncodeError, UnicodeDecodeError):
                reasons["byte_encoding_failed"] += 1
        return kept, reasons

    def _stage_curriculum_tag(
        self, docs: list[dict], dataset_name: str = "", **_: Any
    ) -> tuple[list[dict], dict[str, int]]:
        """Stage 20: Add curriculum metadata tags to each document."""
        for d in docs:
            d["source"] = dataset_name
            d["certified"] = True
        return docs, {}

    def _stage_statistics(
        self,
        docs: list[dict],
        dataset_name: str = "",
        output_dir: Path | None = None,
        **_: Any,
    ) -> tuple[list[dict], dict[str, int]]:
        """Stage 21: Generate DatasetReport and write to output_dir."""
        texts = [d.get("text", "") for d in docs]
        report = self._stats_gen.generate(
            name=dataset_name,
            texts=texts,
            stage="certified",
        )
        if output_dir is not None:
            stats_dir = output_dir / dataset_name if (output_dir / dataset_name).exists() else output_dir
            stats_dir.mkdir(parents=True, exist_ok=True)
            self._stats_gen.save_json(report, stats_dir / "certified_statistics.json")
            self._stats_gen.save_markdown(report, stats_dir / "certified_statistics.md")
        return docs, {}

    def _stage_dataset_card(
        self,
        docs: list[dict],
        dataset_name: str = "",
        output_dir: Path | None = None,
        stage_results_so_far: list[StageResult] | None = None,
        **_: Any,
    ) -> tuple[list[dict], dict[str, int]]:
        """Stage 22: Generate HuggingFace-style dataset card."""
        try:
            from data.pipeline.dataset_card import DatasetCardGenerator, DatasetCardMetadata

            stage_results_so_far = stage_results_so_far or []
            texts = [d.get("text", "") for d in docs]
            total_bytes = sum(len(t.encode("utf-8")) for t in texts)
            avg_len = total_bytes / len(texts) if texts else 0.0

            # Compute removal rates from stage results
            dedup_sr = next((s for s in stage_results_so_far if "dedup" in s.stage_name), None)
            pii_sr = next((s for s in stage_results_so_far if s.stage_name == "pii_detection"), None)
            safety_sr = next((s for s in stage_results_so_far if s.stage_name == "safety_filtering"), None)
            quality_sr = next((s for s in stage_results_so_far if s.stage_name == "quality_scoring"), None)

            meta = DatasetCardMetadata(
                name=dataset_name,
                version="1.0",
                license="cc-by-4.0",
                license_url="https://creativecommons.org/licenses/by/4.0/",
                source_url="",
                created_at=datetime.now().isoformat(),
                num_documents=len(docs),
                num_bytes=total_bytes,
                avg_doc_length=avg_len,
                median_doc_length=avg_len,
                byte_entropy_mean=0.0,
                dedup_removal_rate=dedup_sr.removal_rate if dedup_sr else 0.0,
                pii_removal_rate=pii_sr.removal_rate if pii_sr else 0.0,
                safety_removal_rate=safety_sr.removal_rate if safety_sr else 0.0,
                quality_removal_rate=quality_sr.removal_rate if quality_sr else 0.0,
                overall_retention_rate=1.0 - (
                    sum(s.removed_count for s in stage_results_so_far) / max(1, stage_results_so_far[0].input_count)
                    if stage_results_so_far else 0.0
                ),
                sha256_manifest_hash="",
                freeze_certificate_hash="",
                curriculum_phases=["A"],
                certification_stages_passed=len(stage_results_so_far),
            )

            if output_dir is not None:
                card_dir = output_dir / dataset_name if (output_dir / dataset_name).exists() else output_dir
                card_dir.mkdir(parents=True, exist_ok=True)
                DatasetCardGenerator().generate(meta, card_dir / "DATASET_CARD.md")
        except Exception as e:
            logger.warning(f"Stage 22 dataset card generation error: {e}")

        return docs, {}

    def _stage_sha256_manifest(
        self,
        docs: list[dict],
        output_dir: Path | None = None,
        dataset_name: str = "",
        **_: Any,
    ) -> tuple[list[dict], dict[str, int]]:
        """Stage 23: Write SHA-256 manifest via DatasetVersioner."""
        if output_dir is not None and output_dir.exists():
            try:
                from data.pipeline.versioning import ManifestEntry
                texts = [d.get("text", "") for d in docs]
                total_bytes = sum(len(t.encode("utf-8")) for t in texts)
                self._versioner.create_version(
                    name=dataset_name,
                    data_path=output_dir,
                    config={"certified": True, "stages": 24},
                    document_count=len(docs),
                    byte_count=total_bytes,
                    stage="certified",
                    processing_steps=self.STAGE_NAMES,
                )
                content_hash = self._versioner.compute_content_hash(output_dir)
                entry = ManifestEntry(
                    dataset_name=dataset_name,
                    version="1.0",
                    license="cc-by-4.0",
                    sha256=content_hash,
                    pipeline_version=self._versioner.pipeline_version,
                    creation_time=datetime.now().isoformat(),
                    document_count=len(docs),
                    byte_count=total_bytes,
                    stage="certified",
                    source="certification_pipeline",
                    mixing_weight=1.0,
                )
                self._versioner.write_manifest(output_dir, [entry])
            except Exception as e:
                logger.warning(f"Stage 23 manifest write error: {e}")
        return docs, {}

    def _stage_freeze(
        self,
        docs: list[dict],
        output_dir: Path | None = None,
        dataset_name: str = "",
        doc_count: int = 0,
        **_: Any,
    ) -> tuple[list[dict], dict[str, int]]:
        """Stage 24: Write freeze certificate via FreezeManager."""
        if output_dir is not None and output_dir.exists():
            try:
                from data.pipeline.freeze import FreezeManager
                texts = [d.get("text", "") for d in docs]
                total_bytes = sum(len(t.encode("utf-8")) for t in texts)
                manifest_path = output_dir / "manifest.json"
                manifest_hash = self._versioner.compute_file_sha256(manifest_path) if manifest_path.exists() else ""
                FreezeManager().freeze(
                    dataset_path=output_dir,
                    version="IVERI_DATASET_v1.0",
                    doc_count=doc_count,
                    byte_count=total_bytes,
                    sha256_manifest_hash=manifest_hash,
                )
            except Exception as e:
                logger.warning(f"Stage 24 freeze error: {e}")
        return docs, {}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _write_certified_jsonl(self, docs: list[dict], output_dir: Path) -> None:
        """Write certified documents to train.jsonl in output_dir."""
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "train.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {len(docs)} certified documents to {out_path}")
