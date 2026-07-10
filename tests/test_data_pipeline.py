# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Test suite for Phase 3.0 Data Engineering Pipeline (Stage 0).

Verifies all pipeline modules: data_registry, downloader, license_checker,
byte_encoder, deduplication, language_detector, quality_filter, pii_remover,
splitter, versioning, byte_counter, mixer, sft_validator, statistics, and dataloader.

Every test runs offline and mocks HF network requests.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from configs.base_config import get_base_config

# Import config first
from configs.data_pipeline_config import (
    DataPipelineConfig,
    DedupConfig,
    DownloadConfig,
    QualityConfig,
    SplitConfig,
)
from core.exceptions import ConfigError
from data.pipeline.byte_counter import ByteCounter
from core.constants import BOS_BYTE, EOS_BYTE, PAD_BYTE
from data.pipeline.byte_encoder import ByteEncoder, ByteEncoderConfig
from data.pipeline.data_registry import DataRegistry, DatasetEntry
from data.pipeline.dataloader import CodingByteDataset, PretrainByteDataset, SFTByteDataset
from data.pipeline.deduplication import Deduplicator
from data.pipeline.downloader import DatasetDownloader
from data.pipeline.language_detector import LanguageDetector
from data.pipeline.license_checker import LicenseChecker
from data.pipeline.mixer import DatasetMixer, MixingStrategy
from data.pipeline.pii_remover import PIIRemover
from data.pipeline.provenance import ProvenanceTracker
from data.pipeline.quality_filter import QualityFilter, QualityFilterConfig
from data.pipeline.sft_validator import SFTValidator
from data.pipeline.splitter import DatasetSplitter
from data.pipeline.statistics import DatasetStatisticsGenerator
from data.pipeline.versioning import DatasetVersioner, ManifestEntry


@pytest.fixture
def temp_dir():
    """Temporary directory for test artifacts."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


# ── 1. DataPipelineConfig Tests ──────────────────────────────────────────────


def test_config_defaults():
    """Verify default config values are correct."""
    cfg = DataPipelineConfig()
    assert cfg.download.raw_data_dir == "data/raw"
    assert cfg.quality.min_doc_chars == 100
    assert cfg.dedup.exact_dedup_enabled is True
    assert cfg.split.train_ratio == 0.98
    assert cfg.pii.enabled is True
    assert cfg.language.allowed_languages == ["en"]
    assert cfg.mixing.strategy == "weighted_random"
    assert cfg.report.report_dir == "reports/phase_3_0"


def test_config_validation():
    """Test validator constraints on configurations."""
    with pytest.raises(ConfigError):
        DownloadConfig(max_retries=-1)
    with pytest.raises(ConfigError):
        QualityConfig(min_doc_chars=-1)
    with pytest.raises(ConfigError):
        QualityConfig(min_doc_chars=500, max_doc_chars=100)
    with pytest.raises(ConfigError):
        QualityConfig(min_alpha_ratio=1.5)
    with pytest.raises(ConfigError):
        DedupConfig(near_dedup_threshold=1.5)
    with pytest.raises(ConfigError):
        SplitConfig(train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)


def test_config_serialization(temp_dir):
    """Test config serialization to dict/json/yaml and back."""
    cfg = DataPipelineConfig()
    d = cfg.to_dict()
    assert d["download"]["raw_data_dir"] == "data/raw"

    cfg_from_dict = DataPipelineConfig.from_dict(d)
    assert cfg_from_dict.download.raw_data_dir == "data/raw"

    json_path = temp_dir / "pipeline_config.json"
    cfg.save(json_path)
    loaded = DataPipelineConfig.load(json_path)
    assert loaded.download.raw_data_dir == "data/raw"

    # YAML serialization
    try:
        yaml_str = cfg.to_yaml()
        loaded_yaml = DataPipelineConfig.from_yaml(yaml_str)
        assert loaded_yaml.download.raw_data_dir == "data/raw"
    except ImportError:
        # Ignore if yaml is not installed in the testing environment
        pass


def test_base_config_integration():
    """Verify DataPipelineConfig is integrated into base_config.py's IVERIConfig."""
    base_cfg = get_base_config()
    assert hasattr(base_cfg, "data_pipeline")
    assert isinstance(base_cfg.data_pipeline, DataPipelineConfig)

    # Roundtrip dictionary check
    d = base_cfg.to_dict()
    assert "data_pipeline" in d
    assert d["data_pipeline"]["download"]["raw_data_dir"] == "data/raw"


# ── 2. DataRegistry Tests ────────────────────────────────────────────────────


def test_registry_registration():
    """Test manual dataset registration and retrieval."""
    reg = DataRegistry(auto_discover=False)
    entry = DatasetEntry(
        name="test_ds",
        hf_id="test/ds",
        priority="S",
        format="pretrain",
        source="huggingface",
    )
    reg.register(entry)
    assert reg.get("test_ds") == entry

    with pytest.raises(ValueError):
        # Duplicate registration
        reg.register(entry)

    with pytest.raises(KeyError):
        reg.get("non_existent")


def test_registry_filters():
    """Test list filtering by stage and priority."""
    reg = DataRegistry(auto_discover=False)
    reg.register(DatasetEntry(name="ds1", hf_id="id1", priority="S", stage=1))
    reg.register(DatasetEntry(name="ds2", hf_id="id2", priority="A", stage=2))
    reg.register(DatasetEntry(name="ds3", hf_id="id3", priority="S", stage=2))

    assert len(reg.list_by_stage(2)) == 2
    assert len(reg.list_by_priority("S")) == 2


def test_registry_validation():
    """Verify registry validation checks work correctly."""
    reg = DataRegistry(auto_discover=False)
    with pytest.raises(ValueError):
        # Missing hf_id for HF source
        reg.register(DatasetEntry(name="bad_ds", source="huggingface"))
    with pytest.raises(ValueError):
        # Missing path for local source
        reg.register(DatasetEntry(name="bad_ds", source="local"))


# ── 3. LicenseChecker Tests ──────────────────────────────────────────────────


def test_license_compatibility():
    """Verify compatibility rules for common licenses."""
    checker = LicenseChecker()
    # MIT and Apache-2.0 should be compatible for both
    assert checker.verify("mit_ds") is False  # unknown unless registered, falls back to unknown

    # Mocking registry
    mock_reg = MagicMock()
    mock_reg.get.side_effect = lambda name: DatasetEntry(
        name=name, license="MIT" if "mit" in name else "NVIDIA-Open-Model"
    )

    checker_with_reg = LicenseChecker(mock_reg)
    assert checker_with_reg.verify("mit_ds", "research") is True
    assert checker_with_reg.verify("mit_ds", "commercial") is True
    assert checker_with_reg.verify("nvidia_ds", "research") is True
    assert checker_with_reg.verify("nvidia_ds", "commercial") is False


def test_license_attribution_report():
    """Verify license checker generates Markdown reports correctly."""
    mock_reg = MagicMock()
    mock_reg.get.side_effect = lambda name: DatasetEntry(
        name=name, license="MIT", description="Desc for " + name
    )

    checker = LicenseChecker(mock_reg)
    report = checker.generate_attribution_report(["ds_a", "ds_b"])
    assert "# IVERI CORE — Dataset Attribution Report" in report
    assert "ds_a" in report
    assert "ds_b" in report
    assert "MIT" in report


# ── 4. Versioning Tests ──────────────────────────────────────────────────────


def test_versioning_creation(temp_dir):
    """Test VERSION.json creation, loading, and assertions."""
    versioner = DatasetVersioner()

    dummy_file = temp_dir / "data.txt"
    dummy_file.write_text("dummy content", encoding="utf-8")

    config = {"param1": 100, "param2": "value"}
    info = versioner.create_version(
        name="test_ds",
        data_path=temp_dir,
        config=config,
        document_count=1,
        byte_count=13,
        stage="1",
        processing_steps=["init", "clean"],
    )

    assert info.dataset_name == "test_ds"
    assert (temp_dir / "VERSION.json").exists()

    # Load version back
    loaded = versioner.load_version(temp_dir)
    assert loaded.version_id == info.version_id
    assert loaded.document_count == 1
    assert loaded.byte_count == 13
    assert loaded.processing_steps == ["init", "clean"]

    # Assertion checks
    versioner.assert_version_exists(temp_dir)


def test_manifest_writing(temp_dir):
    """Verify dataset manifests are created and parsed correctly."""
    versioner = DatasetVersioner()
    entry = ManifestEntry(
        dataset_name="ds1",
        version="v1.0",
        license="MIT",
        sha256="abc123sha",
        pipeline_version="3.0.0",
        creation_time="now",
        document_count=100,
        byte_count=5000,
        stage="2",
        source="huggingface",
        mixing_weight=0.5,
    )

    manifest_file = versioner.write_manifest(temp_dir, [entry])
    assert manifest_file.exists()

    loaded = versioner.load_manifest(manifest_file)
    assert len(loaded) == 1
    assert loaded[0].dataset_name == "ds1"
    assert loaded[0].byte_count == 5000


# ── 5. Provenance Tests ──────────────────────────────────────────────────────


def test_provenance_creation():
    """Verify document-level provenance records."""
    tracker = ProvenanceTracker()
    text = "hello world provenance"
    record = tracker.create_record(
        text=text,
        source_dataset="tinystories",
        license="MIT",
        stage="1",
        url="http://tinystories.org",
        language="en",
    )

    assert record.document_hash == tracker.compute_document_hash(text)
    assert record.source_dataset == "tinystories"
    assert record.byte_count == len(text.encode("utf-8"))
    assert len(record.processing_steps) == 0


def test_provenance_steps():
    """Test adding lineage steps and serialization."""
    tracker = ProvenanceTracker()
    text = "lineage text"
    record = tracker.create_record(text, "ds", "MIT", "1")

    tracker.add_step(record, "quality_filter", {"min_len": 10}, result_count=1, removed_count=0)
    assert len(record.processing_steps) == 1
    assert record.processing_steps[0].step_name == "quality_filter"

    d = tracker.to_dict(record)
    assert d["source_dataset"] == "ds"
    assert len(d["processing_steps"]) == 1

    loaded = tracker.from_dict(d)
    assert loaded.document_hash == record.document_hash
    assert len(loaded.processing_steps) == 1


# ── 6. ByteEncoder Tests ─────────────────────────────────────────────────────


def test_byte_encoder_basic():
    """Test basic encoding and decoding of text."""
    encoder = ByteEncoder()
    text = "hello world byte encoder"
    ids = encoder.encode(text)

    # 0-255 bounds check
    assert encoder.validate(ids)
    assert len(ids) == len(text.encode("utf-8")) + 2  # BOS + EOS

    decoded = encoder.decode(ids)
    assert decoded == text


def test_byte_encoder_tensor():
    """Test conversion of byte IDs to tensors with padding and truncation."""
    cfg = ByteEncoderConfig(seq_len=10)
    encoder = ByteEncoder(cfg)

    # Short sequence -> padded
    short_ids = encoder.encode("hi")
    t_padded = encoder.to_tensor(short_ids)
    assert t_padded.shape == (10,)
    assert t_padded[-1].item() == PAD_BYTE

    # Long sequence -> truncated
    long_ids = encoder.encode("this is a very long text sequence")
    t_truncated = encoder.to_tensor(long_ids)
    assert t_truncated.shape == (10,)

    stats = encoder.statistics()
    assert stats.padded_count == 1
    assert stats.truncated_count == 1


def test_byte_encoder_sft():
    """Test SFT sample encoding."""
    encoder = ByteEncoder()
    # Alpaca sample
    sample_alpaca = {"instruction": "test instruction", "output": "test response"}
    t_alpaca = encoder.encode_sft_sample(sample_alpaca)
    assert t_alpaca.dim() == 1

    # Messages sample
    sample_msg = {
        "messages": [
            {"role": "user", "content": "user query"},
            {"role": "assistant", "content": "assistant response"},
        ]
    }
    t_msg = encoder.encode_sft_sample(sample_msg)
    assert t_msg.dim() == 1


# ── 7. Deduplicator Tests ────────────────────────────────────────────────────


def test_exact_deduplication():
    """Verify exact duplicate removal works correctly."""
    deduper = Deduplicator()
    texts = ["apple", "banana", "apple", "cherry", "banana"]
    cleaned, report = deduper.exact_deduplicate(texts)

    assert cleaned == ["apple", "banana", "cherry"]
    assert report.total_input == 5
    assert report.kept == 3
    assert report.removed_exact == 2


def test_near_deduplication():
    """Test near-duplicate detection."""
    deduper = Deduplicator()
    texts = [
        "this is a simple test document for near duplicate detection",
        "this is a simple test document for near duplicate detection indeed",  # near dupe
        "completely different text that should never match",
    ]
    cleaned, report = deduper.deduplicate(texts)

    # If datasketch is not installed, it falls back to exact dedup (keeping all 3 since strings differ)
    # If installed, it should merge the first two based on 0.8 threshold
    if report.near_dedup_available:
        assert len(cleaned) == 2
        assert report.removed_near == 1
    else:
        assert len(cleaned) == 3
        assert report.removed_exact == 0


# ── 8. LanguageDetector Tests ────────────────────────────────────────────────


def test_language_detection_filtering():
    """Test language detection allow/reject filters."""
    detector = LanguageDetector(allowed={"en"}, reject={"fr"})

    # Mocking detect method for deterministic testing without depending on langdetect
    with patch.object(
        detector,
        "detect",
        side_effect=lambda x: "fr" if "bonjour" in x else ("en" if "hello" in x else "unknown"),
    ):
        texts = ["hello world", "bonjour le monde", "plain test", "unknown language"]
        kept, report = detector.filter(texts)

        # Kept: "hello world" (en), "plain test" (unknown), "unknown language" (unknown)
        # Filtered: "bonjour le monde" (fr - rejected)
        assert len(kept) == 3
        assert report.filtered_count == 1
        assert "fr" in report.detected_counts


# ── 9. QualityFilter Tests ───────────────────────────────────────────────────


def test_quality_filter_metrics():
    """Verify individual quality filters reject bad texts."""
    cfg = QualityFilterConfig(min_doc_chars=10, max_doc_chars=50, min_alpha_ratio=0.5)
    filterer = QualityFilter(cfg)

    # Length filters
    assert filterer.min_length_filter("short") is False
    assert filterer.min_length_filter("long text exceeding min characters") is True
    assert (
        filterer.max_length_filter("text exceeding maximum characters configuration limit") is False
    )

    # Alpha ratio
    assert filterer.alpha_ratio_filter("1234567890!") is False
    assert filterer.alpha_ratio_filter("valid text strings") is True

    # Punctuation ratio
    assert filterer.punctuation_ratio_filter("text!!!???@@@###$$$") is False


def test_quality_filter_apply():
    """Verify bulk filtering application and report stats."""
    cfg = QualityFilterConfig(min_doc_chars=10)
    filterer = QualityFilter(cfg)
    texts = [
        "short",  # removed
        "normal valid document",  # kept
        "1234567890!!!!!",  # removed (alpha ratio)
    ]
    kept, report = filterer.apply_with_report(texts)
    assert len(kept) == 1
    assert report.removed_too_short == 1
    assert report.kept == 1


def test_unicode_normalization_and_utf8_repair():
    """Test unicode norm and broken UTF-8 repairs inside quality filter."""
    filterer = QualityFilter()
    # Unicode normalization
    text_ligature = "ﬁle"  # ligature fi
    normalized = filterer.normalize_unicode_text(text_ligature)
    assert normalized == "file"

    # Control char removal
    text_ctrl = "hello\x00world"
    cleaned = filterer.remove_control_characters(text_ctrl)
    assert cleaned == "helloworld"


# ── 10. PIIRemover Tests ─────────────────────────────────────────────────────


def test_pii_scrubbing():
    """Verify email, phones, Aadhaar, PAN, and credentials are redacted."""
    remover = PIIRemover()
    text = (
        "My email is test@domain.com, phone 9876543210. "
        "Aadhaar: 1234 5678 9012. PAN: ABCDE1234F. "
        "AWS secret: aws secret_key='1234567890123456789012345678901234567890'. "
        "OpenAI key: sk-123456789012345678901234567890123456789012345678."
    )
    cleaned = remover.remove(text)
    assert "test@domain.com" not in cleaned
    assert "9876543210" not in cleaned
    assert "ABCDE1234F" not in cleaned
    assert "sk-" not in cleaned
    assert "[REDACTED]" in cleaned

    # Audit check
    report = remover.audit([text])
    assert report.total_docs == 1
    assert report.docs_with_pii == 1
    assert report.match_counts["email"] == 1
    assert report.match_counts["phone_india"] == 1


# ── 11. DatasetSplitter Tests ────────────────────────────────────────────────


def test_splitter_splits():
    """Test train/val/test splits ratios and determinism."""
    splitter = DatasetSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)
    data = list(range(100))

    train1, val1, test1 = splitter.split(data)
    assert len(train1) == 80
    assert len(val1) == 10
    assert len(test1) == 10

    # Determinism check
    train2, val2, test2 = splitter.split(data)
    assert train1 == train2
    assert val1 == val2
    assert test1 == test2


def test_splitter_small_dataset():
    """Verify splitter switches to 90/5/5 for small datasets."""
    splitter = DatasetSplitter(small_threshold=50)
    data_small = list(range(20))
    train, val, test = splitter.auto_split(data_small)

    assert len(train) == 18
    assert len(val) == 1
    assert len(test) == 1

    report = splitter.generate_report(train, val, test, seed=42)
    assert report.total_count == 20
    assert report.ratios == (0.9, 0.05, 0.05)


# ── 12. ByteCounter Tests ────────────────────────────────────────────────────


def test_byte_counter_statistics():
    """Verify byte counter computes sizes and percentiles."""
    counter = ByteCounter()
    texts = ["a", "bb", "ccc", "dddd"]
    stats = counter.count_dataset(texts, "test", "1")

    assert stats.num_documents == 4
    assert stats.total_bytes == 10
    assert stats.avg_bytes == 2.5
    assert stats.min_bytes == 1
    assert stats.max_bytes == 4
    assert stats.median_bytes == 2.5
    assert stats.p25_bytes == 1.75
    assert stats.p75_bytes == 3.25


# ── 13. DatasetMixer Tests ───────────────────────────────────────────────────


def test_dataset_mixer():
    """Test dataset mixing weights, strategies, and curricula."""
    mixer = DatasetMixer(strategy=MixingStrategy.WEIGHTED_RANDOM, seed=42)
    datasets = {
        "ds1": ["a1", "a2", "a3"],
        "ds2": ["b1", "b2", "b3"],
    }
    weights = {"ds1": 0.8, "ds2": 0.2}

    mixed = mixer.mix(datasets, weights, total_samples=100)
    assert len(mixed) == 100

    # ds1 should have higher occurrences
    ds1_count = sum(1 for x in mixed if "a" in x)
    assert ds1_count > 60


def test_mixer_round_robin():
    """Test round-robin mixing strategy."""
    mixer = DatasetMixer(strategy=MixingStrategy.ROUND_ROBIN)
    datasets = {
        "ds1": ["a1", "a2"],
        "ds2": ["b1", "b2"],
    }
    mixed = mixer.round_robin(datasets, n_samples=6)
    assert mixed == ["a1", "b1", "a2", "b2", "a1", "b1"]


def test_mixer_curriculum():
    """Test curriculum mixing step interpolations."""
    mixer = DatasetMixer(strategy=MixingStrategy.CURRICULUM)
    datasets = {
        "ds1": ["a1"],
        "ds2": ["b1"],
    }
    w_start = {"ds1": 1.0, "ds2": 0.0}
    w_end = {"ds1": 0.0, "ds2": 1.0}

    # Step 0: pure ds1
    mixed_start = mixer.curriculum_mix(
        datasets, w_start, w_end, current_step=0, total_steps=100, n_samples=10
    )
    assert all("a" in x for x in mixed_start)

    # Step 100: pure ds2
    mixed_end = mixer.curriculum_mix(
        datasets, w_start, w_end, current_step=100, total_steps=100, n_samples=10
    )
    assert all("b" in x for x in mixed_end)


# ── 14. SFTValidator Tests ───────────────────────────────────────────────────


def test_sft_validation_alpaca():
    """Verify SFT validator checks Alpaca format correctly."""
    validator = SFTValidator()

    # Valid
    ok, msg = validator.validate_alpaca(
        {"instruction": "What is 2NF?", "output": "2NF is defined as..."}
    )
    assert ok is True

    # Missing field
    ok, msg = validator.validate_alpaca({"instruction": "What is 2NF?"})
    assert ok is False
    assert "missing 'output'" in msg

    # Placeholder output
    ok, msg = validator.validate_alpaca({"instruction": "What is 2NF?", "output": "TODO"})
    assert ok is False
    assert "placeholder" in msg


def test_sft_validation_conversation():
    """Verify SFT validator checks multi-turn conversation format."""
    validator = SFTValidator()

    # Valid
    sample_valid = {
        "messages": [
            {"role": "user", "content": "What is normalisation?"},
            {"role": "assistant", "content": "It is the process of organizing data..."},
        ]
    }
    ok, msg = validator.validate_conversation(sample_valid)
    assert ok is True

    # Empty assistant message
    sample_empty_assistant = {
        "messages": [
            {"role": "user", "content": "What is normalisation?"},
            {"role": "assistant", "content": ""},
        ]
    }
    ok, msg = validator.validate_conversation(sample_empty_assistant)
    assert ok is False
    assert "empty content" in msg

    # Non-alternating roles
    sample_consecutive = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Are you there?"},
            {"role": "assistant", "content": "Yes."},
        ]
    }
    ok, msg = validator.validate_conversation(sample_consecutive)
    assert ok is False
    assert "consecutive duplicate role" in msg

    # Ending with user role
    sample_ends_user = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
        ]
    }
    ok, msg = validator.validate_conversation(sample_ends_user)
    assert ok is False
    assert "must end with 'assistant'" in msg


# ── 15. Statistics Tests ─────────────────────────────────────────────────────


def test_statistics_generation(temp_dir):
    """Test generating statistics and saving reports in markdown/json/csv formats."""
    gen = DatasetStatisticsGenerator()
    texts = ["hello world", "longer text string for statistics", "short"]
    report = gen.generate(name="test_stats", texts=texts, stage="1", license_str="MIT")

    assert report.num_documents == 3
    assert report.license == "MIT"

    # Save to files
    md_path = temp_dir / "report.md"
    json_path = temp_dir / "report.json"
    csv_path = temp_dir / "report.csv"

    gen.save_markdown(report, md_path)
    gen.save_json(report, json_path)
    gen.save_csv(report, csv_path)

    assert md_path.exists()
    assert json_path.exists()
    assert csv_path.exists()


# ── 16. Downloader Tests (Offline Mocked) ────────────────────────────────────


@patch("data.pipeline.downloader.DatasetDownloader._load_hf_dataset")
def test_downloader_offline(mock_load_hf, temp_dir):
    """Verify downloader runs completely offline and downloads/saves dataset metadata."""
    # Mocking HF load_dataset return value
    mock_ds = MagicMock()
    mock_ds.__len__.return_value = 10

    # Mock save_to_disk method
    def mock_save(path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "dataset_info.json").write_text("{}", encoding="utf-8")
        (Path(path) / "dummy.arrow").write_text("data content", encoding="utf-8")

    mock_ds.save_to_disk = mock_save
    mock_load_hf.return_value = mock_ds

    downloader = DatasetDownloader(save_dir=temp_dir, verify_checksums=True)
    entry = DatasetEntry(name="stories", hf_id="mock/stories", stage=1)

    meta = downloader.download(name="stories", entry=entry)
    assert meta.status == "success"
    assert meta.num_rows == 10
    assert meta.sha256 is not None
    assert meta.md5 is not None
    assert (temp_dir / "stories" / "metadata.json").exists()


# ── 17. DataLoader Tests ─────────────────────────────────────────────────────


def test_dataloaders_offline(temp_dir):
    """Verify Pretrain, SFT, and Coding dataloaders yield correct shapes."""
    # Create dummy pretrain dataset files
    pretrain_dir = temp_dir / "pretrain"
    pretrain_dir.mkdir()
    (pretrain_dir / "data1.txt").write_text(
        "this is a pretrain sentence for dataloader testing.", encoding="utf-8"
    )

    pretrain_ds = PretrainByteDataset(pretrain_dir, seq_len=16)
    assert len(pretrain_ds) > 0
    x, y = pretrain_ds[0]
    assert x.shape == (16,)
    assert y.shape == (16,)
    assert x.dtype == torch.long
    assert y.dtype == torch.long

    # Create dummy SFT dataset file
    sft_dir = temp_dir / "sft"
    sft_dir.mkdir()
    sft_data = [
        {
            "instruction": "What is normalisation?",
            "output": "Normalisation is defined as DB organization.",
        },
        {
            "instruction": "What is 2NF?",
            "output": "Second Normal Form requires no partial dependencies.",
        },
    ]
    with open(sft_dir / "data.json", "w", encoding="utf-8") as f:
        json.dump(sft_data, f)

    sft_ds = SFTByteDataset(sft_dir, seq_len=32)
    assert len(sft_ds) == 2
    x_sft, y_sft, mask_sft = sft_ds[0]
    assert x_sft.shape == (32,)
    assert y_sft.shape == (32,)
    assert mask_sft.shape == (32,)
    assert mask_sft.dtype == torch.bool


    # Create dummy Coding dataset file
    coding_dir = temp_dir / "coding"
    coding_dir.mkdir()
    (coding_dir / "code.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    coding_ds = CodingByteDataset(coding_dir, seq_len=8)
    assert len(coding_ds) > 0
    x_code, y_code = coding_ds[0]
    assert x_code.shape == (8,)
    assert y_code.shape == (8,)
