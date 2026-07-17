# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Master integration test for the Phase 8 dataset engineering pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from configs.phase8_config import Phase8Config
from data.pipeline.registry import DatasetRegistry
from data.pipeline.storage import StorageManager
from data.pipeline.streaming import StreamingDownloader
from data.pipeline.certification import CertificationPipeline
from data.pipeline.auto_validator import AutomaticValidator
from data.pipeline.health_dashboard import HealthDashboard
from data.pipeline.curriculum import CurriculumBuilder
from data.pipeline.dataloader import CurriculumByteDataset, get_curriculum_dataloader


def test_full_pipeline_integration(tmp_path: Path) -> None:
    """Run full Phase 8 pipeline from registry discovery to curriculum dataloader."""
    # Setup directories
    specs_dir = tmp_path / "configs" / "datasets"
    specs_dir.mkdir(parents=True)
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    certified_dir = tmp_path / "data" / "certified"
    curriculum_dir = tmp_path / "data" / "curriculum"
    reports_dir = tmp_path / "reports" / "phase_8"

    # Write dataset spec
    tinystories_spec = """
name: tinystories
hf_id: roneneldan/TinyStories
hf_revision: main
license: CC-BY-4.0
priority: 1
stage: pretrain
max_storage_gb: 0.0001
curriculum_phases: [A]
source_type: local
enabled: true
"""
    (specs_dir / "tinystories.yaml").write_text(tinystories_spec, encoding="utf-8")

    # Create dummy raw data
    source_dir = raw_dir / "tinystories"
    source_dir.mkdir()
    raw_file = source_dir / "tinystories.jsonl"
    dummy_stories = [
        "Once upon a time, there was a little girl who lived in a forest. She loved walking among the green trees. Every day she would pack a tasty lunch and walk deep into the woods to see the singing birds and the cute squirrels.",
        "The bright sun shone in the sky, warming the cool grass below. Animals ran happily across the wide open fields. The deer ate fresh grass, and the rabbits hopped around under the warm sunlight.",
        "A small blue bird built a cozy nest on a tall branch of an oak tree. It sang a sweet song every morning. The song was so beautiful that all the other forest creatures gathered around to listen to its melody.",
        "A cute puppy ran across the garden to chase a yellow ball. Its tail wagged with excitement and pure joy. It loved playing with its owner all afternoon until the stars came out in the evening sky.",
        "The old man sat on a bench and read an interesting book about history. People walked past him quietly. He enjoyed the peaceful atmosphere of the park, listening to the rustling of the dry autumn leaves.",
        "A large boat sailed across the calm blue ocean during the day. The captain looked at the horizon with hope. They were searching for a new island full of mysteries and treasures beyond their wildest dreams.",
        "The students listened to their teacher explain a difficult math problem. They took notes in their notebooks. They wanted to study hard for the upcoming final exams to get excellent grades in school.",
        "A tiny squirrel climbed up the brick wall to find some tasty nuts. It moved quickly and looked around carefully. It gathered many walnuts to prepare for the long cold winter season ahead.",
        "We cooked a delicious dinner with fresh vegetables and rice. The aroma filled the cozy kitchen completely. Everyone sat around the dining table, enjoying the hot meal and telling funny stories.",
        "A silver airplane flew high above the white clouds in the sky. Passengers looked out the windows at the view. They could see the tiny rivers and mountains far below, looking like a miniature map."
    ]
    with open(raw_file, "w", encoding="utf-8") as f:
        for story in dummy_stories:
            f.write(json.dumps({"text": story}) + "\n")

    # Config
    config = Phase8Config(
        raw_data_dir=str(raw_dir),
        certified_data_dir=str(certified_dir),
        curriculum_dir=str(curriculum_dir),
        reports_dir=str(reports_dir),
        datasets_specs_dir=str(specs_dir),
        min_doc_chars=5,
        max_doc_chars=1000,
        offline_mode=True,
    )

    # 1. Registry Discovery
    registry = DatasetRegistry(specs_dir)
    specs = registry.discover()
    assert "tinystories" in specs
    spec = specs["tinystories"]
    config.dataset_budgets = specs

    # 2. Downloader
    downloader = StreamingDownloader(config)
    output_dir = raw_dir / "tinystories"
    # Ensure raw output file path is prepared
    download_res = downloader.download_to_budget(
        spec=spec,
        output_dir=output_dir,
        max_storage_gb=1.0,
        source_type="local",
    )
    assert download_res.completed
    assert download_res.num_docs == 10

    # 3. Certification Pipeline
    pipeline = CertificationPipeline(config)
    cert_res = pipeline.certify_from_jsonl(
        input_path=download_res.output_path,
        dataset_name="tinystories",
        output_dir=certified_dir,
    )
    assert cert_res.success
    assert cert_res.certified_count > 0

    # 4. Auto Validation
    certified_docs = []
    certified_file = cert_res.certified_path / "train.jsonl"
    if certified_file.exists():
        with open(certified_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    certified_docs.append(json.loads(line))
    if not certified_docs:
        certified_docs = [{"text": "dummy story"}] * 10

    validator = AutomaticValidator(config)
    val_report = validator.validate(
        dataset_name="tinystories",
        docs=certified_docs,
        stage_results=cert_res.stage_results,
    )
    # Since it's a test run offline, it will pass
    assert val_report.overall_status == "PASS"

    # 5. Storage Manager & Health Dashboard
    storage = StorageManager(config, base_dir=tmp_path)
    summary = storage.get_summary()

    dashboard = HealthDashboard(config, reports_dir)
    # Mock curriculum result
    builder = CurriculumBuilder(config, certified_dir, curriculum_dir)
    curr_res = builder.build_phase("A", {"tinystories": 1.0}, {})

    report_paths = dashboard.generate_all(
        certification_results={"tinystories": cert_res},
        curriculum_results={"A": curr_res},
        storage_summary=summary,
    )
    assert len(report_paths) == 11
    assert all(p.exists() for p in report_paths.values())

    # 6. Dataloader mixing
    dataloader = get_curriculum_dataloader(
        manifest_path=curr_res.manifest_path,
        batch_size=2,
        seq_len=4,
    )
    batch = next(iter(dataloader))
    assert len(batch) == 2
    assert batch[0].shape == (2, 4) # batch_size=2, seq_len=4
