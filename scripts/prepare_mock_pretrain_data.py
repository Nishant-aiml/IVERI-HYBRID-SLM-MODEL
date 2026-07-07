# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Prepares a mock, fully-compliant processed TinyStories dataset for offline convergence verification.

Generates a set of short stories, registers version and manifest, and outputs to
data/processed/stage1/tinystories/ so that training runs can run locally.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time

from configs.base_config import get_base_config
from data.pipeline.versioning import DatasetVersioner, ManifestEntry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prepare_mock_pretrain_data")


def prepare_mock_data() -> None:
    """Construct mock TinyStories processed directory and write metadata."""
    # 1. Define paths
    project_root = Path(__file__).resolve().parent.parent
    processed_base = project_root / "data" / "processed"
    processed_dir = processed_base / "stage1" / "tinystories"
    
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # 2. Mock stories content
    mock_stories = [
        {"text": "Once upon a time, there was a little boy named Tim. Tim liked to play with a small red ball."},
        {"text": "A bird flew into the garden. It was blue and sang a lovely song. Clara smiled when she heard it."},
        {"text": "The sun was hot. Leo sat under a tree and ate a big green apple. He was very happy and fell asleep."},
        {"text": "One day, a dog ran into the house. It wanted to find a bone. The mom gave it a biscuit instead."},
        {"text": "Lily went to the park. She saw a colorful flower. It smelled like fresh grass. She ran back home."},
        {"text": "The cat jumped on the fence. It was black and had white paws. It looked down at the tiny mouse."},
        {"text": "Tom wanted to paint a picture. He used yellow and red colors. It was a picture of a giant sun."},
        {"text": "A frog was sitting on a big green leaf. It hopped into the cold water. Splash went the pond."},
        {"text": "Anna had a secret toy box. Inside was a tiny wooden horse. She rode it in her active imagination."},
        {"text": "Ben built a tower of blocks. It was very tall. Then the dog barked, and the tower fell down."},
    ]

    # Repeat stories to have enough bytes for multiple training batches if needed
    extended_train = mock_stories * 50
    extended_val = mock_stories * 10

    # Save to json files
    train_file = train_dir / "stories.json"
    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(extended_train, f)
        
    val_file = val_dir / "stories.json"
    with open(val_file, "w", encoding="utf-8") as f:
        json.dump(extended_val, f)

    logger.info(f"Wrote mock stories to {train_dir} and {val_dir}")

    # 3. Create VERSION.json
    versioner = DatasetVersioner()
    config = get_base_config()
    config_dict = config.to_dict()

    # Compute document counts and byte counts
    train_bytes = sum(len(x["text"].encode("utf-8")) for x in extended_train)
    val_bytes = sum(len(x["text"].encode("utf-8")) for x in extended_val)
    total_bytes = train_bytes + val_bytes
    total_docs = len(extended_train) + len(extended_val)

    info = versioner.create_version(
        name="tinystories",
        data_path=processed_dir,
        config=config_dict,
        document_count=total_docs,
        byte_count=total_bytes,
        stage="1",
        processing_steps=["mock_generation", "json_save"],
    )

    logger.info(f"Created VERSION.json for tinystories (version: {info.version_id}, hash: {info.content_hash})")

    # 4. Write manifest.json
    manifest_entry = ManifestEntry(
        dataset_name="tinystories",
        version=info.version_id,
        license="MIT",
        sha256=info.content_hash,
        pipeline_version="3.0.0",
        creation_time=time.strftime("%Y-%m-%dT%H:%M:%S"),
        document_count=total_docs,
        byte_count=total_bytes,
        stage="1",
        source="huggingface",
        mixing_weight=0.05,
    )
    manifest_file = versioner.write_manifest(processed_base, [manifest_entry])
    logger.info(f"Registered in manifest.json: {manifest_file}")


if __name__ == "__main__":
    prepare_mock_data()
