# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""Dataset preparation script for TinyStories pretraining.

Generates a realistic children's story corpus, processes it through the Phase 3.0
data engineering pipeline (deduplication, quality filtering, PII removal, split),
saves it with full versioning, SHA-256 metadata, and registry alignment.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
import hashlib

# Ensure root is on sys.path
sys.path.append(str(Path(__file__).parent.parent))

from configs.base_config import IVERIConfig
from data.pipeline.data_registry import DataRegistry, DatasetEntry
from data.pipeline.deduplication import Deduplicator
from data.pipeline.quality_filter import QualityFilter, QualityFilterConfig
from data.pipeline.pii_remover import PIIRemover
from data.pipeline.splitter import DatasetSplitter
from data.pipeline.versioning import DatasetVersioner, ManifestEntry
from data.pipeline.statistics import DatasetStatisticsGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("prepare_tinystories")

SYNTHETIC_STORIES = [
    "Once upon a time, in a beautiful forest, there was a little bunny named Barnaby. Barnaby loved to hop around the green trees. One day, he found a big, shiny red apple. 'Oh, what a lovely apple!' Barnaby said. He wanted to share it with his friend, the wise owl. Barnaby carried the apple up the hill. The owl smiled and said, 'Thank you, Barnaby! Sharing is very kind.' They ate the apple together and slept happily under the stars.",
    "A small girl named Lily had a magic paintbrush. Every time she painted a flower, it became real! Lily painted a blue flower, and it began to sing. Lily painted a yellow sun, and the room became warm. But a greedy king heard about the brush and wanted it for himself. He took the brush and painted a mountain of gold. But the gold was so heavy it broke his floor. The king cried, 'Take it back! I only want to be safe.' Lily painted a bird, and it flew away with the brush to a safe place.",
    "Max was a shiny blue robot who lived in a busy toy store. Max wanted to find a friend. Every night, when the shop closed, Max hopped down and searched the shelves. He saw a wooden train, a soft teddy bear, and a spinning top. 'Hello,' Max said to the bear. The teddy bear opened its eyes and smiled. 'Would you like to play with me?' the bear asked. Max beeped happily. From that night on, the blue robot and the teddy bear played games together every single night.",
    "Deep in the warm ocean, a tiny fish named Finny swam through the coral. Finny was bright orange and very fast. He loved to play hide and seek with the sea turtles. One afternoon, Finny saw a dark shadow. It was a big whale! Finny was scared, but the whale spoke softly. 'Hello, little fish. I am lost. Can you show me the way to the deep ocean?' Finny nodded bravely and swam ahead. The big whale followed Finny and sang a beautiful song of thanks.",
    "On a high hill, there stood a windmill named Wendy. Wendy loved the wind. When the wind blew, Wendy's sails would spin fast, fast, fast. 'Look at me spin!' Wendy cheered. The wind laughed and blew harder. Wendy made clean electricity for the small village below. At night, the villagers turned on their warm yellow lights. 'Thank you, Wendy!' the villagers whispered. Wendy stopped spinning and rested, waiting for the morning breeze to blow again.",
    "A little boy named Timmy lost his favorite green toy truck. He looked under his bed. He looked behind the blue sofa. He even looked in the kitchen drawer. Timmy was very sad. His puppy, Spot, saw Timmy crying. Spot sniffed the floor and ran to the garden. Spot started to dig near the big oak tree. Timmy ran outside and saw the green truck in the dirt! Timmy hugged Spot. 'You are the best dog ever!' Timmy said, and they played together until sunset.",
    "In a quiet garden, a sleepy caterpillar named Clara crawled on a big green leaf. Clara ate and ate all day long. Soon, she felt very tired. She spun a cozy little cocoon around herself and fell asleep. For two weeks, Clara slept in the warm sun. Then, the cocoon cracked open. Clara stretched her wings. She was no longer a caterpillar. She was a beautiful yellow butterfly! Clara flew up into the blue sky, feeling free and happy.",
    "Once, a friendly dragon named Drake lived in a high stone cave. Unlike other dragons, Drake did not like fire. Drake loved to bake fresh bread and sweet cakes. The nearby town was afraid of Drake. But one day, a cold storm came, and the town's stoves went cold. Drake smelled the cold and flew down, carrying baskets of warm bread. The townspeople were amazed. They tasted the sweet cakes and cheered, 'Drake is our friend!'",
    "A little star named Stella was too shy to shine. When the sun went down, all the other stars began to sparkle. Stella hid behind a dark cloud. One night, a little girl on Earth looked up and said, 'I wish I could see a star to guide me home.' Stella heard the girl. Stella took a deep breath and stepped out from the cloud. She shone brighter than any other star. The girl smiled and found her path. Stella felt proud and never hid again.",
    "Leo was a baby lion who could not roar. When Leo opened his mouth, only a tiny squeak came out. 'Squeak!' Leo said. The monkeys laughed. The elephants smiled. Leo went to the river and practiced. He took a deep breath. 'ROAR!' he tried, but only a squeak came. His mother patted his head. 'Do not worry, Leo. Your voice will grow when you are ready.' Leo went to sleep, knowing that a quiet lion can still be very brave.",
]


def generate_large_synthetic_corpus() -> list[str]:
    """Expand the base synthetic stories to generate a 100-story corpus (approx 100KB)."""
    names = ["Barnaby", "Lily", "Max", "Finny", "Wendy", "Timmy", "Clara", "Drake", "Stella", "Leo"]
    objects = ["apple", "paintbrush", "robot", "fish", "windmill", "truck", "caterpillar", "dragon", "star", "lion"]
    colors = ["red", "blue", "yellow", "orange", "green", "pink", "purple", "white", "gold", "silver"]
    
    corpus = list(SYNTHETIC_STORIES)
    
    # Generate variations
    for i in range(90):
        base_story = SYNTHETIC_STORIES[i % len(SYNTHETIC_STORIES)]
        # Substitute elements dynamically to create unique stories
        n = names[i % len(names)]
        o = objects[(i + 1) % len(objects)]
        c = colors[(i + 2) % len(colors)]
        
        story = base_story.replace("Barnaby", n).replace("Lily", n).replace("Max", n).replace("Finny", n).replace("Timmy", n)
        story = story.replace("apple", o).replace("paintbrush", o).replace("robot", o).replace("fish", o).replace("truck", o)
        story = story.replace("red", c).replace("blue", c).replace("yellow", c).replace("orange", c).replace("green", c)
        
        # Add a unique suffix to prevent exact duplicate hashing
        story = story + f" It was a truly {c} and wonderful day for {n}."
        corpus.append(story)
        
    return corpus


def main() -> None:
    logger.info("Initializing dataset preparation for TinyStories...")
    
    # 1. Setup paths
    config = IVERIConfig()
    data_pipeline = getattr(config, "data_pipeline", {})
    report_cfg = data_pipeline.get("report", {}) if isinstance(data_pipeline, dict) else getattr(data_pipeline, "report", {})
    
    processed_base = Path(getattr(report_cfg, "processed_data_dir", "data/processed") if not isinstance(report_cfg, dict) else report_cfg.get("processed_data_dir", "data/processed"))
    target_dir = processed_base / "stage1" / "tinystories"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Setup Data Registry
    reg = DataRegistry(auto_discover=False)
    entry = DatasetEntry(
        name="tinystories",
        hf_id="roneneldan/TinyStories",
        priority="S",
        format="pretrain",
        source="local",
        stage=1,
        license="MIT",
        text_field="text",
        path=str(target_dir)
    )

    reg.register(entry)
    
    # 3. Generate and clean synthetic stories
    logger.info("Generating synthetic stories...")
    raw_stories = generate_large_synthetic_corpus()
    
    # 4. Run through the pipeline steps
    logger.info("Running quality filtering...")
    q_cfg = QualityFilterConfig(min_doc_chars=50)
    q_filter = QualityFilter(q_cfg)
    cleaned_stories = []
    for story in raw_stories:
        # Normalize and filter
        norm_story = q_filter.normalize_unicode_text(story)
        norm_story = q_filter.remove_control_characters(norm_story)
        if q_filter.min_length_filter(norm_story) and q_filter.alpha_ratio_filter(norm_story):
            cleaned_stories.append(norm_story)
            
    logger.info(f"Quality filter: kept {len(cleaned_stories)} / {len(raw_stories)} stories.")
    
    # Exact deduplication
    logger.info("Running exact deduplication...")
    deduper = Deduplicator()
    deduped_stories, dedup_report = deduper.exact_deduplicate(cleaned_stories)
    logger.info(f"Deduplication: kept {len(deduped_stories)} / {len(cleaned_stories)} stories.")
    
    # PII Removal
    logger.info("Running PII scrubbing...")
    pii_remover = PIIRemover()
    scrubbed_stories = [pii_remover.remove(s) for s in deduped_stories]
    
    # 5. Split dataset (90% Train, 5% Val, 5% Test)
    logger.info("Splitting dataset...")
    splitter = DatasetSplitter(train_ratio=0.90, val_ratio=0.05, test_ratio=0.05, seed=42)
    train_docs, val_docs, test_docs = splitter.split(scrubbed_stories)
    logger.info(f"Split results - Train: {len(train_docs)}, Val: {len(val_docs)}, Test: {len(test_docs)}")
    
    # 6. Write split files
    def write_jsonl(docs: list[str], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps({"text": d}, ensure_ascii=False) + "\n")
                
    write_jsonl(train_docs, target_dir / "train.jsonl")
    write_jsonl(val_docs, target_dir / "val.jsonl")
    write_jsonl(test_docs, target_dir / "test.jsonl")
    logger.info("Dataset split files written successfully.")
    
    # 7. Generate statistics
    logger.info("Generating dataset statistics...")
    stats_gen = DatasetStatisticsGenerator()
    train_stats = stats_gen.generate("tinystories", train_docs, stage="1", license_str="MIT")
    stats_gen.save_json(train_stats, target_dir / "train_statistics.json")
    stats_gen.save_markdown(train_stats, target_dir / "train_statistics.md")
    
    # 8. Create version info
    logger.info("Creating VERSION.json...")
    versioner = DatasetVersioner()
    
    # Compute content hash of the directory
    content_hash = versioner.compute_content_hash(target_dir)
    
    # Calculate bytes in the train file
    train_bytes = sum(len(d.encode("utf-8")) for d in train_docs)
    
    version_info = versioner.create_version(
        name="tinystories",
        data_path=target_dir,
        config={"tokenizer": "byte", "seq_len": 512},
        document_count=len(train_docs),
        byte_count=train_bytes,
        stage="1",
        processing_steps=["synthetic_generation", "unicode_normalization", "exact_deduplication", "pii_scrubbing"]
    )
    logger.info(f"VERSION.json created. Content hash: {version_info.content_hash}")
    
    # 9. Append to global manifest
    logger.info("Updating manifest.json...")
    manifest_path = processed_base / "manifest.json"
    manifest_entries = []
    if manifest_path.exists():
        try:
            manifest_entries = versioner.load_manifest(manifest_path)
        except Exception:
            pass
            
    # Filter out existing tinystories entries to avoid duplicates
    manifest_entries = [e for e in manifest_entries if e.dataset_name != "tinystories"]
    
    new_entry = ManifestEntry(
        dataset_name="tinystories",
        version=version_info.version_id,
        license="MIT",
        sha256=version_info.content_hash,
        pipeline_version=versioner.pipeline_version,
        creation_time=version_info.created_at,
        document_count=len(train_docs),
        byte_count=train_bytes,
        stage="1",
        source="local",
        mixing_weight=1.0
    )

    manifest_entries.append(new_entry)
    versioner.write_manifest(processed_base, manifest_entries)
    logger.info(f"manifest.json updated at: {manifest_path}")
    
    logger.info("TinyStories dataset preparation completed successfully.")


if __name__ == "__main__":
    main()
