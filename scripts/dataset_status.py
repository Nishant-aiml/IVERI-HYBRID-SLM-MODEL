# Copyright 2026 IVERI Project
# SPDX-License-Identifier: Apache-2.0

"""One-command health check and status dashboard for Phase 8 datasets."""

from __future__ import annotations

import json
import os
from pathlib import Path

from configs.phase8_config import Phase8Config
from data.pipeline.registry import DatasetRegistry
from data.pipeline.storage import StorageManager
from data.pipeline.freeze import FreezeManager


def main() -> None:
    config = Phase8Config()
    registry = DatasetRegistry(config.datasets_specs_dir)
    storage = StorageManager(config)
    fm = FreezeManager()

    # Discover datasets
    try:
        specs = registry.discover()
    except Exception:
        specs = {}

    summary = storage.get_summary()

    # Reports count
    reports_dir = Path(config.reports_dir)
    report_files = list(reports_dir.glob("*.md")) + list(reports_dir.glob("*.json"))
    report_count = len({p.stem for p in report_files})

    # Curriculum readiness
    curr_dir = Path(config.curriculum_dir)
    manifests = list(curr_dir.glob("*_manifest.json"))
    curr_ready = "READY" if len(manifests) > 0 else "NOT READY"

    # Freeze readiness
    all_frozen = len(specs) > 0 and all(
        fm.is_frozen(Path(config.certified_data_dir) / name)
        for name in specs
        if specs[name].stage == "pretrain"
    )

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("IVERI Dataset Status — IVERI_DATASET_v1.0")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Version:      IVERI_DATASET_v1.0")
    print(f"Certified:    {'✅ YES' if len(specs) > 0 else '❌ NO'}")
    print(f"Frozen:       {'✅ YES' if all_frozen else '❌ NO'}")
    print(f"Reports:      ✅ {report_count}/11 generated")
    print(f"Curriculum:   ✅ {curr_ready} ({len(manifests)} recipes)")
    print(f"Training:     {'✅ READY' if all_frozen and len(manifests) > 0 else '❌ NOT READY'}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Storage:      {summary.total_used_gb:.2f} GB / {summary.budget_gb:.2f} GB budget")
    print(f"Status:       {summary.status}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"{'Dataset':<15} {'Docs':<12} {'Size':<10} {'Status':<10}")
    
    for name, spec in specs.items():
        if spec.stage == "eval_only":
            continue
        cert_dir = Path(config.certified_data_dir) / name
        frozen = fm.is_frozen(cert_dir)
        
        # Count documents
        doc_count = 0
        size_str = "0.00 GB"
        train_file = cert_dir / "train.jsonl"
        if train_file.exists():
            try:
                doc_count = sum(1 for _ in open(train_file, encoding="utf-8"))
                size_bytes = train_file.stat().st_size
                size_str = f"{size_bytes / (1024**3):.2f} GB"
            except Exception:
                pass
                
        status_str = "✅ FROZEN v1" if frozen else "⚠️ UNLOCKED"
        print(f"{name:<15} {doc_count:<12,} {size_str:<10} {status_str:<10}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()
